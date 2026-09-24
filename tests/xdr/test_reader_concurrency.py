# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import threading
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from cuphoton.xdr import gds, prefetch, reader
from cuphoton.xdr.gds import HeapReadHandle
from cuphoton.xdr.mock_storage import storage_cache
from cuphoton.xdr.reader import GpuCompImageReader


def test_independent_loader_preserves_active_kvikio_pool(monkeypatch):
    desired_threads = 4
    configured_threads = 1
    resets = []
    pending = False

    def get_default(key):
        assert key == "num_threads"
        return configured_threads

    def set_default(key, value):
        nonlocal configured_threads
        assert key == "num_threads"
        assert not pending, "reset would wait for another loader's I/O"
        resets.append(value)
        configured_threads = value

    class Future:
        def get(self):
            nonlocal pending
            pending = False

    class CuFile:
        def __init__(self, path, mode):
            self.path = path

        def pread(self, *args, **kwargs):
            nonlocal pending
            pending = True
            return Future()

        def close(self):
            pass

    defaults = SimpleNamespace(get=get_default, set=set_default)
    monkeypatch.setitem(sys.modules, "kvikio.defaults", defaults)
    monkeypatch.setitem(
        sys.modules,
        "kvikio",
        SimpleNamespace(defaults=defaults, CuFile=CuFile),
    )
    monkeypatch.setattr(gds, "available_cpu_cores", lambda: desired_threads)
    monkeypatch.setattr(storage_cache, "active", lambda: False)

    with gds.GdsHeapLoader("first.fits") as first:
        future = first._file().pread(object(), size=8, file_offset=0)
        try:
            with gds.GdsHeapLoader("second.fits") as second:
                assert second._file().path == "second.fits"
                assert pending is True
                assert resets == [4]
        finally:
            future.get()

    desired_threads = 8
    with gds.GdsHeapLoader("third.fits") as third:
        assert third._file().path == "third.fits"
    assert resets == [4, 8]


@pytest.fixture
def read_runtime(monkeypatch):
    current = threading.local()
    quarantine = []
    resources = {}

    class Device:
        def __init__(self, device_id=2):
            self.id = device_id

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class Stream:
        done = True
        synchronize_calls = 0

        def __enter__(self):
            self.previous = getattr(current, "stream", None)
            current.stream = self
            return self

        def __exit__(self, *_):
            current.stream = self.previous
            return False

        def synchronize(self):
            self.synchronize_calls += 1
            self.done = True

    class Event:
        done = False

        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            self.done = True

    null_stream = Stream()
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=Device,
                Stream=SimpleNamespace(null=null_stream),
                Event=Event,
            )
        ),
    )
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setattr(
        GpuCompImageReader,
        "prepare_plan",
        lambda *args, **kwargs: {
            "sel_abs_offsets": "offsets",
            "sel_lengths": "lengths",
        },
    )

    def decode(device_input, *args, out, keepalive=None, **kwargs):
        resource = resources[device_input]
        assert current.stream is resource.stream
        resource.decode_calls += 1
        resource.stream.done = False
        if keepalive is not None:
            keepalive.append(resource.scratch)
        return out

    monkeypatch.setattr(
        GpuCompImageReader, "decode_from_device_heap", staticmethod(decode)
    )

    def make_read(*, held=False, owns_loader=False, use_stream=True):
        resource = SimpleNamespace(
            stream=Stream() if use_stream else null_stream,
            output=object(),
            device_input=object(),
            scratch=object(),
            entered=threading.Event(),
            release=threading.Event(),
            load_calls=0,
            decode_calls=0,
            future_done=False,
            done_calls=0,
            get_calls=0,
            get_active=False,
            error=None,
        )
        if not held:
            resource.release.set()

        class Future:
            def done(self):
                resource.done_calls += 1
                assert not resource.get_active, "future polled during get()"
                return resource.future_done

            def get(self):
                assert current.stream is resource.stream
                resource.get_calls += 1
                resource.get_active = True
                try:
                    resource.entered.set()
                    assert resource.release.wait(5), (
                        "I/O wait was not released"
                    )
                    if resource.error is not None:
                        raise resource.error
                    resource.future_done = True
                finally:
                    resource.get_active = False

        resource.handle = HeapReadHandle(
            resource.device_input, "relative-offsets", [Future()]
        )

        class Loader:
            closed = False

            def load_tiles_async(self, offsets, lengths):
                assert current.stream is resource.stream
                resource.load_calls += 1
                return resource.handle

            def close(self):
                self.closed = True

        resource.loader = Loader()
        comp_reader = object.__new__(GpuCompImageReader)
        comp_reader.path = resource.device_input
        resources[resource.device_input] = resource
        resource.read = lambda: comp_reader.read(
            loader=None if owns_loader else resource.loader,
            stream=resource.stream if use_stream else None,
            out=resource.output,
        )
        return resource

    monkeypatch.setattr(
        reader, "GdsHeapLoader", lambda path: resources[path].loader
    )
    return SimpleNamespace(
        make_read=make_read,
        quarantine=quarantine,
        Stream=Stream,
        Event=Event,
    )


def _start_call(call):
    result = {}

    def run():
        try:
            result["value"] = call()
        except BaseException as error:
            result["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


@pytest.mark.parametrize("second_kind", ["standalone", "batch"])
def test_independent_submission_progresses_during_gds_wait(
    monkeypatch, read_runtime, second_kind
):
    first = read_runtime.make_read(held=True)
    second = read_runtime.make_read()
    in_flight = deque()
    first_thread, first_result = _start_call(first.read)
    second_thread = None
    try:
        assert first.entered.wait(5), "first read did not enter I/O wait"
        assert len(read_runtime.quarantine) == 1
        owner = read_runtime.quarantine[0]
        assert owner.actively_awaited is True
        assert owner.device_id == 2
        assert first.loader in owner.keepalive
        assert first.output in owner.keepalive
        assert first.handle in owner.keepalive
        # CUDA can be idle while the GDS futures still own their destination.
        assert first.stream.done is True
        assert owner.event.done is False
        assert first.done_calls == 0

        if second_kind == "standalone":
            second_call = second.read
        else:
            monkeypatch.setattr(
                prefetch, "_consume_prefetched_group", lambda *a, **kw: None
            )

            def second_call():
                prefetch._submit_prefetched_group(
                    [],
                    [second.output],
                    stream=second.stream,
                    owner=second.loader,
                    in_flight=in_flight,
                )
                prefetch._wait_for_all_gpu_batches(in_flight)
                return second.output

        second_thread, second_result = _start_call(second_call)
        second_thread.join(5)
        assert not second_thread.is_alive(), "independent work was blocked"
        assert second_result == {"value": second.output}
        assert first_result == {}
        assert first.decode_calls == first.stream.synchronize_calls == 0
        assert read_runtime.quarantine == [owner]
        assert owner.actively_awaited is True
        assert first.done_calls == 0
    finally:
        first.release.set()
        first_thread.join(5)
        if second_thread is not None:
            second_thread.join(5)

    assert not first_thread.is_alive()
    assert first_result == {"value": first.output}
    assert read_runtime.quarantine == []
    assert not in_flight


@pytest.mark.parametrize(
    "error_type", [KeyboardInterrupt, SystemExit, RuntimeError]
)
@pytest.mark.parametrize("owns_loader", [False, True])
def test_abandoned_pending_future_blocks_submission_on_idle_stream(
    read_runtime, error_type, owns_loader
):
    first = read_runtime.make_read(owns_loader=owns_loader)
    first.error = error_type("I/O wait failed")
    with pytest.raises(error_type) as caught:
        first.read()
    assert caught.value is first.error
    first.error.__traceback__ = None
    owner = read_runtime.quarantine[0]
    assert owner.actively_awaited is False
    assert first.loader in owner.keepalive
    assert first.output in owner.keepalive
    assert first.handle in owner.keepalive
    assert first.stream.done is True
    assert first.done_calls == 0
    assert first.handle.done is False
    assert first.loader.closed is False
    assert first.stream.synchronize_calls == 0

    second = read_runtime.make_read()
    with pytest.raises(RuntimeError, match="previous xDR GPU work"):
        second.read()
    assert second.load_calls == 0
    assert read_runtime.quarantine == [owner]
    assert first.get_calls == 1
    assert first.done_calls >= 2

    pending_probes = first.done_calls
    first.future_done = True
    assert second.read() is second.output
    assert first.get_calls == 1
    assert first.done_calls > pending_probes
    assert read_runtime.quarantine == []


@pytest.mark.parametrize("owns_loader", [False, True])
def test_foreign_unresolved_batch_during_wait_rejects_decode(
    read_runtime, owns_loader
):
    first = read_runtime.make_read(held=True, owns_loader=owns_loader)
    first.stream.done = False
    foreign_event = read_runtime.Event()
    foreign = prefetch._GpuBatchHandle(
        foreign_event, [object()], read_runtime.Stream()
    )
    thread, result = _start_call(first.read)
    publisher = None
    try:
        assert first.entered.wait(5), "first read did not enter I/O wait"
        owner = read_runtime.quarantine[0]
        publisher, published = _start_call(
            lambda: prefetch._quarantine_gpu_batches([foreign])
        )
        publisher.join(5)
        assert not publisher.is_alive(), "I/O wait held the publication lock"
        assert published == {"value": None}
    finally:
        first.release.set()
        thread.join(5)
        if publisher is not None:
            publisher.join(5)

    assert not thread.is_alive()
    assert isinstance(result.get("error"), RuntimeError)
    assert "previous xDR GPU work" in str(result["error"])
    assert first.decode_calls == first.stream.synchronize_calls == 0
    assert owner in read_runtime.quarantine
    assert owner.actively_awaited is False
    assert first.handle in owner.keepalive
    assert first.device_input in owner.keepalive
    assert first.output in owner.keepalive
    assert first.loader.closed is owns_loader

    foreign_event.done = True
    with pytest.raises(RuntimeError, match="previous xDR GPU work"):
        prefetch._ensure_gpu_submissions_safe()
    assert read_runtime.quarantine == [owner]
    first.stream.done = True
    prefetch._ensure_gpu_submissions_safe()
    assert read_runtime.quarantine == []


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("owns_loader", [False, True])
def test_first_gate_exit_signal_abandons_published_io(
    monkeypatch, read_runtime, error_type, owns_loader
):
    first = read_runtime.make_read(owns_loader=owns_loader)
    error = error_type("first gate exit interrupted")
    original_guard = prefetch._gpu_submission_guard
    entries = 0

    @contextmanager
    def interrupt_first_exit():
        nonlocal entries
        entries += 1
        with original_guard():
            yield
        if entries == 1:
            raise error

    monkeypatch.setattr(
        prefetch, "_gpu_submission_guard", interrupt_first_exit
    )
    with pytest.raises(error_type) as caught:
        first.read()
    assert caught.value is error
    assert first.load_calls == 1
    assert first.get_calls == first.decode_calls == 0
    assert first.stream.synchronize_calls == 0
    assert len(read_runtime.quarantine) == 1
    owner = read_runtime.quarantine[0]
    assert owner.actively_awaited is False
    assert first.loader in owner.keepalive
    assert first.handle in owner.keepalive
    assert first.output in owner.keepalive
    assert first.loader.closed is False

    second = read_runtime.make_read()
    with pytest.raises(RuntimeError, match="previous xDR GPU work"):
        second.read()
    assert second.load_calls == 0
    assert read_runtime.quarantine == [owner]
    first.future_done = True
    assert second.read() is second.output
    assert read_runtime.quarantine == []


@pytest.mark.parametrize(
    "error_type", [KeyboardInterrupt, SystemExit, RuntimeError]
)
@pytest.mark.parametrize("owns_loader", [False, True])
def test_second_gate_entry_error_abandons_published_owner(
    monkeypatch, read_runtime, error_type, owns_loader
):
    first = read_runtime.make_read(owns_loader=owns_loader)
    first.stream.done = False
    error = error_type("second gate entry failed")
    original_guard = prefetch._gpu_submission_guard
    entries = 0

    @contextmanager
    def fail_second_entry():
        nonlocal entries
        entries += 1
        if entries == 2:
            raise error
        with original_guard():
            yield

    monkeypatch.setattr(prefetch, "_gpu_submission_guard", fail_second_entry)
    with pytest.raises(error_type) as caught:
        first.read()
    assert caught.value is error
    assert entries == 2
    assert first.handle.done is True
    assert first.decode_calls == first.stream.synchronize_calls == 0
    assert len(read_runtime.quarantine) == 1
    owner = read_runtime.quarantine[0]
    assert owner.actively_awaited is False
    assert first.handle in owner.keepalive
    assert first.device_input in owner.keepalive
    assert first.output in owner.keepalive
    assert first.loader.closed is owns_loader

    with pytest.raises(RuntimeError, match="previous xDR GPU work"):
        prefetch._ensure_gpu_submissions_safe()
    first.stream.done = True
    prefetch._ensure_gpu_submissions_safe()
    assert read_runtime.quarantine == []


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("error_stage", ["load", "wait", "decode"])
def test_stream_none_preserves_control_signal_cleanup(
    monkeypatch, read_runtime, error_type, error_stage
):
    first = read_runtime.make_read(owns_loader=True, use_stream=False)
    error = error_type("interrupted during " + error_stage)

    def interrupt(*args, **kwargs):
        raise error

    if error_stage == "load":
        monkeypatch.setattr(first.loader, "load_tiles_async", interrupt)
    elif error_stage == "wait":
        first.error = error
    else:
        monkeypatch.setattr(
            GpuCompImageReader,
            "decode_from_device_heap",
            staticmethod(interrupt),
        )

    with pytest.raises(error_type) as caught:
        first.read()
    assert caught.value is error
    assert first.loader.closed is (error_stage == "decode")
    assert first.stream.synchronize_calls == 0
    assert len(read_runtime.quarantine) == 1
    completion = read_runtime.quarantine[0]
    assert first.loader in completion.keepalive
    assert first.output in completion.keepalive
    assert completion.actively_awaited is False


def test_stream_none_retains_locked_io_wait(monkeypatch, read_runtime):
    attempted = threading.Event()

    class ObservedRLock:
        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "contender":
                attempted.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_):
            self.lock.release()
            return False

    monkeypatch.setattr(
        prefetch, "_UNRESOLVED_GPU_BATCHES_LOCK", ObservedRLock()
    )
    first = read_runtime.make_read(held=True, use_stream=False)
    second = read_runtime.make_read()
    first_thread, first_result = _start_call(first.read)
    second_thread = None

    def contend():
        threading.current_thread().name = "contender"
        return second.read()

    try:
        assert first.entered.wait(5), "first read did not enter I/O wait"
        second_thread, second_result = _start_call(contend)
        assert attempted.wait(5), "second read did not attempt the gate"
        assert second.load_calls == 0
        assert read_runtime.quarantine == []
    finally:
        first.release.set()
        first_thread.join(5)
        if second_thread is not None:
            second_thread.join(5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert first_result == {"value": first.output}
    assert second_result == {"value": second.output}
    assert first.stream.synchronize_calls == 0
    assert second.stream.synchronize_calls == 1
    assert read_runtime.quarantine == []


def test_concurrent_kvikio_configuration_does_not_reset_active_io(
    monkeypatch,
):
    configured_threads = 1
    pending = False
    resets = []
    first_get = threading.Event()
    contender_attempted = threading.Event()
    first_io = threading.Event()
    release_first = threading.Event()

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "contender":
                contender_attempted.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_):
            self.lock.release()

    def get_default(key):
        assert key == "num_threads"
        observed = configured_threads
        if threading.current_thread().name == "first":
            first_get.set()
            assert contender_attempted.wait(5), "contender did not start"
        else:
            contender_attempted.set()
            assert first_io.wait(5), "first caller did not begin I/O"
        return observed

    def set_default(key, value):
        nonlocal configured_threads
        assert key == "num_threads"
        assert not pending, "pool reset after another caller began I/O"
        resets.append(value)
        configured_threads = value

    defaults = SimpleNamespace(get=get_default, set=set_default)
    monkeypatch.setitem(sys.modules, "kvikio.defaults", defaults)
    monkeypatch.setitem(
        sys.modules, "kvikio", SimpleNamespace(defaults=defaults)
    )
    monkeypatch.setattr(gds, "available_cpu_cores", lambda: 4)
    monkeypatch.setattr(
        gds, "_KVIKIO_DEFAULTS_LOCK", ObservedLock(), raising=False
    )

    def first_call():
        nonlocal pending
        threading.current_thread().name = "first"
        configured = gds.configure_kvikio_parallelism()
        pending = True
        first_io.set()
        assert release_first.wait(5), "first caller was not released"
        return configured

    def second_call():
        threading.current_thread().name = "contender"
        return gds.configure_kvikio_parallelism()

    first_thread, first_result = _start_call(first_call)
    second_thread = None
    try:
        assert first_get.wait(5), "first caller did not inspect the default"
        second_thread, second_result = _start_call(second_call)
        second_thread.join(5)
        assert not second_thread.is_alive(), "configuration did not complete"
        assert second_result == {"value": 4}
        assert pending is True
        assert resets == [4]
    finally:
        release_first.set()
        first_thread.join(5)
        if second_thread is not None:
            second_thread.join(5)

    assert not first_thread.is_alive()
    assert first_result == {"value": 4}
