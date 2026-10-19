# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import sys
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cuphoton.xdr import gds, prefetch


@pytest.fixture
def kvikio_defaults(monkeypatch):
    kvikio = ModuleType("kvikio")
    defaults = ModuleType("kvikio.defaults")
    kvikio.defaults = defaults
    monkeypatch.setitem(sys.modules, "kvikio", kvikio)
    monkeypatch.setitem(sys.modules, "kvikio.defaults", defaults)
    monkeypatch.setattr(gds, "_KVIKIO_NUM_THREADS", None)
    monkeypatch.delenv("KVIKIO_NTHREADS", raising=False)
    return defaults


def test_concurrent_readers_initialize_pool_once(
    monkeypatch, kvikio_defaults
):
    pool_threads = 1
    resets = []
    start = Barrier(8)

    def reset_pool(name, count):
        resets.append((name, count))
        # The real C++ pool reset releases the GIL. Other readers must wait
        # until it finishes, then use this pool without resetting it again.
        time.sleep(0.01)
        nonlocal pool_threads
        pool_threads = count

    def configure_reader():
        start.wait(timeout=5)
        return gds.configure_kvikio_parallelism()

    monkeypatch.setattr(gds, "available_cpu_cores", lambda: 12)
    kvikio_defaults.set = reset_pool
    kvikio_defaults.get = lambda name: pool_threads
    with ThreadPoolExecutor(max_workers=8) as readers:
        futures = [readers.submit(configure_reader) for _ in range(8)]
        assert [future.result(timeout=5) for future in futures] == [12] * 8
    assert resets == [("num_threads", 12)]

    # Later readers cannot resize a pool that earlier readers still use.
    monkeypatch.setattr(gds, "available_cpu_cores", lambda: 24)
    assert gds.configure_kvikio_parallelism() == 12
    assert resets == [("num_threads", 12)]


def test_explicit_kvikio_threads_do_not_reset_pool(
    monkeypatch, kvikio_defaults
):
    monkeypatch.setenv("KVIKIO_NTHREADS", "2")
    kvikio_defaults.get = lambda name: 2

    def reject_reset(*args):
        raise AssertionError("Explicit KvikIO configuration was overwritten")

    kvikio_defaults.set = reject_reset
    monkeypatch.setattr(gds, "available_cpu_cores", reject_reset)

    assert gds.configure_kvikio_parallelism() == 2
    assert gds.configure_kvikio_parallelism() == 2


def test_failed_pool_initialization_can_retry(monkeypatch, kvikio_defaults):
    attempts = []

    def reset_pool(name, count):
        attempts.append(count)
        if len(attempts) == 1:
            raise RuntimeError("Pool initialization failed")

    monkeypatch.setattr(gds, "available_cpu_cores", lambda: 4)
    kvikio_defaults.set = reset_pool
    kvikio_defaults.get = lambda name: 4 if len(attempts) >= 2 else 1

    with pytest.raises(RuntimeError, match="Pool initialization failed"):
        gds.configure_kvikio_parallelism()
    assert gds.configure_kvikio_parallelism() == 4
    assert attempts == [4, 4]


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("short_read", [False, True])
def test_heap_reads_check_each_completion(
    monkeypatch, contiguous, short_read
):
    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(empty=np.empty, uint8=np.uint8)
    )
    calls = []
    waits = []

    def pread(buffer, *, size, file_offset):
        index = len(calls)
        calls.append((file_offset, size))
        assert buffer.nbytes == size

        def get():
            waits.append(index)
            return size - 1 if short_read and index == 0 else size

        return SimpleNamespace(get=get, done=lambda: index in waits)

    loader = gds.GdsHeapLoader("input.fits")
    monkeypatch.setattr(loader, "_file", lambda: SimpleNamespace(pread=pread))
    offsets = [10, 14] if contiguous else [10, 40]
    handle = loader.load_tiles_async(offsets, [4, 6])
    assert waits == []
    assert calls == ([(10, 10)] if contiguous else [(10, 4), (40, 6)])
    if short_read:
        with pytest.raises(OSError, match="xDR read size differs") as first:
            handle.wait()
        with pytest.raises(OSError) as repeated:
            handle.wait()
        assert repeated.value is first.value
    else:
        buffer, relative = handle.wait()
        assert buffer.nbytes == 10
        np.testing.assert_array_equal(relative, [0, 4])
        assert handle.wait()[0] is buffer
    assert waits == list(range(len(calls)))
    assert handle.done


def test_mock_heap_read_reports_completion_before_wait(monkeypatch):
    from cuphoton.xdr.mock_storage import storage_cache

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(empty=np.empty, uint8=np.uint8, asarray=np.asarray),
    )
    contents = np.arange(16, dtype=np.uint8)
    monkeypatch.setattr(storage_cache, "active", lambda: True)
    monkeypatch.setattr(storage_cache, "get_buffer", lambda path: contents)
    with gds.GdsHeapLoader("cached.fits") as loader:
        handle = loader.load_tiles_async([2, 4], [2, 3])
        assert handle.done
        output, offsets = handle.wait()
        np.testing.assert_array_equal(output, contents[2:7])
        np.testing.assert_array_equal(offsets, [0, 2])
        assert handle.done


def test_failed_submission_blocks_reads_before_releasing_guard(monkeypatch):
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", [])
    original_guard = prefetch._gpu_submission_guard
    failure = OSError("read submission failed")

    @contextmanager
    def observed_guard():
        with original_guard():
            try:
                yield
            finally:
                with pytest.raises(
                    RuntimeError, match="buffers remain quarantined"
                ):
                    prefetch._ensure_gpu_submissions_safe()

    monkeypatch.setattr(prefetch, "_gpu_submission_guard", observed_guard)
    with pytest.raises(OSError) as exc_info:
        with gds._read_submission([], owners=(object(),), file=object()):
            raise failure
    assert exc_info.value is failure


@pytest.mark.parametrize(
    "error_type", [OSError, KeyboardInterrupt, SystemExit]
)
def test_heap_io_failure_preserves_unconfirmed_reads_and_error(
    monkeypatch, error_type
):
    quarantine = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    failure = error_type("I/O failed")
    waits = []

    def consumed_done():
        pytest.fail("KvikIO cannot poll a future consumed by failed get")

    def failed_read():
        waits.append("failed")
        raise failure

    def successful_read():
        waits.append("completed")
        return 4

    failed_future = SimpleNamespace(get=failed_read, done=consumed_done)
    other_future = SimpleNamespace(get=successful_read)
    handle = gds.HeapReadHandle(
        object(),
        [0, 4],
        [
            (failed_future, 4),
            (other_future, 4),
        ],
    )
    for _ in range(2):
        with pytest.raises(error_type) as exc_info:
            handle.wait()
        assert exc_info.value is failure
    assert waits == (
        ["failed", "completed"] if error_type is OSError else ["failed"]
    )
    assert not handle.done
    assert len(quarantine) == 1
    assert handle.d_buf in quarantine[0].keepalive
    pending_reads = quarantine[0].event.reads
    assert pending_reads is handle._reads
    assert pending_reads[0][0] is failed_future
    if error_type is OSError:
        assert len(pending_reads) == 1
    else:
        assert len(pending_reads) == 2
        assert pending_reads[1][0] is other_future


@pytest.mark.parametrize(
    "operation",
    [
        "direct",
        "cache",
        "image-loader",
        "heap-handoff",
        "direct-wait-entry",
        "heap-wait-entry",
    ],
)
def test_failed_read_retains_buffer_and_file_after_traceback_is_cleared(
    tmp_path, monkeypatch, operation
):
    from contextlib import nullcontext

    from cuphoton.xdr import reader
    from cuphoton.xdr.mock_storage import StorageCache, storage_cache

    path = tmp_path / "input.bin"
    path.write_bytes(b"1234")
    quarantine = []
    refs = []
    calls = []
    monkeypatch.setattr(prefetch, "_UNRESOLVED_GPU_BATCHES", quarantine)
    monkeypatch.setattr(gds, "configure_kvikio_parallelism", lambda: None)
    monkeypatch.setattr(storage_cache, "active", lambda: False)

    class Future:
        def get(self):
            calls.append("get")
            raise RuntimeError("simulated read failure")

        def done(self):
            raise ValueError("future no longer has shared state")

    class File:
        closed = False

        def __init__(self, *args):
            pass

        def pread(self, buffer, **kwargs):
            refs.extend([weakref.ref(buffer), weakref.ref(self)])
            calls.append("pread")
            return Future()

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "kvikio", SimpleNamespace(CuFile=File))
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=np.empty,
            uint8=np.uint8,
            dtype=np.dtype,
            cuda=SimpleNamespace(Stream=SimpleNamespace(null=nullcontext())),
        ),
    )

    def run():
        if operation.startswith("direct"):
            gds.pread_to_device(path, 0, 4, np.empty(4, dtype=np.uint8))
        elif operation == "cache":
            cache = StorageCache()
            cache._location = "device"
            cache._load(str(path))
        elif operation == "image-loader":
            image = reader.GpuImageReader(
                SimpleNamespace(
                    header={"NAXIS": 1, "BITPIX": 8, "NAXIS1": 4},
                    _file=SimpleNamespace(name=str(path)),
                    _data_offset=0,
                )
            )
            with gds.GdsHeapLoader(path) as loader:
                image.read(loader=loader)
        else:
            with gds.GdsHeapLoader(path) as loader:
                handle = loader.load_tiles_async([0], [4])
                handle.wait()

    if operation == "heap-handoff":

        def fail_handoff(*args, **kwargs):
            raise MemoryError("simulated read failure")

        monkeypatch.setattr(gds, "HeapReadHandle", fail_handoff)
    elif operation.endswith("wait-entry"):

        def interrupt_wait(*args, **kwargs):
            raise KeyboardInterrupt("simulated read failure")

        monkeypatch.setattr(gds, "_wait_for_reads", interrupt_wait)

    with pytest.raises(
        (RuntimeError, MemoryError, KeyboardInterrupt),
        match="simulated read failure",
    ) as caught:
        run()
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert len(quarantine) == 1
    assert all(ref() is not None for ref in refs)
    assert refs[1]().closed is False
    assert any(owner is refs[0]() for owner in quarantine[0].keepalive)
    with pytest.raises(RuntimeError, match="previous xDR GPU work"):
        run()
    assert calls == (
        ["pread", "get"]
        if operation in {"direct", "cache", "image-loader"}
        else ["pread"]
    )
