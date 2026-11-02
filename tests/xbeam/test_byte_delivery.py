# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Byte delivery works without any HDF5 layout or VDIF frame semantics."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam.runtime.batches import WorkQueue
from cuphoton.xbeam.runtime.buffers import copy_bytes
from cuphoton.xbeam.runtime.device_delivery import (
    DeviceLease,
    DeviceReadBatches,
)
from cuphoton.xbeam.runtime.posix_io import (
    RangeReader,
    ShortReadError,
    read_exact_into,
)
from cuphoton.xbeam.runtime.transports import (
    CuFileTransport,
    PosixTransport,
    _ConsumptionFence,
    _CuFileRangeReader,
    _DeviceTarget,
)


def test_positioned_reads_retry_interruptions_and_partial_transfers(
    tmp_path, monkeypatch
):
    path = tmp_path / "raw.bin"
    expected = np.arange(200, dtype="u1")
    path.write_bytes(expected.tobytes())
    original = os.preadv
    calls = 0

    def partial(fd, buffers, offset):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError()
        return original(fd, [buffers[0][:7]], offset)

    monkeypatch.setattr(os, "preadv", partial)
    output = np.empty(111, dtype="u1")
    with path.open("rb", buffering=0) as file:
        file.seek(13)
        read_exact_into(file.fileno(), output, 41)
        assert file.tell() == 13
    np.testing.assert_array_equal(output, expected[41:152])
    assert calls > 2


def test_short_read_keeps_exact_received_count(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"abcdefg")
    output = bytearray([255] * 10)
    with path.open("rb", buffering=0) as file:
        with pytest.raises(ShortReadError) as captured:
            read_exact_into(file.fileno(), output, 3)
    assert (captured.value.expected, captured.value.received) == (10, 4)
    assert output == b"defg" + bytes([255] * 6)


def test_byte_task_failure_drains_unstarted_work_once():
    queue = WorkQueue([(0, 10), (10, 20), (30, 40)])
    assert queue.take() == (0, 10)
    error = OSError("raw read failed")
    assert queue.finish(error) == [(10, 20), (30, 40)]
    assert queue.take() is None
    assert queue.abort(error) == []
    with pytest.raises(RuntimeError, match="underflow"):
        queue.finish(None)


class RawBatches(DeviceReadBatches):
    def _plan_request(self, request):
        offset, length = request
        return {"bytes": ([(offset, length, 0)], length)}

    def _assemble(self, buffers, plans, request):
        return {"bytes": buffers["bytes"][: request[1]]}


class RawPosix(PosixTransport, RawBatches):
    pass


class RawCuFile(CuFileTransport, RawBatches):
    pass


def test_failed_submission_does_not_leave_cleanup_waiting():
    failure = OSError("cannot submit read")

    def submit(*args):
        raise failure

    reader = RawPosix.__new__(RawPosix)
    reader.reader = SimpleNamespace(submit_batch=submit)
    slot = SimpleNamespace(host_buffers={})
    lease = SimpleNamespace(submitted=None)
    with pytest.raises(OSError) as caught:
        reader._enqueue(slot, lease, {}, None)
    assert caught.value is failure
    assert lease.submitted.done()
    with pytest.raises(OSError) as caught:
        lease.submitted.result()
    assert caught.value is failure


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["posix", "cufile"])
def test_binary_bytes_use_same_transports_as_recording_adapters(
    tmp_path, backend
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = tmp_path / "raw.bin"
    expected = np.random.default_rng(91).integers(0, 256, 65536, dtype="u1")
    path.write_bytes(expected.tobytes())
    requests = [(offset, 4096) for offset in range(0, len(expected), 4096)]
    with path.open("rb", buffering=0) as file:
        reader = RangeReader(file.fileno(), workers=2, chunk_bytes=4096)
        try:
            batches = (
                RawPosix(reader, path, requests, 4096)
                if backend == "posix"
                else RawCuFile(path, requests, 4096)
            )
            actual = []
            with batches:
                for lease in batches:
                    lease.wait()
                    actual.append(lease["bytes"].copy())
                    lease.release()
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_array_equal(
                cp.asnumpy(cp.concatenate(actual)), expected
            )
        finally:
            reader.close()


@pytest.mark.parametrize("size", [-1, 9])
def test_copy_bounds_are_rejected_before_cuda_submission(size):
    calls = []
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                memcpyAsync=lambda *args: calls.append(args),
            )
        )
    )
    with pytest.raises(ValueError, match="copy byte count"):
        copy_bytes(
            xp, np.zeros(8, dtype="u1"), np.ones(8, dtype="u1"), size, None
        )
    assert not calls


def test_strided_copy_is_rejected_before_cuda_submission():
    calls = []
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                memcpyAsync=lambda *args: calls.append(args),
            )
        )
    )
    with pytest.raises(ValueError, match="contiguous"):
        copy_bytes(
            xp, np.zeros(8, dtype="u1"), np.ones(16, dtype="u1")[::2], 8, None
        )
    assert not calls


@pytest.mark.parametrize("received", [4096, 0, -5])
def test_cufile_dma_waits_for_consumption_and_checks_count(received):
    calls = []
    reader = _CuFileRangeReader.__new__(_CuFileRangeReader)
    reader.xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda _: nullcontext())
    )
    reader.device, reader.file = 0, object()
    slot = SimpleNamespace(
        buffers={"raw": SimpleNamespace(data=SimpleNamespace(ptr=8192))},
        stream=SimpleNamespace(synchronize=lambda: calls.append("consumed")),
    )

    def read(file, pointer, length, offset, *, buffer_offset):
        calls.append((pointer, length, offset, buffer_offset))
        return received

    reader.session = SimpleNamespace(read=read)
    target = _DeviceTarget(
        slot.buffers["raw"], 8192, _ConsumptionFence(slot.stream)
    )
    if received == 4096:
        reader._read_into(target, 4096, 8192, 4096)
    else:
        with pytest.raises(OSError, match="short/failed read"):
            reader._read_into(target, 4096, 8192, 4096)
    assert calls == ["consumed", (8192, 4096, 4096, 4096)]


def test_consumed_event_fences_actual_consumer_after_all_tile_events():
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append((self.name, "wait", event))

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            calls.append((stream.name, "record", self))

        def synchronize(self):
            calls.append(("host", "synchronize", self))

    consumer, idle = Stream("consumer"), Stream("idle")
    current = [consumer]
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda _: nullcontext(),
            Event=Event,
            get_current_stream=lambda: current[0],
        )
    )
    ready, first_tile, second_tile = Event(), Event(), Event()
    lease = DeviceLease({}, ready, xp, 0)
    lease.wait()
    assert lease.consumer is consumer
    lease.tile_events = [first_tile, second_tile]
    # Releasing on another host-current stream must still protect the
    # stream that actually consumed the slot, including other tile users.
    current[0] = idle
    lease.release()
    assert calls == [
        ("host", "synchronize", ready),
        ("consumer", "wait", ready),
        ("consumer", "wait", first_tile),
        ("consumer", "wait", second_tile),
        ("consumer", "record", lease.consumed),
    ]
    lease.release()
    assert len(calls) == 5


def test_reused_slot_waits_for_consumption_before_enqueuing_new_input():
    calls = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait_event(self, event):
            calls.append(("wait", event))

    class Event:
        def synchronize(self):
            calls.append(("host", self))

    class Reader(RawBatches):
        def _allocate(self, slot, name):
            calls.append(("allocate", name))

        def _enqueue(self, slot, lease, plans, request):
            calls.append(("enqueue", request))

    reader = Reader(None, [], 4096, queue_depth=2)
    reader.xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda _: nullcontext(), Event=Event)
    )
    reader.device = 0
    for slot in reader.slots:
        slot.stream = Stream()
    first = reader.read_request((0, 4096))
    reader.read_request((4096, 4096))
    first.consumed = object()
    first.submitted = SimpleNamespace(
        result=lambda: calls.append(("submission", first))
    )
    calls.clear()
    reader.read_request((8192, 4096))
    assert calls == [
        ("submission", first),
        ("host", first.ready),
        ("wait", first.consumed),
        ("allocate", "bytes"),
        ("enqueue", (8192, 4096)),
    ]
