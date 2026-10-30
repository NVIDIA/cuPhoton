# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submission overlap, ordered publication and common transport contracts."""

import os
from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam import (
    ArraySource,
    CorrelationSpec,
    Hdf5Source,
    correlate_series,
)
from tests.xbeam.test_hdf5_posix import write_input


@pytest.mark.parametrize("rows", [(2, 3, 1), (1, 4, 2), (3, 2, 4)])
def test_device_block_splits_into_exact_requested_tiles_on_cpu(rows):
    from cuphoton.xbeam.adapters.hdf5_batches import DeviceBatches
    from cuphoton.xbeam.runtime.device_delivery import DeviceLease

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

    stream = SimpleNamespace(wait_event=lambda _: None)
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda _: nullcontext(),
            Event=lambda **_: Event(),
            get_current_stream=lambda: stream,
        )
    )
    values = np.arange(sum(rows) * 5).reshape(sum(rows), 5)
    mask = values % 3 != 0
    owner = DeviceLease(
        {"a": (values, mask), "b": (-values, None)}, Event(), xp, 0
    )
    owner.rows = rows
    reader = DeviceBatches.__new__(DeviceBatches)
    reader.closed = False
    reader.tiles = deque()
    reader.pending = deque([owner])
    reader.requests = iter(())
    start = 0
    for count in rows:
        tile = next(reader)
        tile.wait()
        assert (tile.start, tile.stop) == (start, start + count)
        np.testing.assert_array_equal(
            tile["a"][0], values[start : start + count]
        )
        np.testing.assert_array_equal(
            tile["a"][1], mask[start : start + count]
        )
        np.testing.assert_array_equal(
            tile["b"][0], -values[start : start + count]
        )
        assert tile["b"][1] is None
        tile.release()
        tile.release()
        assert tile.released
        start += count
    assert owner.remaining_tiles == 0
    assert owner.consumed is not None
    with pytest.raises(StopIteration):
        next(reader)


def test_both_signals_are_submitted_before_first_signal_finishes(
    tmp_path, monkeypatch
):
    path = tmp_path / "data.h5"
    arrays, _ = write_input(path)
    with Hdf5Source(path, {"a": "a", "b": "b"}, upload_workers=2) as source:
        first = source._layouts["/a"].offset
        second = source._layouts["/b"].offset
        started = Event()
        original = os.preadv

        def read(fd, buffers, offset):
            if offset == first:
                assert started.wait(3)
            if offset == second:
                started.set()
            return original(fd, buffers, offset)

        monkeypatch.setattr(os, "preadv", read)
        with source.prefetch_batches([(("a", "b"), [0], 1024)]) as reader:
            lease = next(reader)
            lease.wait()
            np.testing.assert_array_equal(
                lease["a"][0][0], arrays["a"][:1024]
            )
            np.testing.assert_array_equal(
                lease["b"][0][0], arrays["b"][:1024]
            )
            lease.release()


def test_next_batch_can_read_while_previous_batch_is_incomplete(
    tmp_path, monkeypatch
):
    path = tmp_path / "data.h5"
    arrays, _ = write_input(path)
    with Hdf5Source(
        path, {"a": "a"}, upload_workers=2, io_queue_depth=3
    ) as source:
        base = source._layouts["/a"].offset
        next_started = Event()
        original = os.preadv

        def read(fd, buffers, offset):
            if offset == base:
                assert next_started.wait(3)
            if offset == base + 4096:
                next_started.set()
            return original(fd, buffers, offset)

        monkeypatch.setattr(os, "preadv", read)
        requests = [(("a",), [start], 1024) for start in (0, 1024, 2048)]
        with source.prefetch_batches(requests) as reader:
            retained = []
            for index, lease in enumerate(reader):
                lease.wait()
                retained.append(lease["a"][0])
                np.testing.assert_array_equal(
                    retained[-1][0],
                    arrays["a"][index * 1024 : (index + 1) * 1024],
                )
                assert len(reader.active) <= source.io_queue_depth
                lease.release()
            np.testing.assert_array_equal(retained[0][0], arrays["a"][:1024])
            assert not reader.active


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("workers", [1, 4])
def test_posix_gpu_uses_common_device_slots_and_preserves_cpu_results(
    tmp_path, depth, workers
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from cuphoton.xbeam.adapters.hdf5_batches import DeviceBatches

    path = tmp_path / "data.h5"
    arrays, mask = write_input(path)
    spec = CorrelationSpec(
        (("a", "b"),),
        64,
        hop_samples=37,
        windows_per_result=4,
        batch_windows=4,
        result_groups_per_tile=3,
        output="both",
    )
    expected = list(
        correlate_series(
            ArraySource(
                arrays,
                sample_rate_hz=1000.0,
                origin_seconds=5.0,
                validity={"b": mask},
            ),
            spec,
        )
    )
    # Keep FFT rounding out of the transport comparison: the synchronous
    # GPU path must produce the same bits as the queued device reader.
    with Hdf5Source(
        path,
        {"a": "a", "b": "b"},
        validity_paths={"b": "mask"},
    ) as source:
        gpu_expected = list(
            correlate_series(
                source, replace(spec, backend="cupy", prefetch_batches=0)
            )
        )
    with Hdf5Source(
        path,
        {"a": "a", "b": "b"},
        validity_paths={"b": "mask"},
        io_queue_depth=depth,
        upload_workers=workers,
        read_buffer_bytes=4096,
    ) as source:
        iterator = correlate_series(
            source, replace(spec, backend="cupy", result_location="device")
        )
        first = next(iterator)
        reader = next(iter(source._readers))
        assert isinstance(reader, DeviceBatches)
        assert len(reader.slots) == depth
        assert len(reader.streams) == min(workers, depth)
        assert all(slot.host_buffers for slot in reader.slots)
        actual = [first, *iterator]
    for x, y, gpu in zip(actual, expected, gpu_expected, strict=True):
        host = x.to_host()
        np.testing.assert_array_equal(host.spectrum, gpu.spectrum)
        np.testing.assert_array_equal(host.lag, gpu.lag)
        np.testing.assert_allclose(
            host.spectrum, y.spectrum, rtol=3e-6, atol=2e-5
        )
        np.testing.assert_allclose(host.lag, y.lag, rtol=3e-6, atol=2e-5)
        np.testing.assert_array_equal(
            host.lag_valid_samples, y.lag_valid_samples
        )


@pytest.mark.gpu
def test_posix_gpu_read_failure_is_not_published(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = tmp_path / "data.h5"
    write_input(path)
    with Hdf5Source(path, {"a": "a"}, read_buffer_bytes=4096) as source:
        original = os.preadv
        target = source._layouts["/a"].offset + 4096

        def fail(fd, buffers, offset):
            if offset >= target:
                raise OSError("synthetic POSIX failure")
            return original(fd, buffers, offset)

        monkeypatch.setattr(os, "preadv", fail)
        iterator = correlate_series(
            source,
            CorrelationSpec(
                (("a", "a"),), 1024, backend="cupy", result_location="device"
            ),
        )
        next(iterator)
        with pytest.raises(OSError, match="synthetic POSIX failure"):
            next(iterator)
        iterator.close()
        assert not source._readers


def test_dataset_descriptions_have_independent_positions_and_owned_lifetime(
    tmp_path,
):
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("independent file descriptions require procfs")
    path = tmp_path / "data.h5"
    arrays, _ = write_input(path)
    with Hdf5Source(path, {"a": "a", "b": "b"}) as source:
        reader = source._posix_reader
        first = reader._stream_fd("/a")
        second = reader._stream_fd("/b")
        assert first != second != reader.fd
        positions = [
            os.lseek(fd, 0, os.SEEK_CUR) for fd in (second, reader.fd)
        ]
        os.lseek(first, 123, os.SEEK_SET)
        assert positions == [
            os.lseek(fd, 0, os.SEEK_CUR) for fd in (second, reader.fd)
        ]
        replacement = tmp_path / "replacement.h5"
        write_input(replacement, "i2")
        os.replace(replacement, path)
        with source.prefetch_batches([(("a", "b"), [0], 1024)]) as batches:
            lease = next(batches)
            lease.wait()
            for name in ("a", "b"):
                np.testing.assert_array_equal(
                    lease[name][0][0], arrays[name][:1024]
                )
            lease.release()
        assert reader._stream_fd("/a") == first
        descriptors = [reader.fd, *reader._stream_fds.values()]
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_missing_procfs_keeps_owned_descriptor_fallback(
    tmp_path, monkeypatch
):
    import errno

    path = tmp_path / "data.h5"
    arrays, _ = write_input(path)
    original = os.open

    def no_proc(path, *args, **kwargs):
        if str(path).startswith("/proc/self/fd/"):
            raise FileNotFoundError(errno.ENOENT, "procfs unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_proc)
    with Hdf5Source(path, {"a": "a"}) as source:
        with source.prefetch_batches([(("a",), [0], 1024)]) as batches:
            lease = next(batches)
            lease.wait()
            np.testing.assert_array_equal(
                lease["a"][0][0], arrays["a"][:1024]
            )
            lease.release()
        descriptors = list(source._posix_reader._stream_fds.values())
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_read_blocks_preserve_tiles_and_bound_every_dataset():
    from cuphoton.xbeam.adapters.hdf5_batches import read_blocks
    from cuphoton.xbeam.adapters.hdf5_layout import Extent, plan_batch

    names = {"a": ("/a", None), "b": ("/b", "/mask")}
    layouts = {
        "/a": Extent(4096, np.dtype("f4"), 10000),
        "/b": Extent(8192, np.dtype("f8"), 10000),
        "/mask": Extent(12288, np.dtype("bool"), 10000),
    }
    requests = [(("a", "b"), [i, i + 64], 64) for i in range(0, 768, 128)]
    blocks = list(read_blocks(requests, names, layouts, 4096, 4096))
    assert len(blocks) < len(requests)
    restored = []
    for block in blocks:
        plans = plan_batch(block.request, names, layouts, alignment=4096)
        assert all(needed <= 4096 for _, needed in plans.values())
        signals, starts, size = block.request
        head = 0
        for rows in block.rows:
            restored.append((signals, starts[head : head + rows], size))
            head += rows
        assert head == len(starts)
    assert restored == requests


@pytest.mark.parametrize(
    "starts",
    [
        [[0], [128]],  # Unrequested gap.
        [[128], [0]],  # Reverse order.
        [[0, 128], [192]],  # Noncontiguous request.
        [[0, 32, 80], [112]],  # Irregular overlap needs per-tile gathers.
        [[0, 32], [80, 112]],  # Changing hop at the tile boundary.
    ],
)
def test_read_blocks_do_not_bridge_gaps_or_reorder(starts):
    from cuphoton.xbeam.adapters.hdf5_batches import read_blocks
    from cuphoton.xbeam.adapters.hdf5_layout import Extent

    names = {"a": ("a", None)}
    layouts = {"a": Extent(0, np.dtype("f4"), 1000)}
    requests = [(("a",), row, 64) for row in starts]
    blocks = list(read_blocks(requests, names, layouts, 4096, 1))
    assert [block.request for block in blocks] == requests
    assert all(len(block.rows) == 1 for block in blocks)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["posix", "cufile"])
@pytest.mark.parametrize("depth", [2, 4])
def test_shared_blocks_preserve_gpu_bits_and_reduce_reads(
    tmp_path, backend, depth, monkeypatch
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from cuphoton.xbeam.adapters.hdf5_batches import DeviceBatches
    from tests.xbeam.test_hdf5_cufile import make_input

    path = tmp_path / "aligned.h5"
    make_input(path)
    spec = CorrelationSpec(
        (("a", "b"),),
        64,
        windows_per_result=4,
        batch_windows=4,
        result_groups_per_tile=1,
        output="both",
        backend="cupy",
    )
    calls = []
    original = DeviceBatches.read_request

    def count(self, block):
        calls.append(block.rows)
        return original(self, block)

    monkeypatch.setattr(DeviceBatches, "read_request", count)
    with Hdf5Source(
        path,
        {"a": "a", "b": "b"},
        io_backend=backend,
        read_buffer_bytes=4096,
        io_queue_depth=depth,
    ) as source:
        expected = [item.to_host() for item in correlate_series(source, spec)]
    small = len(calls)
    calls.clear()
    with Hdf5Source(
        path,
        {"a": "a", "b": "b"},
        io_backend=backend,
        read_buffer_bytes=16384,
        io_queue_depth=depth,
    ) as source:
        actual = [item.to_host() for item in correlate_series(source, spec)]
    assert len(calls) < small
    assert any(len(rows) > 1 for rows in calls)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left.spectrum, right.spectrum)
        np.testing.assert_array_equal(left.lag, right.lag)
        np.testing.assert_array_equal(
            left.simultaneous_valid_samples, right.simultaneous_valid_samples
        )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["posix", "cufile"])
def test_storage_block_reuse_waits_for_all_consumer_streams(
    tmp_path, backend
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import h5py

    from tests.xbeam.test_hdf5_cufile import make_input

    path = tmp_path / "aligned.h5"
    make_input(path)
    with h5py.File(path) as file:
        expected = file["a"][:]
    requests = [(("a", "b"), [start], 64) for start in range(0, 4096, 64)]
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    actual = []
    with Hdf5Source(
        path,
        {"a": "a", "b": "b"},
        io_backend=backend,
        read_buffer_bytes=4096,
        io_queue_depth=2,
    ) as source:
        with source.prefetch_device_batches(requests) as reader:
            for index, lease in enumerate(reader):
                with streams[index % len(streams)]:
                    lease.wait()
                    actual.append(cp.array(lease["a"][0]))
                    lease.release()
            if backend == "cufile":
                # GDS does not allocate host staging buffers or transfer
                # its input through the application's POSIX copy path.
                assert all(not slot.host_buffers for slot in reader.slots)
    for stream in streams:
        stream.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(cp.concatenate(actual)).reshape(-1), expected
    )


def test_unaligned_storage_blocks_fit_after_transport_rounding():
    from cuphoton.xbeam.adapters.hdf5_batches import read_blocks
    from cuphoton.xbeam.adapters.hdf5_layout import Extent, plan_batch

    names = {"a": ("/a", None), "b": ("/b", "/mask")}
    layouts = {
        "/a": Extent(4160, np.dtype("f4"), 10000),
        "/b": Extent(8320, np.dtype("f4"), 10000),
        "/mask": Extent(12300, np.dtype("bool"), 10000),
    }
    requests = [
        (("a", "b"), [start], 1024) for start in (13, 1037, 2061, 3085)
    ]
    blocks = list(read_blocks(requests, names, layouts, 8192, 4096))
    # Two adjacent windows have only 8192 raw bytes, but their unaligned
    # envelope needs 12288 bytes; each individual request fits 8192.
    assert [block.request for block in blocks] == requests
    for block in blocks:
        plans = plan_batch(block.request, names, layouts, alignment=4096)
        assert all(needed <= 8192 for _, needed in plans.values())
        assert plans["/a"][1] == 8192
