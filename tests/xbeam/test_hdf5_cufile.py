# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous cuFile reads, slot retirement and aligned HDF5 decoding."""

import os
from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series
from cuphoton.xbeam.adapters.hdf5_cufile import (
    CuFileBatches,
    Extent,
    segments,
)


def test_extent_segments_preserve_windows_and_bound_padding():
    info = Extent(8192, np.dtype("f8"), 4096)
    plan, needed = segments([0, 8, 11, 100, 108], 8, info)
    assert len(plan) == 2 and needed == 8192
    assert [p[-1] for p in plan] == [0, 3]
    assert all(p[0] % 4096 == p[1] % 4096 == p[2] % 4096 == 0 for p in plan)
    with pytest.raises(ValueError):
        segments([4095], 8, info)


def test_all_streams_are_drained_and_released_after_one_fails():
    calls = []

    def sync(index):
        calls.append(("sync", index))
        if index == 0:
            raise OSError("failed I/O stream")

    reader = CuFileBatches("unused", {}, {}, (), 4096)
    reader.device = 0
    reader.xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda _: nullcontext())
    )
    reader.streams = [
        SimpleNamespace(ptr=i, synchronize=lambda i=i: sync(i))
        for i in range(2)
    ]
    reader.session = SimpleNamespace(
        __exit__=lambda *_: calls.append(("session", None))
    )
    with pytest.raises(OSError, match="failed I/O stream"):
        reader.close()
    assert calls == [
        ("sync", 0),
        ("sync", 1),
        ("session", None),
    ]
    assert not reader.streams and not reader.slots and not reader.pending


def make_input(path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(44)
    with h5py.File(
        path, "w", alignment_threshold=1, alignment_interval=4096
    ) as f:
        for key in ["a", "b"]:
            ds = f.create_dataset(
                key, data=rng.normal(size=4096) + 1j * rng.normal(size=4096)
            )
            ds.attrs.update(sample_rate_hz=1000.0, origin_seconds=12.0)
        f.create_dataset("mask", data=rng.random(4096) > 0.1)


@pytest.mark.parametrize("linked", ["signal", "validity"])
def test_external_link_is_posix_only_and_rejected_before_cuda(
    tmp_path, monkeypatch, linked
):
    h5py = pytest.importorskip("h5py")
    path, target = tmp_path / "root.h5", tmp_path / "target.h5"
    make_input(path)
    make_input(target)
    with h5py.File(target, "r+") as f:
        f["a"][:] *= 2
        f["mask"][:] = ~f["mask"][:]
    name = "a" if linked == "signal" else "mask"
    with h5py.File(path, "r+") as f:
        f["external"] = h5py.ExternalLink(target.name, "/" + name)
    config = dict(
        dataset_paths={"a": "external" if linked == "signal" else "a"},
        validity_paths={"a": "external" if linked == "validity" else "mask"},
    )

    def unexpected_cuda(*args):
        pytest.fail("external links must be rejected before CUDA resources")

    monkeypatch.setattr(CuFileBatches, "__enter__", unexpected_cuda)
    with pytest.raises(ValueError, match="owned by the opened HDF5 file"):
        Hdf5Source(path, **config, io_backend="cufile")
    with (
        Hdf5Source(path, **config) as source,
        h5py.File(path) as root,
        h5py.File(target) as other,
    ):
        values, mask = source.read("a", 0, 128)
        expected_values = (other if linked == "signal" else root)["a"][:128]
        expected_mask = (other if linked == "validity" else root)["mask"][
            :128
        ]
        np.testing.assert_array_equal(values, expected_values)
        np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize("link", ["hard", "soft"])
def test_same_file_links_have_valid_cufile_extents(tmp_path, link):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "root.h5"
    make_input(path)
    with h5py.File(path, "r+") as f:
        for name in ["a", "mask"]:
            f[name + "_link"] = (
                f[name] if link == "hard" else h5py.SoftLink("/" + name)
            )
    with Hdf5Source(
        path,
        {"a": "a_link"},
        validity_paths={"a": "mask_link"},
        io_backend="cufile",
    ) as source:
        for dataset in [
            *source._datasets.values(),
            *source._validity.values(),
        ]:
            assert (
                source._layouts[dataset.name].offset
                == dataset.id.get_offset()
            )


def test_source_close_drains_other_readers_and_file_after_failure(tmp_path):
    path = tmp_path / "root.h5"
    make_input(path)
    source = Hdf5Source(path, {"a": "a"})
    calls = []
    first_failure = OSError("first reader failed")

    def close(index):
        calls.append(index)
        raise first_failure if index == 0 else RuntimeError("second failed")

    readers = [SimpleNamespace(close=lambda i=i: close(i)) for i in range(2)]
    source._readers = readers.copy()
    source._sync_reader = readers[1]
    with pytest.raises(OSError) as error:
        source.close()
    assert error.value is first_failure
    assert calls == [0, 1]
    assert not source._readers and source._sync_reader is None
    assert not source._file.id.valid
    source.close()


def test_cufile_rejects_non_descriptor_driver_before_cuda(
    tmp_path, monkeypatch
):
    path = tmp_path / "input.h5"
    make_input(path)

    def unexpected_cuda(*args):
        pytest.fail("file driver must be rejected before CUDA resources")

    monkeypatch.setattr(CuFileBatches, "__enter__", unexpected_cuda)
    with (
        BytesIO(path.read_bytes()) as file,
        pytest.raises(ValueError, match="HDF5 sec2 file driver"),
    ):
        Hdf5Source(file, {"a": "a"}, io_backend="cufile")


@pytest.mark.parametrize("replace_source", [False, True])
def test_cufile_checks_source_identity_before_reads_on_cpu(
    tmp_path, monkeypatch, replace_source
):
    from cuphoton.xbeam.runtime import transports

    class Session:
        def __enter__(self):
            return self

        def open(self, path):
            self.fd = os.open(path, os.O_RDONLY)
            return SimpleNamespace(fd=self.fd)

        def __exit__(self, *args):
            os.close(self.fd)

        def read(self, *args, **kwargs):
            pytest.fail("opening a source must not submit cuFile reads")

    path, replacement = tmp_path / "root.h5", tmp_path / "replacement.h5"
    path.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    stat = path.stat()
    monkeypatch.setattr(transports, "CuFileSession", Session)
    reader = CuFileBatches(
        path, {}, {}, (), 4096, source_identity=(stat.st_dev, stat.st_ino)
    )
    reader.xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda _: nullcontext())
    )
    reader.device = 0
    if replace_source:
        os.replace(replacement, path)
    try:
        expectation = (
            pytest.raises(OSError, match="source changed before cuFile")
            if replace_source
            else nullcontext()
        )
        with expectation:
            reader._open()
    finally:
        reader.close()


@pytest.mark.gpu
@pytest.mark.parametrize("prefetch", [False, True])
def test_cufile_rejects_path_replacement_before_reading(
    tmp_path, prefetch, monkeypatch
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    h5py = pytest.importorskip("h5py")
    from cuphoton.xbeam.cufile_io import CuFileSession

    def unexpected_read(*args, **kwargs):
        pytest.fail("replacement file must be rejected before reading")

    monkeypatch.setattr(CuFileSession, "read", unexpected_read)
    path, replacement = tmp_path / "root.h5", tmp_path / "replacement.h5"
    make_input(path)
    make_input(replacement)
    with h5py.File(replacement, "r+") as file:
        file["a"][:] *= 2
    with Hdf5Source(
        path, {"a": "a"}, io_backend="cufile", read_buffer_bytes=4096
    ) as source:
        os.replace(replacement, path)
        with pytest.raises(OSError, match="source changed before cuFile"):
            if prefetch:
                with source.prefetch_batches([(("a",), [0], 64)]) as reader:
                    next(reader).wait()
            else:
                source.read_windows("a", [0], 64)
        assert not source._readers


@pytest.mark.gpu
@pytest.mark.parametrize("hop", [37, 64, 113])
@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_native_async_matches_posix_and_keeps_retained_results(
    tmp_path, hop, depth, workers
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = tmp_path / "aligned.h5"
    make_input(path)
    config = dict(
        dataset_paths={"a": "a", "b": "b"}, validity_paths={"b": "mask"}
    )
    spec = CorrelationSpec(
        (("a", "b"), ("b", "b")),
        64,
        hop_samples=hop,
        windows_per_result=4,
        batch_windows=4,
        output="both",
        backend="cupy",
        result_groups_per_tile=3,
        result_location="device",
    )
    with Hdf5Source(path, **config) as source:
        expected = [r.to_host() for r in correlate_series(source, spec)]
    stream = cp.cuda.Stream(non_blocking=True)
    with Hdf5Source(
        path,
        **config,
        io_backend="cufile",
        read_buffer_bytes=1024**2,
        io_queue_depth=depth,
        upload_workers=workers,
    ) as source:
        with stream:
            iterator = correlate_series(source, spec)
            first = next(iterator)
            reader = next(iter(source._readers))
            assert reader.worker_count == min(workers, depth)
            assert len(reader.streams) == min(workers, depth)
            assert len(reader.slots) == depth
            assert len({slot.stream.ptr for slot in reader.slots}) == min(
                workers, depth
            )
            actual = [first, *iterator]
    for x, y in zip(actual, expected, strict=True):
        host = x.to_host()
        np.testing.assert_allclose(host.spectrum, y.spectrum, atol=2e-11)
        np.testing.assert_allclose(host.lag, y.lag, atol=2e-12)
        np.testing.assert_array_equal(
            host.lag_valid_samples, y.lag_valid_samples
        )
        np.testing.assert_array_equal(
            host.simultaneous_valid_samples, y.simultaneous_valid_samples
        )


@pytest.mark.gpu
def test_native_sync_early_close_and_capacity_error(tmp_path):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = tmp_path / "aligned.h5"
    make_input(path)
    config = dict(
        dataset_paths={"a": "a", "b": "b"},
        io_backend="cufile",
        read_buffer_bytes=1024**2,
    )
    spec = CorrelationSpec(
        (("a", "b"),), 64, windows_per_result=4, output="both", backend="cupy"
    )
    with Hdf5Source(path, **config) as source:
        iterator = correlate_series(
            source, replace(spec, result_location="device")
        )
        first = next(iterator)
        iterator.close()
        assert not source._readers
        sync = next(
            correlate_series(source, replace(spec, prefetch_batches=0))
        ).to_host()
    np.testing.assert_allclose(
        first.to_host().spectrum, sync.spectrum, atol=2e-11
    )
    with Hdf5Source(
        path,
        dataset_paths={"a": "a", "b": "b"},
        io_backend="cufile",
        read_buffer_bytes=4096,
    ) as source:
        with pytest.raises(ValueError, match="exceeds read_buffer_bytes"):
            list(correlate_series(source, replace(spec, window_samples=1024)))

    # A fresh 4 KiB reader must work after early close and capacity failure,
    # even when it reuses the same CUDA context.
    reopened = tmp_path / "reopened.h5"
    make_input(reopened)
    with Hdf5Source(
        reopened,
        dataset_paths={"a": "a", "b": "b"},
        io_backend="cufile",
        read_buffer_bytes=4096,
    ) as source:
        actual = [item.to_host() for item in correlate_series(source, spec)]
    with Hdf5Source(reopened, dataset_paths={"a": "a", "b": "b"}) as source:
        expected = [item.to_host() for item in correlate_series(source, spec)]
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.spectrum, right.spectrum)
        np.testing.assert_array_equal(left.lag, right.lag)


@pytest.mark.gpu
@pytest.mark.parametrize("received", [0, -5])
def test_short_native_read_is_not_published(tmp_path, monkeypatch, received):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from cuphoton.xbeam.cufile_io import CuFileSession

    path = tmp_path / "aligned.h5"
    make_input(path)
    original = CuFileSession.read

    def short(self, *args, **kwargs):
        original(self, *args, **kwargs)
        return received

    monkeypatch.setattr(CuFileSession, "read", short)
    with (
        pytest.raises(OSError, match="short/failed read"),
        Hdf5Source(path, {"a": "a"}, io_backend="cufile") as source,
        source.prefetch_batches([(("a",), [0], 64)]) as batches,
    ):
        next(batches).wait()
