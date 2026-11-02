# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Raw extent fidelity, concurrent reads and descriptor ownership."""

import os
from dataclasses import fields
from io import BytesIO
from threading import Event, Lock, Thread

import numpy as np
import pytest

from cuphoton.xbeam import (
    ArraySource,
    CorrelationSpec,
    Hdf5Source,
    correlate_series,
)
from cuphoton.xbeam.adapters.hdf5_posix import Extent, PosixReader


def write_input(path, dtype="f4", **options):
    import h5py

    rng = np.random.default_rng(71)
    values = rng.normal(size=4096)
    if np.dtype(dtype).kind == "c":
        values = values + 1j * rng.normal(size=4096)
    arrays = {
        "a": values.astype(dtype),
        "b": np.roll(values, 3).astype(dtype),
    }
    mask = rng.random(4096) > 0.1
    with h5py.File(path, "w") as f:
        for key, data in arrays.items():
            ds = f.create_dataset(key, data=data, **options)
            ds.attrs.update(sample_rate_hz=1000.0, origin_seconds=5.0)
        f.create_dataset("mask", data=mask, **options)
    return arrays, mask


@pytest.mark.parametrize("dtype", ["f4", "f8", "c8", "c16", "i2", "?"])
@pytest.mark.parametrize("hop", [13, 17, 29])
def test_raw_posix_matches_arrays_without_hdf5_data_calls(
    tmp_path, monkeypatch, dtype, hop
):
    import h5py

    path = tmp_path / "input.h5"
    arrays, mask = write_input(path, dtype)
    spec = CorrelationSpec(
        (("a", "b"),),
        17,
        hop_samples=hop,
        windows_per_result=3,
        batch_windows=2,
        result_groups_per_tile=2,
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
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths={"b": "mask"}
    ) as source:
        assert source._posix_reader is not None
        assert source.upload_worker_limit == 4
        fd = source._posix_reader.fd

        def no_hdf5_reads(*args):
            pytest.fail("eligible data must be read outside HDF5's lock")

        monkeypatch.setattr(h5py.Dataset, "__getitem__", no_hdf5_reads)
        actual = list(correlate_series(source, spec))
    for x, y in zip(actual, expected, strict=True):
        for field in fields(x):
            first, second = getattr(x, field.name), getattr(y, field.name)
            if isinstance(first, np.ndarray):
                np.testing.assert_array_equal(first, second)
            else:
                assert first == second
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize(
    "options", [{"compression": "gzip"}, {"chunks": (128,)}]
)
def test_decoded_layout_retains_hdf5_fallback(tmp_path, options):
    path = tmp_path / "input.h5"
    arrays, mask = write_input(path, **options)
    with Hdf5Source(path, {"a": "a"}, validity_paths={"a": "mask"}) as source:
        assert source._posix_reader is None
        assert source.upload_worker_limit == 1
        values, actual_mask = source.read("a", 7, 51)
        np.testing.assert_array_equal(values, arrays["a"][7:51])
        np.testing.assert_array_equal(actual_mask, mask[7:51])


@pytest.mark.parametrize("linked", ["signal", "validity"])
def test_external_and_local_datasets_with_same_name_stay_distinct(
    tmp_path, linked
):
    import h5py

    path, target = tmp_path / "input.h5", tmp_path / "other.h5"
    arrays, mask = write_input(path)
    write_input(target)
    with h5py.File(target, "r+") as other:
        other["a"][:] *= 2
        other["mask"][:] = ~mask
    with h5py.File(path, "r+") as root:
        root["external"] = h5py.ExternalLink(
            target.name, "/a" if linked == "signal" else "/mask"
        )
    datasets = (
        {"a": "external", "b": "a"}
        if linked == "signal"
        else {"a": "a", "b": "b"}
    )
    validity = {
        "a": "external" if linked == "validity" else "mask",
        "b": "mask",
    }
    expected_arrays = (
        {"a": 2 * arrays["a"], "b": arrays["a"]}
        if linked == "signal"
        else arrays
    )
    expected_masks = {
        "a": ~mask if linked == "validity" else mask,
        "b": mask,
    }
    spec = CorrelationSpec(
        (("a", "b"),), 64, windows_per_result=4, output="both"
    )
    expected = list(
        correlate_series(
            ArraySource(
                expected_arrays,
                sample_rate_hz=1000.0,
                origin_seconds=5.0,
                validity=expected_masks,
            ),
            spec,
        )
    )
    with Hdf5Source(path, datasets, validity_paths=validity) as source:
        for name in datasets:
            values, actual_mask = source.read(name, 7, 51)
            np.testing.assert_array_equal(values, expected_arrays[name][7:51])
            np.testing.assert_array_equal(
                actual_mask, expected_masks[name][7:51]
            )
        actual = list(correlate_series(source, spec))
    for first, second in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(first.spectrum, second.spectrum)
        np.testing.assert_array_equal(first.lag, second.lag)
        np.testing.assert_array_equal(
            first.lag_valid_samples, second.lag_valid_samples
        )


def test_raw_prefetch_preserves_custom_window_reads(tmp_path):
    class ScaledSource(Hdf5Source):
        def read_windows(self, signal_id, starts, size):
            values, mask = super().read_windows(signal_id, starts, size)
            return values * 2, mask

    path = tmp_path / "input.h5"
    arrays, _ = write_input(path)
    spec = CorrelationSpec((("a", "b"),), 64, windows_per_result=4)
    expected = list(
        correlate_series(
            ArraySource(
                {name: values * 2 for name, values in arrays.items()},
                sample_rate_hz=1000.0,
                origin_seconds=5.0,
            ),
            spec,
        )
    )
    with ScaledSource(path, {"a": "a", "b": "b"}) as source:
        actual = list(correlate_series(source, spec))
    for first, second in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(first.spectrum, second.spectrum)


def test_unallocated_fill_and_non_native_dtype_use_decoder(tmp_path):
    import h5py

    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as f:
        for name, options in [
            ("fill", dict(shape=(64,), dtype="f4", fillvalue=7)),
            ("endian", dict(data=np.arange(64, dtype=">f4"))),
        ]:
            ds = f.create_dataset(name, **options)
            ds.attrs.update(sample_rate_hz=1.0, origin_seconds=0.0)
    for name, expected in [
        ("fill", np.full(64, 7)),
        ("endian", np.arange(64)),
    ]:
        with Hdf5Source(path, {"a": name}) as source:
            assert source._posix_reader is None
            np.testing.assert_array_equal(
                source.read("a", 0, 64)[0], expected
            )


def test_raw_reads_remain_on_original_inode_after_path_replacement(tmp_path):
    path, replacement = tmp_path / "input.h5", tmp_path / "other.h5"
    arrays, _ = write_input(path)
    write_input(replacement, "i2")
    with Hdf5Source(path, {"a": "a"}) as source:
        os.replace(replacement, path)
        np.testing.assert_array_equal(
            source.read("a", 0, 64)[0], arrays["a"][:64]
        )


def test_raw_chunks_run_concurrently_with_independent_offsets(
    tmp_path, monkeypatch
):
    path = tmp_path / "input.bin"
    data = np.arange(4096, dtype="i4")
    path.write_bytes(data.tobytes())
    first_started, second_started = Event(), Event()
    lock = Lock()
    original = os.preadv
    active = maximum = 0

    def read(fd, buffers, offset):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if offset == 0:
                first_started.set()
                assert second_started.wait(3)
            elif offset == 4096:
                second_started.set()
                assert first_started.wait(3)
            return original(fd, buffers, offset)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(os, "preadv", read)
    fd = os.open(path, os.O_RDONLY)
    reader = PosixReader(fd, workers=2, chunk_bytes=4096)
    os.close(fd)  # Reader owns a duplicated descriptor.
    try:
        actual = reader.read(Extent(0, data.dtype, len(data)), 0, len(data))
        np.testing.assert_array_equal(actual, data)
        assert maximum == 2
        previous = actual.copy()
        reader.read(Extent(0, data.dtype, len(data)), 1, 1025)
        np.testing.assert_array_equal(actual, previous)
    finally:
        reader.close()
        reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read(Extent(0, data.dtype, len(data)), 0, 1)


def test_short_reads_are_completed_and_eof_is_reported(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    data = np.arange(4096, dtype="f4")
    path.write_bytes(data.tobytes())
    original = os.preadv

    def short(fd, buffers, offset):
        return original(fd, [buffers[0][:512]], offset)

    monkeypatch.setattr(os, "preadv", short)
    fd = os.open(path, os.O_RDONLY)
    reader = PosixReader(fd, workers=3, chunk_bytes=4096)
    os.close(fd)
    try:
        np.testing.assert_array_equal(
            reader.read(Extent(0, data.dtype, len(data)), 0, len(data)), data
        )
        with pytest.raises(OSError, match="short POSIX read"):
            reader.read(
                Extent(0, data.dtype, len(data) + 1), 0, len(data) + 1
            )
    finally:
        reader.close()


def test_read_error_drains_other_writers_before_return(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    data = np.arange(4096, dtype="f4")
    path.write_bytes(data.tobytes())
    writing, release, failed, finished = Event(), Event(), Event(), Event()
    original = os.preadv

    def read(fd, buffers, offset):
        if offset == 0:
            assert writing.wait(3)
            failed.set()
            raise OSError("read failed")
        writing.set()
        assert release.wait(3)
        return original(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", read)
    fd = os.open(path, os.O_RDONLY)
    reader = PosixReader(fd, workers=2, chunk_bytes=4096)
    os.close(fd)
    errors = []

    def consume():
        try:
            reader.read(Extent(0, data.dtype, len(data)), 0, len(data))
        except OSError as exc:
            errors.append(str(exc))
        finally:
            finished.set()

    thread = Thread(target=consume)
    thread.start()
    try:
        assert failed.wait(3)
        assert not finished.wait(0.5)
    finally:
        release.set()
        thread.join(3)
        reader.close()
    assert finished.is_set() and not thread.is_alive()
    assert errors == ["read failed"]


def test_source_close_drains_raw_prefetch_before_closing_descriptor(
    tmp_path, monkeypatch
):
    path = tmp_path / "input.h5"
    write_input(path)
    source = Hdf5Source(path, {"a": "a"}, read_buffer_bytes=4096)
    reader = source._posix_reader
    offset = source._layouts["/a"].offset + 4096
    writing, release, closing, closed = Event(), Event(), Event(), Event()
    original = os.preadv

    def read(fd, buffers, where):
        if where == offset:
            writing.set()
            assert release.wait(3)
        return original(fd, buffers, where)

    monkeypatch.setattr(os, "preadv", read)
    iterator = correlate_series(source, CorrelationSpec((("a", "a"),), 1024))
    next(iterator)
    assert writing.wait(3)

    def close():
        closing.set()
        source.close()
        closed.set()

    thread = Thread(target=close)
    thread.start()
    try:
        assert closing.wait(3)
        assert not closed.is_set()
        os.fstat(reader.fd)
    finally:
        release.set()
        thread.join(3)
        iterator.close()
        source.close()
    assert closed.is_set() and not thread.is_alive()
    with pytest.raises(OSError):
        os.fstat(reader.fd)


def test_nonstandard_numeric_encoding_keeps_hdf5_conversion(tmp_path):
    import h5py

    path = tmp_path / "precision.h5"
    with h5py.File(path, "w") as f:
        encoded = h5py.h5t.STD_I16LE.copy()
        encoded.set_precision(12)
        dataset_id = h5py.h5d.create(
            f.id, b"a", encoded, h5py.h5s.create_simple((64,))
        )
        ds = h5py.Dataset(dataset_id)
        ds[:] = np.arange(64, dtype="i2") - 32
        ds.attrs.update(sample_rate_hz=1.0, origin_seconds=0.0)
    with Hdf5Source(path, {"a": "a"}) as source:
        assert source._posix_reader is None
        np.testing.assert_array_equal(
            source.read("a", 0, 64)[0], np.arange(64) - 32
        )


def test_file_object_source_keeps_hdf5_decoding_without_raw_descriptor():
    with BytesIO() as buffer:
        arrays, mask = write_input(buffer)
        with Hdf5Source(
            buffer, {"a": "a", "b": "b"}, validity_paths={"b": "mask"}
        ) as source:
            assert source._file.driver == "fileobj"
            assert source._posix_reader is None
            assert not source._layouts
            for signal in ("a", "b"):
                values, valid = source.read_windows(signal, [3, 19], 8)
                np.testing.assert_array_equal(
                    values,
                    np.stack(
                        [
                            arrays[signal][3:11],
                            arrays[signal][19:27],
                        ]
                    ),
                )
                if signal == "a":
                    assert valid is None
                else:
                    np.testing.assert_array_equal(
                        valid,
                        np.stack(
                            [
                                mask[3:11],
                                mask[19:27],
                            ]
                        ),
                    )
