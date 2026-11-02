# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tiling preserves reduction boundaries and output ownership."""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam import (
    ArraySource,
    CorrelationSpec,
    Hdf5Source,
    correlate_series,
)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("mode", ["linear", "circular"])
@pytest.mark.parametrize("hop", [7, 17, 39])
def test_tiled_masked_results_match_one_group_tiles(dtype, mode, hop):
    rng = np.random.default_rng(63)
    arrays = {
        key: (
            rng.normal(size=801) + 1j * rng.normal(size=801)
            if np.issubdtype(dtype, np.complexfloating)
            else rng.normal(size=801)
        ).astype(dtype)
        for key in ["a", "b"]
    }
    mask = rng.random(801) > 0.3
    source = ArraySource(arrays, sample_rate_hz=2.0, validity={"b": mask})
    spec = CorrelationSpec(
        (("a", "b"), ("a", "a"), ("b", "b")),
        17,
        hop_samples=hop,
        windows_per_result=5,
        batch_windows=3,
        output="both",
        lag_mode=mode,
    )
    expected = list(correlate_series(source, spec))
    for tile in [2, 4, 99]:
        actual = []
        for shard in range(3):
            actual.extend(
                correlate_series(
                    source,
                    replace(
                        spec,
                        result_groups_per_tile=tile,
                        shard_count=3,
                        shard_index=shard,
                    ),
                )
            )
        for x, y in zip(actual, expected, strict=True):
            for f in fields(x):
                xv, yv = getattr(x, f.name), getattr(y, f.name)
                if f.name in {"spectrum", "lag"}:
                    # FFT implementations can choose different vectorized
                    # arithmetic for different batch sizes (notably ARM64).
                    # The window groups are unchanged, but their floating
                    # products need not be bitwise identical.
                    tolerance = 16 * np.finfo(xv.real.dtype).eps
                    np.testing.assert_allclose(
                        xv, yv, rtol=tolerance, atol=tolerance
                    )
                elif isinstance(xv, np.ndarray):
                    np.testing.assert_array_equal(xv, yv)
                else:
                    assert xv == yv


def test_host_result_storage_survives_tile_reuse():
    source = ArraySource({"a": np.arange(256.0)}, sample_rate_hz=1.0)
    iterator = correlate_series(
        source,
        CorrelationSpec(
            (("a", "a"),), 8, result_groups_per_tile=3, output="both"
        ),
    )
    first = next(iterator)
    saved = first.spectrum.copy()
    list(iterator)
    np.testing.assert_array_equal(first.spectrum, saved)
    assert first.to_host() is first


@pytest.mark.parametrize("mode", ["linear", "circular"])
@pytest.mark.parametrize("masked", [False, True])
def test_device_coverage_uploads_vector_once_not_full_tiles(
    monkeypatch, mode, masked
):
    from cuphoton.xbeam.core import series

    class DeviceArray(np.ndarray):
        pass

    class Backend:
        __name__ = "numpy"

        def __init__(self):
            self.uploads = []
            self.cuda = SimpleNamespace(
                Event=lambda: SimpleNamespace(record=lambda _: None),
                Device=lambda: SimpleNamespace(id=0),
                get_current_stream=lambda: None,
            )

        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, value, *args, **kwargs):
            array = np.asarray(value, *args, **kwargs)
            if not isinstance(value, DeviceArray):
                self.uploads.append((array.shape, array.dtype))
            return array.view(DeviceArray)

    values = np.arange(16 * 12, dtype=np.float64)
    source = ArraySource(
        {"a": values, "b": values[::-1]},
        sample_rate_hz=1.0,
        validity={"b": values % 3 != 0} if masked else None,
    )
    spec = CorrelationSpec(
        (("a", "a"), ("a", "b")),
        16,
        windows_per_result=2,
        result_groups_per_tile=2,
        output="both",
        lag_mode=mode,
    )
    expected = list(correlate_series(source, spec))
    backend = Backend()
    monkeypatch.setattr(series, "get_backend", lambda _: backend)
    actual = list(
        series._correlate_series(
            source, replace(spec, backend="cupy", result_location="device")
        )
    )
    assert len(actual) == 6
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            result.lag_valid_samples, reference.lag_valid_samples
        )
        np.testing.assert_array_equal(
            result.simultaneous_valid_samples,
            reference.simultaneous_valid_samples,
        )
        np.testing.assert_array_equal(result.lag, reference.lag)
    lag_count = len(expected[0].lag_samples)
    coverage_uploads = [
        shape
        for shape, dtype in backend.uploads
        if dtype == np.dtype("int64")
    ]
    # One index vector and one coverage vector, independent of tile count.
    assert coverage_uploads.count((lag_count,)) == 2
    assert all(len(shape) < 3 for shape in coverage_uploads)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"result_groups_per_tile": 0},
        {"result_groups_per_tile": True},
        {"result_location": "other"},
        {"result_location": "device"},
    ],
)
def test_bad_tile_or_result_location(kwargs):
    with pytest.raises(ValueError):
        CorrelationSpec((("a", "a"),), 8, **kwargs)


def test_source_backend_requirement_is_checked_before_reading():
    source = ArraySource({"a": np.ones(16)}, sample_rate_hz=1.0)
    source.required_backend = "cupy"
    with pytest.raises(ValueError, match="source requires backend=cupy"):
        list(correlate_series(source, CorrelationSpec((("a", "a"),), 8)))


def test_cufile_layout_rejection_is_cpu_only(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        for key in ["contiguous", "compressed"]:
            ds = f.create_dataset(
                key,
                data=np.arange(4096.0),
                compression="gzip" if key == "compressed" else None,
            )
            ds.attrs.update(sample_rate_hz=1.0, origin_seconds=0.0)
    for key in ["contiguous", "compressed"]:
        with pytest.raises(ValueError, match="cuFile"):
            Hdf5Source(path, {"a": key}, io_backend="cufile")


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_device_results_cross_stream_retention_and_serialization(
    tmp_path, mode
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from cuphoton.xbeam.adapters.hdf5 import write_series_hdf5

    rng = np.random.default_rng(9)
    source = ArraySource(
        {"a": rng.normal(size=307), "b": rng.normal(size=307)},
        sample_rate_hz=1.0,
    )
    spec = CorrelationSpec(
        (("a", "b"),),
        17,
        windows_per_result=3,
        batch_windows=2,
        output="both",
        lag_mode=mode,
    )
    expected = list(correlate_series(source, spec))
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    with producer:
        iterator = correlate_series(
            source,
            replace(
                spec,
                backend="cupy",
                result_location="device",
                result_groups_per_tile=4,
            ),
        )
        first = next(iterator)
    with consumer:
        first.wait(consumer)
        saved = first.spectrum.copy()
        actual = [
            first,
            *iterator,
        ]  # Producer must retain its original stream.
    for x, y in zip(actual, expected, strict=True):
        assert isinstance(x.spectrum, cp.ndarray) and isinstance(
            x.valid, cp.ndarray
        )
        host = x.to_host()
        np.testing.assert_allclose(host.spectrum, y.spectrum, atol=2e-12)
        np.testing.assert_allclose(host.lag, y.lag, atol=2e-13)
        np.testing.assert_array_equal(
            host.lag_valid_samples, y.lag_valid_samples
        )
    consumer.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(first.spectrum), cp.asnumpy(saved)
    )
    write_series_hdf5(tmp_path / "device.h5", iter(actual))
