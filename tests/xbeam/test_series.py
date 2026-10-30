# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent sums validate conventions, masks, streaming, and sharding."""

import builtins
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam import ArraySource, CorrelationSpec, correlate_series
from cuphoton.xbeam.core.series import plan_series_shards


def test_missing_hdf5_dependency_names_the_shared_distribution(monkeypatch):
    from cuphoton.xbeam.adapters.hdf5 import _h5py

    original_import = builtins.__import__

    def without_h5py(name, *args, **kwargs):
        if name == "h5py":
            raise ModuleNotFoundError("h5py is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_h5py)
    with pytest.raises(
        ImportError, match="requires h5py; reinstall cuphoton"
    ):
        _h5py()


def direct_lags(a, b, lags, circular=False):
    n = len(a)
    return np.array(
        [
            sum(
                a[i] * np.conj(b[(i + k) % n])
                for i in range(n)
                if circular or 0 <= i + k < n
            )
            for k in lags
        ]
    )


@pytest.mark.parametrize("n", [1, 7, 8, 19])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_products_match_independent_sums(n, complex_input, mode):
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(2, n))
    if complex_input:
        a = a + 1j * rng.normal(size=n)
        b = b + 1j * rng.normal(size=n)
    source = ArraySource({"a": a, "b": b}, sample_rate_hz=5)
    spec = CorrelationSpec(
        (("a", "b"), ("a", "a")), n, output="both", lag_mode=mode
    )
    (result,) = correlate_series(source, spec)
    for i, (x, y) in enumerate([(a, b), (a, a)]):
        np.testing.assert_allclose(
            result.lag[i],
            direct_lags(x, y, result.lag_samples, mode == "circular"),
            atol=2e-13,
        )
        np.testing.assert_allclose(
            result.spectrum[i],
            np.fft.fft(x, spec.fft_samples)
            * np.conj(np.fft.fft(y, spec.fft_samples)),
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            result.lag_valid_samples[i],
            direct_lags(
                np.ones(n), np.ones(n), result.lag_samples, mode == "circular"
            ).real,
        )
    assert result.valid.all()
    assert result.fft_samples >= n


@pytest.mark.parametrize("shift", [-3, 0, 4])
@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_delayed_impulses_have_positive_lag(shift, mode):
    a, b = np.zeros((2, 17))
    a[6], b[6 + shift] = 1, 1
    source = ArraySource({"a": a, "b": b}, sample_rate_hz=10)
    (result,) = correlate_series(
        source,
        CorrelationSpec((("a", "b"),), 17, output="lag", lag_mode=mode),
    )
    assert result.lag_samples[np.argmax(abs(result.lag[0]))] == shift
    assert result.spectrum is None and result.frequencies_hz is None


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_precision_and_both_output_equivalence(dtype):
    a = np.arange(8).astype(dtype)
    source = ArraySource({"a": a}, sample_rate_hz=1)
    spec = CorrelationSpec((("a", "a"),), 8, output="both")
    (both,) = correlate_series(source, spec)
    (spectral,) = correlate_series(source, replace(spec, output="spectrum"))
    (lag,) = correlate_series(source, replace(spec, output="lag"))
    expected = (
        np.complex64 if dtype in [np.float32, np.complex64] else np.complex128
    )
    assert both.spectrum.dtype == expected
    assert both.lag.dtype == expected
    np.testing.assert_array_equal(both.spectrum, spectral.spectrum)
    np.testing.assert_array_equal(both.lag, lag.lag)
    (high,) = correlate_series(source, replace(spec, precision="float64"))
    assert high.spectrum.dtype == np.complex128


def test_spectrum_only_never_calls_inverse_fft(monkeypatch):
    a = np.arange(8, dtype=float)

    def forbidden(*args, **kwargs):
        raise AssertionError("spectrum-only allocated an inverse transform")

    monkeypatch.setattr(np.fft, "ifft", forbidden)
    assert (
        len(
            list(
                correlate_series(
                    ArraySource({"a": a}, sample_rate_hz=1),
                    CorrelationSpec((("a", "a"),), 8),
                )
            )
        )
        == 1
    )


def test_masked_samples_do_not_renormalize_raw_products():
    a = np.array([1, 2, np.nan, 4, 5.0])
    b = np.arange(5, dtype=float)
    ma, mb = (
        np.array([1, 1, 0, 1, 1], dtype=bool),
        np.array([1, 0, 1, 1, 1], dtype=bool),
    )
    source = ArraySource(
        {"a": a, "b": b, "empty": a},
        sample_rate_hz=1,
        validity={"a": ma, "b": mb, "empty": np.zeros(5, dtype=bool)},
    )
    spec = CorrelationSpec((("a", "b"), ("a", "empty")), 5, output="both")
    (result,) = correlate_series(source, spec)
    np.testing.assert_allclose(
        result.lag[0],
        direct_lags(
            np.where(ma, a, 0), np.where(mb, b, 0), result.lag_samples
        ),
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        result.lag_valid_samples[0],
        direct_lags(ma.astype(int), mb.astype(int), result.lag_samples),
    )
    np.testing.assert_array_equal(result.simultaneous_valid_samples, [3, 0])
    np.testing.assert_array_equal(result.valid, [True, False])
    np.testing.assert_array_equal(result.lag[1], 0)
    np.testing.assert_array_equal(result.lag_valid_samples[1], 0)


def test_overlap_batching_and_shards_preserve_global_windows():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(2, 133))
    source = ArraySource({"a": a, "b": b}, sample_rate_hz=1)
    spec = CorrelationSpec(
        (("a", "b"),),
        17,
        hop_samples=7,
        windows_per_result=3,
        output="both",
        batch_windows=1,
    )
    serial = list(correlate_series(source, spec))
    assert (
        len(serial) == 5
    )  # Two complete windows remain in an incomplete group.
    for i, result in enumerate(serial):
        starts = np.arange(i * 3, (i + 1) * 3) * 7
        np.testing.assert_array_equal(result.window_starts, starts)
        expected = np.mean(
            [
                direct_lags(a[s : s + 17], b[s : s + 17], result.lag_samples)
                for s in starts
            ],
            axis=0,
        )
        np.testing.assert_allclose(result.lag[0], expected, atol=1e-13)
    for batch in [2, 3, 99]:
        batched = list(
            correlate_series(source, replace(spec, batch_windows=batch))
        )
        for x, y in zip(serial, batched, strict=True):
            np.testing.assert_allclose(x.spectrum, y.spectrum, atol=1e-12)
    sharded = []
    for index in range(7):
        sharded.extend(
            correlate_series(
                source, replace(spec, shard_count=7, shard_index=index)
            )
        )
    assert [r.result_index for r in sharded] == list(range(5))
    for x, y in zip(serial, sharded, strict=True):
        np.testing.assert_array_equal(x.spectrum, y.spectrum)
    shards = plan_series_shards(133, replace(spec, shard_count=2))
    assert shards[0].sample_stop > shards[1].sample_start


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_samples": 0},
        {"hop_samples": 0},
        {"batch_windows": True},
        {"windows_per_result": -1},
        {"output": "amplitude"},
        {"lag_mode": "other"},
        {"precision": "float16"},
        {"shard_index": 2, "shard_count": 2},
        {"pairs": (("a", "a"),) * 2},
    ],
)
def test_invalid_spec(kwargs):
    values = dict(pairs=(("a", "a"),), window_samples=8)
    values.update(kwargs)
    with pytest.raises(ValueError):
        CorrelationSpec(**values)


def test_missing_nonfinite_and_short_sources():
    source = ArraySource({"a": np.ones(2)}, sample_rate_hz=1)
    assert (
        list(correlate_series(source, CorrelationSpec((("a", "a"),), 3)))
        == []
    )
    with pytest.raises(ValueError, match="unavailable"):
        list(correlate_series(source, CorrelationSpec((("a", "b"),), 2)))
    bad = ArraySource({"a": np.array([np.nan, 1])}, sample_rate_hz=1)
    with pytest.raises(ValueError, match="nonfinite"):
        list(correlate_series(bad, CorrelationSpec((("a", "a"),), 2)))


def test_core_imports_do_not_load_application_adapters():
    code = """
import sys
from cuphoton.xbeam import CorrelationSpec, correlate_series
bad = [n for n in sys.modules
       if n.startswith('cuphoton.xbeam.adapters')
       or n in ['cuphoton.xbeam.adapters.radio.models',
                'cuphoton.xbeam.adapters.vdif.vdif']]
assert not bad, bad
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


@pytest.mark.gpu
def test_generic_cupy_matches_direct_sums():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(17)
    a = rng.normal(size=64) + 1j * rng.normal(size=64)
    b = rng.normal(size=64) + 1j * rng.normal(size=64)
    source = ArraySource(
        {"a": cp.asarray(a), "b": cp.asarray(b)}, sample_rate_hz=1
    )
    (result,) = correlate_series(
        source,
        CorrelationSpec((("a", "b"),), 64, output="both", backend="cupy"),
    )
    np.testing.assert_allclose(
        result.lag[0], direct_lags(a, b, result.lag_samples), atol=1e-12
    )


def test_bad_series_config_is_actionable(tmp_path):
    from cuphoton.xbeam.adapters.series_cli import run_series_config

    config = tmp_path / "bad.json"
    for value in [[], {}, {"input": "x", "datasets": {}, "correlation": []}]:
        config.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            run_series_config(config, tmp_path / "out.h5")
    assert not (tmp_path / "out.h5").exists()


@pytest.mark.parametrize(
    "field,value", [("input", None), ("datasets", ["a"]), ("validity", [])]
)
def test_series_config_rejects_invalid_source_fields_before_io(
    tmp_path, field, value
):
    from cuphoton.xbeam.adapters.series_cli import run_series_config

    config = {
        "input": "missing.h5",
        "datasets": {"a": "a"},
        "correlation": {"pairs": [["a", "a"]], "window_samples": 8},
        field: value,
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "out.h5"
    with pytest.raises(ValueError, match=field):
        run_series_config(path, output)
    assert not output.exists()


@pytest.mark.parametrize("hop", [3, 8, 29])
@pytest.mark.parametrize("masked", [False, True])
def test_hdf5_batched_reads_preserve_products_and_bound_io(
    tmp_path, hop, masked
):
    h5py = pytest.importorskip("h5py")
    from cuphoton.xbeam import Hdf5Source

    rng = np.random.default_rng(91)
    arrays = {
        "a": rng.normal(size=401) + 1j * rng.normal(size=401),
        "b": rng.normal(size=401) + 1j * rng.normal(size=401),
    }
    validity = {"b": rng.random(401) > 0.2} if masked else {}
    path = tmp_path / "batch.h5"
    with h5py.File(path, "w") as f:
        for key, values in arrays.items():
            ds = f.create_dataset(
                key, data=values, chunks=True, compression="gzip"
            )
            ds.attrs.update(sample_rate_hz=13.0, origin_seconds=5.0)
        if masked:
            f.create_dataset("mask", data=validity["b"])
    spec = CorrelationSpec(
        (("a", "b"), ("b", "b")),
        8,
        hop_samples=hop,
        windows_per_result=7,
        batch_windows=3,
        output="both",
    )
    expected = list(
        correlate_series(
            ArraySource(
                arrays,
                sample_rate_hz=13.0,
                origin_seconds=5.0,
                validity=validity,
            ),
            spec,
        )
    )
    with Hdf5Source(
        path,
        {"a": "a", "b": "b"},
        validity_paths={"b": "mask"} if masked else None,
    ) as source:
        spans = []
        read = source.read

        def counted(key, start, stop):
            spans.append(stop - start)
            return read(key, start, stop)

        source.read = counted
        actual = list(correlate_series(source, spec))
    assert max(spans) <= spec.batch_windows * spec.window_samples
    if hop <= 8:
        assert max(spans) > 8
    else:
        assert set(spans) == {8}
    for x, y in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(x.spectrum, y.spectrum)
        np.testing.assert_array_equal(x.lag, y.lag)
        np.testing.assert_array_equal(
            x.lag_valid_samples, y.lag_valid_samples
        )
        np.testing.assert_array_equal(
            x.simultaneous_valid_samples, y.simultaneous_valid_samples
        )


def test_all_valid_coverage_needs_no_mask_transforms(monkeypatch):
    original = np.fft.fft
    seen = []

    def counted(values, *args, **kwargs):
        seen.append(values.dtype)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.fft, "fft", counted)
    source = ArraySource({"a": np.ones(32, dtype="f4")}, sample_rate_hz=1)
    (result,) = correlate_series(
        source,
        CorrelationSpec(
            (("a", "a"),),
            8,
            windows_per_result=4,
            batch_windows=4,
            output="both",
        ),
    )
    assert seen == [np.dtype("f4")]
    np.testing.assert_array_equal(
        result.lag_valid_samples[0], 4 * (8 - abs(result.lag_samples))
    )


@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_mask_presence_can_change_between_batches(mode):
    source = ArraySource({"a": np.arange(32.0)}, sample_rate_hz=1)
    read = source.read

    def alternating(key, start, stop):
        v, _ = read(key, start, stop)
        return v, None if start < 16 else np.arange(stop - start) % 2 == 0

    source.read = alternating
    (result,) = correlate_series(
        source,
        CorrelationSpec(
            (("a", "a"),),
            8,
            windows_per_result=4,
            batch_windows=1,
            output="both",
            lag_mode=mode,
        ),
    )
    expected = np.zeros(len(result.lag_samples))
    for start in range(0, 32, 8):
        v, m = alternating("a", start, start + 8)
        mask = np.ones(8) if m is None else m.astype(float)
        expected += direct_lags(
            mask, mask, result.lag_samples, mode == "circular"
        ).real
    np.testing.assert_array_equal(result.lag_valid_samples[0], expected)


@pytest.mark.gpu
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_hdf5_gpu_batched_mask_accounting(tmp_path, masked, mode):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    h5py = pytest.importorskip("h5py")
    from cuphoton.xbeam import Hdf5Source

    rng = np.random.default_rng(31)
    a, b = rng.normal(size=(2, 137))
    mask = rng.random(137) > 0.2
    path = tmp_path / "gpu.h5"
    with h5py.File(path, "w") as f:
        for key, v in [("a", a), ("b", b)]:
            ds = f.create_dataset(key, data=v, compression="gzip")
            ds.attrs.update(sample_rate_hz=1.0, origin_seconds=0.0)
        f.create_dataset("mask", data=mask)
    spec = CorrelationSpec(
        (("a", "b"), ("b", "b")),
        17,
        hop_samples=11,
        windows_per_result=3,
        batch_windows=2,
        output="both",
        lag_mode=mode,
    )
    expected = list(
        correlate_series(
            ArraySource(
                {"a": a, "b": b},
                sample_rate_hz=1,
                validity={"b": mask} if masked else None,
            ),
            spec,
        )
    )
    with Hdf5Source(
        path,
        {"a": "a", "b": "b"},
        validity_paths={"b": "mask"} if masked else None,
    ) as source:
        actual = list(correlate_series(source, replace(spec, backend="cupy")))
    for x, y in zip(actual, expected, strict=True):
        np.testing.assert_allclose(x.spectrum, y.spectrum, atol=2e-12)
        np.testing.assert_allclose(x.lag, y.lag, atol=2e-13)
        np.testing.assert_array_equal(
            x.lag_valid_samples, y.lag_valid_samples
        )
        np.testing.assert_array_equal(
            x.simultaneous_valid_samples, y.simultaneous_valid_samples
        )


@pytest.mark.parametrize("invalid_is_masked", [False, True])
def test_hdf5_batch_validation_rejects_only_valid_nonfinite(
    tmp_path, invalid_is_masked
):
    h5py = pytest.importorskip("h5py")
    from cuphoton.xbeam import Hdf5Source

    path = tmp_path / "nonfinite.h5"
    values = np.arange(16.0)
    values[3] = np.nan
    mask = np.ones(16, dtype=bool)
    mask[3] = not invalid_is_masked
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("a", data=values)
        ds.attrs.update(sample_rate_hz=1.0, origin_seconds=0.0)
        f.create_dataset("mask", data=mask)
    with Hdf5Source(path, {"a": "a"}, validity_paths={"a": "mask"}) as source:
        results = correlate_series(
            source,
            CorrelationSpec(
                (("a", "a"),), 8, windows_per_result=2, output="both"
            ),
        )
        if invalid_is_masked:
            (result,) = results
            assert np.isfinite(result.spectrum).all()
            assert result.simultaneous_valid_samples[0] == 15
        else:
            with pytest.raises(ValueError, match="nonfinite"):
                list(results)


@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_first_signal_only_mask_counts_totals_and_invalid_windows(mode):
    a, b = np.arange(16.0).reshape(2, 8)
    mask = np.array([False] * 4 + [True, False, True, True])
    source = ArraySource(
        {"a": a, "b": b}, sample_rate_hz=8, validity={"a": mask}
    )
    (result,) = correlate_series(
        source,
        CorrelationSpec(
            (("a", "b"),),
            4,
            windows_per_result=2,
            output="both",
            lag_mode=mode,
        ),
    )
    expected_sum = direct_lags(
        np.where(mask[4:], a[4:], 0),
        b[4:],
        result.lag_samples,
        mode == "circular",
    )
    counts = direct_lags(
        mask[4:].astype(int),
        np.ones(4),
        result.lag_samples,
        mode == "circular",
    )
    np.testing.assert_allclose(result.lag[0], expected_sum / 2, atol=1e-13)
    np.testing.assert_array_equal(result.lag_valid_samples[0], counts)
    np.testing.assert_array_equal(result.simultaneous_valid_samples, [3])
    np.testing.assert_array_equal(result.contributing_windows, [1])
    np.testing.assert_array_equal(result.valid, [True])
    covered = counts > 0
    np.testing.assert_allclose(
        2 * result.lag[0, covered] / counts[covered],
        expected_sum[covered] / counts[covered],
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "n,mode,expected_lags,fft_samples",
    [
        (7, "linear", list(range(-6, 7)), 16),
        (8, "linear", list(range(-7, 8)), 16),
        (7, "circular", [-3, -2, -1, 0, 1, 2, 3], 7),
        (8, "circular", [-4, -3, -2, -1, 0, 1, 2, 3], 8),
    ],
)
def test_axes_match_sample_and_frequency_grids(
    n, mode, expected_lags, fft_samples
):
    (result,) = correlate_series(
        ArraySource({"a": np.ones(n)}, sample_rate_hz=13),
        CorrelationSpec((("a", "a"),), n, output="both", lag_mode=mode),
    )
    np.testing.assert_array_equal(result.lag_samples, expected_lags)
    np.testing.assert_array_equal(
        result.frequencies_hz, np.fft.fftfreq(fft_samples, d=1 / 13)
    )


def test_series_cli_validates_before_output_and_round_trips(tmp_path):
    h5py = pytest.importorskip("h5py")
    from cuphoton.xbeam.adapters.series_cli import run_series_config

    source_path = tmp_path / "input.h5"
    values = np.arange(16.0)
    with h5py.File(source_path, "w") as source:
        dataset = source.create_dataset("a", data=values)
        dataset.attrs.update(sample_rate_hz=8.0, origin_seconds=2.0)
    config = {
        "input": "input.h5",
        "datasets": {"a": "a"},
        "correlation": {
            "pairs": [["a", "missing"]],
            "window_samples": 4,
            "windows_per_result": 2,
            "output": "both",
        },
    }
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "output.h5"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unavailable"):
        run_series_config(config_path, output_path)
    assert not output_path.exists()
    config["correlation"]["pairs"] = [["a", "a"]]
    config_path.write_text(json.dumps(config))
    run_series_config(config_path, output_path)
    expected = list(
        correlate_series(
            ArraySource({"a": values}, sample_rate_hz=8, origin_seconds=2),
            CorrelationSpec(**config["correlation"]),
        )
    )
    with h5py.File(output_path) as output:
        assert output.attrs["schema"] == "cuphoton-series-v1"
        assert output.attrs["complete"]
        assert len(output["results"]) == len(expected)
        for result in expected:
            group = output[f"results/{result.result_index}"]
            assert json.loads(group.attrs["pairs_json"]) == [["a", "a"]]
            for name in (
                "lag",
                "spectrum",
                "lag_valid_samples",
                "window_starts",
            ):
                np.testing.assert_array_equal(
                    group[name][:], getattr(result, name)
                )
    saved = output_path.read_bytes()
    with pytest.raises(FileExistsError):
        run_series_config(config_path, output_path)
    assert output_path.read_bytes() == saved
