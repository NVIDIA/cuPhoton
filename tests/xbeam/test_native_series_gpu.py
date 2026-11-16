# SPDX-License-Identifier: Apache-2.0
"""Independent numerical coverage of native HDF5 IO and time-series DSP."""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def native_cuda():
    from cuphoton.xbeam import native

    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if directory:
        files = list(Path(directory).rglob("_binding*.so"))
        if len(files) == 1:
            name = "cuphoton.xbeam.native._binding"
            spec = importlib.util.spec_from_file_location(name, files[0])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
    if not native.cuda_available():
        pytest.skip("native CUDA extension is not installed")


def write_signals(path, dtype, *, samples=256, masks=True, bad=None):
    import h5py

    rng = np.random.default_rng(419)
    a = rng.normal(size=samples)
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.normal(size=samples)
    b = np.roll(a, 3) + 0.1 * rng.normal(size=samples)
    ma, mb = rng.random(samples) > 0.3, rng.random(samples) > 0.2
    if masks:
        ma[40:80] = False
        a[~ma] = np.nan
    if bad is not None:
        a[bad] = np.nan
        ma[bad] = True
    with h5py.File(
        path, "w", alignment_threshold=1, alignment_interval=65536
    ) as f:
        for name, values in [("a", a), ("b", b)]:
            ds = f.create_dataset(name, data=values.astype(dtype))
            ds.attrs["sample_rate_hz"] = 128.0
            ds.attrs["origin_seconds"] = 2.25
        f["ma"], f["mb"] = ma, mb
    return {"a": "ma", "b": "mb"} if masks else None


def compare_results(actual, expected, precision):
    assert len(actual) == len(expected)
    tolerance = 3e-5 if precision == "float32" else 3e-12
    for a, b in zip(actual, expected, strict=True):
        for name in (
            "result_index",
            "pairs",
            "sample_start",
            "sample_stop",
            "sample_rate_hz",
            "origin_seconds",
            "fft_samples",
            "lag_mode",
        ):
            assert getattr(a, name) == getattr(b, name), name
        for name in (
            "window_starts",
            "contributing_windows",
            "simultaneous_valid_samples",
            "valid",
            "lag_samples",
            "lag_valid_samples",
            "frequencies_hz",
        ):
            np.testing.assert_array_equal(
                getattr(a, name), getattr(b, name), err_msg=name
            )
        for name in ("spectrum", "lag"):
            x, y = getattr(a, name), getattr(b, name)
            if y is None:
                assert x is None
            else:
                assert x.dtype == y.dtype
                np.testing.assert_allclose(
                    x, y, rtol=tolerance, atol=tolerance, err_msg=name
                )
                reference_norm = np.linalg.norm(y, axis=-1)
                reference_peak = np.max(np.abs(y), axis=-1)
                error_norm = np.linalg.norm(x - y, axis=-1)
                error_peak = np.max(np.abs(x - y), axis=-1)
                assert np.all(error_norm <= 2e-6 * reference_norm)
                assert np.all(error_peak <= 5e-6 * reference_peak)
                if name == "lag":
                    np.testing.assert_array_equal(
                        np.argmax(np.abs(x), axis=-1),
                        np.argmax(np.abs(y), axis=-1),
                    )


@pytest.mark.parametrize(
    "dtype", ["float32", "float64", "complex64", "complex128"]
)
@pytest.mark.parametrize("mode", ["linear", "circular"])
@pytest.mark.parametrize("masks", [False, True])
@pytest.mark.parametrize("output", ["spectrum", "lag", "both"])
def test_native_matches_cpu(tmp_path, dtype, mode, masks, output):
    path = tmp_path / "input.h5"
    validity = write_signals(path, dtype, masks=masks)
    spec = CorrelationSpec(
        (("a", "b"), ("a", "a"), ("b", "a")),
        17,
        hop_samples=11,
        windows_per_result=5,
        batch_windows=3,
        output=output,
        lag_mode=mode,
    )
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths=validity
    ) as source:
        expected = list(correlate_series(source, spec))
        # Raw file IO must stay native, rather than using Python read methods.
        source.read = lambda *a, **k: pytest.fail(
            "native execution read through Python"
        )
        actual = list(
            correlate_series(source, replace(spec, backend="native"))
        )
    precision = (
        "float32"
        if np.dtype(dtype).itemsize <= (8 if "complex" in dtype else 4)
        else "float64"
    )
    compare_results(actual, expected, precision)


@pytest.mark.parametrize("io", ["posix", "cufile"])
def test_native_shards_and_cupy_agree(tmp_path, io):
    if io == "cufile":
        root = os.environ.get("XBEAM_NATIVE_GDS_TEST_ROOT")
        if not root:
            pytest.skip("set XBEAM_NATIVE_GDS_TEST_ROOT to direct-IO storage")
        import tempfile
        from contextlib import ExitStack

        with ExitStack() as stack:
            folder = stack.enter_context(
                tempfile.TemporaryDirectory(dir=root)
            )
            _shards(Path(folder), io)
    else:
        _shards(tmp_path, io)


def _shards(tmp_path, io):
    cupy = pytest.importorskip("cupy")
    path = tmp_path / "input.h5"
    validity = write_signals(path, "complex64", samples=4096)
    spec = CorrelationSpec(
        (("a", "b"), ("b", "a")),
        64,
        windows_per_result=3,
        batch_windows=2,
        output="both",
        backend="cupy",
    )
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths=validity
    ) as source:
        expected = list(correlate_series(source, spec))
    del cupy
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths=validity, io_backend=io
    ) as source:
        actual = [
            r
            for shard in range(3)
            for r in correlate_series(
                source,
                replace(
                    spec, backend="native", shard_count=3, shard_index=shard
                ),
            )
        ]
    compare_results(actual, expected, "float32")


def test_native_rejects_nonfinite_valid_input(tmp_path):
    path = tmp_path / "bad.h5"
    validity = write_signals(path, "float64", bad=0)
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths=validity
    ) as source:
        with pytest.raises(RuntimeError, match="nonfinite"):
            list(
                correlate_series(
                    source,
                    CorrelationSpec((("a", "b"),), 32, backend="native"),
                )
            )


@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_native_lags_match_direct_masked_sums(tmp_path, mode):
    import h5py

    path = tmp_path / "direct.h5"
    validity = write_signals(path, "complex128", samples=128)
    spec = CorrelationSpec(
        (("a", "b"),),
        13,
        hop_samples=7,
        windows_per_result=3,
        batch_windows=2,
        output="both",
        lag_mode=mode,
        backend="native",
    )
    with h5py.File(path, "r") as f:
        a, b, ma, mb = (f[key][:] for key in ("a", "b", "ma", "mb"))
    with Hdf5Source(
        path, {"a": "a", "b": "b"}, validity_paths=validity
    ) as source:
        result = next(correlate_series(source, spec))
    expected, coverage = [], []
    for k in result.lag_samples:
        value, count = 0j, 0
        for start in result.window_starts:
            aa, bb = a[start : start + 13], b[start : start + 13]
            va, vb = ma[start : start + 13], mb[start : start + 13]
            if mode == "circular":
                bb, vb = np.roll(bb, -k), np.roll(vb, -k)
            elif k >= 0:
                aa, bb, va, vb = aa[: 13 - k], bb[k:], va[: 13 - k], vb[k:]
            else:
                aa, bb, va, vb = aa[-k:], bb[: 13 + k], va[-k:], vb[: 13 + k]
            value += np.sum(
                np.where(va, aa, 0) * np.conj(np.where(vb, bb, 0))
            )
            count += np.sum(va & vb)
        expected.append(value / 3)
        coverage.append(count)
    np.testing.assert_allclose(result.lag[0], expected, atol=2e-12)
    np.testing.assert_array_equal(result.lag_valid_samples[0], coverage)


def test_native_generator_close_preserves_owned_results(tmp_path):
    path = tmp_path / "owned.h5"
    write_signals(path, "float32", masks=False)
    spec = CorrelationSpec((("a", "b"),), 16, backend="native")
    with Hdf5Source(path, {"a": "a", "b": "b"}) as source:
        iterator = correlate_series(source, spec)
        first = next(iterator)
        saved = first.spectrum.copy()
        iterator.close()
        list(correlate_series(source, spec))
        np.testing.assert_array_equal(first.spectrum, saved)


def test_native_source_replacement_is_rejected(tmp_path):
    from cuphoton.xbeam.native import _binding
    from cuphoton.xbeam.native.series import _configuration

    path = tmp_path / "identity.h5"
    write_signals(path, "float64", masks=False)
    with Hdf5Source(path, {"a": "a", "b": "b"}) as source:
        config = _configuration(source, CorrelationSpec((("a", "b"),), 16))
        replacement = tmp_path / "replacement.h5"
        write_signals(replacement, "float64", masks=False)
        replacement.replace(path)
        with pytest.raises(RuntimeError, match="source changed"):
            _binding.SeriesExecutor(config)


def test_native_captures_and_restores_device(tmp_path):
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 3:
        pytest.skip("device ownership check requires three visible GPUs")
    path = tmp_path / "devices.h5"
    write_signals(path, "float32", masks=False)
    with Hdf5Source(path, {"a": "a", "b": "b"}) as source:
        with cupy.cuda.Device(1):
            iterator = correlate_series(
                source, CorrelationSpec((("a", "b"),), 32, backend="native")
            )
            first = next(iterator)
            assert cupy.cuda.runtime.getDevice() == 1
        with cupy.cuda.Device(2):
            second = next(iterator)
            assert cupy.cuda.runtime.getDevice() == 2
            iterator.close()
            assert cupy.cuda.runtime.getDevice() == 2
        assert first.spectrum.shape == second.spectrum.shape


@pytest.mark.parametrize("io", ["posix", "cufile"])
def test_native_large_raw_spans_and_read_budget(tmp_path, io):
    import tempfile
    from contextlib import ExitStack

    with ExitStack() as stack:
        if io == "cufile":
            root = os.environ.get("XBEAM_NATIVE_GDS_TEST_ROOT")
            if not root:
                pytest.skip("set XBEAM_NATIVE_GDS_TEST_ROOT")
            tmp_path = Path(
                stack.enter_context(tempfile.TemporaryDirectory(dir=root))
            )
        path = tmp_path / "large.h5"
        validity = write_signals(path, "complex64", samples=262144)
        spec = CorrelationSpec(
            (("a", "b"),),
            65536,
            hop_samples=32768,
            windows_per_result=5,
            batch_windows=16,
            precision="float64",
            output="both",
        )
        with Hdf5Source(
            path, {"a": "a", "b": "b"}, validity_paths=validity
        ) as source:
            expected = list(correlate_series(source, spec))
        with Hdf5Source(
            path,
            {"a": "a", "b": "b"},
            validity_paths=validity,
            io_backend=io,
            read_buffer_bytes=1024**2,
        ) as source:
            actual = list(
                correlate_series(source, replace(spec, backend="native"))
            )
        compare_results(actual, expected, "float64")


def test_native_series_cli_writes_standard_result_file(tmp_path):
    import json

    import h5py

    from cuphoton.xbeam.adapters.series_cli import run_series_config

    path = tmp_path / "input.h5"
    write_signals(path, "float32", masks=False)
    config = tmp_path / "series.json"
    config.write_text(
        json.dumps(
            {
                "input": "input.h5",
                "datasets": {"a": "a", "b": "b"},
                "correlation": {
                    "pairs": [["a", "b"]],
                    "window_samples": 32,
                    "windows_per_result": 2,
                    "output": "both",
                },
            }
        )
    )
    output = tmp_path / "output.h5"
    run_series_config(config, output, "native", upload_workers="2")
    with h5py.File(output) as f:
        assert f.attrs["schema"] == "cuphoton-series-v1"
        assert f.attrs["complete"]
        assert len(f["results"]) == 4
        assert f["results/0/spectrum"].shape == (1, 64)
        assert f["results/0/lag"].shape == (1, 63)
    with pytest.raises(FileExistsError):
        run_series_config(config, output, "native")


def test_native_runtime_does_not_require_cupy(tmp_path):
    import subprocess

    path = tmp_path / "no-cupy.h5"
    write_signals(path, "float32", masks=False)
    script = """
import importlib.util
import os
import sys
from contextlib import closing
from pathlib import Path
sys.modules["cupy"] = None
from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series
name = "cuphoton.xbeam.native._binding"
directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
if directory:
    files = list(Path(directory).rglob("_binding*.so"))
    spec = importlib.util.spec_from_file_location(name, files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
with Hdf5Source(sys.argv[1], {"a": "a", "b": "b"}) as source:
    def forbidden(*args, **kwargs):
        raise AssertionError("native path used Python sample reads")
    source.read = forbidden
    with closing(correlate_series(source, CorrelationSpec(
            (("a", "b"),), 16, backend="native"))) as results:
        assert next(results).spectrum.shape == (1, 32)
assert sys.modules["cupy"] is None
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)], check=True, timeout=60
    )


@pytest.mark.parametrize("mode", ["linear", "circular"])
def test_native_single_sample_windows_with_gaps(tmp_path, mode):
    path = tmp_path / "single.h5"
    write_signals(path, "complex128", samples=32, masks=False)
    spec = CorrelationSpec(
        (("a", "b"),),
        1,
        hop_samples=4,
        windows_per_result=3,
        batch_windows=2,
        output="both",
        lag_mode=mode,
    )
    with Hdf5Source(path, {"a": "a", "b": "b"}) as source:
        expected = list(correlate_series(source, spec))
        actual = list(
            correlate_series(source, replace(spec, backend="native"))
        )
    compare_results(actual, expected, "float64")
