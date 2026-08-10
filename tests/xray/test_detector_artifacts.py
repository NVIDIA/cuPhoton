# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xray.detector_artifacts import (
    DETECTOR_ARRAYS,
    FIT_STATUS_FILE,
    FIT_STATUS_OK,
    FIT_STATUS_SKIPPED,
    FIT_STATUS_UNPROCESSED,
    _clear_detector_artifact_outputs,
    _ensure_cuda_device,
    _fit_error_types,
    _full_detector_trace,
    _integrate_cupy,
    _load_tile_signal,
    _publish_detector_artifact_outputs,
    _row_halo_bounds,
    _tdsfft_cupy,
    _zero_excluded_signal_rows,
    build_detector_artifacts_cupy,
    compare_detector_artifacts,
)
from cuphoton.xray.detector_mask import AxisRange
from cuphoton.xray.linear_prediction import synthetic_trace_batch


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def test_compare_detector_artifacts_uses_candidate_manifest_origin(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()

    shape = (4, 5, 3)
    data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        np.save(reference / f"{name}.npy", data)
        np.save(candidate / f"{name}.npy", data[1:3, 2:4, :])
    amp_sum = np.arange(20, dtype=np.float64).reshape(4, 5)
    np.save(reference / "amp_all_sum_filtered.npy", amp_sum)
    np.save(candidate / "amp_all_sum_filtered.npy", amp_sum[1:3, 2:4])
    (candidate / "manifest.json").write_text(
        json.dumps({"roi_lower": [2, 1]}),
        encoding="utf-8",
    )

    result = compare_detector_artifacts(
        reference_dir=reference,
        candidate_dir=candidate,
        amp_threshold=1.0,
    )

    assert result["comparable"] is True
    assert result["candidate_origin"] == [2, 1]
    for stats in result["arrays"].values():
        assert stats["shape_comparable"] is True
        assert stats["max_abs_diff"] == 0.0
        assert stats["rms_diff"] == 0.0
    assert result["filtered_modes"]["mask_agreement_ratio"] == 1.0


def test_detector_artifact_compare_cli_json(tmp_path, capsys):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        np.save(reference / f"{name}.npy", np.ones((2, 2, 2)))
        np.save(candidate / f"{name}.npy", np.ones((2, 2, 2)))
    np.save(reference / "amp_all_sum_filtered.npy", np.ones((2, 2)))
    np.save(candidate / "amp_all_sum_filtered.npy", np.ones((2, 2)))

    assert (
        main(
            [
                "detector-artifact-compare",
                "--reference-dir",
                str(reference),
                "--candidate-dir",
                str(candidate),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["comparable"] is True
    assert payload["arrays"]["freq_all"]["max_abs_diff"] == 0.0


def test_compare_detector_artifacts_honors_equal_shape_candidate_origin(
    tmp_path,
):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        np.save(reference / f"{name}.npy", np.ones((2, 2, 2)))
        np.save(candidate / f"{name}.npy", np.ones((2, 2, 2)))
    np.save(reference / "amp_all_sum_filtered.npy", np.ones((2, 2)))
    np.save(candidate / "amp_all_sum_filtered.npy", np.ones((2, 2)))

    result = compare_detector_artifacts(
        reference_dir=reference,
        candidate_dir=candidate,
        candidate_origin=(1, 0),
    )

    assert result["comparable"] is False
    assert result["arrays"]["freq_all"]["shape_comparable"] is False


def test_compare_detector_artifacts_reports_nonfinite_value_mismatch(
    tmp_path,
):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    ref = np.array([[[np.inf, np.nan, 1.0]]], dtype=np.float64)
    cand = np.array([[[-np.inf, np.nan, 3.0]]], dtype=np.float64)
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        np.save(reference / f"{name}.npy", ref)
        np.save(candidate / f"{name}.npy", cand)
    np.save(reference / "amp_all_sum_filtered.npy", np.array([[np.inf]]))
    np.save(candidate / "amp_all_sum_filtered.npy", np.array([[-np.inf]]))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = compare_detector_artifacts(
            reference_dir=reference,
            candidate_dir=candidate,
        )

    stats = result["arrays"]["freq_all"]
    assert stats["max_abs_diff"] == 2.0
    assert stats["finite_mismatch_count"] == 0
    assert stats["nonfinite_value_mismatch_count"] == 1
    assert stats["nonzero_reference"] == 1
    assert stats["nonzero_candidate"] == 1
    assert stats["nonfinite_reference"] == 2
    assert stats["nonfinite_candidate"] == 2
    assert (
        result["arrays"]["amp_all_sum_filtered"][
            "nonfinite_value_mismatch_count"
        ]
        == 1
    )
    assert result["arrays"]["amp_all_sum_filtered"]["nonzero_reference"] == 0


def test_detector_artifacts_rejects_unknown_reader_before_gpu_probe(tmp_path):
    try:
        build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=tmp_path / "out",
            hdf5_reader="unknown",
        )
    except ValueError as exc:
        assert str(exc) == (
            "hdf5_reader must be one of: h5py, h5py-threaded, "
            "hdf5-ts-funcwrap"
        )
    else:
        raise AssertionError("expected reader validation")


def test_detector_artifacts_rejects_threaded_reader_with_one_worker(tmp_path):
    try:
        build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=tmp_path / "out",
            hdf5_reader="hdf5-ts-funcwrap",
            hdf5_reader_workers=1,
        )
    except ValueError as exc:
        assert str(exc) == (
            "hdf5_reader_workers must be at least 2 for threaded HDF5 readers"
        )
    else:
        raise AssertionError("expected threaded worker validation")


def test_detector_artifacts_rejects_negative_max_fit_failures(tmp_path):
    try:
        build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=tmp_path / "out",
            max_fit_failures=-1,
        )
    except ValueError as exc:
        assert str(exc) == "max_fit_failures must be non-negative"
    else:
        raise AssertionError("expected max_fit_failures validation")


def test_clear_detector_artifact_outputs_removes_known_files(tmp_path):
    for name in DETECTOR_ARRAYS:
        (tmp_path / f"{name}.npy").write_bytes(b"old")
    (tmp_path / FIT_STATUS_FILE).write_bytes(b"old")
    (tmp_path / "manifest.json").write_text("old", encoding="utf-8")
    (tmp_path / "compare-reference.json").write_text("keep", encoding="utf-8")

    _clear_detector_artifact_outputs(tmp_path)

    for name in DETECTOR_ARRAYS:
        assert not (tmp_path / f"{name}.npy").exists()
    assert not (tmp_path / FIT_STATUS_FILE).exists()
    assert not (tmp_path / "manifest.json").exists()
    assert (tmp_path / "compare-reference.json").exists()


def test_publish_detector_artifact_outputs_replaces_known_files(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    for name in DETECTOR_ARRAYS:
        np.save(source / f"{name}.npy", np.full((1,), 2.0))
        np.save(target / f"{name}.npy", np.full((1,), 1.0))
    np.save(
        source / FIT_STATUS_FILE,
        np.array([FIT_STATUS_OK], dtype=np.uint8),
    )
    np.save(
        target / FIT_STATUS_FILE,
        np.array([FIT_STATUS_UNPROCESSED], dtype=np.uint8),
    )
    (source / "manifest.json").write_text("new", encoding="utf-8")
    (target / "manifest.json").write_text("old", encoding="utf-8")
    (target / "compare-reference.json").write_text("keep", encoding="utf-8")

    _publish_detector_artifact_outputs(source, target)

    for name in DETECTOR_ARRAYS:
        np.testing.assert_allclose(np.load(target / f"{name}.npy"), [2.0])
        assert not (source / f"{name}.npy").exists()
    np.testing.assert_array_equal(
        np.load(target / FIT_STATUS_FILE),
        np.array([FIT_STATUS_OK], dtype=np.uint8),
    )
    assert (target / "manifest.json").read_text(encoding="utf-8") == "new"
    assert (target / "compare-reference.json").read_text(
        encoding="utf-8"
    ) == "keep"


def test_detector_artifacts_rejects_savgol_window_longer_than_samples(
    tmp_path,
):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(tmp_path, samples=16, rows=2, cols=2)
    output = tmp_path / "out"
    output.mkdir()
    (output / "manifest.json").write_text("old", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="savgol_window must be less than or equal to usable samples",
    ):
        build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=output,
            roi_lower=(0, 0),
            roi_dim=(2, 2),
            tile_shape=(2, 2),
            drop_leading=12,
            zero_offset_index=0,
            components=2,
            savgol_window=5,
            savgol_polyorder=3,
        )
    assert (output / "manifest.json").read_text(encoding="utf-8") == "old"


def test_detector_artifact_failure_preserves_existing_outputs(tmp_path):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(tmp_path, samples=48, rows=4, cols=4)
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "manifest.json").write_text("old", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="detector artifact generation produced no successful row fits",
    ):
        build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=output,
            roi_lower=(0, 0),
            roi_dim=(4, 4),
            tile_shape=(2, 2),
            exclude_y=(AxisRange(0, 4),),
            drop_leading=0,
            chunk_frames=8,
            zero_offset_index=0,
            fit_trailing_drop=1,
            integrate_pixels=0,
            components=6,
            savgol_window=5,
            savgol_polyorder=3,
            amp_threshold=0.01,
        )

    assert (output / "manifest.json").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".artifacts.tmp-*"))


def test_ensure_cuda_device_converts_cupy_runtime_error():
    class FakeCUDARuntimeError(Exception):
        pass

    class FakeRuntime:
        CUDARuntimeError = FakeCUDARuntimeError

        @staticmethod
        def getDeviceCount():
            raise FakeCUDARuntimeError("driver unavailable")

    class FakeCupy:
        class cuda:
            runtime = FakeRuntime

    with pytest.raises(RuntimeError, match="no CUDA devices visible"):
        _ensure_cuda_device(FakeCupy)


def test_ensure_cuda_device_rejects_zero_devices():
    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 0

    class FakeCupy:
        class cuda:
            runtime = FakeRuntime

    with pytest.raises(RuntimeError, match="no CUDA devices visible"):
        _ensure_cuda_device(FakeCupy)


def test_fit_error_types_include_cupy_runtime_and_memory_errors():
    class FakeLinalgError(Exception):
        pass

    class FakeRuntimeError(Exception):
        pass

    class FakeOutOfMemoryError(Exception):
        pass

    class FakeDriverError(Exception):
        pass

    class FakeCupy:
        class linalg:
            LinAlgError = FakeLinalgError

        class cuda:
            class runtime:
                CUDARuntimeError = FakeRuntimeError

            class memory:
                OutOfMemoryError = FakeOutOfMemoryError

            class driver:
                CUDADriverError = FakeDriverError

    errors = _fit_error_types(FakeCupy)

    assert FakeLinalgError in errors
    assert FakeRuntimeError in errors
    assert FakeOutOfMemoryError in errors
    assert FakeDriverError in errors


def test_build_detector_artifacts_cupy_smoke_with_threaded_reader(tmp_path):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(tmp_path, samples=48, rows=4, cols=4)
    output = tmp_path / "artifacts"

    result = build_detector_artifacts_cupy(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=output,
        roi_lower=(0, 0),
        roi_dim=(4, 4),
        tile_shape=(2, 2),
        drop_leading=0,
        chunk_frames=8,
        zero_offset_index=0,
        fit_trailing_drop=1,
        integrate_pixels=0,
        components=6,
        savgol_window=5,
        savgol_polyorder=3,
        amp_threshold=0.01,
        hdf5_reader="hdf5-ts-funcwrap",
        hdf5_reader_workers=2,
    )

    assert result.shape == (4, 4, 24)
    assert result.processed_tiles == 4
    assert result.raw_fits == 8
    assert result.failures == 0
    assert (output / "manifest.json").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["hdf5_reader"] == "hdf5-ts-funcwrap"
    assert manifest["hdf5_reader_runtime"]["backend"] == (
        "h5py-worker-threads"
    )
    assert (
        manifest["hdf5_reader_runtime"]["requested_beta_target"]["branch"]
        == "ts_funcwrap_1"
    )
    assert FIT_STATUS_FILE in manifest["arrays"]
    freq_all = np.load(output / "freq_all.npy", mmap_mode="r")
    amp_all = np.load(output / "amp_all.npy", mmap_mode="r")
    amp_sum = np.load(output / "amp_all_sum_filtered.npy", mmap_mode="r")
    fit_status = np.load(output / FIT_STATUS_FILE, mmap_mode="r")
    assert freq_all.shape == (4, 4, 24)
    assert amp_all.shape == (4, 4, 24)
    assert amp_sum.shape == (4, 4)
    assert fit_status.shape == (4, 4)
    assert set(np.unique(fit_status)) == {FIT_STATUS_OK}
    assert np.count_nonzero(amp_all) > 0
    np.testing.assert_allclose(freq_all[:, 0, :], freq_all[:, 1, :])


def test_max_tiles_zeroes_unprocessed_detector_tiles(tmp_path):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(tmp_path, samples=48, rows=4, cols=4)
    output = tmp_path / "artifacts"

    build_detector_artifacts_cupy(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=output,
        roi_lower=(0, 0),
        roi_dim=(4, 4),
        tile_shape=(2, 2),
        drop_leading=0,
        chunk_frames=8,
        zero_offset_index=0,
        fit_trailing_drop=1,
        integrate_pixels=0,
        components=6,
        savgol_window=5,
        savgol_polyorder=3,
        amp_threshold=0.01,
        max_tiles=1,
    )

    amp_all = np.load(output / "amp_all.npy", mmap_mode="r")
    amp_sum = np.load(output / "amp_all_sum_filtered.npy", mmap_mode="r")
    fit_status = np.load(output / FIT_STATUS_FILE, mmap_mode="r")
    assert np.count_nonzero(amp_all[:2, :2, :]) > 0
    assert np.count_nonzero(amp_all[2:, :, :]) == 0
    assert np.count_nonzero(amp_all[:, 2:, :]) == 0
    assert np.count_nonzero(amp_sum[2:, :]) == 0
    assert np.count_nonzero(amp_sum[:, 2:]) == 0
    assert set(np.unique(fit_status[:2, :2])) == {FIT_STATUS_OK}
    assert set(np.unique(fit_status[2:, :])) == {FIT_STATUS_UNPROCESSED}
    assert set(np.unique(fit_status[:, 2:])) == {FIT_STATUS_UNPROCESSED}


def test_excluded_rows_remain_zero_in_detector_artifacts(tmp_path):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(tmp_path, samples=48, rows=4, cols=4)
    output = tmp_path / "artifacts"

    build_detector_artifacts_cupy(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=output,
        roi_lower=(0, 0),
        roi_dim=(4, 4),
        tile_shape=(2, 2),
        exclude_y=(AxisRange(1, 2),),
        drop_leading=0,
        chunk_frames=8,
        zero_offset_index=0,
        fit_trailing_drop=1,
        integrate_pixels=0,
        components=6,
        savgol_window=5,
        savgol_polyorder=3,
        amp_threshold=0.01,
    )

    amp_all = np.load(output / "amp_all.npy", mmap_mode="r")
    amp_sum = np.load(output / "amp_all_sum_filtered.npy", mmap_mode="r")
    fit_status = np.load(output / FIT_STATUS_FILE, mmap_mode="r")
    assert np.count_nonzero(amp_all[:, :, :]) > 0
    assert np.count_nonzero(amp_all[1, :, :]) == 0
    assert np.count_nonzero(amp_sum[1, :]) == 0
    assert set(np.unique(fit_status[1, :])) == {FIT_STATUS_SKIPPED}


def test_integrate_cupy_uses_detector_row_axis_only():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = cupy.asnumpy(_integrate_cupy(cupy, cupy.asarray(data), 1))
    left = np.zeros_like(data)
    left[:, 1:] = data[:, :-1]
    right = np.zeros_like(data)
    right[:, :-1] = data[:, 1:]

    np.testing.assert_allclose(result, data + left + right)


def test_excluded_signal_rows_do_not_feed_neighbor_integration():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    signal = np.array([[1.0, 1000.0, 10.0]], dtype=np.float64)
    _zero_excluded_signal_rows(
        signal,
        y0=0,
        y1=3,
        exclude_y=(AxisRange(1, 2),),
    )

    integrated = cupy.asnumpy(_integrate_cupy(cupy, cupy.asarray(signal), 1))

    np.testing.assert_allclose(signal, [[1.0, 0.0, 10.0]])
    np.testing.assert_allclose(integrated[:, 0], [1.0])
    np.testing.assert_allclose(integrated[:, 2], [10.0])


def test_row_halo_bounds_clip_to_detector_edges():
    assert _row_halo_bounds(16, 32, detector_height=64, pixels=3) == (
        13,
        35,
    )
    assert _row_halo_bounds(0, 16, detector_height=64, pixels=3) == (0, 19)
    assert _row_halo_bounds(48, 64, detector_height=64, pixels=3) == (45, 64)
    assert _row_halo_bounds(16, 32, detector_height=64, pixels=0) == (
        16,
        32,
    )


def test_full_detector_trace_uses_reference_unmasked_detector():
    on = np.ones((2, 3, 2), dtype=np.float64)
    off = np.ones_like(on)
    on[:, 1, :] = 1000.0

    trace = _full_detector_trace(
        _MemoryBlockReader(on, off),
        np.ones(2, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        detector_height=3,
        detector_width=2,
        frame_count=2,
        drop_leading=0,
        chunk_frames=1,
    )

    assert np.all(trace["ratio_minus_one"] > 0.0)


def test_load_tile_signal_uses_shifted_reference_denominator():
    on = np.full((1, 1, 2), 4.0, dtype=np.float64)
    off = np.full((1, 1, 2), 2.0, dtype=np.float64)

    signal = _load_tile_signal(
        _MemoryBlockReader(on, off),
        np.ones(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        shift=10.0,
        drop_leading=0,
        x0=0,
        x1=2,
        y0=0,
        y1=1,
    )

    np.testing.assert_allclose(signal, [[1.0 / 6.0]])


@pytest.mark.parametrize("sample_count", [8, 9])
def test_tdsfft_cupy_matches_numpy_frequency_bins(sample_count):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    sample_spacing = 0.25
    tone_bin = 2
    time = np.arange(sample_count, dtype=np.float64) * sample_spacing
    tone_frequency = tone_bin / (sample_count * sample_spacing)
    trace = np.cos(2.0 * np.pi * tone_frequency * time)

    frequency, value = _tdsfft_cupy(
        cupy,
        cupy.asarray(time),
        cupy.asarray(trace),
    )

    stop = (sample_count + 1) // 2
    expected_frequency = np.fft.fftfreq(
        sample_count,
        d=sample_spacing,
    )[:stop]
    expected_value = np.fft.fft(trace, sample_count)[:stop] / sample_count
    np.testing.assert_allclose(cupy.asnumpy(frequency), expected_frequency)
    np.testing.assert_allclose(
        cupy.asnumpy(value),
        expected_value,
        atol=1e-15,
    )
    assert int(cupy.asnumpy(cupy.argmax(cupy.abs(value)))) == tone_bin


def test_tdsfft_cupy_rejects_short_time_axis_case():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    with pytest.raises(ValueError, match="at least two time samples"):
        _tdsfft_cupy(
            cupy,
            cupy.asarray([0.0]),
            cupy.asarray([1.0]),
        )


class _MemoryBlockReader:
    def __init__(self, on: np.ndarray, off: np.ndarray) -> None:
        self._on = on
        self._off = off

    def read_pair(self, selection):
        return self._on[selection], self._off[selection]


def _write_synthetic_hdf5_pair(
    root: Path,
    *,
    samples: int,
    rows: int,
    cols: int,
) -> None:
    delay, trace_rows = synthetic_trace_batch(samples=samples, traces=rows)
    off = np.ones((samples, rows, cols), dtype=np.float64)
    on = np.ones_like(off)
    for row in range(rows):
        on[:, row, :] = 1.0 + 0.02 * trace_rows[row, :, None]

    for filename, data in (("on.h5", on), ("off.h5", off)):
        with h5py.File(root / filename, "w") as h5:
            h5.create_dataset("ROI", data=np.ones((rows, cols)))
            h5.create_dataset("bin_count", data=np.ones(samples))
            h5.create_dataset("i0", data=np.ones(samples))
            h5.create_dataset("i0_ipm3", data=np.ones(samples))
            h5.create_dataset("imgs", data=data)
            h5.create_dataset("scan_var", data=delay)
