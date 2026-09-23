# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import warnings
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xray.detector_artifacts import (
    DETECTOR_ARRAYS,
    DETECTOR_ARTIFACT_MANIFEST_VERSION,
    FIT_DIAGNOSTICS_FILE,
    FIT_STATUS_FAILED,
    FIT_STATUS_FILE,
    FIT_STATUS_OK,
    FIT_STATUS_SKIPPED,
    FIT_STATUS_UNPROCESSED,
    _clear_detector_artifact_outputs,
    _detector_artifact_config_hash,
    _ensure_cuda_device,
    _fit_detector_row,
    _fit_detector_rows_batched,
    _fit_error_types,
    _FitDiagnosticsWriter,
    _full_detector_trace,
    _integrate_cupy,
    _load_tile_signal,
    _publish_detector_artifact_outputs,
    _row_halo_bounds,
    _stable_json_hash,
    _tdsfft_cupy,
    _zero_excluded_signal_rows,
    build_detector_artifacts_cupy,
    compare_detector_artifacts,
    detector_artifact_complete,
    detector_artifact_resume_identity,
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


def test_detector_artifacts_rejects_unknown_fit_diagnostics_before_gpu(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="fit_diagnostics must be one of: none, summary, full",
    ):
        build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=tmp_path / "out",
            fit_diagnostics="unknown",
        )


@pytest.mark.parametrize("alpha", [-1.0, np.inf, -np.inf, np.nan])
def test_detector_artifacts_rejects_invalid_p2_ridge_before_gpu(
    tmp_path,
    alpha,
):
    with pytest.raises(
        ValueError,
        match="p2_ridge_alpha must be finite and non-negative",
    ):
        build_detector_artifacts_cupy(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=tmp_path / "out",
            p2_ridge_alpha=alpha,
        )


def test_fit_diagnostics_summary_preserves_tile_row_order(tmp_path):
    writer = _FitDiagnosticsWriter(tmp_path, "summary")
    writer.append(_diagnostic_record(x0=0, x1=5, y=2, modes=2))
    writer.append(_diagnostic_record(x0=0, x1=5, y=3, modes=1))
    writer.append(_diagnostic_record(x0=5, x1=10, y=2, modes=3))

    metadata = writer.finalize(source_identity_sha256="a" * 64)

    path = tmp_path / FIT_DIAGNOSTICS_FILE
    assert metadata["record_count"] == 3
    assert (
        metadata["artifact_sha256"] == sha256(path.read_bytes()).hexdigest()
    )
    with np.load(path, allow_pickle=False) as diagnostics:
        np.testing.assert_array_equal(diagnostics["tile_x_start"], [0, 0, 5])
        np.testing.assert_array_equal(diagnostics["detector_y"], [2, 3, 2])
        np.testing.assert_array_equal(
            diagnostics["selected_model_order"], [4, 4, 4]
        )
        np.testing.assert_array_equal(
            diagnostics["fit_status"], [FIT_STATUS_OK] * 3
        )
        assert "trace" not in diagnostics.files


def test_fit_diagnostics_full_serializes_ragged_arrays_without_pickle(
    tmp_path,
):
    writer = _FitDiagnosticsWriter(tmp_path, "full")
    writer.append(_diagnostic_record(x0=0, x1=5, y=0, modes=1, samples=3))
    writer.append(
        _diagnostic_record(
            x0=0,
            x1=5,
            y=1,
            modes=0,
            fit_status=FIT_STATUS_SKIPPED,
        )
    )
    writer.append(_diagnostic_record(x0=0, x1=5, y=2, modes=3, samples=3))

    metadata = writer.finalize(source_identity_sha256="b" * 64)

    assert metadata["array_lengths"]["time"] == 3
    assert metadata["array_lengths"]["trace"] == 6
    assert metadata["array_lengths"]["angular_frequency"] == 4
    with np.load(
        tmp_path / FIT_DIAGNOSTICS_FILE, allow_pickle=False
    ) as diagnostics:
        np.testing.assert_array_equal(diagnostics["time"], [0.0, 0.25, 0.5])
        np.testing.assert_array_equal(
            diagnostics["trace_offsets"], [0, 3, 3, 6]
        )
        np.testing.assert_array_equal(
            diagnostics["mode_offsets"], [0, 1, 1, 4]
        )
        np.testing.assert_array_equal(
            diagnostics["p1_singular_value_offsets"], [0, 2, 2, 6]
        )
        np.testing.assert_array_equal(
            diagnostics["p2_singular_value_offsets"], [0, 3, 3, 10]
        )
        np.testing.assert_array_equal(
            diagnostics["fit_status"],
            [FIT_STATUS_OK, FIT_STATUS_SKIPPED, FIT_STATUS_OK],
        )
        assert np.isnan(diagnostics["trace_std"][1])
        assert diagnostics["selected_model_order"][1] == -1
        assert all(
            not diagnostics[name].dtype.hasobject
            for name in diagnostics.files
        )


def test_fit_diagnostics_full_rejects_different_fitted_time_axes(tmp_path):
    writer = _FitDiagnosticsWriter(tmp_path, "full")
    writer.append(_diagnostic_record(x0=0, x1=2, y=0, modes=1))
    record = _diagnostic_record(x0=0, x1=2, y=1, modes=1)
    record["time"] = np.asarray(record["time"]) + 1.0

    with pytest.raises(ValueError, match="common fitted time axis"):
        writer.append(record)
    writer.close()


def test_detector_artifact_completeness_requires_v2_diagnostic_sidecar(
    tmp_path,
    monkeypatch,
):
    import cuphoton.xray.detector_artifacts as detector_artifacts

    shape = (1, 1, 2)
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        np.save(tmp_path / f"{name}.npy", np.ones(shape))
    np.save(tmp_path / "amp_all_sum_filtered.npy", np.ones((1, 1)))
    np.save(tmp_path / FIT_STATUS_FILE, np.ones((1, 1), dtype=np.uint8))
    input_identity = {"on": {"identity_sha256": "on"}}
    writer = _FitDiagnosticsWriter(tmp_path, "summary")
    writer.append(_diagnostic_record(x0=0, x1=1, y=0, modes=1))
    metadata = writer.finalize(
        source_identity_sha256=_stable_json_hash(input_identity)
    )
    manifest = {
        "kind": "xray-detector-artifacts",
        "manifest_schema_version": DETECTOR_ARTIFACT_MANIFEST_VERSION,
        "input_identity": input_identity,
        "output_shape": list(shape),
        "roi_dim": [1, 1],
        "raw_fits": 1,
        "skipped_fits": 0,
        "failures": 0,
        "p2_ridge_alpha": 0.0,
        "fit_diagnostics": metadata,
    }
    manifest["resume_identity"] = detector_artifact_resume_identity(manifest)
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
    metadata["artifact_resume_identity"] = manifest["resume_identity"]
    metadata["artifact_config_hash"] = manifest["config_hash"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        detector_artifacts,
        "_sha256_file",
        lambda _path: pytest.fail("resume completeness hashed diagnostics"),
    )
    assert detector_artifact_complete(tmp_path) is True
    artifact = tmp_path / FIT_DIAGNOSTICS_FILE
    artifact.write_bytes(artifact.read_bytes() + b"changed size")
    assert detector_artifact_complete(tmp_path) is False
    artifact.write_bytes(artifact.read_bytes()[: -len(b"changed size")])
    (tmp_path / FIT_DIAGNOSTICS_FILE).unlink()
    assert detector_artifact_complete(tmp_path) is False

    manifest["manifest_schema_version"] = 1
    manifest.pop("fit_diagnostics")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert detector_artifact_complete(tmp_path) is True


def test_clear_detector_artifact_outputs_removes_known_files(tmp_path):
    for name in DETECTOR_ARRAYS:
        (tmp_path / f"{name}.npy").write_bytes(b"old")
    (tmp_path / FIT_STATUS_FILE).write_bytes(b"old")
    (tmp_path / FIT_DIAGNOSTICS_FILE).write_bytes(b"old")
    (tmp_path / "manifest.json").write_text("old", encoding="utf-8")
    (tmp_path / "compare-reference.json").write_text("keep", encoding="utf-8")

    _clear_detector_artifact_outputs(tmp_path)

    for name in DETECTOR_ARRAYS:
        assert not (tmp_path / f"{name}.npy").exists()
    assert not (tmp_path / FIT_STATUS_FILE).exists()
    assert not (tmp_path / FIT_DIAGNOSTICS_FILE).exists()
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
    (source / FIT_DIAGNOSTICS_FILE).write_bytes(b"new diagnostics")
    (target / "manifest.json").write_text("old", encoding="utf-8")
    (target / FIT_DIAGNOSTICS_FILE).write_bytes(b"old diagnostics")
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
    assert (target / FIT_DIAGNOSTICS_FILE).read_bytes() == b"new diagnostics"
    assert (target / "compare-reference.json").read_text(
        encoding="utf-8"
    ) == "keep"


def test_publish_detector_artifacts_leaves_no_manifest_when_sidecar_fails(
    tmp_path,
    monkeypatch,
):
    import cuphoton.xray.detector_artifacts as detector_artifacts

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for name in DETECTOR_ARRAYS:
        np.save(source / f"{name}.npy", np.ones((1,)))
    np.save(source / FIT_STATUS_FILE, np.ones((1,), dtype=np.uint8))
    (source / FIT_DIAGNOSTICS_FILE).write_bytes(b"diagnostics")
    (source / "manifest.json").write_text("new", encoding="utf-8")
    (target / "manifest.json").write_text("old", encoding="utf-8")
    real_move = detector_artifacts.shutil.move

    def fail_sidecar(source_path, target_path):
        if Path(source_path).name == FIT_DIAGNOSTICS_FILE:
            raise OSError("interrupted sidecar publication")
        return real_move(source_path, target_path)

    monkeypatch.setattr(detector_artifacts.shutil, "move", fail_sidecar)

    with pytest.raises(OSError, match="interrupted sidecar publication"):
        _publish_detector_artifact_outputs(source, target)
    assert not (target / "manifest.json").exists()


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
            fit_diagnostics="full",
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
        fit_diagnostics="summary",
        hdf5_reader="hdf5-ts-funcwrap",
        hdf5_reader_workers=2,
    )

    assert result.shape == (4, 4, 24)
    assert result.processed_tiles == 4
    assert result.batched_tiles == 0
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
    assert manifest["manifest_schema_version"] == 2
    assert manifest["p2_ridge_alpha"] == 0.0
    assert manifest["fit_diagnostics"]["level"] == "summary"
    assert manifest["fit_diagnostics"]["record_count"] == 8
    assert (
        manifest["fit_diagnostics"]["artifact_resume_identity"]
        == manifest["resume_identity"]
    )
    assert (
        manifest["fit_diagnostics"]["artifact_config_hash"]
        == manifest["config_hash"]
    )
    freq_all = np.load(output / "freq_all.npy", mmap_mode="r")
    amp_all = np.load(output / "amp_all.npy", mmap_mode="r")
    amp_sum = np.load(output / "amp_all_sum_filtered.npy", mmap_mode="r")
    fit_status = np.load(output / FIT_STATUS_FILE, mmap_mode="r")
    assert freq_all.shape == (4, 4, 24)
    assert amp_all.shape == (4, 4, 24)
    assert amp_sum.shape == (4, 4)
    assert fit_status.shape == (4, 4)
    assert set(np.unique(fit_status)) == {FIT_STATUS_OK}
    with np.load(output / FIT_DIAGNOSTICS_FILE, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["fit_status"], [FIT_STATUS_OK] * 8)
        np.testing.assert_array_equal(data["tile_x_start"], [0] * 4 + [2] * 4)
        np.testing.assert_array_equal(data["detector_y"], [0, 1, 2, 3] * 2)
    assert np.count_nonzero(amp_all) > 0
    np.testing.assert_allclose(freq_all[:, 0, :], freq_all[:, 1, :])


@pytest.mark.parametrize(
    "identity",
    [_detector_artifact_config_hash, detector_artifact_resume_identity],
)
def test_detector_artifact_identity_distinguishes_serial_optout(identity):
    legacy = {"kind": "xray-detector-artifacts", "manifest_schema_version": 2}
    assert identity(legacy) == identity({**legacy, "batch_rows": True})
    assert identity(legacy) != identity({**legacy, "batch_rows": False})


def test_detector_artifacts_batches_default_rows_and_allows_optout(
    tmp_path, monkeypatch
):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(tmp_path, samples=48, rows=4, cols=4)
    batch_calls = []
    row_calls = []

    def counted_batch(**kwargs):
        batch_calls.append(
            tuple(int(value) for value in kwargs["traces_gpu"].shape)
        )
        return _fit_detector_rows_batched(**kwargs)

    def counted_row(**kwargs):
        row_calls.append(1)
        return _fit_detector_row(**kwargs)

    monkeypatch.setattr(
        "cuphoton.xray.detector_artifacts._fit_detector_rows_batched",
        counted_batch,
    )
    monkeypatch.setattr(
        "cuphoton.xray.detector_artifacts._fit_detector_row",
        counted_row,
    )
    result = _build_synthetic_detector_artifacts(
        tmp_path,
        tmp_path / "artifacts",
    )

    # A batch that raised would fall back and still report four raw fits,
    # so the row-path call count is what tells the two apart.
    assert batch_calls == [(4, 47)]
    assert row_calls == []
    assert result.batched_tiles == 1
    assert result.raw_fits == 4
    assert result.failures == 0
    manifest = json.loads(
        (tmp_path / "artifacts" / "manifest.json").read_text()
    )
    assert manifest["batched_tiles"] == 1
    assert manifest["batch_rows"] is True

    batch_calls.clear()
    serial = _build_synthetic_detector_artifacts(
        tmp_path,
        tmp_path / "serial",
        batch_rows=False,
    )
    assert batch_calls == []
    assert len(row_calls) == 4
    assert serial.batched_tiles == 0
    assert serial.raw_fits == result.raw_fits
    assert serial.failures == 0
    _assert_detector_outputs_match(
        _load_detector_outputs(tmp_path / "serial"),
        _load_detector_outputs(tmp_path / "artifacts"),
    )
    serial_manifest = json.loads(
        (tmp_path / "serial" / "manifest.json").read_text()
    )
    assert serial_manifest["batch_rows"] is False
    assert serial_manifest["batched_tiles"] == 0
    assert serial_manifest["config_hash"] != manifest["config_hash"]
    assert serial_manifest["resume_identity"] != manifest["resume_identity"]
    assert detector_artifact_complete(
        tmp_path / "serial",
        expected_resume_identity=serial_manifest["resume_identity"],
    )
    assert not detector_artifact_complete(
        tmp_path / "serial",
        expected_resume_identity=manifest["resume_identity"],
    )


@pytest.mark.parametrize("batch_failure", ["exception", "all_rows_failed"])
def test_detector_artifacts_falls_back_when_row_batch_fails(
    tmp_path,
    monkeypatch,
    batch_failure,
):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(tmp_path, samples=48, rows=4, cols=4)
    batched = _build_synthetic_detector_artifacts(
        tmp_path,
        tmp_path / "batched",
    )
    assert batched.raw_fits == 4
    batch_calls = []
    row_calls = []

    def reject_batch(**kwargs):
        batch_calls.append(
            tuple(int(value) for value in kwargs["traces_gpu"].shape)
        )
        if batch_failure == "exception":
            raise ValueError("synthetic batch failure")
        return [None] * len(kwargs["traces_gpu"])

    def counted_row(**kwargs):
        row_calls.append(1)
        return _fit_detector_row(**kwargs)

    monkeypatch.setattr(
        "cuphoton.xray.detector_artifacts._fit_detector_rows_batched",
        reject_batch,
    )
    monkeypatch.setattr(
        "cuphoton.xray.detector_artifacts._fit_detector_row",
        counted_row,
    )
    result = _build_synthetic_detector_artifacts(
        tmp_path,
        tmp_path / "fallback",
    )

    assert batch_calls == [(4, 47)]
    assert len(row_calls) == 4
    assert result.batched_tiles == 0
    assert result.raw_fits == 4
    assert result.failures == 0
    # Falling back must not change what the run writes.
    _assert_detector_outputs_match(
        _load_detector_outputs(tmp_path / "fallback"),
        _load_detector_outputs(tmp_path / "batched"),
    )


def test_detector_artifacts_refit_only_rows_that_fail_in_batch(
    tmp_path,
    monkeypatch,
):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    _write_synthetic_hdf5_pair(
        tmp_path,
        samples=48,
        rows=4,
        cols=4,
        dead_rows=(2,),
    )
    batch_calls = []
    row_calls = []

    def counted_batch(**kwargs):
        batch_calls.append(
            tuple(int(value) for value in kwargs["traces_gpu"].shape)
        )
        return _fit_detector_rows_batched(**kwargs)

    def counted_row(**kwargs):
        row_calls.append(1)
        return _fit_detector_row(**kwargs)

    monkeypatch.setattr(
        "cuphoton.xray.detector_artifacts._fit_detector_rows_batched",
        counted_batch,
    )
    monkeypatch.setattr(
        "cuphoton.xray.detector_artifacts._fit_detector_row",
        counted_row,
    )
    result = _build_synthetic_detector_artifacts(
        tmp_path,
        tmp_path / "batched",
        max_fit_failures=1,
    )

    # Only the dead row leaves the batch and takes the serial path.
    assert batch_calls == [(4, 47)]
    assert len(row_calls) == 1
    assert result.batched_tiles == 1
    assert result.raw_fits == 3
    assert result.failures == 1
    batched = _load_detector_outputs(tmp_path / "batched")
    fit_status = batched[FIT_STATUS_FILE]
    assert set(np.unique(fit_status[2, :])) == {FIT_STATUS_FAILED}
    assert set(np.unique(np.delete(fit_status, 2, axis=0))) == {FIT_STATUS_OK}

    def reject_batch(**kwargs):
        raise ValueError("synthetic batch failure")

    monkeypatch.setattr(
        "cuphoton.xray.detector_artifacts._fit_detector_rows_batched",
        reject_batch,
    )
    fallback = _build_synthetic_detector_artifacts(
        tmp_path,
        tmp_path / "fallback",
        max_fit_failures=1,
    )

    assert fallback.raw_fits == 3
    assert fallback.failures == 1
    _assert_detector_outputs_match(
        batched,
        _load_detector_outputs(tmp_path / "fallback"),
    )


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
        p2_ridge_alpha=0.01,
        fit_diagnostics="full",
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
    with np.load(output / FIT_DIAGNOSTICS_FILE, allow_pickle=False) as data:
        assert data["fit_status"].shape == (2,)
        assert data["time"].shape == (47,)
        np.testing.assert_array_equal(data["trace_offsets"], [0, 47, 94])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["p2_ridge_alpha"] == 0.01


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
        fit_diagnostics="summary",
    )

    amp_all = np.load(output / "amp_all.npy", mmap_mode="r")
    amp_sum = np.load(output / "amp_all_sum_filtered.npy", mmap_mode="r")
    fit_status = np.load(output / FIT_STATUS_FILE, mmap_mode="r")
    assert np.count_nonzero(amp_all[:, :, :]) > 0
    assert np.count_nonzero(amp_all[1, :, :]) == 0
    assert np.count_nonzero(amp_sum[1, :]) == 0
    assert set(np.unique(fit_status[1, :])) == {FIT_STATUS_SKIPPED}
    with np.load(output / FIT_DIAGNOSTICS_FILE, allow_pickle=False) as data:
        assert data["fit_status"].shape == (8,)
        assert np.count_nonzero(data["fit_status"] == FIT_STATUS_SKIPPED) == 2
        skipped = data["fit_status"] == FIT_STATUS_SKIPPED
        assert np.all(np.isnan(data["relative_residual"][skipped]))
        assert np.all(data["selected_model_order"][skipped] == -1)


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


def test_tdsfft_cupy_batches_last_axis():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    time, traces = synthetic_trace_batch(samples=32, traces=3)
    frequency, values = _tdsfft_cupy(
        cupy,
        cupy.asarray(time),
        cupy.asarray(traces),
    )

    stop = (traces.shape[1] + 1) // 2
    np.testing.assert_allclose(
        cupy.asnumpy(frequency),
        np.fft.fftfreq(traces.shape[1], d=time[1] - time[0])[:stop],
    )
    np.testing.assert_allclose(
        cupy.asnumpy(values),
        np.fft.fft(traces, axis=-1)[..., :stop] / traces.shape[1],
        atol=1e-15,
    )


def test_fit_detector_rows_batched_matches_serial_rows():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    time, traces = synthetic_trace_batch(samples=48, traces=3)
    # Rows that select no mode (growing exponential), only a zero-frequency
    # mode (pure decay, which zeroes the spectral grid), and a constant
    # exercise the padded active-mask path with fewer modes than the batch
    # maximum.
    traces = np.concatenate(
        [
            traces,
            np.stack(
                [
                    np.exp(0.08 * time),
                    0.8 * np.exp(-0.1 * time),
                    np.full_like(time, 0.3),
                ]
            ),
        ]
    )
    gpu_time = cupy.asarray(time)
    gpu_traces = cupy.asarray(traces)
    expected = tuple(
        _fit_detector_row(
            cp=cupy,
            time_gpu=gpu_time,
            trace_gpu=trace,
            components=6,
            roots_backend="eigvals",
            padded_length=24,
        )
        for trace in gpu_traces
    )

    actual = _fit_detector_rows_batched(
        cp=cupy,
        time_gpu=gpu_time,
        traces_gpu=gpu_traces,
        components=6,
        padded_length=24,
    )

    assert not np.any(expected[3]["amp"])
    assert expected[4]["amp"][0] > 0.0
    assert not np.any(expected[4]["freq"])
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for name in ("freq", "amp", "fft_freq"):
            np.testing.assert_array_equal(
                actual_row[name],
                expected_row[name],
            )
        # The batched cuFFT plan rounds differently from per-row FFTs.
        np.testing.assert_allclose(
            actual_row["fft"],
            expected_row["fft"],
            rtol=0.0,
            atol=1e-15,
        )


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


def _diagnostic_record(
    *,
    x0: int,
    x1: int,
    y: int,
    modes: int,
    samples: int = 4,
    fit_status: int = FIT_STATUS_OK,
) -> dict[str, object]:
    if fit_status != FIT_STATUS_OK:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "tile_x_start": x0,
            "tile_x_stop": x1,
            "tile_y_start": y,
            "tile_y_stop": y + 1,
            "detector_y": y,
            "fit_status": fit_status,
            "trace_std": None,
            "residual_std": None,
            "relative_residual": None,
            "chi2": None,
            "selected_model_order": -1,
            "mode_count": -1,
            "p1_rank": -1,
            "p1_singular_value_ratio": None,
            "p1_condition": None,
            "p2_rank": -1,
            "p2_singular_value_ratio": None,
            "p2_condition": None,
            "max_amplitude": None,
            "trace": empty,
            "reconstruction": empty,
            "angular_frequency": empty,
            "decay": empty,
            "amplitude": empty,
            "phase": empty,
            "p1_singular_values": empty,
            "p2_singular_values": empty,
        }
    mode_values = np.arange(modes, dtype=np.float64)
    trace = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    return {
        "tile_x_start": x0,
        "tile_x_stop": x1,
        "tile_y_start": y,
        "tile_y_stop": y + 1,
        "detector_y": y,
        "fit_status": FIT_STATUS_OK,
        "trace_std": 2.0,
        "residual_std": 0.5,
        "relative_residual": 0.25,
        "chi2": 0.125,
        "selected_model_order": 4,
        "mode_count": modes,
        "p1_rank": 4,
        "p1_singular_value_ratio": 0.01,
        "p1_condition": 100.0,
        "p2_rank": 2 * modes + 1,
        "p2_singular_value_ratio": 0.1,
        "p2_condition": 10.0,
        "max_amplitude": 3.0,
        "time": np.arange(samples, dtype=np.float64) * 0.25,
        "trace": trace,
        "reconstruction": trace * 0.9,
        "angular_frequency": mode_values + 1.0,
        "decay": mode_values + 0.5,
        "amplitude": mode_values + 2.0,
        "phase": mode_values * 0.1,
        "p1_singular_values": np.arange(modes + 1, dtype=np.float64),
        "p2_singular_values": np.arange(2 * modes + 1, dtype=np.float64),
    }


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
    dead_rows: tuple[int, ...] = (),
) -> None:
    delay, trace_rows = synthetic_trace_batch(samples=samples, traces=rows)
    off = np.ones((samples, rows, cols), dtype=np.float64)
    on = np.ones_like(off)
    for row in range(rows):
        on[:, row, :] = 1.0 + 0.02 * trace_rows[row, :, None]
    for row in dead_rows:
        # No on/off difference leaves an all-zero trace, which P1 rejects
        # with "insufficient finite singular values".
        on[:, row, :] = off[:, row, :]

    for filename, data in (("on.h5", on), ("off.h5", off)):
        with h5py.File(root / filename, "w") as h5:
            h5.create_dataset("ROI", data=np.ones((rows, cols)))
            h5.create_dataset("bin_count", data=np.ones(samples))
            h5.create_dataset("i0", data=np.ones(samples))
            h5.create_dataset("i0_ipm3", data=np.ones(samples))
            h5.create_dataset("imgs", data=data)
            h5.create_dataset("scan_var", data=delay)


def _build_synthetic_detector_artifacts(
    root: Path,
    output_dir: Path,
    **overrides,
):
    options = {
        "h5dir": root,
        "fon": "on.h5",
        "foff": "off.h5",
        "output_dir": output_dir,
        "roi_lower": (0, 0),
        "roi_dim": (4, 4),
        "tile_shape": (4, 4),
        "drop_leading": 0,
        "chunk_frames": 8,
        "zero_offset_index": 0,
        "fit_trailing_drop": 1,
        "integrate_pixels": 0,
        "components": 6,
        "savgol_window": 5,
        "savgol_polyorder": 3,
        "amp_threshold": 0.01,
    }
    options.update(overrides)
    return build_detector_artifacts_cupy(**options)


def _load_detector_outputs(output: Path) -> dict[str, np.ndarray]:
    arrays = {
        name: np.load(output / f"{name}.npy") for name in DETECTOR_ARRAYS
    }
    arrays[FIT_STATUS_FILE] = np.load(output / FIT_STATUS_FILE)
    return arrays


def _assert_detector_outputs_match(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    for name, values in expected.items():
        if name == "fft_all":
            # The batched cuFFT plan rounds differently from per-row FFTs.
            np.testing.assert_allclose(
                actual[name],
                values,
                rtol=0.0,
                atol=1e-15,
            )
        else:
            np.testing.assert_array_equal(actual[name], values)
