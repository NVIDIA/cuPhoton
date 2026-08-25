# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

import cuphoton.xray.detector_distributed as detector_distributed
from cuphoton import __version__
from cuphoton.core.cli import run_component
from cuphoton.xray.detector_artifacts import (
    FIT_DIAGNOSTICS_FILE,
    FIT_STATUS_FAILED,
    FIT_STATUS_FILE,
    FIT_STATUS_OK,
    _append_non_ok_fit_diagnostic,
    _detector_artifact_config_hash,
    _FitDiagnosticsWriter,
    _stable_json_hash,
    detector_artifact_complete,
    detector_artifact_input_identity,
    detector_artifact_resume_identity,
    merge_detector_artifact_shards,
)
from cuphoton.xray.detector_distributed import (
    build_detector_artifact_distributed_plan,
    render_local_shell_script,
)


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def test_detector_artifact_distributed_dry_run_cli_json(tmp_path, capsys):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=4, cols=10)

    assert (
        main(
            [
                "detector-artifact-distributed",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "on.h5",
                "--foff",
                "off.h5",
                "--output-dir",
                str(tmp_path / "out"),
                "--roi-dim",
                "10",
                "4",
                "--tile-shape",
                "4",
                "2",
                "--shard-count",
                "3",
                "--gpus",
                "2",
                "--run-label",
                "dry",
                "--fit-diagnostics",
                "full",
                "--p2-ridge-alpha",
                "0.01",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["executor"] == "dry-run"
    assert payload["global_roi_dim"] == [10, 4]
    assert payload["shard_count"] == 3
    assert [item["roi_dim"] for item in payload["shards"]] == [
        [4, 4],
        [4, 4],
        [2, 4],
    ]
    assert "--shard-index 1" in payload["slurm_script"]
    assert payload["detector_options"]["fit_diagnostics"] == "full"
    assert payload["detector_options"]["p2_ridge_alpha"] == 0.01
    assert "--fit-diagnostics full" in payload["slurm_script"]
    assert "--p2-ridge-alpha 0.01" in payload["slurm_script"]
    assert payload["commands"][0][-1] == "--json"


def test_build_distributed_plan_uses_shard_width(tmp_path):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=3, cols=9)

    plan = build_detector_artifact_distributed_plan(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=tmp_path / "out",
        roi_lower=(1, 0),
        roi_dim=(8, 3),
        tile_shape=(4, 2),
        shard_width=4,
        gpus=2,
        run_label="width",
    )

    assert plan["global_roi_lower"] == [1, 0]
    assert [item["roi_lower"] for item in plan["shards"]] == [[1, 0], [5, 0]]
    assert [item["roi_dim"] for item in plan["shards"]] == [[4, 3], [4, 3]]
    command = plan["commands"][1]
    assert command[command.index("--roi-lower") + 1] == "5"


def test_distributed_resume_identity_includes_diagnostics_and_ridge(tmp_path):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=2, cols=2)
    common = {
        "h5dir": tmp_path,
        "fon": "on.h5",
        "foff": "off.h5",
        "output_dir": tmp_path / "out",
        "roi_dim": (2, 2),
        "shard_count": 1,
        "gpus": 1,
        "run_label": "diagnostics",
    }
    baseline = build_detector_artifact_distributed_plan(**common)
    summary = build_detector_artifact_distributed_plan(
        **common,
        detector_options={"fit_diagnostics": "summary"},
    )
    ridge = build_detector_artifact_distributed_plan(
        **common,
        detector_options={
            "fit_diagnostics": "summary",
            "p2_ridge_alpha": 0.01,
        },
    )

    identities = {
        plan["shards"][0]["resume_identity"]
        for plan in (baseline, summary, ridge)
    }
    assert len(identities) == 3
    assert "--fit-diagnostics" not in baseline["commands"][0]
    assert "--p2-ridge-alpha" not in baseline["commands"][0]
    assert "--fit-diagnostics" in summary["commands"][0]
    assert "--p2-ridge-alpha" in ridge["commands"][0]


@pytest.mark.parametrize("alpha", [-1.0, np.inf, np.nan])
def test_distributed_plan_rejects_invalid_p2_ridge_alpha(tmp_path, alpha):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=2, cols=2)

    with pytest.raises(
        ValueError,
        match="p2_ridge_alpha must be finite and non-negative",
    ):
        build_detector_artifact_distributed_plan(
            h5dir=tmp_path,
            fon="on.h5",
            foff="off.h5",
            output_dir=tmp_path / "out",
            roi_dim=(2, 2),
            shard_count=1,
            detector_options={"p2_ridge_alpha": alpha},
        )


@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("2,5", ("2", "5")),
        (
            "GPU-12345678-1234-1234-1234-123456789abc,"
            "GPU-abcdefab-cdef-cdef-cdef-abcdefabcdef",
            (
                "GPU-12345678-1234-1234-1234-123456789abc",
                "GPU-abcdefab-cdef-cdef-cdef-abcdefabcdef",
            ),
        ),
    ],
)
def test_visible_cuda_device_tokens_preserve_inherited_tokens(
    visible, expected
):
    assert (
        detector_distributed._visible_cuda_device_tokens(
            2, {"CUDA_VISIBLE_DEVICES": visible}
        )
        == expected
    )


@pytest.mark.parametrize("visible", ["", "   ", "-1"])
def test_visible_cuda_device_tokens_reject_empty_visibility(visible):
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES is empty"):
        detector_distributed._visible_cuda_device_tokens(
            1, {"CUDA_VISIBLE_DEVICES": visible}
        )


def test_visible_cuda_device_tokens_reject_impossible_allocation():
    with pytest.raises(ValueError, match="requested 2 GPUs but only 1"):
        detector_distributed._visible_cuda_device_tokens(
            2, {"CUDA_VISIBLE_DEVICES": "GPU-only"}
        )


def test_rendered_local_script_indexes_inherited_visibility(tmp_path):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=2, cols=4)
    plan = build_detector_artifact_distributed_plan(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=tmp_path / "out",
        roi_dim=(4, 2),
        shard_count=2,
        gpus=2,
        run_label="visibility",
    )

    script = render_local_shell_script(plan, merge=False, resume=False)

    assert "CUDA_DEVICE_TOKENS" in script
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_TOKENS[0]}"' in script
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_TOKENS[1]}"' in script
    assert "CUDA_VISIBLE_DEVICES=0 " not in script
    assert "-m cuphoton xray detector-artifacts" in script
    assert "cuphoton.xray.cli" not in script
    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_local_executor_assigns_inherited_uuid_tokens(tmp_path, monkeypatch):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=2, cols=4)
    plan = build_detector_artifact_distributed_plan(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=tmp_path / "out",
        roi_dim=(4, 2),
        shard_count=2,
        gpus=2,
        run_label="uuid-local",
    )
    tokens = ("GPU-first", "GPU-second")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(tokens))
    seen: list[str] = []

    def fake_run(command, shard, *, device, work_dir, resume):
        seen.append(device)
        return {
            "index": shard["index"],
            "returncode": 0,
            "skipped": False,
        }

    monkeypatch.setattr(detector_distributed, "_run_shard_command", fake_run)

    payload = detector_distributed._run_local_plan(
        plan, merge=False, resume=False, submit=True
    )

    assert payload["returncode"] == 0
    assert sorted(seen) == sorted(tokens)


def test_local_executor_rejects_empty_inherited_visibility(
    tmp_path, monkeypatch
):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=2, cols=2)
    plan = build_detector_artifact_distributed_plan(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=tmp_path / "out",
        roi_dim=(2, 2),
        shard_count=1,
        gpus=1,
        run_label="empty-local",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES is empty"):
        detector_distributed._run_local_plan(
            plan, merge=False, resume=False, submit=True
        )


def test_resume_requires_current_input_and_option_identity(tmp_path):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=2, cols=2)
    common = {
        "h5dir": tmp_path,
        "fon": "on.h5",
        "foff": "off.h5",
        "output_dir": tmp_path / "out",
        "roi_dim": (2, 2),
        "shard_count": 1,
        "gpus": 1,
        "run_label": "resume",
    }
    old_plan = build_detector_artifact_distributed_plan(
        **common,
        detector_options={"components": 2},
    )
    new_plan = build_detector_artifact_distributed_plan(
        **common,
        detector_options={"components": 3},
    )
    old_identity = old_plan["shards"][0]["resume_identity"]
    new_identity = new_plan["shards"][0]["resume_identity"]
    assert old_identity != new_identity

    shard_dir = _write_shard(
        Path(old_plan["shards"][0]["output_dir"]),
        index=0,
        count=1,
        roi_lower=(0, 0),
        roi_dim=(2, 2),
        global_roi_dim=(2, 2),
        fill=1.0,
        resume_identity=old_identity,
    )

    assert detector_artifact_complete(
        shard_dir, expected_resume_identity=old_identity
    )
    assert not detector_artifact_complete(
        shard_dir, expected_resume_identity=new_identity
    )


def test_input_identity_redacts_paths_and_changes_with_content(tmp_path):
    on_path = tmp_path / "private-on.h5"
    off_path = tmp_path / "private-off.h5"
    on_path.write_bytes(b"on-v1")
    off_path.write_bytes(b"off-v1")

    before = detector_artifact_input_identity(on_path, off_path)
    encoded = json.dumps(before, sort_keys=True)
    assert str(tmp_path.resolve()) not in encoded

    on_path.write_bytes(b"on-v2")
    after = detector_artifact_input_identity(on_path, off_path)
    assert before["on"]["identity_sha256"] != after["on"]["identity_sha256"]


def test_detector_artifact_normalize_cli_writes_cache(tmp_path, capsys):
    _write_synthetic_hdf5_pair(tmp_path, samples=8, rows=3, cols=3)

    assert (
        main(
            [
                "detector-artifact-normalize",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "on.h5",
                "--foff",
                "off.h5",
                "--output-dir",
                str(tmp_path / "norm"),
                "--drop-leading",
                "0",
                "--zero-offset-index",
                "0",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["detector_shape"] == [3, 3]
    assert payload["zero_offset_index"] == 0
    assert (tmp_path / "norm" / "normalization.json").exists()
    assert (tmp_path / "norm" / "normalization.npz").exists()
    manifest = json.loads(
        (tmp_path / "norm" / "normalization.json").read_text()
    )
    assert manifest["kind"] == "xray-detector-artifact-normalization"
    assert manifest["sample_count"] == 8
    assert manifest["manifest_schema_version"] == 1
    assert "input_identity" in manifest
    assert "h5dir" not in manifest
    assert "fon" not in manifest
    assert "foff" not in manifest
    assert str(tmp_path.resolve()) not in json.dumps(manifest)


def test_merge_detector_artifact_shards(tmp_path):
    left = _write_shard(
        tmp_path / "left",
        index=0,
        count=2,
        roi_lower=(0, 0),
        roi_dim=(3, 2),
        global_roi_dim=(5, 2),
        fill=1.0,
    )
    right = _write_shard(
        tmp_path / "right",
        index=1,
        count=2,
        roi_lower=(3, 0),
        roi_dim=(2, 2),
        global_roi_dim=(5, 2),
        fill=2.0,
    )

    result = merge_detector_artifact_shards(
        shard_dirs=(right, left),
        output_dir=tmp_path / "merged",
    )

    assert result.shape == (2, 5, 4)
    assert detector_artifact_complete(tmp_path / "merged") is True
    amp = np.load(tmp_path / "merged" / "amp_all.npy")
    np.testing.assert_allclose(amp[:, :3, :], 1.0)
    np.testing.assert_allclose(amp[:, 3:, :], 2.0)
    manifest = json.loads((tmp_path / "merged" / "manifest.json").read_text())
    assert manifest["artifact_role"] == "merged-shards"
    assert manifest["roi_lower"] == [0, 0]
    assert manifest["roi_dim"] == [5, 2]
    assert manifest["raw_fits"] == 4
    assert "path" not in manifest["merged_from_shards"][0]
    assert "manifest_path" not in manifest["merged_from_shards"][0]
    assert str(tmp_path.resolve()) not in json.dumps(manifest)
    assert [item["index"] for item in manifest["shard_runtimes"]] == [0, 1]


def test_merge_detector_artifact_shards_combines_diagnostic_records(tmp_path):
    left = _upgrade_shard_with_summary_diagnostics(
        _write_shard(
            tmp_path / "left-diagnostics",
            index=0,
            count=2,
            roi_lower=(0, 0),
            roi_dim=(3, 2),
            global_roi_dim=(5, 2),
            fill=1.0,
        )
    )
    right = _upgrade_shard_with_summary_diagnostics(
        _write_shard(
            tmp_path / "right-diagnostics",
            index=1,
            count=2,
            roi_lower=(3, 0),
            roi_dim=(2, 2),
            global_roi_dim=(5, 2),
            fill=2.0,
        )
    )

    merge_detector_artifact_shards(
        shard_dirs=(right, left),
        output_dir=tmp_path / "merged-diagnostics",
    )

    merged = tmp_path / "merged-diagnostics"
    manifest = json.loads((merged / "manifest.json").read_text())
    assert manifest["fit_diagnostics"]["record_count"] == 4
    assert detector_artifact_complete(merged) is True
    with np.load(merged / FIT_DIAGNOSTICS_FILE, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["tile_x_start"], [0, 0, 3, 3])
        np.testing.assert_array_equal(data["detector_y"], [0, 1, 0, 1])


def test_merge_detector_artifact_shards_combines_full_diagnostics(tmp_path):
    left = _upgrade_shard_with_full_diagnostics(
        _write_shard(
            tmp_path / "left-full-diagnostics",
            index=0,
            count=2,
            roi_lower=(0, 0),
            roi_dim=(3, 2),
            global_roi_dim=(5, 2),
            fill=1.0,
        )
    )
    right = _upgrade_shard_with_full_diagnostics(
        _write_shard(
            tmp_path / "right-full-diagnostics",
            index=1,
            count=2,
            roi_lower=(3, 0),
            roi_dim=(2, 2),
            global_roi_dim=(5, 2),
            fill=2.0,
        )
    )

    merge_detector_artifact_shards(
        shard_dirs=(right, left),
        output_dir=tmp_path / "merged-full-diagnostics",
    )

    merged = tmp_path / "merged-full-diagnostics"
    manifest = json.loads((merged / "manifest.json").read_text())
    assert manifest["fit_diagnostics"]["record_count"] == 4
    assert manifest["fit_diagnostics"]["array_lengths"] == {
        "amplitude": 3,
        "angular_frequency": 3,
        "decay": 3,
        "p1_singular_values": 5,
        "p2_singular_values": 8,
        "phase": 3,
        "reconstruction": 6,
        "time": 3,
        "trace": 6,
    }
    assert detector_artifact_complete(merged) is True
    with np.load(merged / FIT_DIAGNOSTICS_FILE, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["time"], [0.0, 0.25, 0.5])
        np.testing.assert_array_equal(
            data["fit_status"],
            [FIT_STATUS_OK, FIT_STATUS_FAILED] * 2,
        )
        np.testing.assert_array_equal(data["trace_offsets"], [0, 3, 3, 6, 6])
        np.testing.assert_array_equal(data["mode_offsets"], [0, 1, 1, 3, 3])
        np.testing.assert_array_equal(
            data["p1_singular_value_offsets"], [0, 2, 2, 5, 5]
        )
        np.testing.assert_array_equal(
            data["p2_singular_value_offsets"], [0, 3, 3, 8, 8]
        )
        np.testing.assert_array_equal(
            data["trace"], [1.0, 1.25, 1.5, 4.0, 4.25, 4.5]
        )
        np.testing.assert_array_equal(
            data["angular_frequency"], [1.0, 1.0, 2.0]
        )
        np.testing.assert_array_equal(
            data["p1_singular_values"], [0.0, 1.0, 0.0, 1.0, 2.0]
        )
        np.testing.assert_array_equal(
            data["p2_singular_values"],
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        )


def test_merge_rejects_shards_from_different_inputs(tmp_path):
    left = _write_shard(
        tmp_path / "left",
        index=0,
        count=2,
        roi_lower=(0, 0),
        roi_dim=(1, 2),
        global_roi_dim=(2, 2),
        fill=1.0,
        input_identity="dataset-a",
    )
    right = _write_shard(
        tmp_path / "right",
        index=1,
        count=2,
        roi_lower=(1, 0),
        roi_dim=(1, 2),
        global_roi_dim=(2, 2),
        fill=2.0,
        input_identity="dataset-b",
    )

    with pytest.raises(ValueError, match="configuration mismatch"):
        merge_detector_artifact_shards(
            shard_dirs=(left, right),
            output_dir=tmp_path / "merged",
        )


def test_merge_rejects_shard_missing_identity_fields(tmp_path):
    shard = _write_shard(
        tmp_path / "shard",
        index=0,
        count=1,
        roi_lower=(0, 0),
        roi_dim=(2, 2),
        global_roi_dim=(2, 2),
        fill=1.0,
    )
    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["input_identity"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing identity fields"):
        merge_detector_artifact_shards(
            shard_dirs=(shard,), output_dir=tmp_path / "merged"
        )


def test_merge_rejects_invalid_shard_config_hash(tmp_path):
    shard = _write_shard(
        tmp_path / "shard",
        index=0,
        count=1,
        roi_lower=(0, 0),
        roi_dim=(2, 2),
        global_roi_dim=(2, 2),
        fill=1.0,
    )
    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="config_hash mismatch"):
        merge_detector_artifact_shards(
            shard_dirs=(shard,), output_dir=tmp_path / "merged"
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        (
            "p2_ridge_alpha",
            1.0,
            "fit diagnostics artifact resume identity mismatch",
        ),
        (
            "normalization_shift",
            2.0,
            "fit diagnostics artifact config hash mismatch",
        ),
    ),
)
def test_merge_rejects_diagnostics_bound_to_different_config(
    tmp_path, field, value, error
):
    shard = _upgrade_shard_with_summary_diagnostics(
        _write_shard(
            tmp_path / "shard",
            index=0,
            count=1,
            roi_lower=(0, 0),
            roi_dim=(2, 2),
            global_roi_dim=(2, 2),
            fill=1.0,
        )
    )
    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest["resume_identity"] = detector_artifact_resume_identity(manifest)
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        merge_detector_artifact_shards(
            shard_dirs=(shard,), output_dir=tmp_path / "merged"
        )


def test_detector_artifact_merge_cli_reads_plan(tmp_path, capsys):
    left = _write_shard(
        tmp_path / "left",
        index=0,
        count=2,
        roi_lower=(0, 0),
        roi_dim=(1, 2),
        global_roi_dim=(2, 2),
        fill=3.0,
    )
    right = _write_shard(
        tmp_path / "right",
        index=1,
        count=2,
        roi_lower=(1, 0),
        roi_dim=(1, 2),
        global_roi_dim=(2, 2),
        fill=4.0,
    )
    plan = {
        "kind": "xray-detector-artifact-distributed-plan",
        "shards": [
            {"output_dir": str(left)},
            {"output_dir": str(right)},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    assert (
        main(
            [
                "detector-artifact-merge",
                "--shards-manifest",
                str(plan_path),
                "--output-dir",
                str(tmp_path / "merged"),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["shard_count"] == 2
    amp = np.load(tmp_path / "merged" / "amp_all.npy")
    np.testing.assert_allclose(amp[:, 0, :], 3.0)
    np.testing.assert_allclose(amp[:, 1, :], 4.0)


def _write_synthetic_hdf5_pair(
    root: Path,
    *,
    samples: int,
    rows: int,
    cols: int,
) -> None:
    delay = np.linspace(-1.0, 1.0, samples, dtype=np.float64)
    off = np.ones((samples, rows, cols), dtype=np.float64)
    on = np.ones_like(off)
    on += delay[:, None, None] * 0.01
    for filename, data in (("on.h5", on), ("off.h5", off)):
        with h5py.File(root / filename, "w") as h5:
            h5.create_dataset("ROI", data=np.ones((rows, cols)))
            h5.create_dataset("bin_count", data=np.ones(samples))
            h5.create_dataset("i0", data=np.ones(samples))
            h5.create_dataset("i0_ipm3", data=np.ones(samples))
            h5.create_dataset("imgs", data=data)
            h5.create_dataset("scan_var", data=delay)


def _write_shard(
    root: Path,
    *,
    index: int,
    count: int,
    roi_lower: tuple[int, int],
    roi_dim: tuple[int, int],
    global_roi_dim: tuple[int, int],
    fill: float,
    input_identity: str = "dataset-a",
    resume_identity: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    height = roi_dim[1]
    width = roi_dim[0]
    shape = (height, width, 4)
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        np.save(root / f"{name}.npy", np.full(shape, fill))
    np.save(root / "amp_all_sum_filtered.npy", np.full((height, width), fill))
    np.save(
        root / FIT_STATUS_FILE,
        np.full((height, width), FIT_STATUS_OK, dtype=np.uint8),
    )
    manifest = {
        "kind": "xray-detector-artifacts",
        "manifest_schema_version": 1,
        "package_version": __version__,
        "backend": "cupy",
        "dtype": "float64",
        "input_identity": {"identity_sha256": input_identity},
        "normalization_identity": {"kind": "computed-from-input"},
        "schema": "synthetic",
        "image_dataset": "imgs",
        "delay_dataset": "scan_var",
        "normalization_dataset": "i0",
        "drop_leading": 0,
        "chunk_frames": 16,
        "fit_trailing_drop": 1,
        "zero_offset": 0.0,
        "requested_zero_offset_index": 0,
        "zero_offset_index": 0,
        "normalization_shift": 1.0,
        "detector_shape": [2, global_roi_dim[0]],
        "roi_lower": [roi_lower[0], roi_lower[1]],
        "roi_dim": [roi_dim[0], roi_dim[1]],
        "output_shape": [height, width, 4],
        "tile_shape": [1, 1],
        "exclude_y": [],
        "integrate_pixels": 0,
        "components": 2,
        "roots_backend": "eigvals",
        "savgol_window": 5,
        "savgol_polyorder": 3,
        "amp_threshold": 1.6,
        "max_fit_failures": 0,
        "hdf5_reader": "h5py",
        "hdf5_reader_workers": 2,
        "max_tiles": None,
        "processed_tiles": 1,
        "raw_fits": 2,
        "failures": 0,
        "skipped_fits": 0,
        "filtered_fits": 0,
        "elapsed_s": fill,
        "arrays": [
            "freq_all.npy",
            "amp_all.npy",
            "fft_all.npy",
            "fft_freq_all.npy",
            "amp_all_sum_filtered.npy",
            FIT_STATUS_FILE,
        ],
        "shard": {
            "index": index,
            "count": count,
            "global_roi_lower": [0, 0],
            "global_roi_dim": [global_roi_dim[0], global_roi_dim[1]],
            "x_range": [roi_lower[0], roi_lower[0] + roi_dim[0]],
        },
    }
    manifest["resume_identity"] = (
        detector_artifact_resume_identity(manifest)
        if resume_identity is None
        else resume_identity
    )
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _upgrade_shard_with_summary_diagnostics(root: Path) -> Path:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    writer = _FitDiagnosticsWriter(root, "summary")
    x0 = int(manifest["roi_lower"][0])
    x1 = x0 + int(manifest["roi_dim"][0])
    y0 = int(manifest["roi_lower"][1])
    y1 = y0 + int(manifest["roi_dim"][1])
    for y in range(y0, y1):
        writer.append(_summary_diagnostic_record(x0=x0, x1=x1, y=y))
    manifest["manifest_schema_version"] = 2
    manifest["p2_ridge_alpha"] = 0.0
    manifest["fit_diagnostics"] = writer.finalize(
        source_identity_sha256=_stable_json_hash(manifest["input_identity"])
    )
    manifest["resume_identity"] = detector_artifact_resume_identity(manifest)
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
    manifest["fit_diagnostics"]["artifact_resume_identity"] = manifest[
        "resume_identity"
    ]
    manifest["fit_diagnostics"]["artifact_config_hash"] = manifest[
        "config_hash"
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _upgrade_shard_with_full_diagnostics(root: Path) -> Path:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    writer = _FitDiagnosticsWriter(root, "full")
    x0 = int(manifest["roi_lower"][0])
    x1 = x0 + int(manifest["roi_dim"][0])
    y0 = int(manifest["roi_lower"][1])
    y1 = y0 + int(manifest["roi_dim"][1])
    mode_count = int(manifest["shard"]["index"]) + 1
    for row, y in enumerate(range(y0, y1)):
        if row == 0:
            writer.append(
                _full_diagnostic_record(
                    x0=x0,
                    x1=x1,
                    y=y,
                    modes=mode_count,
                )
            )
        else:
            _append_non_ok_fit_diagnostic(
                writer,
                level="full",
                fit_status=FIT_STATUS_FAILED,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                detector_y=y,
            )

    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        values = np.load(root / f"{name}.npy")
        values[1:, ...] = 0.0
        np.save(root / f"{name}.npy", values)
    amp_sum = np.load(root / "amp_all_sum_filtered.npy")
    amp_sum[1:, :] = 0.0
    np.save(root / "amp_all_sum_filtered.npy", amp_sum)
    fit_status = np.load(root / FIT_STATUS_FILE)
    fit_status[1:, :] = FIT_STATUS_FAILED
    np.save(root / FIT_STATUS_FILE, fit_status)

    manifest["manifest_schema_version"] = 2
    manifest["raw_fits"] = 1
    manifest["failures"] = y1 - y0 - 1
    manifest["max_fit_failures"] = y1 - y0 - 1
    manifest["p2_ridge_alpha"] = 0.0
    manifest["fit_diagnostics"] = writer.finalize(
        source_identity_sha256=_stable_json_hash(manifest["input_identity"])
    )
    manifest["resume_identity"] = detector_artifact_resume_identity(manifest)
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
    manifest["fit_diagnostics"]["artifact_resume_identity"] = manifest[
        "resume_identity"
    ]
    manifest["fit_diagnostics"]["artifact_config_hash"] = manifest[
        "config_hash"
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _summary_diagnostic_record(*, x0: int, x1: int, y: int):
    return {
        "tile_x_start": x0,
        "tile_x_stop": x1,
        "tile_y_start": 0,
        "tile_y_stop": 2,
        "detector_y": y,
        "fit_status": FIT_STATUS_OK,
        "trace_std": 1.0,
        "residual_std": 0.1,
        "relative_residual": 0.1,
        "chi2": 0.01,
        "selected_model_order": 2,
        "mode_count": 1,
        "p1_rank": 2,
        "p1_singular_value_ratio": 0.1,
        "p1_condition": 10.0,
        "p2_rank": 3,
        "p2_singular_value_ratio": 0.2,
        "p2_condition": 5.0,
        "max_amplitude": 1.0,
    }


def _full_diagnostic_record(
    *,
    x0: int,
    x1: int,
    y: int,
    modes: int,
):
    time = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)
    mode_values = np.arange(modes, dtype=np.float64)
    return {
        **_summary_diagnostic_record(x0=x0, x1=x1, y=y),
        "mode_count": modes,
        "p2_rank": 2 * modes + 1,
        "time": time,
        "trace": time + x0 + 1.0,
        "reconstruction": time + x0 + 0.9,
        "angular_frequency": mode_values + 1.0,
        "decay": mode_values + 0.5,
        "amplitude": mode_values + 2.0,
        "phase": mode_values * 0.1,
        "p1_singular_values": np.arange(modes + 1, dtype=np.float64),
        "p2_singular_values": np.arange(2 * modes + 1, dtype=np.float64),
    }
