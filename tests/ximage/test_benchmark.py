# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from cuphoton.ximage import benchmark
from cuphoton.ximage.workflows import WorkflowResult


def test_benchmark_devices_reports_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_environment(device: str) -> dict[str, str]:
        return {"requested_device": device}

    def fake_run(
        config_path: Path,
        *,
        device_override: str,
        output_root: Path,
        **kwargs,
    ) -> WorkflowResult:
        calls.append(device_override)
        output_root.mkdir(parents=True, exist_ok=True)
        return WorkflowResult(
            output_root / device_override,
            {"device": device_override},
        )

    samples = iter([1.0, 2.0, 3.0, 3.5])
    monkeypatch.setattr(benchmark, "_device_environment", fake_environment)
    monkeypatch.setattr(benchmark, "run_psf_solve", fake_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(samples))

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("backend: auto\nmethod: adam\n")
    result = benchmark.benchmark_devices(
        workflow="psf-adam",
        config_path=config_path,
        devices=["cpu", "cuda"],
        repeats=1,
        warmup=0,
    )

    assert calls == ["cpu", "cuda"]
    assert result["results"]["cpu"]["median_sec"] == 1.0
    assert result["results"]["cuda"]["median_sec"] == 0.5
    assert result["comparison"] == {
        "faster_device": "cuda",
        "slower_device": "cpu",
        "speedup_vs_slower": 2.0,
    }


def test_benchmark_devices_compares_medians_not_means(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(
        config_path: Path,
        *,
        device_override: str,
        output_root: Path,
        **kwargs,
    ) -> WorkflowResult:
        output_root.mkdir(parents=True, exist_ok=True)
        return WorkflowResult(output_root / device_override, {})

    timestamps = iter(
        [
            0.0,
            1.0,
            10.0,
            11.0,
            20.0,
            120.0,
            130.0,
            132.0,
            140.0,
            142.0,
            150.0,
            152.0,
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "_device_environment",
        lambda device: {"requested_device": device},
    )
    monkeypatch.setattr(benchmark, "run_psf_solve", fake_run)
    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: next(timestamps)
    )

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("backend: auto\nmethod: adam\n")
    result = benchmark.benchmark_devices(
        workflow="psf-adam",
        config_path=config_path,
        devices=["cpu", "cuda"],
        repeats=3,
        warmup=0,
    )

    assert result["results"]["cpu"]["mean_sec"] == 34.0
    assert result["results"]["cpu"]["median_sec"] == 1.0
    assert result["results"]["cuda"]["mean_sec"] == 2.0
    assert result["comparison"] == {
        "faster_device": "cpu",
        "slower_device": "cuda",
        "speedup_vs_slower": 2.0,
    }


def test_benchmark_devices_can_attach_reconstruction_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(
        config_path: Path,
        *,
        device_override: str,
        output_root: Path,
        **kwargs,
    ) -> WorkflowResult:
        output_root.mkdir(parents=True, exist_ok=True)
        return WorkflowResult(output_root / device_override, {})

    def fake_evaluate(*args, **kwargs) -> dict[str, object]:
        return {
            "run_dir": "temporary",
            "image_path": "temporary.npy",
            "restoration": {"sharpness_sf": 1.0, "sky_sigma": 0.5},
            "synthetic_coadd": {
                "psnr_vs_restoration": 30.0,
                "ssim_vs_restoration": 0.9,
            },
            "exposures": {
                "avg_psnr_vs_restoration": 31.0,
                "avg_ssim_vs_restoration": 0.91,
            },
        }

    monkeypatch.setattr(
        benchmark,
        "_device_environment",
        lambda device: {"requested_device": device},
    )
    monkeypatch.setattr(benchmark, "run_reconstruction", fake_run)
    monkeypatch.setattr(
        benchmark, "evaluate_reconstruction_run", fake_evaluate
    )

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("backend: auto\nalgorithm: l2\n")
    result = benchmark.benchmark_devices(
        workflow="reconstruct-l2",
        config_path=config_path,
        devices=["cpu"],
        repeats=1,
        warmup=0,
        evaluate=True,
    )

    evaluation = result["results"]["cpu"]["evaluation"]
    assert evaluation["artifacts_persisted"] is False
    assert "run_dir" not in evaluation
    assert "image_path" not in evaluation


def test_benchmark_devices_rejects_invalid_requests(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("backend: auto\nmethod: adam\n")

    with pytest.raises(ValueError, match="Unsupported device"):
        benchmark.benchmark_devices(
            workflow="psf-adam",
            config_path=config_path,
            devices=["accelerator"],
            repeats=1,
        )
    with pytest.raises(ValueError, match="reconstruction workflows"):
        benchmark.benchmark_devices(
            workflow="psf-mm",
            config_path=config_path,
            devices=["cpu"],
            repeats=1,
            evaluate=True,
        )
