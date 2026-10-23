# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU and CUDA comparison helpers for packaged workflows."""

from __future__ import annotations

import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from cuphoton.ximage.evaluation import (
    clean_evaluation_copy,
    evaluate_reconstruction_run,
)
from cuphoton.ximage.workflows import run_psf_solve, run_reconstruction

SUPPORTED_WORKFLOWS = {
    "psf-adam",
    "psf-mm",
    "reconstruct-l2",
    "reconstruct-rho",
}
SUPPORTED_DEVICES = {"cpu", "cuda"}


def benchmark_devices(
    *,
    workflow: str,
    config_path: Path,
    devices: list[str],
    repeats: int,
    warmup: int = 1,
    output_root: Path | None = None,
    cuda_visible_devices: str | None = None,
    evaluate: bool = False,
    coadd_method: str = "median",
    pipeline_coadd_fits: Path | None = None,
    catalog_parquet: Path | None = None,
    reference_catalog_csv: Path | None = None,
    catalog_band: str = "i",
    catalog_match_radius_px: float = 3.0,
    background_min_fraction: float = 0.5,
    photometry_threshold: float = 0.1,
    photometry_minarea: int = 5,
) -> dict[str, Any]:
    supported_workflows = ", ".join(sorted(SUPPORTED_WORKFLOWS))
    supported_devices = ", ".join(sorted(SUPPORTED_DEVICES))
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ValueError(
            f"Unsupported workflow {workflow!r}. Expected one of: "
            f"{supported_workflows}."
        )
    for device in devices:
        if device not in SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported device {device!r}. Expected one of: "
                f"{supported_devices}."
            )
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if evaluate and not workflow.startswith("reconstruct-"):
        raise ValueError(
            "Evaluation is only supported for reconstruction workflows."
        )
    if coadd_method not in {"median", "mean"}:
        raise ValueError("coadd_method must be 'median' or 'mean'")
    if background_min_fraction <= 0.0 or background_min_fraction > 1.0:
        raise ValueError("background_min_fraction must be in (0, 1]")
    if photometry_threshold <= 0.0:
        raise ValueError("photometry_threshold must be positive")
    if photometry_minarea <= 0:
        raise ValueError("photometry_minarea must be positive")
    if catalog_match_radius_px <= 0.0:
        raise ValueError("catalog_match_radius_px must be positive")

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    runner, runner_kind = _resolve_runner(workflow)
    results: dict[str, Any] = {}
    for device in devices:
        env_info = _device_environment(device)
        timings: list[float] = []
        run_dirs: list[str] = []
        evaluation = None
        total_runs = warmup + repeats
        for idx in range(total_runs):
            keep_artifacts = output_root is not None
            if output_root is not None:
                root = output_root / device / f"iteration-{idx}"
                root.mkdir(parents=True, exist_ok=True)
            else:
                root = Path(
                    tempfile.mkdtemp(
                        prefix=f"ximage-bench-{workflow}-{device}-{idx}-"
                    )
                )
            t0 = time.perf_counter()
            result = runner(
                config_path,
                device_override=device,
                output_root=root,
            )
            dt = time.perf_counter() - t0
            if evaluate and idx >= warmup and evaluation is None:
                evaluation = _finalize_evaluation_result(
                    evaluate_reconstruction_run(
                        result.run_dir,
                        coadd_method=coadd_method,
                        pipeline_coadd_fits=pipeline_coadd_fits,
                        catalog_parquet=catalog_parquet,
                        reference_catalog_csv=reference_catalog_csv,
                        catalog_band=catalog_band,
                        catalog_match_radius_px=catalog_match_radius_px,
                        background_min_fraction=background_min_fraction,
                        photometry_threshold=photometry_threshold,
                        photometry_minarea=photometry_minarea,
                        write_artifacts=keep_artifacts,
                    ),
                    keep_artifacts=keep_artifacts,
                )
            run_dirs.append(str(result.run_dir))
            if idx >= warmup:
                timings.append(dt)
            if not keep_artifacts:
                shutil.rmtree(root, ignore_errors=True)

        mean_sec = statistics.mean(timings)
        stdev_sec = statistics.pstdev(timings) if len(timings) > 1 else 0.0
        results[device] = {
            "device": device,
            "backend": "torch",
            "runner_kind": runner_kind,
            "env": env_info,
            "repeats": repeats,
            "warmup": warmup,
            "timings_sec": timings,
            "mean_sec": mean_sec,
            "median_sec": statistics.median(timings),
            "stdev_sec": stdev_sec,
            "run_dirs": run_dirs if output_root is not None else [],
        }
        if evaluation is not None:
            results[device]["evaluation"] = evaluation

    summary = {
        "workflow": workflow,
        "config_path": str(config_path),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "results": results,
    }
    if len(devices) == 2:
        left, right = devices
        if "median_sec" in results[left] and "median_sec" in results[right]:
            left_time = results[left]["median_sec"]
            right_time = results[right]["median_sec"]
            if left_time > 0 and right_time > 0:
                faster = left if left_time < right_time else right
                slower = right if faster == left else left
                summary["comparison"] = {
                    "faster_device": faster,
                    "slower_device": slower,
                    "speedup_vs_slower": results[slower]["median_sec"]
                    / results[faster]["median_sec"],
                }
        left_eval = results[left].get("evaluation")
        right_eval = results[right].get("evaluation")
        if left_eval and right_eval:
            summary["evaluation_comparison"] = _compare_evaluations(
                left,
                left_eval,
                right,
                right_eval,
            )
    return summary


def _compare_evaluations(
    left_name: str,
    left_eval: dict[str, Any],
    right_name: str,
    right_eval: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "left_device": left_name,
        "right_device": right_name,
        "restoration_sharpness_delta": (
            left_eval["restoration"]["sharpness_sf"]
            - right_eval["restoration"]["sharpness_sf"]
        ),
        "restoration_sky_sigma_delta": (
            left_eval["restoration"]["sky_sigma"]
            - right_eval["restoration"]["sky_sigma"]
        ),
        "synthetic_coadd_psnr_delta": (
            left_eval["synthetic_coadd"]["psnr_vs_restoration"]
            - right_eval["synthetic_coadd"]["psnr_vs_restoration"]
        ),
        "synthetic_coadd_ssim_delta": (
            left_eval["synthetic_coadd"]["ssim_vs_restoration"]
            - right_eval["synthetic_coadd"]["ssim_vs_restoration"]
        ),
        "exposure_avg_psnr_delta": (
            left_eval["exposures"]["avg_psnr_vs_restoration"]
            - right_eval["exposures"]["avg_psnr_vs_restoration"]
        ),
        "exposure_avg_ssim_delta": (
            left_eval["exposures"]["avg_ssim_vs_restoration"]
            - right_eval["exposures"]["avg_ssim_vs_restoration"]
        ),
    }
    left_photo = left_eval.get("photometry")
    right_photo = right_eval.get("photometry")
    if left_photo and right_photo:
        photometry_comparison = {
            "baseline_detection_count_delta": (
                left_photo.get("baseline_detection_count", 0)
                - right_photo.get("baseline_detection_count", 0)
            ),
            "restoration_detection_count_delta": (
                left_photo.get("restoration_detection_count", 0)
                - right_photo.get("restoration_detection_count", 0)
            ),
        }
        if (
            "delta_mag_median" in left_photo
            and "delta_mag_median" in right_photo
        ):
            photometry_comparison["delta_mag_median_delta"] = (
                left_photo["delta_mag_median"]
                - right_photo["delta_mag_median"]
            )
            photometry_comparison["delta_mag_comparable"] = True
        else:
            photometry_comparison["delta_mag_comparable"] = False
        if (
            "faint_quartile_delta_mag_median" in left_photo
            and "faint_quartile_delta_mag_median" in right_photo
        ):
            photometry_comparison["faint_quartile_delta_mag_median_delta"] = (
                left_photo["faint_quartile_delta_mag_median"]
                - right_photo["faint_quartile_delta_mag_median"]
            )
            photometry_comparison["faint_quartile_comparable"] = True
        else:
            photometry_comparison["faint_quartile_comparable"] = False
        result["photometry_comparison"] = photometry_comparison
    left_catalog = (
        left_photo.get("catalog_photometry") if left_photo else None
    )
    right_catalog = (
        right_photo.get("catalog_photometry") if right_photo else None
    )
    if left_catalog and right_catalog:
        catalog_comparison = {
            "available": bool(
                left_catalog.get("available")
                and right_catalog.get("available")
            ),
        }
        for key in (
            "matched_catalog_count",
            "restoration_minus_baseline_mag_median",
            "baseline_residual_mag_median",
            "restoration_residual_mag_median",
        ):
            if key in left_catalog and key in right_catalog:
                catalog_comparison[f"{key}_delta"] = (
                    left_catalog[key] - right_catalog[key]
                )
        if (
            "faint_quartile_restoration_minus_baseline_mag_median"
            in left_catalog
            and "faint_quartile_restoration_minus_baseline_mag_median"
            in right_catalog
        ):
            catalog_comparison[
                "faint_quartile_restoration_minus_baseline_mag_median_delta"
            ] = (
                left_catalog[
                    "faint_quartile_restoration_minus_baseline_mag_median"
                ]
                - right_catalog[
                    "faint_quartile_restoration_minus_baseline_mag_median"
                ]
            )
        result["catalog_photometry_comparison"] = catalog_comparison
    return result


def _finalize_evaluation_result(
    evaluation: dict[str, Any], *, keep_artifacts: bool
) -> dict[str, Any]:
    if keep_artifacts:
        return evaluation

    cleaned = clean_evaluation_copy(evaluation)
    cleaned["artifacts_persisted"] = False
    cleaned.pop("run_dir", None)
    cleaned.pop("image_path", None)
    cleaned.pop("saved", None)
    return cleaned


def _resolve_runner(workflow: str):
    def _run_l2(config_path: Path, device_override: str, output_root: Path):
        return run_reconstruction(
            config_path,
            device_override=device_override,
            output_root=output_root,
            algorithm_override="l2",
        )

    def _run_rho(config_path: Path, device_override: str, output_root: Path):
        return run_reconstruction(
            config_path,
            device_override=device_override,
            output_root=output_root,
            algorithm_override="rho",
        )

    if workflow == "psf-adam":
        return (
            lambda config_path, device_override, output_root: run_psf_solve(
                config_path,
                device_override=device_override,
                output_root=output_root,
                method_override="adam",
            ),
            "psf",
        )
    if workflow == "psf-mm":
        return (
            lambda config_path, device_override, output_root: run_psf_solve(
                config_path,
                device_override=device_override,
                output_root=output_root,
                method_override="mm",
            ),
            "psf",
        )
    if workflow == "reconstruct-l2":
        return (_run_l2, "reconstruction")
    if workflow == "reconstruct-rho":
        return (_run_rho, "reconstruction")
    raise ValueError(f"Unsupported workflow: {workflow}")


def _device_environment(device: str) -> dict[str, Any]:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requested but Torch cannot access a GPU."
        )
    names = (
        [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else []
    )
    return {
        "framework": "torch",
        "version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "requested_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpus": names,
    }
