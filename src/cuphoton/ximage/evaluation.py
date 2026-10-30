# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run loading, reconstruction evaluation, and evaluation artifact output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.ximage.adapters.rubin.evaluation import (
    load_pipeline_coadd_roi,
    photometry_metrics,
    resolve_catalog_parquet,
    resolve_pipeline_coadd_fits,
    resolve_reference_catalog_csv,
    write_catalog_photometry_binned_csv,
    write_catalog_photometry_csv,
)
from cuphoton.ximage.config import ReconstructionConfig, load_config
from cuphoton.ximage.data import XImageBundle, load_ximage_bundle
from cuphoton.ximage.metrics import (
    _evaluate_metric_group,
    build_background_mask,
    project_restoration_to_observation_grid,
    squeeze_exposure_stack,
    squeeze_single_image,
)
from cuphoton.ximage.preprocess import (
    coadd,
    prepare_reconstruction_inputs_roi,
)


@dataclass(frozen=True)
class RestorationRun:
    run_dir: Path
    config: ReconstructionConfig
    summary: dict[str, Any]
    image_path: Path
    image: np.ndarray


def load_reconstruction_run(run_dir: Path) -> RestorationRun:
    run_dir = Path(run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    summary = json.loads(summary_path.read_text())
    config = load_config(config_path, ReconstructionConfig)
    image_path = resolve_saved_path(run_dir, summary["saved"]["image"])
    image = np.load(image_path, allow_pickle=False)
    return RestorationRun(
        run_dir=run_dir,
        config=config,
        summary=summary,
        image_path=image_path,
        image=image,
    )


def evaluate_reconstruction_run(
    run_dir: Path,
    *,
    base_override: Path | None = None,
    coadd_method: str = "median",
    pipeline_coadd_fits: Path | None = None,
    catalog_parquet: Path | None = None,
    reference_catalog_csv: Path | None = None,
    catalog_band: str = "i",
    catalog_match_radius_px: float = 3.0,
    background_min_fraction: float = 0.5,
    photometry_threshold: float = 0.1,
    photometry_minarea: int = 5,
    write_artifacts: bool = False,
    metrics_backend: str = "auto",
) -> dict[str, Any]:
    run = load_reconstruction_run(run_dir)
    data_base = (
        Path(base_override).expanduser()
        if base_override is not None
        else Path(run.config.data.base)
    )
    loaded_bundle = load_ximage_bundle(
        data_base,
        prefer=run.config.data.prefer,
    )
    try:
        return _evaluate_reconstruction_bundle(
            run,
            loaded_bundle,
            coadd_method=coadd_method,
            pipeline_coadd_fits=pipeline_coadd_fits,
            catalog_parquet=catalog_parquet,
            reference_catalog_csv=reference_catalog_csv,
            catalog_band=catalog_band,
            catalog_match_radius_px=catalog_match_radius_px,
            background_min_fraction=background_min_fraction,
            photometry_threshold=photometry_threshold,
            photometry_minarea=photometry_minarea,
            write_artifacts=write_artifacts,
            metrics_backend=metrics_backend,
        )
    finally:
        loaded_bundle.close()


def _evaluate_reconstruction_bundle(
    run: RestorationRun,
    bundle: XImageBundle,
    *,
    coadd_method: str,
    pipeline_coadd_fits: Path | None,
    catalog_parquet: Path | None,
    reference_catalog_csv: Path | None,
    catalog_band: str,
    catalog_match_radius_px: float,
    background_min_fraction: float,
    photometry_threshold: float,
    photometry_minarea: int,
    write_artifacts: bool,
    metrics_backend: str = "auto",
) -> dict[str, Any]:
    selected = run.summary.get("selected_exposures")
    if selected:
        bundle = bundle.select_exposures(selected)
    roi_center = run.config.roi.center
    if roi_center is None:
        roi_center = (
            bundle.images.shape[1] // 2,
            bundle.images.shape[2] // 2,
        )

    y_roi, m_roi, v_roi = prepare_reconstruction_inputs_roi(
        bundle,
        mask_mode=run.config.mask.mode,
        center_roi=roi_center,
        height_roi=run.config.roi.height,
        width_roi=run.config.roi.width,
    )
    if m_roi is None:
        m_roi = np.ones(y_roi.shape, dtype=np.float64)

    target_shape = y_roi.shape[1:3]
    projected = project_restoration_to_observation_grid(
        run.image,
        target_shape=target_shape,
        srf=run.config.srf,
    )
    synthetic = coadd(
        y_roi,
        masks=m_roi,
        variances=v_roi,
        method=coadd_method,
        positive=False,
    )
    synthetic_2d = squeeze_single_image(synthetic)
    exposures = squeeze_exposure_stack(y_roi)
    background_mask = build_background_mask(
        m_roi, min_fraction=background_min_fraction
    )

    coadd_path = resolve_pipeline_coadd_fits(
        explicit_path=pipeline_coadd_fits
    )
    catalog_path = resolve_catalog_parquet(explicit_path=catalog_parquet)
    reference_catalog_path = resolve_reference_catalog_csv(
        explicit_path=reference_catalog_csv,
    )
    metric_images = [projected, synthetic_2d]
    if coadd_path is not None:
        pipeline_coadd = load_pipeline_coadd_roi(
            coadd_path,
            center=roi_center,
            height=target_shape[0],
            width=target_shape[1],
            bundle_metadata=bundle.metadata,
        )
        metric_images.append(pipeline_coadd)
    metrics = _evaluate_metric_group(
        metric_images,
        background_mask=background_mask,
        exposures=exposures,
        backend=metrics_backend,
    )
    result = {
        "run_dir": str(run.run_dir),
        "image_path": str(run.image_path),
        "backend": run.summary.get("backend"),
        "metrics_backend": metrics["backend"],
        "algorithm": run.summary.get("algorithm"),
        "selected_exposures": selected,
        "projection_shape": list(projected.shape),
        "background_fraction": float(np.mean(background_mask)),
        "restoration": metrics["images"][0],
        "synthetic_coadd": {
            "method": coadd_method,
            **metrics["images"][1],
            "psnr_vs_restoration": metrics["comparisons"][0]["psnr"],
            "ssim_vs_restoration": metrics["comparisons"][0]["ssim"],
        },
        "exposures": {
            "count": int(exposures.shape[0]),
            "avg_psnr_vs_restoration": float(
                np.mean([item["psnr"] for item in metrics["exposures"]])
            ),
            "avg_ssim_vs_restoration": float(
                np.mean([item["ssim"] for item in metrics["exposures"]])
            ),
        },
    }

    if coadd_path is not None:
        result["pipeline_coadd"] = {
            "path": str(coadd_path),
            **metrics["images"][2],
            "comparison_valid_fraction": metrics["comparisons"][1][
                "comparison_valid_fraction"
            ],
            "psnr_vs_restoration": metrics["comparisons"][1]["psnr"],
            "ssim_vs_restoration": metrics["comparisons"][1]["ssim"],
        }

    photometry_baseline_name = (
        "pipeline_coadd" if "pipeline_coadd" in result else "synthetic_coadd"
    )
    photometry_baseline = (
        pipeline_coadd if "pipeline_coadd" in result else synthetic_2d
    )
    result["photometry"] = photometry_metrics(
        restoration=projected,
        baseline=photometry_baseline,
        baseline_name=photometry_baseline_name,
        threshold=photometry_threshold,
        minarea=photometry_minarea,
        catalog_parquet=catalog_path,
        reference_catalog_csv=reference_catalog_path,
        catalog_band=catalog_band,
        catalog_match_radius_px=catalog_match_radius_px,
        coadd_fits_path=coadd_path,
        roi_center=roi_center,
        roi_shape=target_shape,
        bundle_metadata=bundle.metadata,
    )
    if write_artifacts:
        result["saved"] = write_evaluation_artifacts(run.run_dir, result)
        strip_internal_artifact_payloads(result)

    return result


def resolve_saved_path(run_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    for marker in ("artifacts", "trace"):
        if marker in path.parts:
            suffix = Path(*path.parts[path.parts.index(marker) :])
            candidate = (run_dir / suffix).resolve()
            if candidate.exists():
                return candidate

    candidate = (run_dir / path).resolve()
    if candidate.exists():
        return candidate

    return candidate


def write_evaluation_artifacts(
    run_dir: Path,
    evaluation: dict[str, Any],
) -> dict[str, str]:
    saved: dict[str, str] = {}
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)

    summary_path = evaluation_dir / "evaluation_summary.json"
    summary_payload = json.loads(
        json.dumps(clean_evaluation_copy(evaluation))
    )
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    saved["evaluation_summary"] = str(summary_path)

    photometry = evaluation.get("photometry")
    if not isinstance(photometry, dict):
        return saved
    catalog = photometry.get("catalog_photometry")
    if not isinstance(catalog, dict):
        return saved
    payload = catalog.get("_artifact_payload")
    if not isinstance(payload, dict):
        return saved

    csv_path = evaluation_dir / "catalog_photometry_matches.csv"
    write_catalog_photometry_csv(csv_path, payload)
    saved["catalog_photometry_matches_csv"] = str(csv_path)

    binned_csv_path = evaluation_dir / "catalog_photometry_binned_summary.csv"
    write_catalog_photometry_binned_csv(binned_csv_path, payload)
    saved["catalog_photometry_binned_csv"] = str(binned_csv_path)
    return saved


def strip_internal_artifact_payloads(evaluation: dict[str, Any]) -> None:
    photometry = evaluation.get("photometry")
    if isinstance(photometry, dict):
        catalog = photometry.get("catalog_photometry")
        if isinstance(catalog, dict):
            catalog.pop("_artifact_payload", None)


def clean_evaluation_copy(evaluation: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(evaluation)
    photometry = cleaned.get("photometry")
    if isinstance(photometry, dict):
        photometry = dict(photometry)
        cleaned["photometry"] = photometry
        catalog = photometry.get("catalog_photometry")
        if isinstance(catalog, dict):
            catalog = dict(catalog)
            photometry["catalog_photometry"] = catalog
            catalog.pop("_artifact_payload", None)
    return json.loads(json.dumps(cleaned, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
