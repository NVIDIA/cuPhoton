# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Quantitative restoration evaluation helpers."""

from __future__ import annotations

import json
import warnings
from csv import DictReader, DictWriter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from cuphoton._photometry import (
    detect_sources,
    elliptical_aperture_flux,
    kron_radius,
    subtract_background,
)
from cuphoton.ximage.config import ReconstructionConfig, load_config
from cuphoton.ximage.data import XImageBundle, load_ximage_bundle
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
        dilation_factor=run.config.mask.dilation_factor,
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

    result = {
        "run_dir": str(run.run_dir),
        "image_path": str(run.image_path),
        "backend": run.summary.get("backend"),
        "algorithm": run.summary.get("algorithm"),
        "selected_exposures": selected,
        "projection_shape": list(projected.shape),
        "background_fraction": float(np.mean(background_mask)),
        "restoration": image_metrics(projected, background_mask),
        "synthetic_coadd": {
            "method": coadd_method,
            **image_metrics(synthetic_2d, background_mask),
            "psnr_vs_restoration": psnr(projected, synthetic_2d),
            "ssim_vs_restoration": ssim(projected, synthetic_2d),
        },
        "exposures": {
            "count": int(exposures.shape[0]),
            "avg_psnr_vs_restoration": float(
                np.mean([psnr(projected, exposure) for exposure in exposures])
            ),
            "avg_ssim_vs_restoration": float(
                np.mean([ssim(projected, exposure) for exposure in exposures])
            ),
        },
    }

    coadd_path = resolve_pipeline_coadd_fits(
        explicit_path=pipeline_coadd_fits,
    )
    catalog_path = resolve_catalog_parquet(
        explicit_path=catalog_parquet,
    )
    reference_catalog_path = resolve_reference_catalog_csv(
        explicit_path=reference_catalog_csv,
    )
    if coadd_path is not None:
        pipeline_coadd = load_pipeline_coadd_roi(
            coadd_path,
            center=roi_center,
            height=target_shape[0],
            width=target_shape[1],
            bundle_metadata=bundle.metadata,
        )
        result["pipeline_coadd"] = {
            "path": str(coadd_path),
            **image_metrics(pipeline_coadd, background_mask),
            "comparison_valid_fraction": float(
                np.mean(np.isfinite(projected) & np.isfinite(pipeline_coadd))
            ),
            "psnr_vs_restoration": psnr(projected, pipeline_coadd),
            "ssim_vs_restoration": ssim(projected, pipeline_coadd),
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


def image_metrics(
    image: np.ndarray,
    background_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report finite coverage; Fourier sharpness needs a complete image."""

    finite = np.isfinite(image)
    complete = bool(np.all(finite))
    result: dict[str, Any] = {
        "sharpness_sf": fourier_sharpness(image) if complete else None,
        "sky_sigma": sky_sigma(image, background_mask=background_mask),
        "finite_pixel_fraction": float(np.mean(finite)),
    }
    if not complete:
        result["sharpness_sf_unavailable_reason"] = (
            "Fourier sharpness requires a complete finite image."
        )
    return result


def fourier_sharpness(image: np.ndarray) -> float:
    image_2d = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(image_2d)):
        return float("nan")
    spectrum = np.fft.fft2(image_2d)
    return float(np.mean(np.log(np.abs(spectrum) + 1e-12)))


def sky_sigma(
    image: np.ndarray,
    *,
    background_mask: np.ndarray | None = None,
) -> float:
    arr = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(arr)
    if background_mask is not None:
        valid &= background_mask
    values = arr[valid]
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if scale == 0:
        return float(np.std(values))
    clipped = values[np.abs(values - median) <= 3.0 * scale]
    if clipped.size == 0:
        clipped = values
    return float(np.std(clipped))


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compare values and dynamic range on jointly finite pixels."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(valid):
        return float("nan")
    ref = ref[valid]
    cand = cand[valid]
    mse = np.mean((ref - cand) ** 2)
    if mse == 0:
        return float("inf")
    data_min = min(float(ref.min()), float(cand.min()))
    data_max = max(float(ref.max()), float(cand.max()))
    data_range = max(data_max - data_min, 1e-12)
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Average SSIM on joint finite support with normalized local moments."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(valid):
        return float("nan")
    data_min = min(float(ref[valid].min()), float(cand[valid].min()))
    data_max = max(float(ref[valid].max()), float(cand[valid].max()))
    data_range = max(data_max - data_min, 1e-12)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    weight: float | np.ndarray = 1.0
    if not np.all(valid):
        ref = np.where(valid, ref, 0.0)
        cand = np.where(valid, cand, 0.0)
        weight = gaussian_filter(
            valid.astype(float), sigma=1.5, mode="reflect"
        )
        weight = np.where(weight > 0.0, weight, 1.0)

    mu_ref = gaussian_filter(ref, sigma=1.5, mode="reflect") / weight
    mu_cand = gaussian_filter(cand, sigma=1.5, mode="reflect") / weight
    mu_ref_sq = mu_ref * mu_ref
    mu_cand_sq = mu_cand * mu_cand
    mu_ref_cand = mu_ref * mu_cand

    sigma_ref_sq = (
        gaussian_filter(ref * ref, sigma=1.5, mode="reflect") / weight
        - mu_ref_sq
    )
    sigma_cand_sq = (
        gaussian_filter(cand * cand, sigma=1.5, mode="reflect") / weight
        - mu_cand_sq
    )
    sigma_ref_cand = (
        gaussian_filter(ref * cand, sigma=1.5, mode="reflect") / weight
        - mu_ref_cand
    )
    sigma_ref_sq = np.maximum(sigma_ref_sq, 0.0)
    sigma_cand_sq = np.maximum(sigma_cand_sq, 0.0)

    numerator = (2 * mu_ref_cand + c1) * (2 * sigma_ref_cand + c2)
    denominator = (mu_ref_sq + mu_cand_sq + c1) * (
        sigma_ref_sq + sigma_cand_sq + c2
    )
    score = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(score[valid]))


def build_background_mask(
    masks: np.ndarray | None,
    *,
    min_fraction: float,
) -> np.ndarray:
    if masks is None:
        raise ValueError("Masks are required to construct a background mask.")
    mask_stack = squeeze_exposure_stack(masks)
    return np.mean(mask_stack, axis=0) >= min_fraction


def photometry_metrics(
    *,
    restoration: np.ndarray,
    baseline: np.ndarray,
    baseline_name: str,
    threshold: float,
    minarea: int,
    catalog_parquet: Path | None = None,
    reference_catalog_csv: Path | None = None,
    catalog_band: str = "i",
    catalog_match_radius_px: float = 3.0,
    coadd_fits_path: Path | None = None,
    roi_center: tuple[int, int] | None = None,
    roi_shape: tuple[int, int] | None = None,
    bundle_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_prepared = _prepare_photometry_image(baseline)
    restoration_prepared = _prepare_photometry_image(restoration)

    try:
        baseline_data = subtract_photometry_background(baseline_prepared)
        restoration_data = subtract_photometry_background(
            restoration_prepared
        )
        baseline_objects = detect_sources(
            baseline_data,
            threshold=threshold,
            minarea=minarea,
        )
        restoration_objects = detect_sources(
            restoration_data,
            threshold=threshold,
            minarea=minarea,
        )
        if len(baseline_objects) == 0:
            return {
                "available": True,
                "baseline_name": baseline_name,
                "threshold": threshold,
                "minarea": minarea,
                "baseline_detection_count": 0,
                "restoration_detection_count": int(len(restoration_objects)),
                "matched_aperture_count": 0,
            }
    except Exception as exc:
        if not is_recoverable_photometry_error(exc):
            raise
        return unavailable_photometry(
            baseline_name=baseline_name,
            threshold=threshold,
            minarea=minarea,
            reason=str(exc),
        )

    x = baseline_objects["x"]
    y = baseline_objects["y"]
    a = np.maximum(baseline_objects["a"], 1e-3)
    b = np.maximum(baseline_objects["b"], 1e-3)
    theta = baseline_objects["theta"]
    try:
        kronrad, kron_flags = kron_radius(
            baseline_data,
            x,
            y,
            a,
            b,
            theta,
            max_radius=6.0,
        )
        aperture_scale = np.where(
            np.isfinite(kronrad) & (kronrad > 0),
            2.5 * kronrad,
            6.0,
        )

        baseline_flux, baseline_flags, restoration_flux, restoration_flags = (
            measure_matched_fluxes(
                baseline_data=baseline_data,
                restoration_data=restoration_data,
                x=x,
                y=y,
                a=a,
                b=b,
                theta=theta,
                aperture_scale=aperture_scale,
            )
        )
    except Exception as exc:
        if not is_recoverable_photometry_error(exc):
            raise
        return unavailable_photometry(
            baseline_name=baseline_name,
            threshold=threshold,
            minarea=minarea,
            reason=str(exc),
        )

    valid = (
        np.isfinite(baseline_flux)
        & np.isfinite(restoration_flux)
        & (baseline_flux > 0)
        & (restoration_flux > 0)
        & (kron_flags == 0)
        & (baseline_flags == 0)
        & (restoration_flags == 0)
    )
    if not np.any(valid):
        return {
            "available": True,
            "baseline_name": baseline_name,
            "threshold": threshold,
            "minarea": minarea,
            "baseline_detection_count": int(len(baseline_objects)),
            "restoration_detection_count": int(len(restoration_objects)),
            "matched_aperture_count": 0,
        }

    base_flux = baseline_flux[valid]
    rest_flux = restoration_flux[valid]
    delta_mag = -2.5 * np.log10(rest_flux / base_flux)
    baseline_mag = -2.5 * np.log10(base_flux)
    faint_cut = np.quantile(baseline_mag, 0.75)
    faint_mask = baseline_mag >= faint_cut
    bright_mask = baseline_mag < faint_cut

    result = {
        "available": True,
        "baseline_name": baseline_name,
        "threshold": threshold,
        "minarea": minarea,
        "baseline_detection_count": int(len(baseline_objects)),
        "restoration_detection_count": int(len(restoration_objects)),
        "matched_aperture_count": int(base_flux.size),
        "excluded_flagged_apertures": int(np.size(valid) - np.sum(valid)),
        "flux_ratio_median": float(np.median(rest_flux / base_flux)),
        "delta_mag_median": float(np.median(delta_mag)),
        "delta_mag_p10": float(np.quantile(delta_mag, 0.1)),
        "delta_mag_p90": float(np.quantile(delta_mag, 0.9)),
        "restoration_brighter_fraction": float(np.mean(delta_mag < 0)),
    }
    if np.any(faint_mask):
        result["faint_quartile_count"] = int(np.sum(faint_mask))
        result["faint_quartile_delta_mag_median"] = float(
            np.median(delta_mag[faint_mask])
        )
        result["faint_quartile_brighter_fraction"] = float(
            np.mean(delta_mag[faint_mask] < 0)
        )
    if np.any(bright_mask):
        result["bright_quartile_count"] = int(np.sum(bright_mask))
        result["bright_quartile_delta_mag_median"] = float(
            np.median(delta_mag[bright_mask])
        )
    if (
        catalog_parquet is not None or reference_catalog_csv is not None
    ) and roi_shape is not None:
        result["catalog_photometry"] = catalog_photometry_metrics(
            baseline_flux=baseline_flux,
            restoration_flux=restoration_flux,
            valid_mask=valid,
            detection_x=x,
            detection_y=y,
            baseline_name=baseline_name,
            catalog_parquet=catalog_parquet,
            reference_catalog_csv=reference_catalog_csv,
            catalog_band=catalog_band,
            catalog_match_radius_px=catalog_match_radius_px,
            coadd_fits_path=coadd_fits_path,
            roi_center=roi_center,
            roi_shape=roi_shape,
            bundle_metadata=bundle_metadata,
        )
    return result


def measure_matched_fluxes(
    *,
    baseline_data: np.ndarray,
    restoration_data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    aperture_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline_flux, _, baseline_flags = elliptical_aperture_flux(
        baseline_data,
        x,
        y,
        a,
        b,
        theta,
        scale=aperture_scale,
    )
    restoration_flux, _, restoration_flags = elliptical_aperture_flux(
        restoration_data,
        x,
        y,
        a,
        b,
        theta,
        scale=aperture_scale,
    )
    return (
        np.asarray(baseline_flux),
        np.asarray(baseline_flags),
        np.asarray(restoration_flux),
        np.asarray(restoration_flags),
    )


def catalog_photometry_metrics(
    *,
    baseline_flux: np.ndarray,
    restoration_flux: np.ndarray,
    valid_mask: np.ndarray,
    detection_x: np.ndarray,
    detection_y: np.ndarray,
    baseline_name: str,
    catalog_band: str,
    catalog_match_radius_px: float,
    roi_center: tuple[int, int] | None,
    roi_shape: tuple[int, int],
    catalog_parquet: Path | None = None,
    reference_catalog_csv: Path | None = None,
    coadd_fits_path: Path | None = None,
    bundle_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if catalog_parquet is not None and reference_catalog_csv is not None:
        raise ValueError(
            "Specify only one catalog source: catalog_parquet or "
            "reference_catalog_csv"
        )
    if reference_catalog_csv is not None:
        catalog = load_reference_catalog_csv_sources(
            reference_catalog_csv,
            roi_center=roi_center,
            roi_shape=roi_shape,
            catalog_band=catalog_band,
            bundle_metadata=bundle_metadata,
        )
    elif catalog_parquet is not None and coadd_fits_path is not None:
        catalog = load_catalog_sources_in_roi(
            catalog_parquet,
            band=catalog_band,
            coadd_fits_path=coadd_fits_path,
            roi_center=roi_center,
            roi_shape=roi_shape,
            bundle_metadata=bundle_metadata,
        )
    else:
        return {
            "available": False,
            "baseline_name": baseline_name,
            "catalog_band": catalog_band,
            "reason": "no compatible catalog source supplied",
        }
    if catalog["count"] == 0:
        return {
            "available": False,
            "baseline_name": baseline_name,
            "catalog_band": catalog_band,
            "catalog_kind": catalog["kind"],
            "catalog_path": catalog["path"],
            "catalog_provenance": catalog.get("provenance"),
            "reason": "no catalog sources in ROI",
        }

    matched = match_detections_to_catalog(
        detection_x=detection_x[valid_mask],
        detection_y=detection_y[valid_mask],
        catalog_x=catalog["x"],
        catalog_y=catalog["y"],
        match_radius_px=catalog_match_radius_px,
    )
    if matched["matched_count"] == 0:
        return {
            "available": False,
            "baseline_name": baseline_name,
            "catalog_band": catalog_band,
            "catalog_kind": catalog["kind"],
            "catalog_path": catalog["path"],
            "catalog_provenance": catalog.get("provenance"),
            "reason": "no matched detections within match radius",
            "catalog_source_count": catalog["count"],
            "match_radius_px": catalog_match_radius_px,
        }

    detection_idx = matched["detection_indices"]
    catalog_idx = matched["catalog_indices"]
    base_flux = baseline_flux[valid_mask][detection_idx]
    rest_flux = restoration_flux[valid_mask][detection_idx]
    ref_flux = catalog["reference_flux"][catalog_idx]
    pointlike = catalog["is_pointlike"][catalog_idx]
    catalog_id = np.asarray(catalog["catalog_id"], dtype=object)[catalog_idx]
    catalog_x = np.asarray(catalog["x"], dtype=np.float64)[catalog_idx]
    catalog_y = np.asarray(catalog["y"], dtype=np.float64)[catalog_idx]
    catalog_ra_deg = np.asarray(
        catalog["catalog_ra_deg"],
        dtype=np.float64,
    )[catalog_idx]
    catalog_dec_deg = np.asarray(
        catalog["catalog_dec_deg"],
        dtype=np.float64,
    )[catalog_idx]

    flux_valid = (
        np.isfinite(base_flux)
        & np.isfinite(rest_flux)
        & np.isfinite(ref_flux)
        & (base_flux > 0)
        & (rest_flux > 0)
        & (ref_flux > 0)
    )
    if not np.any(flux_valid):
        return {
            "available": False,
            "baseline_name": baseline_name,
            "catalog_band": catalog_band,
            "catalog_kind": catalog["kind"],
            "catalog_path": catalog["path"],
            "catalog_provenance": catalog.get("provenance"),
            "reason": "matched sources have no positive finite flux",
            "catalog_source_count": catalog["count"],
            "matched_catalog_count": int(matched["matched_count"]),
            "match_radius_px": catalog_match_radius_px,
        }

    base_flux = base_flux[flux_valid]
    rest_flux = rest_flux[flux_valid]
    ref_flux = ref_flux[flux_valid]
    pointlike = pointlike[flux_valid]
    catalog_id = catalog_id[flux_valid]
    catalog_x = catalog_x[flux_valid]
    catalog_y = catalog_y[flux_valid]
    catalog_ra_deg = catalog_ra_deg[flux_valid]
    catalog_dec_deg = catalog_dec_deg[flux_valid]

    ref_mag = flux_to_mag(ref_flux)
    base_mag = flux_to_mag(base_flux)
    rest_mag = flux_to_mag(rest_flux)
    calib_mask = select_calibration_subset(ref_flux, pointlike)
    zero_point_offset = float(
        np.median(ref_mag[calib_mask] - base_mag[calib_mask])
    )
    base_mag_cal = base_mag + zero_point_offset
    rest_mag_cal = rest_mag + zero_point_offset

    base_resid = base_mag_cal - ref_mag
    rest_resid = rest_mag_cal - ref_mag
    rest_minus_base = rest_mag_cal - base_mag_cal
    faint_mask = ref_mag >= np.quantile(ref_mag, 0.75)
    bright_mask = ref_mag < np.quantile(ref_mag, 0.75)

    result = {
        "available": True,
        "baseline_name": baseline_name,
        "catalog_band": catalog_band,
        "catalog_kind": catalog["kind"],
        "catalog_path": catalog["path"],
        "catalog_provenance": catalog.get("provenance"),
        "catalog_source_count": int(catalog["count"]),
        "matched_catalog_count": int(ref_flux.size),
        "match_radius_px": catalog_match_radius_px,
        "reference_flux_column": catalog["reference_flux_name"],
        "catalog_id_column": catalog["catalog_id_column"],
        "calibration_source_count": int(np.sum(calib_mask)),
        "calibration_mode": (
            "pointlike_bright"
            if np.any(pointlike[calib_mask])
            else "bright_quartile"
        ),
        "zero_point_offset_mag": zero_point_offset,
        "baseline_residual_mag_median": float(np.median(base_resid)),
        "restoration_residual_mag_median": float(np.median(rest_resid)),
        "restoration_minus_baseline_mag_median": float(
            np.median(rest_minus_base)
        ),
    }
    if np.any(faint_mask):
        result["faint_quartile_catalog_count"] = int(np.sum(faint_mask))
        result["faint_quartile_restoration_minus_baseline_mag_median"] = (
            float(np.median(rest_minus_base[faint_mask]))
        )
    if np.any(bright_mask):
        result["bright_quartile_catalog_count"] = int(np.sum(bright_mask))
        result["bright_quartile_restoration_minus_baseline_mag_median"] = (
            float(np.median(rest_minus_base[bright_mask]))
        )
    summarize_catalog_subset(
        result,
        prefix="pointlike",
        mask=pointlike,
        baseline_residual=base_resid,
        restoration_residual=rest_resid,
        restoration_minus_baseline=rest_minus_base,
    )
    summarize_catalog_subset(
        result,
        prefix="extended",
        mask=~pointlike,
        baseline_residual=base_resid,
        restoration_residual=rest_resid,
        restoration_minus_baseline=rest_minus_base,
    )
    result["_artifact_payload"] = {
        "reference_mag": ref_mag,
        "baseline_mag_cal": base_mag_cal,
        "restoration_mag_cal": rest_mag_cal,
        "baseline_residual_mag": base_resid,
        "restoration_residual_mag": rest_resid,
        "restoration_minus_baseline_mag": rest_minus_base,
        "reference_flux": ref_flux,
        "baseline_flux": base_flux,
        "restoration_flux": rest_flux,
        "pointlike": pointlike,
        "catalog_id": catalog_id,
        "catalog_id_column": np.full(
            ref_flux.shape,
            catalog["catalog_id_column"],
            dtype=object,
        ),
        "catalog_x": catalog_x,
        "catalog_y": catalog_y,
        "catalog_ra_deg": catalog_ra_deg,
        "catalog_dec_deg": catalog_dec_deg,
    }
    return result


def load_catalog_sources_in_roi(
    catalog_parquet: Path,
    *,
    band: str,
    coadd_fits_path: Path,
    roi_center: tuple[int, int] | None,
    roi_shape: tuple[int, int],
    bundle_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    table = _load_catalog_table(catalog_parquet, band=band)
    selection = (bundle_metadata or {}).get("selection")
    if (
        bundle_metadata is not None
        and isinstance(selection, dict)
        and selection.get("alignment_policy") == "reference-image-wcs"
    ):
        local_x, local_y, _, _ = project_rubin_catalog_sky_to_store_roi(
            ra=table["coord_ra"],
            dec=table["coord_dec"],
            bundle_metadata=bundle_metadata,
            roi_center=roi_center,
            roi_shape=roi_shape,
        )
    else:
        with fits.open(coadd_fits_path, memmap=True) as hdul:
            image_shape = hdul["IMAGE"].data.shape
            wcs = WCS(hdul["IMAGE"].header)

        if roi_center is None:
            roi_center = (image_shape[0] // 2, image_shape[1] // 2)
        start_h, start_w, _, _ = crop_bounds_about_center(
            center=roi_center,
            height=roi_shape[0],
            width=roi_shape[1],
            image_shape=image_shape,
        )
        full_x, full_y = wcs.world_to_pixel_values(
            table["coord_ra"],
            table["coord_dec"],
        )
        local_x = full_x - start_w
        local_y = full_y - start_h
    in_roi = (
        np.isfinite(local_x)
        & np.isfinite(local_y)
        & (local_x >= 0)
        & (local_x < roi_shape[1])
        & (local_y >= 0)
        & (local_y < roi_shape[0])
        & table["detect_isPrimary"]
        & ~table["xy_flag"]
        & np.isfinite(table["reference_flux"])
        & (table["reference_flux"] > 0)
    )
    return {
        "kind": "lsst_object_table_parquet",
        "path": str(Path(catalog_parquet).expanduser().resolve()),
        "count": int(np.sum(in_roi)),
        "x": np.asarray(local_x[in_roi], dtype=np.float64),
        "y": np.asarray(local_y[in_roi], dtype=np.float64),
        "reference_flux": table["reference_flux"][in_roi],
        "reference_flux_name": table["reference_flux_name"],
        "is_pointlike": table["is_pointlike"][in_roi],
        "catalog_id": table["catalog_id"][in_roi],
        "catalog_id_column": table["catalog_id_column"],
        "catalog_ra_deg": table["coord_ra"][in_roi],
        "catalog_dec_deg": table["coord_dec"][in_roi],
    }


def load_reference_catalog_csv_sources(
    reference_catalog_csv: Path,
    *,
    roi_center: tuple[int, int] | None,
    roi_shape: tuple[int, int],
    catalog_band: str = "r",
    bundle_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load local Rubin reference sources from pixel or sky CSVs."""

    path = Path(reference_catalog_csv).expanduser().resolve()
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Reference catalog CSV has no header: {path}")
        fieldnames = set(reader.fieldnames)
        rows.extend(reader)

    if {"x", "y"} <= fieldnames:
        return load_reference_catalog_pixel_csv_sources(
            path=path,
            rows=rows,
            fieldnames=fieldnames,
            roi_center=roi_center,
            roi_shape=roi_shape,
        )

    return load_reference_catalog_radec_csv_sources(
        path=path,
        rows=rows,
        fieldnames=fieldnames,
        catalog_band=catalog_band,
        bundle_metadata=bundle_metadata,
        roi_center=roi_center,
        roi_shape=roi_shape,
    )


def load_reference_catalog_pixel_csv_sources(
    *,
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: set[str],
    roi_center: tuple[int, int] | None,
    roi_shape: tuple[int, int],
) -> dict[str, Any]:
    """Load same-ROI pixel catalog rows produced by Rubin helpers."""

    flux_name = (
        "aperture_flux"
        if "aperture_flux" in fieldnames
        else "detection_flux"
        if "detection_flux" in fieldnames
        else None
    )
    if flux_name is None:
        raise ValueError(
            "Reference catalog CSV needs aperture_flux or detection_flux"
        )

    x = np.asarray([csv_float(row.get("x")) for row in rows], dtype=float)
    y = np.asarray([csv_float(row.get("y")) for row in rows], dtype=float)
    crop_start_y = 0
    crop_start_x = 0
    if roi_center is not None:
        crop_start_y = int(roi_center[0]) - roi_shape[0] // 2
        crop_start_x = int(roi_center[1]) - roi_shape[1] // 2
        x = x - crop_start_x
        y = y - crop_start_y
    reference_flux = np.asarray(
        [csv_float(row.get(flux_name)) for row in rows],
        dtype=float,
    )
    id_name = reference_catalog_id_column(fieldnames)
    catalog_id = reference_catalog_ids(rows, id_name=id_name)
    radec_names = optional_reference_catalog_radec_columns(fieldnames)
    if radec_names is None:
        catalog_ra_deg = np.full(len(rows), np.nan, dtype=np.float64)
        catalog_dec_deg = np.full(len(rows), np.nan, dtype=np.float64)
    else:
        ra_name, dec_name = radec_names
        catalog_ra_deg = np.asarray(
            [csv_float(row.get(ra_name)) for row in rows],
            dtype=np.float64,
        )
        catalog_dec_deg = np.asarray(
            [csv_float(row.get(dec_name)) for row in rows],
            dtype=np.float64,
        )
    in_roi = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(reference_flux)
        & (reference_flux > 0)
        & (x >= 0)
        & (x < roi_shape[1])
        & (y >= 0)
        & (y < roi_shape[0])
    )
    return {
        "kind": "rubin_reference_catalog_csv",
        "path": str(path),
        "provenance": {"source": "csv"},
        "coordinate_frame": (
            "evaluation_roi_crop" if roi_center is not None else "input_csv"
        ),
        "crop_origin_yx": [crop_start_y, crop_start_x],
        "count": int(np.sum(in_roi)),
        "x": x[in_roi],
        "y": y[in_roi],
        "reference_flux": reference_flux[in_roi],
        "reference_flux_name": flux_name,
        "is_pointlike": np.zeros(int(np.sum(in_roi)), dtype=bool),
        "catalog_id": catalog_id[in_roi],
        "catalog_id_column": id_name,
        "catalog_ra_deg": catalog_ra_deg[in_roi],
        "catalog_dec_deg": catalog_dec_deg[in_roi],
    }


def load_reference_catalog_radec_csv_sources(
    *,
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: set[str],
    catalog_band: str,
    bundle_metadata: dict[str, Any] | None,
    roi_center: tuple[int, int] | None,
    roi_shape: tuple[int, int],
) -> dict[str, Any]:
    """Load a local Rubin catalog with sky coordinates."""

    ra_name, dec_name = reference_catalog_radec_columns(fieldnames)
    flux_name = reference_catalog_flux_column(fieldnames, catalog_band)
    if bundle_metadata is None:
        raise ValueError(
            "Reference catalog CSV with sky coordinates requires Rubin store "
            "metadata to project RA/Dec into the xImage ROI"
        )

    ra = np.asarray(
        [csv_float(row.get(ra_name)) for row in rows], dtype=float
    )
    dec = np.asarray(
        [csv_float(row.get(dec_name)) for row in rows], dtype=float
    )
    x, y, crop_start_y, crop_start_x = project_rubin_catalog_sky_to_store_roi(
        ra=ra,
        dec=dec,
        bundle_metadata=bundle_metadata,
        roi_center=roi_center,
        roi_shape=roi_shape,
    )
    reference_flux = np.asarray(
        [csv_float(row.get(flux_name)) for row in rows],
        dtype=float,
    )
    pointlike = reference_catalog_pointlike_flags(
        rows,
        fieldnames=fieldnames,
        catalog_band=catalog_band,
    )
    id_name = reference_catalog_id_column(fieldnames)
    catalog_id = reference_catalog_ids(rows, id_name=id_name)
    keep = reference_catalog_row_mask(
        rows,
        fieldnames=fieldnames,
        flux_name=flux_name,
    )
    in_roi = (
        keep
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(reference_flux)
        & (reference_flux > 0)
        & (x >= 0)
        & (x < roi_shape[1])
        & (y >= 0)
        & (y < roi_shape[0])
    )
    return {
        "kind": "rubin_catalog_csv",
        "path": str(path),
        "provenance": {"source": "csv"},
        "coordinate_frame": (
            "reference_wcs_evaluation_roi_crop"
            if roi_center is not None
            else "reference_wcs_store_roi"
        ),
        "crop_origin_yx": [crop_start_y, crop_start_x],
        "count": int(np.sum(in_roi)),
        "x": x[in_roi],
        "y": y[in_roi],
        "reference_flux": reference_flux[in_roi],
        "reference_flux_name": flux_name,
        "is_pointlike": pointlike[in_roi],
        "catalog_id": catalog_id[in_roi],
        "catalog_id_column": id_name,
        "catalog_ra_deg": ra[in_roi],
        "catalog_dec_deg": dec[in_roi],
    }


def reference_catalog_id_column(fieldnames: set[str]) -> str | None:
    """Return the first stable catalog identity column."""

    for name in ("objectId", "object_id", "id"):
        if name in fieldnames:
            return name
    return None


def reference_catalog_ids(
    rows: list[dict[str, str]],
    *,
    id_name: str | None,
) -> np.ndarray:
    """Keep catalog identifiers as strings so large IDs stay exact."""

    if id_name is None:
        return np.full(len(rows), None, dtype=object)
    return np.asarray(
        [
            None
            if row.get(id_name) is None or str(row[id_name]).strip() == ""
            else str(row[id_name]).strip()
            for row in rows
        ],
        dtype=object,
    )


def optional_reference_catalog_radec_columns(
    fieldnames: set[str],
) -> tuple[str, str] | None:
    for ra_name, dec_name in (
        ("coord_ra", "coord_dec"),
        ("ra", "dec"),
        ("ra_deg", "dec_deg"),
    ):
        if ra_name in fieldnames and dec_name in fieldnames:
            return ra_name, dec_name
    return None


def reference_catalog_radec_columns(fieldnames: set[str]) -> tuple[str, str]:
    names = optional_reference_catalog_radec_columns(fieldnames)
    if names is not None:
        return names
    raise ValueError(
        "Reference catalog CSV needs x/y columns or sky coordinate columns "
        "(coord_ra/coord_dec, ra/dec, or ra_deg/dec_deg)"
    )


def reference_catalog_flux_column(
    fieldnames: set[str],
    catalog_band: str,
) -> str:
    band = catalog_band
    candidates = [
        f"{band}_psfFlux",
        f"{band}_cModelFlux",
        f"{band}_calibFlux",
        "psfFlux",
        "ap12Flux",
        "ap25Flux",
        "ap17Flux",
        "ap09Flux",
        "ap06Flux",
        "ap03Flux",
        "ap35Flux",
        "aperture_flux",
        "detection_flux",
    ]
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError(
        "Reference catalog CSV needs a supported flux column "
        f"for band {catalog_band!r}"
    )


def reference_catalog_pointlike_flags(
    rows: list[dict[str, str]],
    *,
    fieldnames: set[str],
    catalog_band: str,
) -> np.ndarray:
    candidates = [
        "refExtendedness",
        f"{catalog_band}_extendedness",
        "extendedness",
    ]
    for name in candidates:
        if name in fieldnames:
            values = np.asarray(
                [csv_float(row.get(name)) for row in rows],
                dtype=float,
            )
            return np.isfinite(values) & (values < 0.5)
    return np.zeros(len(rows), dtype=bool)


def reference_catalog_row_mask(
    rows: list[dict[str, str]],
    *,
    fieldnames: set[str],
    flux_name: str,
) -> np.ndarray:
    keep = np.ones(len(rows), dtype=bool)
    if "detect_isPrimary" in fieldnames:
        keep &= np.asarray(
            [
                csv_bool(row.get("detect_isPrimary"), default=True)
                for row in rows
            ],
            dtype=bool,
        )
    flux_flag = f"{flux_name}_flag"
    if flux_flag in fieldnames:
        keep &= ~np.asarray(
            [csv_bool(row.get(flux_flag), default=False) for row in rows],
            dtype=bool,
        )
    return keep


def project_rubin_catalog_sky_to_store_roi(
    *,
    ra: np.ndarray,
    dec: np.ndarray,
    bundle_metadata: dict[str, Any],
    roi_center: tuple[int, int] | None,
    roi_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    selection = bundle_metadata.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Rubin store metadata lacks a selection block")
    if selection.get("alignment_policy") != "reference-image-wcs":
        raise ValueError(
            "Sky-coordinate reference catalogs require a reference-WCS "
            "aligned Rubin store"
        )
    roi_text = selection.get("roi")
    if not isinstance(roi_text, str):
        raise ValueError("Rubin store metadata lacks a selection ROI")

    reference_path = _rubin_reference_path(bundle_metadata)
    with fits.open(reference_path, memmap=True) as hdul:
        reference_wcs = WCS(hdul["IMAGE"].header)
    full_x, full_y = reference_wcs.all_world2pix(ra, dec, 0, quiet=True)

    store_y, store_x = parse_section_text(roi_text)
    x = np.asarray(full_x, dtype=np.float64) - store_x.start
    y = np.asarray(full_y, dtype=np.float64) - store_y.start
    crop_start_y = 0
    crop_start_x = 0
    if roi_center is not None:
        crop_start_y = int(roi_center[0]) - roi_shape[0] // 2
        crop_start_x = int(roi_center[1]) - roi_shape[1] // 2
        x = x - crop_start_x
        y = y - crop_start_y
    return x, y, crop_start_y, crop_start_x


def match_detections_to_catalog(
    *,
    detection_x: np.ndarray,
    detection_y: np.ndarray,
    catalog_x: np.ndarray,
    catalog_y: np.ndarray,
    match_radius_px: float,
) -> dict[str, Any]:
    if detection_x.size == 0 or catalog_x.size == 0:
        return {
            "matched_count": 0,
            "detection_indices": np.array([], dtype=np.int64),
            "catalog_indices": np.array([], dtype=np.int64),
        }
    detection_points = np.column_stack([detection_x, detection_y])
    catalog_points = np.column_stack([catalog_x, catalog_y])
    tree = cKDTree(catalog_points)
    neighbor_lists = tree.query_ball_point(
        detection_points, r=match_radius_px
    )
    counts = np.asarray([len(candidates) for candidates in neighbor_lists])
    if counts.sum() == 0:
        return {
            "matched_count": 0,
            "detection_indices": np.array([], dtype=np.int64),
            "catalog_indices": np.array([], dtype=np.int64),
        }

    det_indices = np.repeat(np.arange(detection_points.shape[0]), counts)
    cat_indices = np.concatenate(
        [
            np.asarray(candidates, dtype=np.int64)
            for candidates in neighbor_lists
            if candidates
        ]
    )
    deltas = detection_points[det_indices] - catalog_points[cat_indices]
    distances = np.sqrt(np.sum(deltas * deltas, axis=1))

    num_det = detection_points.shape[0]
    num_cat = catalog_points.shape[0]
    unmatched_cost = float(match_radius_px) + 0.5
    graph = _build_sparse_match_graph(
        det_indices=det_indices,
        cat_indices=cat_indices,
        distances=distances,
        num_det=num_det,
        num_cat=num_cat,
        unmatched_cost=unmatched_cost,
    )
    row_ind, col_ind = min_weight_full_bipartite_matching(graph)
    matched_mask = col_ind < num_cat
    matched_det = row_ind[matched_mask]
    matched_cat = col_ind[matched_mask]
    return {
        "matched_count": int(matched_det.size),
        "detection_indices": np.asarray(matched_det, dtype=np.int64),
        "catalog_indices": np.asarray(matched_cat, dtype=np.int64),
    }


def _build_sparse_match_graph(
    *,
    det_indices: np.ndarray,
    cat_indices: np.ndarray,
    distances: np.ndarray,
    num_det: int,
    num_cat: int,
    unmatched_cost: float,
) -> csr_matrix:
    row_idx = np.concatenate(
        [det_indices, np.arange(num_det, dtype=np.int64)]
    )
    col_idx = np.concatenate(
        [cat_indices, num_cat + np.arange(num_det, dtype=np.int64)]
    )
    data = np.concatenate(
        [
            distances + 1e-12,
            np.full(num_det, unmatched_cost, dtype=np.float64),
        ]
    )
    return csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(num_det, num_cat + num_det),
    )


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    return -2.5 * np.log10(np.asarray(flux, dtype=np.float64))


def select_calibration_subset(
    reference_flux: np.ndarray,
    pointlike: np.ndarray,
) -> np.ndarray:
    bright_cut = np.quantile(reference_flux, 0.75)
    bright_mask = reference_flux >= bright_cut
    pointlike_bright = bright_mask & pointlike
    if np.sum(pointlike_bright) >= 3:
        return pointlike_bright
    if np.sum(bright_mask) >= 3:
        return bright_mask
    return np.ones(reference_flux.shape, dtype=bool)


def project_restoration_to_observation_grid(
    image: np.ndarray,
    *,
    target_shape: tuple[int, int],
    srf: int,
) -> np.ndarray:
    arr = squeeze_single_image(image)
    if srf > 1:
        arr = average_downsample(arr, factor=srf)
    return center_crop(arr, target_shape)


def average_downsample(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    height = (image.shape[0] // factor) * factor
    width = (image.shape[1] // factor) * factor
    trimmed = image[:height, :width]
    return trimmed.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
    ).mean(axis=(1, 3))


def center_crop(
    image: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_shape
    source_h, source_w = image.shape
    if target_h > source_h or target_w > source_w:
        raise ValueError(
            f"Cannot center-crop {image.shape} to {target_shape}."
        )
    start_h = (source_h - target_h) // 2
    start_w = (source_w - target_w) // 2
    return image[
        start_h : start_h + target_h,
        start_w : start_w + target_w,
    ]


def squeeze_single_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        if arr.shape[0] != 1 or arr.shape[-1] != 1:
            raise ValueError(
                f"Expected a single image, saw shape {arr.shape}."
            )
        return arr[0, :, :, 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected rank-2 or rank-4 image, saw {arr.shape}.")


def squeeze_exposure_stack(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(
            "Expected exposure stack with shape (1, H, W, N), "
            f"saw {arr.shape}."
        )
    return np.transpose(arr[0], (2, 0, 1))


def resolve_pipeline_coadd_fits(
    *,
    explicit_path: Path | None = None,
) -> Path | None:
    if explicit_path is None:
        return None
    path = Path(explicit_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline coadd FITS not found: {path}")
    return path


def resolve_catalog_parquet(
    *,
    explicit_path: Path | None = None,
) -> Path | None:
    if explicit_path is None:
        return None
    path = Path(explicit_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Catalog parquet not found: {path}")
    return path


def resolve_reference_catalog_csv(
    *,
    explicit_path: Path | None = None,
) -> Path | None:
    if explicit_path is None:
        return None
    path = Path(explicit_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference catalog CSV not found: {path}")
    if path.is_dir():
        object_csv = path / "object.csv"
        if object_csv.exists():
            return object_csv
        csv_candidates = sorted(path.glob("*.csv"))
        if len(csv_candidates) == 1:
            return csv_candidates[0]
        if not csv_candidates:
            raise FileNotFoundError(
                f"Reference catalog directory contains no CSV outputs: {path}"
            )
        raise ValueError(
            "Reference catalog directory contains multiple CSV outputs; "
            "specify a CSV file explicitly or include object.csv."
        )
    return path


def load_pipeline_coadd_roi(
    path: Path,
    *,
    center: tuple[int, int] | None,
    height: int,
    width: int,
    bundle_metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    if bundle_metadata is not None:
        aligned = load_rubin_reference_aligned_coadd_roi(
            path,
            center=center,
            height=height,
            width=width,
            bundle_metadata=bundle_metadata,
        )
        if aligned is not None:
            return aligned

    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul["IMAGE"].data, dtype=np.float64)

    if center is None:
        center = (image.shape[0] // 2, image.shape[1] // 2)
    return crop_about_center(
        image,
        center=center,
        height=height,
        width=width,
    )


def load_rubin_reference_aligned_coadd_roi(
    path: Path,
    *,
    center: tuple[int, int] | None,
    height: int,
    width: int,
    bundle_metadata: dict[str, Any],
) -> np.ndarray | None:
    selection = bundle_metadata.get("selection")
    if not isinstance(selection, dict):
        return None
    if selection.get("alignment_policy") != "reference-image-wcs":
        return None
    reference_index = selection.get("reference_index")
    roi_text = selection.get("roi")
    source_dir = bundle_metadata.get("source_dir")
    rubin_files = bundle_metadata.get("rubin_files")
    if (
        not isinstance(reference_index, int)
        or not isinstance(roi_text, str)
        or not isinstance(source_dir, str)
        or not isinstance(rubin_files, list)
        or reference_index >= len(rubin_files)
    ):
        return None
    reference_row = rubin_files[reference_index]
    if not isinstance(reference_row, dict):
        return None
    reference_relative = reference_row.get("relative_path")
    if not isinstance(reference_relative, str):
        return None

    reference_path = Path(source_dir) / reference_relative
    if not reference_path.exists():
        return None

    with fits.open(path, memmap=True) as hdul:
        coadd_hdu = hdul["IMAGE"]
        if (
            "CTYPE1" not in coadd_hdu.header
            or "CTYPE2" not in coadd_hdu.header
        ):
            return None
        coadd_wcs = WCS(coadd_hdu.header)
        coadd_image = np.asarray(coadd_hdu.data, dtype=np.float64)
    with fits.open(reference_path, memmap=True) as hdul:
        reference_wcs = WCS(hdul["IMAGE"].header)

    store_y, store_x = parse_section_text(roi_text)
    if center is None:
        store_center = (
            (store_y.stop - store_y.start) // 2,
            (store_x.stop - store_x.start) // 2,
        )
    else:
        store_center = center
    y0 = store_y.start + store_center[0] - height // 2
    x0 = store_x.start + store_center[1] - width // 2
    y1 = y0 + height
    x1 = x0 + width
    yy, xx = np.mgrid[y0:y1, x0:x1]
    reference_pixels = np.column_stack(
        [xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)]
    )
    world = reference_wcs.all_pix2world(reference_pixels, 0)
    coadd_pixels = coadd_wcs.all_world2pix(world, 0, quiet=True)
    coadd_x = coadd_pixels[:, 0].reshape(xx.shape)
    coadd_y = coadd_pixels[:, 1].reshape(yy.shape)
    image_height, image_width = coadd_image.shape
    valid = (
        np.isfinite(coadd_x)
        & np.isfinite(coadd_y)
        & (coadd_x >= 0)
        & (coadd_x <= image_width - 1)
        & (coadd_y >= 0)
        & (coadd_y <= image_height - 1)
    )
    coords = np.vstack(
        [
            np.where(valid, coadd_y, 0.0).ravel(),
            np.where(valid, coadd_x, 0.0).ravel(),
        ]
    )
    aligned = map_coordinates(
        coadd_image,
        coords,
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape((height, width))
    aligned[~valid] = np.nan
    return aligned


def build_rubin_coadd_reference_catalog(
    *,
    coadd_fits_path: Path,
    store_dir: Path,
    output_csv: Path,
    output_summary: Path | None = None,
    threshold: float = 5.0,
    minarea: int = 5,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Detect coadd sources on the reconstruction pixel grid."""

    coadd_fits_path = Path(coadd_fits_path).expanduser().resolve()
    store_dir = Path(store_dir).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    if output_summary is None:
        output_summary = output_csv.with_suffix(".summary.json")
    else:
        output_summary = Path(output_summary).expanduser().resolve()

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if minarea <= 0:
        raise ValueError("minarea must be positive")
    for path in (output_csv, output_summary):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    metadata_path = store_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing store metadata: {metadata_path}")
    bundle_metadata = json.loads(metadata_path.read_text())
    selection = bundle_metadata.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Store metadata lacks a selection block")
    roi_text = selection.get("roi")
    if not isinstance(roi_text, str):
        raise ValueError("Store metadata lacks a selection ROI")

    store_y, store_x = parse_section_text(roi_text)
    height = store_y.stop - store_y.start
    width = store_x.stop - store_x.start
    aligned = load_rubin_reference_aligned_coadd_roi(
        coadd_fits_path,
        center=None,
        height=height,
        width=width,
        bundle_metadata=bundle_metadata,
    )
    if aligned is None:
        raise ValueError(
            "Could not align coadd to the Rubin reference-WCS store ROI"
        )

    reference_path = _rubin_reference_path(bundle_metadata)
    with fits.open(reference_path, memmap=True) as hdul:
        reference_wcs = WCS(hdul["IMAGE"].header)

    rows = detect_coadd_reference_sources(
        aligned,
        reference_wcs=reference_wcs,
        store_y=store_y,
        store_x=store_x,
        threshold=threshold,
        minarea=minarea,
    )
    fieldnames = [
        "source_id",
        "detection_index",
        "x",
        "y",
        "reference_x",
        "reference_y",
        "ra_deg",
        "dec_deg",
        "a",
        "b",
        "theta",
        "kron_radius",
        "aperture_scale",
        "aperture_flux",
        "aperture_flux_err",
        "aperture_flags",
        "kron_flags",
        "detection_flux",
        "detection_peak",
        "detection_npix",
        "detection_tnpix",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "catalog_kind": "rubin_pipeline_coadd_derived_reference",
        "coadd_fits_path": str(coadd_fits_path),
        "store_dir": str(store_dir),
        "store_roi": roi_text,
        "reference_path": str(reference_path),
        "aligned_shape": [height, width],
        "threshold": float(threshold),
        "minarea": int(minarea),
        "source_count": len(rows),
        "output_csv": str(output_csv),
        "output_summary": str(output_summary),
        "columns": fieldnames,
        "note": (
            "Coadd-derived detections and fluxes share the image noise; "
            "use an independent reference for external validation."
        ),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def detect_coadd_reference_sources(
    image: np.ndarray,
    *,
    reference_wcs: WCS,
    store_y: slice,
    store_x: slice,
    threshold: float,
    minarea: int,
) -> list[dict[str, Any]]:
    prepared = _prepare_photometry_image(image)
    data = subtract_photometry_background(prepared)
    objects = detect_sources(data, threshold=threshold, minarea=minarea)
    if len(objects) == 0:
        return []

    x = objects["x"].astype(np.float64)
    y = objects["y"].astype(np.float64)
    a = np.maximum(objects["a"].astype(np.float64), 1e-3)
    b = np.maximum(objects["b"].astype(np.float64), 1e-3)
    theta = objects["theta"].astype(np.float64)
    kronrad, kron_flags = kron_radius(data, x, y, a, b, theta, max_radius=6.0)
    aperture_scale = np.where(
        np.isfinite(kronrad) & (kronrad > 0),
        2.5 * kronrad,
        6.0,
    )
    flux, flux_err, aperture_flags = elliptical_aperture_flux(
        data,
        x,
        y,
        a,
        b,
        theta,
        scale=aperture_scale,
    )
    reference_x = store_x.start + x
    reference_y = store_y.start + y
    world = reference_wcs.all_pix2world(
        np.column_stack([reference_x, reference_y]),
        0,
    )
    order = np.argsort(
        -np.where(np.isfinite(flux), flux, -np.inf),
        kind="stable",
    )

    rows: list[dict[str, Any]] = []
    for source_id, idx in enumerate(order):
        rows.append(
            {
                "source_id": source_id,
                "detection_index": int(idx),
                "x": finite_float(x[idx]),
                "y": finite_float(y[idx]),
                "reference_x": finite_float(reference_x[idx]),
                "reference_y": finite_float(reference_y[idx]),
                "ra_deg": finite_float(world[idx, 0]),
                "dec_deg": finite_float(world[idx, 1]),
                "a": finite_float(a[idx]),
                "b": finite_float(b[idx]),
                "theta": finite_float(theta[idx]),
                "kron_radius": finite_float(kronrad[idx]),
                "aperture_scale": finite_float(aperture_scale[idx]),
                "aperture_flux": finite_float(flux[idx]),
                "aperture_flux_err": finite_float(flux_err[idx]),
                "aperture_flags": int(aperture_flags[idx]),
                "kron_flags": int(kron_flags[idx]),
                "detection_flux": finite_float(
                    object_field(objects, "flux", idx)
                ),
                "detection_peak": finite_float(
                    object_field(objects, "peak", idx)
                ),
                "detection_npix": int(object_field(objects, "npix", idx, 0)),
                "detection_tnpix": int(
                    object_field(objects, "tnpix", idx, 0)
                ),
            }
        )
    return rows


def _rubin_reference_path(bundle_metadata: dict[str, Any]) -> Path:
    selection = bundle_metadata.get("selection")
    source_dir = bundle_metadata.get("source_dir")
    rubin_files = bundle_metadata.get("rubin_files")
    if (
        not isinstance(selection, dict)
        or not isinstance(source_dir, str)
        or not isinstance(rubin_files, list)
    ):
        raise ValueError("Store metadata lacks Rubin reference information")
    reference_index = selection.get("reference_index")
    if (
        not isinstance(reference_index, int)
        or reference_index < 0
        or reference_index >= len(rubin_files)
    ):
        raise ValueError("Store metadata has an invalid reference index")
    reference_row = rubin_files[reference_index]
    if not isinstance(reference_row, dict):
        raise ValueError("Store metadata has an invalid reference row")
    relative_path = reference_row.get("relative_path")
    if not isinstance(relative_path, str):
        raise ValueError("Store metadata lacks a reference relative path")
    reference_path = Path(source_dir) / relative_path
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference FITS: {reference_path}")
    return reference_path


def object_field(
    objects: np.ndarray,
    name: str,
    index: int,
    default: Any = None,
) -> Any:
    if objects.dtype.names is None or name not in objects.dtype.names:
        return default
    return objects[name][index]


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def csv_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(result):
        return np.nan
    return result


def csv_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_section_text(value: str) -> tuple[slice, slice]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid section: {value}")
    return tuple(parse_slice_text(part) for part in parts)  # type: ignore[return-value]


def parse_slice_text(value: str) -> slice:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError(f"Invalid slice: {value}")
    return slice(int(pieces[0]), int(pieces[1]))


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


def crop_about_center(
    image: np.ndarray,
    *,
    center: tuple[int, int],
    height: int,
    width: int,
) -> np.ndarray:
    start_h = center[0] - height // 2
    start_w = center[1] - width // 2
    stop_h = start_h + height
    stop_w = start_w + width
    if (
        start_h < 0
        or start_w < 0
        or stop_h > image.shape[0]
        or stop_w > image.shape[1]
    ):
        raise ValueError(
            f"Requested ROI {(height, width)} around {center} exceeds "
            f"image bounds {image.shape}."
        )
    cropped = image[start_h:stop_h, start_w:stop_w]
    if cropped.shape != (height, width):
        raise ValueError(
            f"Expected cropped shape {(height, width)}, saw {cropped.shape}."
        )
    return cropped


def crop_bounds_about_center(
    *,
    center: tuple[int, int],
    height: int,
    width: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    start_h = center[0] - height // 2
    start_w = center[1] - width // 2
    stop_h = start_h + height
    stop_w = start_w + width
    if (
        start_h < 0
        or start_w < 0
        or stop_h > image_shape[0]
        or stop_w > image_shape[1]
    ):
        raise ValueError(
            f"Requested ROI {(height, width)} around {center} exceeds "
            f"image bounds {image_shape}."
        )
    return start_h, start_w, stop_h, stop_w


def _prepare_photometry_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image, dtype=np.float32))


def subtract_photometry_background(image: np.ndarray) -> np.ndarray:
    if min(image.shape) < 32:
        return image - np.nanmedian(image)
    try:
        return subtract_background(image)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        warnings.warn(
            "Background estimation failed; falling back to median "
            f"subtraction: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return image - np.nanmedian(image)


def is_recoverable_photometry_error(exc: Exception) -> bool:
    return (
        isinstance(
            exc,
            (ArithmeticError, OSError, RuntimeError, TypeError, ValueError),
        )
        or type(exc) is Exception
    )


def unavailable_photometry(
    *,
    baseline_name: str,
    threshold: float,
    minarea: int,
    reason: str,
) -> dict[str, Any]:
    warnings.warn(
        "Photometry evaluation failed; marking photometry "
        f"unavailable: {reason}",
        RuntimeWarning,
        stacklevel=2,
    )
    return {
        "available": False,
        "baseline_name": baseline_name,
        "threshold": threshold,
        "minarea": minarea,
        "reason": reason,
    }


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


def write_catalog_photometry_csv(
    path: Path,
    payload: dict[str, np.ndarray],
) -> None:
    rows = build_catalog_photometry_rows(payload)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_catalog_photometry_rows(
    payload: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    size = int(np.asarray(payload["reference_mag"]).shape[0])
    for idx in range(size):
        row = {
            "reference_mag": float(payload["reference_mag"][idx]),
            "baseline_mag_cal": float(payload["baseline_mag_cal"][idx]),
            "restoration_mag_cal": float(payload["restoration_mag_cal"][idx]),
            "baseline_residual_mag": float(
                payload["baseline_residual_mag"][idx]
            ),
            "restoration_residual_mag": float(
                payload["restoration_residual_mag"][idx]
            ),
            "restoration_minus_baseline_mag": float(
                payload["restoration_minus_baseline_mag"][idx]
            ),
            "reference_flux": float(payload["reference_flux"][idx]),
            "baseline_flux": float(payload["baseline_flux"][idx]),
            "restoration_flux": float(payload["restoration_flux"][idx]),
            "is_pointlike": bool(payload["pointlike"][idx]),
        }
        for name in (
            "catalog_id",
            "catalog_id_column",
            "catalog_x",
            "catalog_y",
            "catalog_ra_deg",
            "catalog_dec_deg",
        ):
            if name not in payload:
                continue
            value = np.asarray(payload[name], dtype=object)[idx]
            if name.startswith("catalog_") and name not in {
                "catalog_id",
                "catalog_id_column",
            }:
                value = finite_float(value)
            elif value is not None:
                value = str(value)
            row[name] = value
        rows.append(row)
    return rows


def write_catalog_photometry_binned_csv(
    path: Path,
    payload: dict[str, np.ndarray],
) -> None:
    rows = build_catalog_photometry_binned_rows(payload)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_catalog_photometry_binned_rows(
    payload: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    reference_mag = np.asarray(payload["reference_mag"], dtype=np.float64)
    delta_mag = np.asarray(
        payload["restoration_minus_baseline_mag"], dtype=np.float64
    )
    baseline_residual = np.asarray(
        payload["baseline_residual_mag"], dtype=np.float64
    )
    restoration_residual = np.asarray(
        payload["restoration_residual_mag"], dtype=np.float64
    )
    pointlike = np.asarray(payload["pointlike"], dtype=bool)
    if reference_mag.size == 0:
        return []

    num_bins = min(6, max(1, int(np.sqrt(reference_mag.size))))
    if np.allclose(reference_mag.min(), reference_mag.max()):
        bins = np.array([reference_mag.min(), reference_mag.max() + 1e-6])
    else:
        bins = np.linspace(
            reference_mag.min(), reference_mag.max(), num_bins + 1
        )

    rows: list[dict[str, Any]] = []
    for subset_name, subset_mask in (
        ("all", np.ones(reference_mag.shape, dtype=bool)),
        ("pointlike", pointlike),
        ("extended", ~pointlike),
    ):
        if not np.any(subset_mask):
            continue
        for idx in range(len(bins) - 1):
            left = bins[idx]
            right = bins[idx + 1]
            if idx == len(bins) - 2:
                mask = (
                    subset_mask
                    & (reference_mag >= left)
                    & (reference_mag <= right)
                )
            else:
                mask = (
                    subset_mask
                    & (reference_mag >= left)
                    & (reference_mag < right)
                )
            if not np.any(mask):
                continue
            rows.append(
                {
                    "subset": subset_name,
                    "mag_bin_left": float(left),
                    "mag_bin_right": float(right),
                    "count": int(np.sum(mask)),
                    "delta_mag_median": float(np.median(delta_mag[mask])),
                    "baseline_residual_mag_median": float(
                        np.median(baseline_residual[mask])
                    ),
                    "restoration_residual_mag_median": float(
                        np.median(restoration_residual[mask])
                    ),
                }
            )
    return rows


def summarize_catalog_subset(
    result: dict[str, Any],
    *,
    prefix: str,
    mask: np.ndarray,
    baseline_residual: np.ndarray,
    restoration_residual: np.ndarray,
    restoration_minus_baseline: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    result[f"{prefix}_catalog_count"] = int(np.sum(mask))
    result[f"{prefix}_baseline_residual_mag_median"] = float(
        np.median(baseline_residual[mask])
    )
    result[f"{prefix}_restoration_residual_mag_median"] = float(
        np.median(restoration_residual[mask])
    )
    result[f"{prefix}_restoration_minus_baseline_mag_median"] = float(
        np.median(restoration_minus_baseline[mask])
    )


def _load_catalog_table(
    catalog_parquet: Path, *, band: str
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    band_prefix = f"{band}_"
    candidates = [
        (f"{band_prefix}calibFlux", f"{band_prefix}calibFlux_flag"),
        (f"{band_prefix}cModelFlux", f"{band_prefix}cModel_flag"),
        (f"{band_prefix}psfFlux", f"{band_prefix}psfFlux_flag"),
    ]
    base_columns = [
        "coord_ra",
        "coord_dec",
        "detect_isPrimary",
        "xy_flag",
        "refExtendedness",
        f"{band_prefix}extendedness",
    ]
    available_names = set(pq.ParquetFile(catalog_parquet).schema_arrow.names)
    missing = [
        name
        for name in ("coord_ra", "coord_dec")
        if name not in available_names
    ]
    if missing:
        raise ValueError(
            "Catalog parquet is missing required columns: "
            f"{', '.join(missing)}"
        )
    read_columns = [name for name in base_columns if name in available_names]
    id_name = next(
        (
            name
            for name in ("objectId", "object_id", "id")
            if name in available_names
        ),
        None,
    )
    if id_name is not None:
        read_columns.append(id_name)
    for flux_name, flag_name in candidates:
        if flux_name in available_names:
            read_columns.append(flux_name)
        if flag_name in available_names:
            read_columns.append(flag_name)

    table = pq.read_table(catalog_parquet, columns=read_columns)
    arrays = {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in read_columns
    }
    if table.num_rows == 0:
        return {
            "coord_ra": np.array([], dtype=np.float64),
            "coord_dec": np.array([], dtype=np.float64),
            "detect_isPrimary": np.array([], dtype=bool),
            "xy_flag": np.array([], dtype=bool),
            "reference_flux": np.array([], dtype=np.float64),
            "reference_flux_name": "auto",
            "is_pointlike": np.array([], dtype=bool),
            "catalog_id": np.array([], dtype=object),
            "catalog_id_column": id_name,
        }

    pointlike = np.zeros(table.num_rows, dtype=bool)
    if "refExtendedness" in arrays:
        ref_ext = np.asarray(arrays["refExtendedness"], dtype=np.float64)
        pointlike = np.isfinite(ref_ext) & (ref_ext < 0.5)
    elif f"{band_prefix}extendedness" in arrays:
        ext = np.asarray(
            arrays[f"{band_prefix}extendedness"], dtype=np.float64
        )
        pointlike = np.isfinite(ext) & (ext < 0.5)

    fluxes = []
    for flux_name, flag_name in candidates:
        if flux_name not in arrays:
            continue
        flux = np.asarray(arrays[flux_name], dtype=np.float64)
        if flag_name in arrays:
            flag = np.asarray(arrays[flag_name], dtype=bool)
        else:
            flag = np.zeros(flux.shape, dtype=bool)
        valid = np.isfinite(flux) & (flux > 0) & ~flag
        fluxes.append((flux_name, flux, valid))

    reference_flux = np.full(table.num_rows, np.nan, dtype=np.float64)
    reference_name = np.full(table.num_rows, "auto", dtype=object)
    for flux_name, flux, valid in fluxes:
        take = np.isnan(reference_flux) & valid
        reference_flux[take] = flux[take]
        reference_name[take] = flux_name

    return {
        "coord_ra": np.asarray(arrays["coord_ra"], dtype=np.float64),
        "coord_dec": np.asarray(arrays["coord_dec"], dtype=np.float64),
        "detect_isPrimary": np.asarray(
            arrays.get(
                "detect_isPrimary", np.ones(table.num_rows, dtype=bool)
            ),
            dtype=bool,
        ),
        "xy_flag": np.asarray(
            arrays.get("xy_flag", np.zeros(table.num_rows, dtype=bool)),
            dtype=bool,
        ),
        "reference_flux": reference_flux,
        "reference_flux_name": (
            reference_name[0]
            if np.all(reference_name == reference_name[0])
            else "auto"
        ),
        "is_pointlike": pointlike,
        "catalog_id": (
            np.asarray(
                [
                    None if value is None else str(value)
                    for value in arrays[id_name]
                ],
                dtype=object,
            )
            if id_name is not None
            else np.full(table.num_rows, None, dtype=object)
        ),
        "catalog_id_column": id_name,
    }
