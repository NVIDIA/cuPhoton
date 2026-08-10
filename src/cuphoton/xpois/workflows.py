# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow helpers for XPOIS."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton import __version__
from cuphoton.core.runtime import runtime_metadata

from .data import (
    apply_rectangular_cutout,
    load_image_with_wcs,
    load_mask_with_planes,
    load_variance_with_wcs,
)
from .ois import (
    EXPLICIT_BACKENDS,
    GaussianBasisComponent,
    build_compact_source_stamp_mask,
    solve_constant_kernel,
)
from .review import identify_residual_hotspots, write_review_metadata
from .review_bokeh import write_interactive_review_artifact


@dataclass
class WorkflowResult:
    """Persisted XPOIS workflow result.

    Attributes
    ----------
    run_dir
        Directory owning numeric artifacts, metadata, and optional review
        output.
    summary
        JSON-compatible inputs, fit diagnostics, and artifact paths.
    """

    run_dir: Path
    summary: dict[str, Any]


MASK_POLICY_NONE = "none"
MASK_POLICY_STRICT = "strict"
MASK_POLICY_HSC_MASKLITE = "hsc-masklite"
_MASK_POLICY_ALIASES = {
    MASK_POLICY_NONE: MASK_POLICY_NONE,
    MASK_POLICY_STRICT: MASK_POLICY_STRICT,
    MASK_POLICY_HSC_MASKLITE: MASK_POLICY_HSC_MASKLITE,
    "masklite": MASK_POLICY_HSC_MASKLITE,
}
_HSC_MASKLITE_PLANES = (
    "BAD",
    "SAT",
    "INTRP",
    "CR",
    "EDGE",
    "SUSPECT",
    "NO_DATA",
    "CROSSTALK",
    "UNMASKEDNAN",
)


def run_constant_kernel_fit(
    *,
    reference_path: Path,
    target_path: Path,
    output_root: Path,
    name: str | None,
    reference_hdu: int | None,
    target_hdu: int | None,
    kernel_shape: tuple[int, int],
    components: list[GaussianBasisComponent],
    variance_path: Path | None,
    variance_hdu: int | None = None,
    reference_mask_path: Path | None = None,
    target_mask_path: Path | None = None,
    reference_mask_hdu: int | None = None,
    target_mask_hdu: int | None = None,
    mask_policy: str = MASK_POLICY_NONE,
    crop_y0: int | None = None,
    crop_x0: int | None = None,
    crop_height: int | None = None,
    crop_width: int | None = None,
    fit_mask_path: Path | None,
    auto_stamp_mask: bool = False,
    auto_stamp_size: int = 31,
    auto_stamp_count: int = 5,
    auto_peak_percentile: float = 99.5,
    background_degree: int,
    flux_conserve: bool,
    backend: str = "auto",
    workflow_name: str = "fit_kernel",
    run_prefix: str = "fit-kernel",
) -> WorkflowResult:
    """Fit a constant kernel and persist a reproducible subtraction run.

    The workflow loads the reference and target images, applies the requested
    crop and masks, solves the kernel, and writes the matched image and
    ``target - matched`` residual. Image, variance, and mask paths may name
    NumPy arrays or FITS products; HDU selectors apply only to FITS inputs.

    Returns
    -------
    WorkflowResult
        Run directory and JSON-compatible summary of saved artifacts.
    """

    run_dir = _resolve_run_dir(output_root, name, run_prefix)
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir()

    try:
        if variance_hdu is not None and variance_path is None:
            raise ValueError("variance_hdu requires a variance image path")
        if fit_mask_path is not None and auto_stamp_mask:
            raise ValueError(
                "Specify either fit_mask_path or auto_stamp_mask, not both"
            )
        crop_is_none = [
            crop_y0 is None,
            crop_x0 is None,
            crop_height is None,
            crop_width is None,
        ]
        if any(crop_is_none) and not all(crop_is_none):
            raise ValueError(
                "Specify either all crop parameters or none of them"
            )
        if reference_mask_hdu is not None and reference_mask_path is None:
            if reference_path.suffix.lower() == ".npy":
                raise ValueError(
                    "reference_mask_hdu requires a FITS-backed reference "
                    "mask source"
                )
        if target_mask_hdu is not None and target_mask_path is None:
            if target_path.suffix.lower() == ".npy":
                raise ValueError(
                    "target_mask_hdu requires a FITS-backed target mask "
                    "source"
                )
        normalized_mask_policy = _normalize_mask_policy(mask_policy)
        if normalized_mask_policy == MASK_POLICY_NONE and (
            reference_mask_path is not None
            or target_mask_path is not None
            or reference_mask_hdu is not None
            or target_mask_hdu is not None
        ):
            raise ValueError(
                "reference/target mask inputs require a non-'none' "
                "mask_policy"
            )
        reference, _, used_reference_hdu = load_image_with_wcs(
            reference_path,
            hdu=reference_hdu,
        )
        target, _, used_target_hdu = load_image_with_wcs(
            target_path,
            hdu=target_hdu,
        )
        variance = None
        used_variance_hdu = None
        if variance_path is not None:
            variance, _, used_variance_hdu = load_variance_with_wcs(
                variance_path,
                hdu=variance_hdu,
            )
        crop_metadata: dict[str, int] | None = None
        if crop_y0 is not None:
            crop_metadata = {
                "y0": int(crop_y0),
                "x0": int(crop_x0),
                "height": int(crop_height),
                "width": int(crop_width),
            }
            reference = apply_rectangular_cutout(reference, **crop_metadata)
            target = apply_rectangular_cutout(target, **crop_metadata)
            if variance is not None:
                variance = apply_rectangular_cutout(variance, **crop_metadata)
        raw_reference = np.asarray(reference, dtype=np.float64).copy()
        raw_target = np.asarray(target, dtype=np.float64).copy()
        reference_bad_mask = ~np.isfinite(raw_reference)
        target_bad_mask = ~np.isfinite(raw_target)
        preprocessing_metadata: dict[str, Any] | None = None
        if normalized_mask_policy != MASK_POLICY_NONE:
            reference_mask_source = reference_mask_path or reference_path
            target_mask_source = target_mask_path or target_path
            reference_mask, used_reference_mask_hdu, reference_plane_map = (
                load_mask_with_planes(
                    reference_mask_source,
                    hdu=reference_mask_hdu,
                )
            )
            target_mask, used_target_mask_hdu, target_plane_map = (
                load_mask_with_planes(
                    target_mask_source,
                    hdu=target_mask_hdu,
                )
            )
            if crop_metadata is not None:
                reference_mask = apply_rectangular_cutout(
                    reference_mask,
                    **crop_metadata,
                )
                target_mask = apply_rectangular_cutout(
                    target_mask,
                    **crop_metadata,
                )
            reference, reference_bad_mask, reference_mask_metadata = (
                _apply_mask_policy(
                    reference,
                    mask=reference_mask,
                    mask_policy=normalized_mask_policy,
                    plane_map=reference_plane_map,
                )
            )
            target, target_bad_mask, target_mask_metadata = (
                _apply_mask_policy(
                    target,
                    mask=target_mask,
                    mask_policy=normalized_mask_policy,
                    plane_map=target_plane_map,
                )
            )
            if variance is not None:
                variance = np.where(target_bad_mask, np.nan, variance)
            preprocessing_metadata = {
                "mask_policy": normalized_mask_policy,
                "reference_mask_source": str(
                    reference_mask_source.expanduser().resolve()
                ),
                "target_mask_source": str(
                    target_mask_source.expanduser().resolve()
                ),
                "reference_mask_hdu": used_reference_mask_hdu,
                "target_mask_hdu": used_target_mask_hdu,
                "reference_mask_fraction": float(np.mean(reference_bad_mask)),
                "target_mask_fraction": float(np.mean(target_bad_mask)),
                "reference_mask": reference_mask_metadata,
                "target_mask": target_mask_metadata,
            }
        else:
            reference_mask = None
            target_mask = None
            reference_plane_map = None
            target_plane_map = None
            preprocessing_metadata = {
                "mask_policy": normalized_mask_policy,
                "reference_mask_fraction": float(np.mean(reference_bad_mask)),
                "target_mask_fraction": float(np.mean(target_bad_mask)),
            }
        fit_mask = None
        fit_mask_kind: str | None = None
        fit_mask_metadata: dict[str, Any] | None = None
        if fit_mask_path is not None:
            fit_mask = _load_fit_mask(
                fit_mask_path.expanduser().resolve(),
                expected_shape=target.shape,
            )
            fit_mask_kind = "explicit_mask"
        elif auto_stamp_mask:
            selection_image = np.where(
                np.isfinite(reference) & np.isfinite(target),
                0.5 * (reference + target),
                np.nan,
            )
            auto_mask = build_compact_source_stamp_mask(
                selection_image,
                variance=variance,
                stamp_size=auto_stamp_size,
                max_stamps=auto_stamp_count,
                peak_percentile=auto_peak_percentile,
            )
            fit_mask = auto_mask.mask
            fit_mask_kind = "auto_stamp_mask"
            fit_mask_metadata = auto_mask.to_metadata()

        result = solve_constant_kernel(
            reference,
            target,
            components,
            kernel_shape=kernel_shape,
            variance=variance,
            fit_mask=fit_mask,
            background_degree=background_degree,
            flux_conserve=flux_conserve,
            backend=backend,
        )

        saved = _save_artifacts(artifacts_dir, result)
        if fit_mask_metadata is not None:
            fit_mask_metadata_path = artifacts_dir / "fit_mask_metadata.json"
            _write_json(fit_mask_metadata_path, fit_mask_metadata)
            saved["fit_mask_metadata"] = str(
                fit_mask_metadata_path.relative_to(artifacts_dir.parent)
            )
        if preprocessing_metadata is not None:
            preprocessing_metadata_path = (
                artifacts_dir / "input_mask_metadata.json"
            )
            _write_json(preprocessing_metadata_path, preprocessing_metadata)
            saved["input_mask_metadata"] = str(
                preprocessing_metadata_path.relative_to(artifacts_dir.parent)
            )
        fit_region_empty_error = (
            "no finite residual pixels remain inside the fit region"
        )
        fit_region_residual = _finite_values(
            result.residual,
            mask=result.fit_mask,
            empty_error=fit_region_empty_error,
        )
        valid_residual = _finite_values(
            result.residual,
            empty_error="no finite residual pixels remain in the saved run",
        )
        review_metrics = {
            "residual_mean": float(np.mean(fit_region_residual)),
            "residual_std": float(np.std(fit_region_residual)),
            "residual_rms": float(np.sqrt(np.mean(fit_region_residual**2))),
            "residual_median": float(np.median(fit_region_residual)),
            "robust_sigma": float(
                1.4826
                * np.median(
                    np.abs(
                        fit_region_residual - np.median(fit_region_residual)
                    )
                )
            ),
        }
        review_saved, hotspots = write_review_metadata(
            artifacts_dir,
            run_name=run_dir.name,
            residual=result.residual,
            review_metrics=review_metrics,
        )
        saved.update(review_saved)
        interactive_saved = write_interactive_review_artifact(
            artifacts_dir,
            run_name=run_dir.name,
            raw_reference=raw_reference,
            raw_target=raw_target,
            matched=result.matched,
            residual=result.residual,
            fit_mask_metadata=fit_mask_metadata,
            input_mask_metadata=preprocessing_metadata,
            reference_mask_values=reference_mask,
            target_mask_values=target_mask,
            reference_plane_map=reference_plane_map,
            target_plane_map=target_plane_map,
            hotspots=hotspots,
            raw_gray_lo=float(
                np.nanpercentile(
                    np.concatenate(
                        [
                            raw_reference[np.isfinite(raw_reference)],
                            raw_target[np.isfinite(raw_target)],
                        ]
                    ),
                    5.0,
                )
            ),
            raw_gray_hi=float(
                np.nanpercentile(
                    np.concatenate(
                        [
                            raw_reference[np.isfinite(raw_reference)],
                            raw_target[np.isfinite(raw_target)],
                        ]
                    ),
                    99.9,
                )
            ),
            matched_lo=float(
                np.nanpercentile(
                    result.matched[np.isfinite(result.matched)], 1.0
                )
            ),
            matched_hi=float(
                np.nanpercentile(
                    result.matched[np.isfinite(result.matched)], 99.5
                )
            ),
            residual_limit=(
                float(
                    np.nanpercentile(
                        np.abs(
                            result.residual[
                                np.isfinite(result.residual) & result.fit_mask
                            ]
                        ),
                        99.5,
                    )
                )
                if np.any(np.isfinite(result.residual) & result.fit_mask)
                else 1.0
            ),
        )
        saved.update(interactive_saved)

        runtime = runtime_metadata(
            backend=result.backend,
            dtype=str(result.kernel.dtype),
        )
        summary = {
            "workflow": workflow_name,
            "package_version": __version__,
            "reference_path": str(reference_path.expanduser().resolve()),
            "target_path": str(target_path.expanduser().resolve()),
            "reference_hdu": used_reference_hdu,
            "target_hdu": used_target_hdu,
            "variance_hdu": used_variance_hdu,
            "crop": crop_metadata,
            "mask_policy": normalized_mask_policy,
            "kernel_shape": list(kernel_shape),
            "basis": [component.__dict__ for component in components],
            "background_degree": background_degree,
            "flux_conserve": flux_conserve,
            "requested_backend": backend,
            "backend": result.backend,
            "device": runtime["device"],
            "dtype": str(result.kernel.dtype),
            "runtime": runtime,
            "fit_pixel_count": result.fit_pixel_count,
            "chi2": result.chi2,
            "dof": result.dof,
            "kernel_sum": float(result.kernel.sum()),
            "residual_mean": review_metrics["residual_mean"],
            "residual_std": review_metrics["residual_std"],
            "all_pixels_residual_mean": float(np.mean(valid_residual)),
            "all_pixels_residual_std": float(np.std(valid_residual)),
            "fit_region": _fit_region_summary(
                image_shape=target.shape,
                fit_pixel_count=result.fit_pixel_count,
                fit_mask_kind=fit_mask_kind,
                kernel_shape=kernel_shape,
                fit_mask_metadata=fit_mask_metadata,
            ),
            "saved": saved,
        }
        if preprocessing_metadata is not None:
            summary["input_mask"] = preprocessing_metadata
        _write_json(run_dir / "summary.json", summary)
        return WorkflowResult(run_dir=run_dir, summary=summary)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def benchmark_constant_kernel_backends(
    *,
    reference_path: Path,
    target_path: Path,
    output_root: Path,
    name: str | None,
    reference_hdu: int | None,
    target_hdu: int | None,
    kernel_shape: tuple[int, int],
    components: list[GaussianBasisComponent],
    variance_path: Path | None,
    variance_hdu: int | None = None,
    fit_mask_path: Path | None = None,
    crop_y0: int | None = None,
    crop_x0: int | None = None,
    crop_height: int | None = None,
    crop_width: int | None = None,
    background_degree: int = 0,
    flux_conserve: bool = False,
    backends: list[str] | tuple[str, ...] = ("cpu", "cupy"),
    reference_backend: str = "cpu",
    repeats: int = 3,
    warmup: int = 1,
    atol: float = 1e-8,
    rtol: float = 1e-9,
) -> WorkflowResult:
    """Benchmark constant-kernel solve/application backends with parity."""

    backend_names = _normalize_backend_list(backends)
    if reference_backend not in backend_names:
        raise ValueError("reference_backend must be included in backends")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")

    run_dir = _resolve_run_dir(output_root, name, "benchmark-backends")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    try:
        if variance_hdu is not None and variance_path is None:
            raise ValueError("variance_hdu requires a variance image path")
        crop_metadata = _resolve_crop_metadata(
            crop_y0=crop_y0,
            crop_x0=crop_x0,
            crop_height=crop_height,
            crop_width=crop_width,
        )

        load_start = time.perf_counter()
        reference, _, used_reference_hdu = load_image_with_wcs(
            reference_path,
            hdu=reference_hdu,
        )
        target, _, used_target_hdu = load_image_with_wcs(
            target_path,
            hdu=target_hdu,
        )
        variance = None
        used_variance_hdu = None
        if variance_path is not None:
            variance, _, used_variance_hdu = load_variance_with_wcs(
                variance_path,
                hdu=variance_hdu,
            )
        fit_mask = None
        if fit_mask_path is not None:
            fit_mask = _load_fit_mask(
                fit_mask_path.expanduser().resolve(),
                expected_shape=target.shape,
            )
        if crop_metadata is not None:
            reference = apply_rectangular_cutout(reference, **crop_metadata)
            target = apply_rectangular_cutout(target, **crop_metadata)
            if variance is not None:
                variance = apply_rectangular_cutout(
                    variance,
                    **crop_metadata,
                )
            if fit_mask is not None:
                fit_mask = apply_rectangular_cutout(
                    fit_mask,
                    **crop_metadata,
                )
        load_seconds = float(time.perf_counter() - load_start)

        timings_by_backend: dict[str, list[dict[str, float]]] = {}
        results = {}
        saved: dict[str, str] = {}
        for backend in backend_names:
            for _ in range(warmup):
                _sync_backend(backend)
                solve_constant_kernel(
                    reference,
                    target,
                    components,
                    kernel_shape=kernel_shape,
                    variance=variance,
                    fit_mask=fit_mask,
                    background_degree=background_degree,
                    flux_conserve=flux_conserve,
                    backend=backend,
                )
                _sync_backend(backend)

            rows: list[dict[str, float]] = []
            last_result = None
            for _ in range(repeats):
                _sync_backend(backend)
                start = time.perf_counter()
                result = solve_constant_kernel(
                    reference,
                    target,
                    components,
                    kernel_shape=kernel_shape,
                    variance=variance,
                    fit_mask=fit_mask,
                    background_degree=background_degree,
                    flux_conserve=flux_conserve,
                    backend=backend,
                )
                _sync_backend(backend)
                rows.append(
                    {"solve_seconds": float(time.perf_counter() - start)}
                )
                last_result = result

            assert last_result is not None
            timings_by_backend[backend] = rows
            results[backend] = last_result
            saved.update(
                _save_benchmark_result_artifacts(
                    artifacts_dir,
                    backend=backend,
                    result=last_result,
                )
            )

        reference_result = results[reference_backend]
        comparisons = {
            backend: _compare_constant_kernel_results(
                reference_result,
                result,
                atol=atol,
                rtol=rtol,
            )
            for backend, result in results.items()
        }
        parity_ok = all(item["ok"] for item in comparisons.values())

        timings_path = artifacts_dir / "timings.json"
        comparisons_path = artifacts_dir / "comparisons.json"
        _write_json(timings_path, timings_by_backend)
        _write_json(comparisons_path, comparisons)
        saved["timings_json"] = str(timings_path.relative_to(run_dir))
        saved["comparisons_json"] = str(comparisons_path.relative_to(run_dir))

        summary = {
            "workflow": "benchmark-backends",
            "package_version": __version__,
            "created_at_utc": _timestamp(),
            "reference_path": str(reference_path.expanduser().resolve()),
            "target_path": str(target_path.expanduser().resolve()),
            "reference_hdu": used_reference_hdu,
            "target_hdu": used_target_hdu,
            "variance_hdu": used_variance_hdu,
            "fit_mask_path": (
                str(fit_mask_path.expanduser().resolve())
                if fit_mask_path is not None
                else None
            ),
            "crop": crop_metadata,
            "kernel_shape": list(kernel_shape),
            "basis": [component.__dict__ for component in components],
            "background_degree": background_degree,
            "flux_conserve": flux_conserve,
            "backends": backend_names,
            "reference_backend": reference_backend,
            "runtimes": {
                backend: runtime_metadata(
                    backend=result.backend,
                    dtype=str(result.kernel.dtype),
                )
                for backend, result in results.items()
            },
            "repeats": repeats,
            "warmup": warmup,
            "tolerances": {
                "atol": atol,
                "rtol": rtol,
            },
            "setup_timings": {
                "load_seconds": load_seconds,
            },
            "timings": {
                backend: _summarize_timing_rows(rows, "solve_seconds")
                for backend, rows in timings_by_backend.items()
            },
            "parity": {
                "ok": parity_ok,
                "comparisons": comparisons,
            },
            "fit_pixel_count": {
                backend: int(result.fit_pixel_count)
                for backend, result in results.items()
            },
            "chi2": {
                backend: float(result.chi2)
                for backend, result in results.items()
            },
            "saved": saved,
        }
        _write_json(run_dir / "summary.json", summary)
        return WorkflowResult(run_dir=run_dir, summary=summary)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def evaluate_subtraction_run(run_dir: Path) -> dict[str, Any]:
    resolved = run_dir.expanduser().resolve()
    summary_path = resolved / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json in {resolved}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    artifacts_dir = resolved / "artifacts"
    residual = np.load(artifacts_dir / "residual.npy")
    fit_mask_file = artifacts_dir / "fit_mask.npy"
    if fit_mask_file.exists():
        fit_mask = _load_fit_mask(
            fit_mask_file,
            expected_shape=residual.shape,
        )
    else:
        raise FileNotFoundError(
            f"Missing saved fit mask artifact for evaluation: {fit_mask_file}"
        )
    fit_residual = _finite_values(
        residual,
        mask=fit_mask,
        empty_error="no finite residual pixels remain inside the fit region",
    )
    valid_residual = _finite_values(
        residual,
        empty_error="no finite residual pixels remain in the saved run",
    )
    robust_sigma = 1.4826 * np.median(
        np.abs(fit_residual - np.median(fit_residual))
    )
    residual_median = float(np.median(fit_residual))
    deviation = np.abs(fit_residual - residual_median)
    sigma_floor = 1e-12 * max(
        1.0,
        float(np.max(np.abs(fit_residual))),
    )
    if not np.isfinite(robust_sigma) or robust_sigma <= sigma_floor:
        if np.all(deviation <= sigma_floor):
            above_3sigma = 0
            above_5sigma = 0
        else:
            fallback_sigma = float(np.sqrt(np.mean(deviation**2)))
            if (
                not np.isfinite(fallback_sigma)
                or fallback_sigma <= sigma_floor
            ):
                outlier_count = int(np.count_nonzero(deviation > sigma_floor))
                above_3sigma = outlier_count
                above_5sigma = outlier_count
            else:
                above_3sigma = int(
                    np.count_nonzero(deviation > 3.0 * fallback_sigma)
                )
                above_5sigma = int(
                    np.count_nonzero(deviation > 5.0 * fallback_sigma)
                )
    else:
        above_3sigma = int(np.count_nonzero(deviation > 3.0 * robust_sigma))
        above_5sigma = int(np.count_nonzero(deviation > 5.0 * robust_sigma))

    result = {
        "run_dir": str(resolved),
        "kernel_sum": summary.get("kernel_sum"),
        "residual_mean": float(np.mean(fit_residual)),
        "residual_std": float(np.std(fit_residual)),
        "residual_rms": float(np.sqrt(np.mean(fit_residual**2))),
        "residual_median": float(np.median(fit_residual)),
        "robust_sigma": float(robust_sigma),
        "abs_residual_max": float(np.max(np.abs(fit_residual))),
        "all_pixels_residual_mean": float(np.mean(valid_residual)),
        "all_pixels_residual_std": float(np.std(valid_residual)),
        "all_pixels_residual_rms": float(np.sqrt(np.mean(valid_residual**2))),
        "all_pixels_residual_median": float(np.median(valid_residual)),
        "all_pixels_abs_residual_max": float(np.max(np.abs(valid_residual))),
        "pixels_gt_3sigma": above_3sigma,
        "pixels_gt_5sigma": above_5sigma,
        "fit_region_residual_mean": float(np.mean(fit_residual)),
        "fit_region_residual_std": float(np.std(fit_residual)),
        "fit_region_residual_rms": float(np.sqrt(np.mean(fit_residual**2))),
        "fit_region_residual_median": float(np.median(fit_residual)),
        "fit_region_pixel_count": int(fit_mask.sum()),
    }
    _write_json(resolved / "evaluation.json", result)
    return result


def rebuild_interactive_review(run_dir: Path) -> dict[str, Any]:
    resolved = run_dir.expanduser().resolve()
    summary_path = resolved / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json in {resolved}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    artifacts_dir = resolved / "artifacts"
    required = (
        "kernel.npy",
        "matched.npy",
        "residual.npy",
        "fit_mask.npy",
        "background.npy",
    )
    missing = [
        name for name in required if not (artifacts_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required subtraction artifacts in {artifacts_dir}: "
            f"{', '.join(missing)}"
        )

    reference_path = Path(summary["reference_path"])
    target_path = Path(summary["target_path"])
    raw_reference, _, _ = load_image_with_wcs(
        reference_path,
        hdu=summary.get("reference_hdu"),
    )
    raw_target, _, _ = load_image_with_wcs(
        target_path,
        hdu=summary.get("target_hdu"),
    )
    crop = summary.get("crop")
    if crop is not None:
        raw_reference = apply_rectangular_cutout(raw_reference, **crop)
        raw_target = apply_rectangular_cutout(raw_target, **crop)

    preprocessing_metadata_path = artifacts_dir / "input_mask_metadata.json"
    if preprocessing_metadata_path.exists():
        preprocessing_metadata = json.loads(
            preprocessing_metadata_path.read_text(encoding="utf-8")
        )
    else:
        preprocessing_metadata = {
            "mask_policy": summary.get("mask_policy") or MASK_POLICY_NONE
        }

    reference = np.asarray(raw_reference, dtype=np.float64).copy()
    target = np.asarray(raw_target, dtype=np.float64).copy()
    reference_bad_mask = ~np.isfinite(reference)
    target_bad_mask = ~np.isfinite(target)
    reference_mask = None
    target_mask = None
    reference_plane_map = None
    target_plane_map = None
    if preprocessing_metadata.get("mask_policy") != MASK_POLICY_NONE:
        reference_mask_source = Path(
            preprocessing_metadata["reference_mask_source"]
        )
        target_mask_source = Path(
            preprocessing_metadata["target_mask_source"]
        )
        reference_mask, _, reference_plane_map = load_mask_with_planes(
            reference_mask_source,
            hdu=preprocessing_metadata.get("reference_mask_hdu"),
        )
        target_mask, _, target_plane_map = load_mask_with_planes(
            target_mask_source,
            hdu=preprocessing_metadata.get("target_mask_hdu"),
        )
        if crop is not None:
            reference_mask = apply_rectangular_cutout(reference_mask, **crop)
            target_mask = apply_rectangular_cutout(target_mask, **crop)
        reference, reference_bad_mask, _ = _apply_mask_policy(
            reference,
            mask=reference_mask,
            mask_policy=preprocessing_metadata["mask_policy"],
            plane_map=reference_plane_map,
        )
        target, target_bad_mask, _ = _apply_mask_policy(
            target,
            mask=target_mask,
            mask_policy=preprocessing_metadata["mask_policy"],
            plane_map=target_plane_map,
        )

    matched = np.load(artifacts_dir / "matched.npy", allow_pickle=False)
    residual = np.load(artifacts_dir / "residual.npy", allow_pickle=False)
    fit_mask = _load_fit_mask(
        artifacts_dir / "fit_mask.npy",
        expected_shape=residual.shape,
    )
    fit_mask_metadata_path = artifacts_dir / "fit_mask_metadata.json"
    fit_mask_metadata = (
        json.loads(fit_mask_metadata_path.read_text(encoding="utf-8"))
        if fit_mask_metadata_path.exists()
        else None
    )
    evaluation = evaluate_subtraction_run(resolved)
    hotspots = identify_residual_hotspots(
        residual,
        robust_sigma=float(evaluation["robust_sigma"]),
    )

    finite_raw = np.concatenate(
        [
            raw_reference[np.isfinite(raw_reference)],
            raw_target[np.isfinite(raw_target)],
        ]
    )
    raw_gray_lo = float(np.percentile(finite_raw, 5.0))
    raw_gray_hi = float(np.percentile(finite_raw, 99.9))
    matched_finite = matched[np.isfinite(matched)]
    matched_lo = float(np.percentile(matched_finite, 1.0))
    matched_hi = float(np.percentile(matched_finite, 99.5))
    fit_residual = residual[np.isfinite(residual) & fit_mask]
    if fit_residual.size:
        residual_limit = float(np.percentile(np.abs(fit_residual), 99.5))
    else:
        residual_limit = 1.0

    saved = write_interactive_review_artifact(
        artifacts_dir,
        run_name=resolved.name,
        raw_reference=raw_reference,
        raw_target=raw_target,
        matched=matched,
        residual=residual,
        fit_mask_metadata=fit_mask_metadata,
        input_mask_metadata=preprocessing_metadata,
        reference_mask_values=reference_mask,
        target_mask_values=target_mask,
        reference_plane_map=reference_plane_map,
        target_plane_map=target_plane_map,
        hotspots=hotspots,
        raw_gray_lo=raw_gray_lo,
        raw_gray_hi=raw_gray_hi,
        matched_lo=matched_lo,
        matched_hi=matched_hi,
        residual_limit=residual_limit,
    )
    if not saved:
        raise RuntimeError("Interactive review generation requires bokeh")
    summary["saved"].update(saved)
    _write_json(summary_path, summary)
    return {
        "run_dir": str(resolved),
        "saved": saved,
        "fit_region_pixel_count": int(fit_mask.sum()),
        "robust_sigma": float(evaluation["robust_sigma"]),
    }


def _resolve_run_dir(
    output_root: Path,
    name: str | None,
    prefix: str,
) -> Path:
    root = output_root.expanduser().resolve()
    if name:
        label = _validate_run_name(name)
    else:
        label = datetime.now().strftime(f"{prefix}-%Y%m%d-%H%M%S-%f")
    candidate = (root / label).resolve()
    if candidate.parent != root:
        raise ValueError("run name must stay within the output root")
    return candidate


def _validate_run_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise ValueError("run name cannot be empty")
    path = Path(stripped)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ValueError(
            "run name must be a single path component without separators"
        )
    return path.name


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_crop_metadata(
    *,
    crop_y0: int | None,
    crop_x0: int | None,
    crop_height: int | None,
    crop_width: int | None,
) -> dict[str, int] | None:
    crop_is_none = [
        crop_y0 is None,
        crop_x0 is None,
        crop_height is None,
        crop_width is None,
    ]
    if any(crop_is_none) and not all(crop_is_none):
        raise ValueError("Specify either all crop parameters or none of them")
    if crop_y0 is None:
        return None
    return {
        "y0": int(crop_y0),
        "x0": int(crop_x0),
        "height": int(crop_height),
        "width": int(crop_width),
    }


def _normalize_backend_list(
    backends: list[str] | tuple[str, ...],
) -> list[str]:
    names = [
        str(item).strip().lower() for item in backends if str(item).strip()
    ]
    if not names:
        raise ValueError("backends must contain at least one backend")
    unknown = sorted(set(names) - set(EXPLICIT_BACKENDS))
    if unknown:
        raise ValueError("unsupported backend(s): " + ", ".join(unknown))
    seen: set[str] = set()
    unique = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _sync_backend(backend: str) -> None:
    if backend == "numba-cuda":
        from numba import cuda

        cuda.synchronize()
        return
    if backend not in {"cupy", "cutile"}:
        return

    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def _artifact_label(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _save_benchmark_result_artifacts(
    artifacts_dir: Path,
    *,
    backend: str,
    result,
) -> dict[str, str]:
    saved: dict[str, str] = {}
    label = _artifact_label(backend)
    for name, array in (
        ("kernel", result.kernel),
        ("matched", result.matched),
        ("residual", result.residual),
        ("fit_mask", result.fit_mask),
        ("background", result.background),
    ):
        path = artifacts_dir / f"{label}_{name}.npy"
        _save_array(path, array)
        saved[f"{label}_{name}"] = str(path.relative_to(artifacts_dir.parent))
    return saved


def _summarize_timing_rows(
    rows: list[dict[str, float]],
    key: str,
) -> dict[str, float | int]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize empty timing rows")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "best": float(np.min(values)),
    }


def _compare_constant_kernel_results(
    reference,
    candidate,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    array_comparisons = {
        name: _compare_arrays(
            getattr(reference, name),
            getattr(candidate, name),
            atol=atol,
            rtol=rtol,
        )
        for name in ("kernel", "background", "matched", "residual")
    }
    fit_mask_mismatch_count = int(
        np.count_nonzero(reference.fit_mask != candidate.fit_mask)
    )
    chi2_abs_diff = float(abs(reference.chi2 - candidate.chi2))
    chi2_ok = bool(
        np.isclose(reference.chi2, candidate.chi2, atol=atol, rtol=rtol)
    )
    dof_equal = bool(reference.dof == candidate.dof)
    fit_pixel_count_equal = bool(
        reference.fit_pixel_count == candidate.fit_pixel_count
    )
    ok = (
        all(item["ok"] for item in array_comparisons.values())
        and fit_mask_mismatch_count == 0
        and chi2_ok
        and dof_equal
        and fit_pixel_count_equal
    )
    return {
        "ok": ok,
        "arrays": array_comparisons,
        "fit_mask_mismatch_count": fit_mask_mismatch_count,
        "chi2_abs_diff": chi2_abs_diff,
        "chi2_ok": chi2_ok,
        "dof_equal": dof_equal,
        "fit_pixel_count_equal": fit_pixel_count_equal,
    }


def _compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    reference_arr = np.asarray(reference)
    candidate_arr = np.asarray(candidate)
    shape_equal = reference_arr.shape == candidate_arr.shape
    if not shape_equal:
        return {
            "ok": False,
            "shape_equal": False,
            "reference_shape": list(reference_arr.shape),
            "candidate_shape": list(candidate_arr.shape),
            "finite_count": 0,
            "nan_mismatch_count": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    nan_mismatch_count = int(
        np.count_nonzero(np.isnan(reference_arr) != np.isnan(candidate_arr))
    )
    finite = np.isfinite(reference_arr) & np.isfinite(candidate_arr)
    finite_count = int(np.count_nonzero(finite))
    if finite_count:
        diff = np.abs(reference_arr[finite] - candidate_arr[finite])
        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))
    else:
        max_abs_diff = None
        mean_abs_diff = None
    ok = bool(
        nan_mismatch_count == 0
        and np.allclose(
            reference_arr,
            candidate_arr,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
    )
    return {
        "ok": ok,
        "shape_equal": True,
        "reference_shape": list(reference_arr.shape),
        "candidate_shape": list(candidate_arr.shape),
        "finite_count": finite_count,
        "nan_mismatch_count": nan_mismatch_count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def _save_array(path: Path, array: np.ndarray) -> Path:
    np.save(path, np.asarray(array), allow_pickle=False)
    return path


def _finite_values(
    array: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    empty_error: str,
) -> np.ndarray:
    finite_mask = np.isfinite(array)
    if mask is not None:
        finite_mask &= mask
    finite = array[finite_mask]
    if finite.size == 0:
        raise ValueError(empty_error)
    return finite


def _load_fit_mask(
    path: Path,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    mask = np.load(path, allow_pickle=False)
    mask_arr = np.asarray(mask)
    if expected_shape is not None and mask_arr.shape != expected_shape:
        raise ValueError(
            "fit_mask array shape does not match the image shape"
        )
    if mask_arr.dtype == bool:
        return mask_arr
    if not np.issubdtype(mask_arr.dtype, np.number):
        raise ValueError("fit_mask array must be boolean or binary numeric")
    if not np.isfinite(mask_arr).all():
        raise ValueError("fit_mask array must not contain NaN or inf values")
    if not np.all((mask_arr == 0) | (mask_arr == 1)):
        raise ValueError("fit_mask array must contain only 0/1 values")
    return mask_arr.astype(bool)


def _save_artifacts(artifacts_dir: Path, result) -> dict[str, str]:
    saved: dict[str, str] = {}
    for name, array in (
        ("kernel", result.kernel),
        ("matched", result.matched),
        ("residual", result.residual),
        ("fit_mask", result.fit_mask),
        ("background", result.background),
    ):
        path = artifacts_dir / f"{name}.npy"
        _save_array(path, array)
        saved[name] = str(path.relative_to(artifacts_dir.parent))
    return saved


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _fit_region_summary(
    image_shape: tuple[int, int],
    fit_pixel_count: int,
    fit_mask_kind: str | None,
    kernel_shape: tuple[int, int],
    fit_mask_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fit_mask_kind is None:
        margin_y = kernel_shape[0] // 2
        margin_x = kernel_shape[1] // 2
        return {
            "kind": "interior_box",
            "margin_y": margin_y,
            "margin_x": margin_x,
            "pixel_count": fit_pixel_count,
        }
    if fit_mask_kind == "auto_stamp_mask":
        summary = {
            "kind": "auto_stamp_mask",
            "pixel_count": fit_pixel_count,
        }
        if fit_mask_metadata is not None:
            summary.update(fit_mask_metadata)
        return summary
    return {
        "kind": "explicit_mask",
        "pixel_count": fit_pixel_count,
    }


def _normalize_mask_policy(mask_policy: str) -> str:
    policy = str(mask_policy).strip().lower()
    if policy not in _MASK_POLICY_ALIASES:
        raise ValueError(
            "mask_policy must be one of: "
            f"{', '.join(sorted(_MASK_POLICY_ALIASES))}"
        )
    return _MASK_POLICY_ALIASES[policy]


def _apply_mask_policy(
    image: np.ndarray,
    *,
    mask: np.ndarray,
    mask_policy: str,
    plane_map: dict[str, int] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_arr = np.asarray(image, dtype=np.float64)
    mask_arr = np.asarray(mask)
    if mask_arr.shape != image_arr.shape:
        raise ValueError("mask array shape does not match the image shape")

    if mask_policy == MASK_POLICY_STRICT:
        bad_mask = mask_arr != 0
        metadata = {
            "kind": MASK_POLICY_STRICT,
            "masked_plane_names": None,
            "masked_bits": None,
        }
    elif mask_policy == MASK_POLICY_HSC_MASKLITE:
        if plane_map is None:
            raise ValueError(
                "hsc-masklite requires FITS mask metadata with MP_* plane "
                "definitions"
            )
        missing = [
            name for name in _HSC_MASKLITE_PLANES if name not in plane_map
        ]
        if missing:
            raise ValueError(
                "hsc-masklite mask is missing plane definitions for: "
                f"{', '.join(missing)}"
            )
        bitmask = 0
        for name in _HSC_MASKLITE_PLANES:
            bitmask |= 1 << plane_map[name]
        bad_mask = (mask_arr.astype(np.int64) & bitmask) != 0
        metadata = {
            "kind": MASK_POLICY_HSC_MASKLITE,
            "masked_plane_names": list(_HSC_MASKLITE_PLANES),
            "masked_bits": [
                int(plane_map[name]) for name in _HSC_MASKLITE_PLANES
            ],
        }
    else:
        raise ValueError(f"unsupported mask_policy: {mask_policy}")

    masked_image = image_arr.copy()
    masked_image[bad_mask] = np.nan
    metadata["masked_fraction"] = float(np.mean(bad_mask))
    return masked_image, bad_mask, metadata
