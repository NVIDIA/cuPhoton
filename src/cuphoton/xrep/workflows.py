# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow helpers for xRep."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.wcs import WCS

from cuphoton import __version__
from cuphoton.core.cli import ApplicationContext
from cuphoton.core.runtime import runtime_metadata

from .backends import SUPPORTED_BACKENDS, get_backend, resolve_backend
from .geometry import Grid, ReprojectionSpec
from .io import (
    load_fits_image_with_wcs,
    load_fits_mask,
    write_reprojected_fits,
    write_stack_fits,
)
from .mapping import (
    estimate_source_bbox_on_grid,
    make_wcs_mapping,
    prepare_reprojection,
)
from .reproject import (
    _prepare_array_module,
    build_stack_spec_from_fits,
    reproject_stack,
)


@dataclass(slots=True)
class WorkflowResult:
    """Persisted xRep workflow result.

    Attributes
    ----------
    run_dir
        Directory owning the reprojected arrays and optional FITS products.
    summary
        JSON-compatible inputs, resolved backend/device details, timings, grid
        geometry, and artifact paths.
    """

    run_dir: Path
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _BackendVariant:
    label: str
    backend: str
    cupy_lanczos_kernel: str | None = None


def inspect_image(
    path: Path,
    *,
    hdu: int | None = None,
    grid_crval_ra: float | None = None,
    grid_crval_dec: float | None = None,
    pixel_scale_arcsec: float | None = None,
) -> dict[str, Any]:
    """Inspect a FITS image and summarize its default reprojection grid."""

    image, wcs, header, used_hdu = load_fits_image_with_wcs(path, hdu=hdu)
    native_scale = _pixel_scale_arcsec(wcs)
    center_crval = wcs.all_pix2world(
        [[image.shape[1] / 2.0, image.shape[0] / 2.0]],
        0,
    )[0]
    grid = _resolve_grid(
        grid_crval_ra=grid_crval_ra,
        grid_crval_dec=grid_crval_dec,
        pixel_scale_arcsec=pixel_scale_arcsec,
        fallback_wcs=wcs,
        image_shape=image.shape,
    )
    bbox = estimate_source_bbox_on_grid(wcs, shape=image.shape, grid=grid)
    return {
        "path": str(path.expanduser().resolve()),
        "hdu": used_hdu,
        "shape": list(image.shape),
        "native_pixel_scale_arcsec": native_scale,
        "native_center_crval": [
            float(center_crval[0]),
            float(center_crval[1]),
        ],
        "grid": {
            "crval": list(grid.crval),
            "pixel_scale_arcsec": grid.pixel_scale_arcsec,
        },
        "bbox_on_grid": {
            "min_x": bbox.min_x,
            "min_y": bbox.min_y,
            "width": bbox.width,
            "height": bbox.height,
        },
        "wcs": {
            "ctype": [str(value) for value in wcs.wcs.ctype],
            "crpix": [float(value) for value in wcs.wcs.crpix],
            "crval": [float(value) for value in wcs.wcs.crval],
        },
        "header_extname": str(header.get("EXTNAME", "")),
    }


def run_reproject_image(
    *,
    input_path: Path,
    output_root: Path | None,
    name: str | None,
    hdu: int | None,
    mask_path: Path | None,
    mask_hdu: int | None,
    backend: str | None,
    interpolation: str,
    grid_crval_ra: float | None,
    grid_crval_dec: float | None,
    pixel_scale_arcsec: float | None,
    mapping_grid_step: int,
    area_scaling: bool,
    write_fits: bool,
) -> WorkflowResult:
    """Reproject one FITS image and persist run artifacts."""

    requested_backend = backend or "auto"
    backend = resolve_backend(backend)
    run_dir = _resolve_run_dir(output_root, name, "reproject-image")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    grid = None
    result, grid, timings = _execute_single_reprojection(
        input_path=input_path,
        hdu=hdu,
        grid_crval_ra=grid_crval_ra,
        grid_crval_dec=grid_crval_dec,
        pixel_scale_arcsec=pixel_scale_arcsec,
        mask_path=mask_path,
        mask_hdu=mask_hdu,
        interpolation=interpolation,
        backend=backend,
        mapping_grid_step=mapping_grid_step,
        area_scaling=area_scaling,
    )

    saved = {
        "image": _save_array(artifacts_dir / "reprojected.npy", result.image),
    }
    if result.mask is not None:
        saved["mask"] = _save_array(artifacts_dir / "mask.npy", result.mask)
    if write_fits:
        saved["reprojected_fits"] = _save_path(
            write_reprojected_fits(
                artifacts_dir / "reprojected.fits",
                result.image,
                grid=grid,
                bbox=result.bbox,
                mask=result.mask,
                metadata={
                    "backend": backend,
                    "interp": interpolation,
                },
            )
        )

    runtime = _reprojection_runtime(result)
    summary = {
        "workflow": "reproject-image",
        "package_version": __version__,
        "created_at_utc": _timestamp(),
        "input": str(input_path.expanduser().resolve()),
        "requested_backend": requested_backend,
        "backend": backend,
        "device": runtime["device"],
        "dtype": str(result.image.dtype),
        "runtime": runtime,
        "interpolation": interpolation,
        "timings": timings,
        "grid": {
            "crval": list(grid.crval),
            "pixel_scale_arcsec": grid.pixel_scale_arcsec,
        },
        "output_bbox": {
            "min_x": result.bbox.min_x,
            "min_y": result.bbox.min_y,
            "width": result.bbox.width,
            "height": result.bbox.height,
        },
        "saved": _rewrite_saved_paths(saved, run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    return WorkflowResult(run_dir=run_dir, summary=summary)


def benchmark_reproject_image(
    *,
    input_path: Path,
    output_root: Path | None,
    name: str | None,
    hdu: int | None,
    mask_path: Path | None,
    mask_hdu: int | None,
    backend: str | None,
    interpolation: str,
    grid_crval_ra: float | None,
    grid_crval_dec: float | None,
    pixel_scale_arcsec: float | None,
    mapping_grid_step: int,
    area_scaling: bool,
    write_fits: bool,
    repeats: int,
    warmup: int,
) -> WorkflowResult:
    """Benchmark one FITS reprojection and persist timing summaries."""

    requested_backend = backend or "auto"
    backend = resolve_backend(backend)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    run_dir = _resolve_run_dir(output_root, name, "benchmark-reproject-image")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    timings_rows: list[dict[str, float]] = []
    last_result = None
    last_grid = None
    for _ in range(warmup):
        _execute_single_reprojection(
            input_path=input_path,
            hdu=hdu,
            grid_crval_ra=grid_crval_ra,
            grid_crval_dec=grid_crval_dec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            mask_path=mask_path,
            mask_hdu=mask_hdu,
            backend=backend,
            interpolation=interpolation,
            mapping_grid_step=mapping_grid_step,
            area_scaling=area_scaling,
        )
    for _ in range(repeats):
        result, grid, timings = _execute_single_reprojection(
            input_path=input_path,
            hdu=hdu,
            grid_crval_ra=grid_crval_ra,
            grid_crval_dec=grid_crval_dec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            mask_path=mask_path,
            mask_hdu=mask_hdu,
            backend=backend,
            interpolation=interpolation,
            mapping_grid_step=mapping_grid_step,
            area_scaling=area_scaling,
        )
        timings_rows.append(timings)
        last_result = result
        last_grid = grid

    assert last_result is not None
    assert last_grid is not None

    saved = {
        "image": _save_array(
            artifacts_dir / "reprojected.npy",
            last_result.image,
        ),
        "timings_json": _save_json(
            artifacts_dir / "timings.json",
            timings_rows,
        ),
    }
    if last_result.mask is not None:
        saved["mask"] = _save_array(
            artifacts_dir / "mask.npy",
            last_result.mask,
        )
    if write_fits:
        saved["reprojected_fits"] = _save_path(
            write_reprojected_fits(
                artifacts_dir / "reprojected.fits",
                last_result.image,
                grid=last_grid,
                bbox=last_result.bbox,
                mask=last_result.mask,
                metadata={
                    "backend": backend,
                    "interp": interpolation,
                    "benchmark_repeats": repeats,
                },
            )
        )

    runtime = _reprojection_runtime(last_result)
    summary = {
        "workflow": "benchmark-reproject-image",
        "package_version": __version__,
        "created_at_utc": _timestamp(),
        "input": str(input_path.expanduser().resolve()),
        "requested_backend": requested_backend,
        "backend": backend,
        "device": runtime["device"],
        "dtype": str(last_result.image.dtype),
        "runtime": runtime,
        "interpolation": interpolation,
        "repeats": repeats,
        "warmup": warmup,
        "timings": _summarize_timings(timings_rows),
        "grid": {
            "crval": list(last_grid.crval),
            "pixel_scale_arcsec": last_grid.pixel_scale_arcsec,
        },
        "output_bbox": {
            "min_x": last_result.bbox.min_x,
            "min_y": last_result.bbox.min_y,
            "width": last_result.bbox.width,
            "height": last_result.bbox.height,
        },
        "saved": _rewrite_saved_paths(saved, run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    return WorkflowResult(run_dir=run_dir, summary=summary)


def compare_backends_reproject_image(
    *,
    input_path: Path,
    output_root: Path | None,
    name: str | None,
    hdu: int | None,
    mask_path: Path | None,
    mask_hdu: int | None,
    backends: list[str] | tuple[str, ...],
    reference_backend: str,
    interpolation: str,
    grid_crval_ra: float | None,
    grid_crval_dec: float | None,
    pixel_scale_arcsec: float | None,
    mapping_grid_step: int,
    area_scaling: bool,
    write_fits: bool,
    repeats: int,
    warmup: int,
    atol: float,
    rtol: float,
) -> WorkflowResult:
    """Run one FITS reprojection and compare backend outputs."""

    backend_names = _normalize_backend_list(backends)
    if reference_backend not in backend_names:
        raise ValueError("reference_backend must be included in backends")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")

    run_dir = _resolve_run_dir(output_root, name, "compare-backends")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    timings_by_backend: dict[str, list[dict[str, float]]] = {}
    results: dict[str, Any] = {}
    grids: dict[str, Grid] = {}
    saved: dict[str, str] = {}

    for backend in backend_names:
        timings_rows: list[dict[str, float]] = []
        for _ in range(warmup):
            _execute_single_reprojection(
                input_path=input_path,
                hdu=hdu,
                grid_crval_ra=grid_crval_ra,
                grid_crval_dec=grid_crval_dec,
                pixel_scale_arcsec=pixel_scale_arcsec,
                mask_path=mask_path,
                mask_hdu=mask_hdu,
                backend=backend,
                interpolation=interpolation,
                mapping_grid_step=mapping_grid_step,
                area_scaling=area_scaling,
            )
        last_result = None
        last_grid = None
        for _ in range(repeats):
            result, grid, timings = _execute_single_reprojection(
                input_path=input_path,
                hdu=hdu,
                grid_crval_ra=grid_crval_ra,
                grid_crval_dec=grid_crval_dec,
                pixel_scale_arcsec=pixel_scale_arcsec,
                mask_path=mask_path,
                mask_hdu=mask_hdu,
                backend=backend,
                interpolation=interpolation,
                mapping_grid_step=mapping_grid_step,
                area_scaling=area_scaling,
            )
            timings_rows.append(timings)
            last_result = result
            last_grid = grid

        assert last_result is not None
        assert last_grid is not None
        timings_by_backend[backend] = timings_rows
        results[backend] = last_result
        grids[backend] = last_grid

        label = _artifact_label(backend)
        saved[f"{label}_image"] = _save_array(
            artifacts_dir / f"{label}_reprojected.npy",
            last_result.image,
        )
        if last_result.mask is not None:
            saved[f"{label}_mask"] = _save_array(
                artifacts_dir / f"{label}_mask.npy",
                last_result.mask,
            )
        if write_fits:
            saved[f"{label}_reprojected_fits"] = _save_path(
                write_reprojected_fits(
                    artifacts_dir / f"{label}_reprojected.fits",
                    last_result.image,
                    grid=last_grid,
                    bbox=last_result.bbox,
                    mask=last_result.mask,
                    metadata={
                        "backend": backend,
                        "interp": interpolation,
                    },
                )
            )

    reference = results[reference_backend]
    comparisons = {
        backend: _compare_reprojection_results(
            reference,
            result,
            atol=atol,
            rtol=rtol,
        )
        for backend, result in results.items()
    }
    parity_ok = all(item["ok"] for item in comparisons.values())
    timing_summaries = {
        backend: _summarize_timings(rows)
        for backend, rows in timings_by_backend.items()
    }

    saved["timings_json"] = _save_json(
        artifacts_dir / "timings.json",
        timings_by_backend,
    )
    saved["comparisons_json"] = _save_json(
        artifacts_dir / "comparisons.json",
        comparisons,
    )

    grid = grids[reference_backend]
    bbox = reference.bbox
    summary = {
        "workflow": "compare-backends",
        "package_version": __version__,
        "created_at_utc": _timestamp(),
        "input": str(input_path.expanduser().resolve()),
        "backends": backend_names,
        "reference_backend": reference_backend,
        "runtimes": {
            backend: _reprojection_runtime(result)
            for backend, result in results.items()
        },
        "interpolation": interpolation,
        "repeats": repeats,
        "warmup": warmup,
        "tolerances": {
            "atol": atol,
            "rtol": rtol,
        },
        "timings": timing_summaries,
        "parity": {
            "ok": parity_ok,
            "comparisons": comparisons,
        },
        "grid": {
            "crval": list(grid.crval),
            "pixel_scale_arcsec": grid.pixel_scale_arcsec,
        },
        "output_bbox": {
            "min_x": bbox.min_x,
            "min_y": bbox.min_y,
            "width": bbox.width,
            "height": bbox.height,
        },
        "saved": _rewrite_saved_paths(saved, run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    return WorkflowResult(run_dir=run_dir, summary=summary)


def benchmark_backend_variants_reproject_image(
    *,
    input_path: Path,
    output_root: Path | None,
    name: str | None,
    hdu: int | None,
    mask_path: Path | None,
    mask_hdu: int | None,
    variants: list[str] | tuple[str, ...],
    reference_variant: str,
    mask_cases: list[str] | tuple[str, ...],
    interpolation: str,
    grid_crval_ra: float | None,
    grid_crval_dec: float | None,
    pixel_scale_arcsec: float | None,
    mapping_grid_step: int,
    area_scaling: bool,
    write_fits: bool,
    repeats: int,
    warmup: int,
    atol: float,
    rtol: float,
) -> WorkflowResult:
    """Benchmark backend variants with cached geometry and parity reports."""

    variant_specs = _normalize_backend_variant_list(variants)
    variant_labels = [item.label for item in variant_specs]
    if reference_variant not in variant_labels:
        raise ValueError("reference_variant must be included in variants")
    mask_case_names = _normalize_mask_cases(mask_cases)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")

    run_dir = _resolve_run_dir(output_root, name, "benchmark-backends")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    load_start = time.perf_counter()
    image, wcs, _, _ = load_fits_image_with_wcs(input_path, hdu=hdu)
    grid = _resolve_grid(
        grid_crval_ra=grid_crval_ra,
        grid_crval_dec=grid_crval_dec,
        pixel_scale_arcsec=pixel_scale_arcsec,
        fallback_wcs=wcs,
        image_shape=image.shape,
    )
    input_mask = None
    if mask_path is not None:
        input_mask, _, _ = load_fits_mask(mask_path, hdu=mask_hdu)
    load_seconds = float(time.perf_counter() - load_start)

    mapping_start = time.perf_counter()
    mapping, bbox = make_wcs_mapping(input_path, grid, hdu=hdu)
    mapping_seconds = float(time.perf_counter() - mapping_start)

    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=bbox,
        interpolation=interpolation,
        mapping_grid_step=mapping_grid_step,
        area_scaling=area_scaling,
    )
    mask_payloads = _resolve_mask_cases(
        mask_case_names,
        image_shape=image.shape,
        input_mask=input_mask,
    )

    host_timings: dict[str, dict[str, list[dict[str, float]]]] = {}
    device_timings: dict[str, dict[str, list[dict[str, float]]]] = {}
    prepare_seconds: dict[str, float] = {}
    results: dict[str, dict[str, Any]] = {
        case: {} for case in mask_case_names
    }
    saved: dict[str, str] = {}

    for variant in variant_specs:
        with _backend_variant_env(variant):
            backend_impl = get_backend(variant.backend)
            if not backend_impl.is_available():
                raise RuntimeError(
                    f"Requested backend is unavailable: {variant.label}"
                )

            prepare_start = time.perf_counter()
            prepared = prepare_reprojection(
                spec,
                source_shape=image.shape,
                xp=_prepare_array_module(variant.backend),
            )
            _maybe_sync_device(variant.backend)
            prepare_seconds[variant.label] = float(
                time.perf_counter() - prepare_start
            )

            for mask_case, source_mask in mask_payloads.items():
                host_rows: list[dict[str, float]] = []
                for _ in range(warmup):
                    _execute_prepared_backend(
                        backend_impl,
                        variant.backend,
                        image,
                        prepared,
                        spec,
                        source_mask=source_mask,
                    )
                last_result = None
                for _ in range(repeats):
                    result, seconds = _execute_prepared_backend(
                        backend_impl,
                        variant.backend,
                        image,
                        prepared,
                        spec,
                        source_mask=source_mask,
                    )
                    host_rows.append({"backend_seconds": seconds})
                    last_result = result
                assert last_result is not None
                host_timings.setdefault(mask_case, {})[variant.label] = (
                    host_rows
                )
                results[mask_case][variant.label] = last_result

                label = _artifact_label(f"{mask_case}_{variant.label}")
                saved[f"{label}_image"] = _save_array(
                    artifacts_dir / f"{label}_reprojected.npy",
                    last_result.image,
                )
                if last_result.mask is not None:
                    saved[f"{label}_mask"] = _save_array(
                        artifacts_dir / f"{label}_mask.npy",
                        last_result.mask,
                    )
                if write_fits:
                    saved[f"{label}_reprojected_fits"] = _save_path(
                        write_reprojected_fits(
                            artifacts_dir / f"{label}_reprojected.fits",
                            last_result.image,
                            grid=grid,
                            bbox=last_result.bbox,
                            mask=last_result.mask,
                            metadata={
                                "backend": variant.backend,
                                "variant": variant.label,
                                "mask": mask_case,
                            },
                        )
                    )

                if variant.backend == "cupy" and hasattr(
                    backend_impl,
                    "reproject_device",
                ):
                    rows = _benchmark_cupy_device_variant(
                        backend_impl,
                        image,
                        prepared,
                        spec,
                        source_mask=source_mask,
                        repeats=repeats,
                        warmup=warmup,
                    )
                    device_timings.setdefault(mask_case, {})[
                        variant.label
                    ] = rows

    comparisons = {}
    for mask_case, case_results in results.items():
        reference = case_results[reference_variant]
        comparisons[mask_case] = {
            variant.label: _compare_reprojection_results(
                reference,
                case_results[variant.label],
                atol=atol,
                rtol=rtol,
            )
            for variant in variant_specs
        }
    parity_ok = all(
        item["ok"] for case in comparisons.values() for item in case.values()
    )

    saved["host_timings_json"] = _save_json(
        artifacts_dir / "host_timings.json",
        host_timings,
    )
    saved["device_timings_json"] = _save_json(
        artifacts_dir / "device_timings.json",
        device_timings,
    )
    saved["comparisons_json"] = _save_json(
        artifacts_dir / "comparisons.json",
        comparisons,
    )

    summary = {
        "workflow": "benchmark-backend-variants",
        "package_version": __version__,
        "created_at_utc": _timestamp(),
        "input": str(input_path.expanduser().resolve()),
        "variants": [
            {
                "label": item.label,
                "backend": item.backend,
                "cupy_lanczos_kernel": item.cupy_lanczos_kernel,
            }
            for item in variant_specs
        ],
        "reference_variant": reference_variant,
        "runtimes": {
            variant.label: _reprojection_runtime(
                results[mask_case_names[0]][variant.label]
            )
            for variant in variant_specs
        },
        "mask_cases": mask_case_names,
        "interpolation": interpolation,
        "repeats": repeats,
        "warmup": warmup,
        "tolerances": {
            "atol": atol,
            "rtol": rtol,
        },
        "setup_timings": {
            "load_seconds": load_seconds,
            "mapping_seconds": mapping_seconds,
            "prepare_seconds": prepare_seconds,
        },
        "host_timings": {
            case: {
                variant: _summarize_timings(rows)
                for variant, rows in variant_rows.items()
            }
            for case, variant_rows in host_timings.items()
        },
        "device_timings": {
            case: {
                variant: _summarize_timings(rows)
                for variant, rows in variant_rows.items()
            }
            for case, variant_rows in device_timings.items()
        },
        "parity": {
            "ok": parity_ok,
            "comparisons": comparisons,
        },
        "grid": {
            "crval": list(grid.crval),
            "pixel_scale_arcsec": grid.pixel_scale_arcsec,
        },
        "output_bbox": {
            "min_x": bbox.min_x,
            "min_y": bbox.min_y,
            "width": bbox.width,
            "height": bbox.height,
        },
        "saved": _rewrite_saved_paths(saved, run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    return WorkflowResult(run_dir=run_dir, summary=summary)


def run_reproject_stack(
    *,
    input_paths: list[Path],
    output_root: Path | None,
    name: str | None,
    hdu: int | None,
    backend: str | None,
    interpolation: str,
    grid_crval_ra: float | None,
    grid_crval_dec: float | None,
    pixel_scale_arcsec: float | None,
    mapping_grid_step: int,
    area_scaling: bool,
    write_fits: bool,
) -> WorkflowResult:
    """Reproject multiple FITS inputs onto one shared grid."""

    requested_backend = backend or "auto"
    backend = resolve_backend(backend)
    if not input_paths:
        raise ValueError("input_paths must not be empty")
    first_image, first_wcs, _, _ = load_fits_image_with_wcs(
        input_paths[0],
        hdu=hdu,
    )
    grid = _resolve_grid(
        grid_crval_ra=grid_crval_ra,
        grid_crval_dec=grid_crval_dec,
        pixel_scale_arcsec=pixel_scale_arcsec,
        fallback_wcs=first_wcs,
        image_shape=first_image.shape,
    )
    images, spec = build_stack_spec_from_fits(
        input_paths,
        grid=grid,
        hdu=hdu,
        interpolation=interpolation,
        mapping_grid_step=mapping_grid_step,
        area_scaling=area_scaling,
    )
    run_dir = _resolve_run_dir(output_root, name, "reproject-stack")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    result = reproject_stack(images, spec, backend=backend)
    saved = {
        "stack": _save_array(
            artifacts_dir / "reprojected_stack.npy",
            result.images,
        ),
    }
    if result.masks is not None:
        saved["stack_mask"] = _save_array(
            artifacts_dir / "mask_stack.npy",
            result.masks,
        )
    if write_fits:
        saved["stack_fits"] = _save_path(
            write_stack_fits(
                artifacts_dir / "reprojected_stack.fits",
                result.images,
                grid=result.spec.grid,
                bbox=result.spec.output_bbox,
                metadata={
                    "backend": backend,
                    "interp": interpolation,
                    "count": len(input_paths),
                },
            )
        )

    runtime = _reprojection_runtime(result.results[0])
    summary = {
        "workflow": "reproject-stack",
        "package_version": __version__,
        "created_at_utc": _timestamp(),
        "inputs": [str(path.expanduser().resolve()) for path in input_paths],
        "requested_backend": requested_backend,
        "backend": backend,
        "device": runtime["device"],
        "dtype": str(result.images.dtype),
        "runtime": runtime,
        "interpolation": interpolation,
        "grid": {
            "crval": list(result.spec.grid.crval),
            "pixel_scale_arcsec": result.spec.grid.pixel_scale_arcsec,
        },
        "output_bbox": {
            "min_x": result.spec.output_bbox.min_x,
            "min_y": result.spec.output_bbox.min_y,
            "width": result.spec.output_bbox.width,
            "height": result.spec.output_bbox.height,
        },
        "stack_shape": list(result.images.shape),
        "saved": _rewrite_saved_paths(saved, run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    return WorkflowResult(run_dir=run_dir, summary=summary)


def _reprojection_runtime(result: Any) -> dict[str, Any]:
    """Return runtime metadata for a resolved host reprojection result."""

    return runtime_metadata(
        backend=str(result.backend),
        dtype=str(result.image.dtype),
        device=result.metadata.get("device"),
    )


def _resolve_grid(
    *,
    grid_crval_ra: float | None,
    grid_crval_dec: float | None,
    pixel_scale_arcsec: float | None,
    fallback_wcs: WCS,
    image_shape: tuple[int, int],
) -> Grid:
    if (
        grid_crval_ra is None
        and grid_crval_dec is None
        and pixel_scale_arcsec is None
    ):
        center = fallback_wcs.all_pix2world(
            [[image_shape[1] / 2.0, image_shape[0] / 2.0]],
            0,
        )[0]
        pixel_scale = _pixel_scale_arcsec(fallback_wcs)
        return Grid(
            crval=(float(center[0]), float(center[1])),
            pixel_scale_arcsec=pixel_scale,
        )
    if (
        grid_crval_ra is None
        or grid_crval_dec is None
        or pixel_scale_arcsec is None
    ):
        raise ValueError(
            "grid_crval_ra, grid_crval_dec, and pixel_scale_arcsec "
            "must be provided together"
        )
    return Grid(
        crval=(float(grid_crval_ra), float(grid_crval_dec)),
        pixel_scale_arcsec=float(pixel_scale_arcsec),
    )


def _execute_single_reprojection(
    *,
    input_path: Path,
    hdu: int | None,
    grid_crval_ra: float | None,
    grid_crval_dec: float | None,
    pixel_scale_arcsec: float | None,
    mask_path: Path | None,
    mask_hdu: int | None,
    backend: str | None,
    interpolation: str,
    mapping_grid_step: int,
    area_scaling: bool,
) -> tuple[Any, Grid, dict[str, float]]:
    total_start = time.perf_counter()
    backend = resolve_backend(backend)

    load_start = time.perf_counter()
    image, wcs, _, _ = load_fits_image_with_wcs(input_path, hdu=hdu)
    grid = _resolve_grid(
        grid_crval_ra=grid_crval_ra,
        grid_crval_dec=grid_crval_dec,
        pixel_scale_arcsec=pixel_scale_arcsec,
        fallback_wcs=wcs,
        image_shape=image.shape,
    )
    mask = None
    if mask_path is not None:
        mask, _, _ = load_fits_mask(mask_path, hdu=mask_hdu)
    load_seconds = float(time.perf_counter() - load_start)

    mapping_start = time.perf_counter()
    mapping, bbox = make_wcs_mapping(
        input_path,
        grid,
        hdu=hdu,
    )
    mapping_seconds = float(time.perf_counter() - mapping_start)

    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=bbox,
        interpolation=interpolation,
        mapping_grid_step=mapping_grid_step,
        area_scaling=area_scaling,
    )
    prepare_start = time.perf_counter()
    prepared = prepare_reprojection(
        spec,
        source_shape=image.shape,
        xp=_prepare_array_module(backend),
    )
    prepare_seconds = float(time.perf_counter() - prepare_start)

    backend_impl = get_backend(backend)
    if not backend_impl.is_available():
        raise RuntimeError(f"Requested backend is unavailable: {backend}")
    _maybe_sync_device(backend)
    backend_start = time.perf_counter()
    result = backend_impl.reproject(
        image,
        prepared,
        spec,
        source_mask=mask,
    )
    _maybe_sync_device(backend)
    backend_seconds = float(time.perf_counter() - backend_start)
    total_seconds = float(time.perf_counter() - total_start)
    result.metadata.update(
        {
            "path": str(Path(input_path).expanduser().resolve()),
            "grid_crval": list(grid.crval),
            "grid_pixel_scale_arcsec": grid.pixel_scale_arcsec,
        }
    )
    return (
        result,
        grid,
        {
            "load_seconds": load_seconds,
            "mapping_seconds": mapping_seconds,
            "prepare_seconds": prepare_seconds,
            "backend_seconds": backend_seconds,
            "total_seconds": total_seconds,
        },
    )


def _maybe_sync_device(backend: str) -> None:
    """Synchronize the GPU so benchmark timings bound real device work."""
    if backend == "torch":
        try:
            import torch
        except Exception:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif backend == "cupy":
        try:
            import cupy as cp
        except Exception:
            return
        cp.cuda.runtime.deviceSynchronize()


def _summarize_timings(
    timings_rows: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not timings_rows:
        raise ValueError("timings_rows must not be empty")
    summary: dict[str, dict[str, float]] = {}
    for key in timings_rows[0]:
        values = np.asarray(
            [float(row[key]) for row in timings_rows],
            dtype=np.float64,
        )
        summary[key] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _execute_prepared_backend(
    backend_impl,
    backend: str,
    image: np.ndarray,
    prepared,
    spec: ReprojectionSpec,
    *,
    source_mask: np.ndarray | None,
):
    _maybe_sync_device(backend)
    backend_start = time.perf_counter()
    result = backend_impl.reproject(
        image,
        prepared,
        spec,
        source_mask=source_mask,
    )
    _maybe_sync_device(backend)
    return result, float(time.perf_counter() - backend_start)


def _benchmark_cupy_device_variant(
    backend_impl,
    image: np.ndarray,
    prepared,
    spec: ReprojectionSpec,
    *,
    source_mask: np.ndarray | None,
    repeats: int,
    warmup: int,
) -> list[dict[str, float]]:
    import cupy as cp

    source_device = cp.asarray(image, dtype=cp.float64)
    mask_device = None
    if source_mask is not None:
        mask_device = cp.asarray(source_mask)
    cp.cuda.runtime.deviceSynchronize()
    for _ in range(warmup):
        backend_impl.reproject_device(
            source_device,
            prepared,
            spec,
            source_mask=mask_device,
        )
    cp.cuda.runtime.deviceSynchronize()

    rows: list[dict[str, float]] = []
    for _ in range(repeats):
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        backend_impl.reproject_device(
            source_device,
            prepared,
            spec,
            source_mask=mask_device,
        )
        cp.cuda.runtime.deviceSynchronize()
        rows.append({"backend_seconds": float(time.perf_counter() - start)})
    return rows


def _resolve_mask_cases(
    mask_cases: list[str],
    *,
    image_shape: tuple[int, int],
    input_mask: np.ndarray | None,
) -> dict[str, np.ndarray | None]:
    payloads: dict[str, np.ndarray | None] = {}
    synthetic_mask = None
    for item in mask_cases:
        if item == "none":
            payloads[item] = None
        elif item == "mask":
            if input_mask is not None:
                payloads[item] = input_mask
            else:
                if synthetic_mask is None:
                    synthetic_mask = _synthetic_mask(image_shape)
                payloads[item] = synthetic_mask
        else:
            raise ValueError(f"unsupported mask case: {item}")
    return payloads


def _synthetic_mask(image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint16)
    mask[::32, ::32] = 2
    mask[17::64, 19::64] |= 4
    return mask


def _normalize_backend_list(
    backends: list[str] | tuple[str, ...],
) -> list[str]:
    values: list[str] = []
    for item in backends:
        values.extend(part.strip() for part in str(item).split(","))
    values = [item for item in values if item]
    if not values:
        raise ValueError("at least one backend is required")
    unknown = sorted(set(values) - SUPPORTED_BACKENDS)
    if unknown:
        raise ValueError("unsupported backend(s): " + ", ".join(unknown))
    seen = set()
    unique = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _normalize_backend_variant_list(
    variants: list[str] | tuple[str, ...],
) -> list[_BackendVariant]:
    values: list[str] = []
    for item in variants:
        values.extend(part.strip() for part in str(item).split(","))
    values = [item for item in values if item]
    if not values:
        raise ValueError("at least one backend variant is required")
    seen = set()
    unique = []
    for item in values:
        variant = _parse_backend_variant(item)
        if variant.label in seen:
            continue
        seen.add(variant.label)
        unique.append(variant)
    return unique


def _parse_backend_variant(value: str) -> _BackendVariant:
    if value == "cupy-elementwise":
        return _BackendVariant(
            label=value,
            backend="cupy",
            cupy_lanczos_kernel="elementwise",
        )
    if value == "cupy-raw":
        return _BackendVariant(
            label=value,
            backend="cupy",
            cupy_lanczos_kernel="raw",
        )
    if value in SUPPORTED_BACKENDS:
        return _BackendVariant(label=value, backend=value)
    raise ValueError(f"unsupported backend variant: {value}")


def _normalize_mask_cases(
    mask_cases: list[str] | tuple[str, ...],
) -> list[str]:
    values: list[str] = []
    for item in mask_cases:
        values.extend(part.strip() for part in str(item).split(","))
    values = [item for item in values if item]
    if not values:
        raise ValueError("at least one mask case is required")
    unknown = sorted(set(values) - {"none", "mask"})
    if unknown:
        raise ValueError("unsupported mask case(s): " + ", ".join(unknown))
    seen = set()
    unique = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


@contextmanager
def _backend_variant_env(variant: _BackendVariant):
    updates = {}
    if variant.cupy_lanczos_kernel is not None:
        updates["CUPHOTON_XREP_CUPY_LANCZOS_KERNEL"] = (
            variant.cupy_lanczos_kernel
        )
    with _temporary_environ(updates):
        yield


@contextmanager
def _temporary_environ(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compare_reprojection_results(
    reference,
    candidate,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    image = _compare_numeric_arrays(
        reference.image,
        candidate.image,
        atol=atol,
        rtol=rtol,
    )
    mask = _compare_masks(reference.mask, candidate.mask)
    bbox_equal = reference.bbox == candidate.bbox
    return {
        "ok": bool(bbox_equal and image["allclose"] and mask["equal"]),
        "backend": candidate.backend,
        "bbox_equal": bool(bbox_equal),
        "image": image,
        "mask": mask,
    }


def _compare_numeric_arrays(
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
            "allclose": False,
            "shape_equal": False,
            "reference_shape": list(reference_arr.shape),
            "candidate_shape": list(candidate_arr.shape),
            "reference_dtype": str(reference_arr.dtype),
            "candidate_dtype": str(candidate_arr.dtype),
        }

    ref64 = reference_arr.astype(np.float64, copy=False)
    cand64 = candidate_arr.astype(np.float64, copy=False)
    finite = np.isfinite(ref64) & np.isfinite(cand64)
    finite_count = int(np.count_nonzero(finite))
    nan_mismatch_count = int(
        np.count_nonzero(np.isnan(ref64) != np.isnan(cand64))
    )
    finite_mismatch_count = int(
        np.count_nonzero(np.isfinite(ref64) != np.isfinite(cand64))
    )
    allclose_equal_nan = bool(
        np.allclose(ref64, cand64, atol=atol, rtol=rtol, equal_nan=True)
    )

    if finite_count:
        diff = np.abs(ref64[finite] - cand64[finite])
        diff_summary: dict[str, Any] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "p99_abs": float(np.percentile(diff, 99.0)),
        }
    else:
        diff_summary = {
            "max_abs": None,
            "mean_abs": None,
            "p99_abs": None,
        }

    return {
        "allclose": allclose_equal_nan,
        "shape_equal": True,
        "reference_shape": list(reference_arr.shape),
        "candidate_shape": list(candidate_arr.shape),
        "reference_dtype": str(reference_arr.dtype),
        "candidate_dtype": str(candidate_arr.dtype),
        "finite_overlap_count": finite_count,
        "finite_mismatch_count": finite_mismatch_count,
        "nan_mismatch_count": nan_mismatch_count,
        **diff_summary,
    }


def _compare_masks(
    reference: np.ndarray | None,
    candidate: np.ndarray | None,
):
    if reference is None and candidate is None:
        return {
            "equal": True,
            "reference_present": False,
            "candidate_present": False,
            "mismatch_count": 0,
        }
    if reference is None or candidate is None:
        return {
            "equal": False,
            "reference_present": reference is not None,
            "candidate_present": candidate is not None,
            "mismatch_count": None,
        }
    reference_arr = np.asarray(reference)
    candidate_arr = np.asarray(candidate)
    if reference_arr.shape != candidate_arr.shape:
        return {
            "equal": False,
            "reference_present": True,
            "candidate_present": True,
            "reference_shape": list(reference_arr.shape),
            "candidate_shape": list(candidate_arr.shape),
            "mismatch_count": None,
        }
    mismatch_count = int(np.count_nonzero(reference_arr != candidate_arr))
    return {
        "equal": mismatch_count == 0,
        "reference_present": True,
        "candidate_present": True,
        "reference_shape": list(reference_arr.shape),
        "candidate_shape": list(candidate_arr.shape),
        "reference_dtype": str(reference_arr.dtype),
        "candidate_dtype": str(candidate_arr.dtype),
        "mismatch_count": mismatch_count,
    }


def _artifact_label(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value
    )


def _resolve_run_dir(
    output_root: Path | None,
    name: str | None,
    prefix: str,
) -> Path:
    root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else ApplicationContext.for_component("xrep").runs_dir
    )
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = name or f"{prefix}-{timestamp}"
    run_dir = root / run_name
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    return run_dir


def _pixel_scale_arcsec(wcs: WCS) -> float:
    return float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)


def _save_array(path: Path, array: np.ndarray) -> str:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array, allow_pickle=False)
    return str(output)


def _save_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _save_json(path: Path, payload: Any) -> str:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return str(output)


def _rewrite_saved_paths(
    saved: dict[str, str],
    run_dir: Path,
) -> dict[str, str]:
    rewritten = {}
    for key, value in saved.items():
        path = Path(value)
        rewritten[key] = str(path.relative_to(run_dir))
    return rewritten


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
