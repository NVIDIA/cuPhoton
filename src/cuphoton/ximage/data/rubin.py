# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rubin/LSST FITS probing helpers for xImage workflows."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from scipy.ndimage import map_coordinates

from cuphoton.ximage.data.bundle import (
    ARTIFACTS,
    load_ximage_bundle,
    validate_bundle,
)
from cuphoton.ximage.data.rubin_image import (
    RubinImageView,
    open_rubin_image,
    read_rubin_image_bounds,
)

_FITS_SUFFIXES = {".fit", ".fits", ".fz"}
_RUBIN_PRODUCTS = (
    "visit_image",
    "difference_image",
    "template_coadd",
    "deep_coadd",
)
_RUBIN_COADD_PRODUCTS = frozenset({"template_coadd", "deep_coadd"})
_RUBIN_BANDS = {"u", "g", "r", "i", "z", "y"}
_DATE_RE = re.compile(r"^20\d{6}$")
_VISIT_RE = re.compile(r"^20\d{10,}$")
_PHYSICAL_FILTER_RE = re.compile(r"^[ugrizy]_\d+$")
_DETECTOR_RE = re.compile(r"_(R\d{2}_S\d{2})(?:_|\.|$)")

_BITPIX_DTYPE = {
    8: "uint8",
    16: "int16",
    32: "int32",
    64: "int64",
    -32: "float32",
    -64: "float64",
}

_IMAGE_ARTIFACT_KEYS = (
    "images",
    "masks",
    "sky",
    "variances",
    "psfs",
    "mcgauss",
    "exp_times",
)
_MASK_POLICIES = {"rubin-bad-planes", "nonzero-bad", "none-good"}
_EXPOSURE_SUMMARY_KEYS = (
    "expTime",
    "meanVar",
    "nPsfStar",
    "pixelScale",
    "psfArea",
    "psfIxx",
    "psfIxy",
    "psfIyy",
    "psfSigma",
    "skyBg",
    "skyNoise",
    "zeroPoint",
)

_PSF_POLICY = "native-cell-psf-or-exposure-summary-gaussian-or-delta"


def probe_fits_products(
    input_path: str | Path,
    *,
    max_files: int | None = 32,
    read_sample: bool = False,
    sample_section: str | None = None,
) -> dict[str, Any]:
    """Inspect local FITS image products and their headers."""

    files = resolve_fits_paths(input_path, max_files=max_files)
    section = parse_section(sample_section) if sample_section else None
    results = [
        probe_fits_file(path, read_sample=read_sample, section=section)
        for path in files
    ]
    product_counts = Counter(result.get("product") for result in results)
    compression_counts = Counter(
        hdu.get("zcmptype") or "uncompressed"
        for result in results
        for hdu in result["hdus"]
        if hdu.get("is_image")
    )
    return {
        "input": str(Path(input_path).expanduser()),
        "file_count": len(results),
        "max_files": max_files,
        "read_sample": read_sample,
        "sample_section": section_to_text(section),
        "product_counts": dict(sorted(product_counts.items())),
        "image_compression_counts": dict(sorted(compression_counts.items())),
        "files": results,
    }


def build_rubin_manifest(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    inventory_path: str | Path | None = None,
    products: tuple[str, ...] | set[str] | list[str] | None = None,
    sample_per_product: int | None = 0,
    inspect_metadata: bool = True,
    group_prefix: str = "local",
) -> dict[str, Any]:
    """Build a deterministic Rubin FITS manifest for store conversion."""

    selected_products = normalize_products(products)
    if sample_per_product is None:
        sample_per_product = 0
    if sample_per_product < 0:
        raise ValueError("sample_per_product must be non-negative")
    if not group_prefix:
        raise ValueError("group_prefix must not be empty")

    input_candidate = Path(input_path).expanduser()
    if inventory_path is None:
        source_root = manifest_source_root(input_candidate)
        discovered = discover_rubin_product_files(
            input_candidate, selected_products, root=source_root
        )
    else:
        source_root = input_candidate
        discovered = discover_rubin_inventory_files(
            Path(inventory_path).expanduser(),
            selected_products,
            root=source_root,
        )
    rows: list[dict[str, Any]] = []
    discovered_counts: dict[str, int] = {}

    for product in selected_products:
        records = discovered.get(product, [])
        discovered_counts[product] = len(records)
        selected_records = stable_sample_records(
            records, max_count=sample_per_product or None
        )
        for record in selected_records:
            rows.append(
                build_manifest_row(
                    record["path"],
                    root=source_root,
                    group_prefix=group_prefix,
                    inspect_metadata=inspect_metadata,
                    size_bytes=record.get("size_bytes"),
                )
            )

    rows.sort(key=manifest_sort_key)
    product_counts = Counter(row["product"] for row in rows)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_kind": "rubin-local-fits",
        "rubin_image_format": "legacy-fits",
        "source_root": str(source_root),
        "file_count": len(rows),
        "products": selected_products,
        "product_counts": dict(sorted(product_counts.items())),
        "discovered_product_counts": discovered_counts,
        "selection": {
            "sample_per_product": sample_per_product,
            "sampling": "stable-evenly-spaced-by-product",
            "inspect_metadata": inspect_metadata,
            "group_prefix": group_prefix,
            "inventory_path": (
                None if inventory_path is None else str(inventory_path)
            ),
        },
        "metadata_coverage": manifest_metadata_coverage(rows),
        "files": rows,
    }
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest["output_path"] = str(path)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_rubin_store(
    *,
    manifest_path: str | Path,
    input_root: str | Path,
    output_dir: str | Path,
    group: str,
    product: str = "visit_image",
    max_files: int | None = None,
    roi: str | None = None,
    mask_policy: str = "rubin-bad-planes",
    psf_size: int = 25,
    align_wcs: bool = False,
    reference_index: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and validate an xImage NPY store before installing it.

    Existing stores survive input, write, or validation failures, including
    when ``overwrite=True``. Symlink and non-directory destinations are
    rejected.
    """

    if product not in _RUBIN_PRODUCTS:
        raise ValueError(f"Unsupported Rubin product: {product}")
    if product in _RUBIN_COADD_PRODUCTS:
        raise ValueError(
            "Rubin coadds are reference/evaluation products and cannot be "
            "built as independent exposure inputs for the xImage "
            "reconstruction stack."
        )
    if mask_policy not in _MASK_POLICIES:
        raise ValueError(
            "mask_policy must be one of " + ", ".join(sorted(_MASK_POLICIES))
        )
    if psf_size <= 0 or psf_size % 2 == 0:
        raise ValueError("psf_size must be a positive odd integer")
    if reference_index < 0:
        raise ValueError("reference_index must be non-negative")
    if max_files is not None and max_files <= 0:
        max_files = None

    manifest_path = Path(manifest_path).expanduser()
    input_root = Path(input_root).expanduser()
    output_dir = Path(output_dir).expanduser()
    manifest = json.loads(manifest_path.read_text())
    rows = select_manifest_rows(
        manifest,
        group=group,
        product=product,
        max_files=max_files,
    )
    if not rows:
        raise ValueError(
            f"No manifest rows matched group={group!r}, product={product!r}"
        )
    default_source_format = str(
        manifest.get("rubin_image_format", "legacy-fits")
    )
    _require_reconstruction_source_formats(
        rows,
        default_source_format=default_source_format,
    )

    if output_dir.name in {"", ".."}:
        raise ValueError("output_dir must name a store directory")
    if output_dir.is_symlink() or (
        output_dir.exists() and not output_dir.is_dir()
    ):
        raise ValueError(
            "output_dir must be a directory, not a symlink or file"
        )
    if output_dir.exists() and not overwrite:
        raise ValueError(f"{output_dir} already exists. Use --overwrite.")

    resolved_output = output_dir.resolve()
    input_paths = [manifest_path] + [
        input_root / row["relative_path"] for row in rows
    ]
    if any(
        path.resolve().is_relative_to(resolved_output) for path in input_paths
    ):
        raise ValueError(
            "output_dir cannot contain the manifest or selected input files"
        )

    section = parse_section(roi) if roi else None
    arrays, file_summaries = read_manifest_stack(
        rows,
        input_root=input_root,
        section=section,
        mask_policy=mask_policy,
        psf_size=psf_size,
        align_wcs=align_wcs,
        reference_index=reference_index,
        default_source_format=default_source_format,
    )
    psf_source_counts = dict(
        sorted(Counter(item["psf_source"] for item in file_summaries).items())
    )
    delta_count = psf_source_counts.get("delta_fallback", 0)
    if delta_count:
        warnings.warn(
            f"{delta_count} exposure(s) lack a usable PSF model; "
            "the store uses delta placeholder kernels. Supply measured "
            "PSFs before scientific reconstruction.",
            RuntimeWarning,
            stacklevel=2,
        )
    metadata = rubin_store_metadata(
        manifest=manifest,
        manifest_path=manifest_path,
        input_root=input_root,
        output_dir=output_dir,
        rows=rows,
        arrays=arrays,
        group=group,
        product=product,
        roi=section,
        mask_policy=mask_policy,
        psf_size=psf_size,
        align_wcs=align_wcs,
        reference_index=reference_index,
        file_summaries=file_summaries,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent
        )
    )
    staged_store = staging_root / "store"
    previous_store = staging_root / "previous"
    installed = False
    try:
        staged_store.mkdir()
        for key in _IMAGE_ARTIFACT_KEYS:
            np.save(staged_store / ARTIFACTS[key].npy_name, arrays[key])
        (staged_store / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        with load_ximage_bundle(staged_store) as bundle:
            issues = validate_bundle(bundle)
        if issues:
            raise ValueError(
                "Built Rubin store failed xImage validation: "
                + "; ".join(issues)
            )
        if output_dir.is_symlink() or (
            output_dir.exists() and not output_dir.is_dir()
        ):
            raise ValueError("output_dir changed to a symlink or file")
        if output_dir.exists():
            if not overwrite:
                raise ValueError(
                    f"{output_dir} already exists. Use --overwrite."
                )
            output_dir.rename(previous_store)
        try:
            staged_store.rename(output_dir)
        except BaseException:
            if previous_store.exists():
                previous_store.rename(output_dir)
            raise
        installed = True
    finally:
        # Keep the previous store available if restoring its name failed.
        if installed or not previous_store.exists():
            shutil.rmtree(staging_root)
    return {
        "output": str(output_dir),
        "file_count": len(rows),
        "group": group,
        "product": product,
        "roi": section_to_text(section),
        "mask_policy": mask_policy,
        "psf_policy": _PSF_POLICY,
        "psf_source_counts": psf_source_counts,
        "alignment_policy": (
            "reference-image-wcs" if align_wcs else "detector-pixel-roi"
        ),
        "artifacts": {
            key: {
                "path": str(output_dir / ARTIFACTS[key].npy_name),
                "shape": list(arrays[key].shape),
                "dtype": str(arrays[key].dtype),
                "nbytes": int(arrays[key].nbytes),
            }
            for key in _IMAGE_ARTIFACT_KEYS
        },
        "metadata": str(output_dir / "metadata.json"),
    }


def select_manifest_rows(
    manifest: dict[str, Any],
    *,
    group: str,
    product: str,
    max_files: int | None,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in manifest.get("files", [])
        if row.get("group") == group and row.get("product") == product
    ]
    rows.sort(
        key=lambda row: (
            row.get("visit") is None,
            row.get("visit") or 0,
            row.get("detector") is None,
            row.get("detector") or 0,
            row.get("band") or "",
            row.get("relative_path") or "",
        )
    )
    if max_files is not None:
        rows = rows[:max_files]
    return rows


def _require_reconstruction_source_formats(
    rows: list[dict[str, Any]],
    *,
    default_source_format: str,
) -> None:
    coadd_products = sorted(
        {
            str(row["product"])
            for row in rows
            if row.get("product") in _RUBIN_COADD_PRODUCTS
        }
    )
    if coadd_products:
        raise ValueError(
            "Rubin coadds are reference/evaluation products and cannot be "
            "used as independent exposure inputs for the xImage "
            "reconstruction stack; rejected products: "
            + ", ".join(coadd_products)
            + "."
        )
    source_formats = {
        str(row.get("rubin_image_format") or default_source_format)
        for row in rows
    }
    incomplete = sorted(
        source_formats & {"lsst-images-archive", "lsst-images-cutdown"}
    )
    if incomplete:
        raise ValueError(
            "The xImage exposure reconstruction stack accepts only complete "
            "independent exposure products. CellCoadd archives are "
            "reference/evaluation-only and SODA cut-down MaskedImage files "
            "are pixel-transport-only; rejected formats: "
            + ", ".join(incomplete)
            + "."
        )


def read_manifest_stack(
    rows: list[dict[str, Any]],
    *,
    input_root: Path,
    section: tuple[slice, slice] | None,
    mask_policy: str,
    psf_size: int,
    align_wcs: bool = False,
    reference_index: int = 0,
    default_source_format: str = "legacy-fits",
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    _require_reconstruction_source_formats(
        rows,
        default_source_format=default_source_format,
    )
    images: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    skies: list[np.ndarray] = []
    exp_times: list[float] = []
    psf_metrics: list[dict[str, Any] | None] = []
    psf_kernels: list[np.ndarray | None] = []
    file_summaries: list[dict[str, Any]] = []
    if reference_index >= len(rows):
        raise ValueError(
            f"reference_index {reference_index} exceeds "
            f"{len(rows)} selected rows"
        )

    reference_shape: tuple[int, int] | None = None
    reference_view: RubinImageView | None = None
    reference_section: tuple[slice, slice] | None = None
    if align_wcs:
        reference_row = rows[reference_index]
        reference_path = input_root / reference_row["relative_path"]
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Rubin FITS file not found: {reference_path}"
            )
        reference_format = str(
            reference_row.get("rubin_image_format") or default_source_format
        )
        reference_bounds = read_rubin_image_bounds(
            reference_path,
            source_format=reference_format,
        )
        reference_section = section or centered_section(
            reference_bounds.shape,
            size=512,
        )
        reference_view = open_rubin_image(
            reference_path,
            source_format=reference_format,
            section=reference_section,
            mask_policy=mask_policy,
        )
        if reference_view.wcs is None:
            raise ValueError(f"Reference image has no WCS: {reference_path}")

    for exposure_index, row in enumerate(rows):
        path = input_root / row["relative_path"]
        if not path.exists():
            raise FileNotFoundError(f"Rubin FITS file not found: {path}")
        source_format = str(
            row.get("rubin_image_format") or default_source_format
        )
        if align_wcs:
            if reference_view is None or reference_section is None:
                raise RuntimeError("Reference WCS was not initialized.")
            current_section = reference_section
            view = open_rubin_image(
                path,
                source_format=source_format,
                mask_policy=mask_policy,
            )
            image, variance, bad_mask, wcs_summary = read_wcs_aligned_views(
                target=view,
                reference=reference_view,
            )
        else:
            if section is None:
                bounds = read_rubin_image_bounds(
                    path,
                    source_format=source_format,
                )
                current_section = centered_section(
                    bounds.shape,
                    size=512,
                )
            else:
                current_section = section
            view = open_rubin_image(
                path,
                source_format=source_format,
                section=current_section,
                mask_policy=mask_policy,
            )
            image = view.image.astype(np.float32, copy=False)
            variance = view.variance.astype(np.float32, copy=False)
            bad_mask = np.array(view.bad_mask, dtype=bool, copy=True)
            wcs_summary = None
        exp_time = view.exposure_time
        exp_time_source = view.exposure_time_source
        image_bunit = view.unit
        normalization_exp_time, normalization_exp_time_source = (
            effective_exp_time_for_reconstruction(
                image_bunit=image_bunit,
                exp_time=exp_time,
                exp_time_source=exp_time_source,
            )
        )
        photo_calib = view.photo_calib
        exposure_summary = view.exposure_summary

        if reference_shape is None:
            reference_shape = tuple(image.shape)
        elif tuple(image.shape) != reference_shape:
            raise ValueError(
                f"Shape mismatch for {path}: {image.shape} != "
                f"{reference_shape}"
            )

        finite_image = np.isfinite(image)
        finite_variance = np.isfinite(variance)
        positive_variance = variance > 0
        invalid = ~finite_image | ~finite_variance | ~positive_variance
        if np.any(~finite_image):
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~finite_variance | ~positive_variance):
            variance = np.where(
                finite_variance & positive_variance, variance, 1.0
            ).astype(np.float32, copy=False)

        bad_mask |= invalid
        bad_mask_plane_names = list(view.selected_bad_mask_plane_names)
        bad_mask_plane_bits = [
            int(view.mask_plane_schema[name]["bit"])
            for name in bad_mask_plane_names
            if "bit" in view.mask_plane_schema.get(name, {})
        ]
        masks.append(to_ximage_mask(bad_mask))
        skies.append(np.zeros_like(image, dtype=np.float32))
        images.append(image)
        variances.append(variance)
        exp_times.append(normalization_exp_time)
        psf_metrics.append(exposure_summary)
        psf_kernel = view.sample_psf_kernel(psf_size)
        psf_kernels.append(psf_kernel)
        if psf_kernel is not None:
            psf_source = "lsst.images.CellCoadd.psf"
        elif psf_metrics_to_kernel_inputs(exposure_summary) is not None:
            psf_source = "ExposureSummaryStats"
        else:
            psf_source = "delta_fallback"
        file_summaries.append(
            {
                "exposure_index": exposure_index,
                "relative_path": row["relative_path"],
                "rubin_image_format": source_format,
                "visit": row.get("visit"),
                "detector": row.get("detector"),
                "band": row.get("band"),
                "section": section_to_text(current_section),
                "absolute_bounds": view.bounds.to_dict(),
                "alignment": (
                    None
                    if wcs_summary is None
                    else {
                        "policy": "reference-image-wcs",
                        **wcs_summary,
                    }
                ),
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "variance_dtype": str(variance.dtype),
                "image_bunit": image_bunit,
                "exp_time": exp_time,
                "exp_time_source": exp_time_source,
                "normalization_exp_time": normalization_exp_time,
                "normalization_exp_time_source": (
                    normalization_exp_time_source
                ),
                "photo_calib": photo_calib,
                "exposure_summary": exposure_summary,
                "psf_source": psf_source,
                "psf_spatial_sample_xy": (
                    None
                    if psf_kernel is None
                    else [
                        (view.bounds.x_start + view.bounds.x_stop - 1) / 2.0,
                        (view.bounds.y_start + view.bounds.y_stop - 1) / 2.0,
                    ]
                ),
                "psf_sigma": (
                    None
                    if exposure_summary is None
                    else exposure_summary.get("psfSigma")
                ),
                "raw_mask_nonzero_fraction": (view.mask_nonzero_fraction),
                "mask_plane_names": list(view.mask_plane_names),
                "bad_mask_plane_names": bad_mask_plane_names,
                "bad_mask_plane_bits": bad_mask_plane_bits,
                "mask_plane_schema": view.mask_plane_schema,
                "bad_mask_fraction": float(
                    np.count_nonzero(bad_mask) / bad_mask.size
                ),
                "effective_bad_fraction": float(
                    np.count_nonzero(bad_mask) / bad_mask.size
                ),
                "wcs_fidelity": view.wcs_fidelity,
                "requirements_met": view.requirements_met,
                "archive_info": view.archive_info,
                "cell_metadata": view.cell_metadata,
                "provenance_present": view.provenance is not None,
            }
        )

    image_stack = np.stack(images, axis=-1)[np.newaxis, ...]
    variance_stack = np.stack(variances, axis=-1)[np.newaxis, ...]
    mask_stack = np.stack(masks, axis=-1)[np.newaxis, ...]
    sky_stack = np.stack(skies, axis=-1)[np.newaxis, ...]
    psfs = psfs_from_exposure_summaries(psf_size, psf_metrics)
    mcgauss = mcgauss_from_exposure_summaries(psf_size, psf_metrics)
    for index, kernel in enumerate(psf_kernels):
        if kernel is None:
            continue
        psfs[:, :, 0, index] = kernel
        mcgauss[index, :, :] = kernel
    arrays = {
        "images": image_stack,
        "masks": mask_stack,
        "sky": sky_stack,
        "variances": variance_stack,
        "psfs": psfs,
        "mcgauss": mcgauss,
        "exp_times": np.asarray(exp_times, dtype=np.float64),
    }
    return arrays, file_summaries


def read_wcs_aligned_views(
    *,
    target: RubinImageView,
    reference: RubinImageView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample a normalized image view onto another view's exact WCS."""

    if (
        target.source_format != "legacy-fits"
        or reference.source_format != "legacy-fits"
    ):
        raise ValueError(
            "WCS alignment of lsst.images CellCoadd archives or cut-down "
            "MaskedImage transport products is not supported. Use native "
            "bounded CellCoadd reads for reference/evaluation analysis."
        )
    if target.wcs is None or reference.wcs is None:
        raise ValueError("WCS alignment requires WCS on both image views.")
    height, width = reference.image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    world = reference.wcs.pixel_to_world_values(
        xx.astype(np.float64),
        yy.astype(np.float64),
    )
    if not isinstance(world, tuple):
        world = (world,)
    target_pixels = target.wcs.world_to_pixel_values(*world)
    if not isinstance(target_pixels, tuple) or len(target_pixels) != 2:
        raise ValueError("Rubin image WCS is not two-dimensional.")
    target_x = np.asarray(target_pixels[0], dtype=np.float64)
    target_y = np.asarray(target_pixels[1], dtype=np.float64)
    target_height, target_width = target.image.shape
    valid = (
        np.isfinite(target_x)
        & np.isfinite(target_y)
        & (target_x >= 0.0)
        & (target_x <= target_width - 1)
        & (target_y >= 0.0)
        & (target_y <= target_height - 1)
    )
    coords = np.vstack(
        [
            np.where(valid, target_y, 0.0).ravel(),
            np.where(valid, target_x, 0.0).ravel(),
        ]
    )
    image = map_coordinates(
        target.image,
        coords,
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(reference.image.shape)
    variance = _propagate_bilinear_variance(
        target.variance,
        target_y=target_y,
        target_x=target_x,
        valid=valid,
    )
    bad_mask = _propagate_bilinear_mask(
        target.bad_mask,
        target_y=target_y,
        target_x=target_x,
        valid=valid,
    )
    image[~valid] = np.nan
    variance[~valid] = np.nan
    bad_mask[~valid] = 1.0

    finite_x = target_x[valid]
    finite_y = target_y[valid]
    summary = {
        "reference_bounds": reference.bounds.to_dict(),
        "target_valid_fraction": float(np.count_nonzero(valid) / valid.size),
        "variance_resampling": "bilinear-independent-squared-weights",
        "resampling_covariance": "not-represented",
        "mask_resampling": "bilinear-footprint-bitwise-or",
        "target_x_range": (
            None
            if finite_x.size == 0
            else [safe_float(np.min(finite_x)), safe_float(np.max(finite_x))]
        ),
        "target_y_range": (
            None
            if finite_y.size == 0
            else [safe_float(np.min(finite_y)), safe_float(np.max(finite_y))]
        ),
    }
    return (
        image.astype(np.float32),
        variance.astype(np.float32),
        bad_mask.astype(bool),
        summary,
    )


def resolve_fits_paths(
    input_path: str | Path, *, max_files: int | None = 32
) -> list[Path]:
    path = Path(input_path).expanduser()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in _FITS_SUFFIXES
        )
    else:
        raise FileNotFoundError(f"Rubin probe input not found: {path}")

    if not files:
        raise FileNotFoundError(f"No FITS files found under {path}")
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("max_files must be positive")
        files = files[:max_files]
    return files


def normalize_products(
    products: tuple[str, ...] | set[str] | list[str] | None,
) -> tuple[str, ...]:
    if products is None:
        products = ("visit_image", "difference_image")
    normalized = tuple(product for product in products if product)
    if not normalized:
        raise ValueError("products must not be empty")
    unsupported = sorted(set(normalized) - set(_RUBIN_PRODUCTS))
    if unsupported:
        raise ValueError(
            "Unsupported Rubin products: " + ", ".join(unsupported)
        )
    return tuple(
        product for product in _RUBIN_PRODUCTS if product in normalized
    )


def manifest_source_root(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser()
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    raise FileNotFoundError(f"Rubin manifest input not found: {path}")


def discover_rubin_product_files(
    input_path: Path, products: tuple[str, ...], *, root: Path
) -> dict[str, list[dict[str, Any]]]:
    files_by_product: dict[str, list[dict[str, Any]]] = {
        product: [] for product in products
    }
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(
            item
            for item in input_path.rglob("*")
            if item.is_file() and item.suffix.lower() in _FITS_SUFFIXES
        )
    for path in candidates:
        product = infer_product(path)
        if product in files_by_product:
            files_by_product[product].append(
                {"path": path, "size_bytes": path.stat().st_size}
            )
    if not any(files_by_product.values()):
        selected = ", ".join(products)
        raise FileNotFoundError(
            f"No FITS files for Rubin products {selected} under {input_path}"
        )
    for files in files_by_product.values():
        files.sort(
            key=lambda record: manifest_path_sort_key(record["path"], root)
        )
    return files_by_product


def discover_rubin_inventory_files(
    inventory_path: Path, products: tuple[str, ...], *, root: Path
) -> dict[str, list[dict[str, Any]]]:
    files_by_product: dict[str, list[dict[str, Any]]] = {
        product: [] for product in products
    }
    for record in read_inventory_records(inventory_path):
        if not record["path"].is_absolute():
            record["path"] = root / record["path"]
        product = infer_product(record["path"])
        if product in files_by_product:
            files_by_product[product].append(record)
    if not any(files_by_product.values()):
        selected = ", ".join(products)
        raise FileNotFoundError(
            f"No FITS files for Rubin products {selected} in {inventory_path}"
        )
    for files in files_by_product.values():
        files.sort(
            key=lambda record: manifest_path_sort_key(record["path"], root)
        )
    return files_by_product


def read_inventory_records(inventory_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        inventory_path.read_text().splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        size_bytes = None
        if len(parts) == 1:
            path_text = parts[0]
        elif parts[0].isdigit():
            size_bytes = int(parts[0])
            path_text = parts[1]
        elif parts[-1].isdigit():
            size_bytes = int(parts[-1])
            path_text = parts[0]
        else:
            raise ValueError(
                f"Invalid inventory line {line_number}: expected path or "
                "tab-separated size/path"
            )
        path = Path(path_text)
        if path.suffix.lower() not in _FITS_SUFFIXES:
            continue
        records.append({"path": path, "size_bytes": size_bytes})
    if not records:
        raise FileNotFoundError(
            f"No FITS inventory rows found: {inventory_path}"
        )
    return records


def stable_sample_records(
    records: list[dict[str, Any]], *, max_count: int | None
) -> list[dict[str, Any]]:
    if max_count is None or len(records) <= max_count:
        return list(records)
    if max_count <= 0:
        return []
    if max_count == 1:
        return [records[0]]

    last = len(records) - 1
    indices = [
        round(index * last / (max_count - 1)) for index in range(max_count)
    ]
    return [records[index] for index in indices]


def build_manifest_row(
    path: Path,
    *,
    root: Path,
    group_prefix: str,
    inspect_metadata: bool,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    metadata = parse_rubin_path_metadata(path, root=root)
    relative_path = str(path.relative_to(root))
    product = metadata["product"]
    row: dict[str, Any] = {
        "group": f"{group_prefix}_{product}",
        "product": product,
        "rubin_image_format": "legacy-fits",
        "relative_path": relative_path,
        "source_path": str(path),
        **{
            key: value
            for key, value in metadata.items()
            if key != "product" and value is not None
        },
    }
    if size_bytes is None and path.exists():
        size_bytes = path.stat().st_size
    if size_bytes is not None:
        row["size_bytes"] = int(size_bytes)
    if inspect_metadata:
        row.update(probe_manifest_metadata(path))
    return row


def parse_rubin_path_metadata(path: Path, *, root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    parts = relative.parts
    product = infer_product(path)
    product_index = next(
        (index for index, part in enumerate(parts) if part == product),
        -1,
    )
    after_product = (
        parts[product_index + 1 :] if product_index >= 0 else parts
    )
    date = next((int(part) for part in parts if _DATE_RE.match(part)), None)
    visit = next((int(part) for part in parts if _VISIT_RE.match(part)), None)
    band = next((part for part in parts if part in _RUBIN_BANDS), None)
    physical_filter = next(
        (part for part in parts if _PHYSICAL_FILTER_RE.match(part)),
        None,
    )
    detector_match = _DETECTOR_RE.search(path.name)
    detector = detector_match.group(1) if detector_match else None
    tract = None
    patch = None
    if product in {"template_coadd", "deep_coadd"}:
        numeric_parts = [
            int(part)
            for part in after_product
            if part.isdigit() and not _DATE_RE.match(part)
        ]
        if numeric_parts:
            tract = numeric_parts[0]
        if len(numeric_parts) > 1:
            patch = numeric_parts[1]
    return {
        "product": product,
        "date": date,
        "band": band,
        "physical_filter": physical_filter,
        "visit": visit,
        "detector": detector,
        "tract": tract,
        "patch": patch,
    }


def probe_manifest_metadata(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=True) as hdul:
        hdu_names = [str(hdu.name) for hdu in hdul]
        image_shape = None
        try:
            image_shape = normalized_shape(get_named_hdu(hdul, "IMAGE").shape)
        except ValueError:
            pass
        exp_time, exp_time_source = extract_exposure_time(hdul)
        exposure_summary = extract_exposure_summary(hdul)
    return {
        "hdu_count": len(hdu_names),
        "hdu_names": hdu_names,
        "image_shape": image_shape,
        "has_image_mask_variance": all(
            name in hdu_names for name in ("IMAGE", "MASK", "VARIANCE")
        ),
        "has_exptime": exp_time_source != "default_unit",
        "exp_time": exp_time,
        "exp_time_source": exp_time_source,
        "has_exposure_summary": exposure_summary is not None,
        "exposure_summary_keys": (
            [] if exposure_summary is None else sorted(exposure_summary)
        ),
    }


def manifest_path_sort_key(path: Path, root: Path) -> tuple[Any, ...]:
    metadata = parse_rubin_path_metadata(path, root=root)
    return (
        metadata.get("product") or "",
        metadata.get("date") or 0,
        metadata.get("band") or "",
        metadata.get("visit") or 0,
        metadata.get("detector") or "",
        metadata.get("tract") or 0,
        metadata.get("patch") or 0,
        str(path),
    )


def manifest_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("product") or "",
        row.get("date") or 0,
        row.get("band") or "",
        row.get("visit") or 0,
        row.get("detector") or "",
        row.get("tract") or 0,
        row.get("patch") or 0,
        row.get("relative_path") or "",
    )


def manifest_metadata_coverage(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for product in sorted({row["product"] for row in rows}):
        product_rows = [row for row in rows if row["product"] == product]
        coverage[product] = {
            "file_count": len(product_rows),
            "has_exptime": sum(
                1 for row in product_rows if row.get("has_exptime")
            ),
            "has_exposure_summary": sum(
                1 for row in product_rows if row.get("has_exposure_summary")
            ),
            "has_image_mask_variance": sum(
                1
                for row in product_rows
                if row.get("has_image_mask_variance")
            ),
        }
    return coverage


def probe_fits_file(
    path: Path,
    *,
    read_sample: bool = False,
    section: tuple[slice, slice] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        hdus = [
            probe_hdu(hdu, index=i, read_sample=read_sample, section=section)
            for i, hdu in enumerate(hdul)
        ]
    return {
        "path": str(path),
        "product": infer_product(path),
        "size_bytes": path.stat().st_size,
        "hdu_count": len(hdus),
        "hdus": hdus,
    }


def probe_hdu(
    hdu: fits.hdu.base.ExtensionHDU,
    *,
    index: int,
    read_sample: bool,
    section: tuple[slice, slice] | None,
) -> dict[str, Any]:
    header = hdu.header
    compressed_header = compressed_table_header(hdu)
    bitpix = header.get("BITPIX")
    shape = normalized_shape(getattr(hdu, "shape", None))
    info: dict[str, Any] = {
        "index": index,
        "name": hdu.name,
        "class": type(hdu).__name__,
        "xtension": header.get("XTENSION"),
        "bitpix": bitpix,
        "dtype": _BITPIX_DTYPE.get(bitpix),
        "shape": shape,
        "is_image": is_image_hdu(hdu, shape),
        "zcmptype": header.get("ZCMPTYPE")
        or (
            compressed_header.get("ZCMPTYPE")
            if compressed_header is not None
            else None
        ),
    }
    if compressed_header is not None:
        info["zbitpix"] = compressed_header.get("ZBITPIX")
        info["znaxis"] = compressed_header.get("ZNAXIS")

    if read_sample and info["is_image"]:
        info["sample"] = sample_image_hdu(hdu, section=section)
    return info


def compressed_table_header(hdu: Any) -> fits.Header | None:
    table = getattr(hdu, "_bintable", None)
    header = getattr(table, "header", None)
    return header if isinstance(header, fits.Header) else None


def normalized_shape(shape: Any) -> list[int] | None:
    if shape is None:
        return None
    return [int(dim) for dim in tuple(shape)]


def is_image_hdu(hdu: Any, shape: list[int] | None) -> bool:
    if not shape:
        return False
    return isinstance(
        hdu,
        (
            fits.PrimaryHDU,
            fits.ImageHDU,
            fits.CompImageHDU,
        ),
    )


def sample_image_hdu(
    hdu: Any, *, section: tuple[slice, slice] | None
) -> dict[str, Any]:
    shape = tuple(getattr(hdu, "shape", ()) or ())
    if len(shape) != 2:
        return {"available": False, "reason": "only 2D images supported"}
    if section is None:
        section = centered_section(shape, size=31)

    start = time.perf_counter()
    if hasattr(hdu, "section"):
        arr = np.asarray(hdu.section[section])
    else:
        arr = np.asarray(hdu.data[section])
    read_seconds = time.perf_counter() - start

    finite = np.isfinite(arr)
    if np.any(finite):
        finite_values = arr[finite]
        stats = {
            "min": safe_float(np.min(finite_values)),
            "mean": safe_float(np.mean(finite_values)),
            "max": safe_float(np.max(finite_values)),
            "std": safe_float(np.std(finite_values)),
        }
    else:
        stats = {"min": None, "mean": None, "max": None, "std": None}

    return {
        "available": True,
        "section": section_to_text(section),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "read_seconds": read_seconds,
        "finite_count": int(np.sum(finite)),
        "nonfinite_count": int(arr.size - np.sum(finite)),
        **stats,
    }


def _propagate_bilinear_variance(
    variance: np.ndarray,
    *,
    target_y: np.ndarray,
    target_x: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Propagate independent pixel variances through bilinear sampling.

    This returns marginal output variances only. Adjacent resampled pixels
    share input pixels, so their induced covariance remains unrepresented.
    """

    source = np.asarray(variance)
    if source.ndim != 2:
        raise ValueError("Bilinear variance propagation requires a 2-D plane")
    if target_y.shape != target_x.shape or target_y.shape != valid.shape:
        raise ValueError("Bilinear sampling coordinates must have one shape")

    safe_y = np.where(valid, target_y, 0.0)
    safe_x = np.where(valid, target_x, 0.0)
    y0 = np.floor(safe_y).astype(np.intp)
    x0 = np.floor(safe_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    dy = safe_y - y0
    dx = safe_x - x0

    result = np.zeros(target_y.shape, dtype=np.float64)
    for yy, xx, weight in (
        (y0, x0, (1.0 - dy) * (1.0 - dx)),
        (y0, x1, (1.0 - dy) * dx),
        (y1, x0, dy * (1.0 - dx)),
        (y1, x1, dy * dx),
    ):
        weight_squared = weight * weight
        contribution = np.zeros_like(result)
        np.multiply(
            source[yy, xx],
            weight_squared,
            out=contribution,
            where=weight_squared != 0.0,
        )
        result += contribution
    result[~valid] = np.nan
    return result


def _propagate_bilinear_mask(
    mask: np.ndarray,
    *,
    target_y: np.ndarray,
    target_x: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Combine mask values from every pixel with nonzero bilinear weight."""

    source = np.asarray(mask)
    if source.ndim != 2:
        raise ValueError("Bilinear mask propagation requires a 2-D plane")
    if not (
        np.issubdtype(source.dtype, np.bool_)
        or np.issubdtype(source.dtype, np.integer)
    ):
        raise ValueError(
            "Bilinear mask propagation requires bool or integer data"
        )
    if target_y.shape != target_x.shape or target_y.shape != valid.shape:
        raise ValueError("Bilinear sampling coordinates must have one shape")

    safe_y = np.where(valid, target_y, 0.0)
    safe_x = np.where(valid, target_x, 0.0)
    y0 = np.floor(safe_y).astype(np.intp)
    x0 = np.floor(safe_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    dy = safe_y - y0
    dx = safe_x - x0

    result = np.zeros(target_y.shape, dtype=source.dtype)
    zero = np.zeros((), dtype=source.dtype)
    for yy, xx, weight in (
        (y0, x0, (1.0 - dy) * (1.0 - dx)),
        (y0, x1, (1.0 - dy) * dx),
        (y1, x0, dy * (1.0 - dx)),
        (y1, x1, dy * dx),
    ):
        contribution = np.where(weight != 0.0, source[yy, xx], zero)
        np.bitwise_or(result, contribution, out=result)
    result[~valid] = 1
    return result


def get_named_hdu(hdul: fits.HDUList, name: str) -> Any:
    normalized = name.casefold()
    for hdu in hdul:
        if hdu.name.casefold() == normalized:
            return hdu
    raise ValueError(f"FITS file does not contain HDU named {name!r}")


def extract_exposure_time(hdul: fits.HDUList) -> tuple[float, str]:
    for hdu in hdul:
        value = hdu.header.get("EXPTIME")
        if value is None:
            continue
        try:
            exp_time = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(exp_time) and exp_time > 0:
            return exp_time, f"{hdu.name}.header.EXPTIME"
    return 1.0, "default_unit"


def extract_bunit(hdu: Any) -> str | None:
    value = hdu.header.get("BUNIT")
    if value is None:
        return None
    return str(value).strip() or None


def _center_probe_section(
    shape: tuple[int, int],
) -> tuple[slice, slice]:
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Coadd image has invalid shape {shape}.")
    y = shape[0] // 2
    x = shape[1] // 2
    return slice(y, y + 1), slice(x, x + 1)


def _coadd_hdu_metadata(
    path: Path,
    *,
    image_hdu: str,
    required: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hdu_names": [],
        "image_exptime": None,
        "image_mjd_avg": None,
        "primary_bunit": None,
        "primary_exptime": None,
    }
    try:
        with fits.open(path, memmap=True) as hdul:
            result["hdu_names"] = [
                str(hdu.header.get("EXTNAME"))
                for hdu in hdul
                if hdu.header.get("EXTNAME") is not None
            ]
            result["primary_bunit"] = extract_bunit(hdul[0])
            result["primary_exptime"] = safe_float_or_none(
                hdul[0].header.get("EXPTIME")
            )
            try:
                image = hdul[image_hdu]
            except (KeyError, IndexError) as exc:
                if required:
                    raise ValueError(
                        f"Coadd FITS does not contain image HDU {image_hdu!r}"
                    ) from exc
            else:
                result["image_exptime"] = safe_float_or_none(
                    image.header.get("EXPTIME")
                )
                result["image_mjd_avg"] = safe_float_or_none(
                    image.header.get("MJD-AVG")
                )
    except OSError:
        if required:
            raise
    return result


def _native_cell_psf_contract(view: RubinImageView) -> dict[str, Any]:
    x = (view.bounds.x_start + view.bounds.x_stop - 1) / 2.0
    y = (view.bounds.y_start + view.bounds.y_stop - 1) / 2.0
    try:
        kernel = view.sample_psf_kernel_at(x=x, y=y, size=25)
        cell = view.cell_diagnostics_at(x=x, y=y)
    except Exception:
        return {
            "valid": False,
            "native_cell_psf": True,
            "psf_type": (
                None if view.psf is None else type(view.psf).__name__
            ),
            "kernel_shape": None,
            "kernel_sum": None,
            "kernel_normalized": False,
            "cell": None,
        }
    if kernel is None:
        return {
            "valid": False,
            "native_cell_psf": True,
            "psf_type": None,
            "kernel_shape": None,
            "kernel_sum": None,
            "kernel_normalized": False,
            "cell": cell,
        }
    kernel_sum = float(np.sum(kernel))
    normalized = _close_to_unit(kernel_sum)
    return {
        "valid": normalized,
        "native_cell_psf": True,
        "psf_type": type(view.psf).__name__,
        "kernel_shape": list(kernel.shape),
        "kernel_sum": kernel_sum,
        "kernel_normalized": normalized,
        "cell": cell,
    }


def summarize_rubin_coadd_semantics(
    coadd_fits_path: str | Path,
    *,
    image_hdu: str = "IMAGE",
    rubin_image_format: str = "legacy-fits",
    output_path: str | Path | None = None,
    require_coadd_psf: bool = False,
    require_coadd_inputs: bool = False,
    require_visit_info: bool = False,
) -> dict[str, Any]:
    """Summarize a coadd through the normalized Rubin image adapter."""

    coadd_path = Path(coadd_fits_path).expanduser().resolve()
    bounds = read_rubin_image_bounds(
        coadd_path,
        source_format=rubin_image_format,
    )
    probe_section = _center_probe_section(bounds.shape)
    view = open_rubin_image(
        coadd_path,
        source_format=rubin_image_format,
        section=probe_section,
        include_provenance=True,
    )
    hdu_metadata = _coadd_hdu_metadata(
        coadd_path,
        image_hdu=image_hdu,
        required=rubin_image_format == "legacy-fits",
    )
    hdu_names = hdu_metadata["hdu_names"]
    hdu_names_lower = {name.casefold() for name in hdu_names}
    image_shape = list(bounds.shape)
    image_bunit = view.unit
    image_exptime = hdu_metadata["image_exptime"]
    image_mjd_avg = hdu_metadata["image_mjd_avg"]
    primary_bunit = hdu_metadata["primary_bunit"]
    primary_exptime = hdu_metadata["primary_exptime"]

    psf_contract: dict[str, Any] | None
    if rubin_image_format == "lsst-images-archive":
        coadd_inputs_present = view.provenance is not None
        visit_info_present = False
        psf_contract = _native_cell_psf_contract(view)
    else:
        coadd_inputs_present = "coaddinputs" in hdu_names_lower
        visit_info_present = "visitinfo" in hdu_names_lower
        psf_contract = None

    psf_values = psf_contract or {}
    checks = [
        _semantic_check(
            "coadd_image_hdu",
            image_shape is not None,
            f"HDU {image_hdu!r} has image shape {image_shape}",
            f"HDU {image_hdu!r} is not an image",
        ),
        _semantic_check(
            "calibrated_flux_unit",
            image_bunit is not None,
            f"IMAGE BUNIT is {image_bunit!r}",
            "IMAGE BUNIT is missing; flux normalization is ambiguous",
        ),
        _semantic_check(
            "coadd_inputs_present",
            coadd_inputs_present,
            "CoaddInputs HDU is present",
            "CoaddInputs HDU is missing",
            required=require_coadd_inputs,
        ),
        _semantic_check(
            "visit_info_present",
            visit_info_present,
            "VisitInfo HDU is present",
            "VisitInfo HDU is missing",
            required=require_visit_info,
        ),
        _semantic_check(
            "coadd_psf_present",
            view.psf is not None,
            "Native coadd PSF is available from the selected reader",
            "The selected reader does not expose a native coadd PSF",
            required=require_coadd_psf,
        ),
    ]
    if rubin_image_format == "lsst-images-archive":
        checks.extend(
            [
                _semantic_check(
                    "native_cell_psf_kernel_normalized",
                    psf_values.get("kernel_normalized") is True
                    and _close_to_unit(psf_values.get("kernel_sum")),
                    "Native cell PSF kernel is normalized",
                    "Native cell PSF kernel is unavailable or not normalized",
                    required=True,
                ),
                _semantic_check(
                    "native_coadd_provenance_present",
                    view.provenance is not None,
                    "Native CellCoadd provenance is present",
                    "Native CellCoadd provenance is absent",
                    required=True,
                ),
                _semantic_check(
                    "native_exact_wcs",
                    view.wcs_fidelity == "exact-lsst-images"
                    and view.wcs is not None,
                    "Native lsst.images WCS is present",
                    "Native lsst.images WCS is absent",
                    required=True,
                ),
            ]
        )
    ok = all(check["ok"] or not check["required"] for check in checks)
    if rubin_image_format == "lsst-images-archive":
        requirements_met = (
            ok and view.requirements_met and psf_values.get("valid") is True
        )
    else:
        requirements_met = False
    result = {
        "artifact_kind": "rubin_coadd_semantics",
        "coadd_fits_path": str(coadd_path),
        "rubin_image_format": rubin_image_format,
        "product_role": "reference-evaluation-only",
        "transport_role": view.product_role,
        "image_hdu": image_hdu,
        "hdu_names": hdu_names,
        "image": {
            "shape_yx": image_shape,
            "bunit": image_bunit,
            "exptime": image_exptime,
            "mjd_avg": image_mjd_avg,
        },
        "primary": {
            "bunit": primary_bunit,
            "exptime": primary_exptime,
        },
        "coadd_inputs": {
            "present": coadd_inputs_present,
            "source": (
                "lsst.images.CellCoadd.provenance"
                if rubin_image_format == "lsst-images-archive"
                else "CoaddInputs HDU"
            ),
        },
        "visit_info": {
            "present": visit_info_present,
            "source": (
                "represented-by-cell-provenance"
                if rubin_image_format == "lsst-images-archive"
                else "VisitInfo HDU"
            ),
        },
        "coadd_psf": psf_contract,
        "cell_metadata": view.cell_metadata,
        "reconstruction_contract": {
            "raw_exposure_stack_allowed": False,
            "reconstruction_ready": False,
            "reference_product_allowed": True,
            "template_product_allowed": True,
            "normalization_exp_time": 1.0,
            "normalization_exp_time_source": (
                "calibrated_coadd_unit_flux_semantics"
            ),
            "required_metadata": (
                [
                    "native lsst.images WCS",
                    "native cell PSF",
                    "native coadd provenance",
                ]
                if rubin_image_format == "lsst-images-archive"
                else [
                    "CoaddInputs",
                    "VisitInfo",
                    "native coadd PSF",
                ]
            ),
            "note": (
                "Rubin pipeline coadds are derived, calibrated products. "
                "Use them as references/templates with a suitable PSF "
                "model, not as independent raw visit exposures in the "
                "xImage likelihood stack."
            ),
        },
        "checks": checks,
        "ok": ok,
        "requirements_met": requirements_met,
    }
    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        result["output_path"] = str(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def effective_exp_time_for_reconstruction(
    *,
    image_bunit: str | None,
    exp_time: float,
    exp_time_source: str,
) -> tuple[float, str]:
    if image_bunit is not None and image_bunit.casefold() == "njy":
        return 1.0, "IMAGE.header.BUNIT=nJy"
    return exp_time, exp_time_source


def extract_exposure_summary(hdul: fits.HDUList) -> dict[str, Any] | None:
    try:
        hdu = get_named_hdu(hdul, "ExposureSummaryStats")
    except ValueError:
        return None
    data = getattr(hdu, "data", None)
    names = getattr(getattr(hdu, "columns", None), "names", None) or ()
    if data is None or "bytes" not in names:
        return None
    for row in data:
        raw = row["bytes"]
        payload = bytes(np.asarray(raw, dtype=np.uint8).tolist())
        text = payload.decode("utf-8", errors="ignore")
        if "psfSigma:" not in text and "skyBg:" not in text:
            continue
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            continue
        summary = {
            key: safe_float_or_none(parsed.get(key))
            for key in _EXPOSURE_SUMMARY_KEYS
            if key in parsed
        }
        return summary or None
    return None


def to_ximage_mask(bad_mask: np.ndarray) -> np.ndarray:
    masks = np.zeros(bad_mask.shape + (12,), dtype=np.uint8)
    # The reconstruction mask convention uses class 8 for excluded pixels.
    masks[:, :, 8] = bad_mask.astype(np.uint8)
    return masks


def psfs_from_exposure_summaries(
    psf_size: int, summaries: list[dict[str, Any] | None]
) -> np.ndarray:
    psfs = np.zeros((psf_size, psf_size, 1, len(summaries)), dtype=np.float64)
    for index, summary in enumerate(summaries):
        psfs[:, :, 0, index] = psf_kernel_from_summary(psf_size, summary)
    return psfs


def mcgauss_from_exposure_summaries(
    psf_size: int, summaries: list[dict[str, Any] | None]
) -> np.ndarray:
    mcgauss = np.zeros((len(summaries), psf_size, psf_size), dtype=np.float64)
    for index, summary in enumerate(summaries):
        mcgauss[index, :, :] = psf_kernel_from_summary(psf_size, summary)
    return mcgauss


def psf_kernel_from_summary(
    psf_size: int, summary: dict[str, Any] | None
) -> np.ndarray:
    inputs = psf_metrics_to_kernel_inputs(summary)
    if inputs is None:
        kernel = np.zeros((psf_size, psf_size), dtype=np.float64)
        kernel[psf_size // 2, psf_size // 2] = 1.0
        return kernel

    ixx, ixy, iyy, sigma = inputs
    coords = np.arange(psf_size, dtype=np.float64) - psf_size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    det = ixx * iyy - ixy * ixy
    if det > 0:
        inv_xx = iyy / det
        inv_xy = -ixy / det
        inv_yy = ixx / det
        quad = inv_xx * xx * xx + 2 * inv_xy * xx * yy + inv_yy * yy * yy
        kernel = np.exp(-0.5 * quad)
    else:
        kernel = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    total = float(np.sum(kernel))
    if not math.isfinite(total) or total <= 0:
        kernel = np.zeros((psf_size, psf_size), dtype=np.float64)
        kernel[psf_size // 2, psf_size // 2] = 1.0
        return kernel
    return kernel / total


def psf_metrics_to_kernel_inputs(
    summary: dict[str, Any] | None,
) -> tuple[float, float, float, float] | None:
    if summary is None:
        return None
    sigma = summary.get("psfSigma")
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        return None
    ixx = summary.get("psfIxx")
    iyy = summary.get("psfIyy")
    ixy = summary.get("psfIxy") or 0.0
    if not isinstance(ixx, (int, float)) or ixx <= 0:
        ixx = sigma * sigma
    if not isinstance(iyy, (int, float)) or iyy <= 0:
        iyy = sigma * sigma
    if not isinstance(ixy, (int, float)):
        ixy = 0.0
    return float(ixx), float(ixy), float(iyy), float(sigma)


def rubin_store_metadata(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    input_root: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    group: str,
    product: str,
    roi: tuple[slice, slice] | None,
    mask_policy: str,
    psf_size: int,
    align_wcs: bool,
    reference_index: int,
    file_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    source_formats = sorted(
        {str(summary["rubin_image_format"]) for summary in file_summaries}
    )
    artifacts = {}
    for key in _IMAGE_ARTIFACT_KEYS:
        artifact = ARTIFACTS[key]
        arr = arrays[key]
        artifacts[key] = {
            "storage_path": artifact.npy_name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes),
            "axes": list(artifact.axes),
            "chunking": None,
        }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "format": "npy",
        "source_dir": str(input_root.resolve()),
        "source_kind": (
            "rubin-local-fits"
            if source_formats == ["legacy-fits"]
            else "rubin-local-image-products"
        ),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_file_count": manifest.get("file_count"),
        "output_dir": str(output_dir.resolve()),
        "selection": {
            "group": group,
            "product": product,
            "file_count": len(rows),
            "rubin_image_formats": source_formats,
            "roi": section_to_text(roi),
            "mask_policy": mask_policy,
            "psf_policy": _PSF_POLICY,
            "psf_size": psf_size,
            "alignment_policy": (
                "reference-image-wcs" if align_wcs else "detector-pixel-roi"
            ),
            "reference_index": reference_index if align_wcs else None,
            "sky_policy": "zero-placeholder-not-estimated",
            "exp_times_policy": (
                "calibrated-njy-unit-else-fits-header-exptime-or-unit"
            ),
        },
        "artifacts": artifacts,
        "rubin_files": file_summaries,
    }


def parse_section(value: str) -> tuple[slice, slice]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("sample section must be 'y0:y1,x0:x1'")
    return tuple(parse_slice(part) for part in parts)  # type: ignore[return-value]


def parse_slice(value: str) -> slice:
    pieces = value.split(":")
    if len(pieces) != 2 or not pieces[0] or not pieces[1]:
        raise ValueError("sample section slices must use explicit start:stop")
    start = int(pieces[0])
    stop = int(pieces[1])
    if start < 0 or stop <= start:
        raise ValueError(
            "sample section slices must satisfy 0 <= start < stop"
        )
    return slice(start, stop)


def centered_section(
    shape: tuple[int, int], *, size: int
) -> tuple[slice, slice]:
    y, x = shape
    height = min(size, y)
    width = min(size, x)
    y0 = max((y - height) // 2, 0)
    x0 = max((x - width) // 2, 0)
    return (slice(y0, y0 + height), slice(x0, x0 + width))


def section_to_text(section: tuple[slice, slice] | None) -> str | None:
    if section is None:
        return None
    y, x = section
    return f"{y.start}:{y.stop},{x.start}:{x.stop}"


def safe_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def safe_float_or_none(value: Any) -> float | None:
    try:
        return safe_float(value)
    except (TypeError, ValueError):
        return None


def _semantic_check(
    name: str,
    ok: bool,
    pass_message: str,
    fail_message: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
        "message": pass_message if ok else fail_message,
    }


def _close_to_unit(value: Any) -> bool:
    number = safe_float_or_none(value)
    return number is not None and abs(number - 1.0) <= 1.0e-8


def infer_product(path: Path) -> str:
    for part in path.parts:
        if part in _RUBIN_PRODUCTS:
            return part
    name = path.name
    for product in _RUBIN_PRODUCTS:
        if name.startswith(product):
            return product
    return "unknown"
