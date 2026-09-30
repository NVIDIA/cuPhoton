# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Data helpers for xPois."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from astropy.wcs import WCS

from cuphoton.core.cli import ApplicationContext
from cuphoton.core.fits_io import (
    inspect_fits_image,
    inspect_fits_images,
    read_fits_images,
)

FITS_SUFFIXES = {".fits", ".fit", ".fts"}
MASK_EXTENSION_NAMES = {
    "MASK",
}
VARIANCE_EXTENSION_NAMES = {
    "VAR",
    "VARIANCE",
}
ERROR_EXTENSION_NAMES = {
    "ERROR",
    "ERRORS",
    "ERR",
    "ERRS",
    "SIGMA",
}
_SHARED_HSC_ENV = "CUPHOTON_XPOIS_SHARED_HSC_DIR"
_SHARED_HSC_SEARCH_PATH = Path("data/HSC")


def discover_shared_hsc_dir(start_dir: Path | None = None) -> Path | None:
    """Search a directory and its parents for a shared HSC data tree."""

    current = (start_dir or Path.cwd()).expanduser().resolve()
    for base in (current, *current.parents):
        candidate = (base / _SHARED_HSC_SEARCH_PATH).resolve()
        if candidate.exists():
            return candidate
    return None


def default_shared_hsc_dir() -> Path:
    """Resolve the xPois HSC data directory without creating it."""

    explicit = os.environ.get(_SHARED_HSC_ENV)
    if explicit:
        return Path(explicit).expanduser().resolve()
    discovered = discover_shared_hsc_dir()
    if discovered is not None:
        return discovered
    context = ApplicationContext.for_component("xpois")
    return (context.data_dir / "HSC").resolve()


def inspect_hsc_data_tree(base: Path | None = None) -> dict[str, Any]:
    """Summarize conventional HSC bundle and FITS products.

    Parameters
    ----------
    base
        HSC data root. The configured shared directory is used when omitted.

    Returns
    -------
    dict
        Paths, existence flags, and product counts without loading image data.
    """

    root = (base or default_shared_hsc_dir()).expanduser().resolve()
    summary: dict[str, Any] = {
        "base": str(root),
        "exists": root.exists(),
        "bundle_files": {},
        "fits": {
            "coadds": [],
            "warps": [],
            "catalogs": [],
        },
        "counts": {
            "bundle_files": 0,
            "coadds": 0,
            "warps": 0,
            "catalogs": 0,
        },
    }
    if not root.exists():
        return summary

    for name in (
        "HSC_expTimes.pkl",
        "HSC_images.pkl",
        "HSC_masks.pkl",
        "HSC_psfs.pkl",
        "HSC_sky.pkl",
        "HSC_variances.pkl",
        "mcgauss.pkl",
    ):
        path = root / name
        summary["bundle_files"][name] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else None,
        }

    fits_root = root / "FITS"
    if fits_root.exists():
        summary["fits"]["coadds"] = sorted(
            str(path.relative_to(root))
            for path in fits_root.glob("coadd/**/*.fits")
        )
        summary["fits"]["warps"] = sorted(
            str(path.relative_to(root))
            for path in fits_root.glob("warps/**/*.fits")
        )
        summary["fits"]["catalogs"] = sorted(
            str(path.relative_to(root))
            for path in fits_root.glob("catalogs/**/*.parq")
        ) + sorted(
            str(path.relative_to(root))
            for path in fits_root.glob("catalogs/**/*.parquet")
        )
    summary["counts"] = {
        "bundle_files": sum(
            1 for item in summary["bundle_files"].values() if item["exists"]
        ),
        "coadds": len(summary["fits"]["coadds"]),
        "warps": len(summary["fits"]["warps"]),
        "catalogs": len(summary["fits"]["catalogs"]),
    }
    return summary


def load_image_array(
    path: Path,
    hdu: int | None = None,
    *,
    fits_reader: str = "astropy",
    xdr_options=None,
) -> np.ndarray:
    """Load a two-dimensional FITS or NumPy image.

    Parameters
    ----------
    path
        ``.fits`` or ``.npy`` input path.
    hdu
        Explicit FITS HDU index; unsupported for NumPy inputs.

    Returns
    -------
    numpy.ndarray
        Floating-point image data.
    """

    array, _, _ = load_image_with_wcs(
        path, hdu=hdu, fits_reader=fits_reader, xdr_options=xdr_options
    )
    return array


def load_variance_array(
    path: Path,
    hdu: int | None = None,
    *,
    fits_reader: str = "astropy",
    xdr_options=None,
) -> np.ndarray:
    """Load a two-dimensional variance image from FITS or NumPy."""

    array, _, _ = load_variance_with_wcs(
        path, hdu=hdu, fits_reader=fits_reader, xdr_options=xdr_options
    )
    return array


def load_mask_array(
    path: Path,
    hdu: int | None = None,
    *,
    fits_reader: str = "astropy",
    xdr_options=None,
) -> np.ndarray:
    """Load a two-dimensional integer mask from FITS or NumPy."""

    array, _, _ = load_mask_with_planes(
        path, hdu=hdu, fits_reader=fits_reader, xdr_options=xdr_options
    )
    return array


def load_fit_positions(
    path: Path,
    *,
    image_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    """Load strict post-crop ``(y, x)`` fit rows from an NPY file."""

    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".npy":
        raise ValueError("fit_positions must be an NPY file")
    positions = np.asarray(np.load(resolved, allow_pickle=False))
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("fit_positions must have shape (row, 2)")
    if positions.shape[0] == 0:
        raise ValueError("fit_positions must not be empty")
    if positions.dtype == bool or not np.issubdtype(
        positions.dtype, np.integer
    ):
        raise ValueError("fit_positions must contain integers")

    margin_y = kernel_shape[0] // 2
    margin_x = kernel_shape[1] // 2
    sample_y = positions[:, 0]
    sample_x = positions[:, 1]
    if (
        np.any(sample_y < margin_y)
        or np.any(sample_y >= image_shape[0] - margin_y)
        or np.any(sample_x < margin_x)
        or np.any(sample_x >= image_shape[1] - margin_x)
    ):
        raise ValueError(
            "fit_positions must lie inside the valid kernel interior"
        )
    return positions.astype(np.int64, copy=False)


def _load_plane(path, hdu, *, reader, dtype, read_metadata, xdr_options=None):
    result = read_fits_images(
        path, [hdu], reader=reader, xdr_options=xdr_options
    )
    if read_metadata is not None:
        read_metadata.append(result.metadata())
    return np.asarray(result.arrays[0], dtype=dtype)


def load_image_with_wcs(
    path: Path,
    hdu: int | None = None,
    *,
    fits_reader: str = "astropy",
    read_metadata: list[dict[str, Any]] | None = None,
    xdr_options=None,
) -> tuple[np.ndarray, WCS | None, int | None]:
    """Load a host image with optional GPU FITS decompression and its WCS."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() == ".npy":
        if hdu is not None:
            raise ValueError(
                "HDU selectors are not supported for NumPy inputs: "
                f"{resolved}"
            )
        return (
            np.asarray(
                np.load(resolved, allow_pickle=False), dtype=np.float64
            ),
            None,
            None,
        )
    if resolved.suffix.lower() not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported image format: {resolved}")
    info = inspect_fits_image(resolved, hdu=hdu)
    array = _load_plane(
        resolved,
        info.hdu,
        reader=fits_reader,
        dtype=np.float64,
        read_metadata=read_metadata,
        xdr_options=xdr_options,
    )
    return array, WCS(info.header), info.hdu


def load_variance_with_wcs(
    path: Path,
    hdu: int | None = None,
    *,
    fits_reader: str = "astropy",
    read_metadata: list[dict[str, Any]] | None = None,
    xdr_options=None,
) -> tuple[np.ndarray, WCS | None, int | None]:
    """Load variance, preserving the unambiguous named-HDU policy."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() == ".npy" or hdu is not None:
        return load_image_with_wcs(
            resolved,
            hdu,
            fits_reader=fits_reader,
            read_metadata=read_metadata,
            xdr_options=xdr_options,
        )
    if resolved.suffix.lower() not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported image format: {resolved}")
    images = inspect_fits_images(resolved)
    named = [
        info
        for info in images
        if str(info.header.get("EXTNAME", "")).strip().upper()
        in VARIANCE_EXTENSION_NAMES
    ]
    if len(named) > 1:
        raise ValueError(
            "multiple variance-like FITS HDUs found; "
            "specify --variance-hdu explicitly"
        )
    if len(named) == 1:
        selected = named[0]
    elif len(images) == 1:
        selected = images[0]
        if (
            str(selected.header.get("EXTNAME", "")).strip().upper()
            in ERROR_EXTENSION_NAMES
        ):
            raise ValueError(
                "variance FITS uses an error/sigma extension; "
                "specify --variance-hdu explicitly"
            )
    else:
        raise ValueError(
            "variance FITS is ambiguous; specify --variance-hdu explicitly"
        )
    return load_image_with_wcs(
        resolved,
        selected.hdu,
        fits_reader=fits_reader,
        read_metadata=read_metadata,
        xdr_options=xdr_options,
    )


def load_mask_with_planes(
    path: Path,
    hdu: int | None = None,
    *,
    fits_reader: str = "astropy",
    read_metadata: list[dict[str, Any]] | None = None,
    xdr_options=None,
) -> tuple[np.ndarray, int | None, dict[str, int] | None]:
    """Load an integer mask and preserve its FITS mask-plane mapping."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() == ".npy":
        if hdu is not None:
            raise ValueError(
                "HDU selectors are not supported for NumPy mask inputs: "
                f"{resolved}"
            )
        return (
            np.asarray(np.load(resolved, allow_pickle=False), dtype=np.int64),
            None,
            None,
        )
    if resolved.suffix.lower() not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported mask format: {resolved}")
    if hdu is None:
        named = [
            info
            for info in inspect_fits_images(resolved)
            if str(info.header.get("EXTNAME", "")).strip().upper()
            in MASK_EXTENSION_NAMES
        ]
        if len(named) > 1:
            raise ValueError(
                "multiple mask-like FITS HDUs found; "
                "specify a mask HDU explicitly"
            )
        if not named:
            raise ValueError(
                "mask FITS is ambiguous; specify a mask HDU explicitly"
            )
        info = named[0]
    else:
        info = inspect_fits_image(resolved, hdu=hdu)
    plane_map = {}
    for key, value in info.header.items():
        if key.startswith("MP_"):
            try:
                plane_map[key[3:].strip().upper()] = int(value)
            except (TypeError, ValueError):
                continue
    array = _load_plane(
        resolved,
        info.hdu,
        reader=fits_reader,
        dtype=np.int64,
        read_metadata=read_metadata,
        xdr_options=xdr_options,
    )
    return array, info.hdu, plane_map or None


def apply_rectangular_cutout(
    array: np.ndarray,
    *,
    y0: int,
    x0: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Extract a bounded rectangular image cutout.

    Parameters
    ----------
    array
        Source array with spatial axes first.
    y0, x0
        Non-negative cutout origin.
    height, width
        Positive cutout extent.

    Returns
    -------
    numpy.ndarray
        Selected cutout.
    """

    if height <= 0 or width <= 0:
        raise ValueError("cutout height and width must be positive")
    if y0 < 0 or x0 < 0:
        raise ValueError("cutout origin must be non-negative")
    y1 = y0 + height
    x1 = x0 + width
    if y1 > array.shape[0] or x1 > array.shape[1]:
        raise ValueError("requested cutout exceeds image bounds")
    return np.asarray(array[y0:y1, x0:x1])
