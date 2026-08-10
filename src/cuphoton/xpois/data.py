# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Data helpers for XPOIS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from cuphoton.core.cli import ApplicationContext

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
    """Resolve the XPOIS HSC data directory without creating it."""

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


def load_image_array(path: Path, hdu: int | None = None) -> np.ndarray:
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

    array, _, _ = load_image_with_wcs(path, hdu=hdu)
    return array


def load_variance_array(path: Path, hdu: int | None = None) -> np.ndarray:
    """Load a two-dimensional variance image from FITS or NumPy."""

    array, _, _ = load_variance_with_wcs(path, hdu=hdu)
    return array


def load_mask_array(path: Path, hdu: int | None = None) -> np.ndarray:
    """Load a two-dimensional integer mask from FITS or NumPy."""

    array, _, _ = load_mask_with_planes(path, hdu=hdu)
    return array


def load_image_with_wcs(
    path: Path,
    hdu: int | None = None,
) -> tuple[np.ndarray, WCS | None, int | None]:
    """Load a two-dimensional image together with optional FITS WCS.

    Returns
    -------
    array
        Floating-point image.
    wcs
        FITS world-coordinate system, or ``None`` for NumPy input.
    hdu
        Selected FITS HDU index, or ``None`` for NumPy input.
    """

    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".npy":
        if hdu is not None:
            raise ValueError(
                "HDU selectors are not supported for NumPy inputs: "
                f"{resolved}"
            )
        return np.asarray(np.load(resolved), dtype=np.float64), None, None
    if suffix not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported image format: {resolved}")

    with fits.open(resolved) as hdul:
        if hdu is not None:
            if hdu >= len(hdul):
                raise ValueError(f"HDU {hdu} is out of range for {resolved}")
            data = hdul[hdu].data
            if data is None or data.ndim != 2:
                raise ValueError(f"HDU {hdu} in {resolved} is not a 2D image")
            return (
                np.asarray(data, dtype=np.float64),
                WCS(hdul[hdu].header),
                hdu,
            )

        for index, item in enumerate(hdul):
            data = item.data
            if data is not None and data.ndim == 2:
                return (
                    np.asarray(data, dtype=np.float64),
                    WCS(item.header),
                    index,
                )

    raise ValueError(f"Could not find a 2D image HDU in {resolved}")


def load_variance_with_wcs(
    path: Path,
    hdu: int | None = None,
) -> tuple[np.ndarray, WCS | None, int | None]:
    """Load a variance image, preferring an unambiguous named FITS HDU."""

    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".npy":
        if hdu is not None:
            raise ValueError(
                "HDU selectors are not supported for NumPy inputs: "
                f"{resolved}"
            )
        return np.asarray(np.load(resolved), dtype=np.float64), None, None
    if suffix not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported image format: {resolved}")

    with fits.open(resolved) as hdul:
        if hdu is not None:
            return _load_fits_hdu(hdul, resolved, hdu)

        named_candidates: list[int] = []
        image_candidates: list[int] = []
        error_candidates: list[int] = []
        for index, item in enumerate(hdul):
            data = item.data
            if data is None or data.ndim != 2:
                continue
            image_candidates.append(index)
            extname = str(item.header.get("EXTNAME", "")).strip().upper()
            if extname in VARIANCE_EXTENSION_NAMES:
                named_candidates.append(index)
            if extname in ERROR_EXTENSION_NAMES:
                error_candidates.append(index)

        if len(named_candidates) == 1:
            return _load_fits_hdu(hdul, resolved, named_candidates[0])
        if len(named_candidates) > 1:
            raise ValueError(
                "multiple variance-like FITS HDUs found; "
                "specify --variance-hdu explicitly"
            )
        if len(image_candidates) == 1:
            if image_candidates[0] in error_candidates:
                raise ValueError(
                    "variance FITS uses an error/sigma extension; "
                    "specify --variance-hdu explicitly"
                )
            return _load_fits_hdu(hdul, resolved, image_candidates[0])
    raise ValueError(
        "variance FITS is ambiguous; specify --variance-hdu explicitly"
    )


def load_mask_with_planes(
    path: Path,
    hdu: int | None = None,
) -> tuple[np.ndarray, int | None, dict[str, int] | None]:
    """Load an integer mask and any FITS mask-plane mapping."""

    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".npy":
        if hdu is not None:
            raise ValueError(
                "HDU selectors are not supported for NumPy mask inputs: "
                f"{resolved}"
            )
        return np.asarray(np.load(resolved), dtype=np.int64), None, None
    if suffix not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported mask format: {resolved}")

    with fits.open(resolved) as hdul:
        if hdu is not None:
            return _load_mask_hdu(hdul, resolved, hdu)

        named_candidates: list[int] = []
        for index, item in enumerate(hdul):
            data = item.data
            if data is None or data.ndim != 2:
                continue
            extname = str(item.header.get("EXTNAME", "")).strip().upper()
            if extname in MASK_EXTENSION_NAMES:
                named_candidates.append(index)

        if len(named_candidates) == 1:
            return _load_mask_hdu(hdul, resolved, named_candidates[0])
        if len(named_candidates) > 1:
            raise ValueError(
                "multiple mask-like FITS HDUs found; specify a mask HDU "
                "explicitly"
            )
    raise ValueError("mask FITS is ambiguous; specify a mask HDU explicitly")


def _load_fits_hdu(
    hdul: fits.HDUList,
    resolved: Path,
    hdu: int,
) -> tuple[np.ndarray, WCS | None, int]:
    if hdu >= len(hdul):
        raise ValueError(f"HDU {hdu} is out of range for {resolved}")
    data = hdul[hdu].data
    if data is None or data.ndim != 2:
        raise ValueError(f"HDU {hdu} in {resolved} is not a 2D image")
    return (
        np.asarray(data, dtype=np.float64),
        WCS(hdul[hdu].header),
        hdu,
    )


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


def _load_mask_hdu(
    hdul: fits.HDUList,
    resolved: Path,
    hdu: int,
) -> tuple[np.ndarray, int, dict[str, int] | None]:
    if hdu >= len(hdul):
        raise ValueError(f"HDU {hdu} is out of range for {resolved}")
    data = hdul[hdu].data
    if data is None or data.ndim != 2:
        raise ValueError(f"HDU {hdu} in {resolved} is not a 2D mask image")
    plane_map: dict[str, int] = {}
    for key, value in hdul[hdu].header.items():
        if not key.startswith("MP_"):
            continue
        try:
            bit = int(value)
        except Exception:
            continue
        plane_map[key[3:].strip().upper()] = bit
    return (
        np.asarray(data, dtype=np.int64),
        hdu,
        plane_map or None,
    )
