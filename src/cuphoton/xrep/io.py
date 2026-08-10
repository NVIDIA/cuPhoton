# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS I/O helpers for xRep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .geometry import BBox, Grid, bbox_wcs

FITS_SUFFIXES = {".fits", ".fit", ".fts"}


def load_fits_image_with_wcs(
    path: Path,
    *,
    hdu: int | None = None,
) -> tuple[np.ndarray, WCS, fits.Header, int]:
    """Load a 2D FITS image and its WCS."""

    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported FITS image path: {resolved}")

    with fits.open(resolved, memmap=True) as hdul:
        if hdu is not None:
            return _load_2d_image_hdu(hdul, resolved, hdu)

        for index, item in enumerate(hdul):
            data = item.data
            if data is None or data.ndim != 2:
                continue
            return (
                np.asarray(data),
                WCS(item.header),
                item.header.copy(),
                index,
            )

    raise ValueError(f"Could not find a 2D FITS image HDU in {resolved}")


def load_fits_mask(
    path: Path,
    *,
    hdu: int | None = None,
) -> tuple[np.ndarray, fits.Header, int]:
    """Load a 2D FITS mask image."""

    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in FITS_SUFFIXES:
        raise ValueError(f"Unsupported FITS mask path: {resolved}")

    with fits.open(resolved, memmap=True) as hdul:
        if hdu is not None:
            data, header, used = _load_2d_mask_hdu(hdul, resolved, hdu)
            return data, header, used

        for index, item in enumerate(hdul):
            data = item.data
            if data is None or data.ndim != 2:
                continue
            return (
                np.asarray(data),
                item.header.copy(),
                index,
            )

    raise ValueError(f"Could not find a 2D FITS mask HDU in {resolved}")


def write_reprojected_fits(
    path: Path,
    image: np.ndarray,
    *,
    grid: Grid,
    bbox: BBox,
    mask: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one reprojected image (and optional mask) to FITS."""

    header = bbox_wcs(grid, bbox).to_header()
    if metadata:
        for key, value in metadata.items():
            fits_key = str(key).upper()[:8]
            if fits_key in header:
                continue
            try:
                header[fits_key] = value
            except Exception:
                continue
    hdus: list[fits.HDUBase] = [
        fits.PrimaryHDU(),
        fits.ImageHDU(
            data=np.asarray(image, dtype=np.float32),
            header=header,
            name="IMAGE",
        ),
    ]
    if mask is not None:
        mask_data = np.asarray(mask)
        if np.issubdtype(mask_data.dtype, np.bool_):
            mask_data = mask_data.astype(np.uint8)
        hdus.append(
            fits.ImageHDU(
                data=mask_data,
                name="MASK",
            )
        )
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(output, overwrite=True)
    return output


def write_stack_fits(
    path: Path,
    stack: np.ndarray,
    *,
    grid: Grid,
    bbox: BBox,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a stack of reprojected images to FITS."""

    header = bbox_wcs(grid, bbox).to_header()
    if metadata:
        for key, value in metadata.items():
            fits_key = str(key).upper()[:8]
            if fits_key in header:
                continue
            try:
                header[fits_key] = value
            except Exception:
                continue
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(
        data=np.asarray(stack, dtype=np.float32),
        header=header,
    ).writeto(output, overwrite=True)
    return output


def _load_2d_image_hdu(
    hdul: fits.HDUList,
    resolved: Path,
    hdu: int,
) -> tuple[np.ndarray, WCS, fits.Header, int]:
    if hdu >= len(hdul):
        raise ValueError(f"HDU {hdu} is out of range for {resolved}")
    data = hdul[hdu].data
    if data is None or data.ndim != 2:
        raise ValueError(f"HDU {hdu} in {resolved} is not a 2D image")
    header = hdul[hdu].header.copy()
    return (
        np.asarray(data),
        WCS(header),
        header,
        hdu,
    )


def _load_2d_mask_hdu(
    hdul: fits.HDUList,
    resolved: Path,
    hdu: int,
) -> tuple[np.ndarray, fits.Header, int]:
    if hdu >= len(hdul):
        raise ValueError(f"HDU {hdu} is out of range for {resolved}")
    data = hdul[hdu].data
    if data is None or data.ndim != 2:
        raise ValueError(f"HDU {hdu} in {resolved} is not a 2D mask")
    return np.asarray(data), hdul[hdu].header.copy(), hdu
