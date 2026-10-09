# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS I/O helpers for xRep."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from cuphoton.core.fits_io import inspect_fits_image, read_fits_images

from .geometry import BBox, Grid, bbox_wcs

FITS_SUFFIXES = {".fits", ".fit", ".fts"}


def inspect_fits_image_with_wcs(path: Path, *, hdu: int | None = None):
    """Return image geometry and WCS without reading pixels."""
    info = inspect_fits_image(path, hdu=hdu)
    return info.shape, WCS(info.header), info.header, info.hdu


def load_fits_image_with_wcs(
    path: Path,
    *,
    hdu: int | None = None,
    fits_reader: str = "astropy",
    xdr_options: Mapping[str, str] | None = None,
    device: bool = False,
    read_metadata: list[dict[str, Any]] | None = None,
) -> tuple[Any, WCS, fits.Header, int]:
    """Load a FITS plane on the host or device, with its unchanged WCS."""
    info = inspect_fits_image(path, hdu=hdu)
    result = read_fits_images(
        path,
        [info.hdu],
        reader=fits_reader,
        device=device,
        xdr_options=xdr_options,
    )
    if read_metadata is not None:
        read_metadata.append(result.metadata())
    return result.arrays[0], WCS(info.header), info.header, info.hdu


def load_fits_mask(
    path: Path,
    *,
    hdu: int | None = None,
    fits_reader: str = "astropy",
    xdr_options: Mapping[str, str] | None = None,
    device: bool = False,
    read_metadata: list[dict[str, Any]] | None = None,
) -> tuple[Any, fits.Header, int]:
    """Load a FITS mask without narrowing its integer bit representation."""
    info = inspect_fits_image(path, hdu=hdu)
    result = read_fits_images(
        path,
        [info.hdu],
        reader=fits_reader,
        device=device,
        xdr_options=xdr_options,
    )
    if read_metadata is not None:
        read_metadata.append(result.metadata())
    return result.arrays[0], info.header, info.hdu


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

    header = bbox_wcs(grid, bbox).to_header(relax=True)
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

    header = bbox_wcs(grid, bbox).to_header(relax=True)
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
