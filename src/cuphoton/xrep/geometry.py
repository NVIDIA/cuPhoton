# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Core dataclasses and WCS helpers for xRep."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.wcs import WCS

MappingFn = Callable[[np.ndarray], np.ndarray]


@dataclass(slots=True)
class Grid:
    """A fixed north-up sky grid.

    Attributes
    ----------
    crval
        ``(right ascension, declination)`` reference in ICRS-like degrees.
    pixel_scale_arcsec
        Pixel scale in arcseconds per pixel.
    wcs
        Derived two-dimensional TAN world-coordinate system.
    """

    crval: tuple[float, float]
    pixel_scale_arcsec: float
    wcs: WCS = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.wcs = make_north_up_wcs(
            self.crval,
            shape=(1, 1),
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            crpix=(1.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in grid coordinates.

    Attributes
    ----------
    min_x, min_y
        Inclusive, zero-based ``(x, y)`` origin in shared-grid pixels. Values
        may be negative when a source footprint crosses the grid origin.
    width, height
        Positive dimensions in pixels.
    """

    min_x: int
    min_y: int
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""

        return (self.height, self.width)

    @property
    def origin(self) -> tuple[int, int]:
        """Return the zero-based ``(x, y)`` grid origin."""

        return (self.min_x, self.min_y)


@dataclass(frozen=True, slots=True)
class ReprojectionSpec:
    """Backend-neutral reprojection description for one source image.

    Attributes
    ----------
    mapping
        Callable mapping zero-based destination ``(x, y)`` coordinates to
        zero-based source pixels.
    output_bbox
        Destination region on the shared grid.
    source_origin
        Zero-based ``(x, y)`` origin subtracted from mapped source pixels when
        the source array is itself a cutout.
    interpolation
        Interpolation kernel name.
    mapping_grid_step
        Coarse WCS mapping interval in output pixels.
    area_scaling
        Multiply samples by ``|d(source pixel)/d(destination pixel)|`` to
        preserve per-pixel flux under a change in pixel area.
    fill_value
        Image value assigned outside the valid source footprint.
    invalid_mask_value
        Mask bit pattern assigned outside the valid source footprint when a
        source mask is provided.
    lanczos_a
        Lanczos kernel radius.
    two_a_footprint
        Use the validated two-radius footprint convention.
    """

    mapping: MappingFn
    output_bbox: BBox
    source_origin: tuple[int, int] = (0, 0)
    interpolation: str = "lanczos3"
    mapping_grid_step: int = 100
    area_scaling: bool = True
    fill_value: float = float("nan")
    invalid_mask_value: int | bool = 1
    lanczos_a: int = 3
    two_a_footprint: bool = True


@dataclass(frozen=True, slots=True)
class StackReprojectionSpec:
    """Shared-grid reprojection description for multiple source images.

    Attributes
    ----------
    grid
        Shared sky grid.
    output_bbox
        Common destination bounding box.
    members
        Per-source reprojection specifications in source order.
    """

    grid: Grid
    output_bbox: BBox
    members: tuple[ReprojectionSpec, ...]


@dataclass(slots=True)
class DeviceReprojectionResult:
    """Reprojected image payload whose arrays may remain on an accelerator.

    Attributes
    ----------
    image, mask
        Backend-native ``(height, width)`` arrays. Without a source mask,
        ``mask`` is boolean and true outside the valid footprint. With a
        source mask, it is the neighborhood bitwise OR and uses the configured
        invalid value outside the footprint.
    bbox
        Destination bounding box.
    backend, interpolation
        Resolved execution backend and interpolation kernel.
    metadata
        Additional workflow metadata.
    """

    image: Any
    mask: Any | None
    bbox: BBox
    backend: str
    interpolation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReprojectionResult:
    """Host-resident reprojected image payload.

    Attributes
    ----------
    image, mask
        NumPy ``(height, width)`` arrays. Without a source mask, ``mask`` is
        boolean and true outside the valid footprint. With a source mask, it
        is the neighborhood bitwise OR and uses the configured invalid value.
    bbox
        Destination bounding box.
    backend, interpolation
        Resolved execution backend and interpolation kernel.
    metadata
        Additional workflow metadata.
    """

    image: np.ndarray
    mask: np.ndarray | None
    bbox: BBox
    backend: str
    interpolation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaskedReprojectionResult:
    """Host-resident reprojected image, variance, and mask planes.

    Attributes
    ----------
    image, variance, mask
        NumPy ``(height, width)`` arrays. Variance propagation assumes
        independent source pixels and therefore omits interpolation-induced
        covariance. Mask bits are the exact bitwise OR of the contributing
        mask neighborhood.
    bbox
        Destination bounding box.
    backend, interpolation
        Resolved execution backend and interpolation kernel.
    metadata
        Additional workflow metadata, including the variance approximation.
    """

    image: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    bbox: BBox
    backend: str
    interpolation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StackReprojectionResult:
    """Host-resident result for a shared-grid stack reproject.

    Attributes
    ----------
    images, masks
        Arrays with shape ``(sources, height, width)`` in source order.
    spec
        Shared-grid specification used for the operation.
    results
        Individual per-source results.
    backend
        Resolved execution backend.
    metadata
        Additional workflow metadata.
    """

    images: np.ndarray
    masks: np.ndarray | None
    spec: StackReprojectionSpec
    results: tuple[ReprojectionResult, ...]
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


def make_north_up_wcs(
    crval: tuple[float, float] | list[float] | np.ndarray,
    shape: tuple[int, int],
    pixel_scale_arcsec: float,
    crpix: tuple[float, float] | None = None,
) -> WCS:
    """Build a north-up gnomonic world-coordinate system.

    Parameters
    ----------
    crval
        ``(right ascension, declination)`` reference in degrees.
    shape
        Image ``(height, width)`` in pixels.
    pixel_scale_arcsec
        Pixel scale in arcseconds per pixel.
    crpix
        Optional FITS one-based reference pixel.

    Returns
    -------
    astropy.wcs.WCS
        Two-dimensional TAN WCS with right ascension increasing leftward.
    """

    height, width = shape
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = (
        list(crpix)
        if crpix is not None
        else [width / 2 + 0.5, height / 2 + 0.5]
    )
    wcs.wcs.crval = [float(crval[0]), float(crval[1])]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cd = np.array(
        [
            [-pixel_scale_deg, 0.0],
            [0.0, pixel_scale_deg],
        ]
    )
    wcs.wcs.set()
    return wcs


def bbox_wcs(grid: Grid, bbox: BBox) -> WCS:
    """Build a local WCS for a bounding box on a shared grid.

    Parameters
    ----------
    grid
        Parent shared grid.
    bbox
        Local bounding box.

    Returns
    -------
    astropy.wcs.WCS
        WCS whose local pixel coordinates align with the shared grid.
    """

    return make_north_up_wcs(
        grid.crval,
        shape=bbox.shape,
        pixel_scale_arcsec=grid.pixel_scale_arcsec,
        crpix=(1.0 - bbox.min_x, 1.0 - bbox.min_y),
    )


def bbox_union(boxes: list[BBox] | tuple[BBox, ...]) -> BBox:
    """Return the minimal bounding box covering every input.

    Parameters
    ----------
    boxes
        Non-empty sequence of bounding boxes.

    Returns
    -------
    BBox
        Minimal inclusive union.
    """

    if not boxes:
        raise ValueError("bbox_union requires at least one bbox")
    min_x = min(item.min_x for item in boxes)
    min_y = min(item.min_y for item in boxes)
    max_x = max(item.min_x + item.width - 1 for item in boxes)
    max_y = max(item.min_y + item.height - 1 for item in boxes)
    return BBox(
        min_x=min_x,
        min_y=min_y,
        width=max_x - min_x + 1,
        height=max_y - min_y + 1,
    )
