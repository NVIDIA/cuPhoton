# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-position circular aperture photometry on batches of image stamps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any

from cuphoton.core.arrays import array_module

from ._types import ArrayLike


class CircularApertureFlag(IntFlag):
    """Flags for one aperture. Zero denotes complete usable coverage."""

    MASKED = 1
    TRUNCATED = 2
    NO_COVERAGE = 4
    INVALID_GEOMETRY = 8


@dataclass(frozen=True)
class CircularApertureResult:
    """Fluxes, formal standard errors, pixel counts and combined flags.

    Each array has shape ``(batch, number_of_radii)`` on the image backend.
    ``in_bounds_pixels`` counts selected pixel centers within the stamp.
    ``used_pixels`` counts the subset passing the mask and validity checks.
    """

    flux: Any
    flux_error: Any
    used_pixels: Any
    in_bounds_pixels: Any
    flags: Any


def forced_circular_aperture_photometry(
    images: ArrayLike,
    variance: ArrayLike,
    positions: ArrayLike,
    radii: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> CircularApertureResult:
    """Sum pixels within fixed circular apertures at supplied stamp positions.

    ``images`` and ``variance`` have shape ``(batch, height, width)``.
    ``positions`` gives one ``(x, y)`` per stamp in xFit centered pixel
    coordinates. Positive finite ``radii`` in pixels may be a scalar or a
    nonempty vector, shared across the batch and returned in input order.
    A pixel contributes its full value when its squared center distance is
    strictly less than the squared radius in float64, its image and
    positive variance are finite, and ``mask``
    (if supplied, broadcastable to the image shape) is true there.

    Flux is the signed sum of the contributing pixels, and its formal error
    is the square root of their summed variances, assuming independent noise.
    Subtract any sky background before measuring. Partial apertures retain
    their measured sums: MASKED indicates rejected pixels within the aperture
    and TRUNCATED indicates the circle extends past the stamp's outer pixel
    edges. An aperture with no usable pixels returns NaN flux and error and
    NO_COVERAGE. Nonfinite positions and nonpositive or nonfinite radii also
    set INVALID_GEOMETRY for the affected measurements.

    NumPy images return NumPy arrays; CuPy images keep computation and
    results on the device.
    """
    xp = array_module(images)
    data = xp.asarray(images)
    var = xp.asarray(variance)
    if data.ndim != 3 or var.shape != data.shape:
        raise ValueError(
            "images and variance must share shape (batch, height, width)"
        )
    batch, height, width = data.shape
    if height == 0 or width == 0:
        raise ValueError("stamps must have positive height and width")
    xy = xp.asarray(positions, dtype=xp.float64)
    if xy.shape != (batch, 2):
        raise ValueError("positions must have shape (batch, 2)")
    radius_values = xp.atleast_1d(xp.asarray(radii, dtype=xp.float64))
    if radius_values.ndim != 1 or radius_values.size == 0:
        raise ValueError("radii must be a scalar or a nonempty vector")

    good = xp.isfinite(data) & xp.isfinite(var) & (var > 0)
    if mask is not None:
        good &= xp.asarray(mask, dtype=bool)
    x = xp.arange(width, dtype=xp.float64) - (width - 1) / 2
    y = xp.arange(height, dtype=xp.float64) - (height - 1) / 2
    dx = x[None, None, :] - xy[:, 0, None, None]
    dy = y[None, :, None] - xy[:, 1, None, None]
    distance_squared = dx * dx + dy * dy
    edge_distance = xp.minimum(
        width / 2 - xp.abs(xy[:, 0]), height / 2 - xp.abs(xy[:, 1])
    )
    finite_position = xp.isfinite(xy).all(axis=1)
    fluxes, errors, used_counts, in_bounds_counts, flags = [], [], [], [], []
    # A radius at a time bounds working memory independently of radius count.
    for radius in radius_values:
        geometry_ok = finite_position & xp.isfinite(radius) & (radius > 0)
        inside = (distance_squared < radius * radius) & geometry_ok[
            :, None, None
        ]
        usable = inside & good
        axes = (1, 2)
        n_inside = inside.sum(axis=axes)
        n_used = usable.sum(axis=axes)
        flux = xp.where(usable, data, 0).sum(axis=axes, dtype=xp.float64)
        noise = xp.where(usable, var, 0).sum(axis=axes, dtype=xp.float64)
        flag = (
            (n_used < n_inside) * int(CircularApertureFlag.MASKED)
            | (geometry_ok & (radius > edge_distance))
            * int(CircularApertureFlag.TRUNCATED)
            | (n_used == 0) * int(CircularApertureFlag.NO_COVERAGE)
            | (~geometry_ok) * int(CircularApertureFlag.INVALID_GEOMETRY)
        )
        fluxes.append(xp.where(n_used > 0, flux, xp.nan))
        errors.append(xp.where(n_used > 0, xp.sqrt(noise), xp.nan))
        used_counts.append(n_used)
        in_bounds_counts.append(n_inside)
        flags.append(flag.astype(xp.uint8))
    return CircularApertureResult(
        flux=xp.stack(fluxes, axis=1),
        flux_error=xp.stack(errors, axis=1),
        used_pixels=xp.stack(used_counts, axis=1),
        in_bounds_pixels=xp.stack(in_bounds_counts, axis=1),
        flags=xp.stack(flags, axis=1),
    )
