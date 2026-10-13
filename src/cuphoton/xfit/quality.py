# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Quality flags for Gaussian dipole fits.

A valid covariance does not make a fit usable. These flags catch the
failures it does not exclude: widths that have run away from any point
source, lobes outside the stamp, a fit with no lobe at the candidate,
overlapping lobes whose amplitude and separation trade against each other,
and a lobe the data do not support (a single source, brightening or
fading, fitted with a two-lobe model).
"""

from __future__ import annotations

import warnings
from enum import IntFlag
from typing import Any

import numpy as np

from ._types import ArrayLike
from .api import DeviceDipoleFitResult, DipoleFitResult
from .io import _array_module
from .models import GaussianDipoleModel

_FAR = 1.0e6
_GRID_STEP = 6.0
_MIN_NULL_POSITIONS = 12
_CHUNK_ELEMENTS = 1 << 25


class DipoleQualityFlag(IntFlag):
    """Bit flags returned by :func:`dipole_quality_flags`."""

    UNCERTAINTY_INVALID = 1
    WIDTH_OUT_OF_RANGE = 2
    LOBE_OUTSIDE_STAMP = 4
    NO_LOBE_AT_CANDIDATE = 8
    LOBES_OVERLAP = 16
    NEGATIVE_LOBE_UNSUPPORTED = 32
    POSITIVE_LOBE_UNSUPPORTED = 64


def dipole_quality_flags(
    result: DipoleFitResult | DeviceDipoleFitResult,
    images: ArrayLike,
    *,
    variance: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    width_range: tuple[float, float] = (0.5, 5.0),
    center_radius: float = 5.0,
    overlap_separation: float = 2.0,
    lobe_snr: float = 3.0,
    calibrate_noise: bool = True,
) -> Any:
    """Return one integer of :class:`DipoleQualityFlag` bits per fit.

    ``result`` is a Gaussian-model fit of ``images`` in difference mode, with
    the stamp center at the candidate. Coordinates are xFit's centered pixels.

    - ``UNCERTAINTY_INVALID``: xFit returned no valid covariance.
    - ``WIDTH_OUT_OF_RANGE``: a fitted width lies outside
      ``width_range`` pixels. Set the range from the image's point spread
      function, for example 0.5 to 2.5 times its standard deviation.
    - ``LOBE_OUTSIDE_STAMP``: a lobe center lies outside the stamp.
    - ``NO_LOBE_AT_CANDIDATE``: neither lobe lies within ``center_radius``
      pixels of the stamp center, where the detection placed the
      candidate. A fit that has left the source for features elsewhere in
      the stamp sets it.
    - ``LOBES_OVERLAP``: the lobes are closer than ``overlap_separation``
      pixels; only the dipole center and the product of amplitude and
      separation are measured.
    - ``NEGATIVE_LOBE_UNSUPPORTED`` and ``POSITIVE_LOBE_UNSUPPORTED``: for
      lobes that do not overlap, the data do not support the lobe with
      negative or positive flux. Both lobe shapes are fitted to the image
      by weighted linear least squares with free amplitudes; a flag is set
      when its lobe's amplitude, with the lobe's sign, is below
      ``lobe_snr`` standard errors. A candidate with exactly one of the two
      is a single source: brightening when the negative lobe is
      unsupported, fading when the positive lobe is.

    Difference images carry noise that is correlated between neighboring
    pixels, so a lobe-sized sum is noisier than independent pixels predict.
    With ``calibrate_noise``, each stamp's lobe significances are
    divided by the spread of the same statistic at a grid of empty
    positions in that stamp, at least three lobe widths from either lobe.
    The grid has a 6-pixel step along each axis. A stamp whose grid keeps
    fewer than 12 such positions is left uncalibrated; when the stamp size
    alone rules calibration out (below about 33 pixels on a side), a
    warning is issued.

    Without ``variance``, each stamp's noise is its median absolute deviation
    times 1.4826 over the unmasked pixels.

    Computation follows the backend of ``images``: CuPy images, with a
    :class:`DeviceDipoleFitResult` or any result whose arrays CuPy accepts,
    return a CuPy array of flags.
    """

    with (
        np.errstate(invalid="ignore", divide="ignore", over="ignore"),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        xp = _array_module(images)
        data = xp.asarray(images, dtype=xp.float64)
        if data.ndim != 3:
            raise ValueError("images must have shape (batch, height, width)")
        p = xp.asarray(result.parameters, dtype=xp.float64)
        batch, height, width = data.shape
        if p.shape != (batch, 8):
            raise ValueError(
                "result must hold one Gaussian dipole fit per image"
            )
        good = xp.isfinite(data)
        if mask is not None:
            good &= xp.asarray(mask, dtype=bool)
        if variance is None:
            masked = xp.where(good, data, xp.nan)
            center = xp.nanmedian(masked, axis=(1, 2), keepdims=True)
            sigma = 1.4826 * xp.nanmedian(
                xp.abs(masked - center), axis=(1, 2)
            )
            weight = xp.where(good, 1.0 / sigma[:, None, None] ** 2, 0.0)
        else:
            var = xp.asarray(variance, dtype=xp.float64)
            good &= xp.isfinite(var) & (var > 0)
            weight = xp.where(good, 1.0 / xp.where(good, var, 1.0), 0.0)
        image = xp.where(good, data, 0.0)

        amplitude, sx, sy, theta = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        first = (amplitude > 0)[:, None]
        source = xp.where(first, p[:, 4:6], p[:, 6:8])
        other = xp.where(first, p[:, 6:8], p[:, 4:6])
        half_x, half_y = (width - 1) / 2, (height - 1) / 2
        separation = xp.hypot(*(source - other).T)

        flags = xp.zeros(batch, dtype=xp.int32)

        def mark(condition, flag):
            nonlocal flags
            flags = flags | xp.where(condition, int(flag), 0).astype(xp.int32)

        mark(
            ~xp.asarray(result.uncertainty_valid, dtype=bool),
            DipoleQualityFlag.UNCERTAINTY_INVALID,
        )
        lo, hi = width_range
        mark(
            ~((sx >= lo) & (sx <= hi) & (sy >= lo) & (sy <= hi)),
            DipoleQualityFlag.WIDTH_OUT_OF_RANGE,
        )
        mark(
            (xp.abs(p[:, 4::2]) > half_x).any(axis=1)
            | (xp.abs(p[:, 5::2]) > half_y).any(axis=1),
            DipoleQualityFlag.LOBE_OUTSIDE_STAMP,
        )
        nearest = xp.minimum(xp.hypot(*source.T), xp.hypot(*other.T))
        mark(
            ~(nearest <= center_radius),
            DipoleQualityFlag.NO_LOBE_AT_CANDIDATE,
        )
        overlap = separation < overlap_separation
        mark(overlap, DipoleQualityFlag.LOBES_OVERLAP)

        model = GaussianDipoleModel(
            (height, width), backend="numpy" if xp is np else "cupy"
        )

        def lobes(shape, sx, sy, theta, x, y):
            q = xp.stack(
                [xp.ones(shape), sx, sy, theta, x, y]
                + [xp.full(shape, _FAR)] * 2,
                axis=-1,
            )
            q = xp.where(xp.isfinite(q), q, _FAR).reshape(-1, 8)
            # unchecked: the checked path copies parameters to the host
            return model._evaluate_positive_unchecked(q).reshape(
                *shape, height, width
            )

        g1 = lobes((batch,), sx, sy, theta, *source.T)
        g2 = lobes((batch,), sx, sy, theta, *other.T)
        snr1 = _joint_snr(image, weight, g2, g1)
        snr2 = -_joint_snr(image, weight, g1, g2)
        grid_x = xp.arange(-half_x + 4, half_x - 3, _GRID_STEP)
        grid_y = xp.arange(-half_y + 4, half_y - 3, _GRID_STEP)
        count = len(grid_x) * len(grid_y)
        if calibrate_noise and count < _MIN_NULL_POSITIONS:
            warnings.warn(
                f"{height}x{width} stamps hold fewer than "
                f"{_MIN_NULL_POSITIONS} calibration positions; lobe "
                "significance is not calibrated for correlated noise",
                stacklevel=2,
            )
        elif calibrate_noise:
            gx, gy = (a.ravel() for a in xp.meshgrid(grid_x, grid_y))
            chunk = max(1, _CHUNK_ELEMENTS // (count * height * width))
            scale = xp.ones(batch)
            for s in range(0, batch, chunk):
                e = min(s + chunk, batch)
                n = e - s
                keep = 3.0 * xp.maximum(xp.maximum(sx[s:e], sy[s:e]), 2.0)
                far = (
                    xp.hypot(gx - source[s:e, :1], gy - source[s:e, 1:])
                    > keep[:, None]
                ) & (
                    xp.hypot(gx - other[s:e, :1], gy - other[s:e, 1:])
                    > keep[:, None]
                )
                rep = (n, count)
                null = _joint_snr(
                    image[s:e, None],
                    weight[s:e, None],
                    g1[s:e, None],
                    lobes(
                        rep,
                        xp.broadcast_to(sx[s:e, None], rep),
                        xp.broadcast_to(sy[s:e, None], rep),
                        xp.broadcast_to(theta[s:e, None], rep),
                        xp.broadcast_to(gx, rep),
                        xp.broadcast_to(gy, rep),
                    ),
                )
                null = xp.where(far & xp.isfinite(null), null, xp.nan)
                middle = xp.nanmedian(null, axis=1, keepdims=True)
                spread = 1.4826 * xp.nanmedian(xp.abs(null - middle), axis=1)
                enough = (~xp.isnan(null)).sum(axis=1) >= _MIN_NULL_POSITIONS
                scale[s:e] = xp.where(enough, xp.maximum(spread, 1.0), 1.0)
            snr1, snr2 = snr1 / scale, snr2 / scale
        mark(
            ~overlap & ~(snr2 >= lobe_snr),
            DipoleQualityFlag.NEGATIVE_LOBE_UNSUPPORTED,
        )
        mark(
            ~overlap & ~(snr1 >= lobe_snr),
            DipoleQualityFlag.POSITIVE_LOBE_UNSUPPORTED,
        )
        return flags


def _joint_snr(image, weight, g1, g2):
    """Signed amplitude over standard error of lobe ``g2``, fitted jointly
    with lobe ``g1`` by weighted linear least squares (batched over the
    leading axis of ``g2``)."""

    axes = (-2, -1)
    a11 = (weight * g1 * g1).sum(axis=axes)
    a22 = (weight * g2 * g2).sum(axis=axes)
    a12 = (weight * g1 * g2).sum(axis=axes)
    b1 = (weight * g1 * image).sum(axis=axes)
    b2 = (weight * g2 * image).sum(axis=axes)
    det = a11 * a22 - a12 * a12
    return (a11 * b2 - a12 * b1) / det / np.sqrt(a11 / det)
