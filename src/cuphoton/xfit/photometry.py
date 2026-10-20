# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Forced point-spread-function photometry on difference stamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._types import ArrayLike
from .io import _array_module


@dataclass(frozen=True)
class ForcedPhotometryResult:
    """Per-stamp flux, standard error and validity on the input backend."""

    flux: Any
    flux_error: Any
    valid: Any


def forced_psf_photometry(
    images: ArrayLike,
    variance: ArrayLike,
    positions: ArrayLike,
    psf_sigma: float | ArrayLike,
    *,
    mask: ArrayLike | None = None,
    psf: ArrayLike | None = None,
) -> ForcedPhotometryResult:
    """Fit the flux of a circular Gaussian PSF at fixed positions.

    ``images`` and ``variance`` have shape ``(batch, y, x)``. ``positions``
    has shape ``(batch, 2)`` as ``(x, y)`` in xFit's centered pixel
    coordinates, and ``psf_sigma`` is the PSF standard deviation in pixels,
    one value or one per stamp. The PSF is normalized to unit integral, so
    ``flux`` is the source's total flux, as ``2 pi |amplitude| sigma_x
    sigma_y`` is for a Gaussian dipole lobe. ``psf`` replaces the Gaussian
    with ``(batch, y, x)`` model stamps already evaluated at ``positions``
    (such as ``cuphoton.xpois.psf_weights`` with their offsets); the
    estimator and the validity rules are the same.

    The model is linear in the flux, so the weighted least-squares solution
    and its standard error are exact: ``flux = sum(w P D) / sum(w P^2)`` and
    ``flux_error = 1 / sqrt(sum(w P^2))``, with ``w`` the inverse variance on
    usable pixels, ``P`` the PSF and ``D`` the image. A pixel is usable when
    ``mask`` (if given) is true and the image and a positive variance are
    finite there.

    ``flux_error`` is the uncertainty conditional on ``positions`` and
    ``psf_sigma`` being fixed and correct and on ``variance`` describing the
    image noise. A position or width measured on the same noisy image adds
    uncertainty that it does not include, and a wrong width biases the flux;
    correlated noise between pixels also makes it too small.

    Computation follows the backend of ``images``: CuPy images return CuPy
    arrays, NumPy images NumPy arrays.
    """

    xp = _array_module(images)
    data = xp.asarray(images, dtype=xp.float64)
    var = xp.asarray(variance, dtype=xp.float64)
    if data.ndim != 3 or var.shape != data.shape:
        raise ValueError(
            "images and variance must share shape (batch, height, width)"
        )
    batch, height, width = data.shape
    xy = xp.asarray(positions, dtype=xp.float64)
    if xy.shape != (batch, 2):
        raise ValueError("positions must have shape (batch, 2)")
    sigma = xp.broadcast_to(xp.asarray(psf_sigma, dtype=xp.float64), (batch,))
    good = xp.isfinite(data) & xp.isfinite(var) & (var > 0)
    if mask is not None:
        good &= xp.asarray(mask, dtype=bool)
    weight = xp.where(good, 1.0 / xp.where(good, var, 1.0), 0.0)

    positive = xp.isfinite(sigma) & (sigma > 0)
    if psf is None:
        x = xp.arange(width, dtype=xp.float64) - (width - 1) / 2
        y = xp.arange(height, dtype=xp.float64) - (height - 1) / 2
        s2 = xp.where(positive, sigma**2, 1.0)[:, None, None]
        r2 = (x[None, None, :] - xy[:, 0, None, None]) ** 2 + (
            y[None, :, None] - xy[:, 1, None, None]
        ) ** 2
        psf = xp.exp(-0.5 * r2 / s2) / (2.0 * xp.pi * s2)
    else:
        psf = xp.asarray(psf, dtype=xp.float64)
        if psf.shape != data.shape:
            raise ValueError("psf must share the shape of images")

    information = (weight * psf * psf).sum(axis=(1, 2))
    valid = (
        xp.isfinite(information)
        & (information > 0)
        & positive
        & xp.isfinite(xy).all(axis=1)
    )
    safe = xp.where(valid, information, 1.0)
    flux = (weight * psf * xp.where(good, data, 0.0)).sum(axis=(1, 2)) / safe
    return ForcedPhotometryResult(
        flux=xp.where(valid, flux, xp.nan),
        flux_error=xp.where(valid, 1.0 / xp.sqrt(safe), xp.nan),
        valid=valid,
    )
