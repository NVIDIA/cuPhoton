# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measurements on a subtraction that its consumers share.

:func:`kernel_centroid` gives the shift a fitted kernel applied to the
reference, :func:`empty_positions` picks pixels of a residual where nothing
sits, the positions an empirical noise measurement needs, and
:func:`forced_flux_at` measures forced PSF fluxes on a full plane at any
positions through the xFit estimator.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from cuphoton.xfit import ForcedPhotometryResult, forced_psf_photometry

from .correlation import psf_weights
from .stamps import _array_module, _ndimage_module, _to_host, window_stamps

__all__ = [
    "empty_positions",
    "forced_flux_at",
    "kernel_centroid",
]


def kernel_centroid(kernel: Any) -> tuple[float, float]:
    """First moment of a convolution kernel in pixels.

    The moment is taken about the kernel's center pixel, so for the kernel
    of a subtraction it is the shift the kernel applied to the reference.

    Parameters
    ----------
    kernel
        Two-dimensional kernel, NumPy or CuPy.

    Returns
    -------
    tuple of float
        ``(cx, cy)`` in pixels; ``NaN`` for a kernel whose sum is zero or
        not finite, which has no first moment. One host transfer of three
        values.
    """

    xp = _array_module(kernel)
    k = xp.asarray(kernel, dtype=xp.float64)
    if k.ndim != 2:
        raise ValueError("kernel must be a two-dimensional array")
    ny, nx = k.shape
    i = xp.arange(nx, dtype=xp.float64) - nx // 2
    j = xp.arange(ny, dtype=xp.float64) - ny // 2
    cx, cy, total = (
        float(v)
        for v in _to_host(
            xp.stack(
                [
                    (k.sum(axis=0) * i).sum(),
                    (k.sum(axis=1) * j).sum(),
                    k.sum(),
                ]
            )
        )
    )
    if not math.isfinite(total) or total == 0.0:
        return math.nan, math.nan
    return cx / total, cy / total


def empty_positions(
    valid: Any,
    *,
    exclude_xy: Any,
    window: int,
    count: int,
    loud: tuple[Any, float] | None = None,
) -> tuple[Any, int]:
    """Usable pixels of an image where nothing sits, taken evenly.

    A pixel is empty where ``valid`` is true and it lies more than
    ``window`` pixels (Chebyshev distance) from every excluded position
    and from every loud pixel: a window of side ``window`` centered on it
    does not overlap a window centered on any of those. The empty pixels
    are taken in raster order and ``count`` of them are picked at evenly
    spaced ranks, so they spread over the image without a random draw.

    Parameters
    ----------
    valid
        Two-dimensional boolean plane of usable pixels, NumPy or CuPy; the
        result follows its namespace.
    exclude_xy
        ``(M, 2)`` array of ``(x, y)`` pixel positions to keep away from,
        such as candidates and stars. Positions off the image or not finite
        exclude nothing.
    window
        Positive odd side of the window a measurement at a position covers.
    count
        Number of positions wanted. Fewer come back when the image holds
        fewer empty pixels, none when it holds none.
    loud
        Optional ``(residual, n_sigma)``: usable pixels of ``residual`` more
        than ``n_sigma`` robust sigmas (1.4826 times the median absolute
        deviation) from its median are loud, and their surroundings are
        not empty either. Nothing is loud when the spread is zero.

    Returns
    -------
    positions
        ``(K, 2)`` float64 array of ``(x, y)`` integer pixel positions,
        ``K <= count``, in the namespace of ``valid``.
    n_excluded_loud
        Number of usable pixels the loud rule removed that the excluded
        positions had not. The count and ``K`` cross to the host in one
        transfer of two values.
    """

    xp = _array_module(valid)
    usable = xp.asarray(valid, dtype=bool)
    if usable.ndim != 2:
        raise ValueError("valid must be a two-dimensional boolean plane")
    window = int(window)
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd side")
    if int(count) < 0:
        raise ValueError("count must not be negative")
    ny, nx = (int(n) for n in usable.shape)
    xy = xp.asarray(exclude_xy, dtype=xp.float64)
    if xy.size == 0:
        xy = xp.zeros((0, 2), dtype=xp.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("exclude_xy must have shape (M, 2)")
    filt = _ndimage_module(xp).maximum_filter
    reach = 2 * window + 1

    # the excluded pixels, scattered through a bincount so the device never
    # compresses them; a position outside the image goes to a spare bin
    near = xp.zeros((ny, nx), dtype=bool)
    if int(xy.shape[0]):
        finite = xp.isfinite(xy).all(axis=1)
        pixel = xp.rint(xp.where(finite[:, None], xy, -1.0)).astype(xp.int64)
        inside = (
            finite
            & (pixel[:, 0] >= 0)
            & (pixel[:, 0] < nx)
            & (pixel[:, 1] >= 0)
            & (pixel[:, 1] < ny)
        )
        flat = xp.where(inside, pixel[:, 1] * nx + pixel[:, 0], ny * nx)
        near = (
            xp.bincount(flat, minlength=ny * nx + 1)[: ny * nx] > 0
        ).reshape(ny, nx)
    empty = usable & ~(
        filt(near.astype(xp.uint8), size=reach, mode="constant") > 0
    )
    if loud is None:
        noisy = xp.zeros((ny, nx), dtype=bool)
    else:
        residual, n_sigma = loud
        noisy = (
            filt(
                _loud_pixels(xp, residual, usable, float(n_sigma)).astype(
                    xp.uint8
                ),
                size=reach,
                mode="constant",
            )
            > 0
        )
    # the empty pixels in raster order: the k-th one sits where the running
    # count first reaches k, so evenly spaced ranks pick evenly spaced
    # pixels without compressing the plane; the count is the one transfer
    rank = xp.cumsum((empty & ~noisy).ravel().astype(xp.int64))
    n_excluded_loud, total = (
        int(v) for v in _to_host(xp.stack([(empty & noisy).sum(), rank[-1]]))
    )
    picked = min(int(count), total)
    if picked == 0:
        return xp.zeros((0, 2), dtype=xp.float64), n_excluded_loud
    wanted = xp.rint(xp.linspace(1.0, float(total), picked))
    pick = xp.searchsorted(rank, wanted)
    positions = xp.stack([pick % nx, pick // nx], axis=1).astype(xp.float64)
    return positions, n_excluded_loud


def forced_flux_at(
    plane: Any,
    positions: Any,
    psf_sigma: float,
    *,
    window: int,
    variance: Any | None = None,
    local_sky: bool = True,
) -> ForcedPhotometryResult:
    """Forced PSF fluxes of a full plane at positions.

    Windows of side ``window`` are cut around the positions with
    :func:`window_stamps` and handed to
    :func:`cuphoton.xfit.forced_psf_photometry`, the weighted least-squares
    flux of a unit-flux circular Gaussian ``P`` centered on the sub-pixel
    position: ``flux = sum(w P D) / sum(w P^2)`` and
    ``flux_error = 1 / sqrt(sum(w P^2))`` over the usable pixels, ``w`` the
    inverse variance. The Gaussian is :func:`psf_weights` at the position's
    offset from its window's center pixel.

    Parameters
    ----------
    plane
        Two-dimensional image, NumPy or CuPy; the result follows its
        namespace. Non-finite pixels do not contribute.
    positions
        ``(N, 2)`` array of ``(x, y)`` pixel coordinates. A position that is
        not finite gets ``NaN``.
    psf_sigma
        Gaussian standard deviation in pixels.
    window
        Positive odd side of the window around each position.
    variance
        Optional variance plane of ``plane``; pixels where it is not finite
        or not positive do not contribute. Without it every finite pixel
        has unit weight and ``flux_error`` is not a noise estimate.
    local_sky
        Subtract each window's median before the fit, a local sky level for
        a plane that holds sources (the kernel-matched reference, say). The
        residual of a subtraction is measured with ``False``.

    Returns
    -------
    ForcedPhotometryResult
        ``flux``, ``flux_error`` and ``valid`` per position; ``NaN`` where
        no pixel is usable.
    """

    xp = _array_module(plane, variance)
    xy = xp.asarray(positions, dtype=xp.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    # a position that is not finite is cut at the origin and ends as NaN
    # through its offset, without an invalid cast on the way
    at = xp.where(xp.isfinite(xy), xy, 0.0)
    data = window_stamps(plane, at, window)
    if variance is None:
        var = xp.ones_like(data)
    else:
        var = window_stamps(variance, at, window)
    if local_sky:
        usable = xp.isfinite(data) & xp.isfinite(var) & (var > 0)
        with warnings.catch_warnings():
            # a window with no finite pixel has no median and stays NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            sky = xp.nanmedian(
                xp.where(usable, data, xp.nan), axis=(1, 2), keepdims=True
            )
            data = data - sky
    offset = xy - xp.rint(xy)
    psf = psf_weights(psf_sigma, int(window), offset=offset, xp=xp)
    return forced_psf_photometry(data, var, offset, psf_sigma, psf=psf)


def _loud_pixels(xp: Any, residual: Any, valid: Any, n_sigma: float) -> Any:
    """Usable pixels more than ``n_sigma`` robust sigmas from the median;
    nothing when the spread is zero or no pixel is usable. Stays on the
    device: no compress, no scalar transfer.
    """

    values = xp.where(valid, xp.asarray(residual, dtype=xp.float64), xp.nan)
    with warnings.catch_warnings():
        # an all-NaN plane (no usable pixel) gives NaN and marks nothing
        warnings.simplefilter("ignore", RuntimeWarning)
        center = xp.nanmedian(values)
        sigma = 1.4826 * xp.nanmedian(xp.abs(values - center))
    return valid & (xp.abs(values - center) > n_sigma * sigma) & (sigma > 0)
