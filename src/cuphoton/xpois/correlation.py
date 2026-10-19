# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Correlated noise in weighted sums of residual pixels.

The marginal variance ``target_variance + reference_variance * kernel**2``
of :func:`standardize_constant_kernel_residual` treats the resampled
reference pixels as independent. They are not: the interpolation that put
the reference on the target grid correlates neighbors, and the kernel
correlates them further. A statistic that sums residual pixels with weights,
such as a PSF flux, a position or an aperture, therefore has a variance the
diagonal plane under-predicts. This module gives the ratio in closed form
from the resampling weights and the kernel, and measures it on an image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .kernel_uncertainty import _array_module, _fftconvolve, _real_2d_array

__all__ = [
    "CorrelationFactorResult",
    "correlation_factor",
    "empirical_correlation_factor",
    "psf_weights",
    "resampling_variance_factor",
]

_CHUNK = 512


@dataclass(frozen=True)
class CorrelationFactorResult:
    """Correlation factors of weighted sums of residual pixels.

    Attributes
    ----------
    factor
        Per position, the variance of the weighted sum over the diagonal
        plane's prediction ``sum(weights**2 * residual_variance)``. ``NaN``
        where the window is not finite.
    pixel_ratio
        Per position, the variance of one residual pixel over the diagonal
        plane's value at that pixel: the resampling factor of the reference
        term, mixed with the target term it does not touch.
    covariance_scope
        The covariance modeled: correlation of the reference term through
        the resampling weights and the kernel.
    target_pixels_independent
        Always true: target pixels are taken as independent, so the target
        term is ``sum(weights**2 * target_variance)``.
    """

    factor: Any
    pixel_ratio: Any
    covariance_scope: Literal["reference_resampling_and_kernel"] = field(
        default="reference_resampling_and_kernel", init=False
    )
    target_pixels_independent: Literal[True] = field(default=True, init=False)


def psf_weights(
    psf_sigma: float,
    size: int,
    *,
    derivative: Literal["x", "y"] | None = None,
) -> np.ndarray:
    """Weights of a PSF-weighted sum on an odd square stamp.

    The point spread function is a unit-flux, point-sampled circular
    Gaussian centered on the stamp's center pixel. With ``derivative`` the
    weights are its derivative with respect to the source position along
    that axis, the weights that measure a position.

    Parameters
    ----------
    psf_sigma
        Gaussian standard deviation in pixels.
    size
        Positive odd side length.
    derivative
        ``None``, ``"x"`` or ``"y"``.

    Returns
    -------
    numpy.ndarray
        ``(size, size)`` weights.
    """

    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    if psf_sigma <= 0:
        raise ValueError("psf_sigma must be positive")
    u = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-0.5 * (u / psf_sigma) ** 2) / (
        np.sqrt(2.0 * np.pi) * psf_sigma
    )
    dg = g * u / psf_sigma**2
    if derivative is None:
        return g[:, None] * g[None, :]
    if derivative == "x":
        return g[:, None] * dg[None, :]
    if derivative == "y":
        return dg[:, None] * g[None, :]
    raise ValueError("derivative must be None, 'x' or 'y'")


def resampling_variance_factor(
    kernel: Any,
    resampling_weights: Any,
    *,
    is_translation: Any | None = None,
) -> Any:
    """Variance factor of one residual pixel's reference term.

    The reference term of a residual pixel is the kernel applied to the
    resampled reference. Its true variance per unit source variance is the
    squared footprint of the kernel and the resampling weights on the
    source pixels, ``sum((kernel * weights)**2)`` with ``*`` a convolution;
    the diagonal plane gives ``sum(kernel**2) * sum(weights**2)``. The
    factor is their ratio, 1 for independent reference pixels.

    Parameters
    ----------
    kernel
        Two-dimensional fitted kernel.
    resampling_weights
        ``(N, k, k)`` interpolation weights that produced the resampled
        reference pixel at each position from its source neighbors. The
        stamp's orientation does not change the result.
    is_translation
        Optional ``(N,)`` booleans, true where the mapping between the grids
        is locally a translation so one stamp describes the whole kernel
        footprint. Only the translation case is implemented.

    Returns
    -------
    ndarray
        ``(N,)`` factors.
    """

    xp = _array_module(resampling_weights)
    kernel_arr = _real_2d_array(xp, "kernel", kernel)
    stamps = _resampling_stamps(xp, resampling_weights, None)
    _require_translation(xp, is_translation, stamps.shape[0])
    true, diag = _pixel_terms(xp, kernel_arr, stamps)
    return true / diag


def correlation_factor(
    weights: Any,
    *,
    kernel: Any,
    target_variance: Any,
    reference_source_variance: Any,
    resampling_weights: Any,
    residual_variance: Any | None = None,
    is_translation: Any | None = None,
) -> CorrelationFactorResult:
    """Correlation factor of weighted sums of residual pixels.

    For weights ``w`` on a window of residual pixels the factor is

    ``(sum(w**2 V_target) + Var(sum_p w_p (K * R)_p)) / sum(w**2 V_D)``,

    ``V_D`` the diagonal residual variance, ``K`` the kernel and ``R`` the
    resampled reference. The reference part is the source variance, the
    ``w**2``-weighted mean of ``reference_source_variance`` over the
    window, times the squared footprint of ``w`` on the source pixels
    through the kernel and the resampling weights. Target pixels are taken
    as independent.

    Every input is a stack of stamps on the same windows, centered on the
    positions the caller chose; the stamps' side ``s`` sets the window.

    Parameters
    ----------
    weights
        ``(N, s, s)`` weights of the sum at each position: PSF weights for a
        flux, their derivative for a position, a top hat for an aperture.
    kernel
        Two-dimensional fitted kernel.
    target_variance
        ``(N, s, s)`` stamps of the target variance plane, or an array that
        broadcasts to them.
    reference_source_variance
        ``(N, s, s)`` stamps of the reference's source-pixel variance on the
        target grid (the resampled reference variance divided by the
        resampling gain ``sum(resampling_weights**2)``), or an array that
        broadcasts to them.
    resampling_weights
        ``(N, k, k)`` interpolation weights at each position, or one
        ``(k, k)`` stamp for every position.
    residual_variance
        Optional ``(N, s, s)`` stamps of the diagonal residual variance
        plane the factor is measured against, such as
        ``marginal_residual_variance`` of
        :func:`standardize_constant_kernel_residual`. By default it is
        ``target_variance + reference_source_variance * sum(weights**2) *
        sum(kernel**2)``, the same plane under a flat source variance.
    is_translation
        Optional ``(N,)`` booleans, true where the mapping between the grids
        is locally a translation. Only the translation case is implemented;
        a false entry raises ``NotImplementedError``.

    Returns
    -------
    CorrelationFactorResult
        ``factor`` and ``pixel_ratio`` per position, ``NaN`` where any stamp
        of the window is not finite, with the covariance scope recorded.
    """

    xp = _array_module(weights)
    w = xp.asarray(weights, dtype=xp.float64)
    if w.ndim != 3:
        raise ValueError("weights must have shape (N, s, s)")
    count = int(w.shape[0])
    kernel_arr = _real_2d_array(xp, "kernel", kernel)
    stamps = _resampling_stamps(xp, resampling_weights, count)
    _require_translation(xp, is_translation, count)
    target = xp.broadcast_to(
        xp.asarray(target_variance, dtype=xp.float64), w.shape
    )
    source = xp.broadcast_to(
        xp.asarray(reference_source_variance, dtype=xp.float64), w.shape
    )
    true, diag = _pixel_terms(xp, kernel_arr, stamps)
    if residual_variance is None:
        residual = target + source * diag[:, None, None]
    else:
        residual = xp.broadcast_to(
            xp.asarray(residual_variance, dtype=xp.float64), w.shape
        )

    w2 = w**2
    w2_sum = w2.sum(axis=(1, 2))
    target_sum = (w2 * target).sum(axis=(1, 2))
    source_variance = (w2 * source).sum(axis=(1, 2)) / w2_sum
    diagonal_sum = (w2 * residual).sum(axis=(1, 2))
    power = xp.empty(count, dtype=xp.float64)
    flipped = kernel_arr[::-1, ::-1][None]
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        response = _fftconvolve(
            w[start:stop], flipped, mode="full", axes=(1, 2)
        )
        footprint = _fftconvolve(
            response, stamps[start:stop], mode="full", axes=(1, 2)
        )
        power[start:stop] = (footprint**2).sum(axis=(1, 2))

    finite = (
        xp.isfinite(w).all(axis=(1, 2))
        & xp.isfinite(target).all(axis=(1, 2))
        & xp.isfinite(source).all(axis=(1, 2))
        & xp.isfinite(residual).all(axis=(1, 2))
        & (diagonal_sum > 0)
        & (w2_sum > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (target_sum + source_variance * power) / diagonal_sum
        pixel_target = target_sum / w2_sum
        pixel_ratio = (pixel_target + source_variance * true) / (
            pixel_target + source_variance * diag
        )
    return CorrelationFactorResult(
        factor=xp.where(finite, factor, xp.nan),
        pixel_ratio=xp.where(finite, pixel_ratio, xp.nan),
    )


def empirical_correlation_factor(
    residual: Any,
    residual_variance: Any,
    weights: Any,
    positions: Any,
) -> Any:
    """Measured correlation factor of weighted sums at empty positions.

    At each position the weighted sum of the residual window is divided by
    the square root of its diagonal prediction
    ``sum(weights**2 * residual_variance)``. The factor is the squared
    robust spread (1.4826 times the median absolute deviation) of those
    standardized sums over the positions, which is 1 when the diagonal
    plane is right. It checks the closed form of :func:`correlation_factor`
    on any image, and tells whether a variance plane already carries the
    correlation.

    Parameters
    ----------
    residual
        Two-dimensional residual image.
    residual_variance
        Two-dimensional diagonal variance plane of the residual.
    weights
        ``(s, s)`` weights applied at every position, or ``(q, s, s)`` for
        several weightings at once, ``s`` odd.
    positions
        ``(N, 2)`` array of ``(x, y)`` pixel coordinates of positions where
        the residual is pure noise; windows are centered on the nearest
        integer pixel. Positions whose window leaves the image or holds a
        non-finite or non-positive value are dropped.

    Returns
    -------
    ndarray
        The factor, with shape ``()`` for one weighting or ``(q,)`` for
        several; ``NaN`` with fewer than two usable positions.
    """

    xp = _array_module(residual)
    residual_arr = _real_2d_array(xp, "residual", residual)
    variance_arr = _real_2d_array(xp, "residual_variance", residual_variance)
    if variance_arr.shape != residual_arr.shape:
        raise ValueError("residual_variance must match residual shape")
    w = xp.asarray(weights, dtype=xp.float64)
    single = w.ndim == 2
    if single:
        w = w[None]
    if w.ndim != 3 or w.shape[1] != w.shape[2] or w.shape[1] % 2 == 0:
        raise ValueError("weights must be odd square stamps")
    xy = xp.asarray(positions, dtype=xp.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")

    half = int(w.shape[-1]) // 2
    height, width = residual_arr.shape
    center_x = xp.rint(xy[:, 0]).astype(xp.int64)
    center_y = xp.rint(xy[:, 1]).astype(xp.int64)
    inside = (
        (center_x >= half)
        & (center_x < width - half)
        & (center_y >= half)
        & (center_y < height - half)
    )
    safe_x = xp.where(inside, center_x, half)
    safe_y = xp.where(inside, center_y, half)
    offsets = xp.arange(-half, half + 1)
    rows = safe_y[:, None, None] + offsets[None, :, None]
    columns = safe_x[:, None, None] + offsets[None, None, :]
    data = residual_arr[rows, columns]
    variance = variance_arr[rows, columns]
    usable = (
        inside
        & xp.isfinite(data).all(axis=(1, 2))
        & xp.isfinite(variance).all(axis=(1, 2))
        & (variance > 0).all(axis=(1, 2))
    )
    data, variance = data[usable], variance[usable]
    sums = xp.einsum("qyx,nyx->nq", w, data)
    predicted = xp.einsum("qyx,nyx->nq", w**2, variance)
    if data.shape[0] < 2:
        out = xp.full(w.shape[0], xp.nan)
    else:
        z = sums / xp.sqrt(predicted)
        spread = 1.4826 * xp.median(xp.abs(z - xp.median(z, axis=0)), axis=0)
        out = spread**2
    return out[0] if single else out


def _resampling_stamps(xp: Any, resampling_weights: Any, count: int | None):
    stamps = xp.asarray(resampling_weights, dtype=xp.float64)
    if stamps.ndim == 2 and count is not None:
        stamps = xp.broadcast_to(stamps, (count,) + stamps.shape)
    if stamps.ndim != 3 or (count is not None and stamps.shape[0] != count):
        raise ValueError("resampling_weights must have shape (N, k, k)")
    return stamps


def _require_translation(xp: Any, is_translation: Any, count: int) -> None:
    if is_translation is None:
        return
    flags = xp.asarray(is_translation, dtype=bool)
    if flags.shape != (count,):
        raise ValueError("is_translation must have shape (N,)")
    if not bool(flags.all()):
        raise NotImplementedError(
            "the correlation factor is implemented for a locally translated "
            "grid only; positions where the mapping rotates or scales the "
            "grid (is_translation false) are not supported"
        )


def _pixel_terms(xp: Any, kernel: Any, stamps: Any) -> tuple[Any, Any]:
    """True and diagonal reference-term variance of one residual pixel."""

    footprint = _fftconvolve(kernel[None], stamps, mode="full", axes=(1, 2))
    true = (footprint**2).sum(axis=(1, 2))
    diag = (kernel**2).sum() * (stamps**2).sum(axis=(1, 2))
    return true, diag
