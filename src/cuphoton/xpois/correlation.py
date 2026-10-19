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
diagonal plane under-predicts.

:func:`resampling_variance_factor` gives the ratio for one residual pixel,
:func:`correlation_factor` for any weighted sum in closed form from the
resampling weights and the kernel, :func:`empirical_correlation_factor`
measures it on an image, and :func:`psf_weights` builds the weights of a
PSF flux or a PSF position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .kernel_uncertainty import _fftconvolve, _real_2d_array
from .stamps import _array_module, window_stamps

__all__ = [
    "CorrelationFactorResult",
    "EmpiricalCorrelationFactor",
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
        where the window is not finite, where the resampling stamp is all
        zero, or where ``is_translation`` is false.
    pixel_ratio
        Per position, the variance of one residual pixel over the diagonal
        plane's value, both as ``weights**2``-weighted means over the
        window: the resampling factor of the reference term, mixed with the
        target term it does not touch. ``NaN`` where ``factor`` is.
    n_rotated
        Number of positions whose ``is_translation`` flag was false; their
        factors are ``NaN``.
    covariance_scope
        The covariance modeled: correlation of the reference term through
        the resampling weights and the kernel.
    target_pixels_independent
        Always true: target pixels are taken as independent, so the target
        term is ``sum(weights**2 * target_variance)``.
    """

    factor: Any
    pixel_ratio: Any
    n_rotated: int = 0
    covariance_scope: Literal["reference_resampling_and_kernel"] = field(
        default="reference_resampling_and_kernel", init=False
    )
    target_pixels_independent: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True)
class EmpiricalCorrelationFactor:
    """Correlation factor measured at empty positions of an image.

    Attributes
    ----------
    factor
        Squared robust spread (1.4826 times the median absolute deviation)
        of the standardized weighted sums, shape ``()`` for one weighting
        or ``(q,)`` for several; ``NaN`` with fewer than two usable
        positions.
    positions_used
        Number of positions whose window was inside the image, finite and
        of positive variance.
    plain_ratio
        Sample variance of the same standardized sums, the plain estimate
        that outliers inflate; same shape as ``factor``.
    """

    factor: Any
    positions_used: int
    plain_ratio: Any


def psf_weights(
    psf_sigma: float,
    size: int,
    *,
    offset: Any | None = None,
    derivative: Literal["x", "y"] | None = None,
    xp: Any = np,
) -> Any:
    """Weights of a PSF-weighted sum on an odd square stamp.

    The point spread function is a unit-flux, point-sampled circular
    Gaussian centered on the stamp's center pixel, or ``offset`` pixels
    from it. With ``derivative`` the weights are its derivative with
    respect to the source position along that axis, the weights that
    measure a position.

    Parameters
    ----------
    psf_sigma
        Gaussian standard deviation in pixels.
    size
        Positive odd side length.
    offset
        Optional ``(N, 2)`` sub-pixel offsets ``(dx, dy)`` of the center
        from the center pixel, one stamp per row.
    derivative
        ``None``, ``"x"`` or ``"y"``.
    xp
        Array namespace of the result, NumPy or CuPy.

    Returns
    -------
    ndarray
        ``(size, size)`` weights, or ``(N, size, size)`` with ``offset``.
    """

    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    if psf_sigma <= 0:
        raise ValueError("psf_sigma must be positive")
    if derivative not in (None, "x", "y"):
        raise ValueError("derivative must be None, 'x' or 'y'")
    u = xp.arange(size, dtype=xp.float64) - size // 2
    if offset is None:
        dx = dy = xp.zeros(1, dtype=xp.float64)
    else:
        shift = xp.asarray(offset, dtype=xp.float64)
        if shift.ndim != 2 or shift.shape[1] != 2:
            raise ValueError("offset must have shape (N, 2)")
        dx, dy = shift[:, 0], shift[:, 1]
    ux = u[None, :] - dx[:, None]
    uy = u[None, :] - dy[:, None]
    norm = xp.sqrt(2.0 * xp.pi) * psf_sigma
    gx = xp.exp(-0.5 * (ux / psf_sigma) ** 2) / norm
    gy = xp.exp(-0.5 * (uy / psf_sigma) ** 2) / norm
    if derivative == "x":
        gx = gx * ux / psf_sigma**2
    elif derivative == "y":
        gy = gy * uy / psf_sigma**2
    stamps = gy[:, :, None] * gx[:, None, :]
    return stamps[0] if offset is None else stamps


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
        Two-dimensional finite, odd-shaped fitted kernel.
    resampling_weights
        ``(N, k, k)`` interpolation weights that produced the resampled
        reference pixel at each position from its source neighbors, or one
        ``(k, k)`` stamp. The stamp's orientation does not change the
        result.
    is_translation
        Optional ``(N,)`` booleans, true where the mapping between the grids
        is locally a translation so one stamp describes the whole kernel
        footprint. Positions flagged false get ``NaN``.

    Returns
    -------
    ndarray
        ``(N,)`` factors, or shape ``()`` for one ``(k, k)`` stamp; ``NaN``
        where the stamp is all zero or not finite, or where
        ``is_translation`` is false.
    """

    xp = _array_module(resampling_weights)
    kernel_arr = _kernel_array(xp, kernel)
    single = xp.asarray(resampling_weights).ndim == 2
    stamps = _resampling_stamps(xp, resampling_weights, 1 if single else None)
    true, diag = _pixel_terms(xp, kernel_arr, stamps)
    valid = _stamps_valid(xp, stamps) & _translated(
        xp, is_translation, int(stamps.shape[0])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = xp.where(valid, true / xp.where(valid, diag, 1.0), xp.nan)
    return factor[0] if single else factor


def correlation_factor(
    weights: Any,
    *,
    kernel: Any,
    target_variance: Any,
    reference_variance: Any,
    resampling_weights: Any,
    positions: Any | None = None,
    residual_variance: Any | None = None,
    is_translation: Any | None = None,
) -> CorrelationFactorResult:
    """Correlation factor of weighted sums of residual pixels.

    For weights ``w`` on a window of residual pixels the factor is

    ``(sum(w**2 V_target) + Var(sum_p w_p (K * R)_p)) / sum(w**2 V_D)``,

    ``V_D`` the diagonal residual variance, ``K`` the kernel and ``R`` the
    resampled reference. The reference part is the source-pixel variance,
    the ``w**2``-weighted mean over the window of ``reference_variance``
    divided by the resampling gain ``sum(resampling_weights**2)``, times
    the squared footprint of ``w`` on the source pixels through the kernel
    and the resampling weights. Target pixels are taken as independent.

    The variance inputs are either full planes cut here around
    ``positions`` with :func:`window_stamps`, or stamps already cut on the
    same windows; the stamps' side ``s`` sets the window.

    Parameters
    ----------
    weights
        ``(N, s, s)`` weights of the sum at each position, ``s`` odd: PSF
        weights for a flux, their derivative for a position, a top hat for
        an aperture. With ``positions``, one ``(s, s)`` stamp for every
        position is also accepted.
    kernel
        Two-dimensional finite, odd-shaped fitted kernel.
    target_variance
        Target variance: a two-dimensional plane when ``positions`` is
        given, otherwise ``(N, s, s)`` stamps or an array that broadcasts to
        them.
    reference_variance
        Variance of the resampled reference on the target grid, in squared
        reference units, the plane
        :func:`standardize_constant_kernel_residual` takes; the division by
        the resampling gain happens here. Same forms as ``target_variance``.
    resampling_weights
        ``(N, k, k)`` interpolation weights at each position, or one
        ``(k, k)`` stamp for every position.
    positions
        Optional ``(N, 2)`` array of ``(x, y)`` pixel coordinates at which
        the variance planes are cut, each window centered on the nearest
        integer pixel. Without it every variance input is a stack of
        stamps.
    residual_variance
        Optional diagonal residual variance plane the factor is measured
        against, such as ``marginal_residual_variance`` of
        :func:`standardize_constant_kernel_residual`, in the same form as
        ``target_variance``. By default it is
        ``target_variance + reference_variance * sum(kernel**2)``, the same
        plane under a flat reference variance.
    is_translation
        Optional ``(N,)`` booleans, true where the mapping between the grids
        is locally a translation. The closed form holds for a translation;
        positions flagged false get ``NaN`` and are counted in
        ``n_rotated``.

    Returns
    -------
    CorrelationFactorResult
        ``factor`` and ``pixel_ratio`` per position, ``NaN`` where any stamp
        of the window is not finite, where the resampling stamp is all
        zero, or where ``is_translation`` is false, with the covariance
        scope recorded.
    """

    xp = _array_module(weights, target_variance, reference_variance)
    kernel_arr = _kernel_array(xp, kernel)
    w = xp.asarray(weights, dtype=xp.float64)
    if positions is not None:
        xy = xp.asarray(positions, dtype=xp.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        if w.ndim == 2:
            w = xp.broadcast_to(w, (int(xy.shape[0]),) + w.shape)
    if w.ndim != 3 or w.shape[1] % 2 == 0 or w.shape[2] % 2 == 0:
        raise ValueError("weights must be odd-sized stamps (N, s, s)")
    count = int(w.shape[0])
    if positions is not None and int(xy.shape[0]) != count:
        raise ValueError("weights and positions disagree on N")

    def stamps_of(name: str, value: Any) -> Any:
        array = xp.asarray(value, dtype=xp.float64)
        if positions is not None and array.ndim == 2:
            array = window_stamps(
                array, xy, (int(w.shape[1]), int(w.shape[2]))
            )
        try:
            return xp.broadcast_to(array, w.shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a plane with positions, or stamps that "
                "broadcast to the weights"
            ) from exc

    target = stamps_of("target_variance", target_variance)
    resampled = stamps_of("reference_variance", reference_variance)
    stamps = _resampling_stamps(xp, resampling_weights, count)
    true, diag = _pixel_terms(xp, kernel_arr, stamps)
    gain = (stamps**2).sum(axis=(1, 2))
    valid = _stamps_valid(xp, stamps) & _translated(xp, is_translation, count)
    kernel_power = (kernel_arr**2).sum()
    if residual_variance is None:
        residual = target + resampled * kernel_power
    else:
        residual = stamps_of("residual_variance", residual_variance)

    w2 = w**2
    w2_sum = w2.sum(axis=(1, 2))
    target_sum = (w2 * target).sum(axis=(1, 2))
    resampled_sum = (w2 * resampled).sum(axis=(1, 2))
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
        valid
        & xp.isfinite(w).all(axis=(1, 2))
        & xp.isfinite(target).all(axis=(1, 2))
        & xp.isfinite(resampled).all(axis=(1, 2))
        & xp.isfinite(residual).all(axis=(1, 2))
        & (diagonal_sum > 0)
        & (w2_sum > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        source_variance = resampled_sum / (w2_sum * gain)
        factor = (target_sum + source_variance * power) / diagonal_sum
        pixel_ratio = (target_sum + source_variance * true * w2_sum) / (
            diagonal_sum
        )
    return CorrelationFactorResult(
        factor=xp.where(finite, factor, xp.nan),
        pixel_ratio=xp.where(finite, pixel_ratio, xp.nan),
        n_rotated=count - int(_translated(xp, is_translation, count).sum()),
    )


def empirical_correlation_factor(
    residual: Any,
    residual_variance: Any,
    weights: Any,
    positions: Any,
) -> EmpiricalCorrelationFactor:
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
    EmpiricalCorrelationFactor
        The factor, with shape ``()`` for one weighting or ``(q,)`` for
        several, the number of positions used and the plain variance
        ratio.
    """

    xp = _array_module(residual, residual_variance)
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

    side = int(w.shape[-1])
    data = window_stamps(residual_arr, positions, side)
    variance = window_stamps(variance_arr, positions, side)
    usable = (
        xp.isfinite(data).all(axis=(1, 2))
        & xp.isfinite(variance).all(axis=(1, 2))
        & (variance > 0).all(axis=(1, 2))
    )
    data, variance = data[usable], variance[usable]
    used = int(data.shape[0])
    sums = xp.einsum("qyx,nyx->nq", w, data)
    predicted = xp.einsum("qyx,nyx->nq", w**2, variance)
    if used < 2:
        robust = plain = xp.full(w.shape[0], xp.nan)
    else:
        z = sums / xp.sqrt(predicted)
        spread = 1.4826 * xp.median(xp.abs(z - xp.median(z, axis=0)), axis=0)
        robust = spread**2
        plain = z.var(axis=0, ddof=1)
    return EmpiricalCorrelationFactor(
        factor=robust[0] if single else robust,
        positions_used=used,
        plain_ratio=plain[0] if single else plain,
    )


def _kernel_array(xp: Any, kernel: Any) -> Any:
    kernel_arr = _real_2d_array(xp, "kernel", kernel)
    if any(size % 2 == 0 for size in kernel_arr.shape):
        raise ValueError("kernel dimensions must be odd")
    if not bool(xp.isfinite(kernel_arr).all()):
        raise ValueError("kernel must contain only finite values")
    return kernel_arr


def _resampling_stamps(xp: Any, resampling_weights: Any, count: int | None):
    stamps = xp.asarray(resampling_weights, dtype=xp.float64)
    if stamps.ndim == 2 and count is not None:
        stamps = xp.broadcast_to(stamps, (count,) + stamps.shape)
    if stamps.ndim != 3 or (count is not None and stamps.shape[0] != count):
        raise ValueError("resampling_weights must have shape (N, k, k)")
    return stamps


def _stamps_valid(xp: Any, stamps: Any) -> Any:
    """Stamps that are finite and not all zero."""

    return xp.isfinite(stamps).all(axis=(1, 2)) & (
        (stamps**2).sum(axis=(1, 2)) > 0
    )


def _translated(xp: Any, is_translation: Any, count: int) -> Any:
    if is_translation is None:
        return xp.ones(count, dtype=bool)
    flags = xp.asarray(is_translation, dtype=bool)
    if flags.shape != (count,):
        raise ValueError("is_translation must have shape (N,)")
    return flags


def _pixel_terms(xp: Any, kernel: Any, stamps: Any) -> tuple[Any, Any]:
    """True and diagonal reference-term variance of one residual pixel."""

    footprint = _fftconvolve(kernel[None], stamps, mode="full", axes=(1, 2))
    true = (footprint**2).sum(axis=(1, 2))
    diag = (kernel**2).sum() * (stamps**2).sum(axis=(1, 2))
    return true, diag
