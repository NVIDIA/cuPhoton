# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Uncertainty of a fitted constant kernel and what it adds to the residual.

:func:`standardize_constant_kernel_residual` treats the fitted kernel as
fixed. Near a bright reference source the kernel's own uncertainty adds
variance to every residual pixel. This module carries the covariance of the
fitted kernel and background coefficients to a residual pixel, or to a
weighted sum of residual pixels such as a PSF flux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .ois import (
    _accumulate_normal_equations,
    _accumulate_normal_equations_cupy_device,
    _background_design_cupy,
    _default_fit_mask_impl,
    background_design,
    triangular_degree_pairs,
)
from .stamps import _array_module, window_stamps

__all__ = [
    "KernelFitCovariance",
    "kernel_fit_covariance",
    "kernel_fit_variance",
]

_MAX_CONDITION_NUMBER = 1e12
# Element budgets of the batched intermediates: the float64 footprint images
# of one coefficient chunk in the sandwich form (512 MB), and the response
# complex spectra of one position chunk in ``kernel_fit_variance``.
_FOOTPRINT_CHUNK_ELEMENTS = 2**26
_RESPONSE_CHUNK_ELEMENTS = 2**24


@dataclass(frozen=True)
class KernelFitCovariance:
    """Covariance of the fitted kernel and background coefficients.

    Attributes
    ----------
    covariance
        Square covariance matrix over the fit coefficients, the kernel
        basis coefficients first and the background coefficients last, in
        the order of :func:`triangular_degree_pairs`. NumPy or CuPy,
        following the arrays the covariance was built from.
    basis_kernels
        The fit's concrete basis kernels, ``(basis_count, ky, kx)``.
    basis_count
        Number of kernel basis coefficients.
    background_degree
        Total degree of the differential-background polynomial.
    form
        ``"plain"`` for ``(A^T W A)^-1`` with the weights the fit used;
        ``"sandwich"`` when the resampled reference's correlated noise,
        which the fit's residual contains but its weights omit, is added
        through the fit.
    covariance_represented
        Always true: the full coefficient covariance is carried, not a
        diagonal.
    """

    covariance: Any
    basis_kernels: Any
    basis_count: int
    background_degree: int
    form: Literal["plain", "sandwich"]
    covariance_represented: Literal[True] = field(default=True, init=False)


def kernel_fit_covariance(
    fit: Any,
    *,
    reference: Any,
    target_variance: Any,
    reference_variance: Any | None = None,
    resampling_weights: Any | None = None,
) -> KernelFitCovariance:
    """Covariance of the coefficients of a constant-kernel fit.

    The fit solves ``target ~= sum_j c_j (reference * basis_j) + b(x, y)``
    by weighted least squares with ``W = 1 / target_variance`` over the
    fit pixels. The plain covariance is ``(A^T W A)^-1``, ``A`` the design
    matrix over those pixels; no reduced chi-square scaling is applied.
    ``H = A^T W A`` is accumulated in chunks as in the solve, without
    materializing ``A``.

    The resampled reference also carries noise, which the fit's residual
    contains but its weights omit. With ``reference_variance`` and
    ``resampling_weights`` the sandwich form ``H^-1 (H + M) H^-1`` adds it,
    ``M = U diag(V) U^T`` where row ``j`` of ``U`` is the footprint of the
    weighted design column ``W A_j`` on the reference's source pixels,
    through the fitted kernel and the resampling weights, and ``V`` the
    source-pixel variance. The footprint uses one resampling stamp for
    every fit pixel, so on a grid that rotates or scales against the
    reference's this form is an approximation.

    The sandwich holds ``U`` as one full-size image per coefficient
    (``basis_count + background terms``) in memory, so its cost grows with
    the image area times the coefficient count. The footprints are built
    in chunks of coefficients, each chunk's images holding at most
    ``_FOOTPRINT_CHUNK_ELEMENTS`` (2**26) float64 values, with one batched
    convolution per chunk. On a 4096 x 4096 frame with 20 coefficients a
    chunk is 4 coefficient images (512 MB); the chunk's convolution
    temporaries peak near 3.3 GB on top of the 2.5 GB of ``U``, so the
    sandwich peaks near 5.8 GB there. The plain form holds no full-size
    array beyond the inputs.

    Parameters
    ----------
    fit
        A host :class:`ConstantKernelFitResult` or a device
        :class:`DeviceConstantKernelFitResult`. Its ``basis_kernels``,
        ``background_coefficients``, ``fit_mask`` and ``kernel`` are read.
    reference
        Two-dimensional reference image on the target grid, the one the fit
        convolved. Non-finite pixels count as zero, as in the solve.
    target_variance
        Target variance image; its inverse is the fit's weight. It must be
        positive and finite on the fit pixels.
    reference_variance
        Variance of the resampled reference on the target grid, in squared
        reference units, the plane
        :func:`standardize_constant_kernel_residual` takes. The division
        by the resampling gain ``sum(resampling_weights**2)`` that gives
        the source-pixel variance happens here. Non-finite pixels count as
        zero; finite values must be nonnegative. Given together with
        ``resampling_weights``.
    resampling_weights
        One ``(k, k)`` stamp (or ``(1, k, k)``) of the interpolation weights
        that produced a resampled reference pixel from its source
        neighbors, ``resampled[p] = sum_q weights[q] source[p + q - c]``
        with ``c`` the stamp center, taken at a representative position
        such as the image center. A stamp symmetric under point reflection
        gives the same result in either orientation.

    Returns
    -------
    KernelFitCovariance
        The covariance with its form and the coefficient layout.

    Notes
    -----
    The computation runs in CuPy when the fit's arrays or ``reference`` are
    CuPy arrays, and in NumPy otherwise; NumPy inputs are moved to the
    device in the first case.
    """

    xp = _array_module(fit.basis_kernels, fit.fit_mask, reference)
    reference_arr = _real_2d_array(xp, "reference", reference)
    variance_arr = _real_2d_array(xp, "target_variance", target_variance)
    if variance_arr.shape != reference_arr.shape:
        raise ValueError("target_variance must match reference shape")
    if (reference_variance is None) != (resampling_weights is None):
        raise ValueError(
            "reference_variance and resampling_weights are given together "
            "for the sandwich form"
        )

    basis = xp.asarray(fit.basis_kernels, dtype=xp.float64)
    if basis.ndim != 3:
        raise ValueError("fit.basis_kernels must be three-dimensional")
    mask = xp.asarray(fit.fit_mask, dtype=bool)
    if mask.shape != reference_arr.shape:
        raise ValueError("fit.fit_mask must match reference shape")
    mask = mask & _default_fit_mask_impl(xp, mask.shape, basis.shape[1:])
    degree = _background_degree(int(np.size(fit.background_coefficients)))

    fit_variance = variance_arr[mask]
    if not bool(xp.isfinite(fit_variance).all()) or bool(
        (fit_variance <= 0).any()
    ):
        raise ValueError(
            "target_variance must be positive and finite on the fit pixels"
        )
    weights = 1.0 / fit_variance
    reference_model = xp.where(xp.isfinite(reference_arr), reference_arr, 0.0)
    if xp is np:
        background = background_design(reference_arr.shape, degree)
        hessian, _, row_count = _accumulate_normal_equations(
            reference_model,
            xp.zeros_like(reference_model),
            variance_arr,
            mask,
            basis,
            background,
        )
    else:
        background = _background_design_cupy(
            reference_arr.shape, degree, cp=xp
        )
        hessian, _, row_count = _accumulate_normal_equations_cupy_device(
            reference_model,
            xp.zeros_like(reference_model),
            variance_arr,
            mask,
            basis,
            background,
            cp=xp,
        )
    n_coefficients = int(hessian.shape[0])
    if row_count < n_coefficients:
        raise ValueError("fewer fit pixels than coefficients")
    covariance = _inverse(xp, hessian)
    form: Literal["plain", "sandwich"] = "plain"

    if resampling_weights is not None:
        form = "sandwich"
        resampled_arr = _real_2d_array(
            xp, "reference_variance", reference_variance
        )
        if resampled_arr.shape != reference_arr.shape:
            raise ValueError("reference_variance must match reference shape")
        if bool(xp.any(xp.isfinite(resampled_arr) & (resampled_arr < 0.0))):
            raise ValueError(
                "reference_variance must be nonnegative where finite"
            )
        stamp = xp.asarray(resampling_weights, dtype=xp.float64)
        if stamp.ndim == 3 and stamp.shape[0] == 1:
            stamp = stamp[0]
        if stamp.ndim != 2:
            raise ValueError(
                "resampling_weights must be one (k, k) stamp for the "
                "kernel-fit covariance"
            )
        gain = float((stamp**2).sum())
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError(
                "resampling_weights must be finite and not all zero"
            )
        kernel = xp.asarray(fit.kernel, dtype=xp.float64)[::-1, ::-1]
        source_flat = (
            xp.where(xp.isfinite(resampled_arr), resampled_arr, 0.0) / gain
        ).ravel()
        weight_image = xp.zeros(reference_arr.shape, dtype=xp.float64)
        weight_image[mask] = weights
        n_pixels = int(source_flat.size)
        step = max(1, _FOOTPRINT_CHUNK_ELEMENTS // n_pixels)
        footprint = xp.empty((n_coefficients, n_pixels))
        for start in range(0, n_coefficients, step):
            stop = min(start + step, n_coefficients)
            columns = _design_columns(
                xp, reference_model, basis, background, start, stop
            )
            footprint[start:stop] = _fftconvolve(
                _fftconvolve(
                    columns * weight_image[None],
                    kernel[None],
                    mode="same",
                    axes=(1, 2),
                ),
                stamp[None],
                mode="same",
                axes=(1, 2),
            ).reshape(stop - start, n_pixels)
        footprint *= xp.sqrt(source_flat)
        resampled = xp.empty((n_coefficients, n_coefficients))
        for start in range(0, n_coefficients, step):
            stop = min(start + step, n_coefficients)
            resampled[start:stop] = footprint[start:stop] @ footprint.T
        covariance = covariance + covariance @ resampled @ covariance

    return KernelFitCovariance(
        covariance=covariance,
        basis_kernels=basis,
        basis_count=int(basis.shape[0]),
        background_degree=degree,
        form=form,
    )


def kernel_fit_variance(
    cov: KernelFitCovariance,
    *,
    reference: Any,
    positions: Any,
    weights: Any | None = None,
) -> Any:
    """Variance the kernel fit adds to the residual at given positions.

    At a position the residual's response to each coefficient is ``gamma``:
    the reference convolved with each basis kernel, and the background
    design terms. The added variance is ``gamma^T C gamma``. With
    ``weights`` it is the variance of the weighted sum of residual pixels
    around each position, ``g = sum_p w_p gamma_p`` and ``g^T C g``.

    The response and the quadratic form are batched over positions. The
    basis convolutions run in chunks of positions whose spectra hold at
    most ``_RESPONSE_CHUNK_ELEMENTS`` (2**24) complex values; with 20
    coefficients, a 15 x 15 kernel and 15 x 15 weights a chunk is 477
    positions and its temporaries peak near 0.55 GB, independent of the
    frame size.

    Parameters
    ----------
    cov
        The covariance from :func:`kernel_fit_covariance`.
    reference
        Two-dimensional reference image on the target grid, the one the fit
        convolved. Non-finite pixels count as zero.
    positions
        ``(N, 2)`` array of ``(x, y)`` target pixel coordinates. Each stamp
        is centered on the nearest integer pixel.
    weights
        Optional ``(N, s, s)`` or ``(s, s)`` odd-sized weight stamps. Without
        them the variance of the single residual pixel is returned.

    Returns
    -------
    ndarray
        ``(N,)`` variances in squared target units; ``NaN`` where a stamp
        and its kernel footprint leave the image. CuPy when ``cov`` or
        ``reference`` holds CuPy arrays, NumPy otherwise.
    """

    xp = _array_module(cov.covariance, cov.basis_kernels, reference)
    reference_arr = _real_2d_array(xp, "reference", reference)
    basis = xp.asarray(cov.basis_kernels, dtype=xp.float64)
    xy = xp.asarray(positions, dtype=xp.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    count = int(xy.shape[0])
    if weights is None:
        stamps = xp.ones((count, 1, 1), dtype=xp.float64)
    else:
        stamps = xp.asarray(weights, dtype=xp.float64)
        if stamps.ndim == 2:
            stamps = xp.broadcast_to(stamps, (count,) + stamps.shape)
        if (
            stamps.ndim != 3
            or stamps.shape[0] != count
            or stamps.shape[1] != stamps.shape[2]
            or stamps.shape[1] % 2 == 0
        ):
            raise ValueError(
                "weights must be odd square stamps with shape (N, s, s)"
            )
    covariance = xp.asarray(cov.covariance, dtype=xp.float64)
    n_terms = len(triangular_degree_pairs(cov.background_degree))
    if covariance.shape != (cov.basis_count + n_terms,) * 2:
        raise ValueError("covariance does not match basis_count and degree")

    side = int(stamps.shape[-1])
    reference_model = xp.where(xp.isfinite(reference_arr), reference_arr, 0.0)
    patches = window_stamps(
        reference_model,
        xy,
        (side + int(basis.shape[1]) - 1, side + int(basis.shape[2]) - 1),
    )
    inside = xp.isfinite(patches).all(axis=(1, 2))
    patches = xp.where(inside[:, None, None], patches, 0.0)
    half = side // 2
    offsets = xp.arange(-half, half + 1)
    center = xp.rint(xy).astype(xp.int64)
    background = _background_terms(
        xp,
        reference_arr.shape,
        cov.background_degree,
        center[:, 1][:, None, None] + offsets[None, :, None],
        center[:, 0][:, None, None] + offsets[None, None, :],
    )
    projected = xp.empty((count, int(covariance.shape[0])), dtype=xp.float64)
    spectrum = (
        int(basis.shape[0])
        * (patches.shape[1] + basis.shape[1] - 1)
        * (patches.shape[2] + basis.shape[2] - 1)
    )
    step = max(1, _RESPONSE_CHUNK_ELEMENTS // spectrum)
    for start in range(0, count, step):
        stop = min(start + step, count)
        response = _fftconvolve(
            patches[start:stop, None], basis[None], mode="valid", axes=(2, 3)
        )
        projected[start:stop, : basis.shape[0]] = xp.einsum(
            "ncyx,nyx->nc", response, stamps[start:stop]
        )
    projected[:, basis.shape[0] :] = xp.einsum(
        "ncyx,nyx->nc", background, stamps
    )
    variance = xp.einsum("nc,cd,nd->n", projected, covariance, projected)
    return xp.where(inside, variance, xp.nan)


def _inverse(xp: Any, hessian: Any) -> Any:
    """Inverse of the normal equations with the solve's conditioning guard."""

    try:
        condition_number = float(xp.linalg.cond(hessian))
    except xp.linalg.LinAlgError as exc:
        raise ValueError("fit normal equations are singular") from exc
    if (
        not np.isfinite(condition_number)
        or condition_number > _MAX_CONDITION_NUMBER
    ):
        raise ValueError("fit normal equations are ill-conditioned")
    try:
        return xp.linalg.inv(hessian)
    except xp.linalg.LinAlgError as exc:
        raise ValueError("fit normal equations are singular") from exc


def _design_columns(
    xp: Any,
    reference: Any,
    basis: Any,
    background: Any,
    start: int,
    stop: int,
) -> Any:
    """Design columns ``start:stop`` as images, ``(stop - start, ny, nx)``.

    The kernel columns are the reference convolved with each basis kernel of
    the range in one batched convolution; the background columns are the
    design terms.
    """

    n_basis = int(basis.shape[0])
    columns = xp.empty((stop - start,) + reference.shape, dtype=xp.float64)
    split = min(max(start, n_basis), stop)
    if split > start:
        columns[: split - start] = _fftconvolve(
            reference[None], basis[start:split], mode="same", axes=(1, 2)
        )
    if stop > split:
        columns[split - start :] = background[
            split - n_basis : stop - n_basis
        ]
    return columns


def _background_degree(n_terms: int) -> int:
    degree = 0
    while (degree + 1) * (degree + 2) // 2 < n_terms:
        degree += 1
    if (degree + 1) * (degree + 2) // 2 != n_terms:
        raise ValueError(
            "background_coefficients do not fill a triangular degree"
        )
    return degree


def _background_terms(
    xp: Any, shape: tuple[int, int], degree: int, rows: Any, columns: Any
) -> Any:
    """Background design terms at pixel ``rows`` and ``columns``.

    The terms are centered polynomials in the normalized coordinates of
    :func:`background_design`; the result stacks them on axis ``-3`` of
    the broadcast of ``rows`` and ``columns``.
    """

    height, width = shape
    y = -1.0 + 2.0 * rows / max(height - 1, 1)
    x = -1.0 + 2.0 * columns / max(width - 1, 1)
    grid = xp.broadcast(x, y).shape
    terms = [
        xp.broadcast_to((x**deg_x) * (y**deg_y), grid)
        for deg_x, deg_y in triangular_degree_pairs(degree)
    ]
    return xp.stack(terms, axis=-3)


def _fftconvolve(a: Any, k: Any, *, mode: str, axes: Any = None) -> Any:
    """Linear convolution of real arrays by FFT in their own namespace.

    Transforms ``axes`` (every axis by default; the others broadcast) at the
    full linear size, then crops to the full, same (centered on ``a``) or
    valid output.
    """

    xp = _array_module(a)
    axes = tuple(range(a.ndim)) if axes is None else tuple(axes)
    full = [a.shape[i] + k.shape[i] - 1 for i in axes]
    out = xp.fft.irfftn(
        xp.fft.rfftn(a, full, axes) * xp.fft.rfftn(k, full, axes), full, axes
    )
    keep = {
        "full": full,
        "same": [a.shape[i] for i in axes],
        "valid": [a.shape[i] - k.shape[i] + 1 for i in axes],
    }[mode]
    crop = [slice(None)] * out.ndim
    for i, n, m in zip(axes, full, keep, strict=True):
        crop[i] = slice((n - m) // 2, (n - m) // 2 + m)
    return out[tuple(crop)]


def _real_2d_array(xp: Any, name: str, value: Any) -> Any:
    array = xp.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be a real numeric array")
    return xp.asarray(array, dtype=xp.float64)
