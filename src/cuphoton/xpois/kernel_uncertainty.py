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

from .ois import triangular_degree_pairs

__all__ = [
    "KernelFitCovariance",
    "kernel_fit_covariance",
    "kernel_fit_variance",
]


@dataclass(frozen=True)
class KernelFitCovariance:
    """Covariance of the fitted kernel and background coefficients.

    Attributes
    ----------
    covariance
        Square covariance matrix over the fit coefficients, the kernel
        basis coefficients first and the background coefficients last, in
        the order of :func:`triangular_degree_pairs`. NumPy or CuPy,
        following the reference image the covariance was built from.
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
    reference_source_variance: Any | None = None,
    resampling_weights: Any | None = None,
) -> KernelFitCovariance:
    """Covariance of the coefficients of a constant-kernel fit.

    The fit solves ``target ~= sum_j c_j (reference * basis_j) + b(x, y)``
    by weighted least squares with ``W = 1 / target_variance`` over the
    fit pixels. The plain covariance is ``(A^T W A)^-1``, ``A`` the design
    matrix over those pixels; no reduced chi-square scaling is applied.

    The resampled reference also carries noise, which the fit's residual
    contains but its weights omit. With ``reference_source_variance`` and
    ``resampling_weights`` the sandwich form ``H^-1 (H + M) H^-1`` adds it,
    ``H = A^T W A`` and ``M = U^T diag(V_src) U`` where ``U`` is the
    footprint of each weighted design column on the reference's source
    pixels, through the fitted kernel and the resampling weights. The
    footprint uses one resampling stamp for every fit pixel, so on a grid
    that rotates or scales against the reference's this form is an
    approximation.

    Parameters
    ----------
    fit
        A host :class:`ConstantKernelFitResult` or a device
        :class:`DeviceConstantKernelFitResult`. Its ``basis_kernels``,
        ``background_coefficients``, ``fit_mask`` and ``kernel`` are read.
    reference
        Two-dimensional reference image on the target grid, the one the fit
        convolved. Non-finite pixels count as zero, as in the solve. Its
        array namespace (NumPy or CuPy) sets the computation's.
    target_variance
        Target variance image; its inverse is the fit's weight. It must be
        positive and finite on the fit pixels.
    reference_source_variance
        Variance of the reference's source pixels as an image on the target
        grid: the resampled reference variance divided by the resampling
        gain ``sum(resampling_weights**2)``. Non-finite pixels count as
        zero. Given together with ``resampling_weights``.
    resampling_weights
        One ``(k, k)`` stamp (or ``(1, k, k)``) of the interpolation weights
        that produced a resampled reference pixel from its source
        neighbors, taken at a representative position such as the image
        center. The stamp's orientation does not change the result.

    Returns
    -------
    KernelFitCovariance
        The covariance with its form and the coefficient layout.
    """

    xp = _array_module(reference)
    reference_arr = _real_2d_array(xp, "reference", reference)
    variance_arr = _real_2d_array(xp, "target_variance", target_variance)
    if variance_arr.shape != reference_arr.shape:
        raise ValueError("target_variance must match reference shape")
    if (reference_source_variance is None) != (resampling_weights is None):
        raise ValueError(
            "reference_source_variance and resampling_weights are given "
            "together for the sandwich form"
        )

    basis = xp.asarray(fit.basis_kernels, dtype=xp.float64)
    if basis.ndim != 3:
        raise ValueError("fit.basis_kernels must be three-dimensional")
    mask = xp.asarray(fit.fit_mask, dtype=bool)
    if mask.shape != reference_arr.shape:
        raise ValueError("fit.fit_mask must match reference shape")
    degree = _background_degree(int(np.size(fit.background_coefficients)))

    reference_model = xp.where(xp.isfinite(reference_arr), reference_arr, 0.0)
    columns = [
        _fftconvolve(reference_model, kernel, mode="same")[mask]
        for kernel in basis
    ]
    columns += [
        term[mask]
        for term in _background_images(xp, reference_arr.shape, degree)
    ]
    design = xp.stack(columns, axis=1)
    weights = 1.0 / variance_arr[mask]
    if not bool(xp.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError(
            "target_variance must be positive and finite on the fit pixels"
        )
    if design.shape[0] < design.shape[1]:
        raise ValueError("fewer fit pixels than coefficients")
    hessian = design.T @ (design * weights[:, None])
    try:
        covariance = xp.linalg.inv(hessian)
    except Exception as exc:
        raise ValueError("fit normal equations are singular") from exc
    form: Literal["plain", "sandwich"] = "plain"

    if resampling_weights is not None:
        form = "sandwich"
        source_arr = _real_2d_array(
            xp, "reference_source_variance", reference_source_variance
        )
        if source_arr.shape != reference_arr.shape:
            raise ValueError(
                "reference_source_variance must match reference shape"
            )
        stamp = xp.asarray(resampling_weights, dtype=xp.float64)
        if stamp.ndim == 3 and stamp.shape[0] == 1:
            stamp = stamp[0]
        if stamp.ndim != 2:
            raise ValueError(
                "resampling_weights must be one (k, k) stamp for the "
                "kernel-fit covariance"
            )
        kernel = xp.asarray(fit.kernel, dtype=xp.float64)[::-1, ::-1]
        source_flat = xp.where(
            xp.isfinite(source_arr), source_arr, 0.0
        ).ravel()
        footprint = xp.empty((design.shape[1], source_flat.size))
        image = xp.zeros(reference_arr.shape, dtype=xp.float64)
        for j in range(design.shape[1]):
            image[mask] = design[:, j] * weights
            footprint[j] = _fftconvolve(
                _fftconvolve(image, kernel, mode="same"), stamp, mode="same"
            ).ravel()
        resampled = (footprint * source_flat) @ footprint.T
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
        and its kernel footprint leave the image.
    """

    xp = _array_module(reference)
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

    half = int(stamps.shape[-1]) // 2
    kernel_half_y = int(basis.shape[1]) // 2
    kernel_half_x = int(basis.shape[2]) // 2
    height, width = reference_arr.shape
    center_x = xp.rint(xy[:, 0]).astype(xp.int64)
    center_y = xp.rint(xy[:, 1]).astype(xp.int64)
    inside = (
        (center_x >= half + kernel_half_x)
        & (center_x < width - half - kernel_half_x)
        & (center_y >= half + kernel_half_y)
        & (center_y < height - half - kernel_half_y)
    )
    safe_x = xp.where(inside, center_x, half + kernel_half_x)
    safe_y = xp.where(inside, center_y, half + kernel_half_y)
    reference_model = xp.where(xp.isfinite(reference_arr), reference_arr, 0.0)
    patch_y = xp.arange(-half - kernel_half_y, half + kernel_half_y + 1)
    patch_x = xp.arange(-half - kernel_half_x, half + kernel_half_x + 1)
    patches = reference_model[
        safe_y[:, None, None] + patch_y[None, :, None],
        safe_x[:, None, None] + patch_x[None, None, :],
    ]
    response = _fftconvolve(
        patches[:, None], basis[None], mode="valid", axes=(2, 3)
    )
    offsets = xp.arange(-half, half + 1)
    background = _background_terms(
        xp,
        (height, width),
        cov.background_degree,
        safe_y[:, None, None] + offsets[None, :, None],
        safe_x[:, None, None] + offsets[None, None, :],
    )
    gamma = xp.concatenate([response, background], axis=1)
    projected = xp.einsum("ncyx,nyx->nc", gamma, stamps)
    variance = xp.einsum("nc,cd,nd->n", projected, covariance, projected)
    return xp.where(inside, variance, xp.nan)


def _background_degree(n_terms: int) -> int:
    degree = 0
    while (degree + 1) * (degree + 2) // 2 < n_terms:
        degree += 1
    if (degree + 1) * (degree + 2) // 2 != n_terms:
        raise ValueError(
            "background_coefficients do not fill a triangular degree"
        )
    return degree


def _background_images(xp: Any, shape: tuple[int, int], degree: int) -> Any:
    """Background design terms over the image, ``(terms, height, width)``."""

    height, width = shape
    rows = xp.arange(height, dtype=xp.float64)[:, None]
    columns = xp.arange(width, dtype=xp.float64)[None, :]
    return _background_terms(xp, shape, degree, rows, columns)


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


def _array_module(array: Any) -> Any:
    if type(array).__module__.split(".", maxsplit=1)[0] == "cupy":
        import cupy

        return cupy
    return np


def _real_2d_array(xp: Any, name: str, value: Any) -> Any:
    array = xp.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be a real numeric array")
    return xp.asarray(array, dtype=xp.float64)
