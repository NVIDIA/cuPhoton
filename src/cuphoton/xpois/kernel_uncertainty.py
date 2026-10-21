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

from cuphoton.core.arrays import array_module, fft_module

from .ois import (
    _accumulate_normal_equations,
    _accumulate_normal_equations_cupy_device,
    _background_design_cupy,
    _default_fit_mask_impl,
    background_design,
    triangular_degree_pairs,
)
from .stamps import window_stamps

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
    chunk_elements: int | None = None,
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
    ``chunk_elements`` float64 values, with one batched convolution per
    chunk (kernel and resampling stamp composed into one) at the next fast
    FFT length; every chunk has the same size, the last one padded with
    zero images, so one FFT plan serves them all. Array memory on a
    4096 x 4096 frame with 11 coefficients: ``U`` is 1.5 GB, a chunk is 4
    coefficient images (0.5 GB) and its spectra, products and inverse
    transforms at the padded 4320 x 4320 size are 0.6 GB each, so the
    sandwich peaks near 5.2 GB of arrays (the plain form near 1.0 GB). On
    CuPy, cuFFT plan work areas come on top of these figures. Padding to
    the next fast length keeps them small; at an odd full length cuFFT
    would fall back to a Bluestein plan whose work area runs to several
    GB.

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
    chunk_elements
        Element budget of one coefficient chunk of the sandwich footprints,
        by default ``_FOOTPRINT_CHUNK_ELEMENTS`` (2**26, 512 MB of float64
        images). A smaller budget lowers the peak memory and runs more
        chunks.

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

    xp = array_module(fit.basis_kernels, fit.fit_mask, reference)
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

    # one device sync for the whole check instead of a compress and two
    bad = mask & ~(xp.isfinite(variance_arr) & (variance_arr > 0))
    if bool(bad.any()):
        raise ValueError(
            "target_variance must be positive and finite on the fit pixels"
        )
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
        weight_image = xp.where(
            mask, 1.0 / xp.where(mask, variance_arr, 1.0), 0.0
        )
        n_pixels = int(source_flat.size)
        if chunk_elements is None:
            chunk_elements = _FOOTPRINT_CHUNK_ELEMENTS
        step = min(n_coefficients, max(1, chunk_elements // n_pixels))
        shape = (step,) + reference_arr.shape
        # the kernel and the resampling stamp in one pass; its "same" crop
        # is the one the two passes give in turn
        footprint_spectrum = _spectrum(
            shape,
            _fftconvolve(kernel[None], stamp[None], mode="full", axes=(1, 2)),
            axes=(1, 2),
        )
        reference_spectrum = _spectrum(
            basis.shape, reference_model[None], axes=(1, 2)
        )
        footprint = xp.empty((n_coefficients, n_pixels))
        for start in range(0, n_coefficients, step):
            stop = min(start + step, n_coefficients)
            columns = _design_columns(
                xp, reference_spectrum, basis, background, start, stop, step
            )
            columns *= weight_image[None]
            footprint[start:stop] = _fftconvolve(
                columns, footprint_spectrum, mode="same", axes=(1, 2)
            )[: stop - start].reshape(stop - start, n_pixels)
        # source_flat is a variance over a positive gain, so it is not
        # negative and the weighted Gram matrix is a product with itself
        footprint *= xp.sqrt(source_flat)[None]
        resampled = footprint @ footprint.T
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
    chunk_elements: int | None = None,
) -> Any:
    """Variance the kernel fit adds to the residual at given positions.

    At a position the residual's response to each coefficient is ``gamma``:
    the reference convolved with each basis kernel, and the background
    design terms. The added variance is ``gamma^T C gamma``. With
    ``weights`` it is the variance of the weighted sum of residual pixels
    around each position, ``g = sum_p w_p gamma_p`` and ``g^T C g``.
    Several weightings of the same windows (a flux and the two position
    derivatives, say) share one response: pass them on a weights axis.

    The response and the quadratic form are batched over positions. The
    basis convolutions run in chunks of positions whose spectra hold at
    most ``chunk_elements`` complex values; with 11 coefficients, a 15 x 15
    kernel and 21 x 21 weights a chunk is 698 positions and its arrays peak
    near 0.6 GB, independent of the frame size (FFT plan work areas add to
    this on CuPy). Every chunk has the same size, the last one padded with
    zero stamps, so one FFT plan serves them all.

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
        Optional ``(N, s, s)`` or ``(s, s)`` odd-sized weight stamps, or
        ``(N, q, s, s)`` for ``q`` weightings of the same windows. Without
        them the variance of the single residual pixel is returned.
    chunk_elements
        Element budget of one position chunk's response spectra, by default
        ``_RESPONSE_CHUNK_ELEMENTS`` (2**24 complex values). A smaller
        budget lowers the peak memory and runs more chunks.

    Returns
    -------
    ndarray
        ``(N,)`` variances in squared target units, ``(N, q)`` for
        ``(N, q, s, s)`` weights; ``NaN`` where a stamp and its kernel
        footprint leave the image. CuPy when ``cov`` or ``reference`` holds
        CuPy arrays, NumPy otherwise.
    """

    xp = array_module(cov.covariance, cov.basis_kernels, reference)
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
    batched = stamps.ndim == 4
    if not batched:
        stamps = stamps[:, None]
    if (
        stamps.ndim != 4
        or stamps.shape[0] != count
        or stamps.shape[2] != stamps.shape[3]
        or stamps.shape[2] % 2 == 0
    ):
        raise ValueError(
            "weights must be odd square stamps with shape (N, s, s) or "
            "(N, q, s, s)"
        )
    covariance = xp.asarray(cov.covariance, dtype=xp.float64)
    n_terms = len(triangular_degree_pairs(cov.background_degree))
    if covariance.shape != (cov.basis_count + n_terms,) * 2:
        raise ValueError("covariance does not match basis_count and degree")
    if count == 0:
        return xp.empty(0, dtype=xp.float64)

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
    projected = xp.empty(
        (count, int(stamps.shape[1]), int(covariance.shape[0])),
        dtype=xp.float64,
    )
    spectrum = (
        int(basis.shape[0])
        * (patches.shape[1] + basis.shape[1] - 1)
        * (patches.shape[2] + basis.shape[2] - 1)
    )
    if chunk_elements is None:
        chunk_elements = _RESPONSE_CHUNK_ELEMENTS
    step = min(count, max(1, chunk_elements // spectrum))
    padded = -(-count // step) * step
    if padded > count:
        patches = xp.concatenate(
            [patches, xp.zeros((padded - count,) + patches.shape[1:])]
        )
    basis_spectrum = _spectrum(
        (step, 1) + patches.shape[1:], basis[None], axes=(2, 3)
    )
    for start in range(0, count, step):
        stop = min(start + step, count)
        response = _fftconvolve(
            patches[start : start + step, None],
            basis_spectrum,
            mode="valid",
            axes=(2, 3),
        )
        projected[start:stop, :, : basis.shape[0]] = xp.einsum(
            "ncyx,nqyx->nqc", response[: stop - start], stamps[start:stop]
        )
    projected[:, :, basis.shape[0] :] = xp.einsum(
        "ncyx,nqyx->nqc", background, stamps
    )
    variance = xp.einsum("nqc,cd,nqd->nq", projected, covariance, projected)
    variance = xp.where(inside[:, None], variance, xp.nan)
    return variance if batched else variance[:, 0]


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
    reference: _Spectrum,
    basis: Any,
    background: Any,
    start: int,
    stop: int,
    step: int,
) -> Any:
    """Design columns ``start:stop`` as images, ``(step, ny, nx)``.

    The kernel columns are the reference (given as its spectrum for the
    basis shape) convolved with each basis kernel of the range in one
    batched convolution; the background columns are the design terms.
    Images past ``stop - start`` are zero, so every chunk has one shape.
    """

    n_basis = int(basis.shape[0])
    columns = xp.zeros((step,) + reference.shape[1:], dtype=xp.float64)
    split = min(max(start, n_basis), stop)
    if split > start:
        columns[: split - start] = _fftconvolve(
            reference, basis[start:split], mode="same", axes=(1, 2)
        )
    if stop > split:
        columns[split - start : stop - start] = background[
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


@dataclass(frozen=True)
class _Spectrum:
    """The transform of a convolution kernel at fixed fast FFT lengths.

    ``_fftconvolve`` takes it in place of the kernel, so a kernel that
    meets every chunk of a batch is transformed once.
    """

    shape: tuple[int, ...]
    axes: tuple[int, ...]
    full: list[int]
    fast: list[int]
    values: Any


def _spectrum(
    a_shape: tuple[int, ...], k: Any, axes: Any = None
) -> _Spectrum:
    """Transform ``k`` for convolution with arrays of shape ``a_shape``.

    The lengths are the next fast FFT length above the full linear size
    (``next_fast_len`` of ``scipy.fft`` or ``cupyx.scipy.fft``), so odd or
    prime lengths never take the Bluestein plans with their large work
    areas.
    """

    xp = array_module(k)
    next_fast_len = fft_module(xp).next_fast_len
    axes = tuple(range(len(a_shape))) if axes is None else tuple(axes)
    full = [a_shape[i] + k.shape[i] - 1 for i in axes]
    fast = [next_fast_len(n, True) for n in full]
    return _Spectrum(
        tuple(k.shape), axes, full, fast, xp.fft.rfftn(k, fast, axes)
    )


def _fftconvolve(a: Any, k: Any, *, mode: str, axes: Any = None) -> Any:
    """Linear convolution of real arrays by FFT in their own namespace.

    Transforms ``axes`` (every axis by default; the others broadcast) at the
    fast lengths of :func:`_spectrum`, then crops the full linear size to
    the full, same (centered on ``a``) or valid output. Either operand may
    be given as its :class:`_Spectrum` taken for the other's shape.
    """

    if not isinstance(a, _Spectrum) and not isinstance(k, _Spectrum):
        k = _spectrum(a.shape, k, axes)
    spec, other = (a, k) if isinstance(a, _Spectrum) else (k, a)
    xp = array_module(other)
    axes, full, fast = spec.axes, spec.full, spec.fast
    if [a.shape[i] + k.shape[i] - 1 for i in axes] != full:
        raise ValueError("spectrum was taken for another array shape")
    out = xp.fft.irfftn(
        xp.fft.rfftn(other, fast, axes) * spec.values, fast, axes
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


_fft_module = fft_module


def _real_2d_array(xp: Any, name: str, value: Any) -> Any:
    array = xp.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be a real numeric array")
    return xp.asarray(array, dtype=xp.float64)
