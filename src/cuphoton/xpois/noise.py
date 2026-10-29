# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Marginal noise diagnostics for fixed image-subtraction kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .ois import _default_fit_mask_impl
from .stamps import _array_module, _ndimage_module


@dataclass(frozen=True)
class ConstantKernelNoiseResult:
    """Marginal diagonal noise diagnostics for a fixed-kernel residual.

    The scope fields are fixed metadata: they are excluded from the
    constructor and always carry the documented values. The arrays are
    NumPy or CuPy, following the inputs.

    Attributes
    ----------
    propagated_reference_variance
        Reference variance convolved with the squared fixed kernel by direct
        summation. Invalid output pixels are represented by ``NaN``.
    marginal_residual_variance
        Sum of target variance and propagated reference variance. This is a
        per-pixel marginal variance, not a covariance model.
    standardized_residual
        ``residual / sqrt(marginal_residual_variance)`` on ``valid_mask`` and
        ``NaN`` elsewhere. This diagonal standardization does not whiten the
        residual.
    valid_mask
        Pixels with a complete finite reference-variance footprint, positive
        finite target variance, finite residual, and caller authorization.
    variance_scope
        Records that only marginal diagonal variance is represented.
    kernel_treatment
        Records that the supplied kernel is treated as fixed.
    covariance_represented
        Always false: neither spatial nor cross-image covariance is modeled.
    fit_parameter_uncertainty_represented
        Always false: uncertainty in the fitted kernel and background is not
        propagated.
    """

    propagated_reference_variance: Any
    marginal_residual_variance: Any
    standardized_residual: Any
    valid_mask: Any
    variance_scope: Literal["marginal_diagonal"] = field(
        default="marginal_diagonal", init=False
    )
    kernel_treatment: Literal["fixed"] = field(default="fixed", init=False)
    covariance_represented: Literal[False] = field(default=False, init=False)
    fit_parameter_uncertainty_represented: Literal[False] = field(
        default=False, init=False
    )


def standardize_constant_kernel_residual(
    residual: Any,
    *,
    kernel: Any,
    target_variance: Any,
    reference_variance: Any,
    valid_mask: Any | None = None,
) -> ConstantKernelNoiseResult:
    """Propagate diagonal noise through a fixed convolution kernel.

    The residual convention is ``target - (reference ⊗ kernel + background)``.
    Assuming independent input pixels and independent reference and target
    images, the represented marginal variance is

    ``target_variance + reference_variance ⊗ kernel**2``.

    ``reference_variance`` is expressed in squared reference-image units. The
    kernel maps reference pixels into target units, so its square also applies
    any photometric scaling; the kernel is not required to sum to one.

    The calculation treats ``kernel`` as fixed. It does not change solver
    weights or chi-square, and it omits spatial covariance, reference-target
    covariance, resampling covariance, and fitted-parameter uncertainty.
    Consequently, ``standardized_residual`` is a marginal diagnostic rather
    than a whitened residual or calibrated significance image.

    Parameters
    ----------
    residual
        Two-dimensional fixed-kernel residual in target pixel units.
    kernel
        Finite, odd-shaped two-dimensional convolution kernel.
    target_variance
        Target variance image in squared target units. Nonpositive or
        non-finite pixels are excluded from the returned valid mask.
    reference_variance
        Reference variance image in squared reference units. Values must be
        nonnegative where finite. An output is excluded if any pixel in its
        complete rectangular kernel footprint is non-finite. Propagation is a
        direct sum, so a large finite value only affects outputs whose
        footprint contains it; outputs whose propagated variance overflows
        are excluded.
    valid_mask
        Optional boolean or binary ``0``/``1`` output-pixel selection, the
        same contract as the ``fit_mask`` of :func:`solve_constant_kernel`.
        Kernel-boundary pixels are excluded even when selected because they
        lack a complete reference footprint.

    Returns
    -------
    ConstantKernelNoiseResult
        Full-sized arrays (read-only on NumPy) and explicit
        statistical-scope metadata.

    Notes
    -----
    The computation runs in CuPy when any plane or the kernel is a CuPy
    array, and in NumPy otherwise, with no host round trip; NumPy inputs
    are moved to the device in the first case. The convolutions are
    ``scipy.ndimage`` or ``cupyx.scipy.ndimage`` direct sums.
    """

    xp = _array_module(residual, kernel, target_variance, reference_variance)
    residual_arr = _real_2d_array(xp, "residual", residual)
    kernel_arr = _real_2d_array(xp, "kernel", kernel)
    target_variance_arr = _real_2d_array(
        xp, "target_variance", target_variance
    )
    reference_variance_arr = _real_2d_array(
        xp, "reference_variance", reference_variance
    )

    if target_variance_arr.shape != residual_arr.shape:
        raise ValueError("target_variance must match residual shape")
    if reference_variance_arr.shape != residual_arr.shape:
        raise ValueError("reference_variance must match residual shape")
    if any(size % 2 == 0 for size in kernel_arr.shape):
        raise ValueError("kernel dimensions must be odd")
    if any(
        kernel_size > image_size
        for kernel_size, image_size in zip(
            kernel_arr.shape, residual_arr.shape, strict=True
        )
    ):
        raise ValueError("kernel must fit within the residual image")
    if not bool(xp.isfinite(kernel_arr).all()):
        raise ValueError("kernel must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        kernel_squared = kernel_arr**2
    if not bool(xp.isfinite(kernel_squared).all()):
        raise ValueError("squared kernel values must remain finite")
    finite_negative_reference = xp.isfinite(reference_variance_arr) & (
        reference_variance_arr < 0.0
    )
    if bool(xp.any(finite_negative_reference)):
        raise ValueError(
            "reference_variance must be nonnegative where finite"
        )

    selected = _coerce_valid_mask(xp, valid_mask, residual_arr.shape)
    complete_footprint = _default_fit_mask_impl(
        xp, residual_arr.shape, kernel_arr.shape
    )
    reference_finite = xp.isfinite(reference_variance_arr)
    reference_footprint_valid = _footprint_all_true(
        xp, reference_finite, kernel_arr.shape
    )
    output_valid = (
        selected
        & complete_footprint
        & reference_footprint_valid
        & xp.isfinite(residual_arr)
        & xp.isfinite(target_variance_arr)
        & (target_variance_arr > 0.0)
    )

    reference_work = xp.where(reference_finite, reference_variance_arr, 0.0)
    propagated = _convolve_same(xp, reference_work, kernel_squared)
    propagated_finite = xp.isfinite(propagated)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        marginal = target_variance_arr + propagated
        standardized = residual_arr / xp.sqrt(marginal)
    output_valid &= propagated_finite
    output_valid &= xp.isfinite(marginal) & (marginal > 0.0)
    output_valid &= xp.isfinite(standardized)
    if not bool(xp.any(output_valid)):
        raise ValueError("no valid output pixels remain")

    return ConstantKernelNoiseResult(
        propagated_reference_variance=_readonly(
            xp.where(output_valid, propagated, xp.nan)
        ),
        marginal_residual_variance=_readonly(
            xp.where(output_valid, marginal, xp.nan)
        ),
        standardized_residual=_readonly(
            xp.where(output_valid, standardized, xp.nan)
        ),
        valid_mask=_readonly(output_valid),
    )


def _real_2d_array(xp: Any, name: str, value: Any) -> Any:
    array = xp.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.timedelta64)
    ):
        raise TypeError(f"{name} must be a real numeric array")
    return xp.asarray(array, dtype=xp.float64)


def _coerce_valid_mask(
    xp: Any,
    valid_mask: Any | None,
    shape: tuple[int, int],
) -> Any:
    if valid_mask is None:
        return xp.ones(shape, dtype=bool)
    mask = xp.asarray(valid_mask)
    if mask.shape != shape:
        raise ValueError("valid_mask must match residual shape")
    if mask.dtype == bool:
        return mask.copy()
    if not np.issubdtype(mask.dtype, np.number):
        raise ValueError("valid_mask array must be boolean or binary numeric")
    if not bool(xp.isfinite(mask).all()):
        raise ValueError(
            "valid_mask array must not contain NaN or inf values"
        )
    if not bool(xp.all((mask == 0) | (mask == 1))):
        raise ValueError("valid_mask array must contain only 0/1 values")
    return mask.astype(bool)


def _footprint_all_true(
    xp: Any,
    pixel_valid: Any,
    kernel_shape: tuple[int, int],
) -> Any:
    if bool(pixel_valid.all()):
        return _default_fit_mask_impl(xp, pixel_valid.shape, kernel_shape)
    return (
        _ndimage_module(xp).minimum_filter(
            pixel_valid.astype(xp.uint8),
            size=kernel_shape,
            mode="constant",
            cval=0,
        )
        > 0
    )


def _convolve_same(xp: Any, image: Any, kernel: Any) -> Any:
    """Zero-padded true convolution by direct summation.

    FFT convolution spreads round-off proportional to the largest value in
    the plane across every output pixel. Direct summation keeps a large or
    overflowing finite variance local to the outputs whose footprint
    contains it, and a sum of nonnegative terms stays nonnegative.
    """

    return xp.asarray(
        _ndimage_module(xp).convolve(
            image, kernel, mode="constant", cval=0.0
        ),
        dtype=xp.float64,
    )


def _readonly(value: Any) -> Any:
    if hasattr(value, "setflags"):
        value.setflags(write=False)
    return value
