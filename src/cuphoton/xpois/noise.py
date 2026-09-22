# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Marginal noise diagnostics for fixed image-subtraction kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .ois import _coerce_boolean_mask, _default_fit_mask


@dataclass(frozen=True)
class ConstantKernelNoiseResult:
    """Marginal diagonal noise diagnostics for a fixed-kernel residual.

    The scope fields are fixed metadata: they are excluded from the
    constructor and always carry the documented values.

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

    propagated_reference_variance: np.ndarray
    marginal_residual_variance: np.ndarray
    standardized_residual: np.ndarray
    valid_mask: np.ndarray
    variance_scope: Literal["marginal_diagonal"] = field(
        default="marginal_diagonal", init=False
    )
    kernel_treatment: Literal["fixed"] = field(default="fixed", init=False)
    covariance_represented: Literal[False] = field(default=False, init=False)
    fit_parameter_uncertainty_represented: Literal[False] = field(
        default=False, init=False
    )


def standardize_constant_kernel_residual(
    residual: np.ndarray,
    *,
    kernel: np.ndarray,
    target_variance: np.ndarray,
    reference_variance: np.ndarray,
    valid_mask: np.ndarray | None = None,
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
        Full-sized read-only arrays and explicit statistical-scope metadata.
    """

    residual_arr = _real_2d_array("residual", residual)
    kernel_arr = _real_2d_array("kernel", kernel)
    target_variance_arr = _real_2d_array("target_variance", target_variance)
    reference_variance_arr = _real_2d_array(
        "reference_variance", reference_variance
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
    if not np.isfinite(kernel_arr).all():
        raise ValueError("kernel must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        kernel_squared = kernel_arr**2
    if not np.isfinite(kernel_squared).all():
        raise ValueError("squared kernel values must remain finite")
    finite_negative_reference = np.isfinite(reference_variance_arr) & (
        reference_variance_arr < 0.0
    )
    if np.any(finite_negative_reference):
        raise ValueError(
            "reference_variance must be nonnegative where finite"
        )

    selected = _coerce_valid_mask(valid_mask, residual_arr.shape)
    complete_footprint = _default_fit_mask(
        residual_arr.shape, kernel_arr.shape
    )
    reference_finite = np.isfinite(reference_variance_arr)
    reference_footprint_valid = _footprint_all_true(
        reference_finite, kernel_arr.shape
    )
    output_valid = (
        selected
        & complete_footprint
        & reference_footprint_valid
        & np.isfinite(residual_arr)
        & np.isfinite(target_variance_arr)
        & (target_variance_arr > 0.0)
    )

    reference_work = np.where(reference_finite, reference_variance_arr, 0.0)
    propagated = _convolve_same(reference_work, kernel_squared)
    propagated_finite = np.isfinite(propagated)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        marginal = target_variance_arr + propagated
        standardized = residual_arr / np.sqrt(marginal)
    output_valid &= propagated_finite
    output_valid &= np.isfinite(marginal) & (marginal > 0.0)
    output_valid &= np.isfinite(standardized)
    if not np.any(output_valid):
        raise ValueError("no valid output pixels remain")

    propagated_output = np.full(residual_arr.shape, np.nan, dtype=np.float64)
    marginal_output = np.full(residual_arr.shape, np.nan, dtype=np.float64)
    standardized_output = np.full(
        residual_arr.shape, np.nan, dtype=np.float64
    )
    propagated_output[output_valid] = propagated[output_valid]
    marginal_output[output_valid] = marginal[output_valid]
    standardized_output[output_valid] = standardized[output_valid]

    return ConstantKernelNoiseResult(
        propagated_reference_variance=_readonly(propagated_output),
        marginal_residual_variance=_readonly(marginal_output),
        standardized_residual=_readonly(standardized_output),
        valid_mask=_readonly(output_valid),
    )


def _real_2d_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
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
    return np.asarray(array, dtype=np.float64)


def _coerce_valid_mask(
    valid_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(valid_mask)
    if mask.shape != shape:
        raise ValueError("valid_mask must match residual shape")
    return _coerce_boolean_mask(mask, name="valid_mask").copy()


def _footprint_all_true(
    pixel_valid: np.ndarray,
    kernel_shape: tuple[int, int],
) -> np.ndarray:
    from scipy.ndimage import minimum_filter

    if pixel_valid.all():
        return _default_fit_mask(pixel_valid.shape, kernel_shape)
    return np.asarray(
        minimum_filter(
            pixel_valid.astype(np.uint8),
            size=kernel_shape,
            mode="constant",
            cval=0,
        ),
        dtype=bool,
    )


def _convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-padded true convolution by direct summation.

    FFT convolution spreads round-off proportional to the largest value in
    the plane across every output pixel. Direct summation keeps a large or
    overflowing finite variance local to the outputs whose footprint
    contains it, and a sum of nonnegative terms stays nonnegative.
    """

    from scipy.ndimage import convolve

    return np.asarray(
        convolve(image, kernel, mode="constant", cval=0.0), dtype=np.float64
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value
