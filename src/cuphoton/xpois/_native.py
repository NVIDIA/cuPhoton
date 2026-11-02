# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ahead-of-time constant-kernel fitting with CuPy buffer ownership."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cuphoton.core._native_imaging import asarray, extension

from .ois import (
    DeviceConstantKernelFitResult,
    GaussianBasisComponent,
    build_gaussian_polynomial_basis,
    triangular_degree_pairs,
)


def solve_constant_kernel_device(
    reference: Any,
    target: Any,
    components: Sequence[GaussianBasisComponent],
    *,
    kernel_shape: tuple[int, int] = (15, 15),
    variance: Any | None = None,
    fit_mask: Any | None = None,
    background_degree: int = 0,
    flux_conserve: bool = False,
    flux_reference_index: int = 0,
) -> DeviceConstantKernelFitResult:
    """Solve on the active device without CuPy-generated GPU programs.

    The small Gaussian basis is constructed by the shared CPU routine. Image
    preparation, weighted normal equations, SVD conditioning, LU solve,
    convolution and diagnostics execute in the native implementation.
    CuPy supplies allocation, transfers and CUDA Array Interface ownership.
    Both kernel dimensions must be greater than one; length-one kernels
    remain available through the CuPy backend.
    """
    import cupy as cp
    from cupyx.scipy.fft import next_fast_len

    native = extension()
    active_device = int(cp.cuda.runtime.getDevice())
    for name, value in (
        ("reference", reference),
        ("target", target),
        ("variance", variance),
        ("fit_mask", fit_mask),
    ):
        if isinstance(value, cp.ndarray) and value.device.id != active_device:
            raise ValueError(f"{name} must reside on the active CUDA device")

    reference_array = asarray(reference, np.float64)
    target_array = asarray(target, np.float64)
    if reference_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("reference and target must be 2D arrays")
    if reference_array.shape != target_array.shape:
        raise ValueError("reference and target must share the same shape")
    shape = target_array.shape
    variance_array = (
        None if variance is None else asarray(variance, np.float64)
    )
    if variance_array is not None and variance_array.shape != shape:
        raise ValueError("variance must match the image shape")
    mask_array = None
    if fit_mask is not None:
        mask_dtype = np.dtype(
            fit_mask.dtype
            if hasattr(fit_mask, "dtype")
            else np.asarray(fit_mask).dtype
        )
        if mask_dtype.kind not in "buif":
            raise ValueError(
                "native fit_mask must be boolean or real binary numeric"
            )
        mask_array = asarray(
            fit_mask, np.bool_ if mask_dtype.kind == "b" else np.float64
        )
        if mask_array.shape != shape:
            raise ValueError("fit_mask must match the image shape")
    if background_degree < 0:
        raise ValueError("background degree must be non-negative")
    background_count = len(triangular_degree_pairs(background_degree))
    basis_host, basis_terms = build_gaussian_polynomial_basis(
        kernel_shape,
        components,
        flux_conserve=flux_conserve,
        flux_reference_index=flux_reference_index,
    )
    if 1 in basis_host.shape[1:]:
        raise ValueError(
            "native xPois requires both kernel dimensions greater than one; "
            "use backend='cupy' for length-one kernels"
        )
    basis = asarray(basis_host)
    basis_count = basis.shape[0]
    coefficients = cp.empty(basis_count + background_count, dtype=cp.float64)
    kernel = cp.empty(basis.shape[1:], dtype=cp.float64)
    matched = cp.empty(shape, dtype=cp.float64)
    residual = cp.empty(shape, dtype=cp.float64)
    mask = cp.empty(shape, dtype=cp.bool_)
    background = cp.empty(shape, dtype=cp.float64)
    fft_height = next_fast_len(shape[0] + kernel.shape[0] - 1, real=True)
    fft_width = next_fast_len(shape[1] + kernel.shape[1] - 1, real=True)
    receipt = native.solve_constant_kernel(
        (reference_array, target_array, variance_array, mask_array, basis),
        (kernel, matched, residual, mask, background, coefficients),
        background_degree,
        fft_height,
        fft_width,
        cp.cuda.get_current_stream().ptr,
    )
    return DeviceConstantKernelFitResult(
        kernel=kernel,
        matched=matched,
        residual=residual,
        fit_mask=mask,
        background=background,
        kernel_coefficients=coefficients[:basis_count],
        background_coefficients=coefficients[basis_count:],
        basis_kernels=basis,
        basis_terms=basis_terms,
        chi2=float(receipt["chi2"]),
        dof=int(receipt["fit_pixel_count"]) - coefficients.size,
        fit_pixel_count=int(receipt["fit_pixel_count"]),
        flux_conserve=bool(flux_conserve),
        device_id=active_device,
        backend="native",
    )
