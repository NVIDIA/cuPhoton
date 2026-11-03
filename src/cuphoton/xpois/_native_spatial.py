# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ahead-of-time execution of the spatial Gaussian-polynomial model."""

from __future__ import annotations

import numpy as np

from cuphoton.core._native_imaging import extension

from .ois import triangular_degree_pairs
from .spatial_gaussian_polynomial import (
    SpatialGaussianPolynomialKernelConfig,
    SpatialKernelDomain,
)


def solve_native_spatial(
    source: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    basis: np.ndarray,
    indices: np.ndarray,
    relative_precision: np.ndarray | None,
    domain: SpatialKernelDomain,
    config: SpatialGaussianPolynomialKernelConfig,
    output_indices: np.ndarray,
    source_variance: np.ndarray | None,
    diagnostic_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, int]:
    """Execute an already validated fit without CuPy-generated programs.

    CuPy owns allocations and transfers. Native CUDA performs the large
    calculations; the small QR factor uses the shared CPU condition check
    and triangular solve. All arrays returned to the public API are on the
    host, matching the existing CPU and CuPy contracts.
    """
    import cupy as cp

    from .spatial_gaussian_polynomial import (
        _gpu_design_chunk_size,
        _solve_scaled_qr,
    )

    photo_count = len(
        tuple(triangular_degree_pairs(config.photometric_degree))
    )
    shape_count = len(tuple(triangular_degree_pairs(config.shape_degree)))
    background_count = len(
        tuple(triangular_degree_pairs(config.background_degree))
    )
    kernel_columns = photo_count + (basis.shape[0] - 1) * shape_count
    column_count = kernel_columns + background_count
    if indices.size < column_count:
        raise ValueError(
            "fit region is underdetermined for the requested spatial "
            f"model: {indices.size} equations for {column_count} coefficients"
        )
    native = extension()
    if not hasattr(native, "accumulate_spatial_qr"):
        raise RuntimeError(
            "The installed native imaging extension lacks spatial fitting; "
            "rebuild it with CUPHOTON_IMAGING_BUILD_EXT=1"
        )
    chunk_size = _gpu_design_chunk_size(
        cp,
        row_count=max(
            int(indices.size),
            int(output_indices.size),
            0 if diagnostic_mask is None else int(diagnostic_mask.sum()),
        ),
        kernel_shape=basis.shape[1:],
        basis_count=basis.shape[0],
        photometric_term_count=photo_count,
        shape_term_count=shape_count,
        background_term_count=background_count,
    )
    source_device = cp.asarray(source, order="C")
    basis_device = cp.asarray(basis, order="C")
    upper_device = cp.empty((column_count, column_count), dtype=cp.float64)
    rhs_device = cp.empty(column_count, dtype=cp.float64)
    coordinates = (*domain.array_origin_yx, *domain.normalization_bbox)
    stream = cp.cuda.get_current_stream().ptr
    native.accumulate_spatial_qr(
        (
            source_device,
            cp.asarray(target, order="C"),
            cp.asarray(variance, order="C"),
            basis_device,
            cp.asarray(indices, dtype=cp.int64),
            None
            if relative_precision is None
            else cp.asarray(relative_precision, order="C"),
        ),
        (upper_device, rhs_device),
        coordinates,
        config.shape_degree,
        config.photometric_degree,
        config.background_degree,
        chunk_size,
        stream,
    )
    coefficients, condition = _solve_scaled_qr(
        cp.asnumpy(upper_device),
        cp.asnumpy(rhs_device),
        condition_limit=config.condition_limit,
        kernel_column_count=kernel_columns,
    )
    matched_device = cp.empty(source.shape, dtype=cp.float64)
    propagated_device = (
        None
        if source_variance is None
        else cp.empty(source.shape, dtype=cp.float64)
    )
    native.evaluate_spatial_kernel(
        (
            source_device,
            basis_device,
            cp.asarray(coefficients),
            cp.asarray(output_indices, dtype=cp.int64),
            None
            if source_variance is None
            else cp.asarray(source_variance, order="C"),
            None
            if diagnostic_mask is None
            else cp.asarray(np.flatnonzero(diagnostic_mask), dtype=cp.int64),
        ),
        (matched_device, propagated_device),
        coordinates,
        config.shape_degree,
        config.photometric_degree,
        config.background_degree,
        chunk_size,
        stream,
    )
    return (
        coefficients,
        cp.asnumpy(matched_device),
        None if propagated_device is None else cp.asnumpy(propagated_device),
        condition,
        chunk_size,
    )
