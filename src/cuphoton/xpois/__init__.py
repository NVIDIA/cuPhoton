# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Image-subtraction tools in the :mod:`cuphoton.xpois` namespace."""

from cuphoton import __version__

from .correlation import (
    CorrelationFactorResult,
    EmpiricalCorrelationFactor,
    correlation_factor,
    empirical_correlation_factor,
    psf_weights,
    resampling_variance_factor,
)
from .data import inspect_hsc_data_tree, load_image_array
from .kernel_uncertainty import (
    KernelFitCovariance,
    kernel_fit_covariance,
    kernel_fit_variance,
)
from .measurement import empty_positions, forced_flux_at, kernel_centroid
from .noise import (
    ConstantKernelNoiseResult,
    standardize_constant_kernel_residual,
)
from .ois import (
    AutoStampMaskResult,
    BasisTerm,
    ConstantKernelFitResult,
    DeviceConstantKernelFitResult,
    GaussianBasisComponent,
    SeparableKernelFitResult,
    background_design,
    build_compact_source_stamp_mask,
    build_gaussian_polynomial_basis,
    evaluate_background_model,
    make_stamp_mask,
    resolve_backend,
    solve_constant_kernel,
    solve_constant_kernel_device,
    solve_separable_kernel,
    triangular_degree_pairs,
)
from .review import identify_residual_hotspots, residual_hotspot_table
from .spatial_als import (
    SpatialALSConfig,
    SpatialALSFitResult,
    solve_spatial_als,
)
from .stamps import (
    extract_centered_stamp,
    simple_difference_stamp,
    window_stamps,
)
from .statistics import (
    StandardizedResidualStatistics,
    StandardizedResidualSummary,
    summarize_standardized_residuals,
)

__all__ = [
    "__version__",
    "AutoStampMaskResult",
    "BasisTerm",
    "ConstantKernelFitResult",
    "ConstantKernelNoiseResult",
    "CorrelationFactorResult",
    "DeviceConstantKernelFitResult",
    "EmpiricalCorrelationFactor",
    "GaussianBasisComponent",
    "KernelFitCovariance",
    "SeparableKernelFitResult",
    "SpatialALSConfig",
    "SpatialALSFitResult",
    "StandardizedResidualStatistics",
    "StandardizedResidualSummary",
    "background_design",
    "build_compact_source_stamp_mask",
    "build_gaussian_polynomial_basis",
    "correlation_factor",
    "empirical_correlation_factor",
    "empty_positions",
    "evaluate_background_model",
    "extract_centered_stamp",
    "forced_flux_at",
    "identify_residual_hotspots",
    "inspect_hsc_data_tree",
    "kernel_centroid",
    "kernel_fit_covariance",
    "kernel_fit_variance",
    "load_image_array",
    "make_stamp_mask",
    "psf_weights",
    "resampling_variance_factor",
    "residual_hotspot_table",
    "resolve_backend",
    "simple_difference_stamp",
    "solve_constant_kernel",
    "solve_constant_kernel_device",
    "solve_separable_kernel",
    "solve_spatial_als",
    "standardize_constant_kernel_residual",
    "summarize_standardized_residuals",
    "triangular_degree_pairs",
    "window_stamps",
]
