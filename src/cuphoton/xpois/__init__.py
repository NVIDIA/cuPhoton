# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Image-subtraction tools in the :mod:`cuphoton.xpois` namespace."""

from cuphoton import __version__

from .data import inspect_hsc_data_tree, load_image_array
from .ois import (
    AutoStampMaskResult,
    BasisTerm,
    ConstantKernelFitResult,
    GaussianBasisComponent,
    SeparableKernelFitResult,
    background_design,
    build_compact_source_stamp_mask,
    build_gaussian_polynomial_basis,
    evaluate_background_model,
    make_stamp_mask,
    resolve_backend,
    solve_constant_kernel,
    solve_separable_kernel,
    triangular_degree_pairs,
)
from .stamps import extract_centered_stamp, simple_difference_stamp

__all__ = [
    "__version__",
    "AutoStampMaskResult",
    "BasisTerm",
    "ConstantKernelFitResult",
    "GaussianBasisComponent",
    "SeparableKernelFitResult",
    "background_design",
    "build_compact_source_stamp_mask",
    "build_gaussian_polynomial_basis",
    "evaluate_background_model",
    "extract_centered_stamp",
    "inspect_hsc_data_tree",
    "load_image_array",
    "make_stamp_mask",
    "resolve_backend",
    "simple_difference_stamp",
    "solve_constant_kernel",
    "solve_separable_kernel",
    "triangular_degree_pairs",
]
