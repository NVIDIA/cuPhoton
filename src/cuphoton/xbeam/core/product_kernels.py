# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy compilers for generic cross-product kernels."""

from functools import lru_cache
from typing import Any

from ..runtime.cuda import _raw_function
from .cuda_products import _MULTI_VISIBILITY_SOURCE, _VISIBILITY_SOURCE


@lru_cache(maxsize=8)
def _visibility_kernel(double_precision: bool, averaging_mode: int) -> Any:
    precision = "_double" if double_precision else "_float"
    suffix = (
        "_avg1"
        if averaging_mode == 1
        else (
            "_avg4"
            if averaging_mode == 4
            else ("_avg32" if averaging_mode == 32 else "_generic")
        )
    )
    return _raw_function(
        _VISIBILITY_SOURCE, "visibility_product" + precision + suffix
    )


@lru_cache(maxsize=16)
def _batched_visibility_kernel(
    averaging_mode: int,
    double_precision: bool = False,
    accumulate: bool = False,
) -> Any:
    suffix = (
        "_avg1"
        if averaging_mode == 1
        else (
            "_avg4"
            if averaging_mode == 4
            else ("_avg32" if averaging_mode == 32 else "_generic")
        )
    )
    definitions = "#define BATCHED_DOUBLE\n" if double_precision else ""
    if accumulate:
        definitions += "#define BATCHED_ACCUMULATE\n"
    precision = "double" if double_precision else "float"
    return _raw_function(
        definitions + _VISIBILITY_SOURCE,
        "batched_visibility_product_" + precision + suffix,
    )


@lru_cache(maxsize=64)
def _multi_visibility_kernel(
    band_limit: int = 8, product_limit: int = 8
) -> Any:
    """Bound local storage without changing metadata strides or reductions."""

    if not 1 <= band_limit <= 8 or not 2 <= product_limit <= 8:
        raise ValueError("invalid fused visibility specialization bounds")
    # A single-band array lets NVRTC prove both product operands identical
    # and fold the imaginary subtraction, changing legacy float rounding.
    # Preserve the generic arithmetic for this layout.
    if band_limit == 1:
        return _multi_visibility_kernel()
    source = (
        f"#define BAND_LIMIT {band_limit}\n"
        f"#define PRODUCT_LIMIT {product_limit}\n" + _MULTI_VISIBILITY_SOURCE
    )
    return _raw_function(source, "batched_visibility_products_float")
