# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared array-level numerical primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def evaluate_polynomial(
    coefficients: Sequence[float], offset_s: Any, xp: Any
) -> Any:
    """Evaluate ascending-power coefficients using binary64 Horner form."""

    x = xp.asarray(offset_s, dtype=xp.float64)
    result = xp.zeros_like(x, dtype=xp.float64)
    for coefficient in reversed(coefficients):
        result = result * x + xp.float64(coefficient)
    return result


def reduce_cycles(cycles: Any, xp: Any) -> Any:
    """Reduce phase cycles to [-0.5, 0.5) before evaluating sin/cos."""

    value = xp.asarray(cycles, dtype=xp.float64)
    return value - xp.floor(value + xp.float64(0.5))


def cross_power(spectrum_a: Any, spectrum_b: Any, xp: Any) -> Any:
    return xp.asarray(spectrum_a) * xp.conj(xp.asarray(spectrum_b))


def accumulate_products(
    products: Any, block_weights: Any, xp: Any
) -> tuple[Any, Any]:
    """Accumulate product blocks and scalar simultaneous-validity weights."""

    values = xp.asarray(products)
    weights = xp.asarray(block_weights, dtype=xp.float64)
    if values.shape[0] != weights.shape[0]:
        raise ValueError(
            "one block weight is required for every product block"
        )
    return xp.sum(values, axis=0), xp.sum(weights, dtype=xp.float64)


def average_channels(spectrum: Any, channels_per_group: int, xp: Any) -> Any:
    if channels_per_group <= 0:
        raise ValueError("channels_per_group must be positive")
    values = xp.asarray(spectrum)
    count = values.shape[-1]
    if count % channels_per_group:
        raise ValueError("channel count is not divisible by averaging factor")
    grouped = values.reshape(
        *values.shape[:-1], count // channels_per_group, channels_per_group
    )
    return xp.mean(grouped, axis=-1)


def product_validity(
    weight_a: Any, weight_b: Any, *, autocorrelation: bool, xp: Any
) -> Any:
    first = xp.asarray(weight_a, dtype=xp.float64)
    if autocorrelation:
        return first
    return first * xp.asarray(weight_b, dtype=xp.float64)
