# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValueDropResult:
    selected_index: int
    selected_time: float | None
    status: str
    change: str | None
    extrema_index: int | None
    extrema_time: float | None
    local_minima: int
    local_maxima: int
    polynomial_coefficients: tuple[float, float, float, float] | None


def find_value_drop_position(
    time,
    trace,
    *,
    zero_offset: float,
    min_local_minima: int = 11,
) -> ValueDropResult:
    """Find the reference zero-offset index for a normalized trace.

    The reference script uses local extrema and a cubic fit to confirm a
    drop/rise-like shape, then returns the sample closest to the requested
    ``zero_offset``. This function keeps that behavior but makes degenerate
    cases explicit instead of raising from array shape surprises.
    """

    delay = np.asarray(time, dtype=np.float64)
    signal = np.asarray(trace, dtype=np.float64)
    _validate_inputs(delay, signal, min_local_minima)

    import scipy.optimize as opt
    from scipy.signal import argrelmax, argrelmin

    localmin = np.asarray(argrelmin(signal)[0], dtype=np.int64)
    localmax = np.asarray(argrelmax(signal)[0], dtype=np.int64)
    raw_local_minima = int(localmin.shape[0])
    raw_local_maxima = int(localmax.shape[0])
    if raw_local_minima < min_local_minima or raw_local_maxima == 0:
        return _missing_result(
            "insufficient-extrema",
            local_minima=raw_local_minima,
            local_maxima=raw_local_maxima,
        )

    coefficients, _pcov = opt.curve_fit(_cubic, delay, signal)
    fit = tuple(float(value) for value in coefficients)
    drop = fit[3] < 0
    change = "drop" if drop else "rise"

    if drop:
        if (
            localmin.shape[0]
            and localmax.shape[0]
            and localmin[0] < localmax[0]
        ):
            localmin = localmin[1:]
    elif (
        localmin.shape[0] and localmax.shape[0] and localmin[0] > localmax[0]
    ):
        localmax = localmax[1:]

    pair_count = min(localmin.shape[0] // 10, localmax.shape[0])
    if pair_count <= 0:
        return _missing_result(
            "insufficient-paired-extrema",
            local_minima=raw_local_minima,
            local_maxima=raw_local_maxima,
            polynomial_coefficients=fit,
            change=change,
        )

    localmin = localmin[:pair_count]
    localmax = localmax[:pair_count]
    descent = signal[localmax] - signal[localmin]
    extrema_index = int(localmin[int(np.argmax(descent))])
    selected_index = int(np.argmin(np.abs(delay - float(zero_offset))))

    return ValueDropResult(
        selected_index=selected_index,
        selected_time=float(delay[selected_index]),
        status="ok",
        change=change,
        extrema_index=extrema_index,
        extrema_time=float(delay[extrema_index]),
        local_minima=raw_local_minima,
        local_maxima=raw_local_maxima,
        polynomial_coefficients=fit,
    )


def _validate_inputs(delay, signal, min_local_minima: int) -> None:
    if delay.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and trace must be one-dimensional")
    if delay.shape != signal.shape:
        raise ValueError("time and trace must have the same shape")
    if delay.shape[0] == 0:
        raise ValueError("time and trace must contain at least 1 sample")
    if min_local_minima < 1:
        raise ValueError("min_local_minima must be positive")
    if not np.all(np.isfinite(delay)) or not np.all(np.isfinite(signal)):
        raise ValueError("time and trace must contain only finite values")


def _missing_result(
    status: str,
    *,
    local_minima: int,
    local_maxima: int,
    polynomial_coefficients: tuple[float, float, float, float] | None = None,
    change: str | None = None,
) -> ValueDropResult:
    return ValueDropResult(
        selected_index=-1,
        selected_time=None,
        status=status,
        change=change,
        extrema_index=None,
        extrema_time=None,
        local_minima=int(local_minima),
        local_maxima=int(local_maxima),
        polynomial_coefficients=polynomial_coefficients,
    )


def _cubic(x, a0, a1, a2, a3):
    return a3 * x * x * x + a2 * x * x + a1 * x + a0
