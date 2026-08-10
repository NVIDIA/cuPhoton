# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interpolation and mask-propagation helpers for xRep."""

from __future__ import annotations

import numpy as np


def sample_bilinear_array(
    source: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fill_value: float = float("nan"),
    eps: float = 1e-12,
) -> np.ndarray:
    """Sample one source image at floating-point coordinates bilinearly."""

    height, width = source.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    in00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    in10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    in01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    in11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)

    w00 = np.where(in00, w00, 0.0)
    w10 = np.where(in10, w10, 0.0)
    w01 = np.where(in01, w01, 0.0)
    w11 = np.where(in11, w11, 0.0)
    norm = w00 + w10 + w01 + w11

    cx0 = np.clip(x0, 0, width - 1)
    cy0 = np.clip(y0, 0, height - 1)
    cx1 = np.clip(x1, 0, width - 1)
    cy1 = np.clip(y1, 0, height - 1)

    source64 = np.asarray(source, dtype=np.float64)
    value = np.zeros_like(norm, dtype=np.float64)
    for weight, sample in (
        (w00, source64[cy0, cx0]),
        (w10, source64[cy0, cx1]),
        (w01, source64[cy1, cx0]),
        (w11, source64[cy1, cx1]),
    ):
        contribution = np.zeros_like(value)
        np.multiply(weight, sample, out=contribution, where=weight != 0.0)
        value += contribution
    out = np.full_like(value, fill_value, dtype=np.float64)
    np.divide(value, norm, out=out, where=norm > eps)
    return out


def sample_bilinear_variance_array(
    source_variance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fill_value: float = float("nan"),
    eps: float = 1e-12,
) -> np.ndarray:
    """Propagate independent variances through bilinear interpolation."""

    height, width = source_variance.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    in00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    in10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    in01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    in11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)

    w00 = np.where(in00, w00, 0.0)
    w10 = np.where(in10, w10, 0.0)
    w01 = np.where(in01, w01, 0.0)
    w11 = np.where(in11, w11, 0.0)
    norm = w00 + w10 + w01 + w11

    cx0 = np.clip(x0, 0, width - 1)
    cy0 = np.clip(y0, 0, height - 1)
    cx1 = np.clip(x1, 0, width - 1)
    cy1 = np.clip(y1, 0, height - 1)

    variance64 = np.asarray(source_variance, dtype=np.float64)
    value = np.zeros_like(norm, dtype=np.float64)
    for weight, sample in (
        (w00, variance64[cy0, cx0]),
        (w10, variance64[cy0, cx1]),
        (w01, variance64[cy1, cx0]),
        (w11, variance64[cy1, cx1]),
    ):
        contribution = np.zeros_like(value)
        np.multiply(
            np.square(weight),
            sample,
            out=contribution,
            where=weight != 0.0,
        )
        value += contribution
    out = np.full_like(value, fill_value, dtype=np.float64)
    np.divide(value, np.square(norm), out=out, where=norm > eps)
    return out


def sample_lanczos_array(
    source: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    a: int = 3,
    two_a_footprint: bool = True,
    fill_value: float = float("nan"),
    eps: float = 1e-12,
) -> np.ndarray:
    """Sample one source image at floating-point coordinates with Lanczos."""

    if source.ndim != 2:
        raise ValueError("source must be 2D")

    height, width = source.shape
    source64 = np.asarray(source, dtype=np.float64)
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    fx = x - i0
    fy = y - j0

    start = -(a - 1) if two_a_footprint else -a
    win = 2 * a if two_a_footprint else (2 * a + 1)
    x_weights: list[np.ndarray] = []
    y_weights: list[np.ndarray] = []
    sum_wx = np.zeros_like(x, dtype=np.float64)
    sum_wy = np.zeros_like(y, dtype=np.float64)

    for off in range(start, start + win):
        xx = i0 + off
        valid = (xx >= 0) & (xx < width)
        weight = np.where(valid, _lanczos_weight(off - fx, a), 0.0)
        x_weights.append(weight)
        sum_wx += weight

    for off in range(start, start + win):
        yy = j0 + off
        valid = (yy >= 0) & (yy < height)
        weight = np.where(valid, _lanczos_weight(off - fy, a), 0.0)
        y_weights.append(weight)
        sum_wy += weight

    norm = sum_wx * sum_wy
    numerator = np.zeros_like(x, dtype=np.float64)

    for iy, off_y in enumerate(range(start, start + win)):
        yy = j0 + off_y
        valid_y = (yy >= 0) & (yy < height)
        cy = np.clip(yy, 0, height - 1)
        wy = y_weights[iy]
        if not np.any(wy):
            continue

        for ix, off_x in enumerate(range(start, start + win)):
            xx = i0 + off_x
            valid_x = (xx >= 0) & (xx < width)
            weight = wy * x_weights[ix]
            valid = valid_y & valid_x & (weight != 0.0)
            if not np.any(valid):
                continue
            cx = np.clip(xx, 0, width - 1)
            contribution = np.zeros_like(numerator)
            np.multiply(
                weight,
                source64[cy, cx],
                out=contribution,
                where=valid,
            )
            numerator += contribution

    out = np.full_like(numerator, fill_value, dtype=np.float64)
    np.divide(numerator, norm, out=out, where=norm > eps)
    return out


def sample_lanczos_variance_array(
    source_variance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    a: int = 3,
    two_a_footprint: bool = True,
    fill_value: float = float("nan"),
    eps: float = 1e-12,
) -> np.ndarray:
    """Propagate independent variances through normalized Lanczos weights."""

    if source_variance.ndim != 2:
        raise ValueError("source_variance must be 2D")

    height, width = source_variance.shape
    variance64 = np.asarray(source_variance, dtype=np.float64)
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    fx = x - i0
    fy = y - j0

    start = -(a - 1) if two_a_footprint else -a
    win = 2 * a if two_a_footprint else (2 * a + 1)
    x_weights: list[np.ndarray] = []
    y_weights: list[np.ndarray] = []
    sum_wx = np.zeros_like(x, dtype=np.float64)
    sum_wy = np.zeros_like(y, dtype=np.float64)

    for off in range(start, start + win):
        xx = i0 + off
        valid = (xx >= 0) & (xx < width)
        weight = np.where(valid, _lanczos_weight(off - fx, a), 0.0)
        x_weights.append(weight)
        sum_wx += weight

    for off in range(start, start + win):
        yy = j0 + off
        valid = (yy >= 0) & (yy < height)
        weight = np.where(valid, _lanczos_weight(off - fy, a), 0.0)
        y_weights.append(weight)
        sum_wy += weight

    norm = sum_wx * sum_wy
    numerator = np.zeros_like(x, dtype=np.float64)

    for iy, off_y in enumerate(range(start, start + win)):
        yy = j0 + off_y
        valid_y = (yy >= 0) & (yy < height)
        cy = np.clip(yy, 0, height - 1)
        wy = y_weights[iy]
        if not np.any(wy):
            continue

        for ix, off_x in enumerate(range(start, start + win)):
            xx = i0 + off_x
            valid_x = (xx >= 0) & (xx < width)
            weight = wy * x_weights[ix]
            valid = valid_y & valid_x & (weight != 0.0)
            if not np.any(valid):
                continue
            cx = np.clip(xx, 0, width - 1)
            contribution = np.zeros_like(numerator)
            np.multiply(
                np.square(weight),
                variance64[cy, cx],
                out=contribution,
                where=valid,
            )
            numerator += contribution

    out = np.full_like(numerator, fill_value, dtype=np.float64)
    np.divide(
        numerator,
        np.square(norm),
        out=out,
        where=norm > eps,
    )
    return out


def propagate_mask_or(
    source_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    interpolation: str = "bilinear",
    lanczos_a: int = 3,
    two_a_footprint: bool = True,
    invalid_mask_value: int | bool = 1,
) -> np.ndarray:
    """OR mask bits from every nonzero interpolation contributor."""

    mask = np.asarray(source_mask)
    if mask.ndim != 2:
        raise ValueError("source_mask must be 2D")
    if interpolation not in {"bilinear", "lanczos3"}:
        raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")

    height, width = mask.shape
    finite = np.isfinite(x) & np.isfinite(y)
    safe_x = np.where(finite, x, 0.0)
    safe_y = np.where(finite, y, 0.0)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    fx = safe_x - x0
    fy = safe_y - y0

    if interpolation == "bilinear":
        x_contributors = ((0, (1.0 - fx) != 0.0), (1, fx != 0.0))
        y_contributors = ((0, (1.0 - fy) != 0.0), (1, fy != 0.0))
    else:
        start = -(lanczos_a - 1) if two_a_footprint else -lanczos_a
        window = 2 * lanczos_a if two_a_footprint else 2 * lanczos_a + 1
        x_contributors = tuple(
            (
                offset,
                _lanczos_weight_is_nonzero(offset - fx, lanczos_a),
            )
            for offset in range(start, start + window)
        )
        y_contributors = tuple(
            (
                offset,
                _lanczos_weight_is_nonzero(offset - fy, lanczos_a),
            )
            for offset in range(start, start + window)
        )

    out = np.zeros(x.shape, dtype=mask.dtype)
    for offset_y, nonzero_y in y_contributors:
        yy = y0 + offset_y
        cy = np.clip(yy, 0, height - 1)
        valid_y = nonzero_y & (yy >= 0) & (yy < height)
        for offset_x, nonzero_x in x_contributors:
            xx = x0 + offset_x
            cx = np.clip(xx, 0, width - 1)
            contributes = finite & valid_y & nonzero_x
            contributes &= (xx >= 0) & (xx < width)
            out |= np.where(
                contributes,
                mask[cy, cx],
                np.zeros_like(mask[cy, cx]),
            )

    invalid_value = np.asarray(invalid_mask_value, dtype=mask.dtype)

    invalid = ~(
        finite
        & (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )
    return np.where(invalid, out | invalid_value, out).astype(mask.dtype)


def _lanczos_weight_is_nonzero(t: np.ndarray, a: int) -> np.ndarray:
    """Return the mathematical nonzero support of a Lanczos weight."""

    within_support = np.abs(t) < float(a)
    sinc_nonzero = (t == 0.0) | (t != np.trunc(t))
    return within_support & sinc_nonzero


def _sincpi(t: np.ndarray) -> np.ndarray:
    result = np.ones_like(t, dtype=np.float64)
    integer_zero = (t != 0.0) & (t == np.trunc(t))
    result[integer_zero] = 0.0
    mask = (np.abs(t) >= 1e-18) & ~integer_zero
    result[mask] = np.sin(np.pi * t[mask]) / (np.pi * t[mask])
    return result


def _lanczos_weight(t: np.ndarray, a: int) -> np.ndarray:
    abs_t = np.abs(t)
    out = np.zeros_like(t, dtype=np.float64)
    mask = abs_t <= float(a)
    if np.any(mask):
        out[mask] = _sincpi(t[mask]) * _sincpi(t[mask] / float(a))
    return out
