# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Descriptive spatial statistics for standardized subtraction residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .ois import _coerce_boolean_mask

_CARDINAL_LAG1 = ((0, 1), (1, 0))
_DIAGONAL_LAG1 = ((1, -1), (1, 1))
_RADIUS3_LAGS = tuple(
    (dy, dx)
    for dy in range(4)
    for dx in range(-3, 4)
    if not (dy == 0 and dx <= 0)
)


@dataclass(frozen=True)
class StandardizedResidualSummary:
    """Descriptive scalar and spatial statistics for centered residuals.

    The lag covariances and correlations use residuals centered by each
    stamp's valid-pixel mean. Each correlation divides the summed endpoint
    product by the mean of the two endpoint energies, which equals pooling
    both orientations of every lag and makes the value invariant to
    reflecting or transposing the stamps. They are empirical diagnostics,
    not a fitted covariance model.

    Attributes
    ----------
    valid_pixel_count
        Number of valid residual pixels contributing to the scalar metrics.
    standardized_rms
        Root mean square before per-stamp centering.
    demeaned_standardized_std
        Root mean square after subtracting each stamp's valid-pixel mean.
    cardinal_lag1_rho, cardinal_lag1_covariance
        Correlation and mean endpoint product pooled over horizontal and
        vertical lag-one pairs.
    cardinal_lag1_pair_count
        Number of valid cardinal endpoint pairs.
    diagonal_lag1_rho, diagonal_lag1_covariance
        Correlation and mean endpoint product pooled over the two unique
        diagonal lag-one directions.
    diagonal_lag1_pair_count
        Number of valid diagonal endpoint pairs.
    radius3_rho_rms, radius3_covariance_rms
        Unweighted root mean squares of the per-lag correlations and
        covariances over unique nonzero lags through Chebyshev radius three.
        White noise does not drive the correlation RMS to zero. For large
        stamps its sampling contribution is near ``sqrt(mean(1 / N_lag))``
        for per-lag pair counts ``N_lag``. Per-stamp centering also introduces
        negative correlations that persist when many small stamps are pooled;
        compare stamp sizes and masks as well as pair counts.
    radius3_unique_lag_count
        Number of unique radius-three lags, always 24.
    """

    valid_pixel_count: int
    standardized_rms: float
    demeaned_standardized_std: float
    cardinal_lag1_rho: float
    cardinal_lag1_covariance: float
    cardinal_lag1_pair_count: int
    diagonal_lag1_rho: float
    diagonal_lag1_covariance: float
    diagonal_lag1_pair_count: int
    radius3_rho_rms: float
    radius3_covariance_rms: float
    radius3_unique_lag_count: int = field(
        init=False,
        default=len(_RADIUS3_LAGS),
    )


@dataclass(frozen=True)
class StandardizedResidualStatistics:
    """Per-stamp and pixel-pooled descriptive residual statistics.

    ``pixel_pooled`` weights stamps by their valid pixels or valid lag
    endpoint pairs. It is not stamp-balanced. For equal stamp weighting,
    callers must aggregate one metric from each entry in ``per_stamp``.

    The input is assumed to have already been standardized. The summarizer
    centers stamps for its correlation diagnostics but performs no whitening,
    covariance inversion, uncertainty propagation, or significance
    calibration.
    """

    stamp_count: int
    stamp_shape: tuple[int, int]
    pixel_pooled: StandardizedResidualSummary
    per_stamp: tuple[StandardizedResidualSummary, ...]
    centering: Literal["per_stamp_valid_mean"] = "per_stamp_valid_mean"
    rho_definition: Literal["centered_endpoint_energy"] = (
        "centered_endpoint_energy"
    )
    variance_scope: Literal["caller_standardized"] = "caller_standardized"
    pixel_pooling: Literal["valid_pixels_and_lag_endpoint_pairs"] = (
        "valid_pixels_and_lag_endpoint_pairs"
    )
    whitening_applied: Literal[False] = False


def summarize_standardized_residuals(
    standardized_residual_stamps: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> StandardizedResidualStatistics:
    """Summarize spatial structure in standardized residual stamps.

    Parameters
    ----------
    standardized_residual_stamps
        Real numeric array with shape ``(stamp, y, x)``. Values selected by
        ``valid_mask`` must be finite. A single residual image can be supplied
        by adding a leading length-one stamp axis.
    valid_mask
        Optional same-shaped boolean or binary ``0``/``1`` array, the same
        contract as the ``fit_mask`` of :func:`solve_constant_kernel`. False
        pixels, including non-finite placeholders, are excluded from
        centering, scalar metrics, and both endpoints of every lag pair.

    Returns
    -------
    StandardizedResidualStatistics
        Per-stamp and pixel-pooled descriptive statistics.

    Notes
    -----
    Each stamp must define all requested correlations, including the 24 unique
    nonzero lags with Chebyshev radius at most three. Lags are treated as
    unordered endpoint pairs, so the statistics do not change when the stamp
    axes are reflected or transposed. Sparse, constant, non-finite, or
    numerically overflowing diagnostics fail closed rather than returning
    partial results or ``NaN``.

    ``radius3_rho_rms`` averages 24 noisy per-lag estimates without
    weighting, so its white-noise floor grows as per-lag pair counts shrink:
    Gaussian white-noise simulations give about 0.47 at the 4-by-4 minimum,
    where the corner lags rest on a single pair, 0.16 at 8-by-8, and 0.03
    at 32-by-32 for a complete stamp. Judge radius-3 structure only against
    an appropriate white-noise baseline.
    For many complete, independent white-noise stamps with ``N`` pixels
    each, per-stamp centering makes pooled lag correlations approach
    ``-1 / (N - 1)``. Pooling more stamps does not remove that contribution
    to the correlation RMS.
    """

    values, valid = _validated_inputs(
        standardized_residual_stamps,
        valid_mask,
    )
    centered = _center_per_stamp(values, valid)
    per_stamp = tuple(
        _summarize(
            values[index : index + 1],
            centered[index : index + 1],
            valid[index : index + 1],
            context=f"stamp {index}",
        )
        for index in range(values.shape[0])
    )
    pixel_pooled = _summarize(
        values,
        centered,
        valid,
        context="pooled stamps",
    )
    return StandardizedResidualStatistics(
        stamp_count=values.shape[0],
        stamp_shape=(values.shape[1], values.shape[2]),
        pixel_pooled=pixel_pooled,
        per_stamp=per_stamp,
    )


def _validated_inputs(
    residuals: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    residual_array = np.asarray(residuals)
    # np.number excludes bool but admits timedelta64 via signedinteger.
    if (
        not np.issubdtype(residual_array.dtype, np.number)
        or np.issubdtype(residual_array.dtype, np.complexfloating)
        or np.issubdtype(residual_array.dtype, np.timedelta64)
    ):
        raise TypeError(
            "standardized_residual_stamps must be a real numeric array"
        )
    if residual_array.ndim != 3:
        raise ValueError("standardized_residual_stamps must be a 3-D array")
    if residual_array.shape[0] == 0:
        raise ValueError(
            "standardized_residual_stamps must contain at least one stamp"
        )
    if residual_array.shape[1] < 4 or residual_array.shape[2] < 4:
        raise ValueError(
            "each residual stamp must be at least 4 by 4 for radius-3 lags"
        )

    if valid_mask is None:
        valid = np.ones(residual_array.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask)
        if mask.shape != residual_array.shape:
            raise ValueError("valid_mask must match residual shape")
        valid = _coerce_boolean_mask(mask, name="valid_mask")
    values = np.array(residual_array, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(values[valid]).all():
        raise ValueError(
            "standardized_residual_stamps must be finite on valid_mask"
        )
    valid_counts = np.count_nonzero(valid, axis=(1, 2))
    if np.any(valid_counts < 2):
        index = int(np.flatnonzero(valid_counts < 2)[0])
        raise ValueError(
            f"stamp {index} must contain at least two valid pixels"
        )
    return values, valid


def _center_per_stamp(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    counts = np.count_nonzero(valid, axis=(1, 2))
    with np.errstate(over="ignore", invalid="ignore"):
        sums = np.sum(np.where(valid, values, 0.0), axis=(1, 2))
        means = sums / counts
        centered = np.where(valid, values - means[:, None, None], 0.0)
    if not np.isfinite(means).all() or not np.isfinite(centered[valid]).all():
        raise ValueError("valid centered residuals must be finite")
    return centered


def _summarize(
    values: np.ndarray,
    centered: np.ndarray,
    valid: np.ndarray,
    *,
    context: str,
) -> StandardizedResidualSummary:
    selected = values[valid]
    centered_selected = centered[valid]
    # Test constancy on the raw values: the floating-point mean of a
    # non-dyadic constant does not reproduce it exactly, so the centered
    # residuals carry a rounding residue that would correlate perfectly.
    if selected.min() == selected.max():
        raise ValueError(
            f"{context} has zero centered residual variance; "
            "valid pixels are constant and correlation is undefined"
        )
    standardized_rms = _root_mean_square(
        selected,
        context=f"{context} standardized residuals",
    )
    demeaned_standardized_std = _root_mean_square(
        centered_selected,
        context=f"{context} centered standardized residuals",
    )
    if demeaned_standardized_std == 0.0:
        raise ValueError(
            f"{context} has zero centered residual variance; "
            "correlation is undefined"
        )

    cardinal_rho, cardinal_covariance, cardinal_count = _pair_statistics(
        centered,
        valid,
        _CARDINAL_LAG1,
        context=f"{context} cardinal lag-1",
    )
    diagonal_rho, diagonal_covariance, diagonal_count = _pair_statistics(
        centered,
        valid,
        _DIAGONAL_LAG1,
        context=f"{context} diagonal lag-1",
    )

    radius3_rhos: list[float] = []
    radius3_covariances: list[float] = []
    for lag in _RADIUS3_LAGS:
        rho, covariance, _ = _pair_statistics(
            centered,
            valid,
            (lag,),
            context=f"{context} radius-3 lag {lag}",
        )
        radius3_rhos.append(rho)
        radius3_covariances.append(covariance)

    radius3_rho_rms = _root_mean_square(
        np.asarray(radius3_rhos),
        context=f"{context} radius-3 correlations",
    )
    radius3_covariance_rms = _root_mean_square(
        np.asarray(radius3_covariances),
        context=f"{context} radius-3 covariances",
    )
    return StandardizedResidualSummary(
        valid_pixel_count=int(selected.size),
        standardized_rms=standardized_rms,
        demeaned_standardized_std=demeaned_standardized_std,
        cardinal_lag1_rho=cardinal_rho,
        cardinal_lag1_covariance=cardinal_covariance,
        cardinal_lag1_pair_count=cardinal_count,
        diagonal_lag1_rho=diagonal_rho,
        diagonal_lag1_covariance=diagonal_covariance,
        diagonal_lag1_pair_count=diagonal_count,
        radius3_rho_rms=radius3_rho_rms,
        radius3_covariance_rms=radius3_covariance_rms,
    )


def _pair_statistics(
    centered: np.ndarray,
    valid: np.ndarray,
    lags: tuple[tuple[int, int], ...],
    *,
    context: str,
) -> tuple[float, float, int]:
    product_sum = 0.0
    first_square_sum = 0.0
    second_square_sum = 0.0
    pair_count = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for dy, dx in lags:
            first_values, second_values = _shifted_views(centered, dy, dx)
            first_valid, second_valid = _shifted_views(valid, dy, dx)
            pair_valid = first_valid & second_valid
            first = first_values[pair_valid]
            second = second_values[pair_valid]
            # np.dot on 1-D float64 dispatches to BLAS ddot, which OpenBLAS
            # threads above roughly 10k elements at milliseconds per call;
            # einsum keeps these reductions single-threaded and cheap.
            product_sum += float(np.einsum("i,i->", first, second))
            first_square_sum += float(np.einsum("i,i->", first, first))
            second_square_sum += float(np.einsum("i,i->", second, second))
            pair_count += int(first.size)

    if pair_count == 0:
        raise ValueError(f"{context} has no valid endpoint pairs")
    if not np.isfinite(
        (product_sum, first_square_sum, second_square_sum)
    ).all():
        raise ValueError(f"{context} endpoint products must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        # The mean endpoint energy is symmetric in the two endpoints, so
        # pooled lag sets do not depend on which endpoint is labelled first.
        denominator = 0.5 * (first_square_sum + second_square_sum)
    if (
        first_square_sum < 0.0
        or second_square_sum < 0.0
        or denominator <= 0.0
    ):
        raise ValueError(
            f"{context} has invalid mean endpoint energy; "
            "correlation is undefined"
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rho = product_sum / denominator
        covariance = product_sum / pair_count
    if not np.isfinite((denominator, rho, covariance)).all():
        raise ValueError(f"{context} statistics must be finite")
    return float(np.clip(rho, -1.0, 1.0)), float(covariance), pair_count


def _shifted_views(
    array: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = array.shape[-2:]
    if dy >= 0:
        first_y = slice(0, height - dy)
        second_y = slice(dy, height)
    else:
        first_y = slice(-dy, height)
        second_y = slice(0, height + dy)
    if dx >= 0:
        first_x = slice(0, width - dx)
        second_x = slice(dx, width)
    else:
        first_x = slice(-dx, width)
        second_x = slice(0, width + dx)
    return (
        array[..., first_y, first_x],
        array[..., second_y, second_x],
    )


def _root_mean_square(values: np.ndarray, *, context: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        squared_mean = float(np.mean(np.square(values)))
        result = float(np.sqrt(squared_mean))
    if not np.isfinite(result):
        raise ValueError(f"{context} RMS must be finite")
    return result


__all__ = [
    "StandardizedResidualStatistics",
    "StandardizedResidualSummary",
    "summarize_standardized_residuals",
]
