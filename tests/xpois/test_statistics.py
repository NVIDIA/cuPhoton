# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from cuphoton.xpois.noise import standardize_constant_kernel_residual
from cuphoton.xpois.statistics import summarize_standardized_residuals

RADIUS3_LAGS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, -3),
    (1, -2),
    (1, -1),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, -3),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, -3),
    (3, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
)


def _endpoint_pairs(
    stamp: np.ndarray,
    valid: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[list[float], list[float]]:
    """Enumerate valid ``(y, x) -> (y + dy, x + dx)`` pairs pixel by pixel."""

    height, width = stamp.shape
    first: list[float] = []
    second: list[float] = []
    for y in range(height):
        for x in range(width):
            other_y, other_x = y + dy, x + dx
            if not (0 <= other_y < height and 0 <= other_x < width):
                continue
            if valid[y, x] and valid[other_y, other_x]:
                first.append(float(stamp[y, x]))
                second.append(float(stamp[other_y, other_x]))
    return first, second


def _reference_pair_statistics(
    centered_stamps: np.ndarray,
    valid_masks: np.ndarray,
    lags: tuple[tuple[int, int], ...],
) -> tuple[float, float, int]:
    first: list[float] = []
    second: list[float] = []
    for stamp, valid in zip(centered_stamps, valid_masks, strict=True):
        for dy, dx in lags:
            # Pool both orientations of every lag so the reference treats
            # each lag as an unordered endpoint pair.
            for sign in (1, -1):
                head, tail = _endpoint_pairs(
                    stamp,
                    valid,
                    sign * dy,
                    sign * dx,
                )
                first.extend(head)
                second.extend(tail)
    product = math.fsum(a * b for a, b in zip(first, second, strict=True))
    first_energy = math.fsum(a * a for a in first)
    second_energy = math.fsum(b * b for b in second)
    rho = product / math.sqrt(first_energy * second_energy)
    return rho, product / len(first), len(first) // 2


def _reference_summary(
    stamps: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float | int]:
    means = np.asarray(
        [
            np.mean(stamp[valid])
            for stamp, valid in zip(stamps, masks, strict=True)
        ]
    )
    centered = np.where(masks, stamps - means[:, None, None], 0.0)
    cardinal = _reference_pair_statistics(
        centered,
        masks,
        ((0, 1), (1, 0)),
    )
    diagonal = _reference_pair_statistics(
        centered,
        masks,
        ((1, -1), (1, 1)),
    )
    lag_values = [
        _reference_pair_statistics(centered, masks, (lag,))
        for lag in RADIUS3_LAGS
    ]
    return {
        "valid_pixel_count": int(np.count_nonzero(masks)),
        "standardized_rms": float(np.sqrt(np.mean(stamps[masks] ** 2))),
        "demeaned_standardized_std": float(
            np.sqrt(np.mean(centered[masks] ** 2))
        ),
        "cardinal_lag1_rho": cardinal[0],
        "cardinal_lag1_covariance": cardinal[1],
        "cardinal_lag1_pair_count": cardinal[2],
        "diagonal_lag1_rho": diagonal[0],
        "diagonal_lag1_covariance": diagonal[1],
        "diagonal_lag1_pair_count": diagonal[2],
        "radius3_rho_rms": float(
            np.sqrt(np.mean([value[0] ** 2 for value in lag_values]))
        ),
        "radius3_covariance_rms": float(
            np.sqrt(np.mean([value[1] ** 2 for value in lag_values]))
        ),
        "radius3_unique_lag_count": 24,
    }


def _assert_summary_matches_reference(summary, expected) -> None:
    for name, value in expected.items():
        actual = getattr(summary, name)
        if isinstance(value, int):
            assert actual == value
        else:
            assert actual == pytest.approx(value, abs=1.0e-14)


def test_complete_stamps_match_oracle_and_center_separately() -> None:
    generator = np.random.default_rng(20260907)
    stamps = generator.normal(size=(2, 7, 8))
    stamps[0] += 20.0
    stamps[1] -= 35.0
    masks = np.ones(stamps.shape, dtype=bool)

    result = summarize_standardized_residuals(stamps)

    assert result.stamp_count == 2
    assert result.stamp_shape == (7, 8)
    assert result.centering == "per_stamp_valid_mean"
    assert result.rho_definition == "centered_endpoint_energy"
    assert result.variance_scope == "caller_standardized"
    assert result.whitening_applied is False
    _assert_summary_matches_reference(
        result.pixel_pooled,
        _reference_summary(stamps, masks),
    )
    for index, summary in enumerate(result.per_stamp):
        _assert_summary_matches_reference(
            summary,
            _reference_summary(
                stamps[index : index + 1],
                masks[index : index + 1],
            ),
        )

    pooled_centered = np.concatenate(
        [stamp.ravel() - np.mean(stamp) for stamp in stamps]
    )
    assert result.pixel_pooled.demeaned_standardized_std == pytest.approx(
        np.std(pooled_centered)
    )
    assert result.pixel_pooled.standardized_rms > 20.0
    assert result.pixel_pooling == "valid_pixels_and_lag_endpoint_pairs"


def test_incomplete_stamps_use_pairs_with_two_valid_endpoints() -> None:
    generator = np.random.default_rng(20260908)
    stamps = generator.normal(size=(2, 8, 9))
    masks = np.ones(stamps.shape, dtype=bool)
    masks[0, 2, 4] = False
    masks[0, 5, 1] = False
    masks[1, 1, 6] = False
    masks[1, 6, 3] = False
    stamps[~masks] = np.nan

    result = summarize_standardized_residuals(stamps, valid_mask=masks)

    _assert_summary_matches_reference(
        result.pixel_pooled,
        _reference_summary(stamps, masks),
    )
    for index, summary in enumerate(result.per_stamp):
        _assert_summary_matches_reference(
            summary,
            _reference_summary(
                stamps[index : index + 1],
                masks[index : index + 1],
            ),
        )
    # Each isolated interior hole removes two horizontal, two vertical,
    # and four diagonal pairs from its stamp.
    complete_cardinal_pairs = 2 * (8 * (9 - 1) + (8 - 1) * 9)
    complete_diagonal_pairs = 2 * 2 * ((8 - 1) * (9 - 1))
    assert result.pixel_pooled.cardinal_lag1_pair_count == (
        complete_cardinal_pairs - 4 * 4
    )
    assert result.pixel_pooled.diagonal_lag1_pair_count == (
        complete_diagonal_pairs - 4 * 4
    )
    assert result.per_stamp[0].cardinal_lag1_pair_count == (
        complete_cardinal_pairs // 2 - 2 * 4
    )
    assert result.pixel_pooled.valid_pixel_count == int(
        np.count_nonzero(masks)
    )


def test_lag_statistics_are_invariant_to_reflection_and_transpose() -> None:
    generator = np.random.default_rng(20260915)
    stamps = generator.normal(size=(2, 7, 9))
    masks = generator.random(size=stamps.shape) > 0.15
    stamps[~masks] = np.nan

    reference = summarize_standardized_residuals(stamps, valid_mask=masks)

    for transform in (
        lambda array: array[:, :, ::-1],
        lambda array: array[:, ::-1, :],
        lambda array: array.transpose(0, 2, 1),
    ):
        result = summarize_standardized_residuals(
            transform(stamps),
            valid_mask=transform(masks),
        )
        _assert_summary_matches_reference(
            result.pixel_pooled,
            asdict(reference.pixel_pooled),
        )
        for actual, expected in zip(
            result.per_stamp,
            reference.per_stamp,
            strict=True,
        ):
            _assert_summary_matches_reference(actual, asdict(expected))


def test_pixel_pooling_is_not_equal_stamp_weighting() -> None:
    generator = np.random.default_rng(20260914)
    stamps = generator.normal(size=(2, 8, 8))
    stamps[1] *= 10.0
    masks = np.zeros(stamps.shape, dtype=bool)
    masks[0] = True
    masks[1, :4, :4] = True
    stamps[~masks] = np.nan

    result = summarize_standardized_residuals(stamps, valid_mask=masks)

    expected_pixel_pooled_rms = float(
        np.sqrt(np.mean(np.square(stamps[masks])))
    )
    equal_stamp_rms = float(
        np.mean([summary.standardized_rms for summary in result.per_stamp])
    )
    assert result.pixel_pooled.standardized_rms == pytest.approx(
        expected_pixel_pooled_rms
    )
    assert result.pixel_pooled.standardized_rms != pytest.approx(
        equal_stamp_rms
    )

    expected = _reference_summary(stamps, masks)
    equal_stamp_rho = float(
        np.mean([summary.cardinal_lag1_rho for summary in result.per_stamp])
    )
    assert result.pixel_pooled.cardinal_lag1_rho == pytest.approx(
        expected["cardinal_lag1_rho"],
        abs=1.0e-14,
    )
    assert result.pixel_pooled.cardinal_lag1_rho != pytest.approx(
        equal_stamp_rho
    )


def test_results_are_deterministic_frozen_and_inputs_are_unchanged() -> None:
    stamps = np.random.default_rng(20260909).normal(size=(2, 6, 6))
    masks = np.ones(stamps.shape, dtype=bool)
    stamps_before = stamps.copy()
    masks_before = masks.copy()

    first = summarize_standardized_residuals(stamps, valid_mask=masks)
    second = summarize_standardized_residuals(stamps, valid_mask=masks)

    assert first == second
    assert isinstance(first.per_stamp, tuple)
    assert np.array_equal(stamps, stamps_before)
    assert np.array_equal(masks, masks_before)
    with pytest.raises(FrozenInstanceError):
        first.stamp_count = 3
    with pytest.raises(FrozenInstanceError):
        first.pixel_pooled.standardized_rms = 1.0
    assert first.pixel_pooled.radius3_unique_lag_count == 24
    # Python releases differ in the exception used for non-init fields.
    with pytest.raises((ValueError, TypeError), match="init=False"):
        replace(first.pixel_pooled, radius3_unique_lag_count=5)


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (np.ones((4, 4)), ValueError, "3-D"),
        (np.ones((0, 4, 4)), ValueError, "at least one stamp"),
        (np.ones((1, 3, 4)), ValueError, "at least 4 by 4"),
        (
            np.ones((1, 4, 4), dtype=bool),
            TypeError,
            "real numeric array",
        ),
        (
            np.ones((1, 4, 4), dtype=np.complex128),
            TypeError,
            "real numeric array",
        ),
        (
            np.ones((1, 4, 4), dtype="timedelta64[s]"),
            TypeError,
            "real numeric array",
        ),
        (
            np.full((1, 4, 4), "value", dtype=object),
            TypeError,
            "real numeric array",
        ),
    ],
)
def test_residual_stack_type_and_shape_validation(
    value,
    error,
    match,
) -> None:
    with pytest.raises(error, match=match):
        summarize_standardized_residuals(value)


def test_valid_mask_accepts_boolean_or_binary_like_fit_mask() -> None:
    stamps = np.random.default_rng(20260910).normal(size=(2, 8, 8))
    boolean = np.ones(stamps.shape, dtype=bool)
    boolean[0, 1, 2] = False
    boolean[1, 4, 5] = False
    boolean[1, 6, 1] = False
    stamps[~boolean] = np.nan
    reference = summarize_standardized_residuals(stamps, valid_mask=boolean)

    for dtype in (np.uint8, np.int64, np.float32, np.float64):
        binary = boolean.astype(dtype)
        result = summarize_standardized_residuals(stamps, valid_mask=binary)
        assert asdict(result) == asdict(reference)


def test_valid_mask_shape_dtype_and_finite_validation() -> None:
    stamps = np.random.default_rng(20260910).normal(size=(1, 5, 5))

    with pytest.raises(ValueError, match="match residual shape"):
        summarize_standardized_residuals(
            stamps,
            valid_mask=np.ones((1, 5, 4), dtype=bool),
        )
    with pytest.raises(ValueError, match="valid_mask.*boolean or binary"):
        summarize_standardized_residuals(
            stamps,
            valid_mask=np.full(stamps.shape, "1", dtype=object),
        )
    nan_mask = np.ones(stamps.shape, dtype=np.float64)
    nan_mask[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="valid_mask.*NaN or inf"):
        summarize_standardized_residuals(stamps, valid_mask=nan_mask)
    with pytest.raises(ValueError, match="valid_mask.*only 0/1"):
        summarize_standardized_residuals(
            stamps,
            valid_mask=np.full(stamps.shape, 2, dtype=np.int8),
        )

    stamps[0, 2, 2] = np.inf
    with pytest.raises(ValueError, match="finite on valid_mask"):
        summarize_standardized_residuals(stamps)


def test_sparse_constant_and_overflowing_stamps_fail_closed() -> None:
    stamps = np.random.default_rng(20260911).normal(size=(2, 6, 6))
    sparse = np.ones(stamps.shape, dtype=bool)
    sparse[1] = False
    with pytest.raises(ValueError, match="stamp 1 must contain at least two"):
        summarize_standardized_residuals(stamps, valid_mask=sparse)
    sparse[1, 2, 2] = True
    with pytest.raises(ValueError, match="stamp 1 must contain at least two"):
        summarize_standardized_residuals(stamps, valid_mask=sparse)
    sparse[1, :2, :2] = True
    with pytest.raises(ValueError, match="stamp 1 .*no valid endpoint pairs"):
        summarize_standardized_residuals(stamps, valid_mask=sparse)

    # 0.1 and 1/3 are not dyadic, so their floating-point means do not
    # reproduce the constant and the centered residuals are rounding noise.
    for constant in (1.0, 0.1, 1.0 / 3.0, 123.456):
        with pytest.raises(ValueError, match="stamp 0 has zero centered"):
            summarize_standardized_residuals(np.full((1, 6, 6), constant))
        with pytest.raises(ValueError, match="stamp 0 has zero centered"):
            summarize_standardized_residuals(np.full((1, 64, 64), constant))

    mixed = np.random.default_rng(20260916).normal(size=(3, 6, 6))
    mixed[2] = 0.7
    with pytest.raises(ValueError, match="stamp 2 has zero centered"):
        summarize_standardized_residuals(mixed)

    overflowing = np.full((1, 6, 6), np.finfo(np.float64).max)
    overflowing[0, 0, 0] *= -1.0
    with pytest.raises(ValueError, match="centered residuals must be finite"):
        summarize_standardized_residuals(overflowing)


def test_nonfinite_ignored_pixels_are_excluded() -> None:
    stamps = np.random.default_rng(20260912).normal(size=(1, 7, 7))
    valid = np.ones(stamps.shape, dtype=bool)
    valid[0, 3, 3] = False
    stamps[0, 3, 3] = np.inf

    result = summarize_standardized_residuals(stamps, valid_mask=valid)

    finite_stamps = stamps.copy()
    finite_stamps[0, 3, 3] = 0.0
    assert result == summarize_standardized_residuals(
        finite_stamps,
        valid_mask=valid,
    )
    assert np.isfinite(result.pixel_pooled.radius3_covariance_rms)


def test_accepts_fixed_kernel_marginal_standardization() -> None:
    generator = np.random.default_rng(20260913)
    shape = (12, 13)
    noise = standardize_constant_kernel_residual(
        generator.normal(size=shape),
        kernel=np.ones((3, 3)) / 9.0,
        target_variance=np.ones(shape),
        reference_variance=np.ones(shape),
    )

    result = summarize_standardized_residuals(
        noise.standardized_residual[None, ...],
        valid_mask=noise.valid_mask[None, ...],
    )

    assert result.stamp_count == 1
    assert result.stamp_shape == shape
    assert result.pixel_pooled.valid_pixel_count == int(
        noise.valid_mask.sum()
    )
