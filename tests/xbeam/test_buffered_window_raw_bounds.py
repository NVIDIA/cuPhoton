# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.clocks import (
    compile_raw_window_bounds,
    compile_raw_window_payload_spans,
)


def bounds(offsets, extents, *, fft=4):
    offsets = np.asarray(offsets, dtype=np.int64)
    return compile_raw_window_bounds(
        offsets,
        np.zeros(offsets.shape, dtype=np.int64),
        np.full(offsets.shape, 0.5, dtype=np.float64),
        np.asarray(extents, dtype=np.int64),
        sample_rate=1,
        fft_samples=fft,
    )


def test_actual_returned_extent_controls_the_positive_boundary():
    result = bounds([-2, -1, 0, 24, 25, 24], [28, 28, 28, 28, 28, 27])
    assert result.nearest_samples.tolist() == [-2, -1, 0, 24, 25, 24]
    assert result.before_window.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result.after_window.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert result.missing.tolist() == [True, False, False, False, True, True]


def test_empty_and_short_returns_are_not_spare_allocation_capacity():
    result = bounds([0, 0, 0], [0, 3, 4])
    assert result.missing.tolist() == [True, True, False]


def test_upper_bound_comparison_does_not_overflow_int64():
    maximum = np.iinfo(np.int64).max
    result = bounds([maximum, maximum - 4], [maximum, maximum])
    assert result.missing.tolist() == [True, False]
    result = bounds([0, 0], [0, maximum], fft=int(maximum))
    assert result.missing.tolist() == [True, False]


def test_large_origins_preserve_fractional_delay_phase_and_exact_bounds():
    origin = 2**54 + 3
    offsets = [-3, -2, -1, 0, 17, 18, 19]
    output = np.asarray([origin + value for value in offsets], dtype=np.int64)
    head = np.full(output.shape, origin, dtype=np.int64)
    delay = np.asarray([0.49, 0.51, -0.49, -0.51, 0.49, 0.51, -0.51])
    extents = np.full(output.shape, 22, dtype=np.int64)
    result = compile_raw_window_bounds(
        output, head, delay, extents, sample_rate=1, fft_samples=4
    )
    expected = [
        int(Fraction(value) - Fraction(float(d)) + Fraction(1, 2))
        for value, d in zip(offsets, delay, strict=True)
    ]
    assert result.nearest_samples.tolist() == expected
    assert result.missing.tolist() == [n < -1 or n + 4 > 22 for n in expected]


def test_bounds_own_immutable_arrays():
    output = np.asarray([3], dtype=np.int64)
    extents = np.asarray([7], dtype=np.int64)
    result = compile_raw_window_bounds(
        output,
        np.zeros(1, dtype=np.int64),
        np.asarray([0.5]),
        extents,
        sample_rate=1,
        fft_samples=4,
    )
    output[0] = 99
    extents[0] = 0
    assert result.nearest_samples.tolist() == [3]
    assert result.missing.tolist() == [False]
    for array in (
        result.nearest_samples,
        result.before_window,
        result.after_window,
        result.missing,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    "extent, error, message",
    [
        (np.asarray([1.0]), TypeError, "int64 vector"),
        (np.asarray([[1]], dtype=np.int64), TypeError, "int64 vector"),
        (np.asarray([1, 2], dtype=np.int64), ValueError, "shape"),
        (np.asarray([-1], dtype=np.int64), ValueError, "nonnegative"),
    ],
)
def test_invalid_returned_extents_fail_explicitly(extent, error, message):
    with pytest.raises(error, match=message):
        compile_raw_window_bounds(
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1),
            extent,
            sample_rate=1,
            fft_samples=4,
        )


def test_empty_vectors_remain_empty_and_oversized_fft_fails():
    assert bounds([], []).missing.size == 0
    with pytest.raises(OverflowError, match="FFT samples"):
        bounds([0], [4], fft=2**63)


def test_payload_clipping_is_separate_from_fft_admission():
    offsets = np.asarray([-2, -1, 0, 6, 7, 8, 12], dtype=np.int64)
    result = compile_raw_window_payload_spans(
        offsets, np.full(offsets.shape, 10, dtype=np.int64), fft_samples=4
    )
    assert result.first_samples.tolist() == [2, 1, 0, 0, 0, 0, 0]
    assert result.stop_samples.tolist() == [4, 4, 4, 4, 3, 2, 0]
    admitted = bounds([8], [12])
    assert not admitted.missing[0]
    # A wider execution window can admit a partially available decoder FFT.
    assert result.stop_samples[5] == 2


def test_payload_spans_match_integer_interval_intersection_at_limits():
    minimum, maximum = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    offsets = [
        minimum,
        minimum + 1,
        -100,
        -4,
        -3,
        -1,
        0,
        1,
        7,
        8,
        9,
        maximum - 4,
        maximum - 1,
        maximum,
    ]
    for length in [1, 4, int(maximum)]:
        for extent in [0, 2, 8, int(maximum)]:
            result = compile_raw_window_payload_spans(
                np.asarray(offsets, dtype=np.int64),
                np.full(len(offsets), extent, dtype=np.int64),
                fft_samples=length,
            )
            starts = [min(length, max(0, -int(n))) for n in offsets]
            stops = [
                max(a, min(length, max(0, extent - int(n))))
                for a, n in zip(starts, offsets, strict=True)
            ]
            assert result.first_samples.tolist() == starts
            assert result.stop_samples.tolist() == stops


def test_payload_spans_own_readonly_vectors_and_accept_empty_input():
    nearest = np.asarray([2], dtype=np.int64)
    extent = np.asarray([4], dtype=np.int64)
    result = compile_raw_window_payload_spans(nearest, extent, fft_samples=4)
    nearest[0], extent[0] = 99, 99
    assert result.first_samples.tolist() == [0]
    assert result.stop_samples.tolist() == [2]
    for array in (result.first_samples, result.stop_samples):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    empty = np.asarray([], dtype=np.int64)
    assert (
        compile_raw_window_payload_spans(
            empty, empty, fft_samples=4
        ).first_samples.size
        == 0
    )


@pytest.mark.parametrize(
    "extent, error, message",
    [
        (np.asarray([1.0]), TypeError, "int64 vector"),
        (np.asarray([[1]], dtype=np.int64), TypeError, "int64 vector"),
        (np.asarray([1, 2], dtype=np.int64), ValueError, "match"),
        (np.asarray([-1], dtype=np.int64), ValueError, "nonnegative"),
    ],
)
def test_payload_spans_reject_invalid_decoded_extents(extent, error, message):
    with pytest.raises(error, match=message):
        compile_raw_window_payload_spans(
            np.zeros(1, dtype=np.int64), extent, fft_samples=4
        )
