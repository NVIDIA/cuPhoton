# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Own clock arithmetic and signed-boundary tests, without observed tables."""

from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.clocks import (
    LeadingRawHeadContract,
    LeadingRawHeadInputs,
    compile_leading_raw_head_coordinates,
    compile_raw_window_lower_bounds,
)


@pytest.mark.parametrize("origin", [0, 2**53 + 1, 2**60 + 3, -(2**60) + 7])
def test_exact_origins_and_signed_truncation(origin):
    relative = np.array([-2.51, -2.5, -2.49, -1.51, -0.51, 0.49, 0.5, 1.5])
    output = np.full(relative.size, origin, dtype=np.int64)
    head = output.copy()
    delays = -relative
    result = compile_raw_window_lower_bounds(
        output, head, delays, sample_rate=1
    )
    expected = np.array(
        [
            int(Fraction.from_float(float(r)) + Fraction(1, 2))
            for r in relative
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(result.nearest_samples, expected)
    np.testing.assert_array_equal(result.missing, expected < -1)
    np.testing.assert_array_equal(expected, [-2, -2, -1, -1, 0, 0, 1, 2])
    with pytest.raises(ValueError):
        result.nearest_samples.setflags(write=True)
    with pytest.raises(ValueError):
        result.missing.setflags(write=True)


def test_adjacent_float_values_at_negative_lower_bound():
    delays = np.array(
        [np.nextafter(2.5, -np.inf), 2.5, np.nextafter(2.5, np.inf)]
    )
    zeros = np.zeros(3, dtype=np.int64)
    result = compile_raw_window_lower_bounds(
        zeros, zeros, delays, sample_rate=1
    )
    np.testing.assert_array_equal(result.nearest_samples, [-1, -2, -2])
    np.testing.assert_array_equal(result.missing, [False, True, True])


@pytest.mark.parametrize("rate,width", [(7, 3), (11, 17), (4096, 128)])
def test_resumed_fft_uses_its_model_centre_and_returned_head(rate, width):
    # A returned cursor head lies strictly after the first FFT's source;
    # a decreasing linear delay makes the following FFT fully available.
    origin = 2**60 + 5
    output = origin + np.arange(4, dtype=np.int64) * width
    head = np.full(4, origin + 3, dtype=np.int64)
    midpoint = (np.arange(4) * width + (width - 1) / 2) / rate
    delays = -0.25 * midpoint
    product = delays * rate
    result = compile_raw_window_lower_bounds(
        output, head, delays, sample_rate=rate
    )
    expected = [
        int(
            Fraction(int(o) - int(h))
            - Fraction.from_float(float(d))
            + Fraction(1, 2)
        )
        for o, h, d in zip(output, head, product, strict=True)
    ]
    np.testing.assert_array_equal(result.nearest_samples, expected)
    # A lower-bound check deliberately permits far beyond any positive end.
    far = compile_raw_window_lower_bounds(
        np.array([origin + 1000 * width], dtype=np.int64),
        head[:1],
        delays[:1],
        sample_rate=rate,
    )
    assert not far.missing[0]


def test_window_bound_agrees_with_selected_leading_frame_clock():
    output = np.array([0, 1, 987654321, 2**60 + 3], dtype=np.int64)
    endpoints = np.array([0.125, -0.1875, 0.375, -0.125], dtype=np.float64)
    centres = endpoints + np.array([0.125, -0.25, 0.01, 0.0])
    head = compile_leading_raw_head_coordinates(
        LeadingRawHeadInputs(output, endpoints, centres),
        LeadingRawHeadContract(128),
        samples_per_frame=16,
    )
    bounds = compile_raw_window_lower_bounds(
        output, head.head_sample_starts, centres, sample_rate=128
    )
    np.testing.assert_array_equal(
        bounds.nearest_samples, head.nearest_samples
    )
    np.testing.assert_array_equal(bounds.missing, head.missing)


def test_empty_clocks():
    result = compile_raw_window_lower_bounds(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        sample_rate=3,
    )
    assert result.nearest_samples.shape == result.missing.shape == (0,)


@pytest.mark.parametrize("rate", [0, -1, True, 1.5])
def test_invalid_rate(rate):
    with pytest.raises((TypeError, ValueError)):
        compile_raw_window_lower_bounds(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0.0]),
            sample_rate=rate,
        )


@pytest.mark.parametrize("delay", [np.inf, np.nan])
def test_nonfinite_delay_fails(delay):
    with pytest.raises(ValueError, match="finite"):
        compile_raw_window_lower_bounds(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([delay]),
            sample_rate=1,
        )


def test_clock_shape_and_types():
    zero = np.array([0], dtype=np.int64)
    with pytest.raises(TypeError, match="output samples"):
        compile_raw_window_lower_bounds(
            [0], zero, np.array([0.0]), sample_rate=1
        )
    with pytest.raises(TypeError, match="aligned head samples"):
        compile_raw_window_lower_bounds(
            zero, zero.astype(float), np.array([0.0]), sample_rate=1
        )
    with pytest.raises(ValueError, match="equal shape"):
        compile_raw_window_lower_bounds(
            zero, zero, np.array([], dtype=float), sample_rate=1
        )
    with pytest.raises(TypeError, match="centre delays"):
        compile_raw_window_lower_bounds(zero, zero, zero, sample_rate=1)


def test_offset_overflow_and_inexact_delay_phase_fail():
    limits = np.iinfo(np.int64)
    with pytest.raises(OverflowError, match="window sample offset"):
        compile_raw_window_lower_bounds(
            np.array([limits.max], dtype=np.int64),
            np.array([limits.min], dtype=np.int64),
            np.array([0.0]),
            sample_rate=1,
        )
    with pytest.raises(OverflowError, match="sample-phase"):
        compile_raw_window_lower_bounds(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([float(2**53)]),
            sample_rate=1,
        )
