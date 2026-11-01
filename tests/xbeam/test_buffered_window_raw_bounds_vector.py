# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.clocks import (
    compile_raw_window_lower_bounds,
)


def test_signed_window_rounding_against_rational_oracle():
    rng = np.random.default_rng(982)
    count = 10000
    heads = rng.integers(-(2**60), 2**60, count, dtype=np.int64)
    outputs = heads + rng.integers(-10000, 10000, count, dtype=np.int64)
    delays = rng.uniform(-100000, 100000, count)
    result = compile_raw_window_lower_bounds(
        outputs, heads, delays, sample_rate=1
    )
    expected = [
        int(Fraction(int(o) - int(h)) - Fraction(float(d)) + Fraction(1, 2))
        for o, h, d in zip(outputs, heads, delays, strict=True)
    ]
    assert result.nearest_samples.tolist() == expected
    assert result.missing.tolist() == [value < -1 for value in expected]


def test_rounding_ties_and_tiny_negative_values_at_large_delay_magnitudes():
    values = [
        float("-0.0"),
        np.nextafter(0.0, -1.0),
        np.nextafter(0.0, 1.0),
    ]
    for value in [-(2**52), -1048576, -1, 0, 1, 1048576, 2**52]:
        for centre in [float(value), float(value + 0.5)]:
            values.extend(
                [
                    np.nextafter(centre, -np.inf),
                    centre,
                    np.nextafter(centre, np.inf),
                ]
            )
    for offset in [-20, -1, 0, 1, 20]:
        delays = np.asarray(values)
        heads = np.full(delays.shape, 2**54 + 7, dtype=np.int64)
        output = heads + offset
        result = compile_raw_window_lower_bounds(
            output, heads, delays, sample_rate=1
        )
        expected = [
            int(Fraction(offset) - Fraction(float(d)) + Fraction(1, 2))
            for d in values
        ]
        assert result.nearest_samples.tolist() == expected


def test_overflowing_intermediate_difference_can_have_representable_result():
    minimum, maximum = map(
        int, (np.iinfo(np.int64).min, np.iinfo(np.int64).max)
    )
    output = np.asarray([0, -1, maximum, minimum, maximum], dtype=np.int64)
    head = np.asarray([minimum, maximum, 0, 0, maximum], dtype=np.int64)
    delay = np.asarray([8, -8, -0.25, 0.25, 1], dtype=np.float64)
    result = compile_raw_window_lower_bounds(
        output, head, delay, sample_rate=1
    )
    expected = [
        int(Fraction(int(o) - int(h)) - Fraction(float(d)) + Fraction(1, 2))
        for o, h, d in zip(output, head, delay, strict=True)
    ]
    assert result.nearest_samples.tolist() == expected


@pytest.mark.parametrize(
    "output,head,delay", [(0, -(2**63), -1), (-(2**63), 1, 1)]
)
def test_out_of_range_final_offset_rejects_instead_of_wrapping(
    output, head, delay
):
    with pytest.raises(OverflowError, match="window sample offset"):
        compile_raw_window_lower_bounds(
            np.asarray([output], dtype=np.int64),
            np.asarray([head], dtype=np.int64),
            np.asarray([delay], dtype=np.float64),
            sample_rate=1,
        )
