# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.clocks import (
    LeadingRawHeadContract,
    LeadingRawHeadInputs,
    compile_leading_raw_head_coordinates,
)


@pytest.mark.parametrize("origin", [-(1 << 63), (1 << 63) - 1])
def test_head_and_source_accept_exact_int64_endpoints(origin):
    result = compile_leading_raw_head_coordinates(
        LeadingRawHeadInputs(
            np.asarray([origin], dtype=np.int64),
            np.asarray([0.0]),
            np.asarray([0.0]),
        ),
        LeadingRawHeadContract(1),
        samples_per_frame=1,
    )
    assert result.head_sample_starts.tolist() == [origin]
    assert result.source_sample_starts.tolist() == [origin]
    assert result.nearest_samples.tolist() == [0]
    assert result.missing.tolist() == [False]


@pytest.mark.parametrize(
    "origin, endpoint", [(-(1 << 63), 1.0), ((1 << 63) - 1, -1.0)]
)
def test_head_one_sample_outside_int64_fails_closed(origin, endpoint):
    with pytest.raises(
        OverflowError, match="request frame head exceeds int64"
    ):
        compile_leading_raw_head_coordinates(
            LeadingRawHeadInputs(
                np.asarray([origin], dtype=np.int64),
                np.asarray([endpoint]),
                np.asarray([0.0]),
            ),
            LeadingRawHeadContract(1),
            samples_per_frame=1,
        )


@pytest.mark.parametrize(
    "rate, quantum", [(1, 1), (4096000000, 1), (125000000, 8)]
)
def test_head_rounding_preserves_binary_delay_fraction_and_integer_origin(
    rate, quantum
):
    rng = np.random.default_rng(20261006)
    count = 2000
    outputs = rng.integers(-(2**60), 2**60, count, dtype=np.int64)
    endpoint = rng.uniform(-0.001, 0.001, count)
    centre = endpoint + rng.uniform(-30 / rate, 30 / rate, count)
    result = compile_leading_raw_head_coordinates(
        LeadingRawHeadInputs(outputs, endpoint, centre),
        LeadingRawHeadContract(rate, quantum),
        samples_per_frame=32768,
    )
    for index in range(count):
        origin = int(outputs[index])
        delay_ns = int(float(endpoint[index] * (1e9 / quantum))) * quantum
        head = ((origin * 10**9 - delay_ns * rate) // (10**9 * 32768)) * 32768
        offset = int(
            Fraction(origin - head)
            - Fraction.from_float(float(centre[index] * rate))
            + Fraction(1, 2)
        )
        assert result.head_sample_starts[index] == head
        assert result.nearest_samples[index] == offset
        assert result.source_sample_starts[index] == head + max(offset, 0)
        assert result.missing[index] == (offset < -1)


@pytest.mark.parametrize(
    "delay, expected",
    [
        (np.nextafter(7.5, -np.inf), 3),
        (7.5, 3),
        (np.nextafter(7.5, np.inf), 2),
        (np.nextafter(11.5, -np.inf), 0),
        (11.5, -1),
        (np.nextafter(11.5, np.inf), -1),
    ],
)
def test_truncation_at_signed_integer_boundaries(delay, expected):
    result = compile_leading_raw_head_coordinates(
        LeadingRawHeadInputs(
            np.asarray([2**60 + 10], dtype=np.int64),
            np.asarray([0.0]),
            np.asarray([delay]),
        ),
        LeadingRawHeadContract(1),
        samples_per_frame=64,
    )
    assert result.head_sample_starts.tolist() == [2**60]
    assert result.nearest_samples.tolist() == [expected]
    assert result.source_sample_starts.tolist() == [2**60 + max(expected, 0)]
