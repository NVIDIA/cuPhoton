# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit normal-full contract; header admission is not decoded coverage."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import BufferedReaderGeometry
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedNormalFullExtent,
    BufferedWindowGeometry,
)


@pytest.mark.parametrize(
    "frames,frame_bytes,payload,packing,ratio",
    [
        (7, 37, 32, 4, 3),
        (11, 103, 96, Fraction(1, 8), 2),
        (2, 20, 12, Fraction(2, 3), 5),
        (1, 13, 13, Decimal(2), 1),
    ],
)
def test_distinct_exact_framed_admission_and_payload_coverage(
    frames, frame_bytes, payload, packing, ratio
):
    reader = BufferedReaderGeometry(4 * ratio, 4, 17)
    geometry = BufferedWindowGeometry(3, 9, 17, 4, frames * ratio)
    extent = geometry.normal_full_extent(
        reader,
        frame_bytes=frame_bytes,
        payload_bytes=payload,
        samples_per_byte=packing,
    )
    exact = Fraction(packing)
    assert extent.request_frames == frames
    assert extent.read_buffer_ratio == ratio
    assert extent.samples_per_frame == payload * exact
    assert extent.request_framed_bytes == frames * frame_bytes
    assert extent.request_payload_bytes == frames * payload
    assert extent.framed_admission_samples == frames * frame_bytes * exact
    assert extent.decoded_payload_samples == frames * payload * exact
    assert extent.allocation_framed_bytes == frames * ratio * frame_bytes
    assert (
        extent.framed_admission_samples - extent.decoded_payload_samples
        == frames * (frame_bytes - payload) * exact
    )
    assert isinstance(extent.samples_per_byte, Fraction)
    assert isinstance(extent.framed_admission_samples, Fraction)
    with pytest.raises(FrozenInstanceError):
        extent.request_frames = 2


def test_per_lane_admission_is_exact_rational_never_implicitly_rounded():
    extent = BufferedNormalFullExtent(17, 16, 3, 2, Fraction(1, 8))
    assert extent.samples_per_frame == 2
    assert extent.decoded_payload_samples == 6
    assert extent.framed_admission_samples == Fraction(51, 8)
    assert extent.request_framed_bytes == 51


def test_extent_integer_normalization_and_large_exact_products():
    extent = BufferedNormalFullExtent(
        np.int64(17), np.int64(16), np.int64(2**55 + 1), 1, 2
    )
    assert type(extent.request_frames) is int
    assert extent.decoded_payload_samples == 32 * (2**55 + 1)
    assert extent.framed_admission_samples == 34 * (2**55 + 1)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"frame_bytes": True}, TypeError, "integer"),
        ({"request_frames": 1.0}, TypeError, "integer"),
        ({"payload_bytes": 0}, ValueError, "payload_bytes"),
        ({"payload_bytes": 18}, ValueError, "framed"),
        ({"samples_per_byte": 0}, ValueError, "positive"),
        ({"samples_per_byte": -1}, ValueError, "positive"),
        ({"samples_per_byte": 0.5}, TypeError, "exact rational"),
        ({"samples_per_byte": True}, TypeError, "exact rational"),
        ({"samples_per_byte": Decimal("NaN")}, ValueError, "finite"),
        ({"samples_per_byte": Fraction(1, 3)}, ValueError, "integral"),
        ({"frame_bytes": 2**63}, OverflowError, "int64"),
        ({"request_frames": 2**62}, OverflowError, "int64"),
        ({"read_buffer_ratio": 2**62}, OverflowError, "int64"),
        ({"samples_per_byte": 2**62}, OverflowError, "int64"),
    ],
)
def test_invalid_packing_and_extent_overflow_fail_closed(
    kwargs, error, match
):
    inputs = dict(
        frame_bytes=17,
        payload_bytes=16,
        request_frames=3,
        read_buffer_ratio=2,
        samples_per_byte=2,
    )
    inputs.update(kwargs)
    with pytest.raises(error, match=match):
        BufferedNormalFullExtent(**inputs)


@pytest.mark.parametrize(
    "reader,allocation,match",
    [
        (BufferedReaderGeometry(9, 3, 17), 12, "segment"),
        (BufferedReaderGeometry(8, 4, 20), 12, "buffering"),
        (BufferedReaderGeometry(9, 4, 17), 12, "divisible"),
        (BufferedReaderGeometry(12, 4, 17), 13, "whole framed"),
    ],
)
def test_reader_ratio_requires_consistent_whole_request(
    reader, allocation, match
):
    geometry = BufferedWindowGeometry(8, 50, 17, 4, allocation)
    with pytest.raises(ValueError, match=match):
        geometry.normal_full_extent(
            reader, frame_bytes=17, payload_bytes=16, samples_per_byte=2
        )


def test_allowance_geometry_converts_each_return_contract_explicitly():
    reader = BufferedReaderGeometry(12, 4, 17)
    geometry = BufferedWindowGeometry.from_reader(
        reader,
        samples_per_frame=16,
        fft_samples=48,
        sample_rate=240,
        subintegration_s=Decimal(1),
        reference_rate_allowance=0.2,
    )
    extent = geometry.normal_full_extent(
        reader, frame_bytes=19, payload_bytes=16, samples_per_byte=1
    )
    assert extent.request_frames == 20
    assert extent.decoded_payload_samples == 320
    assert extent.framed_admission_samples == 380
    changed = replace(extent, request_frames=2)
    assert changed.decoded_payload_samples == 32
    assert changed.framed_admission_samples == 38
