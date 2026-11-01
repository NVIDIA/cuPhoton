# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    buffered_request_cursor,
)

WIDE = BufferedRequestGeometry(4, 262144, 50, 128, 2)


@pytest.mark.parametrize(
    "local,origin,skipped,delta",
    [
        (7, 0, 0, 7),
        (1023, 128, 0, 511),
        (-1, 0, 0, -1),
        (-4, 65408, 1, 508),
        (-262144, 0, 1, 0),
    ],
)
def test_request_preserves_subbyte_and_alignment_displacement(
    local, origin, skipped, delta
):
    cursor = buffered_request_cursor(local, 0.0, WIDE, sample_rate=4096000000)
    assert (
        cursor.origin_bytes,
        cursor.skipped_ffts,
        cursor.decoder_delta_samples,
    ) == (origin, skipped, delta)


def test_request_wholly_before_window_has_no_decoder_cursor():
    assert (
        buffered_request_cursor(
            -50 * 262144, 0.0, WIDE, sample_rate=4096000000
        )
        is None
    )


@pytest.mark.parametrize(
    "drift,skipped,origin,delta", [(8e-7, 2, 65408, 381), (-8e-7, 1, 0, 65)]
)
def test_delay_drift_can_change_leading_skip_without_losing_phase(
    drift, skipped, origin, delta
):
    cursor = buffered_request_cursor(
        -262144, drift, WIDE, sample_rate=4096000000
    )
    assert (
        cursor.origin_bytes,
        cursor.skipped_ffts,
        cursor.decoder_delta_samples,
    ) == (origin, skipped, delta)


@pytest.mark.parametrize("samples_per_byte", [1, 2, 4, 8])
def test_small_fft_geometry_and_large_integer_origin(samples_per_byte):
    geometry = BufferedRequestGeometry(samples_per_byte, 32, 3, 8, 2)
    first = buffered_request_cursor(34, 0.0, geometry, sample_rate=128)
    translated = buffered_request_cursor(
        2**60 + 34, 0.0, geometry, sample_rate=128
    )
    assert translated.origin_bytes - first.origin_bytes == (
        2**60 // samples_per_byte
    )
    assert translated.decoder_delta_samples == first.decoder_delta_samples
    assert translated.skipped_ffts == first.skipped_ffts == 0
    assert (
        first.origin_bytes * samples_per_byte + first.decoder_delta_samples
        == 34
    )


@pytest.mark.parametrize("drift", [math.inf, -math.inf, math.nan])
def test_nonfinite_drift_fails(drift):
    with pytest.raises(ValueError, match="finite"):
        buffered_request_cursor(0, drift, WIDE, sample_rate=4096000000)


@pytest.mark.parametrize("rate", [0, -1, math.inf, math.nan])
def test_invalid_sample_rate_fails(rate):
    with pytest.raises(ValueError, match="sample rate"):
        buffered_request_cursor(0, 0.0, WIDE, sample_rate=rate)


@pytest.mark.parametrize("values", [(3, 32, 3, 8, 2), (4, 32, 3, 7, 2)])
def test_geometry_requires_complete_payload_bytes_and_words(values):
    with pytest.raises(ValueError):
        BufferedRequestGeometry(*values)


@pytest.mark.parametrize(
    "ratio,fft_samples,alignment,local,skipped,origin,delta",
    [
        (Fraction(1, 8), 32, 32, 34, 0, 256, 2),
        (Fraction(1, 8), 32, 32, -1, 1, 224, 3),
        (Fraction(3, 2), 36, 8, 35, 0, 16, 11),
        (Fraction(2, 3), 32, 24, 34, 0, 48, 2),
    ],
)
def test_rational_lane_packing_preserves_sample_displacement(
    ratio, fft_samples, alignment, local, skipped, origin, delta
):
    geometry = BufferedRequestGeometry(ratio, fft_samples, 3, alignment, 2)
    cursor = buffered_request_cursor(local, 0.0, geometry, sample_rate=128)
    assert cursor.origin_bytes == origin
    assert cursor.skipped_ffts == skipped
    assert cursor.decoder_delta_samples == delta
    assert type(cursor.decoder_delta_samples) is int
    assert (
        Fraction(cursor.origin_bytes) * ratio + cursor.decoder_delta_samples
        == local + skipped * fft_samples
    )


@pytest.mark.parametrize(
    "ratio,fft_samples,alignment",
    [
        (Fraction(1, 8), 32, 32),
        (Fraction(3, 2), 36, 8),
        (Fraction(2, 3), 32, 24),
    ],
)
def test_rational_lane_coordinates_remain_exact_beyond_float_precision(
    ratio, fft_samples, alignment
):
    geometry = BufferedRequestGeometry(ratio, fft_samples, 3, alignment, 2)
    # The translation is an integral number of samples and aligned bytes
    # for every ratio, far beyond float64's exact coordinate range.
    sample_translation = ratio.numerator * alignment * 2**60
    byte_translation = ratio.denominator * alignment * 2**60
    first = buffered_request_cursor(35, 0.0, geometry, sample_rate=128)
    translated = buffered_request_cursor(
        sample_translation + 35, 0.0, geometry, sample_rate=128
    )
    assert translated.origin_bytes == first.origin_bytes + byte_translation
    assert translated.decoder_delta_samples == first.decoder_delta_samples
    assert translated.skipped_ffts == first.skipped_ffts == 0


def test_multichannel_delay_drift_preserves_integer_cursor():
    geometry = BufferedRequestGeometry(Fraction(1, 8), 32, 3, 32, 2)
    cursor = buffered_request_cursor(-32, 0.09375, geometry, sample_rate=128)
    assert cursor.origin_bytes == 192
    assert cursor.skipped_ffts == 2
    assert cursor.decoder_delta_samples == 0


def test_integer_inputs_are_normalized_before_coordinate_arithmetic():
    geometry = BufferedRequestGeometry(
        np.int64(4), np.int64(32), np.int32(3), np.int32(8), np.int32(2)
    )
    assert geometry.samples_per_byte == Fraction(4)
    assert type(geometry.samples_per_byte) is Fraction
    assert type(geometry.fft_samples) is int
    cursor = buffered_request_cursor(
        np.int64(2**60 + 34), 0.0, geometry, sample_rate=128
    )
    assert type(cursor.origin_bytes) is int
    assert type(cursor.decoder_delta_samples) is int
    assert cursor.origin_bytes == (2**60 // 4) + 8
    assert cursor.decoder_delta_samples == 2


@pytest.mark.parametrize("ratio", [0, -1, True, 0.125, Fraction(-1, 8)])
def test_invalid_sample_byte_ratios_fail(ratio):
    with pytest.raises((TypeError, ValueError), match="samples_per_byte"):
        BufferedRequestGeometry(ratio, 32, 3, 32, 2)


def test_fractional_byte_fft_extent_fails():
    with pytest.raises(ValueError, match="complete payload bytes"):
        BufferedRequestGeometry(Fraction(3, 2), 32, 3, 8, 2)


def test_fractional_sample_alignment_fails():
    with pytest.raises(ValueError, match="complete samples"):
        BufferedRequestGeometry(Fraction(1, 8), 32, 3, 4, 2)
