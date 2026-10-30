# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    buffered_request_cursor,
    buffered_request_cursors,
)

WIDE = BufferedRequestGeometry(4, 262144, 50, 128, 2)


def assert_scalar_equivalent(locals_, delays, geometry, rate):
    batch = buffered_request_cursors(
        locals_, delays, geometry, sample_rate=rate
    )
    for index, (local, delay) in enumerate(zip(locals_, delays, strict=True)):
        cursor = buffered_request_cursor(
            int(local), delay, geometry, sample_rate=rate
        )
        assert batch.valid[index] == (cursor is not None)
        if cursor is None:
            assert batch.origin_bytes[index] == 0
            assert batch.skipped_ffts[index] == 0
            assert batch.decoder_delta_samples[index] == 0
        else:
            assert batch.origin_bytes[index] == cursor.origin_bytes
            assert batch.skipped_ffts[index] == cursor.skipped_ffts
            assert batch.decoder_delta_samples[index] == (
                cursor.decoder_delta_samples
            )
    return batch


@pytest.mark.parametrize(
    "geometry",
    [
        WIDE,
        BufferedRequestGeometry(4, 28, 5, 8, 2),
        BufferedRequestGeometry(Fraction(1, 8), 3, 5, 16, 2),
        BufferedRequestGeometry(Fraction(3, 2), 15, 5, 6, 3),
        BufferedRequestGeometry(Fraction(2, 3), 10, 5, 6, 3),
    ],
)
def test_batch_matches_scalar_across_alignment_skip_and_drift_cases(geometry):
    # Boundary grids exercise truncation around zero, word-alignment shifts,
    # exact all-skipped requests and a payload not divisible by word size.
    local_values = sorted(
        {
            skip * geometry.fft_samples + phase
            for skip in range(-geometry.ffts_per_request - 1, 2)
            for phase in (-7, -2, -1, 0, 1, 2, 7)
        }
    )
    drift_values = [-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01]
    locals_ = np.repeat(np.asarray(local_values, dtype=np.int64), 7)
    delays = np.tile(np.asarray(drift_values), len(local_values))
    assert_scalar_equivalent(locals_, delays, geometry, 4096.0)


def test_wide_batch_preserves_calibrated_drift_correction_quirk():
    locals_ = np.asarray([-262144, -262144, -262144, 7], dtype=np.int64)
    delays = np.asarray([8e-7, -8e-7, -0.001, 1e200])
    batch = assert_scalar_equivalent(locals_, delays, WIDE, 4096000000.0)
    assert batch.skipped_ffts[:2].tolist() == [2, 1]
    assert batch.origin_bytes[:2].tolist() == [65408, 0]
    assert batch.decoder_delta_samples[:2].tolist() == [381, 65]
    # A drift correction that does not add an FFT never shifts byte origin;
    # the scalar decoder displacement still incorporates that drift.
    assert batch.origin_bytes[2] == 0
    assert batch.skipped_ffts[2] == 1
    assert batch.valid[3]
    assert batch.decoder_delta_samples[3] == 7


def test_extra_skip_can_still_leave_negative_origin_like_scalar():
    geometry = BufferedRequestGeometry(1, 32, 4, 8, 2)
    batch = assert_scalar_equivalent(
        np.asarray([-1], dtype=np.int64), np.asarray([2.0]), geometry, 128.0
    )
    assert batch.valid[0]
    assert batch.origin_bytes[0] < 0


def test_all_skipped_batch_is_invalid_and_zeroed():
    locals_ = np.asarray([-50 * 262144, -51 * 262144], dtype=np.int64)
    batch = assert_scalar_equivalent(locals_, np.zeros(2), WIDE, 128.0)
    assert not batch.valid.any()


def test_empty_batch_preserves_dtypes_and_shape():
    batch = buffered_request_cursors(
        np.empty(0, dtype=np.int64), np.empty(0), WIDE, sample_rate=128.0
    )
    for array in (
        batch.origin_bytes,
        batch.skipped_ffts,
        batch.decoder_delta_samples,
    ):
        assert array.shape == (0,)
        assert array.dtype == np.int64
    assert batch.valid.shape == (0,)
    assert batch.valid.dtype == np.bool_


@pytest.mark.parametrize(
    "geometry,values",
    [
        (WIDE, [2**60 + 7, 2**60 + 1023, 2**63 - 1]),
        (
            BufferedRequestGeometry(Fraction(1, 8), 32, 3, 32, 2),
            [2**58 + 7, 2**59 - 1],
        ),
        (
            BufferedRequestGeometry(Fraction(3, 2), 36, 3, 8, 2),
            [2**60 + 7, 2**61 - 1],
        ),
    ],
)
def test_large_origins_remain_exact_integer_coordinates(geometry, values):
    batch = assert_scalar_equivalent(
        np.asarray(values, dtype=np.int64),
        np.zeros(len(values)),
        geometry,
        128.0,
    )
    assert batch.valid.all()


def test_noncontiguous_views_and_float32_drift_inputs():
    locals_ = np.asarray([-35, 0, 34, 0, -1, 0], dtype=np.int64)[::2]
    delays = np.asarray(
        [0.001, 0.0, -0.001, 0.0, 0.0, 0.0], dtype=np.float32
    )[::2]
    assert_scalar_equivalent(
        locals_, delays, BufferedRequestGeometry(4, 32, 3, 8, 2), 128.0
    )


@pytest.mark.parametrize(
    "locals_,delays,error,message",
    [
        ([1], np.zeros(1), TypeError, "int64"),
        (np.asarray([1], dtype=np.int32), np.zeros(1), TypeError, "int64"),
        (np.ones((1, 1), dtype=np.int64), np.zeros(1), TypeError, "int64"),
        (np.ones(1, dtype=np.int64), [0.0], TypeError, "float array"),
        (
            np.ones(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            TypeError,
            "float array",
        ),
        (np.ones(1, dtype=np.int64), np.zeros(2), ValueError, "shapes"),
        (
            np.ones(1, dtype=np.int64),
            np.asarray([np.nan]),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_batch_inputs_fail_once(locals_, delays, error, message):
    with pytest.raises(error, match=message):
        buffered_request_cursors(locals_, delays, WIDE, sample_rate=128)


@pytest.mark.parametrize("rate", [0.0, -1.0, np.inf, np.nan])
def test_invalid_sample_rate(rate):
    with pytest.raises(ValueError, match="sample rate"):
        buffered_request_cursors(
            np.zeros(1, dtype=np.int64), np.zeros(1), WIDE, sample_rate=rate
        )


def test_invalid_geometry_type():
    with pytest.raises(TypeError, match="BufferedRequestGeometry"):
        buffered_request_cursors(
            np.zeros(1, dtype=np.int64), np.zeros(1), {}, sample_rate=128
        )


@pytest.mark.parametrize(
    "geometry,local,delay,rate",
    [
        (
            BufferedRequestGeometry(Fraction(1, 8), 32, 3, 32, 2),
            2**60,
            0.0,
            128.0,
        ),
        (
            BufferedRequestGeometry(Fraction(2, 3), 32, 3, 24, 2),
            2**62,
            0.0,
            128.0,
        ),
        (
            BufferedRequestGeometry(1, 4, 3, 2**63 - 1, 2**63 - 1),
            -(2**63),
            0.0,
            128.0,
        ),
        (WIDE, -4, 1e308, 1e308),
        (BufferedRequestGeometry(1, 4, 2, 2, 2), -1, float(2**64), 1.0),
        (BufferedRequestGeometry(1, 4, 2**62, 2, 2), 0, 0.0, 128.0),
    ],
)
def test_arithmetic_and_drift_overflow_fail(geometry, local, delay, rate):
    with pytest.raises(OverflowError, match="int64"):
        buffered_request_cursors(
            np.asarray([local], dtype=np.int64),
            np.asarray([delay]),
            geometry,
            sample_rate=rate,
        )


def test_odd_payload_all_skipped_requests_never_cast_unused_drift():
    geometry = BufferedRequestGeometry(4, 28, 5, 8, 2)
    batch = assert_scalar_equivalent(
        np.asarray([-136], dtype=np.int64),
        np.asarray([1e308]),
        geometry,
        1e308,
    )
    assert not batch.valid[0]


def test_extended_float_drift_conversion_fails_without_wrapping():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform has no extended floating dtype")
    with pytest.raises(OverflowError, match="int64"):
        buffered_request_cursors(
            np.asarray([-4], dtype=np.int64),
            np.asarray([np.finfo(np.longdouble).max], dtype=np.longdouble),
            WIDE,
            sample_rate=128,
        )


@pytest.mark.parametrize(
    "geometry,rate",
    [
        (WIDE, 4096000000.0),
        (BufferedRequestGeometry(4, 28, 5, 8, 2), 128.0),
        (BufferedRequestGeometry(Fraction(1, 8), 32, 3, 32, 2), 128.0),
    ],
)
def test_deterministic_randomized_scalar_parity(geometry, rate):
    generator = np.random.default_rng(20261006)
    width = geometry.ffts_per_request * geometry.fft_samples
    locals_ = generator.integers(
        -width * 2, width * 2, size=1000, dtype=np.int64
    )
    delays = generator.uniform(-1e-6, 1e-6, size=1000)
    assert_scalar_equivalent(locals_, delays, geometry, rate)


def test_float32_rounding_matches_scalar_without_promoting_delay_arithmetic():
    geometry = BufferedRequestGeometry(1, 4, 2, 2, 2)
    delay = np.nextafter(np.float32(0.5), np.float32(0))
    scalar = buffered_request_cursor(-1, delay, geometry, sample_rate=2.0)
    batch = buffered_request_cursors(
        np.asarray([-1], dtype=np.int64),
        np.asarray([delay]),
        geometry,
        sample_rate=2.0,
    )
    assert batch.valid[0]
    assert batch.decoder_delta_samples[0] == scalar.decoder_delta_samples
    assert batch.origin_bytes[0] == scalar.origin_bytes
