# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Float64 storage geometry without decoder publication or model selection."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import (
    BufferedReaderGeometry,
    DelayCoordinateModel,
)
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedWindowGeometry,
    reference_model_rate_allowance,
)
from cuphoton.xbeam.errors import ConfigurationError


def model(station, coefficients, epoch=Decimal(0)):
    return DelayCoordinateModel(
        station,
        epoch,
        tuple(float(c) for c in coefficients),
        (0,),
        (0,),
        (0,),
    )


def geometry(allowance=0.0, *, frames=8, ffts=50, ratio=3):
    return BufferedWindowGeometry.from_reader(
        BufferedReaderGeometry(4 * ratio, 4, 17),
        samples_per_frame=32,
        fft_samples=32 * frames,
        sample_rate=32 * frames * ffts,
        subintegration_s=Decimal(1),
        reference_rate_allowance=allowance,
    )


@pytest.mark.parametrize(
    "allowance,frames,ffts,ratio,expected",
    [
        (0.0, 8, 50, 3, 1206),
        (0.01, 8, 50, 3, 1218),
        (0.006, 8, 50, 3, 1212),
        (1 / 7, 3, 5, 2, 38),
        (1 / 6, 3, 2, 5, 45),
        (np.nextafter(1 / 6, 0), 3, 2, 5, 45),
        (np.float64(1 / 3), 1, 1, 1, 3),
    ],
)
def test_floor_request_before_guards_and_buffer_ratio(
    allowance, frames, ffts, ratio, expected
):
    result = geometry(allowance, frames=frames, ffts=ffts, ratio=ratio)
    assert result.allocation_frames == expected
    assert result.frames_per_fft == frames
    assert result.ffts_per_request == ffts
    assert result.buffered_ffts == 17
    assert result.framed_allocation_bytes(103) == expected * 103


def test_default_geometry_unchanged_and_safe_large_integer_quantums():
    assert geometry() == geometry(0.0) == geometry(np.float64(0))
    count = 2**53 - 4
    result = geometry(0.0, frames=count, ffts=1, ratio=2)
    assert result.allocation_frames == 2 * (count + 2)
    assert result.framed_allocation_bytes(8224) == 2 * (count + 2) * 8224


@pytest.mark.parametrize(
    "frames,rate",
    [(2**53, 0.0), (2**53 - 2, 0.0), (2**53 - 4, 1.0), (8, 1e308)],
)
def test_dangerous_float64_frame_counts_fail_closed(frames, rate):
    with pytest.raises(ConfigurationError, match="frame count"):
        geometry(rate, frames=frames, ffts=1)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        -1.0,
        Fraction(1, 100),
        Decimal(".01"),
        1,
        "0.01",
        float("nan"),
        float("inf"),
        np.float32(0.01),
    ],
)
def test_geometry_requires_explicit_nonnegative_float64_operand(bad):
    with pytest.raises(ConfigurationError, match="allowance"):
        geometry(bad)


@pytest.mark.parametrize(
    "bad,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_framed_bytes_positive_integer_only(bad, error):
    with pytest.raises(error):
        geometry().framed_allocation_bytes(bad)


def test_explicit_model_group_reference_positive_first_derivative_only():
    models = [
        model(0, (4e-6, 0, 900)),
        model(1, (100e-6, -0.01, 900)),
        model(2, (-100e-6, -0.9)),
        model(3, (0, 10)),
        model(4, (1e-6,)),
        model(5, (1e-6, 0.006)),
    ]
    actual = reference_model_rate_allowance(
        models, reference_epoch_seconds=Decimal(0)
    )
    assert type(actual) is float
    assert actual == 0.01
    assert (
        reference_model_rate_allowance(
            [
                replace(m, delay_coefficients_s=m.delay_coefficients_s[:2])
                for m in models
            ],
            reference_epoch_seconds=Decimal(0),
        )
        == actual
    )
    assert (
        reference_model_rate_allowance(
            [model(0, (-1, 100))], reference_epoch_seconds=Decimal(0)
        )
        == 0
    )


@pytest.mark.parametrize(
    "models,epoch,match",
    [
        ([], Decimal(0), "empty"),
        ([model(0, (1,), Decimal(1))], Decimal(0), "epoch"),
        ([model(0, (1,)), model(0, (1,))], Decimal(0), "duplicate"),
        ([model(0, ())], Decimal(0), "finite"),
        ([model(0, (1, float("nan")))], Decimal(0), "finite"),
        ([model(0, (1, 0, float("inf")))], Decimal(0), "finite"),
        ([model(0, (1,))], 0, "Decimal"),
    ],
)
def test_ambiguous_or_invalid_reference_group_fails(models, epoch, match):
    with pytest.raises(ValueError, match=match):
        reference_model_rate_allowance(models, reference_epoch_seconds=epoch)


def test_threshold_arithmetic_increment_before_add_is_explicit():
    for rate, expected in (
        (0.005, 404),
        (0.015, 408),
        (0.0025, 403),
        (np.nextafter(0.0025, 0), 403),
        (np.nextafter(np.nextafter(0.0025, 0), 0), 403),
    ):
        allowance = reference_model_rate_allowance(
            [model(0, (100e-6, -rate))], reference_epoch_seconds=Decimal(0)
        )
        assert geometry(allowance).allocation_frames == 3 * expected
    assert int(400 * (1 + 0.005)) + 2 == 403
    exact = Fraction(0.015) * 400 + 400
    assert exact.numerator // exact.denominator + 2 == 407
    assert int(Decimal(str(np.nextafter(0.0025, 0))) * 400 + 400) + 2 == 402


def test_model_lower_precision_coefficients_rejected():
    lower = replace(
        model(0, (1, 0.01)), delay_coefficients_s=(1.0, np.float32(0.01))
    )
    with pytest.raises(ValueError, match="float64"):
        reference_model_rate_allowance(
            [lower], reference_epoch_seconds=Decimal(0)
        )
