# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declared reference epochs contribute independently of execution time."""

from dataclasses import replace
from decimal import Decimal
from itertools import permutations

import pytest

from cuphoton.xbeam.adapters.radio.models import (
    BufferedReaderGeometry,
    DelayCoordinateModel,
)
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedWindowGeometry,
    declared_model_rate_allowance,
)


def model(station, epoch, constant, rate):
    return DelayCoordinateModel(
        station_id=station,
        epoch_offset_s=Decimal(epoch),
        delay_coefficients_s=(constant, rate),
        u_coefficients_m=(0.0,),
        v_coefficients_m=(0.0,),
        w_coefficients_m=(0.0,),
    )


@pytest.mark.parametrize(
    "first_constant,first_rate,future_constant,future_rate,request_frames",
    [
        (-0.0001, -0.01, 0.0001, -0.01, 406),
        (-0.0001, -0.01, 0.0001, -0.02, 410),
        (0.0001, -0.006, 0.0001, -0.01, 406),
        (0.0001, -0.01, -0.0001, -0.02, 406),
    ],
)
def test_unused_future_epoch_still_contributes_reference_sizing(
    first_constant, first_rate, future_constant, future_rate, request_frames
):
    references = (
        model(0, 0, 0.000004, 0.0),
        model(1, 0, first_constant, first_rate),
        model(0, 3, 0.000004, 0.0),
        model(1, 3, future_constant, future_rate),
    )
    reader = BufferedReaderGeometry(24, 8, 20)
    for ordered in permutations(references):
        geometry = BufferedWindowGeometry.from_reader(
            reader,
            samples_per_frame=32768,
            fft_samples=262144,
            sample_rate=4096000000,
            subintegration_s=Decimal("0.0032"),
            reference_rate_allowance=declared_model_rate_allowance(ordered),
        )
        assert geometry.allocation_frames == 3 * request_frames


def test_reference_aggregation_ignores_curvature_and_nonpositive_constants():
    references = (
        replace(
            model(17, 29, 0.75, 0.125),
            delay_coefficients_s=(0.75, 0.125, -10000.0),
        ),
        model(18, 29, 0.0, -100.0),
        model(17, 103, -1.0, -100.0),
        model(18, 103, 1.0, -0.25),
    )
    assert declared_model_rate_allowance(references) == 0.25


def test_same_station_across_epochs_is_valid_duplicate_epoch_is_not():
    first, later = model(3, 0, 1.0, 0.125), model(3, 100, 1.0, 0.25)
    assert declared_model_rate_allowance((first, later)) == 0.25
    with pytest.raises(ValueError, match="duplicate station"):
        declared_model_rate_allowance((first, later, replace(first)))


@pytest.mark.parametrize("epoch", ["0", Decimal("NaN"), Decimal("Infinity")])
def test_invalid_declared_epoch_fails_before_group_selection(epoch):
    with pytest.raises(ValueError, match="finite Decimal"):
        declared_model_rate_allowance(
            (replace(model(0, 0, 1.0, 0.0), epoch_offset_s=epoch),)
        )


def test_declared_reference_inventory_must_be_nonempty_and_typed():
    with pytest.raises(ValueError, match="empty"):
        declared_model_rate_allowance(())
    with pytest.raises(TypeError, match="DelayCoordinateModel"):
        declared_model_rate_allowance((object(),))
