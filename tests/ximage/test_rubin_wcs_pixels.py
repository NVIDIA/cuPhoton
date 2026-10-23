# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuphoton.ximage.data import (
    RubinImageBounds,
    local_pixel_is_addressable,
    local_pixel_is_inside,
    local_to_absolute_pixel,
)


def test_physical_footprint_and_addressable_pixels_are_distinct() -> None:
    bounds = SimpleNamespace(
        x_start=1000,
        x_stop=1100,
        y_start=-500,
        y_stop=-300,
    )

    assert local_pixel_is_inside(bounds, x=-0.25, y=40.0) is True
    assert local_pixel_is_addressable(bounds, x=-0.25, y=40.0) is False
    assert local_pixel_is_addressable(bounds, x=99.75, y=199.75) is True
    assert local_pixel_is_addressable(bounds, x=100.0, y=40.0) is False
    assert local_to_absolute_pixel(bounds, x=12.5, y=20.25) == (
        1012.5,
        -479.75,
    )


@pytest.mark.parametrize(
    "coordinate",
    [float("inf"), float("-inf"), float("nan")],
)
def test_nonfinite_local_pixels_are_outside(coordinate: float) -> None:
    bounds = RubinImageBounds(y_start=0, y_stop=20, x_start=0, x_stop=10)

    assert local_pixel_is_inside(bounds, x=coordinate, y=0.0) is False
    assert local_pixel_is_addressable(bounds, x=0.0, y=coordinate) is False


def test_physical_and_addressable_boundaries_are_half_open() -> None:
    bounds = RubinImageBounds(y_start=20, y_stop=40, x_start=10, x_stop=20)

    assert local_pixel_is_inside(bounds, x=-0.5, y=-0.5) is True
    assert local_pixel_is_inside(bounds, x=9.5, y=19.5) is False
    assert local_pixel_is_addressable(bounds, x=0.0, y=0.0) is True
    assert local_pixel_is_addressable(bounds, x=10.0, y=20.0) is False


def test_absolute_translation_does_not_validate_containment() -> None:
    bounds = RubinImageBounds(
        y_start=-20,
        y_stop=-10,
        x_start=100,
        x_stop=110,
    )

    assert local_to_absolute_pixel(bounds, x=-2.5, y=12.0) == (97.5, -8.0)
