# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xray.zero_offset import find_value_drop_position


def test_find_value_drop_position_selects_nearest_zero_offset():
    time = np.linspace(-2.0, 2.0, 240)
    trace = np.sin(55.0 * time) - 0.05 * time**3

    result = find_value_drop_position(time, trace, zero_offset=0.0)

    assert result.status == "ok"
    assert result.change == "drop"
    assert result.selected_index == int(np.argmin(np.abs(time)))
    assert result.selected_time == pytest.approx(time[result.selected_index])
    assert result.extrema_index is not None
    assert result.extrema_time == pytest.approx(time[result.extrema_index])
    assert result.polynomial_coefficients is not None


def test_find_value_drop_position_returns_negative_index_for_flat_trace():
    time = np.linspace(-1.0, 1.0, 32)
    trace = np.linspace(2.0, 3.0, 32)

    result = find_value_drop_position(time, trace, zero_offset=0.0)

    assert result.status == "insufficient-extrema"
    assert result.selected_index == -1
    assert result.selected_time is None


def test_find_value_drop_position_validates_shape():
    with pytest.raises(ValueError, match="same shape"):
        find_value_drop_position([0.0, 1.0], [0.0], zero_offset=0.0)
