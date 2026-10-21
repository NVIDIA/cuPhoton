# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.fft
import scipy.ndimage

from cuphoton.core.arrays import (
    array_module,
    fft_module,
    ndimage_module,
    to_host,
)


def test_numpy_inputs_select_numpy_without_importing_cupy() -> None:
    assert array_module() is np
    assert array_module(np.zeros(3), 1.0, None) is np
    assert ndimage_module(np) is scipy.ndimage
    assert fft_module(np) is scipy.fft
    assert "cupy" not in sys.modules or sys.modules["cupy"] is not None


def test_to_host_returns_numpy_views_and_wraps_sequences() -> None:
    array = np.arange(4.0)
    assert to_host(array) is array
    assert to_host([1, 2]).tolist() == [1, 2]
    assert to_host(np.float32(2.5)).dtype == np.float32


def test_cupy_inputs_select_cupy_and_move_to_host() -> None:
    cupy = pytest.importorskip("cupy")
    import cupyx.scipy.fft
    import cupyx.scipy.ndimage

    device = cupy.arange(3)
    assert array_module(np.zeros(2), device) is cupy
    assert ndimage_module(cupy) is cupyx.scipy.ndimage
    assert fft_module(cupy) is cupyx.scipy.fft
    host = to_host(device)
    assert isinstance(host, np.ndarray)
    assert host.tolist() == [0, 1, 2]
