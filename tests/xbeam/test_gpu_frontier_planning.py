# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.frontier_planning import gpu_frontier_bounds
from tests.xbeam.test_gpu_backend import _cupy


@pytest.mark.gpu
@pytest.mark.parametrize("size", [17, 1001, 262144])
def test_gpu_bounds_match_cpu_with_empty_frontiers_and_missing_slots(size):
    cp = _cupy()
    physical = np.arange(size, dtype=np.int64) * 2 + 7
    destinations = (
        np.random.default_rng(42).permutation(size).astype(np.int32)
    )
    stops = np.arange(1, 15, dtype=np.int64) * size // 13
    info = {}
    actual = gpu_frontier_bounds(
        physical, destinations, stops, cp, "gpu", info
    )
    first = 0
    for frontier, stop in enumerate(stops):
        selected = physical[(destinations >= first) & (destinations < stop)]
        expected = (
            [int(selected.min()), int(selected.max()) + 1]
            if selected.size
            else [-1, -1]
        )
        np.testing.assert_array_equal(actual[frontier], expected)
        first = stop
    assert info["gpu_planned_mappings"] == 1


@pytest.mark.gpu
def test_gpu_sort_rejects_nonadjacent_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        gpu_frontier_bounds(
            np.arange(5),
            np.array([2, 0, 1, 2, 3], np.int32),
            np.array([2, 4]),
            _cupy(),
            "gpu",
            {},
        )


@pytest.mark.gpu
def test_gpu_allocation_failure_returns_cpu_fallback(monkeypatch):
    cp = _cupy()

    def fail(*args):
        raise MemoryError("injected sort workspace failure")

    monkeypatch.setattr(cp, "argsort", fail)
    info = {}
    assert (
        gpu_frontier_bounds(
            np.arange(5),
            np.arange(5, dtype=np.int32),
            np.array([2, 5]),
            cp,
            "gpu",
            info,
        )
        is None
    )
    assert info["gpu_planner_fallbacks"] == {"allocation-failed": 1}


@pytest.mark.gpu
def test_gpu_budget_guard_precedes_allocations(monkeypatch):
    cp = _cupy()
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (1, 1))

    def fail(*args):
        pytest.fail("budget rejection allocated device arrays")

    monkeypatch.setattr(cp, "asarray", fail)
    info = {}
    assert (
        gpu_frontier_bounds(
            np.arange(5),
            np.arange(5, dtype=np.int32),
            np.array([2, 5]),
            cp,
            "gpu",
            info,
        )
        is None
    )
    assert info["gpu_planner_fallbacks"] == {"device-budget": 1}


def test_cpu_override_does_not_require_gpu_imports():
    info = {}
    assert (
        gpu_frontier_bounds(
            np.arange(5), np.arange(5), np.array([5]), None, "cpu", info
        )
        is None
    )
    assert info["gpu_planner_fallbacks"] == {"disabled": 1}
