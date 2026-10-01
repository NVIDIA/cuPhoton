# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that wheel acceptance rejects incomplete execution evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "wheels" / "test_stack.py"
SPEC = importlib.util.spec_from_file_location("wheel_stack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)


@pytest.fixture
def workers():
    return [
        {
            "worker_id": worker_id,
            "hostname": "node",
            "pid": 100 + worker_id,
            "scale": 1.7 + worker_id / 10,
        }
        for worker_id in range(2)
    ]


def test_worker_evidence_rejects_missing_worker(workers):
    with pytest.raises(RuntimeError, match="Missing worker"):
        stack.check_workers(workers[:1], 2)


def test_worker_evidence_rejects_duplicated_identity(workers):
    workers[1]["worker_id"] = 0
    with pytest.raises(RuntimeError, match="duplicate worker identities"):
        stack.check_workers(workers, 2)


def test_worker_evidence_rejects_reused_process(workers):
    workers[1]["pid"] = workers[0]["pid"]
    with pytest.raises(RuntimeError, match="distinct processes"):
        stack.check_workers(workers, 2)


def test_worker_evidence_rejects_wrong_numerical_result(workers):
    workers[1]["scale"] = 0
    with pytest.raises(AssertionError):
        stack.check_workers(workers, 2)


def test_cpu_product_fixture_has_known_solution():
    result = stack.solve("cpu", worker_id=1)
    assert result["backend"] == "cpu"
    assert result["scale"] == pytest.approx(1.8, abs=1e-8)
    assert result["fit_pixel_count"] == (48 - 8) * (53 - 8)
