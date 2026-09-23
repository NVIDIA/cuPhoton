# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from cuphoton.core.benchmark import (
    BenchmarkOptions,
    BenchmarkRound,
    build_benchmark_report,
)
from cuphoton.core.bulk import validate_identifier


def _receipts(options):
    return [
        {
            **item.to_payload(),
            "status": "success",
            "batch_wall_sec": duration,
            "worker_wall_max_sec": duration / 2,
            "summary_path": f"rounds/{item.round_id}/summary.json",
        }
        for item, duration in zip(options.rounds(), (100.0, 20.0, 2.0, 4.0))
    ]


def test_report_retains_first_round_and_excludes_only_explicit_warmup():
    options = BenchmarkOptions(warmup_rounds=1, measure_rounds=3)
    receipts = _receipts(options)
    report = build_benchmark_report(options, receipts)
    assert report["status"] == "success"
    assert report["rounds"] == receipts
    assert report["measured_batch_wall_sec"] == {
        "min": 2.0,
        "median": 4.0,
        "max": 20.0,
    }


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "wrong_phase",
        "bool_index",
        "failed",
        "bad_time",
        "extra",
    ],
)
def test_invalid_evidence_has_no_accepted_aggregate(defect):
    options = BenchmarkOptions(warmup_rounds=1, measure_rounds=3)
    receipts = _receipts(options)
    if defect == "missing":
        receipts.pop()
    elif defect == "duplicate":
        receipts[2] = dict(receipts[1])
    elif defect == "wrong_phase":
        receipts[1]["phase"] = "warmup"
    elif defect == "bool_index":
        receipts[1]["index"] = False
    elif defect == "failed":
        receipts[0]["status"] = "failed"
    elif defect == "bad_time":
        receipts[1]["batch_wall_sec"] = -1
    else:
        receipts.append(dict(receipts[-1]))
    original = copy.deepcopy(receipts)
    report = build_benchmark_report(options, receipts)
    assert report["status"] == "failed"
    assert report["measured_batch_wall_sec"] is None
    assert report["rounds"] == original
    assert report["errors"]


def test_cleanup_failure_invalidates_complete_measurements():
    options = BenchmarkOptions(measure_rounds=2)
    report = build_benchmark_report(
        options,
        _receipts(options),
        errors=[{"phase": "close", "message": "worker still running"}],
    )
    assert report["status"] == "failed"
    assert report["measured_batch_wall_sec"] is None
    assert report["errors"][0]["phase"] == "close"


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_rounds", -1),
        ("warmup_rounds", False),
        ("measure_rounds", 0),
        ("measure_rounds", 1.5),
        ("measure_rounds", True),
    ],
)
def test_options_reject_invalid_counts(field, value):
    with pytest.raises(ValueError):
        BenchmarkOptions(**{field: value})


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), True, None, "1"]
)
def test_report_rejects_invalid_durations(value):
    options = BenchmarkOptions()
    receipts = _receipts(options)
    receipts[0]["worker_wall_max_sec"] = value
    assert build_benchmark_report(options, receipts)["status"] == "failed"


def test_round_run_ids_bind_artifacts_to_invocation_and_round():
    first, second = BenchmarkOptions(measure_rounds=2).rounds()
    assert first.run_id("parent") != second.run_id("parent")
    assert first.run_id("parent") != first.run_id("other")
    assert first.run_id("parent") == first.run_id("parent")
    validate_identifier(first.run_id("a" * 128), field="run_id")
    with pytest.raises(ValueError):
        BenchmarkRound("unknown", 0)
