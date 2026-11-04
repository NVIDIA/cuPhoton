# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the notebook's scientific comparisons and run displays."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sc26_demo", Path(__file__).parents[1] / "demo.py"
)
assert _SPEC is not None and _SPEC.loader is not None
demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(demo)


def test_boolean_comparison_requires_exact_flags(capsys):
    flags = np.array([True, False, True])
    receipt = demo.compare_arrays(flags, flags.copy(), label="converged")
    assert receipt["max_abs_error"] == 0
    assert receipt["shape"] == (3,)
    assert capsys.readouterr().out == ""
    with pytest.raises(AssertionError):
        demo.compare_arrays(flags, ~flags, rtol=1, atol=1)


def test_numeric_comparison_keeps_tolerance_and_matching_nans():
    expected = np.array([1.0, np.nan, -2.0])
    actual = np.array([1.000001, np.nan, -2.0])
    receipt = demo.compare_arrays(expected, actual, rtol=1e-5)
    assert receipt["max_abs_error"] == pytest.approx(1e-6)
    with pytest.raises(AssertionError):
        demo.compare_arrays(expected, actual, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    ("executor", "disk_records"),
    [("dragon", False), ("dragon", True), ("processes", False)],
)
def test_summary_shows_completions_and_rejects_changed_workers(
    tmp_path, monkeypatch, executor, disk_records
):
    tables = []
    monkeypatch.setattr(demo, "_show", tables.append)
    provenance = {
        "worker_id": 0,
        "requested_host": "demo-host",
        "requested_gpu_id": 2,
        "hostname": "demo-host",
        "pid": 123,
        "cuda_visible_devices": "2",
        "gpu": {"device_index": 0, "uuid": "GPU-demo"},
    }
    rounds = []
    for phase in ("warmup", "measure"):
        round_id = f"{phase}-0000"
        relative = f"rounds/{round_id}/summary.json"
        rounds.append({"round_id": round_id, "summary_path": relative})
        round_path = tmp_path / relative
        round_path.parent.mkdir(parents=True)
        payload = {
            "status": "success",
            "round_id": round_id,
            "terminal_record_audit": {
                "ok": True,
                "failed_item_ids": [],
                "expected_count": 1,
                "terminal_record_count": 1,
            },
            "records": [
                {
                    "item_id": "pair-0",
                    "worker_id": 0,
                    "status": "success",
                    "worker_seconds": 0.1,
                }
            ],
            "worker_results": [
                {
                    "worker_id": 0,
                    "provenance": provenance,
                    "assigned_item_ids": ["pair-0"],
                    "success_count": 1,
                    "failed_count": 0,
                }
            ],
        }
        if executor == "processes":
            # Local rounds inherit their label from the root benchmark.
            payload.pop("round_id")
        worker = payload["worker_results"][0]
        if disk_records:
            # xPois rounds retain worker and item records in separate files.
            (round_path.parent / "records").mkdir()
            (round_path.parent / "workers").mkdir()
            (round_path.parent / "records" / "pair-0.json").write_text(
                json.dumps(payload.pop("records")[0])
            )
            worker.pop("assigned_item_ids")
            payload.pop("worker_results")
            worker_path = round_path.parent / "workers" / "worker-0000.json"
            worker_path.write_text(json.dumps(worker))
        round_path.write_text(json.dumps(payload))
    summary = {
        "status": "success",
        "executor": executor,
        "benchmark": {"rounds": rounds},
    }
    if executor == "dragon":
        summary["process_exit_audit"] = {"ok": True}
        summary["ready_messages"] = [
            {"worker_id": 0, "provenance": provenance}
        ]
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    placement = demo.summarize_run(tmp_path)

    assert placement["requested_gpu_id"].tolist() == [2]
    assert placement["gpu.device_index"].tolist() == [0]
    completions = next(t for t in tables if "same_as_ready" in t)
    assert completions["round"].tolist() == ["warmup-0000", "measure-0000"]
    if executor == "dragon":
        assert completions["same_as_ready"].all()
    else:
        assert completions["same_as_ready"].isna().all()
    assert completions["item_examples"].tolist() == ["pair-0", "pair-0"]
    records = next(t for t in tables if "item_id" in t)
    assert records["status"].tolist() == ["success"]

    if executor == "processes":
        return
    worker["provenance"]["pid"] = 456
    if disk_records:
        worker_path.write_text(json.dumps(worker))
    else:
        round_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="changed after READY"):
        demo.summarize_run(tmp_path)
