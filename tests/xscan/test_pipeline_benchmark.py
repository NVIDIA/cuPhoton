# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scientific acceptance and process failure gates for the benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xscan.pipeline_benchmark import runner as benchmark
from cuphoton.xscan.pipeline_benchmark.runner import (
    compare_arrays,
    run_child,
)


def test_science_acceptance_preserves_nan_and_row_identity() -> None:
    expected = {"fit": np.array([[1.0, np.nan], [2.0, 3.0]])}
    assert compare_arrays(expected, expected)["passed"]
    reordered = {"fit": expected["fit"][::-1]}
    assert not compare_arrays(expected, reordered)["passed"]
    changed = {"fit": np.array([[1.0, 0.0], [2.0, 3.0]])}
    assert not compare_arrays(expected, changed)["passed"]
    rounded = {"fit": np.array([[1.0, np.nan], [2.0, 3.0 + 1e-12]])}
    assert not compare_arrays(expected, rounded)["passed"]


@pytest.mark.parametrize(
    "actual",
    [
        {"wrong": np.array([1.0])},
        {"fit": np.array([[1.0]])},
        {"fit": np.array([1.0], dtype=np.float32)},
    ],
)
def test_science_rejects_schema_drift(actual: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError):
        compare_arrays({"fit": np.array([1.0])}, actual)


def test_science_discrete_diagnostics_are_exact() -> None:
    assert not compare_arrays(
        {"valid": np.array([True])}, {"valid": np.array([False])}
    )["passed"]


def test_child_failure_retains_diagnostics(tmp_path: Path) -> None:
    log = tmp_path / "child.log"
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_child(
            [
                sys.executable,
                "-c",
                "print('stage failed'); raise SystemExit(7)",
            ],
            log,
            timeout=10,
        )
    assert raised.value.returncode == 7
    assert "stage failed" in log.read_text()


def test_child_timeout_is_not_success(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_child(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            tmp_path / "timeout.log",
            timeout=0.2,
        )


def test_failed_stage_round_retains_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = subprocess.CalledProcessError(7, ["xfit"])
    outcomes = iter([0.1, 0.2, 0.3, 0.4, failure])

    def child(*args: object, **kwargs: object) -> float:
        result = next(outcomes)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(benchmark, "run_child", child)
    args = argparse.Namespace(
        output=tmp_path,
        config=tmp_path / "config.json",
        items=tmp_path / "items.json",
        warmup=1,
        repeat=2,
        timeout=10,
    )
    with pytest.raises(subprocess.CalledProcessError) as raised:
        benchmark.measure_stages(args)
    assert raised.value is failure
    root = tmp_path / "staged"
    previous = json.loads((root / "round-000/timing.json").read_text())
    failed = json.loads((root / "round-001/timing.json").read_text())
    assert previous["status"] == "success"
    assert failed["index"] == 1 and failed["phase"] == "measured"
    assert failed["status"] == "failed"
    assert failed["completed_stages"] == ["xpois"]
    assert failed["stages"]["xpois"]["external_seconds"] == 0.4
    assert failed["stages"]["xfit"]["status"] == "failed"
    assert (
        failed["batch_seconds"]
        >= failed["stages"]["xfit"]["external_seconds"]
        >= 0
    )
    assert failed["failure"]["type"] == "CalledProcessError"
    assert not (root / "round-002").exists()
    assert not (root / "summary.json").exists()


def test_failed_pipeline_round_retains_completed_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuphoton.xscan import device_pipeline

    # Replace CUDA bindings so the failure path can be checked on a CPU.
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=lambda _: SimpleNamespace(use=lambda: None),
                runtime=SimpleNamespace(deviceSynchronize=lambda: None),
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda _: None,
            cuda=SimpleNamespace(
                set_device=lambda _: None, synchronize=lambda _: None
            ),
        ),
    )
    config = SimpleNamespace(
        device="cuda:0", inference_policy={"worker_cpu_threads": 1}
    )
    monkeypatch.setattr(benchmark, "read_inputs", lambda *_: (config, [0, 1]))
    monkeypatch.setattr(
        device_pipeline,
        "DeviceWorkerContext",
        SimpleNamespace(initialize=lambda _: object()),
    )
    result = SimpleNamespace(to_payload=lambda: {"completed": True})
    failure = RuntimeError("injected item failure")
    outcomes = iter([result, result, result, failure])

    def run_item(*_: object) -> object:
        value = next(outcomes)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(device_pipeline, "run_device_pipeline_item", run_item)
    args = argparse.Namespace(
        output=tmp_path, config=None, items=None, warmup=1, repeat=2
    )
    with pytest.raises(RuntimeError, match="injected item failure") as raised:
        benchmark.pipeline_worker(args)
    assert raised.value is failure
    failed = json.loads((tmp_path / "round-001/timing.json").read_text())
    assert failed["index"] == 1 and failed["phase"] == "measured"
    assert failed["status"] == "failed" and failed["completed_items"] == 1
    assert failed["batch_seconds"] >= 0
    assert failed["failure"]["type"] == "RuntimeError"
    assert (tmp_path / "round-001/item-0000.json").is_file()
    assert not (tmp_path / "round-001/item-0001.json").exists()
    assert not (tmp_path / "round-002").exists()
    assert not (tmp_path / "summary.json").exists()
