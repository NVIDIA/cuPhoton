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
        command = args[0]
        stage = command[command.index("--stage") + 1]
        root = Path(command[command.index("--output") + 1]) / stage
        root.mkdir()
        (root / "summary.json").write_text(
            json.dumps({"extra_hashing_seconds": 0.0})
        )
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


@pytest.fixture
def benchmark_fixture(tmp_path: Path):
    from cuphoton.xscan.pipeline_benchmark.fixture import prepare_fixture

    return prepare_fixture(
        tmp_path / "inputs",
        images=1,
        image_size=48,
        candidates=2,
        stamp_size=9,
        seed=2026,
        device="cuda:0",
    )


def test_prepare_fixture_has_repeatable_inputs_and_cpu_model(
    tmp_path: Path, benchmark_fixture
) -> None:
    import torch

    from cuphoton.xscan.pipeline_benchmark.fixture import prepare_fixture

    config, items = benchmark.read_inputs(*benchmark_fixture)
    other = prepare_fixture(
        tmp_path / "repeat",
        images=1,
        image_size=48,
        candidates=2,
        stamp_size=9,
        seed=2026,
        device="cuda:0",
    )
    other_config, other_items = benchmark.read_inputs(*other)
    assert items[0].candidates == other_items[0].candidates
    for name in ("reference", "target", "variance", "fit_mask"):
        left, right = getattr(items[0], name), getattr(other_items[0], name)
        assert left.sha256 == right.sha256
        np.testing.assert_array_equal(np.load(left.path), np.load(right.path))
    models = [
        torch.load(
            Path(value.checkpoint_dir) / "checkpoint.pt", weights_only=False
        )
        for value in (config, other_config)
    ]
    assert models[0]["benchmark_fixture"]["trained"] is False
    for key, value in models[0]["model_state"].items():
        assert value.device.type == "cpu"
        torch.testing.assert_close(
            value, models[1]["model_state"][key], rtol=0, atol=0
        )


@pytest.mark.parametrize("order", ["pipeline-first", "staged-first"])
def test_compare_cpu_orchestration_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_fixture,
    order: str,
) -> None:
    from cuphoton.xscan import device_pipeline
    from cuphoton.xscan.pipeline_benchmark import stages

    config, items = benchmark.read_inputs(*benchmark_fixture)
    science = {name: np.array([0.25, 0.75]) for name in stages.SCIENCE_KEYS}
    predictions = [
        {
            **candidate.to_payload(),
            "logit": 0.25,
            "probability": 0.75,
            "decision": True,
        }
        for candidate in items[0].candidates
    ]
    metadata = {
        "xpois": {"xpois_chi2": 1.0, "basis_terms": []},
        "xfit": {"parameter_names": ["amplitude"], "feature_names": ["flux"]},
        "xscan": {"predictions": predictions},
    }
    # Exercise the real file contracts, stage summaries, round bookkeeping and
    # audit. Only CUDA computation and process launching are replaced on CPU.
    monkeypatch.setattr(stages, "_setup", lambda *_: {})
    monkeypatch.setattr(
        stages,
        "_run_item",
        lambda stage, *_: (
            {
                key: value
                for key, value in science.items()
                if key.startswith(stage + ".")
            },
            metadata[stage],
            {},
        ),
    )
    monkeypatch.setattr(
        device_pipeline,
        "decode_device_pipeline_evidence",
        lambda *_, **__: science,
    )
    launched = []

    def child(command, log, *, timeout):
        stage = command[command.index("--stage") + 1]
        output = Path(command[command.index("--output") + 1])
        launched.append(stage)
        if stage != "pipeline":
            stages.run_stage(stage, config, items, output)
        else:
            for index, phase in enumerate(("warmup", "measured")):
                root = output / f"round-{index:03d}"
                root.mkdir()
                benchmark.write_json(
                    root / "item-0000.json",
                    {
                        "item_id": items[0].item_id,
                        "configuration_sha256": config.configuration_sha256,
                        "predictions": predictions,
                        "xpois": {"chi2": 1.0},
                        "scientific_evidence": {
                            "xpois": {"basis_terms": []},
                            "parameter_names": ["amplitude"],
                            "feature_names": ["flux"],
                        },
                    },
                )
            benchmark.write_json(
                output / "summary.json",
                {
                    "device_provenance": {
                        "gpu_uuid": "GPU-test",
                        "gpu_name": "CPU test double",
                    },
                    "rounds": [
                        {"phase": "warmup", "batch_seconds": 2.0},
                        {"phase": "measured", "batch_seconds": 1.0},
                    ],
                },
            )
        return 0.1

    monkeypatch.setattr(benchmark, "run_child", child)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout="610.57.04\n"),
    )
    args = argparse.Namespace(
        stage="compare",
        output=tmp_path / "compare",
        config=benchmark_fixture[0],
        items=benchmark_fixture[1],
        warmup=1,
        repeat=1,
        timeout=10,
        order=order,
    )
    benchmark.run_benchmark(args)
    report = json.loads((args.output / "report.json").read_text())
    assert report["parity_passed"]
    assert (
        len(json.loads((args.output / "parity.json").read_text())["checks"])
        == 2
    )
    assert launched.count("pipeline") == 1
    assert (
        launched.count("xpois")
        == launched.count("xfit")
        == launched.count("xscan")
        == 2
    )
    assert (launched[0] == "pipeline") == (order == "pipeline-first")
    assert report["provenance"]["device"]["driver_version"] == "610.57.04"
    assert (
        report["fresh_stages_over_warm_pipeline_ratio"]
        < report["raw_fresh_stages_over_warm_pipeline_ratio"]
    )
    for row in report["staged"]["rounds"]:
        assert row["extra_hashing_seconds"] > 0
        assert row["batch_seconds_without_extra_hashing"] == pytest.approx(
            row["batch_seconds"] - row["extra_hashing_seconds"]
        )
    # Candidate identity and exact scientific values both gate acceptance.
    path = args.output / "pipeline/round-001/item-0000.json"
    result = json.loads(path.read_text())
    result["predictions"][0]["candidate_id"] = "changed"
    benchmark.write_json(path, result)
    with pytest.raises(ValueError, match="candidate identity"):
        benchmark.audit(args)
    result["predictions"] = predictions
    benchmark.write_json(path, result)
    monkeypatch.setattr(
        device_pipeline,
        "decode_device_pipeline_evidence",
        lambda *_, **__: {
            name: values + 1 for name, values in science.items()
        },
    )
    assert not benchmark.audit(args)["passed"]
    with pytest.raises(FileExistsError):
        stages.run_stage(
            "xpois", config, items, args.output / "staged/round-001"
        )


def test_provenance_discovers_the_installed_cupy_distribution(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        benchmark.importlib.metadata,
        "packages_distributions",
        lambda: {"cupy": ["cupy"]},
    )
    monkeypatch.setattr(
        benchmark.importlib.metadata, "version", lambda name: "test-" + name
    )
    assert benchmark.provenance()["packages"]["cupy"] == "test-cupy"
