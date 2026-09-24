# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a persistent device pipeline with three file-mediated processes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODULE = "cuphoton.xscan.pipeline_benchmark"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_inputs(config_path: Path, items_path: Path) -> tuple[Any, list[Any]]:
    from cuphoton.xscan.device_pipeline import (
        DevicePipelineConfig,
        DevicePipelineItem,
    )

    config = DevicePipelineConfig.from_payload(
        json.loads(config_path.read_text())
    )
    items = [
        DevicePipelineItem.from_payload(value)
        for value in json.loads(items_path.read_text())
    ]
    if not items or len({item.item_id for item in items}) != len(items):
        raise ValueError("items must be nonempty with unique item IDs")
    return config, items


def compare_arrays(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Require exact values, shapes, dtypes and NaN locations."""
    if expected.keys() != actual.keys():
        raise ValueError("scientific evidence array names differ")
    checks = {}
    for name, left in expected.items():
        right = actual[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{name}: scientific shape or dtype differs")
        equal = np.array_equal(left, right, equal_nan=True)
        finite = np.isfinite(left) & np.isfinite(right)
        error = (
            float(
                np.max(
                    np.abs(
                        left[finite].astype(np.float64)
                        - right[finite].astype(np.float64)
                    )
                )
            )
            if np.any(finite)
            else 0.0
        )
        checks[name] = {"equal": bool(equal), "max_absolute_error": error}
    return {
        "passed": all(v["equal"] for v in checks.values()),
        "arrays": checks,
    }


def run_child(command: list[str], log: Path, *, timeout: float) -> float:
    """Wait for an actual process exit; preserve its output on failure."""
    started = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except BaseException:
            # The session contains only this benchmark child and its children.
            # Clean it up on interrupts as well as ordinary timeouts/failures.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    return time.perf_counter() - started


def pipeline_worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    import cupy as cp
    import torch

    from cuphoton.xscan.device_pipeline import (
        DeviceWorkerContext,
        run_device_pipeline_item,
    )

    config, items = read_inputs(args.config, args.items)
    ordinal = int(config.device.split(":")[1])
    torch.set_num_threads(config.inference_policy["worker_cpu_threads"])
    torch.cuda.set_device(ordinal)
    cp.cuda.Device(ordinal).use()
    context = DeviceWorkerContext.initialize(config)
    cp.cuda.runtime.deviceSynchronize()
    setup_seconds = time.perf_counter() - started
    rounds = []
    for index in range(args.warmup + args.repeat):
        output = args.output / f"round-{index:03d}"
        output.mkdir()
        phase = "warmup" if index < args.warmup else "measured"
        completed_items = 0
        round_started = time.perf_counter()
        try:
            for item_index, item in enumerate(items):
                result = run_device_pipeline_item(item, context)
                write_json(
                    output / f"item-{item_index:04d}.json",
                    result.to_payload(),
                )
                completed_items += 1
            cp.cuda.runtime.deviceSynchronize()
            torch.cuda.synchronize(ordinal)
        except BaseException as exc:
            record = {
                "index": index,
                "phase": phase,
                "status": "failed",
                "batch_seconds": time.perf_counter() - round_started,
                "completed_items": completed_items,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
            try:
                write_json(output / "timing.json", record)
            except OSError as write_error:
                exc.add_note(
                    f"Could not preserve round timing: {write_error}"
                )
            raise
        record = {
            "index": index,
            "phase": phase,
            "status": "success",
            "batch_seconds": time.perf_counter() - round_started,
            "completed_items": completed_items,
        }
        rounds.append(record)
        write_json(output / "timing.json", record)
        print(json.dumps(record), flush=True)
    write_json(
        args.output / "summary.json",
        {
            "setup_seconds": setup_seconds,
            "context_load_seconds": context.load_seconds,
            "worker_seconds": time.perf_counter() - started,
            "configuration_sha256": config.configuration_sha256,
            "rounds": rounds,
            "gpu_name": torch.cuda.get_device_name(ordinal),
        },
    )


def measure_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output / "pipeline"
    output.mkdir()
    elapsed = run_child(
        [
            sys.executable,
            "-m",
            MODULE,
            "--pipeline-worker",
            "--config",
            str(args.config),
            "--items",
            str(args.items),
            "--output",
            str(output),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
        ],
        output / "process.log",
        timeout=args.timeout,
    )
    result = json.loads((output / "summary.json").read_text())
    result["invocation_external_seconds"] = elapsed
    return result


def measure_stages(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output / "staged"
    output.mkdir()
    rounds = []
    for index in range(args.warmup + args.repeat):
        root = output / f"round-{index:03d}"
        root.mkdir()
        phase = "cache_preparation" if index < args.warmup else "measured"
        record: dict[str, Any] = {
            "index": index,
            "phase": phase,
            "stages": {},
            "completed_stages": [],
        }
        started = time.perf_counter()
        try:
            for stage in ("xpois", "xfit", "xscan"):
                stage_started = time.perf_counter()
                elapsed = run_child(
                    [
                        sys.executable,
                        "-m",
                        f"{MODULE}.stages",
                        "--stage",
                        stage,
                        "--config",
                        str(args.config),
                        "--items",
                        str(args.items),
                        "--output-dir",
                        str(root),
                    ],
                    root / f"{stage}.log",
                    timeout=args.timeout,
                )
                record["stages"][stage] = {
                    "external_seconds": elapsed,
                    "status": "success",
                }
                record["completed_stages"].append(stage)
        except BaseException as exc:
            ended = time.perf_counter()
            record.update(
                status="failed",
                batch_seconds=ended - started,
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            record["stages"][stage] = {
                "external_seconds": ended - stage_started,
                "status": "failed",
            }
            try:
                write_json(root / "timing.json", record)
            except OSError as write_error:
                exc.add_note(
                    f"Could not preserve round timing: {write_error}"
                )
            raise
        record["batch_seconds"] = time.perf_counter() - started
        record["status"] = "success"
        rounds.append(record)
        write_json(root / "timing.json", record)
        print(json.dumps({"treatment": "staged", **record}), flush=True)
    result = {"rounds": rounds}
    write_json(output / "summary.json", result)
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from cuphoton.core.artifacts import file_sha256
    from cuphoton.xscan.device_pipeline import (
        _verify_item_hashes,
        decode_device_pipeline_evidence,
    )

    from .stages import load_science_arrays

    config, items = read_inputs(args.config, args.items)
    checks = []
    for index in range(args.warmup + args.repeat):
        root = args.output / "staged" / f"round-{index:03d}"
        stages = {
            stage: json.loads((root / stage / "summary.json").read_text())
            for stage in ("xpois", "xfit", "xscan")
        }
        for summary in stages.values():
            if summary["configuration_sha256"] != config.configuration_sha256:
                raise ValueError("staged configuration differs")
            if [entry["item"] for entry in summary["items"]] != [
                item.to_payload() for item in items
            ]:
                raise ValueError("staged candidate/item identity differs")
        for item_index, item in enumerate(items):
            pipeline_path = (
                args.output
                / "pipeline"
                / f"round-{index:03d}"
                / f"item-{item_index:04d}.json"
            )
            result = json.loads(pipeline_path.read_text())
            if result["item_id"] != item.item_id:
                raise ValueError("pipeline item identity differs")
            if result["configuration_sha256"] != config.configuration_sha256:
                raise ValueError("pipeline configuration differs")
            for candidate, prediction in zip(
                item.candidates, result["predictions"], strict=True
            ):
                for key, value in candidate.to_payload().items():
                    if prediction[key] != value:
                        raise ValueError(
                            "pipeline candidate identity differs"
                        )
                if prediction["decision"] != (
                    prediction["probability"] >= config.decision_threshold
                ):
                    raise ValueError("pipeline decision differs")
            expected = decode_device_pipeline_evidence(
                result["scientific_evidence"], config=config
            )
            metadata = {
                stage: summary["items"][item_index]["metadata"]
                for stage, summary in stages.items()
            }
            for name, value in result["xpois"].items():
                if metadata["xpois"][f"xpois_{name}"] != value:
                    raise ValueError(f"xPOIS {name} differs")
            evidence = result["scientific_evidence"]
            if (
                metadata["xpois"]["basis_terms"]
                != evidence["xpois"]["basis_terms"]
            ):
                raise ValueError("xPOIS basis metadata differs")
            for name in ("parameter_names", "feature_names"):
                if metadata["xfit"][name] != evidence[name]:
                    raise ValueError(f"xFit {name} differs")
            if metadata["xscan"]["predictions"] != result["predictions"]:
                raise ValueError("xScan predictions or identity differs")
            actual = load_science_arrays(root, item_index)
            checks.append(
                {
                    "round": index,
                    "item_id": item.item_id,
                    **compare_arrays(expected, actual),
                }
            )
    for item in items:
        _verify_item_hashes(item, when="after benchmark")
    for path, expected_digest in (
        (
            Path(config.checkpoint_dir) / "checkpoint.pt",
            config.checkpoint_sha256,
        ),
        (Path(config.feature_schema_path), config.feature_schema_sha256),
    ):
        if file_sha256(path) != expected_digest:
            raise ValueError(
                "checkpoint or feature schema changed after execution"
            )
    return {"passed": all(v["passed"] for v in checks), "checks": checks}


def provenance() -> dict[str, Any]:
    from cuphoton.core.artifacts import file_sha256

    package_root = Path(__file__).resolve().parents[2]
    numerical_sources = (
        "xscan/device_pipeline.py",
        "xpois/ois.py",
        "xfit/api.py",
        "xfit/backend.py",
        "xfit/models.py",
        "xfit/solver.py",
        "xscan/xfit_features.py",
        "xscan/training.py",
        "xscan/model.py",
    )
    packages = {}
    for name in ("cuphoton", "numpy", "cupy-cuda13x", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    policy_keys = (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "CUPY_CACHE_DIR",
        "CUDA_CACHE_PATH",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "environment": {key: os.environ.get(key) for key in policy_keys},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "benchmark_source_sha256": {
            path.name: file_sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "numerical_source_sha256": {
            name: file_sha256(package_root / name)
            for name in numerical_sources
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=9)
    parser.add_argument("--stamp-size", type=int, default=17)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--order",
        choices=("pipeline-first", "staged-first"),
        default="pipeline-first",
    )
    parser.add_argument(
        "--pipeline-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 1 or args.timeout <= 0:
        parser.error("repeat, warmup and timeout must be positive")
    if (args.config is None) != (args.items is None):
        parser.error("provide both --config and --items, or neither")
    args.output = args.output.resolve()
    if args.pipeline_worker:
        pipeline_worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.config is None:
            from .fixture import prepare_fixture

            args.config, args.items = prepare_fixture(
                args.output / "input",
                images=args.images,
                image_size=args.image_size,
                candidates=args.candidates,
                stamp_size=args.stamp_size,
                seed=args.seed,
                device=args.device,
            )
        args.config, args.items = args.config.resolve(), args.items.resolve()
        config, items = read_inputs(args.config, args.items)
        report: dict[str, Any] = {
            "schema": "cuphoton.pipeline-stage-benchmark/v1",
            "provenance": provenance(),
            "configuration": config.to_payload(),
            "items": [item.to_payload() for item in items],
            "order": args.order,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "fsync": False,
            "cache_policy": "preserve existing caches",
        }
        write_json(args.output / "manifest.json", report)
        treatments = (
            [("pipeline", measure_pipeline), ("staged", measure_stages)]
            if args.order == "pipeline-first"
            else [("staged", measure_stages), ("pipeline", measure_pipeline)]
        )
        for name, run in treatments:
            report[name] = run(args)
        acceptance = audit(args)
        write_json(args.output / "parity.json", acceptance)
        if not acceptance["passed"]:
            raise ValueError("scientific parity failed; see parity.json")
        report["parity_passed"] = True
        for name in ("pipeline", "staged"):
            values = [
                row["batch_seconds"]
                for row in report[name]["rounds"]
                if row["phase"] == "measured"
            ]
            report[name]["measured_batch_median_seconds"] = statistics.median(
                values
            )
        report["fresh_stages_over_warm_pipeline_ratio"] = (
            report["staged"]["measured_batch_median_seconds"]
            / report["pipeline"]["measured_batch_median_seconds"]
        )
        write_json(args.output / "report.json", report)
        print(
            json.dumps(
                {"report": str(args.output / "report.json"), "parity": True}
            )
        )
    except BaseException as exc:
        write_json(
            args.output / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
