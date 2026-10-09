# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import multiprocessing
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuphoton.core.benchmark import BenchmarkOptions
from cuphoton.core.bulk import WorkItem, read_json_mapping
from cuphoton.core.execution import WorkloadSpec
from cuphoton.xscan import sharing


@pytest.fixture
def cpu_runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    support = importlib.import_module("_sharing_support")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-shared")
    monkeypatch.setattr(sharing, "_process_entry", support.process_entry)
    return support


def workload(tmp_path, cpu_runtime, **options):
    trace = tmp_path / "trace"
    trace.mkdir()
    return WorkloadSpec(
        items=tuple(
            WorkItem(str(index), {"value": index}, 1) for index in range(4)
        ),
        options_payload={"trace": str(trace), **options},
        manifest_payload={"test": True},
        input_identity_payload={},
        manifest_sha256="a" * 64,
        backend="cupy",
        worker_factory=cpu_runtime.factory,
    )


@pytest.mark.parametrize("executor", ["processes", "threads"])
def test_spawn_reuses_workers_and_audits_every_round(
    tmp_path, cpu_runtime, executor
):
    spec = workload(tmp_path, cpu_runtime)
    modules_before = set(sys.modules)
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        run_id="persistent",
        executor=executor,
        workers_per_gpu=2,
        worker_timeout_sec=10,
        benchmark=BenchmarkOptions(warmup_rounds=1, measure_rounds=2),
    )
    assert result.status == "success", result.summary
    assert result.summary["worker_count"] == 2
    traces = [
        path.read_text().splitlines()
        for path in (tmp_path / "trace").iterdir()
    ]
    assert len(traces) == 2
    assert all(
        trace[0] == "initialize" and trace[-1] == "close" for trace in traces
    )
    assert all(len(trace) == 8 for trace in traces)
    for planned in BenchmarkOptions(1, 2).rounds():
        directory = result.run_dir / "rounds" / planned.round_id
        summary = read_json_mapping(directory / "summary.json")
        assert summary["terminal_record_audit"]["ok"]
        assert len(list((directory / "records").glob("*.json"))) == 4
    assert {"cupy", "torch"} & (set(sys.modules) - modules_before) == set()
    assert not multiprocessing.active_children()


def test_threads_own_pools_through_close_and_release(tmp_path, cpu_runtime):
    pools = tmp_path / "pools"
    pools.mkdir()
    spec = workload(tmp_path, cpu_runtime, pool_trace=str(pools))
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor="threads",
        workers_per_gpu=2,
        worker_timeout_sec=10,
    )
    assert result.status == "success", result.summary
    traces = [path.read_text().splitlines() for path in pools.iterdir()]
    assert len(traces) == 2
    for trace in traces:
        assert trace[:3] == ["create", "enter", "initialize"]
        assert len(trace[3:-3]) == 2
        assert all(event.startswith("item:") for event in trace[3:-3])
        assert trace[-3:] == ["close", "release", "exit"]
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("executor", ["processes", "threads"])
@pytest.mark.parametrize(
    "failure", ["initialize", "item", "close", "hang", "exit"]
)
def test_spawn_failures_do_not_report_success_or_leave_children(
    tmp_path, cpu_runtime, executor, failure
):
    spec = workload(tmp_path, cpu_runtime, failure=failure)
    started = time.monotonic()
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor=executor,
        workers_per_gpu=2,
        worker_timeout_sec=0.7 if failure == "hang" else 10,
    )
    assert result.status == "failed"
    assert read_json_mapping(result.summary_path)["status"] == "failed"
    assert time.monotonic() - started < 12
    assert not multiprocessing.active_children()


def test_actual_device_count_and_free_threading_are_verified_after_imports(
    tmp_path, cpu_runtime
):
    spec = workload(tmp_path, cpu_runtime, device_count=2)
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        run_id="two-gpus",
        executor="processes",
        worker_timeout_sec=10,
    )
    assert result.status == "failed"
    assert "exactly one visible GPU" in str(result.summary)
    if sharing._gil_enabled():
        spec = WorkloadSpec(
            **{
                **spec.__dict__,
                "options_payload": {
                    **spec.options_payload,
                    "device_count": 1,
                },
            }
        )
        result = sharing.run_shared_pipeline(
            prepare_workload=lambda rank: spec,
            output_root=tmp_path,
            run_id="gil-enabled",
            executor="threads",
            require_free_threaded=True,
            worker_timeout_sec=10,
        )
        assert result.status == "failed"
        assert "GIL to remain disabled" in str(result.summary)


def test_threads_keep_successful_items_when_a_first_item_fails(
    tmp_path, cpu_runtime
):
    spec = workload(tmp_path, cpu_runtime, fail_item="0")
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor="threads",
        workers_per_gpu=2,
        worker_timeout_sec=10,
    )
    assert result.status == "failed"
    records = [
        read_json_mapping(path)
        for path in (result.run_dir / "records").glob("*.json")
    ]
    assert len(records) == len(spec.items)
    assert {
        record["item_id"]
        for record in records
        if record["status"] == "failed"
    } == {"0"}
    assert result.summary["terminal_record_audit"]["ok"]
    assert "BrokenBarrierError" not in str(result.summary)
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("executor", ["processes", "threads"])
def test_initialization_failure_still_closes_healthy_workers(
    tmp_path, cpu_runtime, executor
):
    spec = workload(
        tmp_path, cpu_runtime, failure="initialize", failure_worker_id=0
    )
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor=executor,
        workers_per_gpu=2,
        worker_timeout_sec=10,
    )
    assert result.status == "failed"
    assert "initialization failed" in str(result.summary)
    traces = sorted(
        path.read_text().splitlines()
        for path in (tmp_path / "trace").iterdir()
    )
    assert traces == [["initialize"], ["initialize", "close"]]
    assert all(
        error["phase"] == "startup" for error in result.summary["errors"]
    )
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("failed_start", [0, 1])
def test_process_start_failure_preserves_error_and_closes_started_workers(
    tmp_path, cpu_runtime, monkeypatch, failed_start
):
    spec = workload(tmp_path, cpu_runtime)
    context = multiprocessing.get_context("spawn")
    created = 0

    def process(**kwargs):
        nonlocal created
        child = context.Process(**kwargs)
        if created == failed_start:

            def fail_start():
                raise OSError("process could not start")

            monkeypatch.setattr(child, "start", fail_start)
        created += 1
        return child

    monkeypatch.setattr(
        sharing.multiprocessing,
        "get_context",
        lambda method: SimpleNamespace(Pipe=context.Pipe, Process=process),
    )
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor="processes",
        workers_per_gpu=2,
        worker_timeout_sec=10,
    )
    assert result.status == "failed"
    assert "process could not start" in str(result.summary)
    assert all(
        error["phase"] == "startup" for error in result.summary["errors"]
    )
    traces = [
        path.read_text().splitlines()
        for path in (tmp_path / "trace").iterdir()
    ]
    assert traces == [["initialize", "close"]] * failed_start
    assert not multiprocessing.active_children()


def test_worker_count_is_capped_by_item_count(tmp_path, cpu_runtime):
    spec = workload(tmp_path, cpu_runtime)
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor="threads",
        workers_per_gpu=8,
        worker_timeout_sec=10,
    )
    assert result.status == "success", result.summary
    assert result.summary["worker_count"] == 4
    assert result.summary["workers_per_gpu"] == 8


@pytest.mark.parametrize(
    "options, message",
    [
        ({"workers_per_gpu": 0}, "positive integer"),
        ({"worker_timeout_sec": float("nan")}, "must be positive"),
        ({"require_free_threaded": True}, "threads executor"),
        (
            {"executor": "threads", "mps_pipe_directory": "/tmp/mps"},
            "processes executor",
        ),
    ],
)
def test_invalid_options_do_not_prepare_workload(tmp_path, options, message):
    def prepare(rank):
        pytest.fail("invalid launch called prepare_workload")

    with pytest.raises(ValueError, match=message):
        sharing.run_shared_pipeline(
            prepare_workload=prepare,
            output_root=tmp_path,
            **{"executor": "processes", **options},
        )


@pytest.mark.parametrize("failure", [None, "mps"])
def test_explicit_mps_requires_verified_client_receipt(
    tmp_path, cpu_runtime, failure
):
    spec = workload(tmp_path, cpu_runtime, failure=failure)
    pipe = str(tmp_path / "mps")
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor="processes",
        mps_pipe_directory=pipe,
        worker_timeout_sec=10,
    )
    if failure:
        assert result.status == "failed"
        assert "MPS client verification failed" in str(result.summary)
    else:
        assert result.status == "success", result.summary
        provenance = read_json_mapping(result.run_dir / "ready.json")[
            "workers"
        ][0]["provenance"]
        assert provenance["mps_pipe_directory"] == pipe
        assert provenance["mps"]["server_pid"] == 41
        assert provenance["mps"]["client_pid"] == provenance["pid"]


def test_timeout_bounds_all_rounds_together(tmp_path, cpu_runtime):
    spec = workload(tmp_path, cpu_runtime, item_delay=0.15)
    result = sharing.run_shared_pipeline(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        executor="processes",
        worker_timeout_sec=3.0,
        benchmark=BenchmarkOptions(measure_rounds=8),
    )
    assert result.status == "failed"
    report = result.summary["benchmark"]
    assert 1 <= len(report["rounds"]) < 8
    assert report["measured_batch_wall_sec"] is None
    assert "timed out" in str(result.summary)
    assert not multiprocessing.active_children()
