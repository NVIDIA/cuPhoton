# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import time
from types import SimpleNamespace

import pytest

from cuphoton.xbeam.sharding import plan_time_shards
from cuphoton.xbeam.upload_resources import _global_cufile_read_token
from cuphoton.xbeam.windowing import CorrelationWindow
from examples.xbeam.benchmark_gds_ingest import (
    _parser,
    _read_token_pipe,
    _run_workers,
    _summarize,
    _windows,
)


def test_replay_uses_production_uneven_shards_and_window_boundaries():
    shards = plan_time_shards(CorrelationWindow("scan", 3, 50), 4)
    windows = [
        window for shard in shards for window in _windows(shard.window, 7)
    ]
    indices = [
        i
        for w in windows
        for i in range(
            w.first_integration, w.first_integration + w.integration_count
        )
    ]
    assert indices == list(range(3, 53))
    assert [s.window.integration_count for s in shards] == [13, 13, 12, 12]


def test_summary_uses_common_span_not_sum_of_device_bandwidths():
    def row(first, last):
        return dict(
            first_read_ns=first,
            last_read_ns=last,
            finished_ns=last + 100_000_000,
            logical_bytes=1_000_000_000,
            requested_bytes=1_000_000_064,
            active_read_seconds=1.0,
            calls=16,
        )

    workers = [
        {"windows": [row(1_000_000_000, 2_000_000_000)]},
        {"windows": [row(2_000_000_000, 3_000_000_000)]},
    ]
    summary = _summarize(workers, 0, 4_000_000_000)
    assert summary["read_gb_per_second"] == 1
    assert summary["mean_read_concurrency"] == 1
    assert summary["complete_gb_per_second"] == 0.5
    assert summary["setup_to_first_read_seconds"] == 1
    assert summary["final_drain_seconds"] == 0.1
    assert summary["steady_state_seconds"] == 2.1


def test_replay_cli_requires_mode_and_positive_windows(tmp_path):
    job = tmp_path / "job.json"
    job.write_text("{}")
    common = [str(job), "--scan", "scan", "--integration-count", "50"]
    with pytest.raises(SystemExit):
        _parser().parse_args(common)
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [*common, "--mode", "read", "--ingest-window-integrations", "0"]
        )
    assert (
        _parser().parse_args([*common, "--mode", "strip"]).workers_per_gpu
        == 4
    )


def test_replay_global_read_tokens_use_production_limiter(monkeypatch):
    monkeypatch.setenv("CUPHOTON_XBEAM_GLOBAL_GDS_READS", "1")
    with _read_token_pipe() as (read_fd, write_fd):
        monkeypatch.setenv(
            "_CUPHOTON_XBEAM_CUFILE_READ_TOKEN_FD", str(read_fd)
        )
        monkeypatch.setenv(
            "_CUPHOTON_XBEAM_CUFILE_RETURN_TOKEN_FD", str(write_fd)
        )
        with pytest.raises(ValueError, match="read failed"):
            with _global_cufile_read_token():
                raise ValueError("read failed")
        os.set_blocking(read_fd, False)
        assert os.read(read_fd, 1) == bytes(1)
    with pytest.raises(OSError):
        os.fstat(read_fd)


@pytest.mark.parametrize("value", ["0", "65", "invalid"])
def test_replay_rejects_invalid_global_limit(monkeypatch, value):
    monkeypatch.setenv("CUPHOTON_XBEAM_GLOBAL_GDS_READS", value)
    with pytest.raises(ValueError):
        with _read_token_pipe():
            pytest.fail("invalid limit accepted")


def _replay_test_worker(arguments, device, window, tokens):
    if arguments.mode == "abrupt" and device == 0:
        os._exit(19)
    if arguments.mode == "exception" and device == 0:
        raise ValueError("injected replay failure")
    if arguments.mode != "success":
        time.sleep(30)
    return {"device": device, "pid": os.getpid()}


def test_replay_starts_exactly_one_process_per_device(monkeypatch):
    context = multiprocessing.get_context("spawn")
    original = context.Process
    spawned = []

    def process(*args, **kwargs):
        result = original(*args, **kwargs)
        spawned.append(result)
        return result

    monkeypatch.setattr(context, "Process", process)
    arguments = SimpleNamespace(devices=(0, 1, 2), mode="success")
    shards = plan_time_shards(CorrelationWindow("scan", 0, 7), 3)
    started = time.perf_counter_ns()
    results, finished = _run_workers(
        arguments, shards, None, function=_replay_test_worker
    )
    stopped = time.perf_counter_ns()
    assert len(spawned) == 3
    assert [row["device"] for row in results] == [0, 1, 2]
    assert len({row["pid"] for row in results}) == 3
    lifecycles = [row["worker_lifecycle"] for row in results]
    for lifecycle in lifecycles:
        assert (
            started
            <= lifecycle["entry_ns"]
            <= lifecycle["result_ready_ns"]
            <= lifecycle["result_received_ns"]
            <= lifecycle["join_observed_ns"]
            <= stopped
        )
    assert finished == max(row["result_ready_ns"] for row in lifecycles)


@pytest.mark.parametrize("mode", ["abrupt", "exception"])
def test_replay_failure_terminates_siblings(mode):
    arguments = SimpleNamespace(devices=(0, 1), mode=mode)
    shards = plan_time_shards(CorrelationWindow("scan", 0, 4), 2)
    previous = {process.pid for process in multiprocessing.active_children()}
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="replay GPU 0") as failure:
        _run_workers(arguments, shards, None, function=_replay_test_worker)
    if mode == "exception":
        assert "injected replay failure" in str(failure.value)
    assert time.perf_counter() - started < 15
    assert {p.pid for p in multiprocessing.active_children()} == previous


@pytest.mark.parametrize("job", ["", ".", "/does-not-exist/job.input"])
def test_replay_rejects_bad_job_before_spawning(job):
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                job,
                "--scan",
                "scan",
                "--integration-count",
                "2",
                "--mode",
                "read",
            ]
        )


def test_phase_partition_accounts_for_full_common_wall_time():
    def worker(first, last, finish, ready, received, entered):
        return {
            "worker_lifecycle": {
                "entry_ns": entered * 10**9,
                "result_ready_ns": ready * 10**9,
                "result_received_ns": received * 10**9,
            },
            "windows": [
                {
                    "first_read_ns": first * 10**9,
                    "last_read_ns": last * 10**9,
                    "finished_ns": finish * 10**9,
                    "logical_bytes": 10**9,
                    "requested_bytes": 10**9,
                    "active_read_seconds": 3,
                    "calls": 2,
                }
            ],
        }

    workers = [worker(2, 5, 6, 8, 9, 1), worker(3, 7, 8, 9, 10, 2)]
    result = _summarize(workers, 0, 12 * 10**9)
    partition = result["wall_time_partition"]
    assert partition == {
        "setup_to_first_read_seconds": 2,
        "first_to_last_read_seconds": 5,
        "last_read_to_ingest_finish_seconds": 1,
        "ingest_finish_to_results_ready_seconds": 1,
        "results_ready_to_received_seconds": 1,
        "results_received_to_parent_stop_seconds": 2,
    }
    assert sum(partition.values()) == result["complete_seconds"]
    assert (
        workers[1]["worker_lifecycle"]["parent_start_to_entry_seconds"] == 2
    )
    assert workers[0]["worker_lifecycle"]["result_delivery_seconds"] == 1


def test_phase_records_elapsed_on_exception(monkeypatch):
    from examples.xbeam.benchmark_gds_ingest import _phase

    stamps = iter([10**9, 3 * 10**9])
    monkeypatch.setattr(time, "perf_counter_ns", lambda: next(stamps))
    timings = {}
    with pytest.raises(ValueError):
        with _phase(timings, "cleanup"):
            raise ValueError("failure")
    assert timings["cleanup"]["seconds"] == 2
