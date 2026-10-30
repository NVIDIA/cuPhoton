# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Production-plan GDS replay; intentionally produces no visibility output."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import statistics
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from multiprocessing.connection import wait
from multiprocessing.reduction import DupFd
from pathlib import Path
from typing import Any

from cuphoton.xbeam.runtime.policy import (
    policy_records,
    resolve_execution_policy,
)
from cuphoton.xbeam.sharding import plan_time_shards
from cuphoton.xbeam.windowing import CorrelationWindow


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be a comma-separated integer list"
        ) from error
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("CUDA devices must be non-negative")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("CUDA devices must be unique")
    return devices


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _job_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not value.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(
            "job must be an existing JSON or DiFX .input file"
        )
    return path.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job", type=_job_file, help="Production job JSON or DiFX .input"
    )
    parser.add_argument("--mode", required=True, choices=("read", "strip"))
    parser.add_argument("--profile")
    parser.add_argument("--scan", required=True)
    parser.add_argument("--first-integration", type=int, default=0)
    parser.add_argument("--integration-count", type=_positive, required=True)
    parser.add_argument("--devices", type=_parse_devices, default=(0,))
    parser.add_argument(
        "--ingest-window-integrations", type=_positive, default=13
    )
    parser.add_argument(
        "--workers-per-gpu", type=int, choices=range(1, 17), default=4
    )
    parser.add_argument(
        "--index-dir", type=Path, help="Existing binary VDIX sidecars"
    )
    parser.add_argument("--max-read-bytes", type=_positive, default=1 << 30)
    parser.add_argument("--trials", type=_positive, default=3)
    parser.add_argument("--report", type=Path)
    return parser


def _windows(window: CorrelationWindow, size: int):
    stop = window.first_integration + window.integration_count
    for first in range(window.first_integration, stop, size):
        yield CorrelationWindow(
            window.scan_id, first, min(size, stop - first)
        )


def _replay(run: Any, mode: str) -> dict[str, Any]:
    from cuphoton.xbeam.adapters.radio.engine import _execute_correlation
    from cuphoton.xbeam.cupy_kernels import reserve_vdif_ingest

    if mode not in {"read", "strip"} or run.vdif_io != "cufile":
        raise ValueError("replay requires read/strip mode and cuFile")
    if (run.ingest_manifest or {}).get("mode") != "binary-sidecar":
        raise ValueError("replay requires a valid binary VDIX sidecar")
    replay = {"mode": mode}
    with reserve_vdif_ingest(int(run.xp.cuda.Device().id)) as token:
        for _ in policy_records(
            run.execution_policy,
            lambda: _execute_correlation(
                run.job,
                xp=run.xp,
                source=run.source,
                timing_groups=list(run.timing_groups),
                batch_blocks=run.batch_blocks,
                autocorrelation_mode=run.autocorrelation_mode,
                vdif_io="cufile",
                subintegration_tiling=run.subintegration_tiling,
                max_tile_subintegrations=run.max_tile_subintegrations,
                tiling_manifest={},
                vdif_upload_workers=run.vdif_upload_workers,
                ingest_manifest=run.ingest_manifest,
                reservation_token=token,
                ingest_replay=replay,
                gpu_warmup=run.gpu_warmup,
            ),
        ):
            raise RuntimeError("ingest replay unexpectedly produced a record")
    return replay


@contextmanager
def _read_token_pipe(policy=None):
    limit = (policy or resolve_execution_policy("cufile")).global_gds_reads
    if limit is None:
        yield None
        return
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, bytes(limit))
        yield read_fd, write_fd
    finally:
        os.close(read_fd)
        os.close(write_fd)


def _worker(arguments, device, window, tokens=None):
    descriptors = []
    try:
        for name, token in zip(
            (
                "_CUPHOTON_XBEAM_CUFILE_READ_TOKEN_FD",
                "_CUPHOTON_XBEAM_CUFILE_RETURN_TOKEN_FD",
            ),
            tokens or (None, None),
            strict=True,
        ):
            os.environ.pop(name, None)
            if token is not None:
                descriptor = token.detach()
                descriptors.append(descriptor)
                os.environ[name] = str(descriptor)
        return _worker_body(arguments, device, window)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _worker_entry(connection, function, arguments, device, window, tokens):
    entered = time.perf_counter_ns()
    try:
        result = function(arguments, device, window, tokens)
        ready = time.perf_counter_ns()
        result["worker_lifecycle"] = {
            "entry_ns": entered,
            "result_ready_ns": ready,
        }
        connection.send((True, result, ready))
    except BaseException:
        connection.send(
            (False, traceback.format_exc(), time.perf_counter_ns())
        )
    finally:
        connection.close()


def _run_workers(arguments, shards, tokens, *, function=_worker):
    """Exactly one process per GPU, with fail-fast exit/result monitoring."""

    context = multiprocessing.get_context("spawn")
    processes, receivers, senders = [], [], []
    results, finished = {}, {}
    successful = False
    try:
        for device, shard in zip(arguments.devices, shards, strict=True):
            receiver, sender = context.Pipe(duplex=False)
            receivers.append(receiver)
            senders.append(sender)
            process = context.Process(
                target=_worker_entry,
                args=(
                    sender,
                    function,
                    arguments,
                    device,
                    shard.window,
                    tuple(DupFd(fd) for fd in tokens) if tokens else None,
                ),
            )
            process.start()
            processes.append(process)
            sender.close()
        pending = set(range(len(processes)))
        while pending:
            ready = wait(
                [receivers[i] for i in pending]
                + [processes[i].sentinel for i in pending]
            )
            for index in tuple(pending):
                connection, process = receivers[index], processes[index]
                if connection in ready or (
                    process.sentinel in ready and connection.poll()
                ):
                    try:
                        ok, result, stopped = connection.recv()
                    except EOFError:
                        ok, result, stopped = (
                            False,
                            "worker exited without a result",
                            0,
                        )
                    if not ok:
                        raise RuntimeError(
                            f"replay GPU {arguments.devices[index]} "
                            f"failed:\n{result}"
                        )
                    result["worker_lifecycle"]["result_received_ns"] = (
                        time.perf_counter_ns()
                    )
                    results[index], finished[index] = result, stopped
                    pending.remove(index)
                elif process.sentinel in ready:
                    raise RuntimeError(
                        f"replay GPU {arguments.devices[index]} exited "
                        f"with status {process.exitcode} without a result"
                    )
        for index, process in enumerate(processes):
            process.join()
            results[index]["worker_lifecycle"]["join_observed_ns"] = (
                time.perf_counter_ns()
            )
            if process.exitcode != 0:
                raise RuntimeError(
                    f"replay worker exited with status {process.exitcode}"
                )
        successful = True
        return [results[i] for i in range(len(processes))], max(
            finished.values()
        )
    finally:
        if not successful:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
        for process in processes:
            process.join()
            process.close()
        for connection in [*receivers, *senders]:
            connection.close()


@contextmanager
def _phase(timings, name):
    """Host elapsed time only; never synchronize CUDA for instrumentation."""
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        timings[name] = {
            "start_ns": start,
            "end_ns": end,
            "seconds": (end - start) / 1e9,
        }


def _worker_body(
    arguments: argparse.Namespace, device: int, window: CorrelationWindow
):
    # Spawned processes own their CUDA contexts and sessions; no CUDA state
    # is inherited from the parent. Never alter filesystem/cache contents.
    phases: dict[str, dict[str, int | float]] = {}
    with _phase(phases, "imports"):
        os.environ["CUFILE_ALLOW_COMPAT_MODE"] = "false"
        import cupy

        from cuphoton.xbeam.adapters.radio.difx_job import load_interop_job
        from cuphoton.xbeam.adapters.radio.engine import prepare_correlation
        from cuphoton.xbeam.adapters.radio.models import load_job
        from cuphoton.xbeam.adapters.vdif.ingest import VdifPlanningCache
        from cuphoton.xbeam.cupy_kernels import clear_cached_cufile_workspaces

    with _phase(phases, "job_load"):
        is_text = arguments.job.suffix.lower() == ".input"
        job = (
            load_interop_job(arguments.job)
            if is_text
            else load_job(arguments.job)
        )
    profile = arguments.profile or (
        "interop-real-usb-v1" if is_text else "analytic-v1"
    )
    rows = []
    planning_cache = VdifPlanningCache()
    device_enter_started = time.perf_counter_ns()
    with cupy.cuda.Device(device):
        device_enter_finished = time.perf_counter_ns()
        phases["device_enter"] = {
            "start_ns": device_enter_started,
            "end_ns": device_enter_finished,
            "seconds": (device_enter_finished - device_enter_started) / 1e9,
        }
        try:
            for selected in _windows(
                window, arguments.ingest_window_integrations
            ):
                window_phases: dict[str, dict[str, int | float]] = {}
                with _phase(window_phases, "prepare"):
                    # Production planning without compute warmup.
                    run = prepare_correlation(
                        job,
                        backend="cupy",
                        profile=profile,
                        window=selected,
                        max_read_bytes=arguments.max_read_bytes,
                        assume_ordered=False,
                        index_dir=str(arguments.index_dir)
                        if arguments.index_dir
                        else None,
                        vdif_io="cufile",
                        start_gpu_warmup=False,
                        vdif_upload_workers=arguments.workers_per_gpu,
                        planning_cache=planning_cache,
                    )
                with _phase(window_phases, "replay"):
                    replay = _replay(run, arguments.mode)
                with _phase(window_phases, "signature_and_statistics"):
                    descriptors = sorted(replay.pop("read_descriptors"))
                    digest = hashlib.sha256(
                        json.dumps(descriptors).encode()
                    ).hexdigest()
                    rows.append(
                        {
                            **asdict(replay["statistics"]),
                            "finished_ns": replay["finished_ns"],
                            "read_signature": digest,
                            "window": asdict(selected),
                            "upload": replay["upload"],
                            "phase_timings": window_phases,
                        }
                    )
                with _phase(window_phases, "release_window"):
                    del run, replay
        finally:
            with _phase(phases, "workspace_cleanup"):
                clear_cached_cufile_workspaces(device)
    from cuphoton.xbeam.profiling import host_timing_report

    return {
        "device": device,
        "windows": rows,
        "phase_timings": phases,
        "host_timing": host_timing_report(),
    }


def _summarize(workers: list[dict], started: int, stopped: int) -> dict:
    rows = [row for worker in workers for row in worker["windows"]]
    if not rows or any(row["first_read_ns"] is None for row in rows):
        raise ValueError("replay performed no reads in a selected window")
    first = min(row["first_read_ns"] for row in rows)
    last = max(row["last_read_ns"] for row in rows)
    finished = max(row["finished_ns"] for row in rows)
    span = (last - first) / 1e9
    logical = sum(row["logical_bytes"] for row in rows)
    active = sum(row["active_read_seconds"] for row in rows)
    partition = {}
    if all("worker_lifecycle" in w for w in workers):
        ready = max(w["worker_lifecycle"]["result_ready_ns"] for w in workers)
        received = max(
            w["worker_lifecycle"]["result_received_ns"] for w in workers
        )
        for worker in workers:
            lifecycle = worker["worker_lifecycle"]
            lifecycle["parent_start_to_entry_seconds"] = (
                lifecycle["entry_ns"] - started
            ) / 1e9
            lifecycle["result_delivery_seconds"] = (
                lifecycle["result_received_ns"] - lifecycle["result_ready_ns"]
            ) / 1e9
        partition = {
            "wall_time_partition": {
                "setup_to_first_read_seconds": (first - started) / 1e9,
                "first_to_last_read_seconds": span,
                "last_read_to_ingest_finish_seconds": (finished - last) / 1e9,
                "ingest_finish_to_results_ready_seconds": (ready - finished)
                / 1e9,
                "results_ready_to_received_seconds": (received - ready) / 1e9,
                "results_received_to_parent_stop_seconds": (
                    stopped - received
                )
                / 1e9,
            },
            "phase_timing_note": (
                "Partition uses common host milestones and sums to "
                "complete_seconds; per-worker/window and host ranges overlap "
                "and must not be summed across workers. Device entry does "
                "not force CUDA initialization. Join timestamps are parent "
                "observations, not exact process exit times."
            ),
        }
    return {
        **partition,
        "logical_bytes": logical,
        "requested_bytes": sum(row["requested_bytes"] for row in rows),
        "calls": sum(row["calls"] for row in rows),
        "read_span_seconds": span,
        "active_read_seconds": active,
        "mean_read_concurrency": active / span if span else 0,
        "read_gb_per_second": logical / span / 1e9 if span else 0,
        "setup_to_first_read_seconds": (first - started) / 1e9,
        "final_drain_seconds": (finished - last) / 1e9,
        "steady_state_seconds": (finished - first) / 1e9,
        "complete_seconds": (stopped - started) / 1e9,
        "complete_gb_per_second": logical / (stopped - started),
        "workers": workers,
    }


def run(arguments: argparse.Namespace) -> dict:
    policy = resolve_execution_policy(
        "cufile", upload_workers=arguments.workers_per_gpu
    )
    arguments.job = _job_file(str(arguments.job))
    window = CorrelationWindow(
        arguments.scan,
        arguments.first_integration,
        arguments.integration_count,
    )
    shards = plan_time_shards(window, len(arguments.devices))
    trials = []
    expected = None
    for trial in range(arguments.trials):
        started = time.perf_counter_ns()
        with _read_token_pipe(policy) as tokens:
            workers, workers_finished = _run_workers(
                arguments, shards, tokens
            )
        stopped = time.perf_counter_ns()
        report = _summarize(workers, started, stopped)
        report["worker_body_seconds"] = (workers_finished - started) / 1e9
        report["parent_cleanup_seconds"] = (stopped - workers_finished) / 1e9
        signatures = [
            row["read_signature"]
            for worker in workers
            for row in worker["windows"]
        ]
        if expected is not None and signatures != expected:
            raise RuntimeError(
                "production read ranges changed between trials"
            )
        expected = signatures
        trials.append(report)
        print(
            f"trial {trial + 1}/{arguments.trials}: "
            f"complete={report['complete_seconds']:.6f}s "
            f"read={report['read_gb_per_second']:.3f} GB/s",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "engine": "cuphoton-gds-ingest-replay",
        "mode": arguments.mode,
        "cache": "unchanged",
        "job": str(arguments.job),
        "devices": list(arguments.devices),
        "workers_per_gpu": arguments.workers_per_gpu,
        "global_gds_reads": policy.global_gds_reads,
        "execution_policy": policy.to_manifest(),
        "trials": trials,
        "median_complete_seconds": statistics.median(
            row["complete_seconds"] for row in trials
        ),
        "median_read_gb_per_second": statistics.median(
            row["read_gb_per_second"] for row in trials
        ),
    }
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(json.dumps(run(arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
