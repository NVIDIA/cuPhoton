# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent pipeline workers sharing one explicitly bound GPU."""

from __future__ import annotations

import importlib
import math
import multiprocessing
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

from cuphoton.core.benchmark import (
    BenchmarkOptions,
    BenchmarkRound,
    build_benchmark_report,
)
from cuphoton.core.bulk import (
    WorkItem,
    atomic_write_json,
    error_payload,
    json_mapping,
    new_run_id,
    partition_byte_balanced,
    read_json_mapping,
    timestamp_utc,
    validate_identifier,
)
from cuphoton.core.execution import (
    ExecutionResult,
    Worker,
    WorkloadSpec,
    audit_worker_provenance,
    execute_worker_round,
    finalize_round,
    prepare_run,
    resolve_worker_factory,
)


def _gil_enabled() -> bool:
    return bool(getattr(sys, "_is_gil_enabled", lambda: True)())


def _check_gil(required: bool) -> bool:
    enabled = _gil_enabled()
    if required and enabled:
        raise RuntimeError(
            "free-threaded execution requires the GIL to remain disabled "
            "after importing the pipeline dependencies"
        )
    return enabled


def _load_cuda() -> Any:
    # Torch establishes CUDA library loading before CuPy touches the driver.
    torch = importlib.import_module("torch")
    cp = importlib.import_module("cupy")
    if (
        int(cp.cuda.runtime.getDeviceCount()) != 1
        or int(torch.cuda.device_count()) != 1
    ):
        raise RuntimeError("GPU sharing requires exactly one visible GPU")
    torch.cuda.set_device(0)
    cp.cuda.Device(0).use()
    return cp


class _FirstUseWorker:
    """Serialize each thread's first item without creating hidden work."""

    def __init__(self, worker: Worker, lock: Any, barrier: Any) -> None:
        self.worker = worker
        self.lock = lock
        self.barrier = barrier
        self.first = True

    def run_item(self, item: WorkItem, output_dir: Path) -> Mapping[str, Any]:
        if self.first:
            self.first = False
            try:
                with self.lock:
                    result = self.worker.run_item(item, output_dir)
            except BaseException:
                self.barrier.abort()
                raise
            self.barrier.wait()
            return result
        return self.worker.run_item(item, output_dir)

    @property
    def gpu_identity(self) -> Mapping[str, Any]:
        return self.worker.gpu_identity

    def close(self) -> None:
        self.worker.close()


def _worker_loop(
    connection: Connection,
    payload: Mapping[str, Any],
    worker_id: int,
    lock: Any = None,
    barrier: Any = None,
) -> None:
    worker: Worker | None = None
    pool = None
    close_error = None
    stage = "ready"
    try:
        with lock if lock is not None else nullcontext():
            cp = _load_cuda()
            pool = cp.cuda.MemoryPool() if lock is not None else None
            allocator = (
                cp.cuda.using_allocator(pool.malloc)
                if pool is not None
                else nullcontext()
            )
            allocator.__enter__()
            factory = resolve_worker_factory(payload["worker_factory"])
            worker = factory(payload["options"])
            if not callable(
                getattr(worker, "run_item", None)
            ) or not callable(getattr(worker, "close", None)):
                raise TypeError("worker requires run_item and close methods")
            gil_enabled = _check_gil(payload["require_free_threaded"])
            mps = None
            if payload["mps_pipe_directory"] is not None:
                from cuphoton.core.mps import require_mps_client

                mps = require_mps_client(os.getpid())
            provenance = {
                "worker_id": worker_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "thread_id": threading.get_native_id(),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "gpu": json_mapping(
                    worker.gpu_identity, field="worker GPU identity"
                ),
                "gil_enabled": gil_enabled,
                "mps_pipe_directory": os.environ.get(
                    "CUDA_MPS_PIPE_DIRECTORY"
                ),
                "mps": mps,
            }
        if lock is not None:
            worker = _FirstUseWorker(worker, lock, barrier)
        connection.send({"phase": "ready", "provenance": provenance})
        items = tuple(
            WorkItem.from_dict(item) for item in payload["shards"][worker_id]
        )
        while True:
            command = connection.recv()
            stage = command["phase"]
            if stage == "close":
                break
            if stage != "round":
                raise ValueError("unknown shared-worker command")
            _check_gil(payload["require_free_threaded"])
            started = time.perf_counter()
            execute_worker_round(
                worker,
                items=items,
                run_id=command["run_id"],
                run_dir=Path(command["run_dir"]),
                manifest_sha256=payload["manifest_sha256"],
                worker_id=worker_id,
                backend=payload["backend"],
                provenance=provenance,
            )
            _check_gil(payload["require_free_threaded"])
            connection.send(
                {
                    "phase": "round",
                    "worker_wall_sec": time.perf_counter() - started,
                }
            )
    except BaseException as exc:
        if barrier is not None:
            barrier.abort()
        try:
            connection.send({"phase": stage, "error": error_payload(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            if worker is not None:
                worker.close()
                worker = None
            if pool is not None:
                pool.free_all_blocks()
        except BaseException as exc:
            close_error = error_payload(exc)
        finally:
            if "allocator" in locals():
                try:
                    allocator.__exit__(None, None, None)
                except BaseException as exc:
                    close_error = error_payload(exc)
        try:
            connection.send({"phase": "closed", "error": close_error})
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


def _process_entry(
    connections: Sequence[Connection],
    payload: Mapping[str, Any],
    worker_ids: Sequence[int],
    threaded: bool,
) -> None:
    if payload["mps_pipe_directory"] is not None:
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = payload["mps_pipe_directory"]
    if threaded:
        lock = threading.Lock()
        barrier = threading.Barrier(len(worker_ids))
        threads = [
            threading.Thread(
                target=_worker_loop,
                args=(connection, payload, worker_id, lock, barrier),
                name=f"cuphoton-pipeline-{worker_id}",
            )
            for connection, worker_id in zip(
                connections, worker_ids, strict=True
            )
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker_loop(connections[0], payload, worker_ids[0])


def _collect(
    connections: Sequence[Connection],
    owners: Sequence[Any],
    phase: str,
    timeout: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    pending = dict(enumerate(connections))
    received: dict[int, dict[str, Any]] = {}
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"shared workers timed out during {phase}")
        ready = wait(list(pending.values()), timeout=min(remaining, 0.1))
        for worker_id, connection in list(pending.items()):
            if connection not in ready:
                if owners[worker_id].exitcode is not None:
                    # Drain a last message sent immediately before exit.
                    if not connection.poll():
                        raise RuntimeError(
                            f"worker {worker_id} exited during {phase}"
                        )
                else:
                    continue
            message = connection.recv()
            if message.get("phase") != phase:
                raise RuntimeError(
                    f"worker {worker_id} sent an unexpected {phase} receipt"
                )
            received[worker_id] = message
            del pending[worker_id]
    return [received[index] for index in range(len(connections))]


def _stop_processes(
    processes: Sequence[Any], timeout: float
) -> list[dict[str, Any]]:
    errors = []
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            errors.append(
                {
                    "phase": "shutdown",
                    "message": "worker required termination",
                }
            )
            process.terminate()
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if process.exitcode != 0:
            errors.append(
                {
                    "phase": "shutdown",
                    "pid": process.pid,
                    "exitcode": process.exitcode,
                }
            )
    return errors


def run_shared_pipeline(
    *,
    prepare_workload: Callable[[int], WorkloadSpec],
    output_root: Path,
    run_id: str | None = None,
    executor: str,
    workers_per_gpu: int = 1,
    worker_timeout_sec: float = 3600,
    benchmark: BenchmarkOptions | None = None,
    require_free_threaded: bool = False,
    mps_pipe_directory: str | None = None,
) -> ExecutionResult:
    """Run whole pipeline items on one GPU with persistent local workers.

    The parent never initializes CUDA. Process workers may attach to an
    externally managed MPS server. Thread workers each own a model, streams,
    and CuPy allocator inside one supervised child process. Initialization
    and each thread's first item are serialized; that first item remains in
    the first round's outputs and elapsed time. One deadline bounds startup,
    all rounds and close. A stuck child is terminated before return.
    """

    if executor not in {"processes", "threads"}:
        raise ValueError("sharing executor must be processes or threads")
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    if (
        isinstance(worker_timeout_sec, bool)
        or not isinstance(worker_timeout_sec, (int, float))
        or not math.isfinite(worker_timeout_sec)
        or worker_timeout_sec <= 0
    ):
        raise ValueError("worker_timeout_sec must be positive")
    if type(require_free_threaded) is not bool:
        raise TypeError("require_free_threaded must be a boolean")
    if require_free_threaded and executor != "threads":
        raise ValueError(
            "require_free_threaded requires the threads executor"
        )
    if mps_pipe_directory is not None:
        if executor != "processes":
            raise ValueError(
                "mps_pipe_directory requires the processes executor"
            )
        if not isinstance(mps_pipe_directory, str) or not mps_pipe_directory:
            raise ValueError("mps_pipe_directory must be a nonempty path")
        mps_pipe_directory = str(
            Path(mps_pipe_directory).expanduser().resolve()
        )
    if benchmark is not None and not isinstance(benchmark, BenchmarkOptions):
        raise TypeError("benchmark must be BenchmarkOptions or None")
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = visibility.split(",")
    if (
        len(tokens) != 1
        or not tokens[0].strip()
        or tokens[0].strip().startswith("-")
    ):
        raise ValueError("set CUDA_VISIBLE_DEVICES to exactly one GPU")
    effective_run_id = run_id or new_run_id("shared-pipeline")
    validate_identifier(effective_run_id, field="run_id")
    started = time.perf_counter()
    started_at = timestamp_utc()
    spec = prepare_workload(0)
    if not isinstance(spec, WorkloadSpec):
        raise TypeError("prepare_workload must return WorkloadSpec")
    worker_count = min(workers_per_gpu, len(spec.items))
    shards = partition_byte_balanced(spec.items, worker_count)
    run_dir = Path(output_root).expanduser().resolve() / effective_run_id
    prepare_run(run_dir, effective_run_id, spec, executor)
    options = {
        "workers_per_gpu": workers_per_gpu,
        "worker_count": worker_count,
        "worker_timeout_sec": worker_timeout_sec,
        "benchmark": benchmark.to_payload() if benchmark else None,
        "require_free_threaded": require_free_threaded,
        "mps_pipe_directory": mps_pipe_directory,
        "serialized_first_item": executor == "threads",
    }
    atomic_write_json(run_dir / "execution-options.json", options)
    payload = {
        **spec.identity_payload(),
        "shards": [[item.to_dict() for item in shard] for shard in shards],
        "require_free_threaded": require_free_threaded,
        "mps_pipe_directory": mps_pipe_directory,
    }
    context = multiprocessing.get_context("spawn")
    pairs = [context.Pipe() for _ in shards]
    parents = [pair[0] for pair in pairs]
    children = [pair[1] for pair in pairs]
    processes: list[Any] = []
    owners: list[Any] = []
    errors: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    last_summary = None
    startup_ready_sec = None
    phase = "startup"
    deadline = time.monotonic() + worker_timeout_sec
    try:
        groups = (
            [list(range(worker_count))]
            if executor == "threads"
            else [[index] for index in range(worker_count)]
        )
        for group in groups:
            process = context.Process(
                target=_process_entry,
                args=(
                    [children[index] for index in group],
                    payload,
                    group,
                    executor == "threads",
                ),
                name=f"cuphoton-{executor}-{group[0]}",
            )
            process.start()
            processes.append(process)
            owners.extend([process] * len(group))
        for connection in children:
            connection.close()
        ready = _collect(
            parents, owners, "ready", deadline - time.monotonic()
        )
        for index, receipt in enumerate(ready):
            if receipt.get("error") is not None:
                raise RuntimeError(
                    f"worker {index} initialization failed: "
                    f"{receipt['error']}"
                )
            provenance = receipt["provenance"]
            if (
                provenance.get("worker_id") != index
                or provenance.get("pid") != owners[index].pid
                or type(provenance.get("thread_id")) is not int
                or provenance["thread_id"] <= 0
            ):
                raise RuntimeError(
                    "worker readiness differs from process ownership"
                )
        if (
            executor == "threads"
            and len({receipt["provenance"]["thread_id"] for receipt in ready})
            != worker_count
        ):
            raise RuntimeError(
                "thread workers require distinct native threads"
            )
        problems = audit_worker_provenance(
            [receipt["provenance"] for receipt in ready],
            backend=spec.backend,
            expected_worker_count=worker_count,
            gpu_groups=[0] * worker_count,
        )
        if problems:
            raise RuntimeError(str(problems))
        atomic_write_json(run_dir / "ready.json", {"workers": ready})
        startup_ready_sec = time.perf_counter() - started
        for planned in (
            benchmark.rounds()
            if benchmark
            else (BenchmarkRound("measure", 0),)
        ):
            phase = planned.round_id
            round_dir = (
                run_dir / "rounds" / planned.round_id
                if benchmark
                else run_dir
            )
            round_id = (
                planned.run_id(effective_run_id)
                if benchmark
                else effective_run_id
            )
            if benchmark:
                prepare_run(round_dir, round_id, spec, executor)
            round_started = time.perf_counter()
            for connection in parents:
                connection.send(
                    {
                        "phase": "round",
                        "run_id": round_id,
                        "run_dir": str(round_dir),
                    }
                )
            results = _collect(
                parents, owners, "round", deadline - time.monotonic()
            )
            collected = time.perf_counter()
            failures = [
                {"phase": phase, "worker_id": index, **result["error"]}
                for index, result in enumerate(results)
                if result.get("error") is not None
            ]
            worker_results = [
                read_json_mapping(
                    round_dir / "workers" / f"worker-{index:04d}.json"
                )
                for index, result in enumerate(results)
                if result.get("error") is None
            ]
            for worker_result in worker_results:
                worker_id = worker_result.get("worker_id")
                if (
                    type(worker_id) is not int
                    or not 0 <= worker_id < worker_count
                    or worker_result.get("provenance")
                    != ready[worker_id]["provenance"]
                ):
                    raise RuntimeError(
                        "worker provenance changed after readiness"
                    )
            last_summary = finalize_round(
                round_dir,
                round_id,
                spec,
                shards,
                worker_results,
                artifact_timeout_sec=0.01,
                gpu_groups=[0] * worker_count,
            )
            last_summary["errors"].extend(failures)
            if failures:
                last_summary["status"] = "failed"
            timing = {
                "batch_wall_sec": collected - round_started,
                "finalization_sec": last_summary["finalization_sec"],
                "worker_wall_max_sec": max(
                    (
                        result.get("worker_wall_sec", 0.0)
                        for result in results
                    ),
                    default=0.0,
                ),
            }
            last_summary.update(timing)
            if benchmark:
                atomic_write_json(round_dir / "summary.json", last_summary)
            rounds.append(
                {
                    **planned.to_payload(),
                    "status": last_summary["status"],
                    "summary_path": str(
                        (round_dir / "summary.json").relative_to(run_dir)
                    ),
                    **timing,
                }
            )
            if last_summary["status"] != "success":
                break
    except BaseException as exc:
        errors.append({"phase": phase, **error_payload(exc)})
        if not isinstance(exc, Exception):
            raise
    finally:
        close_started = time.perf_counter()
        try:
            for connection in parents:
                connection.send({"phase": "close"})
            closed = _collect(
                parents,
                owners,
                "closed",
                min(deadline - time.monotonic(), 1.0)
                if errors
                else deadline - time.monotonic(),
            )
            errors.extend(
                {"phase": "close", "worker_id": index, **receipt["error"]}
                for index, receipt in enumerate(closed)
                if receipt.get("error") is not None
            )
        except Exception as exc:
            errors.append({"phase": "close", **error_payload(exc)})
        errors.extend(
            _stop_processes(
                processes,
                0.0
                if errors
                else max(0.0, min(deadline - time.monotonic(), 5.0)),
            )
        )
        for connection in (*parents, *children):
            connection.close()
        close_sec = time.perf_counter() - close_started
    if benchmark:
        report = build_benchmark_report(benchmark, rounds, errors=errors)
        summary = {
            "schema": "cuphoton.core.execution-benchmark-summary/v1",
            "status": report["status"],
            "benchmark": report,
            "errors": errors,
        }
    else:
        summary = dict(
            last_summary
            or {
                "schema": "cuphoton.core.execution-summary/v1",
                "status": "failed",
                "errors": [],
            }
        )
        summary["errors"].extend(errors)
        if summary["errors"]:
            summary["status"] = "failed"
    summary.update(
        executor=executor,
        run_id=effective_run_id,
        manifest_sha256=spec.manifest_sha256,
        worker_count=worker_count,
        workers_per_gpu=workers_per_gpu,
        physical_gpu_count=1,
        started_at_utc=started_at,
        completed_at_utc=timestamp_utc(),
        startup_ready_sec=startup_ready_sec,
        close_sec=close_sec,
        coordinator_wall_sec=time.perf_counter() - started,
    )
    atomic_write_json(run_dir / "summary.json", summary)
    return ExecutionResult(
        executor,
        effective_run_id,
        run_dir,
        run_dir / "summary.json",
        summary["status"],
        summary,
    )
