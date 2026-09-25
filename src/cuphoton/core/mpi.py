# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collective MPI execution of component-neutral, worker-local workloads."""

from __future__ import annotations

import json
import math
import os
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ._mpi_runtime import (
    _load_mpi_api,
    _mpi_context,
    _mpi_failure_consensus,
    _normalize_mpi_library_version,
    _prebound_visibility,
)
from .benchmark import (
    BenchmarkOptions,
    BenchmarkRound,
    build_benchmark_report,
)
from .bulk import (
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
from .execution import (
    ExecutionResult,
    Worker,
    WorkloadSpec,
    audit_worker_provenance,
    execute_worker_round,
    finalize_round,
    prepare_run,
    resolve_worker_factory,
)


def run_mpi_work_items(
    *,
    prepare_workload: Callable[[int], WorkloadSpec],
    output_root: Path,
    run_id: str | None = None,
    rank_setup_timeout_sec: float = 600.0,
    benchmark: BenchmarkOptions | None = None,
    prepare_on_root: bool = False,
) -> ExecutionResult | None:
    """Run under an external launcher with one prebound GPU per MPI rank.

    Startup, preflight, worker initialization and close failures are agreed
    collectively. Ordinary mode runs once; benchmark mode reuses the
    same worker object for every warmup and measured pass. The launcher owns
    process failure detection and MPI shutdown. With ``prepare_on_root``,
    only rank zero plans the workload; JSON launch descriptors are broadcast
    to peers, while scientific validators and finalizers remain on rank zero.
    """

    start = time.perf_counter()
    started_at = timestamp_utc()
    startup_error = None
    visibility = None
    resolved_root = None
    try:
        if benchmark is not None and not isinstance(
            benchmark, BenchmarkOptions
        ):
            raise TypeError("benchmark must be BenchmarkOptions or None")
        if (
            isinstance(rank_setup_timeout_sec, bool)
            or not isinstance(rank_setup_timeout_sec, (int, float))
            or not math.isfinite(rank_setup_timeout_sec)
            or rank_setup_timeout_sec <= 0
        ):
            raise ValueError("rank_setup_timeout_sec must be positive")
        if run_id is not None:
            validate_identifier(run_id, field="run_id")
        if not callable(prepare_workload):
            raise TypeError("prepare_workload must be callable")
        visibility = _prebound_visibility(os.environ)
        premature = sorted(
            name
            for name in ("cupy", "numba.cuda", "cuda.tile")
            if name in sys.modules
        )
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            premature.append("torch CUDA")
        if premature:
            raise RuntimeError(
                "CUDA was initialized before MPI binding: "
                + ", ".join(premature)
            )
        resolved_root = Path(output_root).expanduser().resolve(strict=False)
    except Exception as exc:
        startup_error = error_payload(exc)
    api = _load_mpi_api()
    comm = api.comm
    _mpi_failure_consensus(comm, "MPI rank-startup validation", startup_error)
    context = _mpi_context(api.MPI, comm, os.environ)
    assert visibility is not None and resolved_root is not None
    spec = None
    preflight_error = None
    try:
        if not prepare_on_root or context.rank == 0:
            spec = prepare_workload(context.rank)
            if not isinstance(spec, WorkloadSpec):
                raise TypeError("prepare_workload must return WorkloadSpec")
            if context.world_size > len(spec.items):
                raise ValueError(
                    "MPI world size cannot exceed the item count"
                )
    except Exception as exc:
        preflight_error = error_payload(exc)
    _mpi_failure_consensus(comm, "MPI workload preflight", preflight_error)
    _agree_payload(comm, {"prepare_on_root": prepare_on_root})
    if prepare_on_root:
        payload = comm.bcast(
            spec.identity_payload() if context.rank == 0 else None, root=0
        )
        descriptor_error = None
        try:
            if context.rank != 0:
                spec = WorkloadSpec(
                    items=tuple(
                        WorkItem.from_dict(item) for item in payload["items"]
                    ),
                    options_payload=payload["options"],
                    manifest_payload=payload["manifest"],
                    input_identity_payload=payload["input_identity"],
                    manifest_sha256=payload["manifest_sha256"],
                    backend=payload["backend"],
                    worker_factory=resolve_worker_factory(
                        payload["worker_factory"]
                    ),
                )
        except Exception as exc:
            descriptor_error = error_payload(exc)
        _mpi_failure_consensus(
            comm, "MPI workload descriptor", descriptor_error
        )
    assert spec is not None
    _agree_payload(
        comm,
        {
            "workload": spec.identity_payload(),
            "run_id": run_id,
            "output_root": str(resolved_root),
            "rank_setup_timeout_sec": rank_setup_timeout_sec,
            "benchmark": benchmark.to_payload() if benchmark else None,
            "mpi4py_version": api.mpi4py_version,
            "mpi_library_version": _normalize_mpi_library_version(
                api.library_version
            ),
        },
    )
    effective_run_id = comm.bcast(
        (run_id or new_run_id("mpi-workload")) if context.rank == 0 else None,
        root=0,
    )
    validate_identifier(effective_run_id, field="run_id")
    run_dir = resolved_root / effective_run_id
    shards = partition_byte_balanced(spec.items, context.world_size)
    worker: Worker | None = None
    ownership = {"claimed": False}
    rounds: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    last_summary: dict[str, Any] | None = None
    startup_ready_sec = None
    phase = "setup"
    try:
        _prepare_directory(
            comm,
            context.rank,
            run_dir,
            effective_run_id,
            spec,
            rank_setup_timeout_sec,
            ownership,
        )
        init_error = None
        provenance = None
        try:
            if context.rank == 0:
                atomic_write_json(
                    run_dir / "execution-options.json",
                    {
                        "benchmark": benchmark.to_payload()
                        if benchmark
                        else None,
                        "rank_setup_timeout_sec": rank_setup_timeout_sec,
                        "mpi4py_version": api.mpi4py_version,
                        "mpi_library_version": api.library_version,
                    },
                    overwrite=False,
                )
            worker = spec.worker_factory(spec.options_payload)
            if not callable(
                getattr(worker, "run_item", None)
            ) or not callable(getattr(worker, "close", None)):
                raise TypeError("worker requires run_item and close methods")
            provenance = {
                "worker_id": context.rank,
                "hostname": context.host,
                "pid": os.getpid(),
                "cuda_visible_devices": visibility,
                "gpu": json_mapping(
                    worker.gpu_identity, field="worker GPU identity"
                ),
                "rank_context": context.to_dict(),
            }
        except Exception as exc:
            init_error = error_payload(exc)
        _mpi_failure_consensus(comm, "MPI worker initialization", init_error)
        ready = comm.gather(provenance, root=0)
        ready_error = None
        if context.rank == 0:
            try:
                problems = audit_worker_provenance(
                    ready,
                    backend=spec.backend,
                    expected_worker_count=context.world_size,
                )
                if problems:
                    raise ValueError(str(problems))
                atomic_write_json(
                    run_dir / "ready.json",
                    {"workers": ready},
                    overwrite=False,
                )
            except Exception as exc:
                ready_error = error_payload(exc)
        _mpi_failure_consensus(comm, "MPI worker readiness", ready_error)
        startup_ready_sec = time.perf_counter() - start
        assert worker is not None and provenance is not None
        planned_rounds = (
            benchmark.rounds()
            if benchmark
            else (BenchmarkRound("measure", 0),)
        )
        for planned in planned_rounds:
            phase = planned.round_id
            round_dir = (
                run_dir / "rounds" / planned.round_id
                if benchmark
                else run_dir
            )
            round_run_id = (
                planned.run_id(effective_run_id)
                if benchmark
                else effective_run_id
            )
            if benchmark:
                _prepare_directory(
                    comm,
                    context.rank,
                    round_dir,
                    round_run_id,
                    spec,
                    rank_setup_timeout_sec,
                )
            round_start = time.perf_counter()
            released = comm.bcast(
                planned.round_id if context.rank == 0 else None, root=0
            )
            release_sec = time.perf_counter() - round_start
            execution_error = None
            receipt = None
            worker_start = time.perf_counter()
            try:
                if released != planned.round_id:
                    raise RuntimeError("MPI round release identity differs")
                receipt = execute_worker_round(
                    worker,
                    items=shards[context.rank],
                    run_id=round_run_id,
                    run_dir=round_dir,
                    manifest_sha256=spec.manifest_sha256,
                    worker_id=context.rank,
                    backend=spec.backend,
                    provenance=provenance,
                )
            except Exception as exc:
                execution_error = error_payload(exc)
            worker_wall = time.perf_counter() - worker_start
            collection_start = time.perf_counter()
            gathered = comm.gather(
                {
                    "worker_result": receipt,
                    "worker_wall_sec": worker_wall,
                    "error": execution_error,
                },
                root=0,
            )
            collected_at = time.perf_counter()
            decision = None
            if context.rank == 0:
                try:
                    execution_errors = [
                        {"worker_id": rank, **value["error"]}
                        for rank, value in enumerate(gathered)
                        if value["error"] is not None
                    ]
                    worker_results = [
                        value["worker_result"]
                        for value in gathered
                        if value["worker_result"] is not None
                    ]
                    last_summary = finalize_round(
                        round_dir,
                        round_run_id,
                        spec,
                        shards,
                        worker_results,
                        artifact_timeout_sec=min(rank_setup_timeout_sec, 0.01)
                        if execution_errors
                        else rank_setup_timeout_sec,
                    )
                    if execution_errors:
                        last_summary["errors"].extend(execution_errors)
                        last_summary["status"] = "failed"
                    timing = {
                        "batch_wall_sec": collected_at - round_start,
                        "finalization_sec": last_summary["finalization_sec"],
                        "worker_wall_max_sec": max(
                            value["worker_wall_sec"] for value in gathered
                        ),
                        "coordinator_timings_sec": {
                            "release_sec": release_sec,
                            "collection_sec": collected_at - collection_start,
                        },
                    }
                    last_summary.update(timing)
                    if benchmark:
                        atomic_write_json(
                            round_dir / "summary.json", last_summary
                        )
                    rounds.append(
                        {
                            **planned.to_payload(),
                            "status": last_summary["status"],
                            "summary_path": str(
                                (round_dir / "summary.json").relative_to(
                                    run_dir
                                )
                            ),
                            **timing,
                        }
                    )
                    decision = {
                        "status": last_summary["status"],
                        "error": None,
                    }
                except Exception as exc:
                    decision = {
                        "status": "failed",
                        "error": error_payload(exc),
                    }
            decision = comm.bcast(decision, root=0)
            if decision["error"] is not None:
                raise RuntimeError(str(decision["error"]))
            if decision["status"] != "success":
                break
    except Exception as exc:
        errors.append({"phase": phase, **error_payload(exc)})
    finally:
        close_start = time.perf_counter()
        close_error = None
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                close_error = error_payload(exc)
        closed = comm.gather(
            {"worker_id": context.rank, "error": close_error}, root=0
        )
        close_sec = time.perf_counter() - close_start
        if context.rank == 0:
            errors.extend(
                {"phase": "close", **record}
                for record in closed
                if record["error"] is not None
            )

    result = None
    decision = None
    if context.rank == 0:
        try:
            if not ownership["claimed"]:
                raise RuntimeError(
                    "MPI run directory was not claimed: "
                    + "; ".join(
                        f"{error['type']}: {error['message']}"
                        for error in errors
                    )
                )
            if benchmark:
                report = build_benchmark_report(
                    benchmark, rounds, errors=errors
                )
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
                executor="mpi",
                run_id=effective_run_id,
                manifest_sha256=spec.manifest_sha256,
                world_size=context.world_size,
                mpi4py_version=api.mpi4py_version,
                mpi_library_version=api.library_version,
                started_at_utc=started_at,
                completed_at_utc=timestamp_utc(),
                startup_ready_sec=startup_ready_sec,
                close_sec=close_sec,
                coordinator_wall_sec=time.perf_counter() - start,
                launcher_exit_note=(
                    "Collective completion does not prove launcher exit."
                ),
            )
            atomic_write_json(run_dir / "summary.json", summary)
            result = ExecutionResult(
                "mpi",
                effective_run_id,
                run_dir,
                run_dir / "summary.json",
                summary["status"],
                summary,
            )
            decision = {"status": result.status, "error": None}
        except Exception as exc:
            decision = {"status": "failed", "error": error_payload(exc)}
    decision = comm.bcast(decision, root=0)
    if decision["error"] is not None:
        raise RuntimeError(
            "cannot persist MPI execution aggregate: "
            + str(decision["error"])
        )
    if context.rank != 0 and decision["status"] != "success":
        raise RuntimeError(f"MPI execution {effective_run_id!r} failed")
    return result


def _agree_payload(comm: Any, payload: Mapping[str, Any]) -> None:
    gathered = comm.gather(payload, root=0)
    error = None
    if int(comm.Get_rank()) == 0:
        try:
            identities = {
                json.dumps(value, sort_keys=True, allow_nan=False)
                for value in gathered
            }
            if len(gathered) != int(comm.Get_size()) or len(identities) != 1:
                raise ValueError(
                    "MPI workload, run options, or benchmark plans differ"
                )
        except Exception as exc:
            error = error_payload(exc)
    _mpi_failure_consensus(comm, "MPI workload agreement", error)


def _prepare_directory(
    comm: Any,
    rank: int,
    path: Path,
    run_id: str,
    spec: WorkloadSpec,
    timeout: float,
    ownership: dict[str, bool] | None = None,
) -> None:
    state = None
    if rank == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
            if ownership is not None:
                ownership["claimed"] = True
            prepare_run(path, run_id, spec, "mpi", directory_claimed=True)
            nonce = secrets.token_hex(32)
            ready = {
                "run_id": run_id,
                "manifest_sha256": spec.manifest_sha256,
                "run_dir": str(path),
                "nonce": nonce,
            }
            atomic_write_json(path / ".ready.json", ready, overwrite=False)
            state = {"error": None, "ready": ready}
        except Exception as exc:
            state = {"error": error_payload(exc), "ready": None}
    state = comm.bcast(state, root=0)
    if state["error"] is not None:
        raise RuntimeError("cannot prepare MPI run: " + str(state["error"]))
    shared_error = None
    deadline = time.monotonic() + timeout
    while True:
        try:
            ready = read_json_mapping(path / ".ready.json")
            if ready != state["ready"]:
                raise RuntimeError(
                    "MPI shared run directory identity differs"
                )
            break
        except OSError as exc:
            if time.monotonic() >= deadline:
                shared_error = error_payload(exc)
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        except Exception as exc:
            shared_error = error_payload(exc)
            break
    _mpi_failure_consensus(
        comm, "MPI shared run-directory validation", shared_error
    )
