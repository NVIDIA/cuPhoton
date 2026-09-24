# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional Dragon execution of persistent, independently placed workloads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cuphoton.core.benchmark import (
    BenchmarkOptions,
    build_benchmark_report,
)
from cuphoton.core.bulk import (
    Placement,
    WorkItem,
    atomic_write_json,
    error_payload,
    item_ids_sha256,
    json_mapping,
    new_run_id,
    partition_byte_balanced,
    regular_file,
    timestamp_utc,
    validate_identifier,
)
from cuphoton.core.bulk import (
    hostnames_match as _hostnames_match,
)
from cuphoton.core.execution import (
    ExecutionResult,
    WorkloadSpec,
    audit_worker_provenance,
    execute_worker_round,
    factory_reference,
    finalize_round,
    prepare_run,
    resolve_worker_factory,
)

_LAUNCH_SCHEMA = "cuphoton.core.dragon-launch/v1"
_TEMPLATE_BUDGET_BYTES = 96 * 1024
_WORKER_POLL_SEC = 0.5


@dataclass(frozen=True)
class _DragonAPI:
    System: Any
    Node: Any
    Policy: Any
    ProcessGroup: Any
    ProcessTemplate: Any
    Queue: Any


def discover_gpu_placements(
    system_type: Callable[[], Any],
    node_type: Callable[[Any], Any],
    *,
    node_ids: Sequence[Any] | None = None,
) -> tuple[Placement, ...]:
    """Enumerate actual Dragon Node.gpus IDs in allocation order."""

    if node_ids is None:
        node_ids = tuple(system_type().nodes)
    placements: list[Placement] = []
    seen: set[tuple[str, int]] = set()
    for node_id in node_ids:
        node = node_type(node_id)
        host = str(node.hostname)
        for gpu_id in node.gpus or []:
            if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
                raise RuntimeError(
                    f"Dragon node {host!r} reported a non-integer GPU ID"
                )
            key = (host, gpu_id)
            if key in seen:
                raise RuntimeError(
                    f"duplicate Dragon GPU placement: {host}:{gpu_id}"
                )
            seen.add(key)
            placements.append(
                Placement(
                    worker_id=len(placements),
                    host=host,
                    gpu_id=gpu_id,
                )
            )
    return tuple(placements)


def _select_gpu_placements(
    placements: Sequence[Placement], worker_count: int
) -> tuple[Placement, ...]:
    """Select GPUs round-robin across hosts, then assign dense worker IDs."""

    if isinstance(worker_count, bool) or not isinstance(worker_count, int):
        raise TypeError("worker_count must be an integer")
    if worker_count <= 0 or worker_count > len(placements):
        raise ValueError("worker_count exceeds available Dragon placements")
    by_host: dict[str, list[Placement]] = {}
    for placement in placements:
        by_host.setdefault(placement.host, []).append(placement)
    selected: list[Placement] = []
    offset = 0
    while len(selected) < worker_count:
        progressed = False
        for host_placements in by_host.values():
            if offset >= len(host_placements):
                continue
            selected.append(host_placements[offset])
            progressed = True
            if len(selected) == worker_count:
                break
        if not progressed:
            raise RuntimeError("could not select requested Dragon placements")
        offset += 1
    return tuple(
        Placement(
            worker_id=worker_id,
            host=placement.host,
            gpu_id=placement.gpu_id,
        )
        for worker_id, placement in enumerate(selected)
    )


def _singleton_cuda_visibility(expected_gpu_id: int | None = None) -> str:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        raise RuntimeError("Dragon worker has no CUDA_VISIBLE_DEVICES")
    tokens = [token.strip() for token in raw.split(",")]
    if (
        any(not token for token in tokens)
        or len(tokens) != 1
        or tokens[0] == "-1"
    ):
        raise RuntimeError(
            f"Dragon worker must see exactly one CUDA device, got {raw!r}"
        )
    if expected_gpu_id is not None and tokens[0] != str(expected_gpu_id):
        raise RuntimeError(
            "Dragon worker GPU placement mismatch: requested "
            f"{expected_gpu_id}, got CUDA_VISIBLE_DEVICES={raw!r}"
        )
    return tokens[0]


def _load_dragon_api() -> _DragonAPI:
    try:
        from dragon.infrastructure.policy import Policy
        from dragon.native.machine import Node, System
        from dragon.native.process import ProcessTemplate
        from dragon.native.process_group import ProcessGroup
        from dragon.native.queue import Queue
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The Dragon executor requires the dragonhpc runtime; "
            "install it in this Python environment on every node "
            "and launch with dragon"
        ) from exc
    return _DragonAPI(
        System=System,
        Node=Node,
        Policy=Policy,
        ProcessGroup=ProcessGroup,
        ProcessTemplate=ProcessTemplate,
        Queue=Queue,
    )


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _stable_gpu_physical_ids(
    value: Any,
) -> frozenset[tuple[str, str]] | None:
    if (
        not isinstance(value, Mapping)
        or "identity_error" not in value
        or value["identity_error"] is not None
    ):
        return None
    identities = frozenset(
        (field, identifier.strip().casefold())
        for field in ("uuid", "pci_bus_id")
        if isinstance((identifier := value.get(field)), str)
        and identifier.strip()
    )
    return identities or None


def _valid_shard_provenance(
    value: Any,
    *,
    placement: Placement,
    allow_loopback_alias: bool,
    backend: str,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    worker_id = _strict_integer(value.get("worker_id"))
    requested_gpu_id = _strict_integer(value.get("requested_gpu_id"))
    pid = _strict_integer(value.get("pid"))
    requested_host = value.get("requested_host")
    hostname = value.get("hostname")
    visibility = value.get("cuda_visible_devices")
    if (
        worker_id != placement.worker_id
        or requested_host != placement.host
        or requested_gpu_id != placement.gpu_id
        or pid is None
        or pid <= 0
        or not isinstance(hostname, str)
        or not hostname
        or not _hostnames_match(
            requested_host,
            hostname,
            allow_loopback_alias=allow_loopback_alias,
        )
        or not isinstance(visibility, str)
        or [token.strip() for token in visibility.split(",") if token.strip()]
        != [str(placement.gpu_id)]
    ):
        return False
    gpu = value.get("gpu")
    if not isinstance(gpu, Mapping) or not gpu:
        return False
    gpu_backend = gpu.get("backend")
    expected_identity_backend = (
        "cupy" if backend in {"cupy", "cutile"} else backend
    )
    return (
        isinstance(gpu_backend, str)
        and bool(gpu_backend)
        and gpu_backend == expected_identity_backend
        and _stable_gpu_physical_ids(gpu) is not None
    )


def _strict_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def run_dragon_work_items(
    *,
    prepare_workload: Callable[[int], WorkloadSpec],
    output_root: Path,
    run_id: str | None = None,
    max_workers: int | None = None,
    result_timeout_sec: float = 60.0,
    worker_timeout_sec: float = 3600.0,
    benchmark: BenchmarkOptions | None = None,
) -> ExecutionResult:
    """Execute a workload with persistent, explicitly placed Dragon workers.

    Worker factories run only after host and singleton CUDA binding checks.
    Complete item descriptors live in hashed shared-filesystem launch files;
    control queues carry bounded round commands and completion receipts.
    """

    invocation_start = time.perf_counter()
    for name, value in (
        ("result_timeout_sec", result_timeout_sec),
        ("worker_timeout_sec", worker_timeout_sec),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    if max_workers is not None and (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers <= 0
    ):
        raise ValueError("max_workers must be a positive integer")
    if benchmark is not None and not isinstance(benchmark, BenchmarkOptions):
        raise TypeError("benchmark must be BenchmarkOptions or None")
    spec = prepare_workload(0)
    if not isinstance(spec, WorkloadSpec):
        raise TypeError("prepare_workload must return WorkloadSpec")
    timings = {
        "workload_preflight_sec": time.perf_counter() - invocation_start
    }
    phase_start = time.perf_counter()
    api = _load_dragon_api()
    node_ids = tuple(api.System().nodes)
    available = discover_gpu_placements(
        api.System, api.Node, node_ids=node_ids
    )
    worker_count = min(
        max_workers or len(available), len(available), len(spec.items)
    )
    if not worker_count:
        raise RuntimeError("Dragon allocation exposes no usable GPUs")
    placements = _select_gpu_placements(available, worker_count)
    shards = partition_byte_balanced(spec.items, worker_count)
    timings["dragon_discovery_sec"] = time.perf_counter() - phase_start
    effective_run_id = run_id or new_run_id("dragon-workload")
    validate_identifier(effective_run_id, field="run_id")
    run_dir = output_root.expanduser().resolve() / effective_run_id
    phase_start = time.perf_counter()
    prepare_run(run_dir, effective_run_id, spec, "dragon")
    for name in ("launch", "startup"):
        (run_dir / name).mkdir()
    timings["run_artifact_setup_sec"] = time.perf_counter() - phase_start
    lifecycle_errors: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    terminal_summary: dict[str, Any] = {}
    command_queues: list[Any] = []
    results_queue: Any | None = None
    group: Any | None = None
    started = False
    joined = False
    exit_status: list[dict[str, int]] = []
    plan = benchmark or BenchmarkOptions()
    phase = "process_setup"
    phase_start = time.perf_counter()
    try:
        results_queue = api.Queue(maxsize=2 * worker_count)
        group = api.ProcessGroup(
            restart=False,
            ignore_error_on_exit=False,
            walltime=worker_timeout_sec,
        )
        for placement, shard in zip(placements, shards):
            policy = api.Policy(
                placement=api.Policy.Placement.HOST_NAME,
                host_name=placement.host,
                gpu_affinity=[placement.gpu_id],
            )
            commands = api.Queue(maxsize=1, policy=policy)
            command_queues.append(commands)
            descriptor = {
                "schema": _LAUNCH_SCHEMA,
                "run_id": effective_run_id,
                "run_dir": str(run_dir),
                "placement": placement.to_dict(),
                "items": [item.to_dict() for item in shard],
                "options": dict(spec.options_payload),
                "manifest_sha256": spec.manifest_sha256,
                "backend": spec.backend,
                "worker_factory": factory_reference(spec.worker_factory),
                "benchmark": benchmark.to_payload() if benchmark else None,
                "allow_loopback_alias": len(node_ids) == 1,
                "worker_timeout_sec": worker_timeout_sec,
                "result_timeout_sec": result_timeout_sec,
            }
            descriptor_path = (
                run_dir / "launch" / f"worker-{placement.worker_id:04d}.json"
            )
            atomic_write_json(descriptor_path, descriptor, overwrite=False)
            encoded = descriptor_path.read_bytes()
            context = {
                "run_dir": str(run_dir),
                "worker_id": placement.worker_id,
                "item_count": len(shard),
                "weight_bytes": sum(item.weight_bytes for item in shard),
                "item_ids_sha256": item_ids_sha256(shard),
                "descriptor_bytes": len(encoded),
                "result_timeout_sec": result_timeout_sec,
            }
            template = api.ProcessTemplate(
                target=_workload_worker,
                args=(
                    effective_run_id,
                    str(descriptor_path),
                    hashlib.sha256(encoded).hexdigest(),
                    context,
                    commands,
                    results_queue,
                ),
                policy=policy,
            )
            if len(template.argdata) > _TEMPLATE_BUDGET_BYTES:
                raise ValueError(
                    "Dragon worker launch arguments exceed 96 KiB"
                )
            group.add_process(nproc=1, template=template)
        phase = "init"
        group.init()
        timings["dragon_process_setup_sec"] = (
            time.perf_counter() - phase_start
        )
        phase = "start"
        phase_start = time.perf_counter()
        worker_deadline = time.monotonic() + worker_timeout_sec
        started = True
        try:
            group.start()
        finally:
            timings["dragon_launch_sec"] = time.perf_counter() - phase_start
        phase = "ready"
        try:
            _collect_messages(
                results_queue,
                ready,
                worker_count=worker_count,
                run_id=effective_run_id,
                kind="ready",
                round_id=None,
                deadline=worker_deadline,
                group=group,
                closed_messages=closed,
            )
            provenances = [message["provenance"] for message in ready]
            ready_errors = audit_worker_provenance(
                provenances,
                backend=spec.backend,
                expected_worker_count=worker_count,
            )
            if ready_errors:
                raise ValueError(
                    f"invalid Dragon READY provenance: {ready_errors}"
                )
            for message in ready:
                if not _valid_shard_provenance(
                    message["provenance"],
                    placement=placements[message["worker_id"]],
                    allow_loopback_alias=len(node_ids) == 1,
                    backend=spec.backend,
                ):
                    raise ValueError(
                        "Dragon READY differs from requested placement"
                    )
        finally:
            timings["readiness_sec"] = time.perf_counter() - invocation_start
        for round_spec in plan.rounds():
            phase = f"round:{round_spec.round_id}"
            round_dir = (
                run_dir / "rounds" / round_spec.round_id
                if benchmark
                else run_dir
            )
            round_run_id = (
                round_spec.run_id(effective_run_id)
                if benchmark
                else effective_run_id
            )
            if benchmark:
                prepare_run(round_dir, round_run_id, spec, "dragon")
            messages: list[dict[str, Any]] = []
            round_errors: list[dict[str, Any]] = []
            round_timings: dict[str, float] = {}
            batch_start = time.perf_counter()
            try:
                try:
                    for commands in command_queues:
                        remaining = min(
                            result_timeout_sec,
                            worker_deadline - time.monotonic(),
                        )
                        if remaining <= 0:
                            raise TimeoutError(
                                "Dragon worker deadline expired"
                            )
                        commands.put(
                            {
                                "run_id": effective_run_id,
                                **round_spec.to_payload(),
                            },
                            timeout=remaining,
                        )
                finally:
                    round_timings["dispatch_sec"] = (
                        time.perf_counter() - batch_start
                    )
                collection_start = time.perf_counter()
                try:
                    _collect_messages(
                        results_queue,
                        messages,
                        worker_count=worker_count,
                        run_id=effective_run_id,
                        kind="round",
                        round_id=round_spec.round_id,
                        deadline=worker_deadline,
                        group=group,
                        closed_messages=closed,
                    )
                finally:
                    round_timings["collection_sec"] = (
                        time.perf_counter() - collection_start
                    )
            except Exception as exc:
                round_errors.append(error_payload(exc))
            batch_wall = time.perf_counter() - batch_start
            worker_results = [
                message["result"]
                for message in messages
                if message.get("kind") == "round"
                and message.get("run_id") == effective_run_id
                and message.get("round_id") == round_spec.round_id
                and isinstance(message.get("result"), Mapping)
            ]
            ready_by_worker = {
                message["worker_id"]: message["provenance"]
                for message in ready
            }
            if any(
                result.get("provenance")
                != ready_by_worker.get(result.get("worker_id"))
                for result in worker_results
            ):
                round_errors.append(
                    {
                        "type": "WorkerIdentityChanged",
                        "message": "worker provenance changed after READY",
                    }
                )
            audit_start = time.perf_counter()
            terminal_summary = finalize_round(
                round_dir,
                round_run_id,
                spec,
                shards,
                worker_results,
                artifact_timeout_sec=min(result_timeout_sec, 0.01)
                if round_errors
                else result_timeout_sec,
            )
            round_timings["artifact_audit_sec"] = max(
                0.0,
                time.perf_counter()
                - audit_start
                - terminal_summary["finalization_sec"],
            )
            receipt = {
                **round_spec.to_payload(),
                "status": "success"
                if terminal_summary["status"] == "success"
                and not round_errors
                else "failed",
                "batch_wall_sec": batch_wall,
                "finalization_sec": terminal_summary["finalization_sec"],
                "worker_wall_max_sec": max(
                    (
                        _duration(message.get("worker_wall_sec"))
                        for message in messages
                    ),
                    default=0.0,
                ),
                "coordinator_timings_sec": round_timings,
                "summary_path": str(
                    (round_dir / "summary.json").relative_to(run_dir)
                ),
            }
            terminal_summary.update(
                receipt,
                parent_run_id=effective_run_id,
                messages=messages,
                round_errors=round_errors,
            )
            if benchmark:
                atomic_write_json(
                    round_dir / "summary.json", terminal_summary
                )
            rounds.append(receipt)
            if receipt["status"] != "success":
                raise RuntimeError(
                    f"Dragon round {round_spec.round_id} failed"
                )
        phase = "worker_close"
        close_start = time.perf_counter()
        for commands in command_queues:
            remaining = min(
                result_timeout_sec, worker_deadline - time.monotonic()
            )
            if remaining <= 0:
                raise TimeoutError("Dragon worker deadline expired")
            commands.put(
                {"run_id": effective_run_id, "kind": "close"},
                timeout=remaining,
            )
        _collect_messages(
            results_queue,
            closed,
            worker_count=worker_count,
            run_id=effective_run_id,
            kind="closed",
            round_id=None,
            deadline=time.monotonic() + result_timeout_sec,
            group=group,
        )
        timings["worker_close_sec"] = time.perf_counter() - close_start
        phase = "join"
        phase_start = time.perf_counter()
        try:
            group.join(timeout=result_timeout_sec)
            joined = True
        finally:
            timings["worker_join_sec"] = time.perf_counter() - phase_start
        phase = "unexpected_result"
        try:
            results_queue.get(timeout=0)
        except (queue.Empty, TimeoutError):
            pass
        else:
            raise ValueError("unexpected extra Dragon worker result")
    except Exception as exc:
        lifecycle_errors.append({"phase": phase, **error_payload(exc)})
    finally:
        timings.setdefault(
            "dragon_process_setup_sec", time.perf_counter() - phase_start
        )
        cleanup_start = time.perf_counter()
        if group is not None:
            if started and not joined:
                try:
                    group.stop(patience=5.0)
                except Exception as exc:
                    lifecycle_errors.append(
                        {"phase": "stop_after_failure", **error_payload(exc)}
                    )
            try:
                exit_status = [
                    {"puid": int(puid), "exit_code": int(code)}
                    for puid, code in group.inactive_puids
                ]
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "exit_status", **error_payload(exc)}
                )
            try:
                group.close(patience=5.0)
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "group_close", **error_payload(exc)}
                )
                cleanup = getattr(group, "_close_no_decorator", None)
                if cleanup is not None:
                    try:
                        cleanup(patience=5.0)
                    except Exception as cleanup_exc:
                        lifecycle_errors.append(
                            {
                                "phase": "forced_close",
                                **error_payload(cleanup_exc),
                            }
                        )
        for channel in [*command_queues, results_queue]:
            if channel is not None:
                try:
                    channel.close()
                except Exception as exc:
                    lifecycle_errors.append(
                        {"phase": "queue_close", **error_payload(exc)}
                    )
        timings["dragon_cleanup_sec"] = time.perf_counter() - cleanup_start
    process_audit = {
        "ok": len(exit_status) == worker_count
        and all(item["exit_code"] == 0 for item in exit_status),
        "expected_count": worker_count,
        "observed_count": len(exit_status),
        "nonzero": [item for item in exit_status if item["exit_code"] != 0],
    }
    if not process_audit["ok"]:
        lifecycle_errors.append(
            {
                "phase": "exit_status",
                "type": "WorkerExitFailure",
                "message": "missing or nonzero Dragon worker exit status",
            }
        )
    report = build_benchmark_report(plan, rounds, errors=lifecycle_errors)
    summary = {
        **(terminal_summary if benchmark is None else {}),
        "schema": "cuphoton.core.dragon-summary/v1",
        "executor": "dragon",
        "run_id": effective_run_id,
        "status": report["status"],
        "manifest_sha256": spec.manifest_sha256,
        "options": dict(spec.options_payload),
        "backend": spec.backend,
        "dragonhpc_version": _distribution_version("dragonhpc"),
        "worker_timeout_sec": worker_timeout_sec,
        "result_timeout_sec": result_timeout_sec,
        "worker_count": worker_count,
        "allocation_node_count": len(node_ids),
        "distinct_host_count": len(
            {placement.host for placement in placements}
        ),
        "placements": [placement.to_dict() for placement in placements],
        "command_queue_placement": "consumer",
        "coordinator_timings_sec": timings,
        "coordinator_wall_sec": time.perf_counter() - invocation_start,
        "coordinator_wall_definition": (
            "Function entry through worker cleanup and terminal audits; "
            "the final summary.json atomic commit is excluded."
        ),
        "completed_at_utc": timestamp_utc(),
        "ready_messages": ready,
        "closed_messages": closed,
        "process_exit_status": exit_status,
        "process_exit_audit": process_audit,
        "lifecycle_errors": lifecycle_errors,
    }
    if benchmark:
        summary["benchmark"] = report
    summary_path = run_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    return ExecutionResult(
        "dragon",
        effective_run_id,
        run_dir,
        summary_path,
        summary["status"],
        summary,
    )


def _duration(value: Any) -> float:
    return (
        float(value)
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        else 0.0
    )


def _collect_messages(
    results_queue: Any,
    messages: list[dict[str, Any]],
    *,
    worker_count: int,
    run_id: str,
    kind: str,
    round_id: str | None,
    deadline: float,
    group: Any,
    closed_messages: list[dict[str, Any]] | None = None,
) -> None:
    seen: dict[int, int] = {}
    failed_workers: set[int] = set()
    observed_exits: list[tuple[int, int]] | None = None
    while len(seen) < worker_count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Dragon {kind} deadline expired")
        try:
            message = json_mapping(
                results_queue.get(
                    timeout=0
                    if observed_exits is not None
                    else min(remaining, _WORKER_POLL_SEC)
                ),
                field=f"Dragon {kind} result",
            )
        except queue.Empty:
            if observed_exits is None:
                # Snapshot exits before draining: a worker sends its receipt
                # before exiting, possibly just after the timed get expires.
                observed_exits = list(group.inactive_puids)
                if not observed_exits:
                    observed_exits = None
                continue
            missing_exits = [
                (puid, code)
                for puid, code in observed_exits
                if puid not in seen.values()
            ]
            if missing_exits:
                raise RuntimeError(
                    f"Dragon worker exited before {kind} completion: "
                    f"{missing_exits}"
                )
            observed_exits = None
            continue
        worker_id = _strict_integer(message.get("worker_id"))
        puid = _strict_integer(message.get("puid"))
        if (
            closed_messages is not None
            and kind != "closed"
            and message.get("kind") == "closed"
        ):
            if (
                message.get("run_id") != run_id
                or message.get("round_id") is not None
                or worker_id not in failed_workers
                or puid != seen.get(worker_id)
                or any(
                    m.get("worker_id") == worker_id for m in closed_messages
                )
            ):
                raise ValueError(
                    "invalid or duplicate Dragon closed identity"
                )
            closed_messages.append(message)
            continue
        messages.append(message)
        if (
            message.get("run_id") != run_id
            or message.get("kind") != kind
            or message.get("round_id") != round_id
            or worker_id is None
            or worker_id not in range(worker_count)
            or worker_id in seen
            or puid is None
            or puid in seen.values()
        ):
            raise ValueError(f"invalid or duplicate Dragon {kind} identity")
        seen[worker_id] = puid
        if message.get("status") != "success":
            failed_workers.add(worker_id)
        if kind == "round" and message.get("status") == "success":
            result = message.get("result")
            duration = message.get("worker_wall_sec")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
                or not isinstance(result, Mapping)
                or _strict_integer(result.get("worker_id")) != worker_id
                or result.get("status") != "success"
            ):
                raise ValueError("invalid Dragon round worker result")
    if failed_workers:
        raise RuntimeError(
            f"Dragon workers {sorted(failed_workers)} {kind} failed"
        )


def _read_descriptor(
    run_id: str,
    descriptor_path_raw: str,
    digest: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = Path(context["run_dir"])
    worker_id = context["worker_id"]
    path = Path(descriptor_path_raw)
    if (
        not run_dir.is_absolute()
        or path != run_dir / "launch" / f"worker-{worker_id:04d}.json"
        or not regular_file(path)
    ):
        raise ValueError("Dragon launch descriptor is not a regular run file")
    with path.open("rb") as handle:
        encoded = handle.read(context["descriptor_bytes"] + 1)
    if (
        len(encoded) != context["descriptor_bytes"]
        or hashlib.sha256(encoded).hexdigest() != digest
    ):
        raise ValueError("Dragon launch descriptor size or SHA-256 differs")
    descriptor = json_mapping(
        json.loads(encoded), field="Dragon launch descriptor"
    )
    if (
        descriptor["schema"] != _LAUNCH_SCHEMA
        or descriptor["run_id"] != run_id
        or descriptor["run_dir"] != str(run_dir)
        or descriptor["placement"]["worker_id"] != worker_id
    ):
        raise ValueError("Dragon launch descriptor identity differs")
    items = tuple(
        WorkItem.from_dict(payload) for payload in descriptor["items"]
    )
    if (
        len(items) != context["item_count"]
        or sum(item.weight_bytes for item in items) != context["weight_bytes"]
        or item_ids_sha256(items) != context["item_ids_sha256"]
    ):
        raise ValueError("Dragon launch descriptor shard identity differs")
    return descriptor


def _validate_binding(
    placement: Placement, *, allow_loopback_alias: bool
) -> tuple[str, str]:
    hostname = socket.gethostname()
    if not _hostnames_match(
        placement.host, hostname, allow_loopback_alias=allow_loopback_alias
    ):
        raise RuntimeError(
            "Dragon worker host differs from requested placement"
        )
    visibility = _singleton_cuda_visibility(placement.gpu_id)
    premature = [
        name
        for name in ("cupy", "numba.cuda", "cuda.tile", "torch")
        if name in sys.modules
    ]
    if premature:
        raise RuntimeError(
            "CUDA modules were imported before Dragon worker placement: "
            + ", ".join(premature)
        )
    return hostname, visibility


def _current_puid() -> int:
    from dragon.infrastructure.parameters import this_process

    return int(this_process.my_puid)


def _workload_worker(
    run_id: str,
    descriptor_path: str,
    digest: str,
    context: Mapping[str, Any],
    commands: Any,
    results_queue: Any,
) -> None:
    worker = None
    worker_id = context["worker_id"]
    puid = _current_puid()
    run_dir = Path(context["run_dir"])
    timeout = context["result_timeout_sec"]
    ready: dict[str, Any] = {
        "kind": "ready",
        "puid": puid,
        "run_id": run_id,
        "round_id": None,
        "worker_id": worker_id,
        "status": "success",
    }
    failed = False
    failure_error = None
    try:
        try:
            descriptor = _read_descriptor(
                run_id, descriptor_path, digest, context
            )
            placement = Placement(**descriptor["placement"])
            hostname, visibility = _validate_binding(
                placement,
                allow_loopback_alias=descriptor["allow_loopback_alias"],
            )
            factory = resolve_worker_factory(descriptor["worker_factory"])
            worker = factory(descriptor["options"])
            provenance = {
                "worker_id": worker_id,
                "requested_host": placement.host,
                "requested_gpu_id": placement.gpu_id,
                "hostname": hostname,
                "pid": os.getpid(),
                "cuda_visible_devices": visibility,
                "gpu": json_mapping(
                    worker.gpu_identity, field="worker GPU identity"
                ),
            }
            ready["provenance"] = provenance
        except Exception as exc:
            ready.update(status="failed", error=error_payload(exc))
        try:
            atomic_write_json(
                run_dir / "startup" / f"worker-{worker_id:04d}.json", ready
            )
        except Exception as exc:
            ready.update(status="failed", artifact_error=error_payload(exc))
        results_queue.put(ready, timeout=timeout)
        if ready["status"] != "success":
            raise RuntimeError("Dragon worker initialization failed")
        benchmark = (
            BenchmarkOptions(**descriptor["benchmark"])
            if descriptor["benchmark"]
            else None
        )
        items = tuple(
            WorkItem.from_dict(payload) for payload in descriptor["items"]
        )
        for round_spec in (benchmark or BenchmarkOptions()).rounds():
            message: dict[str, Any] = {
                "kind": "round",
                "puid": puid,
                "run_id": run_id,
                "round_id": round_spec.round_id,
                "worker_id": worker_id,
                "status": "success",
            }
            round_dir = (
                run_dir / "rounds" / round_spec.round_id
                if benchmark
                else run_dir
            )
            try:
                command = commands.get(
                    timeout=descriptor["worker_timeout_sec"]
                )
                expected = {"run_id": run_id, **round_spec.to_payload()}
                if (
                    not isinstance(command, Mapping)
                    or command != expected
                    or any(
                        type(command.get(key)) is not type(value)
                        for key, value in expected.items()
                    )
                ):
                    raise ValueError("unexpected Dragon round command")
                start = time.perf_counter()
                result = execute_worker_round(
                    worker,
                    items=items,
                    run_id=round_spec.run_id(run_id) if benchmark else run_id,
                    run_dir=round_dir,
                    manifest_sha256=descriptor["manifest_sha256"],
                    worker_id=worker_id,
                    backend=descriptor["backend"],
                    provenance=provenance,
                )
                message.update(
                    status=result["status"],
                    result=result,
                    worker_wall_sec=time.perf_counter() - start,
                )
            except Exception as exc:
                message.update(status="failed", error=error_payload(exc))
                try:
                    atomic_write_json(
                        round_dir / "errors" / f"worker-{worker_id:04d}.json",
                        message,
                    )
                except Exception as write_exc:
                    message["artifact_error"] = error_payload(write_exc)
            results_queue.put(message, timeout=timeout)
            if message["status"] != "success":
                raise RuntimeError(
                    f"Dragon worker round {round_spec.round_id} failed"
                )
        # Closing produces another result-queue message. Wait until the
        # coordinator has collected and audited every final-round receipt.
        command = commands.get(timeout=descriptor["worker_timeout_sec"])
        if command != {"run_id": run_id, "kind": "close"}:
            raise ValueError("unexpected Dragon worker close command")
    except BaseException as exc:
        failed = True
        failure_error = error_payload(exc)
        raise
    finally:
        closed = {
            "kind": "closed",
            "puid": puid,
            "run_id": run_id,
            "round_id": None,
            "worker_id": worker_id,
            "status": "failed" if failed else "success",
        }
        if failure_error is not None:
            closed["error"] = failure_error
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                closed.update(status="failed", error=error_payload(exc))
        try:
            atomic_write_json(
                run_dir / "startup" / f"worker-{worker_id:04d}-closed.json",
                closed,
            )
        except Exception as exc:
            closed.update(status="failed", artifact_error=error_payload(exc))
        try:
            results_queue.put(closed, timeout=timeout)
        except Exception:
            if not failed:
                raise
        if closed["status"] != "success" and not failed:
            raise RuntimeError("Dragon worker cleanup failed")
