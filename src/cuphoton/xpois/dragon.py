# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional Dragon ProcessGroup adapter for XPOIS image pairs."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import queue
import socket
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cuphoton.core.benchmark import BenchmarkOptions, build_benchmark_report
from cuphoton.core.bulk import (
    Placement,
    WorkItem,
    atomic_write_json,
    audit_terminal_records,
    classify_physical_gpu_pair,
    error_payload,
    json_mapping,
    new_run_id,
    partition_byte_balanced,
    read_json_mapping,
    timestamp_utc,
    validate_identifier,
)
from cuphoton.core.bulk import (
    collect_gpu_identity as _collect_gpu_identity,
)
from cuphoton.core.bulk import (
    item_ids_sha256 as _item_ids_sha256,
)
from cuphoton.core.bulk import (
    regular_file as _regular_file,
)

from .batch import (
    BatchFitOptions,
    load_image_pair_manifest,
    preflight_image_pair_manifest,
    run_image_pair_item,
)

_DRAGON_GPU_BACKENDS = frozenset({"cupy", "numba-cuda", "cutile"})
_DRAGON_LAUNCH_SCHEMA = "cuphoton.xpois.dragon-launch/v1"
_DRAGON_TEMPLATE_BUDGET_BYTES = 96 * 1024
_WORKER_POLL_SEC = 0.5


def _dragon_process_id() -> int:
    from dragon.infrastructure.parameters import this_process

    return int(this_process.my_puid)


@dataclass(frozen=True)
class DragonBatchResult:
    """Handle for a terminal XPOIS Dragon batch."""

    run_id: str
    run_dir: Path
    summary_path: Path
    status: str
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a compact command result."""

        return {
            "executor": "dragon",
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "summary_path": str(self.summary_path),
            "status": self.status,
        }


@dataclass(frozen=True)
class _DragonAPI:
    System: Any
    Node: Any
    Policy: Any
    ProcessGroup: Any
    ProcessTemplate: Any
    Queue: Any


def _resolve_worker_target(
    reference: Mapping[str, Any],
) -> Callable[..., None]:
    """Resolve one package-importable worker target."""

    normalized = json_mapping(reference, field="Dragon worker reference")
    if set(normalized) != {"module", "qualname"}:
        raise ValueError("Dragon worker reference has invalid fields")
    module_name = normalized["module"]
    qualname = normalized["qualname"]
    if (
        not isinstance(module_name, str)
        or not module_name
        or module_name == "__main__"
        or not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
    ):
        raise ValueError("Dragon worker target must be package-importable")
    target: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        if not component or component.startswith("<"):
            raise ValueError("Dragon worker target has an invalid qualname")
        target = getattr(target, component)
    if not callable(target):
        raise TypeError("Dragon worker target reference is not callable")
    return target


def _worker_target_reference(
    target: Callable[..., None],
) -> dict[str, str]:
    """Return and verify a stable import reference for a worker target."""

    reference = {
        "module": str(getattr(target, "__module__", "")),
        "qualname": str(getattr(target, "__qualname__", "")),
    }
    try:
        resolved = _resolve_worker_target(reference)
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise ValueError(
            "Dragon worker_target must be a package-importable top-level "
            "callable, not a __main__ or local function"
        ) from exc
    if resolved is not target:
        raise ValueError(
            "Dragon worker_target import reference resolves to a different "
            "callable"
        )
    return reference


def _dragon_launch_worker(
    run_id: str,
    descriptor_path_raw: str,
    descriptor_sha256: str,
    context: Mapping[str, Any],
    results_queue: Any,
    commands: Any | None = None,
) -> None:
    """Load an immutable shard descriptor before importing its worker."""

    started_at = timestamp_utc()
    worker_start = time.perf_counter()
    run_dir = Path(context["run_dir"])
    worker_id = context["worker_id"]
    try:
        descriptor_path = Path(descriptor_path_raw)
        if (
            not run_dir.is_absolute()
            or descriptor_path
            != run_dir / "launch" / f"worker-{worker_id:04d}.json"
            or not _regular_file(descriptor_path)
        ):
            raise ValueError(
                "Dragon launch descriptor is not a regular run file"
            )
        with descriptor_path.open("rb") as handle:
            encoded = handle.read(context["descriptor_bytes"] + 1)
        if len(encoded) != context["descriptor_bytes"]:
            raise ValueError("Dragon launch descriptor size differs")
        if hashlib.sha256(encoded).hexdigest() != descriptor_sha256:
            raise ValueError("Dragon launch descriptor SHA-256 differs")
        descriptor = json_mapping(
            json.loads(encoded), field="Dragon launch descriptor"
        )
        if (
            descriptor["schema"] != _DRAGON_LAUNCH_SCHEMA
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
            or sum(item.weight_bytes for item in items)
            != context["weight_bytes"]
            or _item_ids_sha256(items) != context["item_ids_sha256"]
        ):
            raise ValueError(
                "Dragon launch descriptor shard identity differs"
            )
        target = _resolve_worker_target(descriptor["worker_target"])
        if commands is None:
            target(
                run_id,
                str(run_dir),
                descriptor["placement"],
                descriptor["items"],
                descriptor["options"],
                results_queue,
                descriptor["allow_loopback_alias"],
            )
            return
    except Exception as exc:
        result = {
            "schema": context["shard_schema"],
            "worker_id": worker_id,
            "status": "failed",
            "item_count": context["item_count"],
            "success_count": 0,
            "failed_count": context["item_count"],
            "weight_bytes": context["weight_bytes"],
            "item_ids_sha256": context["item_ids_sha256"],
            "started_at_utc": started_at,
            "completed_at_utc": timestamp_utc(),
            "worker_wall_sec": time.perf_counter() - worker_start,
            "timings_sec": {},
            "provenance": None,
            "record_write_errors": [],
            "error": error_payload(exc),
        }
        if commands is not None:
            result.update(
                kind="ready",
                run_id=run_id,
                round_id=None,
                puid=_dragon_process_id(),
            )
        try:
            atomic_write_json(
                run_dir / "launch" / f"worker-{worker_id:04d}-failure.json",
                result,
                overwrite=False,
            )
        except Exception as write_exc:
            result["artifact_error"] = error_payload(write_exc)
        try:
            results_queue.put(result)
        except Exception as report_exc:
            exc.add_note(f"Dragon launch failure report failed: {report_exc}")
        raise
    # The persistent worker reports its own READY and round failures. Keep its
    # invocation outside this handler to avoid a second READY receipt.
    target(
        run_id,
        str(run_dir),
        descriptor["placement"],
        descriptor["items"],
        descriptor["options"],
        descriptor["benchmark"],
        commands,
        results_queue,
        descriptor["allow_loopback_alias"],
        descriptor["worker_timeout_sec"],
        descriptor["result_timeout_sec"],
    )


def run_dragon_image_pair_batch(
    *,
    manifest_path: Path,
    output_root: Path,
    run_id: str | None,
    max_workers: int | None,
    result_timeout_sec: float,
    options: BatchFitOptions,
    worker_timeout_sec: float = 3600.0,
    benchmark: BenchmarkOptions | None = None,
) -> DragonBatchResult:
    """Run deterministic XPOIS shards with one Dragon worker per GPU."""

    invocation_start = time.perf_counter()
    started_at = timestamp_utc()
    coordinator_timings: dict[str, float] = {}
    if benchmark is not None and not isinstance(benchmark, BenchmarkOptions):
        raise TypeError("benchmark must be BenchmarkOptions or None")
    if options.backend not in _DRAGON_GPU_BACKENDS:
        raise ValueError(
            "Dragon XPOIS workers require an explicit GPU backend"
        )
    phase_start = time.perf_counter()
    manifest = load_image_pair_manifest(manifest_path)
    coordinator_timings["manifest_load_sec"] = (
        time.perf_counter() - phase_start
    )
    phase_start = time.perf_counter()
    preflight_image_pair_manifest(manifest, options)
    coordinator_timings["manifest_preflight_sec"] = (
        time.perf_counter() - phase_start
    )
    return run_dragon_work_items(
        items=manifest.work_items(),
        output_root=output_root,
        run_id=run_id,
        max_workers=max_workers,
        result_timeout_sec=result_timeout_sec,
        worker_timeout_sec=worker_timeout_sec,
        backend=options.backend,
        options_payload=options.to_payload(),
        manifest_payload=manifest.canonical_payload(),
        input_identity_payload=manifest.input_identity_payload(),
        manifest_sha256=manifest.sha256,
        worker_target=_dragon_shard_worker,
        run_prefix="dragon-xpois",
        run_schema="cuphoton.xpois.dragon-run/v1",
        summary_schema="cuphoton.xpois.dragon-summary/v1",
        record_schema="cuphoton.xpois.dragon-item/v1",
        shard_schema="cuphoton.xpois.dragon-shard/v1",
        coordinator_timings=coordinator_timings,
        invocation_start=invocation_start,
        started_at=started_at,
        benchmark=benchmark,
    )


def run_dragon_work_items(
    *,
    items: Sequence[WorkItem],
    output_root: Path,
    run_id: str | None,
    max_workers: int | None,
    result_timeout_sec: float,
    backend: str,
    options_payload: Mapping[str, Any],
    manifest_payload: Mapping[str, Any],
    input_identity_payload: Mapping[str, Any],
    manifest_sha256: str,
    worker_target: Callable[..., None],
    run_prefix: str,
    run_schema: str,
    summary_schema: str,
    record_schema: str,
    shard_schema: str,
    worker_timeout_sec: float = 3600.0,
    benchmark: BenchmarkOptions | None = None,
    coordinator_timings: Mapping[str, float] | None = None,
    invocation_start: float | None = None,
    started_at: str | None = None,
    success_record_validator: (
        Callable[[Mapping[str, Any]], Sequence[str]] | None
    ) = None,
    failed_record_validator: (
        Callable[[Mapping[str, Any]], Sequence[str]] | None
    ) = None,
) -> DragonBatchResult:
    """Run preflighted whole-item shards with one Dragon worker per GPU.

    Callers own workload-specific manifest parsing, item construction, and
    worker targets. This coordinator owns the established Dragon placement,
    lifecycle, durable terminal-record, and audit contracts. Workers must be
    importable callables taking run ID, run directory, placement, item and
    options payloads, result queue, and the loopback-alias flag, in that
    order. Launch descriptors require a shared filesystem.
    """

    if benchmark is not None:
        if not isinstance(benchmark, BenchmarkOptions):
            raise TypeError("benchmark must be BenchmarkOptions or None")
        if worker_target is not _dragon_shard_worker:
            raise ValueError(
                "benchmark rounds currently require the XPOIS worker"
            )
    if invocation_start is None:
        invocation_start = time.perf_counter()
    started_at = started_at or timestamp_utc()
    coordinator_timings = dict(coordinator_timings or {})
    if (
        isinstance(result_timeout_sec, bool)
        or not isinstance(result_timeout_sec, (int, float))
        or not math.isfinite(result_timeout_sec)
        or result_timeout_sec <= 0
    ):
        raise ValueError("result_timeout_sec must be positive")
    if max_workers is not None and (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers <= 0
    ):
        raise ValueError("max_workers must be a positive integer")
    if (
        isinstance(worker_timeout_sec, bool)
        or not isinstance(worker_timeout_sec, (int, float))
        or not math.isfinite(worker_timeout_sec)
        or worker_timeout_sec <= 0
    ):
        raise ValueError("worker_timeout_sec must be positive")
    if backend not in _DRAGON_GPU_BACKENDS:
        raise ValueError("Dragon workers require an explicit GPU backend")
    if not items:
        raise ValueError("Dragon work items must not be empty")
    if not callable(worker_target):
        raise TypeError("worker_target must be callable")
    validate_identifier(run_prefix, field="run prefix")
    for name, value in (
        ("run_schema", run_schema),
        ("summary_schema", summary_schema),
        ("record_schema", record_schema),
        ("shard_schema", shard_schema),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(value not in "0123456789abcdef" for value in manifest_sha256)
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    normalized_options = json_mapping(
        options_payload, field="Dragon worker options"
    )
    if normalized_options.get("backend") != backend:
        raise ValueError(
            "Dragon worker options backend must match the coordinator backend"
        )
    normalized_manifest = json_mapping(
        manifest_payload, field="Dragon manifest payload"
    )
    normalized_input_identity = json_mapping(
        input_identity_payload, field="Dragon input identity payload"
    )
    worker_reference = _worker_target_reference(worker_target)

    phase_start = time.perf_counter()
    api = _load_dragon_api()
    system = api.System()
    node_ids = tuple(system.nodes)
    allocation_node_count = len(node_ids)
    placements = discover_gpu_placements(
        lambda: system,
        api.Node,
        node_ids=node_ids,
    )
    coordinator_timings["dragon_discovery_sec"] = (
        time.perf_counter() - phase_start
    )
    phase_start = time.perf_counter()
    requested_workers = max_workers or len(placements)
    worker_count = min(requested_workers, len(placements), len(items))
    if worker_count <= 0:
        raise RuntimeError("Dragon allocation exposes no usable GPUs")
    selected = _select_gpu_placements(placements, worker_count)
    distinct_host_count = len({placement.host for placement in selected})
    shards = partition_byte_balanced(items, worker_count)
    coordinator_timings["partition_sec"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    join_timeout_sec = float(
        worker_timeout_sec + result_timeout_sec
        if benchmark is None
        else result_timeout_sec
    )
    effective_run_id = run_id or new_run_id(run_prefix)
    validate_identifier(effective_run_id, field="run_id")
    run_dir = output_root.expanduser().resolve() / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    directories = (
        ("items", "launch", "records", "workers")
        if benchmark is None
        else ("launch", "rounds", "startup")
    )
    for name in directories:
        (run_dir / name).mkdir()
    atomic_write_json(run_dir / "manifest.json", normalized_manifest)
    atomic_write_json(
        run_dir / "input-identity.json",
        normalized_input_identity,
    )
    dragon_version = _distribution_version("dragonhpc")
    launch_payload = {
        "schema": run_schema,
        "record_type": "immutable-launch",
        "executor": "dragon",
        "run_id": effective_run_id,
        "started_at_utc": started_at,
        "manifest_sha256": manifest_sha256,
        "dragonhpc_version": dragon_version,
        "worker_count": worker_count,
        "allocation_node_count": allocation_node_count,
        "distinct_host_count": distinct_host_count,
        "worker_timeout_sec": worker_timeout_sec,
        "join_timeout_sec": join_timeout_sec,
        "result_timeout_sec": result_timeout_sec,
        "placements": [placement.to_dict() for placement in selected],
        "options": normalized_options,
        "worker_target": worker_reference,
    }
    if benchmark is not None:
        launch_payload["benchmark"] = benchmark.to_payload()
    atomic_write_json(run_dir / "run.json", launch_payload)
    coordinator_timings["run_artifact_setup_sec"] = (
        time.perf_counter() - phase_start
    )
    if benchmark is not None:
        return _run_dragon_rounds(
            api=api,
            run_dir=run_dir,
            launch_payload=launch_payload,
            placements=selected,
            shards=shards,
            options=BatchFitOptions.from_payload(normalized_options),
            benchmark=benchmark,
            invocation_start=invocation_start,
            coordinator_timings=coordinator_timings,
        )

    lifecycle_errors: list[dict[str, str]] = []
    exit_status: list[dict[str, int]] = []
    shard_results: list[dict[str, Any]] = []
    queue_result_count = 0
    results_queue: Any | None = None
    group: Any | None = None
    start_attempted = False
    join_completed_cleanly = False
    lifecycle_phase = "queue_create"
    process_setup_start = time.perf_counter()
    try:
        results_queue = api.Queue(maxsize=worker_count)
        lifecycle_phase = "group_create"
        group = api.ProcessGroup(
            restart=False,
            ignore_error_on_exit=False,
            walltime=worker_timeout_sec,
        )
        lifecycle_phase = "process_setup"
        for placement, shard in zip(selected, shards):
            descriptor_path = (
                run_dir / "launch" / f"worker-{placement.worker_id:04d}.json"
            )
            atomic_write_json(
                descriptor_path,
                {
                    "schema": _DRAGON_LAUNCH_SCHEMA,
                    "run_id": effective_run_id,
                    "run_dir": str(run_dir),
                    "worker_target": worker_reference,
                    "placement": placement.to_dict(),
                    "items": [item.to_dict() for item in shard],
                    "options": normalized_options,
                    "allow_loopback_alias": allocation_node_count == 1,
                },
                overwrite=False,
            )
            encoded = descriptor_path.read_bytes()
            descriptor_sha256 = hashlib.sha256(encoded).hexdigest()
            context = {
                "run_dir": str(run_dir),
                "worker_id": placement.worker_id,
                "shard_schema": shard_schema,
                "item_count": len(shard),
                "weight_bytes": sum(item.weight_bytes for item in shard),
                "item_ids_sha256": _item_ids_sha256(shard),
                "descriptor_bytes": len(encoded),
            }
            policy = api.Policy(
                placement=api.Policy.Placement.HOST_NAME,
                host_name=placement.host,
                gpu_affinity=[placement.gpu_id],
            )
            template = api.ProcessTemplate(
                target=_dragon_launch_worker,
                args=(
                    effective_run_id,
                    str(descriptor_path),
                    descriptor_sha256,
                    context,
                    results_queue,
                ),
                policy=policy,
            )
            if len(template.argdata) > _DRAGON_TEMPLATE_BUDGET_BYTES:
                raise ValueError(
                    "Dragon worker launch arguments exceed 96 KiB"
                )
            group.add_process(nproc=1, template=template)

        lifecycle_phase = "init"
        group.init()
        coordinator_timings["dragon_process_setup_sec"] = (
            time.perf_counter() - process_setup_start
        )
        lifecycle_phase = "start"
        phase_start = time.perf_counter()
        start_attempted = True
        try:
            group.start()
        finally:
            coordinator_timings["dragon_launch_sec"] = (
                time.perf_counter() - phase_start
            )
        lifecycle_phase = "join"
        phase_start = time.perf_counter()
        try:
            group.join(timeout=join_timeout_sec)
            join_completed_cleanly = True
        except Exception as exc:
            phase = (
                "join_timeout" if isinstance(exc, TimeoutError) else "join"
            )
            lifecycle_errors.append({"phase": phase, **error_payload(exc)})
        try:
            coordinator_timings["worker_join_sec"] = (
                time.perf_counter() - phase_start
            )
            result_collection_start = time.perf_counter()
            lifecycle_phase = "exit_status"
            try:
                # ``exit_status`` is exception-decorated. ``inactive_puids``
                # is the same public state without re-raising a worker
                # exception, so it remains inspectable after a failed join.
                exit_status = [
                    {"puid": int(puid), "exit_code": int(code)}
                    for puid, code in group.inactive_puids
                ]
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "exit_status", **error_payload(exc)}
                )
            deadline = time.monotonic() + result_timeout_sec
            lifecycle_phase = "queue_result"
            while queue_result_count < worker_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    result = results_queue.get(timeout=remaining)
                except (queue.Empty, TimeoutError):
                    break
                queue_result_count += 1
                if isinstance(result, Mapping):
                    try:
                        shard_results.append(
                            json_mapping(result, field="Dragon shard result")
                        )
                    except (TypeError, ValueError) as exc:
                        lifecycle_errors.append(
                            {
                                "phase": "queue_result",
                                "type": "InvalidShardResult",
                                "message": str(exc),
                            }
                        )
                else:
                    lifecycle_errors.append(
                        {
                            "phase": "queue_result",
                            "type": "InvalidShardResult",
                            "message": "worker result was not a mapping",
                        }
                    )
            coordinator_timings["result_collection_sec"] = (
                time.perf_counter() - result_collection_start
            )
        finally:
            coordinator_timings.setdefault(
                "worker_join_sec", time.perf_counter() - phase_start
            )
    except Exception as exc:
        lifecycle_errors.append(
            {"phase": lifecycle_phase, **error_payload(exc)}
        )
    finally:
        coordinator_timings.setdefault(
            "dragon_process_setup_sec",
            time.perf_counter() - process_setup_start,
        )
        cleanup_start = time.perf_counter()
        if group is not None:
            if start_attempted and not join_completed_cleanly:
                try:
                    group.stop(patience=5.0)
                except Exception as exc:
                    lifecycle_errors.append(
                        {"phase": "stop_after_failure", **error_payload(exc)}
                    )
            try:
                group.close(patience=5.0)
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "close", **error_payload(exc)}
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
        if results_queue is not None:
            try:
                results_queue.close()
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "queue_close", **error_payload(exc)}
                )
        coordinator_timings["dragon_cleanup_sec"] = (
            time.perf_counter() - cleanup_start
        )

    audit_start = time.perf_counter()
    records, record_errors = _wait_terminal_artifacts(
        run_dir,
        shards,
        shard_results,
        result_timeout_sec,
    )
    coordinator_timings["artifact_visibility_sec"] = (
        time.perf_counter() - audit_start
    )
    audit_start = time.perf_counter()
    audit = audit_terminal_records([item.item_id for item in items], records)
    expected_records = {
        item.item_id: {
            "worker_id": placement.worker_id,
            "weight_bytes": item.weight_bytes,
            "backend": backend,
            "solver": normalized_options.get("solver"),
        }
        for placement, shard in zip(selected, shards)
        for item in shard
    }
    record_errors.extend(
        _audit_terminal_record_contract(
            run_id=effective_run_id,
            expected=expected_records,
            records=records,
            record_schema=record_schema,
            success_record_validator=success_record_validator,
            failed_record_validator=failed_record_validator,
        )
    )
    shard_audit = _audit_shard_results(
        shards,
        shard_results,
        records,
        placements=selected,
        allow_loopback_alias=allocation_node_count == 1,
        backend=backend,
        shard_schema=shard_schema,
    )
    item_timings, timing_errors = _aggregate_item_timings(records)
    record_errors.extend(timing_errors)
    process_audit = {
        "ok": len(exit_status) == worker_count
        and all(item["exit_code"] == 0 for item in exit_status),
        "expected_count": worker_count,
        "observed_count": len(exit_status),
        "nonzero": [item for item in exit_status if item["exit_code"] != 0],
    }
    status = (
        "success"
        if audit["ok"]
        and not audit["failed_item_ids"]
        and shard_audit["ok"]
        and process_audit["ok"]
        and not lifecycle_errors
        and not record_errors
        else "failed"
    )
    coordinator_timings["terminal_audit_sec"] = (
        time.perf_counter() - audit_start
    )
    setup_phases = (
        "manifest_load_sec",
        "manifest_preflight_sec",
        "dragon_discovery_sec",
        "partition_sec",
        "run_artifact_setup_sec",
        "dragon_process_setup_sec",
        "dragon_launch_sec",
    )
    coordinator_setup_sec = sum(
        coordinator_timings.get(name, 0.0) for name in setup_phases
    )
    coordinator_wall_sec = time.perf_counter() - invocation_start
    summary = {
        "schema": summary_schema,
        "executor": "dragon",
        "status": status,
        "run_id": effective_run_id,
        "started_at_utc": started_at,
        "completed_at_utc": timestamp_utc(),
        "coordinator_wall_sec": coordinator_wall_sec,
        "coordinator_wall_definition": (
            "Function entry through worker cleanup and terminal audits; "
            "the final summary.json atomic commit is excluded."
        ),
        "coordinator_timings_sec": coordinator_timings,
        "coordinator_totals_sec": {"setup": coordinator_setup_sec},
        "manifest_sha256": manifest_sha256,
        "dragonhpc_version": dragon_version,
        "worker_timeout_sec": worker_timeout_sec,
        "join_timeout_sec": join_timeout_sec,
        "result_timeout_sec": result_timeout_sec,
        "allocation_node_count": allocation_node_count,
        "distinct_host_count": distinct_host_count,
        "options": normalized_options,
        "placements": [placement.to_dict() for placement in selected],
        "shards": [
            {
                "worker_id": placement.worker_id,
                "item_count": len(shard),
                "weight_bytes": sum(item.weight_bytes for item in shard),
                "item_ids_sha256": _item_ids_sha256(shard),
            }
            for placement, shard in zip(selected, shards)
        ],
        "shard_results": sorted(shard_results, key=_shard_result_sort_key),
        "queue_result_count": queue_result_count,
        "shard_result_audit": shard_audit,
        "terminal_record_audit": audit,
        "terminal_record_errors": record_errors,
        "process_exit_status": exit_status,
        "process_exit_audit": process_audit,
        "lifecycle_errors": lifecycle_errors,
        "timings_sec": item_timings,
    }
    summary_path = run_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    return DragonBatchResult(
        run_id=effective_run_id,
        run_dir=run_dir,
        summary_path=summary_path,
        status=status,
        summary=summary,
    )


def _run_dragon_rounds(
    *,
    api: _DragonAPI,
    run_dir: Path,
    launch_payload: Mapping[str, Any],
    placements: Sequence[Placement],
    shards: Sequence[Sequence[WorkItem]],
    options: BatchFitOptions,
    benchmark: BenchmarkOptions,
    invocation_start: float,
    coordinator_timings: dict[str, float],
) -> DragonBatchResult:
    """Keep placed workers alive while timing complete production shards."""

    run_id = launch_payload["run_id"]
    worker_count = len(placements)
    worker_timeout = launch_payload["worker_timeout_sec"]
    result_timeout = launch_payload["result_timeout_sec"]
    allow_loopback = launch_payload["allocation_node_count"] == 1
    lifecycle_errors: list[dict[str, Any]] = []
    ready_messages: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    exit_status: list[dict[str, int]] = []
    command_queues: list[Any] = []
    results_queue: Any | None = None
    group: Any | None = None
    start_attempted = False
    joined = False
    phase = "queue_create"
    process_setup_start = time.perf_counter()
    try:
        results_queue = api.Queue(maxsize=worker_count)
        phase = "group_create"
        group = api.ProcessGroup(
            restart=False,
            ignore_error_on_exit=False,
            walltime=worker_timeout,
        )
        phase = "process_setup"
        for placement, shard in zip(placements, shards):
            descriptor_path = (
                run_dir / "launch" / f"worker-{placement.worker_id:04d}.json"
            )
            atomic_write_json(
                descriptor_path,
                {
                    "schema": _DRAGON_LAUNCH_SCHEMA,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "worker_target": _worker_target_reference(
                        _dragon_round_worker
                    ),
                    "placement": placement.to_dict(),
                    "items": [item.to_dict() for item in shard],
                    "options": options.to_payload(),
                    "benchmark": benchmark.to_payload(),
                    "allow_loopback_alias": allow_loopback,
                    "worker_timeout_sec": worker_timeout,
                    "result_timeout_sec": result_timeout,
                },
                overwrite=False,
            )
            encoded = descriptor_path.read_bytes()
            context = {
                "run_dir": str(run_dir),
                "worker_id": placement.worker_id,
                "shard_schema": "cuphoton.xpois.dragon-shard/v1",
                "item_count": len(shard),
                "weight_bytes": sum(item.weight_bytes for item in shard),
                "item_ids_sha256": _item_ids_sha256(shard),
                "descriptor_bytes": len(encoded),
            }
            policy = api.Policy(
                placement=api.Policy.Placement.HOST_NAME,
                host_name=placement.host,
                gpu_affinity=[placement.gpu_id],
            )
            # Blocking receives reside with their consumer. Python TCP's
            # coordinator transport threads must remain free to send.
            commands = api.Queue(maxsize=1, policy=policy)
            command_queues.append(commands)
            template = api.ProcessTemplate(
                target=_dragon_launch_worker,
                args=(
                    run_id,
                    str(descriptor_path),
                    hashlib.sha256(encoded).hexdigest(),
                    context,
                    results_queue,
                    commands,
                ),
                policy=policy,
            )
            if len(template.argdata) > _DRAGON_TEMPLATE_BUDGET_BYTES:
                raise ValueError(
                    "Dragon worker launch arguments exceed 96 KiB"
                )
            group.add_process(nproc=1, template=template)
        phase = "init"
        group.init()
        coordinator_timings["dragon_process_setup_sec"] = (
            time.perf_counter() - process_setup_start
        )
        phase = "start"
        launch_start = time.perf_counter()
        worker_deadline = time.monotonic() + worker_timeout
        start_attempted = True
        try:
            group.start()
        finally:
            coordinator_timings["dragon_launch_sec"] = (
                time.perf_counter() - launch_start
            )
        phase = "ready"
        ready_start = time.perf_counter()
        try:
            _collect_worker_messages(
                results_queue,
                ready_messages,
                worker_count=worker_count,
                run_id=run_id,
                kind="ready",
                round_id=None,
                deadline=worker_deadline,
                group=group,
            )
            for message in ready_messages:
                placement = placements[message["worker_id"]]
                if not _valid_shard_provenance(
                    message.get("provenance"),
                    placement=placement,
                    allow_loopback_alias=allow_loopback,
                    backend=options.backend,
                ):
                    raise ValueError("invalid Dragon READY provenance")
        finally:
            coordinator_timings["worker_ready_wait_sec"] = (
                time.perf_counter() - ready_start
            )
            coordinator_timings["readiness_sec"] = (
                time.perf_counter() - invocation_start
            )
        for spec in benchmark.rounds():
            phase = f"round:{spec.round_id}"
            round_dir = run_dir / "rounds" / spec.round_id
            for name in ("items", "records", "workers"):
                (round_dir / name).mkdir(parents=True, exist_ok=False)
            messages: list[dict[str, Any]] = []
            round_errors: list[dict[str, Any]] = []
            timings: dict[str, float] = {}
            batch_start = time.perf_counter()
            try:
                try:
                    for commands in command_queues:
                        remaining = min(
                            result_timeout,
                            worker_deadline - time.monotonic(),
                        )
                        if remaining <= 0:
                            raise TimeoutError(
                                "Dragon worker deadline expired"
                            )
                        commands.put(
                            {"run_id": run_id, **spec.to_payload()},
                            timeout=remaining,
                        )
                finally:
                    timings["dispatch_sec"] = (
                        time.perf_counter() - batch_start
                    )
                collection_start = time.perf_counter()
                try:
                    _collect_worker_messages(
                        results_queue,
                        messages,
                        worker_count=worker_count,
                        run_id=run_id,
                        kind="round",
                        round_id=spec.round_id,
                        deadline=worker_deadline,
                        group=group,
                    )
                finally:
                    timings["collection_sec"] = (
                        time.perf_counter() - collection_start
                    )
            except Exception as exc:
                round_errors.append(error_payload(exc))
            batch_wall = time.perf_counter() - batch_start
            shard_results = [
                message["result"]
                for message in messages
                if message.get("kind") == "round"
                and message.get("run_id") == run_id
                and message.get("round_id") == spec.round_id
                and isinstance(message.get("result"), Mapping)
            ]
            ready_by_worker = {
                message["worker_id"]: message["provenance"]
                for message in ready_messages
            }
            if any(
                result.get("provenance")
                != ready_by_worker.get(result.get("worker_id"))
                for result in shard_results
            ):
                round_errors.append(
                    {
                        "type": "WorkerIdentityChanged",
                        "message": "worker provenance changed after READY",
                    }
                )
            audit_start = time.perf_counter()
            records, record_errors = _wait_terminal_artifacts(
                round_dir, shards, shard_results, result_timeout
            )
            audit = _audit_batch_records(
                run_id=spec.run_id(run_id),
                shards=shards,
                placements=placements,
                options=options,
                records=records,
                record_errors=record_errors,
                shard_results=shard_results,
                allow_loopback_alias=allow_loopback,
            )
            timings["artifact_audit_sec"] = time.perf_counter() - audit_start
            worker_wall = max(
                (
                    float(message["worker_wall_sec"])
                    for message in messages
                    if isinstance(
                        message.get("worker_wall_sec"), (int, float)
                    )
                    and not isinstance(message["worker_wall_sec"], bool)
                    and math.isfinite(message["worker_wall_sec"])
                    and message["worker_wall_sec"] >= 0
                ),
                default=0.0,
            )
            receipt = {
                **spec.to_payload(),
                "status": (
                    "success"
                    if audit["status"] == "success" and not round_errors
                    else "failed"
                ),
                "batch_wall_sec": batch_wall,
                "worker_wall_max_sec": worker_wall,
                "coordinator_timings_sec": timings,
                "summary_path": f"rounds/{spec.round_id}/summary.json",
            }
            atomic_write_json(
                round_dir / "summary.json",
                {
                    "schema": "cuphoton.xpois.dragon-round/v1",
                    "run_id": spec.run_id(run_id),
                    "parent_run_id": run_id,
                    **audit,
                    **receipt,
                    "messages": messages,
                    "errors": round_errors,
                },
            )
            rounds.append(receipt)
            if receipt["status"] != "success":
                raise RuntimeError(f"Dragon round {spec.round_id} failed")
        phase = "join"
        join_start = time.perf_counter()
        try:
            group.join(timeout=result_timeout)
            joined = True
        finally:
            coordinator_timings["worker_join_sec"] = (
                time.perf_counter() - join_start
            )
        phase = "unexpected_result"
        try:
            extra = results_queue.get(timeout=0)
        except (queue.Empty, TimeoutError):
            pass
        else:
            raise ValueError(f"unexpected Dragon worker result: {extra!r}")
    except Exception as exc:
        lifecycle_errors.append({"phase": phase, **error_payload(exc)})
    finally:
        coordinator_timings.setdefault(
            "dragon_process_setup_sec",
            time.perf_counter() - process_setup_start,
        )
        cleanup_start = time.perf_counter()
        if group is not None:
            if start_attempted and not joined:
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
                    {"phase": "close", **error_payload(exc)}
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
        coordinator_timings["dragon_cleanup_sec"] = (
            time.perf_counter() - cleanup_start
        )
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
    report = build_benchmark_report(
        benchmark, rounds, errors=lifecycle_errors
    )
    summary = {
        **launch_payload,
        "schema": "cuphoton.xpois.dragon-benchmark-summary/v1",
        "record_type": "terminal",
        "status": report["status"],
        "completed_at_utc": timestamp_utc(),
        "coordinator_wall_sec": time.perf_counter() - invocation_start,
        "coordinator_wall_definition": (
            "Function entry through worker cleanup and terminal audits; "
            "the final summary.json atomic commit is excluded."
        ),
        "coordinator_timings_sec": coordinator_timings,
        "ready_messages": ready_messages,
        "command_queue_placement": "consumer",
        "process_exit_status": exit_status,
        "process_exit_audit": process_audit,
        "lifecycle_errors": lifecycle_errors,
        "benchmark": report,
    }
    summary_path = run_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    return DragonBatchResult(
        run_id=run_id,
        run_dir=run_dir,
        summary_path=summary_path,
        status=report["status"],
        summary=summary,
    )


def _collect_worker_messages(
    results_queue: Any,
    messages: list[dict[str, Any]],
    *,
    worker_count: int,
    run_id: str,
    kind: str,
    round_id: str | None,
    deadline: float,
    group: Any,
) -> None:
    """Collect one identified receipt per worker, failing on any replay."""

    seen: set[int] = set()
    seen_puids: set[int] = set()
    failed: list[int] = []
    while len(seen) < worker_count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Dragon {kind} deadline expired")
        try:
            raw = results_queue.get(timeout=min(remaining, _WORKER_POLL_SEC))
        except queue.Empty:
            exits = list(group.inactive_puids)
            # A worker publishes before exiting. Recheck the queue after the
            # exit snapshot so a concurrently delivered receipt is retained.
            try:
                raw = results_queue.get(timeout=0)
            except queue.Empty:
                missing = [
                    entry for entry in exits if entry[0] not in seen_puids
                ]
                if missing:
                    raise RuntimeError(
                        f"Dragon worker exited before {kind} completion: "
                        f"{missing}"
                    ) from None
                continue
        message = json_mapping(raw, field=f"Dragon {kind} result")
        messages.append(message)
        worker_id = _strict_integer(message.get("worker_id"))
        puid = _strict_integer(message.get("puid"))
        if (
            message.get("run_id") != run_id
            or message.get("kind") != kind
            or message.get("round_id") != round_id
            or worker_id is None
            or worker_id not in range(worker_count)
            or worker_id in seen
            or puid is None
            or puid in seen_puids
        ):
            raise ValueError(f"invalid or duplicate Dragon {kind} identity")
        seen.add(worker_id)
        seen_puids.add(puid)
        if message.get("status") != "success":
            failed.append(worker_id)
            continue
        if kind == "round":
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
                raise ValueError("invalid Dragon round shard result")
    if failed:
        raise RuntimeError(f"Dragon workers {failed} {kind} failed")


def _audit_batch_records(
    *,
    run_id: str,
    shards: Sequence[Sequence[WorkItem]],
    placements: Sequence[Placement],
    options: BatchFitOptions,
    records: Sequence[Mapping[str, Any]],
    record_errors: Sequence[Mapping[str, Any]],
    shard_results: Sequence[Mapping[str, Any]],
    allow_loopback_alias: bool,
) -> dict[str, Any]:
    """Apply the same scientific artifact contract to every execution."""

    errors = list(record_errors)
    audit = audit_terminal_records(
        [item.item_id for shard in shards for item in shard], records
    )
    expected = {
        item.item_id: {
            "worker_id": placement.worker_id,
            "weight_bytes": item.weight_bytes,
            "backend": options.backend,
            "solver": options.solver,
        }
        for placement, shard in zip(placements, shards)
        for item in shard
    }
    errors.extend(
        _audit_terminal_record_contract(
            run_id=run_id, expected=expected, records=records
        )
    )
    shard_audit = _audit_shard_results(
        shards,
        shard_results,
        records,
        placements=placements,
        allow_loopback_alias=allow_loopback_alias,
        backend=options.backend,
    )
    item_timings, timing_errors = _aggregate_item_timings(records)
    errors.extend(timing_errors)
    return {
        "status": (
            "success"
            if audit["ok"]
            and not audit["failed_item_ids"]
            and shard_audit["ok"]
            and not errors
            else "failed"
        ),
        "shard_result_audit": shard_audit,
        "terminal_record_audit": audit,
        "terminal_record_errors": errors,
        "timings_sec": item_timings,
    }


def _dragon_round_worker(
    run_id: str,
    run_dir_raw: str,
    placement_payload: Mapping[str, Any],
    item_payloads: Sequence[Mapping[str, Any]],
    options_payload: Mapping[str, Any],
    benchmark_payload: Mapping[str, Any],
    commands: Any,
    results_queue: Any,
    allow_loopback_alias: bool,
    worker_timeout: float,
    result_timeout: float,
) -> None:
    """Initialize once, then execute the normal shard for each release."""

    run_dir = Path(run_dir_raw)
    placement = Placement(**dict(placement_payload))
    ready: dict[str, Any] = {
        "kind": "ready",
        "run_id": run_id,
        "round_id": None,
        "worker_id": placement.worker_id,
        "puid": _dragon_process_id(),
        "status": "success",
    }
    try:
        items = tuple(
            WorkItem.from_dict(payload) for payload in item_payloads
        )
        options = BatchFitOptions.from_payload(options_payload)
        benchmark = BenchmarkOptions(**dict(benchmark_payload))
        actual_host, visibility = _validate_worker_placement(
            placement,
            require_clean_cuda_imports=True,
            allow_loopback_alias=allow_loopback_alias,
        )
        gpu_identity = dict(_collect_gpu_identity(options.backend))
        ready["provenance"] = {
            "worker_id": placement.worker_id,
            "requested_host": placement.host,
            "requested_gpu_id": placement.gpu_id,
            "hostname": actual_host,
            "pid": os.getpid(),
            "cuda_visible_devices": visibility,
            "gpu": gpu_identity,
        }
    except Exception as exc:
        ready.update(status="failed", error=error_payload(exc))
    try:
        atomic_write_json(
            run_dir / "startup" / f"worker-{placement.worker_id:04d}.json",
            ready,
        )
    except Exception as exc:
        ready.update(status="failed", artifact_error=error_payload(exc))
    results_queue.put(ready, timeout=result_timeout)
    if ready["status"] != "success":
        raise RuntimeError("Dragon worker initialization failed")
    for spec in benchmark.rounds():
        round_dir = run_dir / "rounds" / spec.round_id
        message: dict[str, Any] = {
            "kind": "round",
            "run_id": run_id,
            "round_id": spec.round_id,
            "worker_id": placement.worker_id,
            "puid": ready["puid"],
            "status": "success",
        }
        try:
            command = commands.get(timeout=worker_timeout)
            if command != {"run_id": run_id, **spec.to_payload()}:
                raise ValueError("unexpected Dragon round command")
            worker_start = time.perf_counter()
            result = _execute_shard(
                run_id=spec.run_id(run_id),
                run_dir=round_dir,
                placement=placement,
                items=items,
                options=options,
                item_runner=run_image_pair_item,
                gpu_identity_loader=lambda backend: gpu_identity,
                require_clean_cuda_imports=False,
                allow_loopback_alias=allow_loopback_alias,
            )
            message.update(
                status=result["status"],
                result=result,
                worker_wall_sec=time.perf_counter() - worker_start,
            )
        except Exception as exc:
            message.update(status="failed", error=error_payload(exc))
            try:
                atomic_write_json(
                    round_dir
                    / "errors"
                    / f"worker-{placement.worker_id:04d}.json",
                    message,
                )
            except Exception as artifact_exc:
                message["artifact_error"] = error_payload(artifact_exc)
        results_queue.put(message, timeout=result_timeout)
        if message["status"] != "success":
            raise RuntimeError(f"Dragon worker round {spec.round_id} failed")


def _validate_worker_placement(
    placement: Placement,
    *,
    require_clean_cuda_imports: bool,
    allow_loopback_alias: bool,
) -> tuple[str, str]:
    actual_host = socket.gethostname()
    if not _hostnames_match(
        placement.host,
        actual_host,
        allow_loopback_alias=allow_loopback_alias,
    ):
        raise RuntimeError(
            "Dragon worker placement mismatch: requested "
            f"{placement.host!r}, running on {actual_host!r}"
        )
    visibility = _singleton_cuda_visibility(placement.gpu_id)
    premature = sorted(
        name
        for name in ("cupy", "numba.cuda", "cuda.tile", "torch")
        if name in sys.modules
    )
    if require_clean_cuda_imports and premature:
        raise RuntimeError(
            "CUDA modules were imported before Dragon worker placement: "
            + ", ".join(premature)
        )
    return actual_host, visibility


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


def _dragon_shard_worker(
    run_id: str,
    run_dir_raw: str,
    placement_payload: Mapping[str, Any],
    item_payloads: Sequence[Mapping[str, Any]],
    options_payload: Mapping[str, Any],
    results_queue: Any,
    allow_loopback_alias: bool,
) -> None:
    """Dragon target: validate placement, run one shard, report compactly."""

    worker_start = time.perf_counter()
    started_at = timestamp_utc()
    run_dir: Path | None = None
    placement: Placement | None = None
    items: tuple[WorkItem, ...] = ()
    try:
        run_dir = Path(run_dir_raw)
        placement = Placement(**dict(placement_payload))
        items = tuple(
            WorkItem.from_dict(payload) for payload in item_payloads
        )
        options = BatchFitOptions.from_payload(options_payload)
        result = _execute_shard(
            run_id=run_id,
            run_dir=run_dir,
            placement=placement,
            items=items,
            options=options,
            item_runner=run_image_pair_item,
            gpu_identity_loader=_collect_gpu_identity,
            allow_loopback_alias=allow_loopback_alias,
        )
    except Exception as exc:
        worker_id = (
            placement.worker_id
            if placement is not None
            else _strict_integer(
                placement_payload.get("worker_id")
                if isinstance(placement_payload, Mapping)
                else None
            )
        )
        result = {
            "schema": "cuphoton.xpois.dragon-shard/v1",
            "worker_id": worker_id,
            "status": "failed",
            "item_count": len(items),
            "success_count": 0,
            "failed_count": len(items),
            "weight_bytes": sum(item.weight_bytes for item in items),
            "item_ids_sha256": _item_ids_sha256(items),
            "started_at_utc": started_at,
            "completed_at_utc": timestamp_utc(),
            "worker_wall_sec": time.perf_counter() - worker_start,
            "timings_sec": {},
            "provenance": None,
            "record_write_errors": [],
            "error": error_payload(exc),
        }
        if run_dir is not None and worker_id is not None and worker_id >= 0:
            worker_path = run_dir / "workers" / f"worker-{worker_id:04d}.json"
            try:
                # An existing worker file is _execute_shard's complete
                # durable result; this coarse fallback must not replace it.
                if not _regular_file(worker_path):
                    atomic_write_json(worker_path, result)
            except Exception as artifact_exc:
                result["artifact_error"] = error_payload(artifact_exc)
    results_queue.put(result)


def _execute_shard(
    *,
    run_id: str,
    run_dir: Path,
    placement: Placement,
    items: Sequence[WorkItem],
    options: Any,
    item_runner: Callable[[WorkItem, Path, Any], Mapping[str, Any]],
    gpu_identity_loader: Callable[[str], Mapping[str, Any]],
    backend: str | None = None,
    record_schema: str = "cuphoton.xpois.dragon-item/v1",
    shard_schema: str = "cuphoton.xpois.dragon-shard/v1",
    require_clean_cuda_imports: bool = True,
    allow_loopback_alias: bool = False,
) -> dict[str, Any]:
    """Execute one shard with injectable CUDA-free test collaborators."""

    worker_start = time.perf_counter()
    started_at = timestamp_utc()
    actual_host = ""
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_identity: dict[str, Any] | None = None
    setup_exception: Exception | None = None
    setup_error: dict[str, str] | None = None
    resolved_backend = backend or (
        options.get("backend")
        if isinstance(options, Mapping)
        else getattr(options, "backend", None)
    )
    if not isinstance(resolved_backend, str) or not resolved_backend:
        raise ValueError("Dragon shard backend must be a non-empty string")
    try:
        actual_host = socket.gethostname()
        actual_host, visibility = _validate_worker_placement(
            placement,
            require_clean_cuda_imports=require_clean_cuda_imports,
            allow_loopback_alias=allow_loopback_alias,
        )
        gpu_identity = dict(gpu_identity_loader(resolved_backend))
    except Exception as exc:
        setup_exception = exc
        setup_error = error_payload(exc)
    provenance = {
        "worker_id": placement.worker_id,
        "requested_host": placement.host,
        "requested_gpu_id": placement.gpu_id,
        "hostname": actual_host,
        "pid": os.getpid(),
        "cuda_visible_devices": visibility,
        "gpu": gpu_identity,
    }
    success_count = 0
    failed_count = 0
    record_write_errors: list[dict[str, str]] = []
    worker_timings: dict[str, float] = {}
    for item in items:
        item_start = time.perf_counter()
        identity = {
            "schema": record_schema,
            "run_id": run_id,
            "item_id": item.item_id,
            "worker_id": placement.worker_id,
            "weight_bytes": item.weight_bytes,
            "started_at_utc": timestamp_utc(),
        }
        record: dict[str, Any] = dict(identity)
        try:
            if setup_exception is not None:
                raise setup_exception
            item_dir = run_dir / "items" / item.item_id
            metadata = dict(item_runner(item, item_dir, options))
            _validate_success_item_output(item_dir, metadata)
            for field in ("run_dir", "summary_path"):
                if field in metadata:
                    metadata[field] = str(
                        Path(metadata[field]).relative_to(run_dir)
                    )
            if "timings_sec" in metadata:
                metadata["timings_sec"] = _validated_timings(
                    metadata["timings_sec"],
                    field=f"item {item.item_id!r} timings_sec",
                )
            if "wall_sec" in metadata:
                metadata["wall_sec"] = _validated_timings(
                    metadata["wall_sec"],
                    field=f"item {item.item_id!r} wall_sec",
                )
            record.update(metadata)
            record.update(identity)
            record["status"] = "success"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = error_payload(exc)
            notes = getattr(exc, "__notes__", ())
            if notes:
                record["error"]["notes"] = "\n".join(map(str, notes))
        record["worker_seconds"] = time.perf_counter() - item_start
        record["completed_at_utc"] = timestamp_utc()
        for phase, value in record.get("timings_sec", {}).items():
            if isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                worker_timings[str(phase)] = worker_timings.get(
                    str(phase), 0.0
                ) + float(value)
        try:
            atomic_write_json(
                run_dir / "records" / f"{item.item_id}.json", record
            )
        except Exception as exc:
            record_write_errors.append(
                {"item_id": item.item_id, **error_payload(exc)}
            )
            failed_count += 1
        else:
            if record["status"] == "success":
                success_count += 1
            else:
                failed_count += 1

    result = {
        "schema": shard_schema,
        "worker_id": placement.worker_id,
        "status": (
            "success"
            if setup_error is None and failed_count == 0
            else "failed"
        ),
        "item_count": len(items),
        "success_count": success_count,
        "failed_count": failed_count,
        "weight_bytes": sum(item.weight_bytes for item in items),
        "item_ids_sha256": _item_ids_sha256(items),
        "started_at_utc": started_at,
        "completed_at_utc": timestamp_utc(),
        "worker_wall_sec": time.perf_counter() - worker_start,
        "timings_sec": worker_timings,
        "provenance": provenance,
        "record_write_errors": record_write_errors,
        "error": setup_error,
    }
    atomic_write_json(
        run_dir / "workers" / f"worker-{placement.worker_id:04d}.json",
        result,
    )
    return result


def _validate_success_item_output(
    item_dir: Path, metadata: Mapping[str, Any]
) -> None:
    try:
        item_mode = item_dir.lstat().st_mode
    except OSError as exc:
        raise ValueError(
            "item runner did not create a real output directory"
        ) from exc
    if not stat.S_ISDIR(item_mode):
        raise ValueError("item runner did not create a real output directory")
    try:
        recorded_dir = Path(metadata["run_dir"])
        summary_path = Path(metadata["summary_path"])
    except (KeyError, TypeError) as exc:
        raise ValueError("item runner did not report output paths") from exc
    if recorded_dir != item_dir:
        raise ValueError("item runner reported a different output directory")
    if summary_path != item_dir / "summary.json":
        raise ValueError("item runner reported a different summary path")
    try:
        summary_mode = summary_path.lstat().st_mode
    except OSError as exc:
        raise ValueError(
            "item runner did not create a regular summary file"
        ) from exc
    if not stat.S_ISREG(summary_mode):
        raise ValueError("item runner did not create a regular summary file")


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


def _hostnames_match(
    requested: str,
    actual: str,
    *,
    allow_loopback_alias: bool = False,
) -> bool:
    """Accept exact or equivalent short/FQDN scheduler hostnames."""

    requested_normalized = requested.rstrip(".").lower()
    actual_normalized = actual.rstrip(".").lower()
    if allow_loopback_alias and requested_normalized in {
        "localhost",
        "localhost.localdomain",
    }:
        # Dragon 0.14.1 reports ``localhost`` for its single-node system
        # descriptor even though socket.gethostname() exposes the machine
        # hostname inside the launched worker.
        return bool(actual_normalized)
    if requested_normalized == actual_normalized:
        return True
    if "." not in requested_normalized:
        return actual_normalized.startswith(requested_normalized + ".")
    if "." not in actual_normalized:
        return requested_normalized.startswith(actual_normalized + ".")
    return False


def _load_terminal_records(
    run_dir: Path,
    visible_expected: Sequence[Path] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    paths = set((run_dir / "records").glob("*.json"))
    paths.update(visible_expected)
    for path in sorted(paths):
        try:
            if not _regular_file(path):
                raise ValueError(f"record is not a regular file: {path.name}")
            record = read_json_mapping(path)
            if record.get("item_id") != path.stem:
                raise ValueError("record item_id differs from its filename")
            records.append(record)
        except Exception as exc:
            error = {
                "record_path": str(path.relative_to(run_dir)),
                **error_payload(exc),
            }
            if isinstance(exc, OSError):
                error["retryable"] = True
            errors.append(error)
    return records, errors


def _wait_terminal_artifacts(
    run_dir: Path,
    shards: Sequence[Sequence[WorkItem]],
    shard_results: Sequence[Mapping[str, Any]],
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Wait for worker-published records and successful item summaries."""

    assigned = {
        worker_id: {item.item_id for item in shard}
        for worker_id, shard in enumerate(shards)
    }
    unwritten: set[str] = set()
    reporting: set[int] = set()
    for result in shard_results:
        worker_id = _strict_integer(result.get("worker_id"))
        if worker_id is not None:
            reporting.add(worker_id)
        worker_items = assigned.get(worker_id, set())
        if result.get("provenance") is None and isinstance(
            result.get("error"), Mapping
        ):
            unwritten.update(worker_items)
        write_errors = result.get("record_write_errors")
        if isinstance(write_errors, list):
            unwritten.update(
                item_id
                for error in write_errors
                if isinstance(error, Mapping)
                and isinstance((item_id := error.get("item_id")), str)
                and item_id in worker_items
            )
    for worker_id, worker_items in assigned.items():
        if worker_id not in reporting:
            unwritten.update(worker_items)
    expected = {
        item.item_id: run_dir / "records" / f"{item.item_id}.json"
        for shard in shards
        for item in shard
        if item.item_id not in unwritten
    }
    deadline = time.monotonic() + timeout
    delay = 0.05
    records: list[dict[str, Any]] = []
    record_errors: list[dict[str, Any]] = []
    missing: list[str] = []
    missing_summaries: list[str] = []
    while True:
        visible_expected = []
        missing = []
        for path in expected.values():
            if _regular_file(path):
                visible_expected.append(path)
            else:
                missing.append(str(path.relative_to(run_dir)))
        records, raw_errors = _load_terminal_records(
            run_dir, visible_expected
        )
        permanent_errors = [
            {key: value for key, value in error.items() if key != "retryable"}
            for error in raw_errors
            if not error.get("retryable", False)
        ]
        if permanent_errors:
            return records, permanent_errors
        record_errors = [
            {key: value for key, value in error.items() if key != "retryable"}
            for error in raw_errors
        ]
        missing_summaries = [
            f"items/{item_id}/summary.json"
            for record in records
            if record.get("status") == "success"
            and isinstance((item_id := record.get("item_id")), str)
            and item_id in expected
            and not _regular_file(
                run_dir / "items" / item_id / "summary.json"
            )
        ]
        if not missing and not record_errors and not missing_summaries:
            return records, []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 1.0)
    errors = [*record_errors]
    errors.extend(
        {
            "type": "TimeoutError",
            "message": f"Dragon artifact did not become visible: {path}",
        }
        for path in (*missing, *missing_summaries)
    )
    return records, errors


def _audit_terminal_record_contract(
    *,
    run_id: str,
    expected: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    record_schema: str = "cuphoton.xpois.dragon-item/v1",
    success_record_validator: (
        Callable[[Mapping[str, Any]], Sequence[str]] | None
    ) = None,
    failed_record_validator: (
        Callable[[Mapping[str, Any]], Sequence[str]] | None
    ) = None,
) -> list[dict[str, str]]:
    """Validate durable Dragon item records beyond identity/status counts."""

    errors: list[dict[str, str]] = []
    for index, record in enumerate(records):
        item_id_value = record.get("item_id")
        item_id = item_id_value if isinstance(item_id_value, str) else ""
        problems: list[str] = []
        if record.get("schema") != record_schema:
            problems.append("schema")
        if record.get("run_id") != run_id:
            problems.append("run_id")
        assignment = expected.get(item_id)
        if assignment is None:
            problems.append("item_id")
        else:
            worker_id = _strict_integer(record.get("worker_id"))
            if worker_id != assignment["worker_id"]:
                problems.append("worker_id")
            weight_bytes = _strict_integer(record.get("weight_bytes"))
            if weight_bytes != assignment["weight_bytes"]:
                problems.append("weight_bytes")
        started_at = _parse_timezone_aware_timestamp(
            record.get("started_at_utc")
        )
        completed_at = _parse_timezone_aware_timestamp(
            record.get("completed_at_utc")
        )
        if started_at is None:
            problems.append("started_at_utc")
        if completed_at is None:
            problems.append("completed_at_utc")
        elif started_at is not None and completed_at < started_at:
            problems.append("completed_at_utc")
        worker_seconds = record.get("worker_seconds")
        if (
            isinstance(worker_seconds, bool)
            or not isinstance(worker_seconds, (int, float))
            or not math.isfinite(worker_seconds)
            or worker_seconds < 0
        ):
            problems.append("worker_seconds")
        status = record.get("status")
        if status == "success":
            expected_run_dir = f"items/{item_id}"
            if record.get("run_dir") != expected_run_dir:
                problems.append("run_dir")
            expected_summary_path = f"{expected_run_dir}/summary.json"
            if record.get("summary_path") != expected_summary_path:
                problems.append("summary_path")
            expected_backend = (
                assignment.get("backend") if assignment is not None else None
            )
            expected_solver = (
                assignment.get("solver") if assignment is not None else None
            )
            if (
                expected_solver is not None
                and record.get("solver") != expected_solver
            ):
                problems.append("solver")
            for field in ("requested_backend", "backend"):
                if expected_backend is not None:
                    if record.get(field) != expected_backend:
                        problems.append(field)
                elif (
                    not isinstance(record.get(field), str)
                    or not record[field]
                ):
                    problems.append(field)
            if (
                not isinstance(record.get("device"), str)
                or not record["device"]
            ):
                problems.append("device")
            if not isinstance(record.get("runtime"), Mapping):
                problems.append("runtime")
            if not isinstance(record.get("timings_sec"), Mapping):
                problems.append("timings_sec")
            try:
                _validated_timings(
                    record.get("wall_sec"),
                    field=f"item {item_id!r} wall_sec",
                )
            except (TypeError, ValueError):
                problems.append("wall_sec")
            if success_record_validator is not None:
                problems.extend(success_record_validator(record))
        elif status == "failed":
            error = record.get("error")
            if not isinstance(error, Mapping) or not all(
                isinstance(error.get(field), str) and error[field]
                for field in ("type", "message")
            ):
                problems.append("error")
            if failed_record_validator is not None:
                problems.extend(failed_record_validator(record))
        else:
            problems.append("status")
        if problems:
            errors.append(
                {
                    "item_id": item_id,
                    "type": "InvalidTerminalRecord",
                    "message": (
                        f"record {index} has invalid field(s): "
                        + ", ".join(sorted(set(problems)))
                    ),
                }
            )
    return errors


def _audit_shard_results(
    shards: Sequence[Sequence[WorkItem]],
    shard_results: Sequence[Mapping[str, Any]],
    terminal_records: Sequence[Mapping[str, Any]],
    *,
    placements: Sequence[Placement],
    allow_loopback_alias: bool,
    backend: str,
    shard_schema: str = "cuphoton.xpois.dragon-shard/v1",
) -> dict[str, Any]:
    worker_count = len(shards)
    placement_by_worker = {
        placement.worker_id: placement for placement in placements
    }
    valid_results: list[tuple[int, Mapping[str, Any]]] = []
    invalid_results: list[dict[str, Any]] = []
    for index, result in enumerate(shard_results):
        worker_id = _strict_integer(result.get("worker_id"))
        if worker_id is None:
            invalid_results.append(
                {
                    "result_index": index,
                    "field": "worker_id",
                    "message": "worker_id must be a non-boolean integer",
                }
            )
            continue
        valid_results.append((worker_id, result))
    worker_ids = [worker_id for worker_id, _ in valid_results]
    expected = set(range(worker_count))
    observed = set(worker_ids)
    duplicates = sorted(
        worker_id for worker_id in observed if worker_ids.count(worker_id) > 1
    )
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    terminal_statuses: dict[str, list[Any]] = {}
    for record in terminal_records:
        item_id = record.get("item_id")
        if isinstance(item_id, str):
            terminal_statuses.setdefault(item_id, []).append(
                record.get("status")
            )
    mismatched: list[dict[str, Any]] = []
    write_failed: list[dict[str, Any]] = []
    physical_identities: list[
        tuple[int, str, frozenset[tuple[str, str]]]
    ] = []
    for worker_id, result in valid_results:
        if worker_id not in expected:
            continue
        shard = shards[worker_id]
        expected_fields = {
            "item_count": len(shard),
            "weight_bytes": sum(item.weight_bytes for item in shard),
            "item_ids_sha256": _item_ids_sha256(shard),
        }
        fields = {
            field
            for field in ("item_count", "weight_bytes")
            if (value := _strict_integer(result.get(field))) is None
            or value < 0
            or value != expected_fields[field]
        }
        if (
            result.get("item_ids_sha256")
            != expected_fields["item_ids_sha256"]
        ):
            fields.add("item_ids_sha256")
        if result.get("schema") != shard_schema:
            fields.add("schema")
        started_at = _parse_timezone_aware_timestamp(
            result.get("started_at_utc")
        )
        completed_at = _parse_timezone_aware_timestamp(
            result.get("completed_at_utc")
        )
        if started_at is None:
            fields.add("started_at_utc")
        if completed_at is None:
            fields.add("completed_at_utc")
        elif started_at is not None and completed_at < started_at:
            fields.add("completed_at_utc")
        worker_wall_sec = result.get("worker_wall_sec")
        if (
            isinstance(worker_wall_sec, bool)
            or not isinstance(worker_wall_sec, (int, float))
            or not math.isfinite(worker_wall_sec)
            or worker_wall_sec < 0
        ):
            fields.add("worker_wall_sec")
        status = result.get("status")
        if status not in {"success", "failed"}:
            fields.add("status")
        shard_error = result.get("error")
        if shard_error is not None and (
            not isinstance(shard_error, Mapping)
            or any(
                not isinstance(shard_error.get(field), str)
                or not shard_error[field]
                for field in ("type", "message")
            )
        ):
            fields.add("error")
        elif status == "success" and shard_error is not None:
            fields.add("error")
        success_count = _strict_integer(result.get("success_count"))
        failed_count = _strict_integer(result.get("failed_count"))
        if success_count is None or success_count < 0:
            fields.add("success_count")
        if failed_count is None or failed_count < 0:
            fields.add("failed_count")
        if (
            success_count is not None
            and failed_count is not None
            and success_count + failed_count != len(shard)
        ):
            fields.update({"success_count", "failed_count"})
        if success_count is not None and failed_count is not None:
            expected_status = "success" if failed_count == 0 else "failed"
            if status != expected_status:
                fields.add("status")
        record_write_errors = result.get("record_write_errors")
        write_error_ids: set[str] = set()
        write_errors_valid = isinstance(record_write_errors, list)
        if write_errors_valid:
            shard_item_ids = {item.item_id for item in shard}
            for error in record_write_errors:
                if (
                    not isinstance(error, Mapping)
                    or not isinstance(error.get("item_id"), str)
                    or error["item_id"] not in shard_item_ids
                    or error["item_id"] in write_error_ids
                    or not isinstance(error.get("type"), str)
                    or not error["type"]
                    or not isinstance(error.get("message"), str)
                ):
                    write_errors_valid = False
                    break
                write_error_ids.add(error["item_id"])
        if not write_errors_valid:
            fields.add("record_write_errors")
            write_error_ids.clear()
        elif write_error_ids:
            write_failed.append(
                {
                    "worker_id": worker_id,
                    "record_write_errors": record_write_errors,
                }
            )
        # Directory fsync can fail after a record becomes visible. The
        # worker counts that item as a failed write, not a durable outcome.
        terminal_success_count = sum(
            status == "success"
            for item in shard
            if item.item_id not in write_error_ids
            for status in terminal_statuses.get(item.item_id, ())
        )
        terminal_failed_count = len(write_error_ids) + sum(
            status == "failed"
            for item in shard
            if item.item_id not in write_error_ids
            for status in terminal_statuses.get(item.item_id, ())
        )
        if (
            success_count is not None
            and success_count != terminal_success_count
        ):
            fields.add("success_count")
        if failed_count is not None and failed_count != terminal_failed_count:
            fields.add("failed_count")
        placement = placement_by_worker.get(worker_id)
        provenance = result.get("provenance")
        if placement is None or not _valid_shard_provenance(
            provenance,
            placement=placement,
            allow_loopback_alias=allow_loopback_alias,
            backend=backend,
        ):
            fields.add("provenance")
        else:
            assert isinstance(provenance, Mapping)
            physical_identity = _stable_gpu_physical_ids(
                provenance.get("gpu")
            )
            assert physical_identity is not None
            physical_identities.append(
                (worker_id, placement.host, physical_identity)
            )
        try:
            _validated_timings(
                result.get("timings_sec"),
                field=f"worker {worker_id} timings_sec",
            )
        except (TypeError, ValueError):
            fields.add("timings_sec")
        if fields:
            mismatched.append(
                {"worker_id": worker_id, "fields": sorted(fields)}
            )
    duplicate_physical_gpu_worker_ids: set[int] = set()
    incomparable_physical_gpu_worker_ids: set[int] = set()
    for index, (worker_id, host, identity) in enumerate(physical_identities):
        remaining_identities = physical_identities[index + 1 :]
        for (
            other_worker_id,
            other_host,
            other_identity,
        ) in remaining_identities:
            if worker_id == other_worker_id:
                continue
            comparison = classify_physical_gpu_pair(
                dict(identity),
                dict(other_identity),
                same_host=_hostnames_match(host, other_host),
            )
            if comparison == "duplicate":
                duplicate_physical_gpu_worker_ids.update(
                    (worker_id, other_worker_id)
                )
            elif comparison == "incomparable":
                incomparable_physical_gpu_worker_ids.update(
                    (worker_id, other_worker_id)
                )
    invalid_physical_gpu_worker_ids = (
        duplicate_physical_gpu_worker_ids
        | incomparable_physical_gpu_worker_ids
    )
    marked_worker_ids: set[int] = set()
    for mismatch in mismatched:
        worker_id = mismatch["worker_id"]
        if worker_id in invalid_physical_gpu_worker_ids:
            mismatch["fields"] = sorted(
                set(mismatch["fields"]) | {"provenance"}
            )
            marked_worker_ids.add(worker_id)
    for worker_id in sorted(
        invalid_physical_gpu_worker_ids - marked_worker_ids
    ):
        mismatched.append({"worker_id": worker_id, "fields": ["provenance"]})
    return {
        "ok": not missing
        and not duplicates
        and not unexpected
        and not mismatched
        and not write_failed
        and not invalid_results,
        "expected_count": worker_count,
        "observed_count": len(shard_results),
        "missing_worker_ids": missing,
        "duplicate_worker_ids": duplicates,
        "unexpected_worker_ids": unexpected,
        "duplicate_physical_gpu_worker_ids": sorted(
            duplicate_physical_gpu_worker_ids
        ),
        "incomparable_physical_gpu_worker_ids": sorted(
            incomparable_physical_gpu_worker_ids
        ),
        "mismatched_shards": mismatched,
        "write_failed_shards": write_failed,
        "invalid_results": invalid_results,
    }


def _aggregate_item_timings(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    valid: list[dict[str, float]] = []
    errors: list[dict[str, str]] = []
    for record in records:
        item_id = str(record.get("item_id", ""))
        raw = record.get("timings_sec")
        if raw is None:
            continue
        try:
            valid.append(
                _validated_timings(raw, field=f"item {item_id!r} timings_sec")
            )
        except (TypeError, ValueError) as exc:
            errors.append(
                {
                    "item_id": item_id,
                    "type": "InvalidTimingRecord",
                    "message": str(exc),
                }
            )
    phases = sorted({phase for timings in valid for phase in timings})
    result: dict[str, dict[str, float]] = {}
    for phase in phases:
        values = [timings[phase] for timings in valid if phase in timings]
        result[phase] = {
            "sum": sum(values),
            "max": max(values, default=0.0),
            "mean": sum(values) / len(values) if values else 0.0,
        }
    return result, errors


def _validated_timings(value: Any, *, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    timings: dict[str, float] = {}
    for phase, elapsed in value.items():
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"{field} phase names must be non-empty strings")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(
                f"{field} phase {phase!r} must be a finite non-negative "
                "number"
            )
        timings[phase] = float(elapsed)
    return timings


def _parse_timezone_aware_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


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


def _shard_result_sort_key(result: Mapping[str, Any]) -> tuple[int, int]:
    worker_id = _strict_integer(result.get("worker_id"))
    return (worker_id is None, worker_id if worker_id is not None else 0)


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


__all__ = [
    "DragonBatchResult",
    "discover_gpu_placements",
    "run_dragon_image_pair_batch",
]
