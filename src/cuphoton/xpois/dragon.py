# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional Dragon ProcessGroup adapter for XPOIS image pairs."""

from __future__ import annotations

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


def run_dragon_image_pair_batch(
    *,
    manifest_path: Path,
    output_root: Path,
    run_id: str | None,
    max_workers: int | None,
    result_timeout_sec: float,
    options: BatchFitOptions,
    worker_timeout_sec: float = 3600.0,
) -> DragonBatchResult:
    """Run deterministic XPOIS shards with one Dragon worker per GPU."""

    invocation_start = time.perf_counter()
    started_at = timestamp_utc()
    coordinator_timings: dict[str, float] = {}
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
    worker_count = min(
        requested_workers, len(placements), len(manifest.pairs)
    )
    if worker_count <= 0:
        raise RuntimeError("Dragon allocation exposes no usable GPUs")
    selected = _select_gpu_placements(placements, worker_count)
    distinct_host_count = len({placement.host for placement in selected})
    items = manifest.work_items()
    shards = partition_byte_balanced(items, worker_count)
    coordinator_timings["partition_sec"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    join_timeout_sec = float(worker_timeout_sec + result_timeout_sec)
    effective_run_id = run_id or new_run_id("dragon-xpois")
    validate_identifier(effective_run_id, field="run_id")
    run_dir = output_root.expanduser().resolve() / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("items", "records", "workers"):
        (run_dir / name).mkdir()
    atomic_write_json(run_dir / "manifest.json", manifest.canonical_payload())
    atomic_write_json(
        run_dir / "input-identity.json",
        manifest.input_identity_payload(),
    )
    dragon_version = _distribution_version("dragonhpc")
    atomic_write_json(
        run_dir / "run.json",
        {
            "schema": "cuphoton.xpois.dragon-run/v1",
            "record_type": "immutable-launch",
            "executor": "dragon",
            "run_id": effective_run_id,
            "started_at_utc": started_at,
            "manifest_sha256": manifest.sha256,
            "dragonhpc_version": dragon_version,
            "worker_count": worker_count,
            "allocation_node_count": allocation_node_count,
            "distinct_host_count": distinct_host_count,
            "worker_timeout_sec": worker_timeout_sec,
            "join_timeout_sec": join_timeout_sec,
            "result_timeout_sec": result_timeout_sec,
            "placements": [placement.to_dict() for placement in selected],
            "options": options.to_payload(),
        },
    )
    coordinator_timings["run_artifact_setup_sec"] = (
        time.perf_counter() - phase_start
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
            policy = api.Policy(
                placement=api.Policy.Placement.HOST_NAME,
                host_name=placement.host,
                gpu_affinity=[placement.gpu_id],
            )
            template = api.ProcessTemplate(
                target=_dragon_shard_worker,
                args=(
                    effective_run_id,
                    str(run_dir),
                    placement.to_dict(),
                    [item.to_dict() for item in shard],
                    options.to_payload(),
                    results_queue,
                    allocation_node_count == 1,
                ),
                policy=policy,
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
            "backend": options.backend,
        }
        for placement, shard in zip(selected, shards)
        for item in shard
    }
    record_errors.extend(
        _audit_terminal_record_contract(
            run_id=effective_run_id,
            expected=expected_records,
            records=records,
        )
    )
    shard_audit = _audit_shard_results(
        shards,
        shard_results,
        records,
        placements=selected,
        allow_loopback_alias=allocation_node_count == 1,
        backend=options.backend,
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
        "schema": "cuphoton.xpois.dragon-summary/v1",
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
        "manifest_sha256": manifest.sha256,
        "dragonhpc_version": dragon_version,
        "worker_timeout_sec": worker_timeout_sec,
        "join_timeout_sec": join_timeout_sec,
        "result_timeout_sec": result_timeout_sec,
        "allocation_node_count": allocation_node_count,
        "distinct_host_count": distinct_host_count,
        "options": options.to_payload(),
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
    options: BatchFitOptions,
    item_runner: Callable[
        [WorkItem, Path, BatchFitOptions], Mapping[str, Any]
    ],
    gpu_identity_loader: Callable[[str], Mapping[str, Any]],
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
    try:
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
            for name in ("cupy", "numba.cuda", "cuda.tile")
            if name in sys.modules
        )
        if require_clean_cuda_imports and premature:
            raise RuntimeError(
                "CUDA modules were imported before Dragon worker placement: "
                + ", ".join(premature)
            )
        gpu_identity = dict(gpu_identity_loader(options.backend))
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
            "schema": "cuphoton.xpois.dragon-item/v1",
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
        "schema": "cuphoton.xpois.dragon-shard/v1",
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
            records.append(read_json_mapping(path))
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
) -> list[dict[str, str]]:
    """Validate durable Dragon item records beyond identity/status counts."""

    errors: list[dict[str, str]] = []
    for index, record in enumerate(records):
        item_id_value = record.get("item_id")
        item_id = item_id_value if isinstance(item_id_value, str) else ""
        problems: list[str] = []
        if record.get("schema") != "cuphoton.xpois.dragon-item/v1":
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
        elif status == "failed":
            error = record.get("error")
            if not isinstance(error, Mapping) or not all(
                isinstance(error.get(field), str) and error[field]
                for field in ("type", "message")
            ):
                problems.append("error")
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
        if result.get("schema") != "cuphoton.xpois.dragon-shard/v1":
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
