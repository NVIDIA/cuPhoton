# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Whole-item execution contracts shared by optional distributed runtimes."""

from __future__ import annotations

import importlib
import json
import math
import os
import stat
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .bulk import (
    WorkItem,
    atomic_write_json,
    audit_terminal_records,
    classify_physical_gpu_pair,
    error_payload,
    item_ids_sha256,
    json_mapping,
    partition_byte_balanced,
    read_json_mapping,
    regular_file,
    timestamp_utc,
    validate_identifier,
)

ITEM_SCHEMA = "cuphoton.core.execution-item/v1"
WORKER_SCHEMA = "cuphoton.core.execution-worker/v1"
SUMMARY_SCHEMA = "cuphoton.core.execution-summary/v1"


class Worker(Protocol):
    """Worker-local state constructed after placement and CUDA binding."""

    gpu_identity: Mapping[str, Any]

    def run_item(self, item: WorkItem, output_dir: Path) -> Mapping[str, Any]:
        """Execute one item and durably publish its ordinary output files."""

    def close(self) -> None:
        """Complete pending work and release worker-local resources."""


WorkerFactory = Callable[[Mapping[str, Any]], Worker]
RecordValidator = Callable[[Mapping[str, Any], Path], Sequence[str]]
RoundFinalizer = Callable[
    [Path, Sequence[Mapping[str, Any]]], Mapping[str, Any]
]


@dataclass(frozen=True)
class WorkloadSpec:
    """Preflighted JSON descriptors and worker-local execution callbacks.

    Factories are package-importable. Scientific validators and finalizers run
    only on the coordinator; they are never sent to a worker.
    """

    items: Sequence[WorkItem]
    options_payload: Mapping[str, Any]
    manifest_payload: Mapping[str, Any]
    input_identity_payload: Mapping[str, Any]
    manifest_sha256: str
    backend: str
    worker_factory: WorkerFactory
    success_record_validator: RecordValidator | None = None
    failed_record_validator: RecordValidator | None = None
    finalize_round: RoundFinalizer | None = None

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items or not all(isinstance(item, WorkItem) for item in items):
            raise ValueError("workload requires non-empty WorkItems")
        partition_byte_balanced(items, 1)
        object.__setattr__(self, "items", items)
        for field in (
            "options_payload",
            "manifest_payload",
            "input_identity_payload",
        ):
            object.__setattr__(
                self, field, json_mapping(getattr(self, field), field=field)
            )
        if (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.manifest_sha256)
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256")
        if self.backend not in {"cupy", "cutile", "numba-cuda", "torch"}:
            raise ValueError("workload requires an explicit GPU backend")
        factory_reference(self.worker_factory)
        for field in (
            "success_record_validator",
            "failed_record_validator",
            "finalize_round",
        ):
            if getattr(self, field) is not None and not callable(
                getattr(self, field)
            ):
                raise TypeError(f"{field} must be callable or None")

    def identity_payload(self) -> dict[str, Any]:
        """Return launch data for agreement between ranks and provenance."""

        return {
            "items": [item.to_dict() for item in self.items],
            "options": dict(self.options_payload),
            "manifest": dict(self.manifest_payload),
            "input_identity": dict(self.input_identity_payload),
            "manifest_sha256": self.manifest_sha256,
            "backend": self.backend,
            "worker_factory": factory_reference(self.worker_factory),
        }


@dataclass(frozen=True)
class ExecutionResult:
    """Coordinator handle for a terminal invocation."""

    executor: str
    run_id: str
    run_dir: Path
    summary_path: Path
    status: str
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "executor": self.executor,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "summary_path": str(self.summary_path),
            "status": self.status,
        }


def factory_reference(factory: WorkerFactory) -> dict[str, str]:
    """Verify an import reference without serializing a live worker object."""

    reference = {
        "module": str(getattr(factory, "__module__", "")),
        "qualname": str(getattr(factory, "__qualname__", "")),
    }
    if resolve_worker_factory(reference) is not factory:
        raise ValueError(
            "worker factory import resolves to a different callable"
        )
    return reference


def resolve_worker_factory(reference: Mapping[str, Any]) -> WorkerFactory:
    if set(reference) != {"module", "qualname"}:
        raise ValueError("worker factory reference has invalid fields")
    module = reference["module"]
    qualname = reference["qualname"]
    if (
        not isinstance(module, str)
        or not module
        or module == "__main__"
        or not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
    ):
        raise ValueError("worker factory must be package-importable")
    result: Any = importlib.import_module(module)
    for part in qualname.split("."):
        if not part or part.startswith("<"):
            raise ValueError("worker factory qualname is invalid")
        result = getattr(result, part)
    if not callable(result):
        raise TypeError("worker factory is not callable")
    return result


def prepare_run(
    run_dir: Path,
    run_id: str,
    spec: WorkloadSpec,
    executor: str,
    *,
    directory_claimed: bool = False,
) -> None:
    """Claim a run directory and publish its immutable input descriptors."""

    validate_identifier(run_id, field="run_id")
    if directory_claimed:
        if not _real_directory(run_dir) or any(run_dir.iterdir()):
            raise ValueError(
                "claimed execution directory must be real and empty"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("items", "records", "workers"):
        (run_dir / name).mkdir()
    for name, payload in (
        ("manifest.json", spec.manifest_payload),
        ("input-identity.json", spec.input_identity_payload),
        (
            "run.json",
            {
                "schema": "cuphoton.core.execution-run/v1",
                "executor": executor,
                "run_id": run_id,
                "started_at_utc": timestamp_utc(),
                **spec.identity_payload(),
            },
        ),
    ):
        atomic_write_json(run_dir / name, payload, overwrite=False)


def execute_worker_round(
    worker: Worker,
    *,
    items: Sequence[WorkItem],
    run_id: str,
    run_dir: Path,
    manifest_sha256: str,
    worker_id: int,
    backend: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a placed worker's shard and retain every terminal item record.

    The caller measures around this function to include the final worker
    receipt's durable write. Receipts contain no worker objects or device
    values.
    """

    started = time.perf_counter()
    started_at = timestamp_utc()
    successes = failures = 0
    write_errors: list[dict[str, Any]] = []
    for item in items:
        item_started = time.perf_counter()
        identity = {
            "schema": ITEM_SCHEMA,
            "run_id": run_id,
            "manifest_sha256": manifest_sha256,
            "worker_id": worker_id,
            "item_id": item.item_id,
            "weight_bytes": item.weight_bytes,
            "requested_backend": backend,
            "started_at_utc": timestamp_utc(),
        }
        record = dict(identity)
        try:
            item_dir = run_dir / "items" / item.item_id
            metadata = json_mapping(
                worker.run_item(item, item_dir), field="item result metadata"
            )
            _validate_item_output(item_dir, metadata)
            for field in ("run_dir", "summary_path"):
                metadata[field] = str(
                    Path(metadata[field]).relative_to(run_dir)
                )
            record.update(metadata)
            record["status"] = "success"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = error_payload(exc)
            if getattr(exc, "__notes__", None):
                record["error"]["notes"] = "\n".join(map(str, exc.__notes__))
        record.update(identity)
        record.update(
            worker_seconds=time.perf_counter() - item_started,
            completed_at_utc=timestamp_utc(),
        )
        try:
            atomic_write_json(
                run_dir / "records" / f"{item.item_id}.json", record
            )
        except Exception as exc:
            write_errors.append(
                {"item_id": item.item_id, **error_payload(exc)}
            )
            failures += 1
        else:
            successes += record["status"] == "success"
            failures += record["status"] != "success"
    result = {
        "schema": WORKER_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "worker_id": worker_id,
        "assigned_item_ids": [item.item_id for item in items],
        "item_ids_sha256": item_ids_sha256(items),
        "weight_bytes": sum(item.weight_bytes for item in items),
        "success_count": successes,
        "failed_count": failures,
        "status": "failed" if failures else "success",
        "worker_wall_sec": time.perf_counter() - started,
        "started_at_utc": started_at,
        "completed_at_utc": timestamp_utc(),
        "provenance": json_mapping(provenance, field="worker provenance"),
        "record_write_errors": write_errors,
        "error": None,
    }
    try:
        atomic_write_json(
            run_dir / "workers" / f"worker-{worker_id:04d}.json", result
        )
    except Exception as exc:
        result["status"] = "failed"
        result["artifact_error"] = error_payload(exc)
    return result


def audit_worker_provenance(
    provenances: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    expected_worker_count: int,
) -> list[dict[str, Any]]:
    """Require one usable, non-overlapping physical GPU per placed worker."""

    errors: list[dict[str, Any]] = []
    identities: list[tuple[int, str, frozenset[tuple[str, str]]]] = []
    worker_ids = [
        p.get("worker_id") if isinstance(p, Mapping) else None
        for p in provenances
    ]
    if (
        len(provenances) != expected_worker_count
        or any(type(value) is not int for value in worker_ids)
        or set(worker_ids) != set(range(expected_worker_count))
    ):
        errors.append(
            {"phase": "provenance", "message": "worker identities differ"}
        )
    for index, value in enumerate(provenances):
        invalid = []
        if not isinstance(value, Mapping):
            errors.append(
                {
                    "phase": "provenance",
                    "worker_id": index,
                    "message": "invalid provenance",
                }
            )
            continue
        gpu = value.get("gpu")
        host = value.get("hostname")
        visibility = value.get("cuda_visible_devices")
        if type(value.get("pid")) is not int or value["pid"] <= 0:
            invalid.append("pid")
        if not isinstance(host, str) or not host:
            invalid.append("hostname")
        tokens = visibility.split(",") if isinstance(visibility, str) else []
        if (
            len(tokens) != 1
            or not tokens[0].strip()
            or tokens[0].strip().startswith("-")
        ):
            invalid.append("cuda_visible_devices")
        ids = frozenset()
        expected_backend = "cupy" if backend == "cutile" else backend
        if not isinstance(gpu, Mapping):
            invalid.append("gpu")
        else:
            if (
                gpu.get("backend") != expected_backend
                or type(gpu.get("device_index")) is not int
                or gpu["device_index"] != 0
                or "identity_error" not in gpu
                or gpu["identity_error"] is not None
            ):
                invalid.append("gpu")
            ids = frozenset(
                (field, identifier.strip().casefold())
                for field in ("uuid", "pci_bus_id")
                if isinstance((identifier := gpu.get(field)), str)
                and identifier.strip()
            )
            if not ids:
                invalid.append("physical GPU identity")
        if invalid:
            errors.append(
                {
                    "phase": "provenance",
                    "worker_id": value.get("worker_id"),
                    "message": ", ".join(invalid),
                }
            )
        else:
            identities.append((value["worker_id"], host, ids))
    for index, (worker_id, host, ids) in enumerate(identities):
        for other_id, other_host, other_ids in identities[:index]:
            same_host = (
                host.rstrip(".").casefold().split(".")[0]
                == other_host.rstrip(".").casefold().split(".")[0]
            )
            relation = classify_physical_gpu_pair(
                dict(ids), dict(other_ids), same_host=same_host
            )
            if relation != "distinct":
                errors.append(
                    {
                        "phase": "provenance",
                        "worker_ids": [other_id, worker_id],
                        "message": f"{relation} physical GPU identities",
                    }
                )
    return errors


def finalize_round(
    run_dir: Path,
    run_id: str,
    spec: WorkloadSpec,
    shards: Sequence[Sequence[WorkItem]],
    worker_results: Sequence[Mapping[str, Any]],
    *,
    artifact_timeout_sec: float,
) -> dict[str, Any]:
    """Audit durable execution evidence, then run a component's merge step."""

    if (
        not _finite_nonnegative(artifact_timeout_sec)
        or artifact_timeout_sec <= 0
    ):
        raise ValueError("artifact_timeout_sec must be positive")
    errors, records = _read_round_artifacts(
        run_dir, shards, worker_results, artifact_timeout_sec
    )
    expected_items = [item.item_id for shard in shards for item in shard]
    item_audit = audit_terminal_records(
        expected_items,
        [
            {
                **record,
                "status": record.get("status")
                if isinstance(record.get("status"), str)
                else None,
            }
            for record in records
        ],
    )
    if not item_audit["ok"] or item_audit["failed_item_ids"]:
        errors.append(
            {"phase": "items", "message": "terminal item audit failed"}
        )
    expected = {
        item.item_id: (worker_id, item.weight_bytes)
        for worker_id, shard in enumerate(shards)
        for item in shard
    }
    for record in records:
        fields = _record_problems(record, run_dir, run_id, spec, expected)
        validator = (
            spec.success_record_validator
            if record.get("status") == "success"
            else spec.failed_record_validator
        )
        if validator is not None:
            try:
                problems = validator(record, run_dir)
                if (
                    isinstance(problems, (str, bytes))
                    or not isinstance(problems, Sequence)
                    or any(
                        not isinstance(problem, str) for problem in problems
                    )
                ):
                    raise TypeError(
                        "component validator must return field names"
                    )
                fields.extend(problems)
            except Exception as exc:
                fields.append(
                    f"component validator: {type(exc).__name__}: {exc}"
                )
        if fields:
            errors.append(
                {
                    "phase": "records",
                    "item_id": record.get("item_id"),
                    "message": ", ".join(sorted(set(fields))),
                }
            )
    errors.extend(
        _worker_result_problems(run_id, spec, shards, worker_results, records)
    )
    errors.extend(
        audit_worker_provenance(
            [
                result.get("provenance")
                for result in worker_results
                if isinstance(result, Mapping)
            ],
            backend=spec.backend,
            expected_worker_count=len(shards),
        )
    )
    scientific_result = None
    if not errors and spec.finalize_round is not None:
        try:
            by_id = {record["item_id"]: record for record in records}
            scientific_result = json_mapping(
                spec.finalize_round(
                    run_dir, [by_id[item.item_id] for item in spec.items]
                ),
                field="component round finalization",
            )
        except Exception as exc:
            errors.append({"phase": "finalization", **error_payload(exc)})
    summary = {
        "schema": SUMMARY_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": spec.manifest_sha256,
        "backend": spec.backend,
        "status": "failed" if errors else "success",
        "completed_at_utc": timestamp_utc(),
        "terminal_record_audit": item_audit,
        "records": records,
        "worker_results": [dict(result) for result in worker_results],
        "errors": errors,
        "result": scientific_result,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary


def _read_round_artifacts(
    run_dir: Path,
    shards: Sequence[Sequence[WorkItem]],
    results: Sequence[Mapping[str, Any]],
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {
        f"records/{item.item_id}.json" for shard in shards for item in shard
    } | {
        f"workers/worker-{worker_id:04d}.json"
        for worker_id in range(len(shards))
    }
    errors: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, Mapping):
            continue
        worker_id = result.get("worker_id")
        if type(worker_id) is not int or worker_id not in range(len(shards)):
            continue
        if result.get("artifact_error") is not None:
            expected.discard(f"workers/worker-{worker_id:04d}.json")
            errors.append(
                {
                    "phase": "publication",
                    "worker_id": worker_id,
                    "error": result["artifact_error"],
                }
            )
        writes = result.get("record_write_errors")
        if isinstance(writes, list):
            assigned = {item.item_id for item in shards[worker_id]}
            for error in writes:
                if (
                    isinstance(error, Mapping)
                    and error.get("item_id") in assigned
                ):
                    expected.discard(f"records/{error['item_id']}.json")
                    errors.append({"phase": "publication", **dict(error)})
    deadline = time.monotonic() + timeout
    mappings: dict[str, dict[str, Any]] = {}
    while True:
        pending = []
        for label in sorted(expected):
            try:
                path = run_dir / label
                if not regular_file(path):
                    if os.path.lexists(path):
                        raise ValueError("artifact is not a regular file")
                    raise FileNotFoundError(label)
                mapping = json_mapping(read_json_mapping(path), field=label)
                if (
                    label.startswith("records/")
                    and mapping.get("item_id") != Path(label).stem
                ):
                    raise ValueError(
                        "terminal item identity differs from record path"
                    )
                mappings[label] = mapping
                if (
                    label.startswith("records/")
                    and mapping.get("status") == "success"
                ):
                    item_id = Path(label).stem
                    item_dir = run_dir / "items" / item_id
                    if not _real_directory(item_dir) or not regular_file(
                        item_dir / "summary.json"
                    ):
                        if os.path.lexists(item_dir / "summary.json"):
                            raise ValueError("item output is not regular")
                        raise FileNotFoundError(
                            f"items/{item_id}/summary.json"
                        )
            except OSError as exc:
                pending.append(
                    {
                        "phase": "visibility",
                        "path": label,
                        **error_payload(exc),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "phase": "artifacts",
                        "path": label,
                        **error_payload(exc),
                    }
                )
        if not pending or errors or time.monotonic() >= deadline:
            errors.extend(pending)
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    expected_records = {
        f"{item.item_id}.json" for shard in shards for item in shard
    }
    observed_records = {
        path.name for path in (run_dir / "records").glob("*.json")
    }
    if observed_records - expected_records:
        errors.append(
            {"phase": "artifacts", "message": "unexpected terminal records"}
        )
    for result in results:
        if (
            not isinstance(result, Mapping)
            or type(result.get("worker_id")) is not int
        ):
            continue
        label = f"workers/worker-{result['worker_id']:04d}.json"
        if label in mappings and not _same_json(mappings[label], result):
            errors.append(
                {
                    "phase": "artifacts",
                    "path": label,
                    "message": "durable worker receipt differs from "
                    "collected receipt",
                }
            )
    return errors, [
        mapping
        for label, mapping in sorted(mappings.items())
        if label.startswith("records/")
    ]


def _record_problems(
    record: Mapping[str, Any],
    run_dir: Path,
    run_id: str,
    spec: WorkloadSpec,
    assignments: Mapping[str, tuple[int, int]],
) -> list[str]:
    problems = []
    item_id = record.get("item_id")
    assignment = (
        assignments.get(item_id) if isinstance(item_id, str) else None
    )
    for field, expected in (
        ("schema", ITEM_SCHEMA),
        ("run_id", run_id),
        ("manifest_sha256", spec.manifest_sha256),
        ("requested_backend", spec.backend),
    ):
        if record.get(field) != expected:
            problems.append(field)
    if assignment is None:
        problems.append("item_id")
    else:
        for field, value in zip(("worker_id", "weight_bytes"), assignment):
            if type(record.get(field)) is not int or record[field] != value:
                problems.append(field)
    if not _finite_nonnegative(record.get("worker_seconds")):
        problems.append("worker_seconds")
    if not _valid_timestamps(record):
        problems.append("timestamps")
    if record.get("status") == "success":
        if record.get("backend") != spec.backend:
            problems.append("backend")
        if record.get("run_dir") != f"items/{item_id}":
            problems.append("run_dir")
        if record.get("summary_path") != f"items/{item_id}/summary.json":
            problems.append("summary_path")
        if not isinstance(record.get("device"), str) or not record["device"]:
            problems.append("device")
        if not isinstance(record.get("runtime"), Mapping):
            problems.append("runtime")
        for field in ("timings_sec", "wall_sec"):
            if not _valid_timings(record.get(field)):
                problems.append(field)
        if record.get("error") is not None:
            problems.append("error")
    elif record.get("status") == "failed":
        if not _valid_error(record.get("error")):
            problems.append("error")
    else:
        problems.append("status")
    return problems


def _worker_result_problems(
    run_id: str,
    spec: WorkloadSpec,
    shards: Sequence[Sequence[WorkItem]],
    results: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    errors = []
    identifiers = [
        value.get("worker_id") if isinstance(value, Mapping) else None
        for value in results
    ]
    if (
        len(results) != len(shards)
        or any(type(value) is not int for value in identifiers)
        or set(identifiers) != set(range(len(shards)))
    ):
        errors.append(
            {
                "phase": "workers",
                "message": "worker receipt identities differ",
            }
        )
    for result in results:
        if not isinstance(result, Mapping):
            errors.append(
                {"phase": "workers", "message": "invalid worker receipt"}
            )
            continue
        worker_id = result.get("worker_id")
        if type(worker_id) is not int or worker_id not in range(len(shards)):
            continue
        shard = shards[worker_id]
        assigned_records = [
            r
            for r in records
            if type(r.get("worker_id")) is int and r["worker_id"] == worker_id
        ]
        counts = Counter(
            r.get("status") if isinstance(r.get("status"), str) else None
            for r in assigned_records
        )
        writes = result.get("record_write_errors")
        valid_writes = isinstance(writes, list) and all(
            isinstance(error, Mapping)
            and _valid_error(error)
            and error.get("item_id") in {item.item_id for item in shard}
            for error in writes
        )
        failed_count = counts["failed"] + (len(writes) if valid_writes else 0)
        checks = {
            "schema": result.get("schema") == WORKER_SCHEMA,
            "run_id": result.get("run_id") == run_id,
            "manifest_sha256": result.get("manifest_sha256")
            == spec.manifest_sha256,
            "assigned_item_ids": result.get("assigned_item_ids")
            == [item.item_id for item in shard],
            "item_ids_sha256": result.get("item_ids_sha256")
            == item_ids_sha256(shard),
            "weight_bytes": type(result.get("weight_bytes")) is int
            and result["weight_bytes"]
            == sum(item.weight_bytes for item in shard),
            "success_count": type(result.get("success_count")) is int
            and result["success_count"] == counts["success"],
            "failed_count": type(result.get("failed_count")) is int
            and result["failed_count"] == failed_count,
            "terminal_count": counts["success"] + failed_count == len(shard),
            "record_write_errors": valid_writes and not writes,
            "artifact_error": result.get("artifact_error") is None,
            "error": result.get("error") is None,
            "status": result.get("status") == "success" and not failed_count,
            "worker_wall_sec": _finite_nonnegative(
                result.get("worker_wall_sec")
            ),
            "provenance_worker_id": isinstance(
                result.get("provenance"), Mapping
            )
            and type(result["provenance"].get("worker_id")) is int
            and result["provenance"]["worker_id"] == worker_id,
            "timestamps": _valid_timestamps(result),
        }
        invalid = [key for key, ok in checks.items() if not ok]
        if invalid:
            errors.append(
                {
                    "phase": "workers",
                    "worker_id": worker_id,
                    "message": ", ".join(invalid),
                }
            )
    return errors


def _validate_item_output(
    item_dir: Path, metadata: Mapping[str, Any]
) -> None:
    if not _real_directory(item_dir):
        raise ValueError("item runner did not create a real output directory")
    if Path(metadata.get("run_dir", "")) != item_dir:
        raise ValueError("item runner reported a different output directory")
    if Path(metadata.get("summary_path", "")) != item_dir / "summary.json":
        raise ValueError("item runner reported a different summary path")
    if not regular_file(item_dir / "summary.json"):
        raise ValueError("item runner did not create a regular summary file")


def _real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _valid_timings(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and key and _finite_nonnegative(elapsed)
        for key, elapsed in value.items()
    )


def _valid_error(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(value.get(key), str) and value[key]
        for key in ("type", "message")
    )


def _valid_timestamps(value: Mapping[str, Any]) -> bool:
    try:
        started = datetime.fromisoformat(value["started_at_utc"])
        completed = datetime.fromisoformat(value["completed_at_utc"])
        return (
            started.utcoffset() is not None
            and completed.utcoffset() is not None
            and started <= completed
        )
    except (KeyError, TypeError, ValueError):
        return False


def _same_json(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            left, sort_keys=True, allow_nan=False
        ) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False
