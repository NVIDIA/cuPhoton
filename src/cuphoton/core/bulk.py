# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Durable contracts for independent whole-item bulk work."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class WorkItem:
    """One independent, JSON-described unit of work."""

    item_id: str
    payload: Mapping[str, Any]
    weight_bytes: int = 0

    def __post_init__(self) -> None:
        validate_identifier(self.item_id, field="item_id")
        if (
            isinstance(self.weight_bytes, bool)
            or not isinstance(self.weight_bytes, int)
            or self.weight_bytes < 0
        ):
            raise ValueError("weight_bytes must be a non-negative integer")
        object.__setattr__(
            self,
            "payload",
            json_mapping(self.payload, field="work item payload"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible wire representation."""

        return {
            "item_id": self.item_id,
            "payload": dict(self.payload),
            "weight_bytes": self.weight_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkItem:
        """Restore an item from its wire representation."""

        return cls(
            item_id=str(payload["item_id"]),
            payload=payload["payload"],
            weight_bytes=payload.get("weight_bytes", 0),
        )


@dataclass(frozen=True)
class Placement:
    """One explicitly selected Dragon host/GPU worker placement."""

    worker_id: int
    host: str
    gpu_id: int

    def __post_init__(self) -> None:
        if isinstance(self.worker_id, bool) or not isinstance(
            self.worker_id, int
        ):
            raise TypeError("worker_id must be an integer")
        if self.worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        if not self.host:
            raise ValueError("placement host must not be empty")
        if isinstance(self.gpu_id, bool) or not isinstance(self.gpu_id, int):
            raise TypeError("Dragon GPU IDs must be integers")
        if self.gpu_id < 0:
            raise ValueError("Dragon GPU IDs must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible placement record."""

        return asdict(self)


def partition_byte_balanced(
    items: Sequence[WorkItem], worker_count: int
) -> tuple[tuple[WorkItem, ...], ...]:
    """Deterministically assign whole items by descending input bytes."""

    if isinstance(worker_count, bool) or not isinstance(worker_count, int):
        raise TypeError("worker_count must be an integer")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    _require_unique_item_ids(items)
    shards: list[list[WorkItem]] = [[] for _ in range(worker_count)]
    weights = [0] * worker_count
    for item in sorted(
        items,
        key=lambda value: (-value.weight_bytes, value.item_id),
    ):
        worker_id = min(
            range(worker_count),
            key=lambda value: (
                weights[value],
                len(shards[value]),
                value,
            ),
        )
        shards[worker_id].append(item)
        weights[worker_id] += item.weight_bytes
    return tuple(tuple(shard) for shard in shards)


def audit_terminal_records(
    expected_item_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit exactly one valid terminal record for every expected item."""

    expected = list(expected_item_ids)
    expected_counts = Counter(expected)
    if any(count != 1 for count in expected_counts.values()):
        raise ValueError("expected item IDs must be unique")
    observed = [str(record.get("item_id", "")) for record in records]
    observed_counts = Counter(observed)
    expected_set = set(expected)
    missing = sorted(expected_set - set(observed_counts))
    duplicates = sorted(
        item_id for item_id, count in observed_counts.items() if count > 1
    )
    unexpected = sorted(set(observed_counts) - expected_set)
    invalid_status = sorted(
        str(record.get("item_id", ""))
        for record in records
        if record.get("status") not in {"success", "failed"}
    )
    failed = sorted(
        str(record.get("item_id", ""))
        for record in records
        if record.get("status") == "failed"
        and str(record.get("item_id", "")) in expected_set
    )
    ok = (
        not missing
        and not duplicates
        and not unexpected
        and not invalid_status
    )
    return {
        "ok": ok,
        "expected_count": len(expected),
        "terminal_record_count": len(records),
        "missing_item_ids": missing,
        "duplicate_item_ids": duplicates,
        "unexpected_item_ids": unexpected,
        "invalid_status_item_ids": invalid_status,
        "failed_item_ids": failed,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one JSON mapping with a trailing newline."""

    normalized = json_mapping(payload, field=f"payload for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(normalized, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject other JSON value types."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def json_mapping(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Round-trip one mapping through strict JSON."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        encoded = json.dumps(dict(payload), sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-compatible: {exc}") from exc
    if not isinstance(decoded, dict):
        raise TypeError(f"{field} must encode a JSON object")
    return decoded


def validate_identifier(value: str, *, field: str) -> str:
    """Validate a filesystem-safe item or run identity."""

    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must match {_IDENTIFIER.pattern!r}")
    if value in {".", ".."}:
        raise ValueError(f"{field} cannot be {value!r}")
    return value


def new_run_id(prefix: str) -> str:
    """Create a collision-resistant UTC run identity."""

    validate_identifier(prefix, field="run prefix")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def timestamp_utc() -> str:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def error_payload(exc: BaseException) -> dict[str, str]:
    """Return a compact structured exception description."""

    return {"type": type(exc).__name__, "message": str(exc)}


def _require_unique_item_ids(items: Sequence[WorkItem]) -> None:
    counts = Counter(item.item_id for item in items)
    duplicates = sorted(
        item_id for item_id, count in counts.items() if count > 1
    )
    if duplicates:
        raise ValueError("duplicate work item IDs: " + ", ".join(duplicates))


__all__ = [
    "Placement",
    "WorkItem",
    "atomic_write_json",
    "audit_terminal_records",
    "error_payload",
    "json_mapping",
    "new_run_id",
    "partition_byte_balanced",
    "read_json_mapping",
    "timestamp_utc",
    "validate_identifier",
]
