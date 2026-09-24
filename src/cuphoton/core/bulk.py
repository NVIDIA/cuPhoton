# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Durable contracts for independent whole-item bulk work."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PCI_BUS_ID = re.compile(
    r"(?P<domain>[0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):"
    r"(?P<bus>[0-9A-Fa-f]{2}):(?P<device>[0-9A-Fa-f]{2})\."
    r"(?P<function>[0-7])"
)


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


def atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool = True
) -> None:
    """Atomically publish JSON; optionally require a new destination."""

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
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
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


def numba_pci_bus_id(device: Any) -> str:
    """Return a normalized PCI address from Numba CUDA device attributes."""

    components = (
        (device.PCI_DOMAIN_ID, "PCI domain", 0xFFFFFFFF, 8),
        (device.PCI_BUS_ID, "PCI bus", 0xFF, 2),
        (device.PCI_DEVICE_ID, "PCI device", 0x1F, 2),
    )
    normalized: list[str] = []
    for value, field, maximum, width in components:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{field} must be non-negative")
        if value > maximum:
            raise RuntimeError(f"{field} is outside the PCI address range")
        normalized.append(f"{value:0{width}X}")
    return normalize_pci_bus_id(
        f"{normalized[0]}:{normalized[1]}:{normalized[2]}.0"
    )


def normalize_pci_bus_id(value: Any) -> str:
    """Return one canonical CUDA/NVML PCI bus identifier."""

    if not isinstance(value, str) or not (
        match := _PCI_BUS_ID.fullmatch(value.strip())
    ):
        raise ValueError("PCI bus ID must use domain:bus:device.function")
    return (
        f"{int(match['domain'], 16):08X}:"
        f"{int(match['bus'], 16):02X}:"
        f"{int(match['device'], 16):02X}."
        f"{int(match['function'], 16):X}"
    )


def classify_physical_gpu_pair(
    first: Mapping[str, str],
    second: Mapping[str, str],
    *,
    same_host: bool,
) -> Literal["distinct", "duplicate", "incomparable"]:
    """Classify two stable GPU identities without conflating MIG devices."""

    first_uuid = first.get("uuid")
    second_uuid = second.get("uuid")
    if first_uuid is not None and second_uuid is not None:
        return "duplicate" if first_uuid == second_uuid else "distinct"
    if not same_host:
        return "distinct"
    first_pci = first.get("pci_bus_id")
    second_pci = second.get("pci_bus_id")
    if first_pci is None or second_pci is None:
        return "incomparable"
    return "duplicate" if first_pci == second_pci else "distinct"


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
    "classify_physical_gpu_pair",
    "error_payload",
    "json_mapping",
    "new_run_id",
    "normalize_pci_bus_id",
    "numba_pci_bus_id",
    "partition_byte_balanced",
    "read_json_mapping",
    "timestamp_utc",
    "validate_identifier",
]


def collect_gpu_identity(backend: str) -> dict[str, Any]:
    """Collect stable device identity after caller-established placement."""

    if backend in {"auto", "cupy", "cutile"}:
        try:
            return _collect_cupy_identity()
        except (ImportError, OSError, RuntimeError):
            if backend != "auto":
                raise
    return _collect_numba_identity()


def _collect_cupy_identity() -> dict[str, Any]:
    import cupy as cp

    device_index = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device_index)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    uuid_value = properties.get("uuid")
    if isinstance(uuid_value, bytes):
        uuid_value = uuid_value.hex()
    elif uuid_value is not None and not isinstance(uuid_value, str):
        uuid_value = str(uuid_value)
    identity_warnings: list[str] = []
    try:
        pci_bus_id = normalize_pci_bus_id(
            cp.cuda.runtime.deviceGetPCIBusId(device_index)
        )
    except Exception as exc:  # pragma: no cover - runtime-specific
        pci_bus_id = None
        identity_warnings.append(f"pci_bus_id: {type(exc).__name__}: {exc}")
    identity_error = (
        "; ".join(identity_warnings) or "no stable GPU identity was available"
        if uuid_value is None and pci_bus_id is None
        else None
    )
    return {
        "backend": "cupy",
        "device_index": device_index,
        "name": str(name),
        "uuid": uuid_value,
        "pci_bus_id": pci_bus_id,
        "identity_error": identity_error,
        "identity_warnings": identity_warnings,
    }


def _collect_numba_identity() -> dict[str, Any]:
    from numba import cuda

    device = cuda.get_current_device()
    name = getattr(device, "name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    identity_warnings: list[str] = []
    try:
        raw_uuid = device.uuid
        uuid_value = None if raw_uuid is None else str(raw_uuid) or None
    except Exception as exc:  # pragma: no cover - runtime-specific
        uuid_value = None
        identity_warnings.append(f"uuid: {type(exc).__name__}: {exc}")
    try:
        pci_bus_id = numba_pci_bus_id(device)
    except Exception as exc:  # pragma: no cover - runtime-specific
        pci_bus_id = None
        identity_warnings.append(f"pci_bus_id: {type(exc).__name__}: {exc}")
    identity_error = (
        "; ".join(identity_warnings) or "no stable GPU identity was available"
        if uuid_value is None and pci_bus_id is None
        else None
    )
    return {
        "backend": "numba-cuda",
        "device_index": int(getattr(device, "id", 0)),
        "name": str(name),
        "uuid": uuid_value,
        "pci_bus_id": pci_bus_id,
        "identity_error": identity_error,
        "identity_warnings": identity_warnings,
    }


def regular_file(path: Path) -> bool:
    """Return whether the path itself is a regular file, never a symlink."""

    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def item_ids_sha256(items: Sequence[WorkItem]) -> str:
    """Hash ordered work-item identities for shard provenance."""

    return hashlib.sha256(
        "\n".join(item.item_id for item in items).encode()
    ).hexdigest()


def hostnames_match(
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
