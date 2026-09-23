# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from cuphoton.core.bulk import (
    Placement,
    WorkItem,
    atomic_write_json,
    audit_terminal_records,
    classify_physical_gpu_pair,
    collect_gpu_identity,
    normalize_pci_bus_id,
    numba_pci_bus_id,
    partition_byte_balanced,
)


def _item(item_id: str, weight: int) -> WorkItem:
    return WorkItem(item_id, {"path": f"/{item_id}"}, weight)


def test_byte_balanced_partition_is_deterministic() -> None:
    items = (
        _item("small", 1),
        _item("large-b", 8),
        _item("medium", 6),
        _item("large-a", 8),
    )

    first = partition_byte_balanced(items, 2)
    second = partition_byte_balanced(tuple(reversed(items)), 2)

    assert first == second
    assert [[item.item_id for item in shard] for shard in first] == [
        ["large-a", "medium"],
        ["large-b", "small"],
    ]


def test_byte_balanced_partition_distributes_zero_weight_items() -> None:
    shards = partition_byte_balanced(
        tuple(_item(f"item-{index}", 0) for index in range(7)),
        3,
    )

    assert [[item.item_id for item in shard] for shard in shards] == [
        ["item-0", "item-3", "item-6"],
        ["item-1", "item-4"],
        ["item-2", "item-5"],
    ]


def test_byte_balanced_partition_distributes_equal_weight_items() -> None:
    shards = partition_byte_balanced(
        tuple(_item(f"item-{index}", 5) for index in range(6)),
        3,
    )

    assert [[item.item_id for item in shard] for shard in shards] == [
        ["item-0", "item-3"],
        ["item-1", "item-4"],
        ["item-2", "item-5"],
    ]


def test_partition_rejects_duplicate_item_ids() -> None:
    with pytest.raises(ValueError, match="duplicate work item IDs"):
        partition_byte_balanced((_item("same", 1), _item("same", 2)), 2)


@pytest.mark.parametrize("worker_count", [True, 2.0, "2"])
def test_partition_rejects_non_integer_worker_count(worker_count) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        partition_byte_balanced((_item("one", 1),), worker_count)


@pytest.mark.parametrize("worker_id", [False, 1.0, "1"])
def test_placement_rejects_non_integer_worker_id(worker_id) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        Placement(worker_id=worker_id, host="node", gpu_id=0)


def test_terminal_record_audit_reports_all_identity_failures() -> None:
    audit = audit_terminal_records(
        ["a", "b", "c"],
        [
            {"item_id": "a", "status": "success"},
            {"item_id": "a", "status": "failed"},
            {"item_id": "outside", "status": "unknown"},
        ],
    )

    assert audit == {
        "ok": False,
        "expected_count": 3,
        "terminal_record_count": 3,
        "missing_item_ids": ["b", "c"],
        "duplicate_item_ids": ["a"],
        "unexpected_item_ids": ["outside"],
        "invalid_status_item_ids": ["outside"],
        "failed_item_ids": ["a"],
    }


def test_terminal_audit_handles_failed_record_without_item_id() -> None:
    audit = audit_terminal_records(["expected"], [{"status": "failed"}])

    assert audit["ok"] is False
    assert audit["missing_item_ids"] == ["expected"]
    assert audit["unexpected_item_ids"] == [""]
    assert audit["failed_item_ids"] == []


def test_atomic_write_json_replaces_complete_mapping(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "nested" / "record.json"
    fsynced = []
    original_fsync = os.fsync

    def record_fsync(fd):
        fsynced.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)

    atomic_write_json(path, {"status": "starting"})
    atomic_write_json(path, {"status": "success", "count": 2})

    assert json.loads(path.read_text()) == {"status": "success", "count": 2}
    assert list(path.parent.glob(".*.tmp-*")) == []
    assert len(fsynced) == 4


def test_work_item_rejects_non_json_payload() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        WorkItem("item", {"bad": object()})


@pytest.mark.parametrize("symlink", [False, True])
def test_atomic_write_json_can_preserve_existing_destination(
    tmp_path, symlink
):
    path = tmp_path / "record.json"
    if symlink:
        target = tmp_path / "target.json"
        atomic_write_json(target, {"owner": "first"})
        path.symlink_to(target)
    else:
        atomic_write_json(path, {"owner": "first"}, overwrite=False)
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"owner": "second"}, overwrite=False)
    assert json.loads(path.read_text()) == {"owner": "first"}
    assert path.is_symlink() is symlink
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_numba_pci_bus_id_normalizes_cuda_attributes() -> None:
    device = SimpleNamespace(
        PCI_DOMAIN_ID=0,
        PCI_BUS_ID=0xC1,
        PCI_DEVICE_ID=0,
    )

    assert numba_pci_bus_id(device) == "00000000:C1:00.0"


def test_numba_pci_bus_id_accepts_extended_domain() -> None:
    device = SimpleNamespace(
        PCI_DOMAIN_ID=0x10000,
        PCI_BUS_ID=0xC1,
        PCI_DEVICE_ID=0,
    )

    assert numba_pci_bus_id(device) == "00010000:C1:00.0"


@pytest.mark.parametrize("value", ["0000:c1:00.0", "00000000:C1:00.0"])
def test_normalize_pci_bus_id_accepts_cuda_and_nvml_forms(value: str) -> None:
    assert normalize_pci_bus_id(value) == "00000000:C1:00.0"


@pytest.mark.parametrize("value", [None, "c1:00.0", "0000:GG:00.0"])
def test_normalize_pci_bus_id_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="domain:bus:device.function"):
        normalize_pci_bus_id(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("PCI_DOMAIN_ID", True, "PCI domain must be non-negative"),
        ("PCI_DOMAIN_ID", 0x100000000, "PCI domain is outside"),
        ("PCI_BUS_ID", -1, "PCI bus must be non-negative"),
        ("PCI_DEVICE_ID", 0x20, "PCI device is outside"),
    ],
)
def test_numba_pci_bus_id_rejects_invalid_components(
    field, value, message
) -> None:
    values = {
        "PCI_DOMAIN_ID": 0,
        "PCI_BUS_ID": 1,
        "PCI_DEVICE_ID": 0,
    }
    values[field] = value

    with pytest.raises(RuntimeError, match=message):
        numba_pci_bus_id(SimpleNamespace(**values))


@pytest.mark.parametrize(
    ("first", "second", "same_host", "expected"),
    [
        (
            {"uuid": "mig-a", "pci_bus_id": "pci"},
            {"uuid": "mig-b", "pci_bus_id": "pci"},
            True,
            "distinct",
        ),
        (
            {"uuid": "gpu", "pci_bus_id": "pci-a"},
            {"uuid": "gpu", "pci_bus_id": "pci-b"},
            False,
            "duplicate",
        ),
        (
            {"pci_bus_id": "pci"},
            {"uuid": "gpu", "pci_bus_id": "pci"},
            True,
            "duplicate",
        ),
        (
            {"uuid": "gpu"},
            {"pci_bus_id": "pci"},
            True,
            "incomparable",
        ),
        (
            {"uuid": "gpu"},
            {"pci_bus_id": "pci"},
            False,
            "distinct",
        ),
    ],
)
def test_classify_physical_gpu_pair(
    first, second, same_host, expected
) -> None:
    assert (
        classify_physical_gpu_pair(
            first,
            second,
            same_host=same_host,
        )
        == expected
    )


def test_numba_gpu_identity_uses_normalized_pci_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(
        id=0,
        name=b"test GPU",
        uuid=None,
        PCI_DOMAIN_ID=0,
        PCI_BUS_ID=0xC1,
        PCI_DEVICE_ID=0,
    )
    fake_cuda = SimpleNamespace(get_current_device=lambda: device)
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(cuda=fake_cuda))

    identity = collect_gpu_identity("numba-cuda")

    assert identity == {
        "backend": "numba-cuda",
        "device_index": 0,
        "name": "test GPU",
        "uuid": None,
        "pci_bus_id": "00000000:C1:00.0",
        "identity_error": None,
        "identity_warnings": [],
    }


@pytest.mark.parametrize("broken_field", ["uuid", "pci_bus_id"])
def test_numba_gpu_identity_preserves_partial_identity(
    monkeypatch: pytest.MonkeyPatch, broken_field: str
) -> None:
    class PartialIdentity:
        id = 0
        name = b"test GPU"
        PCI_BUS_ID = 1
        PCI_DEVICE_ID = 0

        @property
        def uuid(self):
            if broken_field == "uuid":
                raise RuntimeError("uuid unavailable")
            return "GPU-one"

        @property
        def PCI_DOMAIN_ID(self):
            if broken_field == "pci_bus_id":
                raise RuntimeError("pci unavailable")
            return 0

    fake_cuda = SimpleNamespace(get_current_device=PartialIdentity)
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(cuda=fake_cuda))

    identity = collect_gpu_identity("numba-cuda")

    assert identity["identity_error"] is None
    assert len(identity["identity_warnings"]) == 1
    assert identity["identity_warnings"][0].startswith(broken_field)
    assert identity["uuid"] is not None or identity["pci_bus_id"] is not None


def test_numba_gpu_identity_fails_without_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenIdentity:
        id = 0
        name = b"test GPU"

        @property
        def uuid(self):
            raise RuntimeError("uuid unavailable")

        @property
        def PCI_DOMAIN_ID(self):
            raise RuntimeError("pci unavailable")

    fake_cuda = SimpleNamespace(get_current_device=BrokenIdentity)
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(cuda=fake_cuda))

    identity = collect_gpu_identity("numba-cuda")

    assert identity["uuid"] is None
    assert identity["pci_bus_id"] is None
    assert identity["identity_error"] == "; ".join(
        identity["identity_warnings"]
    )


def test_cupy_gpu_identity_uses_normalized_pci_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {
            "name": b"test GPU",
            "uuid": "GPU-one",
        },
        deviceGetPCIBusId=lambda index: "0000:c1:00.0",
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)),
    )

    identity = collect_gpu_identity("cupy")

    assert identity["pci_bus_id"] == "00000000:C1:00.0"
    assert identity["identity_warnings"] == []


def test_cupy_gpu_identity_preserves_uuid_on_pci_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(index: int) -> str:
        raise RuntimeError("pci unavailable")

    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {
            "name": b"test GPU",
            "uuid": "GPU-one",
        },
        deviceGetPCIBusId=unavailable,
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)),
    )

    identity = collect_gpu_identity("cupy")

    assert identity["uuid"] == "GPU-one"
    assert identity["pci_bus_id"] is None
    assert identity["identity_error"] is None
    assert identity["identity_warnings"] == [
        "pci_bus_id: RuntimeError: pci unavailable"
    ]
