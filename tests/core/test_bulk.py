# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import pytest

from cuphoton.core.bulk import (
    Placement,
    WorkItem,
    atomic_write_json,
    audit_terminal_records,
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
