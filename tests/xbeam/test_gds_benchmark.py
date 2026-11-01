# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from examples.xbeam.benchmark_gds import (
    _ALIGNMENT,
    _assign_tasks,
    _discover_vdif_files,
    _parse_devices,
    _read_tasks,
)


def test_discovers_vdif_files_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "first.vdif"
    second = nested / "second.vdif"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    (nested / "ignored.txt").write_bytes(b"c")

    assert _discover_vdif_files([tmp_path]) == (
        first.resolve(),
        second.resolve(),
    )


def test_read_tasks_cover_logical_bytes_with_aligned_requests(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.vdif"
    source.write_bytes(bytes(2 * _ALIGNMENT + 17))

    tasks = _read_tasks([source], 2 * _ALIGNMENT)

    assert [
        (task.offset, task.logical_bytes, task.request_bytes)
        for task in tasks
    ] == [
        (0, 2 * _ALIGNMENT, 2 * _ALIGNMENT),
        (2 * _ALIGNMENT, 17, _ALIGNMENT),
    ]


def test_assign_tasks_preserves_every_task_once(tmp_path: Path) -> None:
    source = tmp_path / "source.vdif"
    source.write_bytes(bytes(5 * _ALIGNMENT))
    tasks = _read_tasks([source], _ALIGNMENT)

    assignments = _assign_tasks(tasks, 3)

    assert tuple(len(items) for items in assignments) == (2, 2, 1)
    assert sorted(task.offset for items in assignments for task in items) == [
        task.offset for task in tasks
    ]


def test_parse_devices_rejects_duplicates() -> None:
    with pytest.raises(Exception, match="unique"):
        _parse_devices("0,0")
