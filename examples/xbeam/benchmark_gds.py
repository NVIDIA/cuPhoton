# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure aggregate cuFile read bandwidth across VDIF recordings."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from typing import Any

from cuphoton.xbeam.cufile_io import CuFileSession

_ALIGNMENT = 64 * 1024


@dataclass(frozen=True, slots=True)
class ReadTask:
    """One aligned direct read and its logical byte contribution."""

    path: Path
    offset: int
    logical_bytes: int
    request_bytes: int


@dataclass(slots=True)
class _Reader:
    device: int
    tasks: tuple[ReadTask, ...]
    allocation: Any
    buffer: Any
    session: CuFileSession
    files: dict[Path, Any]


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return -(-value // alignment) * alignment


def _discover_vdif_files(inputs: Sequence[Path]) -> tuple[Path, ...]:
    files: set[Path] = set()
    for source in inputs:
        if source.is_dir():
            files.update(
                path for path in source.rglob("*.vdif") if path.is_file()
            )
        elif source.is_file():
            files.add(source)
        else:
            raise FileNotFoundError(f"VDIF input does not exist: {source}")
    if not files:
        raise FileNotFoundError("no .vdif files were found")
    return tuple(sorted(path.resolve() for path in files))


def _read_tasks(
    files: Sequence[Path], chunk_bytes: int
) -> tuple[ReadTask, ...]:
    if chunk_bytes <= 0 or chunk_bytes % _ALIGNMENT:
        raise ValueError(
            "GDS chunk size must be a positive multiple of 64 KiB"
        )
    tasks: list[ReadTask] = []
    for path in files:
        size = path.stat().st_size
        for offset in range(0, size, chunk_bytes):
            logical_bytes = min(chunk_bytes, size - offset)
            tasks.append(
                ReadTask(
                    path,
                    offset,
                    logical_bytes,
                    _align_up(logical_bytes),
                )
            )
    return tuple(tasks)


def _assign_tasks(
    tasks: Sequence[ReadTask], reader_count: int
) -> tuple[tuple[ReadTask, ...], ...]:
    if reader_count <= 0:
        raise ValueError("GDS reader count must be positive")
    assignments: list[list[ReadTask]] = [[] for _ in range(reader_count)]
    for index, task in enumerate(tasks):
        assignments[index % reader_count].append(task)
    return tuple(tuple(items) for items in assignments)


def _aligned_buffer(cupy: Any, size: int) -> tuple[Any, Any]:
    allocation = cupy.empty(size + _ALIGNMENT - 1, dtype=cupy.uint8)
    offset = (-int(allocation.data.ptr)) % _ALIGNMENT
    return allocation, allocation[offset : offset + size]


def _create_reader(
    cupy: Any, device: int, tasks: tuple[ReadTask, ...], chunk_bytes: int
) -> _Reader:
    with cupy.cuda.Device(device):
        allocation, buffer = _aligned_buffer(cupy, chunk_bytes)
        session = CuFileSession()
        session.__enter__()
        try:
            session.register_buffer(int(buffer.data.ptr), int(buffer.nbytes))
            files = {
                path: session.open(path)
                for path in sorted({task.path for task in tasks})
            }
        except BaseException:
            session.close()
            raise
    return _Reader(device, tasks, allocation, buffer, session, files)


def _read_all(reader: _Reader, start: Barrier) -> int:
    start.wait()
    import cupy

    completed = 0
    with cupy.cuda.Device(reader.device):
        pointer = int(reader.buffer.data.ptr)
        for task in reader.tasks:
            received = reader.session.read(
                reader.files[task.path],
                pointer,
                task.request_bytes,
                task.offset,
            )
            if received < task.logical_bytes:
                raise OSError(
                    f"short cuFile read from {task.path} at {task.offset}: "
                    f"expected {task.logical_bytes} bytes, got {received}"
                )
            completed += task.logical_bytes
    return completed


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be a comma-separated integer list"
        ) from error
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("CUDA devices must be non-negative")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("CUDA devices must be unique")
    return devices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="VDIF files or directories searched recursively for *.vdif",
    )
    parser.add_argument("--devices", type=_parse_devices, default=(0,))
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--report", type=Path)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    if arguments.chunk_mib <= 0:
        raise ValueError("--chunk-mib must be positive")
    if arguments.trials <= 0:
        raise ValueError("--trials must be positive")

    # Fail rather than silently measure the POSIX compatibility fallback.
    os.environ["CUFILE_ALLOW_COMPAT_MODE"] = "false"
    import cupy

    files = _discover_vdif_files(arguments.inputs)
    chunk_bytes = arguments.chunk_mib * 1024 * 1024
    tasks = _read_tasks(files, chunk_bytes)
    total_bytes = sum(task.logical_bytes for task in tasks)
    reader_count = len(arguments.devices) * arguments.workers_per_gpu
    assignments = _assign_tasks(tasks, reader_count)

    setup_start = time.perf_counter()
    readers: list[_Reader] = []
    try:
        for index, assigned in enumerate(assignments):
            readers.append(
                _create_reader(
                    cupy,
                    arguments.devices[index % len(arguments.devices)],
                    assigned,
                    chunk_bytes,
                )
            )
        setup_seconds = time.perf_counter() - setup_start
        trial_seconds: list[float] = []
        with ThreadPoolExecutor(max_workers=reader_count) as executor:
            for trial in range(arguments.trials):
                start_barrier = Barrier(reader_count + 1)
                futures = [
                    executor.submit(_read_all, reader, start_barrier)
                    for reader in readers
                ]
                started = time.perf_counter()
                start_barrier.wait()
                completed = sum(future.result() for future in futures)
                elapsed = time.perf_counter() - started
                if completed != total_bytes:
                    raise RuntimeError(
                        "read accounting mismatch: "
                        f"{completed} != {total_bytes}"
                    )
                trial_seconds.append(elapsed)
                print(
                    f"trial {trial + 1}/{arguments.trials}: {elapsed:.6f} s, "
                    f"{total_bytes / elapsed / 1e9:.3f} GB/s "
                    f"({total_bytes / elapsed / (1 << 30):.3f} GiB/s)",
                    flush=True,
                )
    finally:
        for reader in reversed(readers):
            reader.session.close()

    median_seconds = statistics.median(trial_seconds)
    per_device_bytes = {
        str(device): sum(
            sum(task.logical_bytes for task in reader.tasks)
            for reader in readers
            if reader.device == device
        )
        for device in arguments.devices
    }
    per_device_gb_per_second = {
        device: byte_count / median_seconds / 1e9
        for device, byte_count in per_device_bytes.items()
    }
    report = {
        "schema_version": 1,
        "engine": "cuphoton-gds-vdif-read",
        "files": len(files),
        "total_bytes": total_bytes,
        "devices": list(arguments.devices),
        "workers_per_gpu": arguments.workers_per_gpu,
        "chunk_bytes": chunk_bytes,
        "setup_seconds": setup_seconds,
        "trial_seconds": trial_seconds,
        "median_seconds": median_seconds,
        "median_gb_per_second": total_bytes / median_seconds / 1e9,
        "median_gib_per_second": total_bytes / median_seconds / (1 << 30),
        "logical_bytes_per_device": per_device_bytes,
        "median_gb_per_second_by_device": per_device_gb_per_second,
        "inputs": [str(path) for path in files],
    }
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
