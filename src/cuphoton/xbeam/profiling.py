# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in profiling annotations with no dependency on external profilers."""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

_TRUE_VALUES = frozenset({"1", "on", "true", "yes"})
_enabled = os.environ.get("CUPHOTON_XBEAM_NVTX", "").strip().lower() in (
    _TRUE_VALUES
)
_host_timing_enabled = (
    os.environ.get("CUPHOTON_XBEAM_HOST_TIMING", "").strip().lower()
    in _TRUE_VALUES
)
_host_timing_file = os.environ.get(
    "CUPHOTON_XBEAM_HOST_TIMING_FILE", ""
).strip()
if _host_timing_file:
    _host_timing_enabled = True
_nvtx: Any = None
_nvtx_import_attempted = False
_timing_lock = threading.Lock()
_timing_local = threading.local()


@dataclass(slots=True)
class _HostTiming:
    calls: int = 0
    inclusive_ns: int = 0
    self_ns: int = 0
    minimum_ns: int = sys.maxsize
    maximum_ns: int = 0


_host_timings: dict[str, _HostTiming] = {}
_host_timing_written = False


def nvtx_enabled() -> bool:
    """Return whether optional NVTX annotations were requested."""

    return _enabled


def host_timing_enabled() -> bool:
    """Return whether lightweight host range timing was requested."""

    return _host_timing_enabled


def record_host_interval(message: str, started_ns: int) -> None:
    """Record an asynchronous host interval without CUDA synchronization.

    Queue residence is not a nested CPU execution range and may overlap
    other host timings. It deliberately does not manufacture NVTX ranges.
    """

    if _host_timing_enabled:
        elapsed = max(0, time.perf_counter_ns() - started_ns)
        _record_host_timing(message, elapsed, elapsed)


def _record_host_timing(message: str, elapsed_ns: int, self_ns: int) -> None:
    with _timing_lock:
        timing = _host_timings.setdefault(message, _HostTiming())
        timing.calls += 1
        timing.inclusive_ns += elapsed_ns
        timing.self_ns += self_ns
        timing.minimum_ns = min(timing.minimum_ns, elapsed_ns)
        timing.maximum_ns = max(timing.maximum_ns, elapsed_ns)


class _HostTimingRow(TypedDict):
    name: str
    calls: int
    inclusive_seconds: float
    self_seconds: float
    minimum_seconds: float
    maximum_seconds: float


def host_timing_report() -> dict[str, Any]:
    """Return a stable snapshot of the accumulated host range timings."""

    with _timing_lock:
        rows: list[_HostTimingRow] = [
            {
                "name": name,
                "calls": timing.calls,
                "inclusive_seconds": timing.inclusive_ns / 1_000_000_000,
                "self_seconds": timing.self_ns / 1_000_000_000,
                "minimum_seconds": timing.minimum_ns / 1_000_000_000,
                "maximum_seconds": timing.maximum_ns / 1_000_000_000,
            }
            for name, timing in _host_timings.items()
        ]
    rows.sort(key=lambda row: (-row["inclusive_seconds"], row["name"]))
    return {
        "format": "cuphoton-xbeam-host-timing-v1",
        "ranges": rows,
    }


def write_host_timing_report(
    path: str | os.PathLike[str] | None = None,
) -> None:
    """Write accumulated timings as JSON, or as text on standard error."""

    global _host_timing_written
    if not _host_timing_enabled or _host_timing_written:
        return
    report = host_timing_report()
    destination = str(path) if path is not None else _host_timing_file
    if destination:
        Path(destination).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _host_timing_written = True
        return
    print(
        "cuphoton-xbeam host timing (inclusive/self seconds):",
        file=sys.stderr,
    )
    for row in report["ranges"]:
        print(
            f"  {row['inclusive_seconds']:.6f} "
            f"{row['self_seconds']:.6f} "
            f"{row['calls']:6d} {row['name']}",
            file=sys.stderr,
        )
    _host_timing_written = True


def _nvtx_module() -> Any:
    global _nvtx, _nvtx_import_attempted
    if not _enabled:
        return None
    if not _nvtx_import_attempted:
        _nvtx_import_attempted = True
        try:
            from cupy.cuda import nvtx
        except (ImportError, OSError):
            _nvtx = None
        else:
            _nvtx = nvtx
    return _nvtx


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Annotate and optionally time a host range.

    Host timing uses only ``perf_counter_ns``.  It never queries a CuPy
    scalar, records a CUDA event, or synchronizes a device, so enabling it
    does not change the asynchronous execution model.
    """

    nvtx = _nvtx_module()
    if nvtx is None and not _host_timing_enabled:
        yield
        return
    if nvtx is not None:
        nvtx.RangePush(message)
    started_ns = time.perf_counter_ns() if _host_timing_enabled else 0
    stack: list[list[int]] | None = None
    if _host_timing_enabled:
        stack = getattr(_timing_local, "stack", None)
        if stack is None:
            stack = []
            _timing_local.stack = stack
        stack.append([started_ns, 0])
    try:
        yield
    finally:
        if _host_timing_enabled:
            stopped_ns = time.perf_counter_ns()
            assert stack is not None
            frame_started_ns, child_ns = stack.pop()
            elapsed_ns = stopped_ns - frame_started_ns
            _record_host_timing(message, elapsed_ns, elapsed_ns - child_ns)
            if stack:
                stack[-1][1] += elapsed_ns
        if nvtx is not None:
            nvtx.RangePop()


atexit.register(write_host_timing_report)


__all__ = [
    "host_timing_enabled",
    "host_timing_report",
    "nvtx_enabled",
    "nvtx_range",
    "write_host_timing_report",
]
