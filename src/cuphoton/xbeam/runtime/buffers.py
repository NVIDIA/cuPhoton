# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Byte buffers shared by file adapters and streaming ingest workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..cufile_io import CuFileSession

CUFILE_DEVICE_ALIGNMENT = 64 * 1024


def aligned_device_buffer(
    xp: Any, size: int, alignment: int
) -> tuple[Any, Any]:
    """Return an owning allocation and an aligned contiguous byte view."""
    allocation = xp.empty(size + alignment - 1, dtype=xp.uint8)
    offset = (-int(allocation.data.ptr)) % alignment
    return allocation, allocation[offset : offset + size]


def pinned_bytes(size: int) -> np.ndarray:
    """Allocate through the shared pinned allocator, without a host copy."""
    import cupyx

    return cupyx.empty_pinned(size, dtype=np.uint8)


def copy_bytes(
    xp: Any, destination: Any, source: Any, size: int, stream: Any
) -> None:
    """Enqueue contiguous pinned bytes on the caller-owned CUDA stream."""
    if size < 0 or size > source.nbytes or size > destination.nbytes:
        raise ValueError("copy byte count exceeds source or destination")
    if not source.flags.c_contiguous or not destination.flags.c_contiguous:
        raise ValueError("byte copies require contiguous buffers")
    if size == 0:
        return
    runtime = getattr(xp.cuda, "runtime", None)
    copy = getattr(runtime, "memcpyAsync", None)
    if runtime is None or copy is None:
        # Compatible array backends/stream wrappers may expose only set().
        destination[:size].set(source[:size], stream=stream)
        return
    copy(
        destination.data.ptr,
        source.ctypes.data,
        size,
        runtime.memcpyHostToDevice,
        stream.ptr,
    )


@dataclass(slots=True)
class PinnedWorkspace:
    device_identifier: int
    capacity: int
    pinned: tuple[Any, Any]
    staging: tuple[Any, Any]
    copy_stream: Any

    @property
    def retained_bytes(self) -> int:
        return 4 * self.capacity


@dataclass(slots=True)
class RegisteredWorkspace:
    device_identifier: int
    capacity: int
    allocations: tuple[Any, ...]
    buffers: tuple[Any, ...]
    session: CuFileSession

    @property
    def retained_bytes(self) -> int:
        return sum(int(allocation.nbytes) for allocation in self.allocations)
