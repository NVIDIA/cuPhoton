# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, event-owned output copies; no correlation math lives here."""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .profiling import nvtx_range


@dataclass(slots=True)
class _OutputSlot:
    arena: Any
    host: Any
    scratch: Any = None
    pending: Any = None
    compute_stream: Any = None
    compute_done: Any = None
    copied: Any = None
    busy: bool = False


class OutputPipeline:
    """Two arenas whose pinned storage never escapes materialization."""

    def __init__(
        self, xp, arena_factory, *, pinned_factory=None, tiled=False
    ):
        if pinned_factory is None:
            from cupyx import empty_pinned

            pinned_factory = empty_pinned
        self.xp = xp
        self.slots = []
        self.queue = deque()
        self.stream = None
        try:
            with nvtx_range("xbeam:output-pipeline-allocation"):
                for _ in range(2):
                    arena = arena_factory()
                    host = pinned_factory(
                        int(arena.raw.nbytes), dtype="uint8"
                    )
                    scratch = arena_factory() if tiled else None
                    self.slots.append(_OutputSlot(arena, host, scratch))
                self.stream = xp.cuda.Stream(non_blocking=True)
        except BaseException:
            self.slots.clear()
            raise

    def acquire(self):
        for slot in self.slots:
            if not slot.busy:
                slot.busy = True
                slot.compute_stream = self.xp.cuda.get_current_stream()
                slot.arena.spectrum_cursor = 0
                slot.arena.record_cursor = 0
                slot.arena.retained = None
                return slot.arena
        raise RuntimeError("both output slots are still owned")

    def enqueue(self, arena, pending):
        slot = next(slot for slot in self.slots if slot.arena is arena)
        if not slot.busy or slot.pending is not None:
            raise RuntimeError("output slot was not acquired for compute")
        slot.pending = pending
        # Establish ownership before any CUDA operation can fail.
        self.queue.append(slot)
        with nvtx_range("xbeam:output-pipeline-d2h-submit"):
            slot.compute_done = self.xp.cuda.Event(disable_timing=True)
            slot.compute_done.record(slot.compute_stream)
            with self.stream:
                self.stream.wait_event(slot.compute_done)
                arena.raw.get(
                    out=slot.host, stream=self.stream, blocking=False
                )
                slot.copied = self.xp.cuda.Event(disable_timing=True)
                slot.copied.record(self.stream)

    def scratch_for(self, arena):
        """Return scratch owned for the same lifetime as this accumulator."""
        return next(
            slot.scratch for slot in self.slots if slot.arena is arena
        )

    @contextmanager
    def completed(self):
        slot = self.queue[0]
        with nvtx_range("xbeam:output-pipeline-copy-wait"):
            slot.copied.synchronize()
        try:
            yield slot.pending, slot.arena, slot.host
        finally:
            self.queue.popleft()
            slot.pending = None
            slot.compute_done = slot.copied = slot.compute_stream = None
            slot.arena.retained = None
            slot.busy = False

    def close(self, *, best_effort=False):
        error = None
        with nvtx_range("xbeam:output-pipeline-final-drain"):
            # Includes partially submitted copies after an exception.
            streams = [self.stream] if self.stream is not None else []
            streams += [
                slot.compute_stream
                for slot in self.slots
                if slot.busy and slot.compute_stream is not None
            ]
            for stream in streams:
                try:
                    stream.synchronize()
                except BaseException as exc:
                    error = error or exc
        self.queue.clear()
        self.slots.clear()
        self.stream = None
        if error is not None and not best_effort:
            raise error
