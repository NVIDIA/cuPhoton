# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HDF5 layout and compute tiles over generic device byte delivery."""

from collections import deque
from dataclasses import dataclass

from ..runtime.device_delivery import (
    DeviceLease,
    DeviceReadBatches,
    DeviceSlot,
)
from .hdf5_layout import assemble_batch, plan_batch

__all__ = ["DeviceBatches", "DeviceLease", "DeviceSlot", "read_blocks"]


@dataclass
class ReadBlock:
    request: tuple
    rows: tuple


def read_blocks(requests, names, layouts, buffer_bytes, alignment):
    """Combine adjacent requests within the existing per-dataset slot budget.

    Preserve compute tile boundaries and never bridge unrequested sample gaps.
    Only regular window grids merge: irregular grids could otherwise create
    a large gathered array despite a small raw read envelope. Bound metadata
    lookahead even when requests contain very small windows.
    """
    pending = None
    for request in requests:
        signals, starts, size = request
        starts = list(starts)
        hop = starts[1] - starts[0] if len(starts) > 1 else None
        contiguous = bool(starts) and (
            hop is None
            or (
                0 <= hop <= size
                and all(
                    right - left == hop
                    for left, right in zip(starts, starts[1:], strict=False)
                )
            )
        )
        if pending is not None:
            previous, rows = pending
            old_signals, old_starts, old_size = previous
            merge = (
                contiguous
                and old_signals == signals
                and old_size == size
                and old_starts[-1] <= starts[0] <= old_starts[-1] + size
                and len(rows) < 64
                and len(old_starts) + len(starts) <= 65536
            )
            if merge:
                old_hop = (
                    old_starts[1] - old_starts[0]
                    if len(old_starts) > 1
                    else starts[0] - old_starts[-1]
                )
                merge = starts[0] == old_starts[-1] + old_hop and (
                    hop is None or hop == old_hop
                )
            if merge:
                fits = True
                for signal in signals:
                    for name in names[signal]:
                        if name is None:
                            continue
                        info = layouts[name]
                        lo = info.offset + old_starts[0] * info.dtype.itemsize
                        hi = (
                            info.offset
                            + (starts[-1] + size) * info.dtype.itemsize
                        )
                        needed = (
                            (hi + alignment - 1) // alignment * alignment
                            - lo // alignment * alignment
                        )
                        if needed > buffer_bytes:
                            fits = False
                if fits:
                    combined = signals, [*old_starts, *starts], size
                    pending = combined, (*rows, len(starts))
                    continue
            yield ReadBlock(previous, rows)
        pending = ((signals, starts, size), (len(starts),))
        if not contiguous:
            yield ReadBlock(*pending)
            pending = None
    if pending is not None:
        yield ReadBlock(*pending)


class TileLease(dict):
    """A compute tile borrowing a block until all consumers drain."""

    def __init__(self, owner, start, stop):
        super().__init__()
        self.owner, self.start, self.stop = owner, start, stop
        self.consumer = None
        self.released = False

    def wait(self):
        parent = self.owner
        if self.released:
            raise RuntimeError("HDF5 tile lease was released")
        parent.wait()
        self.consumer = parent.consumer
        self.update(
            {
                signal: (
                    values[self.start : self.stop],
                    None if mask is None else mask[self.start : self.stop],
                )
                for signal, (values, mask) in parent.items()
            }
        )

    def release(self):
        if self.released:
            return
        parent = self.owner
        with parent.xp.cuda.Device(parent.device):
            event = parent.xp.cuda.Event()
            event.record(self.consumer or parent.xp.cuda.get_current_stream())
            parent.tile_events.append(event)
            parent.remaining_tiles -= 1
            if parent.remaining_tiles == 0:
                stream = parent.xp.cuda.get_current_stream()
                parent.consumer = stream
                parent.release()
        self.released = True
        self.clear()


class DeviceBatches(DeviceReadBatches):
    label = "HDF5"
    buffer_limit_name = "read_buffer_bytes"

    def __init__(
        self,
        path,
        layouts,
        names,
        requests,
        buffer_bytes,
        queue_depth=2,
        *,
        upload_workers=None,
    ):
        self.layouts, self.names = layouts, names
        self.tiles = deque()
        super().__init__(
            path,
            read_blocks(
                requests, names, layouts, buffer_bytes, self.alignment
            ),
            buffer_bytes,
            queue_depth,
            upload_workers=upload_workers,
        )

    def _plan_request(self, request):
        return plan_batch(
            request, self.names, self.layouts, alignment=self.alignment
        )

    def _assemble(self, buffers, plans, request):
        return assemble_batch(
            buffers, plans, request, self.names, self.layouts, self.xp
        )

    def __next__(self):
        if self.closed:
            raise StopIteration
        if not self.tiles:
            lease = super().__next__()
            if len(lease.rows) == 1:
                return lease
            lease.remaining_tiles = len(lease.rows)
            lease.tile_events = []
            start = 0
            for count in lease.rows:
                self.tiles.append(TileLease(lease, start, start + count))
                start += count
        return self.tiles.popleft()

    def read_request(self, block):
        if not isinstance(block, ReadBlock):
            block = ReadBlock(block, (len(block[1]),))
        lease = super().read_request(block.request)
        lease.rows = block.rows
        return lease

    def _close(self):
        try:
            super()._close()
        finally:
            self.tiles.clear()
