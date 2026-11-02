# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded device byte slots, completion fences and ordered delivery.

Adapters provide byte-range plans and materialize views only after transfer.
This module has no sample geometry, frame decoding or validity rules.
"""

from dataclasses import dataclass, field

from .batches import OrderedBatches
from .buffers import aligned_device_buffer
from .completion import record_completion, wait_for_host
from .uploads import bounded_upload_workers


@dataclass
class DeviceSlot:
    buffers: dict = field(default_factory=dict)
    host_buffers: dict = field(default_factory=dict)
    lease: object = None
    stream: object = None


class DeviceLease(dict):
    def __init__(self, arrays, ready, xp, device):
        super().__init__(arrays)
        self.ready = ready
        self.xp, self.device = xp, device
        self.submitted = None
        self.consumed = None
        self.consumer = None
        self.checked = False

    def wait(self):
        if self.submitted is not None:
            self.submitted.result()
        with self.xp.cuda.Device(self.device):
            if not self.checked:
                self.ready.synchronize()
                self.checked = True
            self.consumer = self.xp.cuda.get_current_stream()
            self.consumer.wait_event(self.ready)

    def release(self):
        if self.consumed is None:
            with self.xp.cuda.Device(self.device):
                stream = (
                    self.consumer
                    if self.consumer is not None
                    else self.xp.cuda.get_current_stream()
                )
                for event in getattr(self, "tile_events", ()):
                    stream.wait_event(event)
                self.consumed = record_completion(
                    self.xp, stream, disable_timing=False
                )


class DeviceReadBatches(OrderedBatches):
    """Transport-independent GPU buffers, stream ordering and cleanup."""

    alignment = 1
    label = "input"
    buffer_limit_name = "buffer_bytes"

    def __init__(
        self,
        path,
        requests,
        buffer_bytes,
        queue_depth=2,
        *,
        upload_workers=None,
    ):
        super().__init__(requests, queue_depth)
        self.path = path
        self.buffer_bytes = buffer_bytes
        self.slots = [DeviceSlot() for _ in range(queue_depth)]
        self.worker_count = bounded_upload_workers(
            upload_workers, automatic=queue_depth, available=queue_depth
        )
        self.index = 0
        self.session = None
        self.streams = []

    def __enter__(self):
        import cupy

        self.xp = cupy
        self.device = int(cupy.cuda.Device().id)
        try:
            self._open()
            for _ in range(self.worker_count):
                stream = cupy.cuda.Stream(non_blocking=True)
                self._register_stream(stream)
                self.streams.append(stream)
            for index, slot in enumerate(self.slots):
                slot.stream = self.streams[index % self.worker_count]
            self._prime()
        except BaseException:
            self.close()
            raise
        return self

    def _register_stream(self, stream):
        pass

    def _deregister_stream(self, stream):
        pass

    def _allocate(self, slot, name):
        if name not in slot.buffers:
            _, buffer = aligned_device_buffer(
                self.xp,
                self.buffer_bytes,
                self.alignment,
            )
            slot.buffers[name] = buffer
            self._register_buffer(buffer)

    def _register_buffer(self, buffer):
        pass

    def _close_transport(self):
        pass

    def _finish(self, slot, lease, plans, request):
        lease.update(self._assemble(slot.buffers, plans, request))
        lease.ready.record(slot.stream)

    def _plan_request(self, request):
        """Map labels to (ranges, bytes); ranges begin (offset, size, dest).

        Trailing range fields are adapter-owned materialization metadata.
        """
        raise NotImplementedError

    def _assemble(self, buffers, plans, request):
        raise NotImplementedError

    def read_request(self, request):
        if self.closed:
            raise RuntimeError("input batch reader is closed")
        slot = self.slots[self.index % len(self.slots)]
        self.index += 1
        plans = self._plan_request(request)
        with self.xp.cuda.Device(self.device), slot.stream:
            if slot.lease is not None:
                if slot.lease.consumed is None:
                    raise RuntimeError("input lease was not released")
                # Host buffers may be overwritten only
                # after the previous I/O finishes; the event below separately
                # protects device data while compute still consumes it.
                wait_for_host(slot.lease.ready, slot.lease.submitted)
                slot.stream.wait_event(slot.lease.consumed)
            for name, (_, needed) in plans.items():
                if needed > self.buffer_bytes:
                    raise ValueError(
                        f"{self.label} batch exceeds "
                        f"{self.buffer_limit_name}; "
                        "reduce the byte request size or increase "
                        f"{self.buffer_limit_name}"
                    )
                self._allocate(slot, name)
            lease = DeviceLease(
                {},
                self.xp.cuda.Event(),
                self.xp,
                self.device,
            )
            slot.lease = lease
            self._enqueue(slot, lease, plans, request)
        return lease

    def close(self):
        if hasattr(self, "device"):
            with self.xp.cuda.Device(self.device):
                self._close()
        else:
            self._close()

    def _close(self):
        if self.closed:
            return
        self.closed = True
        errors = []

        def finish(operation):
            try:
                operation()
            except BaseException as error:
                errors.append(error)

        # CPU read callbacks must finish enqueueing before streams drain.
        for slot in self.slots:
            if slot.lease is not None:
                if slot.lease.submitted is not None:
                    finish(slot.lease.submitted.result)
                if slot.lease.consumer is not None:
                    finish(slot.lease.release)
        finish(self._close_transport)
        for stream in self.streams:
            finish(stream.synchronize)
        for slot in self.slots:
            if slot.lease is not None and slot.lease.consumed is not None:
                finish(slot.lease.consumed.synchronize)
        for stream in self.streams:
            finish(lambda stream=stream: self._deregister_stream(stream))
        if self.session is not None:
            finish(lambda: self.session.__exit__(None, None, None))
        for slot in self.slots:
            if slot.lease is not None:
                slot.lease.clear()
            slot.buffers.clear()
            slot.host_buffers.clear()
            slot.lease = slot.stream = None
        self._discard_queue()
        self.slots.clear()
        self.streams.clear()
        if errors:
            raise errors[0]
