# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""POSIX and cuFile byte transports shared across recording adapters."""

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock

from ..cufile_io import CuFileSession
from .buffers import copy_bytes, pinned_bytes
from .posix_io import RangeReader


class _SubmittedTransport:
    """Publish on a CUDA stream only after byte-checked I/O completes."""

    def _read(self, slot, plans):
        raise NotImplementedError

    def _copy(self, slot, plans):
        pass

    def _enqueue(self, slot, lease, plans, request):
        # These callbacks run on ordinary I/O threads, never CUDA callbacks.
        lease.submitted = Future()
        try:
            reading = self._read(slot, plans)
        except BaseException as error:
            # Shutdown must not wait forever on an unsubmitted transfer.
            lease.submitted.set_exception(error)
            raise

        def publish(completion):
            try:
                completion.result()
                with self.xp.cuda.Device(self.device), slot.stream:
                    self._copy(slot, plans)
                    self._finish(slot, lease, plans, request)
            except BaseException as error:
                lease.submitted.set_exception(error)
            else:
                lease.submitted.set_result(None)

        reading.add_done_callback(publish)


class PosixTransport(_SubmittedTransport):
    """Read pinned host slots, then enqueue copies on shared I/O streams."""

    label = "POSIX"

    def __init__(self, reader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = reader

    def _open(self):
        pass

    def _allocate(self, slot, name):
        super()._allocate(slot, name)
        if name not in slot.host_buffers:
            slot.host_buffers[name] = pinned_bytes(self.buffer_bytes)

    def _read(self, slot, plans):
        return self.reader.submit_batch(slot.host_buffers, plans)

    def _copy(self, slot, plans):
        for name, (_, needed) in plans.items():
            copy_bytes(
                self.xp,
                slot.buffers[name],
                slot.host_buffers[name],
                needed,
                slot.stream,
            )


class _ConsumptionFence:
    """Wait once per slot before workers issue DMA outside its CUDA stream."""

    def __init__(self, stream):
        self.stream = stream
        self.lock = Lock()
        self.ready = False

    def wait(self):
        with self.lock:
            if not self.ready:
                self.stream.synchronize()
                self.ready = True


@dataclass
class _DeviceTarget:
    buffer: object
    nbytes: int
    fence: _ConsumptionFence


class _CuFileRangeReader(RangeReader):
    label = "cuFile"
    thread_name_prefix = "xbeam-cufile"

    def __init__(self, xp, device, session, file, *, workers, chunk_bytes):
        self.xp, self.device = xp, device
        self.session, self.file = session, file
        super().__init__(file.fd, workers=workers, chunk_bytes=chunk_bytes)

    def _stream_fd(self, name):
        return self.fd

    def _read_into(self, output, start, stop, offset, fd=None):
        with self.xp.cuda.Device(self.device):
            output.fence.wait()
            expected = stop - start
            received = self.session.read(
                self.file,
                output.buffer.data.ptr,
                expected,
                offset,
                buffer_offset=start,
            )
            if received != expected:
                raise OSError(
                    "cuFile short/failed read: "
                    f"expected {expected}, received {received}"
                )


class CuFileTransport(_SubmittedTransport):
    """Read registered GPU slots on bounded workers, then publish events.

    cuFileReadAsync can report a full byte count after its compatibility
    copy fails. Blocking cuFile reads on I/O workers retain direct GPU I/O
    and overlap without relying on that stream-async completion signal.
    """

    alignment = 4096
    label = "cuFile"

    def _open(self):
        self.session = CuFileSession().__enter__()
        self.file = self.session.open(self.path)
        self.reader = _CuFileRangeReader(
            self.xp,
            self.device,
            self.session,
            self.file,
            workers=self.worker_count,
            chunk_bytes=min(self.buffer_bytes, 16 * 1024**2),
        )

    def _register_buffer(self, buffer):
        self.session.register_buffer(buffer.data.ptr, buffer.nbytes)

    def _read(self, slot, plans):
        fence = _ConsumptionFence(slot.stream)
        targets = {
            name: _DeviceTarget(buffer, buffer.nbytes, fence)
            for name, buffer in slot.buffers.items()
            if name in plans
        }
        return self.reader.submit_batch(targets, plans)

    def _close_transport(self):
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.close()
