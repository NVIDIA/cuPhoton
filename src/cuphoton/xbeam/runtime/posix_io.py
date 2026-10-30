# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, concurrent positioned reads of immutable byte ranges."""

import errno
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Condition


class ShortReadError(OSError):
    def __init__(self, expected, received):
        self.expected, self.received = expected, received
        super().__init__(
            f"short POSIX read: expected {expected}, received {received}"
        )


def read_exact_into(fd, output, offset):
    """Fill owned contiguous bytes, retrying interrupted/partial reads."""
    view = memoryview(output).cast("B")
    expected, received = len(view), 0
    while view:
        try:
            count = os.preadv(fd, [view], offset)
        except InterruptedError:
            continue
        if count == 0:
            raise ShortReadError(expected, received)
        received += count
        offset += count
        view = view[count:]


@dataclass(eq=False)
class _Read:
    transfers: object
    future: Future = field(default_factory=Future)
    active: int = 0
    exhausted: bool = False
    error: object = None
    completed: bool = False


class RangeReader:
    """Bounded I/O workers shared across signals and queued batches.

    A future becomes ready only after every writer for its batch has drained.
    Queued batches hold iterators, not one executor future per chunk.
    """

    label = "POSIX"
    thread_name_prefix = "xbeam-posix"

    def __init__(self, fd, *, workers, chunk_bytes):
        self.fd = os.dup(fd)
        self.workers, self.chunk_bytes = workers, chunk_bytes
        self._condition = Condition()
        self._closed = False
        self._ready = deque()
        self._reads = set()
        self._stream_fds = {}
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=self.thread_name_prefix
        )
        try:
            self._workers = [
                self._pool.submit(self._work) for _ in range(workers)
            ]
        except BaseException:
            self.close()
            raise

    def _read_into(self, output, start, stop, offset, fd=None):
        fd = self.fd if fd is None else fd
        read_exact_into(fd, memoryview(output).cast("B")[start:stop], offset)

    def _complete(self, read):
        if read.completed or not read.exhausted or read.active:
            return None
        read.completed = True
        self._reads.discard(read)
        return read

    @staticmethod
    def _publish(read):
        if read is not None:
            if read.error is None:
                read.future.set_result(None)
            else:
                read.future.set_exception(read.error)

    def _work(self):
        while True:
            with self._condition:
                while not self._ready and not self._closed:
                    self._condition.wait()
                if not self._ready:
                    return
                read = self._ready.popleft()
                transfer = None
                if not read.exhausted:
                    try:
                        transfer = next(read.transfers)
                    except StopIteration:
                        read.exhausted = True
                    except BaseException as iterator_error:
                        read.error, read.exhausted = iterator_error, True
                if transfer is not None:
                    read.active += 1
                    self._ready.appendleft(read)
                completed = self._complete(read)
            self._publish(completed)
            if transfer is None:
                read = completed = None
                continue
            error = None
            try:
                self._read_into(*transfer)
            except BaseException as failure:
                error = failure
            finally:
                transfer = None
            with self._condition:
                read.active -= 1
                if error is not None:
                    if read.error is None:
                        read.error = error
                    read.exhausted = True
                    read.transfers = iter(())
                completed = self._complete(read)
            self._publish(completed)
            # Idle workers must not retain a completed batch's callbacks,
            # device slots or pinned host buffers in their stack frames.
            read = completed = error = None

    def submit(self, transfers):
        read = _Read(iter(transfers))
        with self._condition:
            if self._closed:
                raise RuntimeError(f"{self.label} reader is closed")
            # Cancellation must not publish readiness while writers still
            # own destination buffers. close() aborts queued work and drains.
            read.future.set_running_or_notify_cancel()
            self._reads.add(read)
            self._ready.append(read)
            self._condition.notify_all()
        return read.future

    def _stream_fd(self, name):
        # dup() shares file-description read-ahead state. On Linux, opening
        # the owned descriptor gives each stream an independent sequential
        # stream without resolving the original, replaceable filename.
        with self._condition:
            if self._closed:
                raise RuntimeError(f"{self.label} reader is closed")
            if name not in self._stream_fds:
                try:
                    fd = os.open(
                        f"/proc/self/fd/{self.fd}",
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                    )
                except OSError as error:
                    if error.errno not in (
                        errno.ENOENT,
                        errno.ENOTDIR,
                        errno.EACCES,
                        errno.EPERM,
                    ):
                        raise
                    # Correctness is unchanged where procfs is unavailable;
                    # the fallback shares the original read-ahead state.
                    fd = os.dup(self.fd)
                self._stream_fds[name] = fd
            return self._stream_fds[name]

    def chunks(self, output, offset, start=0, stop=None, *, fd=None):
        stop = output.nbytes if stop is None else stop
        # Keep the requested workers useful even for a small batch, while
        # avoiding tiny filesystem requests unless explicitly configured.
        portion = max(
            1024**2, (stop - start + self.workers - 1) // self.workers
        )
        chunk = min(self.chunk_bytes, (portion + 4095) // 4096 * 4096)
        for head in range(start, stop, chunk):
            yield (
                output,
                head,
                min(head + chunk, stop),
                offset + head - start,
                fd,
            )

    def submit_batch(self, buffers, plans):
        descriptors = {name: self._stream_fd(name) for name in plans}

        def transfers():
            for name, (plan, _) in plans.items():
                for offset, length, dest, *_ in plan:
                    yield from self.chunks(
                        buffers[name],
                        offset,
                        dest,
                        dest + length,
                        fd=descriptors[name],
                    )

        # Issue contiguous chunks first, without waiting for their writers
        # before another stream/batch can start. This preserves read-ahead
        # locality while keeping all workers supplied with queued work.
        return self.submit(transfers())

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            completed = []
            for read in tuple(self._reads):
                if read.error is None:
                    read.error = RuntimeError(
                        f"{self.label} reader is closed"
                    )
                read.exhausted = True
                read.transfers = iter(())
                done = self._complete(read)
                if done is not None:
                    completed.append(done)
            self._ready.clear()
            self._condition.notify_all()
        for read in completed:
            self._publish(read)
        try:
            self._pool.shutdown(wait=True)
        finally:
            for fd in self._stream_fds.values():
                os.close(fd)
            self._stream_fds.clear()
            os.close(self.fd)
