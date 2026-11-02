# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered bounded submission shared by host and device input transports."""

from collections import deque
from collections.abc import Sequence
from threading import Lock


class OrderedBatches:
    """Reserve lookahead capacity without waiting for input readiness."""

    def __init__(self, requests, depth):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 2:
            raise ValueError("batch queue depth must be at least two")
        self.requests = iter(requests)
        self.depth = depth
        self.pending = deque()
        self.closed = False

    def _next_read(self):
        try:
            request = next(self.requests)
        except StopIteration:
            return None
        return self.read_request(request)

    def _prime(self):
        for _ in range(self.depth - 1):
            lease = self._next_read()
            if lease is None:
                break
            self.pending.append(lease)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or not self.pending:
            raise StopIteration
        lease = self.pending.popleft()
        following = self._next_read()
        if following is not None:
            self.pending.append(following)
        return lease

    def _discard_queue(self):
        self.pending.clear()
        self.requests = iter(())

    def __exit__(self, *args):
        self.close()


class WorkQueue[T]:
    """Dynamically dispatch ordered work; drain pending tasks on failure."""

    def __init__(self, tasks: Sequence[T]) -> None:
        self._tasks = deque(tasks)
        self._lock = Lock()
        self._active = 0
        self._failed = False

    def take(self) -> T | None:
        with self._lock:
            if self._failed or not self._tasks:
                return None
            self._active += 1
            return self._tasks.popleft()

    def finish(self, error: BaseException | None) -> list[T]:
        with self._lock:
            if self._active <= 0:
                raise RuntimeError("work queue completion underflow")
            self._active -= 1
            pending: list[T] = []
            if error is not None and not self._failed:
                self._failed = True
                pending.extend(self._tasks)
                self._tasks.clear()
            return pending

    def abort(self, error: BaseException) -> list[T]:
        """Stop handing out work and return tasks that never started."""

        with self._lock:
            if self._failed:
                return []
            self._failed = True
            pending = list(self._tasks)
            self._tasks.clear()
            return pending
