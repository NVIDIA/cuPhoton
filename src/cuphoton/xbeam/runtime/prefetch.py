# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, ordered host read-ahead with explicit shutdown ownership."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .uploads import bounded_upload_workers


class BatchPrefetch:
    """One current batch and a bounded, ordered read-ahead queue.

    Defaults retain one read worker and one batch ahead. Increasing workers
    never grows the separate ``ahead`` capacity.

    ``read`` must return independently owned host buffers, without doing GPU
    work. Errors propagate when the affected batch is consumed. Closing
    cancels queued work and waits for an in-flight read before releasing
    resources; it cannot interrupt a blocked filesystem operation.
    """

    def __init__(self, read, requests, *, workers=1, ahead=1):
        if isinstance(ahead, bool) or not isinstance(ahead, int) or ahead < 1:
            raise ValueError("read-ahead capacity must be a positive integer")
        self._workers = bounded_upload_workers(
            workers, automatic=1, available=ahead
        )
        self._ahead = ahead
        self._read = read
        self._requests = iter(requests)
        self._pool = None
        self._futures = deque()
        self._closed = False

    def __enter__(self):
        if self._closed or self._pool is not None:
            raise RuntimeError("batch reader cannot be entered twice")
        self._pool = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="xbeam-read"
        )
        try:
            for _ in range(self._ahead):
                self._submit()
        except BaseException:
            self.close()
            raise
        return self

    def _submit(self):
        if self._closed:
            return
        try:
            request = next(self._requests)
        except StopIteration:
            return
        else:
            self._futures.append(self._pool.submit(self._read, request))

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or not self._futures:
            raise StopIteration
        batch = self._futures.popleft().result()
        if self._closed:
            raise StopIteration
        self._submit()
        return batch

    def close(self):
        if self._closed:
            return
        self._closed = True
        for future in self._futures:
            future.cancel()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        self._futures.clear()
        self._pool = self._read = None
        self._requests = iter(())

    def __exit__(self, *args):
        self.close()
