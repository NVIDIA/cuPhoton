# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, ordered host read-ahead with explicit shutdown ownership."""

from concurrent.futures import ThreadPoolExecutor


class BatchPrefetch:
    """One current batch and at most one queued/read-ahead batch.

    ``read`` must return independently owned host buffers, without doing GPU
    work. Errors propagate when the affected batch is consumed. Closing
    cancels queued work and waits for an in-flight read before releasing
    resources; it cannot interrupt a blocked filesystem operation.
    """

    def __init__(self, read, requests):
        self._read = read
        self._requests = iter(requests)
        self._pool = None
        self._future = None
        self._closed = False

    def __enter__(self):
        if self._closed or self._pool is not None:
            raise RuntimeError("batch reader cannot be entered twice")
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xbeam-read"
        )
        try:
            self._submit()
        except BaseException:
            self.close()
            raise
        return self

    def _submit(self):
        try:
            request = next(self._requests)
        except StopIteration:
            self._future = None
        else:
            self._future = self._pool.submit(self._read, request)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._future is None:
            raise StopIteration
        batch = self._future.result()
        self._submit()
        return batch

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        self._future = self._pool = self._read = None
        self._requests = iter(())

    def __exit__(self, *args):
        self.close()
