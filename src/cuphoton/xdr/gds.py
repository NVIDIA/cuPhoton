# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPUDirect Storage wrapper over kvikio.CuFile.

Entry points for the FITS reader:

- `pread_to_device`: read a single byte range straight into a cupy buffer.
  Used by the uncompressed `ImageHDU.to_device()` path.
- `GdsHeapLoader`: persistent CuFile that exposes both a blocking
  `load_tiles()` (backward-compatible) and an async `load_tiles_async()`
  that returns a `HeapReadHandle` for pipelining with downstream compute.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence


def available_cpu_cores() -> int:
    """Return the CPU count available to this process."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    count = (
        process_cpu_count()
        if process_cpu_count is not None
        else os.cpu_count()
    )
    return max(1, int(count or 1))


def configure_kvikio_parallelism() -> int:
    """Configure KvikIO's default thread pool for parallel FITS reads."""
    import kvikio.defaults as defaults

    num_threads = available_cpu_cores()
    defaults.set("num_threads", num_threads)
    return num_threads


@contextmanager
def open_cufile(path: str | Path):
    """Open a kvikio.CuFile (or a MockCuFile when storage_cache is active).

    Context-managed so fds close promptly. When the mock storage cache is
    enabled this returns a MockCuFile that serves preads from the in-memory
    copy — see `cuphoton.xdr.mock_storage`.
    """
    from .mock_storage import MockCuFile, storage_cache

    path_s = str(path)
    if storage_cache.active():
        buf = storage_cache.get_buffer(path_s)
        mf = MockCuFile(path_s, buf, storage_cache.location)
        try:
            yield mf
        finally:
            mf.close()
        return

    import kvikio

    configure_kvikio_parallelism()
    f = kvikio.CuFile(path_s, "r")
    try:
        yield f
    finally:
        f.close()


def pread_to_device(
    path: str | Path, file_offset: int, nbytes: int, out_dbuf
) -> int:
    """Read bytes from `path` at `file_offset` into `out_dbuf`.

    The destination buffer must expose `__cuda_array_interface__` and have at
    least `nbytes` of storage. Returns the number of bytes actually read.
    """
    with open_cufile(path) as f:
        fut = f.pread(out_dbuf, size=nbytes, file_offset=file_offset)
        return fut.get()


class HeapReadHandle:
    """Non-blocking handle returned by `GdsHeapLoader.load_tiles_async`.

    Holds the destination buffer and a list of `IOFuture`s. Call `wait()` to
    block until every pread has completed; returns `(d_buf, rel_offsets)`.
    Idempotent — subsequent calls are no-ops.
    """

    def __init__(self, d_buf, rel_offsets, futures):
        self.d_buf = d_buf
        self.rel_offsets = rel_offsets
        self._futures = futures
        self._waited = False

    def wait(self):
        if not self._waited:
            for fut in self._futures:
                fut.get()
            self._waited = True
            self._futures = None  # drop refs so kvikio can release resources
        return self.d_buf, self.rel_offsets


class GdsHeapLoader:
    """Load compressed tile byte ranges into one device buffer.

    A single `CuFile` is kept open across `load_tiles*()` calls on the same
    loader instance — kvikio's `CompatModeManager` setup is ~22 ms per open,
    so reopening per HDU is a measurable cost.

    Strategy (picked per call):
      - If the tiles are contiguous (`fpack`-style heap), do a single `pread`
        spanning min_offset..max_offset+length. Relative offsets are computed
        from the span start.
      - Otherwise submit one `pread` per tile and concatenate into a single
        allocation with gaps removed.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cufile = None

    def _file(self):
        if self._cufile is None:
            from .mock_storage import MockCuFile, storage_cache

            if storage_cache.active():
                buf = storage_cache.get_buffer(self.path)
                self._cufile = MockCuFile(
                    self.path, buf, storage_cache.location
                )
            else:
                import kvikio

                configure_kvikio_parallelism()
                self._cufile = kvikio.CuFile(self.path, "r")
        return self._cufile

    def close(self):
        if self._cufile is not None:
            try:
                self._cufile.close()
            finally:
                self._cufile = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def load_tiles_async(
        self,
        abs_offsets: Sequence[int],
        lengths: Sequence[int],
    ) -> HeapReadHandle:
        """Issue GDS preads and return before caller-visible completion."""
        import cupy as cp
        import numpy as np

        abs_offsets = np.asarray(abs_offsets, dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        n = abs_offsets.size
        if n == 0:
            return HeapReadHandle(
                cp.empty(0, dtype=cp.uint8),
                np.zeros(0, dtype=np.int64),
                [],
            )

        ends = abs_offsets + lengths
        is_contiguous = bool(np.all(abs_offsets[1:] == ends[:-1]) or (n == 1))

        f = self._file()

        if is_contiguous:
            span_start = int(abs_offsets[0])
            span_len = int(ends[-1]) - span_start
            d_buf = cp.empty(span_len, dtype=cp.uint8)
            fut = f.pread(d_buf, size=span_len, file_offset=span_start)
            rel = (abs_offsets - span_start).astype(np.int64)
            return HeapReadHandle(d_buf, rel, [fut])

        total = int(lengths.sum())
        d_buf = cp.empty(total, dtype=cp.uint8)
        rel = np.zeros(n, dtype=np.int64)
        cursor = 0
        futs = []
        for i in range(n):
            off = int(abs_offsets[i])
            ln = int(lengths[i])
            rel[i] = cursor
            view = d_buf[cursor : cursor + ln]
            futs.append(f.pread(view, size=ln, file_offset=off))
            cursor += ln
        return HeapReadHandle(d_buf, rel, futs)

    def load_tiles(
        self,
        abs_offsets: Sequence[int],
        lengths: Sequence[int],
    ):
        """Blocking variant: issue all preads and wait."""
        return self.load_tiles_async(abs_offsets, lengths).wait()
