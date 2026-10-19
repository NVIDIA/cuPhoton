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
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_KVIKIO_DEFAULTS_LOCK = threading.Lock()
_KVIKIO_NUM_THREADS: int | None = None


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
    """Initialize the shared KvikIO pool before the first cuPhoton reader.

    Keep an explicit KVIKIO_NTHREADS setting; otherwise use available CPUs.
    Resetting this global pool while another reader uses it is unsafe.
    """
    global _KVIKIO_NUM_THREADS
    with _KVIKIO_DEFAULTS_LOCK:
        if _KVIKIO_NUM_THREADS is None:
            import kvikio.defaults as defaults

            num_threads = int(defaults.get("num_threads"))
            if "KVIKIO_NTHREADS" not in os.environ:
                desired_threads = available_cpu_cores()
                if num_threads != desired_threads:
                    defaults.set("num_threads", desired_threads)
                    num_threads = int(defaults.get("num_threads"))
            _KVIKIO_NUM_THREADS = num_threads
        return _KVIKIO_NUM_THREADS


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
            _close_cufile(mf)
        return

    import kvikio

    configure_kvikio_parallelism()
    f = kvikio.CuFile(path_s, "r")
    try:
        yield f
    finally:
        _close_cufile(f)


class _UnresolvedRead:
    """A failed KvikIO get consumes the only completion state we can query."""

    done = False

    def __init__(self, file, reads):
        self.file = file
        self.reads = reads


def _quarantine_read_failure(reads, owners, file, error):
    from .prefetch import (
        _UNRESOLVED_GPU_BATCHES,
        _UNRESOLVED_GPU_BATCHES_LOCK,
        _GpuBatchHandle,
        _quarantine_gpu_batches,
    )

    with _UNRESOLVED_GPU_BATCHES_LOCK:
        if any(
            isinstance(batch.event, _UnresolvedRead)
            and batch.event.reads is reads
            for batch in _UNRESOLVED_GPU_BATCHES
        ):
            return
        completion = _UnresolvedRead(file, reads)
        _quarantine_gpu_batches(
            [
                _GpuBatchHandle(
                    event=completion,
                    keepalive=[file, *owners, reads, error.__traceback__],
                    stream=completion,
                )
            ]
        )


def _close_cufile(file):
    """Defer close when KvikIO tasks may still reference the file handle."""
    from .prefetch import (
        _UNRESOLVED_GPU_BATCHES,
        _UNRESOLVED_GPU_BATCHES_LOCK,
    )

    with _UNRESOLVED_GPU_BATCHES_LOCK:
        if any(
            isinstance(batch.event, _UnresolvedRead)
            and batch.event.file is file
            for batch in _UNRESOLVED_GPU_BATCHES
        ):
            return
        file.close()


@contextmanager
def _read_submission(reads, *, owners, file):
    """Retain owners if submission, handoff, or gate exit fails."""
    from .prefetch import _gpu_submission_guard

    entered = False
    try:
        with _gpu_submission_guard():
            entered = True
            try:
                yield
            except BaseException as error:
                # Publish before another reader can acquire the gate.
                _quarantine_read_failure(reads, owners, file, error)
                raise
    except BaseException as error:
        if entered:
            # Also retain owners if leaving the gate itself is interrupted.
            _quarantine_read_failure(reads, owners, file, error)
        raise


def _pread_into(file, out, nbytes, file_offset):
    reads = []
    try:
        with _read_submission(reads, owners=(out,), file=file):
            reads.append(
                (
                    file.pread(out, size=nbytes, file_offset=file_offset),
                    nbytes,
                )
            )
        _wait_for_reads(reads, owners=(out,), file=file)
    except BaseException as error:
        if reads:
            _quarantine_read_failure(reads, (out,), file, error)
        raise


def pread_to_device(
    path: str | Path, file_offset: int, nbytes: int, out_dbuf
) -> int:
    """Read bytes from `path` at `file_offset` into `out_dbuf`.

    The destination buffer must expose `__cuda_array_interface__` and have at
    least `nbytes` of storage. Returns `nbytes`; an incomplete read raises.
    """
    with open_cufile(path) as f:
        _pread_into(f, out_dbuf, nbytes, file_offset)
        return nbytes


def _wait_for_reads(reads, *, owners=(), file=None):
    """Check counts; quarantine owners permanently if completion is unknown.

    KvikIO consumes its future even when get raises, and sibling tasks may
    still be writing. Neither polling that future nor closing CuFile drains
    those tasks. A failed wait therefore requires a process restart; callers
    must not reuse their destination buffers after it.
    """
    first_error = None
    unresolved = []
    for index, (future, expected) in enumerate(reads):
        try:
            actual = future.get()
        except BaseException as exc:
            unresolved.append((future, expected))
            if not isinstance(exc, Exception):
                unresolved.extend(reads[index + 1 :])
                first_error = exc
                break
            if first_error is None:
                first_error = exc
        else:
            if actual != expected and first_error is None:
                first_error = OSError(
                    "xDR read size differs: "
                    f"expected {expected}, got {actual}"
                )
    reads[:] = unresolved
    if unresolved:
        _quarantine_read_failure(reads, owners, file, first_error)
    if first_error is not None:
        raise first_error


class HeapReadHandle:
    """Non-blocking handle returned by `GdsHeapLoader.load_tiles_async`.

    Holds the destination buffer and `(IOFuture, expected_bytes)` pairs.
    Call `wait()` to check completion and return `(d_buf, rel_offsets)`.
    Repeated calls return the same buffers or raise the cached I/O error.
    """

    def __init__(self, d_buf, rel_offsets, reads, *, file=None):
        self.d_buf = d_buf
        self.rel_offsets = rel_offsets
        self._reads = reads
        self._waited = False
        self._error = None
        self._file = file

    def wait(self):
        if not self._waited and self._error is None:
            try:
                _wait_for_reads(
                    self._reads, owners=(self.d_buf,), file=self._file
                )
            except BaseException as exc:
                self._error = exc
                if self._reads:
                    _quarantine_read_failure(
                        self._reads, (self.d_buf,), self._file, exc
                    )
            self._waited = not self._reads
            if self._waited:
                self._reads = None  # release completed KvikIO resources
        if self._error is not None:
            raise self._error
        return self.d_buf, self.rel_offsets

    @property
    def done(self) -> bool:
        """Report I/O completion without waiting on any read."""
        if self._error is not None and not self._waited:
            # get() consumed the failed future, so done() cannot be queried.
            return False
        return self._waited or all(fut.done() for fut, _ in self._reads)


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
                _close_cufile(self._cufile)
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

        offset_array = np.asarray(abs_offsets, dtype=np.int64)
        length_array = np.asarray(lengths, dtype=np.int64)
        n = offset_array.size
        if n == 0:
            return HeapReadHandle(
                cp.empty(0, dtype=cp.uint8),
                np.zeros(0, dtype=np.int64),
                [],
            )

        ends = offset_array + length_array
        is_contiguous = bool(
            np.all(offset_array[1:] == ends[:-1]) or (n == 1)
        )

        f = self._file()

        if is_contiguous:
            span_start = int(offset_array[0])
            span_len = int(ends[-1]) - span_start
            d_buf = cp.empty(span_len, dtype=cp.uint8)
            reads: list[tuple[Any, int]] = []
            with _read_submission(reads, owners=(d_buf,), file=f):
                reads.append(
                    (
                        f.pread(d_buf, size=span_len, file_offset=span_start),
                        span_len,
                    )
                )
                rel = (offset_array - span_start).astype(np.int64)
                return HeapReadHandle(d_buf, rel, reads, file=f)

        total = int(length_array.sum())
        d_buf = cp.empty(total, dtype=cp.uint8)
        rel = np.zeros(n, dtype=np.int64)
        cursor = 0
        futs: list[tuple[Any, int]] = []
        with _read_submission(futs, owners=(d_buf,), file=f):
            for i in range(n):
                off = int(offset_array[i])
                ln = int(length_array[i])
                rel[i] = cursor
                view = d_buf[cursor : cursor + ln]
                futs.append((f.pread(view, size=ln, file_offset=off), ln))
                cursor += ln
            return HeapReadHandle(d_buf, rel, futs, file=f)

    def load_tiles(
        self,
        abs_offsets: Sequence[int],
        lengths: Sequence[int],
    ):
        """Blocking variant: issue all preads and wait."""
        return self.load_tiles_async(abs_offsets, lengths).wait()
