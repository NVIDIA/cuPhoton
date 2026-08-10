# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mock storage cache for benchmarking the GPU FITS reader without disk I/O.

When the cache is enabled every ``kvikio.CuFile.pread`` call on a registered
file is served from an in-memory copy instead of hitting storage. Lets you
measure decompression + kernel cost in isolation, independent of disk I/O.

Usage:

    from cuphoton.xdr import batch_to_device, mock_storage

    with mock_storage(location="device"):
        for _ in range(N):
            # First iteration populates the cache; the rest run from
            # RAM/VRAM.
            (img,) = batch_to_device([path], hdu_indices=[1])

Location semantics:

- ``"device"`` — whole file cached as a ``cupy.ndarray`` on GPU. Subsequent
  preads become D→D memcpy at HBM bandwidth. Isolates the decompression +
  kernel cost; models an "infinitely fast" GDS path.
- ``"host"`` — whole file cached in pinned host memory. Subsequent preads
  become H→D transfers at PCIe bandwidth — models what a properly-working
  GDS path would deliver on this hardware.

Can also be enabled transparently by setting
``CUPHOTON_XDR_MOCK_STORAGE`` to ``device`` or ``host``. This is
useful for benchmarks that do not explicitly select mock storage.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


class _ImmediateFuture:
    """Drop-in for ``kvikio.cufile.IOFuture`` — already-completed."""

    __slots__ = ("_nbytes",)

    def __init__(self, nbytes: int):
        self._nbytes = int(nbytes)

    def get(self) -> int:
        return self._nbytes


class StorageCache:
    """In-memory cache of FITS file bytes.

    Not thread-safe for enable/disable/clear; benchmark-only utility.
    """

    def __init__(self):
        self._enabled = False
        self._location = "device"
        self._cache = {}
        # path (str) -> buffer on device or pinned host

    # --- lifecycle ---------------------------------------------------------

    def enable(self, location: str = "device") -> None:
        if location not in ("device", "host"):
            raise ValueError(
                f"location must be 'device' or 'host', got {location!r}"
            )
        self._enabled = True
        self._location = location

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self._cache.clear()

    def active(self) -> bool:
        return self._enabled

    @property
    def location(self) -> str:
        return self._location

    def cached_paths(self) -> list[str]:
        return list(self._cache.keys())

    # --- population --------------------------------------------------------

    def preload(self, path: str | Path) -> None:
        """Force the cache to hold this file's bytes now."""
        self.get_buffer(str(path))

    def get_buffer(self, path: str | Path):
        path = str(path)
        buf = self._cache.get(path)
        if buf is None:
            buf = self._load(path)
            self._cache[path] = buf
        return buf

    def _load(self, path: str):
        nbytes = os.path.getsize(path)
        if self._location == "device":
            import cupy as cp
            import kvikio

            buf = cp.empty(nbytes, dtype=cp.uint8)
            # Use a real kvikio read for the one-time population.
            with kvikio.CuFile(path, "r") as f:
                fut = f.pread(buf, size=nbytes, file_offset=0)
                fut.get()
            return buf
        else:  # "host" — pinned host memory
            import cupy as cp
            import numpy as np

            mem = cp.cuda.alloc_pinned_memory(nbytes)
            host = np.frombuffer(mem, dtype=np.uint8, count=nbytes)
            with open(path, "rb") as f:
                # readinto may not fill in one call on some filesystems; loop.
                pos = 0
                view = memoryview(host)
                while pos < nbytes:
                    got = f.readinto(view[pos:])
                    if not got:
                        raise IOError(
                            f"short read on {path} at {pos}/{nbytes}"
                        )
                    pos += got
            return host


# Module-level singleton.
storage_cache = StorageCache()


class MockCuFile:
    """Drop-in ``kvikio.CuFile`` served from a StorageCache."""

    __slots__ = ("path", "_cache_buffer", "_location")

    def __init__(self, path: str, cache_buffer, location: str):
        self.path = path
        self._cache_buffer = cache_buffer
        self._location = location

    def pread(self, buf, size=None, file_offset: int = 0, task_size=None):
        import cupy as cp

        dst = buf.view(cp.uint8).ravel() if hasattr(buf, "view") else buf
        nbytes = int(size) if size is not None else dst.nbytes

        if self._location == "device":
            # Device cache: D→D copy (HBM-bandwidth).
            src = self._cache_buffer[file_offset : file_offset + nbytes]
            dst[:nbytes] = src
        else:
            # Host pinned cache: H→D async copy (PCIe).
            src = self._cache_buffer[file_offset : file_offset + nbytes]
            # cupy will issue cudaMemcpyAsync on the current stream.
            dst[:nbytes] = cp.asarray(src)
        return _ImmediateFuture(nbytes)

    def read(self, buf, size=None, file_offset: int = 0, task_size=None):
        return self.pread(
            buf,
            size=size,
            file_offset=file_offset,
            task_size=task_size,
        ).get()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def mock_storage(location: str = "device"):
    """Context manager: enable the cache for the duration of the block.

    On exit the cache is disabled and cleared; if the cache was already
    enabled when this was called, the prior enable state is restored but
    the buffers populated inside the block are still cleared (to avoid
    leaking device memory across benchmark runs).
    """
    prev_enabled = storage_cache._enabled
    prev_location = storage_cache._location
    storage_cache.enable(location)
    try:
        yield storage_cache
    finally:
        storage_cache.clear()
        if prev_enabled:
            storage_cache._location = prev_location
        else:
            storage_cache.disable()


# Environment-variable opt-in:
# CUPHOTON_XDR_MOCK_STORAGE=device|host. Set it before importing.
# This is useful for running existing benchmarks without edits.
_env = os.environ.get("CUPHOTON_XDR_MOCK_STORAGE", "").lower().strip()
if _env in ("device", "host"):
    storage_cache.enable(_env)
