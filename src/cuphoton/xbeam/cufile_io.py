# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small, optional CUDA Python cuFile resource wrapper."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import ModuleType

from .profiling import nvtx_range


class CuFileUnavailable(RuntimeError):
    """The requested cuFile data path cannot be initialized."""


_RAM_FILESYSTEMS = frozenset({"tmpfs", "ramfs"})
# The kernel escapes space, tab, newline and backslash in mountinfo as \ooo.
_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")


def _filesystem_type(
    path: str | os.PathLike[str],
    mountinfo: str | os.PathLike[str] = "/proc/self/mountinfo",
) -> str | None:
    """Return the filesystem type mounted at path, or None if unknown."""
    try:
        target = os.path.realpath(path)
        best_mount = ""
        best_type = None
        with open(mountinfo, encoding="utf-8") as lines:
            for line in lines:
                head, separator, tail = line.partition(" - ")
                fields = head.split()
                if not separator or len(fields) < 5 or not tail.split():
                    continue
                mount = _MOUNTINFO_ESCAPE.sub(
                    lambda match: chr(int(match[1], 8)), fields[4]
                )
                inside = target == mount or target.startswith(
                    mount.rstrip("/") + "/"
                )
                if inside and len(mount) >= len(best_mount):
                    best_mount, best_type = mount, tail.split()[0]
        return best_type
    except (OSError, UnicodeError):
        return None


def _ram_filesystem_hint(path: str | os.PathLike[str]) -> str:
    """Explain a cuFile failure when the file lives in RAM-backed storage."""
    kind = _filesystem_type(path)
    if kind not in _RAM_FILESYSTEMS:
        return ""
    return (
        f"; {path} is on {kind}, which cuFile does not support. Use a "
        "disk-backed filesystem, for example by pointing TMPDIR (or the "
        "input path) at local disk storage"
    )


_DRIVER_LOCK = Lock()
_DRIVER_USERS = 0


def _cufile_module() -> ModuleType:
    try:
        from cuda.bindings import cufile
    except (ImportError, OSError) as error:
        raise CuFileUnavailable(
            "cuFile ingest requires the optional cuda-python bindings and "
            "libcufile"
        ) from error
    return cufile


def _acquire_driver(cufile: ModuleType) -> None:
    global _DRIVER_USERS
    with _DRIVER_LOCK:
        if _DRIVER_USERS == 0:
            try:
                with nvtx_range("xbeam:cufile-driver-open"):
                    cufile.driver_open()
            except Exception as error:
                raise CuFileUnavailable(
                    f"cuFile driver initialization failed: {error}"
                ) from error
        _DRIVER_USERS += 1


def _release_driver(cufile: ModuleType) -> None:
    global _DRIVER_USERS
    with _DRIVER_LOCK:
        if _DRIVER_USERS <= 0:
            raise RuntimeError("unbalanced cuFile driver release")
        _DRIVER_USERS -= 1
        if _DRIVER_USERS == 0:
            with nvtx_range("xbeam:cufile-driver-close"):
                cufile.driver_close()


@dataclass(frozen=True, slots=True)
class CuFileHandle:
    """Registered cuFile handle paired with its owned POSIX descriptor."""

    path: Path
    fd: int
    handle: int


@dataclass(slots=True)
class _CachedRead:
    file: CuFileHandle
    fingerprint: tuple | None = None
    validate: Callable[[CuFileHandle], None] | None = None
    leases: int = 0


class CuFileSession:
    """Own registered files and device buffers for one ingest worker."""

    def __init__(self) -> None:
        self._cufile = _cufile_module()
        self._files: list[CuFileHandle] = []
        self._cached_files: OrderedDict[Path, _CachedRead] = OrderedDict()
        self._buffers: list[int] = []
        self._entered = False

    def __enter__(self) -> CuFileSession:
        if self._entered:
            raise RuntimeError("cuFile session is already open")
        with nvtx_range("xbeam:cufile-driver-acquire"):
            _acquire_driver(self._cufile)
        self._entered = True
        return self

    def open(self, path: str | os.PathLike[str]) -> CuFileHandle:
        if not self._entered:
            raise RuntimeError("cuFile session is not open")
        direct = getattr(os, "O_DIRECT", None)
        if direct is None:
            raise CuFileUnavailable("cuFile ingest requires Linux O_DIRECT")
        selected = Path(path)
        try:
            fd = os.open(selected, os.O_RDONLY | direct)
        except OSError as error:
            raise CuFileUnavailable(
                f"cannot open {selected} for direct cuFile input: {error}"
                f"{_ram_filesystem_hint(selected)}"
            ) from error
        try:
            descriptor = self._cufile.Descr()
            # Initialize every union member, including unused pointer bytes.
            descriptor.type = self._cufile.FileHandleType.OPAQUE_FD
            descriptor.fs_ops = 0
            descriptor.handle.handle = 0
            descriptor.handle.fd = fd
            handle = int(self._cufile.handle_register(descriptor.ptr))
        except Exception as error:
            os.close(fd)
            raise CuFileUnavailable(
                f"cuFile cannot register {selected}: {error}"
                f"{_ram_filesystem_hint(selected)}"
            ) from error
        result = CuFileHandle(selected, fd, handle)
        self._files.append(result)
        return result

    def open_cached(self, path: str | os.PathLike[str]) -> CuFileHandle:
        """Return a cached handle valid until subsequent cache operations.

        Use lease_file to protect a handle against eviction, or open for
        explicitly owned session-lifetime registrations.
        """

        return self._cached_read(Path(path)).file

    def _cached_read(
        self, selected: Path, fingerprint: tuple | None = None
    ) -> _CachedRead:

        if not self._entered:
            raise RuntimeError("cuFile session is not open")
        cached = self._cached_files.get(selected)
        if cached is not None:
            if fingerprint is not None and cached.fingerprint != fingerprint:
                if cached.leases:
                    raise OSError("cuFile source changed while leased")
                self._evict_cached(cached)
                cached = None
        if cached is not None:
            self._cached_files.move_to_end(selected)
            return cached
        if len(self._cached_files) >= 64:
            victim = next(
                (
                    entry
                    for entry in self._cached_files.values()
                    if not entry.leases
                ),
                None,
            )
            if victim is None:
                raise RuntimeError("all 64 cached cuFile handles are leased")
            self._evict_cached(victim)
        result = _CachedRead(self.open(selected), fingerprint)
        self._cached_files[selected] = result
        return result

    def _evict_cached(self, entry: _CachedRead) -> None:
        if entry.leases:
            raise RuntimeError("cannot evict a leased cuFile handle")
        try:
            if entry.validate is not None:
                entry.validate(entry.file)
        finally:
            self.close_file(entry.file)

    @contextmanager
    def lease_file(
        self,
        path: str | os.PathLike[str],
        *,
        fingerprint: tuple | None = None,
        validate: Callable[[CuFileHandle], None] | None = None,
    ) -> Iterator[CuFileHandle]:
        """Hold one bounded cached registration for synchronous reads."""

        entry = self._cached_read(Path(path), fingerprint)
        entry.leases += 1
        try:
            if validate is not None:
                validate(entry.file)
                entry.validate = validate
            yield entry.file
        finally:
            entry.leases -= 1

    def finish_cached_reads(self) -> None:
        """Validate this window's sources, then forget their validators."""

        first_error = None
        for entry in self._cached_files.values():
            try:
                if entry.leases:
                    raise RuntimeError(
                        "cuFile reads still have active leases"
                    )
                if entry.validate is not None:
                    entry.validate(entry.file)
            except BaseException as error:
                first_error = first_error or error
            finally:
                entry.validate = None
        if first_error is not None:
            raise first_error

    def create(
        self,
        path: str | os.PathLike[str],
        size: int,
        *,
        header: bytes = b"",
    ) -> CuFileHandle:
        """Create and register one preallocated direct-I/O output file."""

        if not self._entered:
            raise RuntimeError("cuFile session is not open")
        direct = getattr(os, "O_DIRECT", None)
        if direct is None:
            raise CuFileUnavailable("cuFile output requires Linux O_DIRECT")
        if size <= 0 or len(header) > size:
            raise ValueError("invalid cuFile output extent")
        selected = Path(path)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        created = False
        try:
            # Populate the small metadata header before enabling O_DIRECT.
            fd = os.open(selected, flags, 0o644)
            created = True
            try:
                os.ftruncate(fd, size)
                if header and os.pwrite(fd, header, 0) != len(header):
                    raise OSError("short sidecar metadata write")
                os.fsync(fd)
            finally:
                os.close(fd)
            fd = os.open(selected, os.O_RDWR | direct)
        except OSError as error:
            if created:
                try:
                    selected.unlink()
                except OSError:
                    pass
            raise CuFileUnavailable(
                f"cannot create {selected} for direct cuFile output: {error}"
            ) from error
        try:
            descriptor = self._cufile.Descr()
            descriptor.type = self._cufile.FileHandleType.OPAQUE_FD
            descriptor.fs_ops = 0
            descriptor.handle.handle = 0
            descriptor.handle.fd = fd
            handle = int(self._cufile.handle_register(descriptor.ptr))
        except Exception as error:
            os.close(fd)
            selected.unlink(missing_ok=True)
            raise CuFileUnavailable(
                f"cuFile cannot register output {selected}: {error}"
            ) from error
        result = CuFileHandle(selected, fd, handle)
        self._files.append(result)
        return result

    def register_buffer(self, pointer: int, size: int) -> None:
        if not self._entered:
            raise RuntimeError("cuFile session is not open")
        if pointer <= 0 or size <= 0:
            raise ValueError(
                "cuFile buffer pointer and size must be positive"
            )
        try:
            with nvtx_range("xbeam:cufile-buffer-register"):
                self._cufile.buf_register(pointer, size, 0)
        except Exception as error:
            raise CuFileUnavailable(
                f"cuFile cannot register a {size}-byte GPU buffer: {error}"
            ) from error
        self._buffers.append(pointer)

    def close_file(self, file: CuFileHandle) -> None:
        """Deregister and close one file before the session itself ends."""

        try:
            index = self._files.index(file)
        except ValueError as error:
            raise ValueError(
                "cuFile handle is not open in this session"
            ) from error
        self._files.pop(index)
        for path, cached in tuple(self._cached_files.items()):
            if cached.file is file:
                if cached.leases:
                    self._files.insert(index, file)
                    raise RuntimeError("cannot close a leased cuFile handle")
                del self._cached_files[path]
        try:
            with nvtx_range("xbeam:cufile-handle-deregister"):
                self._cufile.handle_deregister(file.handle)
        finally:
            with nvtx_range("xbeam:cufile-fd-close"):
                os.close(file.fd)

    def read(
        self,
        file: CuFileHandle,
        pointer: int,
        size: int,
        file_offset: int,
        *,
        buffer_offset: int = 0,
    ) -> int:
        if not self._entered:
            raise RuntimeError("cuFile session is not open")
        try:
            received = int(
                self._cufile.read(
                    file.handle, pointer, size, file_offset, buffer_offset
                )
            )
        except Exception as error:
            raise OSError(
                f"cuFile read failed for {file.path} at byte "
                f"{file_offset}: {error}{_ram_filesystem_hint(file.path)}"
            ) from error
        if received < 0:
            raise OSError(
                f"cuFile read failed for {file.path} at byte "
                f"{file_offset}: status {received}"
                f"{_ram_filesystem_hint(file.path)}"
            )
        return received

    def write(
        self,
        file: CuFileHandle,
        pointer: int,
        size: int,
        file_offset: int,
    ) -> int:
        """Write a device buffer directly to a registered output file."""

        if not self._entered:
            raise RuntimeError("cuFile session is not open")
        try:
            written = int(
                self._cufile.write(file.handle, pointer, size, file_offset, 0)
            )
        except Exception as error:
            raise OSError(
                f"cuFile write failed for {file.path} at byte "
                f"{file_offset}: {error}"
            ) from error
        if written < 0:
            raise OSError(
                f"cuFile write failed for {file.path} at byte "
                f"{file_offset}: status {written}"
            )
        return written

    def close(self) -> None:
        if not self._entered:
            return
        first_error: BaseException | None = None
        for pointer in reversed(self._buffers):
            try:
                with nvtx_range("xbeam:cufile-buffer-deregister"):
                    self._cufile.buf_deregister(pointer)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._buffers.clear()
        for file in reversed(self._files):
            try:
                with nvtx_range("xbeam:cufile-handle-deregister"):
                    self._cufile.handle_deregister(file.handle)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                try:
                    with nvtx_range("xbeam:cufile-fd-close"):
                        os.close(file.fd)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        self._files.clear()
        self._cached_files.clear()
        try:
            with nvtx_range("xbeam:cufile-driver-release"):
                _release_driver(self._cufile)
        except BaseException as error:
            if first_error is None:
                first_error = error
        self._entered = False
        if first_error is not None:
            raise first_error

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise


__all__ = [
    "CuFileHandle",
    "CuFileSession",
    "CuFileUnavailable",
]
