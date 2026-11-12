# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build a NOOA syscall filter using the host's libseccomp.

The filter is passed to bubblewrap before worker startup. It supplements the
namespace, capability and resource limits; it is not a complete syscall allow
list. No library code or compiled filter is distributed with cuPhoton.
"""

from __future__ import annotations

import ctypes
import errno
import socket
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager


class SeccompError(RuntimeError):
    """A fixed diagnostic that does not expose host loader paths."""


class _ArgCmp(ctypes.Structure):
    """The libseccomp scmp_arg_cmp layout for argument-filtered rules."""

    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


# SCMP_ACT_ERRNO(EPERM).
_DENIED_ACTION = 0x00050000 | errno.EPERM


_DENIED_SYSCALLS = (
    # Kernel instrumentation and access to other processes.
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "personality",
    # Keyrings and asynchronous kernel I/O are not used by the worker.
    "add_key",
    "request_key",
    "keyctl",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    # Namespace changes and mount metadata could expose host mount sources.
    "unshare",
    "setns",
    "mount",
    "umount2",
    "pivot_root",
    "open_by_handle_at",
    "open_tree",
    "move_mount",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "mount_setattr",
    "statmount",
    "listmount",
    # The process has no need to administer the host kernel.
    "init_module",
    "finit_module",
    "delete_module",
    "kexec_load",
    "kexec_file_load",
    "reboot",
    "swapon",
    "swapoff",
    "acct",
)


@contextmanager
def seccomp_filter() -> Iterator[int]:
    """Yield a readable filter FD, closing it on every exit path."""
    try:
        library = ctypes.CDLL("libseccomp.so.2")
        library.seccomp_init.argtypes = [ctypes.c_uint32]
        library.seccomp_init.restype = ctypes.c_void_p
        library.seccomp_release.argtypes = [ctypes.c_void_p]
        library.seccomp_release.restype = None
        library.seccomp_arch_native.argtypes = []
        library.seccomp_arch_native.restype = ctypes.c_uint32
        library.seccomp_attr_set.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint32,
        ]
        library.seccomp_attr_set.restype = ctypes.c_int
        library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        library.seccomp_syscall_resolve_name.restype = ctypes.c_int
        library.seccomp_rule_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        library.seccomp_rule_add.restype = ctypes.c_int
        library.seccomp_rule_add_array.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(_ArgCmp),
        ]
        library.seccomp_rule_add_array.restype = ctypes.c_int
        library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.seccomp_export_bpf.restype = ctypes.c_int
    except (OSError, AttributeError) as exc:
        raise SeccompError(
            "NOOA requires the system libseccomp library"
        ) from exc

    # SCMP_ACT_ALLOW; unknown ABIs are killed instead of bypassing rules.
    context = library.seccomp_init(0x7FFF0000)
    if not context:
        raise SeccompError("NOOA could not initialize syscall filtering")
    try:
        # SCMP_FLTATR_ACT_BADARCH = 2, SCMP_ACT_KILL_PROCESS = 0x80000000.
        if library.seccomp_attr_set(context, 2, 0x80000000) != 0:
            raise SeccompError("NOOA could not configure syscall filtering")
        architecture = library.seccomp_arch_native()
        for name in _DENIED_SYSCALLS:
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number == -1 and name in {"statmount", "listmount"}:
                # Ubuntu 24.04's libseccomp predates these syscalls. Their
                # x86-64 and AArch64 UAPI numbers are both 457 and 458:
                # https://github.com/torvalds/linux/blob/v6.8/include/uapi/asm-generic/unistd.h
                # https://github.com/torvalds/linux/blob/v6.8/arch/x86/entry/syscalls/syscall_64.tbl
                if architecture not in {0xC000003E, 0xC00000B7}:
                    raise SeccompError(
                        "NOOA requires libseccomp with mount-query support "
                        "on this architecture"
                    )
                number = 457 if name == "statmount" else 458
            if number == -1:
                raise SeccompError(
                    "NOOA requires a newer system libseccomp library"
                )
            # Negative pseudo-syscall numbers are valid
            # libseccomp inputs for calls absent from the native architecture.
            if (
                library.seccomp_rule_add(context, _DENIED_ACTION, number, 0)
                != 0
            ):
                raise SeccompError(
                    "NOOA could not configure syscall filtering"
                )
        # SCMP_CMP_NE = 1. asyncio needs Unix socketpairs, but other address
        # families (including AF_VSOCK) need not follow network namespaces.
        family = _ArgCmp(0, 1, socket.AF_UNIX, 0)
        for name in ("socket", "socketpair"):
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number == -1:
                raise SeccompError(
                    "NOOA requires a newer system libseccomp library"
                )
            if (
                library.seccomp_rule_add_array(
                    context, _DENIED_ACTION, number, 1, ctypes.byref(family)
                )
                != 0
            ):
                raise SeccompError(
                    "NOOA could not configure syscall filtering"
                )
        with tempfile.TemporaryFile() as program:
            if library.seccomp_export_bpf(context, program.fileno()) != 0:
                raise SeccompError("NOOA could not export syscall filtering")
            program.seek(0)
            yield program.fileno()
    finally:
        library.seccomp_release(context)
