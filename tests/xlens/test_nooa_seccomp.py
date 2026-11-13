# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import ctypes
import errno
import json
import os
import platform
import signal
import subprocess
import sys

import pytest

from cuphoton.xlens import _nooa_seccomp, nooa_backend

# Keep this policy expectation independent of the implementation's rule list.
_EXPECTED_DENIED_SYSCALLS = (
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "personality",
    "add_key",
    "request_key",
    "keyctl",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
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

# Load the exported BPF directly, so these tests need neither NOOA nor bwrap.
_FILTER_LOADER = """
import ctypes
import json
import mmap
import os
import resource
import socket
import sys

class Instruction(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

class Program(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort),
                ("instructions", ctypes.POINTER(Instruction))]

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with os.fdopen(int(sys.argv[1]), "rb") as source:
    data = source.read()
instructions = (Instruction * (len(data) // 8)).from_buffer_copy(data)
program = Program(len(instructions), instructions)
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.restype = ctypes.c_int
libc.syscall.restype = ctypes.c_long
assert libc.prctl(38, 1, 0, 0, 0) == 0, ctypes.get_errno()
assert libc.prctl(22, 2, ctypes.byref(program), 0, 0) == 0, ctypes.get_errno()
"""


def _filtered_child(probe):
    with _nooa_seccomp.seccomp_filter() as descriptor:
        return subprocess.run(
            [sys.executable, "-c", _FILTER_LOADER + probe, str(descriptor)],
            pass_fds=(descriptor,),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )


@pytest.fixture
def system_seccomp():
    if sys.platform != "linux":
        pytest.skip("system libseccomp is a Linux runtime prerequisite")
    try:
        ctypes.CDLL("libseccomp.so.2")
    except OSError:
        if os.environ.get("CUPHOTON_REQUIRE_NOOA_TESTS") == "1":
            raise
        pytest.skip("requires system libseccomp")
    with _nooa_seccomp.seccomp_filter():
        pass


@pytest.fixture
def distinct_filter_errno(system_seccomp, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("syscall policy probes require an unprivileged user")
    # EPERM from capabilities or the host policy must not hide a missing rule.
    monkeypatch.setattr(
        _nooa_seccomp, "_DENIED_ACTION", 0x00050000 | errno.ENOTRECOVERABLE
    )
    return errno.ENOTRECOVERABLE


@pytest.mark.parametrize("name", _EXPECTED_DENIED_SYSCALLS)
def test_each_denied_syscall(distinct_filter_errno, name):
    library = ctypes.CDLL("libseccomp.so.2")
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    number = library.seccomp_syscall_resolve_name(name.encode())
    if number == -1 and platform.machine() in {"x86_64", "aarch64"}:
        number = {"statmount": 457, "listmount": 458}.get(name, -1)
    assert number != -1, f"unrecognized syscall: {name}"
    if number < 0:
        pytest.skip(f"{name} has no native syscall on this architecture")
    # Invalid commands, descriptors, pointers and flags are harmless even if
    # a rule is missing. For personality, -1 is the read-only query operation.
    result = _filtered_child(f"""
ctypes.set_errno(0)
result = libc.syscall({number}, *([ctypes.c_long(-1)] * 6))
print(json.dumps([result, ctypes.get_errno()]))
""")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [-1, distinct_filter_errno]


def test_allows_normal_syscalls_and_unix_sockets(system_seccomp):
    result = _filtered_child("""
assert os.getpid() > 0
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
    pass
left, right = socket.socketpair()
with left, right:
    left.sendall(b"local IPC")
    assert right.recv(9) == b"local IPC"
""")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    # Linux UAPI values also cover builds whose socket module omits AF_VSOCK.
    "family",
    [2, 10, 16, 40],
    ids=["AF_INET", "AF_INET6", "AF_NETLINK", "AF_VSOCK"],
)
@pytest.mark.parametrize("operation", ["socket", "socketpair"])
def test_denies_non_unix_sockets(distinct_filter_errno, family, operation):
    result = _filtered_child(f"""
try:
    socket.{operation}({family}, socket.SOCK_STREAM)
except OSError as exc:
    print(exc.errno)
else:
    raise AssertionError("non-Unix socket family was allowed")
""")
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == distinct_filter_errno


@pytest.mark.skipif(
    platform.machine() != "x86_64", reason="x86 foreign-ABI probes"
)
@pytest.mark.parametrize(
    "code",
    [
        bytes.fromhex("b8 14 00 00 00 cd 80 c3"),  # i386 getpid, int 0x80
        bytes.fromhex("b8 27 00 00 40 0f 05 c3"),  # x32 getpid, syscall
    ],
    ids=["i386", "x32"],
)
def test_foreign_abis_are_killed(system_seccomp, code):
    result = _filtered_child(f"""
code = {code!r}
with mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE) as page:
    page.write(code)
    address = ctypes.addressof(ctypes.c_char.from_buffer(page))
    assert libc.mprotect(ctypes.c_void_p(address), ctypes.c_size_t(len(code)),
                         mmap.PROT_READ | mmap.PROT_EXEC) == 0
    ctypes.CFUNCTYPE(ctypes.c_long)(address)()
""")
    assert result.returncode == -signal.SIGSYS, result.stderr


def test_missing_library_reports_optional_runtime_requirement(monkeypatch):
    def missing_library(name):
        raise OSError("private loader path /home/user/environment")

    monkeypatch.setattr(_nooa_seccomp.ctypes, "CDLL", missing_library)
    with pytest.raises(_nooa_seccomp.SeccompError) as caught:
        with _nooa_seccomp.seccomp_filter():
            raise AssertionError("missing library must not start a worker")
    assert str(caught.value) == "NOOA requires the system libseccomp library"
    with pytest.raises(nooa_backend.NooaError, match="system libseccomp"):
        nooa_backend._probe_namespaces("unused-bwrap")


@pytest.mark.parametrize("fail_inside", [False, True])
def test_filter_descriptor_closes_on_success_and_failure(
    system_seccomp, fail_inside
):
    descriptor = None
    try:
        with _nooa_seccomp.seccomp_filter() as descriptor:
            assert os.read(descriptor, 8)
            if fail_inside:
                raise RuntimeError("caller failed")
    except RuntimeError:
        assert fail_inside
    assert descriptor is not None
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_spawn_failure_closes_filter_descriptor(system_seccomp, monkeypatch):
    descriptors = []
    monkeypatch.setattr(nooa_backend, "_command", lambda: ["bwrap"])

    async def fail_start(*command, **kwargs):
        descriptor = kwargs["pass_fds"][0]
        descriptors.append(descriptor)
        assert command[:3] == ("bwrap", "--seccomp", str(descriptor))
        assert os.fstat(descriptor).st_size > 0
        raise OSError("failed startup")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_start)
    with pytest.raises(nooa_backend.NooaError, match="NOOA worker stopped"):
        asyncio.run(
            nooa_backend.run_nooa(
                {"samples": []},
                "Question",
                "Instructions",
                request=lambda payload: None,
                progress=lambda event: None,
                max_iterations=1,
                max_tool_calls=1,
                timeout_seconds=5,
                max_context_bytes=1024,
                max_response_bytes=1024,
            )
        )
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
