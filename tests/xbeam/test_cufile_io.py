# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import errno
import os
from collections import OrderedDict
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from cuphoton.xbeam import upload_resources
from cuphoton.xbeam.cufile_io import (
    CuFileHandle,
    CuFileSession,
    CuFileUnavailable,
    _CachedRead,
)


def test_session_can_close_one_registered_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.vdif"
    path.write_bytes(b"")
    descriptor = path.open("rb", buffering=0)
    handle = CuFileHandle(path, descriptor.fileno(), 17)
    deregistered = []
    closed = []
    session = CuFileSession.__new__(CuFileSession)
    session._cufile = SimpleNamespace(handle_deregister=deregistered.append)
    session._files = [handle]
    session._cached_files = OrderedDict({path: _CachedRead(handle)})
    monkeypatch.setattr("cuphoton.xbeam.cufile_io.os.close", closed.append)

    session.close_file(handle)

    assert session._files == []
    assert session._cached_files == {}
    assert deregistered == [17]
    assert closed == [descriptor.fileno()]
    descriptor.close()


def test_session_reuses_cached_registered_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.vdif"
    handle = CuFileHandle(path, 3, 17)
    opened = []
    session = CuFileSession.__new__(CuFileSession)
    session._entered = True
    session._cached_files = OrderedDict()

    def open_file(_session, selected):
        opened.append(selected)
        return handle

    monkeypatch.setattr(CuFileSession, "open", open_file)

    assert session.open_cached(path) is handle
    assert session.open_cached(path) is handle
    assert opened == [path]


@pytest.fixture
def fake_session(monkeypatch):
    calls = []
    bindings = SimpleNamespace(
        driver_open=lambda: calls.append("driver-open"),
        driver_close=lambda: calls.append("driver-close"),
        Descr=lambda: SimpleNamespace(handle=SimpleNamespace(), ptr=1),
        FileHandleType=SimpleNamespace(OPAQUE_FD=1),
        handle_register=lambda _: len(calls),
        handle_deregister=lambda _: calls.append("deregister"),
        buf_register=lambda *args: None,
        buf_deregister=lambda _: None,
    )
    monkeypatch.setattr(
        "cuphoton.xbeam.cufile_io._cufile_module", lambda: bindings
    )
    with CuFileSession() as session:
        yield session, calls
    assert calls.count("driver-open") == calls.count("driver-close")


def test_exclusive_create_preserves_existing_file(fake_session, tmp_path):
    session, _ = fake_session
    path = tmp_path / "existing.vdix"
    path.write_bytes(b"keep me")
    with pytest.raises(CuFileUnavailable, match="File exists"):
        session.create(path, 4096)
    assert path.read_bytes() == b"keep me"


def test_create_failure_removes_only_owned_file(
    fake_session, tmp_path, monkeypatch
):
    session, _ = fake_session
    path = tmp_path / "new.vdix"

    def fail(*args):
        raise OSError("injected truncate failure")

    monkeypatch.setattr(os, "ftruncate", fail)
    with pytest.raises(CuFileUnavailable, match="truncate failure"):
        session.create(path, 4096)
    assert not path.exists()


def test_bounded_cache_never_evicts_leases(fake_session, tmp_path):
    session, calls = fake_session
    paths = [tmp_path / f"{i}.vdif" for i in range(75)]
    for path in paths:
        path.touch()
    validated = []
    with session.lease_file(paths[0]) as protected:
        for path in paths[1:]:
            with session.lease_file(
                path, validate=lambda f: validated.append(f.path)
            ):
                assert len(session._cached_files) <= 64
                assert len(session._files) <= 64
                os.fstat(protected.fd)
    assert calls.count("deregister") == 11
    assert paths[1] not in session._cached_files
    assert validated.count(paths[1]) == 2  # acquire and eviction
    with session.lease_file(paths[1]) as revisited:
        assert revisited.path == paths[1]
    session.finish_cached_reads()
    assert all(
        entry.validate is None for entry in session._cached_files.values()
    )


def test_all_handles_leased_rejects_overflow(fake_session, tmp_path):
    session, _ = fake_session
    with ExitStack() as leases:
        for i in range(64):
            path = tmp_path / f"{i}"
            path.touch()
            leases.enter_context(session.lease_file(path))
        overflow = tmp_path / "overflow"
        overflow.touch()
        with pytest.raises(RuntimeError, match="all 64"):
            with session.lease_file(overflow):
                pytest.fail("lease overflow accepted")


def test_new_window_reopens_replaced_inode(fake_session, tmp_path):
    session, _ = fake_session
    path = tmp_path / "source"
    path.touch()
    first_inode = path.stat().st_ino
    with session.lease_file(path, fingerprint=(first_inode,)) as original:
        first_fd = original.fd
    replacement = tmp_path / "replacement"
    replacement.touch()
    replacement.replace(path)
    second_inode = path.stat().st_ino
    with session.lease_file(path, fingerprint=(second_inode,)) as current:
        assert os.fstat(current.fd).st_ino == second_inode
        assert current is not original
    assert first_inode != second_inode
    assert first_fd >= 0


def test_final_validation_reports_mutation_and_clears_callbacks(
    fake_session, tmp_path
):
    session, _ = fake_session
    path = tmp_path / "source"
    path.touch()

    def validate(file):
        if os.fstat(file.fd).st_size:
            raise OSError("source changed")

    with session.lease_file(path, validate=validate):
        path.write_bytes(b"changed")
    with pytest.raises(OSError, match="source changed"):
        session.finish_cached_reads()
    assert all(
        entry.validate is None for entry in session._cached_files.values()
    )


@pytest.mark.parametrize("check", ["lease", "finish"])
@pytest.mark.parametrize("fingerprinted", [False, True])
def test_cached_source_rejects_path_replacement(
    fake_session, tmp_path, check, fingerprinted
):
    session, _ = fake_session
    path = tmp_path / "source.vdif"
    path.write_bytes(b"original")
    before = path.stat()
    reference = (
        str(path),
        0,
        1,
        before.st_size if fingerprinted else None,
        before.st_mtime_ns if fingerprinted else None,
        before.st_dev if fingerprinted else None,
        before.st_ino if fingerprinted else None,
    )
    with upload_resources._lease_validated_cufile_source(session, reference):
        pass
    replacement = tmp_path / "replacement.vdif"
    replacement.write_bytes(b"replaced")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)

    with pytest.raises(OSError, match="source changed"):
        if check == "lease":
            with upload_resources._lease_validated_cufile_source(
                session, reference
            ):
                pytest.fail("replaced source accepted")
        else:
            session.finish_cached_reads()


def test_timed_cleanup_keeps_first_error_and_closes_remaining_resources(
    monkeypatch, tmp_path
):
    from contextlib import contextmanager

    from cuphoton.xbeam import cufile_io

    labels, operations = [], []
    original_error = RuntimeError("buffer deregistration failed")

    @contextmanager
    def timing(name):
        labels.append(name)
        yield

    def deregister(pointer):
        operations.append(("buffer", pointer))
        if pointer == 2:
            raise original_error

    session = object.__new__(CuFileSession)
    session._entered = True
    session._buffers = [1, 2]
    session._files = [CuFileHandle(tmp_path / "file", 42, 7)]
    session._cached_files = OrderedDict()
    session._cufile = SimpleNamespace(
        buf_deregister=deregister,
        handle_deregister=lambda handle: operations.append(
            ("handle", handle)
        ),
    )
    monkeypatch.setattr(cufile_io, "nvtx_range", timing)
    monkeypatch.setattr(
        cufile_io.os, "close", lambda fd: operations.append(("fd", fd))
    )
    monkeypatch.setattr(
        cufile_io,
        "_release_driver",
        lambda module: operations.append(("driver", None)),
    )
    with pytest.raises(RuntimeError) as caught:
        session.close()
    assert caught.value is original_error
    assert operations == [
        ("buffer", 2),
        ("buffer", 1),
        ("handle", 7),
        ("fd", 42),
        ("driver", None),
    ]
    assert not session._entered
    assert not session._buffers and not session._files
    assert "xbeam:cufile-buffer-deregister" in labels
    assert "xbeam:cufile-driver-release" in labels
    session.close()
    assert len(operations) == 5


def test_registered_reads_support_device_offsets(tmp_path):
    calls = []
    session = CuFileSession.__new__(CuFileSession)
    session._entered = True
    session._cufile = SimpleNamespace(
        read=lambda *args: calls.append(args) or 4096,
    )
    handle = CuFileHandle(tmp_path / "raw.bin", 3, 17)
    assert session.read(handle, 65536, 4096, 8192) == 4096
    assert (
        session.read(handle, 65536, 4096, 12288, buffer_offset=4096) == 4096
    )
    assert calls == [
        (17, 65536, 4096, 8192, 0),
        (17, 65536, 4096, 12288, 4096),
    ]
    session._entered = False
    with pytest.raises(RuntimeError, match="not open"):
        session.read(handle, 65536, 4096, 8192)
    assert len(calls) == 2


def test_register_failure_on_tmpfs_names_filesystem(
    fake_session, tmp_path, monkeypatch
):
    session, _ = fake_session
    path = tmp_path / "source.vdif"
    path.touch()

    def refuse(_):
        raise Exception("HANDLE_NOT_REGISTERED")

    session._cufile.handle_register = refuse
    monkeypatch.setattr(
        "cuphoton.xbeam.cufile_io._filesystem_type", lambda _: "tmpfs"
    )
    with pytest.raises(CuFileUnavailable, match="tmpfs.*TMPDIR"):
        session.open(path)


def test_open_failure_on_tmpfs_names_filesystem(
    fake_session, tmp_path, monkeypatch
):
    session, _ = fake_session
    path = tmp_path / "source.vdif"
    path.touch()

    def refuse(*args):
        raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr("cuphoton.xbeam.cufile_io.os.open", refuse)
    monkeypatch.setattr(
        "cuphoton.xbeam.cufile_io._filesystem_type", lambda _: "tmpfs"
    )
    with pytest.raises(
        CuFileUnavailable, match="Invalid argument.*tmpfs.*TMPDIR"
    ):
        session.open(path)


def test_read_failure_on_tmpfs_names_filesystem(
    fake_session, tmp_path, monkeypatch
):
    session, _ = fake_session
    path = tmp_path / "source.vdif"
    handle = CuFileHandle(path, 3, 17)

    def fail(*args):
        raise OSError("cuFile error")

    session._cufile.read = fail
    monkeypatch.setattr(
        "cuphoton.xbeam.cufile_io._filesystem_type", lambda _: "tmpfs"
    )
    with pytest.raises(OSError, match="tmpfs.*TMPDIR"):
        session.read(handle, 0, 16, 0)


def test_failure_on_disk_filesystem_keeps_original_message(
    fake_session, tmp_path, monkeypatch
):
    session, _ = fake_session
    path = tmp_path / "source.vdif"
    path.touch()

    def refuse(_):
        raise Exception("HANDLE_NOT_REGISTERED")

    session._cufile.handle_register = refuse
    monkeypatch.setattr(
        "cuphoton.xbeam.cufile_io._filesystem_type", lambda _: "ext4"
    )
    with pytest.raises(CuFileUnavailable) as raised:
        session.open(path)
    assert "tmpfs" not in str(raised.value)


def test_filesystem_type_uses_longest_mount_prefix(tmp_path):
    from cuphoton.xbeam.cufile_io import _filesystem_type

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 8:1 / / rw - ext4 /dev/sda1 rw\n"
        f"2 1 0:30 / {tmp_path} rw - tmpfs tmpfs rw\n"
        f"3 2 0:31 / {tmp_path}/disk rw - xfs /dev/sdb rw\n"
    )
    (tmp_path / "disk").mkdir()
    assert _filesystem_type(tmp_path / "a", mountinfo) == "tmpfs"
    assert _filesystem_type(tmp_path / "disk" / "a", mountinfo) == "xfs"
    assert _filesystem_type(tmp_path / "a", tmp_path / "missing") is None


def test_filesystem_type_prefers_later_entry_for_same_mount(tmp_path):
    from cuphoton.xbeam.cufile_io import _filesystem_type

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 8:1 / / rw - ext4 /dev/sda1 rw\n"
        f"2 1 0:30 / {tmp_path} rw - xfs /dev/sdb rw\n"
        f"3 1 0:31 / {tmp_path} rw - tmpfs tmpfs rw\n"
    )
    assert _filesystem_type(tmp_path / "a", mountinfo) == "tmpfs"


def test_filesystem_type_decodes_escaped_mount_points(tmp_path):
    from cuphoton.xbeam.cufile_io import _filesystem_type

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 8:1 / / rw - ext4 /dev/sda1 rw\n"
        "2 1 0:30 / /mnt/données\\040raw rw - tmpfs tmpfs rw\n"
    )
    assert _filesystem_type("/mnt/données raw/a", mountinfo) == "tmpfs"


def test_negative_read_status_on_tmpfs_names_filesystem(
    fake_session, tmp_path, monkeypatch
):
    session, _ = fake_session
    path = tmp_path / "source.vdif"
    handle = CuFileHandle(path, 3, 17)
    session._cufile.read = lambda *args: -5
    monkeypatch.setattr(
        "cuphoton.xbeam.cufile_io._filesystem_type", lambda _: "tmpfs"
    )
    with pytest.raises(OSError, match="status -5.*tmpfs.*TMPDIR"):
        session.read(handle, 0, 16, 0)
