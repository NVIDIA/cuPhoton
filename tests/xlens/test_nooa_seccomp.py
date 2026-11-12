# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from cuphoton.xlens import _nooa_seccomp, nooa_backend


@pytest.fixture
def system_seccomp():
    if sys.platform != "linux":
        pytest.skip("system libseccomp is a Linux runtime prerequisite")
    try:
        with _nooa_seccomp.seccomp_filter():
            pass
    except _nooa_seccomp.SeccompError:
        if os.environ.get("CUPHOTON_REQUIRE_NOOA_TESTS") == "1":
            raise
        pytest.skip("requires system libseccomp")


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
