# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from cuphoton.core.mps import require_mps_client


def test_mps_verification_requires_actual_client_membership(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    commands = []

    def control(argv, **kwargs):
        assert argv == ["nvidia-cuda-mps-control"]
        assert 0 < kwargs["timeout"] <= 10
        command = kwargs["input"]
        commands.append(command)
        output = {
            "get_server_list\n": "41\n42\n",
            "get_client_list 41\n": "51\n",
            "get_client_list 42\n": "52\n53\n",
        }[command]
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", control)
    assert require_mps_client(53) == {
        "pipe_directory": str(tmp_path),
        "server_pid": 42,
        "client_pid": 53,
    }
    assert len(commands) == 3
    with pytest.raises(RuntimeError, match="not connected"):
        require_mps_client(99)


@pytest.mark.parametrize("failure", ["unavailable", "malformed", "timeout"])
def test_mps_query_failure_is_not_silent_fallback(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))

    def control(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return SimpleNamespace(
            returncode=1 if failure == "unavailable" else 0,
            stdout="Server error",
            stderr="No daemon",
        )

    monkeypatch.setattr(subprocess, "run", control)
    with pytest.raises(RuntimeError, match="MPS"):
        require_mps_client(53)
