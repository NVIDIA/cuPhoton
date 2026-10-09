# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only connection checks for an externally managed MPS v2 service."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def require_mps_client(pid: int) -> dict[str, str | int]:
    """Confirm an initialized CUDA worker appears in the selected service.

    The caller sets ``CUDA_MPS_PIPE_DIRECTORY`` before initializing CUDA.
    Client and service must report PIDs in the same namespace. This function
    never starts a daemon, terminates a client, or changes GPU policy.
    """

    if type(pid) is not int or pid <= 0:
        raise ValueError("MPS client PID must be a positive integer")
    directory = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if not directory or not Path(directory).is_dir():
        raise RuntimeError("selected MPS pipe directory does not exist")
    deadline = time.monotonic() + 10.0

    def query(command: str) -> list[int]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("MPS connection verification timed out")
        try:
            result = subprocess.run(
                ["nvidia-cuda-mps-control"],
                input=command + "\n",
                text=True,
                capture_output=True,
                timeout=remaining,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "cannot query selected MPS v2 service"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "cannot query selected MPS v2 service: "
                + result.stderr.strip()
            )
        tokens = result.stdout.split()
        if any(
            not token.isascii() or not token.isdecimal() for token in tokens
        ):
            raise RuntimeError(
                "unexpected MPS v2 PID list: " + result.stdout.strip()
            )
        return [int(token) for token in tokens]

    for server_pid in query("get_server_list"):
        if server_pid > 0 and pid in query(f"get_client_list {server_pid}"):
            return {
                "pipe_directory": directory,
                "server_pid": server_pid,
                "client_pid": pid,
            }
    raise RuntimeError(
        f"worker PID {pid} is not connected to the selected MPS v2 service"
    )
