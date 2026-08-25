# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "cuphoton-openmpi-rank-exec"


def _environment(**updates: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "GPU-a,MIG-b,2,3",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "2",
        }
    )
    environment.update(updates)
    return environment


def _run(
    *command: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *command],
        check=False,
        capture_output=True,
        text=True,
        env=environment or _environment(),
    )


def test_script_is_executable_and_has_valid_bash_syntax() -> None:
    assert os.access(SCRIPT, os.X_OK)
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_uses_bash_42_compatible_expansions() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    # Bash 4.2 lacks @Q and treats an empty array expansion as unset under -u.
    assert "@Q}" not in source
    assert "seen_devices=()" not in source


def test_target_python_starts_with_bound_visibility(tmp_path: Path) -> None:
    sentinel = tmp_path / "python-startup-cuda-visible-devices.txt"
    (tmp_path / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text("
        "os.environ['CUDA_VISIBLE_DEVICES'] + '\\n' + "
        "os.environ['CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES'] + '\\n', "
        "encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = _environment(PYTHONPATH=str(tmp_path))

    result = _run(
        "--",
        sys.executable,
        "-c",
        "pass",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert sentinel.read_text(encoding="utf-8") == (
        "MIG-b\nGPU-a,MIG-b,2,3\n"
    )


def test_helper_does_not_start_python_before_non_python_target(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "unexpected-python-startup"
    (tmp_path / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n",
        encoding="utf-8",
    )

    result = _run(
        "--",
        "/bin/true",
        environment=_environment(PYTHONPATH=str(tmp_path)),
    )

    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()


def test_helper_ignores_bash_env_until_after_binding(tmp_path: Path) -> None:
    sentinel = tmp_path / "bash-startup-cuda-visible-devices.txt"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(
        'printf "%s\\n" "${CUDA_VISIBLE_DEVICES}" '
        '>> "${BASH_ENV_SENTINEL}"\n',
        encoding="utf-8",
    )
    environment = _environment(
        BASH_ENV=str(bash_env),
        BASH_ENV_SENTINEL=str(sentinel),
    )

    result = _run(
        "--",
        "/bin/true",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()

    result = _run(
        "--",
        "/bin/bash",
        "-c",
        'source "$BASH_ENV"',
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    sourced_devices = sentinel.read_text(encoding="utf-8").splitlines()
    assert sourced_devices
    assert set(sourced_devices) == {"MIG-b"}


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"CUDA_VISIBLE_DEVICES": ""}, "exposes no devices"),
        ({"CUDA_VISIBLE_DEVICES": "0,1\n,2"}, "newline"),
        ({"CUDA_VISIBLE_DEVICES": "0\nGPU-a"}, "newline"),
        ({"CUDA_VISIBLE_DEVICES": "-1"}, "exposes no devices"),
        ({"CUDA_VISIBLE_DEVICES": "-2"}, "negative device token"),
        ({"CUDA_VISIBLE_DEVICES": "0,-1,2"}, "negative device token"),
        ({"CUDA_VISIBLE_DEVICES": "0,-2,2"}, "negative device token"),
        ({"CUDA_VISIBLE_DEVICES": "0,,2"}, "empty token"),
        ({"CUDA_VISIBLE_DEVICES": "0, 1"}, "invalid CUDA_VISIBLE_DEVICES"),
        ({"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-a"}, "duplicate"),
        ({"OMPI_COMM_WORLD_LOCAL_RANK": "-1"}, "non-negative integer"),
        ({"OMPI_COMM_WORLD_LOCAL_RANK": "rank"}, "non-negative integer"),
        ({"OMPI_COMM_WORLD_LOCAL_RANK": "9" * 100}, "non-negative integer"),
        ({"OMPI_COMM_WORLD_LOCAL_SIZE": "0"}, "positive integer"),
        ({"OMPI_COMM_WORLD_LOCAL_SIZE": "9" * 100}, "positive integer"),
        (
            {
                "OMPI_COMM_WORLD_LOCAL_RANK": "2",
                "OMPI_COMM_WORLD_LOCAL_SIZE": "2",
            },
            "must be smaller",
        ),
        (
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "OMPI_COMM_WORLD_LOCAL_SIZE": "3",
            },
            "requests 3 ranks",
        ),
    ),
)
def test_helper_rejects_invalid_launcher_state(
    updates: dict[str, str], message: str
) -> None:
    result = _run("--", "/bin/true", environment=_environment(**updates))

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    "missing",
    (
        "CUDA_VISIBLE_DEVICES",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
    ),
)
def test_helper_requires_launcher_variables(missing: str) -> None:
    environment = _environment()
    environment.pop(missing)

    result = _run("--", "/bin/true", environment=environment)

    assert result.returncode == 2
    assert missing in result.stderr


def test_helper_requires_target_command() -> None:
    result = _run("--")

    assert result.returncode == 2
    assert "target command" in result.stderr


def test_helper_requires_target_delimiter() -> None:
    result = _run("/bin/true")

    assert result.returncode == 2
    assert "preceded by --" in result.stderr
