# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _executable(path: Path, content: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)
    return path


@pytest.fixture
def build_environment(tmp_path):
    root = tmp_path / "source checkout"
    script = root / "src/cuphoton/xdr/src/build.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2]
    shutil.copy2(source / "src/cuphoton/xdr/src/build.sh", script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("dirname", "env"):
        (bin_dir / name).symlink_to(shutil.which(name))
    _executable(
        bin_dir / "uv",
        "#!/bin/sh\n"
        'printf "%s\\n" "$CUPHOTON_XDR_BUILD_EXT" "$@" '
        '> "$XDR_BUILD_ARGS"\n',
    )
    env = {
        "PATH": str(bin_dir),
        "XDR_BUILD_ARGS": str(tmp_path / "args"),
    }
    return root, script, bin_dir, env


def _run_build(script, env, cwd=None):
    return subprocess.run(
        [shutil.which("bash"), str(script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


@pytest.mark.parametrize(
    "selection", ["explicit", "active", "repository", "system"]
)
def test_build_selects_interpreter_and_preserves_install_arguments(
    build_environment, selection
):
    root, script, bin_dir, env = build_environment
    system = _executable(bin_dir / "python3")
    expected = system
    if selection in {"explicit", "active", "repository"}:
        expected = _executable(root / ".venv/bin/python")
    if selection in {"explicit", "active"}:
        expected = _executable(root.parent / "active env/bin/python")
        env["VIRTUAL_ENV"] = str(expected.parents[1])
    if selection == "explicit":
        expected = _executable(root.parent / "explicit python")
        env["PYTHON"] = str(expected)
    result = _run_build(script, env)
    assert result.returncode == 0, result.stderr
    assert str(expected) in result.stdout
    assert Path(env["XDR_BUILD_ARGS"]).read_text().splitlines() == [
        "1",
        "pip",
        "install",
        "--python",
        str(expected),
        "--no-build-isolation",
        "-e",
        f"{root}[gpu]",
    ]


def test_build_accepts_interpreter_command_on_path(build_environment):
    _, script, bin_dir, env = build_environment
    expected = _executable(bin_dir / "selected-python")
    env["PYTHON"] = "selected-python"
    result = _run_build(script, env)
    assert result.returncode == 0, result.stderr
    assert str(expected) in Path(env["XDR_BUILD_ARGS"]).read_text()


def test_build_resolves_relative_interpreter_before_changing_directory(
    build_environment,
):
    root, script, _, env = build_environment
    expected = _executable(root.parent / "selected env/bin/python")
    env["PYTHON"] = "selected env/bin/python"
    result = _run_build(script, env, cwd=root.parent)
    assert result.returncode == 0, result.stderr
    assert str(expected) in Path(env["XDR_BUILD_ARGS"]).read_text()


def test_build_propagates_installer_failure(build_environment):
    _, script, bin_dir, env = build_environment
    _executable(bin_dir / "python3")
    _executable(bin_dir / "uv", "#!/bin/sh\nexit 23\n")
    result = _run_build(script, env)
    assert result.returncode == 23


@pytest.mark.parametrize("selection", ["missing", "empty", "active", "none"])
def test_build_rejects_missing_selected_interpreter(
    build_environment, selection
):
    root, script, bin_dir, env = build_environment
    if selection != "none":
        _executable(root / ".venv/bin/python")
        _executable(bin_dir / "python3")
    if selection == "missing":
        env["PYTHON"] = str(root / "missing-python")
    elif selection == "empty":
        env["PYTHON"] = ""
    elif selection == "active":
        env["VIRTUAL_ENV"] = str(root / "missing-env")
    result = _run_build(script, env)
    assert result.returncode != 0
    assert "Python interpreter" in result.stderr
    assert not Path(env["XDR_BUILD_ARGS"]).exists()


def test_build_reports_missing_uv(build_environment):
    _, script, bin_dir, env = build_environment
    (bin_dir / "uv").unlink()
    result = _run_build(script, env)
    assert result.returncode != 0
    assert "requires uv on PATH" in result.stderr
    assert not Path(env["XDR_BUILD_ARGS"]).exists()
