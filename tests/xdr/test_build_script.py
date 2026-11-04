# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
    for name in ("dirname", "env", "mktemp", "rm"):
        (bin_dir / name).symlink_to(shutil.which(name))
    (bin_dir / "xdr-python").symlink_to(sys.executable)
    _executable(
        bin_dir / "uv",
        "#!/usr/bin/env xdr-python\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        "native = os.environ.get('CUPHOTON_XDR_BUILD_EXT', '')\n"
        "phase = ('export' if arguments[0] == 'export' else\n"
        "         'build' if native == '1' else 'dependencies')\n"
        "call = {'arguments': arguments, 'native': native}\n"
        "if '--requirement' in arguments:\n"
        "    path = arguments[arguments.index('--requirement') + 1]\n"
        "    call['requirements'] = Path(path).read_text()\n"
        "with Path(os.environ['XDR_BUILD_ARGS']).open('a') as log:\n"
        "    log.write(json.dumps(call) + '\\n')\n"
        "if os.environ.get('UV_FAIL_PHASE') == phase:\n"
        "    sys.exit(23)\n"
        "if phase == 'export':\n"
        "    path = arguments[arguments.index('--output-file') + 1]\n"
        "    Path(path).write_text('locked-build-tool==1.2.3\\n')\n",
    )
    env = {
        "PATH": str(bin_dir),
        "XDR_BUILD_ARGS": str(tmp_path / "args"),
    }
    return root, script, bin_dir, env


def _build_calls(env):
    return [
        json.loads(line)
        for line in Path(env["XDR_BUILD_ARGS"]).read_text().splitlines()
    ]


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
    export, dependencies, build = _build_calls(env)
    exported_arguments = export["arguments"]
    requirements = Path(exported_arguments[-1])
    assert exported_arguments == [
        "export",
        "--locked",
        "--only-group",
        "native-build",
        "--no-emit-project",
        "--format",
        "requirements.txt",
        "--python",
        str(expected),
        "--output-file",
        str(requirements),
    ]
    assert dependencies["arguments"] == [
        "pip",
        "install",
        "--python",
        str(expected),
        "--no-deps",
        "--require-hashes",
        "--requirement",
        str(requirements),
    ]
    assert dependencies["requirements"] == "locked-build-tool==1.2.3\n"
    assert build["native"] == "1"
    assert build["arguments"] == [
        "pip",
        "install",
        "--python",
        str(expected),
        "--no-cache",
        "--reinstall",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(root),
    ]
    assert not requirements.exists()


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


@pytest.mark.parametrize("phase", ["export", "dependencies", "build"])
def test_build_stops_after_failure_and_cleans_export(
    build_environment, phase
):
    _, script, bin_dir, env = build_environment
    _executable(bin_dir / "python3")
    env["UV_FAIL_PHASE"] = phase
    result = _run_build(script, env)
    assert result.returncode == 23
    calls = _build_calls(env)
    assert len(calls) == ["export", "dependencies", "build"].index(phase) + 1
    assert not Path(calls[0]["arguments"][-1]).exists()


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
