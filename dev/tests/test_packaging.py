# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build distributions and exercise developer opt-in in clean environments."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(
    *arguments: str,
    cwd: Path,
    environment: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("SETUPTOOLS_SCM_") or key in {
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "UV_PROJECT_ENVIRONMENT",
        }:
            env.pop(key)
    for component in ("XDR", "XFIT", "XRAY", "IMAGING"):
        env[f"CUPHOTON_{component}_BUILD_EXT"] = "0"
    if environment is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = tmp_path_factory.mktemp("product-source")
    files = run(
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        cwd=ROOT,
    ).stdout.split("\0")
    for name in files:
        if not name or not (ROOT / name).is_file():
            continue
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    run("git", "init", "--initial-branch=main", cwd=source)
    run("git", "config", "user.name", "Packaging Tests", cwd=source)
    run("git", "config", "user.email", "tests@example.org", cwd=source)
    run("git", "config", "commit.gpgsign", "false", cwd=source)
    run("git", "config", "tag.gpgsign", "false", cwd=source)
    run("git", "add", ".", cwd=source)
    run("git", "commit", "-m", "Packaging fixture", cwd=source)
    run("git", "tag", "v0.2.0", cwd=source)
    return source


def build(source: Path, output: Path) -> tuple[Path, Path]:
    run(
        "uv",
        "build",
        "--wheel",
        "--sdist",
        "--out-dir",
        str(output),
        cwd=source,
    )
    return next(output.glob("*.whl")), next(output.glob("*.tar.gz"))


def probe(python: Path, cwd: Path, *, installed: bool) -> None:
    run(
        str(python),
        "-c",
        "import importlib.metadata as m, importlib.util, io; "
        "from cuphoton.core.cli import main; "
        f"installed = {installed!r}; "
        "assert (importlib.util.find_spec('cuphoton_dev') is not None) "
        "== installed; "
        "names = {d.metadata['Name'] for d in m.distributions()}; "
        "assert ('cuphoton-dev' in names) == installed; "
        "out, err = io.StringIO(), io.StringIO(); "
        "assert main(['--help'], out=out, err=err) == 0; "
        "assert ('  dev  ' in out.getvalue()) == installed; "
        "assert not err.getvalue(); "
        "out, err = io.StringIO(), io.StringIO(); "
        "assert main(['dev', '--help'], out=out, err=err) "
        "== (0 if installed else 2)",
        cwd=cwd,
    )


def test_product_artifacts_exclude_developer_project(
    source: Path, tmp_path: Path
) -> None:
    wheel, sdist = build(source, tmp_path / "initial")
    with zipfile.ZipFile(wheel) as archive:
        original_members = set(archive.namelist())
        assert not any("cuphoton_dev" in name for name in original_members)
        assert not any(name.startswith("dev/") for name in original_members)
    with tarfile.open(sdist) as archive:
        assert not any(
            "dev" in Path(name).parts[1:2]
            or "cuphoton_dev" in Path(name).parts
            for name in archive.getnames()
        )
        archive.extractall(tmp_path / "extracted", filter="data")
    extracted = next((tmp_path / "extracted").iterdir())
    # Conda uses this same extracted sdist as its complete source payload.
    assert (extracted / "packaging/conda/recipe.yaml").is_file()
    rebuilt, _ = build(extracted, tmp_path / "rebuilt")
    with zipfile.ZipFile(rebuilt) as archive:
        assert set(archive.namelist()) == original_members

    exported = subprocess.run(
        ["git", "archive", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(exported.stdout)) as archive:
        assert not any(
            name == "dev" or name.startswith("dev/")
            for name in archive.getnames()
        )


def test_installed_editable_sync_and_explicit_opt_in(
    source: Path, tmp_path: Path
) -> None:
    product_env = tmp_path / "product-env"
    developer_env = tmp_path / "developer-env"
    python = product_env / "bin/python"
    wheel, _ = build(source, tmp_path / "before")
    run(
        "uv", "venv", str(product_env), "--python", sys.executable, cwd=source
    )
    run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        str(wheel),
        cwd=source,
    )
    probe(python, source, installed=False)
    run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "-e",
        ".",
        cwd=source,
    )
    probe(python, source, installed=False)
    run("uv", "sync", "--locked", cwd=source, environment=product_env)
    probe(python, source, installed=False)
    run(
        "uv",
        "sync",
        "--locked",
        "--extra",
        "dev",
        cwd=source,
        environment=product_env,
    )
    probe(python, source, installed=False)
    run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "-e",
        ".",
        "-e",
        "dev",
        cwd=source,
    )
    probe(python, source, installed=True)
    after, _ = build(source, tmp_path / "after")
    with zipfile.ZipFile(wheel) as before, zipfile.ZipFile(after) as current:
        assert set(before.namelist()) == set(current.namelist())
    run(
        "uv",
        "pip",
        "uninstall",
        "--python",
        str(python),
        "cuphoton-dev",
        cwd=source,
    )
    probe(python, source, installed=False)
    run(
        "uv",
        "sync",
        "--locked",
        "--project",
        "dev",
        cwd=source,
        environment=developer_env,
    )
    run("uv", "sync", "--locked", cwd=source, environment=product_env)
    probe(python, source, installed=False)
    probe(developer_env / "bin/python", source, installed=True)
