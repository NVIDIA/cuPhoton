# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real version configuration in small, isolated source trees."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, cwd: Path, env: dict[str, str] | None = None):
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    package = source / "src" / "cuphoton"
    package.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "src/cuphoton/__init__.py", package / "__init__.py"
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lines = [
        "[build-system]",
        f"requires = {json.dumps(config['build-system']['requires'])}",
        'build-backend = "setuptools.build_meta"',
        "[project]",
        'name = "cuphoton"',
        'dynamic = ["version"]',
        "[tool.setuptools.packages.find]",
        'where = ["src"]',
        "[tool.setuptools_scm]",
    ]
    lines.extend(
        f"{key} = {json.dumps(value)}"
        for key, value in config["tool"]["setuptools_scm"].items()
    )
    (source / "pyproject.toml").write_text("\n".join(lines) + "\n")
    for name in (".gitignore", "MANIFEST.in"):
        shutil.copyfile(ROOT / name, source / name)
    for name in (
        "AGENTS.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "RELEASING.md",
        ".github/workflows/example.yml",
        "scripts/excluded.py",
        "examples/excluded.ipynb",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded source-only file\n")
    _run("git", "init", cwd=source)
    _run("git", "config", "user.name", "Packaging Tests", cwd=source)
    _run("git", "config", "user.email", "tests@example.org", cwd=source)
    _run("git", "config", "commit.gpgsign", "false", cwd=source)
    _run("git", "config", "tag.gpgsign", "false", cwd=source)
    _run("git", "add", ".", cwd=source)
    _run("git", "commit", "-m", "Initial source", cwd=source)
    return source


def _build(
    source: Path,
    output: Path,
    kind: str,
    *,
    override: str | None = None,
) -> Path:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SETUPTOOLS_SCM_")
    }
    if override is not None:
        env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUPHOTON"] = override
    _run(
        "uv",
        "build",
        f"--{kind}",
        "--out-dir",
        str(output),
        str(source),
        cwd=source,
        env=env,
    )
    suffix = "*.whl" if kind == "wheel" else "*.tar.gz"
    (artifact,) = output.glob(suffix)
    return artifact


def _wheel_version(wheel: Path, output: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        (metadata_path,) = (
            name for name in archive.namelist() if name.endswith("/METADATA")
        )
        version = Parser().parsestr(archive.read(metadata_path).decode())[
            "Version"
        ]
        archive.extractall(output)
    result = _run(
        sys.executable,
        "-I",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import cuphoton; from importlib.metadata import version; "
        "assert cuphoton.__version__ == version('cuphoton'); "
        "print(cuphoton.__version__)",
        str(output),
        cwd=output,
    )
    assert result.stdout.strip() == version
    return version


@pytest.mark.parametrize("version", ["0.1.3rc0", "0.1.3"])
def test_tag_version_survives_gitless_sdist(
    source: Path, tmp_path: Path, version: str
) -> None:
    _run("git", "tag", f"v{version}", cwd=source)
    sdist = _build(source, tmp_path / "sdist", "sdist")
    unpacked = tmp_path / "unpacked"
    with tarfile.open(sdist) as archive:
        names = [Path(name).parts[1:] for name in archive.getnames()]
        assert ("src", "cuphoton", "_version.py") in names
        assert not any(
            parts
            and (
                parts[0] == ".github"
                or parts[-1]
                in {"AGENTS.md", "CHANGELOG.md", "CLAUDE.md", "RELEASING.md"}
                or parts[-1].endswith(".ipynb")
                or parts == ("scripts", "excluded.py")
            )
            for parts in names
        )
        archive.extractall(unpacked, filter="data")
    (gitless_source,) = unpacked.iterdir()
    assert not (gitless_source / ".git").exists()
    wheel = _build(gitless_source, tmp_path / "wheel", "wheel")
    assert _wheel_version(wheel, tmp_path / "installed") == version


def test_event_version_selects_rc_or_final_on_same_commit(
    source: Path, tmp_path: Path
) -> None:
    _run("git", "tag", "v0.1.3rc0", cwd=source)
    _run("git", "tag", "v0.1.3", cwd=source)
    head = _run("git", "rev-parse", "HEAD", cwd=source).stdout
    for version in ("0.1.3rc0", "0.1.3"):
        wheel = _build(source, tmp_path / version, "wheel", override=version)
        assert _wheel_version(wheel, tmp_path / f"installed-{version}") == (
            version
        )
    assert _run("git", "rev-parse", "HEAD", cwd=source).stdout == head


def test_untagged_descendant_is_next_development_version(
    source: Path, tmp_path: Path
) -> None:
    _run("git", "tag", "v0.1.2", cwd=source)
    (source / "README.md").write_text("Development change\n")
    _run("git", "add", "README.md", cwd=source)
    _run("git", "commit", "-m", "Development change", cwd=source)
    wheel = _build(source, tmp_path / "wheel", "wheel")
    version = Version(_wheel_version(wheel, tmp_path / "installed"))
    assert version.release == (0, 1, 3)
    assert version.dev == 1
    assert version.local is not None


def test_unrecognized_release_tag_is_rejected(
    source: Path, tmp_path: Path
) -> None:
    _run("git", "tag", "v0.1.3beta0", cwd=source)
    with pytest.raises(subprocess.CalledProcessError):
        _build(source, tmp_path / "wheel", "wheel")


def test_unbuilt_source_has_development_fallback(source: Path) -> None:
    result = _run(
        sys.executable,
        "-I",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import cuphoton; print(cuphoton.__version__)",
        str(source / "src"),
        cwd=source,
    )
    assert result.stdout.strip() == "0.0.0.dev0"


def test_unversioned_archive_does_not_build_as_fallback(
    source: Path, tmp_path: Path
) -> None:
    shutil.rmtree(source / ".git")
    with pytest.raises(subprocess.CalledProcessError):
        _build(source, tmp_path / "wheel", "wheel")
