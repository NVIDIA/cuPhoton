# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check source identity and artifact promotion without a conda build."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/conda/build.py"
SPEC = importlib.util.spec_from_file_location("conda_build", SCRIPT)
conda_build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conda_build)


def source_archive(directory, version):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "release-source.tar.gz"
    contents = {
        "PKG-INFO": f"Name: cuphoton\nVersion: {version}\n".encode(),
        "packaging/conda/recipe.yaml": b"source:\n  path: ../source\n",
        "src/cuphoton/__init__.py": b"# Packaged source, without Git.\n",
    }
    with tarfile.open(archive, "w:gz") as target:
        for name, data in contents.items():
            member = tarfile.TarInfo(f"cuphoton-source/{name}")
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
    return archive, contents


def invoke(monkeypatch, archive, output, python=None):
    arguments = [str(SCRIPT), str(archive), "--output-dir", str(output)]
    if python is not None:
        arguments.extend(("--python", python))
    monkeypatch.setattr(sys, "argv", arguments)
    conda_build.main()


def write_packages(directory, version, versions):
    packages = {}
    for python in versions:
        tag = f"py{python.replace('.', '')}h123_0"
        relative = f"linux-64/cuphoton-{version}-{tag}.conda"
        payload = f"{version}: native package for Python {python}\n".encode()
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        packages[relative] = payload
    return packages


@pytest.mark.parametrize("version", ["0.1.3", "0.1.3rc0"])
@pytest.mark.parametrize("python", [None, "3.14"])
def test_sdist_identity_and_bytes_survive_neutral_build_staging(
    monkeypatch, tmp_path, version, python
):
    private = tmp_path / "private-home"
    private_tmp = private / "temporary"
    private_tmp.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(private_tmp))
    monkeypatch.setattr(conda_build.tempfile, "tempdir", str(private_tmp))
    monkeypatch.setenv("CUPHOTON_CONDA_VERSION", "0.0.0")
    archive, contents = source_archive(private / "input", version)
    output = private / "artifacts"
    builds = []
    expected = {}

    def build(command, *, env, check):
        assert env["CUPHOTON_CONDA_VERSION"] == version
        recipe = Path(command[command.index("--recipe") + 1])
        build_output = Path(command[command.index("--output-dir") + 1])
        stage = recipe.parent
        assert stage.parent == Path("/tmp")
        assert build_output.is_relative_to(stage)
        assert not stage.is_relative_to(private)
        assert not (stage / "source/.git").exists()
        for name, data in contents.items():
            assert (stage / "source" / name).read_bytes() == data
        assert (recipe / "recipe.yaml").read_bytes() == contents[
            "packaging/conda/recipe.yaml"
        ]
        builds.append(stage)
        expected.update(
            write_packages(
                build_output,
                version,
                [python] if python else ["3.12", "3.13", "3.14"],
            )
        )

    monkeypatch.setattr(conda_build.subprocess, "run", build)

    invoke(monkeypatch, archive, output, python)

    assert len(builds) == 1
    assert not builds[0].exists()
    assert {
        str(path.relative_to(output)): path.read_bytes()
        for path in output.glob("linux-*/*.conda")
    } == expected
    receipt = json.loads((output / "provenance.json").read_text())
    assert receipt == {
        "version": version,
        "source_archive": archive.name,
        "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "packages": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in expected.items()
        },
    }


def test_stale_output_is_rejected_before_building(monkeypatch, tmp_path):
    archive, _ = source_archive(tmp_path / "input", "0.1.3rc0")
    output = tmp_path / "output"
    original = write_packages(output, "0.1.2", ["3.12"])
    receipt = output / "provenance.json"
    receipt.write_bytes(b"previous build receipt")

    def unexpected_build(*args, **kwargs):
        pytest.fail(
            "A stale output directory must be rejected before building"
        )

    monkeypatch.setattr(conda_build.subprocess, "run", unexpected_build)

    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, archive, output)

    assert error.value.code == 2
    assert receipt.read_bytes() == b"previous build receipt"
    assert {
        str(path.relative_to(output)): path.read_bytes()
        for path in output.glob("linux-*/*.conda")
    } == original


@pytest.mark.parametrize("python", [None, "3.14"])
def test_incomplete_matrix_is_not_promoted(monkeypatch, tmp_path, python):
    archive, _ = source_archive(tmp_path / "input", "0.1.3rc0")
    output = tmp_path / "output"

    def build(command, *, env, check):
        build_output = Path(command[command.index("--output-dir") + 1])
        versions = ["3.12", "3.13"] if python is None else []
        write_packages(build_output, "0.1.3rc0", versions)
        # A stale version must not fill the gap in this build's matrix.
        write_packages(build_output, "0.1.2", ["3.14"])

    monkeypatch.setattr(conda_build.subprocess, "run", build)

    with pytest.raises(RuntimeError, match="Expected .* cuPhoton packages"):
        invoke(monkeypatch, archive, output, python)

    assert not output.exists()
