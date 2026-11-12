# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise source and renderer isolation through real external processes."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cuphoton_dev import build, renderers
from cuphoton_dev.transforms import check_snapshot

from .test_transforms import rendered_guide

# One fake executable implements uv, its provisioned Python, and Quarto. Git
# cloning, checkout, mutation detection and HTML postprocessing remain real.
TOOL = r"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

checkout = Path(os.environ["CUPHOTON_REPO"])
source = Path(os.environ["BUILD_TEST_SOURCE"])
assert checkout != source
quarto_keys = {key for key in os.environ if key.startswith("QUARTO_")}
assert quarto_keys == {"QUARTO_PYTHON"}
assert os.environ["QUARTO_PYTHON"] == str(checkout / ".venv/bin/python")
assert "PYTHONPATH" not in os.environ
assert "GITHUB_TOKEN" not in os.environ
assert os.environ["UV_PROJECT_ENVIRONMENT"] == str(checkout / ".venv")
mode = Path(sys.argv[0]).name
with Path(os.environ["BUILD_TEST_LOG"]).open("a", encoding="utf-8") as log:
    log.write(mode + " " + " ".join(sys.argv[1:]) + "\n")

if mode == "quarto" and sys.argv[1:] == ["--version"]:
    print("1.9.36")
    sys.exit(0)
assert Path.cwd() == checkout
lock = (checkout / "uv.lock").read_text(encoding="utf-8")
assert lock == "committed lock\n"
text = (checkout / "source.txt").read_text(encoding="utf-8")
assert text == "committed source\n"
if mode == "uv":
    assert "--locked" in sys.argv, "render environment must honor target lock"
    assert not (checkout / ".venv").exists()
    assert not (checkout / "docs/guide/_book").exists()
    assert not (checkout / ".git/objects/info/alternates").exists()
    def object_inodes(repo):
        return {(p.stat().st_dev, p.stat().st_ino)
                for p in (repo / ".git/objects").rglob("*") if p.is_file()}
    shared = object_inodes(checkout) & object_inodes(source)
    assert not shared, "shared Git objects"
    python = checkout / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    shutil.copyfile(__file__, python)
    python.chmod(0o755)
elif mode == "python":
    assert sys.argv[1:5] == ["-m", "ipykernel", "install", "--prefix"]
    assert sys.argv[5] == str(checkout / ".venv")
    (checkout / ".venv/kernel-installed").touch()
elif mode == "quarto":
    assert (checkout / ".venv/kernel-installed").exists()
    assert "--execute" in sys.argv and "--no-cache" in sys.argv
    shutil.copytree(
        checkout / "render-fixture", checkout / "docs/guide/_book"
    )
    mutation = os.environ.get("BUILD_TEST_MUTATION")
    if mutation:
        (checkout / "source.txt").write_text(
            "render mutation\n", encoding="utf-8"
        )
        if mutation == "staged":
            subprocess.run(["git", "add", "source.txt"], check=True)
else:
    raise AssertionError(mode)
"""


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        text=True,
        encoding="utf-8",
        stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def build_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        renderers, "urlopen", lambda *a, **k: pytest.fail("must not download")
    )
    source = tmp_path / "source"
    guide = source / "docs/guide"
    guide.mkdir(parents=True)
    (guide / "_quarto.yml").write_text(
        "project:\n  type: book\n", encoding="utf-8"
    )
    workflow = source / ".github/workflows/docs.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n  build:\n    steps:\n"
        "      # https://github.com/quarto-dev/quarto-cli/releases/download/"
        "v1.8.0/quarto-1.8.0-linux-amd64.tar.gz\n"
        "      - run: curl https://github.com/quarto-dev/quarto-cli/releases/download/"
        "v1.9.36/quarto-1.9.36-linux-amd64.tar.gz\n",
        encoding="utf-8",
    )
    (source / "uv.lock").write_text("committed lock\n", encoding="utf-8")
    (source / "source.txt").write_text("committed source\n", encoding="utf-8")
    (source / ".gitignore").write_text(".venv/\n_book/\n", encoding="utf-8")
    rendered_guide(source / "render-fixture")
    git(source, "init", "-q", "--initial-branch=main")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "commit.gpgsign", "false")
    git(source, "add", ".")
    git(source, "commit", "-qm", "committed guide")
    git(source, "tag", "v0.2.0")
    (source / "later.txt").write_text(
        "development commit\n", encoding="utf-8"
    )
    git(source, "add", ".")
    git(source, "commit", "-qm", "development change")
    # None of this controller state may enter the rendered snapshot.
    (source / "uv.lock").write_text(
        "dirty controller lock\n", encoding="utf-8"
    )
    (source / "source.txt").write_text(
        "dirty controller source\n", encoding="utf-8"
    )
    (source / ".venv").mkdir()
    (guide / "_book").mkdir()

    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("uv", "quarto"):
        executable = tools / name
        executable.write_text(f"#!{sys.executable}\n{TOOL}", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CUPHOTON_DOCS_UV", str(tools / "uv"))
    monkeypatch.setenv("BUILD_TEST_SOURCE", str(source))
    log = tmp_path / "tools.log"
    monkeypatch.setenv("BUILD_TEST_LOG", str(log))
    for name in (
        "QUARTO_FORCE_VERSION",
        "QUARTO_BIN_PATH",
        "QUARTO_SHARE_PATH",
        "QUARTO_PROFILE",
        "QUARTO_PYTHON",
        "PYTHONPATH",
        "GITHUB_TOKEN",
    ):
        monkeypatch.setenv(name, "controller-only")
    return SimpleNamespace(source=source, log=log, output=tmp_path / "output")


@pytest.mark.parametrize(
    "ref", ["v0.2.0", "tags/v0.2.0", "refs/tags/v0.2.0", "HEAD"]
)
def test_build_uses_committed_source_and_isolated_locked_environment(
    build_source, ref
):
    fixture = build_source
    # Both the branch and a higher-precedence ref differ from the actual tag.
    git(fixture.source, "branch", "v0.2.0", "HEAD")
    git(fixture.source, "update-ref", "refs/v0.2.0", "HEAD")
    before = git(fixture.source, "status", "--porcelain")
    result = build.docs_build(
        ref=ref, output=fixture.output, repo=fixture.source
    )
    expected_version = "main" if ref == "HEAD" else "v0.2.0"
    assert result["version"] == expected_version
    expected_ref = "HEAD" if ref == "HEAD" else "refs/tags/v0.2.0"
    assert result["source_sha"] == git(
        fixture.source, "rev-parse", expected_ref
    )
    assert result["renderer"] == "1.9.36"
    assert (
        result["lock_sha256"]
        == hashlib.sha256(b"committed lock\n").hexdigest()
    )
    assert check_snapshot(fixture.output)["version"] == expected_version
    assert f"/cuPhoton/{expected_version}/index.html" in (
        fixture.output / "index.html"
    ).read_text(encoding="utf-8")
    assert git(fixture.source, "status", "--porcelain") == before
    assert (fixture.source / "source.txt").read_text(
        encoding="utf-8"
    ) == "dirty controller source\n"


def test_semantic_version_branch_without_tag_is_a_main_preview(build_source):
    fixture = build_source
    git(fixture.source, "branch", "v0.2.1", "HEAD")
    result = build.docs_build(
        ref="v0.2.1", output=fixture.output, repo=fixture.source
    )
    assert result["version"] == "main"
    assert result["source_sha"] == git(fixture.source, "rev-parse", "HEAD")
    assert "/cuPhoton/main/index.html" in (
        fixture.output / "index.html"
    ).read_text(encoding="utf-8")


def test_commented_renderer_pin_cannot_build(build_source):
    fixture = build_source
    workflow = fixture.source / ".github/workflows/docs.yml"
    workflow.write_text(
        "# https://github.com/quarto-dev/quarto-cli/releases/download/"
        "v1.9.36/quarto-1.9.36-linux-amd64.tar.gz\n",
        encoding="utf-8",
    )
    git(fixture.source, "add", str(workflow))
    git(fixture.source, "commit", "-qm", "remove active renderer install")
    with pytest.raises(ValueError, match="no recognized pinned Quarto"):
        build.docs_build(
            ref="HEAD", output=fixture.output, repo=fixture.source
        )
    assert not fixture.output.exists()
    assert not fixture.log.exists()


@pytest.mark.parametrize("mutation", ["unstaged", "staged"])
def test_build_rejects_tracked_render_mutations(
    build_source, monkeypatch, mutation
):
    fixture = build_source
    monkeypatch.setenv("BUILD_TEST_MUTATION", mutation)
    with pytest.raises(RuntimeError, match="git failed"):
        build.docs_build(
            ref="v0.2.0", output=fixture.output, repo=fixture.source
        )
    assert not fixture.output.exists()
    assert (fixture.source / "source.txt").read_text(
        encoding="utf-8"
    ) == "dirty controller source\n"
    assert (
        fixture.log.read_text(encoding="utf-8")
        .splitlines()[-1]
        .startswith("quarto render")
    )
