# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/dev_changes.py"
SPEC = importlib.util.spec_from_file_location("dev_changes", SCRIPT)
assert SPEC and SPEC.loader
changes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(changes)


@pytest.mark.parametrize(
    "path",
    [
        "dev/uv.lock",
        "src/cuphoton/core/cli.py",
        "src/cuphoton/__main__.py",
        "pyproject.toml",
        "uv.lock",
        "MANIFEST.in",
        ".gitattributes",
        "tests/conftest.py",
        "docs/guide/install.qmd",
        ".github/workflows/release-docs.yml",
        ".github/workflows/docs-preview.yml",
        "new-unknown-build-file",
    ],
)
def test_dependency_paths(path):
    assert changes.affects_dev(path)


@pytest.mark.parametrize(
    "path",
    ["src/cuphoton/xray/models.py", "docs/xfit/README.md", "README.md"],
)
def test_unrelated_paths(path):
    assert not changes.affects_dev(path)


def test_vetted_copy_uses_actual_pr_target(monkeypatch):
    head, target = "a" * 40, "b" * 40
    monkeypatch.setattr(
        changes,
        "command",
        lambda *args: json.dumps(
            {"head": {"sha": head}, "base": {"sha": target}}
        ),
    )
    assert changes.comparison(
        {"after": head, "before": "c" * 40},
        "push",
        "refs/heads/pull-request/19",
        "NVIDIA/cuPhoton",
    ) == (target, head, True)
    assert changes.classify(
        {"after": "d" * 40},
        "push",
        "refs/heads/pull-request/19",
        "NVIDIA/cuPhoton",
    )["dev"]


def test_unknown_ranges_are_conservative():
    assert changes.classify({}, "push", "refs/heads/main", "NVIDIA/cuPhoton")[
        "dev"
    ]
    assert changes.classify(
        {}, "workflow_dispatch", "refs/heads/main", "NVIDIA/cuPhoton"
    )["dev"]


def test_git_ranges_cover_divergence_renames_and_deletions(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/input.py").write_text("input\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-qb", "target")
    (tmp_path / "pyproject.toml").write_text("target-only\n")
    git("add", ".")
    git("commit", "-qm", "target")
    target = git("rev-parse", "HEAD")
    git("checkout", "-qb", "topic", base)
    git("mv", "dev/input.py", "moved.py")
    git("commit", "-qm", "rename")
    head = git("rev-parse", "HEAD")
    # PR comparison excludes unrelated target-branch changes and includes the
    # removed name even when Git would otherwise report a rename.
    assert set(changes.changed_paths(target, head, True)) == {
        "dev/input.py",
        "moved.py",
    }
    # Push comparisons include the actual before/after trees, even on rewinds.
    assert set(changes.changed_paths(target, head, False)) == {
        "dev/input.py",
        "moved.py",
        "pyproject.toml",
    }
    git("rm", "moved.py")
    git("commit", "-qm", "delete")
    assert changes.changed_paths(head, git("rev-parse", "HEAD"), False) == [
        "moved.py"
    ]


def test_preview_cancellation_is_separate_from_publication_lock():
    workflows = SCRIPT.parents[2] / ".github/workflows"
    preview = yaml.safe_load((workflows / "docs-preview.yml").read_text())
    publication = yaml.safe_load((workflows / "docs.yml").read_text())
    # PyYAML interprets the Actions `on` key as YAML 1.1's boolean true.
    assert set(preview[True]) == {"pull_request"}
    assert "pull_request" not in publication[True]
    assert preview["permissions"] == {"contents": "read"}
    assert preview["concurrency"]["cancel-in-progress"] is True
    assert "queue" not in preview["concurrency"]
    assert publication["concurrency"]["cancel-in-progress"] is False
    assert publication["concurrency"]["queue"] == "max"
    assert (
        preview["concurrency"]["group"] != publication["concurrency"]["group"]
    )
