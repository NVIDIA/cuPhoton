# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from cuphoton.xray import synthetic_validation


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=cuPhoton tests",
                "-c",
                "user.email=test@nvidia.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

    git("init", "-q")
    source = tmp_path / "src/cuphoton/xray/synthetic_validation.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source checkout\n")
    (tmp_path / ".gitignore").write_text(".venv/\n")
    git("add", "src", ".gitignore")
    git("commit", "-qm", "Initial test source")
    monkeypatch.setattr(synthetic_validation, "__file__", str(source))
    return source, git


def test_clean_tracked_source_reports_its_revision(source_checkout):
    source, git = source_checkout
    revision = git("rev-parse", "HEAD")
    assert synthetic_validation._source_provenance() == (revision, False)
    assert synthetic_validation.source_revision() == revision
    # Generated run files do not change the tracked source provenance.
    (source.parent / "summary.json").write_text("{}\n")
    assert synthetic_validation._source_provenance() == (revision, False)


@pytest.mark.parametrize("staged", [False, True])
def test_changed_tracked_source_is_dirty(source_checkout, staged):
    source, git = source_checkout
    source.write_text("# changed source\n")
    if staged:
        git("add", "src")
    assert synthetic_validation._source_provenance() == (
        git("rev-parse", "HEAD"),
        True,
    )


def test_dirty_tracks_changes_outside_the_module_directory(source_checkout):
    source, git = source_checkout
    checkout = source.parents[3]
    (checkout / ".gitignore").write_text(".venv/\n*.json\n")
    assert synthetic_validation._source_provenance() == (
        git("rev-parse", "HEAD"),
        True,
    )


@pytest.mark.parametrize("directory", [".venv", "untracked"])
def test_untracked_install_does_not_report_enclosing_checkout(
    source_checkout, monkeypatch, directory
):
    source, _ = source_checkout
    installed = (
        source.parents[3]
        / directory
        / "lib/python3.12/site-packages/cuphoton/xray/synthetic_validation.py"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text("# installed package\n")
    monkeypatch.setattr(synthetic_validation, "__file__", str(installed))
    assert synthetic_validation._source_provenance() == (None, None)
    assert synthetic_validation.source_revision() is None


def test_non_git_install_has_unknown_provenance(tmp_path, monkeypatch):
    source = tmp_path / "synthetic_validation.py"
    source.write_text("# installed package\n")
    monkeypatch.setattr(synthetic_validation, "__file__", str(source))
    assert synthetic_validation._source_provenance() == (None, None)


def test_git_unavailable_has_unknown_provenance(source_checkout, monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert synthetic_validation._source_provenance() == (None, None)


@pytest.mark.parametrize("dirty", [False, True])
def test_summary_includes_tracked_source_provenance(source_checkout, dirty):
    source, git = source_checkout
    if dirty:
        source.write_text("# changed source\n")
    sweep = synthetic_validation.validation_sweep(snr_db=(30,), trials=2)
    runtime = synthetic_validation.build_summary([sweep])["runtime"]
    assert runtime["source_revision"] == git("rev-parse", "HEAD")
    assert runtime["source_dirty"] is dirty
