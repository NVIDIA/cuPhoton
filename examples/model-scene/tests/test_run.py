# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execution receipts must retain lock failures and reject changed sources."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as runner


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    here = repo / "examples/model-scene"
    here.mkdir(parents=True)
    source = repo / "src/cuphoton/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# package source\n")
    (here / "scene.py").write_text("# frozen science\n")
    (here / "protocol.json").write_text(
        json.dumps(
            dict(
                frozen=True,
                development_seeds=[1000],
                test_seed_start=3000,
                calibration_seed_start=5000,
                noise_realizations=50,
                science_source_hashes={
                    "scene.py": runner.digest(here / "scene.py")
                },
            )
        )
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for key, value in (
        ("user.name", "Test"),
        ("user.email", "test@example.com"),
    ):
        subprocess.run(["git", "config", key, value], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "Fixture"],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(runner, "HERE", here)
    monkeypatch.setattr(runner.cuphoton, "__file__", str(source))
    monkeypatch.setattr(runner, "version", lambda name: "test")
    return repo, source


@pytest.mark.parametrize("failure", ["lock_open", "source_edit"])
def test_failure_before_science_retains_receipt(
    source_tree, tmp_path, monkeypatch, failure
):
    _, source = source_tree
    output = tmp_path / "output"
    lock = (
        tmp_path / "missing/lock"
        if failure == "lock_open"
        else tmp_path / "lock"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--output",
            str(output),
            "--profile",
            "reference",
            "--gpu-lock",
            str(lock),
        ],
    )

    def edit_while_waiting(*args):
        source.write_text("# changed while waiting for the GPU lock\n")

    monkeypatch.setattr(runner.fcntl, "flock", edit_while_waiting)
    error = FileNotFoundError if failure == "lock_open" else RuntimeError
    with pytest.raises(error):
        runner.main()
    receipt = json.loads((output / "execution.json").read_text())
    assert receipt["completed"] is False
    assert receipt["error_type"] == error.__name__
    assert receipt["source_tree_clean"] is (failure == "lock_open")
    assert not (output / "controls-cpu").exists()


def test_source_guard_rejects_clean_revision_change(source_tree):
    repo, source = source_tree
    receipt = {
        "profile": "reference",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
    }
    source.write_text("# new committed package source\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Changed source",
        ],
        cwd=repo,
        check=True,
    )
    with pytest.raises(RuntimeError, match="Source revision changed"):
        runner.check_source(repo, receipt)
    assert receipt["source_tree_clean"] is True
