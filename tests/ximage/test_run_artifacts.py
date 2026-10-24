# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from cuphoton.ximage import run_artifacts
from cuphoton.ximage.config import OutputConfig
from cuphoton.ximage.run_artifacts import (
    create_run_paths,
    current_git_sha,
    summarize_trace,
)


def test_create_run_paths_detects_collision(tmp_path: Path) -> None:
    output = OutputConfig(root_dir=tmp_path, name="run", overwrite=False)
    create_run_paths(output, "psf-adam")
    with pytest.raises(FileExistsError):
        create_run_paths(output, "psf-adam")


def test_summarize_trace_dict_and_list_shapes() -> None:
    summary, arrays = summarize_trace(
        {
            "iterations": [0, 1],
            "loss": [3.0, 1.5],
            "u_score": [0.2, 0.1],
        }
    )
    assert summary["iterations_count"] == 2
    assert "loss" in arrays

    list_summary, list_arrays = summarize_trace(
        [{"iterations": [0, 1], "loss": [2.0, 1.0], "u_score": [0.4, 0.2]}]
    )
    assert list_summary["entries"] == 1
    assert "psf_0_loss" in list_arrays


def test_current_git_sha_uses_package_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    repository_root = tmp_path / "repository"
    source_package_dir = repository_root / "src" / "cuphoton"
    source_package_dir.parent.mkdir(parents=True)
    source_package_dir.symlink_to(
        Path(run_artifacts.__file__).resolve().parents[1],
        target_is_directory=True,
    )
    (repository_root / ".git").mkdir()

    def fake_check_output(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return "abc123\n"

    monkeypatch.delenv("CUPHOTON_XIMAGE_GIT_SHA", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_artifacts.subprocess, "check_output", fake_check_output
    )
    monkeypatch.setattr(
        run_artifacts,
        "_PACKAGE_REPOSITORY_ROOT",
        repository_root,
    )

    assert current_git_sha() == "abc123"
    assert captured["command"] == ["git", "rev-parse", "HEAD"]
    assert captured["cwd"] == repository_root


def test_current_git_sha_ignores_unrelated_install_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUPHOTON_XIMAGE_GIT_SHA", raising=False)
    monkeypatch.setattr(run_artifacts, "_PACKAGE_REPOSITORY_ROOT", tmp_path)

    def unexpected_check_output(*args, **kwargs):
        raise AssertionError("git must not run for an installed package")

    monkeypatch.setattr(
        run_artifacts.subprocess,
        "check_output",
        unexpected_check_output,
    )

    assert current_git_sha() is None


def test_overwrite_only_replaces_the_selected_run(tmp_path: Path) -> None:
    old_run = tmp_path / "run"
    old_run.mkdir()
    (old_run / "old.txt").write_text("old run")
    sibling = tmp_path / "other-run"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("other run")
    output = OutputConfig(root_dir=tmp_path, name="run", overwrite=True)

    paths = create_run_paths(output, "psf-adam")
    (paths.run_dir / "new.txt").write_text("new run")
    assert (old_run / "old.txt").read_text() == "old run"
    assert run_artifacts.finalize_run_paths(paths) == old_run
    assert (old_run / "new.txt").read_text() == "new run"
    assert not (old_run / "old.txt").exists()
    assert (sibling / "keep.txt").read_text() == "other run"


def test_run_creation_rejects_an_existing_directory_alias(
    tmp_path: Path,
) -> None:
    target = tmp_path / "original"
    target.mkdir()
    alias = tmp_path / "run"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        create_run_paths(
            OutputConfig(root_dir=tmp_path, name="run", overwrite=True),
            "psf-adam",
        )
    assert target.is_dir()
    assert alias.is_symlink()


def test_run_creation_preserves_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.write_text("unrelated file")
    with pytest.raises(ValueError, match="not a directory"):
        create_run_paths(
            OutputConfig(root_dir=tmp_path, name="run", overwrite=True),
            "psf-adam",
        )
    assert target.read_text() == "unrelated file"


def test_overwrite_staging_collision_preserves_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedClock:
        @staticmethod
        def now(tz):
            from datetime import datetime

            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(run_artifacts, "datetime", FixedClock)
    output = OutputConfig(root_dir=tmp_path, name="run", overwrite=True)
    (tmp_path / "run").mkdir()
    first = create_run_paths(output, "psf-adam")
    (first.run_dir / "keep.txt").write_text("staged result")
    with pytest.raises(FileExistsError):
        create_run_paths(output, "psf-adam")
    assert (first.run_dir / "keep.txt").read_text() == "staged result"
    run_artifacts.cleanup_failed_run(first)
    assert (tmp_path / "run").is_dir()
