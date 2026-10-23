# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import cuphoton
import cuphoton.ximage
import cuphoton.ximage.workflows as workflows
from cuphoton.ximage.workflows import run_psf_solve, run_reconstruction


class _FailingBackend:
    def solve_psf(self, *args, **kwargs):
        raise NotImplementedError


class _FakePsfBackend:
    name = "torch"

    def solve_psf(self, bundle, config):
        return {
            "psfs": np.zeros((3, 3, 1, 1), dtype=np.float64),
            "trace": {},
        }


class _FakeReconstructionBackend:
    def reconstruct(self, bundle, config):
        return {
            "image": np.zeros((1, 4, 4, 1), dtype=np.float64),
            "trace": {},
        }


class _FailingReconstructionBackend:
    def reconstruct(self, bundle, config):
        raise RuntimeError("reconstruction failed")


class _CloseTracker:
    def __init__(self) -> None:
        self.close_calls = 0


class _FakeBundle:
    def __init__(
        self, store_path: Path, tracker: _CloseTracker | None = None
    ):
        self.images = np.zeros((4, 4, 1, 2), dtype=np.float64)
        self.psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
        self.metadata = {}
        self.store_path = store_path
        self.tracker = tracker or _CloseTracker()

    def __getitem__(self, key: str):
        return getattr(self, key)

    def select_exposures(self, indices):
        selected = _FakeBundle(self.store_path, self.tracker)
        selected.images = self.images[:, :, :, indices]
        selected.psfs = self.psfs[:, :, :, indices]
        selected.metadata = {"selected_exposures": list(indices)}
        return selected

    def close(self) -> None:
        self.tracker.close_calls += 1


class _FailingSelectionBundle(_FakeBundle):
    def select_exposures(self, indices):
        raise ValueError("selection failed")


def test_mm_override_revalidates_l1_config(tmp_path: Path) -> None:
    config_path = tmp_path / "psf.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "method": "adam",
                "loss": "L1",
                "data": {"base": str(tmp_path / "data")},
                "output": {"root_dir": str(tmp_path / "runs")},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supports only"):
        run_psf_solve(config_path, method_override="mm")


def _patch_psf_failure(
    monkeypatch: pytest.MonkeyPatch, store_path: Path
) -> _CloseTracker:
    bundle = _FakeBundle(store_path)
    monkeypatch.setattr(
        workflows,
        "load_ximage_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        workflows,
        "get_backend",
        lambda backend: _FailingBackend(),
    )
    return bundle.tracker


def test_run_psf_solve_cleans_new_run_dir_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _patch_psf_failure(monkeypatch, tmp_path / "data")
    config_path = tmp_path / "psf.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "torch",
                "method": "mm",
                "data": {"base": str(tmp_path / "data")},
                "output": {
                    "root_dir": str(tmp_path),
                    "name": "psf-fail",
                    "overwrite": False,
                },
                "psf_artifact": "psfs",
            },
            sort_keys=False,
        )
    )

    with pytest.raises(NotImplementedError):
        run_psf_solve(config_path)

    assert not (tmp_path / "psf-fail").exists()
    assert tracker.close_calls == 1


def test_run_psf_solve_preserves_existing_run_dir_on_overwrite_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _patch_psf_failure(monkeypatch, tmp_path / "data")
    existing = tmp_path / "psf-existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep")

    config_path = tmp_path / "psf-existing.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "torch",
                "method": "mm",
                "data": {"base": str(tmp_path / "data")},
                "output": {
                    "root_dir": str(tmp_path),
                    "name": "psf-existing",
                    "overwrite": True,
                },
                "psf_artifact": "psfs",
            },
            sort_keys=False,
        )
    )

    with pytest.raises(NotImplementedError):
        run_psf_solve(config_path)

    assert existing.exists()
    assert sentinel.read_text() == "keep"
    assert tracker.close_calls == 1


def test_run_reconstruction_persists_resolved_psf_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psf_path = tmp_path / "visit_piffpsfs.npy"
    np.save(
        psf_path,
        np.ones((3, 3, 1, 2), dtype=np.float64),
        allow_pickle=False,
    )
    bundle = _FakeBundle(tmp_path / "data")
    monkeypatch.setattr(
        workflows, "load_ximage_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(
        workflows,
        "get_backend",
        lambda backend: _FakeReconstructionBackend(),
    )
    config_path = tmp_path / "reconstruct.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "torch",
                "data": {"base": str(tmp_path / "data")},
                "output": {
                    "root_dir": str(tmp_path),
                    "name": "reconstruct-l2",
                    "overwrite": True,
                },
                "algorithm": "l2",
                "psf_path": "visit_piffpsfs.npy",
            },
            sort_keys=False,
        )
    )

    monkeypatch.chdir(tmp_path)
    result = run_reconstruction(config_path)

    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["psf_path"] == str(psf_path.resolve())
    assert summary["package_version"] == cuphoton.ximage.__version__
    assert summary["cuphoton_core_version"] == cuphoton.__version__
    assert bundle.tracker.close_calls == 1


def test_run_psf_solve_closes_loaded_bundle_after_selected_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _FakeBundle(tmp_path / "data")
    monkeypatch.setattr(
        workflows, "load_ximage_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(
        workflows, "get_backend", lambda backend: _FakePsfBackend()
    )
    config_path = tmp_path / "psf-success.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "torch",
                "method": "mm",
                "data": {
                    "base": str(tmp_path / "data"),
                    "exposure_indices": [1],
                },
                "output": {
                    "root_dir": str(tmp_path),
                    "name": "psf-success",
                },
                "psf_artifact": "psfs",
            },
            sort_keys=False,
        )
    )

    result = run_psf_solve(config_path)

    assert result.summary["package_version"] == cuphoton.ximage.__version__
    assert result.summary["cuphoton_core_version"] == cuphoton.__version__
    assert result.summary["selected_exposures"] == [1]
    assert bundle.tracker.close_calls == 1


def test_run_reconstruction_closes_loaded_bundle_after_selected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _FakeBundle(tmp_path / "data")
    monkeypatch.setattr(
        workflows, "load_ximage_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(
        workflows,
        "get_backend",
        lambda backend: _FailingReconstructionBackend(),
    )
    config_path = tmp_path / "reconstruct-failure.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "torch",
                "algorithm": "l2",
                "data": {
                    "base": str(tmp_path / "data"),
                    "exposure_indices": [0],
                },
                "output": {
                    "root_dir": str(tmp_path),
                    "name": "reconstruct-failure",
                },
            },
            sort_keys=False,
        )
    )

    with pytest.raises(RuntimeError, match="reconstruction failed"):
        run_reconstruction(config_path)

    assert bundle.tracker.close_calls == 1
    assert not (tmp_path / "reconstruct-failure").exists()


def test_run_reconstruction_closes_bundle_when_exposure_selection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _FailingSelectionBundle(tmp_path / "data")
    monkeypatch.setattr(
        workflows, "load_ximage_bundle", lambda *args, **kwargs: bundle
    )
    config_path = tmp_path / "selection-failure.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "torch",
                "algorithm": "l2",
                "data": {
                    "base": str(tmp_path / "data"),
                    "exposure_indices": [0],
                },
                "output": {
                    "root_dir": str(tmp_path),
                    "name": "selection-failure",
                },
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ValueError, match="selection failed"):
        run_reconstruction(config_path)

    assert bundle.tracker.close_calls == 1
    assert not (tmp_path / "selection-failure").exists()
