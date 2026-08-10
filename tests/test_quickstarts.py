# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import pytest


def _load_quickstart_module() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "run_quickstarts.py"
    spec = importlib.util.spec_from_file_location("run_quickstarts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load quickstart runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_quickstarts = _load_quickstart_module()


def test_quickstart_component_registry() -> None:
    expected = ("xfit", "xpois", "xscan", "xrep", "xray")

    assert run_quickstarts.COMPONENTS == expected
    assert tuple(run_quickstarts.RUNNERS) == expected


def test_xfit_quickstart_uses_only_safe_npz_arrays(tmp_path: Path) -> None:
    output_dir = tmp_path / "xfit"

    summary = run_quickstarts.run_quickstarts(
        output_dir=output_dir,
        profile="cpu",
        require_gpu=False,
        components=["xfit"],
    )

    result = summary["results"]["xfit"]
    assert result["status"] == "ok"
    assert result["backend"] == "numpy"
    assert result["device"] == "cpu"
    assert result["metrics"]["finite"] is True
    assert result["metrics"]["converged_count"] == 3
    assert not list(output_dir.rglob("*.npy"))
    assert sorted(path.name for path in output_dir.rglob("*.npz")) == [
        "dipoles.npz",
        "fit-arrays.npz",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (float("nan"), None), (0.75, 0.75)],
)
def test_quickstart_normalizes_nullable_metrics(value, expected) -> None:
    assert run_quickstarts._finite_float_or_none(value) == expected


def _cpu_hardware() -> dict[str, Any]:
    return {
        "python": "test",
        "platform": "test",
        "cuphoton": "test",
        "torch": {"available": True, "cuda_available": False},
        "cupy": {"available": False, "cuda_available": False},
        "numba_cuda_available": False,
        "gpu_available": False,
    }


def test_quickstart_summary_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_runner(
        root: Path,
        component_dir: Path,
        **_: Any,
    ) -> dict[str, Any]:
        calls.append(component_dir.name)
        artifact = component_dir / "result.npy"
        artifact.write_bytes(b"synthetic")
        return {
            "backend": "cpu",
            "device": "cpu",
            "dtype": "float32",
            "inputs": {},
            "artifacts": {
                "result": artifact.relative_to(root).as_posix(),
            },
            "metrics": {"finite": True},
        }

    monkeypatch.setattr(run_quickstarts, "detect_hardware", _cpu_hardware)
    monkeypatch.setitem(run_quickstarts.RUNNERS, "xray", fake_runner)
    monkeypatch.setitem(run_quickstarts.RUNNERS, "xrep", fake_runner)
    output_dir = tmp_path / "quickstarts"

    summary = run_quickstarts.run_quickstarts(
        output_dir=output_dir,
        profile="cpu",
        require_gpu=False,
        components=["xray", "xray", "xrep"],
    )

    persisted = json.loads((output_dir / "summary.json").read_text())
    assert persisted == summary
    assert summary["status"] == "ok"
    assert summary["components"] == ["xray", "xrep"]
    assert calls == ["xray", "xrep"]
    for component in calls:
        component_summary = json.loads(
            (output_dir / component / "summary.json").read_text()
        )
        assert component_summary == summary["results"][component]
        assert component_summary["status"] == "ok"
        assert component_summary["artifacts"]["summary"] == (
            f"{component}/summary.json"
        )


def test_quickstart_main_prints_persisted_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(run_quickstarts, "detect_hardware", _cpu_hardware)
    output_dir = tmp_path / "xray"

    status = run_quickstarts.main(
        [
            "--profile",
            "cpu",
            "--component",
            "xray",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    assert (
        capsys.readouterr().out == (output_dir / "summary.json").read_text()
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    result = summary["results"]["xray"]
    assert result["status"] == "ok"
    assert result["backend"] == "cpu"
    assert result["metrics"]["finite"] is True
    for value in result["inputs"].values():
        with h5py.File(output_dir / value, "r") as handle:
            assert "imgs" in handle
            assert "scan_var" in handle
    for value in result["artifacts"].values():
        assert (output_dir / value).is_file()


def test_cpu_profile_rejects_required_gpu(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        run_quickstarts.run_quickstarts(
            output_dir=tmp_path / "not-created",
            profile="cpu",
            require_gpu=True,
            components=["xray"],
        )


def test_quickstart_output_directory_must_be_new(tmp_path: Path) -> None:
    output_dir = tmp_path / "exists"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_quickstarts.run_quickstarts(
            output_dir=output_dir,
            profile="cpu",
            require_gpu=False,
            components=["xray"],
        )
