# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run artifact helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cuphoton.ximage.config import Model, OutputConfig

_PACKAGE_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RunPaths:
    run_dir: Path
    final_run_dir: Path
    artifacts_dir: Path
    trace_dir: Path
    config_path: Path
    summary_path: Path
    target_existed: bool
    is_staging: bool


def create_run_paths(output: OutputConfig, operation: str) -> RunPaths:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = OutputConfig.require_run_name(
        output.name or f"{operation}-{stamp}"
    )
    assert name is not None
    output_root = output.root_dir.expanduser().resolve()
    final_run_dir = output_root / name
    if final_run_dir.is_symlink():
        raise ValueError("Run directory must not be a symbolic link")
    target_existed = final_run_dir.exists()
    if target_existed and not final_run_dir.is_dir():
        raise ValueError(f"Run path is not a directory: {final_run_dir}")
    if target_existed and not output.overwrite:
        raise FileExistsError(
            f"{final_run_dir} already exists. Set overwrite=true to reuse it."
        )
    if target_existed and output.overwrite:
        run_dir = output_root / f".{name}.tmp-{stamp}"
        is_staging = True
    else:
        run_dir = final_run_dir
        is_staging = False
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = run_dir / "artifacts"
    trace_dir = run_dir / "trace"
    artifacts_dir.mkdir(exist_ok=True)
    trace_dir.mkdir(exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        final_run_dir=final_run_dir,
        artifacts_dir=artifacts_dir,
        trace_dir=trace_dir,
        config_path=run_dir / "config.yaml",
        summary_path=run_dir / "summary.json",
        target_existed=target_existed,
        is_staging=is_staging,
    )


def finalize_run_paths(paths: RunPaths) -> Path:
    if paths.is_staging:
        if (
            paths.final_run_dir.is_symlink()
            or paths.run_dir.resolve().parent != paths.final_run_dir.parent
        ):
            raise ValueError("Run overwrite target changed outside its root")
        if paths.final_run_dir.exists():
            shutil.rmtree(paths.final_run_dir)
        paths.run_dir.rename(paths.final_run_dir)
        return paths.final_run_dir
    return paths.run_dir


def cleanup_failed_run(paths: RunPaths) -> None:
    if paths.is_staging:
        shutil.rmtree(paths.run_dir, ignore_errors=True)
        return
    if not paths.target_existed:
        shutil.rmtree(paths.run_dir, ignore_errors=True)


def current_git_sha() -> str | None:
    env_sha = os.environ.get("CUPHOTON_XIMAGE_GIT_SHA")
    if env_sha:
        return env_sha
    package_dir = Path(__file__).resolve().parents[1]
    source_package_dir = _PACKAGE_REPOSITORY_ROOT / "src" / "cuphoton"
    try:
        is_source_checkout = (
            source_package_dir.samefile(package_dir)
            and (_PACKAGE_REPOSITORY_ROOT / ".git").exists()
        )
    except OSError:
        is_source_checkout = False
    if not is_source_checkout:
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PACKAGE_REPOSITORY_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def write_config(path: Path, config: Model) -> None:
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )


def save_array(path: Path, array: Any) -> Path:
    arr = array.numpy() if hasattr(array, "numpy") else np.asarray(array)
    np.save(path, arr, allow_pickle=False)
    return (
        path.with_suffix(path.suffix + ".npy")
        if path.suffix != ".npy"
        else path
    )


def summarize_trace(
    trace: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    if isinstance(trace, dict):
        for key in ("iterations", "loss", "u_score"):
            value = trace.get(key)
            if value is None:
                continue
            arr = _coerce_numeric_array(value)
            if arr is not None:
                arrays[key] = arr
                summary[f"{key}_count"] = int(arr.shape[0]) if arr.ndim else 1
                if arr.size:
                    summary[f"{key}_last"] = float(arr.reshape(-1)[-1])
            else:
                summary[f"{key}_type"] = type(value).__name__
        background = _coerce_numeric_array(trace.get("background"))
        if background is not None:
            arrays["background"] = background
            summary["background_count"] = (
                int(background.shape[0]) if background.ndim else 1
            )
            if background.size:
                last = np.asarray(background[-1]).reshape(-1)
                summary["background_last_min"] = float(np.min(last))
                summary["background_last_median"] = float(np.median(last))
                summary["background_last_max"] = float(np.max(last))
        tiling = trace.get("tiling")
        if isinstance(tiling, dict):
            summary["tiling_enabled"] = bool(tiling.get("enabled"))
            summary["tiling_tile_count"] = tiling.get("tile_count")
            summary["tiling_halo"] = tiling.get("halo")
        return summary, arrays

    if isinstance(trace, list):
        summary["entries"] = len(trace)
        for idx, entry in enumerate(trace):
            if not isinstance(entry, dict):
                continue
            for key in ("iterations", "loss", "u_score"):
                value = entry.get(key)
                arr = _coerce_numeric_array(value)
                if arr is not None:
                    arrays[f"psf_{idx}_{key}"] = arr
        return summary, arrays

    summary["trace_type"] = type(trace).__name__
    return summary, arrays


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _coerce_numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, list):
        coerced = []
        for item in value:
            if hasattr(item, "numpy"):
                item = item.numpy()
            if isinstance(item, np.ndarray) and item.size == 1:
                coerced.append(item.reshape(-1)[0])
            elif np.isscalar(item):
                coerced.append(item)
            else:
                return None
        return np.asarray(coerced)
    if isinstance(value, np.ndarray):
        return value
    if np.isscalar(value):
        return np.asarray([value])
    return None
