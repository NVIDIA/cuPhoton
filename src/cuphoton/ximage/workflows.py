# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PSF calibration and scene reconstruction workflows for xImage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton import __version__ as CUPHOTON_CORE_VERSION
from cuphoton.ximage import __version__ as XIMAGE_VERSION
from cuphoton.ximage.backends import get_backend
from cuphoton.ximage.config import (
    PsfSolveConfig,
    ReconstructionConfig,
    load_config,
)
from cuphoton.ximage.data import ARTIFACTS, load_ximage_bundle
from cuphoton.ximage.data.bundle import normalize_exposure_indices
from cuphoton.ximage.run_artifacts import (
    cleanup_failed_run,
    create_run_paths,
    current_git_sha,
    finalize_run_paths,
    save_array,
    summarize_trace,
    write_config,
    write_summary,
)


@dataclass
class WorkflowResult:
    """Location and JSON-compatible summary of a completed workflow run."""

    run_dir: Path
    summary: dict[str, Any]


def run_psf_solve(
    config_path: Path,
    *,
    backend_override: str | None = None,
    device_override: str | None = None,
    output_root: Path | None = None,
    default_output_root: Path | None = None,
    method_override: str | None = None,
) -> WorkflowResult:
    config = load_config(
        config_path,
        PsfSolveConfig,
        default_output_root=default_output_root,
    )
    if backend_override is not None:
        config = config.model_copy(update={"backend": backend_override})
    if device_override is not None:
        config = config.model_copy(update={"device": device_override})
    if output_root is not None:
        config = config.model_copy(
            update={
                "output": config.output.model_copy(
                    update={"root_dir": output_root}
                )
            }
        )
    if method_override is not None:
        config = config.model_copy(update={"method": method_override})
    config = PsfSolveConfig.model_validate(config.model_dump())
    if config.method is None:
        raise ValueError(
            "PSF solve config must specify a method or use a CLI "
            "subcommand that sets one."
        )
    if config.method == "mm" and config.loss != "L2":
        raise ValueError("method='mm' supports only loss='L2'")
    paths = create_run_paths(config.output, f"psf-{config.method}")
    loaded_bundle = None
    try:
        backend = get_backend(config.backend)
        loaded_bundle = load_ximage_bundle(
            config.data.base,
            prefer=config.data.prefer,
            artifacts=(config.psf_artifact,),
        )
        bundle = loaded_bundle
        exposure_dim = ARTIFACTS[config.psf_artifact].exposure_dim
        if exposure_dim is None:
            raise ValueError("PSF artifact must have an exposure axis")
        indices = normalize_exposure_indices(
            config.data.exposure_indices,
            max_exposures=config.data.max_exposures,
            total_exposures=bundle[config.psf_artifact].shape[exposure_dim],
        )
        if indices is not None:
            bundle = bundle.select_exposures(indices)
        result = backend.solve_psf(bundle, config)
        write_config(paths.config_path, config)

        saved = {
            "psfs": str(
                save_array(paths.artifacts_dir / "psfs.npy", result["psfs"])
            ),
        }
        if "padded_psfs" in result:
            saved["padded_psfs"] = str(
                save_array(
                    paths.artifacts_dir / "psfs_padded.npy",
                    result["padded_psfs"],
                )
            )
        if "gaussian_psf" in result:
            saved["gaussian_psf"] = str(
                save_array(
                    paths.artifacts_dir / "gaussian_psf.npy",
                    result["gaussian_psf"],
                )
            )
        trace_summary, trace_arrays = summarize_trace(result["trace"])
        for key, arr in trace_arrays.items():
            saved[f"trace_{key}"] = str(
                save_array(paths.trace_dir / f"{key}.npy", arr)
            )

        summary = {
            "workflow": "psf_solve",
            "method": config.method,
            "backend": getattr(backend, "name", config.backend),
            "requested_backend": config.backend,
            "requested_device": config.device,
            "dtype": config.dtype,
            "package_version": XIMAGE_VERSION,
            "cuphoton_core_version": CUPHOTON_CORE_VERSION,
            "git_sha": current_git_sha(),
            "data_store": str(bundle.store_path),
            "selected_exposures": bundle.metadata.get("selected_exposures"),
            "psfs_shape": list(np.asarray(result["psfs"]).shape),
            "saved": saved,
            "trace_summary": trace_summary,
        }
        if "performance" in result:
            summary["performance"] = result["performance"]
        summary["saved"] = _rewrite_saved_paths(
            saved,
            paths.run_dir,
            paths.final_run_dir,
        )
        write_summary(paths.summary_path, summary)
        final_run_dir = finalize_run_paths(paths)
        return WorkflowResult(final_run_dir, summary)
    except Exception:
        cleanup_failed_run(paths)
        raise
    finally:
        if loaded_bundle is not None:
            loaded_bundle.close()


def run_reconstruction(
    config_path: Path,
    *,
    backend_override: str | None = None,
    device_override: str | None = None,
    output_root: Path | None = None,
    default_output_root: Path | None = None,
    algorithm_override: str | None = None,
) -> WorkflowResult:
    config = load_config(
        config_path,
        ReconstructionConfig,
        default_output_root=default_output_root,
    )
    if backend_override is not None:
        config = config.model_copy(update={"backend": backend_override})
    if device_override is not None:
        config = config.model_copy(update={"device": device_override})
    if output_root is not None:
        config = config.model_copy(
            update={
                "output": config.output.model_copy(
                    update={"root_dir": output_root}
                )
            }
        )
    if algorithm_override is not None:
        config = config.model_copy(update={"algorithm": algorithm_override})
    if config.algorithm is None:
        raise ValueError(
            "Reconstruction config must specify an algorithm or use "
            "a CLI subcommand that sets one."
        )
    paths = create_run_paths(config.output, f"reconstruct-{config.algorithm}")
    loaded_bundle = None
    try:
        workflow_timings: dict[str, float] = {}
        start = time.perf_counter()
        loaded_bundle = load_ximage_bundle(
            config.data.base, prefer=config.data.prefer
        )
        bundle = loaded_bundle
        workflow_timings["data_load"] = time.perf_counter() - start
        indices = normalize_exposure_indices(
            config.data.exposure_indices,
            max_exposures=config.data.max_exposures,
            total_exposures=bundle.images.shape[3],
        )
        if indices is not None:
            start = time.perf_counter()
            bundle = bundle.select_exposures(indices)
            workflow_timings["exposure_select"] = time.perf_counter() - start
        backend = get_backend(config.backend)
        start = time.perf_counter()
        result = backend.reconstruct(bundle, config)
        workflow_timings["backend_reconstruct"] = time.perf_counter() - start
        write_config(paths.config_path, config)

        start = time.perf_counter()
        saved = {
            "image": str(
                save_array(paths.artifacts_dir / "image.npy", result["image"])
            ),
        }
        if "auxiliary" in result:
            saved["auxiliary"] = str(
                save_array(
                    paths.artifacts_dir / "auxiliary.npy", result["auxiliary"]
                )
            )
        trace_summary, trace_arrays = summarize_trace(result["trace"])
        for key, arr in trace_arrays.items():
            saved[f"trace_{key}"] = str(
                save_array(paths.trace_dir / f"{key}.npy", arr)
            )
        workflow_timings["artifact_write"] = time.perf_counter() - start

        summary = {
            "workflow": "reconstruction",
            "algorithm": config.algorithm,
            "backend": getattr(backend, "name", config.backend),
            "requested_backend": config.backend,
            "requested_device": config.device,
            "dtype": config.dtype,
            "package_version": XIMAGE_VERSION,
            "cuphoton_core_version": CUPHOTON_CORE_VERSION,
            "git_sha": current_git_sha(),
            "data_store": str(bundle.store_path),
            "psf_path": (
                str(Path(config.psf_path).expanduser().resolve())
                if config.psf_path is not None
                else None
            ),
            "selected_exposures": bundle.metadata.get("selected_exposures"),
            "mask_mode": config.mask.mode,
            "effective_mask_mu": config.effective_mask_mu,
            "exposure_batch_size": config.exposure_batch_size,
            "background_model": config.background_model,
            "tiling": config.tiling.model_dump(mode="json"),
            "roi": config.roi.model_dump(mode="json"),
            "image_shape": list(np.asarray(result["image"]).shape),
            "saved": saved,
            "trace_summary": trace_summary,
            "workflow_timings_sec": workflow_timings,
        }
        if "performance" in result:
            performance = dict(result["performance"])
            performance["workflow_timings_sec"] = workflow_timings
            performance.setdefault("timings_sec", {})
            performance["timings_sec"]["artifact_write"] = workflow_timings[
                "artifact_write"
            ]
            performance["timings_sec"]["data_load"] = workflow_timings[
                "data_load"
            ]
            performance["timings_sec"]["workflow_total"] = sum(
                workflow_timings.values()
            )
            summary["performance"] = performance
        summary["saved"] = _rewrite_saved_paths(
            saved,
            paths.run_dir,
            paths.final_run_dir,
        )
        write_summary(paths.summary_path, summary)
        final_run_dir = finalize_run_paths(paths)
        return WorkflowResult(final_run_dir, summary)
    except Exception:
        cleanup_failed_run(paths)
        raise
    finally:
        if loaded_bundle is not None:
            loaded_bundle.close()


def _rewrite_saved_paths(
    saved: dict[str, str], old_root: Path, new_root: Path
) -> dict[str, str]:
    rewritten: dict[str, str] = {}
    for key, value in saved.items():
        path = Path(value)
        if str(path).startswith(str(old_root)):
            rewritten[key] = str(new_root / path.relative_to(old_root))
        else:
            rewritten[key] = value
    return rewritten
