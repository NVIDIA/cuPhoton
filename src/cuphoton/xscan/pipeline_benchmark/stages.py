# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Matched numerical stages with real process/file boundaries.

This benchmark reuses private device-pipeline validation and stamp helpers;
those helpers are not new public APIs. Each invocation processes the complete
ordered item manifest. Stage timers explicitly synchronize; they must not be
compared as GPU-only timings with the pipeline's asynchronous host timers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cuphoton.core.artifacts import file_sha256
from cuphoton.xscan import device_pipeline as pipeline

SCHEMA = "cuphoton.xscan.pipeline-benchmark.stage/v1"
STAGES = ("xpois", "xfit", "xscan")
XFIT_FIELDS = (
    *pipeline._XFIT_SCALAR_EVIDENCE_FIELDS,
    "parameters",
    "standard_errors",
    "covariance",
)
SCIENCE_KEYS = (
    "xpois.kernel_coefficients",
    "xpois.background_coefficients",
    *(f"xfit.{name}" for name in XFIT_FIELDS),
    "xfit.features",
    "xscan.logits",
    "xscan.probabilities",
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _measure(
    timings: dict[str, float],
    name: str,
    function: Callable[[], Any],
    synchronize: Callable[[], None] | None = None,
) -> Any:
    if synchronize is not None:
        synchronize()
    started = time.perf_counter()
    value = function()
    if synchronize is not None:
        synchronize()
    timings[name] = time.perf_counter() - started
    return value


def _read_summary(root: Path, stage: str) -> dict[str, Any]:
    value = json.loads((root / stage / "summary.json").read_text())
    if (
        value.get("schema") != SCHEMA
        or value.get("stage") != stage
        or value.get("status") != "success"
    ):
        raise ValueError(f"invalid completed {stage} summary")
    expected_upstream = set(STAGES[: STAGES.index(stage)])
    if set(value.get("upstream_summary_sha256", {})) != expected_upstream:
        raise ValueError(f"{stage} upstream summary links are incomplete")
    records = value.get("items")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{stage} item manifest is empty or invalid")
    if _json_hash([record["item"] for record in records]) != value.get(
        "items_sha256"
    ):
        raise ValueError(f"{stage} item metadata or order changed")
    return value


def _artifact_hash(
    path: Path, timings: dict[str, float] | None = None
) -> str:
    started = time.perf_counter()
    digest = file_sha256(path)
    if timings is not None:
        timings["artifact_hash_seconds"] = (
            timings.get("artifact_hash_seconds", 0.0)
            + time.perf_counter()
            - started
        )
    return digest


def _read_array(
    root: Path,
    descriptor: dict[str, Any],
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    path = (root / descriptor["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("intermediate artifact escapes its round directory")
    if _artifact_hash(path, timings) != descriptor["sha256"]:
        raise ValueError(f"intermediate artifact hash changed: {path.name}")
    array = np.load(path, allow_pickle=False)
    if (
        not isinstance(array, np.ndarray)
        or list(array.shape) != descriptor["shape"]
        or array.dtype.str != descriptor["dtype"]
        or array.nbytes != descriptor["nbytes"]
        or not array.flags.c_contiguous
    ):
        raise ValueError(
            f"intermediate artifact contract changed: {path.name}"
        )
    if _artifact_hash(path, timings) != descriptor["sha256"]:
        raise ValueError(
            f"intermediate artifact changed while reading: {path.name}"
        )
    return array


def _write_arrays(
    root: Path,
    directory: Path,
    arrays: dict[str, np.ndarray],
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    directory.mkdir()
    artifacts = {}
    for name, values in arrays.items():
        array = np.ascontiguousarray(values)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"unsupported numeric artifact dtype: {name}")
        path = directory / f"{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        artifacts[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _artifact_hash(path, timings),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": array.nbytes,
            "file_bytes": path.stat().st_size,
        }
    return artifacts


def _setup(
    stage: str, config: pipeline.DevicePipelineConfig
) -> dict[str, Any]:
    from cuphoton.xscan.xfit_features import FEATURE_NAMES

    schema = pipeline._load_feature_schema_contract(
        Path(config.feature_schema_path),
        expected_sha256=config.feature_schema_sha256,
    )
    pipeline._validate_feature_schema_source(
        schema, config=config, feature_names=FEATURE_NAMES
    )
    checkpoint_path = Path(config.checkpoint_dir) / "checkpoint.pt"
    if file_sha256(checkpoint_path) != config.checkpoint_sha256:
        raise ValueError("checkpoint hash changed")
    runtime: dict[str, Any] = {"cp": None, "torch": None}
    if stage == "xscan":
        import torch

        from cuphoton.xscan.config import PerformanceConfig
        from cuphoton.xscan.training import load_model_from_checkpoint

        torch.set_num_threads(config.inference_policy["worker_cpu_threads"])
        torch.cuda.set_device(config.device_id)
        runtime.update(torch=torch, device=torch.device(config.device))
        model, checkpoint, performance = load_model_from_checkpoint(
            Path(config.checkpoint_dir),
            device=runtime["device"],
            performance_override=PerformanceConfig(**config.inference_policy),
        )
        if asdict(performance) != config.inference_policy:
            raise ValueError("inference policy changed during normalization")
        pipeline._validate_checkpoint_contract(
            checkpoint,
            stamp_shape=config.stamp_shape,
            feature_names=FEATURE_NAMES,
            feature_schema_sha256=config.feature_schema_sha256,
            feature_schema=schema,
        )
        pipeline._validate_loaded_model_contract(
            model, feature_names=FEATURE_NAMES
        )
        runtime.update(model=model, performance=performance)
        torch.cuda.synchronize(runtime["device"])
    else:
        import cupy as cp

        cp.cuda.Device(config.device_id).use()
        cp.cuda.Device(config.device_id).synchronize()
        runtime.update(cp=cp)
    if file_sha256(checkpoint_path) != config.checkpoint_sha256:
        raise ValueError("checkpoint changed during stage initialization")
    if (
        file_sha256(config.feature_schema_path)
        != config.feature_schema_sha256
    ):
        raise ValueError("feature schema changed during stage initialization")
    return runtime


def _run_item(
    stage: str,
    item: pipeline.DevicePipelineItem,
    config: pipeline.DevicePipelineConfig,
    runtime: dict[str, Any],
    root: Path,
    priors: dict[str, dict[str, Any]],
    index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    cp, torch = runtime["cp"], runtime["torch"]

    def sync() -> None:
        if stage == "xscan":
            torch.cuda.synchronize(runtime["device"])
        else:
            cp.cuda.Device().synchronize()

    timings: dict[str, float] = {}

    def previous(which: str, name: str) -> np.ndarray:
        return _read_array(
            root, priors[which]["items"][index]["artifacts"][name], timings
        )

    if stage == "xpois":
        from cuphoton.xpois import (
            GaussianBasisComponent,
            solve_constant_kernel_device,
        )

        host = _measure(
            timings, "read_seconds", lambda: pipeline._read_item_inputs(item)
        )
        device = _measure(
            timings,
            "upload_seconds",
            lambda: {name: cp.asarray(value) for name, value in host.items()},
            sync,
        )

        def compute() -> tuple[dict[str, Any], dict[str, Any]]:
            result = solve_constant_kernel_device(
                device["reference"],
                device["target"],
                [
                    GaussianBasisComponent(sigma=s, degree=d)
                    for s, d in zip(
                        config.xpois.basis_sigmas,
                        config.xpois.basis_degrees,
                        strict=True,
                    )
                ],
                kernel_shape=config.xpois.kernel_shape,
                variance=device.get("variance"),
                fit_mask=device.get("fit_mask"),
                background_degree=config.xpois.background_degree,
                flux_conserve=config.xpois.flux_conserve,
                flux_reference_index=config.xpois.flux_reference_index,
            )
            stamps, difference = pipeline._extract_stamps(
                cp=cp,
                target=device["target"],
                reference=device["reference"],
                residual=result.residual,
                candidates=item.candidates,
                stamp_shape=config.stamp_shape,
            )
            arrays = {
                f"xpois.{name}": getattr(result, name)
                for name in (
                    "kernel_coefficients",
                    "background_coefficients",
                )
            }
            arrays.update(stamps=stamps, difference=difference)
            return arrays, {
                "xpois_chi2": float(result.chi2),
                "xpois_dof": result.dof,
                "xpois_fit_pixel_count": result.fit_pixel_count,
                "basis_terms": [asdict(term) for term in result.basis_terms],
                "solver": result.solver,
                "backend": result.backend,
                "flux_conserve": result.flux_conserve,
            }

        arrays, metadata = _measure(timings, "compute_seconds", compute, sync)
    elif stage == "xfit":
        from cuphoton.xfit import LMConfig, fit_dipoles_device
        from cuphoton.xscan.xfit_features import (
            transform_xfit_result_features_device,
        )

        host = _measure(
            timings, "read_seconds", lambda: previous("xpois", "difference")
        )
        difference = _measure(
            timings, "upload_seconds", lambda: cp.asarray(host), sync
        )

        def compute() -> tuple[dict[str, Any], dict[str, Any]]:
            result = fit_dipoles_device(
                difference,
                model=config.xfit.model,
                mode=config.xfit.mode,
                config=LMConfig(**config.xfit.solver_payload()),
            )
            features = transform_xfit_result_features_device(
                result,
                image_shape=config.stamp_shape,
                variance_present=False,
            )
            return {
                **{
                    f"xfit.{name}": getattr(result, name)
                    for name in XFIT_FIELDS
                },
                "xfit.features": features.values,
            }, {
                "parameter_names": list(result.parameter_names),
                "feature_names": list(features.feature_names),
                "model": result.model,
                "mode": result.mode,
                "backend": result.backend,
                "solver": result.solver,
                "dtype": result.dtype,
                "variance_present": False,
            }

        arrays, metadata = _measure(timings, "compute_seconds", compute, sync)
    else:
        from cuphoton.xscan.training import predict_tensors

        host = _measure(
            timings,
            "read_seconds",
            lambda: {
                "images": previous("xpois", "stamps"),
                "xfit_features": previous("xfit", "xfit.features"),
            },
        )
        device = _measure(
            timings,
            "upload_seconds",
            lambda: {
                name: torch.from_numpy(value).to(runtime["device"])
                for name, value in host.items()
            },
            sync,
        )
        prediction = _measure(
            timings,
            "compute_seconds",
            lambda: predict_tensors(
                model=runtime["model"],
                **device,
                device=runtime["device"],
                performance=runtime["performance"],
            ),
            sync,
        )
        arrays = {
            f"xscan.{name}": value for name, value in prediction.items()
        }
        metadata = {"inference_policy": config.inference_policy}

    host_arrays = _measure(
        timings,
        "download_seconds",
        lambda: {
            name: (
                value.detach().cpu().numpy()
                if stage == "xscan"
                else cp.asnumpy(value)
            )
            for name, value in arrays.items()
        },
        sync,
    )
    if stage == "xscan":
        if not all(
            np.isfinite(value).all() for value in host_arrays.values()
        ):
            raise ValueError("non-finite xScan predictions")
        metadata["predictions"] = [
            {
                **candidate.to_payload(),
                "logit": float(host_arrays["xscan.logits"][position]),
                "probability": float(
                    host_arrays["xscan.probabilities"][position]
                ),
                "decision": bool(
                    host_arrays["xscan.probabilities"][position]
                    >= config.decision_threshold
                ),
            }
            for position, candidate in enumerate(item.candidates)
        ]
    return host_arrays, metadata, timings


def run_stage(
    stage: str,
    config: pipeline.DevicePipelineConfig,
    items: list[pipeline.DevicePipelineItem],
    root: Path,
) -> dict[str, Any]:
    """Run one stage, refusing stale output or mismatched inputs."""
    started = time.perf_counter()
    if stage not in STAGES or not items:
        raise ValueError("a known stage and nonempty item list are required")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("item IDs must be unique within a round")
    root = Path(root).resolve()
    stage_dir = root / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    item_payloads = [item.to_payload() for item in items]
    items_sha256 = _json_hash(item_payloads)
    identity = {
        "configuration_sha256": config.configuration_sha256,
        "checkpoint_sha256": config.checkpoint_sha256,
        "feature_schema_sha256": config.feature_schema_sha256,
        "items_sha256": items_sha256,
    }
    priors = {}
    for previous in STAGES[: STAGES.index(stage)]:
        prior = _read_summary(root, previous)
        if any(prior.get(key) != value for key, value in identity.items()):
            raise ValueError(
                f"{previous} input/configuration identity mismatch"
            )
        if [entry["item"] for entry in prior["items"]] != item_payloads:
            raise ValueError(f"{previous} item order or metadata mismatch")
        priors[previous] = prior
    for item in items:
        pipeline._validate_item_contract(item, config=config)
    runtime = _setup(stage, config)
    setup_seconds = time.perf_counter() - started
    records = []
    for index, item in enumerate(items):
        item_started = time.perf_counter()
        arrays, metadata, timings = _run_item(
            stage,
            item,
            config,
            runtime,
            root,
            priors,
            index,
        )
        artifacts = _measure(
            timings,
            "write_seconds",
            lambda: _write_arrays(
                root,
                stage_dir / f"item-{index:04d}",
                arrays,
                timings,
            ),
        )
        _measure(
            timings,
            "input_recheck_seconds",
            lambda: pipeline._verify_item_hashes(
                item,
                when=f"during {stage} execution",
            ),
        )
        # The pipeline and xPOIS both hash original inputs on read and once
        # after execution. Only the later stages' rechecks are additional.
        timings["extra_hashing_seconds"] = timings.get(
            "artifact_hash_seconds", 0.0
        ) + (timings["input_recheck_seconds"] if stage != "xpois" else 0.0)
        timings["wall_seconds"] = time.perf_counter() - item_started
        records.append(
            {
                "item_id": item.item_id,
                "item": item.to_payload(),
                "metadata": metadata,
                "artifacts": artifacts,
                "timings_seconds": timings,
            }
        )
    summary = {
        "schema": SCHEMA,
        "status": "success",
        "stage": stage,
        **identity,
        "device": config.device,
        "setup_seconds": setup_seconds,
        "wall_seconds": time.perf_counter() - started,
        "upstream_summary_sha256": {
            name: file_sha256(root / name / "summary.json") for name in priors
        },
        "items": records,
        "extra_hashing_seconds": sum(
            record["timings_seconds"]["extra_hashing_seconds"]
            for record in records
        ),
        "timing_note": (
            "Setup starts inside run_stage; parent process wall includes "
            "imports and shutdown. Compute/upload/download synchronize "
            "explicitly; these are host elapsed, not GPU-only. Wall excludes "
            "this final summary write. NPY writes close without fsync. "
            "Extra hashing counts intermediate digests and original-input "
            "rechecks in xFit/xScan; input hashes shared with the pipeline "
            "remain in the adjusted comparison."
        ),
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def load_science_arrays(root: Path, index: int) -> dict[str, np.ndarray]:
    """Read all 22 arrays in the pipeline decoder's float64 form."""
    root = Path(root).resolve()
    summaries = {stage: _read_summary(root, stage) for stage in STAGES}
    first = summaries["xpois"]
    for stage, summary in summaries.items():
        for key in (
            "configuration_sha256",
            "checkpoint_sha256",
            "feature_schema_sha256",
            "items_sha256",
        ):
            if summary[key] != first[key]:
                raise ValueError("scientific stage identities disagree")
        if [entry["item"] for entry in summary["items"]] != [
            entry["item"] for entry in first["items"]
        ]:
            raise ValueError("scientific stage item identities disagree")
        for upstream, digest in summary["upstream_summary_sha256"].items():
            if file_sha256(root / upstream / "summary.json") != digest:
                raise ValueError(f"{stage} upstream summary changed")
    return {
        name: _read_array(
            root,
            summaries[name.split(".")[0]]["items"][index]["artifacts"][name],
        ).astype(np.float64)
        for name in SCIENCE_KEYS
    }
