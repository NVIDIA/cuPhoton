# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Candidate-sharded execution of the ordinary xFit numerical workflow."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.core.artifacts import array_sha256, file_sha256
from cuphoton.core.bulk import (
    WorkItem,
    atomic_write_json,
    collect_gpu_identity,
    json_mapping,
    read_json_mapping,
)

from .api import DipoleFitResult, _floating_dtype, fit_dipoles
from .io import (
    XFitDataset,
    _broadcast_image_auxiliary,
    load_xfit_dataset,
    write_fit_artifacts,
)
from .models import GaussianDipoleModel, StampDipoleModel
from .solver import LMConfig

_SCHEMA = "cuphoton.xfit.executor-chunk/v1"
_META_FIELDS = {
    "parameter_names",
    "backend",
    "device",
    "dtype",
    "model",
    "mode",
}
_ARRAY_FIELDS = tuple(
    field.name
    for field in fields(DipoleFitResult)
    if field.name not in _META_FIELDS
)
_DEFAULTS = {
    "model": "gaussian",
    "mode": "difference",
    "backend": "cupy",
    "compute_dtype": "input",
    "stamp_evaluation": "bilinear",
    "stamp_scale": 1.0,
    "f_tol": None,
    "x_tol": None,
    "g_tol": None,
    "max_evaluations": None,
    "use_finite_difference": False,
}


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def plan_xfit_chunks(
    input_path: Path, *, chunk_size: int, fit_options: Mapping[str, Any]
) -> tuple[tuple[WorkItem, ...], dict[str, Any]]:
    """Freeze candidate ranges independently of the eventual worker count."""

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    if set(fit_options) - set(_DEFAULTS):
        raise ValueError("unsupported xFit execution option")
    settings = json_mapping(
        {**_DEFAULTS, **fit_options}, field="xFit options"
    )
    if settings["compute_dtype"] not in {"input", "float32", "float64"}:
        raise ValueError("unsupported xFit compute dtype")
    dataset = load_xfit_dataset(
        input_path, model=settings["model"], mode=settings["mode"]
    )
    if not dataset.batch_size:
        raise ValueError("distributed xFit requires at least one candidate")
    options = {
        "input_path": str(dataset.path),
        "input_sha256": dataset.input_archive_sha256,
        "candidate_count": dataset.batch_size,
        "result_dtype": (
            str(_floating_dtype(dataset.images))
            if settings["compute_dtype"] == "input"
            else settings["compute_dtype"]
        ),
        "fit_options": settings,
    }
    stat = dataset.path.stat()
    options["input_stat"] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    options["configuration_sha256"] = _digest(options)
    items = tuple(
        WorkItem(
            f"candidates-{start:08d}-{stop:08d}",
            {
                "start": start,
                "stop": stop,
                "candidate_ids_sha256": array_sha256(
                    dataset.candidate_id[start:stop]
                ),
                "configuration_sha256": options["configuration_sha256"],
            },
            int(dataset.images[start:stop].nbytes),
        )
        for start in range(0, dataset.batch_size, chunk_size)
        for stop in (min(start + chunk_size, dataset.batch_size),)
    )
    return items, options


def _load_input(options: Mapping[str, Any]) -> XFitDataset:
    settings = options["fit_options"]
    dataset = load_xfit_dataset(
        options["input_path"], model=settings["model"], mode=settings["mode"]
    )
    if dataset.input_archive_sha256 != options["input_sha256"]:
        raise ValueError("xFit input archive changed after planning")
    if file_sha256(dataset.path) != options["input_sha256"]:
        raise ValueError("xFit input archive changed while loading")
    return dataset


def _chunk_dataset(
    dataset: XFitDataset, start: int, stop: int
) -> XFitDataset:
    auxiliaries = {
        name: (
            None
            if (array := getattr(dataset, name)) is None
            else _broadcast_image_auxiliary(array, dataset.images)[start:stop]
        )
        for name in ("mask", "variance")
    }
    initial = dataset.initial
    if initial is not None and initial.ndim == 2:
        initial = initial[start:stop]
    return replace(
        dataset,
        candidate_id=dataset.candidate_id[start:stop],
        images=dataset.images[start:stop],
        initial=initial,
        **auxiliaries,
    )


class XFitWorker:
    """Retain validated input and solver configuration per worker."""

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.options = dict(options)
        self.dataset = _load_input(options)
        self.settings = options["fit_options"]
        self.gpu_identity = collect_gpu_identity(self.settings["backend"])
        self.config = LMConfig(
            **{
                name: self.settings[name]
                for name in ("f_tol", "x_tol", "g_tol", "max_evaluations")
            },
            use_finite_difference=(
                self.settings["use_finite_difference"]
                or self.settings["model"] == "stamp"
            ),
        )
        self.model: Any = "gaussian"
        if self.settings["model"] == "stamp":
            basis = np.asarray(self.dataset.stamp_basis)
            self.model = StampDipoleModel(
                basis[0] if basis.ndim == 3 else basis,
                image_shape=tuple(self.dataset.images.shape[-2:]),
                evaluation=self.settings["stamp_evaluation"],
                scale=self.settings["stamp_scale"],
            )

    def run_item(self, item: WorkItem, item_dir: Path) -> Mapping[str, Any]:
        started = time.perf_counter()
        start, stop = _range(item, self.options)
        dataset = _chunk_dataset(self.dataset, start, stop)
        if (
            array_sha256(dataset.candidate_id)
            != item.payload["candidate_ids_sha256"]
        ):
            raise ValueError("xFit candidate identity differs from work item")
        _check_input_stat(self.options)
        images = dataset.images
        if self.settings["compute_dtype"] != "input":
            images = images.astype(self.settings["compute_dtype"], copy=False)
        solve_start = time.perf_counter()
        result = fit_dipoles(
            images,
            model=self.model,
            initial=dataset.initial,
            mask=dataset.mask,
            variance=dataset.variance,
            mode=self.settings["mode"],
            backend=self.settings["backend"],
            config=self.config,
        )
        solve_seconds = time.perf_counter() - solve_start
        _check_input_stat(self.options)
        item_dir.mkdir(parents=True, exist_ok=False)
        archive = item_dir / "result.npz"
        np.savez_compressed(
            archive,
            candidate_id=dataset.candidate_id,
            **{
                name: np.asarray(getattr(result, name))
                for name in _ARRAY_FIELDS
            },
        )
        summary = {
            "schema": _SCHEMA,
            "item_id": item.item_id,
            **dict(item.payload),
            "input_sha256": self.options["input_sha256"],
            "result_sha256": file_sha256(archive),
            "result_metadata": {
                name: list(result.parameter_names)
                if name == "parameter_names"
                else getattr(result, name)
                for name in sorted(_META_FIELDS)
            },
        }
        atomic_write_json(item_dir / "summary.json", summary)
        return {
            "run_dir": str(item_dir),
            "summary_path": str(item_dir / "summary.json"),
            "requested_backend": self.settings["backend"],
            "backend": result.backend,
            "device": result.device,
            "runtime": {
                "configuration_sha256": self.options["configuration_sha256"]
            },
            "timings_sec": {"fit_sec": solve_seconds},
            "wall_sec": {"item_runner": time.perf_counter() - started},
        }

    def close(self) -> None:
        """Release host input ownership; CUDA caches belong to the process."""
        self.dataset = None
        self.model = None


def create_xfit_worker(options: Mapping[str, Any]) -> XFitWorker:
    """Construct a worker after its executor has bound the GPU."""
    return XFitWorker(options)


def _check_input_stat(options: Mapping[str, Any]) -> None:
    stat = Path(options["input_path"]).stat()
    if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != options[
        "input_stat"
    ]:
        raise ValueError("xFit input archive changed during execution")


def _range(item: WorkItem, options: Mapping[str, Any]) -> tuple[int, int]:
    start, stop = item.payload.get("start"), item.payload.get("stop")
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= options["candidate_count"]
        or item.payload.get("configuration_sha256")
        != options["configuration_sha256"]
    ):
        raise ValueError("invalid xFit chunk identity")
    return start, stop


def _read_result(
    item: WorkItem, round_dir: Path, options: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    item_dir = round_dir / "items" / item.item_id
    summary = read_json_mapping(item_dir / "summary.json")
    expected = {
        "schema": _SCHEMA,
        "item_id": item.item_id,
        **dict(item.payload),
        "input_sha256": options["input_sha256"],
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("xFit chunk summary identity mismatch")
    archive = item_dir / "result.npz"
    if file_sha256(archive) != summary.get("result_sha256"):
        raise ValueError("xFit chunk result checksum mismatch")
    with np.load(archive, allow_pickle=False) as loaded:
        if set(loaded.files) != {*_ARRAY_FIELDS, "candidate_id"}:
            raise ValueError("xFit chunk result fields mismatch")
        arrays = {name: loaded[name] for name in loaded.files}
    start, stop = _range(item, options)
    if any(
        array.ndim < 1 or array.shape[0] != stop - start
        for array in arrays.values()
    ):
        raise ValueError("xFit chunk result row count mismatch")
    if (
        array_sha256(arrays["candidate_id"])
        != item.payload["candidate_ids_sha256"]
    ):
        raise ValueError("xFit chunk result candidate identity mismatch")
    metadata = summary.get("result_metadata")
    if not isinstance(metadata, dict) or set(metadata) != _META_FIELDS:
        raise ValueError("xFit chunk result metadata mismatch")
    settings = options["fit_options"]
    model_type = (
        GaussianDipoleModel
        if settings["model"] == "gaussian"
        else StampDipoleModel
    )
    expected_metadata = {
        "model": settings["model"],
        "mode": settings["mode"],
        "backend": settings["backend"],
        "dtype": options["result_dtype"],
        "parameter_names": list(model_type.parameter_names),
    }
    if any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise ValueError("xFit chunk result configuration mismatch")
    return metadata, arrays


def _effective_config(
    options: Mapping[str, Any],
    dataset: XFitDataset,
    result: DipoleFitResult,
    output: Path,
) -> dict[str, Any]:
    settings = options["fit_options"]
    tolerance = float(np.sqrt(np.finfo(np.dtype(result.dtype)).eps))
    finite_difference = (
        settings["use_finite_difference"] or settings["model"] == "stamp"
    )
    config = LMConfig(
        **{
            name: settings[name]
            for name in ("f_tol", "x_tol", "g_tol", "max_evaluations")
        },
        use_finite_difference=finite_difference,
    )
    return {
        "schema_version": 1,
        "command": "fit-dipoles",
        "input": str(dataset.path),
        "output_dir": str(output),
        "model": settings["model"],
        "mode": settings["mode"],
        "backend": settings["backend"],
        "compute_dtype": {
            "requested": settings["compute_dtype"],
            "resolved": result.dtype,
        },
        "stamp": (
            {
                "evaluation": settings["stamp_evaluation"],
                "scale": settings["stamp_scale"],
                "basis": "input:stamp_basis",
            }
            if settings["model"] == "stamp"
            else None
        ),
        "solver": {
            **{
                name: tolerance if settings[name] is None else settings[name]
                for name in ("f_tol", "x_tol", "g_tol")
            },
            "max_evaluations": config.resolved_max_evaluations(
                len(result.parameter_names)
            ),
            "initial_damping": config.initial_damping,
            "damping_increase": config.damping_increase,
            "damping_decrease": config.damping_decrease,
            "finite_difference_step": tolerance
            if config.finite_difference_step is None
            else config.finite_difference_step,
            "use_finite_difference": finite_difference,
        },
    }


def finalize_xfit_round(
    round_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    items: Sequence[WorkItem],
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Restore candidate order and publish ordinary xFit artifacts."""
    expected = {item.item_id for item in items}
    observed = [record.get("item_id") for record in records]
    if (
        len(observed) != len(expected)
        or set(observed) != expected
        or any(record.get("status") != "success" for record in records)
    ):
        raise ValueError(
            "xFit merge requires one successful result per chunk"
        )
    dataset = _load_input(options)
    ordered = sorted(items, key=lambda item: item.payload["start"])
    chunks = []
    metadata = None
    next_row = 0
    for item in ordered:
        start, stop = _range(item, options)
        if start != next_row:
            raise ValueError("xFit chunks overlap or omit candidate rows")
        current_metadata, arrays = _read_result(item, round_dir, options)
        if metadata is not None and current_metadata != metadata:
            raise ValueError("xFit result metadata differs across chunks")
        metadata = current_metadata
        if not np.array_equal(
            arrays["candidate_id"], dataset.candidate_id[start:stop]
        ):
            raise ValueError("xFit merged candidate order differs from input")
        chunks.append(arrays)
        next_row = stop
    if next_row != dataset.batch_size or metadata is None:
        raise ValueError("xFit chunks do not cover the complete input")
    values = {
        name: np.concatenate([chunk[name] for chunk in chunks])
        for name in _ARRAY_FIELDS
    }
    values["uncertainty_reason"] = tuple(
        values["uncertainty_reason"].tolist()
    )
    result = DipoleFitResult(
        **values,
        **{**metadata, "parameter_names": tuple(metadata["parameter_names"])},
    )
    output = round_dir / "scientific"
    summary = write_fit_artifacts(
        output,
        dataset=dataset,
        result=result,
        effective_config=_effective_config(options, dataset, result, output),
    )
    return {
        "summary_path": "scientific/summary.json",
        "candidate_count": dataset.batch_size,
        "artifact_sha256": summary["artifact_sha256"],
    }


def prepare_xfit_workload(
    *, input_path: Path, chunk_size: int = 256, fit_options: Mapping[str, Any]
):
    """Preflight one standalone fit for either shared GPU executor."""
    from cuphoton.core.execution import WorkloadSpec

    items, options = plan_xfit_chunks(
        input_path, chunk_size=chunk_size, fit_options=fit_options
    )
    backend = options["fit_options"]["backend"]
    if backend not in {"cupy", "cutile"}:
        raise ValueError("distributed xFit requires backend cupy or cutile")
    by_id = {item.item_id: item for item in items}

    def validate(record, round_dir):
        try:
            _read_result(by_id[record["item_id"]], round_dir, options)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

    manifest = {
        "schema": "cuphoton.xfit.executor-manifest/v1",
        "items": [item.to_dict() for item in items],
        "options": options,
    }
    return WorkloadSpec(
        items=items,
        options_payload=options,
        manifest_payload=manifest,
        input_identity_payload={
            "input": options["input_path"],
            "sha256": options["input_sha256"],
        },
        manifest_sha256=_digest(manifest),
        backend=backend,
        worker_factory=create_xfit_worker,
        success_record_validator=validate,
        finalize_round=lambda round_dir, records: finalize_xfit_round(
            round_dir, records, items=items, options=options
        ),
    )
