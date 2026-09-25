# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dragon orchestration for the persistent XPOIS-to-XScan device seam."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from cuphoton.core.artifacts import file_sha256
from cuphoton.core.bulk import (
    Placement,
    WorkItem,
    atomic_write_json,
    error_payload,
    item_ids_sha256,
    json_mapping,
    read_json_mapping,
    timestamp_utc,
)
from cuphoton.xpois import dragon as _dragon

from .device_pipeline import (
    DEVICE_PIPELINE_RESULT_SCHEMA,
    DevicePipelineConfig,
    DevicePipelineItem,
    DeviceWorkerContext,
    NpyArrayDescriptor,
    decode_device_pipeline_evidence,
    run_device_pipeline_item,
)

DRAGON_DEVICE_PIPELINE_MANIFEST_SCHEMA = (
    "cuphoton.xscan.device-pipeline.dragon-manifest/v1"
)
DRAGON_DEVICE_PIPELINE_INPUT_IDENTITY_SCHEMA = (
    "cuphoton.xscan.device-pipeline.dragon-input-identity/v1"
)
DRAGON_DEVICE_PIPELINE_OPTIONS_SCHEMA = (
    "cuphoton.xscan.device-pipeline.dragon-options/v1"
)
DRAGON_DEVICE_PIPELINE_RUN_SCHEMA = (
    "cuphoton.xscan.device-pipeline.dragon-run/v1"
)
DRAGON_DEVICE_PIPELINE_SUMMARY_SCHEMA = (
    "cuphoton.xscan.device-pipeline.dragon-summary/v1"
)
DRAGON_DEVICE_PIPELINE_ITEM_SCHEMA = (
    "cuphoton.xscan.device-pipeline.dragon-item/v1"
)
DRAGON_DEVICE_PIPELINE_SHARD_SCHEMA = (
    "cuphoton.xscan.device-pipeline.dragon-shard/v1"
)

_BACKEND = "cupy"
_RUNTIME_SCHEMA = "cuphoton.xscan.device-pipeline.worker-runtime/v1"
_RESULT_METADATA_FIELDS = frozenset(
    {
        "schema",
        "summary_sha256",
        "result_sha256",
        "configuration_sha256",
        "checkpoint_sha256",
        "feature_schema_sha256",
        "input_sha256",
        "candidate_count",
        "positive_count",
        "scientific_evidence_sha256",
        "scientific_evidence_bytes",
        "transfers",
    }
)
_RESULT_IDENTITY_FIELDS = (
    "schema",
    "item_id",
    "device",
    "checkpoint_sha256",
    "feature_schema_sha256",
    "configuration_sha256",
    "input_sha256",
    "xpois",
    "predictions",
    "scientific_evidence",
    "transfers",
)
_RESULT_PAYLOAD_FIELDS = frozenset(
    (*_RESULT_IDENTITY_FIELDS, "timings", "result_sha256")
)
_PREDICTION_FIELDS = frozenset(
    {
        "candidate_id",
        "center_x",
        "center_y",
        "source_index",
        "logit",
        "probability",
        "decision",
    }
)
_XPOIS_RESULT_FIELDS = frozenset({"chi2", "dof", "fit_pixel_count"})
_TIMING_FIELDS = frozenset({"stages_seconds", "total_seconds", "semantics"})
_TIMING_STAGES = frozenset(
    {
        "input_read",
        "input_h2d",
        "xpois",
        "stamp_extraction",
        "xfit",
        "feature_transform",
        "evidence_pack",
        "dlpack_handoff",
        "xscan",
        "terminal_pack",
        "terminal_d2h",
    }
)


def _collect_device_pipeline_gpu_identity(backend: str) -> Mapping[str, Any]:
    """Load Torch before CuPy touches CUDA in the placed worker.

    Torch establishes its CUDA library load order during import. A prior CuPy
    runtime call can make the later cuBLASLt model initialization fail.
    """

    importlib.import_module("torch")
    return _dragon._collect_gpu_identity(backend)


_TRANSFER_FIELDS = frozenset(
    {
        "input_h2d_bytes",
        "terminal_d2h_bytes",
        "dlpack_shared_bytes",
        "pipeline_full_array_d2h_bytes",
        "pipeline_compact_h2d_bytes",
        "terminal_d2h_calls",
        "accounting_scope",
        "device_stage_internal_transfers",
    }
)
_XFIT_SCALAR_EVIDENCE_SEGMENTS = (
    "xfit.status_codes",
    "xfit.converged",
    "xfit.evaluations",
    "xfit.residual_norm",
    "xfit.chi_square",
    "xfit.valid_pixel_count",
    "xfit.valid_pixel_fraction",
    "xfit.null_chi_square",
    "xfit.delta_chi_square",
    "xfit.fractional_null_improvement",
    "xfit.degrees_of_freedom",
    "xfit.reduced_chi_square",
    "xfit.uncertainty_valid",
    "xfit.uncertainty_reason_codes",
)
_EVIDENCE_SEGMENTS = (
    "xpois.kernel_coefficients",
    "xpois.background_coefficients",
    *_XFIT_SCALAR_EVIDENCE_SEGMENTS,
    "xfit.parameters",
    "xfit.standard_errors",
    "xfit.covariance",
    "xfit.features",
    "xscan.logits",
    "xscan.probabilities",
)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_mapping(
    value: Any, *, expected: frozenset[str], field: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    result = dict(value)
    if set(result) != expected:
        raise ValueError(f"{field} has invalid fields")
    return result


def _finite_float(
    value: Any, *, field: str, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{field} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _strict_integer(
    value: Any, *, field: str, minimum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _strict_name_list(value: Any, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(name, str) or not name for name in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return value


def _expected_input_h2d_bytes(item: DevicePipelineItem) -> int:
    return sum(
        math.prod(descriptor.shape)
        * (
            np.dtype(np.bool_).itemsize
            if name == "fit_mask"
            else np.dtype(np.float64).itemsize
        )
        for name, descriptor in _item_descriptors(item)
    )


def _expected_dlpack_shared_bytes(
    decoded: Mapping[str, np.ndarray],
    *,
    config: DevicePipelineConfig,
    candidate_count: int,
) -> int:
    stamp_bytes = (
        candidate_count
        * 3
        * math.prod(config.stamp_shape)
        * np.dtype(np.float32).itemsize
    )
    feature_bytes = (
        decoded["xfit.features"].size * np.dtype(np.float32).itemsize
    )
    evidence_bytes = sum(
        values.size * np.dtype(np.float64).itemsize
        for name, values in decoded.items()
        if name not in {"xscan.logits", "xscan.probabilities"}
    )
    return int(stamp_bytes + feature_bytes + evidence_bytes)


def _strict_device_pipeline_result_payload(
    payload: Mapping[str, Any],
    *,
    item: DevicePipelineItem,
    config: DevicePipelineConfig,
) -> dict[str, Any]:
    """Validate one complete result/v2 and its canonical evidence receipt."""

    values = json_mapping(
        payload, field=f"device pipeline result {item.item_id!r}"
    )
    if set(values) != _RESULT_PAYLOAD_FIELDS:
        raise ValueError("device pipeline result has invalid fields")
    if (
        values["schema"] != DEVICE_PIPELINE_RESULT_SCHEMA
        or values["item_id"] != item.item_id
        or values["device"] != config.device
        or values["configuration_sha256"] != config.configuration_sha256
        or values["checkpoint_sha256"] != config.checkpoint_sha256
        or values["feature_schema_sha256"] != config.feature_schema_sha256
        or values["input_sha256"] != _input_sha256(item)
    ):
        raise ValueError("device pipeline result identity differs")

    xpois = _exact_mapping(
        values["xpois"],
        expected=_XPOIS_RESULT_FIELDS,
        field="device pipeline XPOIS result",
    )
    _finite_float(
        xpois["chi2"], field="device pipeline XPOIS chi2", minimum=0.0
    )
    dof = _strict_integer(xpois["dof"], field="device pipeline XPOIS dof")
    fit_pixel_count = _strict_integer(
        xpois["fit_pixel_count"],
        field="device pipeline XPOIS fit_pixel_count",
        minimum=1,
    )

    raw_predictions = values["predictions"]
    if not isinstance(raw_predictions, list) or len(raw_predictions) != len(
        item.candidates
    ):
        raise ValueError("device pipeline predictions and candidates differ")
    predictions: list[dict[str, Any]] = []
    for index, (raw_prediction, candidate) in enumerate(
        zip(raw_predictions, item.candidates, strict=True)
    ):
        prediction = _exact_mapping(
            raw_prediction,
            expected=_PREDICTION_FIELDS,
            field=f"device pipeline prediction {index}",
        )
        if (
            type(prediction["candidate_id"])
            is not type(candidate.candidate_id)
            or prediction["candidate_id"] != candidate.candidate_id
        ):
            raise ValueError(
                "device pipeline prediction candidate ID differs"
            )
        for field, expected in (
            ("center_x", candidate.center_x),
            ("center_y", candidate.center_y),
            ("source_index", candidate.source_index),
        ):
            if (
                _strict_integer(
                    prediction[field],
                    field=f"device pipeline prediction {index} {field}",
                )
                != expected
            ):
                raise ValueError(
                    f"device pipeline prediction {index} {field} differs"
                )
        _finite_float(
            prediction["logit"],
            field=f"device pipeline prediction {index} logit",
        )
        probability = _finite_float(
            prediction["probability"],
            field=f"device pipeline prediction {index} probability",
            minimum=0.0,
        )
        if probability > 1.0:
            raise ValueError(
                "device pipeline prediction probability exceeds one"
            )
        if not isinstance(prediction["decision"], bool):
            raise TypeError(
                "device pipeline prediction decision must be boolean"
            )
        if prediction["decision"] is not (
            probability >= config.decision_threshold
        ):
            raise ValueError("device pipeline prediction decision differs")
        predictions.append(prediction)

    transfers = _exact_mapping(
        values["transfers"],
        expected=_TRANSFER_FIELDS,
        field="device pipeline transfer receipt",
    )
    for name in (
        "input_h2d_bytes",
        "terminal_d2h_bytes",
        "dlpack_shared_bytes",
        "pipeline_full_array_d2h_bytes",
        "pipeline_compact_h2d_bytes",
        "terminal_d2h_calls",
    ):
        _strict_integer(
            transfers[name],
            field=f"device pipeline transfer receipt {name}",
            minimum=0,
        )
    if (
        transfers["pipeline_full_array_d2h_bytes"] != 0
        or transfers["pipeline_compact_h2d_bytes"] != 0
        or transfers["terminal_d2h_calls"] != 1
        or transfers["accounting_scope"] != "pipeline-owned"
        or transfers["device_stage_internal_transfers"] != "not-instrumented"
    ):
        raise ValueError("device pipeline transfer receipt differs")

    timings = _exact_mapping(
        values["timings"],
        expected=_TIMING_FIELDS,
        field="device pipeline timing receipt",
    )
    stages = _exact_mapping(
        timings["stages_seconds"],
        expected=_TIMING_STAGES,
        field="device pipeline stage timings",
    )
    for name, value in stages.items():
        _finite_float(
            value,
            field=f"device pipeline stage timing {name}",
            minimum=0.0,
        )
    _finite_float(
        timings["total_seconds"],
        field="device pipeline total timing",
        minimum=0.0,
    )
    if timings["semantics"] != "host-elapsed; terminal-d2h-is-blocking":
        raise ValueError("device pipeline timing semantics differ")

    evidence = values["scientific_evidence"]
    if not isinstance(evidence, Mapping):
        raise TypeError(
            "device pipeline scientific evidence must be a mapping"
        )
    decoded = decode_device_pipeline_evidence(evidence, config=config)
    if tuple(decoded) != _EVIDENCE_SEGMENTS:
        raise ValueError(
            "device pipeline scientific evidence segments differ"
        )
    candidate_count = _strict_integer(
        evidence.get("candidate_count"),
        field="device pipeline scientific evidence candidate_count",
        minimum=1,
    )
    if candidate_count != len(predictions):
        raise ValueError(
            "device pipeline scientific evidence candidate count differs"
        )
    parameter_names = _strict_name_list(
        evidence.get("parameter_names"),
        field="device pipeline scientific evidence parameter_names",
    )
    feature_names = _strict_name_list(
        evidence.get("feature_names"),
        field="device pipeline scientific evidence feature_names",
    )
    expected_shapes = {
        **{
            name: (candidate_count,)
            for name in _XFIT_SCALAR_EVIDENCE_SEGMENTS
        },
        "xfit.parameters": (candidate_count, len(parameter_names)),
        "xfit.standard_errors": (candidate_count, len(parameter_names)),
        "xfit.covariance": (
            candidate_count,
            len(parameter_names),
            len(parameter_names),
        ),
        "xfit.features": (candidate_count, len(feature_names)),
        "xscan.logits": (candidate_count,),
        "xscan.probabilities": (candidate_count,),
    }
    if any(
        tuple(decoded[name].shape) != shape
        for name, shape in expected_shapes.items()
    ):
        raise ValueError("device pipeline scientific evidence shape differs")
    kernel_coefficients = decoded["xpois.kernel_coefficients"]
    background_coefficients = decoded["xpois.background_coefficients"]
    if (
        kernel_coefficients.ndim != 1
        or kernel_coefficients.size < 1
        or background_coefficients.ndim != 1
        or background_coefficients.size < 1
    ):
        raise ValueError("device pipeline XPOIS evidence shape differs")
    if dof != fit_pixel_count - int(
        kernel_coefficients.size + background_coefficients.size
    ):
        raise ValueError("device pipeline XPOIS degrees of freedom differ")
    for index, prediction in enumerate(predictions):
        if (
            float(decoded["xscan.logits"][index]) != prediction["logit"]
            or float(decoded["xscan.probabilities"][index])
            != prediction["probability"]
        ):
            raise ValueError(
                "device pipeline predictions and scientific evidence differ"
            )
    if transfers["terminal_d2h_bytes"] != evidence.get("packed_byte_count"):
        raise ValueError(
            "device pipeline evidence and terminal transfer bytes differ"
        )
    if transfers["input_h2d_bytes"] != _expected_input_h2d_bytes(item):
        raise ValueError("device pipeline input H2D byte count differs")
    if transfers["dlpack_shared_bytes"] != _expected_dlpack_shared_bytes(
        decoded,
        config=config,
        candidate_count=candidate_count,
    ):
        raise ValueError("device pipeline DLPack shared byte count differs")

    result_sha256 = values["result_sha256"]
    if not _is_sha256(result_sha256):
        raise ValueError("device pipeline result SHA-256 is invalid")
    identity = {name: values[name] for name in _RESULT_IDENTITY_FIELDS}
    if _json_sha256(identity) != result_sha256:
        raise ValueError("device pipeline result SHA-256 differs")
    return values


def _item_descriptors(
    item: DevicePipelineItem,
) -> tuple[tuple[str, NpyArrayDescriptor], ...]:
    values: list[tuple[str, NpyArrayDescriptor]] = [
        ("reference", item.reference),
        ("target", item.target),
    ]
    if item.variance is not None:
        values.append(("variance", item.variance))
    if item.fit_mask is not None:
        values.append(("fit_mask", item.fit_mask))
    return tuple(values)


def _input_sha256(item: DevicePipelineItem) -> dict[str, str]:
    return {
        name: descriptor.sha256
        for name, descriptor in _item_descriptors(item)
    }


def _observe_content_file(
    path: Path,
    *,
    expected_sha256: str,
    description: str,
    cache: dict[Path, tuple[str, int]],
    validate_content: bool = True,
) -> int:
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(
            f"{description} is not readable: {path}"
        ) from exc
    observed = cache.get(path)
    if observed is None:
        observed = (
            file_sha256(path) if validate_content else expected_sha256,
            size_bytes,
        )
        cache[path] = observed
    if observed != (expected_sha256, size_bytes):
        raise RuntimeError(f"{description} changed before Dragon launch")
    return size_bytes


def _preflight_items(
    items: Sequence[DevicePipelineItem],
    *,
    validate_content: bool = True,
) -> tuple[tuple[WorkItem, ...], list[dict[str, Any]]]:
    if not items:
        raise ValueError("Dragon device pipeline items must not be empty")
    if any(not isinstance(item, DevicePipelineItem) for item in items):
        raise TypeError("items must contain only DevicePipelineItem values")
    item_ids = [item.item_id for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Dragon device pipeline item IDs must be unique")

    observed_files: dict[Path, tuple[str, int]] = {}
    work_items: list[WorkItem] = []
    identities: list[dict[str, Any]] = []
    for item in items:
        by_path: dict[str, dict[str, Any]] = {}
        for role, descriptor in _item_descriptors(item):
            path = Path(descriptor.path)
            size_bytes = _observe_content_file(
                path,
                expected_sha256=descriptor.sha256,
                description=f"{item.item_id} {role} input",
                cache=observed_files,
                validate_content=validate_content,
            )
            descriptor_payload = descriptor.to_payload()
            existing = by_path.get(descriptor.path)
            if existing is None:
                by_path[descriptor.path] = {
                    **descriptor_payload,
                    "size_bytes": size_bytes,
                    "roles": [role],
                }
            else:
                observed_descriptor = {
                    name: existing[name]
                    for name in ("path", "sha256", "shape", "dtype")
                }
                if observed_descriptor != descriptor_payload:
                    raise ValueError(
                        f"item {item.item_id!r} describes one input path "
                        "with conflicting identities"
                    )
                existing["roles"].append(role)
        files = [by_path[path] for path in sorted(by_path)]
        weight_bytes = sum(value["size_bytes"] for value in files)
        work_items.append(
            WorkItem(
                item_id=item.item_id,
                payload=item.to_payload(),
                weight_bytes=weight_bytes,
            )
        )
        identities.append({"item_id": item.item_id, "files": files})
    return tuple(work_items), identities


def _preflight(
    items: Sequence[DevicePipelineItem],
    *,
    config: DevicePipelineConfig,
    validate_content: bool = True,
) -> tuple[
    tuple[WorkItem, ...],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    if not isinstance(config, DevicePipelineConfig):
        raise TypeError("config must be a DevicePipelineConfig")
    if config.device != "cuda:0":
        raise ValueError(
            "Dragon device workers require config.device='cuda:0'; "
            "Dragon maps each physical GPU affinity to local CUDA ordinal 0"
        )

    observed_files: dict[Path, tuple[str, int]] = {}
    checkpoint_path = Path(config.checkpoint_dir) / "checkpoint.pt"
    checkpoint_size = _observe_content_file(
        checkpoint_path,
        expected_sha256=config.checkpoint_sha256,
        description="XScan checkpoint",
        cache=observed_files,
        validate_content=validate_content,
    )
    feature_schema_path = Path(config.feature_schema_path)
    feature_schema_size = _observe_content_file(
        feature_schema_path,
        expected_sha256=config.feature_schema_sha256,
        description="xFit feature schema",
        cache=observed_files,
        validate_content=validate_content,
    )
    work_items, item_identities = _preflight_items(
        items, validate_content=validate_content
    )
    manifest_payload = {
        "schema": DRAGON_DEVICE_PIPELINE_MANIFEST_SCHEMA,
        "backend": _BACKEND,
        "configuration": config.to_payload(),
        "configuration_sha256": config.configuration_sha256,
        "items": [item.to_dict() for item in work_items],
    }
    manifest_sha256 = _json_sha256(manifest_payload)
    input_identity_payload = {
        "schema": DRAGON_DEVICE_PIPELINE_INPUT_IDENTITY_SCHEMA,
        "content_sha256": {
            "checkpoint": config.checkpoint_sha256,
            "feature_schema": config.feature_schema_sha256,
        },
        "configuration_sha256": config.configuration_sha256,
        "configuration_files": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "size_bytes": checkpoint_size,
            },
            "feature_schema": {
                "path": str(feature_schema_path),
                "size_bytes": feature_schema_size,
            },
        },
        "items": item_identities,
    }
    return (
        work_items,
        manifest_payload,
        input_identity_payload,
        manifest_sha256,
    )


def _strict_worker_options(
    payload: Mapping[str, Any],
) -> DevicePipelineConfig:
    options = json_mapping(payload, field="Dragon device pipeline options")
    expected = {
        "schema",
        "backend",
        "configuration",
        "configuration_sha256",
    }
    if set(options) != expected:
        raise ValueError("Dragon device pipeline options have invalid fields")
    if options["schema"] != DRAGON_DEVICE_PIPELINE_OPTIONS_SCHEMA:
        raise ValueError("unsupported Dragon device pipeline options schema")
    if options["backend"] != _BACKEND:
        raise ValueError("Dragon device pipeline backend must be 'cupy'")
    configuration = options["configuration"]
    if not isinstance(configuration, Mapping):
        raise TypeError(
            "Dragon device pipeline configuration must be a mapping"
        )
    config = DevicePipelineConfig.from_payload(configuration)
    if config.device != "cuda:0":
        raise ValueError(
            "Dragon device pipeline worker device must be cuda:0"
        )
    if options["configuration_sha256"] != config.configuration_sha256:
        raise ValueError(
            "Dragon device pipeline configuration SHA-256 differs"
        )
    return config


def _strict_work_item(payload: Mapping[str, Any]) -> WorkItem:
    values = json_mapping(payload, field="Dragon device pipeline work item")
    if set(values) != {"item_id", "payload", "weight_bytes"}:
        raise ValueError(
            "Dragon device pipeline work item has invalid fields"
        )
    if not isinstance(values["item_id"], str):
        raise TypeError(
            "Dragon device pipeline work item ID must be a string"
        )
    return WorkItem.from_dict(values)


def _publish_item_result(
    *,
    work_item: WorkItem,
    item_dir: Path,
    config: DevicePipelineConfig,
    context: DeviceWorkerContext,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    item = DevicePipelineItem.from_payload(work_item.payload)
    if item.item_id != work_item.item_id:
        raise ValueError("Dragon work item and device item IDs differ")
    result = run_device_pipeline_item(item, context)
    if not callable(getattr(result, "to_payload", None)):
        raise TypeError("device pipeline runner returned an invalid result")
    payload = _strict_device_pipeline_result_payload(
        result.to_payload(), item=item, config=config
    )

    item_dir.mkdir(parents=True, exist_ok=False)
    summary_path = item_dir / "summary.json"
    try:
        atomic_write_json(summary_path, payload)
    except BaseException as exc:
        try:
            summary_path.unlink(missing_ok=True)
            item_dir.rmdir()
        except OSError as cleanup_error:
            exc.add_note(
                "device pipeline item cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise

    predictions = cast(list[Mapping[str, Any]], payload["predictions"])
    timing = cast(Mapping[str, Any], payload["timings"])
    evidence = cast(Mapping[str, Any], payload["scientific_evidence"])
    return {
        "run_dir": str(item_dir),
        "summary_path": str(summary_path),
        "requested_backend": _BACKEND,
        "backend": _BACKEND,
        "device": config.device,
        "runtime": {
            "schema": _RUNTIME_SCHEMA,
            "context_load_seconds": context.load_seconds,
            "configuration_sha256": config.configuration_sha256,
            "checkpoint_sha256": config.checkpoint_sha256,
            "feature_schema_sha256": config.feature_schema_sha256,
        },
        "timings_sec": timing["stages_seconds"],
        "wall_sec": {"item_runner": time.perf_counter() - started},
        "device_pipeline": {
            "schema": DEVICE_PIPELINE_RESULT_SCHEMA,
            "summary_sha256": file_sha256(summary_path),
            "result_sha256": payload["result_sha256"],
            "configuration_sha256": config.configuration_sha256,
            "checkpoint_sha256": config.checkpoint_sha256,
            "feature_schema_sha256": config.feature_schema_sha256,
            "input_sha256": payload["input_sha256"],
            "candidate_count": len(predictions),
            "positive_count": sum(
                prediction.get("decision") is True
                for prediction in predictions
            ),
            "scientific_evidence_sha256": evidence["packed_sha256"],
            "scientific_evidence_bytes": evidence["packed_byte_count"],
            "transfers": payload["transfers"],
        },
    }


def _dragon_device_pipeline_worker(
    run_id: str,
    run_dir_raw: str,
    placement_payload: Mapping[str, Any],
    item_payloads: Sequence[Mapping[str, Any]],
    options_payload: Mapping[str, Any],
    results_queue: Any,
    allow_loopback_alias: bool,
) -> None:
    """Dragon target that owns one persistent context for one whole shard."""

    worker_started = time.perf_counter()
    started_at = timestamp_utc()
    run_dir: Path | None = None
    placement: Placement | None = None
    items: tuple[WorkItem, ...] = ()
    try:
        run_dir = Path(run_dir_raw)
        placement = Placement(**dict(placement_payload))
        items = tuple(_strict_work_item(payload) for payload in item_payloads)
        config = _strict_worker_options(options_payload)
        context_attempted = False
        context: DeviceWorkerContext | None = None
        context_error: Exception | None = None

        def item_runner(
            work_item: WorkItem,
            item_dir: Path,
            _options: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            nonlocal context_attempted, context, context_error
            if not context_attempted:
                context_attempted = True
                try:
                    context = DeviceWorkerContext.initialize(config)
                except Exception as exc:
                    context_error = exc
            if context_error is not None:
                raise context_error
            if context is None:
                raise RuntimeError(
                    "Dragon device worker context is unavailable"
                )
            return _publish_item_result(
                work_item=work_item,
                item_dir=item_dir,
                config=config,
                context=context,
            )

        result = _dragon._execute_shard(
            run_id=run_id,
            run_dir=run_dir,
            placement=placement,
            items=items,
            options=json_mapping(
                options_payload, field="Dragon device pipeline options"
            ),
            item_runner=item_runner,
            gpu_identity_loader=_collect_device_pipeline_gpu_identity,
            backend=_BACKEND,
            record_schema=DRAGON_DEVICE_PIPELINE_ITEM_SCHEMA,
            shard_schema=DRAGON_DEVICE_PIPELINE_SHARD_SCHEMA,
            allow_loopback_alias=allow_loopback_alias,
        )
    except Exception as exc:
        result = {
            "schema": DRAGON_DEVICE_PIPELINE_SHARD_SCHEMA,
            "worker_id": (
                placement.worker_id if placement is not None else None
            ),
            "status": "failed",
            "item_count": len(items),
            "success_count": 0,
            "failed_count": len(items),
            "weight_bytes": sum(item.weight_bytes for item in items),
            "item_ids_sha256": item_ids_sha256(items),
            "started_at_utc": started_at,
            "completed_at_utc": timestamp_utc(),
            "worker_wall_sec": time.perf_counter() - worker_started,
            "timings_sec": {},
            "provenance": None,
            "record_write_errors": [],
            "error": error_payload(exc),
        }
        if run_dir is not None and placement is not None:
            worker_path = (
                run_dir / "workers" / f"worker-{placement.worker_id:04d}.json"
            )
            try:
                if not os.path.lexists(worker_path):
                    atomic_write_json(worker_path, result)
            except Exception as artifact_error:
                result["artifact_error"] = error_payload(artifact_error)
    results_queue.put(result)


def _finite_nonnegative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        # JSON integers need not fit the float conversion used by isfinite.
        return False


def _success_record_problems(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    config: DevicePipelineConfig,
    items: Mapping[str, DevicePipelineItem],
    run_dir: Path | None = None,
) -> tuple[str, ...]:
    problems: set[str] = set()
    item_id = record.get("item_id")
    item = items.get(item_id) if isinstance(item_id, str) else None
    record_device = record.get("device")
    if record_device != config.device:
        problems.add("device")
    runtime = record.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "schema",
            "context_load_seconds",
            "configuration_sha256",
            "checkpoint_sha256",
            "feature_schema_sha256",
        }
        or runtime.get("schema") != _RUNTIME_SCHEMA
        or not _finite_nonnegative(runtime.get("context_load_seconds"))
        or runtime.get("configuration_sha256") != config.configuration_sha256
        or runtime.get("checkpoint_sha256") != config.checkpoint_sha256
        or runtime.get("feature_schema_sha256")
        != config.feature_schema_sha256
    ):
        problems.add("runtime")

    metadata = record.get("device_pipeline")
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != _RESULT_METADATA_FIELDS
        or metadata.get("schema") != DEVICE_PIPELINE_RESULT_SCHEMA
        or not _is_sha256(metadata.get("summary_sha256"))
        or not _is_sha256(metadata.get("result_sha256"))
        or metadata.get("configuration_sha256") != config.configuration_sha256
        or metadata.get("checkpoint_sha256") != config.checkpoint_sha256
        or metadata.get("feature_schema_sha256")
        != config.feature_schema_sha256
        or not _is_sha256(metadata.get("scientific_evidence_sha256"))
        or isinstance(metadata.get("scientific_evidence_bytes"), bool)
        or not isinstance(metadata.get("scientific_evidence_bytes"), int)
        or metadata["scientific_evidence_bytes"] <= 0
        or item is None
        or metadata.get("input_sha256") != _input_sha256(item)
    ):
        problems.add("device_pipeline")
        return tuple(sorted(problems))

    assert item is not None
    candidate_count = metadata.get("candidate_count")
    positive_count = metadata.get("positive_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != len(item.candidates)
        or isinstance(positive_count, bool)
        or not isinstance(positive_count, int)
        or not 0 <= positive_count <= candidate_count
    ):
        problems.add("device_pipeline")
    transfers = metadata.get("transfers")
    if (
        not isinstance(transfers, Mapping)
        or transfers.get("pipeline_full_array_d2h_bytes") != 0
        or transfers.get("pipeline_compact_h2d_bytes") != 0
        or transfers.get("terminal_d2h_calls") != 1
    ):
        problems.add("device_pipeline")

    run_id = record.get("run_id")
    summary_relative = record.get("summary_path")
    expected_summary = f"items/{item.item_id}/summary.json"
    if not isinstance(run_id, str) or summary_relative != expected_summary:
        problems.add("summary_path")
        return tuple(sorted(problems))
    assert isinstance(summary_relative, str)
    summary_path = (run_dir or output_root / run_id) / summary_relative
    try:
        summary = read_json_mapping(summary_path)
        if file_sha256(summary_path) != metadata["summary_sha256"]:
            problems.add("device_pipeline")
    except Exception:
        problems.add("summary_path")
        return tuple(sorted(problems))

    try:
        summary = _strict_device_pipeline_result_payload(
            summary, item=item, config=config
        )
    except (TypeError, ValueError):
        problems.add("device_pipeline")
        return tuple(sorted(problems))
    evidence = cast(Mapping[str, Any], summary["scientific_evidence"])
    timing = cast(Mapping[str, Any], summary["timings"])
    predictions = cast(list[Mapping[str, Any]], summary["predictions"])
    if record_device != summary["device"]:
        problems.add("device")
    if (
        summary["result_sha256"] != metadata["result_sha256"]
        or evidence["packed_sha256"] != metadata["scientific_evidence_sha256"]
        or evidence["packed_byte_count"]
        != metadata["scientific_evidence_bytes"]
        or summary.get("transfers") != transfers
        or sum(prediction["decision"] is True for prediction in predictions)
        != positive_count
        or record.get("timings_sec") != timing["stages_seconds"]
    ):
        problems.add("device_pipeline")
    return tuple(sorted(problems))


def _failed_record_problems(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    items: Mapping[str, DevicePipelineItem],
    run_dir: Path | None = None,
) -> tuple[str, ...]:
    item_id = record.get("item_id")
    run_id = record.get("run_id")
    if (
        not isinstance(item_id, str)
        or item_id not in items
        or not isinstance(run_id, str)
    ):
        return ()
    item_dir = (run_dir or output_root / run_id) / "items" / item_id
    return ("failed_item_output",) if os.path.lexists(item_dir) else ()


def run_dragon_device_pipeline(
    *,
    items: Sequence[DevicePipelineItem],
    config: DevicePipelineConfig,
    output_root: Path,
    run_id: str | None = None,
    max_workers: int | None = None,
    result_timeout_sec: float = 120.0,
    worker_timeout_sec: float = 3600.0,
) -> _dragon.DragonBatchResult:
    """Run compact file-backed pipeline items on persistent Dragon workers."""

    invocation_start = time.perf_counter()
    started_at = timestamp_utc()
    preflight_started = time.perf_counter()
    normalized_items = tuple(items)
    (
        work_items,
        manifest_payload,
        input_identity_payload,
        manifest_sha256,
    ) = _preflight(normalized_items, config=config)
    preflight_seconds = time.perf_counter() - preflight_started
    options_payload = {
        "schema": DRAGON_DEVICE_PIPELINE_OPTIONS_SCHEMA,
        "backend": _BACKEND,
        "configuration": config.to_payload(),
        "configuration_sha256": config.configuration_sha256,
    }
    by_id = {item.item_id: item for item in normalized_items}
    resolved_output_root = output_root.expanduser().resolve()
    return _dragon.run_dragon_work_items(
        items=work_items,
        output_root=resolved_output_root,
        run_id=run_id,
        max_workers=max_workers,
        result_timeout_sec=result_timeout_sec,
        worker_timeout_sec=worker_timeout_sec,
        backend=_BACKEND,
        options_payload=options_payload,
        manifest_payload=manifest_payload,
        input_identity_payload=input_identity_payload,
        manifest_sha256=manifest_sha256,
        worker_target=_dragon_device_pipeline_worker,
        run_prefix="dragon-device-pipeline",
        run_schema=DRAGON_DEVICE_PIPELINE_RUN_SCHEMA,
        summary_schema=DRAGON_DEVICE_PIPELINE_SUMMARY_SCHEMA,
        record_schema=DRAGON_DEVICE_PIPELINE_ITEM_SCHEMA,
        shard_schema=DRAGON_DEVICE_PIPELINE_SHARD_SCHEMA,
        coordinator_timings={"manifest_preflight_sec": preflight_seconds},
        invocation_start=invocation_start,
        started_at=started_at,
        success_record_validator=lambda record: _success_record_problems(
            record,
            output_root=resolved_output_root,
            config=config,
            items=by_id,
        ),
        failed_record_validator=lambda record: _failed_record_problems(
            record,
            output_root=resolved_output_root,
            items=by_id,
        ),
    )


__all__ = [
    "DRAGON_DEVICE_PIPELINE_INPUT_IDENTITY_SCHEMA",
    "DRAGON_DEVICE_PIPELINE_ITEM_SCHEMA",
    "DRAGON_DEVICE_PIPELINE_MANIFEST_SCHEMA",
    "DRAGON_DEVICE_PIPELINE_OPTIONS_SCHEMA",
    "DRAGON_DEVICE_PIPELINE_RUN_SCHEMA",
    "DRAGON_DEVICE_PIPELINE_SHARD_SCHEMA",
    "DRAGON_DEVICE_PIPELINE_SUMMARY_SCHEMA",
    "run_dragon_device_pipeline",
]
