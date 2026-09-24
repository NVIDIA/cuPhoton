# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven device pipeline execution through Dragon or MPI."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cuphoton.core.bulk import WorkItem, collect_gpu_identity, json_mapping

from .device_pipeline import DevicePipelineConfig, DevicePipelineItem
from .dragon_pipeline import (
    DRAGON_DEVICE_PIPELINE_OPTIONS_SCHEMA,
    _failed_record_problems,
    _preflight,
    _publish_item_result,
    _strict_worker_options,
    _success_record_problems,
)

if TYPE_CHECKING:
    from cuphoton.core.execution import ExecutionResult, WorkloadSpec

PIPELINE_MANIFEST_SCHEMA = "cuphoton.xscan.pipeline-manifest/v1"


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def load_pipeline_manifest(
    path: Path,
) -> tuple[DevicePipelineConfig, tuple[DevicePipelineItem, ...]]:
    """Read config/items and resolve paths beside the manifest."""

    path = path.expanduser().resolve()
    payload = json_mapping(
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_mapping,
        ),
        field="device pipeline manifest",
    )
    if set(payload) != {"schema", "configuration", "items"}:
        raise ValueError(
            "pipeline manifest requires schema, configuration, items"
        )
    if payload["schema"] != PIPELINE_MANIFEST_SCHEMA:
        raise ValueError("unsupported pipeline manifest schema")

    def resolve(raw: Any) -> str:
        if not isinstance(raw, str) or not raw:
            raise ValueError("manifest paths must be nonempty strings")
        value = Path(raw).expanduser()
        return str((path.parent / value).resolve())

    configuration = json_mapping(
        payload["configuration"], field="configuration"
    )
    for field in ("checkpoint_dir", "feature_schema_path"):
        if field in configuration:
            configuration[field] = resolve(configuration[field])
    config = DevicePipelineConfig.from_payload(configuration)
    if not isinstance(payload["items"], list):
        raise ValueError("pipeline items must be a list")
    items = []
    for raw in payload["items"]:
        item = json_mapping(raw, field="pipeline item")
        for role in ("reference", "target", "variance", "fit_mask"):
            if item.get(role) is not None:
                descriptor = json_mapping(item[role], field=role)
                if "path" in descriptor:
                    descriptor["path"] = resolve(descriptor["path"])
                item[role] = descriptor
        items.append(DevicePipelineItem.from_payload(item))
    return config, tuple(items)


class _PipelineWorker:
    def __init__(self, options: Mapping[str, Any]) -> None:
        self.config = _strict_worker_options(options)
        # Torch must establish its CUDA library load order before CuPy calls
        # CUDA. The executor has already bound this process to one GPU.
        torch = importlib.import_module("torch")
        cp = importlib.import_module("cupy")

        from .device_pipeline import DeviceWorkerContext

        torch.cuda.set_device(0)
        cp.cuda.Device(0).use()
        self.gpu_identity = collect_gpu_identity("cupy")
        self.context = DeviceWorkerContext.initialize(self.config)
        self.closed = False

    def run_item(self, item: WorkItem, output_dir: Path) -> Mapping[str, Any]:
        if self.closed:
            raise RuntimeError("pipeline worker is closed")
        return _publish_item_result(
            work_item=item,
            item_dir=output_dir,
            config=self.config,
            context=self.context,
        )

    def close(self) -> None:
        self.closed = True
        failure = None
        for stream in (
            self.context.producer_stream,
            self.context.consumer_stream,
        ):
            try:
                stream.synchronize()
            except Exception as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(str(exc))
        if failure is not None:
            raise RuntimeError(
                "pipeline worker stream cleanup failed"
            ) from failure


def create_pipeline_worker(options: Mapping[str, Any]) -> _PipelineWorker:
    """Create a process-local context after executor placement."""

    return _PipelineWorker(options)


def prepare_pipeline_workload(
    items: Sequence[DevicePipelineItem],
    config: DevicePipelineConfig,
    *,
    rank: int = 0,
) -> WorkloadSpec:
    """Prepare descriptors with component-owned scientific validation."""

    from cuphoton.core.execution import WorkloadSpec

    items = tuple(items)
    work, manifest, identities, digest = _preflight(
        items, config=config, validate_content=rank == 0
    )
    by_id = {item.item_id: item for item in items}

    def validate_success(record: Mapping[str, Any], round_dir: Path):
        return _success_record_problems(
            record,
            output_root=round_dir.parent,
            run_dir=round_dir,
            config=config,
            items=by_id,
        )

    def validate_failure(record: Mapping[str, Any], round_dir: Path):
        return _failed_record_problems(
            record,
            output_root=round_dir.parent,
            run_dir=round_dir,
            items=by_id,
        )

    return WorkloadSpec(
        items=work,
        options_payload={
            "schema": DRAGON_DEVICE_PIPELINE_OPTIONS_SCHEMA,
            "backend": "cupy",
            "configuration": config.to_payload(),
            "configuration_sha256": config.configuration_sha256,
        },
        manifest_payload=manifest,
        input_identity_payload=identities,
        manifest_sha256=digest,
        backend="cupy",
        worker_factory=create_pipeline_worker,
        success_record_validator=validate_success,
        failed_record_validator=validate_failure,
    )


def run_pipeline_manifest(
    *,
    executor: str,
    manifest_path: Path,
    output_root: Path,
    run_id: str | None = None,
    **options: Any,
) -> ExecutionResult | None:
    """Run an XPOIS/xFit/XScan manifest through the selected executor."""

    from cuphoton.core.executors import run_workload

    def prepare(rank: int):
        config, items = load_pipeline_manifest(manifest_path)
        return prepare_pipeline_workload(items, config, rank=rank)

    return run_workload(
        executor=executor,
        prepare_workload=prepare,
        output_root=output_root,
        run_id=run_id,
        **options,
    )
