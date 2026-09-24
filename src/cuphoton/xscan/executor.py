# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered inference shards with one checkpoint load per placed worker."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from cuphoton.core.artifacts import array_sha256, file_sha256
from cuphoton.core.bulk import WorkItem, atomic_write_json, read_json_mapping

from .hsc import prediction_group_indices
from .metrics import sigmoid

_SCHEMA = "cuphoton.xscan.executor-chunk/v1"
_OUTPUT_ARRAYS = ("logits", "labels", "probabilities", "sample_index")


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _input_files(options: Mapping[str, Any]) -> list[Path]:
    dataset = Path(options["dataset_dir"])
    files = [Path(options["run_dir"]) / "checkpoint.pt"]
    files.extend(
        dataset / name
        for name in (
            "search.npy",
            "template.npy",
            "labels.npy",
            "split.npy",
            "metadata.jsonl",
        )
    )
    if (dataset / "difference.npy").exists():
        files.append(dataset / "difference.npy")
    if options["xfit_feature_dir"] is not None:
        feature_root = Path(options["xfit_feature_dir"])
        files.extend(
            feature_root / name
            for name in (
                "candidate-id.npy",
                "features.npy",
                "input-image-sha256.npy",
                "schema.json",
            )
        )
    return files


def _check_inputs(
    options: Mapping[str, Any], *, hash_content: bool = False
) -> None:
    for identity in options["input_files"]:
        path = Path(identity["path"])
        stat = path.stat()
        if (
            stat.st_size != identity["size"]
            or stat.st_mtime_ns != identity["mtime_ns"]
        ):
            raise ValueError(f"XScan input changed: {path.name}")
        if hash_content and file_sha256(path) != identity["sha256"]:
            raise ValueError(f"XScan input checksum changed: {path.name}")


def _indices(options: Mapping[str, Any]) -> np.ndarray:
    return prediction_group_indices(
        np.load(
            Path(options["dataset_dir"]) / "split.npy", allow_pickle=False
        ),
        options["split"],
    )


def plan_inference_chunks(
    *,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    batch_size: int = 32,
    task_batches: int = 16,
    num_workers: int | None = None,
    xfit_feature_dir: Path | None = None,
    use_xfit_features: bool = False,
) -> tuple[tuple[WorkItem, ...], dict[str, Any]]:
    """Describe complete original minibatches without importing Torch."""

    for name, value in (
        ("batch_size", batch_size),
        ("task_batches", task_batches),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_workers is not None and (
        type(num_workers) is not int or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    if type(use_xfit_features) is not bool or use_xfit_features != (
        xfit_feature_dir is not None
    ):
        raise ValueError(
            "xFit fusion requires both enablement and a feature directory"
        )
    options = {
        "run_dir": str(run_dir.expanduser().resolve()),
        "dataset_dir": str(dataset_dir.expanduser().resolve()),
        "split": split,
        "batch_size": batch_size,
        "task_batches": task_batches,
        "num_workers": num_workers,
        "use_xfit_features": use_xfit_features,
        "xfit_feature_dir": str(xfit_feature_dir.expanduser().resolve())
        if xfit_feature_dir is not None
        else None,
    }
    selected = _indices(options)
    if not selected.size:
        raise ValueError(
            "distributed XScan inference requires a non-empty split"
        )
    options["sample_count"] = int(selected.size)
    options["sample_indices_sha256"] = array_sha256(selected)
    identities = []
    for path in _input_files(options):
        before = path.stat()
        checksum = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("XScan input changed during planning")
        identities.append(
            {
                "path": str(path),
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": checksum,
            }
        )
    options["input_files"] = identities
    options["configuration_sha256"] = _digest(options)
    task_size = batch_size * task_batches
    items = tuple(
        WorkItem(
            f"samples-{start:08d}-{stop:08d}",
            {
                "start": start,
                "stop": stop,
                "sample_indices_sha256": array_sha256(selected[start:stop]),
                "configuration_sha256": options["configuration_sha256"],
            },
            stop - start,
        )
        for start in range(0, int(selected.size), task_size)
        for stop in (min(start + task_size, int(selected.size)),)
    )
    return items, options


def _device():
    import torch

    return torch.device("cuda:0")


def _gpu_identity() -> Mapping[str, Any]:
    import torch

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    if not uuid:
        raise RuntimeError("Torch did not expose a stable GPU UUID")
    return {
        "backend": "torch",
        "device_index": index,
        "name": properties.name,
        "uuid": str(uuid),
        "pci_bus_id": None,
        "identity_error": None,
    }


class XScanInferenceWorker:
    """Retain the model, performance policy and dataset in one process."""

    def __init__(self, options: Mapping[str, Any]) -> None:
        from .training import (
            load_model_from_checkpoint,
            normalize_performance_config,
            xfit_fit_coverage,
        )
        from .workflows import (
            _checkpoint_dataset,
            _checkpoint_xfit_feature_matrix,
        )

        self.options = dict(options)
        _check_inputs(options, hash_content=True)
        self.device = _device()
        self.gpu_identity = _gpu_identity()
        self.model, checkpoint, self.performance = load_model_from_checkpoint(
            Path(options["run_dir"]), device=self.device
        )
        if options["num_workers"] is not None:
            self.performance = normalize_performance_config(
                replace(self.performance, num_workers=options["num_workers"]),
                device=self.device,
            )
        feature_dir = (
            Path(options["xfit_feature_dir"])
            if options["xfit_feature_dir"] is not None
            else None
        )
        feature_matrix = _checkpoint_xfit_feature_matrix(
            checkpoint=checkpoint,
            dataset_dir=Path(options["dataset_dir"]),
            xfit_feature_dir=feature_dir,
            use_xfit_features=options["use_xfit_features"],
        )
        self.dataset = _checkpoint_dataset(
            checkpoint=checkpoint,
            dataset_dir=Path(options["dataset_dir"]),
            split=options["split"],
            xfit_feature_matrix=feature_matrix,
        )
        if (
            array_sha256(self.dataset.indices)
            != options["sample_indices_sha256"]
        ):
            raise ValueError(
                "XScan worker split differs from planned sample order"
            )
        self.summary = {
            "workflow": "infer",
            "run_dir": options["run_dir"],
            "dataset_dir": options["dataset_dir"],
            "split": options["split"],
            "batch_size": options["batch_size"],
            "performance": asdict(self.performance),
            "xfit_features": {
                "enabled": options["use_xfit_features"],
                "feature_dir": options["xfit_feature_dir"],
                "feature_names": list(self.dataset.xfit_feature_names),
                "bundle_identity": self.dataset.xfit_feature_bundle_identity,
                "join_diagnostics": self.dataset.xfit_join_diagnostics,
                "fit_coverage": xfit_fit_coverage(
                    self.dataset, split=options["split"]
                )
                if options["use_xfit_features"]
                else None,
            },
        }
        _check_inputs(options, hash_content=True)

    def run_item(self, item: WorkItem, item_dir: Path) -> Mapping[str, Any]:
        from .training import predict_dataset

        started = time.perf_counter()
        start, stop = _range(item, self.options)
        _check_inputs(self.options)
        dataset = copy.copy(self.dataset)
        dataset.indices = self.dataset.indices[start:stop]
        if (
            array_sha256(dataset.indices)
            != item.payload["sample_indices_sha256"]
        ):
            raise ValueError(
                "XScan item sample order differs from its descriptor"
            )
        payload = predict_dataset(
            model=self.model,
            dataset=dataset,
            batch_size=self.options["batch_size"],
            device=self.device,
            performance=self.performance,
        )
        _check_inputs(self.options)
        prediction_seconds = time.perf_counter() - started
        item_dir.mkdir(parents=True, exist_ok=False)
        arrays = {
            name: payload[name]
            for name in ("logits", "labels", "probabilities")
        }
        arrays["sample_index"] = dataset.indices
        checksums = {}
        for name, array in arrays.items():
            path = item_dir / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            checksums[name] = file_sha256(path)
        atomic_write_json(
            item_dir / "summary.json",
            {
                "schema": _SCHEMA,
                "item_id": item.item_id,
                **dict(item.payload),
                "artifact_sha256": checksums,
                "metadata_rows": payload["metadata_rows"],
                "inference_summary": self.summary,
            },
        )
        return {
            "run_dir": str(item_dir),
            "summary_path": str(item_dir / "summary.json"),
            "requested_backend": "torch",
            "backend": "torch",
            "device": str(self.device),
            "runtime": {
                "configuration_sha256": self.options["configuration_sha256"]
            },
            "timings_sec": {"predict_sec": prediction_seconds},
            "wall_sec": {"item_runner": time.perf_counter() - started},
        }

    def close(self) -> None:
        """Release model and dataset ownership after the final round."""
        self.model = None
        self.dataset = None


def create_xscan_worker(options: Mapping[str, Any]) -> XScanInferenceWorker:
    """Construct the Torch worker after executor device binding."""
    return XScanInferenceWorker(options)


def _range(item: WorkItem, options: Mapping[str, Any]) -> tuple[int, int]:
    start, stop = item.payload.get("start"), item.payload.get("stop")
    batch_size = options["batch_size"]
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= options["sample_count"]
        or start % batch_size
        or (stop != options["sample_count"] and stop % batch_size)
        or item.payload.get("configuration_sha256")
        != options["configuration_sha256"]
    ):
        raise ValueError("invalid XScan chunk or minibatch boundary")
    return start, stop


def _read_result(item: WorkItem, round_dir: Path, options: Mapping[str, Any]):
    item_dir = round_dir / "items" / item.item_id
    summary = read_json_mapping(item_dir / "summary.json")
    expected = {
        "schema": _SCHEMA,
        "item_id": item.item_id,
        **dict(item.payload),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("XScan chunk summary identity mismatch")
    start, stop = _range(item, options)
    arrays = {}
    for name in _OUTPUT_ARRAYS:
        path = item_dir / f"{name}.npy"
        if file_sha256(path) != summary["artifact_sha256"].get(name):
            raise ValueError("XScan chunk result checksum mismatch")
        values = np.load(path, allow_pickle=False)
        if values.shape != (stop - start,):
            raise ValueError("XScan chunk row count mismatch")
        arrays[name] = values
    if (
        array_sha256(arrays["sample_index"])
        != item.payload["sample_indices_sha256"]
    ):
        raise ValueError("XScan chunk sample identity mismatch")
    if not np.all(np.isfinite(arrays["logits"])) or not np.array_equal(
        arrays["probabilities"], sigmoid(arrays["logits"])
    ):
        raise ValueError(
            "XScan chunk probabilities differ from the original host sigmoid"
        )
    return summary, arrays


def finalize_inference_round(
    round_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    items: Sequence[WorkItem],
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Merge ordered predictions without fitting thresholds or metrics."""
    observed = [record.get("item_id") for record in records]
    if (
        len(observed) != len(items)
        or set(observed) != {item.item_id for item in items}
        or any(record.get("status") != "success" for record in records)
    ):
        raise ValueError(
            "XScan merge requires one successful result per chunk"
        )
    _check_inputs(options, hash_content=True)
    selected = _indices(options)
    metadata_path = Path(options["dataset_dir"]) / "metadata.jsonl"
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    ]
    labels = np.load(
        Path(options["dataset_dir"]) / "labels.npy", allow_pickle=False
    )
    chunks = []
    inference_summary = None
    next_row = 0
    for item in sorted(items, key=lambda item: item.payload["start"]):
        start, stop = _range(item, options)
        if start != next_row:
            raise ValueError("XScan chunks overlap or omit sample rows")
        summary, arrays = _read_result(item, round_dir, options)
        expected_rows = [
            {**metadata[int(index)], "sample_index": int(index)}
            for index in selected[start:stop]
        ]
        if (
            summary["metadata_rows"] != expected_rows
            or not np.array_equal(
                arrays["sample_index"], selected[start:stop]
            )
            or not np.array_equal(
                arrays["labels"],
                labels[selected[start:stop]].astype(np.int64),
            )
        ):
            raise ValueError(
                "XScan chunk sample identities or labels differ from input"
            )
        current_summary = summary["inference_summary"]
        if (
            inference_summary is not None
            and current_summary != inference_summary
        ):
            raise ValueError(
                "XScan checkpoint, feature or performance settings "
                "differ across workers"
            )
        inference_summary = current_summary
        chunks.append(arrays)
        next_row = stop
    if next_row != options["sample_count"] or inference_summary is None:
        raise ValueError("XScan chunks do not cover the selected split")
    output = round_dir / "scientific"
    output.mkdir(exist_ok=False)
    checksums = {}
    for name in _OUTPUT_ARRAYS:
        path = output / f"{name}.npy"
        np.save(
            path,
            np.concatenate([chunk[name] for chunk in chunks]),
            allow_pickle=False,
        )
        checksums[name] = file_sha256(path)
    # Paths in this execution-owned scientific summary are relative to itself,
    # rather than to the read-only training run directory.
    final_summary = {
        **inference_summary,
        "output_dir": str(output),
        "saved": {name: f"{name}.npy" for name in _OUTPUT_ARRAYS},
        "artifact_sha256": checksums,
        "configuration_sha256": options["configuration_sha256"],
    }
    atomic_write_json(output / "summary.json", final_summary)
    return {
        "summary_path": "scientific/summary.json",
        "sample_count": next_row,
        "artifact_sha256": checksums,
    }


def prepare_inference_workload(**kwargs):
    """Preflight inference descriptors without importing Torch."""
    from cuphoton.core.execution import WorkloadSpec

    items, options = plan_inference_chunks(**kwargs)
    by_id = {item.item_id: item for item in items}

    def validate(record, round_dir):
        try:
            _read_result(by_id[record["item_id"]], round_dir, options)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

    manifest = {
        "schema": "cuphoton.xscan.executor-manifest/v1",
        "items": [item.to_dict() for item in items],
        "options": options,
    }
    return WorkloadSpec(
        items=items,
        options_payload=options,
        manifest_payload=manifest,
        input_identity_payload={"files": options["input_files"]},
        manifest_sha256=_digest(manifest),
        backend="torch",
        worker_factory=create_xscan_worker,
        success_record_validator=validate,
        finalize_round=lambda round_dir, records: finalize_inference_round(
            round_dir, records, items=items, options=options
        ),
    )
