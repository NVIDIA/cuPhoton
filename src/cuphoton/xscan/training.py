# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Training and inference helpers for XScan."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cuphoton import __version__

from .config import PerformanceConfig, TrainingConfig, dump_config
from .dataset import StampDataset, load_metadata_rows
from .metrics import ensure_finite_scores, evaluate_predictions, sigmoid
from .model import build_model
from .types import (
    CollatedDatasetBatch,
    CollatedModelFeatures,
    EarlyStoppingMetric,
    ModelFeatures,
    TrainingMode,
    XFitCoverage,
)

# ROC/PR AUC are ranking metrics (higher is better) and undefined for a
# single-class validation split; val_loss is always defined and minimized, so
# it is the selection metric to fall back to when a split lacks both classes.
AUC_SELECTION_METRICS = {"val_roc_auc", "val_pr_auc"}
LOSS_SELECTION_METRICS = {"val_loss"}
EARLY_STOPPING_METRICS = AUC_SELECTION_METRICS | LOSS_SELECTION_METRICS
LSSTCOMCAM_PLACEHOLDER_LABEL_SOURCE = "unlabeled_lsstcomcam_smoke_placeholder"
_STABLE_ENTITY_FIELD_ALIASES = {
    "candidate_diaObjectId": "dia_object_id",
    "diaObjectId": "dia_object_id",
    "dia_object_id": "dia_object_id",
}
XFIT_COVERAGE_MISMATCH_THRESHOLD = 0.05


def val_bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean BCE-with-logits loss, computed in the numerically stable form.

    Matches ``nn.BCEWithLogitsLoss`` (mean reduction) but stays in numpy so it
    can be evaluated from the prediction payload; unlike ROC/PR AUC it is
    defined for a single-class validation split.
    """
    x = np.asarray(logits, dtype=np.float64)
    z = np.asarray(labels, dtype=np.float64)
    return float(
        np.mean(np.maximum(x, 0.0) - x * z + np.log1p(np.exp(-np.abs(x))))
    )


def selection_value(metric_name: str, raw_value: object) -> float | None:
    """Direction-normalized selection score (higher is always better).

    Loss metrics are negated so the shared ``> best`` comparison maximizes
    them too; ``None`` (an undefined metric) propagates so callers can reject
    it rather than compare against a fabricated number.
    """
    if raw_value is None:
        return None
    value = float(raw_value)
    return -value if metric_name in LOSS_SELECTION_METRICS else value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def normalize_performance_config(
    config: PerformanceConfig,
    *,
    device: torch.device,
) -> PerformanceConfig:
    worker_start_method = config.worker_start_method
    if worker_start_method is not None:
        worker_start_method = str(worker_start_method).strip().lower()
        if worker_start_method in {"", "none"}:
            worker_start_method = None
    if (
        worker_start_method is None
        and int(config.num_workers) > 0
        and device.type == "cuda"
    ):
        worker_start_method = "spawn"
    if worker_start_method not in {None, "spawn", "forkserver", "fork"}:
        raise ValueError(
            "worker_start_method must be one of: spawn, forkserver, fork, "
            "none"
        )
    worker_cpu_threads = int(config.worker_cpu_threads)
    if worker_cpu_threads < 0:
        raise ValueError("worker_cpu_threads must be non-negative")
    compile_threads = config.compile_threads
    if compile_threads is not None:
        compile_threads = int(compile_threads)
        if compile_threads <= 0:
            raise ValueError("compile_threads must be positive when provided")
    compile_worker_start_method = config.compile_worker_start_method
    if compile_worker_start_method is not None:
        compile_worker_start_method = (
            str(compile_worker_start_method).strip().lower()
        )
        if compile_worker_start_method in {"", "none"}:
            compile_worker_start_method = None
    if (
        compile_worker_start_method is None
        and bool(config.compile)
        and device.type == "cuda"
    ):
        compile_worker_start_method = "spawn"
    if compile_worker_start_method not in {
        None,
        "spawn",
        "fork",
        "subprocess",
    }:
        raise ValueError(
            "compile_worker_start_method must be one of: spawn, fork, "
            "subprocess, none"
        )
    pin_memory = bool(config.pin_memory and device.type == "cuda")
    persistent_workers = bool(
        config.persistent_workers and config.num_workers > 0
    )
    non_blocking = bool(config.non_blocking_transfers and pin_memory)
    return PerformanceConfig(
        amp_dtype=config.amp_dtype,
        allow_tf32=bool(config.allow_tf32),
        cudnn_benchmark=bool(config.cudnn_benchmark),
        compile=bool(config.compile),
        compile_mode=config.compile_mode,
        compile_backend=config.compile_backend,
        compile_threads=compile_threads,
        compile_worker_start_method=compile_worker_start_method,
        num_workers=int(config.num_workers),
        worker_start_method=worker_start_method,
        worker_cpu_threads=worker_cpu_threads,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        non_blocking_transfers=non_blocking,
    )


def resolve_amp_dtype(
    config: PerformanceConfig,
    *,
    device: torch.device,
) -> torch.dtype | None:
    mode = str(config.amp_dtype).strip().lower()
    if mode in {"off", "none", "false"}:
        return None
    if device.type != "cuda":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be one of: off, bf16, fp16")


def autocast_context(
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def make_dataloader(
    dataset: StampDataset,
    *,
    batch_size: int,
    shuffle: bool,
    performance: PerformanceConfig,
) -> DataLoader[CollatedDatasetBatch]:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": performance.num_workers,
        "pin_memory": performance.pin_memory,
        "persistent_workers": performance.persistent_workers,
    }
    if (
        performance.num_workers > 0
        and performance.worker_start_method is not None
    ):
        kwargs["multiprocessing_context"] = mp.get_context(
            performance.worker_start_method
        )
    # PyTorch types DataLoader by its uncollated Dataset item even though the
    # default collator changes this tuple sample into the list batch above.
    return cast(
        DataLoader[CollatedDatasetBatch],
        DataLoader(dataset, **kwargs),
    )


def move_feature_batch(
    features: CollatedModelFeatures,
    *,
    device: torch.device,
    non_blocking: bool,
) -> ModelFeatures:
    """Move an image-only or image/xFit feature batch to one device."""

    if isinstance(features, (list, tuple)):
        if len(features) != 2 or any(
            not isinstance(value, torch.Tensor) for value in features
        ):
            raise TypeError(
                "fusion feature batches must contain two Torch tensors"
            )
        return (
            features[0].to(device, non_blocking=non_blocking),
            features[1].to(device, non_blocking=non_blocking),
        )
    if not isinstance(features, torch.Tensor):
        raise TypeError("feature batches must contain Torch tensors")
    return features.to(device, non_blocking=non_blocking)


def forward_feature_batch(
    model: torch.nn.Module,
    features: ModelFeatures,
) -> torch.Tensor:
    """Dispatch image-only or late-fusion batches without changing callers."""

    if isinstance(features, tuple):
        if len(features) != 2:
            raise ValueError(
                "fusion batches must contain images and xFit features"
            )
        return model(features[0], xfit_features=features[1])
    return model(features)


def configure_worker_environment(
    performance: PerformanceConfig,
) -> dict[str, str]:
    if performance.num_workers <= 0 or performance.worker_cpu_threads <= 0:
        return {}
    thread_value = str(performance.worker_cpu_threads)
    updated: dict[str, str] = {}
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) in {None, ""}:
            os.environ[name] = thread_value
            updated[name] = thread_value
    return updated


def configure_compile_environment(
    performance: PerformanceConfig,
) -> dict[str, str]:
    if not performance.compile:
        return {}
    updated: dict[str, str] = {}
    if performance.compile_threads is not None:
        value = str(performance.compile_threads)
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = value
        updated["TORCHINDUCTOR_COMPILE_THREADS"] = value
    if performance.compile_worker_start_method is not None:
        value = str(performance.compile_worker_start_method)
        os.environ["TORCHINDUCTOR_WORKER_START"] = value
        updated["TORCHINDUCTOR_WORKER_START"] = value
    return updated


def configure_runtime(
    *,
    performance: PerformanceConfig,
    device: torch.device,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "device_type": device.type,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "requested": asdict(performance),
    }
    if device.type != "cuda":
        runtime.update(
            {
                "gpu_name": None,
                "gpu_compute_capability": None,
                "blackwell_like": False,
                "worker_environment": {},
            }
        )
        return runtime

    torch.backends.cuda.matmul.allow_tf32 = bool(performance.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(performance.allow_tf32)
    torch.backends.cudnn.benchmark = bool(performance.cudnn_benchmark)
    torch.set_float32_matmul_precision(
        "high" if performance.allow_tf32 else "highest"
    )
    props = torch.cuda.get_device_properties(device)
    runtime.update(
        {
            "gpu_name": props.name,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "blackwell_like": bool(props.major >= 12),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "worker_environment": {},
        }
    )
    return runtime


# label_source values that carry no real provenance and must be rejected.
_UNTRUSTED_SOURCES = frozenset({"", "missing", "none", "null", "nan"})
# Substrings that mark placeholder / unlabeled provenance in a label_source.
_PLACEHOLDER_SOURCE_MARKERS = ("placeholder", "unlabeled")


def _is_binary_label(value: Any) -> bool:
    # A metadata label must be exactly 0 or 1 (no rounding): 0.49 is invalid.
    if isinstance(value, bool):
        return True
    if not isinstance(value, (int, float)):
        return False
    return bool(np.isfinite(value)) and float(value) in (0.0, 1.0)


def _is_untrusted_source(raw_source: Any) -> bool:
    if not isinstance(raw_source, str):
        return True  # None, bool, numeric -> no real provenance
    normalized = raw_source.strip().lower()
    if normalized in _UNTRUSTED_SOURCES:
        return True
    if normalized == LSSTCOMCAM_PLACEHOLDER_LABEL_SOURCE.strip().lower():
        return True
    return any(marker in normalized for marker in _PLACEHOLDER_SOURCE_MARKERS)


def _stable_entity_identity(value: object) -> str | None:
    """Normalize numeric stable IDs across integer and string encodings."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    if isinstance(value, int):
        return None if value == 0 else f"integer:{value}"
    normalized = value.strip()
    if not normalized:
        return None
    try:
        integer = int(normalized, 10)
    except ValueError:
        return f"string:{normalized}"
    return None if integer == 0 else f"integer:{integer}"


def check_training_label_provenance(dataset_dir: Path) -> dict[str, Any]:
    rows = load_metadata_rows(dataset_dir)
    raw_labels = np.load(dataset_dir / "labels.npy")
    sample_count = int(raw_labels.shape[0])
    # Validate the ORIGINAL label domain before any cast: labels must be
    # integer-valued and drawn from {0, 1}. Casting to int first would
    # silently truncate floats like 0.9/1.1 and hide a third class such as 2.
    labels_binary = bool(
        np.all(np.isfinite(raw_labels))
        and np.all(raw_labels == np.round(raw_labels))
        and np.all(np.isin(raw_labels, (0, 1)))
    )
    labels = np.round(raw_labels).astype(np.int64) if labels_binary else None
    label_counts = (
        np.bincount(labels, minlength=2) if labels is not None else None
    )
    split_names: list[str] | None = None
    split_storage_invalid = False
    split_path = dataset_dir / "split.npy"
    if split_path.is_file():
        raw_split = np.load(split_path, allow_pickle=False)
        if (
            raw_split.ndim != 1
            or raw_split.shape[0] != sample_count
            or not np.all(np.isin(raw_split, (0, 1, 2)))
        ):
            split_storage_invalid = True
        else:
            split_names = [
                ("train", "val", "test")[int(value)] for value in raw_split
            ]

    label_sources: dict[str, int] = {}
    target_available = {"true": 0, "false": 0, "missing": 0, "invalid": 0}
    untrusted_source_count = 0
    label_mismatch_count = 0
    label_field_missing_count = 0
    label_field_not_binary_count = 0
    split_group_splits: dict[str, set[str]] = {}
    split_group_values: dict[str, Any] = {}
    invalid_split_group_count = 0
    invalid_split_count = 0
    entity_splits: dict[tuple[str, str], set[str]] = {}
    entity_values: dict[tuple[str, str], Any] = {}
    for index, row in enumerate(rows):
        raw_source = row.get("label_source")
        source_key = "missing" if raw_source is None else str(raw_source)
        label_sources[source_key] = label_sources.get(source_key, 0) + 1
        if _is_untrusted_source(raw_source):
            untrusted_source_count += 1
        available = row.get("target_label_available")
        if available is None:
            target_available["missing"] += 1
        elif available is True:
            target_available["true"] += 1
        elif available is False:
            target_available["false"] += 1
        else:
            # Non-bool (e.g. the string "false") is invalid provenance.
            target_available["invalid"] += 1
        # Every row must carry a raw binary label equal to labels.npy at its
        # index -- no rounding, no "only when the field is present".
        if "label" not in row:
            label_field_missing_count += 1
        elif not _is_binary_label(row["label"]):
            label_field_not_binary_count += 1
        elif (
            labels is not None
            and index < sample_count
            and int(row["label"]) != int(labels[index])
        ):
            label_mismatch_count += 1
        if split_names is None or index >= len(split_names):
            continue
        split = split_names[index]
        metadata_split = row.get("split")
        if metadata_split is not None and metadata_split != split:
            invalid_split_count += 1
        split_group = row.get("split_group")
        if (
            isinstance(split_group, bool)
            or not isinstance(split_group, (str, int))
            or (isinstance(split_group, str) and not split_group.strip())
        ):
            invalid_split_group_count += 1
        else:
            group_key = f"{type(split_group).__name__}:{split_group}"
            split_group_values[group_key] = split_group
            split_group_splits.setdefault(group_key, set()).add(split)
        for field, canonical_field in _STABLE_ENTITY_FIELD_ALIASES.items():
            entity = row.get(field)
            normalized_entity = _stable_entity_identity(entity)
            if normalized_entity is None:
                continue
            entity_key = (
                canonical_field,
                normalized_entity,
            )
            entity_values[entity_key] = entity
            entity_splits.setdefault(entity_key, set()).add(split)

    placeholder_count = label_sources.get(
        LSSTCOMCAM_PLACEHOLDER_LABEL_SOURCE,
        0,
    )
    errors = []
    if placeholder_count:
        errors.append("lsstcomcam_placeholder_labels_present")
    if not labels_binary:
        errors.append("labels_not_binary")
    if len(rows) != sample_count:
        # load_metadata_rows returns [] when metadata.jsonl is absent, so a
        # metadata-stripped dataset previously passed silently.
        errors.append("metadata_incomplete")
    if untrusted_source_count:
        errors.append("label_source_untrusted")
    if target_available["false"]:
        errors.append("target_label_unavailable")
    if target_available["invalid"]:
        errors.append("target_label_available_invalid")
    if label_field_missing_count:
        errors.append("metadata_label_missing")
    if label_field_not_binary_count:
        errors.append("metadata_label_not_binary")
    if label_mismatch_count:
        errors.append("metadata_label_mismatch")
    if invalid_split_count:
        errors.append("metadata_split_mismatch")
    if split_storage_invalid:
        errors.append("split_assignments_invalid")
    if invalid_split_group_count:
        errors.append("metadata_split_group_invalid")
    cross_split_groups = [
        {
            "split_group": split_group_values[key],
            "splits": sorted(splits),
        }
        for key, splits in sorted(split_group_splits.items())
        if len(splits) > 1
    ]
    cross_split_entities = [
        {
            "field": key[0],
            "value": entity_values[key],
            "splits": sorted(splits),
        }
        for key, splits in sorted(entity_splits.items())
        if len(splits) > 1
    ]
    if cross_split_groups:
        errors.append("split_group_crosses_splits")
    if cross_split_entities:
        errors.append("stable_entity_crosses_splits")
    if labels is not None and int(label_counts[0]) == 0:
        errors.append("negative_labels_missing")
    if labels is not None and int(label_counts[1]) == 0:
        errors.append("positive_labels_missing")

    return {
        "dataset_dir": str(dataset_dir.expanduser().resolve()),
        "ok": not errors,
        "errors": errors,
        "metadata_row_count": len(rows),
        "sample_count": sample_count,
        "labels_binary": labels_binary,
        "label_counts": {
            "negative": int(label_counts[0]) if labels is not None else None,
            "positive": int(label_counts[1]) if labels is not None else None,
        },
        "label_sources": dict(sorted(label_sources.items())),
        "placeholder_label_source": LSSTCOMCAM_PLACEHOLDER_LABEL_SOURCE,
        "placeholder_label_count": int(placeholder_count),
        "untrusted_source_count": int(untrusted_source_count),
        "metadata_label_mismatch_count": int(label_mismatch_count),
        "metadata_label_missing_count": int(label_field_missing_count),
        "metadata_label_not_binary_count": int(label_field_not_binary_count),
        "invalid_split_count": int(invalid_split_count),
        "invalid_split_group_count": int(invalid_split_group_count),
        "cross_split_groups": cross_split_groups,
        "cross_split_entities": cross_split_entities,
        "target_label_available": target_available,
    }


def require_training_label_provenance(dataset_dir: Path) -> dict[str, Any]:
    provenance = check_training_label_provenance(dataset_dir)
    placeholder_count = int(provenance["placeholder_label_count"])
    if placeholder_count:
        raise ValueError(
            "refusing to train on LSSTComCam smoke placeholder labels: "
            f"{placeholder_count}/{provenance['metadata_row_count']} "
            "metadata rows still use "
            f"label_source={LSSTCOMCAM_PLACEHOLDER_LABEL_SOURCE!r}. "
            "Review or truth-link all rows and materialize labels with "
            "review-apply before running pair/triplet ablations."
        )
    if provenance["cross_split_groups"] or provenance["cross_split_entities"]:
        raise ValueError(
            "refusing to train with group leakage across train/validation/"
            "test splits: "
            f"{len(provenance['cross_split_groups'])} split_group value(s), "
            f"{len(provenance['cross_split_entities'])} stable entity "
            "value(s). Rebuild group-aware splits before training."
        )
    if provenance["errors"]:
        raise ValueError(
            "refusing to train on labels that are not real-bogus usable: "
            + ", ".join(provenance["errors"])
        )
    return provenance


def maybe_compile_model(
    model: torch.nn.Module,
    *,
    performance: PerformanceConfig,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    compile_requested = bool(performance.compile)
    info = {
        "requested": compile_requested,
        "enabled": False,
        "mode": performance.compile_mode,
        "backend": performance.compile_backend,
    }
    if not compile_requested:
        return model, info
    if not hasattr(torch, "compile"):
        info["reason"] = "torch.compile unavailable"
        return model, info

    kwargs: dict[str, Any] = {}
    if performance.compile_mode is not None:
        kwargs["mode"] = performance.compile_mode
    if performance.compile_backend is not None:
        kwargs["backend"] = performance.compile_backend
    compiled = torch.compile(model, **kwargs)
    info["enabled"] = True
    return compiled, info


def normalize_early_stopping_metric(
    value: object,
) -> EarlyStoppingMetric | None:
    metric = value
    if metric is not None:
        metric = str(metric).strip().lower()
        if metric in {"", "none", "off", "false"}:
            metric = None
    if metric is None:
        return None
    if metric not in EARLY_STOPPING_METRICS:
        raise ValueError(
            "early_stopping_metric must be one of: "
            + ", ".join(sorted(EARLY_STOPPING_METRICS))
        )
    return cast(EarlyStoppingMetric, metric)


def normalize_early_stopping_config(config: TrainingConfig) -> dict[str, Any]:
    metric = normalize_early_stopping_metric(config.early_stopping_metric)
    patience = config.early_stopping_patience
    enabled = metric is not None or patience is not None
    if not enabled:
        return {
            "enabled": False,
            "metric": "val_roc_auc",
            "patience": None,
            "min_delta": float(config.early_stopping_min_delta),
            "stopped_early": False,
            "stop_epoch": None,
            "best_epoch": None,
        }

    metric = metric or "val_roc_auc"
    patience = 0 if patience is None else int(patience)
    if patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    min_delta = float(config.early_stopping_min_delta)
    if min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    return {
        "enabled": True,
        "metric": metric,
        "patience": patience,
        "min_delta": min_delta,
        "stopped_early": False,
        "stop_epoch": None,
        "best_epoch": None,
    }


def build_runtime_summary(
    *,
    performance: PerformanceConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    compile_info: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "requested": asdict(performance),
        "runtime": {
            **runtime,
            "amp_enabled": bool(amp_dtype is not None),
            "amp_dtype": (
                "bf16"
                if amp_dtype == torch.bfloat16
                else "fp16"
                if amp_dtype == torch.float16
                else "off"
            ),
            "compile": compile_info,
            "compile_threads": performance.compile_threads,
            "compile_worker_start_method": (
                performance.compile_worker_start_method
            ),
            "pin_memory": bool(performance.pin_memory),
            "persistent_workers": bool(performance.persistent_workers),
            "num_workers": int(performance.num_workers),
            "worker_start_method": performance.worker_start_method,
            "worker_cpu_threads": int(performance.worker_cpu_threads),
            "non_blocking_transfers": bool(
                performance.non_blocking_transfers
            ),
        },
    }


def normalize_training_mode(config: TrainingConfig) -> TrainingMode:
    mode = str(config.training_mode).strip().lower()
    if mode not in {"scratch", "fine_tune"}:
        raise ValueError("training_mode must be one of: scratch, fine_tune")
    if mode == "scratch" and config.pretrain_checkpoint is not None:
        raise ValueError(
            "pretrain_checkpoint is only valid when training_mode is "
            "'fine_tune'"
        )
    if mode == "fine_tune" and not config.pretrain_checkpoint:
        raise ValueError(
            "training_mode 'fine_tune' requires pretrain_checkpoint"
        )
    return cast(TrainingMode, mode)


def initialize_from_pretrain(
    model: torch.nn.Module,
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    resolved = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(
        resolved,
        map_location="cpu",
        weights_only=True,
    )
    if "model_state" not in checkpoint:
        raise ValueError("pretrain checkpoint does not contain 'model_state'")
    source_state = checkpoint["model_state"]
    if not isinstance(source_state, dict):
        raise ValueError("pretrain checkpoint model_state must be a mapping")

    current_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    loaded_parameter_count = 0
    for key, source_value in source_state.items():
        target_value = current_state.get(key)
        if target_value is None:
            skipped_missing += 1
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped_shape += 1
            continue
        matched[key] = source_value.detach().cpu().clone()
        loaded_parameter_count += int(source_value.numel())
    if not matched:
        raise ValueError(
            "pretrain checkpoint shares no compatible tensors with the "
            "requested model"
        )
    current_state.update(matched)
    model.load_state_dict(current_state)
    source_model_config = checkpoint.get("model_config")
    return {
        "enabled": True,
        "checkpoint_path": str(resolved),
        "loaded_tensor_count": len(matched),
        "loaded_parameter_count": loaded_parameter_count,
        "skipped_missing_tensor_count": skipped_missing,
        "skipped_shape_mismatch_tensor_count": skipped_shape,
        "source_model_config": source_model_config,
    }


def freeze_encoder_stages(
    model: torch.nn.Module,
    *,
    stage_indices: list[int],
) -> list[int]:
    if not stage_indices:
        return []
    encoder = getattr(getattr(model, "inner", None), "encoder", None)
    if encoder is None:
        raise ValueError(
            "model does not expose an encoder for stage freezing"
        )
    for attr in ("img_projs", "blocks", "norms", "exchanges", "stage_count"):
        if not hasattr(encoder, attr):
            raise ValueError(
                "model encoder does not expose the expected stage layout"
            )
    normalized = sorted({int(value) for value in stage_indices})
    for value in normalized:
        if value < 0 or value >= int(encoder.stage_count):
            raise ValueError(
                "freeze_encoder_stages values must satisfy 0 <= stage < "
                f"{int(encoder.stage_count)}"
            )
    for index in normalized:
        modules = (
            encoder.img_projs[index],
            encoder.blocks[index],
            encoder.norms[index],
            encoder.exchanges[index],
        )
        for module in modules:
            module.requires_grad_(False)
    return normalized


def xfit_fit_coverage(
    dataset: StampDataset,
    *,
    split: str,
) -> XFitCoverage:
    """Summarize fit availability for one split of a fusion dataset."""

    if dataset.xfit_features is None:
        raise ValueError("fit coverage requires an xFit fusion dataset")
    try:
        present_index = dataset.xfit_feature_names.index("fit_present")
    except ValueError as exc:
        raise ValueError(
            "xFit feature schema does not contain fit_present"
        ) from exc
    selected = dataset.xfit_features[dataset.indices, present_index]
    fit_present_count = int(np.count_nonzero(selected == 1.0))
    sample_count = int(selected.size)
    return {
        "split": split,
        "sample_count": sample_count,
        "fit_present_count": fit_present_count,
        "fit_coverage": (
            float(fit_present_count / sample_count)
            if sample_count > 0
            else 0.0
        ),
    }


def train_classifier(
    config: TrainingConfig,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    performance = normalize_performance_config(
        config.performance,
        device=device,
    )
    amp_dtype = resolve_amp_dtype(performance, device=device)
    early_stopping = normalize_early_stopping_config(config)
    selection_metric = str(early_stopping["metric"])
    requested_selection_metric = normalize_early_stopping_metric(
        config.early_stopping_metric
    )
    training_mode = normalize_training_mode(config)
    runtime = configure_runtime(performance=performance, device=device)
    runtime["worker_environment"] = configure_worker_environment(performance)
    runtime["compile_environment"] = configure_compile_environment(
        performance
    )
    dataset_dir = Path(config.dataset_dir).expanduser().resolve()
    label_provenance = require_training_label_provenance(dataset_dir)
    xfit_feature_dir = (
        Path(config.xfit_feature_dir).expanduser().resolve()
        if config.xfit_feature_dir is not None
        else None
    )
    requested_xfit_names = tuple(config.model.xfit_feature_names)
    if not isinstance(config.use_xfit_features, bool):
        raise ValueError("use_xfit_features must be a boolean")
    if config.use_xfit_features and xfit_feature_dir is None:
        raise ValueError(
            "use_xfit_features=true requires top-level xfit_feature_dir"
        )
    if not config.use_xfit_features and xfit_feature_dir is not None:
        raise ValueError("xfit_feature_dir requires use_xfit_features=true")
    if not config.use_xfit_features and requested_xfit_names:
        raise ValueError(
            "model.xfit_feature_names requires use_xfit_features=true"
        )
    shared_xfit_features = None
    if config.use_xfit_features:
        from .xfit_features import load_xfit_feature_matrix

        assert xfit_feature_dir is not None
        shared_xfit_features = load_xfit_feature_matrix(
            dataset_dir=dataset_dir,
            feature_dir=xfit_feature_dir,
            expected_feature_names=requested_xfit_names or None,
        )
        config.model.xfit_feature_names = list(
            shared_xfit_features.feature_names
        )
    train_dataset = StampDataset(
        dataset_dir,
        input_mode=config.model.input_mode,
        split=config.train_split,
        xfit_feature_matrix=shared_xfit_features,
        xfit_feature_names=config.model.xfit_feature_names or None,
    )
    val_dataset = StampDataset(
        dataset_dir,
        input_mode=config.model.input_mode,
        split=config.val_split,
        xfit_feature_matrix=shared_xfit_features,
        xfit_feature_names=config.model.xfit_feature_names or None,
    )
    if shared_xfit_features is not None and (
        train_dataset.xfit_features is not val_dataset.xfit_features
    ):
        raise RuntimeError(
            "training and validation did not share the xFit feature matrix"
        )
    if len(train_dataset) == 0:
        raise ValueError(
            f"training split '{config.train_split}' is empty for "
            f"{dataset_dir}"
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"validation split '{config.val_split}' is empty for "
            f"{dataset_dir}"
        )
    split_fit_coverage = (
        {
            "train": xfit_fit_coverage(
                train_dataset,
                split=config.train_split,
            ),
            "validation": xfit_fit_coverage(
                val_dataset,
                split=config.val_split,
            ),
        }
        if shared_xfit_features is not None
        else None
    )
    # ROC/PR AUC checkpoint selection is undefined for a single-class split.
    # An explicit AUC request remains an error. When no metric was requested,
    # select checkpoints by loss without changing whether early stopping is
    # enabled or its patience.
    val_classes = np.unique(
        np.asarray(val_dataset.labels)[val_dataset.indices]
    )
    if val_classes.size < 2 and selection_metric in AUC_SELECTION_METRICS:
        if requested_selection_metric in AUC_SELECTION_METRICS:
            raise ValueError(
                f"validation split '{config.val_split}' is single-class "
                f"(labels={val_classes.tolist()}); ROC/PR AUC checkpoint "
                f"selection ('{selection_metric}') is undefined for it. Set "
                "early_stopping_metric to 'val_loss' (loss-based selection "
                "is defined for a single-class validation split) or provide "
                "a two-class validation split."
            )
        selection_metric = "val_loss"
        early_stopping["metric"] = selection_metric
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        performance=performance,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        performance=performance,
    )
    base_model = build_model(**asdict(config.model)).to(device)
    transfer = {
        "enabled": False,
        "checkpoint_path": None,
        "loaded_tensor_count": 0,
        "loaded_parameter_count": 0,
        "skipped_missing_tensor_count": 0,
        "skipped_shape_mismatch_tensor_count": 0,
        "source_model_config": None,
    }
    if training_mode == "fine_tune":
        transfer = initialize_from_pretrain(
            base_model,
            checkpoint_path=str(config.pretrain_checkpoint),
        )
    frozen_stages = freeze_encoder_stages(
        base_model,
        stage_indices=list(config.freeze_encoder_stages),
    )
    model, compile_info = maybe_compile_model(
        base_model,
        performance=performance,
    )
    trainable_parameters = [
        parameter
        for parameter in base_model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError(
            "no trainable parameters remain after stage freezing"
        )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(device.type == "cuda" and amp_dtype == torch.float16),
    )

    history: list[dict[str, float | None]] = []
    best_state: dict[str, torch.Tensor] | None = None
    # best_selection is the direction-normalized score we maximize;
    # best_metric_value is the raw selection-metric value at that epoch.
    best_selection = -float("inf")
    best_metric_value: float | None = None
    best_val_auc: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    total_train_time_seconds = 0.0
    peak_gpu_memory_allocated_bytes = None
    peak_gpu_memory_reserved_bytes = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(config.epochs):
        model.train()
        # float64 accumulator: summing many finite float32 losses can itself
        # overflow to inf even when every batch loss is finite.
        running_loss = torch.zeros((), dtype=torch.float64, device=device)
        sample_count = 0
        epoch_start = time.perf_counter()
        for features, labels in train_loader:
            features = move_feature_batch(
                features,
                device=device,
                non_blocking=performance.non_blocking_transfers,
            )
            labels = labels.to(
                device,
                non_blocking=performance.non_blocking_transfers,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device=device, amp_dtype=amp_dtype):
                logits = forward_feature_batch(model, features)
                loss = criterion(logits, labels)
            # GradScaler handles non-finite gradients, but a non-finite
            # forward loss invalidates the run itself. Reject it on every
            # precision path rather than silently changing the training set.
            if not bool(torch.isfinite(loss)):
                raise ValueError(
                    f"non-finite training loss at epoch {epoch + 1}"
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            batch_size = int(labels.shape[0])
            # Accumulate in float64; materialize once per epoch (below).
            running_loss = running_loss + loss.detach().double() * batch_size
            sample_count += batch_size

        # Fence GPU work before reading the wall clock so the final
        # backward/optimizer/accumulation tail is included in the epoch time.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time_seconds = float(time.perf_counter() - epoch_start)
        total_train_time_seconds += epoch_time_seconds
        if sample_count == 0:
            raise ValueError(
                "every training batch produced a non-finite loss at epoch "
                f"{epoch + 1}"
            )
        train_loss = float(running_loss.item()) / sample_count
        if not np.isfinite(train_loss):
            raise ValueError(
                f"non-finite mean training loss at epoch {epoch + 1}"
            )
        val_payload = predict_dataset(
            model=model,
            dataset=val_dataset,
            batch_size=config.batch_size,
            device=device,
            performance=performance,
            _loader=val_loader,
        )
        val_metrics = evaluate_predictions(
            y_true=val_payload["labels"],
            logits=val_payload["logits"],
            metadata_rows=val_payload["metadata_rows"],
        )
        # ROC/PR AUC are None for a single-class split; keep them None (not
        # float(None)) so history stays valid JSON; selection uses val_loss.
        val_roc_auc = val_metrics["roc_auc"]
        val_pr_auc = val_metrics["pr_auc"]
        val_loss = val_bce_with_logits(
            val_payload["logits"], val_payload["labels"]
        )
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_roc_auc": (
                None if val_roc_auc is None else float(val_roc_auc)
            ),
            "val_pr_auc": None if val_pr_auc is None else float(val_pr_auc),
            "epoch_time_seconds": epoch_time_seconds,
            "train_samples_per_second": float(
                sample_count / max(epoch_time_seconds, 1e-12)
            ),
        }
        history.append(epoch_metrics)
        current_selection = selection_value(
            selection_metric, epoch_metrics[selection_metric]
        )
        if current_selection is None:
            raise ValueError(
                f"selection metric '{selection_metric}' is undefined at "
                f"epoch {epoch + 1} (single-class validation split); set "
                "early_stopping_metric to 'val_loss'"
            )
        improved = bool(
            current_selection
            > best_selection + float(early_stopping["min_delta"])
        )
        if improved:
            best_selection = current_selection
            best_metric_value = float(epoch_metrics[selection_metric])
            best_val_auc = epoch_metrics["val_roc_auc"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in base_model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if early_stopping["enabled"] and (
                epochs_without_improvement > int(early_stopping["patience"])
            ):
                early_stopping["stopped_early"] = True
                early_stopping["stop_epoch"] = epoch + 1
                break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in base_model.state_dict().items()
        }
        best_epoch = len(history)
        last = history[-1] if history else None
        best_metric_value = (
            float(last[selection_metric]) if last is not None else None
        )
        best_val_auc = last["val_roc_auc"] if last is not None else None
    early_stopping["best_epoch"] = best_epoch
    checkpoint = {
        "model_state": best_state,
        "model_config": asdict(config.model),
        "train_config": asdict(config),
        "xfit_feature_bundle": train_dataset.xfit_feature_bundle_identity,
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    dump_config(run_dir / "config.yaml", config)
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "package_version": __version__,
        "dataset_dir": str(dataset_dir),
        "backend": "torch",
        "device": str(device),
        "dtype": "float32",
        "training_mode": training_mode,
        "benchmark_regime_name": config.benchmark_regime_name,
        "label_provenance": label_provenance,
        "xfit_features": {
            "enabled": config.use_xfit_features,
            "feature_dir": (
                str(xfit_feature_dir)
                if xfit_feature_dir is not None
                else None
            ),
            "feature_names": list(config.model.xfit_feature_names),
            "schema_version": (
                train_dataset.xfit_feature_schema.get("schema_version")
                if train_dataset.xfit_feature_schema is not None
                else None
            ),
            "join_diagnostics": train_dataset.xfit_join_diagnostics,
            "bundle_identity": train_dataset.xfit_feature_bundle_identity,
            "fit_coverage_by_split": split_fit_coverage,
        },
        "epochs": config.epochs,
        "epochs_completed": len(history),
        "batch_size": config.batch_size,
        "best_val_roc_auc": best_val_auc,
        "best_metric": {
            "name": selection_metric,
            "value": best_metric_value,
        },
        "early_stopping": early_stopping,
        "total_train_time_seconds": total_train_time_seconds,
        "transfer": transfer,
        "frozen_encoder_stages": frozen_stages,
        "performance": build_runtime_summary(
            performance=performance,
            device=device,
            amp_dtype=amp_dtype,
            compile_info=compile_info,
            runtime=runtime,
        ),
        "saved": {
            "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            "config": "config.yaml",
            "history": "history.json",
        },
    }
    if device.type == "cuda":
        peak_gpu_memory_allocated_bytes = int(
            torch.cuda.max_memory_allocated(device)
        )
        peak_gpu_memory_reserved_bytes = int(
            torch.cuda.max_memory_reserved(device)
        )
        summary["gpu_memory"] = {
            "peak_allocated_bytes": peak_gpu_memory_allocated_bytes,
            "peak_reserved_bytes": peak_gpu_memory_reserved_bytes,
            "peak_allocated_gib": float(
                peak_gpu_memory_allocated_bytes / 1024**3
            ),
            "peak_reserved_gib": float(
                peak_gpu_memory_reserved_bytes / 1024**3
            ),
        }
    return summary


def load_model_from_checkpoint(
    run_dir: Path,
    *,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], PerformanceConfig]:
    checkpoint = torch.load(
        run_dir / "checkpoint.pt",
        map_location=device or "cpu",
        weights_only=True,
    )
    model = build_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    target_device = device or torch.device("cpu")
    model.to(target_device)
    performance_payload = checkpoint.get("train_config", {}).get(
        "performance", {}
    )
    performance = PerformanceConfig(**(performance_payload or {}))
    performance = normalize_performance_config(
        performance,
        device=target_device,
    )
    model, _compile_info = maybe_compile_model(
        model,
        performance=performance,
    )
    model.eval()
    return model, checkpoint, performance


@torch.no_grad()
def predict_dataset(
    *,
    model: torch.nn.Module,
    dataset: StampDataset,
    batch_size: int,
    device: torch.device,
    performance: PerformanceConfig | None = None,
    _loader: DataLoader[CollatedDatasetBatch] | None = None,
) -> dict[str, Any]:
    resolved_performance = normalize_performance_config(
        performance or PerformanceConfig(),
        device=device,
    )
    amp_dtype = resolve_amp_dtype(resolved_performance, device=device)
    loader = _loader or make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        performance=resolved_performance,
    )
    # Run inference in eval mode: @torch.no_grad() only disables autograd;
    # it does NOT deactivate Dropout or switch BatchNorm to running stats.
    # Without this, per-epoch validation ran with dropout active and
    # BatchNorm updating its running buffers from the validation set
    # (leaking val data into the saved checkpoint). Restore the caller's
    # mode afterwards.
    # Snapshot every submodule's mode and restore each individually: a caller
    # may intentionally hold some submodules (e.g. specific BatchNorm layers)
    # in eval, which a recursive model.train(was_training) would clobber.
    module_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    logits_chunks = []
    label_chunks = []
    try:
        for features, labels in loader:
            features = move_feature_batch(
                features,
                device=device,
                non_blocking=resolved_performance.non_blocking_transfers,
            )
            with autocast_context(device=device, amp_dtype=amp_dtype):
                logits = forward_feature_batch(model, features)
            logits = logits.detach().float().cpu().numpy()
            logits_chunks.append(logits)
            label_chunks.append(labels.detach().cpu().numpy())
    finally:
        for module, was_training in module_modes:
            module.training = was_training
    logits = (
        np.concatenate(logits_chunks, axis=0)
        if logits_chunks
        else np.zeros((0,))
    )
    logits = ensure_finite_scores(logits, name="prediction logits")
    labels = (
        np.concatenate(label_chunks, axis=0)
        if label_chunks
        else np.zeros((0,))
    )
    rows = load_metadata_rows(dataset.dataset_dir)
    selected_rows = []
    for idx in dataset.indices.tolist():
        row = dict(rows[int(idx)])
        row["sample_index"] = int(idx)
        selected_rows.append(row)
    return {
        "logits": logits,
        "labels": labels.astype(np.int64),
        "metadata_rows": selected_rows,
        "probabilities": ensure_finite_scores(
            sigmoid(logits),
            name="prediction probabilities",
        ),
    }
