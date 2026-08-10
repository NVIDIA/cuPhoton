# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""YAML-backed workflow config helpers for XScan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .types import EarlyStoppingMetric, InputMode, TrainingMode


@dataclass(slots=True)
class ModelConfig:
    """Model settings for the Inada transformer family.

    Image sizes are pixels; depth, head, and embedding lists are ordered from
    the first encoder stage to the last. ``input_mode`` selects pair or
    triplet channel ownership in
    :class:`~cuphoton.xscan.dataset.StampDataset`.
    """

    input_mode: InputMode = "pair"
    image_size: int = 51
    depths: list[int] = field(default_factory=lambda: [3, 3, 3, 3, 3, 3])
    num_heads: list[int] = field(default_factory=lambda: [2, 2, 4, 8, 8, 8])
    embed_dims: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 512, 512]
    )
    decoder_embedding_dim: int = 64
    pos_dim: int = 16
    output_nc: int = 2
    drop_rate: float = 0.2
    attn_drop: float = 0.2
    xfit_feature_names: list[str] = field(default_factory=list)
    xfit_hidden_dim: int = 32
    xfit_dropout: float = 0.0
    xfit_modality_dropout: float = 0.0


@dataclass(slots=True)
class PerformanceConfig:
    """Device, compilation, precision, and data-loader settings.

    ``amp_dtype`` accepts the runtime precision policy, while worker fields
    control process ownership and CPU thread counts for PyTorch data loading.
    """

    amp_dtype: str = "off"
    allow_tf32: bool = False
    cudnn_benchmark: bool = False
    compile: bool = False
    compile_mode: str | None = None
    compile_backend: str | None = None
    compile_threads: int | None = None
    compile_worker_start_method: str | None = None
    num_workers: int = 0
    worker_start_method: str | None = None
    worker_cpu_threads: int = 1
    pin_memory: bool = False
    persistent_workers: bool = False
    non_blocking_transfers: bool = False


@dataclass(slots=True)
class TrainingConfig:
    """Complete configuration for one XScan training workflow.

    Paths are interpreted relative to the invoking process unless already
    absolute. ``device="auto"`` prefers CUDA and otherwise selects CPU.
    xFit fusion requires both ``use_xfit_features=True`` and a feature
    directory; neither setting implicitly enables the other.
    """

    dataset_dir: str
    output_root: str | None = None
    run_name: str | None = None
    benchmark_regime_name: str | None = None
    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"
    training_mode: TrainingMode = "scratch"
    pretrain_checkpoint: str | None = None
    freeze_encoder_stages: list[int] = field(default_factory=list)
    train_split: str = "train"
    val_split: str = "val"
    eval_split: str = "test"
    early_stopping_metric: EarlyStoppingMetric | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    model: ModelConfig = field(default_factory=ModelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    # Fusion options are intentionally appended after every pre-existing
    # field so positional TrainingConfig construction retains its old meaning.
    xfit_feature_dir: str | None = None
    use_xfit_features: bool = False


def _checked_kwargs(
    cls: type, payload: Any, *, path: Path, section: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: {section} section must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown {section} option(s): {', '.join(unknown)}; "
            f"valid options: {', '.join(sorted(known))}"
        )
    return payload


def load_training_config(path: Path) -> TrainingConfig:
    """Load a YAML training configuration.

    Parameters
    ----------
    path
        YAML document containing top-level training values and optional nested
        ``model`` and ``performance`` mappings.

    Returns
    -------
    TrainingConfig
        Valid dataclass configuration using defaults for omitted fields.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_payload = payload.pop("model", {}) or {}
    performance_payload = payload.pop("performance", {}) or {}
    model = ModelConfig(
        **_checked_kwargs(
            ModelConfig, model_payload, path=path, section="model"
        )
    )
    performance = PerformanceConfig(
        **_checked_kwargs(
            PerformanceConfig,
            performance_payload,
            path=path,
            section="performance",
        )
    )
    _checked_kwargs(TrainingConfig, payload, path=path, section="top-level")
    return TrainingConfig(model=model, performance=performance, **payload)


def dump_config(path: Path, config: Any) -> None:
    """Write a dataclass or mapping as human-readable YAML."""

    if is_dataclass(config):
        payload = asdict(config)
    else:
        payload = config
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
