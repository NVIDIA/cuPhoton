# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-class validation must not fabricate a checkpoint-selection metric.

ROC/PR AUC are undefined for a single-class validation split. The training
loop now (a) raises an actionable error when AUC selection is requested for
such a split, pointing to ``val_loss``, and (b) selects on ``val_loss`` (a
defined, minimized metric) when that is configured instead.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cuphoton.xscan import training as xscan_training  # noqa: E402
from cuphoton.xscan.config import load_training_config  # noqa: E402
from cuphoton.xscan.training import (  # noqa: E402
    AUC_SELECTION_METRICS,
    EARLY_STOPPING_METRICS,
    normalize_early_stopping_config,
    selection_value,
    train_classifier,
    val_bce_with_logits,
)
from cuphoton.xscan.workflows import evaluate_workflow  # noqa: E402


def test_val_loss_matches_torch_bce() -> None:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(64).astype(np.float64)
    labels = rng.integers(0, 2, size=64)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits),
        torch.tensor(labels, dtype=torch.float64),
    ).item()
    assert val_bce_with_logits(logits, labels) == pytest.approx(expected)


def test_selection_value_normalizes_direction() -> None:
    # Loss is minimized, so it is negated to fit the shared > comparison.
    assert selection_value("val_loss", 0.3) == pytest.approx(-0.3)
    assert selection_value("val_roc_auc", 0.9) == pytest.approx(0.9)
    assert selection_value("val_roc_auc", None) is None


def _early_stopping(metric):
    return SimpleNamespace(
        early_stopping_metric=metric,
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
    )


def test_val_loss_is_a_valid_selection_metric() -> None:
    assert "val_loss" in EARLY_STOPPING_METRICS
    assert "val_loss" not in AUC_SELECTION_METRICS
    assert (
        normalize_early_stopping_config(_early_stopping("val_loss"))["metric"]
        == "val_loss"
    )
    with pytest.raises(ValueError):
        normalize_early_stopping_config(_early_stopping("not_a_metric"))


def _full_row(label: int, index: int) -> dict:
    return {
        "candidate_id": f"c{index}",
        "exposure_id": index,
        "ccd_id": 1,
        "band": "i",
        "x": 10 + index,
        "y": 10 + index,
        "split_group": f"g{index}",
        "label": int(label),
        "label_source": "reviewed",
        "center_source": "catalog",
        "catalog_pool_role": "positive" if label == 1 else "negative",
        "catalog_flux": 100.0,
        "catalog_extendedness": 0.1,
        "fake_id": f"f{index}" if label == 1 else None,
        "autoscan_score": 0.5,
        "diff_snr": 5.0,
        "snr": 5.0,
        "flux_ratio": 0.2,
        "center_offset_radius": 0.0,
        "search_valid_fraction": 1.0,
        "difference_context_valid_fraction": 1.0,
    }


def _write_dataset(directory, *, val_labels) -> None:
    # train has both classes; val is caller-controlled; test has both classes.
    labels = [0, 1, 0, 1] + list(val_labels) + [0, 1]
    splits = [0, 0, 0, 0] + [1] * len(val_labels) + [2, 2]
    n = len(labels)
    rng = np.random.default_rng(0)
    for name in ("search", "template"):
        np.save(
            directory / f"{name}.npy",
            rng.standard_normal((n, 17, 17)).astype(np.float32),
        )
    np.save(directory / "labels.npy", np.array(labels, dtype=np.int64))
    np.save(directory / "split.npy", np.array(splits, dtype=np.int64))
    with open(directory / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for index, label in enumerate(labels):
            handle.write(json.dumps(_full_row(label, index)) + "\n")


def _config(directory, output_root, *, metric=None, epochs=1):
    lines = [
        f"dataset_dir: {directory}",
        f"output_root: {output_root}",
        "run_name: guard",
        f"epochs: {epochs}",
        "batch_size: 4",
        "learning_rate: 0.001",
        "weight_decay: 0.0",
        "seed: 0",
        "device: cpu",
        "train_split: train",
        "val_split: val",
        "eval_split: test",
    ]
    if metric is not None:
        lines.append(f"early_stopping_metric: {metric}")
    lines += [
        "model:",
        "  input_mode: pair",
        "  image_size: 17",
        "  depths: [1, 1]",
        "  num_heads: [1, 1]",
        "  embed_dims: [8, 16]",
        "  decoder_embedding_dim: 8",
        "  pos_dim: 4",
        "  output_nc: 2",
        "  drop_rate: 0.0",
        "  attn_drop: 0.0",
        "",
    ]
    path = directory / "config.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return load_training_config(path)


def test_single_class_val_with_auc_selection_raises(tmp_path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    _write_dataset(dataset, val_labels=[0, 0])  # single-class val
    config = _config(
        dataset,
        tmp_path / "runs",
        metric="val_roc_auc",
    )
    with pytest.raises(ValueError, match="val_loss"):
        train_classifier(config, run_dir=tmp_path / "runs" / "guard")


def test_single_class_val_defaults_to_loss_without_early_stopping(
    tmp_path,
) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    _write_dataset(dataset, val_labels=[0, 0])
    config = _config(dataset, tmp_path / "runs", epochs=2)
    run_dir = tmp_path / "runs" / "guard"
    run_dir.mkdir(parents=True)

    summary = train_classifier(config, run_dir=run_dir)

    assert summary["epochs_completed"] == 2
    assert summary["best_metric"]["name"] == "val_loss"
    assert summary["early_stopping"]["enabled"] is False
    assert summary["early_stopping"]["patience"] is None
    evaluation = evaluate_workflow(
        run_dir=run_dir,
        dataset_dir=dataset,
        split="test",
        batch_size=2,
    )
    diagnostics = evaluation.summary["threshold_diagnostics"]
    assert evaluation.summary["threshold_selection_undefined_reason"] == (
        "single_class_validation_split"
    )
    assert diagnostics["validation_selection_undefined_reason"] == (
        "single_class_validation_split"
    )
    assert "validation_selected" not in diagnostics


def test_disabled_metric_token_uses_loss_without_enabling_early_stopping(
    tmp_path,
) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    _write_dataset(dataset, val_labels=[0, 0])
    config = _config(dataset, tmp_path / "runs", epochs=2)
    config.early_stopping_metric = "off"
    run_dir = tmp_path / "runs" / "guard"
    run_dir.mkdir(parents=True)

    summary = train_classifier(config, run_dir=run_dir)

    assert summary["epochs_completed"] == 2
    assert summary["best_metric"]["name"] == "val_loss"
    assert summary["early_stopping"]["enabled"] is False


def test_single_class_val_with_loss_selection_trains(tmp_path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    _write_dataset(dataset, val_labels=[0, 0])  # single-class val
    config = _config(dataset, tmp_path / "runs", metric="val_loss")
    run_dir = tmp_path / "runs" / "guard"
    run_dir.mkdir(parents=True)
    summary = train_classifier(config, run_dir=run_dir)
    assert summary["best_metric"]["name"] == "val_loss"
    assert summary["best_metric"]["value"] is not None
    # AUC is undefined for the single-class val split -> reported as null.
    assert summary["best_val_roc_auc"] is None
    history = json.loads((run_dir / "history.json").read_text())
    assert history[0]["val_roc_auc"] is None
    assert history[0]["val_loss"] is not None


def test_nonfinite_loss_raises_when_scaler_is_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    class NonFiniteFirstBatch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))
            self.training_forward_count = 0

        def forward(self, features):
            logits = self.bias.expand(features.shape[0])
            if self.training and self.training_forward_count == 0:
                self.training_forward_count += 1
                return logits + float("inf")
            self.training_forward_count += int(self.training)
            return logits

    class EnabledScaler:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def is_enabled(self) -> bool:
            return True

        def scale(self, loss):
            return loss

        def step(self, optimizer) -> None:
            optimizer.step()

        def update(self) -> None:
            return None

    dataset = tmp_path / "ds"
    dataset.mkdir()
    _write_dataset(dataset, val_labels=[0, 1])
    config = _config(dataset, tmp_path / "runs")
    config.batch_size = 2
    run_dir = tmp_path / "runs" / "guard"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        xscan_training,
        "build_model",
        lambda **_kwargs: NonFiniteFirstBatch(),
    )
    monkeypatch.setattr(xscan_training.torch.amp, "GradScaler", EnabledScaler)

    with pytest.raises(
        ValueError,
        match="non-finite training loss at epoch 1",
    ):
        train_classifier(config, run_dir=run_dir)
