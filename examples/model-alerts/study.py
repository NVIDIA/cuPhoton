# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development-only study preparation using the native xScan training API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from evaluation import confusion, paired_field_bootstrap, recall_threshold


def prepare_training(
    raw_dir, output, *, normalization=None, eligible_only=True
):
    """Keep binary unambiguous non-challenge rows; fit scaling on train only.

    Original complete detection products remain in raw_dir for coverage and
    challenge reporting. Row identities survive selection unchanged.
    """
    from cuphoton.xscan.dataset import validate_dataset_dir
    from cuphoton.xscan.training import require_training_label_provenance

    raw_dir, output = Path(raw_dir), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("training output must be absent or empty")
    rows = [
        json.loads(line)
        for line in (raw_dir / "metadata.jsonl").read_text().splitlines()
    ]
    eligible = np.array(
        [
            (not r["ambiguous"] and not r["challenge"]) or not eligible_only
            for r in rows
        ]
    )
    selected = np.flatnonzero(eligible)
    split = np.load(raw_dir / "split.npy")[selected]
    arrays = [
        np.load(raw_dir / f"{name}.npy", mmap_mode="r")[selected]
        for name in ("search", "template", "difference")
    ]
    if normalization is None:
        if not np.any(split == 0):
            raise ValueError("training rows required to fit scaling")
        pair_train = np.concatenate(
            [a[split == 0].ravel() for a in arrays[:2]]
        )
        pair_center = float(np.median(pair_train))
        pair_scale = float(np.std(pair_train))
        diff_scale = float(np.std(arrays[2][split == 0]))
    else:
        pair_center = float(normalization["pair_center"])
        pair_scale = float(normalization["pair_scale"])
        diff_scale = float(normalization["difference_scale"])
    if min(pair_scale, diff_scale) <= 0:
        raise ValueError("nondegenerate training arrays required")
    output.mkdir(parents=True, exist_ok=True)
    for index, (name, array) in enumerate(
        zip(("search", "template", "difference"), arrays, strict=False)
    ):
        center, scale = (
            (pair_center, pair_scale) if index < 2 else (0, diff_scale)
        )
        np.save(
            output / f"{name}.npy",
            ((array - center) / scale).astype(np.float32),
        )
    variance_path = raw_dir / "variance.npy"
    if variance_path.is_file():
        np.save(
            output / "variance.npy",
            np.load(variance_path, mmap_mode="r")[selected] / diff_scale**2,
        )
    np.save(output / "labels.npy", np.load(raw_dir / "labels.npy")[selected])
    np.save(output / "split.npy", split)
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(rows[i]) + "\n" for i in selected)
    )
    receipt = dict(
        pair_center=pair_center,
        pair_scale=pair_scale,
        difference_center=0,
        difference_scale=diff_scale,
        fitted_split="train",
        raw_rows=len(rows),
        selected_rows=len(selected),
        ambiguous_rows=sum(r["ambiguous"] for r in rows),
        challenge_rows=sum(r["challenge"] for r in rows),
        selected_raw_indices=selected.tolist(),
    )
    (output / "normalization.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    validate_dataset_dir(output)
    require_training_label_provenance(output)
    return receipt


def train_development(
    dataset_dir,
    run_dir,
    *,
    mode,
    seed,
    epochs=12,
    device="cuda:0",
    xfit_feature_dir=None,
):
    """Fit native xScan from scratch and freeze a validation threshold.

    This function never accesses a test split or generates reserved fields.
    """
    import torch

    from cuphoton.xscan.config import ModelConfig, TrainingConfig
    from cuphoton.xscan.dataset import StampDataset
    from cuphoton.xscan.training import (
        load_model_from_checkpoint,
        predict_dataset,
        train_classifier,
    )

    run_dir, dataset_dir = Path(run_dir), Path(dataset_dir)
    if run_dir.exists():
        raise ValueError("run directory must not exist")
    run_dir.mkdir(parents=True)
    size = np.load(dataset_dir / "search.npy", mmap_mode="r").shape[-1]
    config = TrainingConfig(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=64,
        learning_rate=3e-4,
        seed=seed,
        device=device,
        early_stopping_metric="val_loss",
        use_xfit_features=xfit_feature_dir is not None,
        xfit_feature_dir=str(xfit_feature_dir) if xfit_feature_dir else None,
        model=ModelConfig(
            input_mode=mode,
            image_size=size,
            depths=[1, 1, 1],
            num_heads=[1, 2, 4],
            embed_dims=[16, 32, 64],
            decoder_embedding_dim=16,
            pos_dim=8,
            drop_rate=0.1,
            attn_drop=0.1,
        ),
    )
    summary = train_classifier(config, run_dir=run_dir)
    model, _, performance = load_model_from_checkpoint(
        run_dir, device=torch.device(device)
    )
    validation = StampDataset(
        dataset_dir,
        input_mode=mode,
        split="val",
        xfit_feature_dir=xfit_feature_dir,
    )
    prediction = predict_dataset(
        model=model,
        dataset=validation,
        batch_size=64,
        device=torch.device(device),
        performance=performance,
    )
    labels, scores = prediction["labels"], prediction["probabilities"]
    threshold = recall_threshold(labels, scores)
    fields = np.array([r["field"] for r in prediction["metadata_rows"]])
    interval = paired_field_bootstrap(
        labels, scores, scores, fields, threshold, threshold
    )
    from cuphoton.xscan.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    receipt = dict(
        validation_roc_auc=auc(fpr, tpr),
        mode=mode,
        seed=seed,
        threshold=threshold,
        validation=confusion(labels, scores, threshold),
        validation_fpr_interval=interval["a"],
        validation_fields=int(len(np.unique(fields))),
        epochs_completed=summary["epochs_completed"],
    )
    np.savez(
        run_dir / "validation.npz",
        labels=labels,
        scores=scores,
        fields=fields,
    )
    (run_dir / "operating-point.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
