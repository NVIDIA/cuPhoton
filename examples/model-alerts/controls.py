# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development controls that retain native provenance guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARRAYS = ("search", "template", "difference", "labels", "split")


def copy_rows(source, output, selected):
    """Copy explicit rows to a fresh directory and retain provenance."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("control output must not exist")
    rows = [
        json.loads(line)
        for line in (source / "metadata.jsonl").read_text().splitlines()
    ]
    selected = np.asarray(selected, dtype=int)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("nonempty row selection required")
    output.mkdir(parents=True)
    for name in ARRAYS:
        np.save(
            output / f"{name}.npy",
            np.load(source / f"{name}.npy", mmap_mode="r")[selected],
        )
    if (source / "variance.npy").is_file():
        np.save(
            output / "variance.npy",
            np.load(source / "variance.npy", mmap_mode="r")[selected],
        )
    chosen = [rows[i] for i in selected]
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in chosen)
    )
    return chosen


def learning_curve_subset(raw, output, field_count):
    """Select first field IDs; keep validation fixed and omit all test rows.

    Run prepare_training on this raw subset so its normalization is fit on
    only the subset training fields. The final test is never consulted.
    """
    raw = Path(raw)
    rows = [
        json.loads(line)
        for line in (raw / "metadata.jsonl").read_text().splitlines()
    ]
    fields = sorted({r["field"] for r in rows if r["split"] == "train"})
    if not 1 <= field_count <= len(fields):
        raise ValueError("field count must fit the development training pool")
    selected_fields = set(fields[:field_count])
    selected = [
        i
        for i, r in enumerate(rows)
        if r["split"] == "val"
        or (r["split"] == "train" and r["field"] in selected_fields)
    ]
    copy_rows(raw, output, selected)
    return selected_fields


def permutation_control(prepared, output, *, seed=991):
    """Permute training labels once; retain validation truth.

    A permutation is not a new scientific label. Every changed row explicitly
    records its original label and the negative-control provenance. Validation
    labels and images stay unchanged. Never select a seed from results.
    """
    from cuphoton.xscan.training import require_training_label_provenance

    prepared, output = Path(prepared), Path(output)
    split = np.load(prepared / "split.npy")
    selected = np.flatnonzero(split != 2)
    rows = copy_rows(prepared, output, selected)
    labels = np.load(output / "labels.npy")
    train = np.flatnonzero(np.load(output / "split.npy") == 0)
    labels[train] = np.random.default_rng(seed).permutation(labels[train])
    for i in train:
        rows[i]["original_label"] = rows[i]["label"]
        rows[i]["label"] = int(labels[i])
        rows[i]["label_source"] = "permuted_training_negative_control"
    np.save(output / "labels.npy", labels)
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    require_training_label_provenance(output)
    return dict(
        seed=seed, permuted_rows=len(train), validation_truth_unchanged=True
    )


def rejected_split_control(prepared, output):
    """Prove that native training refuses candidate-level field leakage."""
    from cuphoton.xscan.training import require_training_label_provenance

    prepared, output = Path(prepared), Path(output)
    split = np.load(prepared / "split.npy")
    rows = copy_rows(prepared, output, np.flatnonzero(split != 2))
    split = np.load(output / "split.npy")
    train = np.flatnonzero(split == 0)
    first = int(train[0])
    if sum(r["split_group"] == rows[first]["split_group"] for r in rows) < 2:
        raise ValueError("need multiple detections from a training field")
    split[first] = 1
    rows[first]["split"] = "val"
    np.save(output / "split.npy", split)
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    try:
        require_training_label_provenance(output)
    except ValueError as error:
        if "group leakage" not in str(error):
            raise
        return dict(rejected=True, reason=str(error))
    raise AssertionError(
        "native provenance guard accepted invalid field split"
    )
