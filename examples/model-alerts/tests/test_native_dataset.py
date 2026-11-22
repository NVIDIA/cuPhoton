# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator import Design, write_dataset
from study import prepare_training


def test_native_provenance_and_training_only_normalization(tmp_path):
    from cuphoton.xscan.training import require_training_label_provenance

    raw, first, second = (tmp_path / n for n in ("raw", "first", "second"))
    write_dataset(raw, {10: "train", 11: "val"}, design=Design(epochs=1))
    receipt = prepare_training(raw, first)
    assert require_training_label_provenance(first)["ok"]
    for name in ("search", "template", "difference"):
        values = np.load(raw / f"{name}.npy")
        values[np.load(raw / "split.npy") == 1] += 100000
        np.save(raw / f"{name}.npy", values)
    perturbed = prepare_training(raw, second)
    for key in ("pair_center", "pair_scale", "difference_scale"):
        assert perturbed[key] == receipt[key]
    # Deliberately invalid split: retain a field's identity while moving only
    # one candidate across splits. The native training guard must reject it.
    rows = [
        json.loads(line)
        for line in (first / "metadata.jsonl").read_text().splitlines()
    ]
    split = np.load(first / "split.npy")
    rows[0]["split"] = "val"
    split[0] = 1
    np.save(first / "split.npy", split)
    (first / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    with pytest.raises(ValueError, match="group leakage"):
        require_training_label_provenance(first)
