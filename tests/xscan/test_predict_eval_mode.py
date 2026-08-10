# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""predict_dataset must run inference in eval mode and restore per-submodule.

Regression guards: validation previously ran with the model in ``train()``
(dropout active, BatchNorm updating running stats from the validation set),
and mode restoration used a recursive ``model.train(flag)`` that clobbered
submodules a caller intentionally held in eval.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cuphoton.xscan.dataset import StampDataset  # noqa: E402
from cuphoton.xscan.training import predict_dataset  # noqa: E402


class _TinyModel(torch.nn.Module):
    """Stand-in with Dropout + BatchNorm exposing train/eval behavior."""

    def __init__(self, raise_in_forward: bool = False) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.drop = torch.nn.Dropout(p=0.5)
        self.head = torch.nn.Linear(4, 1)
        self.raise_in_forward = raise_in_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.raise_in_forward:
            raise RuntimeError("boom")
        x = self.bn(self.conv(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.drop(x)
        return self.head(x).squeeze(1)


def _write_pair_dataset(directory) -> None:
    n, h, w = 4, 8, 8
    rng = np.random.default_rng(0)
    for name in ("search", "template"):
        np.save(
            directory / f"{name}.npy",
            rng.standard_normal((n, h, w)).astype(np.float32),
        )
    labels = [0, 1, 0, 1]
    np.save(directory / "labels.npy", np.array(labels, dtype=np.int64))
    np.save(directory / "split.npy", np.full(n, 2, dtype=np.int64))  # 2==test
    with open(directory / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for i, label in enumerate(labels):
            handle.write(
                json.dumps({"candidate_id": f"c{i}", "label": int(label)})
                + "\n"
            )


def _dataset(tmp_path):
    _write_pair_dataset(tmp_path)
    return StampDataset(tmp_path, input_mode="pair", split="test")


def test_predict_dataset_runs_in_eval_mode(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    model = _TinyModel()
    dropout_flags: list[bool] = []
    model.drop.register_forward_hook(
        lambda module, inputs, output: dropout_flags.append(module.training)
    )
    model.train()
    out = predict_dataset(
        model=model, dataset=dataset, batch_size=2, device=torch.device("cpu")
    )
    assert dropout_flags and not any(dropout_flags)
    assert model.training  # restored
    # Eval mode is deterministic (no dropout): repeat runs agree.
    again = predict_dataset(
        model=model, dataset=dataset, batch_size=2, device=torch.device("cpu")
    )
    np.testing.assert_allclose(out["logits"], again["logits"])


def test_predict_dataset_does_not_update_batchnorm(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    model = _TinyModel()
    model.train()
    before_mean = model.bn.running_mean.clone()
    before_var = model.bn.running_var.clone()
    predict_dataset(
        model=model, dataset=dataset, batch_size=2, device=torch.device("cpu")
    )
    assert torch.equal(model.bn.running_mean, before_mean)
    assert torch.equal(model.bn.running_var, before_var)


def test_predict_dataset_restores_mode_on_exception(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    model = _TinyModel(raise_in_forward=True)
    model.train()
    with pytest.raises(RuntimeError):
        predict_dataset(
            model=model,
            dataset=dataset,
            batch_size=2,
            device=torch.device("cpu"),
        )
    assert model.training  # restored even though the forward raised


def test_predict_dataset_preserves_per_submodule_mode(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    model = _TinyModel()
    model.train()
    model.bn.eval()  # caller intentionally holds this submodule in eval
    predict_dataset(
        model=model, dataset=dataset, batch_size=2, device=torch.device("cpu")
    )
    assert model.training is True  # root restored to train
    assert model.bn.training is False  # NOT clobbered back to train
