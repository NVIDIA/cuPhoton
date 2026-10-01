# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import multiprocessing.process
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cuphoton.core.cli import run_component  # noqa: E402
from cuphoton.xscan import commands, workflows  # noqa: E402
from cuphoton.xscan.model import build_model  # noqa: E402


def test_inference_worker_override_preserves_checkpoint_and_predictions(
    tmp_path, monkeypatch
):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    dataset_dir.mkdir()
    run_dir.mkdir()
    rng = np.random.default_rng(4)
    for name in ("search", "template"):
        np.save(dataset_dir / f"{name}.npy", rng.normal(size=(5, 17, 17)))
    np.save(dataset_dir / "labels.npy", np.array([0, 1, 0, 1, 1]))
    np.save(dataset_dir / "split.npy", np.array([2, 0, 2, 2, 2]))
    rows = [{"candidate_id": f"c{i}"} for i in range(5)]
    (dataset_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    model_config = dict(
        input_mode="pair",
        image_size=17,
        depths=[1, 1],
        num_heads=[1, 1],
        embed_dims=[8, 16],
        decoder_embedding_dim=8,
        pos_dim=4,
        output_nc=2,
        drop_rate=0.0,
        attn_drop=0.0,
    )
    torch.manual_seed(0)
    model = build_model(**model_config)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_config": model_config,
            "model_state": model.state_dict(),
            "train_config": {
                "performance": {
                    "num_workers": 1,
                    "persistent_workers": True,
                    "worker_start_method": "spawn",
                    "amp_dtype": "off",
                }
            },
        },
        checkpoint_path,
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    monkeypatch.setattr(
        workflows, "resolve_device", lambda _: torch.device("cpu")
    )
    original_load = workflows.load_model_from_checkpoint
    loaded_settings = []

    def record_load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        loaded_settings.append(result[2])
        return result

    monkeypatch.setattr(workflows, "load_model_from_checkpoint", record_load)
    original_predict = workflows.predict_dataset
    predictions = []

    def record_predict(**kwargs):
        payload = original_predict(**kwargs)
        predictions.append(payload)
        return payload

    monkeypatch.setattr(workflows, "predict_dataset", record_predict)
    inherited = workflows.infer_workflow(
        run_dir=run_dir, dataset_dir=dataset_dir, split="test", batch_size=2
    )
    assert inherited.summary["performance"]["num_workers"] == 1
    assert inherited.summary["performance"]["persistent_workers"] is True

    def unexpected_worker(*args, **kwargs):
        pytest.fail("num_workers=0 must not start a process")

    monkeypatch.setattr(
        multiprocessing.process.BaseProcess, "start", unexpected_worker
    )
    override = workflows.infer_workflow(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        num_workers=0,
    )
    assert override.summary["performance"]["num_workers"] == 0
    assert override.summary["performance"]["persistent_workers"] is False
    assert override.summary["batch_size"] == 2
    assert all(config.num_workers == 1 for config in loaded_settings)
    assert all(config.persistent_workers for config in loaded_settings)
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    for key in ("logits", "probabilities", "labels"):
        np.testing.assert_array_equal(
            predictions[0][key], predictions[1][key]
        )
        saved = np.load(run_dir / override.summary["saved"][key])
        np.testing.assert_array_equal(saved, predictions[1][key])
    for prediction in predictions:
        assert [
            row["candidate_id"] for row in prediction["metadata_rows"]
        ] == ["c0", "c2", "c3", "c4"]
    saved_summary = json.loads(
        (override.run_dir / "summary.json").read_text()
    )
    assert saved_summary["performance"] == override.summary["performance"]
    assert asdict(loaded_settings[0]) == asdict(loaded_settings[1])


@pytest.mark.parametrize("workers", [None, 0, 2])
def test_cli_forwards_optional_inference_workers(
    tmp_path, monkeypatch, capsys, workers
):
    calls = []

    def infer(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(summary={})

    monkeypatch.setattr(commands, "infer_workflow", infer)
    argv = [
        "infer-real-bogus",
        "--run-dir",
        str(tmp_path),
        "--dataset-dir",
        str(tmp_path),
    ]
    if workers is not None:
        argv.extend(["--num-workers", str(workers)])
    assert run_component("xscan", argv) == 0
    assert calls[0]["num_workers"] == workers
    assert calls[0]["batch_size"] == 32
    capsys.readouterr()


def test_inference_rejects_negative_workers_before_checkpoint_loading(
    tmp_path, monkeypatch
):
    def unexpected_load(*args, **kwargs):
        pytest.fail("negative workers must fail before checkpoint loading")

    monkeypatch.setattr(
        workflows, "load_model_from_checkpoint", unexpected_load
    )
    with pytest.raises(ValueError, match="num_workers must be non-negative"):
        workflows.infer_workflow(
            run_dir=tmp_path,
            dataset_dir=tmp_path,
            split="test",
            num_workers=-1,
        )


def test_cli_rejects_negative_inference_workers(
    tmp_path, monkeypatch, capsys
):
    def unexpected_inference(**kwargs):
        pytest.fail("invalid worker count must not reach inference")

    monkeypatch.setattr(commands, "infer_workflow", unexpected_inference)
    assert (
        run_component(
            "xscan",
            [
                "infer-real-bogus",
                "--run-dir",
                str(tmp_path),
                "--dataset-dir",
                str(tmp_path),
                "--num-workers=-1",
            ],
        )
        != 0
    )
    assert "non-negative" in capsys.readouterr().err
