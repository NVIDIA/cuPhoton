# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cuphoton.core import execution  # noqa: E402
from cuphoton.core.artifacts import file_sha256  # noqa: E402
from cuphoton.core.bulk import WorkItem  # noqa: E402
from cuphoton.xscan import executor, training, workflows  # noqa: E402
from cuphoton.xscan.model import build_model  # noqa: E402


def _inputs(tmp_path):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "model"
    dataset_dir.mkdir()
    run_dir.mkdir()
    rng = np.random.default_rng(19)
    for name in ("search", "template"):
        np.save(
            dataset_dir / f"{name}.npy",
            rng.normal(size=(7, 17, 17)).astype(np.float32),
        )
    np.save(dataset_dir / "labels.npy", np.array([0, 1, 0, 1, 1, 0, 1]))
    np.save(dataset_dir / "split.npy", np.array([2, 0, 2, 2, 2, 0, 2]))
    (dataset_dir / "metadata.jsonl").write_text(
        "".join(
            json.dumps({"candidate_id": f"candidate-{6 - index}"}) + "\n"
            for index in range(7)
        )
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
    torch.manual_seed(2)
    model = build_model(**model_config)
    torch.save(
        {
            "model_config": model_config,
            "model_state": model.state_dict(),
            "train_config": {
                "performance": {"num_workers": 0, "amp_dtype": "off"}
            },
        },
        run_dir / "checkpoint.pt",
    )
    return run_dir, dataset_dir


def _cpu_worker(monkeypatch):
    monkeypatch.setattr(executor, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        executor,
        "_gpu_identity",
        lambda: {"backend": "torch", "device_index": 0, "uuid": "test"},
    )


def test_persistent_workers_preserve_original_batches_predictions_and_order(
    tmp_path, monkeypatch
):
    run_dir, dataset_dir = _inputs(tmp_path)
    _cpu_worker(monkeypatch)
    monkeypatch.setattr(
        workflows, "resolve_device", lambda _: torch.device("cpu")
    )
    ordinary = workflows.infer_workflow(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        num_workers=0,
    )
    expected = {
        name: np.load(ordinary.run_dir / f"{name}.npy")
        for name in ("logits", "labels", "probabilities")
    }
    checkpoint_bytes = (run_dir / "checkpoint.pt").read_bytes()
    ordinary_summary = (ordinary.run_dir / "summary.json").read_bytes()
    spec = executor.prepare_inference_workload(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        task_batches=1,
        num_workers=0,
    )
    items, options = spec.items, spec.options_payload
    assert [
        (item.payload["start"], item.payload["stop"]) for item in items
    ] == [(0, 2), (2, 4), (4, 5)]
    loads = []
    original_load = training.load_model_from_checkpoint

    def load(*args, **kwargs):
        loads.append(True)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(training, "load_model_from_checkpoint", load)
    workers = [executor.create_xscan_worker(options) for _ in range(2)]
    for round_index in range(2):
        round_dir = tmp_path / f"round-{round_index}"
        round_id = f"round-{round_index}"
        execution.prepare_run(round_dir, round_id, spec, "test")
        shards = ((items[2], items[0]), (items[1],))
        results = [
            execution.execute_worker_round(
                worker,
                items=shards[index],
                run_id=round_id,
                run_dir=round_dir,
                manifest_sha256=spec.manifest_sha256,
                worker_id=index,
                backend="torch",
                provenance={
                    "worker_id": index,
                    "pid": os.getpid(),
                    "hostname": "test-host",
                    "cuda_visible_devices": str(index),
                    "gpu": {
                        "backend": "torch",
                        "device_index": 0,
                        "uuid": f"GPU-{index}",
                        "identity_error": None,
                    },
                },
            )
            for index, worker in enumerate(workers)
        ]
        audited = execution.finalize_round(
            round_dir,
            round_id,
            spec,
            shards,
            results,
            artifact_timeout_sec=0.1,
        )
        assert audited["status"] == "success", audited["errors"]
        receipt = audited["result"]
        assert receipt["sample_count"] == 5
        for name, values in expected.items():
            actual = np.load(round_dir / "scientific" / f"{name}.npy")
            np.testing.assert_array_equal(actual, values)
        np.testing.assert_array_equal(
            np.load(round_dir / "scientific" / "sample_index.npy"),
            [0, 2, 3, 4, 6],
        )
        summary = json.loads(
            (round_dir / "scientific" / "summary.json").read_text()
        )
        for key in (
            "performance",
            "xfit_features",
            "batch_size",
            "dataset_dir",
            "split",
        ):
            assert summary[key] == ordinary.summary[key]
        assert summary["saved"]["logits"] == "logits.npy"
    assert len(loads) == 2
    assert (run_dir / "checkpoint.pt").read_bytes() == checkpoint_bytes
    assert (
        ordinary.run_dir / "summary.json"
    ).read_bytes() == ordinary_summary
    for worker in workers:
        worker.close()


@pytest.mark.parametrize(
    "defect",
    ["missing", "duplicate", "failed", "wrong_probability", "wrong_identity"],
)
def test_merge_rejects_incomplete_or_invalid_scientific_results(
    tmp_path, monkeypatch, defect
):
    run_dir, dataset_dir = _inputs(tmp_path)
    _cpu_worker(monkeypatch)
    items, options = executor.plan_inference_chunks(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        task_batches=1,
    )
    worker = executor.create_xscan_worker(options)
    round_dir = tmp_path / "round"
    records = []
    for item in items:
        worker.run_item(item, round_dir / "items" / item.item_id)
        records.append({"item_id": item.item_id, "status": "success"})
    if defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records[1] = dict(records[0])
    elif defect == "failed":
        records[0]["status"] = "failed"
    else:
        item_dir = round_dir / "items" / items[0].item_id
        summary_path = item_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if defect == "wrong_probability":
            np.save(item_dir / "probabilities.npy", np.array([0.123, 0.456]))
            summary["artifact_sha256"]["probabilities"] = file_sha256(
                item_dir / "probabilities.npy"
            )
        else:
            summary["metadata_rows"][0]["candidate_id"] = "wrong"
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        executor.finalize_inference_round(
            round_dir, records, items=items, options=options
        )
    assert not (round_dir / "scientific").exists()


def test_worker_rejects_split_batches_and_changed_input(
    tmp_path, monkeypatch
):
    run_dir, dataset_dir = _inputs(tmp_path)
    _cpu_worker(monkeypatch)
    items, options = executor.plan_inference_chunks(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        task_batches=1,
    )
    worker = executor.create_xscan_worker(options)
    malformed = WorkItem(items[0].item_id, {**items[0].payload, "stop": 1})
    with pytest.raises(ValueError, match="minibatch boundary"):
        worker.run_item(malformed, tmp_path / "unused")
    np.save(dataset_dir / "labels.npy", np.zeros(7, dtype=np.int64))
    with pytest.raises(ValueError, match="input changed"):
        worker.run_item(items[0], tmp_path / "unused")


def test_import_and_planning_do_not_import_torch(tmp_path):
    run_dir, dataset_dir = _inputs(tmp_path)
    command = (
        "import sys; from pathlib import Path; "
        "from cuphoton.xscan.executor import plan_inference_chunks; "
        "assert 'torch' not in sys.modules; "
        f"plan_inference_chunks(run_dir=Path({str(run_dir)!r}), "
        f"dataset_dir=Path({str(dataset_dir)!r}), split='test'); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", command], check=True, env=os.environ.copy()
    )


def test_feature_matrix_is_reused_with_global_coverage(tmp_path, monkeypatch):
    from cuphoton.xscan.xfit_features import XFitFeatureMatrix

    run_dir, dataset_dir = _inputs(tmp_path)
    _cpu_worker(monkeypatch)
    monkeypatch.setattr(
        workflows, "resolve_device", lambda _: torch.device("cpu")
    )
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    names = ("fit_present", "separation_over_stamp")
    checkpoint["model_config"]["xfit_feature_names"] = list(names)
    model = build_model(**checkpoint["model_config"])
    torch.nn.init.ones_(model.xfit_head[-1].weight)
    checkpoint["model_state"] = model.state_dict()
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    values = np.column_stack(
        (np.array([1, 0, 1, 0, 1, 0, 1]), np.arange(7) / 10)
    ).astype(np.float32)
    for name, array in {
        "candidate-id.npy": np.arange(7),
        "features.npy": values,
        "input-image-sha256.npy": np.zeros(7, dtype="S64"),
    }.items():
        np.save(feature_dir / name, array)
    (feature_dir / "schema.json").write_text("{}")
    matrix = XFitFeatureMatrix(
        dataset_dir,
        np.arange(7),
        values,
        names,
        {},
        {"identity": "retained"},
        {},
    )
    loads = []

    def features(**kwargs):
        assert kwargs["use_xfit_features"] is True
        assert kwargs["xfit_feature_dir"] == feature_dir
        loads.append(True)
        return matrix

    # Bundle validation already has dedicated tests; this exercises the
    # adapter's reuse and original-index slicing with real fusion inference.
    monkeypatch.setattr(
        workflows, "_checkpoint_xfit_feature_matrix", features
    )
    ordinary = workflows.infer_workflow(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        use_xfit_features=True,
        xfit_feature_dir=feature_dir,
    )
    items, options = executor.plan_inference_chunks(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        task_batches=1,
        use_xfit_features=True,
        xfit_feature_dir=feature_dir,
    )
    worker = executor.create_xscan_worker(options)
    assert worker.dataset.xfit_features is matrix.values
    round_dir = tmp_path / "round"
    for item in items:
        worker.run_item(item, round_dir / "items" / item.item_id)
    executor.finalize_inference_round(
        round_dir,
        [
            {"item_id": item.item_id, "status": "success"}
            for item in reversed(items)
        ],
        items=items,
        options=options,
    )
    assert len(loads) == 2  # One ordinary call, one retained worker.
    for name in ("logits", "probabilities", "labels"):
        np.testing.assert_array_equal(
            np.load(round_dir / "scientific" / f"{name}.npy"),
            np.load(ordinary.run_dir / f"{name}.npy"),
        )
    summary = json.loads(
        (round_dir / "scientific" / "summary.json").read_text()
    )
    assert summary["xfit_features"] == ordinary.summary["xfit_features"]
    assert summary["xfit_features"]["fit_coverage"]["sample_count"] == 5
    assert summary["xfit_features"]["fit_coverage"]["fit_coverage"] == 0.8
