# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cuphoton.core import execution  # noqa: E402
from cuphoton.core.artifacts import file_sha256  # noqa: E402
from cuphoton.core.bulk import WorkItem  # noqa: E402
from cuphoton.xscan import executor, training, workflows  # noqa: E402
from cuphoton.xscan.model import build_model  # noqa: E402


def _mark_loader_ready(worker_id, *, directory):
    time.sleep(0.1)
    (directory / f"ready-{os.getpid()}").write_text(str(worker_id))


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


@pytest.mark.parametrize("runtime", ["dragon", "mpi"])
def test_cli_dispatches_collective_preflight_and_preserves_batches(
    tmp_path, monkeypatch, capsys, runtime
):
    from types import SimpleNamespace

    from cuphoton.core import executors
    from cuphoton.core.cli import run_component

    run_dir, dataset_dir = _inputs(tmp_path)
    output_dir = tmp_path / "distributed"
    calls = []

    def run(**kwargs):
        spec = kwargs["prepare_workload"](0)
        calls.append((kwargs, spec))
        return SimpleNamespace(
            status="success", to_dict=lambda: {"status": "success"}
        )

    monkeypatch.setattr(executors, "run_workload", run)
    assert (
        run_component(
            "xscan",
            [
                "infer-real-bogus",
                "--run-dir",
                str(run_dir),
                "--dataset-dir",
                str(dataset_dir),
                "--executor",
                runtime,
                "--output-dir",
                str(output_dir),
                "--batch-size",
                "2",
                "--task-batches",
                "1",
                "--num-workers",
                "0",
                "--warmup-rounds",
                "1",
                "--measure-rounds",
                "2",
            ],
        )
        == 0
    )
    kwargs, spec = calls[0]
    assert kwargs["run_id"] == "distributed"
    assert kwargs["output_root"] == tmp_path
    assert kwargs["benchmark"].warmup_rounds == 1
    assert len(spec.items) == 3
    assert spec.options_payload["batch_size"] == 2
    assert spec.options_payload["num_workers"] == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"


def test_cli_requires_separate_distributed_output(tmp_path, capsys):
    from cuphoton.core.cli import run_component

    assert (
        run_component(
            "xscan",
            [
                "infer-real-bogus",
                "--run-dir",
                str(tmp_path),
                "--dataset-dir",
                str(tmp_path),
                "--executor",
                "mpi",
            ],
        )
        != 0
    )
    assert "--output-dir is required" in capsys.readouterr().err


@pytest.mark.parametrize("num_workers", [0, 1])
def test_persistent_workers_preserve_original_batches_predictions_and_order(
    tmp_path, monkeypatch, num_workers
):
    run_dir, dataset_dir = _inputs(tmp_path)
    if num_workers:
        # Spawned workers must be able to import the readiness test callback.
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
        checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True)
        checkpoint["train_config"]["performance"]["worker_start_method"] = (
            "spawn"
        )
        torch.save(checkpoint, run_dir / "checkpoint.pt")
    _cpu_worker(monkeypatch)
    monkeypatch.setattr(
        workflows, "resolve_device", lambda _: torch.device("cpu")
    )
    ordinary = workflows.infer_workflow(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=2,
        num_workers=num_workers,
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
        num_workers=num_workers,
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
    from cuphoton.xscan import dataset as dataset_module

    metadata_reads = []
    loader_creations = []
    original_metadata = dataset_module.load_metadata_rows
    original_loader = training.make_dataloader

    def metadata(path):
        metadata_reads.append(path)
        return original_metadata(path)

    def loader(*args, **kwargs):
        loader_creations.append(True)
        result = original_loader(*args, **kwargs)
        result.worker_init_fn = partial(
            _mark_loader_ready, directory=tmp_path
        )
        return result

    monkeypatch.setattr(dataset_module, "load_metadata_rows", metadata)
    monkeypatch.setattr(training, "make_dataloader", loader)
    workers = [executor.create_xscan_worker(options) for _ in range(2)]
    children = [
        process
        for worker in workers
        for process in getattr(worker.loader._iterator, "_workers", [])
    ]
    assert len(children) == 2 * num_workers
    assert len(list(tmp_path.glob("ready-*"))) == 2 * num_workers
    assert all(process.is_alive() for process in children)
    from multiprocessing.process import BaseProcess

    def unexpected_process_start(_):
        pytest.fail("loader processes must start before READY")

    monkeypatch.setattr(BaseProcess, "start", unexpected_process_start)

    def unexpected_metadata_read(_):
        pytest.fail("metadata must be retained across tasks")

    monkeypatch.setattr(
        training, "load_metadata_rows", unexpected_metadata_read
    )
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
            "xfit_features",
            "batch_size",
            "dataset_dir",
            "split",
        ):
            assert summary[key] == ordinary.summary[key]
        assert summary["performance"] == {
            **ordinary.summary["performance"],
            "persistent_workers": num_workers > 0,
        }
        assert summary["saved"]["logits"] == "logits.npy"
    assert len(loads) == 2
    assert len(metadata_reads) == 2
    assert len(loader_creations) == 2
    assert all(process.is_alive() for process in children)
    assert children == [
        process
        for worker in workers
        for process in getattr(worker.loader._iterator, "_workers", [])
    ]
    assert (run_dir / "checkpoint.pt").read_bytes() == checkpoint_bytes
    assert (
        ordinary.run_dir / "summary.json"
    ).read_bytes() == ordinary_summary
    for worker in workers:
        worker.close()
    assert all(not process.is_alive() for process in children)


def test_loader_defaults_and_prediction_timer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    run_dir, dataset_dir = _inputs(tmp_path)
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    checkpoint["train_config"]["performance"]["num_workers"] = 2
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    _cpu_worker(monkeypatch)
    items, options = executor.plan_inference_chunks(
        run_dir=run_dir, dataset_dir=dataset_dir, split="test", batch_size=2
    )
    worker = executor.create_xscan_worker(options)
    assert worker.performance.num_workers == 0
    clock = [0.0]
    original_check = executor._check_inputs
    original_predict = training.predict_dataset

    def check(*args, **kwargs):
        clock[0] += 100
        return original_check(*args, **kwargs)

    def predict(**kwargs):
        clock[0] += 7
        return original_predict(**kwargs)

    monkeypatch.setattr(
        executor, "time", SimpleNamespace(perf_counter=lambda: clock[0])
    )
    monkeypatch.setattr(executor, "_check_inputs", check)
    monkeypatch.setattr(training, "predict_dataset", predict)
    result = worker.run_item(items[0], tmp_path / "item")
    assert result["timings_sec"]["predict_sec"] == 7
    assert result["wall_sec"]["item_runner"] == 207
    worker.close()


@pytest.mark.parametrize("ulps", [1, 4, 5])
def test_probability_audit_tolerates_only_bounded_host_rounding(
    tmp_path, monkeypatch, ulps
):
    run_dir, dataset_dir = _inputs(tmp_path)
    _cpu_worker(monkeypatch)
    items, options = executor.plan_inference_chunks(
        run_dir=run_dir, dataset_dir=dataset_dir, split="test"
    )
    worker = executor.create_xscan_worker(options)
    item = items[0]
    round_dir = tmp_path / "round"
    item_dir = round_dir / "items" / item.item_id
    worker.run_item(item, item_dir)
    path = item_dir / "probabilities.npy"
    values = np.load(path)
    for _ in range(ulps):
        values = np.nextafter(values, np.inf)
    np.save(path, values)
    summary_path = item_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["artifact_sha256"]["probabilities"] = file_sha256(path)
    summary_path.write_text(json.dumps(summary))
    if ulps > 4:
        with pytest.raises(ValueError, match="more than 4 ULP"):
            executor._read_result(item, round_dir, options)
    else:
        _, arrays = executor._read_result(item, round_dir, options)
        np.testing.assert_array_equal(arrays["probabilities"], values)
    worker.close()


@pytest.mark.parametrize(
    "defect,message",
    [
        ("missing", "one successful result"),
        ("duplicate", "one successful result"),
        ("failed", "one successful result"),
        ("wrong_probability", "more than 4 ULP"),
        ("wrong_identity", "identities or labels differ"),
        ("checksum", "checksum mismatch"),
        ("wrong_labels", "identities or labels differ"),
        ("short_array", "row count mismatch"),
        ("divergent_summary", "differ across workers"),
        ("nonfinite", "finite float64"),
        ("wrong_dtype", "finite float64"),
    ],
)
def test_merge_rejects_incomplete_or_invalid_scientific_results(
    tmp_path, monkeypatch, defect, message
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
        if defect == "wrong_identity":
            summary["metadata_rows"][0]["candidate_id"] = "wrong"
        elif defect == "divergent_summary":
            summary["inference_summary"]["performance"]["num_workers"] = 10
        else:
            name = "probabilities"
            values = np.array([0.123, 0.456])
            if defect == "wrong_labels":
                name = "labels"
                values = 1 - np.load(item_dir / "labels.npy")
            elif defect == "short_array":
                name = "logits"
                values = np.load(item_dir / "logits.npy")[:-1]
            elif defect == "nonfinite":
                values[0] = np.nan
            elif defect == "wrong_dtype":
                values = np.load(item_dir / "probabilities.npy").astype(
                    np.float32
                )
            path = item_dir / f"{name}.npy"
            np.save(path, values)
            if defect != "checksum":
                summary["artifact_sha256"][name] = file_sha256(path)
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=message):
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
