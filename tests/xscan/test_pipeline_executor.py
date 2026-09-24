# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.core.artifacts import file_sha256
from cuphoton.core.bulk import WorkItem
from cuphoton.core.cli import run_component
from cuphoton.xscan import pipeline_executor as pipeline
from cuphoton.xscan.device_pipeline import (
    DevicePipelineCandidate,
    DevicePipelineConfig,
    DevicePipelineItem,
    DeviceWorkerContext,
    DeviceXPOISPipelineConfig,
    NpyArrayDescriptor,
)


def _manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.pt").write_bytes(b"checkpoint")
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    image = tmp_path / "image.npy"
    array = np.ones((96, 96), dtype=np.float64)
    np.save(image, array)
    descriptor = NpyArrayDescriptor(
        path=str(image),
        sha256=file_sha256(image),
        shape=array.shape,
        dtype=array.dtype.str,
    )
    config = DevicePipelineConfig(
        device="cuda:0",
        checkpoint_dir=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint / "checkpoint.pt"),
        feature_schema_path=str(schema),
        feature_schema_sha256=file_sha256(schema),
        stamp_shape=(63, 63),
        decision_threshold=0.5,
        xpois=DeviceXPOISPipelineConfig(
            kernel_shape=(3, 3),
            basis_sigmas=(1.0,),
            basis_degrees=(0,),
        ),
    )
    item = DevicePipelineItem(
        item_id="pair",
        reference=descriptor,
        target=descriptor,
        candidates=(
            DevicePipelineCandidate(
                candidate_id="candidate",
                center_x=47,
                center_y=47,
                source_index=0,
            ),
        ),
    )
    payload = {
        "schema": pipeline.PIPELINE_MANIFEST_SCHEMA,
        "configuration": config.to_payload(),
        "items": [item.to_payload()],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path, config, item


def test_manifest_paths_are_relative_to_manifest_not_cwd(
    tmp_path, monkeypatch
):
    path, config, item = _manifest(tmp_path)
    data = json.loads(path.read_text())
    data["configuration"]["checkpoint_dir"] = "checkpoint"
    data["configuration"]["feature_schema_path"] = "schema.json"
    for role in ("reference", "target"):
        data["items"][0][role]["path"] = "image.npy"
    path.write_text(json.dumps(data))
    monkeypatch.chdir(tmp_path.parent)
    assert pipeline.load_pipeline_manifest(path) == (config, (item,))


@pytest.mark.parametrize(
    "defect", ["duplicate", "unknown", "schema", "items"]
)
def test_manifest_rejects_ambiguous_or_invalid_descriptors(tmp_path, defect):
    path, _, _ = _manifest(tmp_path)
    data = json.loads(path.read_text())
    if defect == "unknown":
        data["unexpected"] = 1
    elif defect == "schema":
        data["schema"] = "wrong/v1"
    elif defect == "items":
        data["items"] = {}
    path.write_text(json.dumps(data))
    if defect == "duplicate":
        path.write_text(
            path.read_text().replace('"items":', '"items": [], "items":')
        )
    with pytest.raises(ValueError):
        pipeline.load_pipeline_manifest(path)


def test_preflight_agrees_across_ranks_and_hashes_only_on_root(
    tmp_path, monkeypatch
):
    _, config, item = _manifest(tmp_path)
    root = pipeline.prepare_pipeline_workload((item,), config)

    def unexpected_hash(path):
        pytest.fail(f"non-root preflight hashed {path}")

    monkeypatch.setattr(
        "cuphoton.xscan.dragon_pipeline.file_sha256", unexpected_hash
    )
    peer = pipeline.prepare_pipeline_workload((item,), config, rank=1)
    assert root.identity_payload() == peer.identity_payload()
    assert peer.worker_factory is pipeline.create_pipeline_worker


def test_preflight_rejects_changed_input_on_root(tmp_path):
    _, config, item = _manifest(tmp_path)
    image = tmp_path / "image.npy"
    image.write_bytes(image.read_bytes()[:-1] + b"x")
    with pytest.raises(RuntimeError, match="changed before"):
        pipeline.prepare_pipeline_workload((item,), config)


@pytest.mark.parametrize("executor", ["dragon", "mpi"])
def test_entrypoint_prepares_inside_runtime_callback(
    tmp_path, monkeypatch, executor
):
    path, config, items = _manifest(tmp_path)
    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        spec = kwargs["prepare_workload"](1)
        assert spec.items[0].item_id == items.item_id
        assert spec.options_payload["configuration"] == config.to_payload()
        return "result"

    monkeypatch.setattr("cuphoton.core.executors.run_workload", run)
    assert (
        pipeline.run_pipeline_manifest(
            executor=executor,
            manifest_path=path,
            output_root=tmp_path / "runs",
            run_id="example",
        )
        == "result"
    )
    assert seen["executor"] == executor
    assert seen["run_id"] == "example"


def test_worker_reuses_context_and_cleans_both_streams(tmp_path, monkeypatch):
    _, config, item = _manifest(tmp_path)
    events = []
    context = SimpleNamespace(
        producer_stream=SimpleNamespace(
            synchronize=lambda: events.append("producer")
        ),
        consumer_stream=SimpleNamespace(
            synchronize=lambda: events.append("consumer")
        ),
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda n: events.append(("torch", n)))
    )
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda n: SimpleNamespace(
                use=lambda: events.append(("cupy", n))
            )
        )
    )

    def import_runtime(name):
        events.append(name)
        return {"torch": torch, "cupy": cupy}[name]

    monkeypatch.setattr(pipeline.importlib, "import_module", import_runtime)
    monkeypatch.setattr(
        pipeline, "_strict_worker_options", lambda options: config
    )
    monkeypatch.setattr(
        pipeline, "collect_gpu_identity", lambda backend: {"backend": backend}
    )

    def initialize(cls, seen):
        assert seen == config
        events.append("initialize")
        return context

    monkeypatch.setattr(
        DeviceWorkerContext, "initialize", classmethod(initialize)
    )

    def publish(**kwargs):
        assert kwargs["context"] is context
        events.append("item")
        return {"status": "success"}

    monkeypatch.setattr(pipeline, "_publish_item_result", publish)
    worker = pipeline.create_pipeline_worker({})
    work = WorkItem(item.item_id, item.to_payload())
    for index in range(3):
        worker.run_item(work, tmp_path / str(index))
    worker.close()
    assert events == [
        "torch",
        "cupy",
        ("torch", 0),
        ("cupy", 0),
        "initialize",
        "item",
        "item",
        "item",
        "producer",
        "consumer",
    ]
    with pytest.raises(RuntimeError, match="closed"):
        worker.run_item(work, tmp_path / "closed")


def test_close_attempts_second_stream_after_first_failure():
    calls = []

    def fail():
        calls.append("producer")
        raise RuntimeError("cannot synchronize")

    worker = object.__new__(pipeline._PipelineWorker)
    worker.context = SimpleNamespace(
        producer_stream=SimpleNamespace(synchronize=fail),
        consumer_stream=SimpleNamespace(
            synchronize=lambda: calls.append("consumer")
        ),
    )
    with pytest.raises(RuntimeError, match="cleanup"):
        worker.close()
    assert calls == ["producer", "consumer"]
    assert worker.closed


@pytest.mark.parametrize("executor", ["dragon", "mpi"])
def test_pipeline_command_forwards_execution_options(
    tmp_path, monkeypatch, executor, capsys
):
    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            status="success", to_dict=lambda: {"executor": executor}
        )

    monkeypatch.setattr(pipeline, "run_pipeline_manifest", run)
    assert (
        run_component(
            "xscan",
            [
                "run-pipeline",
                "--executor",
                executor,
                "--manifest",
                str(tmp_path / "m.json"),
                "--output-dir",
                str(tmp_path / "runs"),
                "--name",
                "qualified",
                "--warmup-rounds",
                "1",
                "--measure-rounds",
                "3",
            ],
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"executor": executor}
    assert seen["run_id"] == "qualified"
    assert seen["benchmark"].to_payload() == {
        "warmup_rounds": 1,
        "measure_rounds": 3,
    }


def test_pipeline_command_rejects_wrong_executor_limits_before_runtime(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pipeline,
        "run_pipeline_manifest",
        lambda **kwargs: pytest.fail("invalid args reached runtime"),
    )
    assert (
        run_component(
            "xscan",
            [
                "run-pipeline",
                "--executor",
                "mpi",
                "--max-workers",
                "2",
                "--manifest",
                str(tmp_path / "m.json"),
                "--output-dir",
                str(tmp_path),
            ],
        )
        != 0
    )
