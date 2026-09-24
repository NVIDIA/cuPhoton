# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from cuphoton.core import execution
from cuphoton.core.bulk import WorkItem, atomic_write_json, read_json_mapping


class FakeWorker:
    def __init__(self, options):
        self.options = options
        self.closed = False
        self.gpu_identity = {
            "backend": options.get("backend", "cupy"),
            "device_index": 0,
            "uuid": options.get("uuid", "GPU-0"),
            "identity_error": None,
        }

    def run_item(self, item, output_dir):
        if item.payload.get("fail"):
            raise ValueError("scientific failure")
        output_dir.mkdir()
        atomic_write_json(output_dir / "summary.json", item.payload)
        return {
            "run_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "backend": self.options.get("backend", "cupy"),
            "device": "cuda:0",
            "runtime": {},
            "timings_sec": {"work": 0.0},
            "wall_sec": {"item": 0.0},
        }

    def close(self):
        self.closed = True


def worker_factory(options):
    return FakeWorker(options)


def workload(**overrides):
    values = {
        "items": (
            WorkItem("one", {"value": 1}, 1),
            WorkItem("two", {"value": 2}, 2),
        ),
        "options_payload": {},
        "manifest_payload": {"items": ["one", "two"]},
        "input_identity_payload": {},
        "manifest_sha256": "a" * 64,
        "backend": "cupy",
        "worker_factory": worker_factory,
    }
    values.update(overrides)
    return execution.WorkloadSpec(**values)


def provenance(worker_id, gpu):
    return {
        "worker_id": worker_id,
        "hostname": "host",
        "pid": os.getpid(),
        "cuda_visible_devices": str(worker_id),
        "gpu": gpu,
    }


def stage(tmp_path, spec=None):
    spec = spec or workload()
    execution.prepare_run(tmp_path, "run", spec, "test")
    shards = ((spec.items[1],), (spec.items[0],))
    results = []
    for worker_id, shard in enumerate(shards):
        worker = FakeWorker(
            {"uuid": f"GPU-{worker_id}", "backend": spec.backend}
        )
        results.append(
            execution.execute_worker_round(
                worker,
                items=shard,
                run_id="run",
                run_dir=tmp_path,
                manifest_sha256=spec.manifest_sha256,
                worker_id=worker_id,
                backend=spec.backend,
                provenance=provenance(worker_id, worker.gpu_identity),
            )
        )
    return spec, shards, results


def test_round_audits_then_finalizes_in_manifest_order(tmp_path):
    calls = []

    def finalize(run_dir, records):
        calls.append([record["item_id"] for record in records])
        return {"ordered": calls[-1]}

    spec, shards, results = stage(
        tmp_path / "run", workload(finalize_round=finalize)
    )
    report = execution.finalize_round(
        tmp_path / "run",
        "run",
        spec,
        shards,
        results,
        artifact_timeout_sec=0.01,
    )
    assert report["status"] == "success"
    assert report["terminal_record_audit"]["ok"]
    assert calls == [["one", "two"]]
    assert report["result"] == {"ordered": ["one", "two"]}
    assert read_json_mapping(tmp_path / "run" / "summary.json") == report


@pytest.mark.parametrize(
    "tamper",
    ["record", "receipt", "missing", "extra", "gpu", "round", "callback"],
)
def test_round_audit_rejects_invalid_evidence_before_component_merge(
    tmp_path, tamper
):
    finalized = []
    spec, shards, results = stage(
        tmp_path / "run",
        workload(
            finalize_round=lambda directory, records: (
                finalized.append(True) or {}
            )
        ),
    )
    run_dir = tmp_path / "run"
    if tamper == "record":
        path = run_dir / "records" / "one.json"
        record = read_json_mapping(path)
        record["worker_id"] = 0
        atomic_write_json(path, record)
    elif tamper == "receipt":
        results[0]["success_count"] = 99
    elif tamper == "missing":
        (run_dir / "records" / "one.json").unlink()
    elif tamper == "extra":
        atomic_write_json(run_dir / "records" / "extra.json", {})
    elif tamper == "gpu":
        results[1]["provenance"]["gpu"]["uuid"] = "GPU-0"
        atomic_write_json(
            run_dir / "workers" / "worker-0001.json", results[1]
        )
    elif tamper == "round":
        results[0]["run_id"] = "other-round"
        atomic_write_json(
            run_dir / "workers" / "worker-0000.json", results[0]
        )
    elif tamper == "callback":
        spec = replace(
            spec,
            success_record_validator=lambda record, directory: [
                "bad science"
            ],
        )
    report = execution.finalize_round(
        run_dir, "run", spec, shards, results, artifact_timeout_sec=0.01
    )
    assert report["status"] == "failed"
    assert report["errors"]
    assert report["result"] is None
    assert not finalized


def test_failed_item_and_publication_are_retained(tmp_path, monkeypatch):
    spec = workload(
        items=(WorkItem("one", {"fail": True}), WorkItem("two", {}))
    )
    original = execution.atomic_write_json

    def write(path, payload, **kwargs):
        if path.name == "two.json":
            raise OSError("record disk failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(execution, "atomic_write_json", write)
    spec, shards, results = stage(tmp_path / "run", spec)
    report = execution.finalize_round(
        tmp_path / "run",
        "run",
        spec,
        shards,
        results,
        artifact_timeout_sec=0.01,
    )
    assert report["status"] == "failed"
    assert "scientific failure" in json.dumps(report)
    assert "record disk failure" in json.dumps(report)


def test_torch_worker_does_not_require_cupy(tmp_path):
    spec, shards, results = stage(tmp_path / "run", workload(backend="torch"))
    report = execution.finalize_round(
        tmp_path / "run",
        "run",
        spec,
        shards,
        results,
        artifact_timeout_sec=0.01,
    )
    assert report["status"] == "success"


def test_factory_reference_is_importable_and_payload_is_json():
    spec = workload()
    reference = execution.factory_reference(worker_factory)
    assert execution.resolve_worker_factory(reference) is worker_factory
    assert (
        json.loads(json.dumps(spec.identity_payload()))
        == spec.identity_payload()
    )
    with pytest.raises(ValueError, match="package-importable"):
        workload(worker_factory=lambda options: FakeWorker(options))


def test_run_directory_is_never_reused(tmp_path):
    run_dir = tmp_path / "run"
    execution.prepare_run(run_dir, "run", workload(), "test")
    with pytest.raises(FileExistsError):
        execution.prepare_run(run_dir, "run", workload(), "test")
