# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import queue
from types import SimpleNamespace

import numpy as np
import pytest

import cuphoton.xpois.dragon as dragon_module
from cuphoton.core.bulk import (
    Placement,
    WorkItem,
    atomic_write_json,
    audit_terminal_records,
)
from cuphoton.xpois.batch import (
    BatchFitOptions,
    load_image_pair_manifest,
    run_image_pair_item,
)
from cuphoton.xpois.dragon import (
    _aggregate_item_timings,
    _audit_shard_results,
    _audit_terminal_record_contract,
    _dragon_shard_worker,
    _DragonAPI,
    _execute_shard,
    _hostnames_match,
    _load_terminal_records,
    _select_gpu_placements,
    _singleton_cuda_visibility,
    discover_gpu_placements,
    run_dragon_image_pair_batch,
)


class _System:
    nodes = (7, 11)


class _Node:
    def __init__(self, node_id):
        self.hostname = f"node-{node_id}"
        self.gpus = [2, 5] if node_id == 7 else [4]


def _options(**overrides) -> BatchFitOptions:
    values = {
        "kernel_shape": (9, 9),
        "basis_sigmas": (1.5,),
        "basis_degrees": (0,),
        "backend": "cupy",
    }
    values.update(overrides)
    return BatchFitOptions(**values)


def _test_placements(worker_count: int) -> tuple[Placement, ...]:
    return tuple(
        Placement(worker_id=index, host=f"node-{index}", gpu_id=index + 3)
        for index in range(worker_count)
    )


def _valid_shard_result(
    shard: tuple[WorkItem, ...],
    placement: Placement,
    *,
    backend: str = "cupy",
) -> dict[str, object]:
    identity_backend = "cupy" if backend in {"cupy", "cutile"} else backend
    return {
        "schema": "cuphoton.xpois.dragon-shard/v1",
        "worker_id": placement.worker_id,
        "status": "success",
        "item_count": len(shard),
        "success_count": len(shard),
        "failed_count": 0,
        "weight_bytes": sum(item.weight_bytes for item in shard),
        "item_ids_sha256": dragon_module._item_ids_sha256(shard),
        "started_at_utc": "2026-08-21T00:00:00+00:00",
        "completed_at_utc": "2026-08-21T00:00:01+00:00",
        "worker_wall_sec": 1.0,
        "timings_sec": {},
        "provenance": {
            "worker_id": placement.worker_id,
            "requested_host": placement.host,
            "requested_gpu_id": placement.gpu_id,
            "hostname": placement.host,
            "pid": 1234,
            "cuda_visible_devices": str(placement.gpu_id),
            "gpu": {
                "backend": identity_backend,
                "name": "fake-gpu",
                "uuid": f"GPU-worker-{placement.worker_id}",
                "pci_bus_id": f"0000:{placement.gpu_id:02x}:00.0",
                "identity_error": None,
            },
        },
        "record_write_errors": [],
    }


def test_discovery_uses_actual_noncontiguous_node_gpu_ids() -> None:
    assert discover_gpu_placements(_System, _Node) == (
        Placement(worker_id=0, host="node-7", gpu_id=2),
        Placement(worker_id=1, host="node-7", gpu_id=5),
        Placement(worker_id=2, host="node-11", gpu_id=4),
    )


def test_worker_selection_round_robins_across_hosts() -> None:
    placements = discover_gpu_placements(_System, _Node)

    assert _select_gpu_placements(placements, 2) == (
        Placement(worker_id=0, host="node-7", gpu_id=2),
        Placement(worker_id=1, host="node-11", gpu_id=4),
    )


def test_singleton_visibility_rejects_multiple_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    with pytest.raises(RuntimeError, match="exactly one"):
        _singleton_cuda_visibility()


def test_singleton_visibility_rejects_wrong_placed_device(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

    with pytest.raises(RuntimeError, match="placement mismatch"):
        _singleton_cuda_visibility(2)


def test_hostname_validation_accepts_short_fqdn_aliases() -> None:
    assert _hostnames_match("gpu-node", "gpu-node.example.com")
    assert _hostnames_match("GPU-NODE.EXAMPLE.COM.", "gpu-node")
    assert not _hostnames_match(
        "localhost", dragon_module.socket.gethostname()
    )
    assert _hostnames_match(
        "localhost",
        dragon_module.socket.gethostname(),
        allow_loopback_alias=True,
    )
    assert not _hostnames_match("gpu-node-1", "gpu-node-2")
    assert not _hostnames_match(
        "gpu-node.dc1.example.com", "gpu-node.dc2.example.com"
    )


def test_shard_rejects_wrong_host_before_gpu_initialization(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    initialized = False

    def identity_loader(backend):
        nonlocal initialized
        initialized = True
        return {"backend": backend}

    with pytest.raises(RuntimeError, match="placement mismatch"):
        _execute_shard(
            run_id="wrong-host",
            run_dir=tmp_path,
            placement=Placement(
                worker_id=0,
                host="definitely-not-this-host",
                gpu_id=0,
            ),
            items=(),
            options=_options(),
            item_runner=lambda item, output, options: {},
            gpu_identity_loader=identity_loader,
            require_clean_cuda_imports=False,
        )
    assert initialized is False


def test_shard_rejects_wrong_gpu_before_gpu_initialization(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    initialized = False

    def identity_loader(backend):
        nonlocal initialized
        initialized = True
        return {"backend": backend}

    with pytest.raises(RuntimeError, match="GPU placement mismatch"):
        _execute_shard(
            run_id="wrong-gpu",
            run_dir=tmp_path,
            placement=Placement(
                worker_id=0,
                host=dragon_module.socket.gethostname(),
                gpu_id=2,
            ),
            items=(),
            options=_options(),
            item_runner=lambda item, output, options: {},
            gpu_identity_loader=identity_loader,
            require_clean_cuda_imports=False,
        )
    assert initialized is False


def test_shard_rejects_cuda_import_before_worker_preflight(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setitem(dragon_module.sys.modules, "cupy", object())
    initialized = False

    def identity_loader(backend):
        nonlocal initialized
        initialized = True
        return {"backend": backend}

    with pytest.raises(RuntimeError, match="before Dragon worker placement"):
        _execute_shard(
            run_id="premature-cuda",
            run_dir=tmp_path,
            placement=Placement(
                worker_id=0,
                host=dragon_module.socket.gethostname(),
                gpu_id=2,
            ),
            items=(),
            options=_options(),
            item_runner=lambda item, output, options: {},
            gpu_identity_loader=identity_loader,
        )
    assert initialized is False


def test_shard_persists_success_and_failure_then_continues(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    run_dir = tmp_path / "run"
    for name in ("items", "records", "workers"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    items = (
        WorkItem("good", {"id": "good"}, 10),
        WorkItem("bad", {"id": "bad"}, 20),
    )
    visited = []

    def item_runner(item, output_dir, options):
        visited.append(item.item_id)
        if item.item_id == "bad":
            output_dir.mkdir(parents=True)
            (output_dir / "partial.npy").write_bytes(b"partial")
            error = ValueError("injected ordinary failure")
            error.add_note("post-item input identity verification failed")
            raise error
        return {
            "run_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "timings_sec": {"solve": 1.25},
        }

    result = _execute_shard(
        run_id="test-run",
        run_dir=run_dir,
        placement=Placement(
            worker_id=0,
            host=dragon_module.socket.gethostname(),
            gpu_id=5,
        ),
        items=items,
        options=_options(),
        item_runner=item_runner,
        gpu_identity_loader=lambda backend: {
            "backend": backend,
            "uuid": "GPU-test",
            "pci_bus_id": "0000:01:00.0",
        },
        require_clean_cuda_imports=False,
    )

    assert visited == ["good", "bad"]
    assert result["status"] == "failed"
    assert result["success_count"] == 1
    assert result["failed_count"] == 1
    good = json.loads((run_dir / "records" / "good.json").read_text())
    bad = json.loads((run_dir / "records" / "bad.json").read_text())
    assert good["status"] == "success"
    assert good["summary_path"] == "items/good/summary.json"
    assert bad["status"] == "failed"
    assert bad["error"] == {
        "type": "ValueError",
        "message": "injected ordinary failure",
        "notes": "post-item input identity verification failed",
    }
    assert (run_dir / "items" / "bad" / "partial.npy").is_file()
    assert not list((run_dir / "records").glob(".*.tmp-*"))
    assert result["provenance"]["cuda_visible_devices"] == "5"
    assert result["provenance"]["gpu"]["uuid"] == "GPU-test"


def test_shard_result_audit_rejects_missing_and_duplicate_workers() -> None:
    shards = (
        (WorkItem("zero", {}, 10),),
        (WorkItem("one", {}, 20),),
        (WorkItem("two", {}, 30),),
    )
    placements = _test_placements(len(shards))
    incomplete = _valid_shard_result(shards[0], placements[0])
    for field in ("item_count", "weight_bytes", "item_ids_sha256"):
        incomplete.pop(field)
    audit = _audit_shard_results(
        shards,
        [
            incomplete,
            dict(incomplete),
            {"worker_id": 4},
        ],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["missing_worker_ids"] == [1, 2]
    assert audit["duplicate_worker_ids"] == [0]
    assert audit["unexpected_worker_ids"] == [4]
    assert audit["mismatched_shards"] == [
        {
            "worker_id": 0,
            "fields": ["item_count", "item_ids_sha256", "weight_bytes"],
        },
        {
            "worker_id": 0,
            "fields": ["item_count", "item_ids_sha256", "weight_bytes"],
        },
    ]


def test_shard_result_audit_quarantines_invalid_worker_id() -> None:
    shards = ((WorkItem("zero", {}, 10),),)
    placements = _test_placements(1)

    audit = _audit_shard_results(
        shards,
        [{"worker_id": "not-an-integer"}],
        [],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["missing_worker_ids"] == [0]
    assert audit["invalid_results"] == [
        {
            "result_index": 0,
            "field": "worker_id",
            "message": "worker_id must be a non-boolean integer",
        }
    ]


def test_shard_result_audit_rejects_status_count_mismatch() -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    result["status"] = "failed"

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["status"]}
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("item_count", True),
        ("item_count", 1.0),
        ("weight_bytes", True),
        ("weight_bytes", 10.0),
    ],
)
def test_shard_result_audit_requires_strict_integral_totals(
    field, value
) -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    result[field] = value

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["mismatched_shards"] == [{"worker_id": 0, "fields": [field]}]


def test_shard_result_audit_accepts_honest_reported_failure() -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    result.update(status="failed", success_count=0, failed_count=1)

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "failed"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is True
    assert audit["mismatched_shards"] == []
    assert audit["write_failed_shards"] == []


def test_shard_result_audit_cross_checks_terminal_status_counts() -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "failed"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["failed_count", "success_count"]}
    ]


def test_shard_result_audit_separates_record_write_errors() -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    result["record_write_errors"] = [{"item_id": "zero"}]

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["mismatched_shards"] == []
    assert audit["write_failed_shards"] == [
        {"worker_id": 0, "record_write_errors": [{"item_id": "zero"}]}
    ]


@pytest.mark.parametrize("value", [None, {"item_id": "zero"}, ["zero"]])
def test_shard_result_audit_rejects_malformed_record_write_errors(
    value,
) -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    result["record_write_errors"] = value

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["write_failed_shards"] == []
    assert audit["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["record_write_errors"]}
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("started_at_utc", "2026-08-21T00:00:00"),
        ("completed_at_utc", "not-a-timestamp"),
        ("worker_wall_sec", True),
        ("worker_wall_sec", -1.0),
        ("worker_wall_sec", float("inf")),
    ],
)
def test_shard_result_audit_rejects_invalid_worker_timing(
    field, value
) -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    result[field] = value

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["ok"] is False
    assert audit["mismatched_shards"] == [{"worker_id": 0, "fields": [field]}]


def test_shard_result_audit_rejects_decreasing_timestamps() -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    result["completed_at_utc"] = "2026-08-20T23:59:59+00:00"

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["completed_at_utc"]}
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_id", 1),
        ("requested_host", "different-node"),
        ("requested_gpu_id", 99),
        ("hostname", "different-node"),
        ("pid", 0),
        ("cuda_visible_devices", "3,7"),
        ("gpu", {}),
        ("gpu", {"backend": "", "name": "fake-gpu"}),
        ("gpu", {"backend": "numba-cuda", "name": "fake-gpu"}),
    ],
)
def test_shard_result_audit_rejects_invalid_provenance(field, value) -> None:
    shard = (WorkItem("zero", {}, 10),)
    placements = _test_placements(1)
    result = _valid_shard_result(shard, placements[0])
    provenance = result["provenance"]
    assert isinstance(provenance, dict)
    provenance[field] = value

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=placements,
        allow_loopback_alias=False,
        backend="cupy",
    )

    assert audit["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["provenance"]}
    ]


@pytest.mark.parametrize("backend", ["cupy", "numba-cuda", "cutile"])
def test_shard_result_audit_accepts_backend_identity(backend) -> None:
    shard = (WorkItem("zero", {}, 10),)
    placement = Placement(worker_id=0, host="localhost", gpu_id=3)
    result = _valid_shard_result(shard, placement, backend=backend)
    provenance = result["provenance"]
    assert isinstance(provenance, dict)
    provenance["hostname"] = "actual-node.example.com"

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=(placement,),
        allow_loopback_alias=True,
        backend=backend,
    )

    assert audit["ok"] is True


@pytest.mark.parametrize("backend", ["cupy", "numba-cuda", "cutile"])
@pytest.mark.parametrize(
    ("uuid", "pci_bus_id", "identity_error"),
    [
        (None, None, None),
        ("GPU-test", "0000:03:00.0", "identity lookup failed"),
    ],
)
def test_shard_result_audit_requires_stable_gpu_identity(
    backend, uuid, pci_bus_id, identity_error
) -> None:
    shard = (WorkItem("zero", {}, 10),)
    placement = Placement(worker_id=0, host="node", gpu_id=3)
    result = _valid_shard_result(shard, placement, backend=backend)
    provenance = result["provenance"]
    assert isinstance(provenance, dict)
    gpu = provenance["gpu"]
    assert isinstance(gpu, dict)
    gpu.update(
        uuid=uuid,
        pci_bus_id=pci_bus_id,
        identity_error=identity_error,
    )

    audit = _audit_shard_results(
        (shard,),
        [result],
        [{"item_id": "zero", "status": "success"}],
        placements=(placement,),
        allow_loopback_alias=False,
        backend=backend,
    )

    assert audit["ok"] is False
    assert audit["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["provenance"]}
    ]


@pytest.mark.parametrize("backend", ["cupy", "numba-cuda", "cutile"])
@pytest.mark.parametrize("identity_field", ["uuid", "pci_bus_id"])
def test_shard_result_audit_rejects_duplicate_physical_gpu_identity(
    backend, identity_field
) -> None:
    shards = (
        (WorkItem("zero", {}, 10),),
        (WorkItem("one", {}, 20),),
    )
    placements = (
        Placement(worker_id=0, host="node.example.com", gpu_id=3),
        Placement(worker_id=1, host="node.example.com", gpu_id=7),
    )
    results = [
        _valid_shard_result(shard, placement, backend=backend)
        for shard, placement in zip(shards, placements)
    ]
    for result in results:
        provenance = result["provenance"]
        assert isinstance(provenance, dict)
        gpu = provenance["gpu"]
        assert isinstance(gpu, dict)
        gpu[identity_field] = "shared-physical-id"

    audit = _audit_shard_results(
        shards,
        results,
        [
            {"item_id": "zero", "status": "success"},
            {"item_id": "one", "status": "success"},
        ],
        placements=placements,
        allow_loopback_alias=False,
        backend=backend,
    )

    assert audit["ok"] is False
    assert audit["duplicate_physical_gpu_worker_ids"] == [0, 1]
    assert audit["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["provenance"]},
        {"worker_id": 1, "fields": ["provenance"]},
    ]


def test_shard_continues_after_terminal_record_write_failure(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    run_dir = tmp_path / "run"
    for name in ("items", "records", "workers"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    original_atomic_write_json = dragon_module.atomic_write_json

    def flaky_atomic_write_json(path, payload):
        if path.parent.name == "records" and path.name == "first.json":
            raise OSError("injected record write failure")
        original_atomic_write_json(path, payload)

    monkeypatch.setattr(
        dragon_module, "atomic_write_json", flaky_atomic_write_json
    )
    visited = []

    def item_runner(item, output_dir, options):
        del options
        visited.append(item.item_id)
        return {
            "run_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
        }

    result = _execute_shard(
        run_id="record-write-failure",
        run_dir=run_dir,
        placement=Placement(
            worker_id=0,
            host=dragon_module.socket.gethostname(),
            gpu_id=5,
        ),
        items=(
            WorkItem("first", {}, 10),
            WorkItem("second", {}, 20),
        ),
        options=_options(),
        item_runner=item_runner,
        gpu_identity_loader=lambda backend: {"backend": backend},
        require_clean_cuda_imports=False,
    )

    assert visited == ["first", "second"]
    assert result["status"] == "failed"
    assert result["success_count"] == 1
    assert result["failed_count"] == 1
    assert result["record_write_errors"] == [
        {
            "item_id": "first",
            "type": "OSError",
            "message": "injected record write failure",
        }
    ]
    assert not (run_dir / "records" / "first.json").exists()
    assert (run_dir / "records" / "second.json").is_file()
    assert (run_dir / "workers" / "worker-0000.json").is_file()


def test_real_item_record_matches_coordinator_contract(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    reference = np.zeros((64, 64), dtype=np.float64)
    reference[20:40, 20:40] = 1.0
    target = reference.copy()
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    np.save(reference_path, reference, allow_pickle=False)
    np.save(target_path, target, allow_pickle=False)
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": "cpu-pair",
                        "reference": str(reference_path),
                        "target": str(target_path),
                    }
                ],
            }
        )
    )
    item = load_image_pair_manifest(manifest_path).work_items()[0]
    run_dir = tmp_path / "run"
    for name in ("items", "records", "workers"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    options = BatchFitOptions(
        kernel_shape=(9, 9),
        basis_sigmas=(1.5,),
        basis_degrees=(0,),
        backend="cpu",
    )

    result = _execute_shard(
        run_id="real-item-contract",
        run_dir=run_dir,
        placement=Placement(
            worker_id=0,
            host=dragon_module.socket.gethostname(),
            gpu_id=0,
        ),
        items=(item,),
        options=options,
        item_runner=run_image_pair_item,
        gpu_identity_loader=lambda backend: {"backend": backend},
        require_clean_cuda_imports=False,
    )
    records, load_errors = _load_terminal_records(run_dir)

    assert result["status"] == "success"
    assert load_errors == []
    assert (
        _audit_terminal_record_contract(
            run_id="real-item-contract",
            expected={
                item.item_id: {
                    "worker_id": 0,
                    "weight_bytes": item.weight_bytes,
                }
            },
            records=records,
        )
        == []
    )


def test_item_timing_aggregation_quarantines_malformed_mapping() -> None:
    timings, errors = _aggregate_item_timings(
        [
            {"item_id": "good", "timings_sec": {"solve_sec": 1.0}},
            {"item_id": "bad", "timings_sec": "not-a-mapping"},
        ]
    )

    assert timings["solve_sec"] == {"sum": 1.0, "max": 1.0, "mean": 1.0}
    assert errors == [
        {
            "item_id": "bad",
            "type": "InvalidTimingRecord",
            "message": "item 'bad' timings_sec must be a mapping",
        }
    ]


def test_terminal_record_contract_rejects_incomplete_success() -> None:
    errors = _audit_terminal_record_contract(
        run_id="run",
        expected={"item": {"worker_id": 0, "weight_bytes": 10}},
        records=[{"item_id": "item", "status": "success"}],
    )

    assert len(errors) == 1
    assert errors[0]["type"] == "InvalidTerminalRecord"
    assert "schema" in errors[0]["message"]
    assert "worker_id" in errors[0]["message"]
    assert "summary_path" in errors[0]["message"]
    assert "device" in errors[0]["message"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("started_at_utc", "2026-08-21T00:00:00"),
        ("completed_at_utc", "not-a-timestamp"),
    ],
)
def test_terminal_record_contract_rejects_invalid_timestamps(
    field, value
) -> None:
    record = {
        "schema": "cuphoton.xpois.dragon-item/v1",
        "run_id": "run",
        "item_id": "item",
        "worker_id": 0,
        "weight_bytes": 10,
        "started_at_utc": "2026-08-21T00:00:00+00:00",
        "completed_at_utc": "2026-08-21T00:00:01+00:00",
        "worker_seconds": 1.0,
        "status": "failed",
        "error": {"type": "RuntimeError", "message": "failed"},
    }
    record[field] = value

    errors = _audit_terminal_record_contract(
        run_id="run",
        expected={"item": {"worker_id": 0, "weight_bytes": 10}},
        records=[record],
    )

    assert len(errors) == 1
    assert errors[0]["message"].endswith(field)


def test_terminal_record_contract_rejects_decreasing_timestamps() -> None:
    errors = _audit_terminal_record_contract(
        run_id="run",
        expected={"item": {"worker_id": 0, "weight_bytes": 10}},
        records=[
            {
                "schema": "cuphoton.xpois.dragon-item/v1",
                "run_id": "run",
                "item_id": "item",
                "worker_id": 0,
                "weight_bytes": 10,
                "started_at_utc": "2026-08-21T00:00:01+00:00",
                "completed_at_utc": "2026-08-21T00:00:00+00:00",
                "worker_seconds": 1.0,
                "status": "failed",
                "error": {"type": "RuntimeError", "message": "failed"},
            }
        ],
    )

    assert len(errors) == 1
    assert errors[0]["message"].endswith("completed_at_utc")


@pytest.mark.parametrize(
    "wall_sec",
    [
        {"fit": True},
        {"fit": -1.0},
        {"fit": float("inf")},
        {"fit": "one"},
    ],
)
def test_terminal_record_contract_validates_wall_timings(wall_sec) -> None:
    errors = _audit_terminal_record_contract(
        run_id="run",
        expected={
            "item": {
                "worker_id": 0,
                "weight_bytes": 10,
                "backend": "cupy",
            }
        },
        records=[
            {
                "schema": "cuphoton.xpois.dragon-item/v1",
                "run_id": "run",
                "item_id": "item",
                "worker_id": 0,
                "weight_bytes": 10,
                "started_at_utc": "2026-08-21T00:00:00+00:00",
                "completed_at_utc": "2026-08-21T00:00:01+00:00",
                "worker_seconds": 1.0,
                "status": "success",
                "run_dir": "items/item",
                "summary_path": "items/item/summary.json",
                "requested_backend": "cupy",
                "backend": "cupy",
                "device": "fake-gpu",
                "runtime": {},
                "timings_sec": {},
                "wall_sec": wall_sec,
            }
        ],
    )

    assert len(errors) == 1
    assert errors[0]["message"].endswith("wall_sec")


def test_terminal_loader_exposes_unexpected_record_files(tmp_path) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    atomic_write_json(
        records_dir / "expected.json",
        {"item_id": "expected", "status": "success"},
    )
    atomic_write_json(
        records_dir / "unexpected.json",
        {"item_id": "unexpected", "status": "success"},
    )

    records, errors = _load_terminal_records(tmp_path)
    audit = audit_terminal_records(["expected"], records)

    assert errors == []
    assert audit["ok"] is False
    assert audit["unexpected_item_ids"] == ["unexpected"]


class _FakeQueue(queue.Queue):
    def close(self):
        return None


class _FakePolicy:
    Placement = SimpleNamespace(HOST_NAME="host-name")

    def __init__(self, *, placement, host_name, gpu_affinity):
        self.placement = placement
        self.host_name = host_name
        self.gpu_affinity = gpu_affinity


class _FakeTemplate:
    def __init__(self, *, target, args, policy):
        self.target = target
        self.args = args
        self.policy = policy


class _FakeGroup:
    last_walltime = None
    last_join_timeout = None
    last_closed = False
    last_stopped = False

    def __init__(self, *, restart, ignore_error_on_exit, walltime):
        assert restart is False
        assert ignore_error_on_exit is False
        type(self).last_walltime = walltime
        type(self).last_join_timeout = None
        type(self).last_closed = False
        type(self).last_stopped = False
        self.templates = []
        self.exit_status = []

    def add_process(self, *, nproc, template):
        assert nproc == 1
        self.templates.append(template)

    def init(self):
        return None

    def start(self):
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for index, template in enumerate(self.templates):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    template.policy.gpu_affinity[0]
                )
                template.target(*template.args)
                self.exit_status.append((1000 + index, 0))
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original

    def join(self, timeout=None):
        type(self).last_join_timeout = timeout
        return None

    def stop(self, patience=5.0):
        assert patience == 5.0
        type(self).last_stopped = True
        return None

    def close(self, patience=5.0):
        assert patience == 5.0
        type(self).last_closed = True
        return None

    @property
    def inactive_puids(self):
        return self.exit_status


class _FakeSystem:
    nodes = (1,)


class _FakeNode:
    hostname = "fake-node"
    gpus = [3, 7]

    def __init__(self, node_id):
        assert node_id == 1


class _FakeFailedGroup(_FakeGroup):
    def start(self):
        self.exit_status = [(1000, -9)]


class _FakeSetupFailedGroup(_FakeGroup):
    def add_process(self, *, nproc, template):
        del nproc, template
        raise RuntimeError("process setup failed")


class _FakeJoinTimedOutGroup(_FakeGroup):
    def start(self):
        return None

    def join(self, timeout=None):
        type(self).last_join_timeout = timeout
        raise TimeoutError("worker join deadline exceeded")


class _FakeStartFailedGroup(_FakeGroup):
    def start(self):
        raise RuntimeError("partial worker start failed")


class _FakeDecoratedJoinFailedGroup(_FakeGroup):
    last_forced_closed = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).last_forced_closed = False

    def start(self):
        return None

    def join(self, timeout=None):
        type(self).last_join_timeout = timeout
        raise RuntimeError("decorated worker failure")

    def stop(self, patience=5.0):
        assert patience == 5.0
        type(self).last_stopped = True
        raise RuntimeError("decorated worker failure")

    def close(self, patience=5.0):
        assert patience == 5.0
        type(self).last_closed = True
        raise RuntimeError("decorated worker failure")

    def _close_no_decorator(self, patience=5.0):
        assert patience == 5.0
        type(self).last_forced_closed = True


def _fake_worker(
    run_id,
    run_dir_raw,
    placement_payload,
    item_payloads,
    options_payload,
    results_queue,
    allow_loopback_alias,
):
    del allow_loopback_alias
    run_dir = dragon_module.Path(run_dir_raw)
    worker_id = placement_payload["worker_id"]
    requested_host = placement_payload["host"]
    requested_gpu_id = placement_payload["gpu_id"]
    backend = options_payload["backend"]
    identity_backend = "cupy" if backend in {"cupy", "cutile"} else backend
    for payload in item_payloads:
        item = WorkItem.from_dict(payload)
        atomic_write_json(
            run_dir / "records" / f"{item.item_id}.json",
            {
                "schema": "cuphoton.xpois.dragon-item/v1",
                "run_id": run_id,
                "item_id": item.item_id,
                "worker_id": worker_id,
                "weight_bytes": item.weight_bytes,
                "started_at_utc": "2026-08-21T00:00:00+00:00",
                "completed_at_utc": "2026-08-21T00:00:01+00:00",
                "worker_seconds": 1.0,
                "status": "success",
                "run_dir": f"items/{item.item_id}",
                "summary_path": f"items/{item.item_id}/summary.json",
                "requested_backend": "cupy",
                "backend": "cupy",
                "device": "fake-gpu",
                "runtime": {},
                "timings_sec": {"solve": 0.5},
                "wall_sec": {"item_runner": 0.75},
            },
        )
    results_queue.put(
        {
            "schema": "cuphoton.xpois.dragon-shard/v1",
            "worker_id": worker_id,
            "status": "success",
            "item_count": len(item_payloads),
            "success_count": len(item_payloads),
            "failed_count": 0,
            "weight_bytes": sum(
                WorkItem.from_dict(payload).weight_bytes
                for payload in item_payloads
            ),
            "item_ids_sha256": dragon_module._item_ids_sha256(
                [WorkItem.from_dict(payload) for payload in item_payloads]
            ),
            "started_at_utc": "2026-08-21T00:00:00+00:00",
            "completed_at_utc": "2026-08-21T00:00:01+00:00",
            "worker_wall_sec": 1.0,
            "timings_sec": {"solve": 0.5 * len(item_payloads)},
            "provenance": {
                "worker_id": worker_id,
                "requested_host": requested_host,
                "requested_gpu_id": requested_gpu_id,
                "hostname": requested_host,
                "pid": 1234,
                "cuda_visible_devices": str(requested_gpu_id),
                "gpu": {
                    "backend": identity_backend,
                    "name": "fake-gpu",
                    "uuid": f"GPU-worker-{worker_id}",
                    "pci_bus_id": f"0000:{requested_gpu_id:02x}:00.0",
                    "identity_error": None,
                },
            },
            "record_write_errors": [],
        }
    )


def _fake_corrupt_worker(
    run_id,
    run_dir_raw,
    placement_payload,
    item_payloads,
    options_payload,
    results_queue,
    allow_loopback_alias,
):
    local_results = _FakeQueue()
    _fake_worker(
        run_id,
        run_dir_raw,
        placement_payload,
        item_payloads,
        options_payload,
        local_results,
        allow_loopback_alias,
    )
    local_results.get_nowait()
    for payload in item_payloads:
        item = WorkItem.from_dict(payload)
        record_path = (
            dragon_module.Path(run_dir_raw)
            / "records"
            / f"{item.item_id}.json"
        )
        record = json.loads(record_path.read_text())
        record["timings_sec"] = "corrupt"
        atomic_write_json(record_path, record)
    results_queue.put(["corrupt"])


def _fake_inconsistent_shard_worker(
    run_id,
    run_dir_raw,
    placement_payload,
    item_payloads,
    options_payload,
    results_queue,
    allow_loopback_alias,
):
    local_results = _FakeQueue()
    _fake_worker(
        run_id,
        run_dir_raw,
        placement_payload,
        item_payloads,
        options_payload,
        local_results,
        allow_loopback_alias,
    )
    result = local_results.get_nowait()
    result["status"] = "failed"
    result["success_count"] = 0
    result["failed_count"] = len(item_payloads)
    results_queue.put(result)


def _fake_record_write_failure_shard_worker(
    run_id,
    run_dir_raw,
    placement_payload,
    item_payloads,
    options_payload,
    results_queue,
    allow_loopback_alias,
):
    local_results = _FakeQueue()
    _fake_worker(
        run_id,
        run_dir_raw,
        placement_payload,
        item_payloads,
        options_payload,
        local_results,
        allow_loopback_alias,
    )
    result = local_results.get_nowait()
    result["record_write_errors"] = [
        {
            "item_id": item_payloads[0]["item_id"],
            "type": "OSError",
            "message": "injected record write failure",
        }
    ]
    results_queue.put(result)


def _write_fake_manifest(tmp_path, *, item_ids=("one",)):
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    np.save(reference, np.zeros((16, 16)), allow_pickle=False)
    np.save(target, np.zeros((16, 16)), allow_pickle=False)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "cuphoton.xpois.image-pairs/v1",
                "pairs": [
                    {
                        "id": item_id,
                        "reference": str(reference),
                        "target": str(target),
                    }
                    for item_id in item_ids
                ],
            }
        )
    )
    return manifest


def test_dragon_shard_worker_round_trips_wire_payloads(
    monkeypatch, tmp_path
) -> None:
    seen = {}

    def fake_execute_shard(**kwargs):
        seen.update(kwargs)
        return {
            "worker_id": kwargs["placement"].worker_id,
            "status": "success",
        }

    monkeypatch.setattr(dragon_module, "_execute_shard", fake_execute_shard)
    results = _FakeQueue()
    placement = Placement(worker_id=3, host="node", gpu_id=7)
    item = WorkItem("wire-item", {"id": "wire-item"}, 42)

    _dragon_shard_worker(
        "wire-run",
        str(tmp_path),
        placement.to_dict(),
        [item.to_dict()],
        _options().to_payload(),
        results,
        True,
    )

    assert results.get_nowait() == {"worker_id": 3, "status": "success"}
    assert seen["run_id"] == "wire-run"
    assert seen["placement"] == placement
    assert seen["items"] == (item,)
    assert seen["options"] == _options()
    assert seen["allow_loopback_alias"] is True


def test_coordinator_audits_fake_dragon_process_group(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)
    monkeypatch.setattr(dragon_module, "_dragon_shard_worker", _fake_worker)

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "runs",
        run_id="fake-dragon",
        max_workers=2,
        result_timeout_sec=1.0,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "success"
    assert result.summary["terminal_record_audit"]["ok"] is True
    assert result.summary["process_exit_audit"]["ok"] is True
    assert result.summary["shard_result_audit"]["ok"] is True
    assert result.summary["placements"] == [
        {"worker_id": 0, "host": "fake-node", "gpu_id": 3}
    ]
    assert result.summary["distinct_host_count"] == 1
    assert result.summary["coordinator_wall_sec"] >= 0
    coordinator_timings = result.summary["coordinator_timings_sec"]
    assert "coordinator_setup_sec" not in coordinator_timings
    assert result.summary["coordinator_totals_sec"]["setup"] >= 0
    assert _FakeGroup.last_walltime == 30.0
    assert _FakeGroup.last_join_timeout == 31.0
    assert _FakeGroup.last_stopped is False
    assert _FakeGroup.last_closed is True
    launch = json.loads((result.run_dir / "run.json").read_text())
    assert launch["record_type"] == "immutable-launch"
    assert "status" not in launch
    assert (result.run_dir / "input-identity.json").is_file()


def test_coordinator_audits_two_worker_fanout(monkeypatch, tmp_path) -> None:
    manifest = _write_fake_manifest(
        tmp_path, item_ids=("one", "two", "three")
    )
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)
    monkeypatch.setattr(dragon_module, "_dragon_shard_worker", _fake_worker)

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "multi-runs",
        run_id="fake-dragon-multi",
        max_workers=2,
        result_timeout_sec=1.0,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "success"
    assert result.summary["placements"] == [
        {"worker_id": 0, "host": "fake-node", "gpu_id": 3},
        {"worker_id": 1, "host": "fake-node", "gpu_id": 7},
    ]
    assert result.summary["queue_result_count"] == 2
    assert result.summary["terminal_record_audit"]["expected_count"] == 3
    assert result.summary["terminal_record_audit"]["ok"] is True
    assert result.summary["process_exit_audit"] == {
        "ok": True,
        "expected_count": 2,
        "observed_count": 2,
        "nonzero": [],
    }
    assert result.summary["shard_result_audit"]["ok"] is True
    expected_shards = {
        shard["worker_id"]: shard for shard in result.summary["shards"]
    }
    observed_shards = {
        shard["worker_id"]: shard for shard in result.summary["shard_results"]
    }
    assert set(expected_shards) == set(observed_shards) == {0, 1}
    assert sum(shard["item_count"] for shard in expected_shards.values()) == 3
    for worker_id, expected in expected_shards.items():
        observed = observed_shards[worker_id]
        assert observed["item_count"] == expected["item_count"]
        assert observed["weight_bytes"] == expected["weight_bytes"]
        assert observed["item_ids_sha256"] == expected["item_ids_sha256"]


def test_coordinator_rejects_failed_shard_with_successful_records(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)
    monkeypatch.setattr(
        dragon_module,
        "_dragon_shard_worker",
        _fake_inconsistent_shard_worker,
    )

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "failed-shard-runs",
        run_id="fake-failed-shard",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "failed"
    assert result.summary["terminal_record_audit"]["ok"] is True
    assert result.summary["shard_result_audit"]["mismatched_shards"] == [
        {"worker_id": 0, "fields": ["failed_count", "success_count"]}
    ]


def test_coordinator_rejects_declared_record_write_errors(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)
    monkeypatch.setattr(
        dragon_module,
        "_dragon_shard_worker",
        _fake_record_write_failure_shard_worker,
    )

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "write-failed-shard-runs",
        run_id="fake-write-failed-shard",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "failed"
    shard_audit = result.summary["shard_result_audit"]
    assert shard_audit["ok"] is False
    assert shard_audit["mismatched_shards"] == []
    assert shard_audit["write_failed_shards"] == [
        {
            "worker_id": 0,
            "record_write_errors": [
                {
                    "item_id": "one",
                    "type": "OSError",
                    "message": "injected record write failure",
                }
            ],
        }
    ]


def test_coordinator_finalizes_corrupt_worker_payloads(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)
    monkeypatch.setattr(
        dragon_module, "_dragon_shard_worker", _fake_corrupt_worker
    )

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "corrupt-runs",
        run_id="fake-corrupt",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "failed"
    assert result.summary_path.is_file()
    assert result.summary["shard_result_audit"]["missing_worker_ids"] == [0]
    assert result.summary["queue_result_count"] == 1
    assert result.summary["lifecycle_errors"] == [
        {
            "phase": "queue_result",
            "type": "InvalidShardResult",
            "message": "worker result was not a mapping",
        }
    ]
    assert result.summary["terminal_record_errors"] == [
        {
            "item_id": "one",
            "type": "InvalidTerminalRecord",
            "message": "record 0 has invalid field(s): timings_sec",
        },
        {
            "item_id": "one",
            "type": "InvalidTimingRecord",
            "message": "item 'one' timings_sec must be a mapping",
        },
    ]


def test_coordinator_rejects_abrupt_worker_exit(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeFailedGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "failed-runs",
        run_id="fake-worker-killed",
        max_workers=1,
        result_timeout_sec=0.001,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "failed"
    assert result.summary["process_exit_audit"]["nonzero"] == [
        {"puid": 1000, "exit_code": -9}
    ]
    assert result.summary["terminal_record_audit"]["missing_item_ids"] == [
        "one"
    ]
    assert result.summary["shard_result_audit"]["missing_worker_ids"] == [0]


def test_coordinator_finalizes_process_setup_failure(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeSetupFailedGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "setup-failed-runs",
        run_id="fake-setup-failed",
        max_workers=1,
        result_timeout_sec=1.0,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "failed"
    assert result.summary_path.is_file()
    assert result.summary["lifecycle_errors"] == [
        {
            "phase": "process_setup",
            "type": "RuntimeError",
            "message": "process setup failed",
        }
    ]
    assert result.summary["terminal_record_audit"]["missing_item_ids"] == [
        "one"
    ]


def test_coordinator_finalizes_join_timeout(monkeypatch, tmp_path) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeJoinTimedOutGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "join-timeout-runs",
        run_id="fake-join-timeout",
        max_workers=1,
        result_timeout_sec=0.001,
        worker_timeout_sec=0.01,
        options=_options(),
    )

    assert result.status == "failed"
    assert result.summary_path.is_file()
    assert result.summary["lifecycle_errors"] == [
        {
            "phase": "join_timeout",
            "type": "TimeoutError",
            "message": "worker join deadline exceeded",
        }
    ]
    assert _FakeJoinTimedOutGroup.last_join_timeout == pytest.approx(0.011)
    assert _FakeJoinTimedOutGroup.last_stopped is True
    assert _FakeJoinTimedOutGroup.last_closed is True


def test_coordinator_stops_after_partial_start_failure(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeStartFailedGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "start-failed-runs",
        run_id="fake-start-failed",
        max_workers=1,
        result_timeout_sec=0.001,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "failed"
    assert result.summary["lifecycle_errors"] == [
        {
            "phase": "start",
            "type": "RuntimeError",
            "message": "partial worker start failed",
        }
    ]
    assert _FakeStartFailedGroup.last_stopped is True
    assert _FakeStartFailedGroup.last_closed is True


def test_coordinator_preserves_join_and_cleanup_errors(
    monkeypatch, tmp_path
) -> None:
    manifest = _write_fake_manifest(tmp_path)
    api = _DragonAPI(
        System=_FakeSystem,
        Node=_FakeNode,
        Policy=_FakePolicy,
        ProcessGroup=_FakeDecoratedJoinFailedGroup,
        ProcessTemplate=_FakeTemplate,
        Queue=_FakeQueue,
    )
    monkeypatch.setattr(dragon_module, "_load_dragon_api", lambda: api)

    result = run_dragon_image_pair_batch(
        manifest_path=manifest,
        output_root=tmp_path / "join-failed-runs",
        run_id="fake-join-failed",
        max_workers=1,
        result_timeout_sec=0.001,
        worker_timeout_sec=30.0,
        options=_options(),
    )

    assert result.status == "failed"
    assert result.summary["lifecycle_errors"] == [
        {
            "phase": "join",
            "type": "RuntimeError",
            "message": "decorated worker failure",
        },
        {
            "phase": "stop_after_failure",
            "type": "RuntimeError",
            "message": "decorated worker failure",
        },
        {
            "phase": "close",
            "type": "RuntimeError",
            "message": "decorated worker failure",
        },
    ]
    assert _FakeDecoratedJoinFailedGroup.last_stopped is True
    assert _FakeDecoratedJoinFailedGroup.last_closed is True
    assert _FakeDecoratedJoinFailedGroup.last_forced_closed is True


@pytest.mark.parametrize("backend", ["auto", "cpu"])
def test_coordinator_requires_explicit_gpu_backend(
    monkeypatch, tmp_path, backend
) -> None:
    monkeypatch.setattr(
        dragon_module,
        "_load_dragon_api",
        lambda: pytest.fail("Dragon must not load before backend validation"),
    )

    with pytest.raises(ValueError, match="explicit GPU backend"):
        run_dragon_image_pair_batch(
            manifest_path=tmp_path / "missing.json",
            output_root=tmp_path / "runs",
            run_id="invalid-backend",
            max_workers=1,
            result_timeout_sec=1.0,
            worker_timeout_sec=30.0,
            options=_options(backend=backend),
        )
