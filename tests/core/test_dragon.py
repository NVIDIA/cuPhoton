# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the persistent coordinator with threads and ordinary file IO."""

from __future__ import annotations

import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from cuphoton.core import dragon
from cuphoton.core.benchmark import BenchmarkOptions
from cuphoton.core.bulk import WorkItem, atomic_write_json
from cuphoton.core.execution import WorkloadSpec

_STATE = None


class _Worker:
    def __init__(self, options):
        self.options = options
        self.placement = _STATE.local.placement
        self.calls = 0
        _STATE.factories.append(self.placement.worker_id)
        _STATE.instances.append(self)
        self.gpu_identity = {
            "backend": options["backend"],
            "device_index": 0,
            "name": "fake-gpu",
            "uuid": f"GPU-{self.placement.worker_id}",
            "pci_bus_id": f"0000:{self.placement.gpu_id:02x}:00.0",
            "identity_error": None,
        }

    def run_item(self, item, output_dir):
        self.calls += 1
        _STATE.calls.append(
            (
                self.placement.worker_id,
                item.item_id,
                output_dir,
                threading.get_ident(),
            )
        )
        if _STATE.item_failure:
            raise ValueError("synthetic scientific failure")
        output_dir.mkdir()
        atomic_write_json(
            output_dir / "summary.json", {"value": item.payload["value"]}
        )
        return {
            "run_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "requested_backend": self.options["backend"],
            "backend": self.options["backend"],
            "device": "fake-gpu",
            "runtime": {},
            "timings_sec": {"solve": 0.1},
            "wall_sec": {},
        }

    def close(self):
        _STATE.closed.append(self.placement.worker_id)
        if _STATE.close_failure:
            raise RuntimeError("synthetic worker close failure")


def _factory(options):
    return _Worker(options)


def _install_runtime(
    monkeypatch, *, mutate=None, item_failure=False, close_failure=False
):
    global _STATE
    state = _STATE = SimpleNamespace(
        local=threading.local(),
        groups=[],
        queues=[],
        factories=[],
        instances=[],
        calls=[],
        closed=[],
        bindings=[],
        item_failure=item_failure,
        close_failure=close_failure,
    )

    class Queue(queue.Queue):
        def __init__(self, maxsize=0, policy=None):
            super().__init__(maxsize=maxsize)
            self.policy = policy
            self.closed = False
            state.queues.append(self)

        def put(self, value, block=True, timeout=None):
            values = (
                mutate(value)
                if mutate is not None and self.policy is None
                else [value]
            )
            for message in values:
                super().put(message, block=block, timeout=timeout)

        def close(self):
            self.closed = True

    class Policy:
        Placement = SimpleNamespace(HOST_NAME="host-name")

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Template:
        def __init__(self, target, args, policy):
            self.target, self.args, self.policy = target, args, policy
            self.argdata = repr(args).encode()

    class Group:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.templates, self.threads, self.inactive_puids = [], [], []
            self.stopped = self.closed = False
            state.groups.append(self)

        def add_process(self, *, nproc, template):
            assert nproc == 1
            self.templates.append(template)

        def init(self):
            pass

        def start(self):
            def invoke(index, template):
                try:
                    template.target(*template.args)
                except Exception:
                    self.inactive_puids.append((1000 + index, 1))
                else:
                    self.inactive_puids.append((1000 + index, 0))

            for index, template in enumerate(self.templates):
                thread = threading.Thread(
                    target=invoke, args=(index, template)
                )
                self.threads.append(thread)
                thread.start()

        def join(self, timeout):
            deadline = time.monotonic() + timeout
            for thread in self.threads:
                thread.join(max(0, deadline - time.monotonic()))
            if any(thread.is_alive() for thread in self.threads):
                raise TimeoutError("group join timed out")

        def stop(self, patience):
            self.stopped = True
            for template in self.templates:
                try:
                    template.args[4].put_nowait(None)
                except queue.Full:
                    pass
            for thread in self.threads:
                thread.join(patience)
            assert not any(thread.is_alive() for thread in self.threads)

        def close(self, patience):
            self.closed = True

    def binding(placement, *, allow_loopback_alias):
        assert allow_loopback_alias
        state.local.placement = placement
        state.bindings.append(placement.worker_id)
        return placement.host, str(placement.gpu_id)

    monkeypatch.setattr(dragon, "_validate_binding", binding)
    monkeypatch.setattr(
        dragon,
        "_load_dragon_api",
        lambda: dragon._DragonAPI(
            System=lambda: SimpleNamespace(nodes=(1,)),
            Node=lambda node_id: SimpleNamespace(
                hostname="fake-node.example", gpus=[3, 7]
            ),
            Policy=Policy,
            ProcessGroup=Group,
            ProcessTemplate=Template,
            Queue=Queue,
        ),
    )
    return state


def _run(
    tmp_path,
    *,
    benchmark=None,
    worker_count=2,
    worker_timeout=10,
    large=False,
    validator=None,
    finalizer=None,
):
    spec = WorkloadSpec(
        items=(
            WorkItem("one", {"value": "x" * 300_000 if large else 1}, 12),
            WorkItem("two", {"value": 2}, 6),
        ),
        options_payload={"backend": "cupy"},
        manifest_payload={"schema": "example"},
        input_identity_payload={"schema": "example-identity"},
        manifest_sha256="0" * 64,
        backend="cupy",
        worker_factory=_factory,
        success_record_validator=validator,
        finalize_round=finalizer,
    )
    return dragon.run_dragon_work_items(
        prepare_workload=lambda rank: spec,
        output_root=tmp_path,
        run_id="run",
        max_workers=worker_count,
        result_timeout_sec=0.2,
        worker_timeout_sec=worker_timeout,
        benchmark=benchmark,
    )


def test_persistent_workers_use_bounded_descriptors_and_all_rounds(
    monkeypatch, tmp_path
):
    state = _install_runtime(monkeypatch)
    validated, finalized = [], []

    def validator(record, round_dir):
        validated.append((record["item_id"], round_dir.name))
        return ()

    def finalizer(round_dir, records):
        finalized.append(round_dir.name)
        return {"item_ids": [record["item_id"] for record in records]}

    result = _run(
        tmp_path,
        benchmark=BenchmarkOptions(1, 2),
        large=True,
        validator=validator,
        finalizer=finalizer,
    )

    assert result.status == "success", result.summary
    assert sorted(state.factories) == sorted(state.bindings) == [0, 1]
    assert sorted(state.closed) == [0, 1]
    assert len(state.instances) == 2
    assert all(worker.calls == 3 for worker in state.instances)
    assert len(state.groups) == 1
    assert state.groups[0].closed and not state.groups[0].stopped
    assert all(channel.closed for channel in state.queues)
    assert all(
        channel.policy.host_name == "fake-node.example"
        for channel in state.queues
        if channel.policy
    )
    expected = ["warmup-0000", "measure-0000", "measure-0001"]
    assert finalized == expected
    assert len(validated) == 6
    for worker_id in range(2):
        calls = [call for call in state.calls if call[0] == worker_id]
        assert [call[2].parent.parent.name for call in calls] == expected
        assert len({call[3] for call in calls}) == 1
    for template in state.groups[0].templates:
        assert len(template.args) == 6
        assert len(template.argdata) < 2000
    descriptor = json.loads(
        (result.run_dir / "launch/worker-0000.json").read_text()
    )
    assert len(descriptor["items"][0]["payload"]["value"]) == 300_000
    report = result.summary["benchmark"]
    assert report["measured_batch_wall_sec"] is not None
    run_ids = set()
    for receipt in report["rounds"]:
        summary = json.loads(
            (result.run_dir / receipt["summary_path"]).read_text()
        )
        assert summary["terminal_record_audit"]["ok"]
        assert summary["result"] == {"item_ids": ["one", "two"]}
        run_ids.add(summary["run_id"])
    assert len(run_ids) == 3


def test_ordinary_workload_retains_root_artifacts(monkeypatch, tmp_path):
    state = _install_runtime(monkeypatch)
    result = _run(tmp_path)
    assert result.status == "success", result.summary
    assert "benchmark" not in result.summary
    assert (result.run_dir / "records/one.json").is_file()
    assert not (result.run_dir / "rounds").exists()
    assert all(worker.calls == 1 for worker in state.instances)
    assert sorted(state.closed) == [0, 1]


@pytest.mark.parametrize(
    "failure",
    [
        "wrong-round",
        "duplicate-ready",
        "missing-round",
        "invalid-duration",
        "changed-provenance",
        "scientific",
        "close",
    ],
)
def test_failures_keep_evidence_and_invalidate_aggregates(
    monkeypatch, tmp_path, failure
):
    def mutate(message):
        if failure == "duplicate-ready" and message["kind"] == "ready":
            return [message, message]
        if message["kind"] == "round":
            if failure == "wrong-round":
                return [{**message, "round_id": "measure-9999"}]
            if failure == "missing-round":
                return []
            if failure == "invalid-duration":
                return [{**message, "worker_wall_sec": True}]
            if failure == "changed-provenance":
                result = dict(message["result"])
                result["provenance"] = {**result["provenance"], "pid": 99999}
                return [{**message, "result": result}]
        return [message]

    state = _install_runtime(
        monkeypatch,
        mutate=mutate,
        item_failure=failure == "scientific",
        close_failure=failure == "close",
    )
    result = _run(
        tmp_path,
        benchmark=BenchmarkOptions(0, 1),
        worker_count=1,
        worker_timeout=0.5 if failure == "missing-round" else 10,
    )
    assert result.status == "failed"
    assert result.summary["benchmark"]["measured_batch_wall_sec"] is None
    assert result.summary["lifecycle_errors"]
    assert state.groups[0].stopped and state.groups[0].closed
    assert all(channel.closed for channel in state.queues)
    assert (result.run_dir / "summary.json").is_file()
    if failure == "close":
        assert result.summary["benchmark"]["rounds"][0]["status"] == "success"
        assert (
            result.summary["closed_messages"][0]["error"]["message"]
            == "synthetic worker close failure"
        )


def test_descriptor_tampering_fails_before_worker_factory(
    monkeypatch, tmp_path
):
    state = _install_runtime(monkeypatch)
    read_descriptor = dragon._read_descriptor

    def invalid_digest(run_id, descriptor_path, digest, context):
        return read_descriptor(run_id, descriptor_path, "f" * 64, context)

    monkeypatch.setattr(dragon, "_read_descriptor", invalid_digest)
    result = _run(tmp_path, worker_count=1)
    assert result.status == "failed"
    assert not state.factories
    assert not state.calls
    startup = result.summary["ready_messages"][0]
    assert startup["status"] == "failed"
    assert "SHA-256" in startup["error"]["message"]
    assert (result.run_dir / "startup/worker-0000.json").is_file()
    assert state.groups[0].closed


def test_later_failure_keeps_successful_warmup(monkeypatch, tmp_path):
    def corrupt_measured(message):
        if (
            message["kind"] == "round"
            and message["round_id"] == "measure-0000"
        ):
            return [{**message, "round_id": "warmup-0000"}]
        return [message]

    _install_runtime(monkeypatch, mutate=corrupt_measured)
    result = _run(tmp_path, benchmark=BenchmarkOptions(1, 2), worker_count=1)
    assert result.status == "failed"
    rounds = result.summary["benchmark"]["rounds"]
    assert [entry["round_id"] for entry in rounds] == [
        "warmup-0000",
        "measure-0000",
    ]
    assert [entry["status"] for entry in rounds] == ["success", "failed"]
    assert (result.run_dir / "rounds/warmup-0000/summary.json").is_file()
    assert result.summary["benchmark"]["measured_batch_wall_sec"] is None


def test_scientific_validator_failure_suppresses_component_finalizer(
    monkeypatch, tmp_path
):
    _install_runtime(monkeypatch)
    finalized = []
    result = _run(
        tmp_path,
        worker_count=1,
        validator=lambda record, path: ("scientific receipt differs",),
        finalizer=lambda path, records: finalized.append(path) or {},
    )
    assert result.status == "failed"
    assert not finalized
    assert any(
        "scientific receipt differs" in error["message"]
        for error in result.summary["errors"]
    )
