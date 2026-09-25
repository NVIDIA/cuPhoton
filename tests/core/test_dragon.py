# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the persistent coordinator with threads and ordinary file IO."""

from __future__ import annotations

import json
import queue
import sys
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
    monkeypatch,
    *,
    mutate=None,
    item_failure=False,
    close_failure=False,
    interleave_close=False,
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
        fast_result=threading.Event(),
        fast_waiting_for_close=threading.Event(),
        release_slow_result=threading.Event(),
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
                if (
                    interleave_close
                    and self.policy is None
                    and message["kind"] == "round"
                    and message["worker_id"] == 1
                ):
                    assert state.release_slow_result.wait(timeout=5)
                super().put(message, block=block, timeout=timeout)
                if interleave_close and self.policy is None:
                    if (
                        message["kind"] == "round"
                        and message["worker_id"] == 0
                    ):
                        state.fast_result.set()
                    elif (
                        message["kind"] == "closed"
                        and message["worker_id"] == 0
                    ):
                        state.release_slow_result.set()

        def get(self, block=True, timeout=None):
            if (
                interleave_close
                and self.policy is not None
                and self.policy.gpu_affinity == [3]
                and state.fast_result.is_set()
            ):
                state.fast_waiting_for_close.set()
                state.release_slow_result.set()
            return super().get(block=block, timeout=timeout)

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
                    state.local.puid = 1000 + index
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
    monkeypatch.setattr(dragon, "_current_puid", lambda: state.local.puid)
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
    result_timeout=0.2,
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
        result_timeout_sec=result_timeout,
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
        assert summary["finalization_sec"] == receipt["finalization_sec"]
        assert receipt["finalization_sec"] >= 0
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


def test_fast_worker_waits_for_close_until_all_round_results_arrive(
    monkeypatch, tmp_path
):
    state = _install_runtime(monkeypatch, interleave_close=True)
    result = _run(tmp_path)

    assert result.status == "success", result.summary
    assert state.fast_waiting_for_close.is_set()
    assert sorted(state.closed) == [0, 1]
    assert len(result.summary["messages"]) == 2
    assert len(result.summary["closed_messages"]) == 2
    assert state.groups[0].closed and not state.groups[0].stopped


@pytest.mark.parametrize(
    "kind, exit_code",
    [
        ("ready", -9),
        ("round", -9),
        ("closed", -9),
        ("ready", 0),
        ("round", 0),
        ("closed", 0),
    ],
)
def test_collector_detects_native_exit_on_first_bounded_poll(kind, exit_code):
    timeouts = []

    class EmptyQueue:
        def get(self, *, timeout):
            timeouts.append(timeout)
            raise queue.Empty

    with pytest.raises(RuntimeError, match="worker exited"):
        dragon._collect_messages(
            EmptyQueue(),
            [],
            worker_count=1,
            run_id="run",
            kind=kind,
            round_id=None,
            deadline=time.monotonic() + 2,
            group=SimpleNamespace(inactive_puids=[(123, exit_code)]),
        )
    assert len(timeouts) == 2
    assert 0 < timeouts[0] <= 1
    assert timeouts[1] == 0


def test_collector_keeps_queued_results_before_reporting_peer_exit():
    receipt = {
        "kind": "ready",
        "run_id": "run",
        "round_id": None,
        "worker_id": 0,
        "puid": 1000,
        "status": "success",
    }

    class ResultsQueue:
        calls = 0

        def get(self, *, timeout):
            self.calls += 1
            if self.calls == 1:
                return receipt
            raise queue.Empty

    messages = []
    with pytest.raises(RuntimeError, match="worker exited"):
        dragon._collect_messages(
            ResultsQueue(),
            messages,
            worker_count=2,
            run_id="run",
            kind="ready",
            round_id=None,
            deadline=time.monotonic() + 2,
            group=SimpleNamespace(inactive_puids=[(123, -9)]),
        )
    assert messages == [receipt]


def test_closed_collection_allows_normal_exit_before_delayed_receipt():
    receipt = {
        "kind": "closed",
        "run_id": "run",
        "round_id": None,
        "worker_id": 0,
        "puid": 1000,
        "status": "success",
    }

    class DelayedQueue:
        calls = 0

        def get(self, *, timeout):
            self.calls += 1
            if self.calls == 1:
                raise queue.Empty
            return receipt

    channel = DelayedQueue()
    messages = []
    dragon._collect_messages(
        channel,
        messages,
        worker_count=1,
        run_id="run",
        kind="closed",
        round_id=None,
        deadline=time.monotonic() + 2,
        group=SimpleNamespace(inactive_puids=[(123, 0)]),
    )
    assert channel.calls == 2
    assert messages == [receipt]


def test_native_worker_crash_stops_group_without_ready_receipt(
    monkeypatch, tmp_path
):
    state = _install_runtime(monkeypatch)

    def crash(*args):
        raise RuntimeError("synthetic native crash")

    monkeypatch.setattr(dragon, "_workload_worker", crash)
    result = _run(tmp_path, worker_count=1)
    assert result.status == "failed"
    assert not result.summary["ready_messages"]
    assert any(
        "worker exited before ready" in error["message"]
        for error in result.summary["lifecycle_errors"]
    )
    assert state.groups[0].stopped and state.groups[0].closed


@pytest.mark.parametrize("close_failure", [False, True])
def test_terminal_summary_is_absent_until_cleanup(
    monkeypatch, tmp_path, close_failure
):
    state = _install_runtime(monkeypatch, close_failure=close_failure)
    close = _Worker.close
    group_type = dragon._load_dragon_api().ProcessGroup
    join = group_type.join
    group_close = group_type.close
    observations = []

    def inspect_close(worker):
        observations.append(
            ("worker_close", (tmp_path / "run/summary.json").exists())
        )
        return close(worker)

    def inspect_join(group, timeout):
        observations.append(
            ("join", (tmp_path / "run/summary.json").exists())
        )
        return join(group, timeout)

    def inspect_group_close(group, patience):
        observations.append(
            ("group_close", (tmp_path / "run/summary.json").exists())
        )
        return group_close(group, patience)

    monkeypatch.setattr(_Worker, "close", inspect_close)
    monkeypatch.setattr(group_type, "join", inspect_join)
    monkeypatch.setattr(group_type, "close", inspect_group_close)
    result = _run(tmp_path)
    assert not any(exists for _, exists in observations)
    assert [phase for phase, _ in observations].count("worker_close") == 2
    assert ("join", False) in observations or close_failure
    assert ("group_close", False) in observations
    assert result.status == ("failed" if close_failure else "success")
    assert len(result.summary["closed_messages"]) == 2
    assert all(not thread.is_alive() for thread in state.groups[0].threads)
    assert json.loads(result.summary_path.read_text()) == result.summary


def test_failed_round_collects_slow_peer_after_failed_worker_closes(
    monkeypatch, tmp_path
):
    failed_closed = threading.Event()

    def observe(message):
        if message["kind"] == "closed" and message["worker_id"] == 0:
            failed_closed.set()
        return [message]

    state = _install_runtime(monkeypatch, mutate=observe)
    run_item = _Worker.run_item

    def run_with_failure(worker, item, output_dir):
        if worker.placement.worker_id == 0:
            raise ValueError("first worker failed")
        assert failed_closed.wait(timeout=2)
        return run_item(worker, item, output_dir)

    monkeypatch.setattr(_Worker, "run_item", run_with_failure)
    result = _run(tmp_path, benchmark=BenchmarkOptions(0, 2))
    assert result.status == "failed"
    assert len(result.summary["benchmark"]["rounds"]) == 1
    summary = json.loads(
        (result.run_dir / "rounds/measure-0000/summary.json").read_text()
    )
    assert len(summary["worker_results"]) == 2
    assert summary["terminal_record_audit"]["ok"]
    assert len(summary["terminal_record_audit"]["failed_item_ids"]) == 1
    assert sorted(record["status"] for record in summary["records"]) == [
        "failed",
        "success",
    ]
    assert all(not thread.is_alive() for thread in state.groups[0].threads)
    assert not any(
        error["phase"] == "stop_after_failure"
        for error in result.summary["lifecycle_errors"]
    )


def test_collector_drains_receipt_after_exit_snapshot_before_failing():
    receipt = {
        "kind": "ready",
        "run_id": "run",
        "round_id": None,
        "worker_id": 0,
        "puid": 123,
        "status": "failed",
    }
    timeouts = []

    class RacingQueue:
        def get(self, *, timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise queue.Empty
            return receipt

    messages = []
    with pytest.raises(RuntimeError, match="ready failed"):
        dragon._collect_messages(
            RacingQueue(),
            messages,
            worker_count=1,
            run_id="run",
            kind="ready",
            round_id=None,
            deadline=time.monotonic() + 2,
            group=SimpleNamespace(inactive_puids=[(123, 1)]),
        )
    assert messages == [receipt]
    assert timeouts[1] == 0


@pytest.mark.parametrize("kind", ["ready", "round", "closed"])
def test_failed_receipt_does_not_abort_slow_peer_on_reported_exit(kind):
    first = {
        "kind": kind,
        "run_id": "run",
        "round_id": None,
        "worker_id": 0,
        "puid": 1000,
        "status": "failed",
    }
    peer = {**first, "worker_id": 1, "puid": 1001, "status": "failed"}
    sequence = iter((first, queue.Empty, queue.Empty, peer))

    class DelayedQueue:
        def get(self, *, timeout):
            value = next(sequence)
            if value is queue.Empty:
                raise queue.Empty
            return value

    messages = []
    with pytest.raises(RuntimeError, match=kind + " failed"):
        dragon._collect_messages(
            DelayedQueue(),
            messages,
            worker_count=2,
            run_id="run",
            kind=kind,
            round_id=None,
            deadline=time.monotonic() + 2,
            group=SimpleNamespace(inactive_puids=[(1000, 1)]),
        )
    assert messages == [first, peer]


def test_known_round_failure_uses_short_artifact_timeout(
    monkeypatch, tmp_path
):
    _install_runtime(
        monkeypatch,
        mutate=lambda message: (
            [{**message, "status": "failed"}]
            if message["kind"] == "round"
            else [message]
        ),
    )
    finalize = dragon.finalize_round
    observed = []

    def inspect_finalize(*args, **kwargs):
        observed.append(kwargs["artifact_timeout_sec"])
        return finalize(*args, **kwargs)

    monkeypatch.setattr(dragon, "finalize_round", inspect_finalize)
    result = _run(tmp_path, worker_count=1, result_timeout=30)
    assert result.status == "failed"
    assert observed == [0.01]


@pytest.mark.parametrize("reject_terminal_put", [False, True])
def test_interrupted_worker_keeps_failed_close_evidence(
    monkeypatch, tmp_path, reject_terminal_put
):
    state = _install_runtime(monkeypatch)
    _run(tmp_path, worker_count=1)
    args = list(state.groups[0].templates[0].args)
    state.local.puid = 1000

    class InterruptedCommands:
        def get(self, **kwargs):
            raise KeyboardInterrupt("synthetic interrupt")

    class Results:
        def put(self, message, **kwargs):
            if reject_terminal_put and message["kind"] == "closed":
                raise queue.Full

    args[4:] = [InterruptedCommands(), Results()]
    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        dragon._workload_worker(*args)
    closed = json.loads(
        (tmp_path / "run/startup/worker-0000-closed.json").read_text()
    )
    assert closed["status"] == "failed"
    assert closed["error"]["type"] == "KeyboardInterrupt"


@pytest.mark.parametrize("initialized", [False, True])
def test_binding_allows_torch_import_without_cuda(monkeypatch, initialized):
    for name in ("cupy", "numba.cuda", "cuda.tile"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_initialized=lambda: initialized)
        ),
    )
    monkeypatch.setattr(
        dragon.socket, "gethostname", lambda: "worker.example"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    placement = dragon.Placement(0, "worker.example", 3)
    if initialized:
        with pytest.raises(RuntimeError, match="torch CUDA"):
            dragon._validate_binding(placement, allow_loopback_alias=False)
    else:
        assert dragon._validate_binding(
            placement, allow_loopback_alias=False
        ) == ("worker.example", "3")
