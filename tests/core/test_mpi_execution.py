# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from types import SimpleNamespace

import pytest

from cuphoton.core import mpi
from cuphoton.core._mpi_runtime import _MPIAPI
from cuphoton.core.benchmark import BenchmarkOptions
from cuphoton.core.bulk import read_json_mapping
from cuphoton.core.execution import ExecutionResult

from .test_execution import FakeWorker, workload

_STATE = local()


class TrackingWorker(FakeWorker):
    def __init__(self, options):
        self.rank = _STATE.rank
        self.trace = _STATE.trace
        self.trace.append(("initialize", self.rank))
        if options["failure"] == "initialize" and self.rank == 1:
            raise ValueError("worker initialization failed")
        super().__init__({**options, "uuid": f"GPU-{self.rank}"})

    def run_item(self, item, output_dir):
        round_id = output_dir.parent.parent.name
        self.trace.append(("execute", self.rank, round_id, id(self)))
        if self.options["failure"] == "item" and self.rank == 1:
            raise ValueError("item execution failed")
        return super().run_item(item, output_dir)

    def close(self):
        self.trace.append(("close", self.rank))
        if self.options["failure"] == "close" and self.rank == 1:
            raise ValueError("worker close failed")
        super().close()


def tracking_factory(options):
    return TrackingWorker(options)


@pytest.mark.parametrize(
    "failure",
    [None, "preflight", "plan", "initialize", "item", "close", "unexpected"],
)
def test_two_rank_persistent_lifecycle_and_collective_failures(
    monkeypatch, tmp_path, failure
):
    for name in tuple(os.environ):
        if name.startswith(
            (
                "OMPI_",
                "PMI_",
                "PMIX_",
                "SLURM_",
                "MPI_LOCAL",
                "CUPHOTON_ALLOCATED",
            )
        ):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for name in ("cupy", "numba.cuda", "cuda.tile", "torch"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    barrier = Barrier(2, timeout=5)
    shared = SimpleNamespace(gathered=[None, None], broadcast=None)
    trace = []

    class Comm:
        def __init__(self, rank):
            self.rank = rank

        def Get_rank(self):
            return self.rank

        def Get_size(self):
            return 2

        def Split_type(self, split_type, key):
            return SimpleNamespace(
                Get_rank=lambda: self.rank, Free=lambda: None
            )

        def gather(self, value, root):
            shared.gathered[self.rank] = value
            barrier.wait()
            result = list(shared.gathered) if self.rank == root else None
            barrier.wait()
            return result

        def bcast(self, value, root):
            if self.rank == root:
                shared.broadcast = value
            barrier.wait()
            result = shared.broadcast
            barrier.wait()
            return result

    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: _MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            Comm(_STATE.rank),
            "test-mpi4py",
            "test-MPI",
        ),
    )
    if failure == "unexpected":
        execute = mpi.execute_worker_round

        def fail_outside_item(*args, **kwargs):
            if _STATE.rank == 1:
                raise RuntimeError("unexpected worker loop failure")
            return execute(*args, **kwargs)

        monkeypatch.setattr(mpi, "execute_worker_round", fail_outside_item)

    def prepare(rank):
        trace.append(("preflight", rank))
        if failure == "preflight" and rank == 1:
            raise ValueError("bad rank input")
        return workload(
            worker_factory=tracking_factory,
            options_payload={"failure": failure},
        )

    def run(rank):
        _STATE.rank = rank
        _STATE.trace = trace
        try:
            return mpi.run_mpi_work_items(
                prepare_workload=prepare,
                output_root=tmp_path,
                run_id="parallel",
                rank_setup_timeout_sec=0.01,
                benchmark=None
                if failure == "plan" and rank == 1
                else BenchmarkOptions(warmup_rounds=1, measure_rounds=2),
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        root, peer = list(pool.map(run, (0, 1)))
    if failure in {"preflight", "plan"}:
        assert isinstance(root, RuntimeError)
        assert isinstance(peer, RuntimeError)
        assert not (tmp_path / "parallel").exists()
        assert not any(event[0] == "initialize" for event in trace)
        return
    assert isinstance(root, ExecutionResult), repr(root)
    assert sorted(event for event in trace if event[0] == "initialize") == [
        ("initialize", 0),
        ("initialize", 1),
    ]
    for rank in (0, 1):
        instance_ids = {
            event[3]
            for event in trace
            if event[0] == "execute" and event[1] == rank
        }
        assert len(instance_ids) <= 1
    closed = sorted(event[1] for event in trace if event[0] == "close")
    assert closed == ([0] if failure == "initialize" else [0, 1])
    report = root.summary["benchmark"]
    assert read_json_mapping(root.summary_path) == root.summary
    if failure is None:
        assert root.status == "success"
        assert peer is None
        assert len(report["rounds"]) == 3
        assert report["measured_batch_wall_sec"] is not None
        assert len([event for event in trace if event[0] == "execute"]) == 6
    else:
        assert root.status == "failed"
        assert isinstance(peer, RuntimeError)
        assert report["measured_batch_wall_sec"] is None
        assert report["errors"]
        if failure == "close":
            assert len(report["rounds"]) == 3
            assert all(
                round_result["status"] == "success"
                for round_result in report["rounds"]
            )
        elif failure == "initialize":
            assert report["rounds"] == []
        else:
            assert len(report["rounds"]) == 1
            assert report["rounds"][0]["status"] == "failed"


def test_ordinary_mode_and_rank_binding_before_worker_factory(
    monkeypatch, tmp_path
):
    class Comm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Split_type(self, split_type, key):
            return SimpleNamespace(Get_rank=lambda: 0, Free=lambda: None)

        def gather(self, value, root):
            return [value]

        def bcast(self, value, root):
            return value

    for name in tuple(os.environ):
        if name.startswith(
            (
                "OMPI_",
                "PMI_",
                "PMIX_",
                "SLURM_",
                "MPI_LOCAL",
                "CUPHOTON_ALLOCATED",
            )
        ):
            monkeypatch.delenv(name, raising=False)
    for name in ("cupy", "numba.cuda", "cuda.tile", "torch"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    comm = Comm()
    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: _MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1), comm, "test", "test"
        ),
    )
    _STATE.rank = 0
    _STATE.trace = []
    result = mpi.run_mpi_work_items(
        prepare_workload=lambda rank: workload(
            worker_factory=tracking_factory, options_payload={"failure": None}
        ),
        output_root=tmp_path,
        run_id="once",
    )
    assert result.status == "success"
    assert "benchmark" not in result.summary
    assert not (result.run_dir / "rounds").exists()
    assert (
        len([event for event in _STATE.trace if event[0] == "execute"]) == 2
    )
    assert _STATE.trace[-1] == ("close", 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="one CUDA device"):
        mpi.run_mpi_work_items(
            prepare_workload=lambda rank: pytest.fail(
                "preflight before binding"
            ),
            output_root=tmp_path,
            run_id="unbound",
        )
    assert not (tmp_path / "unbound").exists()
