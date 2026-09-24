# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, local
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.core import mpi
from cuphoton.core._mpi_runtime import _MPIAPI
from cuphoton.core.benchmark import BenchmarkOptions


@pytest.mark.parametrize("component", ["xfit", "xscan"])
@pytest.mark.parametrize("repeated", [False, True])
def test_two_rank_adapters_publish_scientific_results(
    tmp_path, monkeypatch, component, repeated
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
    for name in ("cupy", "numba.cuda", "cuda.tile"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    state = local()
    barrier = Barrier(2, timeout=10)
    messages = [None, None]

    class Comm:
        def Get_rank(self):
            return state.rank

        def Get_size(self):
            return 2

        def Split_type(self, split_type, key):
            return SimpleNamespace(
                Get_rank=lambda: state.rank, Free=lambda: None
            )

        def exchange(self, kind, value, root):
            messages[state.rank] = (kind, root, pickle.dumps(value))
            barrier.wait()
            assert all(message[:2] == (kind, root) for message in messages)
            values = [pickle.loads(message[2]) for message in messages]
            barrier.wait()
            return values

        def gather(self, value, root):
            values = self.exchange("gather", value, root)
            return values if state.rank == root else None

        def bcast(self, value, root):
            return self.exchange("bcast", value, root)[root]

    monkeypatch.setattr(
        mpi,
        "_load_mpi_api",
        lambda: _MPIAPI(
            SimpleNamespace(COMM_TYPE_SHARED=1),
            Comm(),
            "test-mpi4py",
            "test-MPI",
        ),
    )

    def gpu_identity(backend):
        return {
            "backend": backend,
            "device_index": 0,
            "uuid": f"GPU-{state.rank}",
            "identity_error": None,
        }

    if component == "xfit":
        from cuphoton.xfit import executor, fit_dipoles

        from ..xfit.test_executor import _input

        path = _input(tmp_path)

        def cpu_fit(*args, **kwargs):
            result = fit_dipoles(*args, **{**kwargs, "backend": "numpy"})
            return replace(result, backend="cupy", device="cuda:0")

        monkeypatch.setattr(executor, "fit_dipoles", cpu_fit)
        monkeypatch.setattr(executor, "collect_gpu_identity", gpu_identity)

        def prepare(rank):
            return executor.prepare_xfit_workload(
                input_path=path,
                chunk_size=2,
                fit_options={"backend": "cupy", "max_evaluations": 2},
            )

    else:
        torch = pytest.importorskip("torch")
        from cuphoton.xscan import executor

        from ..xscan.test_executor import _inputs

        run_dir, dataset_dir = _inputs(tmp_path)
        monkeypatch.setattr(executor, "_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(
            executor, "_gpu_identity", lambda: gpu_identity("torch")
        )

        def prepare(rank):
            return executor.prepare_inference_workload(
                run_dir=run_dir,
                dataset_dir=dataset_dir,
                split="test",
                batch_size=2,
                task_batches=1,
            )

    def run(rank):
        state.rank = rank
        return mpi.run_mpi_work_items(
            prepare_workload=prepare,
            output_root=tmp_path,
            run_id="distributed",
            rank_setup_timeout_sec=1,
            benchmark=BenchmarkOptions(warmup_rounds=1, measure_rounds=2)
            if repeated
            else None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        root, peer = list(pool.map(run, (0, 1)))
    assert peer is None
    assert root.status == "success", root.summary
    run_dir = tmp_path / "distributed"
    rounds = (
        [
            run_dir / "rounds" / name
            for name in ("warmup-0000", "measure-0000", "measure-0001")
        ]
        if repeated
        else [run_dir]
    )
    for round_dir in rounds:
        scientific = round_dir / "scientific"
        assert (scientific / "summary.json").is_file()
        if component == "xfit":
            import pyarrow.parquet as pq

            assert pq.read_table(scientific / "fits.parquet")[
                "candidate_id"
            ].to_pylist() == ["z", "a", "m"]
        else:
            np.testing.assert_array_equal(
                np.load(scientific / "sample_index.npy"), [0, 2, 3, 4, 6]
            )
