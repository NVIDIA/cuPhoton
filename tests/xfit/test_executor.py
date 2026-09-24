# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pyarrow.parquet as pq
import pytest

from cuphoton.xfit import (
    DipoleFitResult,
    GaussianDipoleModel,
    StampDipoleModel,
    executor,
    fit_dipoles,
)
from cuphoton.xfit.io import load_xfit_dataset


def _input(tmp_path, *, mode="difference", auxiliary="candidate"):
    model = GaussianDipoleModel((7, 9), dtype=np.float64)
    truth = np.array(
        [
            [4, 1.2, 1.4, 0.1, 2, 2, 4, 2.5],
            [5, 1.3, 1.1, -0.1, 2.1, 1.9, 4.1, 2.6],
            [3, 1.1, 1.3, 0.2, 2.2, 2.2, 4.2, 2.4],
        ],
        dtype=np.float64,
    )
    images = np.asarray(model.evaluate(truth, mode=mode))
    variance = np.full((3, 7, 9), 2.0)
    variance[1] = 3.0
    mask = np.ones_like(variance, dtype=bool)
    mask[1, 0, 0] = False
    if mode == "split" and auxiliary == "plane":
        variance = variance[None, :]
        mask = mask[None, :]
    elif auxiliary == "scalar":
        variance = np.asarray(2.0)
        mask = np.asarray(True)
    path = tmp_path / "input.npz"
    np.savez_compressed(
        path,
        candidate_id=np.array(["z", "a", "m"]),
        images=images,
        initial=truth + 0.01,
        variance=variance,
        mask=mask,
    )
    return path


@pytest.mark.parametrize("runtime", ["dragon", "mpi"])
def test_cli_dispatches_collective_preflight_and_preserves_options(
    tmp_path, monkeypatch, capsys, runtime
):
    from types import SimpleNamespace

    from cuphoton.core import executors
    from cuphoton.core.cli import run_component

    path = _input(tmp_path)
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
            "xfit",
            [
                "fit-dipoles",
                "--input",
                str(path),
                "--model",
                "gaussian",
                "--backend",
                "cupy",
                "--executor",
                runtime,
                "--chunk-size",
                "2",
                "--output-dir",
                str(output_dir),
                "--max-evaluations",
                "10",
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
    assert kwargs["benchmark"].measure_rounds == 2
    assert len(spec.items) == 2
    assert spec.options_payload["fit_options"]["max_evaluations"] == 10
    assert json.loads(capsys.readouterr().out)["status"] == "success"


@pytest.mark.parametrize(
    "mode,auxiliary",
    [
        ("difference", "candidate"),
        ("split", "candidate"),
        ("split", "plane"),
        ("difference", "scalar"),
    ],
)
def test_chunks_preserve_numerics_broadcasting_and_order(
    tmp_path, monkeypatch, mode, auxiliary
):
    path = _input(tmp_path, mode=mode, auxiliary=auxiliary)
    items, options = executor.plan_xfit_chunks(
        path,
        chunk_size=2,
        fit_options={"backend": "numpy", "mode": mode, "max_evaluations": 10},
    )
    identities = []
    monkeypatch.setattr(
        executor,
        "collect_gpu_identity",
        lambda backend: identities.append(backend) or {"backend": "numpy"},
    )
    worker = executor.create_xfit_worker(options)
    dataset = load_xfit_dataset(path, mode=mode, model="gaussian")
    ordinary = fit_dipoles(
        dataset.images,
        initial=dataset.initial,
        mask=dataset.mask,
        variance=dataset.variance,
        mode=mode,
        model="gaussian",
        backend="numpy",
        config=worker.config,
    )
    for round_index in range(2):
        round_dir = tmp_path / f"round-{round_index}"
        records = []
        for item in reversed(items):
            worker.run_item(item, round_dir / "items" / item.item_id)
            records.append({"item_id": item.item_id, "status": "success"})
        receipt = executor.finalize_xfit_round(
            round_dir, records, items=items, options=options
        )
        assert receipt["candidate_count"] == 3
        chunks = [
            executor._read_result(item, round_dir, options)[1]
            for item in items
        ]
        for field in fields(DipoleFitResult):
            if field.name in executor._META_FIELDS:
                continue
            actual = np.concatenate([chunk[field.name] for chunk in chunks])
            expected = np.asarray(getattr(ordinary, field.name))
            if expected.dtype.kind in "biuUS":
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
                )
        rows = pq.read_table(
            round_dir / "scientific" / "fits.parquet"
        ).to_pylist()
        assert [row["candidate_id"] for row in rows] == ["z", "a", "m"]
        assert [row["candidate_index"] for row in rows] == [0, 1, 2]
        with np.load(
            round_dir / "scientific" / "fit-arrays.npz", allow_pickle=False
        ) as saved:
            np.testing.assert_allclose(
                saved["residuals"], ordinary.residuals, atol=1e-12
            )
        summary = json.loads(
            (round_dir / "scientific" / "summary.json").read_text()
        )
        assert (
            summary["inputs"]["input_archive_sha256"]
            == dataset.input_archive_sha256
        )
        assert summary["metrics"]["candidate_count"] == 3
    assert identities == ["numpy"]
    worker.close()


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "failed", "corrupt"]
)
def test_merge_rejects_incomplete_or_changed_scientific_results(
    tmp_path, monkeypatch, defect
):
    path = _input(tmp_path)
    items, options = executor.plan_xfit_chunks(
        path,
        chunk_size=2,
        fit_options={"backend": "numpy", "max_evaluations": 2},
    )
    monkeypatch.setattr(executor, "collect_gpu_identity", lambda _: {})
    worker = executor.create_xfit_worker(options)
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
        records[1]["status"] = "failed"
    else:
        (round_dir / "items" / items[0].item_id / "result.npz").write_bytes(
            b"changed"
        )
    with pytest.raises(ValueError):
        executor.finalize_xfit_round(
            round_dir, records, items=items, options=options
        )
    assert not (round_dir / "scientific").exists()


def test_retained_worker_rejects_changed_source_archive(
    tmp_path, monkeypatch
):
    path = _input(tmp_path)
    items, options = executor.plan_xfit_chunks(
        path, chunk_size=2, fit_options={"backend": "numpy"}
    )
    monkeypatch.setattr(executor, "collect_gpu_identity", lambda _: {})
    worker = executor.create_xfit_worker(options)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="input archive changed"):
        worker.run_item(items[0], tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_stamp_chunks_retain_shared_basis_and_forced_finite_difference(
    tmp_path, monkeypatch
):
    y, x = np.mgrid[-4:5, -4:5]
    basis = np.exp(-(x * x + y * y) / 3.0)
    model = StampDipoleModel(basis, image_shape=(7, 9))
    truth = np.array(
        [[-1.1, 0.3, 1.2, -0.2, 4.0], [-1.3, 0.2, 1.4, -0.3, 3.0]]
    )
    images = model.evaluate(truth)
    initial = truth + 0.01
    path = tmp_path / "stamp.npz"
    np.savez_compressed(
        path,
        candidate_id=np.array([9, 2]),
        images=images,
        initial=initial,
        stamp_basis=basis[None, :],
    )
    items, options = executor.plan_xfit_chunks(
        path,
        chunk_size=1,
        fit_options={
            "backend": "numpy",
            "model": "stamp",
            "max_evaluations": 8,
        },
    )
    monkeypatch.setattr(executor, "collect_gpu_identity", lambda _: {})
    worker = executor.create_xfit_worker(options)
    assert worker.config.use_finite_difference is True
    ordinary = fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="numpy",
        config=worker.config,
    )
    round_dir = tmp_path / "round"
    for item in items:
        worker.run_item(item, round_dir / "items" / item.item_id)
    executor.finalize_xfit_round(
        round_dir,
        [{"item_id": item.item_id, "status": "success"} for item in items],
        items=items,
        options=options,
    )
    chunks = [
        executor._read_result(item, round_dir, options)[1] for item in items
    ]
    np.testing.assert_allclose(
        np.concatenate([chunk["parameters"] for chunk in chunks]),
        ordinary.parameters,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk["evaluations"] for chunk in chunks]),
        ordinary.evaluations,
    )
