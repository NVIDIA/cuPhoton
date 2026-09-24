# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from dataclasses import fields, replace

import numpy as np
import pyarrow.parquet as pq
import pytest

from cuphoton.core.artifacts import file_sha256
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


@pytest.mark.parametrize(
    "model,finite_difference,message",
    [
        ("stamp", False, "only the Gaussian model"),
        ("gaussian", True, "does not support finite-difference fitting"),
    ],
)
def test_preflight_rejects_unsupported_cutile_settings(
    tmp_path, monkeypatch, model, finite_difference, message
):
    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid settings must fail before reading input")

    monkeypatch.setattr(executor, "load_xfit_dataset", unexpected_load)
    with pytest.raises(ValueError, match=message):
        executor.prepare_xfit_workload(
            input_path=tmp_path / "missing.npz",
            fit_options={
                "backend": "cutile",
                "model": model,
                "use_finite_difference": finite_difference,
            },
        )


@pytest.mark.parametrize("backend", ["auto", "numpy"])
def test_preflight_rejects_cpu_backend_before_input_reads(tmp_path, backend):
    with pytest.raises(ValueError, match="requires backend cupy or cutile"):
        executor.prepare_xfit_workload(
            input_path=tmp_path / "missing.npz",
            fit_options={"backend": backend},
        )


def test_chunk_size_is_part_of_configuration_identity(tmp_path):
    path = _input(tmp_path)
    _, first = executor.plan_xfit_chunks(path, chunk_size=1, fit_options={})
    _, second = executor.plan_xfit_chunks(path, chunk_size=2, fit_options={})
    assert first["chunk_size"] == 1
    assert second["chunk_size"] == 2
    assert first["configuration_sha256"] != second["configuration_sha256"]


def test_finalizer_retains_planning_input_and_still_checks_content(
    tmp_path, monkeypatch
):
    path = _input(tmp_path)
    loads = []
    original_load = executor.load_xfit_dataset

    def load(*args, **kwargs):
        loads.append(True)
        return original_load(*args, **kwargs)

    def cpu_fit(*args, **kwargs):
        result = fit_dipoles(*args, **{**kwargs, "backend": "numpy"})
        return replace(result, backend="cupy", device="cuda:0")

    monkeypatch.setattr(executor, "load_xfit_dataset", load)
    monkeypatch.setattr(executor, "fit_dipoles", cpu_fit)
    monkeypatch.setattr(executor, "collect_gpu_identity", lambda _: {})
    spec = executor.prepare_xfit_workload(
        input_path=path,
        chunk_size=2,
        fit_options={"backend": "cupy", "max_evaluations": 2},
    )
    worker = spec.worker_factory(spec.options_payload)
    for index in range(2):
        round_dir = tmp_path / f"round-{index}"
        records = []
        for item in spec.items:
            worker.run_item(item, round_dir / "items" / item.item_id)
            records.append({"item_id": item.item_id, "status": "success"})
        assert spec.finalize_round(round_dir, records)["candidate_count"] == 3
    assert len(loads) == 2  # One planning load and one worker load.
    before = path.stat()
    changed = bytearray(path.read_bytes())
    changed[-1] ^= 1
    path.write_bytes(changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="archive changed after planning"):
        spec.finalize_round(tmp_path / "unused", records)
    assert len(loads) == 2
    worker.close()


def test_preflight_accepts_analytic_gaussian_cutile(tmp_path):
    spec = executor.prepare_xfit_workload(
        input_path=_input(tmp_path), fit_options={"backend": "cutile"}
    )
    assert spec.backend == "cutile"


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
    "defect,message",
    [
        ("missing", "one successful result"),
        ("duplicate", "one successful result"),
        ("failed", "one successful result"),
        ("corrupt", "checksum mismatch"),
        ("checksum", "checksum mismatch"),
        ("wrong_candidates", "candidate identity mismatch"),
        ("wrong_summary", "summary identity mismatch"),
        ("wrong_model", "configuration mismatch"),
    ],
)
def test_merge_rejects_incomplete_or_changed_scientific_results(
    tmp_path, monkeypatch, defect, message
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
    elif defect == "corrupt":
        (round_dir / "items" / items[0].item_id / "result.npz").write_bytes(
            b"changed"
        )
    else:
        item_dir = round_dir / "items" / items[0].item_id
        summary_path = item_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if defect == "wrong_summary":
            summary["configuration_sha256"] = "changed"
        elif defect == "wrong_model":
            summary["result_metadata"]["model"] = "stamp"
        else:
            archive = item_dir / "result.npz"
            with np.load(archive, allow_pickle=False) as loaded:
                values = dict(loaded)
            if defect == "checksum":
                values["parameters"][0, 0] += 1
            else:
                values["candidate_id"] = values["candidate_id"][::-1]
            np.savez_compressed(archive, **values)
            if defect != "checksum":
                summary["result_sha256"] = file_sha256(archive)
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=message):
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
