# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from cuphoton.core.cli import run_component
from cuphoton.xfit import GaussianDipoleModel, StampDipoleModel


def _write_input(path, *, mode="difference", model="gaussian"):
    if model == "gaussian":
        selected = GaussianDipoleModel((11, 15), dtype=np.float64)
        truth = np.asarray(
            [
                [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
                [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
            ]
        )
        initial = truth + np.asarray(
            [0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1]
        )
        extra = {}
    else:
        y, x = np.mgrid[-3:4, -2:3]
        basis = np.exp(-0.5 * (x * x + (y / 1.2) ** 2))
        selected = StampDipoleModel(basis, image_shape=(11, 15))
        truth = np.asarray([[-2.1, 0.6, 2.2, -0.4, 5.0]])
        initial = truth
        extra = {"stamp_basis": basis}
    images = selected.evaluate(truth, mode=mode)
    mask = np.ones(images.shape, dtype=bool)
    mask[..., 0, 0] = False
    np.savez_compressed(
        path,
        candidate_id=np.asarray(
            [f"candidate-{i}" for i in range(len(truth))]
        ),
        images=images,
        initial=initial,
        mask=mask,
        variance=np.full_like(images, 0.5),
        **extra,
    )


def _arguments(
    input_path, output_dir, *, backend, model="gaussian", mode="difference"
):
    return [
        "fit-dipoles",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--mode",
        mode,
        "--backend",
        backend,
    ]


@pytest.mark.parametrize("mode", ["difference", "split"])
def test_numba_mlir_cli_fits_and_writes_matching_artifacts(
    tmp_path, capsys, mode
):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("numba_cuda_mlir.cuda")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status != 100:  # cudaErrorNoDevice
            raise
        pytest.skip("CUDA device is unavailable")
    if count < 1:
        pytest.skip("CUDA device is unavailable")
    input_path = tmp_path / "synthetic.npz"
    _write_input(input_path, mode=mode)
    tables = {}
    for backend in ("cupy", "numba-cuda-mlir"):
        output = tmp_path / backend
        rc = run_component(
            "xfit", _arguments(input_path, output, backend=backend, mode=mode)
        )
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert captured.err == ""
        summary = json.loads(captured.out)
        assert summary["backend"] == backend
        assert summary["requested_backend"] == backend
        assert summary["metrics"]["converged_count"] == 2
        assert json.loads((output / "summary.json").read_text()) == summary
        config = yaml.safe_load(
            (output / "effective-config.yaml").read_text()
        )
        assert config["backend"] == backend
        assert config["mode"] == mode
        tables[backend] = pq.read_table(output / "fits.parquet").to_pydict()
        assert tables[backend]["backend"] == [backend, backend]

    reference = tables["cupy"]
    result = tables["numba-cuda-mlir"]
    for field in (
        "candidate_id",
        "status",
        "converged",
        "evaluations",
        "valid_pixel_count",
        "degrees_of_freedom",
        "uncertainty_valid",
        "uncertainty_reason",
    ):
        assert result[field] == reference[field]
    assert reference["uncertainty_valid"] == [True, True]
    assert min(reference["evaluations"]) > 1
    for field in (
        *GaussianDipoleModel.parameter_names,
        "chi_square",
        "residual_norm",
    ):
        np.testing.assert_allclose(
            result[field], reference[field], rtol=5e-12, atol=5e-12
        )
    with (
        np.load(
            tmp_path / "cupy" / "fit-arrays.npz", allow_pickle=False
        ) as baseline,
        np.load(
            tmp_path / "numba-cuda-mlir" / "fit-arrays.npz",
            allow_pickle=False,
        ) as fused,
    ):
        np.testing.assert_array_equal(
            fused["candidate_id"], baseline["candidate_id"]
        )
        for field in ("covariance", "residuals"):
            np.testing.assert_allclose(
                fused[field], baseline[field], rtol=5e-12, atol=5e-12
            )
        assert np.abs(baseline["covariance"]).max() > 1e-3


def test_numba_mlir_cli_reports_missing_dependency_before_cuda(
    tmp_path, capsys, monkeypatch
):
    import cuphoton.xfit.backend as backend_module

    input_path = tmp_path / "synthetic.npz"
    output = tmp_path / "fit"
    _write_input(input_path)
    monkeypatch.setitem(sys.modules, "numba_cuda_mlir", None)
    monkeypatch.setattr(
        backend_module,
        "_cupy_backend",
        lambda **kwargs: pytest.fail("missing MLIR dependency reached CUDA"),
    )
    rc = run_component(
        "xfit", _arguments(input_path, output, backend="numba-cuda-mlir")
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "Numba-CUDA-MLIR is not installed" in captured.err
    assert "numba-cuda-mlir extra" in captured.err
    assert not output.exists()


@pytest.mark.parametrize(
    ("model", "options", "message"),
    [
        ("stamp", [], "supports only the Gaussian model"),
        (
            "gaussian",
            ["--use-finite-difference"],
            "does not support finite-difference",
        ),
        ("gaussian", ["--max-evaluations", "0"], "must be at least 1"),
    ],
)
def test_numba_mlir_cli_rejects_invalid_requests_before_cuda(
    tmp_path, capsys, monkeypatch, model, options, message
):
    import cuphoton.xfit.api as api

    input_path = tmp_path / "synthetic.npz"
    output = tmp_path / "fit"
    _write_input(input_path, model=model)
    monkeypatch.setattr(
        api,
        "resolve_backend",
        lambda *args, **kwargs: pytest.fail("invalid request reached CUDA"),
    )
    rc = run_component(
        "xfit",
        _arguments(input_path, output, backend="numba-cuda-mlir", model=model)
        + options,
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert message in captured.err
    assert not output.exists()
