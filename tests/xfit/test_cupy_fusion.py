# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import numpy as np
import pytest
import yaml

import cuphoton.xfit.api as xfit_api
from cuphoton.core.cli import run_component
from cuphoton.xfit import (
    GaussianDipoleModel,
    LMConfig,
    StampDipoleModel,
    fit_dipoles,
    fit_dipoles_device,
)


def _require_cupy():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    if count < 1:
        pytest.skip("test requires a visible CUDA device")
    return cp


def _truth(dtype=np.float64):
    return np.asarray(
        [
            [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
            [4.2, 1.4, 2.0, 0.1, -1.7, 0.8, 1.9, -0.9],
        ],
        dtype=dtype,
    )


def _assert_numeric(actual, expected, dtype):
    # Match the existing same-backend Gaussian normal-equation tolerances.
    tolerance = 3e-5 if np.dtype(dtype) == np.dtype(np.float32) else 5e-12
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )


def _assert_result(actual, expected):
    for field in fields(expected):
        observed = getattr(actual, field.name)
        reference = getattr(expected, field.name)
        if isinstance(reference, np.ndarray):
            if reference.dtype.kind == "f":
                _assert_numeric(observed, reference, expected.dtype)
            else:
                np.testing.assert_array_equal(
                    observed, reference, err_msg=field.name
                )
        else:
            assert observed == reference, field.name


def _no_cuda(*args, **kwargs):
    pytest.fail("unsupported fusion settings reached CUDA/backend discovery")


@pytest.mark.parametrize("backend", ["numpy", "auto", "cutile", "native"])
def test_fusion_rejects_unsupported_backend_before_cuda(monkeypatch, backend):
    monkeypatch.setattr(xfit_api, "resolve_backend", _no_cuda)
    with pytest.raises(ValueError, match="fusion requires backend='cupy'"):
        fit_dipoles(
            np.zeros((1, 5, 7)),
            model="gaussian",
            backend=backend,
            fusion=True,
        )


@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("case", ["stamp", "finite_difference", "nonboolean"])
def test_fusion_rejects_unsupported_configuration_before_cuda(
    monkeypatch, device, case
):
    model = "gaussian"
    config = None
    fusion = True
    exception = ValueError
    match = "finite-difference"
    if case == "stamp":
        model = StampDipoleModel(np.ones((3, 3)), image_shape=(5, 7))
        match = "Gaussian model"
    elif case == "finite_difference":
        config = LMConfig(use_finite_difference=True)
    else:
        fusion = 1
        exception = TypeError
        match = "fusion must be boolean"
    monkeypatch.setattr(xfit_api, "resolve_backend", _no_cuda)
    monkeypatch.setattr(xfit_api, "_load_cupy", _no_cuda)
    kwargs = {} if device else {"backend": "cupy"}
    with pytest.raises(exception, match=match):
        (fit_dipoles_device if device else fit_dipoles)(
            np.zeros((1, 5, 7)),
            model=model,
            config=config,
            fusion=fusion,
            **kwargs,
        )


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--backend", "numpy"], "fusion requires backend='cupy'"),
        (["--backend", "cupy", "--model", "stamp"], "Gaussian model"),
        (
            ["--backend", "cupy", "--use-finite-difference"],
            "finite-difference",
        ),
    ],
)
def test_fusion_cli_rejects_options_before_loading_or_backend_probe(
    tmp_path, capsys, monkeypatch, arguments, message
):
    import cuphoton.xfit.backend as backend
    from cuphoton.xfit.commands import FitDipolesCommand

    monkeypatch.setattr(backend, "resolve_backend", _no_cuda)
    monkeypatch.setattr(xfit_api, "resolve_backend", _no_cuda)
    monkeypatch.setattr(FitDipolesCommand, "_load_dataset", _no_cuda)
    input_path = tmp_path / "unread-manifest.json"
    input_path.write_text("{}")
    rc = run_component(
        "xfit",
        [
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--model",
            "gaussian",
            "--fusion",
            *arguments,
        ],
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert message in captured.err
    assert not (tmp_path / "output").exists()


def _reference_callbacks(cp, parameters, images, weights, indices, mode):
    physical = parameters.copy()
    physical[:, 1:3] = cp.exp(parameters[:, 1:3])
    model = GaussianDipoleModel(
        images.shape[-2:], backend="cupy", dtype=images.dtype
    )
    prediction = model._evaluate_positive_unchecked(physical, mode=mode)
    residual = ((prediction - images[indices]) * weights[indices]).reshape(
        parameters.shape[0], -1
    )
    jacobian = model._jacobian_positive_unchecked(physical, mode=mode)
    chain_shape = (parameters.shape[0], 2) + (1,) * (jacobian.ndim - 2)
    jacobian[:, 1:3] *= physical[:, 1:3].reshape(chain_shape)
    jacobian *= weights[indices, None, ...]
    return residual, jacobian.reshape(parameters.shape[0], 8, -1)


@pytest.mark.parametrize("shape", [(11, 15), (8, 10)])
@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_fusion_callbacks_preserve_weighting_indices_and_layout(
    shape, mode, dtype
):
    cp = _require_cupy()
    from cuphoton.xfit._cupy_fused import GaussianFusedProblem

    rng = np.random.default_rng(1219)
    truth = _truth(dtype)
    truth[2, 0] *= -1
    model = GaussianDipoleModel(shape, dtype=dtype)
    prediction = model.evaluate(truth, mode=mode)
    images = cp.asarray(
        prediction + rng.normal(scale=0.2, size=prediction.shape), dtype=dtype
    )[..., ::-1]
    weights = cp.asarray(
        rng.uniform(0.25, 2.0, size=prediction.shape), dtype=dtype
    )[..., ::-1]
    weights[..., ::3, ::2] = 0
    indices = cp.asarray([2, 0], dtype=cp.int64)
    parameters = cp.asarray(truth)[indices]
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    problem = GaussianFusedProblem(
        cp, images, weights, image_shape=shape, mode=mode
    )
    expected, jacobian = _reference_callbacks(
        cp, parameters, images, weights, indices, mode
    )
    actual = problem.residual(parameters, indices=indices)
    _assert_numeric(cp.asnumpy(actual), cp.asnumpy(expected), dtype)

    # A supplied residual must be consumed, not silently recomputed.
    residual = expected + dtype(0.03)
    gradient, hessian = problem.normal_equations(
        parameters, residual, indices=indices
    )
    expected_gradient = cp.einsum("knm,km->kn", jacobian, residual)
    expected_hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)
    _assert_numeric(
        cp.asnumpy(gradient), cp.asnumpy(expected_gradient), dtype
    )
    _assert_numeric(cp.asnumpy(hessian), cp.asnumpy(expected_hessian), dtype)
    np.testing.assert_array_equal(
        cp.asnumpy(hessian), cp.asnumpy(hessian).swapaxes(1, 2)
    )


def test_cupy_fusion_preserves_cancelling_hessian_terms():
    cp = _require_cupy()
    from cuphoton.xfit._cupy_fused import GaussianFusedProblem

    shape = (51, 51)
    parameters = cp.asarray(
        [[2**20, 8.0, 4.0, 0.0, -0.25, 0.0, 0.25, 0.0]],
        dtype=cp.float64,
    )
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    images = cp.zeros((1, *shape), dtype=cp.float64)
    coordinate = cp.arange(51, dtype=cp.float64) - 25
    radius_squared = coordinate[:, None] ** 2 + coordinate[None, :] ** 2
    weights = (1 + radius_squared[None, ...] / 4096).copy()
    indices = cp.asarray([0], dtype=cp.int64)
    _, jacobian = _reference_callbacks(
        cp, parameters, images, weights, indices, "difference"
    )
    residual = cp.ones((1, 51 * 51), dtype=cp.float64)
    expected_gradient = cp.einsum("knm,km->kn", jacobian, residual)
    expected_hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)

    # Symmetry makes these bright-lobe derivative products cancel. A small
    # relative error in each partial sum can change the near-zero Hessian.
    absolute_sum = float(
        cp.sum(cp.abs(jacobian[0, 4] * jacobian[0, 5])).item()
    )
    assert absolute_sum > 1e8
    assert abs(float(expected_hessian[0, 4, 5].item())) < 1e-12 * absolute_sum

    problem = GaussianFusedProblem(
        cp, images, weights, image_shape=shape, mode="difference"
    )
    gradient, hessian = problem.normal_equations(
        parameters, residual, indices=indices
    )
    _assert_numeric(
        cp.asnumpy(gradient), cp.asnumpy(expected_gradient), np.float64
    )
    _assert_numeric(
        cp.asnumpy(hessian), cp.asnumpy(expected_hessian), np.float64
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_fusion_extreme_widths_preserve_nonfinite_results(dtype):
    cp = _require_cupy()
    from cuphoton.xfit._cupy_fused import GaussianFusedProblem

    parameters = cp.asarray(_truth(dtype))
    parameters[:, 3] = 0
    parameters[:, 4:8] = cp.asarray([-1, 0, 1, 0], dtype=dtype)
    small = 1e-16 if dtype is np.float32 else 1e-120
    parameters[:, 1:3] = cp.asarray(
        [[-1000, -1000], [1000, 1000], [np.log(small), np.log(small)]],
        dtype=dtype,
    )
    images = cp.zeros((3, 3, 5, 7), dtype=dtype)
    weights = cp.ones_like(images)
    weights[0] = 0
    indices = cp.arange(3, dtype=cp.int64)
    problem = GaussianFusedProblem(
        cp, images, weights, image_shape=(5, 7), mode="split"
    )
    residual, jacobian = _reference_callbacks(
        cp, parameters, images, weights, indices, "split"
    )
    actual = problem.residual(parameters, indices=indices)
    _assert_numeric(cp.asnumpy(actual), cp.asnumpy(residual), dtype)
    gradient, hessian = problem.normal_equations(
        parameters, residual, indices=indices
    )
    _assert_numeric(
        cp.asnumpy(gradient),
        cp.asnumpy(cp.einsum("knm,km->kn", jacobian, residual)),
        dtype,
    )
    _assert_numeric(
        cp.asnumpy(hessian),
        cp.asnumpy(cp.einsum("knm,kpm->knp", jacobian, jacobian)),
        dtype,
    )


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_fusion_full_results_preserve_compacted_rows(mode, dtype):
    _require_cupy()
    truth = _truth(dtype)
    model = GaussianDipoleModel(
        (11, 15), split_weights=(1.0, 0.25, 1.75), dtype=dtype
    )
    images = model.evaluate(truth, mode=mode)
    initial = truth + np.asarray(
        [0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1], dtype=dtype
    )
    initial[0] = truth[0]
    mask = np.ones(images.shape, dtype=bool)
    mask[..., 0, :3] = False
    images[..., 0, :3] = np.nan
    variance = np.linspace(0.5, 1.5, images.size, dtype=dtype).reshape(
        images.shape
    )
    variance[~mask] = np.nan
    kwargs = dict(
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        mode=mode,
        backend="cupy",
    )
    expected = fit_dipoles(images, **kwargs)
    actual = fit_dipoles(images, fusion=True, **kwargs)
    assert expected.evaluations[0] == 1
    assert np.all(expected.evaluations[1:] > 1)
    assert expected.uncertainty_valid.all()
    assert np.max(np.abs(expected.covariance)) > 1e-3
    _assert_result(actual, expected)


@pytest.mark.parametrize("case", ["masked", "rank_deficient", "exhausted"])
def test_cupy_fusion_preserves_degenerate_and_budget_results(case):
    _require_cupy()
    truth = _truth()[:1]
    model = GaussianDipoleModel((5, 7), split_weights=(1.0, 0.0, 0.5))
    if case == "rank_deficient":
        truth[:, 0] = 0
    images = model.evaluate(truth, mode="split")
    mask = np.ones(images.shape, dtype=bool)
    if case == "masked":
        mask[:] = False
        images[:] = np.nan
    initial = truth.copy()
    config = None
    if case == "exhausted":
        initial[:, 0] += 0.2
        config = LMConfig(max_evaluations=1)
    kwargs = dict(
        model=model,
        initial=initial,
        mask=mask,
        mode="split",
        config=config,
        backend="cupy",
    )
    expected = fit_dipoles(images, **kwargs)
    actual = fit_dipoles(images, fusion=True, **kwargs)
    assert not actual.uncertainty_valid.any()
    _assert_result(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "amplitude"), [(np.float32, 1e25), (np.float64, 1e200)]
)
@pytest.mark.parametrize("nonzero_residual", [False, True])
def test_cupy_fusion_preserves_finite_jacobian_overflow_results(
    dtype, amplitude, nonzero_residual
):
    cp = _require_cupy()
    model = GaussianDipoleModel((5, 7), backend="cupy", dtype=dtype)
    # Coincident stars give an exactly zero difference while retaining finite,
    # large position derivatives. Only the middle row overflows J.T @ J.
    initial = np.asarray(
        [
            [5, 1, 1, 0, 0, 0, 0, 0],
            [amplitude, 1, 1, 0, 0, 0, 0, 0],
            [3, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=dtype,
    )
    images = np.zeros((3, 5, 7), dtype=dtype)
    if nonzero_residual:
        # A noncentral pixel also overflows J.T @ r, exercising the solver
        # after its gradient stopping check rather than only at zero residual.
        images[1, 1, 2] = amplitude
    jacobian = model.jacobian(cp.asarray(initial)).reshape(3, 8, -1)
    assert bool(cp.isfinite(jacobian).all())
    hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)
    np.testing.assert_array_equal(
        cp.asnumpy(cp.isfinite(hessian).all(axis=(1, 2))),
        [True, False, True],
    )

    kwargs = dict(model=model, initial=initial, backend="cupy")
    expected = fit_dipoles(images, **kwargs)
    actual = fit_dipoles(images, fusion=True, **kwargs)
    assert expected.status.tolist() == [
        "converged_g_tol",
        "singular" if nonzero_residual else "converged_g_tol",
        "converged_g_tol",
    ]
    np.testing.assert_array_equal(expected.evaluations, [1, 1, 1])
    np.testing.assert_array_equal(actual.parameters, expected.parameters)
    _assert_result(actual, expected)


def test_cupy_fusion_repeated_calls_and_independent_worker_streams():
    cp = _require_cupy()
    device = cp.cuda.runtime.getDevice()

    def run_worker(offset):
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True):
            saved = []
            for shift in (offset, offset + 0.4):
                truth = _truth()[:2].copy()
                truth[:, 4:8] += shift
                model = GaussianDipoleModel((9, 13))
                images = cp.asarray(model.evaluate(truth))
                initial = cp.asarray(truth + 0.025)
                expected = fit_dipoles(
                    images, model=model, initial=initial, backend="cupy"
                )
                actual = fit_dipoles_device(
                    images, model=model, initial=initial, fusion=True
                )
                snapshot = cp.asnumpy(actual.parameters)
                _assert_numeric(snapshot, expected.parameters, np.float64)
                saved.append((actual, snapshot))
            for actual, snapshot in saved:
                np.testing.assert_array_equal(
                    cp.asnumpy(actual.parameters), snapshot
                )
            cp.cuda.get_current_stream().synchronize()
            return saved[-1][1]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_worker, [0.0, 0.3]))
    assert not np.array_equal(results[0], results[1])


def test_cupy_fusion_cli_runs_selected_mode_and_persists_outputs(
    tmp_path, capsys
):
    _require_cupy()
    truth = _truth()[:2]
    model = GaussianDipoleModel((9, 13))
    images = model.evaluate(truth)
    initial = truth + 0.025
    input_path = tmp_path / "input.npz"
    np.savez(
        input_path,
        candidate_id=np.asarray(["first", "second"]),
        images=images,
        initial=initial,
        variance=np.ones_like(images),
    )
    output = tmp_path / "fused"
    rc = run_component(
        "xfit",
        [
            "fit-dipoles",
            "--input",
            str(input_path),
            "--output-dir",
            str(output),
            "--model",
            "gaussian",
            "--backend",
            "cupy",
            "--fusion",
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    configuration = yaml.safe_load(
        (output / "effective-config.yaml").read_text()
    )
    assert configuration["fusion"] is True
    assert configuration["backend"] == "cupy"
    expected = fit_dipoles(
        images,
        model=model,
        initial=initial,
        variance=np.ones_like(images),
        backend="cupy",
    )
    with np.load(output / "fit-arrays.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(
            result["candidate_id"], ["first", "second"]
        )
        _assert_numeric(result["residuals"], expected.residuals, np.float64)
        _assert_numeric(result["covariance"], expected.covariance, np.float64)
