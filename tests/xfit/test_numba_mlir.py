# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scientific and stream-lifetime coverage for the optional MLIR backend."""

from __future__ import annotations

import gc
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import ModuleType

import numpy as np
import pytest

from cuphoton.xfit import (
    GaussianDipoleModel,
    LMConfig,
    LMStatus,
    fit_dipoles,
    fit_dipoles_device,
)

_BACKEND = "numba-cuda-mlir"


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    pytest.importorskip("numba_cuda_mlir")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status != 100:  # cudaErrorNoDevice
            raise
        pytest.skip("CUDA device is unavailable")
    if count < 1:
        pytest.skip("CUDA device is unavailable")
    return cp


def test_dispatcher_creation_ignores_partial_experimental_import(monkeypatch):
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    from cuphoton.xfit._numba_mlir import _kernels

    # Load compiler prerequisites through the real public JIT, without
    # compiling a signature or initializing a device.
    _kernels(cuda, np.float64, 7, 9)
    name = "numba_cuda_mlir.cuda.experimental"
    # Another thread can publish this module before defining its helpers.
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for kernel in _kernels(cuda, np.float64, 7, 9):
        assert not kernel.signatures


def _truth(dtype=np.float64):
    return np.asarray(
        [
            [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
            [4.2, 1.4, 2.0, 0.1, -1.7, 0.8, 1.9, -0.9],
        ],
        dtype=dtype,
    )


def _tolerance(dtype):
    return 3.0e-5 if np.dtype(dtype) == np.dtype(np.float32) else 5.0e-12


def _assert_fit_equal(actual, expected, dtype):
    for name in (
        "status",
        "converged",
        "evaluations",
        "valid_pixel_count",
        "degrees_of_freedom",
        "uncertainty_valid",
    ):
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(expected, name), err_msg=name
        )
    assert actual.uncertainty_reason == expected.uncertainty_reason
    tolerance = _tolerance(dtype)
    for name in (
        "parameters",
        "residuals",
        "residual_norm",
        "chi_square",
        "valid_pixel_fraction",
        "null_chi_square",
        "delta_chi_square",
        "fractional_null_improvement",
        "reduced_chi_square",
        "covariance",
        "standard_errors",
    ):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
            err_msg=name,
        )


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_fit_preserves_scientific_outputs(gpu, mode, dtype):
    truth = _truth(dtype)[:2]
    model = GaussianDipoleModel((11, 15), dtype=dtype)
    images = model.evaluate(truth, mode=mode)
    initial = truth + np.asarray(
        [0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1], dtype=dtype
    )
    # Explicit variance keeps covariance appreciable for noiseless images.
    options = dict(
        model=model,
        initial=initial,
        variance=np.full(images.shape, 0.5, dtype=dtype),
        mode=mode,
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend=_BACKEND, **options)

    assert actual.backend == _BACKEND
    assert actual.converged.all()
    assert actual.uncertainty_valid.all()
    assert np.abs(expected.covariance).max() > 1.0e-3
    _assert_fit_equal(actual, expected, dtype)


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_boundary_preserves_weights_and_compacted_indices(
    gpu, mode, dtype
):
    from cuphoton.xfit._numba_mlir import GaussianWorkspace

    cp = gpu
    rng = np.random.default_rng(1219)
    physical = cp.asarray(_truth(dtype))
    parameters = physical.copy()
    parameters[:, 1:3] = cp.log(physical[:, 1:3])
    model = GaussianDipoleModel((8, 11), backend="cupy", dtype=dtype)
    prediction = model.evaluate(physical, mode=mode)
    images = prediction + cp.asarray(
        rng.normal(scale=0.2, size=prediction.shape), dtype=dtype
    )
    weights = cp.asarray(
        rng.uniform(0.25, 2.0, size=prediction.shape), dtype=dtype
    )
    weights.reshape(3, -1)[:, ::7] = 0
    residuals = ((prediction - images) * weights).reshape(3, -1)
    jacobian = model.jacobian(physical, mode=mode)
    chain_shape = (3, 2) + (1,) * (jacobian.ndim - 2)
    jacobian[:, 1:3] *= physical[:, 1:3].reshape(chain_shape)
    jacobian = (jacobian * weights[:, None, ...]).reshape(3, 8, -1)
    tolerance = _tolerance(dtype)

    # FP64 inputs and sums are deliberately not exactly representable in
    # FP32: reductions must not narrow their intermediate values.
    with GaussianWorkspace(
        images, weights, image_shape=(8, 11), mode=mode
    ) as workspace:
        for rows in ((0, 1, 2), (2, 0), (1,)):
            indices = cp.asarray(rows, dtype=cp.int64)
            active = parameters[indices]
            actual_residual = workspace.residual(active, indices=indices)
            gradient, hessian = workspace.normal_equations(
                active, actual_residual, indices=indices
            )
            expected_gradient = cp.einsum(
                "knm,km->kn", jacobian[indices], residuals[indices]
            )
            expected_hessian = cp.einsum(
                "knm,kpm->knp", jacobian[indices], jacobian[indices]
            )
            for actual, expected in (
                (actual_residual, residuals[indices]),
                (gradient, expected_gradient),
                (hessian, expected_hessian),
            ):
                np.testing.assert_allclose(
                    cp.asnumpy(actual),
                    cp.asnumpy(expected),
                    rtol=tolerance,
                    atol=tolerance,
                )


def test_fused_boundary_preserves_cancelling_hessian_terms(gpu):
    from cuphoton.xfit._numba_mlir import GaussianWorkspace

    cp = gpu
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

    physical = parameters.copy()
    physical[:, 1:3] = cp.exp(parameters[:, 1:3])
    model = GaussianDipoleModel(shape, backend="cupy", dtype=np.float64)
    jacobian = model.jacobian(physical, mode="difference")
    jacobian[:, 1:3] *= physical[:, 1:3, None, None]
    jacobian = (jacobian * weights[:, None, ...]).reshape(1, 8, -1)
    residual = cp.ones((1, 51 * 51), dtype=cp.float64)
    expected_gradient = cp.einsum("knm,km->kn", jacobian, residual)
    expected_hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)

    # Symmetry cancels products of bright-lobe position derivatives.
    absolute_sum = float(
        cp.sum(cp.abs(jacobian[0, 4] * jacobian[0, 5])).item()
    )
    assert absolute_sum > 1e8
    assert abs(float(expected_hessian[0, 4, 5].item())) < 1e-12 * absolute_sum

    with GaussianWorkspace(
        images, weights, image_shape=shape, mode="difference"
    ) as workspace:
        gradient, hessian = workspace.normal_equations(
            parameters, residual, indices=indices
        )
        for actual, expected in (
            (gradient, expected_gradient),
            (hessian, expected_hessian),
        ):
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(
                cp.asnumpy(actual),
                cp.asnumpy(expected),
                rtol=_tolerance(np.float64),
                atol=_tolerance(np.float64),
                equal_nan=True,
            )


def test_fit_masks_variance_and_independent_termination(gpu):
    model = GaussianDipoleModel(
        (11, 15), split_weights=(1.0, 0.25, 1.75), dtype=np.float64
    )
    truth = _truth()
    images = model.evaluate(truth, mode="split")
    initial = truth + np.asarray([0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1])
    initial[0] = truth[0]
    mask = np.ones(images.shape, dtype=bool)
    mask[0, 0, 0, :3] = False
    mask[1, 1, 2:4, 5:8] = False
    mask[2, 2, 7:, 10:] = False
    y, x = np.mgrid[:11, :15]
    variance = np.empty(images.shape)
    for candidate in range(3):
        for plane in range(3):
            variance[candidate, plane] = (
                0.75 + 0.03 * x + 0.05 * y + 0.2 * candidate + 0.1 * plane
            )
    images[~mask] = np.nan
    variance[~mask] = np.nan
    options = dict(
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        mode="split",
    )
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend=_BACKEND, **options)

    assert actual.evaluations[0] == 1
    assert np.all(actual.evaluations[1:] > 1)
    _assert_fit_equal(actual, expected, np.float64)


@pytest.mark.parametrize("failure", ["rank", "budget"])
def test_fit_preserves_failure_diagnostics(gpu, failure):
    model = GaussianDipoleModel((7, 9))
    truth = _truth()[:1]
    if failure == "rank":
        truth[:, 0] = 0
        initial = truth.copy()
        config = LMConfig()
    else:
        initial = truth + np.asarray(
            [0.3, 0.1, -0.1, 0.05, 0.1, -0.1, -0.1, 0.1]
        )
        config = LMConfig(max_evaluations=1)
    images = model.evaluate(truth)
    options = dict(model=model, initial=initial, config=config)
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend=_BACKEND, **options)

    assert not actual.uncertainty_valid.any()
    assert np.isnan(actual.covariance).all()
    if failure == "budget":
        assert actual.status.tolist() == ["max_evaluations"]
        assert actual.evaluations.tolist() == [1]
    else:
        assert actual.uncertainty_reason == ("rank-deficient Jacobian",)
    _assert_fit_equal(actual, expected, np.float64)


@pytest.mark.parametrize(
    "log_width", [1000.0, -1000.0, np.inf, -np.inf, np.nan]
)
def test_nonfinite_trial_widths_produce_invalid_residuals(gpu, log_width):
    from cuphoton.xfit._numba_mlir import GaussianWorkspace

    cp = gpu
    model = GaussianDipoleModel((7, 9), backend="cupy")
    physical = cp.asarray(_truth()[:1])
    parameters = physical.copy()
    parameters[:, 1:3] = log_width
    physical[:, 1:3] = cp.exp(parameters[:, 1:3])
    images = cp.zeros((1, 7, 9), dtype=cp.float64)
    weights = cp.ones_like(images)
    weights[:, 0, 0] = 0
    expected = (
        model._evaluate_positive_unchecked(physical, mode="difference")
        * weights
    ).reshape(1, -1)
    with GaussianWorkspace(
        images, weights, image_shape=(7, 9), mode="difference"
    ) as workspace:
        indices = cp.asarray([0], dtype=cp.int64)
        actual = workspace.residual(parameters, indices=indices)
        gradient, hessian = workspace.normal_equations(
            parameters, actual, indices=indices
        )
        np.testing.assert_array_equal(
            cp.asnumpy(actual), cp.asnumpy(expected)
        )
        assert not cp.isfinite(actual).any().item()
        assert not cp.isfinite(gradient).any().item()
        assert not cp.isfinite(hessian).any().item()


@pytest.mark.parametrize("mode", ["difference", "split"])
@pytest.mark.parametrize(
    ("dtype", "sigma"), [(np.float32, 1e-16), (np.float64, 1e-110)]
)
def test_finite_tiny_widths_preserve_sigma_cube_semantics(
    gpu, mode, dtype, sigma
):
    from cuphoton.xfit._numba_mlir import GaussianWorkspace

    cp = gpu
    initial = np.asarray(
        [[1, sigma, sigma, 0, -0.25, 0.25, 0.25, -0.25]], dtype=dtype
    )
    parameters = cp.asarray(initial)
    parameters[:, 1:3] = cp.log(parameters[:, 1:3])
    physical = parameters.copy()
    physical[:, 1:3] = cp.exp(parameters[:, 1:3])
    model = GaussianDipoleModel((2, 2), backend="cupy", dtype=dtype)
    prediction = model.evaluate(physical, mode=mode)
    images = cp.zeros_like(prediction)
    mask = cp.ones(images.shape, dtype=cp.bool_)
    mask.reshape(1, -1)[:, 0] = False
    weights = mask.astype(dtype)
    if mode == "split":
        weights *= cp.asarray(model.split_weights, dtype=dtype).reshape(
            1, 3, 1, 1
        )
    residuals = (prediction * weights).reshape(1, -1)
    jacobian = model.jacobian(physical, mode=mode)
    chain_shape = (1, 2) + (1,) * (jacobian.ndim - 2)
    jacobian[:, 1:3] *= physical[:, 1:3].reshape(chain_shape)
    jacobian = (jacobian * weights[:, None, ...]).reshape(1, 8, -1)
    gradient = cp.einsum("knm,km->kn", jacobian, residuals)
    hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)

    # Finite positive widths yield finite residuals, but sigma**3 rounds to
    # zero. Preserve the reference's 0/0 derivatives, including masked pixels;
    # cancelling sigma algebraically would change the solver's failure status.
    assert cp.isfinite(physical).all().item()
    assert cp.all(physical[:, 1:3] ** 3 == 0).item()
    assert cp.isfinite(residuals).all().item()
    assert not cp.isfinite(jacobian).all().item()
    with GaussianWorkspace(
        images, weights, image_shape=(2, 2), mode=mode
    ) as workspace:
        indices = cp.asarray([0], dtype=cp.int64)
        actual_residuals = workspace.residual(parameters, indices=indices)
        actual_gradient, actual_hessian = workspace.normal_equations(
            parameters, actual_residuals, indices=indices
        )
        for actual, expected in (
            (actual_residuals, residuals),
            (actual_gradient, gradient),
            (actual_hessian, hessian),
        ):
            np.testing.assert_array_equal(
                cp.asnumpy(actual), cp.asnumpy(expected)
            )

    options = dict(model=model, initial=initial, mask=mask, mode=mode)
    expected_fit = fit_dipoles(images, backend="cupy", **options)
    actual_fit = fit_dipoles(images, backend=_BACKEND, **options)
    assert expected_fit.status.tolist() == ["invalid_residual"]
    _assert_fit_equal(actual_fit, expected_fit, dtype)


@pytest.mark.parametrize(
    ("dtype", "amplitude"), [(np.float32, 1e25), (np.float64, 1e200)]
)
@pytest.mark.parametrize("nonzero_residual", [False, True])
def test_finite_jacobian_overflow_preserves_solver_outcomes(
    gpu, dtype, amplitude, nonzero_residual
):
    cp = gpu
    model = GaussianDipoleModel((5, 7), backend="cupy", dtype=dtype)
    # Coincident stars have zero difference but finite position derivatives.
    # Only the middle row overflows the normal equations.
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
        images[1, 1, 2] = amplitude
    jacobian = model.jacobian(cp.asarray(initial)).reshape(3, 8, -1)
    assert cp.isfinite(jacobian).all().item()
    hessian = cp.einsum("knm,kpm->knp", jacobian, jacobian)
    np.testing.assert_array_equal(
        cp.asnumpy(cp.isfinite(hessian).all(axis=(1, 2))),
        [True, False, True],
    )

    options = dict(model=model, initial=initial)
    expected = fit_dipoles(images, backend="cupy", **options)
    actual = fit_dipoles(images, backend=_BACKEND, **options)
    assert expected.status.tolist() == [
        "converged_g_tol",
        "singular" if nonzero_residual else "converged_g_tol",
        "converged_g_tol",
    ]
    np.testing.assert_array_equal(expected.evaluations, [1, 1, 1])
    np.testing.assert_array_equal(actual.parameters, expected.parameters)
    _assert_fit_equal(actual, expected, dtype)


def test_device_fit_retains_results_after_repeated_concurrent_calls(gpu):
    cp = gpu
    workers = 4
    barrier = Barrier(workers)
    model = GaussianDipoleModel((11, 15))
    truth = _truth()[:2]
    images = model.evaluate(truth)
    initial = truth + np.asarray([0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1])
    cases = {}
    for worker in range(workers):
        for iteration in range(3):
            scale = 1.0 + 0.1 * worker + 0.05 * iteration
            scaled_initial = initial.copy()
            scaled_initial[:, 0] *= scale
            options = dict(
                initial=scaled_initial,
                variance=np.ones_like(images) * scale**2,
            )
            expected = fit_dipoles(
                images * scale, model="gaussian", backend=_BACKEND, **options
            )
            cases[worker, iteration] = (images * scale, options, expected)
    device_id = cp.cuda.runtime.getDevice()

    def fit_on_owned_stream(worker):
        with cp.cuda.Device(device_id), cp.cuda.Stream(non_blocking=True):
            barrier.wait(timeout=120)
            results = []
            for iteration in range(3):
                # Different amplitudes expose accidental cross-worker or
                # cross-call reuse of model observations and scratch arrays.
                scaled_images, options, _ = cases[worker, iteration]
                result = fit_dipoles_device(
                    cp.asarray(scaled_images),
                    model="gaussian",
                    initial=cp.asarray(options["initial"]),
                    variance=cp.asarray(options["variance"]),
                    backend=_BACKEND,
                )
                results.append(result)
            gc.collect()
            portable = []
            for result in results:
                assert result.backend == _BACKEND
                assert isinstance(result.parameters, cp.ndarray)
                portable.append(
                    (
                        cp.asnumpy(result.parameters),
                        cp.asnumpy(result.status_codes),
                        cp.asnumpy(result.evaluations),
                        cp.asnumpy(result.covariance),
                    )
                )
            return portable

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fit_on_owned_stream, range(workers)))
    for worker, worker_results in enumerate(results):
        for iteration, result in enumerate(worker_results):
            parameters, status_codes, evaluations, covariance = result
            _, _, expected = cases[worker, iteration]
            # Scheduling alone must not change any numerical result.
            np.testing.assert_array_equal(parameters, expected.parameters)
            np.testing.assert_array_equal(covariance, expected.covariance)
            np.testing.assert_array_equal(evaluations, expected.evaluations)
            status = [
                LMStatus(int(code)).name.lower() for code in status_codes
            ]
            assert status == expected.status.tolist()


def test_workspace_orders_delayed_producer_and_retains_input_owners(gpu):
    from cuphoton.xfit._numba_mlir import GaussianWorkspace

    cp = gpu
    shape = (8, 11)
    model = GaussianDipoleModel(shape)
    truth = _truth()[:2]
    expected = model.evaluate(truth)
    jacobian = model.jacobian(truth)
    jacobian[:, 1:3] *= truth[:, 1:3, None, None]
    jacobian = jacobian.reshape(2, 8, -1)
    expected_gradient = np.einsum(
        "knm,km->kn", jacobian, expected.reshape(2, -1)
    )
    expected_hessian = np.einsum("knm,kpm->knp", jacobian, jacobian)
    log_parameters = truth.copy()
    log_parameters[:, 1:3] = np.log(log_parameters[:, 1:3])
    delay = cp.RawKernel(
        r"""
        extern "C" __global__ void delay(unsigned int repetitions) {
            for (unsigned int i = 0; i < repetitions; ++i) {
                __nanosleep(1000000);
            }
        }
        """,
        "delay",
    )
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    with consumer:
        delay((1,), (1,), (np.uint32(0),))
        with GaussianWorkspace(
            cp.zeros_like(cp.asarray(expected)),
            cp.ones_like(cp.asarray(expected)),
            image_shape=shape,
            mode="difference",
        ) as warm:
            parameters = cp.asarray(log_parameters)
            indices = cp.arange(2, dtype=cp.int64)
            residuals = warm.residual(parameters, indices=indices)
            warm.normal_equations(parameters, residuals, indices=indices)

    with producer:
        source_parameters = cp.asarray(log_parameters)
        source_indices = cp.arange(2, dtype=cp.int64)
    producer.synchronize()
    for _ in range(3):
        with producer:
            images = cp.empty(expected.shape, dtype=cp.float64)
            weights = cp.empty_like(images)
            parameters = cp.empty(log_parameters.shape, dtype=cp.float64)
            indices = cp.empty(2, dtype=cp.int64)
            delay((1,), (1,), (np.uint32(100),))
            images.fill(0)
            weights.fill(1)
            cp.copyto(parameters, source_parameters)
            cp.copyto(indices, source_indices)
            ready = cp.cuda.Event()
            ready.record(producer)
        with consumer:
            consumer.wait_event(ready)
            with GaussianWorkspace(
                images, weights, image_shape=shape, mode="difference"
            ) as workspace:
                residuals = workspace.residual(parameters, indices=indices)
                gradient, hessian = workspace.normal_equations(
                    parameters, residuals, indices=indices
                )
                del images, weights, parameters, indices
                gc.collect()
            # Closing waits for pending work before relinquishing owners;
            # the returned arrays remain usable after the workspace exits.
            np.testing.assert_allclose(
                cp.asnumpy(residuals).reshape(expected.shape),
                expected,
                rtol=5.0e-12,
                atol=5.0e-12,
            )
            np.testing.assert_allclose(
                cp.asnumpy(gradient),
                expected_gradient,
                rtol=5.0e-12,
                atol=5.0e-12,
            )
            np.testing.assert_allclose(
                cp.asnumpy(hessian),
                expected_hessian,
                rtol=5.0e-12,
                atol=5.0e-12,
            )
