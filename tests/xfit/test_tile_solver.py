# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Contracts and numerical checks for public Tile/SIMT Gaussian fitting."""

from dataclasses import fields, replace

import numpy as np
import pytest

from cuphoton.xfit import (
    GaussianDipoleModel,
    LMConfig,
    LMStatus,
    _tile_solver,
    api,
)
from cuphoton.xfit.backend import Backend


@pytest.mark.parametrize(
    "entrypoint", [api.fit_dipoles, api.fit_dipoles_device]
)
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (LMConfig(solver_version="v0.1.3"), "solver_version='v0.2' only"),
        (LMConfig(use_finite_difference=True), "finite-difference"),
    ],
)
def test_rejects_unsupported_policy_before_loading_cuda(
    monkeypatch, entrypoint, settings, message
):
    monkeypatch.setattr(
        api,
        "resolve_backend",
        lambda *args: pytest.fail("initialized an unsupported CUDA request"),
    )
    with pytest.raises(ValueError, match=message):
        entrypoint(
            np.zeros((1, 7, 9)),
            model="gaussian",
            backend="tile-simt",
            config=settings,
        )


def test_rejects_custom_gaussian_before_loading_cuda(monkeypatch):
    class CustomGaussian(GaussianDipoleModel):
        pass

    model = CustomGaussian((7, 9))
    monkeypatch.setattr(
        api, "resolve_backend", lambda *args: pytest.fail("initialized CUDA")
    )
    with pytest.raises(ValueError, match="custom Gaussian"):
        api.fit_dipoles(np.zeros((1, 7, 9)), model=model, backend="tile-simt")


def test_standard_api_dispatch_preserves_mask_variance_and_config(
    monkeypatch,
):
    images = np.zeros((1, 7, 9))
    mask = np.ones_like(images, dtype=bool)
    variance = np.full_like(images, 2.0)
    config = LMConfig(max_evaluations=1)
    reference = api._fit_dipoles_backend(
        images,
        model="gaussian",
        mask=mask,
        variance=variance,
        config=config,
        resolved=Backend("numpy", np, "cpu"),
    )

    def fit(data, **kwargs):
        assert data is images
        assert kwargs["mask"] is mask
        assert kwargs["variance"] is variance
        assert kwargs["config"] is config
        return replace(reference, backend="tile-simt", device="cuda:0")

    monkeypatch.setattr(_tile_solver, "fit_gaussian_pipeline", fit)
    monkeypatch.setattr(
        api, "resolve_backend", lambda name: Backend(name, np, "cuda:0")
    )
    result = api.fit_dipoles(
        images,
        model="gaussian",
        mask=mask,
        variance=variance,
        config=config,
        backend="tile-simt",
    )
    assert result.backend == "tile-simt"
    assert result.solver_version == "v0.2"
    np.testing.assert_array_equal(result.evaluations, reference.evaluations)
    assert not np.shares_memory(result.parameters, reference.parameters)


def test_rejects_fp32_before_compiling(monkeypatch):
    monkeypatch.setattr(
        _tile_solver, "_runtime", lambda: pytest.fail("initialized CUDA")
    )
    with pytest.raises(TypeError, match="float64"):
        _tile_solver._gaussian_iterations(
            np.zeros((1, 8), dtype=np.float32),
            np.zeros((1, 63), dtype=np.float32),
            np.ones((1, 63), dtype=np.float32),
            image_shape=(7, 9),
            mode="difference",
        )


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("nvmath.tile.linalg")
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"CUDA is unavailable: {exc}")
    if not count:
        pytest.skip("CUDA device is unavailable")
    yield cp
    cp.cuda.get_current_stream().synchronize()


def _fixture(shape, mode):
    model = GaussianDipoleModel(shape, split_weights=(1.0, 0.7, 0.3))
    truth = np.array(
        [
            [5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [3.0, 1.6, 1.1, -0.3, -1.0, -1.0, 2.0, 1.0],
        ]
    )
    images = model.evaluate(truth, mode=mode)
    initial = truth + np.array([0.2, 0.1, -0.1, 0.03, 0.1, -0.1, -0.1, 0.1])
    mask = np.ones_like(images, dtype=bool)
    mask[..., 0, 0] = False
    images[..., 0, 0] = np.nan
    variance = np.full_like(images, 2.0)
    return model, truth, images, initial, mask, variance


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("shape", "mode"),
    [((11, 15), "difference"), ((11, 15), "split"), ((51, 51), "difference")],
)
def test_fused_fit_preserves_science_and_input_ownership(
    runtime, shape, mode
):
    cp = runtime
    model, truth, images, initial, mask, variance = _fixture(shape, mode)
    device_images = cp.asarray(images)
    initial_device = cp.asarray(initial)
    original_images = device_images.copy()
    original_initial = initial_device.copy()
    config = LMConfig(max_evaluations=100)
    result = api.fit_dipoles_device(
        device_images,
        model=model,
        initial=initial_device,
        mask=mask,
        variance=variance,
        mode=mode,
        backend="tile-simt",
        config=config,
    )
    reference = api.fit_dipoles(
        images,
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        mode=mode,
        backend="numpy",
        config=config,
    )
    assert result.backend == "tile-simt"
    assert result.solver_version == "v0.2"
    assert isinstance(result.parameters, cp.ndarray)
    np.testing.assert_allclose(
        cp.asnumpy(result.parameters), truth, rtol=2e-8, atol=2e-8
    )
    np.testing.assert_array_equal(cp.asnumpy(result.converged), [True, True])
    np.testing.assert_allclose(
        cp.asnumpy(result.covariance),
        reference.covariance,
        rtol=2e-7,
        atol=2e-8,
    )
    factor = cp.asnumpy(result.covariance_factor)
    np.testing.assert_allclose(
        factor @ factor.swapaxes(1, 2),
        cp.asnumpy(result.covariance),
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_array_equal(
        cp.asnumpy(device_images), cp.asnumpy(original_images)
    )
    np.testing.assert_array_equal(
        cp.asnumpy(initial_device), cp.asnumpy(original_initial)
    )
    assert not cp.shares_memory(result.parameters, initial_device)
    assert not cp.shares_memory(result.residuals, device_images)


@pytest.mark.gpu
def test_budget_one_preserves_terminal_rows_and_current_diagnostics(runtime):
    cp = runtime
    model, _, images, initial, mask, variance = _fixture(
        (11, 15), "difference"
    )
    result = api.fit_dipoles(
        images,
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        backend="tile-simt",
        config=LMConfig(max_evaluations=1),
    )
    reference = api.fit_dipoles(
        images,
        model=model,
        initial=initial,
        mask=mask,
        variance=variance,
        backend="numpy",
        config=LMConfig(max_evaluations=1),
    )
    np.testing.assert_array_equal(result.evaluations, [1, 1])
    np.testing.assert_array_equal(result.status, reference.status)
    np.testing.assert_allclose(
        result.parameters, reference.parameters, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        result.chi_square, reference.chi_square, rtol=2e-8, atol=2e-8
    )
    assert not result.uncertainty_valid.any()
    assert np.isnan(result.covariance).all()
    cp.cuda.get_current_stream().synchronize()


@pytest.mark.gpu
def test_overflowing_cost_does_not_imply_gradient_convergence(runtime):
    model = GaussianDipoleModel((11, 15))
    initial = np.array([[5.0, 1.0, 2.0, 0.0, -2.0, 0.0, 2.0, 0.0]])
    truth = initial.copy()
    truth[:, 0] = 1e200
    images = model.evaluate(truth)
    config = LMConfig(max_evaluations=2)
    with np.errstate(over="ignore", invalid="ignore"):
        reference = api.fit_dipoles(
            images,
            model=model,
            initial=initial,
            backend="numpy",
            config=config,
        )
    result = api.fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="tile-simt",
        config=config,
    )
    np.testing.assert_array_equal(result.status, reference.status)
    np.testing.assert_array_equal(result.evaluations, [2])
    np.testing.assert_array_equal(result.converged, [False])


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [4, 100])
def test_rejected_trials_preserve_and_refresh_normal_equations(
    runtime, monkeypatch, budget
):
    cp = runtime
    model = GaussianDipoleModel((11, 15))
    truth = np.array([[5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7]])
    initial = np.array([[1.0, 1.2, 1.0, 0.5, -2.4, 0.4, 3.2, -3.2]])
    images = model.evaluate(truth)
    config = LMConfig(initial_damping=1e-12, max_evaluations=budget)
    iterations = _tile_solver._gaussian_iterations
    states = []

    def capture(*args, **kwargs):
        state = iterations(*args, **kwargs)
        states.append(state)
        return state

    monkeypatch.setattr(_tile_solver, "_gaussian_iterations", capture)
    result = api.fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="tile-simt",
        config=config,
    )
    reference = api.fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="numpy",
        config=config,
    )
    np.testing.assert_allclose(
        result.parameters, reference.parameters, rtol=2e-7, atol=2e-8
    )
    np.testing.assert_array_equal(result.converged, reference.converged)
    if budget == 4:
        # The first step is accepted, then at least two trials are rejected.
        # The terminal row must retain diagnostics for that accepted point.
        assert int(cp.asnumpy(states[0]["rejected_steps"])[0]) >= 2
        assert bool(cp.asnumpy(states[0]["diagnostics_current"])[0])
        np.testing.assert_array_equal(result.evaluations, [budget])
    else:
        # Later accepted steps must invalidate the cached equations so the
        # same fit can leave the rejection plateau and reach its solution.
        np.testing.assert_array_equal(result.converged, [True])
        np.testing.assert_allclose(
            model.evaluate(result.parameters), images, rtol=2e-7, atol=2e-8
        )


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [4, 100])
def test_singleton_loop_preserves_mixed_terminal_rows(
    runtime, monkeypatch, budget
):
    cp = runtime
    if cp.cuda.Device().compute_capability not in {"100", "120"}:
        pytest.skip("specialized singleton schedule requires SM100 or SM120")
    model = GaussianDipoleModel((51, 51))
    truth = np.array([[5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7]])
    images = np.concatenate((model.evaluate(truth), np.zeros((2, 51, 51))))
    initial = np.array(
        [
            [1.0, 1.2, 1.0, 0.5, -2.4, 0.4, 3.2, -3.2],
            [0.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [1.0, 1e-200, 1.0, 0.0, 0.25, 0.0, -0.25, 0.0],
        ]
    )
    config = LMConfig(initial_damping=1e-12, max_evaluations=budget)
    iterations = _tile_solver._gaussian_iterations
    states = []

    def capture(*args, **kwargs):
        state = iterations(*args, **kwargs)
        states.append(state)
        return state

    monkeypatch.setattr(_tile_solver, "_gaussian_iterations", capture)
    result = api.fit_dipoles(
        images,
        model=model,
        initial=initial,
        backend="tile-simt",
        config=config,
    )
    # A solved row and a finite-residual/nonfinite-J row leave one eligible
    # fit. Their flags and budgets must survive its remaining transitions.
    np.testing.assert_array_equal(
        result.status[1:], ["converged_g_tol", "invalid_residual"]
    )
    np.testing.assert_array_equal(result.evaluations[1:], [1, 1])
    np.testing.assert_array_equal(
        cp.asnumpy(states[0]["diagnostics_current"])[1:], [True, False]
    )
    if budget == 4:
        np.testing.assert_allclose(
            result.parameters[0], initial[0], rtol=2e-15, atol=0
        )
        assert result.evaluations[0] == budget
        assert bool(cp.asnumpy(states[0]["diagnostics_current"])[0])
        assert int(cp.asnumpy(states[0]["rejected_steps"])[0]) >= 2
    else:
        assert result.converged[0]
        assert result.evaluations[0] > 17
        np.testing.assert_allclose(
            result.parameters[0], truth[0], rtol=2e-7, atol=2e-8
        )


@pytest.mark.parametrize("drain_fails", [False, True])
@pytest.mark.parametrize(
    "failure_stage", ["prepare", "launch", "synchronize"]
)
def test_launch_failure_drains_stream_and_preserves_exception(
    monkeypatch, drain_fails, failure_stage
):
    from types import SimpleNamespace

    drained = []

    def synchronize():
        drained.append(True)
        if failure_stage == "synchronize" and len(drained) == 1:
            raise KeyboardInterrupt("interrupted after enqueue")
        if drain_fails:
            raise RuntimeError("secondary synchronization failure")

    stream = SimpleNamespace(synchronize=synchronize)

    class ArrayRuntime:
        cuda = SimpleNamespace(get_current_stream=lambda: stream)

        def __getattr__(self, name):
            return getattr(np, name)

    def launch(*args):
        if failure_stage == "launch":
            raise KeyboardInterrupt("interrupted after enqueue")

    monkeypatch.setattr(
        _tile_solver,
        "_runtime",
        lambda: (ArrayRuntime(), SimpleNamespace(launch=launch)),
    )

    def kernels(*args):
        if failure_stage == "prepare":
            raise KeyboardInterrupt("interrupted after enqueue")
        return object()

    monkeypatch.setattr(_tile_solver, "_kernels", kernels)
    with pytest.raises(KeyboardInterrupt, match="interrupted after enqueue"):
        _tile_solver._gaussian_iterations(
            np.zeros((1, 8)),
            np.zeros((1, 63)),
            np.ones((1, 63)),
            image_shape=(7, 9),
            mode="difference",
        )
    assert drained == [True] * (2 if failure_stage == "synchronize" else 1)


@pytest.mark.gpu
@pytest.mark.parametrize("nonblocking", [False, True])
@pytest.mark.parametrize("budget", [18, 25, 60])
def test_graph_preserves_terminal_rows_budget_and_singleton_tail(
    runtime, monkeypatch, nonblocking, budget
):
    cp = runtime
    capability = cp.cuda.Device().compute_capability
    if capability not in {"100", "120"}:
        pytest.skip("bounded graph schedule requires SM100 or SM120")
    _, ct = _tile_solver._runtime()
    sm100_normals = capability == "100"
    if budget == 18 and not sm100_normals:
        pytest.skip("accepted-Jacobian graph caching requires SM100")
    control_kernel = (
        _tile_solver._singleton_graph_control(True)
        if sm100_normals
        else _tile_solver._singleton_graph_control()
    )
    iteration_kernel = (
        _tile_solver._kernels(
            True,
            normal_recipe="sm100-gemmsn",
            known_non_singleton_cost=False,
        )
        if sm100_normals
        else _tile_solver._kernels(True)
    )
    cached_kernel = (
        _tile_solver._kernels(
            True,
            normal_recipe="sm100-mma",
            known_non_singleton_cost=False,
            cache_normals=True,
        )
        if sm100_normals
        else None
    )
    launch = ct.launch
    stream = (
        cp.cuda.Stream(non_blocking=True)
        if nonblocking
        else cp.cuda.Stream.null
    )
    captured_controls = []
    singleton_tail = []
    cached_diagnostics = []

    def record_launch(target_stream, grid, kernel, arguments):
        if kernel is cached_kernel and target_stream.ptr == stream.ptr:
            active = (arguments[9] == 0) & (arguments[10] < budget)
            assert not cp.asnumpy(arguments[11])[cp.asnumpy(active)].any()
            # The first real graph step must initialize every readable cost.
            arguments[-3].fill(cp.nan)
            cached_diagnostics.append(arguments[11])
        if kernel is control_kernel and target_stream.ptr != stream.ptr:
            captured_controls.append(True)
        if (
            captured_controls
            and target_stream.ptr == stream.ptr
            and kernel is iteration_kernel
            and arguments[21] == 2
        ):
            singleton_tail.append(True)
        return launch(target_stream, grid, kernel, arguments)

    monkeypatch.setattr(ct, "launch", record_launch)
    model = GaussianDipoleModel((51, 51))
    truth = np.array([[5.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7]])
    image = model.evaluate(truth)
    images = np.concatenate((image, image, np.zeros((2, 51, 51))))
    initial = np.array(
        [
            [1.0, 1.2, 1.0, 0.5, -2.4, 0.4, 3.2, -3.2],
            [1.0, 1.3, 2.6, -0.6, 3.0, -3.5, 1.3, -0.8],
            [0.0, 1.3, 1.8, 0.25, -2.0, 0.5, 2.2, -0.7],
            [1.0, 1e-200, 1.0, 0.0, 0.25, 0.0, -0.25, 0.0],
        ]
    )
    with stream:
        device_images = cp.asarray(images)
        device_initial = cp.asarray(initial)
        result = api.fit_dipoles_device(
            device_images,
            model=model,
            initial=device_initial,
            backend="tile-simt",
            config=LMConfig(initial_damping=1e-12, max_evaluations=budget),
        )
        status = cp.asnumpy(result.status_codes)
        evaluations = cp.asnumpy(result.evaluations)
        converged = cp.asnumpy(result.converged)
        parameters = cp.asnumpy(result.parameters)
        np.testing.assert_array_equal(cp.asnumpy(device_images), images)
        np.testing.assert_array_equal(cp.asnumpy(device_initial), initial)
        assert not cp.shares_memory(result.parameters, device_initial)
        assert not cp.shares_memory(result.residuals, device_images)
        if sm100_normals:
            assert len(cached_diagnostics) == 1
            if budget == 18:
                # Acceptance at the budget limit leaves final diagnostics
                # stale; the public finalizer must refresh the returned fit.
                assert not bool(cp.asnumpy(cached_diagnostics[0])[0])
            if budget == 25:
                # Rejection preserves current diagnostics at the budget limit.
                np.testing.assert_array_equal(
                    cp.asnumpy(cached_diagnostics[0]),
                    [False, True, True, False],
                )
    assert captured_controls, "the long-fit fixture did not enter capture"
    if budget == 60:
        assert singleton_tail, "the remaining fit did not return to extent2"
    # Budget 18 exhausts after the warm transition. Later budgets let one
    # fit converge inside the first replay while the other reaches its limit.
    np.testing.assert_array_equal(
        converged, [budget != 18, False, True, False]
    )
    np.testing.assert_array_equal(
        status[1:],
        [
            LMStatus.MAX_EVALUATIONS,
            LMStatus.CONVERGED_G_TOL,
            LMStatus.INVALID_RESIDUAL,
        ],
    )
    np.testing.assert_array_equal(evaluations[1:], [budget, 1, 1])
    assert evaluations[0] <= budget
    if budget == 18:
        assert status[0] == LMStatus.MAX_EVALUATIONS
        assert evaluations[0] == budget
    else:
        np.testing.assert_allclose(
            parameters[0], truth[0], rtol=2e-7, atol=2e-8
        )
    if sm100_normals:
        # Ordinary transitions recompute accepted derivatives. Graph reuse
        # must preserve every public result, including final diagnostics.
        with monkeypatch.context() as reference_patch, stream:
            reference_patch.setattr(ct, "launch", launch)
            reference_patch.setattr(
                _tile_solver, "_GRAPH_MIN_HOST_TRANSITIONS", budget + 1
            )
            reference = api.fit_dipoles_device(
                device_images,
                model=model,
                initial=device_initial,
                backend="tile-simt",
                config=LMConfig(
                    initial_damping=1e-12, max_evaluations=budget
                ),
            )
            for field in fields(result):
                actual = getattr(result, field.name)
                expected = getattr(reference, field.name)
                if isinstance(actual, cp.ndarray):
                    assert isinstance(expected, cp.ndarray), field.name
                    actual = cp.asnumpy(actual)
                    expected = cp.asnumpy(expected)
                    assert actual.shape == expected.shape, field.name
                    assert actual.dtype == expected.dtype, field.name
                    assert actual.tobytes() == expected.tobytes(), field.name
                else:
                    assert type(actual) is type(expected), field.name
                    assert actual == expected, field.name
