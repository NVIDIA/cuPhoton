# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest

from cuphoton.xray.iterative_fit import (
    IterativeFitOptions,
    _evaluate,
    _normal_equations,
    iterative_fit,
)


def _fixture():
    time = np.linspace(0.0, 60.0, 241)
    frequency = np.array([0.12, 0.27])
    decay = np.array([0.035, 0.017])
    amplitude = np.array([2.0, 0.8])
    phase = np.array([0.25, -0.6])
    signal = (
        np.sum(
            amplitude
            * np.exp(-time[:, None] * decay)
            * np.cos(2.0 * np.pi * frequency * time[:, None] + phase),
            axis=1,
        )
        + 0.15
    )
    parameters = np.empty(9)
    parameters[0:-1:4] = np.log(decay)
    parameters[1:-1:4] = amplitude
    parameters[2:-1:4] = phase
    parameters[3:-1:4] = np.tan(np.pi * (frequency / 0.5 - 0.5))
    parameters[-1] = 0.15
    return time, signal, parameters


def test_jacobian_and_regularized_gradient_match_finite_differences():
    time, trace, parameters = _fixture()
    parameters += np.array(
        [0.1, 0.1, 0.05, 0.02, -0.1, -0.05, -0.05, -0.02, 0.02]
    )
    model, jacobian = _evaluate(np, time, parameters, 2, 0.0, np.pi)
    l2 = 0.07
    _normal, gradient, cost = _normal_equations(
        np, jacobian, model - trace, parameters, l2
    )
    assert cost == pytest.approx(
        0.5 * np.sum((model - trace) ** 2)
        + l2 * np.sum(parameters[1:-1:4] ** 2)
    )
    numeric_jacobian = np.empty_like(jacobian)
    numeric_gradient = np.empty_like(parameters)
    for index in range(len(parameters)):
        step = np.zeros_like(parameters)
        step[index] = 1e-6
        shifted = []
        costs = []
        for trial in (parameters + step, parameters - step):
            values, _ = _evaluate(np, time, trial, 2, 0.0, np.pi)
            shifted.append(values)
            costs.append(
                0.5 * np.sum((values - trace) ** 2)
                + l2 * np.sum(trial[1:-1:4] ** 2)
            )
        numeric_jacobian[:, index] = (shifted[0] - shifted[1]) / 2e-6
        numeric_gradient[index] = (costs[0] - costs[1]) / 2e-6
    np.testing.assert_allclose(
        jacobian, numeric_jacobian, atol=2e-8, rtol=2e-7
    )
    np.testing.assert_allclose(
        gradient, numeric_gradient, atol=5e-7, rtol=2e-7
    )


def test_fft_initialized_fit_recovers_two_damped_modes():
    time, trace, _ = _fixture()
    result = iterative_fit(
        time, trace, 2, options=IterativeFitOptions(max_frequency=0.5)
    )
    assert result.converged
    assert result.status == "converged"
    np.testing.assert_allclose(
        result.angular_frequency / (2.0 * np.pi), [0.12, 0.27], atol=2e-7
    )
    np.testing.assert_allclose(result.amplitude, [2.0, 0.8], atol=2e-6)
    np.testing.assert_allclose(result.decay, [0.035, 0.017], atol=2e-7)
    np.testing.assert_allclose(result.phase, [0.25, -0.6], atol=2e-6)
    np.testing.assert_allclose(result.reconstruction, trace, atol=2e-7)
    assert result.offset == pytest.approx(0.15, abs=2e-7)
    assert result.chi2 < 1e-14
    np.testing.assert_allclose(
        result.time_components.sum(axis=1) + result.offset,
        result.reconstruction,
    )


def test_regularization_cost_and_diagnostics_describe_returned_model():
    time, trace, parameters = _fixture()
    result = iterative_fit(
        time,
        trace,
        2,
        options=IterativeFitOptions(amplitude_l2=0.01, max_frequency=0.5),
        initial_parameters=parameters,
    )
    assert result.converged
    assert result.amplitude[0] < 2.0
    residual = result.reconstruction - trace
    assert result.cost == pytest.approx(
        0.5 * np.sum(residual**2) + 0.01 * np.sum(result.amplitude**2)
    )
    assert result.chi2 == pytest.approx(np.mean(residual**2))
    assert result.trace_std == pytest.approx(np.std(trace))
    assert result.residual_std == pytest.approx(np.std(residual))
    assert result.relative_residual == pytest.approx(
        np.std(residual) / np.std(trace)
    )


def test_absolute_time_origin_does_not_change_phase_or_reconstruction():
    time, trace, _ = _fixture()
    options = IterativeFitOptions(max_frequency=0.5)
    baseline = iterative_fit(time, trace, 2, options=options)
    shifted = iterative_fit(time + 17.25, trace, 2, options=options)
    np.testing.assert_allclose(shifted.time, time + 17.25)
    np.testing.assert_allclose(
        shifted.reconstruction, baseline.reconstruction, atol=1e-12
    )
    np.testing.assert_allclose(shifted.phase, baseline.phase, atol=1e-12)


def test_negative_modes_are_canonicalized_without_changing_model():
    time = np.linspace(0, 10, 81)
    trace = (
        -2 * np.exp(-0.05 * time) * np.cos(-2 * np.pi * 0.2 * time + 0.3)
        + 0.4
    )
    u = np.tan(np.pi * ((-0.2 + 0.5) / 1.0 - 0.5))
    result = iterative_fit(
        time,
        trace,
        1,
        options=IterativeFitOptions(min_frequency=-0.5, max_frequency=0.5),
        initial_parameters=[np.log(0.05), -2.0, 0.3, u, 0.4],
    )
    assert result.converged
    np.testing.assert_allclose(result.angular_frequency / (2 * np.pi), [0.2])
    np.testing.assert_allclose(result.amplitude, [2.0])
    np.testing.assert_allclose(result.reconstruction, trace, atol=1e-14)


@pytest.mark.parametrize("phase", [1e10, 1e16])
def test_large_initial_phase_has_consistent_model_and_cost(phase):
    time = np.linspace(0, 10, 50)
    parameters = np.array([np.log(0.2), 1.0, phase, 0.0, 0.1])
    # This is the review regression: a trace generated without reducing a
    # large phase can differ from the model after phase reduction.
    trace, _ = _evaluate(np, time, parameters, 1, 0, 2 * np.pi)
    result = iterative_fit(
        time,
        trace,
        1,
        options=IterativeFitOptions(max_frequency=1.0),
        initial_parameters=parameters,
    )
    residual = result.reconstruction - trace
    assert result.cost == pytest.approx(0.5 * np.sum(residual**2), abs=1e-25)
    assert result.chi2 == pytest.approx(np.mean(residual**2), abs=1e-25)
    assert np.all(np.abs(result.phase) <= np.pi)


@pytest.mark.parametrize("phase", [1e10, 1e16])
def test_initial_phase_is_reduced_before_model_evaluation(phase):
    time = np.linspace(0, 10, 50)
    parameters = np.array([np.log(0.2), 1.0, phase, 0.0, 0.1])
    reduced = parameters.copy()
    reduced[2] = (phase + np.pi) % (2 * np.pi) - np.pi
    trace, _ = _evaluate(np, time, reduced, 1, 0, 2 * np.pi)
    result = iterative_fit(
        time,
        trace,
        1,
        options=IterativeFitOptions(max_frequency=1.0),
        initial_parameters=parameters,
    )
    assert result.converged
    assert result.iterations == 0
    assert result.cost < 1e-25
    np.testing.assert_allclose(result.reconstruction, trace, atol=1e-14)


def test_trial_phase_is_reduced_before_model_evaluation(monkeypatch):
    time = np.linspace(0, 10, 50)
    parameters = np.array([np.log(0.2), 1.0, 0.0, 0.0, 0.1])
    target = parameters.copy()
    target[2] = (1e16 + np.pi) % (2 * np.pi) - np.pi
    trace, _ = _evaluate(np, time, target, 1, 0, 2 * np.pi)

    def large_phase_step(_matrix, rhs):
        step = np.zeros_like(rhs)
        step[2] = 1e16
        return step

    monkeypatch.setattr(np.linalg, "solve", large_phase_step)
    result = iterative_fit(
        time,
        trace,
        1,
        options=IterativeFitOptions(max_iterations=1, max_frequency=1.0),
        initial_parameters=parameters,
    )
    assert result.converged
    assert result.iterations == 1
    assert result.cost < 1e-25
    np.testing.assert_allclose(result.reconstruction, trace, atol=1e-14)


@pytest.mark.parametrize("value", [0.0, 3.25])
def test_constant_trace_has_exact_offset_and_no_modes(value):
    time = np.arange(20, dtype=float)
    result = iterative_fit(time, np.full(len(time), value), 2)
    assert result.converged and result.status == "constant"
    assert result.iterations == 0
    assert result.offset == value
    assert result.time_components.shape == (len(time), 0)
    assert result.amplitude.size == 0
    assert result.chi2 == result.cost == result.gradient_norm == 0
    assert result.relative_residual == 0
    np.testing.assert_array_equal(result.reconstruction, value)


def test_float32_rounded_delay_grid_is_supported():
    time = np.arange(300, dtype=np.float32) * np.float32(0.7)
    trace = np.exp(-0.02 * time) * np.cos(2.0 * np.pi * 0.13 * time)
    result = iterative_fit(time.astype(float), trace.astype(float), 1)
    assert result.converged
    assert result.chi2 < 1e-9


def test_iteration_limit_does_not_claim_convergence():
    time, trace, parameters = _fixture()
    parameters[3] += 0.1
    result = iterative_fit(
        time,
        trace,
        2,
        options=IterativeFitOptions(max_iterations=1, max_frequency=0.5),
        initial_parameters=parameters,
    )
    assert not result.converged
    assert result.status == "iteration-limit"
    assert result.iterations == 1
    assert result.gradient_norm > 0


def test_rejected_zero_steps_do_not_claim_convergence(monkeypatch):
    time, trace, parameters = _fixture()
    parameters[1] *= 0.5
    monkeypatch.setattr(
        np.linalg, "solve", lambda _matrix, rhs: np.zeros_like(rhs)
    )
    result = iterative_fit(
        time,
        trace,
        2,
        options=IterativeFitOptions(max_frequency=0.5),
        initial_parameters=parameters,
    )
    assert not result.converged
    assert result.status == "stalled"
    assert result.gradient_norm > 0


def test_noisy_stationary_fit_converges_at_float64_cost_resolution():
    time, _trace, parameters = _fixture()
    values, jacobian = _evaluate(np, time, parameters, 2, 0.0, np.pi)
    orthogonal, _ = np.linalg.qr(jacobian)
    noise = np.random.default_rng(4).normal(size=time.shape)
    noise -= orthogonal @ (orthogonal.T @ noise)
    noise *= 2.0 / np.linalg.norm(noise)
    # Start just above the requested gradient tolerance, with a decrease
    # smaller than float64 can reliably distinguish from the noisy cost.
    trace = values + noise + 2.0 * 1.05e-8 / np.sqrt(len(time))
    normal, gradient, _cost = _normal_equations(
        np, jacobian, values - trace, parameters, 0.0
    )
    relative_gradient = np.max(
        np.abs(gradient) / np.sqrt(np.diag(normal))
    ) / np.linalg.norm(values - trace)
    assert 1e-8 < relative_gradient < np.sqrt(np.finfo(float).eps)

    result = iterative_fit(
        time,
        trace,
        2,
        options=IterativeFitOptions(max_frequency=0.5),
        initial_parameters=parameters,
    )

    assert result.converged
    assert result.status == "converged"
    assert result.cost == pytest.approx(2.0, abs=2e-15)
    assert result.cost == pytest.approx(
        0.5 * np.sum((result.reconstruction - trace) ** 2)
    )


def test_nonfinite_frequency_transform_is_not_accepted(monkeypatch):
    time = np.arange(16, dtype=float) * 0.5
    trace = np.exp(-0.05 * time) * np.cos(2.0 * np.pi * time)

    def infinite_frequency_step(_matrix, rhs):
        step = np.zeros_like(rhs)
        step[3] = np.inf
        return step

    monkeypatch.setattr(np.linalg, "solve", infinite_frequency_step)
    result = iterative_fit(
        time,
        trace,
        1,
        initial_parameters=[np.log(0.05), 1.0, 0.0, 0.0, 0.0],
    )
    assert not result.converged
    assert result.status == "nonfinite"
    assert np.all(np.isfinite(result.reconstruction))
    assert result.cost > 0


@pytest.mark.parametrize(
    "updates",
    [
        {"max_iterations": 0},
        {"max_iterations": 1.5},
        {"max_iterations": True},
        {"tolerance": 0},
        {"tolerance": float("nan")},
        {"amplitude_l2": -1},
        {"amplitude_l2": float("inf")},
        {"min_frequency": float("nan")},
        {"max_frequency": 0},
        {"min_frequency": 0.2, "max_frequency": 0.1},
    ],
)
def test_options_reject_invalid_values(updates):
    with pytest.raises(ValueError):
        IterativeFitOptions(**updates).validate()


def test_options_serialization_is_json_safe():
    values = IterativeFitOptions(
        max_iterations=np.int64(20), tolerance=np.float32(1e-6)
    ).to_dict()
    assert json.loads(json.dumps(values)) == values
    assert values["max_frequency"] is None


@pytest.mark.parametrize(
    "time,trace,components",
    [
        (np.arange(10), np.arange(9), 1),
        (np.arange(4), np.arange(4), 1),
        (np.arange(10).reshape(2, 5), np.arange(10), 1),
        (np.arange(10), np.full(10, np.nan), 1),
        (np.arange(10), np.arange(10) + 1j, 1),
        (np.arange(10)[::-1], np.arange(10), 1),
        (np.arange(10) ** 2, np.arange(10), 1),
        (np.zeros(10), np.arange(10), 1),
        (np.arange(10), np.arange(10), 0),
        (np.arange(10), np.arange(10), 3),
        (np.arange(10), np.arange(10), True),
        (np.arange(10), np.arange(10), 1.5),
    ],
)
def test_fit_rejects_invalid_arrays_and_model_size(time, trace, components):
    with pytest.raises(ValueError):
        iterative_fit(time, trace, components)


@pytest.mark.parametrize(
    "parameters",
    [np.ones(8), np.full(9, np.inf), [1000, 1, 0, 0, 0, 1, 0, 0, 0]],
)
def test_fit_rejects_invalid_initial_parameters(parameters):
    time, trace, _ = _fixture()
    with pytest.raises(ValueError, match="initial_parameters"):
        iterative_fit(time, trace, 2, initial_parameters=parameters)


@pytest.mark.parametrize(
    "options",
    [
        IterativeFitOptions(max_frequency=3.0),
        IterativeFitOptions(min_frequency=3.0),
        IterativeFitOptions(min_frequency=-3.0),
    ],
)
def test_fit_rejects_frequency_bounds_beyond_nyquist(options):
    time, trace, _ = _fixture()
    with pytest.raises(ValueError, match="frequency bounds"):
        iterative_fit(time, trace, 2, options=options)


def test_fit_rejects_unknown_backend():
    time, trace, _ = _fixture()
    with pytest.raises(ValueError, match="backend"):
        iterative_fit(time, trace, 2, backend="jax")


def test_gpu_fit_matches_cpu():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    time, trace, parameters = _fixture()
    parameters[1] *= 0.95
    options = IterativeFitOptions(max_frequency=0.5, amplitude_l2=0.01)
    cpu = iterative_fit(
        time, trace, 2, options=options, initial_parameters=parameters
    )
    gpu = iterative_fit(
        cupy.asarray(time),
        cupy.asarray(trace),
        2,
        options=options,
        backend="gpu",
        initial_parameters=cupy.asarray(parameters),
    )
    assert cpu.converged and gpu.converged
    assert gpu.backend == "gpu"
    assert isinstance(gpu.reconstruction, np.ndarray)
    np.testing.assert_allclose(
        gpu.reconstruction, cpu.reconstruction, atol=2e-7, rtol=2e-7
    )
    np.testing.assert_allclose(
        gpu.angular_frequency, cpu.angular_frequency, atol=2e-7, rtol=2e-7
    )
    assert gpu.cost == pytest.approx(cpu.cost, rel=1e-8, abs=1e-10)
