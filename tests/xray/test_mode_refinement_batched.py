# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from cuphoton.xray.mode_refinement import _model_and_jacobian, refine_modes
from cuphoton.xray.mode_refinement_batched import (
    _canonical,
    benchmark_refinement,
    model_and_jacobian_batched,
    refine_modes_batched,
)
from cuphoton.xray.synthetic_validation import (
    modes_to_theta,
    synthetic_modes_trace,
)


def test_batched_model_and_jacobian_match_the_serial_ones():
    fx = synthetic_modes_trace(96)
    theta = modes_to_theta(fx.modes, fx.constant)
    batch = np.stack([theta, theta * 1.01, theta * 0.98])
    model, jac = model_and_jacobian_batched(np, batch, fx.time, 2)
    for i in range(3):
        m, j = _model_and_jacobian(batch[i], fx.time, 2)
        np.testing.assert_allclose(model[i], m, rtol=0, atol=1e-12)
        np.testing.assert_allclose(jac[i], j, rtol=0, atol=1e-12)


def test_batched_large_phase_matches_wrapped_scalar_model():
    time = np.linspace(0, 10, 50)
    start = np.array([[1.0, 0.2, 2.0, 1e16, 0.1]])
    wrapped = start.copy()
    wrapped[0, 3] = (start[0, 3] + np.pi) % (2 * np.pi) - np.pi
    trace, expected_jac = _model_and_jacobian(wrapped[0], time, 1)
    model, jac = model_and_jacobian_batched(np, start, time, 1)
    np.testing.assert_allclose(model[0], trace, atol=1e-14)
    np.testing.assert_allclose(jac[0], expected_jac, atol=1e-14)
    result = refine_modes_batched(np, time, trace[None, :], start, 1)
    assert result.converged.tolist() == [True]
    assert result.residual_rms[0] < 1e-14
    np.testing.assert_allclose(result.theta, wrapped, atol=1e-14)


def test_batched_refinement_matches_scipy_per_trace():
    fx = synthetic_modes_trace(96)
    rng = np.random.default_rng(1)
    traces = 12
    y = fx.clean[None, :] + rng.normal(0.0, 0.02, (traces, fx.time.size))
    truth = modes_to_theta(fx.modes, fx.constant)
    theta0 = truth[None, :] * (1.0 + 0.05 * rng.standard_normal((traces, 1)))
    res = refine_modes_batched(np, fx.time, y, theta0, 2)
    assert bool(np.all(res.converged))
    got = _canonical(res.theta, 2)
    for i in range(traces):
        init = [tuple(theta0[i, 4 * k : 4 * k + 4]) for k in range(2)]
        r = refine_modes(fx.time, y[i], init, float(theta0[i, -1]))
        j = int(np.argmin(np.abs(r.angular_frequency - 2.4)))
        assert abs(got[i, 2] - r.angular_frequency[j]) < 1e-7
        assert abs(got[i, 1] - r.decay[j]) < 1e-7
        assert abs(got[i, -1] - r.constant) < 1e-7


def test_batched_refinement_rejects_bad_shapes():
    fx = synthetic_modes_trace(96)
    with pytest.raises(ValueError):
        refine_modes_batched(np, fx.time, fx.trace, np.zeros((1, 9)), 2)
    with pytest.raises(ValueError):
        refine_modes_batched(
            np, fx.time, fx.trace[None, :], np.zeros((1, 7)), 2
        )


@pytest.mark.parametrize("time_size,trace_size", [(1, 96), (96, 1), (0, 0)])
def test_batched_refinement_rejects_broadcast_or_empty_samples(
    time_size, trace_size
):
    with pytest.raises(ValueError, match="time|sample"):
        refine_modes_batched(
            np,
            np.arange(time_size),
            np.zeros((1, trace_size)),
            np.zeros((1, 9)),
            2,
        )


@pytest.mark.parametrize("input_name", ["time", "traces", "theta0"])
@pytest.mark.parametrize("invalid_value", [np.nan, 1j])
def test_batched_refinement_rejects_invalid_input(input_name, invalid_value):
    fx = synthetic_modes_trace(96)
    inputs = {
        "time": fx.time.copy(),
        "traces": fx.trace[None, :].copy(),
        "theta0": modes_to_theta(fx.modes, fx.constant)[None, :],
    }
    inputs[input_name] = inputs[input_name].astype(type(invalid_value))
    inputs[input_name].flat[0] = invalid_value
    with pytest.raises(ValueError, match=input_name):
        refine_modes_batched(np, **inputs, n_modes=2)


@pytest.mark.parametrize(
    "options", [{"tol": np.inf}, {"tol": 0}, {"max_iter": -1}, {"n_modes": 0}]
)
def test_batched_refinement_rejects_invalid_stopping_options(options):
    fx = synthetic_modes_trace(96)
    arguments = {"n_modes": 2, **options}
    with pytest.raises(ValueError, match=next(iter(options))):
        refine_modes_batched(
            np,
            fx.time,
            fx.trace[None, :],
            modes_to_theta(fx.modes, fx.constant)[None, :],
            **arguments,
        )


def test_batched_refinement_accepts_an_exact_start_without_improvement():
    fx = synthetic_modes_trace(96)
    truth = modes_to_theta(fx.modes, fx.constant)
    result = refine_modes_batched(
        np, fx.time, fx.trace[None, :], truth[None, :], 2
    )
    assert result.converged.tolist() == [True]
    assert result.iterations == 1
    assert result.residual_rms[0] < 1e-14


def test_zero_amplitude_start_does_not_abort_other_rows():
    fx = synthetic_modes_trace(96)
    truth = modes_to_theta(fx.modes, fx.constant)
    starts = np.stack([truth, truth])
    starts[1, 4] = 0.0
    traces = np.stack([fx.trace, fx.trace])
    result = refine_modes_batched(np, fx.time, traces, starts, 2)
    assert result.converged.tolist() == [True, True]
    np.testing.assert_allclose(
        result.theta, np.stack([truth, truth]), atol=1e-7
    )
    assert np.all(result.residual_rms < 1e-8)


@pytest.mark.parametrize("lambda0", [1e10, 1e30])
def test_large_damping_does_not_make_a_nonstationary_start_converged(lambda0):
    fx = synthetic_modes_trace(96)
    start = modes_to_theta(fx.modes, fx.constant)[None, :]
    start[0, -1] += 1.0
    result = refine_modes_batched(
        np, fx.time, fx.trace[None, :], start, 2, lambda0=lambda0, max_iter=1
    )
    assert result.converged.tolist() == [False]


@pytest.mark.parametrize("time_scale", [1e-9, 1.0, 1e9])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_batched_convergence_is_independent_of_time_units(
    time_scale, backend
):
    xp = np
    if backend == "cupy":
        xp = pytest.importorskip("cupy")
        try:
            xp.cuda.runtime.getDeviceCount()
        except Exception:  # pragma: no cover - no device
            pytest.skip("no CUDA device")
    fx = synthetic_modes_trace(96)
    truth = modes_to_theta(fx.modes, fx.constant)
    start = truth.copy()
    start[[1, 2, 5, 6]] /= time_scale
    start[[2, 6]] *= 1.05
    result = refine_modes_batched(
        xp, fx.time * time_scale, fx.trace[None, :], start[None, :], 2
    )
    assert result.converged.tolist() == [True]
    assert result.residual_rms[0] < 1e-8
    recovered = result.theta[0].copy()
    if backend == "cupy":
        recovered = xp.asnumpy(recovered)
    recovered[[1, 2, 5, 6]] *= time_scale
    np.testing.assert_allclose(recovered, truth, atol=1e-8)


@pytest.mark.parametrize("lambda0", [0.0, -1.0, np.nan, np.inf])
def test_batched_refinement_requires_positive_finite_damping(lambda0):
    fx = synthetic_modes_trace(96)
    truth = modes_to_theta(fx.modes, fx.constant)[None, :]
    with pytest.raises(ValueError, match="lambda0"):
        refine_modes_batched(
            np, fx.time, fx.trace[None, :], truth, 2, lambda0=lambda0
        )


def test_benchmark_runs_on_cpu_and_reports_agreement():
    r = benchmark_refinement(traces=8, run_gpu=False)
    assert r.traces == 8
    assert r.max_abs_theta_diff_numpy < 1e-6
    assert r.cupy_batched_s is None and r.gpu_error is None


def test_benchmark_reports_unconverged_fits_and_residuals(monkeypatch):
    from cuphoton.xray import mode_refinement_batched

    def one_iteration(*args, **kwargs):
        return refine_modes_batched(*args, **kwargs, max_iter=1)

    monkeypatch.setattr(
        mode_refinement_batched, "refine_modes_batched", one_iteration
    )
    result = benchmark_refinement(traces=2, repeat=1, run_gpu=False)
    payload = result.to_dict()
    assert payload["cpu_serial_converged"] == 2
    assert payload["numpy_batched_converged"] == 0
    assert payload["cupy_batched_converged"] is None
    assert (
        payload["numpy_batched_max_residual_rms"]
        > payload["cpu_serial_max_residual_rms"]
    )
    json.dumps(payload, allow_nan=False)
    assert "numpy_batched_converged=0/2" in str(result)
    assert "gpu_status=skipped" in str(result)


@pytest.mark.parametrize("options", [{"traces": 0}, {"n_modes": 3}])
def test_benchmark_rejects_empty_or_unavailable_fixtures(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        benchmark_refinement(**options, run_gpu=False)


@pytest.mark.parametrize("time_scale", [1e-9, 1.0, 1e9])
def test_batched_refinement_on_cupy_matches_numpy(time_scale):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - no device
        pytest.skip("no CUDA device")
    fx = synthetic_modes_trace(96)
    rng = np.random.default_rng(2)
    y = fx.clean[None, :] + rng.normal(0.0, 0.02, (32, fx.time.size))
    truth = modes_to_theta(fx.modes, fx.constant)
    theta0 = truth[None, :] * (1.0 + 0.05 * rng.standard_normal((32, 1)))
    theta0[:, [1, 2, 5, 6]] /= time_scale
    # Resolve noisy optima to a practical gradient tolerance; at 1e-9,
    # rounding of cost differences can prevent further accepted steps.
    a = refine_modes_batched(np, fx.time * time_scale, y, theta0, 2, tol=1e-8)
    b = refine_modes_batched(cp, fx.time * time_scale, y, theta0, 2, tol=1e-8)
    got = cp.asnumpy(b.theta)
    expected = a.theta.copy()
    got[:, [1, 2, 5, 6]] *= time_scale
    expected[:, [1, 2, 5, 6]] *= time_scale
    assert np.all(a.converged)
    assert np.all(cp.asnumpy(b.converged))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        cp.asnumpy(b.residual_rms), a.residual_rms, rtol=1e-10
    )
