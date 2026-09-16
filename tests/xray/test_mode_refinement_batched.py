# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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


def test_benchmark_runs_on_cpu_and_reports_agreement():
    r = benchmark_refinement(traces=8, run_gpu=False)
    assert r.traces == 8
    assert r.max_abs_theta_diff_numpy < 1e-6
    assert r.cupy_batched_s is None and r.gpu_error is None


def test_batched_refinement_on_cupy_matches_numpy():
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
    a = refine_modes_batched(np, fx.time, y, theta0, 2)
    b = refine_modes_batched(cp, fx.time, y, theta0, 2)
    np.testing.assert_allclose(cp.asnumpy(b.theta), a.theta, atol=1e-7)
