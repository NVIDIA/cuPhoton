# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xray.mode_refinement import (
    _model_and_jacobian,
    linear_prediction_refined,
    refine_modes,
    seed_from_residual,
)
from cuphoton.xray.synthetic_validation import (
    cramer_rao_bounds,
    synthetic_modes_trace,
    validation_sweep,
)


def test_analytic_jacobian_matches_finite_differences():
    fx = synthetic_modes_trace(96)
    theta = np.array([1.25, 0.09, 2.4, 0.3, 0.45, 0.03, 0.9, -0.75, 0.15])
    _, jac = _model_and_jacobian(theta, fx.time, 2)
    h = 1e-6
    for k in range(theta.size):
        e = np.zeros_like(theta)
        e[k] = h
        num = (
            _model_and_jacobian(theta + e, fx.time, 2)[0]
            - _model_and_jacobian(theta - e, fx.time, 2)[0]
        ) / (2 * h)
        np.testing.assert_allclose(jac[:, k], num, atol=1e-7)


def test_refinement_recovers_exact_modes_from_a_perturbed_start():
    fx = synthetic_modes_trace(96)
    start = [(1.1, 0.05, 2.5, 0.0), (0.5, 0.0, 0.8, 0.0)]
    r = refine_modes(fx.time, fx.trace, start, 0.0)
    assert r.converged
    np.testing.assert_allclose(r.angular_frequency, [2.4, 0.9], atol=1e-8)
    np.testing.assert_allclose(r.decay, [0.09, 0.03], atol=1e-8)
    np.testing.assert_allclose(r.amplitude, [1.25, 0.45], atol=1e-8)
    assert abs(r.constant - 0.15) < 1e-8
    assert r.residual_rms < 1e-10


def test_seed_from_residual_finds_the_missing_mode():
    fx = synthetic_modes_trace(256, duration=25.5)
    weak = 0.45 * np.exp(-0.03 * fx.time) * np.cos(0.9 * fx.time - 0.75)
    amp, dec, freq, _ = seed_from_residual(fx.time, weak)
    assert abs(freq - 0.9) < 0.05
    assert dec == 0.0 and amp > 0


def test_refined_estimator_keeps_the_weak_mode_under_noise():
    fx = synthetic_modes_trace(96)
    sigma = float(np.std(fx.clean - fx.clean.mean())) / 100
    rng = np.random.default_rng(0)
    lost = 0
    for _ in range(60):
        y = fx.clean + rng.normal(0.0, sigma, size=fx.time.shape)
        r = linear_prediction_refined(fx.time, y, 6, 2)
        if np.min(np.abs(r.angular_frequency - 0.9)) > 0.05:
            lost += 1
    assert lost == 0


def test_refined_sweep_sits_near_the_cramer_rao_bound():
    sweep = validation_sweep(
        snr_db=(30.0,),
        trials=60,
        seed=2,
        estimator=lambda t, y, k: linear_prediction_refined(t, y, k, 2),
        backend="cpu-refined",
    )
    level = sweep.levels[0]
    for mode in level.modes:
        assert mode.loss_rate == 0.0
        assert 0.7 < mode.frequency_ratio < 1.4
        assert 0.7 < mode.decay_ratio < 1.4


def test_refinement_rejects_empty_initial_modes():
    fx = synthetic_modes_trace(96)
    with pytest.raises(ValueError):
        refine_modes(fx.time, fx.trace, [], 0.0)


def test_bounds_are_the_reference_for_the_refined_uncertainties():
    # the per-fit one-sigma from the Jacobian should agree with the bound
    # to within the scatter of a single residual-variance estimate
    fx = synthetic_modes_trace(96)
    sigma = float(np.std(fx.clean - fx.clean.mean())) / 1000
    b = cramer_rao_bounds(fx.modes, fx.constant, fx.time, sigma)
    y = fx.clean + np.random.default_rng(5).normal(0.0, sigma, fx.time.shape)
    r = linear_prediction_refined(fx.time, y, 6, 2)
    ratio = r.sigma_angular_frequency / b["angular_frequency"]
    assert np.all(ratio > 0.6) and np.all(ratio < 1.6)
