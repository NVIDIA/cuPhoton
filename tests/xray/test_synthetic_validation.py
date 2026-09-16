# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xray.linear_prediction import (
    linear_prediction_numpy,
    synthetic_trace,
)
from cuphoton.xray.synthetic_validation import (
    PARAMETERS,
    DampedMode,
    build_summary,
    cramer_rao_bounds,
    distort_trace,
    match_modes,
    synthetic_modes_trace,
    validation_sweep,
    write_validation_run,
)


def test_fixture_reproduces_synthetic_trace_at_zero_noise():
    time, trace = synthetic_trace(96)
    fx = synthetic_modes_trace(96)
    np.testing.assert_allclose(fx.time, time)
    np.testing.assert_allclose(fx.trace, trace)
    assert fx.noise_sigma == 0.0


def test_fixture_noise_is_reproducible_and_has_the_requested_scale():
    a = synthetic_modes_trace(256, noise_sigma=0.05, seed=7)
    b = synthetic_modes_trace(256, noise_sigma=0.05, seed=7)
    np.testing.assert_array_equal(a.trace, b.trace)
    resid = a.trace - a.clean
    assert abs(resid.std() - 0.05) < 0.01


def test_estimator_recovers_known_modes_exactly_at_zero_noise():
    fx = synthetic_modes_trace(96)
    r = linear_prediction_numpy(fx.time, fx.trace, 6)
    hit = match_modes(
        fx.modes,
        np.asarray(r.angular_frequency),
        np.asarray(r.decay),
        tolerance=0.05,
    )
    assert all(h is not None for h in hit)
    for m, h in zip(fx.modes, hit):
        assert abs(float(r.angular_frequency[h]) - m.angular_frequency) < 1e-6
        assert abs(float(r.decay[h]) - m.decay) < 1e-6


def test_cramer_rao_bound_scales_with_noise_and_shrinks_with_trace_length():
    fx = synthetic_modes_trace(96)
    b1 = cramer_rao_bounds(fx.modes, fx.constant, fx.time, 0.01)
    b2 = cramer_rao_bounds(fx.modes, fx.constant, fx.time, 0.02)
    np.testing.assert_allclose(
        b2["angular_frequency"], 2 * b1["angular_frequency"], rtol=1e-6
    )
    longer = synthetic_modes_trace(192, duration=19.0)
    b3 = cramer_rao_bounds(longer.modes, longer.constant, longer.time, 0.01)
    assert np.all(b3["angular_frequency"] < b1["angular_frequency"])


def test_single_mode_bound_matches_the_undamped_closed_form():
    # Real sinusoid A cos(w t + phi) in white Gaussian noise, unknown A, w and
    # phi, samples at t = n dt for n = 0..N-1 (Kay, Fundamentals of
    # Statistical Signal Processing: Estimation Theory, 1993, sinusoidal
    # parameter estimation): var(w) >= 24 sigma^2 / (A^2 dt^2 N (N^2 - 1)).
    # The complex-exponential form has 12 in place of 24. The numerical
    # bound also frees the decay and the constant, so it may sit slightly
    # above the closed form, never below it.
    mode = (DampedMode(1.0, 0.0, 2.0, 0.3),)
    fx = synthetic_modes_trace(256, modes=mode, constant=0.0, duration=25.5)
    dt = fx.time[1] - fx.time[0]
    n = fx.time.size
    sigma = 0.01
    closed = np.sqrt(24 * sigma**2 / (1.0**2 * dt**2 * n * (n**2 - 1)))
    numerical = cramer_rao_bounds(mode, 0.0, fx.time, sigma)[
        "angular_frequency"
    ][0]
    assert numerical >= closed * 0.999
    assert numerical < closed * 1.03


def test_validation_sweep_reports_bound_ratios_and_loss_rates():
    sweep = validation_sweep(snr_db=(40.0, 20.0), trials=40, seed=1)
    assert len(sweep.levels) == 2
    strong = sweep.levels[0].modes[0]
    w = strong.angular_frequency
    assert w.crlb_std > 0 and w.std > 0
    assert w.crlb_variance == pytest.approx(w.crlb_std**2)
    assert w.std_over_crlb_std >= 1.0  # an estimator cannot beat the bound
    assert w.std_over_crlb_std < 10.0
    assert abs(w.bias) < 0.01
    assert w.rmse >= abs(w.bias)
    for name in PARAMETERS:
        assert getattr(strong, name).truth == getattr(sweep.modes[0], name)
    assert abs(strong.phase.bias) < 0.05
    for level in sweep.levels:
        assert level.trials_attempted == 40
        assert level.trials_successful + level.trials_failed == 40
        assert level.estimator_errors == 0
        for m in level.modes:
            assert 0.0 <= m.loss_rate <= 1.0
            assert m.recovered_trials == round(40 * (1 - m.loss_rate))
    d = sweep.to_dict()
    assert d["levels"][0]["snr_db"] == 40.0


def test_estimator_errors_count_as_failed_trials():
    def broken(t, y, k):
        raise RuntimeError("no fit")

    sweep = validation_sweep(snr_db=(30.0,), trials=3, estimator=broken)
    level = sweep.levels[0]
    assert level.estimator_errors == 3
    assert level.trials_failed == 3 and level.trials_successful == 0
    assert all(m.loss_rate == 1.0 for m in level.modes)
    assert np.isnan(level.residual_ratio)


def test_summary_schema_and_run_artifacts(tmp_path):
    sweep = validation_sweep(snr_db=(30.0,), trials=6, seed=3)
    summary = build_summary([sweep])
    assert summary["schema_version"] == 1
    assert summary["command"] == "linear-prediction-validate"
    config = summary["config"]
    assert config["truth"]["modes"][0]["angular_frequency"] == 2.4
    assert config["sampling"]["samples"] == 96
    assert config["seed"] == 3 and config["trials_per_level"] == 6
    assert config["n_components"] == 6
    assert config["mode_matching"]["tolerance_rad_per_time"] == 0.3
    assert summary["runtime"]["backend"] == "cpu"
    assert summary["runtime"]["numpy_version"]
    result = summary["results"][0]
    assert result["trials"]["attempted"] == 6
    mode = result["modes"][0]
    assert mode["statistics_over"] == "recovered_trials"
    for name in PARAMETERS:
        assert set(mode[name]) == {
            "truth",
            "bias",
            "std",
            "rmse",
            "crlb_std",
            "crlb_variance",
            "std_over_crlb_std",
        }
    out = tmp_path / "run"
    written = write_validation_run(out, [sweep])
    assert (out / "summary.json").exists()
    assert written["artifacts"]["summary"] == "summary.json"
    figure = written["artifacts"]["figure"]
    assert figure is None or (out / figure).exists()
    with pytest.raises(ValueError):
        build_summary([])


def test_invalid_inputs_are_rejected():
    fx = synthetic_modes_trace(96)
    with pytest.raises(ValueError):
        cramer_rao_bounds(fx.modes, fx.constant, fx.time, 0.0)
    with pytest.raises(ValueError):
        validation_sweep(trials=0, snr_db=(40.0,))
    with pytest.raises(ValueError):
        validation_sweep(trials=5, snr_db=())
    with pytest.raises(ValueError):
        synthetic_modes_trace(8)


def test_undistorted_residual_ratio_is_near_one():
    sweep = validation_sweep(snr_db=(30.0,), trials=30, seed=9)
    assert 0.7 < sweep.levels[0].residual_ratio < 1.6


def test_distortions_raise_the_residual_ratio():
    for kind, amount in [
        ("chirp", 0.2),
        ("gaussian_envelope", 6.0),
        ("baseline_drift", 0.3),
        ("clip", 1.0),
    ]:
        sweep = validation_sweep(
            snr_db=(30.0,), trials=20, seed=9, distortion=(kind, amount)
        )
        assert sweep.levels[0].residual_ratio > 4.0, kind
        assert sweep.distortion == (kind, amount)


def test_distort_trace_kinds_and_rejection():
    fx = synthetic_modes_trace(96)
    for kind, amount in [
        ("chirp", 0.1),
        ("gaussian_envelope", 5.0),
        ("baseline_drift", 0.2),
        ("clip", 1.0),
        ("glitch", 0.5),
    ]:
        y = distort_trace(fx.time, fx.modes, fx.constant, kind, amount)
        assert y.shape == fx.time.shape
        assert not np.allclose(y, fx.clean), kind
    with pytest.raises(ValueError):
        distort_trace(fx.time, fx.modes, fx.constant, "square", 1.0)
