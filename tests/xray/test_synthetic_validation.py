# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from types import SimpleNamespace

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


def test_integer_mode_parameters_have_the_same_bounds_as_floats():
    time = np.linspace(0, 25.5, 256)
    actual = cramer_rao_bounds((DampedMode(1, 0, 2, 0),), 0, time, 0.01)
    expected = cramer_rao_bounds(
        (DampedMode(1.0, 0.0, 2.0, 0.0),), 0.0, time, 0.01
    )
    for name in (*PARAMETERS, "constant"):
        np.testing.assert_allclose(actual[name], expected[name])
        assert np.all(np.isfinite(actual[name]))


@pytest.mark.parametrize("start", [0.0, 3.0])
def test_chirp_integrates_the_requested_frequency_ramp(start):
    time = np.linspace(start, start + 10.0, 256)
    # Integrating omega(t) = 2 + .04 * (t - start) gives this phase.
    expected = np.cos(2 * time + 0.02 * (time - start) ** 2 + 0.3)
    actual = distort_trace(time, (DampedMode(1, 0, 2, 0.3),), 0, "chirp", 0.2)
    np.testing.assert_allclose(actual, expected)


def test_matching_recovers_close_modes_independent_of_truth_order():
    modes = (DampedMode(1, 0, 1.0), DampedMode(1, 0, 1.15))
    fitted = np.array([1.05, 0.85])
    decay = np.zeros(2)
    assert match_modes(modes, fitted, decay, tolerance=0.2) == [1, 0]
    assert match_modes(modes[::-1], fitted, decay, tolerance=0.2) == [0, 1]
    assert match_modes(modes, fitted[:1], decay[:1], tolerance=0.2) == [
        0,
        None,
    ]
    assert match_modes(
        modes, np.array([np.nan]), decay[:1], tolerance=0.2
    ) == [None, None]


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


@pytest.mark.parametrize("amplitude", [-1.0, 1.0])
@pytest.mark.parametrize("phase", [0.3, 0.3 + 4 * np.pi])
def test_sweep_uses_the_same_amplitude_phase_convention_for_truth_and_fit(
    amplitude, phase
):
    mode = DampedMode(amplitude, 0.1, 2.0, phase)
    fixture = synthetic_modes_trace(modes=(mode,))

    def exact_estimator(t, y, k):
        return SimpleNamespace(
            amplitude=np.array([mode.amplitude]),
            decay=np.array([mode.decay]),
            angular_frequency=np.array([mode.angular_frequency]),
            phase=np.array([mode.phase]),
            reconstruction=fixture.clean,
        )

    sweep = validation_sweep(
        modes=(mode,),
        snr_db=(30.0,),
        trials=2,
        n_components=1,
        estimator=exact_estimator,
    )
    result = sweep.levels[0].modes[0]
    assert result.recovered_trials == 2
    for name in PARAMETERS:
        stats = getattr(result, name)
        assert stats.bias == pytest.approx(0.0, abs=1e-14)
        assert stats.rmse == pytest.approx(0.0, abs=1e-14)
        assert stats.truth == getattr(sweep.modes[0], name)

    canonical = synthetic_modes_trace(modes=sweep.modes)
    np.testing.assert_allclose(canonical.clean, fixture.clean, atol=1e-14)
    summary = build_summary([sweep])
    truth = summary["config"]["truth"]["modes"][0]
    assert truth["amplitude"] == 1.0
    expected_phase = 0.3 - np.pi if amplitude < 0 else 0.3
    assert truth["phase"] == pytest.approx(expected_phase)
    for name in PARAMETERS:
        assert summary["results"][0]["modes"][0][name]["truth"] == truth[name]


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


@pytest.mark.parametrize("trials", [1, 2])
def test_small_sweeps_write_strict_json_and_two_trials_have_std(
    tmp_path, trials
):
    sweep = validation_sweep(snr_db=(30.0,), trials=trials, seed=3)
    summary = write_validation_run(tmp_path, [sweep])
    encoded = json.dumps(summary, allow_nan=False)
    assert json.loads(encoded) == json.loads(
        (tmp_path / "summary.json").read_text()
    )
    strong = summary["results"][0]["modes"][0]
    assert strong["recovered_trials"] == trials
    assert (strong["angular_frequency"]["std"] is None) == (trials == 1)


def test_failed_sweep_statistics_are_null_in_summary():
    def broken(t, y, k):
        raise RuntimeError("no fit")

    sweep = validation_sweep(snr_db=(30.0,), trials=2, estimator=broken)
    summary = build_summary([sweep])
    json.dumps(summary, allow_nan=False)
    result = summary["results"][0]
    assert result["residual_rms_over_sigma_median"] is None
    assert result["modes"][0]["angular_frequency"]["bias"] is None


@pytest.mark.parametrize(
    "change", [{"samples": 192}, {"seed": 77}, {"match_tolerance": 0.1}]
)
def test_summary_rejects_incompatible_sweep_provenance(change):
    sweep = validation_sweep(snr_db=(30.0,), trials=2)
    with pytest.raises(ValueError, match="summary sweeps must share"):
        build_summary([sweep, replace(sweep, **change)])


def test_summary_preserves_distortion_signal_scale():
    sweep = validation_sweep(snr_db=(30.0,), trials=2)
    distorted = replace(
        sweep, backend="other", distortion=("clip", 0.5), signal_rms=0.1
    )
    summary = build_summary([sweep, distorted])
    assert summary["results"][0]["signal_rms"] == sweep.signal_rms
    assert summary["results"][1]["signal_rms"] == 0.1


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
