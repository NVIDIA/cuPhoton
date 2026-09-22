# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from cuphoton.xray import linear_prediction as lp
from cuphoton.xray.linear_prediction import (
    amplitudes_and_phases_cupy,
    amplitudes_and_phases_numpy,
    benchmark_linear_prediction_fixed_stages,
    benchmark_linear_prediction_p1_batch,
    benchmark_linear_prediction_p2,
    benchmark_linear_prediction_runtime_bridge,
    benchmark_linear_prediction_savgol,
    benchmark_linear_prediction_variable_artifacts,
    benchmark_linear_prediction_variable_p2,
    benchmark_linear_prediction_variable_stages,
    benchmark_prediction_roots,
    compare_cpu_gpu,
    linear_prediction_batched_legacy_rows_cupy,
    linear_prediction_batched_legacy_tiles_cupy,
    linear_prediction_legacy_rows_from_artifacts_cupy,
    linear_prediction_legacy_rows_from_artifacts_numpy,
    linear_prediction_mode_batch_from_roots_cupy,
    linear_prediction_mode_batch_from_roots_numpy,
    linear_prediction_modes_from_roots_cupy,
    linear_prediction_modes_from_roots_numpy,
    linear_prediction_numpy,
    linear_prediction_p2_artifacts_numpy,
    linear_prediction_variable_artifacts_cupy_batched,
    model_order_sweep,
    synthetic_prediction_coefficients,
    synthetic_trace,
    synthetic_trace_batch,
)


def test_amplitudes_and_phases_numpy_all_coefficient_cases():
    amplitudes, phases = amplitudes_and_phases_numpy(
        [0, 3, 0, 3],
        [0, 0, 4, 4],
    )

    np.testing.assert_allclose(amplitudes, [0, 3, 4, 5])
    np.testing.assert_allclose(phases, [0, 0, pi / 2, np.arctan2(4, 3)])


def test_amplitudes_and_phases_numpy_negative_single_coefficients():
    amplitudes, phases = amplitudes_and_phases_numpy(
        [-3, 0],
        [0, -4],
    )

    np.testing.assert_allclose(amplitudes, [3, 4])
    np.testing.assert_allclose(phases, [pi, -pi / 2])


def test_amplitudes_and_phases_cupy_matches_numpy():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    amplitudes, phases = amplitudes_and_phases_cupy(
        cupy.asarray([0, 3, 0, 3]),
        cupy.asarray([0, 0, 4, 4]),
    )

    np.testing.assert_allclose(cupy.asnumpy(amplitudes), [0, 3, 4, 5])
    np.testing.assert_allclose(
        cupy.asnumpy(phases),
        [0, 0, pi / 2, np.arctan2(4, 3)],
    )


def test_linear_prediction_numpy_reconstructs_synthetic_trace():
    time, trace = synthetic_trace(samples=96)

    result = linear_prediction_numpy(time, trace, n_components=8)

    assert result.backend == "cpu"
    assert result.reconstruction.shape == trace.shape
    assert result.frequency.shape == (1000,)
    assert result.chi2 < 1e-20
    assert 0 < result.selected_model_order <= 8
    assert result.elapsed_s > 0
    assert result.roots_stats is not None
    assert result.roots_stats.backend == "eigvals"
    assert result.roots_stats.array_module == "numpy"
    assert result.roots_stats.matrix_size > 0
    assert result.roots_stats.row_count == 1
    assert result.roots_stats.failures == 0
    assert result.decaying_root_count >= len(result.decay)
    assert np.all(result.decay >= 0)
    np.testing.assert_allclose(result.reconstruction, trace, atol=1e-10)


def test_linear_prediction_numpy_reports_fit_diagnostics():
    time, trace = synthetic_trace(samples=32)

    result = linear_prediction_numpy(
        time,
        trace,
        n_components=4,
        fit_diagnostics=True,
    )

    diagnostics = result.diagnostics
    assert diagnostics is not None
    residual = (result.reconstruction - trace)[:-1]
    assert diagnostics.trace_std == pytest.approx(np.std(trace))
    assert diagnostics.residual_std == pytest.approx(np.std(residual))
    assert diagnostics.relative_residual == pytest.approx(
        np.std(residual) / np.std(trace)
    )
    assert diagnostics.chi2 == result.chi2
    assert diagnostics.selected_model_order == result.selected_model_order
    assert diagnostics.mode_count == len(result.angular_frequency)
    assert diagnostics.max_amplitude == pytest.approx(
        np.max(result.amplitude)
    )

    p1_threshold = np.max(result.singular_values) * 1e-12
    assert diagnostics.p1_rank == np.count_nonzero(
        np.isfinite(result.singular_values)
        & (result.singular_values > p1_threshold)
    )
    assert diagnostics.p1_rank > diagnostics.selected_model_order
    assert diagnostics.p1_singular_value_ratio == pytest.approx(
        np.min(result.singular_values) / np.max(result.singular_values)
    )
    assert diagnostics.p1_condition == pytest.approx(
        np.max(result.singular_values) / np.min(result.singular_values)
    )

    design, _fit_time = lp._p2_design_matrix(
        np,
        np.asarray(time),
        result.decay,
        result.angular_frequency,
        dtype=np.float64,
    )
    _coefficients, _residuals, rank, singular_values = np.linalg.lstsq(
        design,
        trace,
        rcond=None,
    )
    assert diagnostics.p2_rank == rank
    np.testing.assert_allclose(
        diagnostics.p2_singular_values,
        singular_values,
    )
    assert diagnostics.p2_singular_value_ratio == pytest.approx(
        singular_values[-1] / singular_values[0]
    )
    assert diagnostics.p2_condition == pytest.approx(
        singular_values[0] / singular_values[-1]
    )


def test_fit_diagnostics_handle_zero_variance_trace():
    time = np.linspace(0.0, 3.1, 32)
    trace = np.full_like(time, 3.0)

    result = linear_prediction_numpy(
        time,
        trace,
        n_components=4,
        fit_diagnostics=True,
    )

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.trace_std == 0.0
    assert np.isfinite(diagnostics.residual_std)
    assert diagnostics.relative_residual is None


def test_singular_value_diagnostics_handle_empty_and_zero_spectra():
    assert (
        lp._numerical_rank_from_singular_values(
            [],
            relative_tolerance=1e-12,
        )
        == 0
    )
    assert lp._singular_value_ratio_and_condition([]) == (None, None)

    assert (
        lp._numerical_rank_from_singular_values(
            [4.0, 1e-13, 0.0],
            relative_tolerance=1e-12,
        )
        == 1
    )
    ratio, condition = lp._singular_value_ratio_and_condition([4.0, 0.0])
    assert ratio == 0.0
    assert condition == float("inf")

    ratio, condition = lp._singular_value_ratio_and_condition([4.0, np.nan])
    assert ratio is None
    assert condition == float("inf")


def test_p2_ridge_alpha_zero_preserves_exact_results():
    time, trace = synthetic_trace(samples=64)

    default = linear_prediction_numpy(time, trace, n_components=8)
    explicit_zero = linear_prediction_numpy(
        time,
        trace,
        n_components=8,
        p2_ridge_alpha=0.0,
    )

    for name in (
        "time_components",
        "reconstruction",
        "frequency",
        "spectrum_components",
        "spectrum_total",
        "angular_frequency",
        "decay",
        "amplitude",
        "phase",
        "singular_values",
    ):
        np.testing.assert_array_equal(
            getattr(explicit_zero, name),
            getattr(default, name),
        )
    assert explicit_zero.chi2 == default.chi2
    assert explicit_zero.selected_model_order == default.selected_model_order
    assert explicit_zero.decaying_root_count == default.decaying_root_count
    assert explicit_zero.diagnostics is None
    assert default.diagnostics is None

    default_artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        default.decay,
        default.angular_frequency,
    )
    explicit_zero_artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        default.decay,
        default.angular_frequency,
        p2_ridge_alpha=0.0,
    )
    for name in (
        "coefficients",
        "amplitude",
        "phase",
        "time_components",
        "reconstruction",
        "frequency",
        "spectrum_components",
        "spectrum_total",
        "frequency_centers",
        "chi2",
    ):
        np.testing.assert_array_equal(
            getattr(explicit_zero_artifacts, name),
            getattr(default_artifacts, name),
        )


@pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf, -np.inf])
def test_p2_ridge_rejects_negative_and_nonfinite_alpha(alpha):
    time, trace = synthetic_trace(samples=32)

    with pytest.raises(ValueError, match="finite nonnegative"):
        linear_prediction_numpy(
            time,
            trace,
            n_components=4,
            p2_ridge_alpha=alpha,
        )
    with pytest.raises(ValueError, match="finite nonnegative"):
        linear_prediction_p2_artifacts_numpy(
            time,
            trace,
            np.asarray([0.1]),
            np.asarray([1.0]),
            p2_ridge_alpha=alpha,
        )


def test_p2_ridge_rejects_nonfinite_scaled_penalty():
    time, trace = synthetic_trace(samples=32)

    with pytest.raises(ValueError, match="nonfinite scaled penalty"):
        linear_prediction_numpy(
            time,
            trace,
            n_components=4,
            p2_ridge_alpha=np.finfo(np.float64).max,
        )


def test_p2_ridge_does_not_penalize_constant_intercept():
    time = np.linspace(0.0, 6.3, 64)
    trace = np.full_like(time, 4.25)

    artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        np.asarray([0.1]),
        np.asarray([0.0]),
        p2_ridge_alpha=1.0,
    )

    assert artifacts.coefficients[-1] == pytest.approx(4.25)
    np.testing.assert_allclose(artifacts.coefficients[:-1], 0.0, atol=1e-14)
    np.testing.assert_allclose(artifacts.reconstruction, trace, atol=1e-14)


def test_p2_ridge_stabilizes_ill_conditioned_known_modes():
    time = np.linspace(0.0, 20.0, 160)
    decay = np.asarray([0.03, 0.03])
    angular_frequency = np.asarray([1.2, 1.2 + 1e-10])
    clean = 0.4 + 1.7 * np.exp(-decay[0] * time) * np.cos(
        angular_frequency[0] * time + 0.35
    )
    trace = clean + 1e-5 * np.sin(2.73 * time) + 7e-6 * np.cos(3.11 * time)

    design, _fit_time = lp._p2_design_matrix(
        np,
        time,
        decay,
        angular_frequency,
        dtype=np.float64,
    )
    assert np.linalg.cond(design) > 1e8

    unregularized = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        decay,
        angular_frequency,
    )
    alpha = 1e-6
    regularized = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        decay,
        angular_frequency,
        p2_ridge_alpha=alpha,
    )

    modal_design = design[:, :-1]
    penalty_weight = alpha * np.mean(np.sum(modal_design**2, axis=0))
    penalty_rows = np.zeros((modal_design.shape[1], design.shape[1]))
    penalty_rows[:, :-1] = np.eye(modal_design.shape[1]) * np.sqrt(
        penalty_weight
    )
    expected_coefficients = np.linalg.lstsq(
        np.concatenate((design, penalty_rows), axis=0),
        np.concatenate((trace, np.zeros(modal_design.shape[1]))),
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        regularized.coefficients,
        expected_coefficients,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.linalg.norm(regularized.coefficients[:-1]) < (
        np.linalg.norm(unregularized.coefficients[:-1]) / 100
    )
    assert np.max(regularized.amplitude) < 2.0
    assert np.max(unregularized.amplitude) > 100.0
    assert np.sum(regularized.amplitude) == pytest.approx(1.7, rel=1e-5)
    assert np.sqrt(np.mean((regularized.reconstruction - clean) ** 2)) < 1e-5


def test_p2_ridge_diagnostics_describe_unregularized_design():
    time, trace = synthetic_trace(samples=64)

    unregularized = linear_prediction_numpy(
        time,
        trace,
        n_components=6,
        fit_diagnostics=True,
    )
    regularized = linear_prediction_numpy(
        time,
        trace,
        n_components=6,
        p2_ridge_alpha=1e-2,
        fit_diagnostics=True,
    )

    assert unregularized.diagnostics is not None
    assert regularized.diagnostics is not None
    assert (
        regularized.diagnostics.p2_rank == unregularized.diagnostics.p2_rank
    )
    np.testing.assert_array_equal(
        regularized.diagnostics.p2_singular_values,
        unregularized.diagnostics.p2_singular_values,
    )
    assert (
        regularized.diagnostics.p2_condition
        == unregularized.diagnostics.p2_condition
    )
    assert not np.array_equal(
        regularized.reconstruction,
        unregularized.reconstruction,
    )


def test_p2_ridge_skips_diagnostics_only_solve_by_default(monkeypatch):
    time, trace = synthetic_trace(samples=64)
    real_lstsq = np.linalg.lstsq
    calls = []

    def counted_lstsq(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return real_lstsq(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", counted_lstsq)

    default = linear_prediction_numpy(
        time,
        trace,
        n_components=6,
        p2_ridge_alpha=1e-2,
    )

    assert default.diagnostics is None
    assert len(calls) == 1

    calls.clear()
    diagnosed = linear_prediction_numpy(
        time,
        trace,
        n_components=6,
        p2_ridge_alpha=1e-2,
        fit_diagnostics=True,
    )

    assert diagnosed.diagnostics is not None
    assert len(calls) == 2


def test_synthetic_trace_batch_shape_and_variation():
    time, traces = synthetic_trace_batch(samples=32, traces=4)

    assert time.shape == (32,)
    assert traces.shape == (4, 32)
    assert not np.allclose(traces[0], traces[1])


def test_synthetic_prediction_coefficients_shape():
    coefficients = synthetic_prediction_coefficients(
        samples=32,
        traces=4,
        n_components=4,
    )

    assert coefficients.shape == (4, 23)


def test_p1_batch_benchmark_can_skip_gpu():
    result = benchmark_linear_prediction_p1_batch(
        samples=32,
        traces=4,
        n_components=4,
        repeat=1,
        run_gpu=False,
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.fit.requested_components == 4
    assert result.fit.matrix_size == 23
    assert result.fit.root_count == 23
    assert result.fit.selected_model_order == 4
    assert result.fit.selected_root_count > 0
    assert result.fit.decaying_root_count >= result.fit.selected_root_count
    assert result.fit.filtered_root_count >= 0
    assert result.fit.chi2 >= 0
    assert result.fit.rms_residual >= 0
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.gpu_error is None


def test_p1_batch_benchmark_roots_backend_can_skip_gpu():
    result = benchmark_linear_prediction_p1_batch(
        samples=32,
        traces=4,
        n_components=4,
        repeat=1,
        run_gpu=False,
        roots_backend="roots",
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.fit.requested_components == 4
    assert result.fit.matrix_size == 23
    assert result.fit.selected_model_order == 4
    assert result.fit.decaying_root_count >= result.fit.selected_root_count
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.max_abs_eigenvalue_diff is None
    assert result.gpu_error is None


def test_p2_benchmark_can_skip_gpu():
    result = benchmark_linear_prediction_p2(
        samples=32,
        traces=4,
        n_components=4,
        repeat=1,
        run_gpu=False,
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.modes > 0
    assert result.design_columns == result.modes * 2 + 1
    assert result.fit.selected_root_count == result.modes
    assert result.fit.decaying_root_count >= result.fit.selected_root_count
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.gpu_batch_speedup is None
    assert result.max_abs_reconstruction_diff is None
    assert result.rms_reconstruction_diff is None
    assert result.gpu_error is None


def test_savgol_benchmark_can_skip_gpu():
    result = benchmark_linear_prediction_savgol(
        samples=32,
        traces=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.window_length == 7
    assert result.polyorder == 3
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.gpu_batch_speedup is None
    assert result.max_abs_filter_diff is None
    assert result.rms_filter_diff is None
    assert result.gpu_error is None


def test_fixed_stages_benchmark_can_skip_gpu_case():
    result = benchmark_linear_prediction_fixed_stages(
        samples=32,
        traces=4,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.modes >= 0
    assert result.design_columns == result.modes * 2 + 1
    assert result.fit.selected_root_count == result.modes
    assert result.fit.decaying_root_count >= result.fit.selected_root_count
    assert result.window_length == 7
    assert result.polyorder == 3
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.gpu_batch_speedup is None
    assert result.max_abs_filter_diff is None
    assert result.rms_filter_diff is None
    assert result.max_abs_reconstruction_diff is None
    assert result.rms_reconstruction_diff is None
    assert result.gpu_error is None


def test_variable_p2_benchmark_can_skip_gpu():
    result = benchmark_linear_prediction_variable_p2(
        samples=32,
        traces=4,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.batched_solver == "pinv"
    assert result.window_length == 7
    assert result.polyorder == 3
    assert len(result.fits) == 4
    assert all(fit.requested_components == 4 for fit in result.fits)
    assert all(
        fit.decaying_root_count >= fit.selected_root_count
        for fit in result.fits
    )
    assert result.mode_count_min >= 0
    assert result.mode_count_max >= result.mode_count_min
    assert result.mode_count_unique
    assert tuple(group.mode_count for group in result.mode_groups) == (
        result.mode_count_unique
    )
    assert sum(group.trace_count for group in result.mode_groups) == (
        result.traces
    )
    assert all(
        group.design_columns == group.mode_count * 2 + 1
        for group in result.mode_groups
    )
    assert result.max_design_columns == result.mode_count_max * 2 + 1
    assert result.padded_design_entries == (
        result.traces * result.max_design_columns
    )
    assert result.grouped_design_entries == sum(
        group.trace_count * group.design_columns
        for group in result.mode_groups
    )
    assert result.padding_overhead_ratio == pytest.approx(
        result.padded_design_entries / result.grouped_design_entries
    )
    assert result.mode_reference_elapsed_s > 0
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.gpu_batch_speedup is None
    assert result.max_abs_reconstruction_diff is None
    assert result.rms_reconstruction_diff is None
    assert result.gpu_error is None


def test_variable_p2_benchmark_grouped_solver_can_skip_gpu():
    result = benchmark_linear_prediction_variable_p2(
        samples=32,
        traces=4,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
        batched_solver="grouped-pinv",
    )

    assert result.batched_solver == "grouped-pinv"
    assert result.mode_count_unique
    assert result.gpu_batched_best_s is None
    assert result.gpu_error is None


def test_variable_p2_benchmark_grouped_normal_can_skip_gpu():
    result = benchmark_linear_prediction_variable_p2(
        samples=32,
        traces=4,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
        batched_solver="grouped-normal",
    )

    assert result.batched_solver == "grouped-normal"
    assert result.mode_count_unique
    assert result.gpu_batched_best_s is None
    assert result.gpu_error is None


def test_variable_p2_benchmark_grouped_qr_can_skip_gpu():
    result = benchmark_linear_prediction_variable_p2(
        samples=32,
        traces=4,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
        batched_solver="grouped-qr",
    )

    assert result.batched_solver == "grouped-qr"
    assert result.mode_count_unique
    assert result.gpu_batched_best_s is None
    assert result.gpu_error is None


def test_variable_p2_benchmark_accepts_trace_batch():
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)

    result = benchmark_linear_prediction_variable_p2(
        time=time,
        trace_rows=trace_rows,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
        batched_solver="grouped-pinv",
    )

    assert result.samples == 32
    assert result.traces == 2
    assert result.batched_solver == "grouped-pinv"
    assert result.mode_count_unique
    assert result.gpu_error is None


def test_variable_stages_benchmark_can_skip_gpu():
    result = benchmark_linear_prediction_variable_stages(
        samples=32,
        traces=4,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
        batched_solver="grouped-pinv",
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.batched_solver == "grouped-pinv"
    assert result.mode_count_unique
    assert len(result.fits) == 4
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.gpu_batch_speedup is None
    assert result.max_abs_filter_diff is None
    assert result.rms_filter_diff is None
    assert result.max_abs_reconstruction_diff is None
    assert result.rms_reconstruction_diff is None
    assert result.gpu_error is None


def test_variable_artifacts_benchmark_can_skip_gpu():
    result = benchmark_linear_prediction_variable_artifacts(
        samples=32,
        traces=4,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
        batched_solver="grouped-pinv",
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.batched_solver == "grouped-pinv"
    assert result.mode_count_unique
    assert len(result.fits) == 4
    assert result.cpu_serial_best_s > 0
    assert result.gpu_serial_best_s is None
    assert result.gpu_batched_best_s is None
    assert result.gpu_batch_speedup is None
    assert result.max_abs_filter_diff is None
    assert result.max_abs_coefficient_diff is None
    assert result.max_abs_amplitude_diff is None
    assert result.max_abs_phase_diff is None
    assert result.max_abs_frequency_center_diff is None
    assert result.max_abs_time_component_diff is None
    assert result.max_abs_reconstruction_diff is None
    assert result.rms_reconstruction_diff is None
    assert result.max_abs_spectrum_component_diff is None
    assert result.max_abs_spectrum_total_diff is None
    assert result.max_abs_chi2_diff is None
    assert result.gpu_error is None


def test_runtime_bridge_benchmark_can_skip_gpu():
    result = benchmark_linear_prediction_runtime_bridge(
        samples=32,
        tiles=2,
        rows_per_tile=2,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=False,
        batched_solver="grouped-pinv",
    )

    assert result.samples == 32
    assert result.tiles == 2
    assert result.rows_per_tile == 2
    assert result.traces == 4
    assert result.components == 4
    assert result.batched_solver == "grouped-pinv"
    assert result.mode_count_unique
    assert len(result.fits) == 4
    assert result.p1_reference_elapsed_s > 0
    assert result.gpu_serial_per_tile_best_s is None
    assert result.gpu_row_batched_per_tile_best_s is None
    assert result.gpu_multi_tile_grouped_best_s is None
    assert result.row_batched_speedup is None
    assert result.multi_tile_speedup is None
    assert result.multi_vs_row_batched_speedup is None
    assert result.max_abs_frequency_center_diff is None
    assert result.max_abs_time_component_diff is None
    assert result.max_abs_reconstruction_diff is None
    assert result.max_abs_amplitude_diff is None
    assert result.max_abs_phase_diff is None
    assert result.max_abs_chi2_diff is None
    assert result.gpu_error is None


def test_p2_artifact_chi2_matches_linear_prediction_result():
    time, trace = synthetic_trace(samples=32)
    reference = linear_prediction_numpy(time, trace, n_components=4)

    artifacts = linear_prediction_p2_artifacts_numpy(
        time=time,
        trace=trace,
        decay=reference.decay,
        angular_frequency=reference.angular_frequency,
    )

    np.testing.assert_allclose(artifacts.chi2, reference.chi2, rtol=1e-12)


def test_modes_from_roots_numpy_matches_linear_prediction_result():
    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )
    reference = linear_prediction_numpy(time, trace, n_components=4)

    modes = linear_prediction_modes_from_roots_numpy(
        time,
        p1.eigenvalues,
        singular_value_count=len(p1.singular_values),
    )

    np.testing.assert_allclose(modes.decay, reference.decay)
    np.testing.assert_allclose(
        modes.angular_frequency,
        reference.angular_frequency,
    )
    assert modes.decaying_root_count == reference.decaying_root_count


def test_modes_from_roots_rejects_bad_inputs_case():
    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )

    with pytest.raises(ValueError, match="time must contain at least two"):
        linear_prediction_modes_from_roots_numpy(
            [0.0],
            p1.eigenvalues,
            singular_value_count=len(p1.singular_values),
        )
    with pytest.raises(ValueError, match="singular_value_count"):
        linear_prediction_modes_from_roots_numpy(
            time,
            p1.eigenvalues,
            singular_value_count=-1,
        )


def test_mode_batch_from_roots_numpy_pads_selected_modes():
    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )
    expected = linear_prediction_modes_from_roots_numpy(
        time,
        p1.eigenvalues,
        singular_value_count=len(p1.singular_values),
    )

    batch = linear_prediction_mode_batch_from_roots_numpy(
        time,
        (p1.eigenvalues, p1.eigenvalues),
        (0, p1.singular_values),
    )

    assert batch.decay.shape == (2, expected.decay.shape[0])
    assert batch.angular_frequency.shape == batch.decay.shape
    np.testing.assert_array_equal(batch.mode_counts, [0, len(expected.decay)])
    np.testing.assert_array_equal(
        batch.decaying_root_counts,
        [0, expected.decaying_root_count],
    )
    np.testing.assert_array_equal(
        batch.decay[0],
        np.zeros_like(expected.decay),
    )
    np.testing.assert_allclose(batch.decay[1], expected.decay)
    np.testing.assert_allclose(
        batch.angular_frequency[1],
        expected.angular_frequency,
    )


def test_mode_batch_from_roots_rejects_count_length_mismatch():
    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )

    with pytest.raises(ValueError, match="same length"):
        linear_prediction_mode_batch_from_roots_numpy(
            time,
            (p1.eigenvalues, p1.eigenvalues),
            (p1.singular_values,),
        )


def test_mode_batch_from_roots_numpy_uses_array_length_for_counts():
    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )
    expected = linear_prediction_modes_from_roots_numpy(
        time,
        p1.eigenvalues,
        singular_value_count=1,
    )

    batch = linear_prediction_mode_batch_from_roots_numpy(
        time,
        (p1.eigenvalues,),
        (np.asarray([999.0]),),
    )

    np.testing.assert_array_equal(batch.mode_counts, [len(expected.decay)])
    np.testing.assert_allclose(batch.decay[0], expected.decay)
    np.testing.assert_allclose(
        batch.angular_frequency[0],
        expected.angular_frequency,
    )


def test_modes_from_roots_cupy_matches_numpy_when_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )
    expected = linear_prediction_modes_from_roots_numpy(
        time,
        p1.eigenvalues,
        singular_value_count=len(p1.singular_values),
    )
    actual = linear_prediction_modes_from_roots_cupy(
        cupy.asarray(time),
        cupy.asarray(p1.eigenvalues),
        singular_value_count=len(p1.singular_values),
    )

    np.testing.assert_allclose(cupy.asnumpy(actual.decay), expected.decay)
    np.testing.assert_allclose(
        cupy.asnumpy(actual.angular_frequency),
        expected.angular_frequency,
    )
    assert actual.decaying_root_count == expected.decaying_root_count


def test_mode_batch_from_roots_cupy_matches_numpy_when_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )
    expected = linear_prediction_mode_batch_from_roots_numpy(
        time,
        (p1.eigenvalues, p1.eigenvalues),
        (0, p1.singular_values),
    )

    actual = linear_prediction_mode_batch_from_roots_cupy(
        cupy.asarray(time),
        (cupy.asarray(p1.eigenvalues), cupy.asarray(p1.eigenvalues)),
        (cupy.asarray(0), cupy.asarray(p1.singular_values)),
    )

    np.testing.assert_allclose(cupy.asnumpy(actual.decay), expected.decay)
    np.testing.assert_allclose(
        cupy.asnumpy(actual.angular_frequency),
        expected.angular_frequency,
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(actual.mode_counts),
        expected.mode_counts,
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(actual.decaying_root_counts),
        expected.decaying_root_counts,
    )


def test_p2_artifact_public_numpy_matches_linear_prediction_result():
    time, trace = synthetic_trace(samples=32)
    reference = linear_prediction_numpy(time, trace, n_components=4)

    artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        reference.decay,
        reference.angular_frequency,
    )

    np.testing.assert_allclose(artifacts.amplitude, reference.amplitude)
    np.testing.assert_allclose(artifacts.phase, reference.phase)
    np.testing.assert_allclose(
        artifacts.frequency_centers,
        reference.frequency[np.argmax(reference.spectrum_components, axis=0)],
    )
    np.testing.assert_allclose(
        artifacts.reconstruction,
        reference.reconstruction,
    )
    np.testing.assert_allclose(
        artifacts.spectrum_components,
        reference.spectrum_components,
    )
    np.testing.assert_allclose(
        artifacts.spectrum_total,
        reference.spectrum_total,
    )
    np.testing.assert_allclose(artifacts.chi2, reference.chi2)


def test_legacy_rows_from_artifacts_numpy_matches_linear_prediction_result():
    time, trace = synthetic_trace(samples=32)
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=trace,
        n_components=4,
    )
    modes = linear_prediction_mode_batch_from_roots_numpy(
        time,
        (p1.eigenvalues,),
        (p1.singular_values,),
    )
    mode_count = int(modes.mode_counts[0])
    artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        modes.decay[0, :mode_count],
        modes.angular_frequency[0, :mode_count],
    )
    batched_artifacts = lp._stack_p2_artifact_rows(
        np,
        [artifacts],
        max_modes=int(modes.decay.shape[1]),
    )
    reference = linear_prediction_numpy(time, trace, n_components=4)

    rows = linear_prediction_legacy_rows_from_artifacts_numpy(
        time,
        batched_artifacts,
        modes.decay,
        modes.angular_frequency,
        modes.mode_counts,
        (p1.singular_values,),
    )

    assert len(rows) == 1
    legacy = rows[0]
    assert len(legacy) == 13
    np.testing.assert_allclose(legacy[0], artifacts.frequency_centers)
    np.testing.assert_allclose(legacy[1], reference.time)
    np.testing.assert_allclose(legacy[2], reference.time_components)
    np.testing.assert_allclose(legacy[3], reference.reconstruction)
    np.testing.assert_allclose(legacy[4], reference.frequency)
    np.testing.assert_allclose(legacy[5], reference.spectrum_components)
    np.testing.assert_allclose(legacy[6], reference.spectrum_total)
    np.testing.assert_allclose(legacy[7], reference.angular_frequency)
    np.testing.assert_allclose(legacy[8], reference.decay)
    np.testing.assert_allclose(legacy[9], reference.amplitude)
    np.testing.assert_allclose(legacy[10], reference.phase)
    np.testing.assert_allclose(legacy[11], reference.chi2)
    np.testing.assert_allclose(legacy[12], reference.singular_values)


def test_legacy_rows_from_artifacts_numpy_preserves_zero_mode_spectrum():
    time, trace = synthetic_trace(samples=32)
    artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    )
    batched_artifacts = lp._stack_p2_artifact_rows(
        np,
        [artifacts],
        max_modes=0,
    )

    rows = linear_prediction_legacy_rows_from_artifacts_numpy(
        time,
        batched_artifacts,
        np.zeros((1, 0), dtype=np.float64),
        np.zeros((1, 0), dtype=np.float64),
        np.asarray([0], dtype=np.int64),
    )

    legacy = rows[0]
    assert legacy[0].shape == (0,)
    assert legacy[2].shape == (time.shape[0], 0)
    assert legacy[5].shape == (1000, 1)
    np.testing.assert_allclose(legacy[5][:, 0], legacy[4])
    assert legacy[7].shape == (0,)
    assert legacy[8].shape == (0,)
    assert legacy[9].shape == (0,)
    assert legacy[10].shape == (0,)
    assert legacy[12].shape == (0,)


def test_legacy_rows_from_artifacts_rejects_singular_value_mismatch():
    time, trace = synthetic_trace(samples=32)
    artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    )
    batched_artifacts = lp._stack_p2_artifact_rows(
        np,
        [artifacts],
        max_modes=0,
    )

    with pytest.raises(ValueError, match="singular_value_rows"):
        linear_prediction_legacy_rows_from_artifacts_numpy(
            time,
            batched_artifacts,
            np.zeros((1, 0), dtype=np.float64),
            np.zeros((1, 0), dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            (),
        )


def test_legacy_rows_from_artifacts_rejects_sample_shape_mismatch():
    time, trace = synthetic_trace(samples=32)
    artifacts = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    )
    batched_artifacts = lp._stack_p2_artifact_rows(
        np,
        [artifacts],
        max_modes=0,
    )

    with pytest.raises(ValueError, match="reconstruction"):
        linear_prediction_legacy_rows_from_artifacts_numpy(
            time[:-1],
            batched_artifacts,
            np.zeros((1, 0), dtype=np.float64),
            np.zeros((1, 0), dtype=np.float64),
            np.asarray([0], dtype=np.int64),
        )


def test_p2_artifact_chi2_rejects_two_sample_trace():
    with pytest.raises(ValueError, match="at least three samples"):
        lp._linear_prediction_p2_artifacts_impl(
            xp=np,
            time=np.asarray([0.0, 1.0]),
            trace=np.asarray([1.0, 1.0]),
            decay=np.asarray([]),
            angular_frequency=np.asarray([]),
        )


def test_variable_artifacts_benchmark_runs_gpu_path_when_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    result = benchmark_linear_prediction_variable_artifacts(
        samples=32,
        traces=2,
        n_components=4,
        window_length=7,
        polyorder=3,
        repeat=1,
        run_gpu=True,
        batched_solver="grouped-pinv",
    )

    assert result.gpu_error is None
    assert result.gpu_serial_best_s is not None
    assert result.gpu_batched_best_s is not None
    assert result.gpu_batch_speedup is not None
    assert result.gpu_batch_speedup > 0
    assert result.max_abs_filter_diff is not None
    assert result.max_abs_filter_diff < 1e-12
    assert result.max_abs_coefficient_diff is not None
    assert result.max_abs_coefficient_diff < 1e-12
    assert result.max_abs_amplitude_diff is not None
    assert result.max_abs_amplitude_diff < 1e-12
    assert result.max_abs_phase_diff is not None
    assert result.max_abs_phase_diff < 1e-12
    assert result.max_abs_frequency_center_diff == 0.0
    assert result.max_abs_time_component_diff is not None
    assert result.max_abs_time_component_diff < 1e-12
    assert result.max_abs_reconstruction_diff is not None
    assert result.max_abs_reconstruction_diff < 1e-12
    assert result.rms_reconstruction_diff is not None
    assert result.rms_reconstruction_diff < 1e-12
    assert result.max_abs_spectrum_component_diff is not None
    assert result.max_abs_spectrum_component_diff < 1e-9
    assert result.max_abs_spectrum_total_diff is not None
    assert result.max_abs_spectrum_total_diff < 1e-9
    assert result.max_abs_chi2_diff is not None
    assert result.max_abs_chi2_diff < 1e-12


def test_public_variable_artifacts_cupy_batched_matches_cpu_reference():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    from scipy.signal import savgol_filter

    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    filtered_rows = savgol_filter(trace_rows, 7, 3, axis=1)
    p1_rows = [
        lp._linear_prediction_p1_impl(
            xp=np,
            time=time,
            trace=filtered,
            n_components=4,
        )
        for filtered in filtered_rows
    ]
    references = [
        linear_prediction_numpy(time, filtered, n_components=4)
        for filtered in filtered_rows
    ]
    modes = linear_prediction_mode_batch_from_roots_cupy(
        cupy.asarray(time),
        tuple(cupy.asarray(item.eigenvalues) for item in p1_rows),
        tuple(cupy.asarray(item.singular_values) for item in p1_rows),
    )

    filtered, artifacts = linear_prediction_variable_artifacts_cupy_batched(
        time,
        trace_rows,
        modes.decay,
        modes.angular_frequency,
        modes.mode_counts,
        solver="grouped-pinv",
        window_length=7,
        polyorder=3,
    )
    cupy.cuda.Stream.null.synchronize()
    mode_counts = cupy.asnumpy(modes.mode_counts)

    np.testing.assert_allclose(
        cupy.asnumpy(filtered),
        filtered_rows,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(artifacts.reconstruction[0]),
        references[0].reconstruction,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(artifacts.amplitude[0, : mode_counts[0]]),
        references[0].amplitude,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(artifacts.chi2[0]),
        references[0].chi2,
        atol=1e-12,
    )

    legacy_rows = linear_prediction_legacy_rows_from_artifacts_cupy(
        cupy.asarray(time),
        artifacts,
        modes.decay,
        modes.angular_frequency,
        modes.mode_counts,
        tuple(cupy.asarray(item.singular_values) for item in p1_rows),
    )
    np.testing.assert_allclose(
        cupy.asnumpy(legacy_rows[0][0]),
        references[0].frequency[
            np.argmax(references[0].spectrum_components, axis=0)
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(legacy_rows[0][3]),
        references[0].reconstruction,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(legacy_rows[0][8]),
        references[0].decay,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(legacy_rows[0][9]),
        references[0].amplitude,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(legacy_rows[0][10]),
        references[0].phase,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(legacy_rows[0][11]),
        references[0].chi2,
        atol=1e-12,
    )

    batched_filtered, batched_legacy_rows = (
        linear_prediction_batched_legacy_rows_cupy(
            cupy.asarray(time),
            cupy.asarray(trace_rows),
            tuple(cupy.asarray(item.eigenvalues) for item in p1_rows),
            tuple(cupy.asarray(item.singular_values) for item in p1_rows),
            solver="grouped-pinv",
            window_length=7,
            polyorder=3,
        )
    )
    cupy.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(
        cupy.asnumpy(batched_filtered),
        filtered_rows,
        atol=1e-12,
    )
    assert len(batched_legacy_rows) == len(legacy_rows)
    for actual, expected in zip(
        batched_legacy_rows,
        legacy_rows,
        strict=True,
    ):
        np.testing.assert_allclose(
            cupy.asnumpy(actual[0]),
            cupy.asnumpy(expected[0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            cupy.asnumpy(actual[3]),
            cupy.asnumpy(expected[3]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            cupy.asnumpy(actual[9]),
            cupy.asnumpy(expected[9]),
            atol=1e-12,
        )


def test_public_batched_legacy_tiles_cupy_matches_per_tile_rows():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    from scipy.signal import savgol_filter

    time, trace_rows = synthetic_trace_batch(samples=32, traces=4)
    filtered_rows = savgol_filter(trace_rows, 7, 3, axis=1)
    p1_rows = [
        lp._linear_prediction_p1_impl(
            xp=np,
            time=time,
            trace=filtered,
            n_components=4,
        )
        for filtered in filtered_rows
    ]
    gpu_time = cupy.asarray(time)
    tile_traces = (
        cupy.asarray(trace_rows[:2]),
        cupy.asarray(trace_rows[2:]),
    )
    tile_roots = (
        tuple(cupy.asarray(item.eigenvalues) for item in p1_rows[:2]),
        tuple(cupy.asarray(item.eigenvalues) for item in p1_rows[2:]),
    )
    tile_singular_values = (
        tuple(cupy.asarray(item.singular_values) for item in p1_rows[:2]),
        tuple(cupy.asarray(item.singular_values) for item in p1_rows[2:]),
    )

    expected_tiles = tuple(
        linear_prediction_batched_legacy_rows_cupy(
            gpu_time,
            traces,
            roots,
            singular_values,
            solver="grouped-pinv",
            window_length=7,
            polyorder=3,
        )
        for traces, roots, singular_values in zip(
            tile_traces,
            tile_roots,
            tile_singular_values,
            strict=True,
        )
    )
    actual_filtered, actual_rows = (
        linear_prediction_batched_legacy_tiles_cupy(
            gpu_time,
            tile_traces,
            tile_roots,
            tile_singular_values,
            solver="grouped-pinv",
            window_length=7,
            polyorder=3,
        )
    )
    cupy.cuda.Stream.null.synchronize()

    expected_filtered = tuple(item[0] for item in expected_tiles)
    expected_rows = tuple(item[1] for item in expected_tiles)
    for actual, expected in zip(
        actual_filtered,
        expected_filtered,
        strict=True,
    ):
        np.testing.assert_allclose(
            cupy.asnumpy(actual),
            cupy.asnumpy(expected),
            atol=1e-12,
        )
    for actual_tile, expected_tile in zip(
        actual_rows,
        expected_rows,
        strict=True,
    ):
        for actual_row, expected_row in zip(
            actual_tile,
            expected_tile,
            strict=True,
        ):
            for slot in (0, 2, 3, 9, 10, 11):
                np.testing.assert_allclose(
                    cupy.asnumpy(actual_row[slot]),
                    cupy.asnumpy(expected_row[slot]),
                    atol=1e-12,
                )


def test_batched_legacy_rows_cupy_rejects_trace_count_mismatch():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    filtered = trace_rows[0]
    p1 = lp._linear_prediction_p1_impl(
        xp=np,
        time=time,
        trace=filtered,
        n_components=4,
    )

    with pytest.raises(ValueError, match="one row per P1 row"):
        linear_prediction_batched_legacy_rows_cupy(
            cupy.asarray(time),
            cupy.asarray(trace_rows),
            (cupy.asarray(p1.eigenvalues),),
            (cupy.asarray(p1.singular_values),),
            solver="grouped-pinv",
            window_length=7,
            polyorder=3,
        )


def test_variable_p2_grouped_solver_matches_padded_pinv():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    time = cupy.linspace(0.0, 1.0, 24, dtype=cupy.float64)
    traces = cupy.stack(
        [
            cupy.cos(2.0 * time) + 0.1,
            cupy.cos(1.5 * time) + 0.25 * cupy.cos(3.5 * time),
            cupy.cos(2.2 * time + 0.1) - 0.05,
        ]
    )
    decay = cupy.asarray(
        [
            [0.05, 0.0],
            [0.03, 0.07],
            [0.02, 0.0],
        ],
        dtype=cupy.float64,
    )
    angular_frequency = cupy.asarray(
        [
            [2.0, 0.0],
            [1.5, 3.5],
            [2.2, 0.0],
        ],
        dtype=cupy.float64,
    )
    mode_counts = cupy.asarray([1, 2, 1], dtype=cupy.int64)

    padded = lp._variable_p2_cupy_batched_pinv(
        time,
        traces,
        decay,
        angular_frequency,
        mode_counts,
    )
    grouped = lp._variable_p2_cupy_batched_solver(
        time,
        traces,
        decay,
        angular_frequency,
        mode_counts,
        solver="grouped-pinv",
    )
    grouped_normal = lp._variable_p2_cupy_batched_solver(
        time,
        traces,
        decay,
        angular_frequency,
        mode_counts,
        solver="grouped-normal",
    )
    grouped_qr = lp._variable_p2_cupy_batched_solver(
        time,
        traces,
        decay,
        angular_frequency,
        mode_counts,
        solver="grouped-qr",
    )
    cupy.cuda.Stream.null.synchronize()

    np.testing.assert_allclose(
        cupy.asnumpy(grouped),
        cupy.asnumpy(padded),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(grouped_normal),
        cupy.asnumpy(padded),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(grouped_qr),
        cupy.asnumpy(padded),
        atol=1e-9,
    )


def test_prediction_roots_benchmark_cpu_backends():
    result = benchmark_prediction_roots(
        samples=32,
        traces=4,
        n_components=4,
        repeat=1,
        backends=("numpy-eigvals", "numpy-roots"),
        run_gpu=False,
    )

    assert result.samples == 32
    assert result.traces == 4
    assert result.components == 4
    assert result.repeat == 1
    assert result.fit.requested_components == 4
    assert result.fit.matrix_size == 23
    assert result.fit.selected_model_order == 4
    assert [backend.backend for backend in result.backends] == [
        "numpy-eigvals",
        "numpy-roots",
    ]
    for backend in result.backends:
        assert backend.array_module == "numpy"
        assert backend.matrix_size == 23
        assert backend.row_count == 4
        assert backend.failures == 0
        assert backend.best_s is not None
        assert backend.best_s > 0
        assert backend.max_abs_root_diff is not None
        assert backend.max_abs_root_diff < 1e-10
        assert backend.error is None


def test_model_order_sweep_reports_best_order():
    time, trace = synthetic_trace(samples=48)

    result = model_order_sweep(
        time,
        trace,
        components=(2, 4, 6),
        relative_tolerance=0.05,
    )

    assert result.samples == 48
    assert result.roots_backend == "eigvals"
    assert result.best_components in {2, 4, 6}
    assert result.best_selected_model_order > 0
    assert result.best_rms_residual >= 0
    assert result.best_reconstruction_rms_error >= 0
    assert [entry.components for entry in result.entries] == [2, 4, 6]
    for entry in result.entries:
        assert entry.matrix_size == 35
        assert entry.root_count == 35
        assert 0 < entry.selected_model_order <= entry.components
        assert entry.selected_root_count >= 0
        assert entry.decaying_root_count >= entry.selected_root_count
        assert entry.filtered_root_count >= 0
        assert entry.chi2 >= 0
        assert entry.rms_residual >= 0
        assert entry.reconstruction_rms_error == entry.rms_residual
        assert entry.singular_value_head
        assert entry.singular_value_tail
    assert any(
        entry.decaying_root_count != entry.selected_root_count
        for entry in result.entries
    )


def test_model_order_sweep_chooses_smallest_qualified_order():
    time, trace = synthetic_trace(samples=48)

    result = model_order_sweep(
        time,
        trace,
        components=(6, 4, 2),
        relative_tolerance=1e16,
    )

    assert [entry.components for entry in result.entries] == [6, 4, 2]
    assert result.best_components == 2
    assert result.best_selected_model_order == 2


def test_compare_cpu_gpu_can_skip_gpu():
    comparison = compare_cpu_gpu(
        samples=64,
        n_components=6,
        run_gpu=False,
    )

    assert comparison.cpu.backend == "cpu"
    assert comparison.gpu is None
    assert comparison.max_abs_reconstruction_diff is None
    assert comparison.rms_reconstruction_diff is None


def test_compare_cpu_gpu_reconstruction_matches_when_gpu_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    comparison = compare_cpu_gpu(
        samples=64,
        n_components=6,
        run_gpu=True,
    )

    assert comparison.gpu is not None
    assert comparison.gpu.backend == "gpu"
    assert comparison.max_abs_reconstruction_diff is not None
    assert comparison.max_abs_reconstruction_diff < 1e-8
    assert comparison.rms_reconstruction_diff is not None
    assert comparison.rms_reconstruction_diff < 1e-9


def test_p2_ridge_cupy_matches_numpy_when_gpu_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    time, trace = synthetic_trace(samples=64)
    p1_reference = linear_prediction_numpy(time, trace, n_components=4)
    cpu = linear_prediction_p2_artifacts_numpy(
        time,
        trace,
        p1_reference.decay,
        p1_reference.angular_frequency,
        p2_ridge_alpha=1e-4,
    )
    gpu = lp._linear_prediction_p2_artifacts_impl(
        xp=cupy,
        time=cupy.asarray(time),
        trace=cupy.asarray(trace),
        decay=cupy.asarray(p1_reference.decay),
        angular_frequency=cupy.asarray(p1_reference.angular_frequency),
        p2_ridge_alpha=1e-4,
    )
    gpu_numpy = lp._p2_artifacts_to_numpy(cupy, gpu)

    np.testing.assert_allclose(
        gpu_numpy.coefficients,
        cpu.coefficients,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gpu_numpy.amplitude,
        cpu.amplitude,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gpu_numpy.phase,
        cpu.phase,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gpu_numpy.reconstruction,
        cpu.reconstruction,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(gpu_numpy.chi2, cpu.chi2, rtol=1e-10)


def test_cupy_roots_helpers_preserve_complex_coefficients():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    coefficients = np.asarray(
        [
            [1.2 + 0.1j, -0.4 + 0.3j, 0.05 - 0.2j],
            [0.8 - 0.2j, 0.1 + 0.5j, -0.3 + 0.1j],
        ],
        dtype=np.complex128,
    )
    baseline = lp._roots_numpy_serial(coefficients, backend="eigvals")

    serial = lp._roots_cupy_serial(coefficients, backend="eigvals")
    batched = lp._roots_cupy_batched_eigvals(coefficients)

    np.testing.assert_allclose(serial, baseline, atol=1e-12)
    np.testing.assert_allclose(batched, baseline, atol=1e-12)


def test_p2_impl_applies_ridge_through_shared_solver():
    time, trace = synthetic_trace(samples=48)
    first = linear_prediction_numpy(time, trace, n_components=4)
    decay = first.decay
    angular_frequency = first.angular_frequency
    design, _fit_time = lp._p2_design_matrix(
        np,
        np.asarray(time, dtype=np.float64),
        np.asarray(decay, dtype=np.float64),
        np.asarray(angular_frequency, dtype=np.float64),
        dtype=np.float64,
    )
    for alpha in (0.0, 0.25):
        result = lp._linear_prediction_p2_impl(
            xp=np,
            time=time,
            trace=trace,
            decay=decay,
            angular_frequency=angular_frequency,
            p2_ridge_alpha=alpha,
        )
        expected = lp._p2_lstsq(
            np,
            design,
            np.asarray(trace, dtype=np.float64),
            p2_ridge_alpha=alpha,
            diagnostics=False,
        ).coefficients
        np.testing.assert_allclose(result.coefficients, expected)
    unregularized = lp._linear_prediction_p2_impl(
        xp=np,
        time=time,
        trace=trace,
        decay=decay,
        angular_frequency=angular_frequency,
    )
    ridged = lp._linear_prediction_p2_impl(
        xp=np,
        time=time,
        trace=trace,
        decay=decay,
        angular_frequency=angular_frequency,
        p2_ridge_alpha=0.25,
    )
    assert not np.allclose(unregularized.coefficients, ridged.coefficients)
