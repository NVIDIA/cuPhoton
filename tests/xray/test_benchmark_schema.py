# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict

from cuphoton.xray.linear_prediction import (
    benchmark_linear_prediction_fixed_stages,
    benchmark_linear_prediction_p1_batch,
    benchmark_linear_prediction_p2,
    benchmark_linear_prediction_savgol,
    benchmark_linear_prediction_variable_artifacts,
    benchmark_linear_prediction_variable_p2,
    benchmark_linear_prediction_variable_stages,
    benchmark_prediction_roots,
    model_order_sweep,
    synthetic_trace,
)
from cuphoton.xray.subspace import compare_subspace_methods

FIT_KEYS = {
    "requested_components",
    "selected_model_order",
    "matrix_size",
    "root_count",
    "selected_root_count",
    "decaying_root_count",
    "filtered_root_count",
    "chi2",
    "rms_residual",
}

ROOTS_BACKEND_KEYS = {
    "backend",
    "array_module",
    "batched",
    "matrix_size",
    "row_count",
    "failures",
    "best_s",
    "max_abs_root_diff",
    "error",
}

MODE_GROUP_KEYS = {
    "mode_count",
    "trace_count",
    "design_columns",
}

MODEL_ORDER_ENTRY_KEYS = {
    "components",
    "selected_model_order",
    "matrix_size",
    "root_count",
    "selected_root_count",
    "decaying_root_count",
    "filtered_root_count",
    "chi2",
    "rms_residual",
    "reconstruction_rms_error",
    "singular_value_head",
    "singular_value_tail",
    "singular_value_ratio",
    "elapsed_s",
}

SUBSPACE_METHOD_KEYS = {
    "method",
    "svd_backend",
    "samples",
    "model_order",
    "pencil_rows",
    "svd_rank",
    "singular_value_head",
    "rms_residual",
    "max_abs_reconstruction_diff",
    "elapsed_s",
    "roots_real",
    "roots_imag",
}


def test_p1_batch_benchmark_schema_is_stable_case():
    payload = asdict(
        benchmark_linear_prediction_p1_batch(
            samples=32,
            traces=4,
            n_components=4,
            repeat=1,
            run_gpu=False,
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "components",
        "repeat",
        "fit",
        "cpu_serial_best_s",
        "gpu_serial_best_s",
        "gpu_batched_best_s",
        "gpu_batch_speedup",
        "max_abs_coefficient_diff",
        "max_abs_eigenvalue_diff",
        "gpu_error",
    }
    assert set(payload["fit"]) == FIT_KEYS


def test_p2_benchmark_schema_is_stable():
    payload = asdict(
        benchmark_linear_prediction_p2(
            samples=32,
            traces=4,
            n_components=4,
            repeat=1,
            run_gpu=False,
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "components",
        "modes",
        "design_columns",
        "repeat",
        "fit",
        "cpu_serial_best_s",
        "gpu_serial_best_s",
        "gpu_batched_best_s",
        "gpu_batch_speedup",
        "max_abs_reconstruction_diff",
        "rms_reconstruction_diff",
        "gpu_error",
    }
    assert set(payload["fit"]) == FIT_KEYS


def test_savgol_benchmark_schema_is_stable():
    payload = asdict(
        benchmark_linear_prediction_savgol(
            samples=32,
            traces=4,
            window_length=7,
            polyorder=3,
            repeat=1,
            run_gpu=False,
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "window_length",
        "polyorder",
        "repeat",
        "cpu_serial_best_s",
        "gpu_serial_best_s",
        "gpu_batched_best_s",
        "gpu_batch_speedup",
        "max_abs_filter_diff",
        "rms_filter_diff",
        "gpu_error",
    }


def test_fixed_stages_benchmark_schema_is_stable():
    payload = asdict(
        benchmark_linear_prediction_fixed_stages(
            samples=32,
            traces=4,
            n_components=4,
            window_length=7,
            polyorder=3,
            repeat=1,
            run_gpu=False,
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "components",
        "modes",
        "design_columns",
        "window_length",
        "polyorder",
        "repeat",
        "fit",
        "cpu_serial_best_s",
        "gpu_serial_best_s",
        "gpu_batched_best_s",
        "gpu_batch_speedup",
        "max_abs_filter_diff",
        "rms_filter_diff",
        "max_abs_reconstruction_diff",
        "rms_reconstruction_diff",
        "gpu_error",
    }
    assert set(payload["fit"]) == FIT_KEYS


def test_variable_p2_benchmark_schema_is_stable():
    payload = asdict(
        benchmark_linear_prediction_variable_p2(
            samples=32,
            traces=4,
            n_components=4,
            window_length=7,
            polyorder=3,
            repeat=1,
            run_gpu=False,
            batched_solver="grouped-pinv",
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "components",
        "batched_solver",
        "window_length",
        "polyorder",
        "repeat",
        "fits",
        "mode_count_min",
        "mode_count_max",
        "mode_count_unique",
        "mode_groups",
        "max_design_columns",
        "padded_design_entries",
        "grouped_design_entries",
        "padding_overhead_ratio",
        "mode_reference_elapsed_s",
        "cpu_serial_best_s",
        "gpu_serial_best_s",
        "gpu_batched_best_s",
        "gpu_batch_speedup",
        "max_abs_reconstruction_diff",
        "rms_reconstruction_diff",
        "gpu_error",
    }
    assert len(payload["fits"]) == 4
    for fit in payload["fits"]:
        assert set(fit) == FIT_KEYS
    for group in payload["mode_groups"]:
        assert set(group) == MODE_GROUP_KEYS


def test_variable_stages_benchmark_schema_is_stable():
    payload = asdict(
        benchmark_linear_prediction_variable_stages(
            samples=32,
            traces=4,
            n_components=4,
            window_length=7,
            polyorder=3,
            repeat=1,
            run_gpu=False,
            batched_solver="grouped-pinv",
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "components",
        "batched_solver",
        "window_length",
        "polyorder",
        "repeat",
        "fits",
        "mode_count_min",
        "mode_count_max",
        "mode_count_unique",
        "mode_groups",
        "max_design_columns",
        "padded_design_entries",
        "grouped_design_entries",
        "padding_overhead_ratio",
        "mode_reference_elapsed_s",
        "cpu_serial_best_s",
        "gpu_serial_best_s",
        "gpu_batched_best_s",
        "gpu_batch_speedup",
        "max_abs_filter_diff",
        "rms_filter_diff",
        "max_abs_reconstruction_diff",
        "rms_reconstruction_diff",
        "gpu_error",
    }
    assert len(payload["fits"]) == 4
    for fit in payload["fits"]:
        assert set(fit) == FIT_KEYS
    for group in payload["mode_groups"]:
        assert set(group) == MODE_GROUP_KEYS


def test_variable_artifacts_benchmark_schema_is_stable():
    payload = asdict(
        benchmark_linear_prediction_variable_artifacts(
            samples=32,
            traces=4,
            n_components=4,
            window_length=7,
            polyorder=3,
            repeat=1,
            run_gpu=False,
            batched_solver="grouped-pinv",
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "components",
        "batched_solver",
        "window_length",
        "polyorder",
        "repeat",
        "fits",
        "mode_count_min",
        "mode_count_max",
        "mode_count_unique",
        "mode_groups",
        "max_design_columns",
        "mode_reference_elapsed_s",
        "cpu_serial_best_s",
        "gpu_serial_best_s",
        "gpu_batched_best_s",
        "gpu_batch_speedup",
        "max_abs_filter_diff",
        "rms_filter_diff",
        "max_abs_coefficient_diff",
        "max_abs_amplitude_diff",
        "max_abs_phase_diff",
        "max_abs_frequency_center_diff",
        "max_abs_time_component_diff",
        "max_abs_reconstruction_diff",
        "rms_reconstruction_diff",
        "max_abs_spectrum_component_diff",
        "max_abs_spectrum_total_diff",
        "max_abs_chi2_diff",
        "gpu_error",
    }
    assert len(payload["fits"]) == 4
    for fit in payload["fits"]:
        assert set(fit) == FIT_KEYS
    for group in payload["mode_groups"]:
        assert set(group) == MODE_GROUP_KEYS


def test_prediction_roots_benchmark_schema_is_stable():
    payload = asdict(
        benchmark_prediction_roots(
            samples=32,
            traces=4,
            n_components=4,
            repeat=1,
            backends=("numpy-eigvals", "numpy-roots"),
            run_gpu=False,
        )
    )

    assert set(payload) == {
        "samples",
        "traces",
        "components",
        "repeat",
        "fit",
        "backends",
    }
    assert set(payload["fit"]) == FIT_KEYS
    assert len(payload["backends"]) == 2
    for backend in payload["backends"]:
        assert set(backend) == ROOTS_BACKEND_KEYS


def test_model_order_sweep_schema_is_stable():
    time, trace = synthetic_trace(samples=48)
    payload = asdict(
        model_order_sweep(
            time,
            trace,
            components=(2, 4),
            relative_tolerance=0.05,
        )
    )

    assert set(payload) == {
        "samples",
        "roots_backend",
        "relative_tolerance",
        "best_components",
        "best_selected_model_order",
        "best_rms_residual",
        "best_reconstruction_rms_error",
        "entries",
    }
    assert len(payload["entries"]) == 2
    for entry in payload["entries"]:
        assert set(entry) == MODEL_ORDER_ENTRY_KEYS


def test_subspace_benchmark_schema_is_stable_case():
    time, trace = synthetic_trace(samples=64)
    payload = asdict(
        compare_subspace_methods(
            time,
            trace,
            model_order=5,
            components=6,
            methods=("matrix-pencil", "esprit"),
            svd_backends=("full",),
        )
    )

    assert set(payload) == {
        "samples",
        "model_order",
        "components",
        "baseline_chi2",
        "baseline_rms_residual",
        "methods",
    }
    assert len(payload["methods"]) == 2
    for method in payload["methods"]:
        assert set(method) == SUBSPACE_METHOD_KEYS
