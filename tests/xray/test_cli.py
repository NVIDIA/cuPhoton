# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest

from cuphoton.core.cli import (
    build_component_cli,
    get_component,
    run_component,
)
from cuphoton.xray.linear_prediction import (
    synthetic_trace,
    synthetic_trace_batch,
)


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def _assert_cli_error(capsys, args: list[str], message: str) -> None:
    assert main(args) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_cli_help(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "GPU-accelerated X-ray detector analysis tools" in captured.out


def test_app_spec_uses_canonical_namespace():
    spec = get_component("xray")

    assert spec.import_name == "cuphoton.xray"
    assert spec.program_name == "cuphoton xray"
    assert spec.group == "xray"
    assert spec.app_dir == "xray"
    assert spec.module_name == "cuphoton.xray.commands"
    assert spec.log_env == "CUPHOTON_XRAY_LOG_FILE"
    assert spec.default_log_filename == "xray.log"


def test_core_command_registry_loads_xray_commands():
    cli = build_component_cli("xray")

    expected_aliases = {
        "data-probe": "dp",
        "detector-artifact-compare": "dac",
        "detector-artifact-distributed": "dad",
        "detector-artifact-merge": "dam",
        "detector-artifact-normalize": "dan",
        "detector-artifacts": "da",
        "detector-mask": "dm",
        "doctor": None,
        "extract-trace": "et",
        "gpu-policy": "gp",
        "linear-prediction-benchmark": "lpb",
        "linear-prediction-fixed-stages-benchmark": "lpfsb",
        "linear-prediction-p2-benchmark": "lppb",
        "linear-prediction-profile-summary": "lpps",
        "linear-prediction-runtime-bridge-benchmark": "lprbb",
        "linear-prediction-savgol-benchmark": "lpsb",
        "linear-prediction-smoke": "lps",
        "linear-prediction-variable-artifacts-acceptance": "lpvaa",
        "linear-prediction-variable-artifacts-benchmark": "lpvab",
        "linear-prediction-variable-p2-acceptance": "lpvpa",
        "linear-prediction-variable-p2-benchmark": "lpvpb",
        "linear-prediction-variable-stages-acceptance": "lpvsa",
        "linear-prediction-variable-stages-benchmark": "lpvsb",
        "model-order-sweep": "mos",
        "phonon-viz": "pv",
        "prediction-roots-benchmark": "prb",
        "report": None,
        "roi-candidates": "rc",
        "subspace-acceptance": "sa",
        "subspace-benchmark": "sb",
        "trace-smoke": "ts",
        "validation-viz": "vv",
        "workflow-viz": "wv",
    }

    assert set(cli.command_names) == set(expected_aliases)
    assert {
        name: (
            commandline.shortname
            if commandline.shortname != commandline.name
            else None
        )
        for name, commandline in cli._commands_by_name.items()
    } == expected_aliases


def test_help_subcommand_delegates_to_xray_parser(capsys):
    assert main(["help", "doctor"]) == 0
    captured = capsys.readouterr()
    assert "usage: cuphoton xray doctor" in captured.out
    assert "--json" in captured.out


def test_umbrella_program_name_reaches_xray_parser(capsys):
    assert main(["help", "doctor"], program_name="cuphoton xray") == 0
    captured = capsys.readouterr()
    assert "usage: cuphoton xray doctor" in captured.out


def test_nonnegative_option_preserves_parser_error(capsys):
    assert (
        main(
            [
                "linear-prediction-profile-summary",
                "dummy.log",
                "--steady-state-skip-profile-lines",
                "-1",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "usage: cuphoton xray linear-prediction-profile-summary"
    )
    assert captured.err.endswith(
        "cuphoton xray linear-prediction-profile-summary: error: argument "
        "--steady-state-skip-profile-lines: must be non-negative\n"
    )


def test_gpu_policy(capsys):
    assert main(["gpu-policy"]) == 0
    captured = capsys.readouterr()
    assert "NVIDIA GPU execution" in captured.out


def test_doctor_json(capsys):
    assert main(["doctor", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["runtime"]["python_version"]
    assert payload["runtime"]["cuphoton_version"]
    assert "imports" in payload
    assert "executables" in payload
    assert {probe["category"] for probe in payload["imports"]} >= {
        "required",
        "gpu",
        "optional",
    }
    assert payload["cuda_visibility"] in {"unset", "set", "empty"}
    assert "config" not in payload
    assert all("path" not in item for item in payload["executables"])


def test_validation_error_is_concise(capsys):
    assert (
        main(
            [
                "linear-prediction-smoke",
                "--samples",
                "0",
                "--no-gpu",
                "--json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "/src/cuphoton" not in captured.err
    assert "at least 16" in captured.err


def test_linear_prediction_profile_summary_json(tmp_path, capsys):
    log_path = tmp_path / "profile.log"
    log_path.write_text(
        "\n".join(
            [
                "DEBUG:\t20.000s: Processing 1 tiles took   8.500s",
                "linearpred_cupy: ROI [0:16,0:16] raw_fits = 16, "
                "failures = 0, skipped_fits = 0, filtered_fits = 1, "
                "amp_threshold = 1.6",
                "linearpred_profile: ROI [0:16,0:16] "
                "linearpred_total=2.000000s/1, "
                "savgol_cupy=0.400000s/16, "
                "eigvals_cupy=0.700000s/16",
                "real 12.25",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "linear-prediction-profile-summary",
                str(log_path),
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["sources"] == [str(log_path)]
    assert payload["profile_lines"] == 1
    assert payload["fit_lines"] == 1
    assert payload["processing_tiles"] == 1
    assert payload["tile_processing_seconds"] == 8.5
    assert payload["command_real_seconds"] == 12.25
    assert payload["raw_fits"] == 16
    assert payload["failures"] == 0
    assert payload["filtered_fits"] == 1
    assert payload["stages"][0]["stage"] == "linearpred_total"


def test_linear_prediction_profile_summary_steady_state_json(
    tmp_path,
    capsys,
):
    log_path = tmp_path / "profile.log"
    log_path.write_text(
        "\n".join(
            [
                "linearpred_profile: ROI [0:16,0:16] "
                "linearpred_total=10.000000s/1, "
                "eigvals_cupy=8.000000s/16",
                "linearpred_profile: ROI [0:16,16:32] "
                "linearpred_total=2.000000s/1, "
                "eigvals_cupy=0.700000s/16",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "linear-prediction-profile-summary",
                str(log_path),
                "--steady-state-skip-profile-lines",
                "1",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["profile_lines"] == 2
    assert payload["stages"][0]["seconds"] == 12.0
    steady_state = payload["steady_state"]
    assert steady_state["skip_profile_lines_per_source"] == 1
    assert steady_state["profile_lines"] == 2
    assert steady_state["skipped_profile_lines"] == 1
    assert steady_state["included_profile_lines"] == 1
    assert steady_state["stages"][0]["stage"] == "linearpred_total"
    assert steady_state["stages"][0]["seconds"] == 2.0


def test_linear_prediction_smoke_json_without_gpu(capsys):
    assert (
        main(
            [
                "linear-prediction-smoke",
                "--samples",
                "32",
                "--components",
                "4",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["samples"] == 32
    assert payload["components"] == 4
    assert payload["cpu"]["backend"] == "cpu"
    assert 0 < payload["cpu"]["selected_model_order"] <= 4
    assert payload["cpu"]["decaying_root_count"] >= payload["cpu"]["modes"]
    assert payload["cpu"]["reconstruction_samples"] == 32
    assert payload["cpu"]["roots"]["backend"] == "eigvals"
    assert payload["cpu"]["roots"]["matrix_size"] > 0
    assert payload["gpu"] is None


def test_linear_prediction_smoke_json_with_roots_backend(capsys):
    assert (
        main(
            [
                "linear-prediction-smoke",
                "--samples",
                "32",
                "--components",
                "4",
                "--roots-backend",
                "roots",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["cpu"]["roots"]["backend"] == "roots"
    assert payload["cpu"]["roots"]["array_module"] == "numpy"


def test_linear_prediction_benchmark_json_without_gpu(capsys):
    assert (
        main(
            [
                "linear-prediction-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["repeat"] == 1
    assert payload["fit"]["requested_components"] == 4
    assert payload["fit"]["matrix_size"] == 23
    assert payload["fit"]["root_count"] == 23
    assert payload["fit"]["selected_model_order"] == 4
    assert payload["fit"]["selected_root_count"] > 0
    assert (
        payload["fit"]["decaying_root_count"]
        >= payload["fit"]["selected_root_count"]
    )
    assert payload["cpu_serial_best_s"] > 0
    assert payload["gpu_serial_best_s"] is None
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_p2_benchmark_json_without_gpu(capsys):
    assert (
        main(
            [
                "linear-prediction-p2-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["repeat"] == 1
    assert payload["modes"] > 0
    assert payload["design_columns"] == payload["modes"] * 2 + 1
    assert payload["fit"]["requested_components"] == 4
    assert payload["fit"]["matrix_size"] == 23
    assert payload["fit"]["selected_model_order"] == 4
    assert payload["fit"]["selected_root_count"] == payload["modes"]
    assert (
        payload["fit"]["decaying_root_count"]
        >= payload["fit"]["selected_root_count"]
    )
    assert payload["fit"]["filtered_root_count"] >= 0
    assert payload["fit"]["chi2"] >= 0
    assert payload["fit"]["rms_residual"] >= 0
    assert payload["cpu_serial_best_s"] > 0
    assert payload["gpu_serial_best_s"] is None
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_batch_speedup"] is None
    assert payload["max_abs_reconstruction_diff"] is None
    assert payload["rms_reconstruction_diff"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_savgol_benchmark_json_without_gpu(capsys):
    assert (
        main(
            [
                "linear-prediction-savgol-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["window_length"] == 7
    assert payload["polyorder"] == 3
    assert payload["repeat"] == 1
    assert payload["cpu_serial_best_s"] > 0
    assert payload["gpu_serial_best_s"] is None
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_batch_speedup"] is None
    assert payload["max_abs_filter_diff"] is None
    assert payload["rms_filter_diff"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_fixed_stages_benchmark_json_without_gpu(capsys):
    assert (
        main(
            [
                "linear-prediction-fixed-stages-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["window_length"] == 7
    assert payload["polyorder"] == 3
    assert payload["repeat"] == 1
    assert payload["fit"]["requested_components"] == 4
    assert payload["fit"]["matrix_size"] == 23
    assert payload["fit"]["selected_model_order"] == 4
    fit_modes = payload["fit"]["selected_root_count"]
    assert fit_modes >= 0
    assert payload["fit"]["decaying_root_count"] >= fit_modes
    assert payload.get("modes") == fit_modes
    assert payload["design_columns"] == fit_modes * 2 + 1
    assert payload["fit"]["filtered_root_count"] >= 0
    assert payload["fit"]["chi2"] >= 0
    assert payload["fit"]["rms_residual"] >= 0
    assert payload["cpu_serial_best_s"] > 0
    assert payload["gpu_serial_best_s"] is None
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_batch_speedup"] is None
    assert payload["max_abs_filter_diff"] is None
    assert payload["rms_filter_diff"] is None
    assert payload["max_abs_reconstruction_diff"] is None
    assert payload["rms_reconstruction_diff"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_p2_benchmark_json_without_gpu(capsys):
    assert (
        main(
            [
                "linear-prediction-variable-p2-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["batched_solver"] == "pinv"
    assert payload["window_length"] == 7
    assert payload["polyorder"] == 3
    assert payload["repeat"] == 1
    assert len(payload["fits"]) == 4
    assert all(fit["requested_components"] == 4 for fit in payload["fits"])
    assert all(fit["matrix_size"] == 23 for fit in payload["fits"])
    assert all(
        fit["decaying_root_count"] >= fit["selected_root_count"]
        for fit in payload["fits"]
    )
    assert all(fit["filtered_root_count"] >= 0 for fit in payload["fits"])
    assert all(fit["chi2"] >= 0 for fit in payload["fits"])
    assert all(fit["rms_residual"] >= 0 for fit in payload["fits"])
    assert payload["mode_count_min"] >= 0
    assert payload["mode_count_max"] >= payload["mode_count_min"]
    assert payload["mode_count_unique"]
    assert [
        group["mode_count"] for group in payload["mode_groups"]
    ] == payload["mode_count_unique"]
    assert (
        sum(group["trace_count"] for group in payload["mode_groups"])
        == (payload["traces"])
    )
    assert all(
        group["design_columns"] == group["mode_count"] * 2 + 1
        for group in payload["mode_groups"]
    )
    assert (
        sorted({fit["selected_root_count"] for fit in payload["fits"]})
        == payload["mode_count_unique"]
    )
    assert payload["max_design_columns"] == payload["mode_count_max"] * 2 + 1
    assert payload["padded_design_entries"] == (
        payload["traces"] * payload["max_design_columns"]
    )
    assert payload["grouped_design_entries"] == sum(
        group["trace_count"] * group["design_columns"]
        for group in payload["mode_groups"]
    )
    assert payload["padding_overhead_ratio"] >= 1
    assert payload["mode_reference_elapsed_s"] > 0
    assert payload["cpu_serial_best_s"] > 0
    assert payload["gpu_serial_best_s"] is None
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_batch_speedup"] is None
    assert payload["max_abs_reconstruction_diff"] is None
    assert payload["rms_reconstruction_diff"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_p2_benchmark_grouped_json(capsys):
    assert (
        main(
            [
                "linear-prediction-variable-p2-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["batched_solver"] == "grouped-pinv"
    assert payload["mode_count_unique"]
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_stages_benchmark_json_without_gpu(capsys):
    assert (
        main(
            [
                "linear-prediction-variable-stages-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["batched_solver"] == "grouped-pinv"
    assert payload["mode_count_unique"]
    assert len(payload["fits"]) == 4
    assert payload["gpu_batched_best_s"] is None
    assert payload["max_abs_filter_diff"] is None
    assert payload["max_abs_reconstruction_diff"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_stages_benchmark_trace_dir(
    tmp_path,
    capsys,
):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for name, trace in zip(
        ("trace-y10.npz", "trace-y2.npz"),
        trace_rows,
        strict=True,
    ):
        np.savez(trace_dir / name, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-stages-benchmark",
                "--trace-dir",
                str(trace_dir),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["source"]["trace_count"] == 2
    assert [
        source["path"].rsplit("/", 1)[-1]
        for source in payload["source"]["traces"]
    ] == ["trace-y2.npz", "trace-y10.npz"]
    assert payload["traces"] == 2
    assert len(payload["fits"]) == 2


def test_linear_prediction_variable_artifacts_benchmark_json_without_gpu(
    capsys,
):
    assert (
        main(
            [
                "linear-prediction-variable-artifacts-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["batched_solver"] == "grouped-pinv"
    assert payload["mode_count_unique"]
    assert len(payload["fits"]) == 4
    assert payload["gpu_batched_best_s"] is None
    assert payload["max_abs_coefficient_diff"] is None
    assert payload["max_abs_amplitude_diff"] is None
    assert payload["max_abs_phase_diff"] is None
    assert payload["max_abs_frequency_center_diff"] is None
    assert payload["max_abs_time_component_diff"] is None
    assert payload["max_abs_reconstruction_diff"] is None
    assert payload["max_abs_spectrum_component_diff"] is None
    assert payload["max_abs_spectrum_total_diff"] is None
    assert payload["max_abs_chi2_diff"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_runtime_bridge_benchmark_json_without_gpu(
    capsys,
):
    assert (
        main(
            [
                "linear-prediction-runtime-bridge-benchmark",
                "--samples",
                "32",
                "--tiles",
                "2",
                "--rows-per-tile",
                "2",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["samples"] == 32
    assert payload["tiles"] == 2
    assert payload["rows_per_tile"] == 2
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["batched_solver"] == "grouped-pinv"
    assert payload["mode_count_unique"]
    assert len(payload["fits"]) == 4
    assert payload["gpu_serial_per_tile_best_s"] is None
    assert payload["gpu_row_batched_per_tile_best_s"] is None
    assert payload["gpu_multi_tile_grouped_best_s"] is None
    assert payload["row_batched_speedup"] is None
    assert payload["multi_tile_speedup"] is None
    assert payload["multi_vs_row_batched_speedup"] is None
    assert payload["max_abs_frequency_center_diff"] is None
    assert payload["max_abs_time_component_diff"] is None
    assert payload["max_abs_reconstruction_diff"] is None
    assert payload["max_abs_amplitude_diff"] is None
    assert payload["max_abs_phase_diff"] is None
    assert payload["max_abs_chi2_diff"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_artifacts_benchmark_trace_dir(
    tmp_path,
    capsys,
):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for name, trace in zip(
        ("trace-y10.npz", "trace-y2.npz"),
        trace_rows,
        strict=True,
    ):
        np.savez(trace_dir / name, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-artifacts-benchmark",
                "--trace-dir",
                str(trace_dir),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["source"]["trace_count"] == 2
    assert [
        source["path"].rsplit("/", 1)[-1]
        for source in payload["source"]["traces"]
    ] == ["trace-y2.npz", "trace-y10.npz"]
    assert payload["traces"] == 2
    assert len(payload["fits"]) == 2


def test_linear_prediction_variable_artifacts_acceptance_without_gpu(
    tmp_path,
    capsys,
):
    trace_paths = []
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for index, trace in enumerate(trace_rows):
        trace_path = tmp_path / f"trace-{index}.npz"
        np.savez(trace_path, time=time, trace=trace)
        trace_paths.append(trace_path)

    args = [
        "linear-prediction-variable-artifacts-acceptance",
        "--components",
        "4",
        "--window-length",
        "7",
        "--polyorder",
        "3",
        "--batched-solver",
        "grouped-pinv",
        "--repeat",
        "1",
        "--no-gpu",
        "--json",
    ]
    for trace_path in trace_paths:
        args.extend(["--trace-npz", str(trace_path)])

    assert main(args) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is False
    assert payload["accepted_solvers"] == []
    assert payload["trace_count"] == 2
    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["thresholds"]["max_filter_diff_ratio"] == 1e-09
    assert payload["thresholds"]["max_time_component_diff_ratio"] == 1e-09
    assert payload["thresholds"]["max_reconstruction_diff_ratio"] == 1e-09
    assert payload["thresholds"]["max_spectrum_component_diff"] == 1e-08
    assert payload["thresholds"]["min_gpu_speedup"] == 1.0
    assert len(payload["solvers"]) == 1
    solver = payload["solvers"][0]
    assert solver["batched_solver"] == "grouped-pinv"
    assert solver["accepted"] is False
    assert solver["filter_diff_ratio"] == float("inf")
    assert solver["time_component_diff_ratio"] == float("inf")
    assert solver["reconstruction_diff_ratio"] == float("inf")
    assert solver["artifact_checks"]["coefficient_diff"] is False
    assert solver["artifact_checks"]["gpu_speedup"] is False
    assert solver["gpu_batched_best_s"] is None


def test_variable_artifacts_acceptance_requires_all_solvers(
    tmp_path,
    capsys,
):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-artifacts-acceptance",
                "--trace-npz",
                str(trace_path),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "pinv",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is False
    assert payload["accepted_solvers"] == []
    assert [solver["batched_solver"] for solver in payload["solvers"]] == [
        "pinv",
        "grouped-pinv",
    ]


def test_linear_prediction_variable_artifacts_acceptance_text(
    tmp_path,
    capsys,
):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-artifacts-acceptance",
                "--trace-npz",
                str(trace_path),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--repeat",
                "1",
                "--no-gpu",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out

    assert "status=rejected" in output
    assert "trace_count=1" in output
    assert "solver=grouped-pinv" in output
    assert "accepted=False" in output
    assert "gpu_batch_speedup=None" in output
    assert "filter_diff_ratio=inf" in output
    assert "time_component_diff_ratio=inf" in output
    assert "reconstruction_diff_ratio=inf" in output


def test_variable_artifacts_acceptance_rejects_thresholds(tmp_path, capsys):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    _assert_cli_error(
        capsys,
        [
            "linear-prediction-variable-artifacts-acceptance",
            "--trace-npz",
            str(trace_path),
            "--max-coefficient-diff",
            "-1",
        ],
        "--max-coefficient-diff must be non-negative",
    )


def test_variable_artifacts_acceptance_requires_trace_input(capsys):
    _assert_cli_error(
        capsys,
        ["linear-prediction-variable-artifacts-acceptance"],
        "provide --trace-npz or --trace-dir",
    )


def test_variable_artifacts_acceptance_accepts_gpu_path(
    tmp_path,
    capsys,
):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA devices visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    trace_paths = []
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for index, trace in enumerate(trace_rows):
        trace_path = tmp_path / f"trace-{index}.npz"
        np.savez(trace_path, time=time, trace=trace)
        trace_paths.append(trace_path)

    args = [
        "linear-prediction-variable-artifacts-acceptance",
        "--components",
        "4",
        "--window-length",
        "7",
        "--polyorder",
        "3",
        "--repeat",
        "1",
        "--min-gpu-speedup",
        "0",
        "--json",
    ]
    for trace_path in trace_paths:
        args.extend(["--trace-npz", str(trace_path)])

    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is True
    assert payload["accepted_solvers"] == ["grouped-pinv"]
    solver = payload["solvers"][0]
    assert solver["accepted"] is True
    assert solver["gpu_error"] is None
    assert solver["gpu_batch_speedup"] > 0
    assert solver["filter_diff_ratio"] < 1e-12
    assert solver["time_component_diff_ratio"] < 1e-12
    assert solver["reconstruction_diff_ratio"] < 1e-12
    assert all(solver["artifact_checks"].values())


def test_linear_prediction_variable_stages_acceptance_without_gpu(
    tmp_path,
    capsys,
):
    trace_paths = []
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for index, trace in enumerate(trace_rows):
        trace_path = tmp_path / f"trace-{index}.npz"
        np.savez(trace_path, time=time, trace=trace)
        trace_paths.append(trace_path)

    args = [
        "linear-prediction-variable-stages-acceptance",
        "--components",
        "4",
        "--window-length",
        "7",
        "--polyorder",
        "3",
        "--batched-solver",
        "grouped-pinv",
        "--repeat",
        "1",
        "--no-gpu",
        "--json",
    ]
    for trace_path in trace_paths:
        args.extend(["--trace-npz", str(trace_path)])

    assert main(args) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is False
    assert payload["accepted_solvers"] == []
    assert payload["trace_count"] == 2
    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["thresholds"] == {
        "max_filter_diff_ratio": 1e-09,
        "max_reconstruction_diff_ratio": 1e-09,
        "min_gpu_speedup": 1.0,
    }
    assert len(payload["solvers"]) == 1
    solver = payload["solvers"][0]
    assert solver["batched_solver"] == "grouped-pinv"
    assert solver["accepted"] is False
    assert solver["filter_diff_ratio"] == float("inf")
    assert solver["reconstruction_diff_ratio"] == float("inf")
    assert solver["gpu_batched_best_s"] is None


def test_variable_stages_acceptance_requires_all_solvers(tmp_path, capsys):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-stages-acceptance",
                "--trace-npz",
                str(trace_path),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "pinv",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is False
    assert payload["accepted_solvers"] == []
    assert [solver["batched_solver"] for solver in payload["solvers"]] == [
        "pinv",
        "grouped-pinv",
    ]


def test_linear_prediction_variable_stages_acceptance_text(tmp_path, capsys):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-stages-acceptance",
                "--trace-npz",
                str(trace_path),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--repeat",
                "1",
                "--no-gpu",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out

    assert "status=rejected" in output
    assert "trace_count=1" in output
    assert "solver=grouped-pinv" in output
    assert "accepted=False" in output
    assert "gpu_batch_speedup=None" in output
    assert "filter_diff_ratio=inf" in output
    assert "reconstruction_diff_ratio=inf" in output


def test_variable_stages_acceptance_rejects_thresholds(tmp_path, capsys):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    _assert_cli_error(
        capsys,
        [
            "linear-prediction-variable-stages-acceptance",
            "--trace-npz",
            str(trace_path),
            "--max-filter-diff-ratio",
            "-1",
        ],
        "--max-filter-diff-ratio must be non-negative",
    )
    _assert_cli_error(
        capsys,
        [
            "linear-prediction-variable-stages-acceptance",
            "--trace-npz",
            str(trace_path),
            "--max-reconstruction-diff-ratio",
            "-1",
        ],
        "--max-reconstruction-diff-ratio must be non-negative",
    )


def test_variable_stages_acceptance_requires_trace_input(capsys):
    _assert_cli_error(
        capsys,
        ["linear-prediction-variable-stages-acceptance"],
        "provide --trace-npz or --trace-dir",
    )


def test_linear_prediction_variable_p2_benchmark_grouped_normal_json(capsys):
    assert (
        main(
            [
                "linear-prediction-variable-p2-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-normal",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["batched_solver"] == "grouped-normal"
    assert payload["mode_count_unique"]
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_p2_benchmark_grouped_qr_json(capsys):
    assert (
        main(
            [
                "linear-prediction-variable-p2-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-qr",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["batched_solver"] == "grouped-qr"
    assert payload["mode_count_unique"]
    assert payload["gpu_batched_best_s"] is None
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_p2_benchmark_trace_npz_batch(
    tmp_path,
    capsys,
):
    trace_paths = []
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for index, trace in enumerate(trace_rows):
        trace_path = tmp_path / f"trace-{index}.npz"
        np.savez(trace_path, time=time, trace=trace)
        trace_paths.append(trace_path)

    args = [
        "linear-prediction-variable-p2-benchmark",
        "--components",
        "4",
        "--window-length",
        "7",
        "--polyorder",
        "3",
        "--batched-solver",
        "grouped-pinv",
        "--repeat",
        "1",
        "--no-gpu",
        "--json",
    ]
    for trace_path in trace_paths:
        args.extend(["--trace-npz", str(trace_path)])

    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["source"]["trace_count"] == 2
    assert [source["samples"] for source in payload["source"]["traces"]] == [
        32,
        32,
    ]
    assert payload["samples"] == 32
    assert payload["traces"] == 2
    assert payload["batched_solver"] == "grouped-pinv"
    assert len(payload["fits"]) == 2
    assert payload["gpu_error"] is None


def test_linear_prediction_variable_p2_benchmark_trace_dir(tmp_path, capsys):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for name, trace in zip(
        ("trace-y10.npz", "trace-y2.npz"),
        trace_rows,
        strict=True,
    ):
        np.savez(trace_dir / name, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-p2-benchmark",
                "--trace-dir",
                str(trace_dir),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "grouped-pinv",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["source"]["trace_count"] == 2
    assert [
        source["path"].rsplit("/", 1)[-1]
        for source in payload["source"]["traces"]
    ] == ["trace-y2.npz", "trace-y10.npz"]
    assert payload["traces"] == 2
    assert len(payload["fits"]) == 2


def test_linear_prediction_variable_p2_acceptance_without_gpu(
    tmp_path,
    capsys,
):
    trace_paths = []
    time, trace_rows = synthetic_trace_batch(samples=32, traces=2)
    for index, trace in enumerate(trace_rows):
        trace_path = tmp_path / f"trace-{index}.npz"
        np.savez(trace_path, time=time, trace=trace)
        trace_paths.append(trace_path)

    args = [
        "linear-prediction-variable-p2-acceptance",
        "--components",
        "4",
        "--window-length",
        "7",
        "--polyorder",
        "3",
        "--batched-solver",
        "grouped-pinv",
        "--repeat",
        "1",
        "--no-gpu",
        "--json",
    ]
    for trace_path in trace_paths:
        args.extend(["--trace-npz", str(trace_path)])

    assert main(args) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is False
    assert payload["accepted_solvers"] == []
    assert payload["trace_count"] == 2
    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["thresholds"] == {
        "max_diff_ratio": 1e-09,
        "min_gpu_speedup": 1.0,
    }
    assert len(payload["solvers"]) == 1
    solver = payload["solvers"][0]
    assert solver["batched_solver"] == "grouped-pinv"
    assert solver["accepted"] is False
    assert solver["gpu_batched_best_s"] is None


def test_variable_p2_acceptance_requires_all_solvers(tmp_path, capsys):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-p2-acceptance",
                "--trace-npz",
                str(trace_path),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--batched-solver",
                "pinv",
                "--batched-solver",
                "grouped-pinv",
                "--batched-solver",
                "grouped-normal",
                "--batched-solver",
                "grouped-qr",
                "--repeat",
                "1",
                "--no-gpu",
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is False
    assert payload["accepted_solvers"] == []
    assert [solver["batched_solver"] for solver in payload["solvers"]] == [
        "pinv",
        "grouped-pinv",
        "grouped-normal",
        "grouped-qr",
    ]


def test_linear_prediction_variable_p2_acceptance_text(tmp_path, capsys):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "linear-prediction-variable-p2-acceptance",
                "--trace-npz",
                str(trace_path),
                "--components",
                "4",
                "--window-length",
                "7",
                "--polyorder",
                "3",
                "--repeat",
                "1",
                "--no-gpu",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out

    assert "status=rejected" in output
    assert "trace_count=1" in output
    assert "solver=grouped-pinv" in output
    assert "accepted=False" in output
    assert "gpu_batch_speedup=None" in output


def test_variable_p2_acceptance_rejects_thresholds(tmp_path, capsys):
    time, trace = synthetic_trace(samples=32)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    _assert_cli_error(
        capsys,
        [
            "linear-prediction-variable-p2-acceptance",
            "--trace-npz",
            str(trace_path),
            "--max-diff-ratio",
            "-1",
        ],
        "--max-diff-ratio must be non-negative",
    )


def test_variable_p2_acceptance_requires_trace_input(capsys):
    _assert_cli_error(
        capsys,
        ["linear-prediction-variable-p2-acceptance"],
        "provide --trace-npz or --trace-dir",
    )


def test_prediction_roots_benchmark_json_without_gpu(capsys):
    assert (
        main(
            [
                "prediction-roots-benchmark",
                "--samples",
                "32",
                "--traces",
                "4",
                "--components",
                "4",
                "--repeat",
                "1",
                "--backend",
                "numpy-eigvals",
                "--backend",
                "numpy-roots",
                "--no-gpu",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["samples"] == 32
    assert payload["traces"] == 4
    assert payload["components"] == 4
    assert payload["repeat"] == 1
    assert payload["fit"]["requested_components"] == 4
    assert payload["fit"]["matrix_size"] == 23
    assert payload["fit"]["selected_model_order"] == 4
    assert (
        payload["fit"]["decaying_root_count"]
        >= payload["fit"]["selected_root_count"]
    )
    assert [backend["backend"] for backend in payload["backends"]] == [
        "numpy-eigvals",
        "numpy-roots",
    ]
    for backend in payload["backends"]:
        assert backend["matrix_size"] == 23
        assert backend["row_count"] == 4
        assert backend["failures"] == 0
        assert backend["error"] is None


def test_model_order_sweep_json(capsys):
    assert (
        main(
            [
                "model-order-sweep",
                "--samples",
                "48",
                "--component",
                "2",
                "--component",
                "4",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["samples"] == 48
    assert payload["roots_backend"] == "eigvals"
    assert payload["best_components"] in {2, 4}
    assert payload["best_selected_model_order"] > 0
    assert payload["best_reconstruction_rms_error"] >= 0
    assert [entry["components"] for entry in payload["entries"]] == [2, 4]
    for entry in payload["entries"]:
        assert entry["matrix_size"] == 35
        assert entry["root_count"] == 35
        assert 0 < entry["selected_model_order"] <= entry["components"]
        assert entry["selected_root_count"] >= 0
        assert entry["decaying_root_count"] >= entry["selected_root_count"]
        assert entry["rms_residual"] >= 0
        assert entry["reconstruction_rms_error"] == entry["rms_residual"]
    assert any(
        entry["decaying_root_count"] != entry["selected_root_count"]
        for entry in payload["entries"]
    )


def test_model_order_sweep_json_from_trace_npz(tmp_path, capsys):
    time, trace = synthetic_trace(48)
    trace_path = tmp_path / "trace.npz"
    summary = {"source": "fixture", "roi": [1, 2, 3, 4]}
    np.savez(trace_path, time=time, trace=trace, summary=json.dumps(summary))

    assert (
        main(
            [
                "model-order-sweep",
                "--trace-npz",
                str(trace_path),
                "--component",
                "2",
                "--component",
                "4",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["source"]["kind"] == "trace-npz"
    assert payload["source"]["path"] == str(trace_path)
    assert payload["source"]["samples"] == 48
    assert payload["source"]["summary"] == summary
    assert payload["samples"] == 48
    assert [entry["components"] for entry in payload["entries"]] == [2, 4]


def test_model_order_sweep_json_from_trace_npz_batch(tmp_path, capsys):
    trace_paths = []
    time, trace_rows = synthetic_trace_batch(samples=48, traces=2)
    for index, trace in enumerate(trace_rows):
        trace_path = tmp_path / f"trace-{index}.npz"
        summary = {"source": "fixture", "row": index}
        np.savez(
            trace_path,
            time=time,
            trace=trace,
            summary=json.dumps(summary),
        )
        trace_paths.append(trace_path)

    args = [
        "model-order-sweep",
        "--component",
        "2",
        "--component",
        "4",
        "--json",
    ]
    for trace_path in trace_paths:
        args.extend(["--trace-npz", str(trace_path)])

    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["source"]["trace_count"] == 2
    assert payload["trace_count"] == 2
    assert payload["samples"] == 48
    assert payload["component_counts"] == [2, 4]
    assert len(payload["best_components_by_trace"]) == 2
    assert payload["best_components_unique"]
    assert len(payload["best_selected_model_orders_by_trace"]) == 2
    assert payload["best_selected_model_orders_unique"]
    assert payload["best_reconstruction_rms_error_min"] >= 0
    assert (
        payload["best_reconstruction_rms_error_max"]
        >= payload["best_reconstruction_rms_error_min"]
    )
    assert len(payload["traces"]) == 2
    for index, trace_payload in enumerate(payload["traces"]):
        assert trace_payload["trace_index"] == index
        assert trace_payload["source"]["path"] == str(trace_paths[index])
        assert trace_payload["source"]["summary"]["row"] == index
        components = [
            entry["components"] for entry in trace_payload["entries"]
        ]
        assert components == [2, 4]
    assert payload["traces"][1]["source"]["time_matches_first"] is True


def test_model_order_sweep_json_from_trace_dir(tmp_path, capsys):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, trace_rows = synthetic_trace_batch(samples=48, traces=2)
    for name, trace in zip(
        ("trace-y10.npz", "trace-y2.npz"),
        trace_rows,
        strict=True,
    ):
        np.savez(trace_dir / name, time=time, trace=trace)

    assert (
        main(
            [
                "model-order-sweep",
                "--trace-dir",
                str(trace_dir),
                "--component",
                "2",
                "--component",
                "4",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["trace_count"] == 2
    assert [
        trace_payload["source"]["path"].rsplit("/", 1)[-1]
        for trace_payload in payload["traces"]
    ] == ["trace-y2.npz", "trace-y10.npz"]


def test_trace_npz_rejects_hdf5_selection_options(tmp_path, capsys):
    time, trace = synthetic_trace(48)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    _assert_cli_error(
        capsys,
        [
            "model-order-sweep",
            "--trace-npz",
            str(trace_path),
            "--roi-lower",
            "0",
            "0",
            "--roi-dim",
            "8",
            "8",
        ],
        "--trace-npz cannot be combined with HDF5 input options",
    )


def test_trace_dir_rejects_explicit_trace_npz(tmp_path, capsys):
    time, trace = synthetic_trace(48)
    trace_path = tmp_path / "trace.npz"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    np.savez(trace_path, time=time, trace=trace)
    np.savez(trace_dir / "trace.npz", time=time, trace=trace)

    _assert_cli_error(
        capsys,
        [
            "model-order-sweep",
            "--trace-npz",
            str(trace_path),
            "--trace-dir",
            str(trace_dir),
        ],
        "--trace-dir cannot be combined with --trace-npz",
    )


def test_subspace_benchmark_json(capsys):
    assert (
        main(
            [
                "subspace-benchmark",
                "--samples",
                "64",
                "--model-order",
                "5",
                "--components",
                "6",
                "--method",
                "matrix-pencil",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["source"]["kind"] == "synthetic"
    assert payload["samples"] == 64
    assert payload["model_order"] == 5
    assert payload["components"] == 6
    assert payload["baseline_chi2"] >= 0
    assert len(payload["methods"]) == 1
    method = payload["methods"][0]
    assert method["method"] == "matrix-pencil"
    assert method["svd_backend"] == "full"
    assert method["svd_rank"] == 5
    assert len(method["singular_value_head"]) > 0
    assert method["samples"] == 64
    assert method["model_order"] == 5
    assert method["rms_residual"] >= 0
    assert len(method["roots_real"]) == 5


def test_subspace_benchmark_json_compares_svd_backends(capsys):
    pytest.importorskip("scipy.sparse.linalg")

    assert (
        main(
            [
                "subspace-benchmark",
                "--samples",
                "64",
                "--model-order",
                "5",
                "--components",
                "6",
                "--method",
                "matrix-pencil",
                "--svd-backend",
                "full",
                "--svd-backend",
                "randomized",
                "--svd-backend",
                "partial",
                "--random-seed",
                "7",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["svd_backends"] == ["full", "randomized", "partial"]
    assert [
        (method["method"], method["svd_backend"])
        for method in payload["methods"]
    ] == [
        ("matrix-pencil", "full"),
        ("matrix-pencil", "randomized"),
        ("matrix-pencil", "partial"),
    ]
    for method in payload["methods"]:
        assert method["svd_rank"] == 5
        assert len(method["singular_value_head"]) > 0


def test_subspace_benchmark_json_from_trace_npz(tmp_path, capsys):
    time, trace = synthetic_trace(64)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "subspace-benchmark",
                "--trace-npz",
                str(trace_path),
                "--model-order",
                "5",
                "--components",
                "6",
                "--method",
                "esprit",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["source"]["kind"] == "trace-npz"
    assert payload["source"]["path"] == str(trace_path)
    assert payload["source"]["samples"] == 64
    assert payload["samples"] == 64
    assert len(payload["methods"]) == 1
    assert payload["methods"][0]["method"] == "esprit"


def test_subspace_benchmark_json_from_trace_dir(tmp_path, capsys):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, trace_rows = synthetic_trace_batch(samples=64, traces=2)
    for name, trace in zip(
        ("trace-y10.npz", "trace-y2.npz"),
        trace_rows,
        strict=True,
    ):
        np.savez(trace_dir / name, time=time, trace=trace)

    assert (
        main(
            [
                "subspace-benchmark",
                "--trace-dir",
                str(trace_dir),
                "--model-order",
                "5",
                "--components",
                "6",
                "--method",
                "matrix-pencil",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "trace-npz-batch"
    assert payload["source"]["trace_count"] == 2
    assert payload["trace_count"] == 2
    assert payload["samples"] == 64
    assert payload["model_order"] == 5
    assert payload["components"] == 6
    assert payload["baseline_rms_residual_min"] >= 0
    assert (
        payload["baseline_rms_residual_max"]
        >= payload["baseline_rms_residual_min"]
    )
    assert len(payload["method_summary"]) == 1
    summary = payload["method_summary"][0]
    assert set(summary) == {
        "method",
        "svd_backend",
        "trace_count",
        "rms_residual_min",
        "rms_residual_max",
        "max_abs_reconstruction_diff_max",
        "elapsed_s_total",
    }
    assert summary["method"] == "matrix-pencil"
    assert summary["svd_backend"] == "full"
    assert summary["trace_count"] == 2
    assert summary["rms_residual_min"] >= 0
    assert summary["rms_residual_max"] >= summary["rms_residual_min"]
    assert summary["max_abs_reconstruction_diff_max"] >= 0
    assert summary["elapsed_s_total"] > 0
    assert [
        trace_payload["source"]["path"].rsplit("/", 1)[-1]
        for trace_payload in payload["traces"]
    ] == ["trace-y2.npz", "trace-y10.npz"]
    for trace_payload in payload["traces"]:
        assert trace_payload["source"]["kind"] == "trace-npz"
        assert trace_payload["samples"] == 64
        assert len(trace_payload["methods"]) == 1


def test_subspace_acceptance_json_from_trace_npz(tmp_path, capsys):
    trace_paths = []
    for index in range(2):
        time, trace = synthetic_trace(64)
        trace_path = tmp_path / f"trace-{index}.npz"
        np.savez(trace_path, time=time, trace=trace)
        trace_paths.append(trace_path)

    args = [
        "subspace-acceptance",
        "--model-order",
        "5",
        "--components",
        "6",
        "--json",
    ]
    for trace_path in trace_paths:
        args.extend(["--trace-npz", str(trace_path)])

    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is True
    assert payload["accepted_methods"] == ["matrix-pencil", "esprit"]
    assert payload["trace_count"] == 2
    assert payload["svd_backends"] == ["full"]
    assert payload["thresholds"] == {
        "max_diff_ratio": 0.05,
        "max_rms_ratio": 1.05,
    }
    assert [trace["samples"] for trace in payload["traces"]] == [64, 64]
    for summary in payload["method_summary"]:
        assert summary["svd_backend"] == "full"
        assert summary["accepted"] is True
        assert summary["accepted_traces"] == 2
        assert summary["trace_count"] == 2


def test_subspace_acceptance_json_from_trace_dir(tmp_path, capsys):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for index in range(2):
        time, trace = synthetic_trace(64)
        np.savez(trace_dir / f"trace-{index}.npz", time=time, trace=trace)

    assert (
        main(
            [
                "subspace-acceptance",
                "--trace-dir",
                str(trace_dir),
                "--model-order",
                "5",
                "--components",
                "6",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is True
    assert payload["trace_count"] == 2
    assert [
        trace["source"]["path"].rsplit("/", 1)[-1]
        for trace in payload["traces"]
    ] == ["trace-0.npz", "trace-1.npz"]


def test_subspace_acceptance_json_compares_svd_backends(tmp_path, capsys):
    pytest.importorskip("scipy.sparse.linalg")

    time, trace = synthetic_trace(64)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "subspace-acceptance",
                "--trace-npz",
                str(trace_path),
                "--model-order",
                "5",
                "--components",
                "6",
                "--method",
                "matrix-pencil",
                "--svd-backend",
                "full",
                "--svd-backend",
                "randomized",
                "--svd-backend",
                "partial",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is True
    assert payload["accepted_methods"] == [
        "matrix-pencil:full",
        "matrix-pencil:randomized",
        "matrix-pencil:partial",
    ]
    assert [
        (summary["method"], summary["svd_backend"])
        for summary in payload["method_summary"]
    ] == [
        ("matrix-pencil", "full"),
        ("matrix-pencil", "randomized"),
        ("matrix-pencil", "partial"),
    ]


def test_subspace_acceptance_rejects_strict_diff_ratio(tmp_path, capsys):
    time, trace = synthetic_trace(64)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "subspace-acceptance",
                "--trace-npz",
                str(trace_path),
                "--model-order",
                "5",
                "--components",
                "6",
                "--max-diff-ratio",
                "0.0",
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is False
    assert payload["accepted_methods"] == []
    for summary in payload["method_summary"]:
        assert summary["accepted"] is False
        assert summary["accepted_traces"] == 0


def test_subspace_acceptance_rejects_with_text_status(tmp_path, capsys):
    time, trace = synthetic_trace(64)
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, time=time, trace=trace)

    assert (
        main(
            [
                "subspace-acceptance",
                "--trace-npz",
                str(trace_path),
                "--model-order",
                "5",
                "--components",
                "6",
                "--max-diff-ratio",
                "0.0",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out

    assert "status=rejected" in output
    assert "trace_count=1" in output
    assert "accepted=False" in output
