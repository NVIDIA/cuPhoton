# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.core.cli import ApplicationContext, get_component, run_component
from cuphoton.xpois.spatial_als import SpatialALSConfig


def _run_cli(argv: list[str]) -> int:
    return run_component("xpois", argv)


def test_component_registry_uses_canonical_namespace() -> None:
    component = get_component("xpois")

    assert component.group == "xpois"
    assert component.import_name == "cuphoton.xpois"


def test_group_program_name_is_used_in_help_output(capsys) -> None:
    rc = _run_cli(["help", "data-inspect"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois data-inspect" in captured.out
    assert "--base" in captured.out
    assert "-c FILE, --conf=FILE" not in captured.out


def test_help_for_fit_kernel_command(capsys) -> None:
    rc = _run_cli(["help", "fit-kernel"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois fit-kernel" in captured.out
    assert "--reference" in captured.out
    assert "--variance-hdu" in captured.out
    assert "--basis-sigmas" in captured.out
    assert "--backend" in captured.out
    assert "--solver" in captured.out
    unwrapped_help = " ".join(captured.out.split())
    assert "Kernel model to fit: constant or spatial-als." in unwrapped_help
    assert "spatial-als auto tries cupy, then cpu." in unwrapped_help
    assert "--spatial-degree" in captured.out
    assert "--als-iterations" in captured.out
    assert "--als-tolerance" in captured.out
    assert "--als-regularization" in captured.out
    assert "[default: auto]" in captured.out
    assert f"[default: {SpatialALSConfig.max_iterations}]" in captured.out
    assert f"[default: {SpatialALSConfig.tolerance}]" in captured.out


def test_help_for_benchmark_backends_command_case(capsys) -> None:
    rc = _run_cli(["help", "benchmark-backends"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois benchmark-backends" in captured.out
    assert "--backends" in captured.out
    assert "--reference-backend" in captured.out
    assert "--repeats" in captured.out
    assert "--solver" in captured.out
    assert "--spatial-degree" in captured.out
    assert "--als-iterations" in captured.out
    assert "--als-tolerance" in captured.out
    assert "--als-regularization" in captured.out


def test_help_for_fit_batch_command(capsys) -> None:
    rc = _run_cli(["help", "fit-batch"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois fit-batch" in captured.out
    assert "--executor {dragon,mpi}" in captured.out
    assert "--manifest" in captured.out
    assert "--max-workers" in captured.out
    assert "--result-timeout-sec" in captured.out
    assert "--worker-timeout-sec" in captured.out
    assert "--aggregation-mode" in captured.out
    assert "--rank-setup-timeout-sec" in captured.out
    assert "--rank-timeout-sec" in captured.out
    assert "--attempt-id" in captured.out
    assert "--backend {cupy,cutile,numba-cuda}" in captured.out
    assert "Dragon default: 60.0" in captured.out
    assert "MPI-only rank-metadata aggregation mode" in captured.out
    assert "[MPI default:" in captured.out
    assert "--reference" not in captured.out
    assert "--target" not in captured.out
    assert "--variance" not in captured.out
    assert "--fit-mask" not in captured.out
    assert "--solver" in captured.out
    assert "--spatial-degree" in captured.out
    assert "--als-iterations" in captured.out
    assert "--als-tolerance" in captured.out
    assert "--als-regularization" in captured.out


def test_fit_kernel_forwards_spatial_solver_options(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    reference.touch()
    target.touch()
    seen = {}

    def fake_fit(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(summary={"solver": kwargs["solver"]})

    monkeypatch.setattr(commands, "run_constant_kernel_fit", fake_fit)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--solver",
            "spatial-als",
            "--backend",
            "cupy",
            "--spatial-degree",
            "3",
            "--als-iterations",
            "17",
            "--als-tolerance",
            "2e-7",
            "--als-regularization",
            "4e-5",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == {"solver": "spatial-als"}
    assert seen["solver"] == "spatial-als"
    assert seen["backend"] == "cupy"
    assert seen["spatial_degree"] == 3
    assert seen["als_iterations"] == 17
    assert seen["als_tolerance"] == pytest.approx(2e-7)
    assert seen["als_regularization"] == pytest.approx(4e-5)


def test_benchmark_backends_forwards_spatial_solver_options(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    reference.touch()
    target.touch()
    seen = {}

    def fake_benchmark(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(summary={"solver": kwargs["solver"]})

    monkeypatch.setattr(
        commands,
        "benchmark_constant_kernel_backends",
        fake_benchmark,
    )

    rc = _run_cli(
        [
            "benchmark-backends",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--backends",
            "cpu",
            "--solver",
            "spatial-als",
            "--spatial-degree",
            "3",
            "--als-iterations",
            "17",
            "--als-tolerance",
            "2e-7",
            "--als-regularization",
            "4e-5",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == {"solver": "spatial-als"}
    assert seen["solver"] == "spatial-als"
    assert seen["spatial_degree"] == 3
    assert seen["als_iterations"] == 17
    assert seen["als_tolerance"] == pytest.approx(2e-7)
    assert seen["als_regularization"] == pytest.approx(4e-5)


def test_subtract_forwards_unset_spatial_options_for_constant_solver(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    reference.touch()
    target.touch()
    seen = {}

    def fake_fit(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(summary={"solver": kwargs["solver"]})

    monkeypatch.setattr(commands, "run_constant_kernel_fit", fake_fit)

    rc = _run_cli(
        [
            "subtract",
            "--reference",
            str(reference),
            "--target",
            str(target),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == {"solver": "constant"}
    assert seen["solver"] == "constant"
    assert seen["spatial_degree"] is None
    assert seen["als_iterations"] is None
    assert seen["als_tolerance"] is None
    assert seen["als_regularization"] is None


def _write_cli_spatial_inputs(tmp_path) -> tuple[str, str]:
    rng = np.random.default_rng(91)
    source = rng.normal(size=(33, 35))
    coordinates = np.arange(15, dtype=np.float64) - 7.0
    line = np.exp(-(coordinates**2) / (2.0 * 1.5**2))
    line /= line.sum()
    kernel = np.outer(line, line)
    patches = np.lib.stride_tricks.sliding_window_view(source, (15, 15))
    target = np.zeros_like(source)
    target[7:-7, 7:-7] = (
        np.einsum(
            "yxvu,vu->yx",
            patches,
            kernel[::-1, ::-1],
            optimize=True,
        )
        + 0.1
    )
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    np.save(reference_path, source, allow_pickle=False)
    np.save(target_path, target, allow_pickle=False)
    return str(reference_path), str(target_path)


def test_fit_kernel_runs_spatial_solver_with_cli_defaults(
    tmp_path, capsys
) -> None:
    reference_path, target_path = _write_cli_spatial_inputs(tmp_path)
    output_dir = tmp_path / "runs"

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference_path),
            "--target",
            str(target_path),
            "--solver",
            "spatial-als",
            "--output-dir",
            str(output_dir),
            "--name",
            "spatial-defaults",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    summary = json.loads(captured.out)
    assert summary["solver"] == "spatial-als"
    assert summary["requested_backend"] == "auto"
    assert summary["basis"] == [
        {"sigma": 1.5, "degree": 2},
        {"sigma": 3.0, "degree": 1},
        {"sigma": 6.0, "degree": 0},
    ]
    assert summary["flux_conserve"] is False
    assert summary["spatial_als"]["spatial_degree"] == 2
    assert summary["spatial_als"]["converged"] is True
    assert summary["converged"] is True
    assert summary["backend"] == "cpu"
    assert "WARNING" not in captured.err


@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_fit_kernel_reports_single_sweep_convergence(
    tmp_path, capsys, noise
) -> None:
    reference_path, target_path = _write_cli_spatial_inputs(tmp_path)
    if noise:
        target = np.load(target_path)
        target += np.random.default_rng(921).normal(
            scale=noise, size=target.shape
        )
        np.save(target_path, target, allow_pickle=False)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            reference_path,
            "--target",
            target_path,
            "--solver",
            "spatial-als",
            "--als-iterations",
            "1",
            "--output-dir",
            str(tmp_path / "runs"),
            "--name",
            "spatial-one-sweep",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    summary = json.loads(captured.out)
    assert summary["converged"] is (noise == 0.0)
    assert summary["iterations"] == 1
    if noise:
        assert "WARNING: spatial ALS did not converge in 1 of 1 sweeps" in (
            captured.err
        )
    else:
        assert "WARNING" not in captured.err


def _load_dragon_example():
    example_path = (
        Path(__file__).parents[2] / "examples" / "xpois" / "dragon_batch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cuphoton_test_dragon_batch", example_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arguments",
    [
        ["--e=mpi"],
        ["--exec", "mpi"],
        ["--exec=mpi"],
        ["--executor"],
        ["--executor=mpi"],
    ],
)
def test_dragon_example_rejects_executor_override(
    arguments, monkeypatch
) -> None:
    module = _load_dragon_example()
    monkeypatch.setattr(
        module,
        "run_component",
        lambda *args, **kwargs: pytest.fail("CLI must not run"),
    )

    with pytest.raises(SystemExit, match="fixes --executor=dragon"):
        module.main(arguments)


@pytest.mark.parametrize(
    "arguments", [["--backend", "numba-cuda"], ["--backend=cutile"]]
)
def test_dragon_example_preserves_explicit_backend(
    arguments, monkeypatch
) -> None:
    module = _load_dragon_example()
    seen = {}

    def run_component(component, arguments):
        seen["component"] = component
        seen["arguments"] = arguments
        return 7

    monkeypatch.setattr(module, "run_component", run_component)

    assert module.main(arguments) == 7
    assert seen == {
        "component": "xpois",
        "arguments": [
            "fit-batch",
            "--executor",
            "dragon",
            *arguments,
        ],
    }


def test_dragon_example_requires_explicit_backend(capsys) -> None:
    module = _load_dragon_example()

    assert module.main(["--manifest", "pairs.yaml"]) == 2
    captured = capsys.readouterr()
    assert "--backend" in captured.err


def test_fit_batch_dispatches_dragon_with_effective_defaults(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    manifest = tmp_path / "pairs.yaml"
    manifest.write_text("schema: cuphoton.xpois.image-pairs/v1\n")
    summary = tmp_path / "run" / "summary.json"
    seen = {}

    def run_dragon(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            status="success",
            summary_path=summary,
            to_dict=lambda: {
                "executor": "dragon",
                "run_id": "dragon-run",
                "status": "success",
            },
        )

    monkeypatch.setattr(commands, "_run_dragon_image_pair_batch", run_dragon)

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "dragon",
            "--manifest",
            str(manifest),
            "--backend",
            "cupy",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == {
        "executor": "dragon",
        "run_id": "dragon-run",
        "status": "success",
    }
    assert seen["manifest_path"] == manifest
    assert seen["options"].backend == "cupy"
    assert seen["options"].solver == "constant"
    assert seen["max_workers"] is None
    assert seen["result_timeout_sec"] == 60.0
    assert seen["worker_timeout_sec"] == 3600.0


def test_fit_batch_dispatches_explicit_dragon_limits(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    seen = {}

    def run_dragon(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            status="success",
            summary_path=tmp_path / "run" / "summary.json",
            to_dict=lambda: {"executor": "dragon", "status": "success"},
        )

    monkeypatch.setattr(commands, "_run_dragon_image_pair_batch", run_dragon)

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "dragon",
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--backend",
            "cupy",
            "--max-workers",
            "7",
            "--result-timeout-sec",
            "12.5",
            "--worker-timeout-sec",
            "34.5",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out)["executor"] == "dragon"
    assert seen["max_workers"] == 7
    assert seen["result_timeout_sec"] == 12.5
    assert seen["worker_timeout_sec"] == 34.5


def test_fit_batch_dispatches_mpi_with_effective_defaults(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    manifest = tmp_path / "pairs.yaml"
    manifest.write_text("schema: cuphoton.xpois.image-pairs/v1\n")
    seen = {}

    def run_mpi(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            status="success",
            summary_path=tmp_path / "run" / "summary.json",
            to_dict=lambda: {"executor": "mpi", "status": "success"},
        )

    monkeypatch.setattr(commands, "_run_mpi_image_pair_batch", run_mpi)

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "mpi",
            "--manifest",
            str(manifest),
            "--backend",
            "numba-cuda",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == {
        "executor": "mpi",
        "status": "success",
    }
    assert seen["manifest_path"] == manifest
    assert seen["options"].backend == "numba-cuda"
    assert seen["aggregation_mode"] == "mpi"
    assert seen["rank_setup_timeout_sec"] == 600.0
    assert seen["rank_timeout_sec"] is None
    assert seen["attempt_id"] is None


def test_fit_batch_mpi_nonroot_emits_nothing(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    monkeypatch.setattr(
        commands,
        "_run_mpi_image_pair_batch",
        lambda **kwargs: None,
    )

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "mpi",
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--backend",
            "cupy",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == ""
    assert captured.err == ""


def test_fit_batch_dispatches_explicit_mpi_file_aggregation(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    seen = {}

    def run_mpi(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            status="success",
            summary_path=tmp_path / "run" / "summary.json",
            to_dict=lambda: {"executor": "mpi", "status": "success"},
        )

    monkeypatch.setattr(commands, "_run_mpi_image_pair_batch", run_mpi)

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "mpi",
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--backend",
            "cupy",
            "--aggregation-mode",
            "files",
            "--name",
            "mpi-file-run",
            "--attempt-id",
            "attempt-1",
            "--rank-setup-timeout-sec",
            "17.5",
            "--rank-timeout-sec",
            "23.5",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out)["executor"] == "mpi"
    assert seen["run_id"] == "mpi-file-run"
    assert seen["aggregation_mode"] == "files"
    assert seen["rank_setup_timeout_sec"] == 17.5
    assert seen["rank_timeout_sec"] == 23.5
    assert seen["attempt_id"] == "attempt-1"


@pytest.mark.parametrize("missing", ["--name", "--attempt-id"])
def test_fit_batch_mpi_files_requires_explicit_identity_before_runtime(
    missing, monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    monkeypatch.setattr(
        commands,
        "BatchFitOptions",
        lambda **kwargs: pytest.fail("batch options must not be constructed"),
    )
    monkeypatch.setattr(
        commands,
        "_run_mpi_image_pair_batch",
        lambda **kwargs: pytest.fail("MPI runtime must not be called"),
    )
    arguments = [
        "fit-batch",
        "--executor",
        "mpi",
        "--manifest",
        str(tmp_path / "pairs.yaml"),
        "--backend",
        "cupy",
        "--aggregation-mode",
        "files",
        "--name",
        "mpi-file-run",
        "--attempt-id",
        "attempt-1",
    ]
    index = arguments.index(missing)
    del arguments[index : index + 2]

    rc = _run_cli(arguments)
    captured = capsys.readouterr()

    assert rc == 1
    assert (
        f"{missing} must be provided with --aggregation-mode files"
        in captured.err
    )
    assert "Traceback" not in captured.err


def test_fit_batch_wraps_invalid_options(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    manifest = tmp_path / "pairs.yaml"
    manifest.write_text("schema: cuphoton.xpois.image-pairs/v1\n")
    called = False

    def should_not_run(**kwargs):
        del kwargs
        nonlocal called
        called = True

    monkeypatch.setattr(
        commands, "_run_dragon_image_pair_batch", should_not_run
    )

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "dragon",
            "--manifest",
            str(manifest),
            "--backend",
            "cupy",
            "--kernel-height",
            "8",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert called is False
    assert "kernel_shape must contain two positive odd values" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (["--backend", "auto"], "invalid choice: 'auto'"),
        (["--backend", "dragon"], "invalid choice: 'dragon'"),
        (["--executor", "cupy"], "invalid choice: 'cupy'"),
        (
            ["--executor", "mpi", "--aggregation-mode", "auto"],
            "invalid choice: 'auto'",
        ),
    ),
)
def test_fit_batch_rejects_executor_backend_category_errors(
    arguments, message, tmp_path, capsys
) -> None:
    base = [
        "fit-batch",
        "--executor",
        "dragon",
        "--manifest",
        str(tmp_path / "pairs.yaml"),
        "--backend",
        "cupy",
    ]

    rc = _run_cli([*base, *arguments])
    captured = capsys.readouterr()

    assert rc == 2
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_fit_batch_emits_failed_result_before_nonzero_exit(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    summary_path = tmp_path / "failed" / "summary.json"
    monkeypatch.setattr(
        commands,
        "_run_dragon_image_pair_batch",
        lambda **kwargs: SimpleNamespace(
            status="failed",
            summary_path=summary_path,
            to_dict=lambda: {"executor": "dragon", "status": "failed"},
        ),
    )

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "dragon",
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--backend",
            "cupy",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert json.loads(captured.out) == {
        "executor": "dragon",
        "status": "failed",
    }
    assert f"Dragon batch failed; inspect {summary_path}" in captured.err


def test_fit_batch_rejects_executor_provenance_mismatch(
    monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    monkeypatch.setattr(
        commands,
        "_run_dragon_image_pair_batch",
        lambda **kwargs: SimpleNamespace(
            status="success",
            summary_path=tmp_path / "run" / "summary.json",
            to_dict=lambda: {"executor": "mpi", "status": "success"},
        ),
    )

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "dragon",
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--backend",
            "cupy",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    assert "dragon executor returned invalid provenance" in captured.err


@pytest.mark.parametrize("missing", ["executor", "backend"])
def test_fit_batch_requires_executor_and_backend(
    missing, tmp_path, capsys
) -> None:
    arguments = [
        "fit-batch",
        "--executor",
        "dragon",
        "--manifest",
        str(tmp_path / "pairs.yaml"),
        "--backend",
        "cupy",
    ]
    flag = f"--{missing}"
    index = arguments.index(flag)
    del arguments[index : index + 2]

    rc = _run_cli(arguments)
    captured = capsys.readouterr()

    assert rc == 2
    assert flag in captured.err


@pytest.mark.parametrize(
    ("executor", "option", "value"),
    (
        ("dragon", "--aggregation-mode", "mpi"),
        ("dragon", "--rank-setup-timeout-sec", "12"),
        ("dragon", "--rank-timeout-sec", "12"),
        ("dragon", "--attempt-id", "attempt-1"),
        ("mpi", "--max-workers", "2"),
        ("mpi", "--result-timeout-sec", "12"),
        ("mpi", "--worker-timeout-sec", "12"),
    ),
)
def test_fit_batch_rejects_wrong_executor_options_before_runtime(
    executor, option, value, monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    def should_not_construct_options(**kwargs):
        del kwargs
        pytest.fail("batch options must not be constructed")

    def should_not_run(**kwargs):
        del kwargs
        pytest.fail("executor runtime must not be called")

    monkeypatch.setattr(
        commands, "BatchFitOptions", should_not_construct_options
    )
    monkeypatch.setattr(
        commands, "_run_dragon_image_pair_batch", should_not_run
    )
    monkeypatch.setattr(commands, "_run_mpi_image_pair_batch", should_not_run)

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            executor,
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--backend",
            "cupy",
            option,
            value,
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert (
        f"{option} cannot be used with --executor {executor}" in captured.err
    )
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--rank-timeout-sec", "12"),
        ("--attempt-id", "attempt-1"),
    ),
)
def test_fit_batch_rejects_file_options_with_mpi_aggregation(
    option, value, monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    monkeypatch.setattr(
        commands,
        "_run_mpi_image_pair_batch",
        lambda **kwargs: pytest.fail("MPI runtime must not be called"),
    )

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            "mpi",
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--backend",
            "cupy",
            option,
            value,
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert (
        f"{option} can be used only with --aggregation-mode files"
        in captured.err
    )
    assert "Traceback" not in captured.err


def test_help_for_evaluate_subtraction_command(capsys) -> None:
    rc = _run_cli(["help", "evaluate-subtraction"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois evaluate-subtraction" in captured.out
    assert "--run-dir" in captured.out


def test_help_for_review_bokeh_command(capsys) -> None:
    rc = _run_cli(["help", "review-bokeh"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois review-bokeh" in captured.out
    assert "--run-dir" in captured.out


def test_zero_arg_command_rejects_positional_argument(capsys) -> None:
    rc = _run_cli(["data-inspect", "extra"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "invalid number of arguments" in captured.err


def test_top_level_help_exits_zero_to_stdout_case(capsys) -> None:
    rc = _run_cli(["help"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Available subcommands" in captured.out
    assert captured.err == ""


def test_missing_subcommand_help_is_nonzero(capsys) -> None:
    rc = _run_cli(["help", "missing-command"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Unknown subcommand 'missing-command'" in captured.err


def test_data_inspect_uses_env_shared_hsc_root(
    monkeypatch, tmp_path, capsys
) -> None:
    hsc_root = tmp_path / "shared-hsc"
    hsc_root.mkdir()
    monkeypatch.setenv("CUPHOTON_XPOIS_SHARED_HSC_DIR", str(hsc_root))

    rc = _run_cli(["data-inspect"])
    captured = capsys.readouterr()

    assert rc == 0
    assert str(hsc_root) in captured.out


def test_shared_hsc_discovery_is_domain_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from cuphoton.xpois.data import default_shared_hsc_dir

    hsc_root = tmp_path / "data" / "HSC"
    nested = tmp_path / "work" / "nested"
    hsc_root.mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.delenv(
        "CUPHOTON_XPOIS_SHARED_HSC_DIR",
        raising=False,
    )
    monkeypatch.chdir(nested)

    assert default_shared_hsc_dir() == hsc_root.resolve()


def test_shared_hsc_fallback_does_not_create_xdg_data_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from cuphoton.xpois.data import default_shared_hsc_dir

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    data_home = tmp_path / "data-home"
    monkeypatch.delenv(
        "CUPHOTON_XPOIS_SHARED_HSC_DIR",
        raising=False,
    )
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.chdir(work_dir)

    expected = data_home / "cuphoton" / "xpois" / "HSC"
    assert default_shared_hsc_dir() == expected.resolve()
    assert not data_home.exists()


def test_xdg_paths_follow_env_after_import(monkeypatch, tmp_path) -> None:
    state_home = tmp_path / "xdg-state"
    config_home = tmp_path / "xdg-config"
    data_home = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    context = ApplicationContext.for_component("xpois")
    component_state = (state_home / "cuphoton" / "xpois").resolve()

    assert context.state_dir == component_state
    assert context.workspace_dir == component_state
    assert context.runs_dir == component_state / "runs"
    assert (
        context.config_dir == (config_home / "cuphoton" / "xpois").resolve()
    )
    assert context.data_dir == (data_home / "cuphoton" / "xpois").resolve()
    assert context.log_file == component_state / "logs" / "xpois.log"
    assert not component_state.exists()


def test_cli_smoke_fit_subtract_and_evaluate_case(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    output_root = tmp_path / "runs"
    arr = [[0.0] * 64 for _ in range(64)]
    for row in range(20, 40):
        for col in range(20, 40):
            arr[row][col] = 1.0
    import numpy as np

    np.save(reference, np.asarray(arr, dtype=np.float64), allow_pickle=False)
    np.save(target, np.asarray(arr, dtype=np.float64), allow_pickle=False)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(output_root),
            "--name",
            "fit-run",
            "--backend",
            "cpu",
        ]
    )
    captured = capsys.readouterr()
    fit_summary = json.loads(captured.out)
    assert rc == 0
    assert fit_summary["workflow"] == "fit_kernel"
    assert fit_summary["backend"] == "cpu"
    assert (output_root / "fit-run" / "summary.json").exists()
    assert (output_root / "fit-run" / "artifacts" / "residual.npy").exists()

    rc = _run_cli(
        [
            "subtract",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(output_root),
            "--name",
            "subtract-run",
        ]
    )
    captured = capsys.readouterr()
    subtract_summary = json.loads(captured.out)
    assert rc == 0
    assert subtract_summary["workflow"] == "subtract"
    assert subtract_summary["requested_backend"] == "auto"
    assert subtract_summary["backend"] == "cpu"
    assert (output_root / "subtract-run" / "summary.json").exists()

    rc = _run_cli(
        [
            "benchmark-backends",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(output_root),
            "--name",
            "benchmark-run",
            "--backends",
            "cpu",
            "--repeats",
            "2",
            "--warmup",
            "0",
        ]
    )
    captured = capsys.readouterr()
    benchmark_summary = json.loads(captured.out)
    assert rc == 0
    assert benchmark_summary["workflow"] == "benchmark-backends"
    assert benchmark_summary["backends"] == ["cpu"]
    assert benchmark_summary["parity"]["ok"] is True
    assert (output_root / "benchmark-run" / "summary.json").exists()

    rc = _run_cli(
        [
            "evaluate-subtraction",
            "--run-dir",
            str(output_root / "subtract-run"),
        ]
    )
    captured = capsys.readouterr()
    evaluation = json.loads(captured.out)
    assert rc == 0
    assert evaluation["fit_region_pixel_count"] > 0
    assert (output_root / "subtract-run" / "evaluation.json").exists()


def test_cli_benchmark_backends_default_basis_handles_synthetic_cutout(
    tmp_path, capsys
) -> None:
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    output_root = tmp_path / "runs"

    y_coords, x_coords = np.meshgrid(
        np.arange(80, dtype=np.float64),
        np.arange(80, dtype=np.float64),
        indexing="ij",
    )
    image = np.zeros((80, 80), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (22.0, 24.0, 3.0, 90.0),
        (50.0, 44.0, 5.0, 130.0),
        (60.0, 62.0, 2.5, 70.0),
    ):
        image += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    background = 0.05 + 0.01 * x_coords / image.shape[1]
    np.save(reference, image, allow_pickle=False)
    np.save(target, image + background, allow_pickle=False)
    np.save(variance, np.ones_like(image), allow_pickle=False)

    rc = _run_cli(
        [
            "benchmark-backends",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--output-dir",
            str(output_root),
            "--name",
            "default-basis-synthetic-cutout",
            "--backends",
            "cpu",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert rc == 0
    assert summary["workflow"] == "benchmark-backends"
    assert summary["basis"] == [
        {"sigma": 1.5, "degree": 2},
        {"sigma": 3.0, "degree": 1},
        {"sigma": 6.0, "degree": 0},
    ]
    assert summary["parity"]["ok"] is True
    assert summary["fit_pixel_count"]["cpu"] > 0


def test_cli_fit_kernel_supports_auto_stamp_mask(tmp_path, capsys) -> None:
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    y_coords, x_coords = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="ij",
    )
    arr = np.zeros((64, 64), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ):
        arr += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--auto-stamp-mask",
            "--auto-stamp-size",
            "15",
            "--auto-stamp-count",
            "2",
            "--auto-peak-percentile",
            "98.0",
            "--output-dir",
            str(tmp_path / "runs"),
            "--name",
            "auto-mask-run",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["fit_region"]["kind"] == "auto_stamp_mask"
    assert summary["fit_region"]["selected_count"] == 2
    assert "fit_mask_metadata" in summary["saved"]


@pytest.mark.parametrize("command", ["fit-kernel", "benchmark-backends"])
def test_cli_fit_kernel_supports_fits_mask_policy_crop_and_auto_stamps(
    tmp_path, capsys, command
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    y_coords, x_coords = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="ij",
    )
    arr = np.zeros((64, 64), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ):
        arr += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    variance = np.ones_like(arr, dtype=np.float64)
    mask = np.zeros_like(arr, dtype=np.int64)
    mask[12:16, 12:16] = 1 << 9  # CROSSTALK
    mask[20:24, 20:24] = 1 << 5  # DETECTED only; should survive masklite.

    def write_fits(path):
        image_hdu = fits.ImageHDU(arr, name="IMAGE")
        mask_hdu = fits.ImageHDU(mask, name="MASK")
        for key, bit in (
            ("MP_BAD", 0),
            ("MP_SAT", 1),
            ("MP_INTRP", 2),
            ("MP_CR", 3),
            ("MP_EDGE", 4),
            ("MP_DETECTED", 5),
            ("MP_DETECTED_NEGATIVE", 6),
            ("MP_SUSPECT", 7),
            ("MP_NO_DATA", 8),
            ("MP_CROSSTALK", 9),
            ("MP_NOT_DEBLENDED", 10),
            ("MP_UNMASKEDNAN", 11),
        ):
            header_key = f"HIERARCH {key}" if len(key) > 8 else key
            mask_hdu.header[header_key] = bit
        variance_hdu = fits.ImageHDU(variance, name="VARIANCE")
        fits.HDUList(
            [fits.PrimaryHDU(), image_hdu, mask_hdu, variance_hdu]
        ).writeto(path)

    write_fits(reference)
    write_fits(target)

    rc = _run_cli(
        [
            command,
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--reference-hdu",
            "1",
            "--target-hdu",
            "1",
            "--variance",
            str(target),
            "--variance-hdu",
            "3",
            "--mask-policy",
            "hsc-masklite",
            "--auto-stamp-mask",
            "--auto-stamp-size",
            "15",
            "--auto-stamp-count",
            "2",
            "--auto-peak-percentile",
            "98.0",
            "--crop-y0",
            "8",
            "--crop-x0",
            "8",
            "--crop-height",
            "48",
            "--crop-width",
            "48",
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
            "--name",
            "fits-auto-mask-run",
        ]
        + (
            ["--backends", "cpu", "--repeats", "1", "--warmup", "0"]
            if command == "benchmark-backends"
            else []
        )
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["mask_policy"] == "hsc-masklite"
    assert summary["crop"] == {
        "y0": 8,
        "x0": 8,
        "height": 48,
        "width": 48,
    }
    assert summary["fit_region"]["kind"] == "auto_stamp_mask"
    assert summary["fit_region"]["selected_count"] >= 1
    assert "fit_mask_metadata" in summary["saved"]
    assert "input_mask_metadata" in summary["saved"]
    assert summary["input_mask"]["reference_mask_fraction"] > 0.0


@pytest.mark.parametrize(
    ("command", "no_review"),
    [("fit-kernel", False), ("fit-kernel", True), ("subtract", True)],
)
def test_cli_review_bokeh_rebuilds_saved_run_case(
    tmp_path, capsys, command, no_review
) -> None:
    pytest.importorskip("bokeh")
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    y_coords, x_coords = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="ij",
    )
    arr = np.zeros((64, 64), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ):
        arr += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)

    mask_path = tmp_path / "mask.npy"
    mask = np.zeros(arr.shape, dtype=np.uint16)
    mask[10, 10] = 1
    np.save(mask_path, mask)

    rc = _run_cli(
        [
            command,
            *(["--no-review"] if no_review else []),
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--reference-mask",
            str(mask_path),
            "--target-mask",
            str(mask_path),
            "--mask-policy",
            "strict",
            "--variance",
            str(variance),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--auto-stamp-mask",
            "--auto-stamp-size",
            "15",
            "--auto-stamp-count",
            "2",
            "--auto-peak-percentile",
            "98.0",
            "--output-dir",
            str(tmp_path / "runs"),
            "--name",
            "review-source-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    summary = json.loads(captured.out)
    run_dir = tmp_path / "runs" / "review-source-run"
    assert summary["review_enabled"] is not no_review
    if no_review:
        assert not any(key.startswith("review_") for key in summary["saved"])
    else:
        review_bokeh_path = run_dir / summary["saved"]["review_bokeh_html"]
        review_bokeh_path.unlink()

    rc = _run_cli(
        [
            "review-bokeh",
            "--run-dir",
            str(run_dir),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert "review_bokeh_html" in payload["saved"]
    assert (run_dir / payload["saved"]["review_bokeh_html"]).exists()


def test_prime_preserves_falsey_loaded_options() -> None:
    from cuphoton.xpois import commands as xpois_commands

    command = xpois_commands.FitKernelCommand()
    command.options = {}
    command.load_order = [
        "reference_hdu",
        "target_hdu",
        "background_degree",
        "flux_conserve",
    ]
    command.reference_hdu = 0
    command.target_hdu = 0
    command.background_degree = 0
    command.flux_conserve = False

    primed = command.prime(xpois_commands.SubtractCommand)

    assert primed.reference_hdu == 0
    assert primed.target_hdu == 0
    assert primed.background_degree == 0
    assert primed.flux_conserve is False


def test_fit_kernel_accepts_hdu_zero_and_bad_hdu_is_user_facing(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    fits.PrimaryHDU(np.ones((32, 32), dtype=np.float64)).writeto(reference)
    fits.PrimaryHDU(np.ones((32, 32), dtype=np.float64)).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--reference-hdu",
            "0",
            "--target-hdu",
            "0",
            "--kernel-height",
            "15",
            "--kernel-width",
            "15",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "6",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "reference_hdu" not in captured.err.lower()
    assert "target_hdu" not in captured.err.lower()

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--reference-hdu",
            "99",
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs-2"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "out of range" in captured.err.lower()


def test_fit_kernel_supports_variance_hdu_selection(tmp_path, capsys) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    variance = tmp_path / "variance.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((32, 32), dtype=np.float64) * 0.5),
        ]
    ).writeto(variance)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--variance-hdu",
            "1",
            "--kernel-height",
            "15",
            "--kernel-width",
            "15",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "6",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert '"variance_hdu": 1' not in captured.err


def test_fit_kernel_rejects_ambiguous_variance_fits_without_hdu(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    variance = tmp_path / "variance.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)
    fits.HDUList(
        [
            fits.PrimaryHDU(arr),
            fits.ImageHDU(arr * 0.5),
        ]
    ).writeto(variance)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "specify --variance-hdu explicitly" in captured.err


def test_fit_kernel_rejects_variance_hdu_without_variance(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance-hdu",
            "1",
            "--kernel-height",
            "15",
            "--kernel-width",
            "15",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "6",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "variance_hdu requires a variance image path" in captured.err


def test_fit_kernel_rejects_invalid_basis_sigma_tokens(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--basis-sigmas",
            "1.5,bad",
            "--basis-degrees",
            "0,0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "--basis-sigmas must contain only numeric values" in captured.err


def test_fit_kernel_rejects_empty_basis_csvs_case(tmp_path, capsys) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--basis-sigmas",
            " , ",
            "--basis-degrees",
            " , ",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "must provide at least one basis term" in captured.err


def test_fit_kernel_rejects_all_non_finite_pixels(tmp_path, capsys) -> None:
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    np.save(reference, np.full((32, 32), np.nan, dtype=np.float64))
    np.save(target, np.full((32, 32), np.nan, dtype=np.float64))

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "no finite pixels remain within the fit region" in captured.err


def test_fit_kernel_wraps_filesystem_errors(
    monkeypatch, tmp_path, capsys
) -> None:
    import numpy as np

    from cuphoton.xpois import commands as xpois_commands

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.ones((32, 32), dtype=np.float64)
    np.save(reference, arr)
    np.save(target, arr)

    def fail(**kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(xpois_commands, "run_constant_kernel_fit", fail)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "permission denied" in captured.err


def test_data_inspect_missing_base_reports_zero_counts(
    tmp_path, capsys
) -> None:
    missing = tmp_path / "missing-hsc"
    rc = _run_cli(["data-inspect", "--base", str(missing)])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"counts": {' in captured.out
    assert '"bundle_files": 0' in captured.out


@pytest.mark.parametrize("executor", ["dragon", "mpi"])
def test_fit_batch_forwards_spatial_solver_options(
    executor, monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    manifest = tmp_path / "pairs.yaml"
    manifest.write_text("schema: cuphoton.xpois.image-pairs/v1\n")
    seen = {}

    def run_batch(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            status="success",
            summary_path=tmp_path / "run" / "summary.json",
            to_dict=lambda: {"executor": executor, "status": "success"},
        )

    monkeypatch.setattr(
        commands,
        f"_run_{executor}_image_pair_batch",
        run_batch,
    )

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            executor,
            "--manifest",
            str(manifest),
            "--backend",
            "cupy",
            "--solver",
            "spatial-als",
            "--spatial-degree",
            "3",
            "--als-iterations",
            "17",
            "--als-tolerance",
            "2e-7",
            "--als-regularization",
            "4e-5",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    assert json.loads(captured.out) == {
        "executor": executor,
        "status": "success",
    }
    options = seen["options"]
    assert options.backend == "cupy"
    assert options.solver == "spatial-als"
    assert options.spatial_degree == 3
    assert options.als_iterations == 17
    assert options.als_tolerance == pytest.approx(2e-7)
    assert options.als_regularization == pytest.approx(4e-5)


@pytest.mark.parametrize("executor", ["dragon", "mpi"])
def test_fit_batch_rejects_spatial_als_with_cutile(
    executor, monkeypatch, tmp_path, capsys
) -> None:
    from cuphoton.xpois import commands

    called = False

    def should_not_run(**kwargs):
        del kwargs
        nonlocal called
        called = True

    monkeypatch.setattr(
        commands,
        f"_run_{executor}_image_pair_batch",
        should_not_run,
    )

    rc = _run_cli(
        [
            "fit-batch",
            "--executor",
            executor,
            "--manifest",
            str(tmp_path / "pairs.yaml"),
            "--solver",
            "spatial-als",
            "--backend",
            "cutile",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert called is False
    assert "supports only auto, cpu, and cupy backends" in captured.err
    assert "Traceback" not in captured.err
