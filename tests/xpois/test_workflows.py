# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import importlib.util
import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.core import _native_imaging
from cuphoton.xpois import workflows
from cuphoton.xpois.ois import GaussianBasisComponent
from cuphoton.xpois.spatial_als import SpatialALSConfig


def _compact_source_image(shape: tuple[int, int]) -> np.ndarray:
    y_coords, x_coords = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        indexing="ij",
    )
    image = np.zeros(shape, dtype=np.float64)
    stars = [
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ]
    for cy, cx, sigma, amp in stars:
        image += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    return image


def _write_spatial_als_inputs(
    tmp_path: Path,
    *,
    noise: float = 0.0,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(3829)
    source = rng.normal(size=(31, 33))
    coordinates = np.arange(5, dtype=np.float64) - 2.0
    line = np.exp(-(coordinates**2) / (2.0 * 0.9**2))
    line /= line.sum()
    kernel = np.outer(line, line)
    patches = np.lib.stride_tricks.sliding_window_view(source, (5, 5))
    target = np.zeros_like(source)
    target[2:-2, 2:-2] = (
        np.einsum(
            "yxvu,vu->yx",
            patches,
            kernel[::-1, ::-1],
            optimize=True,
        )
        + 0.25
    )
    if noise:
        target += rng.normal(scale=noise, size=target.shape)
    reference_path = tmp_path / "spatial-reference.npy"
    target_path = tmp_path / "spatial-target.npy"
    np.save(reference_path, source, allow_pickle=False)
    np.save(target_path, target, allow_pickle=False)
    return reference_path, target_path


def test_run_constant_kernel_fit_cleans_failed_run_dir(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    np.save(
        reference, np.ones((16, 16), dtype=np.float64), allow_pickle=False
    )
    np.save(target, np.ones((16, 16), dtype=np.float64), allow_pickle=False)

    with pytest.raises(ValueError):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path,
            name="fit-fail",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(15, 15),
            components=[GaussianBasisComponent(sigma=1.5, degree=6)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=False,
        )

    assert not (tmp_path / "fit-fail").exists()


def test_run_constant_kernel_fit_uses_workflow_prefix_and_name(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="subtract-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
        workflow_name="subtract",
        run_prefix="subtract",
    )

    assert result.summary["workflow"] == "subtract"
    assert result.summary["requested_backend"] == "auto"
    assert result.summary["backend"] == "cpu"
    assert result.summary["device"] == "cpu"
    assert result.summary["dtype"] == "float64"
    assert result.summary["runtime"]["package_version"]
    assert result.run_dir.name == "subtract-run"


def test_run_constant_kernel_fit_rejects_spatial_als_unsupported_backend(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="spatial-als.*only auto, cpu, and cupy",
    ):
        workflows.run_constant_kernel_fit(
            reference_path=tmp_path / "missing-reference.npy",
            target_path=tmp_path / "missing-target.npy",
            output_root=tmp_path / "runs",
            name="spatial-gpu-run",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(5, 5),
            components=[GaussianBasisComponent(sigma=0.9, degree=1)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=True,
            backend="numba-cuda",
            solver="spatial-als",
        )

    assert not (tmp_path / "runs" / "spatial-gpu-run").exists()


def test_run_constant_kernel_fit_reports_cupy_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def without_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ModuleNotFoundError("No module named 'cupy'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cupy)
    reference, target = _write_spatial_als_inputs(tmp_path)
    with pytest.raises(ImportError, match="uv sync --extra gpu"):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            reference_hdu=None,
            target_hdu=None,
            output_root=tmp_path,
            name="missing-cupy",
            kernel_shape=(5, 5),
            components=[GaussianBasisComponent(sigma=0.9, degree=0)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=False,
            backend="cupy",
            review=False,
        )
    assert not (tmp_path / "missing-cupy").exists()


@pytest.mark.parametrize(
    "option",
    [
        {"spatial_degree": 2},
        {"als_iterations": 11},
        {"als_tolerance": 1e-6},
        {"als_regularization": 0.0},
    ],
)
def test_run_constant_kernel_fit_rejects_spatial_options_for_constant(
    tmp_path: Path,
    option: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="require solver='spatial-als'"):
        workflows.run_constant_kernel_fit(
            reference_path=tmp_path / "missing-reference.npy",
            target_path=tmp_path / "missing-target.npy",
            output_root=tmp_path / "runs",
            name="constant-with-als-option",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(5, 5),
            components=[GaussianBasisComponent(sigma=0.9, degree=1)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=False,
            solver="constant",
            **option,
        )

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("weighted", [False, True])
def test_spatial_gaussian_polynomial_crop_preserves_model_and_artifacts(
    tmp_path: Path,
    weighted: bool,
) -> None:
    from scipy.signal import convolve2d

    from cuphoton.xpois.ois import build_gaussian_polynomial_basis

    rng = np.random.default_rng(812)
    source = rng.normal(size=(41, 43))
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, source.shape[0]),
        np.linspace(-1.0, 1.0, source.shape[1]),
        indexing="ij",
    )
    components = [GaussianBasisComponent(sigma=0.9, degree=2)]
    basis, _ = build_gaussian_polynomial_basis(
        (5, 5),
        components,
        flux_conserve=True,
    )
    scale = 1.0 + 0.1 * xx - 0.05 * yy
    shape_fields = np.stack(
        [
            0.02 * (index + 1) * xx - 0.01 * yy
            for index in range(len(basis) - 1)
        ]
    )
    target = scale * convolve2d(source, basis[0], mode="same")
    for kernel, field in zip(basis[1:], shape_fields, strict=True):
        target += field * convolve2d(source, kernel, mode="same")
    target += 0.2 + 0.03 * yy
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    fit_mask_path = tmp_path / "fit-mask.npy"
    fit_mask = np.ones(source.shape, dtype=bool)
    fit_mask[15:18, 17:21] = False
    np.save(reference_path, source)
    np.save(target_path, target)
    np.save(fit_mask_path, fit_mask)
    variance_path = None
    variance = 2.0 + 0.2 * xx
    if weighted:
        variance_path = tmp_path / "variance.npy"
        np.save(variance_path, variance)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference_path,
        target_path=target_path,
        reference_hdu=None,
        target_hdu=None,
        output_root=tmp_path / "runs",
        name="spatial-gp",
        kernel_shape=(5, 5),
        components=components,
        variance_path=variance_path,
        fit_mask_path=fit_mask_path,
        background_degree=1,
        flux_conserve=False,
        solver="spatial-gaussian-polynomial",
        spatial_degree=1,
        photometric_degree=1,
        backend="cpu",
        review=False,
        crop_y0=6,
        crop_x0=7,
        crop_height=27,
        crop_width=29,
    )
    summary = result.summary
    model = summary["spatial_gaussian_polynomial"]
    assert model["spatial_domain"] == {
        "array_origin_yx": [6, 7],
        "normalization_bbox": [0, 41, 0, 43],
    }
    assert summary["fit_weighting"] == (
        "target_variance" if weighted else "uniform"
    )
    assert "chi2" not in summary
    assert "flux_conserve" not in summary
    arrays = {
        name: np.load(result.run_dir / path)
        for name, path in summary["saved"].items()
        if path.endswith(".npy")
    }
    np.testing.assert_allclose(
        arrays["matched"][2:-2, 2:-2],
        target[8:31, 9:34],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        arrays["photometric_scale"],
        scale[6:33, 7:36],
        atol=1e-12,
    )
    np.testing.assert_allclose(arrays["basis_kernels"], basis)
    assert arrays["shape_coefficients"].shape == (len(basis) - 1, 3)
    assert arrays["photometric_coefficients"].shape == (3,)
    assert arrays["background_coefficients"].shape == (3,)
    assert not arrays["fit_mask"][9:12, 10:14].any()
    residual = arrays["residual"][arrays["fit_mask"]]
    objective = np.square(residual)
    if weighted:
        objective /= variance[6:33, 7:36][arrays["fit_mask"]]
        np.testing.assert_allclose(
            arrays["marginal_residual_variance"][2:-2, 2:-2],
            variance[8:31, 9:34],
        )
    else:
        assert "marginal_residual_variance" not in arrays
    assert summary["fit_objective"] == pytest.approx(
        float(objective.sum()),
        rel=1e-12,
        abs=1e-30,
    )
    evaluation = workflows.evaluate_subtraction_run(result.run_dir)
    assert evaluation["fit_weighting"] == summary["fit_weighting"]
    assert "converged" not in evaluation


def test_run_spatial_als_resolves_unset_options_from_config_defaults(
    tmp_path: Path,
) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path / "runs",
        name="spatial-als-defaults",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(5, 5),
        components=[GaussianBasisComponent(sigma=0.9, degree=1)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
        solver="spatial-als",
    )
    defaults = SpatialALSConfig()
    recorded = result.summary["spatial_als"]

    assert recorded["spatial_degree"] == defaults.spatial_degree
    assert recorded["max_iterations"] == defaults.max_iterations
    assert recorded["tolerance"] == defaults.tolerance
    assert recorded["regularization"] == defaults.regularization
    assert "flux_scale" not in recorded
    assert (
        "not a standalone photometric scale"
        in (recorded["vertical_reference_scale_interpretation"])
    )


def test_run_spatial_als_normalizes_coordinates_over_the_crop(
    tmp_path: Path,
) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path / "runs",
        name="spatial-als-crop",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(5, 5),
        components=[GaussianBasisComponent(sigma=0.9, degree=1)],
        variance_path=None,
        fit_mask_path=None,
        crop_y0=3,
        crop_x0=2,
        crop_height=25,
        crop_width=27,
        background_degree=0,
        flux_conserve=True,
        solver="spatial-als",
    )

    assert result.summary["crop"] == {
        "y0": 3,
        "x0": 2,
        "height": 25,
        "width": 27,
    }
    assert result.summary["image_shape"] == [25, 27]
    normalization = result.summary["spatial_als"]["coordinate_normalization"]
    assert "image_shape" in normalization
    assert "crop" in normalization
    center = np.load(result.run_dir / "artifacts" / "kernel_center.npy")
    assert center.shape == (5, 5)


def test_run_spatial_als_surfaces_non_convergence(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path, noise=0.05)

    with caplog.at_level(logging.WARNING, logger="cuphoton.xpois"):
        result = workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path / "runs",
            name="spatial-als-stalled",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(5, 5),
            components=[GaussianBasisComponent(sigma=0.9, degree=1)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=True,
            solver="spatial-als",
            als_iterations=2,
            als_tolerance=0.0,
        )

    summary = result.summary
    assert summary["converged"] is False
    assert summary["iterations"] == 2
    assert summary["spatial_als"]["converged"] is False
    change = summary["spatial_als"]["final_relative_objective_change"]
    assert change is not None and 0.0 < change < 1.0
    assert any(
        record.levelno == logging.WARNING
        and "did not converge in 2 of 2 sweeps" in record.getMessage()
        for record in caplog.records
    )

    evaluation = workflows.evaluate_subtraction_run(result.run_dir)
    assert evaluation["converged"] is False
    assert evaluation["iterations"] == 2
    written = json.loads(
        (result.run_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert written["converged"] is False


@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_run_spatial_als_single_sweep_reports_no_objective_change(
    tmp_path: Path,
    noise: float,
) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path, noise=noise)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path / "runs",
        name="spatial-als-one-sweep",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(5, 5),
        components=[GaussianBasisComponent(sigma=0.9, degree=1)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=True,
        solver="spatial-als",
        als_iterations=1,
    )

    assert result.summary["converged"] is (noise == 0.0)
    assert result.summary["iterations"] == 1
    assert (
        result.summary["spatial_als"]["final_relative_objective_change"]
        is None
    )


def test_run_spatial_als_cleans_failed_run_dir(tmp_path: Path) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path)

    with pytest.raises(ValueError, match="underdetermined"):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path / "runs",
            name="spatial-fail",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(15, 15),
            components=[GaussianBasisComponent(sigma=1.5, degree=4)],
            variance_path=None,
            fit_mask_path=None,
            crop_y0=0,
            crop_x0=0,
            crop_height=21,
            crop_width=21,
            background_degree=2,
            flux_conserve=True,
            backend="cpu",
            solver="spatial-als",
        )

    assert not (tmp_path / "runs" / "spatial-fail").exists()


@pytest.mark.parametrize("review_enabled", [False, True])
@pytest.mark.parametrize("backend", ["auto", "cupy"])
def test_run_spatial_als_writes_solver_specific_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    review_enabled: bool,
    backend: str,
) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path)
    if backend == "cupy":
        # Check dispatch on CPU; real GPU parity is tested separately.
        spatial_solver = workflows.solve_spatial_als

        def cpu_spatial_solver(*args, **kwargs):
            assert kwargs.pop("backend") == "cupy"
            return spatial_solver(*args, backend="cpu", **kwargs)

        def unexpected_device_preparation():
            pytest.fail("spatial ALS must retain its own preparation path")

        monkeypatch.setattr(
            workflows, "solve_spatial_als", cpu_spatial_solver
        )
        monkeypatch.setattr(
            workflows, "_load_cupy", unexpected_device_preparation
        )

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path / "runs",
        name="spatial-als-run",
        review=review_enabled,
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(5, 5),
        components=[GaussianBasisComponent(sigma=0.9, degree=1)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=True,
        backend=backend,
        solver="spatial-als",
        spatial_degree=1,
        als_iterations=12,
        als_tolerance=1e-9,
        als_regularization=1e-8,
    )

    assert result.summary["solver"] == "spatial-als"
    assert result.summary["requested_backend"] == backend
    assert result.summary["backend"] == "cpu"
    assert result.summary["kernel_sum_center"] == pytest.approx(
        result.summary["spatial_als"]["flux_scale"],
        abs=1e-10,
    )
    assert result.summary["spatial_als"]["spatial_degree"] == 1
    assert result.summary["spatial_als"][
        "vertical_reference_scale_interpretation"
    ] == ("position-independent signed kernel sum")
    saved = result.summary["saved"]
    assert "kernel" not in saved
    assert not (result.run_dir / "artifacts" / "kernel.npy").exists()
    for name in (
        "kernel_center",
        "horizontal_coefficients",
        "vertical_coefficients",
        "background_coefficients",
        "objective_history",
    ):
        assert name in saved
        assert (result.run_dir / saved[name]).exists()

    evaluation = workflows.evaluate_subtraction_run(result.run_dir)
    assert evaluation["solver"] == "spatial-als"
    assert evaluation["kernel_sum_center"] == pytest.approx(
        result.summary["kernel_sum_center"]
    )
    assert evaluation["converged"] is True
    assert evaluation["iterations"] == result.summary["iterations"]
    assert "kernel_sum" not in evaluation

    if importlib.util.find_spec("bokeh") is not None:
        review = workflows.rebuild_interactive_review(result.run_dir)
        assert review["run_dir"] == str(result.run_dir.resolve())
        assert review["fit_region_pixel_count"] > 0


def test_benchmark_constant_kernel_backends_writes_parity_artifacts(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.benchmark_constant_kernel_backends(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="backend-benchmark",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
        backends=["cpu"],
        reference_backend="cpu",
        repeats=2,
        warmup=0,
    )

    assert result.summary["workflow"] == "benchmark-backends"
    assert result.summary["backends"] == ["cpu"]
    assert result.summary["runtimes"]["cpu"]["device"] == "cpu"
    assert result.summary["parity"]["ok"] is True
    assert result.summary["timings"]["cpu"]["count"] == 2
    assert result.summary["timings"]["cpu"]["best"] >= 0.0
    assert result.summary["parity"]["comparisons"]["cpu"]["ok"] is True
    assert (result.run_dir / result.summary["saved"]["timings_json"]).exists()
    assert (
        result.run_dir / result.summary["saved"]["comparisons_json"]
    ).exists()
    assert (result.run_dir / result.summary["saved"]["cpu_kernel"]).exists()


def test_benchmark_spatial_als_writes_portable_parity_artifacts(
    tmp_path: Path,
) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path)

    result = workflows.benchmark_constant_kernel_backends(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path / "runs",
        name="spatial-backend-benchmark",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(5, 5),
        components=[GaussianBasisComponent(sigma=0.9, degree=1)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=True,
        backends=["cpu"],
        reference_backend="cpu",
        repeats=2,
        warmup=1,
        solver="spatial-als",
        spatial_degree=1,
        als_iterations=12,
        als_tolerance=1e-9,
        als_regularization=1e-8,
    )

    summary = result.summary
    assert summary["solver"] == "spatial-als"
    assert summary["runtimes"]["cpu"]["device"] == "cpu"
    assert summary["timings"]["cpu"]["count"] == 2
    assert summary["timings"]["cpu"]["median"] >= 0.0
    assert summary["first_solve_timings"]["cpu"]["solve_seconds"] >= 0.0
    assert summary["warm_timings"]["cpu"]["count"] == 2
    assert summary["median_speedup_vs_reference"]["cpu"] == pytest.approx(1.0)
    assert summary["warm_median_speedup_vs_reference"][
        "cpu"
    ] == pytest.approx(1.0)
    assert summary["parity"]["ok"] is True
    comparison = summary["parity"]["comparisons"]["cpu"]
    assert comparison["arrays"]["horizontal_coefficients"]["ok"] is True
    assert comparison["diagnostics"]["objective_history"]["ok"] is True
    assert comparison["arrays"]["realized_kernels"]["ok"] is True
    facts = summary["solver_facts"]["cpu"]
    assert facts["resolved_backend"] == "cpu"
    assert facts["iterations"] >= 1
    assert facts["unique_fit_pixel_count"] > 0
    assert facts["design_chunk_size"] > 0
    assert summary["spatial_als"]["coordinate_order"] == "y,x"
    assert summary["spatial_als"]["term_degree_order"] == "x,y"
    assert len(summary["kernel_sample_positions_yx"]) == 5
    for name in (
        "cpu_horizontal_reference",
        "cpu_horizontal_basis",
        "cpu_horizontal_coefficients",
        "cpu_vertical_reference",
        "cpu_vertical_basis",
        "cpu_vertical_coefficients",
        "cpu_background_coefficients",
        "cpu_flux_scale",
        "cpu_objective_history",
        "cpu_kernel_sample_positions_yx",
        "cpu_realized_kernels",
    ):
        assert name in summary["saved"]
        assert (result.run_dir / summary["saved"][name]).exists()


def test_benchmark_spatial_als_rejects_unsupported_backend_before_run(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="spatial-als.*unsupported: numba-cuda",
    ):
        workflows.benchmark_constant_kernel_backends(
            reference_path=tmp_path / "missing-reference.npy",
            target_path=tmp_path / "missing-target.npy",
            output_root=tmp_path / "runs",
            name="spatial-unsupported-backend",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(5, 5),
            components=[GaussianBasisComponent(sigma=0.9, degree=1)],
            variance_path=None,
            backends=["cpu", "numba-cuda"],
            solver="spatial-als",
        )

    assert not (tmp_path / "runs" / "spatial-unsupported-backend").exists()


def test_benchmark_reports_unusable_gpu_backend_before_solving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, target = _write_spatial_als_inputs(tmp_path)
    solves: list[str] = []
    original_solve = workflows._solve_benchmark_model

    def tracked_solve(**kwargs):
        solves.append(kwargs["backend"])
        return original_solve(**kwargs)

    def failing_sync(backend: str) -> None:
        if backend == "cupy":
            raise RuntimeError("cudaErrorNoDevice: no CUDA-capable device")

    monkeypatch.setattr(workflows, "_solve_benchmark_model", tracked_solve)
    monkeypatch.setattr(workflows, "_sync_backend", failing_sync)

    with pytest.raises(
        RuntimeError,
        match="backend='cupy' requires a usable CUDA device",
    ):
        workflows.benchmark_constant_kernel_backends(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path / "runs",
            name="spatial-no-device",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(5, 5),
            components=[GaussianBasisComponent(sigma=0.9, degree=1)],
            variance_path=None,
            backends=["cupy", "cpu"],
            reference_backend="cpu",
            repeats=1,
            warmup=0,
            solver="spatial-als",
        )

    assert solves == []
    assert not (tmp_path / "runs" / "spatial-no-device").exists()


def test_spatial_parity_keeps_optimizer_diagnostics_non_gating(
    tmp_path: Path,
) -> None:
    reference_path, target_path = _write_spatial_als_inputs(tmp_path)
    source = np.load(reference_path, allow_pickle=False)
    target = np.load(target_path, allow_pickle=False)
    reference = workflows.solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=0.9, degree=1)],
        kernel_shape=(5, 5),
        backend="cpu",
    )
    candidate = replace(
        reference,
        objective_history=np.append(reference.objective_history, 1.0),
        condition_number=reference.condition_number + 100.0,
        iterations=reference.iterations + 1,
        converged=not reference.converged,
    )

    comparison = workflows._compare_spatial_als_results(
        reference,
        candidate,
        atol=1.0e-9,
        rtol=1.0e-8,
    )

    assert comparison["ok"] is True
    assert comparison["diagnostics"]["objective_history"]["ok"] is False
    assert comparison["diagnostics"]["condition_number"]["ok"] is False
    assert comparison["diagnostics"]["iterations_equal"] is False
    assert comparison["diagnostics"]["converged_equal"] is False


def test_benchmark_spatial_als_cupy_matches_cpu(tmp_path: Path) -> None:
    cp = pytest.importorskip("cupy")
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    if device_count < 1:
        pytest.skip("CuPy CUDA runtime has no visible device")

    reference, target = _write_spatial_als_inputs(tmp_path)
    result = workflows.benchmark_constant_kernel_backends(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path / "runs",
        name="spatial-cupy-benchmark",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(5, 5),
        components=[GaussianBasisComponent(sigma=0.9, degree=1)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=True,
        backends=["cpu", "cupy"],
        reference_backend="cpu",
        repeats=1,
        warmup=1,
        solver="spatial-als",
        spatial_degree=1,
        als_iterations=12,
        als_tolerance=1e-9,
        als_regularization=1e-8,
    )

    summary = result.summary
    assert summary["parity"]["ok"] is True
    assert summary["runtimes"]["cupy"]["backend"] == "cupy"
    first_solve = summary["first_solve_timings"]["cupy"]
    assert "cuda_event_seconds" in first_solve
    assert "gpu_total_bytes_after" in first_solve
    assert "cupy_pool_reserved_bytes_after" in first_solve
    assert "cuda_event" in summary["timings"]["cupy"]
    assert summary["median_speedup_vs_reference"]["cupy"] > 0.0
    assert summary["solver_facts"]["cupy"]["resolved_backend"] == "cupy"
    assert summary["solver_facts"]["cupy"]["design_chunk_size"] > 0


@pytest.mark.parametrize(
    ("backend", "selection_backend"),
    [
        ("cpu", "cpu"),
        ("auto", "cpu"),
        pytest.param("cupy", "cupy", marks=pytest.mark.gpu),
        pytest.param("auto", "cupy", marks=pytest.mark.gpu),
    ],
)
def test_run_constant_kernel_fit_auto_stamp_mask_records_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    selection_backend: str,
) -> None:
    if selection_backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
        if device_count < 1:
            pytest.skip("CuPy CUDA runtime has no visible device")
        monkeypatch.setattr(
            "cuphoton.xpois.ois._backend_available",
            lambda candidate: candidate in {"cpu", "cupy"},
        )
    elif backend == "auto":
        # Exercise CPU fallback independently of installed GPU extras.
        monkeypatch.setattr(
            "cuphoton.xpois.ois._backend_available", lambda backend: False
        )

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    arr = _compact_source_image((64, 64))
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)
    selector = workflows.build_compact_source_stamp_mask
    expected_mask = selector(
        cp.asarray(arr) if selection_backend == "cupy" else arr,
        variance=(
            cp.asarray(np.ones_like(arr))
            if selection_backend == "cupy"
            else np.ones_like(arr)
        ),
        stamp_size=15,
        max_stamps=2,
        peak_percentile=98.0,
    )
    expected_mask = replace(
        expected_mask, mask=workflows.to_host(expected_mask.mask)
    )
    selector_calls = []
    selected_inputs = {}
    device_solve_calls = []
    device_solver = workflows.solve_constant_kernel_device

    def tracked_selector(image, **kwargs):
        selector_calls.append(type(image).__module__.split(".")[0])
        selected = selector(image, **kwargs)
        selected_inputs.update(
            variance=kwargs["variance"], mask=selected.mask
        )
        return selected

    def tracked_device_solver(reference, target, *args, **kwargs):
        assert isinstance(reference, cp.ndarray)
        assert isinstance(target, cp.ndarray)
        assert kwargs["variance"] is selected_inputs["variance"]
        assert kwargs["fit_mask"] is selected_inputs["mask"]
        device_solve_calls.append(True)
        return device_solver(reference, target, *args, **kwargs)

    monkeypatch.setattr(
        workflows, "build_compact_source_stamp_mask", tracked_selector
    )
    monkeypatch.setattr(
        workflows, "solve_constant_kernel_device", tracked_device_solver
    )

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="auto-stamps-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=variance,
        variance_hdu=None,
        fit_mask_path=None,
        auto_stamp_mask=True,
        auto_stamp_size=15,
        auto_stamp_count=2,
        auto_peak_percentile=98.0,
        background_degree=0,
        flux_conserve=False,
        backend=backend,
        review=False,
    )

    assert selector_calls == [
        "cupy" if selection_backend == "cupy" else "numpy"
    ]
    assert device_solve_calls == (
        [True] if selection_backend == "cupy" else []
    )
    expected = workflows.solve_constant_kernel(
        arr,
        arr,
        [GaussianBasisComponent(sigma=1.5, degree=0)],
        kernel_shape=(9, 9),
        variance=np.ones_like(arr),
        fit_mask=expected_mask.mask,
        backend=selection_backend,
    )
    for name in ("fit_mask", "kernel", "matched", "residual", "background"):
        actual = np.load(result.run_dir / result.summary["saved"][name])
        if selection_backend == "cupy" and name != "fit_mask":
            np.testing.assert_allclose(
                actual, getattr(expected, name), rtol=1e-9, atol=1e-8
            )
        else:
            np.testing.assert_array_equal(actual, getattr(expected, name))
    assert result.summary["fit_pixel_count"] == expected.fit_pixel_count
    assert result.summary["dof"] == expected.dof
    assert result.summary["chi2"] == pytest.approx(
        expected.chi2, rel=1e-9, abs=1e-8
    )
    fit_region = result.summary["fit_region"]
    assert fit_region["kind"] == "auto_stamp_mask"
    assert fit_region["selected_count"] == 2
    assert fit_region["stamp_size"] == 15
    assert (
        fit_region["centers_yx"] == expected_mask.to_metadata()["centers_yx"]
    )
    assert "fit_mask_metadata" in result.summary["saved"]
    metadata_path = (
        result.run_dir / result.summary["saved"]["fit_mask_metadata"]
    )
    assert (
        json.loads(metadata_path.read_text()) == expected_mask.to_metadata()
    )


@pytest.mark.parametrize(
    ("backend", "explicit_mask", "use_gpu"),
    [
        pytest.param("cupy", True, True, marks=pytest.mark.gpu),
        pytest.param("auto", False, True, marks=pytest.mark.gpu),
        pytest.param("native", True, True, marks=pytest.mark.gpu),
        pytest.param("native", False, True, marks=pytest.mark.gpu),
        pytest.param(
            "native",
            None,
            True,
            marks=pytest.mark.gpu,
            id="native-auto-stamps",
        ),
        pytest.param("cupy", True, False, id="cpu-dispatch"),
    ],
)
def test_run_constant_kernel_fit_device_solve_preserves_host_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    explicit_mask: bool | None,
    use_gpu: bool,
) -> None:
    if use_gpu:
        cp = pytest.importorskip("cupy")
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
        if device_count < 1:
            pytest.skip("CuPy CUDA runtime has no visible device")
        if backend == "native":
            if importlib.util.find_spec("cuphoton.core._imaging_ext") is None:
                pytest.skip("native imaging extension is not built")
            _native_imaging.extension()
    else:
        # Exercise preparation, dispatch and artifacts in CPU CI as well.
        uploads = []

        def record_upload(value, dtype):
            uploaded = np.array(value, dtype=dtype, copy=True)
            uploads.append(uploaded)
            return uploaded

        cp = SimpleNamespace(
            asarray=record_upload, float64=np.float64, bool_=np.bool_
        )
        monkeypatch.setattr(workflows, "_load_cupy", lambda: (cp, None))
    monkeypatch.setattr(
        "cuphoton.xpois.ois._backend_available",
        lambda candidate: candidate in {"cpu", "cupy"},
    )

    reference = _compact_source_image((64, 64))
    y, x = np.indices(reference.shape, dtype=np.float64)
    target = 1.05 * reference + 0.01 * x + 0.02 * y
    reference_path, target_path = (
        tmp_path / "reference.npy",
        tmp_path / "target.npy",
    )
    np.save(reference_path, reference)
    np.save(target_path, target)
    variance = np.ones_like(reference)
    variance_path = tmp_path / "variance.npy"
    np.save(variance_path, variance)
    mask, mask_path = None, None
    if explicit_mask:
        mask = np.zeros(reference.shape, dtype=bool)
        mask[6:-6, 6:-6] = True
        mask_path = tmp_path / "mask.npy"
        np.save(mask_path, mask.astype(np.uint8))
    auto_stamps = explicit_mask is None
    if auto_stamps:
        mask = cp.asnumpy(
            workflows.build_compact_source_stamp_mask(
                cp.asarray(0.5 * (reference + target)),
                variance=cp.asarray(variance),
                stamp_size=11,
            ).mask
        )
    components = [GaussianBasisComponent(sigma=1.5, degree=0)]
    options = dict(
        kernel_shape=(9, 9),
        background_degree=1,
        flux_conserve=True,
    )
    expected = workflows.solve_constant_kernel(
        reference,
        target,
        components,
        variance=variance,
        fit_mask=mask,
        backend="cupy" if use_gpu else "cpu",
        **options,
    )
    device_solver = workflows.solve_constant_kernel_device
    device_calls = []
    native_selector_calls = []
    native_selection_calls = []
    if backend == "native":
        selection_image = _native_imaging.selection_image
        selector = workflows.build_compact_source_stamp_mask

        def checked_selection_image(*args, **kwargs):
            native_selection_calls.append(True)
            return selection_image(*args, **kwargs)

        def checked_selector(*args, **kwargs):
            assert kwargs["backend"] == "native"
            native_selector_calls.append(True)
            return selector(*args, **kwargs)

        monkeypatch.setattr(
            _native_imaging, "selection_image", checked_selection_image
        )
        monkeypatch.setattr(
            workflows, "build_compact_source_stamp_mask", checked_selector
        )

    def checked_device_solver(reference, target, *args, **kwargs):
        inputs = [reference, target, kwargs["variance"]]
        assert all(value.dtype == np.float64 for value in inputs)
        if kwargs["fit_mask"] is not None:
            assert kwargs["fit_mask"].dtype == np.bool_
            inputs.append(kwargs["fit_mask"])
        device_calls.append(True)
        if use_gpu:
            assert all(isinstance(value, cp.ndarray) for value in inputs)
            assert kwargs.get("backend", "cupy") == (
                "native" if backend == "native" else "cupy"
            )
            return device_solver(reference, target, *args, **kwargs)
        assert all(
            any(value is uploaded for uploaded in uploads) for value in inputs
        )
        assert kwargs.pop("backend", "cupy") == "cupy"
        return replace(
            workflows.solve_constant_kernel(
                reference, target, *args, backend="cpu", **kwargs
            ),
            backend="cupy",
        )

    monkeypatch.setattr(
        workflows, "solve_constant_kernel_device", checked_device_solver
    )
    fitted = workflows.run_constant_kernel_fit(
        reference_path=reference_path,
        target_path=target_path,
        reference_hdu=None,
        target_hdu=None,
        output_root=tmp_path,
        name="device-fit-run",
        components=components,
        variance_path=variance_path,
        fit_mask_path=mask_path,
        backend=backend,
        auto_stamp_mask=auto_stamps,
        auto_stamp_size=11,
        review=False,
        **options,
    )

    assert device_calls == [True]
    assert fitted.summary["backend"] == (
        "native" if backend == "native" else "cupy"
    )
    if backend == "native":
        assert native_selector_calls == ([True] if auto_stamps else [])
        assert native_selection_calls == ([True] if auto_stamps else [])
    assert fitted.summary["fit_pixel_count"] == expected.fit_pixel_count
    assert fitted.summary["dof"] == expected.dof
    assert fitted.summary["chi2"] == pytest.approx(
        expected.chi2, rel=1e-9, abs=1e-8
    )
    assert fitted.summary["fit_region"]["kind"] == (
        "auto_stamp_mask"
        if auto_stamps
        else "explicit_mask"
        if explicit_mask
        else "interior_box"
    )
    for name in ("kernel", "matched", "residual", "background", "fit_mask"):
        actual = np.load(fitted.run_dir / fitted.summary["saved"][name])
        assert actual.dtype == getattr(expected, name).dtype
        if name == "fit_mask":
            np.testing.assert_array_equal(actual, expected.fit_mask)
        else:
            np.testing.assert_array_equal(
                np.isnan(actual), np.isnan(getattr(expected, name))
            )
            np.testing.assert_allclose(
                actual, getattr(expected, name), rtol=1e-9, atol=1e-8
            )
    evaluated = workflows.evaluate_subtraction_run(fitted.run_dir)
    assert evaluated["fit_region_pixel_count"] == expected.fit_pixel_count


def test_run_constant_kernel_fit_writes_interactive_and_numeric_review(
    tmp_path: Path,
) -> None:
    has_bokeh = importlib.util.find_spec("bokeh") is not None

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    arr = _compact_source_image((64, 64))
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="review-artifacts-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=variance,
        variance_hdu=None,
        fit_mask_path=None,
        auto_stamp_mask=True,
        auto_stamp_size=15,
        auto_stamp_count=2,
        auto_peak_percentile=98.0,
        background_degree=0,
        flux_conserve=False,
    )

    saved = result.summary["saved"]
    assert "review_hotspots_metadata" in saved
    metadata_path = result.run_dir / saved["review_hotspots_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_name"] == "review-artifacts-run"
    assert "review_metrics" in metadata
    assert isinstance(metadata["hotspots"], list)
    for key in (
        "review_overview",
        "review_stamps",
        "review_kernel",
        "review_sigma",
        "review_hotspots",
        "review_html",
        "review_manifest",
    ):
        assert key not in saved
    assert not list((result.run_dir / "artifacts").glob("review_*.png"))
    if has_bokeh:
        assert "review_bokeh_html" in saved
        assert (result.run_dir / saved["review_bokeh_html"]).exists()
    else:
        assert "review_bokeh_html" not in saved


@pytest.mark.parametrize("bokeh_available", [False, True])
@pytest.mark.parametrize("solver", ["constant", "spatial-als"])
def test_no_review_preserves_fit_and_skips_review_work(
    tmp_path: Path, monkeypatch, bokeh_available: bool, solver: str
) -> None:
    if bokeh_available:
        pytest.importorskip("bokeh")
    else:
        original_import = builtins.__import__

        def without_bokeh(name, *args, **kwargs):
            if name == "bokeh" or name.startswith("bokeh."):
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_bokeh)

    image = _compact_source_image((64, 64))
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    np.save(reference, image)
    np.save(target, image + np.random.default_rng(7).normal(size=image.shape))
    mask_path = tmp_path / "mask.npy"
    mask = np.zeros(image.shape, dtype=np.uint16)
    mask[10, 10] = 1
    np.save(mask_path, mask)
    options = dict(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        solver=solver,
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        reference_mask_path=mask_path,
        target_mask_path=mask_path,
        mask_policy="strict",
        background_degree=0,
        flux_conserve=False,
        backend="cpu",
    )
    reviewed = workflows.run_constant_kernel_fit(name="reviewed", **options)
    assert reviewed.summary["review_enabled"] is True
    assert "review_hotspots_metadata" in reviewed.summary["saved"]
    assert (
        "review_bokeh_html" in reviewed.summary["saved"]
    ) == bokeh_available

    def unexpected_review(*args, **kwargs):
        pytest.fail("Review-only work ran with review disabled")

    monkeypatch.setattr(workflows, "write_review_metadata", unexpected_review)
    monkeypatch.setattr(
        workflows, "write_interactive_review_artifact", unexpected_review
    )
    monkeypatch.setattr(np, "nanpercentile", unexpected_review)
    deferred = workflows.run_constant_kernel_fit(
        name="deferred", review=False, **options
    )
    assert deferred.summary["review_enabled"] is False
    assert (
        deferred.summary["timings_sec"]["review_generation_and_write_sec"]
        == 0.0
    )
    assert not list((deferred.run_dir / "artifacts").glob("review_*"))
    for key, relative_path in deferred.summary["saved"].items():
        assert not key.startswith("review_")
        assert (deferred.run_dir / relative_path).read_bytes() == (
            reviewed.run_dir / reviewed.summary["saved"][key]
        ).read_bytes()
    for key in (
        "fit_pixel_count",
        "chi2",
        "dof",
        "kernel_sum" if solver == "constant" else "kernel_sum_center",
        "residual_mean",
        "residual_std",
        "all_pixels_residual_mean",
        "all_pixels_residual_std",
        "fit_region",
    ):
        assert deferred.summary[key] == reviewed.summary[key]
    if solver == "spatial-als":
        assert (
            deferred.summary["spatial_als"] == reviewed.summary["spatial_als"]
        )


def test_evaluate_subtraction_run_reports_fit_region_metrics(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="eval-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )
    evaluation = workflows.evaluate_subtraction_run(result.run_dir)

    assert evaluation["fit_region_pixel_count"] == (64 - 8) * (64 - 8)
    assert "fit_region_residual_std" in evaluation


def test_evaluate_subtraction_run_uses_saved_explicit_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    mask = np.zeros_like(arr, dtype=bool)
    mask[10:30, 10:30] = True
    np.save(mask_path, mask, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="masked-eval-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=mask_path,
        background_degree=0,
        flux_conserve=False,
    )
    evaluation = workflows.evaluate_subtraction_run(result.run_dir)

    clipped_mask = mask.copy()
    clipped_mask[:4, :] = False
    clipped_mask[-4:, :] = False
    clipped_mask[:, :4] = False
    clipped_mask[:, -4:] = False
    assert evaluation["fit_region_pixel_count"] == int(clipped_mask.sum())


def test_evaluate_subtraction_run_works_after_run_dir_move(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    mask = np.zeros_like(arr, dtype=bool)
    mask[12:28, 12:28] = True
    np.save(mask_path, mask, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="movable-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=mask_path,
        background_degree=0,
        flux_conserve=False,
    )

    moved = tmp_path / "moved-run"
    shutil.copytree(result.run_dir, moved)
    evaluation = workflows.evaluate_subtraction_run(moved)

    clipped_mask = mask.copy()
    clipped_mask[:4, :] = False
    clipped_mask[-4:, :] = False
    clipped_mask[:, :4] = False
    clipped_mask[:, -4:] = False
    assert evaluation["fit_region_pixel_count"] == int(clipped_mask.sum())
    summary = (moved / "summary.json").read_text(encoding="utf-8")
    assert '"fit_mask": "artifacts/fit_mask.npy"' in summary


def test_evaluate_subtraction_run_rejects_missing_explicit_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    mask = np.zeros_like(arr, dtype=bool)
    mask[10:30, 10:30] = True
    np.save(mask_path, mask, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="missing-mask-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        variance_hdu=None,
        fit_mask_path=mask_path,
        background_degree=0,
        flux_conserve=False,
    )
    (result.run_dir / "artifacts" / "fit_mask.npy").unlink()

    with pytest.raises(FileNotFoundError):
        workflows.evaluate_subtraction_run(result.run_dir)


def test_run_constant_kernel_fit_rejects_malformed_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    mask = np.zeros_like(arr, dtype=np.float64)
    mask[10:30, 10:30] = 0.5
    np.save(mask_path, mask, allow_pickle=False)

    with pytest.raises(ValueError):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path,
            name="bad-mask-run",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(9, 9),
            components=[GaussianBasisComponent(sigma=1.5, degree=0)],
            variance_path=None,
            fit_mask_path=mask_path,
            background_degree=0,
            flux_conserve=False,
        )


def test_evaluate_subtraction_run_rejects_missing_interior_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="missing-interior-mask-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )
    (result.run_dir / "artifacts" / "fit_mask.npy").unlink()

    with pytest.raises(FileNotFoundError):
        workflows.evaluate_subtraction_run(result.run_dir)


def test_evaluate_subtraction_run_rejects_malformed_saved_fit_mask(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bad-fit-mask-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    fit_mask = np.zeros((16, 16), dtype=np.float64)
    fit_mask[4:12, 4:12] = 2.0
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        workflows.evaluate_subtraction_run(run_dir)


def test_evaluate_subtraction_run_rejects_mismatched_saved_fit_mask_shape(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bad-fit-mask-shape-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    fit_mask = np.ones((15, 15), dtype=bool)
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        workflows.evaluate_subtraction_run(run_dir)


def test_run_constant_kernel_fit_uses_timestamp_name_by_default(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name=None,
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    assert result.run_dir.name.startswith("fit-kernel-")
    assert result.run_dir.name != "fit-kernel"


def test_run_constant_kernel_fit_rejects_unsafe_run_name(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    with pytest.raises(ValueError):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path,
            name="../escape",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(9, 9),
            components=[GaussianBasisComponent(sigma=1.5, degree=0)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=False,
        )


def test_non_finite_inputs_are_excluded_from_saved_statistics(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    reference_arr = arr.copy()
    target_arr = arr.copy()
    reference_arr[1, 1] = np.nan
    reference_arr[24, 24] = np.nan
    target_arr[30, 30] = np.nan
    np.save(reference, reference_arr, allow_pickle=False)
    np.save(target, target_arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="non-finite-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    residual = np.load(result.run_dir / "artifacts" / "residual.npy")
    assert np.isnan(residual[0, 0])
    assert np.isnan(residual[24, 24])
    assert np.isnan(residual[30, 30])
    assert np.isfinite(result.summary["residual_mean"])
    assert np.isfinite(result.summary["residual_std"])

    evaluation = workflows.evaluate_subtraction_run(result.run_dir)

    assert np.isfinite(evaluation["residual_mean"])
    assert np.isfinite(evaluation["residual_std"])
    assert np.isfinite(evaluation["abs_residual_max"])


def test_saved_residual_excludes_convolution_margin(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="margin-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    residual = np.load(result.run_dir / "artifacts" / "residual.npy")

    assert np.isnan(residual[0, 0])
    assert np.isnan(residual[3, 3])
    assert np.isfinite(residual[4, 4])


def test_run_constant_kernel_fit_records_effective_variance_hdu(
    tmp_path: Path,
) -> None:
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    variance = tmp_path / "variance.fits"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                np.ones((64, 64), dtype=np.float64),
                name="VARIANCE",
            ),
        ]
    ).writeto(variance)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="variance-hdu-run",
        reference_hdu=0,
        target_hdu=0,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=variance,
        variance_hdu=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    assert result.summary["variance_hdu"] == 1


def test_evaluate_subtraction_run_counts_spike_when_mad_is_zero(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "spike-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    residual[8, 8] = 25.0
    fit_mask = np.ones_like(residual, dtype=bool)
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    evaluation = workflows.evaluate_subtraction_run(run_dir)

    assert evaluation["pixels_gt_3sigma"] == 1
    assert evaluation["pixels_gt_5sigma"] == 1


def test_evaluate_subtraction_run_reports_zero_outliers_for_constant_field(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "constant-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    fit_mask = np.ones_like(residual, dtype=bool)
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    evaluation = workflows.evaluate_subtraction_run(run_dir)

    assert evaluation["pixels_gt_3sigma"] == 0
    assert evaluation["pixels_gt_5sigma"] == 0


@pytest.mark.parametrize("workflow", ["fit", "benchmark"])
@pytest.mark.parametrize(
    "mismatched_input, message",
    [
        ("reference", "reference and target must share the same shape"),
        ("variance", "variance must match the image shape"),
        ("reference_mask", "reference mask.*shape"),
        ("target_mask", "target mask.*shape"),
    ],
)
def test_cropped_workflows_reject_full_frame_shape_mismatches(
    tmp_path, workflow, mismatched_input, message
):
    reference = _compact_source_image((64, 64))
    arrays = {
        "reference": reference,
        "target": 1.1 * reference + 0.2,
        "variance": np.ones_like(reference),
        "reference_mask": np.zeros(reference.shape, dtype=np.int16),
        "target_mask": np.zeros(reference.shape, dtype=np.int16),
    }
    arrays[mismatched_input] = np.pad(arrays[mismatched_input], 8)
    paths = {}
    for label, array in arrays.items():
        paths[f"{label}_path"] = tmp_path / f"{label}.npy"
        np.save(paths[f"{label}_path"], array, allow_pickle=False)
    options = dict(
        **paths,
        output_root=tmp_path,
        name=workflow,
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(3, 3),
        components=[GaussianBasisComponent(sigma=1.0, degree=0)],
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
        mask_policy="strict",
        crop_y0=8,
        crop_x0=8,
        crop_height=48,
        crop_width=48,
    )
    with pytest.raises(ValueError, match=message):
        if workflow == "fit":
            workflows.run_constant_kernel_fit(
                **options, backend="cpu", review=False
            )
        else:
            workflows.benchmark_constant_kernel_backends(
                **options, backends=["cpu"], repeats=1, warmup=0
            )


@pytest.mark.parametrize("selection", ["full", "auto", "explicit", "cropped"])
@pytest.mark.parametrize("mask_policy", ["strict", "masklite"])
def test_benchmark_preprocessing_matches_fitting(
    tmp_path, selection, mask_policy, monkeypatch
):
    from astropy.io import fits

    arr = _compact_source_image((64, 64))
    reference_mask = np.zeros(arr.shape, dtype=np.int16)
    target_mask = reference_mask.copy()
    reference_mask[22, 23] = 1
    target_mask[35, 30] = 1
    target_mask[27, 28] = 1 << 9
    target = 1.1 * arr + 0.2
    target[35, 30] = 1e5
    for name, image, mask in (
        ("reference", arr, reference_mask),
        ("target", target, target_mask),
    ):
        mask_hdu = fits.ImageHDU(mask)
        for bit, plane in enumerate(workflows._HSC_MASKLITE_PLANES):
            mask_hdu.header[f"MP_{plane}"] = bit
        mask_hdu.header["MP_DETECTED"] = 9
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(image),
                mask_hdu,
                fits.ImageHDU(np.ones_like(arr)),
            ]
        ).writeto(tmp_path / f"{name}.fits")
    options = dict(
        reference_path=tmp_path / "reference.fits",
        target_path=tmp_path / "target.fits",
        reference_hdu=1,
        target_hdu=1,
        reference_mask_path=tmp_path / "reference.fits",
        target_mask_path=tmp_path / "target.fits",
        reference_mask_hdu=2,
        target_mask_hdu=2,
        variance_path=tmp_path / "target.fits",
        variance_hdu=3,
        mask_policy=mask_policy,
        crop_y0=8,
        crop_x0=8,
        crop_height=48,
        crop_width=48,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
        output_root=tmp_path,
    )
    if selection == "auto":
        options.update(
            auto_stamp_mask=True,
            auto_stamp_size=15,
            auto_stamp_count=2,
            auto_peak_percentile=98.0,
        )
    elif selection in {"explicit", "cropped"}:
        mask = np.zeros(arr.shape, dtype=bool)
        mask[16:40, 16:40] = True
        if selection == "cropped":
            mask = mask[8:56, 8:56]
        path = tmp_path / "selection.npy"
        np.save(path, mask)
        options["fit_mask_path"] = path
    monkeypatch.setattr(
        workflows, "write_interactive_review_artifact", lambda *a, **kw: {}
    )
    fitted = workflows.run_constant_kernel_fit(
        **options, name="fit", backend="cpu"
    )
    benchmark = workflows.benchmark_constant_kernel_backends(
        **options, name="benchmark", backends=["cpu"], repeats=1, warmup=0
    )
    for name in ("kernel", "matched", "residual", "fit_mask"):
        expected = np.load(fitted.run_dir / fitted.summary["saved"][name])
        actual = np.load(
            benchmark.run_dir / benchmark.summary["saved"][f"cpu_{name}"]
        )
        np.testing.assert_array_equal(actual, expected)
    mask = np.load(
        benchmark.run_dir / benchmark.summary["saved"]["cpu_fit_mask"]
    )
    assert not mask[35 - 8, 30 - 8]
    if selection in {"explicit", "cropped"}:
        assert mask[22, 22]
        outside = mask.copy()
        outside[8:32, 8:32] = False
        assert not outside.any()
    assert mask.sum() == benchmark.summary["fit_pixel_count"]["cpu"]
    assert benchmark.summary["fit_region"] == fitted.summary["fit_region"]
    assert benchmark.summary["input_mask"] == fitted.summary["input_mask"]
    assert benchmark.summary["setup_timings"]["load_seconds"] >= 0
    assert benchmark.summary["setup_timings"]["preprocess_seconds"] > 0
    if selection == "auto":
        assert mask.sum() <= 2 * 15**2
        assert "fit_mask_metadata" in benchmark.summary["saved"]


@pytest.mark.parametrize(
    "options, message",
    [
        (
            {"fit_mask_path": Path("mask.npy"), "auto_stamp_mask": True},
            "either fit_mask_path",
        ),
        ({"reference_mask_path": Path("mask.npy")}, "non-'none'"),
        ({"variance_hdu": 2}, "variance_hdu requires"),
        ({"reference_mask_hdu": 2}, "FITS-backed reference"),
        ({"mask_policy": "strict"}, "reference_mask_path is required"),
        ({"mask_policy": "masklite"}, "reference_mask_path is required"),
        (
            {
                "mask_policy": "strict",
                "reference_mask_path": Path("reference-mask.npy"),
            },
            "target_mask_path is required",
        ),
        (
            {
                "mask_policy": "masklite",
                "reference_mask_path": Path("reference-mask.fits"),
            },
            "target_mask_path is required",
        ),
        ({"crop_y0": 8}, "all crop parameters"),
    ],
)
def test_benchmark_rejects_invalid_preprocessing_before_loading(
    tmp_path, monkeypatch, options, message
):
    def unexpected_call(*args, **kwargs):
        pytest.fail("invalid options must fail before loading or solving")

    monkeypatch.setattr(workflows, "load_image_with_wcs", unexpected_call)
    monkeypatch.setattr(workflows, "solve_constant_kernel", unexpected_call)
    with pytest.raises(ValueError, match=message):
        workflows.benchmark_constant_kernel_backends(
            reference_path=tmp_path / "reference.npy",
            target_path=tmp_path / "target.npy",
            reference_hdu=None,
            target_hdu=None,
            variance_path=None,
            output_root=tmp_path,
            name="invalid",
            kernel_shape=(9, 9),
            components=[GaussianBasisComponent(sigma=1.5, degree=0)],
            backends=["cpu"],
            **options,
        )
    assert not (tmp_path / "invalid").exists()
