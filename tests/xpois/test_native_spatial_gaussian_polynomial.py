# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native spatial fits preserve held-out signals and noise diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import numpy as np
import pytest

import cuphoton.xpois.spatial_gaussian_polynomial as spatial_model
from cuphoton.core import _native_imaging
from cuphoton.xpois import (
    GaussianBasisComponent,
    build_gaussian_polynomial_basis,
)
from cuphoton.xpois.spatial_gaussian_polynomial import (
    SpatialGaussianPolynomialKernelConfig,
    SpatialGaussianPolynomialKernelFitSamples,
    SpatialKernelDomain,
    solve_spatial_gaussian_polynomial_kernel,
)
from tests.xpois.test_cli import _run_cli, _write_cli_spatial_inputs
from tests.xpois.test_spatial_gaussian_polynomial import (
    _chebyshev_field,
    _spatial_target,
)


@pytest.fixture
def native_cupy() -> Any:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("native spatial xPois requires an actual CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if importlib.util.find_spec("cuphoton.core._imaging_ext") is None:
        pytest.skip("native imaging extension is not built")
    _native_imaging.extension()
    return cp


def _assert_same_fit(actual, expected) -> None:
    for name in (
        "photometric_coefficients",
        "shape_coefficients",
        "background_coefficients",
        "basis_kernels",
        "photometric_scale",
        "background",
        "matched",
        "residual",
        "propagated_source_variance",
        "marginal_residual_variance",
        "marginal_standardized_residual",
    ):
        left, right = getattr(actual, name), getattr(expected, name)
        if right is None:
            assert left is None
        else:
            np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
            np.testing.assert_allclose(
                left, right, rtol=2e-8, atol=2e-9, equal_nan=True
            )
    np.testing.assert_array_equal(actual.fit_mask, expected.fit_mask)
    np.testing.assert_allclose(
        [actual.fit_objective, actual.condition_number],
        [expected.fit_objective, expected.condition_number],
        rtol=2e-8,
        atol=2e-9,
    )
    for name in (
        "basis_terms",
        "photometric_terms",
        "shape_terms",
        "background_terms",
        "image_shape",
        "spatial_domain",
        "explicit_fit_samples",
        "fit_selection_kind",
        "fit_weighting",
        "dof",
        "fit_pixel_count",
        "unique_fit_pixel_count",
        "requested_fit_pixel_count",
        "excluded_requested_fit_pixel_count",
        "source_variance_included",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def test_native_spatial_recovers_held_out_nonseparable_signal(
    native_cupy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        spatial_model, "_gpu_design_chunk_size", lambda *args, **kwargs: 256
    )
    shape = (56, 62)
    source = np.random.default_rng(113).normal(size=(112, 124))[::2, ::2]
    assert not source.flags.c_contiguous
    components = [GaussianBasisComponent(1.2, 2)]
    basis, terms = build_gaussian_polynomial_basis(
        (7, 9), components, flux_conserve=True
    )
    domain = SpatialKernelDomain((112, 518), (100, 180, 500, 596))
    photometric = np.array([1.03, 0.02, -0.015])
    background = np.array([0.04, -0.01, 0.006])
    coefficients = np.zeros((basis.shape[0] - 1, 6))
    # The mixed xy term produces off-axis shape and a nonseparable kernel.
    mixed = next(
        index
        for index, term in enumerate(terms)
        if (term.poly_u_degree, term.poly_v_degree) == (1, 1)
    )
    coefficients[mixed - 1] = [0.12, 0.035, -0.03, 0.01, 0.02, -0.01]
    coefficients[0] = [0.025, -0.01, 0.006, 0.003, -0.002, 0.001]
    truth = _spatial_target(
        source,
        basis,
        photometric,
        coefficients,
        background,
        photometric_degree=1,
        shape_degree=2,
        background_degree=1,
        domain=domain,
    )
    transient = np.zeros(shape)
    ty, tx = np.mgrid[-3:4, -3:4]
    profile = np.exp(-(tx**2 + ty**2) / 2.0)
    transient[22:29, 25:32] = 7.25 * profile / profile.sum()
    target = truth + transient
    variance = 0.4 + np.linspace(0.0, 0.3, source.size).reshape(shape)
    fit_mask = np.zeros(shape, dtype=bool)
    fit_mask[4:-4, 5:-5] = True
    # Keep the evaluation patch and a full kernel margin out of the fit.
    fit_mask[15:34, 18:39] = False
    held_out = np.zeros(shape, dtype=bool)
    held_out[19:30, 22:35] = True
    source[10, 11] = np.nan
    target[43, 44] = np.nan
    variance[36, 45] = np.inf
    options = dict(
        kernel_shape=(7, 9),
        variance=variance,
        fit_mask=fit_mask,
        spatial_domain=domain,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=2, photometric_degree=1, background_degree=1
        ),
    )
    cpu = solve_spatial_gaussian_polynomial_kernel(
        source, target, components, backend="cpu", **options
    )
    native = solve_spatial_gaussian_polynomial_kernel(
        source, target, components, backend="native", **options
    )
    cupy = solve_spatial_gaussian_polynomial_kernel(
        source, target, components, backend="cupy", **options
    )
    assert native.backend == "native"
    assert native.design_chunk_size == 256
    assert native.fit_pixel_count > 256
    assert native.fit_pixel_count % 256 != 0
    _assert_same_fit(native, cpu)
    _assert_same_fit(native, cupy)
    assert not np.any(native.fit_mask & held_out)
    np.testing.assert_allclose(
        native.matched[held_out], truth[held_out], rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        native.residual[held_out], transient[held_out], rtol=1e-9, atol=1e-9
    )
    assert native.residual[held_out].sum() == pytest.approx(7.25, abs=1e-8)
    np.testing.assert_allclose(
        native.shape_coefficients, coefficients, rtol=1e-8, atol=1e-9
    )
    for y, x in ((8.5, 9.25), (25.0, 28.0), (46.5, 50.25)):
        kernel = native.kernel_at_local(y, x)
        assert np.linalg.matrix_rank(kernel, tol=1e-10) > 1
        np.testing.assert_allclose(
            kernel, cpu.kernel_at_local(y, x), rtol=2e-8, atol=2e-9
        )
        np.testing.assert_allclose(
            kernel, native.kernel_at_parent(y + 112, x + 518), atol=1e-12
        )
        assert kernel.sum() == pytest.approx(
            native.photometric_scale_at_parent(y + 112, x + 518), abs=1e-12
        )
    expected_scale = _chebyshev_field(shape, photometric, 1, domain)
    np.testing.assert_allclose(
        native.photometric_scale, expected_scale, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize(
    "degrees,chunk_size",
    [((3, 2, 0), 17), ((1, 0, 2), 53), ((2, 0, 0), 53)],
)
def test_native_spatial_preserves_weighting_and_marginal_variance(
    native_cupy: Any,
    monkeypatch: pytest.MonkeyPatch,
    degrees: tuple[int, int, int],
    chunk_size: int,
) -> None:
    shape_degree, photometric_degree, background_degree = degrees
    monkeypatch.setattr(
        spatial_model,
        "_gpu_design_chunk_size",
        lambda *args, **kwargs: chunk_size,
    )
    shape = (34, 38)
    rng = np.random.default_rng(127)
    source = rng.normal(size=shape)
    components = [GaussianBasisComponent(1.1, 1)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7), components, flux_conserve=True
    )
    term_counts = [(degree + 1) * (degree + 2) // 2 for degree in degrees]
    photometric = np.linspace(0.015, -0.01, term_counts[1])
    photometric[0] = 1.02
    target = _spatial_target(
        source,
        basis,
        photometric,
        rng.normal(scale=0.01, size=(basis.shape[0] - 1, term_counts[0])),
        np.linspace(0.05, -0.01, term_counts[2]),
        photometric_degree=photometric_degree,
        shape_degree=shape_degree,
        background_degree=background_degree,
    )
    target += rng.normal(scale=0.01, size=shape)
    variance = 0.3 + np.linspace(0.0, 0.2, source.size).reshape(shape)
    source_variance = 0.5 + np.linspace(0.0, 0.3, source.size).reshape(shape)
    # Shared validation accepts column-major noise planes; the native
    # adapter must normalize their layout before its row-major CUDA kernels.
    variance = np.asfortranarray(variance)
    source_variance = np.asfortranarray(source_variance)
    assert not variance.flags.c_contiguous
    assert not source_variance.flags.c_contiguous
    source_variance[17, 19] = np.nan
    mask = np.zeros(shape, dtype=bool)
    mask[4:-4:2, 5:-5:2] = True
    positions = np.column_stack(np.nonzero(mask))
    samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=positions[::-1],
        relative_precision=np.linspace(0.2, 3.0, len(positions)),
    )
    options = dict(
        kernel_shape=(7, 7),
        variance=variance,
        fit_samples=samples,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=shape_degree,
            photometric_degree=photometric_degree,
            background_degree=background_degree,
        ),
    )
    cpu = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        source_variance=source_variance,
        backend="cpu",
        **options,
    )
    native = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        source_variance=source_variance,
        backend="native",
        **options,
    )
    without_source_variance = solve_spatial_gaussian_polynomial_kernel(
        source, target, components, backend="native", **options
    )
    cupy = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        source_variance=source_variance,
        backend="cupy",
        **options,
    )
    _assert_same_fit(native, cpu)
    _assert_same_fit(native, cupy)
    assert native.backend == "native"
    assert native.design_chunk_size == chunk_size
    assert native.fit_pixel_count > chunk_size
    assert native.fit_pixel_count % chunk_size != 0
    if degrees == (3, 2, 0):
        column_count = (
            term_counts[1]
            + (basis.shape[0] - 1) * term_counts[0]
            + term_counts[2]
        )
        assert chunk_size < column_count
    assert native.fit_weighting == "target_variance_relative_precision"
    np.testing.assert_array_equal(native.fit_mask, mask)
    np.testing.assert_allclose(
        native.matched, without_source_variance.matched, equal_nan=True
    )
    assert native.fit_objective == without_source_variance.fit_objective
    assert np.isfinite(native.residual[17, 19])
    assert np.isnan(native.marginal_residual_variance[17, 19])
    for y, x in ((8, 9), (25, 28)):
        kernel = native.kernel_at_local(y, x)
        patch = source_variance[y - 3 : y + 4, x - 3 : x + 4]
        propagated = np.sum(patch * kernel[::-1, ::-1] ** 2)
        marginal = variance[y, x] + propagated
        assert native.propagated_source_variance[y, x] == pytest.approx(
            propagated, rel=2e-10, abs=1e-12
        )
        assert native.marginal_residual_variance[y, x] == pytest.approx(
            marginal, rel=2e-10, abs=1e-12
        )
        assert native.marginal_standardized_residual[y, x] == pytest.approx(
            native.residual[y, x] / np.sqrt(marginal), rel=2e-10, abs=1e-12
        )


@pytest.mark.parametrize("case", ["underdetermined", "singular", "condition"])
def test_native_spatial_rejects_unconstrained_models(
    native_cupy: Any, case: str
) -> None:
    source = np.random.default_rng(137).normal(size=(32, 36))
    components = [GaussianBasisComponent(1.2, 1)]
    options: dict[str, Any] = dict(kernel_shape=(7, 7))
    if case == "underdetermined":
        options["fit_samples"] = SpatialGaussianPolynomialKernelFitSamples(
            positions_yx=np.array([[8, 9], [12, 15], [22, 25]])
        )
        message = "underdetermined"
    elif case == "singular":
        source.fill(1.0)
        components = [GaussianBasisComponent(1.2, 0)]
        options["config"] = SpatialGaussianPolynomialKernelConfig(
            shape_degree=0, photometric_degree=1, background_degree=1
        )
        message = "ill-conditioned"
    else:
        options["config"] = SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
            condition_limit=1.01,
        )
        message = "ill-conditioned"
    for backend in ("cpu", "native"):
        with pytest.raises(ValueError, match=message):
            solve_spatial_gaussian_polynomial_kernel(
                source, source + 0.1, components, backend=backend, **options
            )


def test_native_spatial_workspace_reuses_streams_and_isolates_threads(
    native_cupy: Any,
) -> None:
    cp = native_cupy
    device = cp.cuda.runtime.getDevice()
    barrier = Barrier(2)
    components = [GaussianBasisComponent(1.1, 1)]
    basis, _ = build_gaussian_polynomial_basis(
        (5, 5), components, flux_conserve=True
    )

    def worker(seed):
        rng = np.random.default_rng(211 + seed)
        with cp.cuda.Device(device):
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
            for index, (shape, degrees) in enumerate(
                (
                    ((24 + seed, 27), (1, seed, 2 - seed)),
                    ((38 + seed, 41), (3, 2 - seed, seed)),
                    ((24 + seed, 27), (1, seed, 2 - seed)),
                )
            ):
                shape_degree, photometric_degree, background_degree = degrees
                counts = [
                    (degree + 1) * (degree + 2) // 2 for degree in degrees
                ]
                source = rng.normal(size=shape)
                photometric = np.linspace(0.02, -0.01, counts[1])
                photometric[0] = 1.03
                target = _spatial_target(
                    source,
                    basis,
                    photometric,
                    rng.normal(
                        scale=0.01, size=(basis.shape[0] - 1, counts[0])
                    ),
                    np.linspace(0.04, -0.02, counts[2]),
                    photometric_degree=photometric_degree,
                    shape_degree=shape_degree,
                    background_degree=background_degree,
                )
                options = dict(
                    kernel_shape=(5, 5),
                    variance=np.ones(shape),
                    source_variance=np.full(shape, 0.5),
                    config=SpatialGaussianPolynomialKernelConfig(
                        shape_degree=shape_degree,
                        photometric_degree=photometric_degree,
                        background_degree=background_degree,
                    ),
                )
                expected = solve_spatial_gaussian_polynomial_kernel(
                    source, target, components, backend="cpu", **options
                )
                barrier.wait(timeout=30)
                with streams[index % 2]:
                    actual = solve_spatial_gaussian_polynomial_kernel(
                        source,
                        target,
                        components,
                        backend="native",
                        **options,
                    )
                _assert_same_fit(actual, expected)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, seed) for seed in range(2)]
        for future in futures:
            future.result(timeout=60)


def test_native_spatial_without_compilation_in_fresh_process(
    native_cupy: Any, tmp_path
) -> None:
    source = textwrap.dedent("""
        import numpy as np
        import cupy as cp
        from cupy._core import core
        from cupy.cuda import compiler
        from cuphoton.xpois import GaussianBasisComponent
        from cuphoton.xpois.spatial_gaussian_polynomial import (
            solve_spatial_gaussian_polynomial_kernel,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("CuPy runtime compilation attempted")

        core.compile_with_cache = forbidden
        compiler._compile_module_with_cache = forbidden
        compiler.compile_using_nvrtc = forbidden
        compiler.compile_using_nvcc = forbidden
        compiler.nvrtc.compileProgram = forbidden
        image = np.random.default_rng(193).normal(size=(24, 27))
        result = solve_spatial_gaussian_polynomial_kernel(
            image, image * 1.03 + 0.04,
            [GaussianBasisComponent(1.1, 1)], kernel_shape=(5, 5),
            variance=np.ones_like(image),
            source_variance=np.full_like(image, 0.5), backend="native",
        )
        assert result.backend == "native"
        assert np.isfinite(result.matched[2:-2, 2:-2]).all()
        assert np.isfinite(
            result.propagated_source_variance[2:-2, 2:-2]
        ).all()
    """)
    environment = os.environ.copy()
    environment["CUPY_CACHE_DIR"] = str(tmp_path / "cupy-empty")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not list((tmp_path / "cupy-empty").glob("**/*.cubin"))


def test_cli_native_spatial_matches_cpu(native_cupy: Any, tmp_path, capsys):
    reference, target_path = _write_cli_spatial_inputs(tmp_path)
    target = np.load(target_path)
    scale = 1.0 + 0.2 * np.linspace(-1.0, 1.0, target.shape[1])[None, :]
    np.save(target_path, (target - 0.1) * scale + 0.3, allow_pickle=False)
    results = {}
    for backend in ("cpu", "native"):
        rc = _run_cli(
            [
                "fit-kernel",
                "--reference",
                reference,
                "--target",
                target_path,
                "--solver",
                "spatial-gaussian-polynomial",
                "--backend",
                backend,
                "--basis-sigmas",
                "1.5",
                "--basis-degrees",
                "2",
                "--spatial-degree",
                "1",
                "--photometric-degree",
                "1",
                "--background-degree",
                "0",
                "--no-review",
                "--output-dir",
                str(tmp_path / "runs"),
                "--name",
                backend,
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        summary = json.loads(captured.out)
        assert summary["backend"] == backend
        model = summary["spatial_gaussian_polynomial"]
        assert (
            model["shape_degree"],
            model["photometric_degree"],
            model["background_degree"],
        ) == (1, 1, 0)
        results[backend] = {
            name: np.load(tmp_path / "runs" / backend / filename)
            for name, filename in summary["saved"].items()
            if filename.endswith(".npy")
        }
    for name in (
        "matched",
        "residual",
        "photometric_scale",
        "kernel_center",
        "photometric_coefficients",
        "shape_coefficients",
        "background_coefficients",
    ):
        np.testing.assert_allclose(
            results["native"][name],
            results["cpu"][name],
            rtol=2e-8,
            atol=2e-9,
            equal_nan=True,
        )
