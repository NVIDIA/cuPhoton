# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native spatial fits preserve held-out signals and noise diagnostics."""

from __future__ import annotations

import importlib.util
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
from tests.xpois.test_native_constant_kernel import _forbid_cupy_compilation
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
    with _forbid_cupy_compilation(monkeypatch):
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


def test_native_spatial_preserves_weighting_and_marginal_variance(
    native_cupy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        spatial_model, "_gpu_design_chunk_size", lambda *args, **kwargs: 53
    )
    shape = (34, 38)
    rng = np.random.default_rng(127)
    source = rng.normal(size=shape)
    components = [GaussianBasisComponent(1.1, 1)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7), components, flux_conserve=True
    )
    target = _spatial_target(
        source,
        basis,
        np.array([1.02, 0.015, -0.01]),
        np.array([[0.03, 0.004, -0.003], [-0.02, -0.002, 0.003]]),
        np.array([0.05, -0.01, 0.02]),
        photometric_degree=1,
        shape_degree=1,
        background_degree=1,
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
            shape_degree=1, photometric_degree=1, background_degree=1
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
    with _forbid_cupy_compilation(monkeypatch):
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
    assert native.design_chunk_size == 53
    assert native.fit_pixel_count > 53
    assert native.fit_pixel_count % 53 != 0
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
    native_cupy: Any, monkeypatch: pytest.MonkeyPatch, case: str
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
        with (
            _forbid_cupy_compilation(monkeypatch),
            pytest.raises(ValueError, match=message),
        ):
            solve_spatial_gaussian_polynomial_kernel(
                source, source + 0.1, components, backend=backend, **options
            )
