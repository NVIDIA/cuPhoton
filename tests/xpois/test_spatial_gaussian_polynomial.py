# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from scipy.signal import fftconvolve

import cuphoton.xpois.spatial_als as spatial_als_module
import cuphoton.xpois.spatial_gaussian_polynomial as spatial_model_module
from cuphoton.xpois import (
    GaussianBasisComponent,
    SpatialALSConfig,
    build_gaussian_polynomial_basis,
    solve_constant_kernel,
    solve_spatial_als,
    triangular_degree_pairs,
)
from cuphoton.xpois.spatial_gaussian_polynomial import (
    SpatialGaussianPolynomialKernelConfig,
    SpatialGaussianPolynomialKernelFitSamples,
    SpatialKernelDomain,
    solve_spatial_gaussian_polynomial_kernel,
)


def _random_source(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


def _cupy_device_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    if device_count < 1:
        pytest.skip("CuPy CUDA runtime has no visible device")
    return cp


def _assert_cupy_matches_cpu(cpu, gpu) -> None:
    assert gpu.backend == "cupy"
    assert (
        1
        <= gpu.design_chunk_size
        <= spatial_model_module._GPU_MAX_DESIGN_CHUNK_SIZE
    )
    assert np.array_equal(gpu.fit_mask, cpu.fit_mask)
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
        cpu_value = getattr(cpu, name)
        gpu_value = getattr(gpu, name)
        if cpu_value is None:
            assert gpu_value is None
        else:
            np.testing.assert_allclose(
                gpu_value,
                cpu_value,
                rtol=2.0e-8,
                atol=2.0e-9,
                equal_nan=True,
            )
    np.testing.assert_allclose(
        [gpu.fit_objective, gpu.condition_number],
        [cpu.fit_objective, cpu.condition_number],
        rtol=2.0e-8,
        atol=2.0e-9,
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
        assert getattr(gpu, name) == getattr(cpu, name)


def _chebyshev_field(
    shape: tuple[int, int],
    coefficients: np.ndarray,
    degree: int,
    domain: SpatialKernelDomain | None = None,
) -> np.ndarray:
    terms = triangular_degree_pairs(degree)
    if domain is None:
        y = np.linspace(-1.0, 1.0, shape[0], dtype=np.float64)
        x = np.linspace(-1.0, 1.0, shape[1], dtype=np.float64)
    else:
        origin_y, origin_x = domain.array_origin_yx
        y0, y1, x0, x1 = domain.normalization_bbox
        y = 2.0 * (np.arange(shape[0]) + origin_y - y0) / (y1 - y0 - 1) - 1.0
        x = 2.0 * (np.arange(shape[1]) + origin_x - x0) / (x1 - x0 - 1) - 1.0
    y_values = np.polynomial.chebyshev.chebvander(y, degree)
    x_values = np.polynomial.chebyshev.chebvander(x, degree)
    result = np.zeros(shape, dtype=np.float64)
    for coefficient, (degree_x, degree_y) in zip(
        coefficients,
        terms,
        strict=True,
    ):
        result += coefficient * np.outer(
            y_values[:, degree_y],
            x_values[:, degree_x],
        )
    return result


def _spatial_target(
    source: np.ndarray,
    basis: np.ndarray,
    photometric_coefficients: np.ndarray,
    shape_coefficients: np.ndarray,
    background_coefficients: np.ndarray,
    *,
    photometric_degree: int,
    shape_degree: int,
    background_degree: int,
    domain: SpatialKernelDomain | None = None,
) -> np.ndarray:
    basis_images = np.stack(
        [fftconvolve(source, kernel, mode="same") for kernel in basis],
        axis=0,
    )
    target = (
        _chebyshev_field(
            source.shape,
            photometric_coefficients,
            photometric_degree,
            domain,
        )
        * basis_images[0]
    )
    for coefficients, basis_image in zip(
        shape_coefficients,
        basis_images[1:],
        strict=True,
    ):
        target += (
            _chebyshev_field(
                source.shape,
                coefficients,
                shape_degree,
                domain,
            )
            * basis_image
        )
    target += _chebyshev_field(
        source.shape,
        background_coefficients,
        background_degree,
        domain,
    )
    return target


@pytest.mark.parametrize(
    ("flux_scale", "variance_scale"),
    [
        (1.0, 1.0),
        (1.0e-12, 1.0),
        (1.0e11, 1.0),
        (1.0e-12, 1.0e-24),
        (1.0e11, 1.0e22),
    ],
)
def test_spatial_model_recovers_decoupled_spatial_fields(
    flux_scale: float, variance_scale: float
) -> None:
    shape = (60, 64)
    source = flux_scale * _random_source(shape, seed=11)
    components = [GaussianBasisComponent(sigma=1.2, degree=1)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    photometric_coefficients = np.array([1.02, 0.015, -0.01])
    shape_coefficients = np.array(
        [
            [0.02, 0.004, -0.003],
            [-0.01, 0.002, 0.001],
        ]
    )
    background_coefficients = flux_scale * np.array([0.1, 0.03, -0.02])
    target = _spatial_target(
        source,
        basis,
        photometric_coefficients,
        shape_coefficients,
        background_coefficients,
        photometric_degree=1,
        shape_degree=1,
        background_degree=1,
    )
    variance = 0.5 + np.linspace(0.0, 0.4, shape[1])[None, :]
    variance = variance_scale * np.broadcast_to(variance, shape).copy()

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=1,
            photometric_degree=1,
            background_degree=1,
        ),
        backend="cpu",
    )

    assert result.backend == "cpu"
    assert result.design_chunk_size == spatial_model_module._DESIGN_CHUNK_SIZE
    assert result.fit_selection_kind == "all_valid"
    assert result.fit_weighting == "target_variance"
    assert result.explicit_fit_samples is None
    assert result.fit_pixel_count == result.unique_fit_pixel_count
    assert result.excluded_requested_fit_pixel_count == 0
    assert result.dof > 0
    assert result.condition_number < 10.0
    assert np.allclose(
        result.photometric_coefficients,
        photometric_coefficients,
        atol=1.0e-11,
    )
    assert np.allclose(
        result.shape_coefficients,
        shape_coefficients,
        atol=1.0e-11,
    )
    assert np.allclose(
        result.background_coefficients / flux_scale,
        background_coefficients / flux_scale,
        atol=1.0e-11,
    )
    assert (
        np.max(np.abs(result.residual[result.fit_mask])) / flux_scale
        < 1.0e-12
    )
    assert result.fit_objective * variance_scale / flux_scale**2 < 1.0e-20

    y, x = 17.5, 22.25
    kernel = result.kernel_at_local(y, x)
    assert np.isclose(kernel.sum(), result.photometric_scale_at_local(y, x))
    normalized_y = (2.0 * y / (shape[0] - 1)) - 1.0
    normalized_x = (2.0 * x / (shape[1] - 1)) - 1.0
    assert np.isclose(
        result.photometric_scale_at_local(y, x),
        photometric_coefficients[0]
        + photometric_coefficients[1] * normalized_y
        + photometric_coefficients[2] * normalized_x,
    )


def test_spatial_model_recovers_multiple_gaussian_widths() -> None:
    shape = (40, 44)
    source = _random_source(shape, seed=83)
    components = [
        GaussianBasisComponent(sigma=0.9, degree=1),
        GaussianBasisComponent(sigma=1.7, degree=0),
    ]
    basis, _ = build_gaussian_polynomial_basis(
        (9, 9), components, flux_conserve=True
    )
    photometric_coefficients = np.array([1.03, 0.02, -0.015])
    shape_coefficients = np.array(
        [[0.02, 0.005, -0.003], [-0.01, 0.002, 0.004], [0.25, 0.03, -0.02]]
    )
    background_coefficients = np.array([0.04, -0.008, 0.006])
    # The oracle convolves each fixed basis kernel independently before
    # applying its spatial field; it does not use the chunked fit design.
    target = _spatial_target(
        source,
        basis,
        photometric_coefficients,
        shape_coefficients,
        background_coefficients,
        photometric_degree=1,
        shape_degree=1,
        background_degree=1,
    )

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(9, 9),
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=1,
            photometric_degree=1,
            background_degree=1,
        ),
    )

    for actual, expected in (
        (result.photometric_coefficients, photometric_coefficients),
        (result.shape_coefficients, shape_coefficients),
        (result.background_coefficients, background_coefficients),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(
        result.matched[result.fit_mask],
        target[result.fit_mask],
        rtol=0.0,
        atol=1e-12,
    )
    for y, x in ((4, 4), (19, 21), (35, 39)):
        terms = np.array([1.0, 2.0 * y / 39.0 - 1.0, 2.0 * x / 43.0 - 1.0])
        expected = (photometric_coefficients @ terms) * basis[0]
        expected += np.tensordot(shape_coefficients @ terms, basis[1:], 1)
        np.testing.assert_allclose(
            result.kernel_at_local(y, x), expected, rtol=0.0, atol=1e-12
        )
        assert np.linalg.matrix_rank(expected, tol=1e-12) > 1


def test_spatial_model_uses_explicit_parent_spatial_domain() -> None:
    shape = (40, 44)
    source = _random_source(shape, seed=13)
    components = [GaussianBasisComponent(sigma=1.2, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    domain = SpatialKernelDomain(
        array_origin_yx=(112, 518),
        normalization_bbox=(100, 180, 500, 596),
    )
    local_y, local_x = np.mgrid[: shape[0], : shape[1]]
    normalized_y = 2.0 * (local_y + 12) / 79.0 - 1.0
    normalized_x = 2.0 * (local_x + 18) / 95.0 - 1.0
    photometric_coefficients = np.array([1.02, 0.03, -0.02])
    background_coefficients = np.array([0.08, -0.01, 0.015])
    photometric_scale = (
        photometric_coefficients[0]
        + photometric_coefficients[1] * normalized_y
        + photometric_coefficients[2] * normalized_x
    )
    background = (
        background_coefficients[0]
        + background_coefficients[1] * normalized_y
        + background_coefficients[2] * normalized_x
    )
    target = (
        photometric_scale
        * fftconvolve(
            source,
            basis[0],
            mode="same",
        )
        + background
    )

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        spatial_domain=domain,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=1,
            background_degree=1,
        ),
        backend="cpu",
    )

    assert result.spatial_domain == domain
    assert np.allclose(
        result.photometric_coefficients,
        photometric_coefficients,
        atol=1.0e-11,
    )
    assert np.allclose(
        result.background_coefficients,
        background_coefficients,
        atol=1.0e-11,
    )
    y, x = 17.5, 22.25
    expected_scale = (
        photometric_coefficients[0]
        + photometric_coefficients[1] * (2.0 * (y + 12) / 79.0 - 1.0)
        + photometric_coefficients[2] * (2.0 * (x + 18) / 95.0 - 1.0)
    )
    assert result.photometric_scale_at_local(y, x) == pytest.approx(
        expected_scale
    )
    assert result.photometric_scale_at_parent(
        y + 112,
        x + 518,
    ) == pytest.approx(expected_scale)
    assert np.array_equal(
        result.kernel_at_local(y, x),
        result.kernel_at_parent(y + 112, x + 518),
    )
    assert np.max(np.abs(result.residual[result.fit_mask])) < 1.0e-12

    # Parent-frame evaluation covers the whole normalization domain, so a
    # cutout extrapolates its fields beyond the fitted array.
    corner_scale = result.photometric_scale_at_parent(100, 500)
    assert corner_scale == pytest.approx(
        photometric_coefficients[0]
        - photometric_coefficients[1]
        - photometric_coefficients[2]
    )
    assert np.isclose(result.kernel_at_parent(100, 500).sum(), corner_scale)
    with pytest.raises(ValueError, match="inside the local array"):
        result.kernel_at_local(-12, -18)
    with pytest.raises(ValueError, match="inside the parent pixel domain"):
        result.photometric_scale_at_parent(180, 500)


def test_spatial_model_resolves_default_spatial_domain() -> None:
    shape = (36, 38)
    source = _random_source(shape, seed=15)
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        source.copy(),
        components,
        kernel_shape=(7, 7),
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )

    assert result.spatial_domain == SpatialKernelDomain(
        array_origin_yx=(0, 0),
        normalization_bbox=(0, shape[0], 0, shape[1]),
    )

    with pytest.raises(ValueError, match="inside the local array"):
        result.kernel_at_local(-0.5, 5.0)
    with pytest.raises(ValueError, match="inside the local array"):
        result.photometric_scale_at_local(5.0, float("nan"))
    with pytest.raises(ValueError, match="inside the parent pixel domain"):
        result.kernel_at_parent(shape[0], 5.0)


def test_spatial_model_degree_zero_matches_constant_solver() -> None:
    shape = (56, 60)
    source = _random_source(shape, seed=17)
    components = [GaussianBasisComponent(sigma=1.3, degree=2)]
    basis, _ = build_gaussian_polynomial_basis(
        (9, 9),
        components,
        flux_conserve=True,
    )
    true_coefficients = np.array([0.98, 0.02, -0.01, 0.0, 0.006, 0.0])
    kernel = np.tensordot(true_coefficients, basis, axes=(0, 0))
    y_gradient = np.linspace(-1.0, 1.0, shape[0])[:, None]
    x_gradient = np.linspace(-1.0, 1.0, shape[1])[None, :]
    # Noise with spatially varying variance makes the weighted solution
    # depend on the row weights, so the normal-equation constant solver is an
    # independent oracle for the weight exponent, not only for the design.
    variance = 0.3 + 0.5 * np.random.default_rng(18).random(shape)
    noise = (
        0.1 * np.sqrt(variance) * np.random.default_rng(19).normal(size=shape)
    )
    target = (
        fftconvolve(source, kernel, mode="same")
        + 0.08
        + 0.02 * y_gradient
        - 0.01 * x_gradient
        + noise
    )

    constant = solve_constant_kernel(
        source,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        background_degree=1,
        flux_conserve=True,
        backend="cpu",
    )
    spatial = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=1,
        ),
        backend="cpu",
    )

    assert spatial.fit_objective > 1.0
    assert np.allclose(
        spatial.kernel_at_local(20, 20),
        constant.kernel,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(
        spatial.matched[spatial.fit_mask],
        constant.matched[constant.fit_mask],
        rtol=1e-10,
        atol=1e-12,
    )
    assert spatial.fit_objective == pytest.approx(constant.chi2, rel=1e-10)
    # A wrong weight exponent moves the minimizer by ~1e-4 on this problem.
    wrong_weights = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance**2,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=1,
        ),
    )
    assert (
        np.abs(wrong_weights.kernel_at_local(20, 20) - constant.kernel).max()
        > 1e-6
    )


@pytest.mark.parametrize("chunk_size", [1, 5, 4096])
def test_accumulate_qr_matches_dense_weighted_least_squares(
    chunk_size: int,
) -> None:
    shape = (30, 34)
    source = _random_source(shape, seed=77)
    components = [GaussianBasisComponent(sigma=1.2, degree=1)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    kernel = 1.05 * basis[0] + 0.03 * basis[1] - 0.02 * basis[2]
    noise = 0.1 * np.random.default_rng(78).normal(size=shape)
    target = fftconvolve(source, kernel, mode="same") + 0.05 + noise
    variance = 0.4 + 0.6 * np.random.default_rng(79).random(shape)
    domain = SpatialKernelDomain(
        array_origin_yx=(0, 0),
        normalization_bbox=(0, shape[0], 0, shape[1]),
    )
    fit_mask = np.zeros(shape, dtype=bool)
    fit_mask[3:-3, 3:-3] = True
    positions = np.column_stack(np.nonzero(fit_mask))
    precision = np.linspace(0.5, 2.0, positions.shape[0])
    sample_y = positions[:, 0]
    sample_x = positions[:, 1]
    terms = tuple(triangular_degree_pairs(1))

    patches, basis_flat, margin_y, margin_x = (
        spatial_model_module._patch_design_inputs(source, basis)
    )
    basis_values = (
        patches[sample_y - margin_y, sample_x - margin_x].reshape(
            sample_y.size, -1
        )
        @ basis_flat.T
    )
    design = spatial_model_module._model_design(
        basis_values,
        sample_y,
        sample_x,
        domain,
        terms,
        terms,
        terms,
    )
    row_scale = np.sqrt(precision / variance[sample_y, sample_x])
    weighted_design = design * row_scale[:, None]
    weighted_target = target[sample_y, sample_x] * row_scale
    expected, *_ = np.linalg.lstsq(
        weighted_design,
        weighted_target,
        rcond=None,
    )
    expected_condition = np.linalg.cond(
        weighted_design / np.linalg.norm(weighted_design, axis=0)
    )
    assert sample_y.size > 2 * chunk_size or chunk_size == 4096

    upper, transformed_rhs, row_count = spatial_model_module._accumulate_qr(
        source,
        target,
        variance,
        sample_y * shape[1] + sample_x,
        precision,
        domain,
        basis,
        terms,
        terms,
        terms,
        chunk_size=chunk_size,
    )
    coefficients, condition_number = spatial_model_module._solve_scaled_qr(
        upper,
        transformed_rhs,
        condition_limit=1.0e12,
    )

    assert row_count == sample_y.size
    assert upper.shape == (design.shape[1], design.shape[1])
    assert np.allclose(coefficients, expected, rtol=1e-12, atol=1e-14)
    assert condition_number == pytest.approx(expected_condition, rel=1e-12)

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        fit_samples=SpatialGaussianPolynomialKernelFitSamples(
            positions_yx=positions,
            relative_precision=precision,
        ),
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=1,
            photometric_degree=1,
            background_degree=1,
        ),
    )
    fitted = np.concatenate(
        (
            result.photometric_coefficients,
            result.shape_coefficients.ravel(),
            result.background_coefficients,
        )
    )
    assert np.allclose(fitted, expected, rtol=1e-12, atol=1e-14)
    assert result.condition_number == pytest.approx(
        expected_condition,
        rel=1e-12,
    )
    assert result.fit_objective == pytest.approx(
        np.sum((weighted_design @ expected - weighted_target) ** 2),
        rel=1e-12,
    )


def test_spatial_model_represents_rank_two_kernel_that_als_cannot() -> None:
    shape = (72, 76)
    source = _random_source(shape, seed=23)
    components = [GaussianBasisComponent(sigma=1.3, degree=2)]
    basis, _ = build_gaussian_polynomial_basis(
        (9, 9),
        components,
        flux_conserve=True,
    )
    true_coefficients = np.array([1.0, 0.2, 0.0, 0.0, -0.15, 0.0])
    true_kernel = np.tensordot(true_coefficients, basis, axes=(0, 0))
    target = fftconvolve(source, true_kernel, mode="same") + 0.07
    variance = np.ones(shape, dtype=np.float64)

    linear = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )
    als = solve_spatial_als(
        source,
        target,
        components,
        kernel_shape=(9, 9),
        variance=variance,
        config=SpatialALSConfig(
            spatial_degree=0,
            background_degree=0,
            max_iterations=30,
            tolerance=1.0e-10,
            regularization=0.0,
            flux_conserve=True,
        ),
        backend="cpu",
    )

    assert np.linalg.matrix_rank(true_kernel, tol=1.0e-12) == 2
    assert linear.fit_objective < 1.0e-20
    assert als.chi2 > 10.0
    assert als.chi2 > linear.fit_objective + 10.0


def test_spatial_model_excludes_invalid_inputs_and_respects_mask() -> None:
    shape = (48, 52)
    source = _random_source(shape, seed=29)
    components = [GaussianBasisComponent(sigma=1.2, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    target = fftconvolve(source, basis[0], mode="same") + 0.1
    variance = np.ones(shape, dtype=np.float64)
    fit_mask = np.zeros(shape, dtype=bool)
    fit_mask[8:40, 9:44] = True
    source[20, 21] = np.nan
    target[30, 31] = np.nan
    variance[34, 35] = np.inf

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        fit_mask=fit_mask,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )

    assert not result.fit_mask[20, 21]
    assert not result.fit_mask[30, 31]
    assert not result.fit_mask[34, 35]
    assert np.isnan(result.matched[20, 21])
    assert np.isnan(result.residual[30, 31])
    assert result.fit_pixel_count < int(fit_mask.sum())
    assert result.fit_selection_kind == "mask"
    assert result.requested_fit_pixel_count == int(fit_mask.sum())
    assert result.excluded_requested_fit_pixel_count == (
        int(fit_mask.sum()) - result.fit_pixel_count
    )


def test_footprint_masks_are_exact_beyond_int16_counts() -> None:
    shape = (300, 300)
    kernel_shape = (183, 183)
    assert kernel_shape[0] * kernel_shape[1] > np.iinfo(np.int16).max
    image = np.full(shape, np.nan)
    image[150, 150] = 1.0

    invalid = spatial_model_module._invalid_input_mask(
        image,
        np.zeros(shape),
        np.ones(shape),
        kernel_shape,
    )
    finite = spatial_model_module._finite_footprint_mask(image, kernel_shape)

    assert invalid.all()
    assert not finite.any()

    single = np.ones(shape)
    single[100, 120] = np.nan
    expected = np.ones(shape, dtype=bool)
    expected[100 - 91 : 100 + 92, 120 - 91 : 120 + 92] = False
    assert np.array_equal(
        spatial_model_module._finite_footprint_mask(single, kernel_shape),
        expected,
    )
    assert np.array_equal(
        spatial_model_module._invalid_input_mask(
            single,
            np.ones(shape),
            np.ones(shape),
            kernel_shape,
        ),
        ~expected,
    )


def test_spatial_model_explicit_samples_match_equivalent_mask() -> None:
    shape = (48, 52)
    source = _random_source(shape, seed=53)
    components = [GaussianBasisComponent(sigma=1.2, degree=1)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    true_kernel = np.tensordot(
        np.array([1.03, 0.04, -0.025]),
        basis,
        axes=(0, 0),
    )
    noise = np.random.default_rng(55).normal(scale=0.01, size=shape)
    target = fftconvolve(source, true_kernel, mode="same") + 0.06 + noise
    variance = 0.4 + np.linspace(0.0, 0.3, source.size).reshape(shape)
    fit_mask = np.zeros(shape, dtype=bool)
    fit_mask[6:-6:2, 7:-7:2] = True
    expected_positions = np.column_stack(np.nonzero(fit_mask))
    input_positions = expected_positions[::-1].astype(np.float64)
    samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=input_positions
    )
    input_positions[:] = -1.0

    config = SpatialGaussianPolynomialKernelConfig(
        shape_degree=0,
        photometric_degree=0,
        background_degree=0,
    )
    domain = SpatialKernelDomain(
        array_origin_yx=(100, 500),
        normalization_bbox=(100, 148, 500, 552),
    )
    masked = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        fit_mask=fit_mask,
        spatial_domain=domain,
        config=config,
        backend="cpu",
    )
    explicit = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        fit_samples=samples,
        spatial_domain=domain,
        config=config,
        backend="cpu",
    )

    assert np.array_equal(samples.positions_yx, expected_positions)
    assert not samples.positions_yx.flags.writeable
    assert explicit.explicit_fit_samples is samples
    assert masked.explicit_fit_samples is None
    assert explicit.fit_selection_kind == "samples"
    assert masked.fit_selection_kind == "mask"
    assert explicit.fit_weighting == "target_variance"
    assert explicit.requested_fit_pixel_count == expected_positions.shape[0]
    assert explicit.excluded_requested_fit_pixel_count == 0
    assert explicit.fit_pixel_count == explicit.unique_fit_pixel_count
    assert explicit.dof == masked.dof
    assert np.array_equal(explicit.fit_mask, masked.fit_mask)
    assert np.array_equal(
        explicit.photometric_coefficients,
        masked.photometric_coefficients,
    )
    assert np.array_equal(
        explicit.shape_coefficients,
        masked.shape_coefficients,
    )
    assert np.array_equal(
        explicit.background_coefficients,
        masked.background_coefficients,
    )
    assert explicit.fit_objective == pytest.approx(masked.fit_objective)
    assert np.allclose(explicit.matched, masked.matched, equal_nan=True)


def test_relative_precision_matches_adjusted_variance() -> None:
    shape = (42, 46)
    source = _random_source(shape, seed=57)
    components = [GaussianBasisComponent(sigma=1.1, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    noise = np.random.default_rng(59).normal(scale=0.02, size=shape)
    target = fftconvolve(source, 1.04 * basis[0], mode="same") + 0.03 + noise
    variance = 0.5 + np.linspace(0.0, 0.2, source.size).reshape(shape)
    fit_mask = np.zeros(shape, dtype=bool)
    fit_mask[5:-5:2, 6:-6:2] = True
    positions = np.column_stack(np.nonzero(fit_mask))
    precision = np.linspace(0.5, 2.0, positions.shape[0])
    input_precision = precision[::-1].copy()
    weighted_samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=positions[::-1],
        relative_precision=input_precision,
    )
    input_precision[:] = 9.0
    unit_samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=positions
    )
    adjusted_variance = variance.copy()
    adjusted_variance[positions[:, 0], positions[:, 1]] /= precision
    config = SpatialGaussianPolynomialKernelConfig(
        shape_degree=0,
        photometric_degree=0,
        background_degree=0,
    )

    weighted = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        fit_samples=weighted_samples,
        config=config,
        backend="cpu",
    )
    adjusted = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=adjusted_variance,
        fit_samples=unit_samples,
        config=config,
        backend="cpu",
    )
    relative_only = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        fit_samples=weighted_samples,
        config=config,
        backend="cpu",
    )

    assert weighted.fit_weighting == "target_variance_relative_precision"
    assert adjusted.fit_weighting == "target_variance"
    assert relative_only.fit_weighting == "relative_precision"
    assert relative_only.dof == weighted.dof
    assert (
        relative_only.fit_pixel_count == relative_only.unique_fit_pixel_count
    )
    assert weighted_samples.relative_precision is not None
    assert np.array_equal(weighted_samples.relative_precision, precision)
    assert not weighted_samples.relative_precision.flags.writeable
    assert np.allclose(
        weighted.photometric_coefficients,
        adjusted.photometric_coefficients,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert np.allclose(
        weighted.background_coefficients,
        adjusted.background_coefficients,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert weighted.fit_objective == pytest.approx(adjusted.fit_objective)
    sample_y = weighted_samples.positions_yx[:, 0]
    sample_x = weighted_samples.positions_yx[:, 1]
    expected_objective = np.sum(
        weighted.residual[sample_y, sample_x] ** 2
        * weighted_samples.relative_precision
        / variance[sample_y, sample_x]
    )
    assert weighted.fit_objective == pytest.approx(expected_objective)
    expected_relative_only = np.sum(
        relative_only.residual[sample_y, sample_x] ** 2
        * weighted_samples.relative_precision
    )
    assert relative_only.fit_objective == pytest.approx(
        expected_relative_only
    )


@pytest.mark.parametrize(
    ("residual", "variance", "precision"),
    [
        (1.0e200, 1.0e200, None),
        (1.0e-200, 1.0e-200, None),
        (1.0e200, 1.0, 1.0e-200),
        (1.0e-200, 1.0, 1.0e200),
        (1.0e-160, 1.0e-320, None),
        (1.0e154, 1.0e308, None),
        (1.0e160, 1.0e200, 1.0e-200),
        (1.0e-200, 1.0e-200, 1.0e200),
    ],
)
def test_fit_objective_weights_before_squaring(
    residual: float,
    variance: float,
    precision: float | None,
) -> None:
    relative_precision = None if precision is None else np.array([precision])

    objective = spatial_model_module._evaluate_fit_objective(
        np.array([[residual]]),
        np.array([[variance]]),
        np.array([0]),
        relative_precision,
    )

    expected = Decimal.from_float(residual) ** 2 / Decimal.from_float(
        variance
    )
    if precision is not None:
        expected *= Decimal.from_float(precision)
    assert np.isfinite(objective)
    assert np.isclose(objective, float(expected), rtol=1.0e-12, atol=0.0)


def test_spatial_model_reports_target_only_noise_diagnostics() -> None:
    shape = (44, 48)
    source = _random_source(shape, seed=37)
    components = [GaussianBasisComponent(sigma=1.1, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    target = fftconvolve(source, 1.04 * basis[0], mode="same") + 0.03
    variance = 0.4 + np.linspace(0.0, 0.3, source.size).reshape(shape)

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )

    valid = np.isfinite(result.residual)
    assert not result.source_variance_included
    assert result.fit_weighting == "target_variance"
    assert result.propagated_source_variance is None
    assert result.marginal_residual_variance is not None
    assert result.marginal_standardized_residual is not None
    assert np.array_equal(
        result.marginal_residual_variance[valid],
        variance[valid],
    )
    assert np.allclose(
        result.marginal_standardized_residual[valid],
        result.residual[valid] / np.sqrt(variance[valid]),
    )


def test_spatial_model_omits_noise_diagnostics_without_variance() -> None:
    shape = (38, 42)
    source = _random_source(shape, seed=39)
    components = [GaussianBasisComponent(sigma=1.1, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    target = fftconvolve(source, basis[0], mode="same")

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )

    assert not result.source_variance_included
    assert result.fit_weighting == "uniform"
    assert result.propagated_source_variance is None
    assert result.marginal_residual_variance is None
    assert result.marginal_standardized_residual is None


def test_spatial_model_propagates_source_variance() -> None:
    shape = (46, 50)
    source = _random_source(shape, seed=41)
    components = [GaussianBasisComponent(sigma=1.2, degree=1)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    photometric_coefficients = np.array([1.01, 0.02, -0.015])
    shape_coefficients = np.array(
        [
            [0.025, 0.006, -0.004],
            [-0.018, 0.003, 0.005],
        ]
    )
    background_coefficients = np.array([0.04])
    domain = SpatialKernelDomain(
        array_origin_yx=(111, 517),
        normalization_bbox=(100, 180, 500, 596),
    )
    target = _spatial_target(
        source,
        basis,
        photometric_coefficients,
        shape_coefficients,
        background_coefficients,
        photometric_degree=1,
        shape_degree=1,
        background_degree=0,
        domain=domain,
    )
    target_variance = 0.3 + np.linspace(0.0, 0.2, source.size).reshape(shape)
    source_variance = 0.5 + np.linspace(0.0, 0.4, source.size).reshape(shape)

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=target_variance,
        source_variance=source_variance,
        spatial_domain=domain,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=1,
            photometric_degree=1,
            background_degree=0,
        ),
        backend="cpu",
    )

    assert result.source_variance_included
    assert result.propagated_source_variance is not None
    assert result.marginal_residual_variance is not None
    assert result.marginal_standardized_residual is not None
    for y, x in ((8, 9), (22, 25), (37, 40)):
        kernel = result.kernel_at_local(y, x)
        variance_patch = source_variance[y - 3 : y + 4, x - 3 : x + 4]
        expected_propagated = np.sum(variance_patch * kernel[::-1, ::-1] ** 2)
        expected = target_variance[y, x] + expected_propagated
        assert result.propagated_source_variance[y, x] == pytest.approx(
            expected_propagated
        )
        assert result.marginal_residual_variance[y, x] == pytest.approx(
            expected
        )
        assert result.marginal_standardized_residual[y, x] == pytest.approx(
            result.residual[y, x] / np.sqrt(expected)
        )


def test_source_variance_invalidates_only_affected_diagnostics() -> None:
    shape = (42, 46)
    source = _random_source(shape, seed=43)
    components = [GaussianBasisComponent(sigma=1.1, degree=0)]
    target_variance = np.ones(shape)
    source_variance = np.ones(shape)
    source_variance[20, 21] = np.nan

    baseline = solve_spatial_gaussian_polynomial_kernel(
        source,
        source.copy(),
        components,
        kernel_shape=(7, 7),
        variance=target_variance,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )
    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        source.copy(),
        components,
        kernel_shape=(7, 7),
        variance=target_variance,
        source_variance=source_variance,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )

    affected = np.s_[17:24, 18:25]
    assert np.array_equal(result.fit_mask, baseline.fit_mask)
    assert np.array_equal(
        result.photometric_coefficients,
        baseline.photometric_coefficients,
    )
    assert np.array_equal(
        result.background_coefficients,
        baseline.background_coefficients,
    )
    assert result.fit_mask[affected].all()
    assert np.isfinite(result.residual[affected]).all()
    assert result.propagated_source_variance is not None
    assert result.marginal_residual_variance is not None
    assert result.marginal_standardized_residual is not None
    assert np.isnan(result.propagated_source_variance[affected]).all()
    assert np.isnan(result.marginal_residual_variance[affected]).all()
    assert np.isnan(result.marginal_standardized_residual[affected]).all()


@pytest.mark.parametrize("backend", ["cpu", "cupy"])
@pytest.mark.parametrize(
    ("scale", "target_variance", "propagation_overflows"),
    [(2.0, 1.0, True), (1.0, 1.0e308, False)],
    ids=["propagated-variance", "marginal-variance-sum"],
)
def test_source_variance_overflow_invalidates_only_noise_diagnostics(
    scale: float,
    target_variance: float,
    propagation_overflows: bool,
    backend: str,
) -> None:
    if backend == "cupy":
        _cupy_device_or_skip()
    rng = np.random.default_rng(20260920)
    source = rng.normal(size=(9, 9))
    target = scale * source + rng.normal(scale=0.1, size=source.shape)
    variance = np.full(source.shape, target_variance)
    source_variance = np.zeros(source.shape)
    source_variance[4, 4] = 1.0e308
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]
    config = SpatialGaussianPolynomialKernelConfig(
        shape_degree=0, photometric_degree=0, background_degree=0
    )
    assert np.isfinite(source_variance).all()
    assert np.isfinite(variance).all()
    baseline = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(1, 1),
        variance=variance,
        config=config,
        backend=backend,
    )
    with np.errstate(over="ignore"):
        result = solve_spatial_gaussian_polynomial_kernel(
            source,
            target,
            components,
            kernel_shape=(1, 1),
            variance=variance,
            source_variance=source_variance,
            config=config,
            backend=backend,
        )

    assert result.backend == backend
    assert np.array_equal(result.fit_mask, baseline.fit_mask)
    assert np.array_equal(result.matched, baseline.matched)
    assert np.array_equal(result.residual, baseline.residual)
    assert result.fit_objective == baseline.fit_objective
    assert np.isfinite(result.fit_objective)
    assert np.isfinite(result.residual).all()
    assert result.residual[4, 4] != 0.0
    assert result.propagated_source_variance is not None
    assert result.marginal_residual_variance is not None
    assert result.marginal_standardized_residual is not None
    if propagation_overflows:
        assert np.isnan(result.propagated_source_variance[4, 4])
    else:
        assert np.isfinite(result.propagated_source_variance[4, 4])
    assert np.isnan(result.marginal_residual_variance[4, 4])
    assert np.isnan(result.marginal_standardized_residual[4, 4])

    unaffected = np.ones(source.shape, dtype=bool)
    unaffected[4, 4] = False
    assert np.all(result.propagated_source_variance[unaffected] == 0.0)
    assert np.isfinite(result.marginal_residual_variance[unaffected]).all()
    assert np.array_equal(
        result.marginal_standardized_residual[unaffected],
        baseline.marginal_standardized_residual[unaffected],
    )


def test_spatial_model_rejects_invalid_source_variance() -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=47)
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]

    with pytest.raises(ValueError, match="requires target variance"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            components,
            source_variance=np.ones(shape),
            backend="cpu",
        )
    with pytest.raises(ValueError, match="match the image shape"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            components,
            variance=np.ones(shape),
            source_variance=np.ones((31, 34)),
            backend="cpu",
        )
    invalid = np.ones(shape)
    invalid[12, 13] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            components,
            variance=np.ones(shape),
            source_variance=invalid,
            backend="cpu",
        )


@pytest.mark.parametrize(
    ("origin", "bbox", "message"),
    [
        ((0,), (0, 10, 0, 10), "2-tuple"),
        ((0, 0), (0, 10, 0), "4-tuple"),
        ((False, 0), (0, 10, 0, 10), "integers"),
        ((0, 0), (0, 0, 0, 10), "positive extent"),
    ],
)
def test_spatial_kernel_domain_rejects_invalid_values(
    origin: tuple[int, ...],
    bbox: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialKernelDomain(
            array_origin_yx=origin,  # type: ignore[arg-type]
            normalization_bbox=bbox,  # type: ignore[arg-type]
        )


def test_spatial_model_singleton_axis_requires_degree_zero() -> None:
    shape = (1, 30)
    source = _random_source(shape, seed=75)
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (1, 5),
        components,
        flux_conserve=True,
    )
    target = fftconvolve(source, 1.05 * basis[0], mode="same") + 0.02

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(1, 5),
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=2,
            photometric_degree=0,
            background_degree=0,
        ),
    )

    assert result.spatial_domain.normalization_bbox == (0, 1, 0, 30)
    assert np.allclose(result.photometric_scale, 1.05, atol=1.0e-9)
    assert np.allclose(result.background, 0.02, atol=1.0e-9)
    assert result.photometric_scale_at_local(0, 12.5) == pytest.approx(1.05)

    for name, degree in (
        ("photometric_degree", 1),
        ("background_degree", 2),
    ):
        degrees = {
            "shape_degree": 0,
            "photometric_degree": 0,
            "background_degree": 0,
            name: degree,
        }
        with pytest.raises(ValueError, match="extent 1 along y"):
            solve_spatial_gaussian_polynomial_kernel(
                source,
                target,
                components,
                kernel_shape=(1, 5),
                config=SpatialGaussianPolynomialKernelConfig(**degrees),
            )
    with pytest.raises(ValueError, match="extent 1 along y"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            target,
            [
                GaussianBasisComponent(sigma=1.0, degree=0),
                GaussianBasisComponent(sigma=1.5, degree=0),
            ],
            kernel_shape=(1, 5),
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=1,
                photometric_degree=0,
                background_degree=0,
            ),
        )


def test_spatial_model_accepts_array_on_spatial_domain_edges() -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=49)
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]
    domain = SpatialKernelDomain(
        array_origin_yx=(100, 500),
        normalization_bbox=(100, 132, 500, 534),
    )

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        source.copy(),
        components,
        kernel_shape=(7, 7),
        spatial_domain=domain,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="cpu",
    )

    assert result.spatial_domain == domain


@pytest.mark.parametrize(
    "origin",
    [
        (99, 500),
        (100, 499),
        (101, 500),
        (100, 501),
    ],
)
def test_spatial_model_rejects_array_outside_spatial_domain(
    origin: tuple[int, int],
) -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=51)
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]
    domain = SpatialKernelDomain(
        array_origin_yx=origin,
        normalization_bbox=(100, 132, 500, 534),
    )

    with pytest.raises(ValueError, match="array extent"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            components,
            spatial_domain=domain,
            backend="cpu",
        )


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.array([4, 5]), "shape"),
        (np.empty((0, 2)), "must not be empty"),
        (np.array([["4", "5"]]), "numeric"),
        (np.array([[4.0, np.nan]]), "NaN or inf"),
        (np.array([[4.5, 5.0]]), "integer values"),
        (np.array([[4, 5], [4, 5]]), "duplicate pixels"),
        (np.array([[1.0e20, 5.0]]), "supported range"),
    ],
)
def test_spatial_kernel_fit_samples_rejects_invalid_positions(
    positions: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialGaussianPolynomialKernelFitSamples(positions_yx=positions)


@pytest.mark.parametrize(
    ("precision", "message"),
    [
        (np.ones((2, 1)), "one value"),
        (np.array(["1", "2"]), "numeric"),
        (np.array([1.0 + 0.0j, 2.0 + 0.0j]), "numeric"),
        (np.array([True, False]), "numeric"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, 0.0]), "strictly positive"),
        (np.array([1.0, -1.0]), "strictly positive"),
    ],
)
def test_spatial_kernel_fit_samples_rejects_invalid_precision(
    precision: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialGaussianPolynomialKernelFitSamples(
            positions_yx=np.array([[4, 5], [6, 7]]),
            relative_precision=precision,
        )


def test_spatial_model_rejects_conflicting_fit_selectors() -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=61)
    samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=np.array([[10, 11]])
    )

    with pytest.raises(ValueError, match="either fit_mask or fit_samples"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            [GaussianBasisComponent(sigma=1.0, degree=0)],
            fit_mask=np.ones(shape, dtype=bool),
            fit_samples=samples,
            backend="cpu",
        )


def test_spatial_model_rejects_samples_outside_kernel_interior() -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=63)
    samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=np.array([[0, 0]])
    )

    with pytest.raises(ValueError, match="valid kernel interior"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            [GaussianBasisComponent(sigma=1.0, degree=0)],
            kernel_shape=(7, 7),
            fit_samples=samples,
            backend="cpu",
        )


@pytest.mark.parametrize("invalid_input", ["source", "target", "variance"])
def test_spatial_model_rejects_samples_with_invalid_inputs(
    invalid_input: str,
) -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=65)
    target = source.copy()
    variance = np.ones(shape)
    if invalid_input == "source":
        source[16, 17] = np.nan
    elif invalid_input == "target":
        target[16, 17] = np.nan
    else:
        variance[16, 17] = np.inf
    samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=np.array([[16, 17]])
    )

    with pytest.raises(ValueError, match="finite target and variance"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            target,
            [GaussianBasisComponent(sigma=1.0, degree=0)],
            kernel_shape=(7, 7),
            variance=variance,
            fit_samples=samples,
            backend="cpu",
        )


def test_spatial_model_rejects_invalid_array_contracts() -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=69)
    target = source.copy()
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]
    config = SpatialGaussianPolynomialKernelConfig(
        shape_degree=0,
        photometric_degree=0,
        background_degree=0,
    )

    def solve(
        source_image: np.ndarray = source,
        target_image: np.ndarray = target,
        **overrides: object,
    ):
        arguments: dict[str, object] = {
            "kernel_shape": (7, 7),
            "config": config,
        }
        arguments.update(overrides)
        return solve_spatial_gaussian_polynomial_kernel(
            source_image,
            target_image,
            components,
            **arguments,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="share the same shape"):
        solve(target_image=target[:-1])
    with pytest.raises(ValueError, match="2D arrays"):
        solve(source.ravel(), target.ravel())
    for kernel_shape, message in (
        ((6, 7), "odd dimensions"),
        ((7.0, 7), "dimensions must be integers"),
        ((0, 7), "dimensions must be positive"),
        ((33, 7), "no valid interior pixels"),
        (7, "must be a 2-tuple"),
        ((7, 7, 7), "must be a 2-tuple"),
    ):
        with pytest.raises(ValueError, match=message):
            solve(kernel_shape=kernel_shape)
    with pytest.raises(ValueError, match="strictly positive"):
        solve(variance=np.zeros(shape))
    with pytest.raises(ValueError, match="variance must match the image"):
        solve(variance=np.ones((32, 33)))
    with pytest.raises(ValueError, match="fit_mask must match the image"):
        solve(fit_mask=np.ones((32, 33), dtype=bool))
    with pytest.raises(ValueError, match="boolean or binary numeric"):
        solve(fit_mask=np.full(shape, "1"))
    for bad_value, message in (
        (np.nan, "NaN or inf values"),
        (2.0, "only 0/1 values"),
    ):
        numeric_mask = np.ones(shape)
        numeric_mask[10, 10] = bad_value
        with pytest.raises(ValueError, match=message):
            solve(fit_mask=numeric_mask)
    with pytest.raises(ValueError, match="no finite pixels remain"):
        solve(fit_mask=np.zeros(shape, dtype=bool))
    with pytest.raises(TypeError, match="FitSamples instance"):
        solve(fit_samples=np.array([[10, 11]]))
    with pytest.raises(ValueError, match="valid kernel interior"):
        solve(
            fit_samples=SpatialGaussianPolynomialKernelFitSamples(
                positions_yx=np.array([[-1, 5]])
            )
        )

    boolean_mask = np.zeros(shape, dtype=bool)
    boolean_mask[8:24, 9:26] = True
    from_boolean = solve(fit_mask=boolean_mask)
    from_numeric = solve(fit_mask=boolean_mask.astype(np.float64))
    assert np.array_equal(from_numeric.fit_mask, from_boolean.fit_mask)
    assert from_numeric.fit_selection_kind == "mask"


def test_spatial_model_precision_does_not_increase_dof() -> None:
    shape = (32, 34)
    source = _random_source(shape, seed=67)
    samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=np.array([[16, 17]]),
        relative_precision=np.array([1.0e100]),
    )

    with pytest.raises(ValueError, match="underdetermined"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            [GaussianBasisComponent(sigma=1.0, degree=0)],
            kernel_shape=(7, 7),
            fit_samples=samples,
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=0,
                photometric_degree=0,
                background_degree=0,
            ),
            backend="cpu",
        )


def test_spatial_model_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spatial_als_module,
        "_cupy_is_available",
        lambda: False,
    )
    shape = (15, 17)
    source = _random_source(shape, seed=71)
    components = [GaussianBasisComponent(sigma=1.0, degree=0)]
    basis, _ = build_gaussian_polynomial_basis(
        (3, 3),
        components,
        flux_conserve=True,
    )
    target = fftconvolve(source, basis[0], mode="same") + 0.02

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(3, 3),
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
        backend="auto",
    )

    assert result.backend == "cpu"
    assert result.design_chunk_size == spatial_model_module._DESIGN_CHUNK_SIZE


def test_spatial_model_auto_prefers_cupy_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spatial_als_module,
        "_cupy_is_available",
        lambda: True,
    )

    assert spatial_model_module._resolve_spatial_backend("auto") == "cupy"


def test_spatial_model_explicit_cupy_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable():
        raise ImportError("test CuPy failure")

    monkeypatch.setattr(spatial_model_module, "_load_cupy", unavailable)
    image = np.ones((9, 9), dtype=np.float64)

    with pytest.raises(ImportError, match="test CuPy failure"):
        solve_spatial_gaussian_polynomial_kernel(
            image,
            image,
            [GaussianBasisComponent(sigma=1.0, degree=0)],
            kernel_shape=(3, 3),
            backend="cupy",
        )


def test_spatial_model_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="auto, cpu, cupy"):
        solve_spatial_gaussian_polynomial_kernel(
            np.ones((9, 9)),
            np.ones((9, 9)),
            [GaussianBasisComponent(sigma=1.0, degree=0)],
            kernel_shape=(3, 3),
            backend="numba-cuda",
        )


def _fake_cupy(free_bytes: int):
    class Runtime:
        @staticmethod
        def memGetInfo() -> tuple[int, int]:
            return (free_bytes, 80 * 1024**3)

    class Cuda:
        runtime = Runtime()

    class FakeCupy:
        cuda = Cuda()

    return FakeCupy()


def test_spatial_model_gpu_design_chunk_size_is_memory_bounded() -> None:
    # A 21x21 kernel with 12 bases and 6/6/3 terms spans 75 columns. The
    # variance loop dominates the estimate: 5 * 441 + 12 + 6 + 6 + 32 = 2261
    # FP64 values, or 18088 bytes, per row.
    bytes_per_row = 18088
    model = dict(
        kernel_shape=(21, 21),
        basis_count=12,
        photometric_term_count=6,
        shape_term_count=6,
        background_term_count=3,
    )
    chunk = spatial_model_module._gpu_design_chunk_size

    scarce = chunk(_fake_cupy(64 * 1024**2), row_count=10_000_000, **model)
    ample = chunk(_fake_cupy(80 * 1024**3), row_count=10_000_000, **model)
    exhausted = chunk(_fake_cupy(1), row_count=10_000_000, **model)
    few_rows = chunk(_fake_cupy(80 * 1024**3), row_count=500, **model)
    capped = chunk(
        _fake_cupy(80 * 1024**3),
        row_count=10_000_000,
        kernel_shape=(3, 3),
        basis_count=1,
        photometric_term_count=1,
        shape_term_count=1,
        background_term_count=1,
    )
    # A 3x3 kernel with 6 bases and 10/10/10 terms spans 70 columns, so the
    # QR loop dominates: 2 * 9 + 6 + 10 + 10 + 10 + 8 * 70 + 32 = 646 FP64
    # values, or 5168 bytes, per row.
    column_heavy = chunk(
        _fake_cupy(80 * 1024**3),
        row_count=10_000_000,
        kernel_shape=(3, 3),
        basis_count=6,
        photometric_term_count=10,
        shape_term_count=10,
        background_term_count=10,
    )

    assert scarce == (64 * 1024**2 // 4) // bytes_per_row == 927
    assert (
        ample
        == spatial_model_module._GPU_SCRATCH_BUDGET_BYTES // bytes_per_row
        == 14840
    )
    assert scarce < ample < spatial_model_module._GPU_MAX_DESIGN_CHUNK_SIZE
    assert exhausted == 1
    assert few_rows == 500
    assert capped == spatial_model_module._GPU_MAX_DESIGN_CHUNK_SIZE
    assert (
        column_heavy
        == spatial_model_module._GPU_SCRATCH_BUDGET_BYTES // 5168
        == 51941
    )


def test_spatial_model_cupy_matches_rank_two_parent_domain_cpu_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cupy_device_or_skip()
    monkeypatch.setattr(
        spatial_model_module,
        "_gpu_design_chunk_size",
        lambda *args, **kwargs: 37,
    )
    shape = (38, 42)
    source_parent = _random_source((76, 84), seed=73)
    source = source_parent[::2, ::2]
    assert not source.flags.c_contiguous
    components = [GaussianBasisComponent(sigma=1.2, degree=2)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    photometric_coefficients = np.array([1.03, 0.02, -0.015])
    shape_coefficients = np.zeros((basis.shape[0] - 1, 6))
    shape_coefficients[0] = [0.12, 0.01, -0.008, 0.004, -0.003, 0.002]
    shape_coefficients[3] = [-0.08, -0.006, 0.009, -0.003, 0.002, -0.001]
    background_coefficients = np.array([0.04, -0.01, 0.006])
    domain = SpatialKernelDomain(
        array_origin_yx=(112, 518),
        normalization_bbox=(100, 180, 500, 596),
    )
    target = _spatial_target(
        source,
        basis,
        photometric_coefficients,
        shape_coefficients,
        background_coefficients,
        photometric_degree=1,
        shape_degree=2,
        background_degree=1,
        domain=domain,
    )
    variance = 0.4 + np.linspace(0.0, 0.3, source.size).reshape(shape)
    source[9, 10] = np.nan
    target[20, 22] = np.nan
    variance[25, 26] = np.nan
    config = SpatialGaussianPolynomialKernelConfig(
        shape_degree=2,
        photometric_degree=1,
        background_degree=1,
    )

    cpu = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        spatial_domain=domain,
        config=config,
        backend="cpu",
    )
    gpu = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=variance,
        spatial_domain=domain,
        config=config,
        backend="cupy",
    )

    assert np.linalg.matrix_rank(cpu.kernel_at_local(19, 21)) == 2
    _assert_cupy_matches_cpu(cpu, gpu)
    assert gpu.design_chunk_size == 37
    assert not gpu.fit_mask[9, 10]
    assert not gpu.fit_mask[20, 22]
    assert not gpu.fit_mask[25, 26]
    for parent_y, parent_x in ((115.5, 521.25), (131.5, 539.25), (148, 558)):
        np.testing.assert_allclose(
            gpu.kernel_at_parent(parent_y, parent_x),
            cpu.kernel_at_parent(parent_y, parent_x),
            rtol=2.0e-8,
            atol=2.0e-9,
        )


def test_spatial_model_cupy_matches_weighting_and_source_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cupy_device_or_skip()
    # 29 divides neither the 182 fit rows nor the 847 diagnostic rows, so
    # the QR, reconstruction, and variance-propagation loops all span
    # several chunks with a short final chunk.
    monkeypatch.setattr(
        spatial_model_module,
        "_gpu_design_chunk_size",
        lambda *args, **kwargs: 29,
    )
    shape = (34, 38)
    source = _random_source(shape, seed=79)
    components = [GaussianBasisComponent(sigma=1.1, degree=1)]
    basis, _ = build_gaussian_polynomial_basis(
        (7, 7),
        components,
        flux_conserve=True,
    )
    photometric_coefficients = np.array([1.02, 0.015, -0.01])
    shape_coefficients = np.array(
        [
            [0.03, 0.004, -0.003],
            [-0.02, -0.002, 0.003],
        ]
    )
    background_coefficients = np.array([0.05])
    noise = np.random.default_rng(83).normal(scale=0.01, size=shape)
    target = _spatial_target(
        source,
        basis,
        photometric_coefficients,
        shape_coefficients,
        background_coefficients,
        photometric_degree=1,
        shape_degree=1,
        background_degree=0,
    )
    target += noise
    target_variance = 0.3 + np.linspace(0.0, 0.2, source.size).reshape(shape)
    source_variance = 0.5 + np.linspace(0.0, 0.3, source.size).reshape(shape)
    source_variance[17, 19] = np.nan
    fit_mask = np.zeros(shape, dtype=bool)
    fit_mask[4:-4:2, 5:-5:2] = True
    positions = np.column_stack(np.nonzero(fit_mask))
    samples = SpatialGaussianPolynomialKernelFitSamples(
        positions_yx=positions[::-1],
        relative_precision=np.linspace(0.5, 2.0, positions.shape[0]),
    )
    config = SpatialGaussianPolynomialKernelConfig(
        shape_degree=1,
        photometric_degree=1,
        background_degree=0,
    )

    cpu = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=target_variance,
        source_variance=source_variance,
        fit_samples=samples,
        config=config,
        backend="cpu",
    )
    gpu = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        kernel_shape=(7, 7),
        variance=target_variance,
        source_variance=source_variance,
        fit_samples=samples,
        config=config,
        backend="cupy",
    )

    _assert_cupy_matches_cpu(cpu, gpu)
    assert gpu.design_chunk_size == 29
    assert gpu.fit_pixel_count == 182
    assert np.isfinite(gpu.propagated_source_variance).sum() == 847
    assert gpu.explicit_fit_samples is samples


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("shape_degree", -1, "non-negative"),
        ("photometric_degree", 1.5, "integer"),
        ("background_degree", True, "integer"),
        ("condition_limit", np.inf, "finite"),
        ("condition_limit", 1.0, "greater than one"),
    ],
)
def test_spatial_model_config_rejects_invalid_values(
    name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialGaussianPolynomialKernelConfig(**{name: value})


@pytest.mark.parametrize(
    "components",
    [
        [
            GaussianBasisComponent(sigma=1.2, degree=0),
            GaussianBasisComponent(sigma=1.2, degree=0),
        ],
        [
            GaussianBasisComponent(sigma=1.2, degree=2),
            GaussianBasisComponent(sigma=1.2, degree=0),
        ],
        [
            GaussianBasisComponent(sigma=2.5, degree=1),
            GaussianBasisComponent(sigma=2.5, degree=1),
        ],
    ],
)
def test_spatial_model_rejects_numerically_null_basis_kernels(
    components: list[GaussianBasisComponent],
) -> None:
    shape = (48, 52)
    source = _random_source(shape, seed=71)
    basis, _ = build_gaussian_polynomial_basis(
        (15, 15),
        components,
        flux_conserve=True,
    )
    target = fftconvolve(source, 1.1 * basis[0], mode="same") + 0.05
    norms = np.linalg.norm(basis.reshape(basis.shape[0], -1), axis=1)

    # The duplicate of the reference is rounding residue, not an exact zero.
    assert 0.0 < norms.min() < 1.0e-14 * norms.max()
    with pytest.raises(ValueError, match="numerically zero after the flux"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            target,
            components,
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=0,
                photometric_degree=0,
                background_degree=0,
            ),
        )
    with pytest.raises(ValueError, match="ill-conditioned"):
        solve_constant_kernel(
            source,
            target,
            components,
            flux_conserve=True,
            backend="cpu",
        )


def test_spatial_model_fits_near_duplicate_components() -> None:
    shape = (48, 52)
    source = _random_source(shape, seed=73)
    components = [
        GaussianBasisComponent(sigma=1.2, degree=0),
        GaussianBasisComponent(sigma=1.21, degree=0),
    ]
    basis, _ = build_gaussian_polynomial_basis(
        (15, 15),
        components,
        flux_conserve=True,
    )
    kernel = 1.1 * basis[0] + 0.02 * basis[1]
    target = fftconvolve(source, kernel, mode="same")

    result = solve_spatial_gaussian_polynomial_kernel(
        source,
        target,
        components,
        config=SpatialGaussianPolynomialKernelConfig(
            shape_degree=0,
            photometric_degree=0,
            background_degree=0,
        ),
    )

    assert result.condition_number < 10.0
    assert np.allclose(result.photometric_coefficients, [1.1], atol=1.0e-9)
    assert np.allclose(result.shape_coefficients, [[0.02]], atol=1.0e-9)


def test_solve_scaled_qr_rejects_null_and_non_finite_columns() -> None:
    upper = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    rhs = np.array([1.0, 2.0, 3.0])
    coefficients, condition_number = spatial_model_module._solve_scaled_qr(
        upper,
        rhs,
        condition_limit=1.0e12,
    )

    assert np.allclose(coefficients, np.linalg.solve(upper, rhs))
    assert np.isfinite(condition_number)

    null_column = upper.copy()
    null_column[:, 2] = 1.0e-12
    with pytest.raises(ValueError, match="numerically null column"):
        spatial_model_module._solve_scaled_qr(
            null_column,
            rhs,
            condition_limit=1.0e12,
        )
    with pytest.raises(ValueError, match="numerically null column"):
        spatial_model_module._solve_scaled_qr(
            np.zeros((3, 3)),
            rhs,
            condition_limit=1.0e12,
        )
    overflow = upper.copy()
    overflow[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite column norms"):
        spatial_model_module._solve_scaled_qr(
            overflow,
            rhs,
            condition_limit=1.0e12,
        )


def test_spatial_model_fails_closed_at_condition_limit() -> None:
    shape = (40, 44)
    source = _random_source(shape, seed=31)
    components = [GaussianBasisComponent(sigma=1.2, degree=1)]

    def solve(condition_limit: float):
        return solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            components,
            kernel_shape=(7, 7),
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=0,
                photometric_degree=0,
                background_degree=0,
                condition_limit=condition_limit,
            ),
        )

    baseline = solve(SpatialGaussianPolynomialKernelConfig().condition_limit)
    assert 1.0 < baseline.condition_number < 10.0
    accepted = solve(1.001 * baseline.condition_number)
    assert accepted.condition_number == baseline.condition_number
    with pytest.raises(ValueError, match="ill-conditioned"):
        solve(0.999 * baseline.condition_number)


@pytest.mark.parametrize("flux_scale", [1.0e-12, 1.0, 1.0e11])
def test_spatial_model_fails_closed_on_degenerate_designs(
    flux_scale: float,
) -> None:
    shape = (40, 44)
    components = [GaussianBasisComponent(sigma=1.2, degree=1)]

    # A constant source makes the photometric and background fields exactly
    # collinear, so the default condition limit must reject the model.
    constant_source = flux_scale * np.ones(shape)
    with pytest.raises(ValueError, match="ill-conditioned"):
        solve_spatial_gaussian_polynomial_kernel(
            constant_source,
            constant_source + 0.1 * flux_scale,
            [GaussianBasisComponent(sigma=1.2, degree=0)],
            kernel_shape=(7, 7),
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=0,
                photometric_degree=1,
                background_degree=1,
            ),
        )
    # With zero-sum shape bases the same source also yields rounding-level
    # shape columns, which the null-column guard rejects first.
    with pytest.raises(ValueError, match="empty or numerically null column"):
        solve_spatial_gaussian_polynomial_kernel(
            constant_source,
            constant_source + 0.1 * flux_scale,
            components,
            kernel_shape=(7, 7),
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=0,
                photometric_degree=0,
                background_degree=0,
            ),
        )
    # A zero source leaves every kernel column exactly empty.
    zero_source = np.zeros(shape)
    with pytest.raises(ValueError, match="empty or numerically null column"):
        solve_spatial_gaussian_polynomial_kernel(
            zero_source,
            zero_source + 0.1 * flux_scale,
            components,
            kernel_shape=(7, 7),
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=0,
                photometric_degree=0,
                background_degree=0,
            ),
            backend="cpu",
        )


def test_spatial_model_cupy_fails_closed_at_condition_limit() -> None:
    _cupy_device_or_skip()
    shape = (40, 44)
    source = _random_source(shape, seed=31)
    components = [GaussianBasisComponent(sigma=1.2, degree=1)]

    with pytest.raises(ValueError, match="ill-conditioned"):
        solve_spatial_gaussian_polynomial_kernel(
            source,
            source.copy(),
            components,
            kernel_shape=(7, 7),
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=0,
                photometric_degree=0,
                background_degree=0,
                condition_limit=1.01,
            ),
            backend="cupy",
        )
