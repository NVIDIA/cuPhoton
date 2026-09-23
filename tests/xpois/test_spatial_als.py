# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import cuphoton.xpois.spatial_als as spatial_als_module
from cuphoton.xpois.ois import GaussianBasisComponent, solve_separable_kernel
from cuphoton.xpois.spatial_als import (
    SpatialALSConfig,
    SpatialALSFitResult,
    solve_spatial_als,
)

_SPATIAL_TERMS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (2, 0),
)
_BACKGROUND_TERMS = ((0, 0), (0, 1), (1, 0))


@dataclass(frozen=True)
class _RecoveryCase:
    result: SpatialALSFitResult
    source: np.ndarray
    target: np.ndarray
    horizontal_reference: np.ndarray
    vertical_reference: np.ndarray
    horizontal_basis: np.ndarray
    vertical_basis: np.ndarray
    horizontal_coefficients: np.ndarray
    vertical_coefficients: np.ndarray
    flux_scale: float


def _chebyshev_design(
    shape: tuple[int, int],
    terms: tuple[tuple[int, int], ...],
) -> np.ndarray:
    y_grid, x_grid = np.mgrid[: shape[0], : shape[1]]
    normalized_x = 2.0 * x_grid / (shape[1] - 1) - 1.0
    normalized_y = 2.0 * y_grid / (shape[0] - 1) - 1.0
    maximum_degree = max(max(term) for term in terms)
    x_values = np.polynomial.chebyshev.chebvander(
        normalized_x.ravel(),
        maximum_degree,
    ).reshape(*shape, maximum_degree + 1)
    y_values = np.polynomial.chebyshev.chebvander(
        normalized_y.ravel(),
        maximum_degree,
    ).reshape(*shape, maximum_degree + 1)
    return np.stack(
        [
            x_values[..., degree_x] * y_values[..., degree_y]
            for degree_x, degree_y in terms
        ],
        axis=-1,
    )


def _line_model(
    length: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(length, dtype=np.float64) - length // 2
    gaussian = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    reference = gaussian / gaussian.sum()
    corrections = []
    for degree in (1, 2):
        correction = gaussian * coordinates**degree
        correction -= correction.sum() * reference
        corrections.append(correction / np.linalg.norm(correction))
    return reference, np.stack(corrections)


def _render_spatial_target(
    source: np.ndarray,
    horizontal_reference: np.ndarray,
    vertical_reference: np.ndarray,
    horizontal_basis: np.ndarray,
    vertical_basis: np.ndarray,
    horizontal_coefficients: np.ndarray,
    vertical_coefficients: np.ndarray,
    flux_scale: float,
    background_coefficients: np.ndarray,
) -> np.ndarray:
    spatial = _chebyshev_design(source.shape, _SPATIAL_TERMS)
    background = _chebyshev_design(source.shape, _BACKGROUND_TERMS)
    kernel_height = vertical_reference.size
    kernel_width = horizontal_reference.size
    margin_y = kernel_height // 2
    margin_x = kernel_width // 2
    target = np.zeros(source.shape, dtype=np.float64)
    for y in range(margin_y, source.shape[0] - margin_y):
        for x in range(margin_x, source.shape[1] - margin_x):
            horizontal = horizontal_reference + np.einsum(
                "is,s,iu->u",
                horizontal_coefficients,
                spatial[y, x],
                horizontal_basis,
                optimize=True,
            )
            vertical = flux_scale * vertical_reference + np.einsum(
                "is,s,iv->v",
                vertical_coefficients,
                spatial[y, x],
                vertical_basis,
                optimize=True,
            )
            kernel = np.outer(vertical, horizontal)
            patch = source[
                y - margin_y : y + margin_y + 1,
                x - margin_x : x + margin_x + 1,
            ]
            target[y, x] = (
                np.sum(patch * kernel[::-1, ::-1])
                + background[y, x] @ background_coefficients
            )
    return target


@pytest.fixture(scope="module")
def recovery_case() -> _RecoveryCase:
    image_shape = (31, 33)
    kernel_shape = (5, 5)
    sigma = 1.1
    generator = np.random.default_rng(20260902)
    source = generator.normal(size=image_shape)
    horizontal_reference, horizontal_basis = _line_model(
        kernel_shape[1],
        sigma,
    )
    vertical_reference, vertical_basis = _line_model(
        kernel_shape[0],
        sigma,
    )
    horizontal_coefficients = np.array(
        [
            [0.040, -0.025, 0.012, 0.018, -0.008, 0.005],
            [-0.030, 0.016, -0.009, 0.011, 0.006, -0.004],
        ]
    )
    vertical_coefficients = np.array(
        [
            [-0.025, 0.012, -0.006, 0.014, 0.005, 0.003],
            [0.020, -0.010, 0.007, -0.008, 0.004, 0.002],
        ]
    )
    flux_scale = 1.08
    target = _render_spatial_target(
        source,
        horizontal_reference,
        vertical_reference,
        horizontal_basis,
        vertical_basis,
        horizontal_coefficients,
        vertical_coefficients,
        flux_scale,
        np.array([0.070, 0.020, -0.015]),
    )
    result = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=sigma, degree=2)],
        kernel_shape=kernel_shape,
        config=SpatialALSConfig(
            spatial_degree=2,
            background_degree=1,
            max_iterations=40,
            tolerance=0.0,
            regularization=0.0,
            flux_conserve=True,
        ),
    )
    return _RecoveryCase(
        result=result,
        source=source,
        target=target,
        horizontal_reference=horizontal_reference,
        vertical_reference=vertical_reference,
        horizontal_basis=horizontal_basis,
        vertical_basis=vertical_basis,
        horizontal_coefficients=horizontal_coefficients,
        vertical_coefficients=vertical_coefficients,
        flux_scale=flux_scale,
    )


def _true_kernel_at(case: _RecoveryCase, y: int, x: int) -> np.ndarray:
    spatial = _chebyshev_design(case.source.shape, _SPATIAL_TERMS)[y, x]
    horizontal = case.horizontal_reference + np.einsum(
        "is,s,iu->u",
        case.horizontal_coefficients,
        spatial,
        case.horizontal_basis,
        optimize=True,
    )
    vertical = case.flux_scale * case.vertical_reference + np.einsum(
        "is,s,iv->v",
        case.vertical_coefficients,
        spatial,
        case.vertical_basis,
        optimize=True,
    )
    return np.outer(vertical, horizontal)


def test_spatial_als_recovers_nonconstant_degree_two_model(
    recovery_case: _RecoveryCase,
) -> None:
    case = recovery_case
    result = case.result
    interior = np.zeros(case.source.shape, dtype=bool)
    interior[2:-2, 2:-2] = True

    assert result.fit_pixel_count == int(interior.sum())
    assert result.unique_fit_pixel_count == int(interior.sum())
    assert np.sqrt(np.mean(result.residual[interior] ** 2)) < 1.0e-10
    assert np.allclose(
        result.horizontal_coefficients,
        case.horizontal_coefficients,
        atol=1.0e-10,
    )
    assert np.allclose(
        result.vertical_coefficients,
        case.vertical_coefficients,
        atol=1.0e-10,
    )
    assert result.flux_scale == pytest.approx(case.flux_scale, abs=1.0e-10)
    assert np.all(np.diff(result.objective_history) <= 0.0)
    assert np.isnan(result.matched[~interior]).all()
    assert np.isnan(result.residual[~interior]).all()

    for y, x in ((2, 2), (15, 16), (28, 30)):
        assert np.allclose(
            result.kernel_at(y, x),
            _true_kernel_at(case, y, x),
            atol=1.0e-10,
        )
    assert not np.allclose(
        result.kernel_at(2, 2),
        result.kernel_at(28, 30),
    )


def test_exact_fit_converges_at_machine_precision_floor(
    recovery_case: _RecoveryCase,
) -> None:
    result = recovery_case.result
    target_samples = recovery_case.target[result.fit_mask]
    accumulation_length = (
        result.vertical_reference.size * result.horizontal_reference.size
        + len(result.background_terms)
    )
    objective_floor = (
        np.finfo(np.float64).eps ** 2
        * accumulation_length**2
        * float(np.sum(target_samples**2))
        * max(result.condition_number, 1.0)
    )

    assert result.converged
    assert result.iterations < 40
    assert np.all(np.diff(result.objective_history) <= 0.0)
    assert result.objective_history[-1] <= objective_floor


def test_degree_zero_matches_separable_solver() -> None:
    from scipy.signal import fftconvolve

    generator = np.random.default_rng(825)
    source = generator.normal(size=(25, 27))
    axis = np.arange(-2, 3, dtype=float)
    gaussian = np.exp(-(axis**2) / (2.0 * 1.1**2))
    horizontal = gaussian * (1.0 + 0.09 * axis + 0.04 * axis**2)
    horizontal /= horizontal.sum()
    vertical = gaussian * (1.0 - 0.08 * axis + 0.02 * axis**2)
    vertical *= 1.3 / vertical.sum()
    y_grid, x_grid = np.indices(source.shape)
    target = (
        fftconvolve(source, np.outer(vertical, horizontal), mode="same")
        + 0.15
        + 0.003 * x_grid
        - 0.005 * y_grid
        + 0.002 * generator.normal(size=source.shape)
    )
    variance = 0.7 + 0.02 * x_grid + 0.01 * y_grid
    fit_mask = np.ones(source.shape, dtype=bool)
    fit_mask[7:10, 12:16] = False
    components = [GaussianBasisComponent(sigma=1.1, degree=2)]

    spatial = solve_spatial_als(
        source,
        target,
        components,
        kernel_shape=(5, 5),
        variance=variance,
        fit_mask=fit_mask,
        config=SpatialALSConfig(
            spatial_degree=0,
            background_degree=1,
            max_iterations=50,
            tolerance=1.0e-12,
            regularization=0.0,
            flux_conserve=False,
        ),
    )
    separable = solve_separable_kernel(
        source,
        target,
        components,
        kernel_shape=(5, 5),
        variance=variance,
        fit_mask=fit_mask,
        background_degree=1,
        max_iterations=50,
        tolerance=1.0e-12,
        flux_conserve=False,
    )

    assert spatial.converged and separable.converged
    np.testing.assert_array_equal(spatial.fit_mask, separable.fit_mask)
    for y, x in ((2, 2), (12, 13), (22, 24)):
        np.testing.assert_allclose(
            spatial.kernel_at(y, x), separable.kernel, rtol=0.0, atol=2e-9
        )
    for field in ("matched", "residual", "background"):
        np.testing.assert_allclose(
            getattr(spatial, field),
            getattr(separable, field),
            rtol=0.0,
            atol=2e-9,
            equal_nan=True,
        )
    assert spatial.chi2 == pytest.approx(separable.chi2, rel=1.0e-9)


def test_exact_fit_converges_with_a_single_sweep_budget() -> None:
    from scipy.signal import fftconvolve

    source = np.random.default_rng(551).normal(size=(15, 17))
    line = np.exp(-(np.arange(-2, 3, dtype=float) ** 2) / 2)
    line /= line.sum()
    target = (
        1.2 * fftconvolve(source, np.outer(line, line), mode="same") + 0.15
    )

    result = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=1.0, degree=0)],
        kernel_shape=(5, 5),
        config=SpatialALSConfig(
            spatial_degree=0,
            background_degree=0,
            flux_conserve=True,
            max_iterations=1,
        ),
    )

    assert np.max(np.abs(result.residual[result.fit_mask])) < 1.0e-14
    assert result.iterations == 1
    assert result.converged


@pytest.mark.parametrize("scale", [1.0e-18, 1.0e-8, 1.0e8])
def test_convergence_floor_preserves_recovery_across_image_units(
    recovery_case: _RecoveryCase, scale: float
) -> None:
    case = recovery_case
    result = solve_spatial_als(
        case.source * scale,
        case.target * scale,
        [GaussianBasisComponent(sigma=1.1, degree=2)],
        kernel_shape=(5, 5),
        config=SpatialALSConfig(
            spatial_degree=2,
            background_degree=1,
            max_iterations=40,
            tolerance=0.0,
            regularization=0.0,
            flux_conserve=True,
        ),
    )

    assert result.converged
    assert result.iterations > 1
    assert np.max(np.abs(result.residual[result.fit_mask])) / scale < 1.0e-10
    assert np.allclose(
        result.horizontal_coefficients,
        case.horizontal_coefficients,
        atol=1.0e-10,
    )


def test_objective_increase_never_reports_convergence() -> None:
    assert not spatial_als_module._objective_has_converged(
        1.0,
        1.0 + np.finfo(np.float64).eps,
        tolerance=1.0,
        objective_floor=1.0e-30,
    )


def test_overflowed_target_energy_does_not_end_the_first_sweep(
    recovery_case: _RecoveryCase,
) -> None:
    case = recovery_case
    with np.errstate(over="ignore"):
        result = solve_spatial_als(
            case.source * 1.0e155,
            case.target * 1.0e155,
            [GaussianBasisComponent(sigma=1.0, degree=2)],
            kernel_shape=(5, 5),
            variance=np.full(case.source.shape, 1.0e308),
            config=SpatialALSConfig(
                spatial_degree=2,
                background_degree=1,
                max_iterations=2,
                tolerance=0.0,
                flux_conserve=True,
            ),
        )

    assert result.iterations == 2
    assert not result.converged
    assert np.isfinite(result.objective_history).all()
    assert result.objective_history[-1] < result.objective_history[0]


@pytest.mark.parametrize("prior", [np.inf, np.nan])
def test_nonfinite_prior_objective_does_not_establish_convergence(
    prior: float,
) -> None:
    assert not spatial_als_module._objective_has_converged(
        prior,
        1.0,
        tolerance=1.0,
        objective_floor=0.0,
    )


def test_objective_jitter_below_numerical_floor_reports_convergence() -> None:
    assert spatial_als_module._objective_has_converged(
        1.0e-28,
        2.0e-28,
        tolerance=0.0,
        objective_floor=1.0e-27,
    )


def test_flux_conserving_kernel_sum_is_position_independent(
    recovery_case: _RecoveryCase,
) -> None:
    result = recovery_case.result
    for y, x in ((0, 0), (2, 30), (15, 16), (30, 32)):
        assert result.horizontal_at(y, x).sum() == pytest.approx(1.0)
        assert result.vertical_at(y, x).sum() == pytest.approx(
            result.flux_scale
        )
        assert result.kernel_at(y, x).sum() == pytest.approx(
            result.flux_scale
        )


def _render_constant_target(
    source: np.ndarray,
    kernel: np.ndarray,
    background: float,
) -> np.ndarray:
    margin_y = kernel.shape[0] // 2
    margin_x = kernel.shape[1] // 2
    target = np.zeros_like(source)
    for y in range(margin_y, source.shape[0] - margin_y):
        for x in range(margin_x, source.shape[1] - margin_x):
            patch = source[
                y - margin_y : y + margin_y + 1,
                x - margin_x : x + margin_x + 1,
            ]
            target[y, x] = np.sum(patch * kernel[::-1, ::-1]) + background
    return target


def _exact_constant_pair(
    shape: tuple[int, int],
    kernel_shape: tuple[int, int],
    *,
    sigma: float,
    seed: int,
    sky: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    source = amplitude * generator.normal(size=shape)
    line, _ = _line_model(kernel_shape[0], sigma)
    kernel = np.outer(line, line)
    margin_y = kernel_shape[0] // 2
    margin_x = kernel_shape[1] // 2
    patches = np.lib.stride_tricks.sliding_window_view(source, kernel_shape)
    target = np.zeros_like(source)
    target[margin_y:-margin_y, margin_x:-margin_x] = (
        np.einsum("yxvu,vu->yx", patches, kernel[::-1, ::-1], optimize=True)
        + amplitude * sky
    )
    return source, target


@pytest.mark.parametrize("amplitude", [1.0, 100.0])
def test_exact_fit_convergence_is_independent_of_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
    amplitude: float,
) -> None:
    source, target = _exact_constant_pair(
        (41, 43),
        (5, 5),
        sigma=0.9,
        seed=0,
        sky=0.25,
        amplitude=amplitude,
    )
    outcomes = []
    for chunk_size in (4096, 5):
        monkeypatch.setattr(
            spatial_als_module,
            "_DESIGN_CHUNK_SIZE",
            chunk_size,
        )
        result = solve_spatial_als(
            source,
            target,
            [GaussianBasisComponent(sigma=0.9, degree=1)],
            kernel_shape=(5, 5),
            config=SpatialALSConfig(spatial_degree=2),
        )
        outcomes.append((result.converged, result.iterations))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0]
    assert outcomes[0][1] <= 3


_CLI_COMPONENTS = (
    GaussianBasisComponent(sigma=1.5, degree=2),
    GaussianBasisComponent(sigma=3.0, degree=1),
    GaussianBasisComponent(sigma=6.0, degree=0),
)


def _noisy_varying_width_pair(
    shape: tuple[int, int],
    kernel_shape: tuple[int, int],
    *,
    flux: float,
    noise: float = 0.05,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact sources matched by a separable kernel whose widths vary."""

    generator = np.random.default_rng(seed)
    height, width = shape
    y_grid, x_grid = np.mgrid[:height, :width]
    source = np.zeros(shape, dtype=np.float64)
    for _ in range(40):
        center_y = generator.uniform(4.0, height - 5.0)
        center_x = generator.uniform(4.0, width - 5.0)
        amplitude = generator.uniform(20.0, 200.0)
        source += amplitude * np.exp(
            -((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
            / (2.0 * 1.3**2)
        )
    kernel_height, kernel_width = kernel_shape
    margin_y = kernel_height // 2
    margin_x = kernel_width // 2
    normalized_y = 2.0 * y_grid / (height - 1) - 1.0
    normalized_x = 2.0 * x_grid / (width - 1) - 1.0
    sigma_y = 1.4 + 0.35 * normalized_y + 0.15 * normalized_x
    sigma_x = 1.5 - 0.3 * normalized_x + 0.2 * normalized_y * normalized_x
    v_coords = np.arange(kernel_height, dtype=np.float64) - margin_y
    u_coords = np.arange(kernel_width, dtype=np.float64) - margin_x
    vertical = np.exp(-(v_coords**2) / (2.0 * sigma_y[..., None] ** 2))
    horizontal = np.exp(-(u_coords**2) / (2.0 * sigma_x[..., None] ** 2))
    vertical /= vertical.sum(axis=-1, keepdims=True)
    horizontal /= horizontal.sum(axis=-1, keepdims=True)
    kernels = flux * vertical[..., :, None] * horizontal[..., None, :]
    patches = np.lib.stride_tricks.sliding_window_view(source, kernel_shape)
    interior = (
        slice(margin_y, height - margin_y),
        slice(margin_x, width - margin_x),
    )
    target = np.zeros(shape, dtype=np.float64)
    target[interior] = np.einsum(
        "yxvu,yxvu->yx",
        patches,
        kernels[interior][..., ::-1, ::-1],
        optimize=True,
    )
    target += generator.normal(scale=noise, size=shape)
    return source, target, np.full(shape, noise**2)


def test_flux_ratio_fit_is_near_optimal_within_default_sweeps() -> None:
    source, target, variance = _noisy_varying_width_pair(
        (64, 60),
        (9, 9),
        flux=5.0,
    )
    default_budget = solve_spatial_als(
        source,
        target,
        _CLI_COMPONENTS,
        kernel_shape=(9, 9),
        variance=variance,
        config=SpatialALSConfig(flux_conserve=True),
    )
    converged = solve_spatial_als(
        source,
        target,
        _CLI_COMPONENTS,
        kernel_shape=(9, 9),
        variance=variance,
        config=SpatialALSConfig(
            max_iterations=80,
            tolerance=1.0e-10,
            flux_conserve=True,
        ),
    )

    assert converged.converged
    assert converged.flux_scale == pytest.approx(5.0, rel=1.0e-3)
    assert default_budget.converged
    assert default_budget.iterations < SpatialALSConfig().max_iterations
    assert default_budget.flux_scale == pytest.approx(5.0, rel=1.0e-2)
    assert default_budget.objective_history[-1] <= (
        1.01 * converged.objective_history[-1]
    )
    assert np.all(np.diff(default_budget.objective_history) <= 0.0)


@pytest.mark.parametrize("scale", [1.0e-8, 1.0e8])
def test_spatial_als_condition_gate_is_unit_invariant(scale: float) -> None:
    source, target = _exact_constant_pair(
        (21, 23),
        (5, 5),
        sigma=0.9,
        seed=5,
        sky=0.25,
    )
    components = [GaussianBasisComponent(sigma=0.9, degree=1)]
    config = SpatialALSConfig(spatial_degree=1)
    unit = solve_spatial_als(
        source,
        target,
        components,
        kernel_shape=(5, 5),
        config=config,
    )
    scaled = solve_spatial_als(
        source * scale,
        target * scale,
        components,
        kernel_shape=(5, 5),
        variance=np.full(source.shape, scale**2),
        config=config,
    )

    assert unit.converged and scaled.converged
    assert unit.condition_number < 1.0e3
    assert scaled.condition_number == pytest.approx(
        unit.condition_number,
        rel=1.0e-6,
    )
    assert scaled.flux_scale == pytest.approx(unit.flux_scale, abs=1.0e-10)
    assert np.allclose(
        scaled.horizontal_coefficients,
        unit.horizontal_coefficients,
        atol=1.0e-10,
    )
    assert np.allclose(
        scaled.vertical_coefficients,
        unit.vertical_coefficients,
        atol=1.0e-10,
    )
    assert np.allclose(
        scaled.background_coefficients,
        scale * unit.background_coefficients,
        rtol=1.0e-10,
    )


def test_repeated_sample_positions_act_as_multiplicity_weights() -> None:
    shape = (17, 19)
    generator = np.random.default_rng(7)
    source = generator.normal(size=shape)
    reference, _ = _line_model(5, 1.2)
    target = _render_constant_target(
        source, np.outer(reference, reference), 0.1
    )
    repeated_position = np.array([8, 9])
    target[tuple(repeated_position)] += 3.0
    unique_positions = np.array(
        [
            (y, x)
            for y in range(2, shape[0] - 2)
            for x in range(2, shape[1] - 2)
        ]
    )
    multiplicity = 8
    repeated_positions = np.concatenate(
        (
            unique_positions,
            np.repeat(
                repeated_position[None, :],
                multiplicity - 1,
                axis=0,
            ),
        )
    )
    config = SpatialALSConfig(
        spatial_degree=0,
        background_degree=0,
        max_iterations=12,
        tolerance=1.0e-12,
        regularization=1.0e-8,
    )
    repeated = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=1.2, degree=1)],
        kernel_shape=(5, 5),
        sample_positions=repeated_positions,
        config=config,
    )
    equivalent_variance = np.ones(shape)
    equivalent_variance[tuple(repeated_position)] = 1.0 / multiplicity
    weighted = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=1.2, degree=1)],
        kernel_shape=(5, 5),
        variance=equivalent_variance,
        sample_positions=unique_positions,
        config=config,
    )
    deduplicated = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=1.2, degree=1)],
        kernel_shape=(5, 5),
        sample_positions=unique_positions,
        config=config,
    )

    assert repeated.fit_pixel_count == unique_positions.shape[0] + 7
    assert repeated.unique_fit_pixel_count == unique_positions.shape[0]
    assert np.allclose(
        repeated.kernel_at(*repeated_position),
        weighted.kernel_at(*repeated_position),
        atol=1.0e-12,
    )
    assert np.allclose(
        repeated.background_coefficients,
        weighted.background_coefficients,
        atol=1.0e-12,
    )
    assert (
        abs(
            repeated.matched[tuple(repeated_position)]
            - deduplicated.matched[tuple(repeated_position)]
        )
        > 0.1
    )


def test_repeated_positions_do_not_satisfy_the_rank_guard() -> None:
    source, target = _exact_constant_pair(
        (21, 23),
        (5, 5),
        sigma=0.9,
        seed=5,
        sky=0.25,
    )
    positions = np.tile(np.array([[7, 7], [8, 9], [12, 3]]), (70, 1))

    with pytest.raises(
        ValueError,
        match=r"underdetermined.*3 unique pixels \(210 sample rows\)",
    ):
        solve_spatial_als(
            source,
            target,
            [GaussianBasisComponent(sigma=0.9, degree=1)],
            kernel_shape=(5, 5),
            sample_positions=positions,
            config=SpatialALSConfig(spatial_degree=1),
        )


def test_spatial_als_builds_designs_and_patches_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_size = 7
    shape = (13, 15)
    generator = np.random.default_rng(17)
    source = generator.normal(size=shape)
    reference, _ = _line_model(3, 1.2)
    target = _render_constant_target(
        source,
        np.outer(reference, reference),
        0.03,
    )
    design_sizes: list[int] = []
    patch_sizes: list[int] = []
    original_design = spatial_als_module._chebyshev_design
    original_contract = spatial_als_module._contract_patches

    def tracked_design(
        sample_y: np.ndarray,
        sample_x: np.ndarray,
        image_shape: tuple[int, int],
        terms: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        design_sizes.append(sample_y.size)
        assert sample_y.size <= chunk_size
        return original_design(sample_y, sample_x, image_shape, terms)

    def tracked_contract(
        patches: np.ndarray,
        vertical: np.ndarray,
        horizontal: np.ndarray,
    ) -> np.ndarray:
        patch_sizes.append(patches.shape[0])
        assert patches.shape[0] <= chunk_size
        return original_contract(patches, vertical, horizontal)

    monkeypatch.setattr(spatial_als_module, "_DESIGN_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(
        spatial_als_module,
        "_chebyshev_design",
        tracked_design,
    )
    monkeypatch.setattr(
        spatial_als_module,
        "_contract_patches",
        tracked_contract,
    )

    result = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=1.2, degree=0)],
        kernel_shape=(3, 3),
        config=SpatialALSConfig(
            spatial_degree=0,
            background_degree=0,
            max_iterations=4,
        ),
    )

    assert np.isfinite(result.chi2)
    assert max(design_sizes) == chunk_size
    assert max(patch_sizes) == chunk_size


def test_spatial_als_accepts_noncontiguous_image_cutouts() -> None:
    generator = np.random.default_rng(29)
    source_parent = generator.normal(size=(26, 30))
    source = source_parent[::2, 1::2]
    reference, _ = _line_model(3, 1.2)
    target_parent = np.zeros((26, 30), dtype=np.float64)
    target = target_parent[::2, 1::2]
    target[...] = _render_constant_target(
        source,
        np.outer(reference, reference),
        0.02,
    )
    variance_parent = np.ones((26, 30), dtype=np.float64)
    variance = variance_parent[::2, 1::2]

    assert not source.flags.c_contiguous
    assert not target.flags.c_contiguous
    assert not variance.flags.c_contiguous

    result = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=1.2, degree=0)],
        kernel_shape=(3, 3),
        variance=variance,
        config=SpatialALSConfig(
            spatial_degree=0,
            background_degree=0,
            max_iterations=4,
        ),
    )

    assert result.converged
    assert np.sqrt(np.mean(result.residual[result.fit_mask] ** 2)) < 1.0e-12


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("spatial_degree", 1.5, "integer"),
        ("background_degree", True, "integer"),
        ("max_iterations", 0, "positive"),
        ("tolerance", np.nan, "finite"),
        ("regularization", np.inf, "finite"),
        ("flux_conserve", 1, "boolean"),
    ],
)
def test_spatial_als_config_rejects_invalid_values(
    keyword: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialALSConfig(**{keyword: value})


def test_spatial_als_config_defaults_to_nonconserving_flux() -> None:
    assert not SpatialALSConfig().flux_conserve


def test_spatial_als_accepts_reference_keyword_like_sibling_solvers() -> None:
    generator = np.random.default_rng(31)
    source = generator.normal(size=(11, 13))
    reference, _ = _line_model(3, 1.2)
    target = _render_constant_target(
        source,
        np.outer(reference, reference),
        0.05,
    )
    components = [GaussianBasisComponent(sigma=1.2, degree=0)]
    config = SpatialALSConfig(spatial_degree=0, max_iterations=4)

    result = solve_spatial_als(
        reference=source,
        target=target,
        components=components,
        kernel_shape=(3, 3),
        config=config,
    )

    assert np.sqrt(np.mean(result.residual[result.fit_mask] ** 2)) < 1e-12


def test_spatial_als_rejects_invalid_array_contracts() -> None:
    source = np.ones((9, 9))
    target = source.copy()
    components = [GaussianBasisComponent(sigma=1.2, degree=0)]

    with pytest.raises(ValueError, match="share the same shape"):
        solve_spatial_als(source, target[:-1], components)
    with pytest.raises(ValueError, match="odd dimensions"):
        solve_spatial_als(source, target, components, kernel_shape=(4, 5))
    with pytest.raises(ValueError, match="dimensions must be integers"):
        solve_spatial_als(source, target, components, kernel_shape=(3.0, 3))
    with pytest.raises(ValueError, match="strictly positive"):
        solve_spatial_als(
            source,
            target,
            components,
            kernel_shape=(3, 3),
            variance=np.zeros_like(target),
        )
    with pytest.raises(
        ValueError, match="either fit_mask or sample_positions"
    ):
        solve_spatial_als(
            source,
            target,
            components,
            kernel_shape=(3, 3),
            fit_mask=np.ones_like(target, dtype=bool),
            sample_positions=np.array([[4, 4]]),
        )
    with pytest.raises(ValueError, match="integer values"):
        solve_spatial_als(
            source,
            target,
            components,
            kernel_shape=(3, 3),
            sample_positions=np.array([[4.5, 4.0]]),
        )


def test_binary_numeric_fit_mask_selects_interior_pixels() -> None:
    source, target = _exact_constant_pair(
        (21, 23),
        (5, 5),
        sigma=0.9,
        seed=5,
        sky=0.25,
    )
    generator = np.random.default_rng(3)
    mask = (generator.uniform(size=source.shape) < 0.6).astype(np.int8)
    interior = np.zeros(source.shape, dtype=bool)
    interior[2:-2, 2:-2] = True
    expected = mask.astype(bool) & interior

    result = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=0.9, degree=1)],
        kernel_shape=(5, 5),
        fit_mask=mask,
        config=SpatialALSConfig(spatial_degree=1),
    )

    assert result.fit_pixel_count == int(expected.sum())
    assert result.unique_fit_pixel_count == int(expected.sum())
    assert np.array_equal(result.fit_mask, expected)
    assert np.sqrt(np.mean(result.residual[expected] ** 2)) < 1.0e-12
    assert np.isfinite(result.matched[interior]).all()
    assert np.isnan(result.matched[~interior]).all()


def test_automatic_selection_omits_invalid_pixels() -> None:
    source, target = _exact_constant_pair(
        (21, 23),
        (5, 5),
        sigma=0.9,
        seed=5,
        sky=0.25,
    )
    variance = np.ones(source.shape)
    source[6, 7] = np.nan
    target[12, 15] = np.nan
    variance[15, 4] = np.nan
    interior = np.zeros(source.shape, dtype=bool)
    interior[2:-2, 2:-2] = True
    expected = interior.copy()
    expected[4:9, 5:10] = False
    expected[12, 15] = False
    expected[15, 4] = False

    result = solve_spatial_als(
        source,
        target,
        [GaussianBasisComponent(sigma=0.9, degree=1)],
        kernel_shape=(5, 5),
        variance=variance,
        config=SpatialALSConfig(spatial_degree=1),
    )

    assert np.array_equal(result.fit_mask, expected)
    assert result.fit_pixel_count == int(expected.sum())
    assert np.isnan(result.matched[4:9, 5:10]).all()
    assert np.isnan(result.matched[12, 15])
    assert np.isnan(result.matched[15, 4])
    assert np.isnan(result.residual[~expected]).all()
    assert np.isfinite(result.matched[expected]).all()
    assert np.sqrt(np.mean(result.residual[expected] ** 2)) < 1.0e-12


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.array([[4.0, np.nan]]), "NaN or inf"),
        (np.array([[0, 4]]), "inside the valid kernel interior"),
        (np.array([[4, 8]]), "inside the valid kernel interior"),
        (np.array([4, 4]), r"shape \(row, 2\)"),
        (np.zeros((0, 2), dtype=np.int64), "must not be empty"),
        (np.array([[True, True]]), "numeric values"),
    ],
)
def test_spatial_als_rejects_malformed_sample_positions(
    positions: np.ndarray,
    message: str,
) -> None:
    source = np.ones((9, 9))
    target = source.copy()

    with pytest.raises(ValueError, match=message):
        solve_spatial_als(
            source,
            target,
            [GaussianBasisComponent(sigma=1.2, degree=0)],
            kernel_shape=(3, 3),
            sample_positions=positions,
            config=SpatialALSConfig(spatial_degree=0),
        )


def test_non_conserving_spatial_model_is_recovered_exactly() -> None:
    image_shape = (31, 33)
    kernel_shape = (5, 5)
    components = [GaussianBasisComponent(sigma=1.1, degree=2)]
    generator = np.random.default_rng(20260902)
    source = generator.normal(size=image_shape)
    raw_basis = spatial_als_module._build_line_basis(5, components)
    reference, basis = spatial_als_module._prepare_line_basis(
        raw_basis,
        flux_conserve=False,
    )
    horizontal_coefficients = np.array(
        [
            [0.040, -0.025, 0.012, 0.018, -0.008, 0.005],
            [-0.030, 0.016, -0.009, 0.011, 0.006, -0.004],
        ]
    )
    vertical_coefficients = np.array(
        [
            [-0.025, 0.012, -0.006, 0.014, 0.005, 0.003],
            [0.020, -0.010, 0.007, -0.008, 0.004, 0.002],
        ]
    )
    flux_scale = 1.08
    target = _render_spatial_target(
        source,
        reference,
        reference,
        basis,
        basis,
        horizontal_coefficients,
        vertical_coefficients,
        flux_scale,
        np.array([0.070, 0.020, -0.015]),
    )

    result = solve_spatial_als(
        source,
        target,
        components,
        kernel_shape=kernel_shape,
        config=SpatialALSConfig(
            spatial_degree=2,
            background_degree=1,
            max_iterations=80,
            tolerance=0.0,
            regularization=0.0,
            flux_conserve=False,
        ),
    )

    assert not result.flux_conserve
    assert result.converged
    assert np.sqrt(np.mean(result.residual[result.fit_mask] ** 2)) < 1e-10
    assert np.allclose(
        result.horizontal_coefficients,
        horizontal_coefficients,
        atol=1.0e-8,
    )
    assert np.allclose(
        result.vertical_coefficients,
        vertical_coefficients,
        atol=1.0e-8,
    )
    assert result.flux_scale == pytest.approx(flux_scale, abs=1.0e-8)
    kernel_sums = [
        result.kernel_at(y, x).sum() for y, x in ((2, 2), (15, 16), (28, 30))
    ]
    assert not np.allclose(kernel_sums, kernel_sums[0])


@pytest.mark.parametrize("invalid_input", ["source", "target", "variance"])
def test_spatial_als_rejects_invalid_explicit_sample_positions(
    invalid_input: str,
) -> None:
    source = np.ones((9, 9))
    target = source.copy()
    variance = np.ones_like(target)
    if invalid_input == "source":
        source[3, 3] = np.nan
    elif invalid_input == "target":
        target[4, 4] = np.nan
    else:
        variance[4, 4] = np.nan

    with pytest.raises(
        ValueError,
        match=(
            "finite target and variance values with finite source footprints"
        ),
    ):
        solve_spatial_als(
            source,
            target,
            [GaussianBasisComponent(sigma=1.2, degree=0)],
            kernel_shape=(3, 3),
            variance=variance,
            sample_positions=np.array([[4, 4]]),
            config=SpatialALSConfig(spatial_degree=0),
        )


def test_spatial_als_rejects_invalid_or_underdetermined_basis() -> None:
    image = np.ones((9, 9))

    with pytest.raises(ValueError, match="sigma must be positive"):
        solve_spatial_als(
            image,
            image,
            [GaussianBasisComponent(sigma=np.nan, degree=0)],
            kernel_shape=(3, 3),
        )
    with pytest.raises(ValueError, match="basis degree must be an integer"):
        solve_spatial_als(
            image,
            image,
            [GaussianBasisComponent(sigma=1.2, degree=1.5)],
            kernel_shape=(3, 3),
        )
    with pytest.raises(ValueError, match="underdetermined"):
        solve_spatial_als(
            np.ones((5, 5)),
            np.ones((5, 5)),
            [GaussianBasisComponent(sigma=1.2, degree=0)],
            kernel_shape=(5, 5),
            config=SpatialALSConfig(spatial_degree=0),
        )


def test_spatial_als_rejects_rank_deficient_line_bases() -> None:
    image = np.ones((9, 9))
    components = [
        GaussianBasisComponent(sigma=1.2, degree=0),
        GaussianBasisComponent(sigma=1.2, degree=0),
    ]

    with pytest.raises(ValueError, match="rank-deficient"):
        solve_spatial_als(
            image,
            image,
            components,
            kernel_shape=(3, 3),
            config=SpatialALSConfig(
                spatial_degree=0,
                regularization=1.0e6,
            ),
        )


@pytest.mark.parametrize(
    ("components", "kernel_shape", "message"),
    [
        (
            [GaussianBasisComponent(sigma=1.2, degree=4)],
            (3, 3),
            "5 functions but the kernel width is 3",
        ),
        (
            list(_CLI_COMPONENTS),
            (5, 5),
            "6 functions but the kernel width is 5",
        ),
        (
            list(_CLI_COMPONENTS),
            (5, 7),
            "6 functions but the kernel height is 5",
        ),
    ],
)
def test_spatial_als_rejects_line_bases_longer_than_the_kernel_axis(
    components: list[GaussianBasisComponent],
    kernel_shape: tuple[int, int],
    message: str,
) -> None:
    image = np.ones((21, 21))

    with pytest.raises(ValueError, match=message):
        solve_spatial_als(image, image, components, kernel_shape=kernel_shape)


def test_kernel_at_rejects_coordinates_outside_image(
    recovery_case: _RecoveryCase,
) -> None:
    with pytest.raises(ValueError, match="outside the image"):
        recovery_case.result.kernel_at(-1, 0)
