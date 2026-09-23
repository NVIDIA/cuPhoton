# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Held-out model-selection checks for spatial kernel families."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve

from cuphoton.xpois import (
    GaussianBasisComponent,
    SpatialALSConfig,
    build_gaussian_polynomial_basis,
    solve_constant_kernel,
    solve_spatial_als,
)
from cuphoton.xpois.spatial_gaussian_polynomial import (
    SpatialGaussianPolynomialKernelConfig,
    solve_spatial_gaussian_polynomial_kernel,
)

_IMAGE_SHAPE = (96, 100)
_KERNEL_SHAPE = (9, 9)
_NOISE_SIGMA = 0.05
_REALIZATION_SEEDS = tuple(range(20261000, 20261005))
_MODEL_NAMES = ("constant", "spatial_als", "gaussian_polynomial")
_TRUTH_NAMES = ("constant", "spatial_rank_one", "spatial_rank_two")


def _blocked_masks() -> tuple[np.ndarray, np.ndarray]:
    fit = np.zeros(_IMAGE_SHAPE, dtype=bool)
    evaluate = np.zeros(_IMAGE_SHAPE, dtype=bool)
    block_step = 20
    footprint_gap = _KERNEL_SHAPE[0] - 1
    border = 5

    for block_y, y0 in enumerate(
        range(border, _IMAGE_SHAPE[0] - border, block_step)
    ):
        for block_x, x0 in enumerate(
            range(border, _IMAGE_SHAPE[1] - border, block_step)
        ):
            y1 = min(
                y0 + block_step - footprint_gap,
                _IMAGE_SHAPE[0] - border,
            )
            x1 = min(
                x0 + block_step - footprint_gap,
                _IMAGE_SHAPE[1] - border,
            )
            destination = evaluate if (block_y + block_x) % 3 == 0 else fit
            destination[y0:y1, x0:x1] = True

    return fit, evaluate


def _basis_index(terms, u_degree: int, v_degree: int) -> int:
    return next(
        index
        for index, term in enumerate(terms)
        if (term.poly_u_degree, term.poly_v_degree) == (u_degree, v_degree)
    )


def _truth_images(
    source: np.ndarray,
    basis: np.ndarray,
    terms,
) -> dict[str, np.ndarray]:
    convolved = np.stack(
        [fftconvolve(source, kernel, mode="same") for kernel in basis]
    )
    normalized_y = np.linspace(-1.0, 1.0, _IMAGE_SHAPE[0])[:, None]
    normalized_x = np.linspace(-1.0, 1.0, _IMAGE_SHAPE[1])[None, :]
    u_index = _basis_index(terms, 1, 0)
    v_index = _basis_index(terms, 0, 1)
    uv_index = _basis_index(terms, 1, 1)
    first_field = 0.10 + 0.04 * normalized_x + 0.02 * normalized_y
    second_field = -0.08 + 0.03 * normalized_y - 0.02 * normalized_x

    assert (
        np.linalg.matrix_rank(
            basis[0] + 0.10 * basis[u_index],
            tol=1.0e-12,
        )
        == 1
    )
    assert (
        np.linalg.matrix_rank(
            basis[0] + 0.10 * basis[v_index] - 0.08 * basis[uv_index],
            tol=1.0e-12,
        )
        == 2
    )

    return {
        "constant": convolved[0] + 0.03,
        "spatial_rank_one": convolved[0]
        + first_field * convolved[u_index]
        + 0.03,
        "spatial_rank_two": convolved[0]
        + first_field * convolved[v_index]
        + second_field * convolved[uv_index]
        + 0.03,
    }


def _held_out_mse(
    source: np.ndarray,
    target: np.ndarray,
    variance: np.ndarray,
    fit_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    components: tuple[GaussianBasisComponent, ...],
) -> dict[str, float]:
    results = {
        "constant": solve_constant_kernel(
            source,
            target,
            components,
            kernel_shape=_KERNEL_SHAPE,
            variance=variance,
            fit_mask=fit_mask,
            background_degree=0,
            flux_conserve=True,
            backend="cpu",
        ),
        "spatial_als": solve_spatial_als(
            source,
            target,
            components,
            kernel_shape=_KERNEL_SHAPE,
            variance=variance,
            fit_mask=fit_mask,
            config=SpatialALSConfig(
                spatial_degree=1,
                background_degree=0,
                max_iterations=30,
                tolerance=1.0e-10,
                regularization=0.0,
                flux_conserve=True,
            ),
            backend="cpu",
        ),
        "gaussian_polynomial": solve_spatial_gaussian_polynomial_kernel(
            source,
            target,
            components,
            kernel_shape=_KERNEL_SHAPE,
            variance=variance,
            fit_mask=fit_mask,
            config=SpatialGaussianPolynomialKernelConfig(
                shape_degree=1,
                photometric_degree=0,
                background_degree=0,
            ),
            backend="cpu",
        ),
    }

    scores: dict[str, float] = {}
    for name, result in results.items():
        np.testing.assert_array_equal(result.fit_mask, fit_mask)
        residual = result.residual[evaluation_mask]
        assert np.all(np.isfinite(residual))
        scores[name] = float(np.mean(residual**2))
    return scores


def test_spatially_blocked_holdout_selects_model_family_by_truth() -> None:
    """Select added structure only when held-out error supports it.

    The constructed cases exercise one shared rank-one mode and one
    Gaussian-polynomial rank-two mode; they do not imply family nesting.
    """

    fit_mask, evaluation_mask = _blocked_masks()
    assert not np.any(fit_mask & evaluation_mask)

    # Two 9x9 source footprints overlap exactly when their output centers are
    # at most eight pixels apart along both axes. The 17x17 maximum filter
    # therefore proves that no fit and evaluation rows share source pixels.
    nearby_fit_center = maximum_filter(
        fit_mask,
        size=2 * (_KERNEL_SHAPE[0] - 1) + 1,
        mode="constant",
        cval=0,
    )
    assert not np.any(evaluation_mask & nearby_fit_center)

    components = (GaussianBasisComponent(sigma=1.3, degree=2),)
    basis, terms = build_gaussian_polynomial_basis(
        _KERNEL_SHAPE,
        components,
        flux_conserve=True,
    )
    variance = np.full(_IMAGE_SHAPE, _NOISE_SIGMA**2)
    scores = {
        truth: {model: [] for model in _MODEL_NAMES} for truth in _TRUTH_NAMES
    }

    for seed in _REALIZATION_SEEDS:
        generator = np.random.default_rng(seed)
        source = generator.normal(size=_IMAGE_SHAPE)
        for truth_name, noiseless_target in _truth_images(
            source,
            basis,
            terms,
        ).items():
            target = noiseless_target + generator.normal(
                scale=_NOISE_SIGMA,
                size=_IMAGE_SHAPE,
            )
            realization = _held_out_mse(
                source,
                target,
                variance,
                fit_mask,
                evaluation_mask,
                components,
            )
            for model_name, mse in realization.items():
                scores[truth_name][model_name].append(mse)

    means = {
        truth: {
            model: float(np.mean(realizations))
            for model, realizations in models.items()
        }
        for truth, models in scores.items()
    }

    constant = means["constant"]
    assert constant["constant"] < constant["spatial_als"]
    assert constant["constant"] < constant["gaussian_polynomial"]
    assert constant["spatial_als"] < 1.02 * constant["constant"]
    assert constant["gaussian_polynomial"] < 1.02 * constant["constant"]

    rank_one = means["spatial_rank_one"]
    assert rank_one["spatial_als"] < 0.55 * rank_one["constant"]
    assert rank_one["gaussian_polynomial"] < 0.55 * rank_one["constant"]
    shared_mode_ratio = (
        rank_one["gaussian_polynomial"] / rank_one["spatial_als"]
    )
    assert 0.95 < shared_mode_ratio < 1.05

    rank_two = means["spatial_rank_two"]
    assert rank_two["gaussian_polynomial"] < 0.30 * rank_two["spatial_als"]
    assert rank_two["gaussian_polynomial"] < 0.50 * rank_two["constant"]
    assert np.all(
        np.asarray(scores["spatial_rank_two"]["gaussian_polynomial"])
        < np.asarray(scores["spatial_rank_two"]["spatial_als"])
    )
