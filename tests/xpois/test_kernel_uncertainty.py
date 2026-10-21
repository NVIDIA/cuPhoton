# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.ndimage import correlate
from scipy.signal import fftconvolve

from cuphoton.xpois import (
    GaussianBasisComponent,
    KernelFitCovariance,
    background_design,
    kernel_fit_covariance,
    kernel_fit_variance,
    solve_constant_kernel,
    window_stamps,
)
from cuphoton.xpois.kernel_uncertainty import _fftconvolve

SIZE = 96
KERNEL_SHAPE = (9, 9)
COMPONENTS = (GaussianBasisComponent(sigma=1.0, degree=1),)
TARGET_VARIANCE = 4.0
SOURCE_VARIANCE = 1.0


def _gaussian(shape, sigma):
    y = np.arange(shape[0]) - shape[0] // 2
    x = np.arange(shape[1]) - shape[1] // 2
    k = np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2) / sigma**2)
    return k / k.sum()


def _resampling_stamp():
    taps = np.array([0.1, 0.4, 0.4, 0.1])
    return taps[:, None] * taps[None, :]


def _scene(seed=0):
    """Noiseless source image with bright stars and its resampled copy."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    raw = np.full((SIZE, SIZE), 10.0)
    stars = rng.uniform(20, SIZE - 20, (12, 2))
    for x0, y0 in stars:
        raw += 3000.0 * np.exp(
            -0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5**2
        )
    stamp = _resampling_stamp()
    reference = correlate(raw, stamp, mode="nearest")
    target = fftconvolve(reference, _gaussian(KERNEL_SHAPE, 1.0), "same")
    target += 2.0
    fit_mask = np.zeros((SIZE, SIZE), dtype=bool)
    fit_mask[12:-12, 12:-12] = True
    return raw, reference, target, fit_mask, stamp, stars


def _fit(reference, target, fit_mask, *, background_degree=0, variance=None):
    if variance is None:
        variance = np.full(reference.shape, TARGET_VARIANCE)
    return solve_constant_kernel(
        reference,
        target,
        COMPONENTS,
        kernel_shape=KERNEL_SHAPE,
        variance=variance,
        fit_mask=fit_mask,
        background_degree=background_degree,
        backend="cpu",
    )


def _design(fit, reference, degree):
    columns = [
        fftconvolve(reference, b, mode="same")[fit.fit_mask]
        for b in fit.basis_kernels
    ]
    columns += [
        t[fit.fit_mask] for t in background_design(reference.shape, degree)
    ]
    return np.column_stack(columns)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize(
    "shape, kshape", [((32, 32), (5, 5)), ((31, 33), (4, 6))]
)
def test_fftconvolve_matches_scipy(mode, shape, kshape):
    rng = np.random.default_rng(4)
    a, k = rng.normal(size=shape), rng.normal(size=kshape)
    np.testing.assert_allclose(
        _fftconvolve(a, k, mode=mode),
        fftconvolve(a, k, mode=mode),
        atol=1e-12,
    )


@pytest.mark.parametrize("flat_variance", [True, False])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_plain_form_is_the_weighted_least_squares_covariance(
    degree, flat_variance
):
    _, reference, target, fit_mask, _, _ = _scene()
    if flat_variance:
        variance = np.full(reference.shape, TARGET_VARIANCE)
    else:
        variance = np.random.default_rng(degree).uniform(
            1.0, 9.0, reference.shape
        )
    fit = _fit(
        reference,
        target,
        fit_mask,
        background_degree=degree,
        variance=variance,
    )
    cov = kernel_fit_covariance(
        fit, reference=reference, target_variance=variance
    )
    assert isinstance(cov, KernelFitCovariance)
    assert cov.form == "plain"
    assert cov.covariance_represented is True
    assert cov.basis_count == fit.basis_kernels.shape[0] == 3
    assert cov.background_degree == degree
    design = _design(fit, reference, degree)
    expected = np.linalg.inv(
        design.T @ (design / variance[fit.fit_mask][:, None])
    )
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    np.testing.assert_allclose(
        cov.covariance / scale, expected / scale, rtol=1e-8, atol=1e-10
    )


def test_variance_of_a_delta_weight_is_the_quadratic_form():
    _, reference, target, fit_mask, _, stars = _scene()
    fit = _fit(reference, target, fit_mask, background_degree=1)
    cov = kernel_fit_covariance(
        fit,
        reference=reference,
        target_variance=np.full(reference.shape, TARGET_VARIANCE),
    )
    positions = np.rint(stars[:4])
    pixel = kernel_fit_variance(cov, reference=reference, positions=positions)
    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    weighted = kernel_fit_variance(
        cov, reference=reference, positions=positions, weights=delta
    )
    np.testing.assert_allclose(weighted, pixel, rtol=1e-10)
    background = background_design(reference.shape, 1)
    for (x, y), value in zip(positions.astype(int), pixel, strict=True):
        gamma = np.r_[
            [
                fftconvolve(reference, b, "same")[y, x]
                for b in fit.basis_kernels
            ],
            background[:, y, x],
        ]
        assert value == pytest.approx(
            gamma @ cov.covariance @ gamma, rel=1e-8
        )
    assert pixel.min() > 0
    # a stamp that leaves the image is NaN
    edge = kernel_fit_variance(
        cov, reference=reference, positions=np.array([[2.0, 40.0]])
    )
    assert np.isnan(edge[0])


def test_weighted_sum_variance_is_the_projected_quadratic_form():
    _, reference, target, fit_mask, _, stars = _scene()
    fit = _fit(reference, target, fit_mask)
    cov = kernel_fit_covariance(
        fit,
        reference=reference,
        target_variance=np.full(reference.shape, TARGET_VARIANCE),
    )
    weights = _gaussian((11, 11), 2.0)
    positions = np.rint(stars[:3])
    got = kernel_fit_variance(
        cov, reference=reference, positions=positions, weights=weights
    )
    columns = [fftconvolve(reference, b, "same") for b in fit.basis_kernels]
    columns += list(background_design(reference.shape, 0))
    for (x, y), value in zip(positions.astype(int), got, strict=True):
        patch = [c[y - 5 : y + 6, x - 5 : x + 6] for c in columns]
        g = np.array([(weights * p).sum() for p in patch])
        assert value == pytest.approx(g @ cov.covariance @ g, rel=1e-8)


@pytest.mark.parametrize("weights", [None, np.ones((3, 3))])
def test_variance_with_no_positions_returns_empty(weights):
    covariance = KernelFitCovariance(
        covariance=np.eye(2),
        basis_kernels=np.ones((1, 1, 1)),
        basis_count=1,
        background_degree=0,
        form="plain",
    )
    result = kernel_fit_variance(
        covariance,
        reference=np.ones((9, 9)),
        positions=np.empty((0, 2)),
        weights=weights,
    )
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_per_position_weights_match_one_call_per_position():
    _, reference, target, fit_mask, _, stars = _scene()
    fit = _fit(reference, target, fit_mask, background_degree=1)
    cov = kernel_fit_covariance(
        fit,
        reference=reference,
        target_variance=np.full(reference.shape, TARGET_VARIANCE),
    )
    positions = np.rint(stars[:5])
    weights = np.random.default_rng(7).normal(size=(5, 7, 7))
    got = kernel_fit_variance(
        cov, reference=reference, positions=positions, weights=weights
    )
    each = [
        kernel_fit_variance(
            cov,
            reference=reference,
            positions=positions[i : i + 1],
            weights=w,
        )[0]
        for i, w in enumerate(weights)
    ]
    np.testing.assert_allclose(got, each, rtol=1e-12)
    assert not np.allclose(got, got[0])


def test_reference_variance_is_in_resampled_units():
    """Doubling the resampled reference variance doubles the added term."""

    _, reference, target, fit_mask, stamp, _ = _scene()
    fit = _fit(reference, target, fit_mask)
    variance = np.full(reference.shape, TARGET_VARIANCE)
    plain = kernel_fit_covariance(
        fit, reference=reference, target_variance=variance
    )
    one, two = (
        kernel_fit_covariance(
            fit,
            reference=reference,
            target_variance=variance,
            reference_variance=np.full(reference.shape, scale),
            resampling_weights=stamp,
        ).covariance
        for scale in (1.0, 2.0)
    )
    np.testing.assert_allclose(
        two - plain.covariance, 2.0 * (one - plain.covariance), rtol=1e-10
    )
    # the stamp's orientation does not matter for a point-symmetric stamp
    flipped = kernel_fit_covariance(
        fit,
        reference=reference,
        target_variance=variance,
        reference_variance=np.ones(reference.shape),
        resampling_weights=stamp[::-1, ::-1],
    ).covariance
    np.testing.assert_allclose(flipped, one, rtol=1e-10)


@pytest.mark.parametrize(
    "target_variance, source_variance",
    [(4.0, 1.0), (1.0, 16.0)],
    ids=["target-dominated", "reference-dominated"],
)
def test_sandwich_matches_monte_carlo(target_variance, source_variance):
    """Fresh target and reference noise on one scene, the fit mask fixed.

    The scatter of the fitted model at bright pixels against the predicted
    kernel-fit variance: the sandwich form agrees within 7%, the plain form
    omits the reference noise and under-predicts. The reference-dominated
    regime makes the reference term most of the variance, so a wrong scale
    of the added term fails the test.
    """

    raw, _, target_clean, fit_mask, stamp, stars = _scene()
    rng = np.random.default_rng(1)
    positions = np.rint(stars)
    xs, ys = positions[:, 0].astype(int), positions[:, 1].astype(int)
    reference_clean = correlate(raw, stamp, mode="nearest")
    model = []
    coefficients = []
    covariances = None
    for _ in range(1500):
        noisy_raw = raw + rng.normal(0.0, np.sqrt(source_variance), raw.shape)
        reference = correlate(noisy_raw, stamp, mode="nearest")
        target = target_clean + rng.normal(
            0.0, np.sqrt(target_variance), raw.shape
        )
        variance = np.full(raw.shape, target_variance)
        fit = _fit(reference, target, fit_mask, variance=variance)
        if covariances is None:
            resampled_variance = np.full(
                raw.shape, source_variance * (stamp**2).sum()
            )
            plain = kernel_fit_covariance(
                fit, reference=reference, target_variance=variance
            )
            sandwich = kernel_fit_covariance(
                fit,
                reference=reference,
                target_variance=variance,
                reference_variance=resampled_variance,
                resampling_weights=stamp[None],
            )
            assert sandwich.form == "sandwich"
            half_added = dataclasses.replace(
                sandwich,
                covariance=0.5 * (plain.covariance + sandwich.covariance),
            )
            covariances = {
                name: kernel_fit_variance(
                    cov, reference=reference, positions=positions
                )
                for name, cov in (
                    ("plain", plain),
                    ("sandwich", sandwich),
                    ("half_added", half_added),
                )
            }
        coefficients.append(
            np.r_[fit.kernel_coefficients, fit.background_coefficients]
        )
        model.append(
            (
                fftconvolve(reference_clean, fit.kernel, "same")
                + fit.background
            )[ys, xs]
        )
    empirical = np.var(np.array(model), axis=0, ddof=1)
    ratio = {k: empirical.sum() / v.sum() for k, v in covariances.items()}
    assert ratio["sandwich"] == pytest.approx(1.0, abs=0.07)
    assert ratio["plain"] > ratio["sandwich"] * 1.1
    if source_variance > target_variance:
        assert ratio["half_added"] > 1.5
    coefficient_variance = np.var(np.array(coefficients), axis=0, ddof=1)
    np.testing.assert_allclose(
        coefficient_variance / np.diag(sandwich.covariance), 1.0, atol=0.08
    )


def test_argument_validation():
    _, reference, target, fit_mask, stamp, _ = _scene()
    fit = _fit(reference, target, fit_mask)
    variance = np.full(reference.shape, TARGET_VARIANCE)
    with pytest.raises(ValueError, match="together"):
        kernel_fit_covariance(
            fit,
            reference=reference,
            target_variance=variance,
            resampling_weights=stamp,
        )
    with pytest.raises(ValueError, match="one \\(k, k\\) stamp"):
        kernel_fit_covariance(
            fit,
            reference=reference,
            target_variance=variance,
            reference_variance=variance,
            resampling_weights=np.stack([stamp, stamp]),
        )
    with pytest.raises(ValueError, match="not all zero"):
        kernel_fit_covariance(
            fit,
            reference=reference,
            target_variance=variance,
            reference_variance=variance,
            resampling_weights=np.zeros_like(stamp),
        )
    with pytest.raises(ValueError, match="match reference shape"):
        kernel_fit_covariance(
            fit, reference=reference, target_variance=variance[:-1]
        )
    y, x = np.nonzero(fit_mask)
    zero = variance.copy()
    zero[y[0], x[0]] = 0.0
    with pytest.raises(ValueError, match="positive and finite"):
        kernel_fit_covariance(fit, reference=reference, target_variance=zero)
    infinite = variance.copy()
    infinite[y[-1], x[-1]] = np.inf
    with pytest.raises(ValueError, match="positive and finite"):
        kernel_fit_covariance(
            fit, reference=reference, target_variance=infinite
        )
    outside = variance.copy()
    outside[~fit_mask] = 0.0
    kernel_fit_covariance(fit, reference=reference, target_variance=outside)
    duplicated = dataclasses.replace(
        fit,
        basis_kernels=np.concatenate(
            [fit.basis_kernels, fit.basis_kernels[:1]]
        ),
    )
    with pytest.raises(ValueError, match="ill-conditioned"):
        kernel_fit_covariance(
            duplicated, reference=reference, target_variance=variance
        )
    cov = kernel_fit_covariance(
        fit, reference=reference, target_variance=variance
    )
    with pytest.raises(ValueError, match="positions"):
        kernel_fit_variance(cov, reference=reference, positions=[1.0, 2.0])
    with pytest.raises(ValueError, match="odd square"):
        kernel_fit_variance(
            cov,
            reference=reference,
            positions=[[40.0, 40.0]],
            weights=np.ones((4, 4)),
        )


def test_sandwich_rejects_negative_reference_variance():
    _, reference, target, fit_mask, stamp, _ = _scene()
    fit = _fit(reference, target, fit_mask)
    reference_variance = np.ones_like(reference)
    reference_variance[0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        kernel_fit_covariance(
            fit,
            reference=reference,
            target_variance=np.full(reference.shape, TARGET_VARIANCE),
            reference_variance=reference_variance,
            resampling_weights=stamp,
        )


def test_window_stamps_cuts_x_y_windows_with_a_fill():
    plane = np.arange(5 * 8, dtype=float).reshape(5, 8)
    windows = window_stamps(plane, [[6.0, 2.0], [0.4, 3.6]], 3)
    assert windows.shape == (2, 3, 3)
    np.testing.assert_array_equal(windows[0], plane[1:4, 5:8])
    assert np.isnan(windows[1][:, 0]).all()
    assert windows[1][0, 1] == plane[3, 0] and windows[1][1, 2] == plane[4, 1]
    assert np.isnan(windows[1][2]).all()
    tall = window_stamps(plane, [[3.0, 2.0]], (5, 3), fill=0.0)
    assert tall.shape == (1, 5, 3)
    np.testing.assert_array_equal(tall[0], plane[0:5, 2:5])
    with pytest.raises(ValueError, match="odd"):
        window_stamps(plane, [[3.0, 2.0]], 4)
    with pytest.raises(ValueError, match="positions"):
        window_stamps(plane, [3.0, 2.0], 3)


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_window_stamps_fills_windows_from_empty_images(shape):
    windows = window_stamps(np.empty(shape), [[2.0, 1.0]], (3, 5), fill=-2.0)
    np.testing.assert_array_equal(windows, np.full((1, 3, 5), -2.0))
    empty = window_stamps(np.empty(shape), np.empty((0, 2)), 3)
    assert empty.shape == (0, 3, 3)
