# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


def _fit(reference, target, fit_mask, *, background_degree=0):
    return solve_constant_kernel(
        reference,
        target,
        COMPONENTS,
        kernel_shape=KERNEL_SHAPE,
        variance=np.full(reference.shape, TARGET_VARIANCE),
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


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_plain_form_is_the_weighted_least_squares_covariance(degree):
    _, reference, target, fit_mask, _, _ = _scene()
    fit = _fit(reference, target, fit_mask, background_degree=degree)
    cov = kernel_fit_covariance(
        fit,
        reference=reference,
        target_variance=np.full(reference.shape, TARGET_VARIANCE),
    )
    assert isinstance(cov, KernelFitCovariance)
    assert cov.form == "plain"
    assert cov.covariance_represented is True
    assert cov.basis_count == fit.basis_kernels.shape[0] == 3
    assert cov.background_degree == degree
    design = _design(fit, reference, degree)
    expected = np.linalg.inv(design.T @ design / TARGET_VARIANCE)
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


def test_sandwich_matches_monte_carlo():
    """Fresh target and reference noise on one scene, the fit mask fixed.

    The scatter of the fitted model at bright pixels against the predicted
    kernel-fit variance: the sandwich form agrees within 5%, the plain form
    omits the reference noise and under-predicts.
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
        noisy_raw = raw + rng.normal(0.0, np.sqrt(SOURCE_VARIANCE), raw.shape)
        reference = correlate(noisy_raw, stamp, mode="nearest")
        target = target_clean + rng.normal(
            0.0, np.sqrt(TARGET_VARIANCE), raw.shape
        )
        fit = _fit(reference, target, fit_mask)
        if covariances is None:
            source_variance = np.full(raw.shape, SOURCE_VARIANCE)
            variance = np.full(raw.shape, TARGET_VARIANCE)
            plain = kernel_fit_covariance(
                fit, reference=reference, target_variance=variance
            )
            sandwich = kernel_fit_covariance(
                fit,
                reference=reference,
                target_variance=variance,
                reference_source_variance=source_variance,
                resampling_weights=stamp[None],
            )
            assert sandwich.form == "sandwich"
            covariances = {
                name: kernel_fit_variance(
                    cov, reference=reference, positions=positions
                )
                for name, cov in (("plain", plain), ("sandwich", sandwich))
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
    assert ratio["sandwich"] == pytest.approx(1.0, abs=0.05)
    assert ratio["plain"] > ratio["sandwich"] * 1.1
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
            reference_source_variance=variance,
            resampling_weights=np.stack([stamp, stamp]),
        )
    with pytest.raises(ValueError, match="match reference shape"):
        kernel_fit_covariance(
            fit, reference=reference, target_variance=variance[:-1]
        )
    bad = variance.copy()
    bad[fit_mask][0] = 0.0
    bad[20, 20] = np.inf
    with pytest.raises(ValueError, match="positive and finite"):
        kernel_fit_covariance(fit, reference=reference, target_variance=bad)
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
