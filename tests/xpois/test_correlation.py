# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import correlate
from scipy.signal import fftconvolve

from cuphoton.xpois import (
    CorrelationFactorResult,
    correlation_factor,
    empirical_correlation_factor,
    psf_weights,
    resampling_variance_factor,
    standardize_constant_kernel_residual,
)


def _gaussian(size, sigma):
    i = np.arange(size) - size // 2
    k = np.exp(-0.5 * (i[:, None] ** 2 + i[None, :] ** 2) / sigma**2)
    return k / k.sum()


def _lanczos_stamp(frac_x, frac_y, a=3):
    """Normalized Lanczos weights at a sub-pixel phase, one 2-D stamp."""

    taps = np.arange(-(a - 1), a + 1, dtype=np.float64)

    def weights(frac):
        u = taps - frac
        w = np.sinc(u) * np.sinc(u / a)
        return w / w.sum()

    return weights(frac_y)[:, None] * weights(frac_x)[None, :]


DELTA = np.zeros((1, 1, 1))
DELTA[0, 0, 0] = 1.0


def test_toy_strip_sum_variance_is_the_exact_filtered_power():
    """1-D strip: unit white noise through a 3-tap filter, then a 5-pixel
    box sum. The diagonal prediction is 1.875, the exact variance 4.25."""

    taps = np.array([[[0.25, 0.5, 0.25]]])
    box = np.ones((1, 1, 5))
    kernel = np.ones((1, 1))
    result = correlation_factor(
        box,
        kernel=kernel,
        target_variance=0.0,
        reference_source_variance=1.0,
        resampling_weights=taps,
    )
    assert isinstance(result, CorrelationFactorResult)
    assert result.covariance_scope == "reference_resampling_and_kernel"
    assert result.target_pixels_independent is True
    assert result.factor[0] == pytest.approx(4.25 / 1.875)
    assert result.pixel_ratio[0] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(20_000, 7))
    filtered = np.stack(
        [noise[:, i : i + 3] @ taps[0, 0, ::-1] for i in range(5)], axis=1
    )
    assert filtered.sum(1).var() == pytest.approx(4.25, rel=0.05)
    assert filtered[:, 0].var() == pytest.approx(0.375, rel=0.05)


def test_identity_resampling_gives_one():
    kernel = _gaussian(15, 1.2)
    factor = resampling_variance_factor(kernel, np.ones((3, 1, 1)))
    np.testing.assert_allclose(factor, 1.0, rtol=1e-12)
    delta = np.zeros((15, 15))
    delta[7, 7] = 1.0
    result = correlation_factor(
        np.broadcast_to(psf_weights(2.0, 21), (2, 21, 21)),
        kernel=delta,
        target_variance=4.0,
        reference_source_variance=9.0,
        resampling_weights=np.ones((1, 1)),
    )
    np.testing.assert_allclose(result.factor, 1.0, rtol=1e-10)
    np.testing.assert_allclose(result.pixel_ratio, 1.0, rtol=1e-10)


def test_white_target_only_is_one():
    result = correlation_factor(
        psf_weights(2.0, 21)[None],
        kernel=_gaussian(15, 1.2),
        target_variance=4.0,
        reference_source_variance=0.0,
        resampling_weights=_lanczos_stamp(0.5, 0.25),
    )
    np.testing.assert_allclose(result.factor, 1.0, rtol=1e-10)


def test_resampling_factor_exceeds_one_at_a_half_pixel_phase():
    stamps = np.stack([_lanczos_stamp(0.0, 0.0), _lanczos_stamp(0.5, 0.5)])
    factor = resampling_variance_factor(_gaussian(15, 1.2), stamps)
    assert factor[0] == pytest.approx(1.0, abs=1e-12)
    assert factor[1] > 1.2


def test_factor_equals_the_direct_quadratic_form():
    """The footprint power equals w^T C w with C the autocovariance of
    white source noise through the resampling stamp and the kernel."""

    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.5, 0.2)
    sigma, half = 1.5, 8
    size = 2 * half + 1
    w = psf_weights(sigma, size)
    var_t, s_t2 = 3.0, 5.0
    result = correlation_factor(
        w[None],
        kernel=kernel,
        target_variance=var_t,
        reference_source_variance=s_t2,
        resampling_weights=stamp,
    )
    h = fftconvolve(kernel, stamp, mode="full")
    acf = fftconvolve(h, h[::-1, ::-1], mode="full")
    center = np.array(acf.shape) // 2
    i = np.arange(-half, half + 1)
    pts = np.array([(a, b) for a in i for b in i])
    lag = pts[:, None, :] - pts[None, :, :]
    cov = s_t2 * acf[lag[..., 0] + center[0], lag[..., 1] + center[1]]
    cov[np.arange(size**2), np.arange(size**2)] += var_t
    flat = w.ravel()
    diag = var_t + s_t2 * (stamp**2).sum() * (kernel**2).sum()
    expected = (flat @ cov @ flat) / (diag * (flat**2).sum())
    assert result.factor[0] == pytest.approx(expected, rel=1e-8)
    assert result.factor[0] > 1.3


def test_translation_matches_monte_carlo():
    """White source noise resampled by a Lanczos stamp at a half-pixel
    phase and convolved with the kernel, plus white target noise: the
    variance of PSF-weighted sums over the diagonal prediction matches the
    closed form, and the diagonal plane is the one xPOIS propagates."""

    kernel = _gaussian(15, 1.3)
    stamp = _lanczos_stamp(0.5, 0.5)
    n, var_t, var_s = 160, 1.0, 2.0
    rng = np.random.default_rng(3)
    half = 10
    size = 2 * half + 1
    w = psf_weights(2.0, size)
    centers = np.arange(half + 8, n - half - 8, size)
    gx, gy = np.meshgrid(centers, centers)
    positions = np.column_stack([gx.ravel(), gy.ravel()])
    sums = []
    kept = []
    for _ in range(400):
        resampled = correlate(
            rng.normal(0.0, np.sqrt(var_s), (n, n)), stamp, mode="wrap"
        )
        residual = rng.normal(0.0, np.sqrt(var_t), (n, n)) - fftconvolve(
            resampled, kernel, mode="same"
        )
        for x, y in positions:
            sums.append(
                (
                    w
                    * residual[
                        y - half : y + half + 1, x - half : x + half + 1
                    ]
                ).sum()
            )
        if len(kept) < 20:
            kept.append(residual)
    noise = standardize_constant_kernel_residual(
        residual,
        kernel=kernel,
        target_variance=np.full((n, n), var_t),
        reference_variance=np.full((n, n), var_s * (stamp**2).sum()),
    )
    diagonal = noise.marginal_residual_variance
    x0, y0 = positions[0]
    window = diagonal[y0 - half : y0 + half + 1, x0 - half : x0 + half + 1]
    measured = np.var(sums) / (w**2 * window).sum()
    result = correlation_factor(
        w[None],
        kernel=kernel,
        target_variance=var_t,
        reference_source_variance=var_s,
        resampling_weights=stamp,
        residual_variance=window[None],
    )
    default = correlation_factor(
        w[None],
        kernel=kernel,
        target_variance=var_t,
        reference_source_variance=var_s,
        resampling_weights=stamp,
    )
    assert result.factor[0] == pytest.approx(default.factor[0], rel=1e-10)
    assert result.factor[0] > 1.3
    assert measured == pytest.approx(result.factor[0], rel=0.05)
    # the measured factor on 20 realizations stacked into one tall image
    tall = np.concatenate(kept, axis=0)
    tall_positions = np.concatenate(
        [positions + [0, i * n] for i in range(len(kept))]
    ).astype(float)
    empirical = empirical_correlation_factor(
        tall, np.tile(diagonal, (len(kept), 1)), w, tall_positions
    )
    assert empirical.shape == ()
    assert empirical == pytest.approx(result.factor[0], rel=0.15)


def test_empirical_factor_of_white_noise_is_one():
    rng = np.random.default_rng(1)
    residual = rng.normal(size=(600, 600))
    centers = np.arange(20, 580, 24)
    gx, gy = np.meshgrid(centers, centers)
    positions = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    weights = np.stack(
        [psf_weights(2.0, 21), psf_weights(2.0, 21, derivative="x")]
    )
    factor = empirical_correlation_factor(
        residual, np.ones_like(residual), weights, positions
    )
    assert factor.shape == (2,)
    np.testing.assert_allclose(factor, 1.0, atol=0.15)
    scaled = empirical_correlation_factor(
        4.0 * residual, np.ones_like(residual), weights, positions
    )
    np.testing.assert_allclose(scaled, 16.0 * factor)


def test_empirical_factor_drops_unusable_windows():
    rng = np.random.default_rng(2)
    residual = rng.normal(size=(64, 64))
    variance = np.ones_like(residual)
    variance[30, 30] = np.nan
    positions = np.array([[30.0, 30.0], [3.0, 3.0], [50.0, 50.0]])
    one = empirical_correlation_factor(
        residual, variance, psf_weights(1.5, 11), positions
    )
    assert np.isnan(one)
    positions = np.array([[20.0, 20.0], [40.0, 40.0], [50.0, 20.0]])
    assert np.isfinite(
        empirical_correlation_factor(
            residual, variance, psf_weights(1.5, 11), positions
        )
    )


def test_partially_invalid_window_gives_nan():
    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.5, 0.25)
    w = np.broadcast_to(psf_weights(2.0, 21), (2, 21, 21))
    target = np.full((2, 21, 21), 4.0)
    target[0, 3, 5] = np.nan
    result = correlation_factor(
        w,
        kernel=kernel,
        target_variance=target,
        reference_source_variance=9.0,
        resampling_weights=stamp,
    )
    assert np.isnan(result.factor[0]) and np.isnan(result.pixel_ratio[0])
    assert np.isfinite(result.factor[1]) and np.isfinite(
        result.pixel_ratio[1]
    )


def test_pixel_ratio_mixes_the_resampling_factor_with_the_target_term():
    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.3, 0.6)
    resampling = resampling_variance_factor(kernel, stamp[None])[0]
    var_t, s_t2 = 4.0, 9.0
    result = correlation_factor(
        np.ones((1, 1, 1)),
        kernel=kernel,
        target_variance=var_t,
        reference_source_variance=s_t2,
        resampling_weights=stamp,
    )
    diag = s_t2 * (stamp**2).sum() * (kernel**2).sum()
    expected = (var_t + resampling * diag) / (var_t + diag)
    assert result.pixel_ratio[0] == pytest.approx(expected, rel=1e-10)
    assert result.factor[0] == pytest.approx(expected, rel=1e-10)


def test_position_weights_are_derivatives_along_their_own_axis():
    g = psf_weights(1.5, 13)
    dx = psf_weights(1.5, 13, derivative="x")
    dy = psf_weights(1.5, 13, derivative="y")
    np.testing.assert_allclose(dx, dy.T)
    assert g.sum() == pytest.approx(1.0, abs=1e-4)
    assert dx[6, 6] == 0.0 and dx[6, 7] > 0 and dx[6, 5] < 0
    assert dy[7, 6] > 0 and dy[5, 6] < 0
    with pytest.raises(ValueError):
        psf_weights(1.5, 12)


def test_rotated_positions_are_not_implemented():
    kernel = _gaussian(15, 1.2)
    stamps = np.stack([_lanczos_stamp(0.5, 0.25)] * 2)
    with pytest.raises(NotImplementedError, match="translat"):
        resampling_variance_factor(
            kernel, stamps, is_translation=[True, False]
        )
    with pytest.raises(NotImplementedError, match="translat"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel,
            target_variance=1.0,
            reference_source_variance=1.0,
            resampling_weights=stamps,
            is_translation=[False, True],
        )
    with pytest.raises(ValueError, match="resampling_weights"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel,
            target_variance=1.0,
            reference_source_variance=1.0,
            resampling_weights=stamps[:1],
        )
