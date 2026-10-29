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
    EmpiricalCorrelationFactor,
    correlation_factor,
    empirical_correlation_factor,
    psf_weights,
    resampling_variance_factor,
    standardize_constant_kernel_residual,
    window_stamps,
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


def _gain(stamp):
    return (stamp**2).sum()


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
        reference_variance=_gain(taps),
        resampling_weights=taps,
    )
    assert isinstance(result, CorrelationFactorResult)
    assert result.covariance_scope == "reference_resampling_and_kernel"
    assert result.target_pixels_independent is True
    assert result.n_rotated == 0
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
    assert resampling_variance_factor(kernel, np.ones((1, 1))).shape == ()
    delta = np.zeros((15, 15))
    delta[7, 7] = 1.0
    result = correlation_factor(
        np.broadcast_to(psf_weights(2.0, 21), (2, 21, 21)),
        kernel=delta,
        target_variance=4.0,
        reference_variance=9.0,
        resampling_weights=np.ones((1, 1)),
    )
    np.testing.assert_allclose(result.factor, 1.0, rtol=1e-10)
    np.testing.assert_allclose(result.pixel_ratio, 1.0, rtol=1e-10)


def test_white_target_only_is_one():
    result = correlation_factor(
        psf_weights(2.0, 21)[None],
        kernel=_gaussian(15, 1.2),
        target_variance=4.0,
        reference_variance=0.0,
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
        reference_variance=s_t2 * _gain(stamp),
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
    reference_variance = np.full((n, n), var_s * _gain(stamp))
    noise = standardize_constant_kernel_residual(
        residual,
        kernel=kernel,
        target_variance=np.full((n, n), var_t),
        reference_variance=reference_variance,
    )
    diagonal = noise.marginal_residual_variance
    x0, y0 = positions[0]
    window = diagonal[y0 - half : y0 + half + 1, x0 - half : x0 + half + 1]
    measured = np.var(sums) / (w**2 * window).sum()
    result = correlation_factor(
        w[None],
        kernel=kernel,
        target_variance=var_t,
        reference_variance=var_s * _gain(stamp),
        resampling_weights=stamp,
        residual_variance=window[None],
    )
    default = correlation_factor(
        w[None],
        kernel=kernel,
        target_variance=var_t,
        reference_variance=var_s * _gain(stamp),
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
    assert isinstance(empirical, EmpiricalCorrelationFactor)
    assert empirical.factor.shape == ()
    assert empirical.positions_used == len(tall_positions)
    assert empirical.factor == pytest.approx(result.factor[0], rel=0.15)
    assert empirical.plain_ratio == pytest.approx(result.factor[0], rel=0.15)


def test_planes_with_positions_match_the_stamps_form():
    """Full planes cut at positions give the stamps result, with the window
    on the nearest pixel in (x, y) order on a non-square image."""

    rng = np.random.default_rng(5)
    height, width, size = 40, 70, 9
    kernel = _gaussian(7, 1.0)
    stamp = _lanczos_stamp(0.3, 0.7)
    target_plane = rng.uniform(1.0, 4.0, (height, width))
    reference_plane = rng.uniform(1.0, 4.0, (height, width))
    residual_plane = rng.uniform(5.0, 9.0, (height, width))
    positions = np.array([[50.4, 10.6], [12.0, 30.0], [66.0, 20.0]])
    w = psf_weights(1.5, size, offset=positions - np.rint(positions))
    planes = correlation_factor(
        w,
        kernel=kernel,
        target_variance=target_plane,
        reference_variance=reference_plane,
        resampling_weights=stamp,
        positions=positions,
        residual_variance=residual_plane,
    )
    windows = [
        window_stamps(plane, positions, size)
        for plane in (target_plane, reference_plane, residual_plane)
    ]
    y, x = 11, 50
    np.testing.assert_array_equal(
        windows[0][0], target_plane[y - 4 : y + 5, x - 4 : x + 5]
    )
    stamps = correlation_factor(
        w,
        kernel=kernel,
        target_variance=windows[0],
        reference_variance=windows[1],
        resampling_weights=stamp,
        residual_variance=windows[2],
    )
    np.testing.assert_allclose(planes.factor, stamps.factor, rtol=1e-12)
    np.testing.assert_allclose(
        planes.pixel_ratio, stamps.pixel_ratio, rtol=1e-12
    )
    assert np.isfinite(planes.factor[:2]).all()
    assert np.isnan(planes.factor[2])  # the window leaves the image
    transposed = correlation_factor(
        w[:1],
        kernel=kernel,
        target_variance=target_plane.T,
        reference_variance=reference_plane.T,
        resampling_weights=stamp,
        positions=positions[:1, ::-1],
        residual_variance=residual_plane.T,
    )
    assert transposed.factor[0] != pytest.approx(planes.factor[0])
    shared = correlation_factor(
        psf_weights(1.5, size),
        kernel=kernel,
        target_variance=target_plane,
        reference_variance=reference_plane,
        resampling_weights=stamp,
        positions=positions,
    )
    assert shared.factor.shape == (3,)


def test_per_position_stamps_match_one_call_per_position():
    rng = np.random.default_rng(6)
    n, size = 7, 11
    kernel = _gaussian(9, 1.1)
    w = rng.normal(size=(n, size, size))
    target = rng.uniform(1.0, 2.0, (n, size, size))
    reference = rng.uniform(1.0, 2.0, (n, size, size))
    stamps = np.stack(
        [_lanczos_stamp(fx, fy) for fx, fy in rng.uniform(0, 1, (n, 2))]
    )
    batch = correlation_factor(
        w,
        kernel=kernel,
        target_variance=target,
        reference_variance=reference,
        resampling_weights=stamps,
    )
    each = [
        correlation_factor(
            w[i : i + 1],
            kernel=kernel,
            target_variance=target[i : i + 1],
            reference_variance=reference[i : i + 1],
            resampling_weights=stamps[i : i + 1],
        ).factor[0]
        for i in range(n)
    ]
    np.testing.assert_allclose(batch.factor, each, rtol=1e-12)
    assert np.std(batch.factor) > 1e-3
    np.testing.assert_allclose(
        resampling_variance_factor(kernel, stamps),
        [resampling_variance_factor(kernel, s) for s in stamps],
        rtol=1e-12,
    )


def test_weights_axis_matches_one_call_per_weighting():
    """``(N, q, s, s)`` weights on shared windows give the ``q`` separate
    results as ``(N, q)``, with planes cut once and the position's mask,
    rotation flag and all-zero stamp applied to every weighting."""

    rng = np.random.default_rng(9)
    height, width, size, n = 40, 50, 9, 6
    kernel = _gaussian(7, 1.0)
    target_plane = rng.uniform(1.0, 4.0, (height, width))
    reference_plane = rng.uniform(1.0, 4.0, (height, width))
    valid_plane = rng.uniform(size=(height, width)) > 0.1
    positions = rng.uniform(8, 32, (n, 2))
    offset = positions - np.rint(positions)
    weights = np.stack(
        [
            psf_weights(1.5, size, offset=offset),
            psf_weights(1.5, size, offset=offset, derivative="x"),
            psf_weights(1.5, size, offset=offset, derivative="y"),
        ],
        axis=1,
    )
    stamps = np.stack(
        [_lanczos_stamp(fx, fy) for fx, fy in rng.uniform(0, 1, (n, 2))]
    )
    stamps[-1] = 0.0
    is_translation = np.array([True] * (n - 2) + [False, True])
    common = dict(
        kernel=kernel,
        target_variance=target_plane,
        reference_variance=reference_plane,
        resampling_weights=stamps,
        positions=positions,
        is_translation=is_translation,
        valid=valid_plane,
    )
    batch = correlation_factor(weights, **common)
    assert batch.factor.shape == batch.pixel_ratio.shape == (n, 3)
    assert batch.n_masked.shape == (n,) and batch.n_rotated == 1
    for q in range(3):
        each = correlation_factor(weights[:, q], **common)
        np.testing.assert_allclose(
            batch.factor[:, q], each.factor, rtol=1e-12
        )
        np.testing.assert_allclose(
            batch.pixel_ratio[:, q], each.pixel_ratio, rtol=1e-12
        )
        np.testing.assert_array_equal(batch.n_masked, each.n_masked)
    assert np.isfinite(batch.factor[: n - 2]).all()
    assert np.isnan(batch.factor[n - 2 :]).all()
    assert np.std(batch.factor[: n - 2], axis=1).min() > 1e-6


def test_chunk_boundary_is_seamless():
    n = 600
    kernel = _gaussian(7, 1.0)
    w = np.broadcast_to(psf_weights(1.5, 9), (n, 9, 9))
    stamps = np.stack([_lanczos_stamp(0.5, 0.5)] * (n - 1) + [np.eye(6)])
    result = correlation_factor(
        w,
        kernel=kernel,
        target_variance=1.0,
        reference_variance=2.0,
        resampling_weights=stamps,
    )
    assert result.factor.shape == (n,)
    np.testing.assert_allclose(result.factor[:-1], result.factor[0])
    assert result.factor[-1] != pytest.approx(result.factor[0])


@pytest.mark.parametrize(
    "name", ["target_variance", "reference_variance", "residual_variance"]
)
@pytest.mark.parametrize("planes", [False, True])
def test_negative_variance_invalidates_only_its_window(name, planes):
    shape = (9, 9) if planes else (2, 3, 3)
    variances = {
        "target_variance": np.ones(shape),
        "reference_variance": np.ones(shape),
        "residual_variance": np.full(shape, 2.0),
    }
    variances[name][(2, 2) if planes else (0, 1, 1)] = -0.5
    result = correlation_factor(
        np.ones((2, 3, 3)),
        kernel=np.ones((1, 1)),
        resampling_weights=np.ones((1, 1)),
        positions=np.array([[2, 2], [6, 6]]) if planes else None,
        **variances,
    )
    assert np.isnan(result.factor[0])
    assert np.isnan(result.pixel_ratio[0])
    assert result.factor[1] == pytest.approx(1.0)
    assert result.pixel_ratio[1] == pytest.approx(1.0)


def test_residual_variance_plane_scales_the_factor():
    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.5, 0.25)
    w = psf_weights(2.0, 21)[None]
    kwargs = dict(
        kernel=kernel,
        target_variance=4.0,
        reference_variance=9.0,
        resampling_weights=stamp,
    )
    default = correlation_factor(w, **kwargs)
    explicit = correlation_factor(
        w, residual_variance=4.0 + 9.0 * (kernel**2).sum(), **kwargs
    )
    doubled = correlation_factor(
        w, residual_variance=2.0 * (4.0 + 9.0 * (kernel**2).sum()), **kwargs
    )
    assert explicit.factor[0] == pytest.approx(default.factor[0], rel=1e-12)
    assert doubled.factor[0] == pytest.approx(
        default.factor[0] / 2.0, rel=1e-12
    )
    assert doubled.pixel_ratio[0] == pytest.approx(
        default.pixel_ratio[0] / 2.0, rel=1e-12
    )


def test_zero_resampling_stamps_give_nan():
    kernel = _gaussian(7, 1.0)
    stamps = np.stack([_lanczos_stamp(0.5, 0.5), np.zeros((6, 6))])
    with np.errstate(all="raise"):
        factor = resampling_variance_factor(kernel, stamps)
        result = correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel,
            target_variance=1.0,
            reference_variance=1.0,
            resampling_weights=stamps,
        )
    assert np.isfinite(factor[0]) and np.isnan(factor[1])
    assert np.isfinite(result.factor[0]) and np.isnan(result.factor[1])
    assert np.isnan(result.pixel_ratio[1])
    with np.errstate(all="raise"):
        empty = correlation_factor(
            np.zeros((1, 5, 5)),
            kernel=kernel,
            target_variance=1.0,
            reference_variance=1.0,
            resampling_weights=stamps[:1],
        )
    assert np.isnan(empty.factor[0])


def test_empirical_factor_of_white_noise_is_one():
    rng = np.random.default_rng(1)
    residual = rng.normal(size=(600, 600))
    centers = np.arange(20, 580, 24)
    gx, gy = np.meshgrid(centers, centers)
    positions = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    weights = np.stack(
        [psf_weights(2.0, 21), psf_weights(2.0, 21, derivative="x")]
    )
    result = empirical_correlation_factor(
        residual, np.ones_like(residual), weights, positions
    )
    assert result.factor.shape == result.plain_ratio.shape == (2,)
    assert result.positions_used == len(positions)
    np.testing.assert_allclose(result.factor, 1.0, atol=0.15)
    np.testing.assert_allclose(result.plain_ratio, 1.0, atol=0.15)
    scaled = empirical_correlation_factor(
        4.0 * residual, np.ones_like(residual), weights, positions
    )
    np.testing.assert_allclose(scaled.factor, 16.0 * result.factor)


def test_empirical_factor_drops_unusable_windows():
    rng = np.random.default_rng(2)
    residual = rng.normal(size=(64, 64))
    variance = np.ones_like(residual)
    variance[30, 30] = np.nan
    positions = np.array([[30.0, 30.0], [3.0, 3.0], [50.0, 50.0]])
    one = empirical_correlation_factor(
        residual, variance, psf_weights(1.5, 11), positions
    )
    assert one.positions_used == 1
    assert np.isnan(one.factor) and np.isnan(one.plain_ratio)
    positions = np.array([[20.0, 20.0], [40.0, 40.0], [50.0, 20.0]])
    three = empirical_correlation_factor(
        residual, variance, psf_weights(1.5, 11), positions
    )
    assert three.positions_used == 3
    assert np.isfinite(three.factor)


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
        reference_variance=9.0,
        resampling_weights=stamp,
    )
    assert np.isnan(result.factor[0]) and np.isnan(result.pixel_ratio[0])
    assert np.isfinite(result.factor[1]) and np.isfinite(
        result.pixel_ratio[1]
    )


@pytest.mark.parametrize("derivative", [None, "x", "y"])
def test_masked_window_keeps_a_factor_near_the_full_window_value(derivative):
    """A masked column (or a masked block from the column to the edge) at
    1 to 10 px from the center: the factor over the remaining pixels stays
    within 10% of the full-window value from 4 px on, and within 20% from
    1 to 3 px, where the mask cuts into the weights' core; ``n_masked``
    counts the pixels left out."""

    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.5, 0.25)
    size = 21
    weights = psf_weights(2.0, size, derivative=derivative)[None]
    common = dict(
        kernel=kernel,
        target_variance=4.0,
        reference_variance=9.0,
        resampling_weights=stamp,
    )
    full = correlation_factor(weights, **common)
    assert full.n_masked.tolist() == [0]
    for d in range(1, 11):
        column = np.ones((1, size, size), bool)
        column[0, :, size // 2 + d] = False
        block = np.ones((1, size, size), bool)
        block[0, :, size // 2 + d :] = False
        for valid, masked in (
            (column, size),
            (block, size * (size // 2 - d + 1)),
        ):
            result = correlation_factor(weights, valid=valid, **common)
            assert result.n_masked.tolist() == [masked]
            assert masked < size * size / 2
            ratio = result.factor[0] / full.factor[0]
            assert abs(ratio - 1.0) < (0.2 if d <= 3 else 0.1), (d, ratio)
    # more than half masked: the caller's limit, from n_masked
    half = np.ones((1, size, size), bool)
    half[0, :, size // 2 :] = False
    result = correlation_factor(weights, valid=half, **common)
    assert result.n_masked[0] > size * size / 2
    assert np.isfinite(result.factor[0])
    none = correlation_factor(
        weights, valid=np.zeros((1, size, size), bool), **common
    )
    assert np.isnan(none.factor[0]) and none.n_masked.tolist() == [size**2]


def test_masked_pixels_need_no_finite_variance_and_match_zero_weights():
    """Masked pixels with NaN variance keep the factor, which equals the
    factor of the same weights set to zero there with finite stamps; the
    plane form cuts the validity plane around the positions."""

    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.5, 0.25)
    size = 21
    w = np.broadcast_to(psf_weights(2.0, size), (2, size, size)).copy()
    target = np.full((2, size, size), 4.0)
    valid = np.ones((2, size, size), bool)
    valid[0, 3:8, 5] = False
    target[0, 3:8, 5] = np.nan
    masked = correlation_factor(
        w,
        kernel=kernel,
        target_variance=target,
        reference_variance=9.0,
        resampling_weights=stamp,
        valid=valid,
    )
    assert np.isfinite(masked.factor).all()
    assert masked.n_masked.tolist() == [5, 0]
    zeroed = correlation_factor(
        np.where(valid, w, 0.0),
        kernel=kernel,
        target_variance=np.where(valid, target, 4.0),
        reference_variance=9.0,
        resampling_weights=stamp,
    )
    np.testing.assert_allclose(masked.factor, zeroed.factor, rtol=1e-12)
    assert masked.factor[0] != pytest.approx(masked.factor[1])

    plane_target = np.full((80, 80), 4.0)
    plane_valid = np.ones((80, 80), bool)
    plane_valid[28:33, 25] = False  # stamp rows 3 to 7, column 5
    positions = np.array([[30.0, 35.0], [60.0, 60.0]])
    cut = correlation_factor(
        psf_weights(2.0, size),
        kernel=kernel,
        target_variance=plane_target,
        reference_variance=np.full((80, 80), 9.0),
        resampling_weights=stamp,
        positions=positions,
        valid=plane_valid,
    )
    assert cut.n_masked.tolist() == [5, 0]
    np.testing.assert_allclose(cut.factor, masked.factor, rtol=1e-12)
    with pytest.raises(ValueError, match="valid must be"):
        correlation_factor(
            w,
            kernel=kernel,
            target_variance=target,
            reference_variance=9.0,
            resampling_weights=stamp,
            valid=np.ones((3, size, size), bool),
        )


def test_plane_valid_counts_off_image_window_pixels_as_masked():
    """With a boolean plane, a window that leaves the image keeps a factor:
    the off-image pixels count in ``n_masked`` and equal the stamp form
    with those pixels set invalid; without ``valid`` the same window is
    ``NaN``."""

    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.5, 0.25)
    size = 21
    plane_target = np.full((80, 80), 4.0)
    plane_reference = np.full((80, 80), 9.0)
    # window of the first position spans rows -4..16: 4 rows off the image
    positions = np.array([[30.0, 6.0], [40.0, 40.0]])
    common = dict(
        kernel=kernel,
        target_variance=plane_target,
        reference_variance=plane_reference,
        resampling_weights=stamp,
        positions=positions,
    )
    cut = correlation_factor(
        psf_weights(2.0, size), valid=np.ones((80, 80), bool), **common
    )
    assert cut.n_masked.tolist() == [4 * size, 0]
    assert np.isfinite(cut.factor).all()
    bare = correlation_factor(psf_weights(2.0, size), **common)
    assert np.isnan(bare.factor[0]) and np.isfinite(bare.factor[1])

    valid = np.ones((2, size, size), bool)
    valid[0, :4] = False
    stamps = correlation_factor(
        np.broadcast_to(psf_weights(2.0, size), (2, size, size)),
        kernel=kernel,
        target_variance=4.0,
        reference_variance=9.0,
        resampling_weights=stamp,
        valid=valid,
    )
    assert stamps.n_masked.tolist() == cut.n_masked.tolist()
    np.testing.assert_allclose(cut.factor, stamps.factor, rtol=1e-12)
    np.testing.assert_allclose(cut.factor[1], bare.factor[1], rtol=1e-12)


def test_pixel_ratio_mixes_the_resampling_factor_with_the_target_term():
    kernel = _gaussian(15, 1.2)
    stamp = _lanczos_stamp(0.3, 0.6)
    resampling = resampling_variance_factor(kernel, stamp)
    var_t, s_t2 = 4.0, 9.0
    result = correlation_factor(
        np.ones((1, 1, 1)),
        kernel=kernel,
        target_variance=var_t,
        reference_variance=s_t2 * _gain(stamp),
        resampling_weights=stamp,
    )
    diag = s_t2 * (stamp**2).sum() * (kernel**2).sum()
    expected = (var_t + resampling * diag) / (var_t + diag)
    assert result.pixel_ratio[0] == pytest.approx(expected, rel=1e-10)
    assert result.factor[0] == pytest.approx(expected, rel=1e-10)


def test_psf_weights_are_unit_flux_gaussians_with_offsets():
    g = psf_weights(1.5, 13)
    dx = psf_weights(1.5, 13, derivative="x")
    dy = psf_weights(1.5, 13, derivative="y")
    np.testing.assert_allclose(dx, dy.T)
    assert g.sum() == pytest.approx(1.0, abs=1e-4)
    assert dx[6, 6] == 0.0 and dx[6, 7] > 0 and dx[6, 5] < 0
    assert dy[7, 6] > 0 and dy[5, 6] < 0
    shifted = psf_weights(1.5, 13, offset=[[0.0, 0.0], [0.4, -0.3]])
    assert shifted.shape == (2, 13, 13)
    np.testing.assert_allclose(shifted[0], g)
    u = np.arange(13) - 6
    expected = np.exp(
        -0.5 * ((u[:, None] + 0.3) ** 2 + (u[None, :] - 0.4) ** 2) / 1.5**2
    ) / (2.0 * np.pi * 1.5**2)
    np.testing.assert_allclose(shifted[1], expected, rtol=1e-12)
    # the x derivative of a shifted PSF is the finite difference in x
    eps = 1e-6
    numeric = (
        psf_weights(1.5, 13, offset=[[0.4 + eps, -0.3]])[0]
        - psf_weights(1.5, 13, offset=[[0.4 - eps, -0.3]])[0]
    ) / (2 * eps)
    analytic = psf_weights(1.5, 13, offset=[[0.4, -0.3]], derivative="x")[0]
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)
    with pytest.raises(ValueError):
        psf_weights(1.5, 12)
    with pytest.raises(ValueError, match="offset"):
        psf_weights(1.5, 13, offset=[0.4, -0.3])
    with pytest.raises(ValueError, match="derivative"):
        psf_weights(1.5, 13, derivative="z")


def test_rotated_positions_are_nan_and_counted():
    kernel = _gaussian(15, 1.2)
    stamps = np.stack([_lanczos_stamp(0.5, 0.25)] * 2)
    factor = resampling_variance_factor(
        kernel, stamps, is_translation=[True, False]
    )
    assert np.isfinite(factor[0]) and np.isnan(factor[1])
    result = correlation_factor(
        np.ones((2, 5, 5)),
        kernel=kernel,
        target_variance=1.0,
        reference_variance=1.0,
        resampling_weights=stamps,
        is_translation=[False, True],
    )
    assert np.isnan(result.factor[0]) and np.isfinite(result.factor[1])
    assert np.isnan(result.pixel_ratio[0])
    assert result.n_rotated == 1
    with pytest.raises(ValueError, match="is_translation"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel,
            target_variance=1.0,
            reference_variance=1.0,
            resampling_weights=stamps,
            is_translation=[False],
        )


def test_argument_validation():
    kernel = _gaussian(15, 1.2)
    stamps = np.stack([_lanczos_stamp(0.5, 0.25)] * 2)
    kwargs = dict(target_variance=1.0, reference_variance=1.0)
    with pytest.raises(ValueError, match="resampling_weights"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel,
            resampling_weights=stamps[:1],
            **kwargs,
        )
    with pytest.raises(ValueError, match="odd"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel[:-1],
            resampling_weights=stamps,
            **kwargs,
        )
    with pytest.raises(ValueError, match="finite"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel * np.nan,
            resampling_weights=stamps,
            **kwargs,
        )
    with pytest.raises(ValueError, match="odd-sized"):
        correlation_factor(
            np.ones((2, 4, 4)),
            kernel=kernel,
            resampling_weights=stamps,
            **kwargs,
        )
    with pytest.raises(ValueError, match="broadcast"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel,
            resampling_weights=stamps,
            target_variance=np.ones((3, 5, 5)),
            reference_variance=1.0,
        )
    with pytest.raises(ValueError, match="positions"):
        correlation_factor(
            np.ones((2, 5, 5)),
            kernel=kernel,
            resampling_weights=stamps,
            positions=np.zeros((3, 2)),
            **kwargs,
        )
