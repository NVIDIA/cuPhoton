# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Host (NumPy) checks of the stage operations against cuPhoton's reference
code.
"""

import numpy as np
import pytest
from transient_chain import ops


def _residual(seed=0, shape=(600, 800), n=60):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 1.0, shape)
    ys, xs = (
        rng.integers(20, shape[0] - 20, n),
        rng.integers(20, shape[1] - 20, n),
    )
    yy, xx = np.mgrid[-4:5, -4:5]
    blob = 20.0 * np.exp(-(xx**2 + yy**2) / 4.0)
    for y, x, s in zip(ys, xs, rng.choice([-1, 1], n), strict=True):
        r[y - 4 : y + 5, x - 4 : x + 5] += s * blob
    r[100:104, 200:204] = 30.0  # plateau: exercises the peak tie-break
    r[5, 5] = np.nan
    return r


def test_hotspots_match_reference():
    from cuphoton.xpois.review import identify_residual_hotspots

    r = _residual()
    sigma = ops.robust_sigma(r)
    ours = ops.hotspots(r, robust_sigma=sigma, threshold_sigma=5.0)
    ref = identify_residual_hotspots(
        r, robust_sigma=sigma, threshold_sigma=5.0, max_regions=10_000
    )
    assert len(ref) == len(ours["pixel_count"]) > 40
    for i, h in enumerate(ref):
        assert h["pixel_count"] == ours["pixel_count"][i]
        assert tuple(h["peak_yx"]) == (ours["peak_y"][i], ours["peak_x"][i])
        assert h["centroid_yx"] == pytest.approx(
            [ours["centroid_y"][i], ours["centroid_x"][i]]
        )
        assert h["peak_abs_sigma"] == pytest.approx(ours["peak_abs_sigma"][i])
        assert h["peak_residual"] == pytest.approx(ours["peak_residual"][i])
        assert h["mean_residual"] == pytest.approx(ours["mean_residual"][i])


def test_hotspots_empty():
    out = ops.hotspots(np.zeros((10, 10)), robust_sigma=1.0)
    assert out["pixel_count"].size == 0


def _scene(seed=1, shape=(700, 900), n=120):
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 3.0, shape)
    yy, xx = np.mgrid[-6:7, -6:7]
    for _ in range(n):
        y, x = (
            rng.integers(30, shape[0] - 30),
            rng.integers(30, shape[1] - 30),
        )
        amp, s = rng.uniform(50, 3000), rng.uniform(1.4, 2.2)
        img[y - 6 : y + 7, x - 6 : x + 7] += amp * np.exp(
            -(xx**2 + yy**2) / (2 * s * s)
        )
    img[10:20, 10:20] = np.nan
    return img


def test_compact_stamp_mask_matches_reference():
    from cuphoton.xpois import build_compact_source_stamp_mask

    img = _scene()
    var = np.full(img.shape, 9.0)
    mask, diag = ops.compact_source_stamp_mask(
        img, variance=var, stamp_size=31, max_stamps=5, peak_percentile=99.5
    )
    ref = build_compact_source_stamp_mask(
        img, variance=var, stamp_size=31, max_stamps=5, peak_percentile=99.5
    )
    assert diag["centers"] == list(ref.centers)
    assert np.array_equal(mask, ref.mask)


def test_find_stars_recovers_planted_positions():
    rng = np.random.default_rng(3)
    img = rng.normal(0.0, 1.0, (500, 500))
    truth = np.array(
        [[100.3, 120.7], [300.2, 80.4], [420.6, 410.1], [50.5, 450.5]]
    )  # x, y
    yy, xx = np.mgrid[-8:9, -8:9]
    for x, y in truth:
        ix, iy = int(round(x)), int(round(y))
        img[iy - 8 : iy + 9, ix - 8 : ix + 9] += 500 * np.exp(
            -((xx - (x - ix)) ** 2 + (yy - (y - iy)) ** 2) / (2 * 1.7**2)
        )
    stars = ops.find_stars(img, fwhm=4.0, nsigma=20.0, max_n=50)
    assert len(stars) == 4
    a, b = ops.match_stars(truth, stars, r=2.0)
    assert len(a) == 4
    assert np.abs(a - b).max() < 0.1
    d = ops.find_stars_daofind(img, fwhm=4.0, nsigma=20.0, max_n=50)
    a2, b2 = ops.match_stars(truth, d, r=2.0)
    assert len(a2) == 4 and np.abs(a2 - b2).max() < 0.1


def test_affine_and_xcorr():
    src = np.random.default_rng(0).uniform(0, 1000, (40, 2))
    A, b = (
        np.array([[1.0001, 0.0002], [-0.0001, 0.9999]]),
        np.array([4.15, 0.21]),
    )
    A2, b2, rms = ops.fit_affine(src, src @ A.T + b)
    assert np.allclose(A2, A) and np.allclose(b2, b) and rms < 1e-9
    img = _scene(seed=5, shape=(256, 256), n=30)
    shifted = np.roll(np.roll(img, 3, axis=0), -5, axis=1)
    dx, dy = ops.xcorr_offset(shifted, img)
    assert (round(dx), round(dy)) == (-5, 3)


def test_stamps_and_noise_factor():
    rng = np.random.default_rng(2)
    search = rng.normal(10, 5, (200, 200))
    resid = rng.normal(0, 2, (200, 200))
    resid[50, 50] = np.nan
    stamps, diff = ops.extract_stamps(
        search, resid, np.array([50, 120]), np.array([50, 130]), 51
    )
    assert (
        stamps.shape == (2, 3, 51, 51)
        and stamps.dtype == np.float32
        and diff.dtype == np.float64
    )
    assert np.isfinite(stamps).all() and stamps[0, 2, 25, 25] == 0.0
    # builder convention: template keeps the search value where the residual
    # was NaN
    assert stamps[0, 1, 25, 25] == stamps[0, 0, 25, 25]
    assert np.allclose(stamps[:, 1], stamps[:, 0] - stamps[:, 2])
    f, n_bad = ops.noise_match_factors(stamps[:, 0], 114.7)
    assert f.shape == (2,) and np.all(f > 0) and n_bad == 0
    assert np.allclose(
        ops.robust_sigma(stamps[0, 0] * f[0]), 114.7, rtol=1e-3
    )


def test_noise_factor_flat_stamps_fall_back_to_one():
    f, n_bad = ops.noise_match_factors(np.zeros((2, 51, 51)), 1.0)
    assert n_bad == 2 and np.all(f == 1.0)
    mixed = np.zeros((2, 51, 51))
    mixed[1] = np.random.default_rng(0).normal(0, 2.0, (51, 51))
    f, n_bad = ops.noise_match_factors(mixed, 2.0)
    assert n_bad == 1 and np.isfinite(f).all() and f[0] == pytest.approx(f[1])


def test_extract_stamps_rejects_even_size():
    with pytest.raises(ValueError, match="odd"):
        ops.extract_stamps(
            np.zeros((100, 100)),
            np.zeros((100, 100)),
            np.array([50]),
            np.array([50]),
            50,
        )


def test_find_stars_keeps_the_brightest_when_precutting():
    rng = np.random.default_rng(7)
    img = rng.normal(0.0, 1.0, (600, 600))
    yy, xx = np.mgrid[-12:13, -12:13]
    compact = [(80 + 60 * i, 90 + 55 * i) for i in range(6)]
    for y, x in compact:
        img[y - 12 : y + 13, x - 12 : x + 13] += 400 * np.exp(
            -(xx**2 + yy**2) / (2 * 1.2**2)
        )
    img[480 - 12 : 480 + 13, 480 - 12 : 480 + 13] += 120 * np.exp(
        -(xx**2 + yy**2) / (2 * 4.0**2)
    )  # broad: largest box flux
    stars = ops.find_stars(img, fwhm=4.0, nsigma=20.0, max_n=3)
    assert len(stars) == 3
    assert (
        np.hypot(stars[0, 0] - 480, stars[0, 1] - 480) < 1.0
    )  # ranked first by flux, not cut by the peak pre-cut
