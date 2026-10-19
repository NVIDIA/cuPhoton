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


def test_match_stars_keeps_only_the_closest_pair_per_template_star():
    visit = np.array([[10.4, 10.0], [10.1, 10.0], [20.1, 20.0], [99, 99]])
    template = np.array([[10.0, 10.0], [20.0, 20.0]])
    matched_visit, matched_template = ops.match_stars(visit, template, 1.0)
    np.testing.assert_array_equal(matched_visit, visit[[1, 2]])
    np.testing.assert_array_equal(matched_template, template)


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
    var = np.full((200, 200), 4.0)
    var[60, 52] = 0.0
    stamps, diff, valid, vstamps = ops.extract_stamps(
        search, resid, np.array([50, 120]), np.array([50, 130]), 51, var
    )
    assert valid.dtype == bool and vstamps.dtype == np.float64
    assert not valid[0, 25, 25] and not valid[0, 35, 27]
    assert valid.sum() == 2 * 51 * 51 - 2
    _, _, valid_only, none = ops.extract_stamps(
        search, resid, np.array([50]), np.array([50]), 51
    )
    assert none is None and valid_only.sum() == 51 * 51 - 1
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


def _render(size, xs, ys, fluxes, sigma, sky=400.0):
    """Pixel-integrated circular Gaussians on a flat sky."""
    from scipy.special import erf

    img = np.full((size, size), sky)
    edges = np.arange(size + 1) - 0.5
    for x, y, f in zip(xs, ys, fluxes, strict=True):
        cx = 0.5 * np.diff(erf((edges - x) / (np.sqrt(2) * sigma)))
        cy = 0.5 * np.diff(erf((edges - y) / (np.sqrt(2) * sigma)))
        img += f * np.outer(cy, cx)
    return img


STAR_FINDER = dict(fwhm=4.0, nsigma=20.0, max_n=600, box=15, iterations=4)


def test_star_centroid_error_matches_monte_carlo():
    clean = _render(64, [31.3], [32.6], [2.0e4], 2.0)
    var = clean + 25.0
    rng = np.random.default_rng(1)
    xs, us = [], []
    for _ in range(300):
        img = clean + rng.normal(0.0, np.sqrt(var))
        c = ops.find_stars(img, **STAR_FINDER)
        assert len(c) == 1
        xs.append(c[0])
        us.append(
            ops.star_measurements(img, var, c, box=15).centroid_error[0]
        )
    scatter, pred = (
        np.std(np.array(xs), axis=0),
        np.mean(np.array(us), axis=0),
    )
    np.testing.assert_allclose(scatter, pred, rtol=0.15)


def test_star_width_is_the_pixel_level_gaussian_width():
    img = _render(64, [31.3], [32.6], [1.0e5], 2.0, sky=0.0)
    m = ops.star_measurements(
        img, np.ones_like(img), np.array([[31.3, 32.6]]), box=15
    )
    assert m.width[0] == pytest.approx(np.sqrt(4.0 + 1.0 / 12.0), rel=2e-3)
    assert m.snr[0] > 100 and m.flux[0] == pytest.approx(1.0e5, rel=1e-2)
    none = ops.star_measurements(img, None, np.array([[31.3, 32.6]]), box=15)
    assert np.all(none.centroid_error == 0) and np.isnan(none.snr[0])


def _field(xy, d=0.2, half=256.0, c=255.5):
    X, Y = (xy[:, 0] - c) / half, (xy[:, 1] - c) / half
    return np.column_stack([d * (X**2 - Y**2), d * 2 * X * Y])


def test_fit_misalignment_recovers_the_quadratic_field_and_its_variance():
    rng = np.random.default_rng(2)
    A, b = (
        np.array([[1.0002, -0.0003], [0.0001, 0.9998]]),
        np.array([0.3, -0.2]),
    )
    src = rng.uniform(20, 490, (200, 2))
    noise = np.full((200, 2), 0.01**2)
    dst = src @ A.T + b + _field(src) + rng.normal(0, 0.01, src.shape)
    m = ops.fit_misalignment(src, dst, noise, shape=(512, 512))
    assert m.mode == "polynomial" and m.used.all()
    np.testing.assert_allclose(m.matrix, A, atol=2e-4)
    probe = rng.uniform(40, 470, (50, 2))
    true = probe @ A.T + b + _field(probe) - (probe @ m.matrix.T + m.offset)
    p, vp = m.excess(probe[:, 0], probe[:, 1])
    assert np.max(np.abs(p - true)) < 0.01
    assert np.all(m.excess_variance < 0.003**2)
    # held-out stars: measured scatter around the affine agrees with the
    # model's variance plus the centroid noise
    held = rng.uniform(20, 490, (2000, 2))
    meas = held @ A.T + b + _field(held) + rng.normal(0, 0.01, held.shape)
    e = meas - (held @ m.matrix.T + m.offset)
    z = e / np.sqrt(m.variance(held[:, 0], held[:, 1]) + 0.01**2)
    assert np.std(z) == pytest.approx(1.0, abs=0.15)
    var = m.variance(probe[:, 0], probe[:, 1])
    aff = m.affine_variance(probe[:, 0], probe[:, 1])
    np.testing.assert_allclose(
        var, aff + p**2 + m.excess_variance, rtol=1e-12
    )


def test_fit_misalignment_rms_mode_and_clipping():
    rng = np.random.default_rng(3)
    src = rng.uniform(20, 490, (20, 2))
    noise = np.full((20, 2), 0.01**2)
    dst = src + rng.normal(0, 0.02, src.shape)
    dst[0] += 1.0  # a blended pair
    m = ops.fit_misalignment(src, dst, noise, shape=(512, 512))
    assert m.mode == "rms" and not m.used[0] and m.used[1:].all()
    assert np.sqrt(m.excess_variance).mean() == pytest.approx(
        np.sqrt(3) * 0.01, rel=0.6
    )
    var = m.variance([100.0, 400.0], [100.0, 400.0])
    assert var.shape == (2, 2) and np.all(var >= m.excess_variance)
    p, vp = m.excess([100.0], [100.0])
    assert np.all(p == 0) and np.all(vp == 0)
    assert ops.fit_misalignment(
        src, dst, noise, shape=(512, 512), clip=None
    ).used.all()
    assert (
        ops.fit_misalignment(
            src, dst, np.zeros_like(noise), shape=(512, 512)
        ).excess_variance.min()
        > 0
    )


def test_affine_covariance_matches_monte_carlo():
    rng = np.random.default_rng(4)
    src = rng.uniform(20, 490, (40, 2))
    sig = rng.uniform(0.005, 0.03, (40, 2))
    probe = np.array([[255.5, 255.5], [40.0, 470.0]])
    preds = []
    for _ in range(400):
        A, b, _ = ops.fit_affine(
            src, src + sig * rng.standard_normal(src.shape)
        )
        preds.append(probe @ A.T + b - probe)
    m = ops.fit_misalignment(
        src, src, sig**2, shape=(512, 512), min_poly=10**9
    )
    np.testing.assert_allclose(
        np.var(np.array(preds), axis=0),
        m.affine_variance(probe[:, 0], probe[:, 1]),
        rtol=0.2,
    )


def test_neighbor_free_drops_stars_with_a_close_peak():
    xs, ys = [40.0, 150.0, 158.0, 100.0], [40.0, 150.0, 152.0, 160.0]
    img = _render(200, xs, ys, [3e4, 3e4, 1e4, 3e4], 1.7, sky=0.0)
    img += np.random.default_rng(5).normal(0, 1.0, img.shape)
    xy = np.column_stack([xs, ys])
    np.testing.assert_array_equal(
        ops.neighbor_free(img, xy, 15.0), [True, False, False, True]
    )
    assert ops.neighbor_free(img, np.empty((0, 2)), 15.0).shape == (0,)


def test_psf_width_clips_and_reports_a_standard_error():
    rng = np.random.default_rng(5)
    w = np.concatenate([rng.normal(2.02, 0.01, 60), [3.5, 4.0, 5.0, np.nan]])
    r = ops.psf_width(w)
    assert r.n == 63 and r.n_used == 60
    assert r.sigma_px == pytest.approx(2.02, abs=0.005)
    assert r.standard_error_px == pytest.approx(
        1.2533 * 0.01 / np.sqrt(60), rel=0.3
    )
    assert r.optical_sigma_px == pytest.approx(
        np.sqrt(r.sigma_px**2 - 1 / 12)
    )
    with pytest.raises(ValueError):
        ops.psf_width([np.nan])
