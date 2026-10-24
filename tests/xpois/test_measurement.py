# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from cuphoton.xfit import forced_psf_photometry
from cuphoton.xpois import (
    empty_positions,
    forced_flux_at,
    kernel_centroid,
    psf_weights,
    window_stamps,
)
from cuphoton.xpois.measurement import _loud_pixels

SHAPE = (60, 80)
WINDOW = 7


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - no CUDA runtime
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    return cp


def _scene(seed=0, n_sources=6, flux=500.0, sigma=1.6, noise=1.0, sky=0.0):
    """A plane of Gaussian sources on a flat sky with white noise."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    xy = rng.uniform(10, 50, (n_sources, 2))
    plane = np.full(SHAPE, sky)
    for x0, y0 in xy:
        plane += (
            flux
            * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma**2)
            / (2 * np.pi * sigma**2)
        )
    plane += rng.normal(0.0, noise, SHAPE)
    return plane, np.full(SHAPE, noise**2), xy, sigma


def test_kernel_centroid_is_the_first_moment():
    k = np.zeros((7, 9))
    k[3, 6] = 2.0
    assert kernel_centroid(k) == pytest.approx((2.0, 0.0))
    k = np.zeros((7, 9))
    k[2, 5] = 1.0
    assert kernel_centroid(k) == pytest.approx((1.0, -1.0))
    g = psf_weights(1.3, 11)
    assert kernel_centroid(g) == pytest.approx((0.0, 0.0), abs=1e-12)
    # two sources of flux 1 and 0.5 at offsets 0 and 2 px, zero padded
    two = np.pad(g, 3) + 0.5 * np.pad(g, ((3, 3), (5, 1)))
    assert kernel_centroid(two) == pytest.approx((1.0 / 1.5, 0.0), abs=1e-9)


def test_kernel_centroid_of_a_zero_or_non_finite_sum_is_nan():
    k = np.zeros((5, 5))
    k[2, 1], k[2, 3] = 1.0, -1.0
    assert all(math.isnan(v) for v in kernel_centroid(k))
    k[2, 3] = np.nan
    assert all(math.isnan(v) for v in kernel_centroid(k))
    assert kernel_centroid(np.ones((3, 3))) == pytest.approx((0.0, 0.0))
    with pytest.raises(ValueError, match="two-dimensional"):
        kernel_centroid(np.ones(5))


def test_kernel_centroid_on_the_device_matches_the_host():
    cp = _cupy()
    k = psf_weights(1.3, 11) + 0.5 * np.roll(psf_weights(1.3, 11), 2, axis=1)
    assert kernel_centroid(cp.asarray(k)) == pytest.approx(kernel_centroid(k))


def _hand_selection(valid, sources, window, count, loud_plane=None):
    near = np.zeros(valid.shape, bool)
    near[sources[:, 1].astype(int), sources[:, 0].astype(int)] = True
    reach = 2 * window + 1
    empty = valid & ~maximum_filter(near, size=reach)
    if loud_plane is not None:
        empty &= ~maximum_filter(loud_plane, size=reach)
    flat = np.flatnonzero(empty)
    take = min(count, len(flat))
    return flat, flat[
        np.rint(np.linspace(0, len(flat) - 1, take)).astype(int)
    ]


def test_empty_positions_are_evenly_spaced_usable_pixels_away_from_sources():
    """The positions are usable pixels more than one window from every
    excluded position, taken evenly over the empty pixels in raster order,
    up to the number requested; off-image and non-finite exclusions
    exclude nothing."""

    rng = np.random.default_rng(1)
    valid = rng.uniform(size=SHAPE) > 0.05
    valid[20:30, 40:55] = False
    sources = np.rint(rng.uniform(5, 55, (12, 2)))
    given = np.vstack([sources + 0.3, [[-5.0, 10.0], [np.nan, 3.0]]])
    pos, n_loud = empty_positions(
        valid, exclude_xy=given, window=WINDOW, count=150
    )
    assert pos.shape == (150, 2) and pos.dtype == np.float64 and n_loud == 0
    ix, iy = pos[:, 0].astype(int), pos[:, 1].astype(int)
    np.testing.assert_array_equal(pos, np.column_stack([ix, iy]))
    assert valid[iy, ix].all()
    for x, y in zip(ix, iy, strict=True):
        assert np.abs(sources - (x, y)).max(axis=1).min() > WINDOW
    flat, want = _hand_selection(valid, sources, WINDOW, 150)
    np.testing.assert_array_equal(iy * SHAPE[1] + ix, want)
    # more requested than there are: every empty pixel, in raster order
    every, _ = empty_positions(
        valid, exclude_xy=sources, window=WINDOW, count=10 * len(flat)
    )
    assert every.shape == (len(flat), 2)
    np.testing.assert_array_equal(every[:, 1] * SHAPE[1] + every[:, 0], flat)
    # no excluded position at all, and no usable pixel at all
    free, _ = empty_positions(
        valid, exclude_xy=np.zeros((0, 2)), window=WINDOW, count=5
    )
    assert free.shape == (5, 2)
    none, _ = empty_positions(
        np.zeros(SHAPE, bool), exclude_xy=sources, window=WINDOW, count=5
    )
    assert none.shape == (0, 2)
    assert empty_positions(valid, exclude_xy=sources, window=WINDOW, count=0)[
        0
    ].shape == (0, 2)


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (5, 0)])
def test_empty_positions_accepts_planes_with_no_pixels(shape):
    positions, n_loud = empty_positions(
        np.zeros(shape, dtype=bool),
        exclude_xy=np.empty((0, 2)),
        window=3,
        count=4,
    )
    assert positions.shape == (0, 2)
    assert positions.dtype == np.float64
    assert n_loud == 0


def test_loud_pixels_and_their_surroundings_are_not_empty():
    """A hot pixel removes the usable pixels within one window of it and
    the count says how many; a flat residual marks nothing."""

    rng = np.random.default_rng(2)
    valid = np.ones(SHAPE, bool)
    residual = rng.normal(size=SHAPE)
    quiet, n_quiet = empty_positions(
        valid,
        exclude_xy=np.zeros((0, 2)),
        window=WINDOW,
        count=5000,
        loud=(residual, 5.0),
    )
    assert n_quiet == 0 and quiet.shape[0] == valid.sum()
    residual[30, 40] = 1000.0
    pos, n_loud = empty_positions(
        valid,
        exclude_xy=np.zeros((0, 2)),
        window=WINDOW,
        count=5000,
        loud=(residual, 5.0),
    )
    assert n_loud == (2 * WINDOW + 1) ** 2
    assert pos.shape[0] == valid.sum() - n_loud
    assert np.abs(pos - (40, 30)).max(axis=1).min() > WINDOW
    loud_plane = _loud_pixels(np, residual, valid, 5.0)
    assert loud_plane.sum() == 1 and loud_plane[30, 40]
    _, want = _hand_selection(
        valid, np.zeros((0, 2)), WINDOW, 5000, loud_plane
    )
    np.testing.assert_array_equal(pos[:, 1] * SHAPE[1] + pos[:, 0], want)
    # pixels already excluded by a position are not counted as loud
    _, n_both = empty_positions(
        valid,
        exclude_xy=np.array([[40.0, 30.0]]),
        window=WINDOW,
        count=10,
        loud=(residual, 5.0),
    )
    assert n_both == 0
    # a zero spread marks nothing, as does an all-NaN residual
    assert not _loud_pixels(np, np.ones(SHAPE), valid, 5.0).any()
    assert not _loud_pixels(np, np.full(SHAPE, np.nan), valid, 5.0).any()


def test_empty_positions_argument_validation():
    valid = np.ones(SHAPE, bool)
    with pytest.raises(ValueError, match="odd"):
        empty_positions(valid, exclude_xy=np.zeros((0, 2)), window=4, count=1)
    with pytest.raises(ValueError, match="exclude_xy"):
        empty_positions(valid, exclude_xy=np.zeros((3, 3)), window=5, count=1)
    with pytest.raises(ValueError, match="two-dimensional"):
        empty_positions(
            valid[0], exclude_xy=np.zeros((0, 2)), window=5, count=1
        )
    with pytest.raises(ValueError, match="negative"):
        empty_positions(
            valid, exclude_xy=np.zeros((0, 2)), window=5, count=-1
        )


def test_empty_positions_on_the_device_match_the_host():
    cp = _cupy()
    rng = np.random.default_rng(3)
    valid = rng.uniform(size=SHAPE) > 0.05
    sources = rng.uniform(5, 55, (12, 2))
    residual = rng.normal(size=SHAPE)
    residual[10, 70] = 500.0
    kwargs = dict(window=WINDOW, count=200)
    host = empty_positions(
        valid, exclude_xy=sources, loud=(residual, 5.0), **kwargs
    )
    device = empty_positions(
        cp.asarray(valid),
        exclude_xy=sources,
        loud=(cp.asarray(residual), 5.0),
        **kwargs,
    )
    assert type(device[0]).__module__ == "cupy"
    np.testing.assert_array_equal(cp.asnumpy(device[0]), host[0])
    assert device[1] == host[1] > 0


def test_forced_flux_at_recovers_planted_fluxes():
    plane, variance, xy, sigma = _scene()
    result = forced_flux_at(
        plane,
        xy,
        sigma,
        window=2 * int(math.ceil(5 * sigma)) + 1,
        variance=variance,
        local_sky=False,
    )
    assert result.valid.all()
    np.testing.assert_allclose(result.flux, 500.0, rtol=0.03)
    assert np.all(np.abs(result.flux - 500.0) < 4 * result.flux_error)
    # local sky: the window median takes a constant offset of the plane
    # back out (and a little of the source's wings with it)
    offset = forced_flux_at(
        plane + 40.0, xy, sigma, window=17, variance=variance, local_sky=True
    )
    raw = forced_flux_at(
        plane + 40.0, xy, sigma, window=17, variance=variance, local_sky=False
    )
    np.testing.assert_allclose(offset.flux, result.flux, rtol=0.05)
    assert np.all(raw.flux > result.flux + 500.0)


@pytest.mark.parametrize("invalid_variance", [0.0, -1.0, np.nan, np.inf])
def test_forced_flux_local_sky_ignores_unusable_pixels(invalid_variance):
    plane = 100.0 * psf_weights(1.0, 5) + 10.0
    variance = np.ones_like(plane)
    variance[:3] = invalid_variance
    positions = np.array([[2.0, 2.0]])
    plane[:3] = np.nan
    expected = forced_flux_at(
        plane, positions, 1.0, window=5, variance=variance
    )
    plane[:3] = 1e6
    actual = forced_flux_at(
        plane, positions, 1.0, window=5, variance=variance
    )
    assert actual.valid.all()
    np.testing.assert_allclose(actual.flux, expected.flux)
    np.testing.assert_allclose(actual.flux_error, expected.flux_error)


def test_forced_flux_at_is_the_xfit_estimator_on_cut_windows():
    """The same numbers as ``forced_psf_photometry`` on windows cut by
    ``window_stamps`` with ``psf_weights`` at the sub-pixel offsets; without
    a variance the weights are one; a non-finite position is NaN."""

    plane, variance, xy, sigma = _scene(seed=4)
    plane[12:14, :] = np.nan
    window = 15
    xy = np.vstack([xy, [[np.nan, 20.0], [2.0, 2.0]]])
    result = forced_flux_at(
        plane, xy, sigma, window=window, variance=variance, local_sky=False
    )
    offset = xy - np.rint(xy)
    at = np.where(np.isfinite(xy), xy, 0.0)
    expected = forced_psf_photometry(
        window_stamps(plane, at, window),
        window_stamps(variance, at, window),
        offset,
        sigma,
        psf=psf_weights(sigma, window, offset=offset),
    )
    np.testing.assert_array_equal(result.flux, expected.flux)
    np.testing.assert_array_equal(result.flux_error, expected.flux_error)
    np.testing.assert_array_equal(result.valid, expected.valid)
    assert not result.valid[-2] and np.isnan(result.flux[-2])
    assert result.valid[-1]  # a window partly off the image keeps its pixels
    unit = forced_flux_at(
        plane, xy[:3], sigma, window=window, local_sky=False
    )
    weighted = forced_psf_photometry(
        window_stamps(plane, xy[:3], window),
        np.ones((3, window, window)),
        offset[:3],
        sigma,
    )
    np.testing.assert_allclose(unit.flux, weighted.flux, rtol=1e-12)
    with pytest.raises(ValueError, match="positions"):
        forced_flux_at(plane, xy[:, :1], sigma, window=window)


def test_forced_flux_at_on_the_device_matches_the_host():
    cp = _cupy()
    plane, variance, xy, sigma = _scene(seed=5)
    host = forced_flux_at(plane, xy, sigma, window=15, variance=variance)
    device = forced_flux_at(
        cp.asarray(plane),
        cp.asarray(xy),
        sigma,
        window=15,
        variance=cp.asarray(variance),
    )
    assert type(device.flux).__module__ == "cupy"
    np.testing.assert_allclose(cp.asnumpy(device.flux), host.flux, rtol=1e-12)
    np.testing.assert_allclose(
        cp.asnumpy(device.flux_error), host.flux_error, rtol=1e-12
    )
