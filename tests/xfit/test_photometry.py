# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xfit import forced_psf_photometry

SHAPE = (31, 31)


def _sources(count, *, flux=200.0, sigma=1.8, noise=1.5, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-2.0, 2.0, (count, 2))
    x = np.arange(SHAPE[1]) - (SHAPE[1] - 1) / 2
    y = np.arange(SHAPE[0]) - (SHAPE[0] - 1) / 2
    r2 = (x[None, None, :] - positions[:, 0, None, None]) ** 2 + (
        y[None, :, None] - positions[:, 1, None, None]
    ) ** 2
    clean = flux * np.exp(-0.5 * r2 / sigma**2) / (2 * np.pi * sigma**2)
    variance = np.full((count, *SHAPE), noise**2)
    images = clean + rng.normal(0.0, noise, clean.shape)
    return images, variance, positions


def test_noiseless_flux_is_exact():
    clean, variance, positions = _sources(4, noise=1e-12)
    result = forced_psf_photometry(clean, variance, positions, 1.8)
    assert result.valid.all()
    np.testing.assert_allclose(result.flux, 200.0, rtol=1e-6)


def test_flux_error_is_calibrated():
    images, variance, positions = _sources(4000, seed=1)
    result = forced_psf_photometry(images, variance, positions, 1.8)
    z = (result.flux - 200.0) / result.flux_error
    assert abs(np.mean(np.abs(z) <= 1.959964) - 0.95) < 0.015
    assert abs(np.std(z) - 1.0) < 0.05


def test_masked_pixels_are_excluded():
    images, variance, positions = _sources(3, seed=2)
    mask = np.ones(images.shape, dtype=bool)
    mask[:, :5, :] = False
    corrupted = images.copy()
    corrupted[~mask] = np.nan
    full = forced_psf_photometry(images * mask, variance, positions, 1.8)
    masked = forced_psf_photometry(
        corrupted, variance, positions, 1.8, mask=mask
    )
    assert masked.valid.all()
    assert np.all(masked.flux_error >= full.flux_error)
    np.testing.assert_allclose(masked.flux, full.flux, rtol=1e-3)


def test_invalid_rows_return_nan():
    images, variance, positions = _sources(3, seed=3)
    variance[1] = np.nan
    result = forced_psf_photometry(
        images, variance, positions, np.array([1.8, 1.8, 0.0])
    )
    assert result.valid.tolist() == [True, False, False]
    assert np.isnan(result.flux[1:]).all()
    assert np.isnan(result.flux_error[1:]).all()


def test_shape_validation():
    images, variance, positions = _sources(2)
    with pytest.raises(ValueError, match="positions"):
        forced_psf_photometry(images, variance, positions[:1], 1.8)
    with pytest.raises(ValueError, match="variance"):
        forced_psf_photometry(images, variance[:1], positions, 1.8)


def test_cupy_matches_numpy():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("no visible CUDA device")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    images, variance, positions = _sources(16, seed=4)
    host = forced_psf_photometry(images, variance, positions, 1.8)
    device = forced_psf_photometry(
        cp.asarray(images), cp.asarray(variance), cp.asarray(positions), 1.8
    )
    assert isinstance(device.flux, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(device.flux), host.flux)
    np.testing.assert_allclose(cp.asnumpy(device.flux_error), host.flux_error)


def test_supplied_psf_stamps_replace_the_gaussian():
    images, variance, positions = _sources(3, seed=5)
    x = np.arange(SHAPE[1]) - (SHAPE[1] - 1) / 2
    y = np.arange(SHAPE[0]) - (SHAPE[0] - 1) / 2
    r2 = (x[None, None, :] - positions[:, 0, None, None]) ** 2 + (
        y[None, :, None] - positions[:, 1, None, None]
    ) ** 2
    psf = np.exp(-0.5 * r2 / 1.8**2) / (2 * np.pi * 1.8**2)
    built = forced_psf_photometry(images, variance, positions, 1.8)
    given = forced_psf_photometry(images, variance, positions, 1.8, psf=psf)
    np.testing.assert_allclose(given.flux, built.flux, rtol=1e-12)
    np.testing.assert_allclose(given.flux_error, built.flux_error, rtol=1e-12)
    wider = forced_psf_photometry(
        images, variance, positions, 1.8, psf=np.roll(psf, 3, axis=2)
    )
    assert not np.allclose(wider.flux, built.flux)
    with pytest.raises(ValueError, match="psf"):
        forced_psf_photometry(images, variance, positions, 1.8, psf=psf[:1])
