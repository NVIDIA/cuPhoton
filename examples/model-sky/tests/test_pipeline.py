# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioral controls for the Model Sky measurements."""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline  # noqa: E402


def test_trail_mask_leaves_a_localized_galaxy_alone():
    yy, xx = np.indices((240, 240))
    rng = np.random.default_rng(41)
    galaxy = 80 * np.exp(-(((xx - 65) / 18) ** 2) - ((yy - 100) / 2) ** 2)
    galaxy += rng.normal(size=galaxy.shape)
    mask, _ = pipeline.detect_trail(galaxy, np.ones_like(galaxy))
    assert not mask.any()
    trail = 30 * np.exp(-(((yy - 0.45 * xx - 45) / 5) ** 2))
    mask, result = pipeline.detect_trail(trail + galaxy, np.ones_like(galaxy))
    assert result["detected"]
    assert mask[99, 120]
    assert not mask[200, 20]


@pytest.mark.parametrize("exposure_s,eccentricity", [(30, 0), (180, 0.12)])
def test_physical_transit_depth_and_time_recovery(exposure_s, eccentricity):
    from model_sky import SceneConfig
    from scene import transit_flux

    times = np.arange(120) - 59.5
    rng = np.random.default_rng(18)
    config = SceneConfig(
        radius_ratio=0.116, exposure_s=exposure_s, eccentricity=eccentricity
    )
    shape = transit_flux(times - 3.5, config)
    error = np.full(120, 0.001)
    data = shape + rng.normal(scale=error)
    fit = pipeline.fit_transit(
        times,
        data,
        error,
        np.ones(120, bool),
        dict(
            exposure_s=exposure_s,
            eccentricity=eccentricity,
            limb_darkening=[
                config.limb_darkening_u1,
                config.limb_darkening_u2,
            ],
        ),
    )
    assert abs(fit["radius_ratio"] - 0.116) < 3 * fit["radius_ratio_error"]
    assert abs(fit["midtime_min"] - 3.5) < 3 * fit["midtime_error_min"]


def test_rejects_insufficient_photometry():
    with pytest.raises(ValueError, match="not enough"):
        pipeline.fit_transit(
            np.arange(10), np.ones(10), np.ones(10), np.ones(10, bool), {}
        )


def test_cuda_motion_recovers_shift_with_broader_seeing():
    pytest.importorskip("cupy")
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [53, -28]
    wcs.wcs.crpix = [128, 128]
    wcs.wcs.cdelt = [-0.2 / 3600, 0.2 / 3600]
    rng = np.random.default_rng(8)
    yy, xx = np.indices((256, 256))
    old = np.zeros((256, 256))
    for x, y, flux in [
        (60, 60, 8000),
        (120, 60, 6000),
        (190, 60, 5000),
        (60, 190, 4000),
        (190, 190, 3000),
    ]:
        old += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 8)
    new = old.copy()
    old += 5000 * np.exp(-((xx - 130) ** 2 + (yy - 130) ** 2) / 8)
    new += 5000 * np.exp(-((xx - 135) ** 2 + (yy - 132) ** 2) / 8)
    new = gaussian_filter(new, 1)

    def exposure(data):
        return {
            "science": data + rng.normal(0, 10, data.shape),
            "variance": np.full(data.shape, 100.0),
            "mask": np.zeros(data.shape, dtype="uint32"),
            "header": wcs.to_header(),
        }

    result = pipeline.measure_motion(
        exposure(old), exposure(new), np.array([130.0, 130.0])
    )
    assert result["backend"] == "cupy"
    np.testing.assert_allclose(result["new_xy"], [135, 132], atol=0.1)
    np.testing.assert_allclose(result["old_xy"], [130, 130], atol=0.1)
    assert abs(result["separation_pixels"] - np.hypot(5, 2)) < 0.1


def test_fully_masked_forced_fit_is_invalid():
    cp = pytest.importorskip("cupy")
    from cuphoton.xfit import forced_psf_photometry

    result = forced_psf_photometry(
        cp.ones((1, 21, 21)),
        cp.ones((1, 21, 21)),
        cp.zeros((1, 2)),
        2,
        mask=cp.zeros((1, 21, 21), dtype=bool),
    )
    assert not result.valid.item()
    assert cp.isnan(result.flux).item()
