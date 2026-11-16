# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical checks for the workshop's calibration-unit simulation."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_sky import SceneConfig, prepare  # noqa: E402
from scene import (  # noqa: E402
    add_star,
    binary_quality_mask,
    draw_exposure,
    load_exposure,
    normalized_psf,
    prepare_bundle,
    transit_flux,
)


def test_preparation_refuses_existing_data_without_changing_it(tmp_path):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    calibration = provenance / "calibration.json"
    calibration.write_text('{"original": true}\n')
    with pytest.raises(ValueError, match="absent or empty"):
        prepare(tmp_path)
    assert calibration.read_text() == '{"original": true}\n'
    assert sorted(p.name for p in provenance.iterdir()) == [calibration.name]


@pytest.mark.parametrize("scale", [0.25, 0.562, 2.0, 10.0])
@pytest.mark.parametrize("n_average", [1, 16])
def test_calibrated_noise_mean_and_variance(scale, n_average):
    # A Monte Carlo check distinguishes a*M from the reciprocal-gain mistake.
    sample, declared = draw_exposure(
        np.full(200_000, 100.0),
        316.0,
        scale,
        np.random.default_rng(721),
        n_average,
    )
    expected_variance = (316 + scale * 100) / n_average
    assert abs(sample.mean() - 100) < 5 * np.sqrt(
        expected_variance / sample.size
    )
    assert sample.var() / expected_variance == pytest.approx(1, abs=0.012)
    assert declared[0] == pytest.approx(expected_variance)


def test_detected_pixels_are_usable_but_saturation_is_bad():
    raw = np.array([[0, 1 << 5, 1 << 1, (1 << 1) | (1 << 5)]])
    result = binary_quality_mask(
        raw, {"mask_planes": {"DETECTED": 5, "SAT": 1}}
    )
    np.testing.assert_array_equal(result, [[0, 0, 1, 1]])


def test_subpixel_source_preserves_flux_and_centroid():
    yy, xx = np.indices((25, 25))
    psf = normalized_psf(np.exp(-((xx - 12) ** 2 + (yy - 12) ** 2) / 8), 1.2)
    image = np.zeros((128, 128))
    xy = np.array([63.27, 62.61])
    add_star(image, psf, xy, 12345.0)
    yy, xx = np.indices(image.shape)
    assert image.sum() == pytest.approx(12345.0)
    np.testing.assert_allclose(
        [(image * xx).sum() / image.sum(), (image * yy).sum() / image.sum()],
        xy,
        atol=0.002,
    )


def test_geometric_transit_has_ingress_and_52_minute_duration():
    config = SceneConfig()
    times = np.arange(-60, 60, 0.02)
    flux = transit_flux(times, config)
    assert flux[0] == flux[-1] == 1.0
    assert 0.015 < 1 - flux.min() < 0.02
    active = times[flux < 1 - 1e-7]
    assert 51 < active[-1] - active[0] < 53
    assert np.all(np.diff(flux[(times > -24) & (times < -20)]) < 0)


def test_bundle_preserves_astrometry_and_keeps_truth_separate(
    tmp_path, monkeypatch
):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [64.5, 64.5]
    wcs.wcs.crval = [53.0, -28.0]
    wcs.wcs.cdelt = [-0.2 / 3600, 0.2 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    yy, xx = np.indices((25, 25))
    np.savez(
        calibration / "calibration.npz",
        background=np.zeros((128, 128)),
        base_psf=np.exp(-((xx - 12) ** 2 + (yy - 12) ** 2) / 8),
        sky_variance=np.full((128, 128), 316.0),
        noise_scale=np.full((128, 128), 0.562),
        quality_mask=np.zeros((128, 128), dtype=np.uint8),
    )
    fits.PrimaryHDU(header=wcs.to_header()).writeto(
        calibration / "calibration.fits"
    )
    (calibration / "calibration.json").write_text(
        json.dumps(
            {
                "pixel_scale_arcsec": 0.2,
                "quality_mask_is_binary": True,
            }
        )
    )
    config = SceneConfig(n_epochs=8, trail_epoch=3)
    output = tmp_path / "bundle"
    manifest = prepare_bundle(calibration, output, config=config)
    truth = json.loads((output / manifest["truth"]["json"]).read_text())
    old = load_exposure(output / manifest["files"]["reference_old"])
    current = load_exposure(output / manifest["files"]["reference_current"])
    sky_old = WCS(old["header"]).pixel_to_world(*truth["old_target_xy"])
    sky_new = WCS(current["header"]).pixel_to_world(
        *truth["current_target_xy"]
    )
    assert sky_old.separation(sky_new).arcsec == pytest.approx(
        0.948, abs=0.001
    )
    assert current["header"]["SIMULATE"]
    assert current["header"]["NCOMBINE"] == 16
    assert "radius_ratio" not in manifest["assumptions"]
    assert "config" not in manifest["fingerprint"]
    with np.load(output / manifest["truth"]["npz"]) as latent:
        for epoch, xy in zip(
            manifest["epochs"], latent["epoch_target_xy"], strict=True
        ):
            exposure = load_exposure(output / epoch["path"])
            sky_epoch = WCS(exposure["header"]).pixel_to_world(*xy)
            assert sky_epoch.separation(sky_new).arcsec < 1e-6
            assert np.all(exposure["variance"] > 0)
            assert set(np.unique(exposure["mask"])) <= {0, 1}
    assert prepare_bundle(calibration, output, config=config) == manifest
    reference = output / manifest["files"]["reference_old"]
    reference.write_bytes(reference.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_bundle(calibration, output, config=config)

    import scene

    write_exposure = scene._write_exposure

    def fail_after_write(*args, **kwargs):
        write_exposure(*args, **kwargs)
        raise RuntimeError("interrupted regeneration")

    with monkeypatch.context() as patch:
        patch.setattr(scene, "_write_exposure", fail_after_write)
        with pytest.raises(RuntimeError, match="interrupted regeneration"):
            prepare_bundle(calibration, output, config=config, force=True)
    assert not (output / "bundle.json").exists()
    assert prepare_bundle(calibration, output, config=config) == manifest
