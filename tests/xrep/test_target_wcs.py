# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS, DistortionLookupTable, Sip

from cuphoton.core.cli import run_component
from cuphoton.xrep import (
    BBox,
    Grid,
    ReprojectionSpec,
    make_north_up_wcs,
    make_wcs_pair_transform,
    reproject_array,
)
from cuphoton.xrep.backends import get_backend
from cuphoton.xrep.geometry import bbox_wcs
from cuphoton.xrep.workflows import run_reproject_stack


def _wcs(shape=(17, 21)):
    return make_north_up_wcs((150.0, 2.0), shape, 0.2)


def _write_image(path, image, wcs):
    fits.PrimaryHDU(image, header=wcs.to_header(relax=True)).writeto(path)
    return path


def _run(tmp_path, source, target, **kwargs):
    options = {
        "input_paths": [source],
        "target_wcs_path": target,
        "output_root": tmp_path / "runs",
        "name": "target",
        "hdu": 0,
        "backend": "cpu",
        "interpolation": "bilinear",
        "grid_crval_ra": None,
        "grid_crval_dec": None,
        "pixel_scale_arcsec": None,
        "mapping_grid_step": 1,
        "area_scaling": False,
        "write_fits": True,
    }
    options.update(kwargs)
    return run_reproject_stack(**options)


def _sip(wcs):
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    a[2, 0] = 0.002
    b[0, 2] = -0.003
    wcs.sip = Sip(a, b, None, None, wcs.wcs.crpix)
    wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    return wcs


@pytest.mark.parametrize(
    "kind", ["identity", "rotated", "sip", "partial", "none"]
)
@pytest.mark.parametrize("area_scaling", [False, True])
def test_target_grid_matches_direct_mapping(tmp_path, kind, area_scaling):
    image = np.arange(17 * 21, dtype=np.float64).reshape(17, 21)
    source_wcs = _wcs()
    target_wcs = source_wcs.deepcopy()
    shape = image.shape
    if kind == "rotated":
        angle = np.deg2rad(25.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        target_wcs.wcs.cd = source_wcs.wcs.cd @ rotation
        target_wcs.wcs.crpix += [0.25, -0.6]
        shape = (19, 23)
    elif kind == "sip":
        target_wcs = _sip(target_wcs)
    elif kind == "partial":
        target_wcs.wcs.crpix += [10, 0]
    elif kind == "none":
        target_wcs.wcs.crpix += [100, 0]
    source = _write_image(tmp_path / "source.fits", image, source_wcs)
    target = _write_image(
        tmp_path / "target.fits", np.zeros(shape), target_wcs
    )

    result = _run(tmp_path, source, target, area_scaling=area_scaling)
    stack = np.load(result.run_dir / result.summary["saved"]["stack"])
    expected = reproject_array(
        image,
        ReprojectionSpec(
            mapping=make_wcs_pair_transform(target_wcs, source_wcs),
            output_bbox=BBox(0, 0, shape[1], shape[0]),
            interpolation="bilinear",
            mapping_grid_step=1,
            area_scaling=area_scaling,
        ),
        backend="cpu",
    )
    assert stack.shape == (1, *shape)
    np.testing.assert_allclose(stack[0], expected.image, atol=1e-7, rtol=1e-8)
    masks = np.load(result.run_dir / result.summary["saved"]["stack_mask"])
    np.testing.assert_array_equal(masks[0], expected.mask)
    if kind == "identity":
        np.testing.assert_allclose(
            stack[0, 2:-2, 2:-2], image[2:-2, 2:-2], atol=1e-7
        )
    elif kind == "partial":
        assert np.isnan(stack).any() and np.isfinite(stack).any()
    elif kind == "none":
        assert np.isnan(stack).all() and masks.all()
    assert result.summary["output_bbox"] == {
        "min_x": 0,
        "min_y": 0,
        "height": shape[0],
        "width": shape[1],
    }
    provenance = result.summary["target_wcs"]
    assert provenance["hdu"] == 0
    assert provenance["shape"] == list(shape)
    points = [[0, 0], [5.3, 8.1], [shape[1] - 1, shape[0] - 1]]
    with fits.open(
        result.run_dir / result.summary["saved"]["stack_fits"]
    ) as hdul:
        assert hdul[0].data.shape == stack.shape
        saved_wcs = WCS(hdul[0].header, naxis=2)
        np.testing.assert_allclose(
            saved_wcs.all_pix2world(points, 0),
            target_wcs.all_pix2world(points, 0),
            atol=1e-12,
            rtol=0,
        )
        assert (saved_wcs.sip is not None) == (kind == "sip")
    header_wcs = WCS(fits.Header.fromstring(provenance["wcs_header"]))
    np.testing.assert_allclose(
        header_wcs.all_pix2world(points, 0),
        target_wcs.all_pix2world(points, 0),
        atol=1e-12,
        rtol=0,
    )


def test_target_hdu_cli_preserves_shape(tmp_path: Path, capsys):
    source = _write_image(tmp_path / "source.fits", np.ones((17, 21)), _wcs())
    target = tmp_path / "target.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((7, 9)), header=_wcs((7, 9)).to_header()),
            fits.ImageHDU(
                np.zeros((11, 13)), header=_wcs((11, 13)).to_header()
            ),
        ]
    ).writeto(target)
    rc = run_component(
        "xrep",
        [
            "reproject-stack",
            "--inputs",
            f"{source},{source}",
            "--target-wcs",
            str(target),
            "--target-hdu",
            "2",
            "--backend",
            "cpu",
            "--mapping-grid-step",
            "1",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stack_shape"] == [2, 11, 13]
    assert summary["target_wcs"]["hdu"] == 2


@pytest.mark.parametrize(
    "option", ["grid_crval_ra", "grid_crval_dec", "pixel_scale_arcsec"]
)
def test_target_rejects_synthetic_grid_options(tmp_path, option):
    source = _write_image(tmp_path / "source.fits", np.ones((17, 21)), _wcs())
    with pytest.raises(ValueError, match="conflicts"):
        _run(tmp_path, source, source, **{option: 1.0})
    assert not (tmp_path / "runs").exists()


def test_target_hdu_requires_target_path(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        _run(tmp_path, tmp_path / "source.fits", None, target_hdu=0)


@pytest.mark.parametrize("hdu", [0, 2])
def test_target_rejects_empty_or_absent_hdu(tmp_path, hdu):
    target = tmp_path / "target.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((2, 2)))]).writeto(
        target
    )
    with pytest.raises(ValueError, match="not a 2D image|out of range"):
        _run(tmp_path, tmp_path / "source.fits", target, target_hdu=hdu)


def test_target_requires_celestial_wcs(tmp_path):
    target = _write_image(
        tmp_path / "target.fits", np.ones((2, 2)), WCS(naxis=2)
    )
    with pytest.raises(ValueError, match="celestial"):
        _run(tmp_path, tmp_path / "source.fits", target)


def test_grid_from_wcs_copies_and_shifts_sip_reference():
    wcs = _sip(_wcs())
    grid = Grid.from_wcs(wcs)
    wcs.wcs.crpix += [20, 30]
    box = BBox(-4, 3, 10, 9)
    local = bbox_wcs(grid, box)
    points = np.array([[0.0, 0.0], [3.25, 4.75]])
    np.testing.assert_allclose(
        local.all_pix2world(points, 0),
        grid.wcs.all_pix2world(points + box.origin, 0),
        atol=1e-12,
        rtol=0,
    )
    assert local.array_shape == box.shape
    assert not np.array_equal(grid.wcs.wcs.crpix, wcs.wcs.crpix)


def test_target_rejects_lookup_table_distortion():
    wcs = _wcs()
    wcs.cpdis1 = DistortionLookupTable(
        np.zeros((2, 2), dtype=np.float32), (1, 1), (1, 1), (1, 1)
    )
    with pytest.raises(ValueError, match="lookup-table"):
        Grid.from_wcs(wcs)


def test_target_sip_with_coarse_grid_warns(tmp_path):
    image = np.ones((17, 21), dtype=np.float64)
    source = _write_image(tmp_path / "source.fits", image, _wcs())
    target = _write_image(tmp_path / "target.fits", image, _sip(_wcs()))
    with pytest.warns(RuntimeWarning, match="SIP distortion"):
        _run(tmp_path, source, target, mapping_grid_step=4)


@pytest.mark.parametrize("sip, step", [(True, 1), (False, 4)])
def test_target_exact_or_undistorted_grid_does_not_warn(tmp_path, sip, step):
    image = np.ones((17, 21), dtype=np.float64)
    source = _write_image(tmp_path / "source.fits", image, _wcs())
    target_wcs = _sip(_wcs()) if sip else _wcs()
    target = _write_image(tmp_path / "target.fits", image, target_wcs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _run(tmp_path, source, target, mapping_grid_step=step)


@pytest.mark.parametrize("backend", ["torch", "cupy"])
def test_target_sip_backend_parity(tmp_path, backend, monkeypatch):
    monkeypatch.setenv("CUPHOTON_XREP_TORCH_DEVICE", "cpu")
    if not get_backend(backend).is_available():
        pytest.skip(f"{backend} backend unavailable")
    image = np.arange(17 * 21, dtype=np.float64).reshape(17, 21)
    source = _write_image(tmp_path / "source.fits", image, _wcs())
    target_wcs = _sip(_wcs())
    target_wcs.wcs.crpix += [0.3, -0.4]
    target = _write_image(tmp_path / "target.fits", image, target_wcs)
    reference = _run(tmp_path, source, target, name="cpu", area_scaling=True)
    candidate = _run(
        tmp_path,
        source,
        target,
        name=backend,
        backend=backend,
        area_scaling=True,
    )
    for key in ("stack", "stack_mask"):
        expected = np.load(
            reference.run_dir / reference.summary["saved"][key]
        )
        actual = np.load(candidate.run_dir / candidate.summary["saved"][key])
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-10)


@pytest.mark.parametrize(
    "units",
    [("deg", "deg"), ("arcsec", "arcsec"), ("rad", "rad"), ("arcsec", "rad")],
)
def test_grid_metadata_normalizes_preserved_wcs_units(units):
    header = _wcs().to_header()
    for axis, unit in enumerate(units, start=1):
        factor = u.deg.to(u.Unit(unit))
        header[f"CUNIT{axis}"] = unit
        header[f"CRVAL{axis}"] *= factor
        header[f"CDELT{axis}"] *= factor
    wcs = WCS(header, preserve_units=True)
    grid = Grid.from_wcs(wcs)
    np.testing.assert_allclose(grid.crval, [150, 2], atol=1e-12, rtol=0)
    assert grid.pixel_scale_arcsec == pytest.approx(0.2, abs=1e-12)
    assert tuple(str(unit) for unit in grid.wcs.wcs.cunit) == units
    np.testing.assert_array_equal(
        grid.wcs.all_pix2world([[0, 0], [2.5, 6.3]], 0),
        wcs.all_pix2world([[0, 0], [2.5, 6.3]], 0),
    )
