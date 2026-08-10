# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import cuphoton.xrep.mapping as mapping_module
import cuphoton.xrep.reproject as reproject_module
from cuphoton.xrep import (
    BBox,
    Grid,
    ReprojectionSpec,
    estimate_source_bbox_on_grid,
    make_north_up_wcs,
    make_wcs_pair_transform,
    make_wcs_pair_transform_with_offset,
    prepare_reprojection,
)


class _RadianLowLevelWcs:
    """WCS whose legacy numeric world API uses radians."""

    def __init__(self, delegate: WCS) -> None:
        self._delegate = delegate

    def pixel_to_world(self, x, y):
        return self._delegate.pixel_to_world(x, y)

    def world_to_pixel(self, world):
        return self._delegate.world_to_pixel(world)

    def all_pix2world(self, coords, origin):
        return np.deg2rad(self._delegate.all_pix2world(coords, origin))

    def all_world2pix(self, world, origin, **kwargs):
        return self._delegate.all_world2pix(
            np.rad2deg(world),
            origin,
            **kwargs,
        )


class _LegacyOnlyWcs:
    """Compatibility stub exposing only the original numeric WCS methods."""

    def __init__(self, delegate: WCS) -> None:
        self._delegate = delegate

    def all_pix2world(self, coords, origin):
        return self._delegate.all_pix2world(coords, origin)

    def all_world2pix(self, world, origin, **kwargs):
        return self._delegate.all_world2pix(world, origin, **kwargs)

    @property
    def pixel_scale_matrix(self):
        return self._delegate.pixel_scale_matrix


def test_make_wcs_pair_transform_identity_round_trips() -> None:
    wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=(11, 11),
        pixel_scale_arcsec=0.2,
    )
    mapping = make_wcs_pair_transform(wcs, wcs)
    coords = np.array([[0.0, 0.0], [5.0, 7.0], [10.0, 10.0]])
    mapped = mapping(coords)

    assert np.allclose(mapped, coords)


def test_make_wcs_pair_transform_supports_legacy_only_wcs() -> None:
    native = make_north_up_wcs(
        (150.0, 2.0),
        shape=(11, 11),
        pixel_scale_arcsec=0.2,
    )
    legacy = _LegacyOnlyWcs(native)
    coords = np.array([[0.0, 0.0], [5.0, 7.0], [10.0, 10.0]])

    assert np.allclose(
        make_wcs_pair_transform(legacy, legacy)(coords),
        coords,
        atol=1e-8,
    )


def test_make_wcs_pair_transform_supports_multi_object_world_values() -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [0.5, 2.0]
    wcs.wcs.ctype = ["LINEAR", "LINEAR"]
    wcs.wcs.set()
    coords = np.array([[0.0, 0.0], [5.0, 7.0], [10.0, 10.0]])

    assert np.allclose(
        make_wcs_pair_transform(wcs, wcs)(coords),
        coords,
        atol=1e-8,
    )


def test_make_wcs_pair_transform_is_unit_aware() -> None:
    degree_wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=(11, 11),
        pixel_scale_arcsec=0.2,
    )
    radian_wcs = _RadianLowLevelWcs(degree_wcs)
    coords = np.array([[0.0, 0.0], [5.0, 7.0], [10.0, 10.0]])

    assert np.allclose(
        radian_wcs.all_pix2world(coords, 0),
        np.deg2rad(degree_wcs.all_pix2world(coords, 0)),
    )
    assert np.allclose(
        make_wcs_pair_transform(radian_wcs, degree_wcs)(coords),
        coords,
        atol=1e-8,
    )
    assert np.allclose(
        make_wcs_pair_transform(degree_wcs, radian_wcs)(coords),
        coords,
        atol=1e-8,
    )
    assert np.allclose(
        make_wcs_pair_transform_with_offset(
            radian_wcs,
            degree_wcs,
            offset=(2, 3),
        )(coords),
        coords + np.array([2.0, 3.0]),
        atol=1e-8,
    )

    grid = Grid(crval=(150.0, 2.0), pixel_scale_arcsec=0.2)
    assert estimate_source_bbox_on_grid(
        radian_wcs,
        shape=(11, 11),
        grid=grid,
        perimeter_step=2,
    ) == estimate_source_bbox_on_grid(
        degree_wcs,
        shape=(11, 11),
        grid=grid,
        perimeter_step=2,
    )


def test_prepare_reprojection_identity_builds_expected_coordinates() -> None:
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=4, height=3),
        interpolation="bilinear",
        mapping_grid_step=1,
        area_scaling=True,
    )

    prepared = prepare_reprojection(spec, source_shape=(3, 4))

    expected_x = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )
    expected_y = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )

    assert np.allclose(prepared.x_local, expected_x)
    assert np.allclose(prepared.y_local, expected_y)
    assert np.allclose(prepared.relative_area, 1.0)
    assert np.all(prepared.valid)


def test_grid_builds_north_up_tan_wcs() -> None:
    grid = Grid(crval=(150.0, 2.0), pixel_scale_arcsec=0.168)

    assert isinstance(grid.wcs, WCS)
    assert tuple(grid.wcs.wcs.ctype) == ("RA---TAN", "DEC--TAN")
    assert grid.wcs.wcs.cd[0, 0] < 0.0
    assert grid.wcs.wcs.cd[1, 1] > 0.0


@pytest.mark.parametrize("shape", [(11, 9), (10, 8)])
def test_north_up_wcs_places_crval_at_geometric_pixel_center(
    shape: tuple[int, int],
) -> None:
    wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=shape,
        pixel_scale_arcsec=0.2,
    )
    center = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)

    world = wcs.pixel_to_world(*center)

    assert isinstance(world, SkyCoord)
    assert world.ra.deg == pytest.approx(150.0)
    assert world.dec.deg == pytest.approx(2.0)
    assert np.allclose(wcs.wcs.crpix, np.asarray(center) + 1.0)


@pytest.mark.parametrize("shape", [(11, 9), (10, 8)])
def test_default_grid_uses_zero_based_geometric_pixel_center(
    shape: tuple[int, int],
) -> None:
    source_wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=shape,
        pixel_scale_arcsec=0.2,
    )
    center = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    expected = source_wcs.pixel_to_world(*center)

    grid = reproject_module._default_grid_from_wcs(source_wcs, shape)

    assert grid.crval[0] == pytest.approx(expected.ra.deg)
    assert grid.crval[1] == pytest.approx(expected.dec.deg)


def test_default_grid_supports_legacy_only_wcs() -> None:
    shape = (10, 8)
    native = make_north_up_wcs(
        (150.0, 2.0),
        shape=shape,
        pixel_scale_arcsec=0.2,
    )
    expected = native.all_pix2world(
        [[(shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0]],
        0,
    )[0]

    grid = reproject_module._default_grid_from_wcs(
        _LegacyOnlyWcs(native),
        shape,
    )

    assert grid.crval == pytest.approx(tuple(expected))


def test_bbox_estimate_ignores_nonfinite_perimeter_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def projected(source_wcs, destination_wcs, perimeter):
        del source_wcs, destination_wcs
        values = np.full(perimeter.shape, np.nan, dtype=np.float64)
        values[-2:] = [[1.2, 3.4], [5.6, 7.8]]
        return values

    monkeypatch.setattr(
        mapping_module,
        "_high_level_pixel_to_pixel",
        projected,
    )
    wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=(5, 5),
        pixel_scale_arcsec=0.2,
    )

    bbox = estimate_source_bbox_on_grid(
        wcs,
        shape=(5, 5),
        grid=Grid(crval=(150.0, 2.0), pixel_scale_arcsec=0.2),
    )

    assert bbox == BBox(min_x=1, min_y=3, width=6, height=6)


def test_bbox_estimate_rejects_all_nonfinite_perimeter_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mapping_module,
        "_high_level_pixel_to_pixel",
        lambda source_wcs, destination_wcs, perimeter: np.full(
            perimeter.shape,
            np.nan,
        ),
    )
    wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=(5, 5),
        pixel_scale_arcsec=0.2,
    )

    with pytest.raises(ValueError, match="no finite projection"):
        estimate_source_bbox_on_grid(
            wcs,
            shape=(5, 5),
            grid=Grid(crval=(150.0, 2.0), pixel_scale_arcsec=0.2),
        )
