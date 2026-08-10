# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xrep import (
    BBox,
    Grid,
    ReprojectionSpec,
    prepare_reprojection,
    reproject_array,
    reproject_fits,
    reproject_masked_array,
)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_torch_backend_matches_cpu_for_identity(interpolation: str) -> None:
    source = np.arange(36, dtype=np.float64).reshape(6, 6)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=6, height=6),
        interpolation=interpolation,
        mapping_grid_step=1,
    )

    cpu = reproject_array(source, spec, backend="cpu")
    torch_result = reproject_array(source, spec, backend="torch")

    assert np.allclose(torch_result.image, cpu.image, equal_nan=True)
    assert np.array_equal(torch_result.mask, cpu.mask)


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_torch_backend_handles_nonnative_byteorder(byteorder: str) -> None:
    # FITS arrays are big-endian; torch.as_tensor rejects non-native order,
    # so the backend must byteswap. Native and big-endian must agree.
    native = np.arange(36, dtype=np.float64).reshape(6, 6)
    source = native.astype(np.dtype(f"{byteorder}f8"))
    spec = ReprojectionSpec(
        mapping=lambda coords: (
            np.asarray(coords, dtype=np.float64) + np.array([0.3, -0.4])
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=6, height=6),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )
    cpu = reproject_array(native, spec, backend="cpu")
    result = reproject_array(source, spec, backend="torch")
    assert np.allclose(result.image, cpu.image, equal_nan=True)


def test_torch_lanczos_matches_cpu_for_float32_subpixel_mapping() -> None:
    source = np.arange(64, dtype=np.float32).reshape(8, 8)
    offset = np.array([0.25, 0.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) + offset,
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation="lanczos3",
        mapping_grid_step=1,
    )

    cpu = reproject_array(source, spec, backend="cpu")
    torch_result = reproject_array(source, spec, backend="torch")

    assert np.allclose(
        torch_result.image,
        cpu.image,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.array_equal(torch_result.mask, cpu.mask)


def test_reproject_fits_torch_on_real_fits(tmp_path) -> None:
    # Regression: reproject_fits(backend="torch") used to fail on FITS-native
    # big-endian arrays with a byte-order ValueError.
    fits = pytest.importorskip("astropy.io.fits")
    from astropy.wcs import WCS

    size = 24
    data = np.random.default_rng(0).normal(100.0, 5.0, (size, size))
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cd = np.array([[-5e-5, 0.0], [0.0, 5e-5]])
    wcs.wcs.set()
    path = tmp_path / "src.fits"
    fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(path)

    grid = Grid(crval=(150.0, 2.0), pixel_scale_arcsec=0.18)
    cpu = reproject_fits(path, grid=grid, backend="cpu")
    tor = reproject_fits(path, grid=grid, backend="torch")
    assert tor.image.shape == cpu.image.shape
    assert np.allclose(tor.image, cpu.image, equal_nan=True)


def test_torch_legacy_and_plane_aware_mask_policies_are_distinct() -> None:
    image = np.ones((9, 9), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.zeros(image.shape, dtype=np.uint64)
    mask[2, 2] = np.uint64(1) << np.uint64(40)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.broadcast_to(
            np.array([4.25, 4.375]),
            np.asarray(coords).shape,
        ).copy(),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="lanczos3",
        mapping_grid_step=1,
        area_scaling=False,
    )

    legacy = reproject_array(
        image,
        spec,
        source_mask=mask,
        backend="torch",
    )
    plane_aware = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="torch",
    )

    assert legacy.mask[0, 0] == 0
    assert plane_aware.mask[0, 0] == np.uint64(1) << np.uint64(40)


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_torch_masked_reproject_matches_cpu(interpolation: str) -> None:
    rng = np.random.default_rng(42)
    image = rng.normal(size=(9, 9))
    variance = rng.uniform(0.5, 3.0, size=(9, 9))
    mask = np.arange(81, dtype=np.uint64).reshape(9, 9)
    mask[0, 0] = np.uint64(1) << np.uint64(63)
    spec = ReprojectionSpec(
        mapping=lambda coords: (
            np.asarray(coords, dtype=np.float64) + np.array([0.25, 0.375])
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=7, height=7),
        interpolation=interpolation,
        mapping_grid_step=1,
    )

    cpu = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
    )
    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="torch",
    )

    assert np.allclose(
        result.image,
        cpu.image,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        result.variance,
        cpu.variance,
        atol=1e-10,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.array_equal(result.mask, cpu.mask)


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_torch_ignores_zero_weight_nonfinite_neighbors(
    interpolation: str,
) -> None:
    image = np.array([[5.0, np.nan], [np.inf, -np.inf]])
    variance = np.array([[2.0, np.inf], [np.inf, np.inf]])
    mask = np.zeros((2, 2), dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64),
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation=interpolation,
        mapping_grid_step=1,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="torch",
    )

    assert result.image[0, 0] == pytest.approx(5.0)
    assert result.variance[0, 0] == pytest.approx(2.0)


def test_torch_uses_variance_fill_independent_from_image_fill() -> None:
    image = np.ones((2, 2), dtype=np.float64)
    variance = np.ones_like(image)
    mask = np.zeros(image.shape, dtype=np.uint16)
    spec = ReprojectionSpec(
        mapping=lambda coords: np.asarray(coords, dtype=np.float64) - 10.0,
        output_bbox=BBox(min_x=0, min_y=0, width=1, height=1),
        interpolation="bilinear",
        mapping_grid_step=1,
        fill_value=-99.0,
    )

    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="torch",
        variance_fill_value=123.0,
    )

    assert result.image[0, 0] == -99.0
    assert result.variance[0, 0] == 123.0


@pytest.mark.parametrize("interpolation", ["bilinear", "lanczos3"])
def test_torch_masked_reproject_matches_cpu_with_nonunit_area(
    interpolation: str,
) -> None:
    rng = np.random.default_rng(7)
    image = rng.normal(size=(12, 12))
    variance = rng.uniform(0.5, 3.0, size=image.shape)
    mask = np.arange(image.size, dtype=np.uint64).reshape(image.shape)
    scale = np.array([1.2, 0.8], dtype=np.float64)
    offset = np.array([2.25, 2.375], dtype=np.float64)
    spec = ReprojectionSpec(
        mapping=lambda coords: (
            np.asarray(coords, dtype=np.float64) * scale + offset
        ),
        output_bbox=BBox(min_x=0, min_y=0, width=5, height=5),
        interpolation=interpolation,
        mapping_grid_step=1,
        area_scaling=True,
    )

    prepared = prepare_reprojection(spec, source_shape=image.shape)
    cpu = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="cpu",
        prepared=prepared,
    )
    result = reproject_masked_array(
        image,
        variance,
        mask,
        spec,
        backend="torch",
    )

    assert np.allclose(prepared.relative_area, 0.96)
    assert np.allclose(result.image, cpu.image, atol=1e-10, rtol=1e-12)
    assert np.allclose(
        result.variance,
        cpu.variance,
        atol=1e-10,
        rtol=1e-12,
    )
    assert np.array_equal(result.mask, cpu.mask)
