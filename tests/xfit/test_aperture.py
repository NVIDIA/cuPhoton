# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xfit import (
    CircularApertureFlag as Flag,
)
from cuphoton.xfit import (
    forced_circular_aperture_photometry,
)


def test_aperture_membership_signed_flux_and_incomplete_coverage():
    image = -np.ones((6, 5, 5))
    variance = np.full_like(image, 4.0)
    good = np.ones_like(image, dtype=bool)
    image[1, 2, 2] = np.nan
    variance[1, 1, 2] = 0
    good[1, 2, 1] = False
    good[3] = False
    offsets = [(0, 0), (0, 0), (-2, -2), (0, 0), (20, 0), (np.nan, 0)]
    result = forced_circular_aperture_photometry(
        image, variance, offsets, [1.5, 1], mask=good
    )
    np.testing.assert_array_equal(
        result.used_pixels, [[9, 1], [6, 0], [4, 1], [0, 0], [0, 0], [0, 0]]
    )
    np.testing.assert_array_equal(
        result.in_bounds_pixels,
        [[9, 1], [9, 1], [4, 1], [9, 1], [0, 0], [0, 0]],
    )
    np.testing.assert_equal(
        result.flux,
        np.where(result.used_pixels > 0, -result.used_pixels, np.nan),
    )
    np.testing.assert_allclose(result.flux_error, 2 * np.sqrt(-result.flux))
    masked_empty = Flag.MASKED | Flag.NO_COVERAGE
    outside = Flag.TRUNCATED | Flag.NO_COVERAGE
    invalid = Flag.INVALID_GEOMETRY | Flag.NO_COVERAGE
    np.testing.assert_array_equal(
        result.flags,
        [
            [0, 0],
            [Flag.MASKED, masked_empty],
            [Flag.TRUNCATED, 0],
            [masked_empty, masked_empty],
            [outside, outside],
            [invalid, invalid],
        ],
    )


def test_complete_center_aperture_crosses_pixel_edge():
    for width in (5, 13):
        image = np.ones((1, width, width))
        result = forced_circular_aperture_photometry(
            image, image, [[0, 0]], 2.6
        )
        assert (
            result.in_bounds_pixels.item() == result.used_pixels.item() == 21
        )
        assert result.flags.item() == 0


@pytest.mark.parametrize("shape", [(5, 5), (6, 4)])
def test_truncated_means_a_selected_center_is_outside(shape):
    height, width = shape
    positions = [(2.625, 0.5), (-0.5, -2.625), (8.5, -9.25), (0, 0)]
    radii = [0.1, 0.625, np.nextafter(0.625, 1), 2.6, 3.0]
    image = np.ones((len(positions), height, width))
    result = forced_circular_aperture_photometry(
        image, image, positions, radii
    )
    # Extend the grid to contain every circle, including the far corner.
    y, x = np.mgrid[-15 : height + 15, -15 : width + 15]
    outside = (x < 0) | (x >= width) | (y < 0) | (y >= height)
    x = x[outside] - (width - 1) / 2
    y = y[outside] - (height - 1) / 2
    for i, (cx, cy) in enumerate(positions):
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        expected = [np.any(d2 < r * r) for r in radii]
        np.testing.assert_array_equal(
            (result.flags[i] & Flag.TRUNCATED) != 0, expected
        )


def test_unusable_pixels_outside_aperture_do_not_set_masked():
    image = np.ones((1, 5, 5))
    variance = np.ones_like(image)
    mask = np.ones_like(image, dtype=bool)
    image[0, 0, 0] = np.nan
    variance[0, -1, -1] = 0
    mask[0, 0, -1] = False
    result = forced_circular_aperture_photometry(
        image, variance, [[0, 0]], 1.5, mask=mask
    )
    assert result.in_bounds_pixels.item() == result.used_pixels.item() == 9
    assert result.flags.item() == 0


def test_subpixel_centers_match_photutils():
    aperture = pytest.importorskip("photutils.aperture")
    rng = np.random.default_rng(41)
    shape = (3, 14, 18)
    images = rng.normal(size=shape)
    variance = rng.uniform(0.1, 2.0, size=shape)
    good = rng.random(size=shape) > 0.15
    positions = np.array([[-0.2, 0.45], [0.5, -0.5], [2.11, -1.37]])
    radii = [3, 1, 3.2]
    got = forced_circular_aperture_photometry(
        images, variance, positions, radii, mask=good
    )
    for i, offset in enumerate(positions):
        xy = offset + [(shape[2] - 1) / 2, (shape[1] - 1) / 2]
        for j, radius in enumerate(radii):
            aperture_sum, aperture_error = aperture.CircularAperture(
                xy, radius
            ).do_photometry(
                images[i],
                error=np.sqrt(variance[i]),
                mask=~good[i],
                method="center",
            )
            assert got.flux[i, j] == pytest.approx(aperture_sum[0])
            assert got.flux_error[i, j] == pytest.approx(aperture_error[0])


def test_radius_cutoff_at_adjacent_float64_values():
    aperture = pytest.importorskip("photutils.aperture")
    root_two = np.sqrt(2.0)
    radii = [np.nextafter(root_two, 0), root_two, np.nextafter(root_two, 2)]
    image = np.ones((1, 5, 5))
    result = forced_circular_aperture_photometry(
        image, image, [[0, 0]], radii
    )
    expected = [
        aperture.CircularAperture((2, 2), r)
        .to_mask(method="center")
        .to_image((5, 5))
        .sum()
        for r in radii
    ]
    assert expected == [5, 9, 9]
    np.testing.assert_array_equal(result.flux[0], expected)


def test_scalar_invalid_radii_empty_batches_and_shapes():
    image = np.ones((1, 4, 6))
    xy = [[0, 0]]
    tiny = forced_circular_aperture_photometry(image, image, xy, 0.1)
    assert tiny.flux.shape == (1, 1)
    assert tiny.flags[0, 0] == Flag.NO_COVERAGE
    result = forced_circular_aperture_photometry(
        image, image, xy, [0, -2, np.inf, np.nan]
    )
    assert np.all(result.flags == Flag.INVALID_GEOMETRY | Flag.NO_COVERAGE)
    assert np.all(np.isnan(result.flux))
    empty = forced_circular_aperture_photometry(
        image[:0], image[:0], np.empty((0, 2)), [1, 2]
    )
    assert empty.flux.shape == empty.flags.shape == (0, 2)
    with pytest.raises(ValueError, match="positions"):
        forced_circular_aperture_photometry(image, image, [0, 0], 1)
    with pytest.raises(ValueError, match="variance"):
        forced_circular_aperture_photometry(image, image[0], xy, 1)
    with pytest.raises(ValueError, match="radii"):
        forced_circular_aperture_photometry(image, image, xy, [[1, 2]])


def test_gpu_parity_with_device_radii_and_stream():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no visible CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    rng = np.random.default_rng(13)
    image = rng.normal(size=(7, 13, 16)).astype(np.float32)
    variance = rng.uniform(1, 4, size=image.shape).astype(np.float32)
    positions = rng.uniform(-4, 4, size=(len(image), 2))
    mask = rng.random(image.shape) > 0.1
    radii = [1, 4, 3, 1]
    host = forced_circular_aperture_photometry(
        image, variance, positions, radii, mask=mask
    )
    with cp.cuda.Stream(non_blocking=True) as stream:
        device = forced_circular_aperture_photometry(
            cp.asarray(image),
            cp.asarray(variance),
            cp.asarray(positions),
            cp.asarray(radii),
            mask=cp.asarray(mask),
        )
    stream.synchronize()
    for name in (
        "flux",
        "flux_error",
        "used_pixels",
        "in_bounds_pixels",
        "flags",
    ):
        value = getattr(device, name)
        assert isinstance(value, cp.ndarray)
        np.testing.assert_allclose(
            value.get(), getattr(host, name), rtol=1e-12
        )
