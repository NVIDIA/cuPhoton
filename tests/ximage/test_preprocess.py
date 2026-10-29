# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pytest

import cuphoton.ximage.preprocess as preprocess
from cuphoton.ximage.data import XImageBundle, validate_bundle
from cuphoton.ximage.preprocess import (
    prepare_reconstruction_inputs_roi,
    round_down_to_odd,
)
from cuphoton.ximage.psf import gaussian_psf


@pytest.mark.parametrize("value", [0, 0.5])
def test_round_down_to_odd_rejects_values_below_one(value) -> None:
    with pytest.raises(ValueError):
        round_down_to_odd(value)


def test_round_down_to_odd_rounds_even_values() -> None:
    assert round_down_to_odd(10) == 9
    assert round_down_to_odd(11) == 11
    assert round_down_to_odd(1) == 1


@pytest.mark.parametrize(
    "prepare",
    [
        preprocess.prepare_reconstruction_inputs,
        prepare_reconstruction_inputs_roi,
    ],
)
def test_prepare_reconstruction_inputs_validates_exposure_times(prepare):
    bundle = XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays={
            "images": np.full((1, 2, 2, 2), 6.0),
            "variances": np.full((1, 2, 2, 2), 36.0),
            "exp_times": np.array([2.0, 3.0]),
        },
        metadata={},
    )
    images, masks, variances = prepare(bundle, mask_mode="none")
    np.testing.assert_allclose(images[..., 0], 3.0)
    np.testing.assert_allclose(images[..., 1], 2.0)
    np.testing.assert_allclose(variances[..., 0], 9.0)
    np.testing.assert_allclose(variances[..., 1], 4.0)
    assert masks is None

    for invalid in (
        [0.0, 1.0],
        [-1.0, 1.0],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [1.0],
        [1.0, 1.0, 1.0],
        [[1.0, 1.0]],
    ):
        bundle.arrays["exp_times"] = np.asarray(invalid)
        with pytest.raises(ValueError, match="exposure times"):
            prepare(bundle, mask_mode="none")


@pytest.mark.parametrize(
    "prepare",
    [
        preprocess.prepare_reconstruction_inputs,
        prepare_reconstruction_inputs_roi,
    ],
)
def test_prepare_rejects_invalid_mask_mode_before_bundle_access(prepare):
    bundle = XImageBundle(
        store_format="npy", store_path=Path("."), arrays={}, metadata={}
    )
    with pytest.raises(ValueError, match="Unsupported mask mode: invalid"):
        prepare(bundle, mask_mode="invalid")


def test_validate_bundle_rejects_wrong_rank() -> None:
    bundle = XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays={
            "images": np.zeros((4, 4, 3), dtype=np.float64),
            "masks": np.ones((1, 4, 4, 3), dtype=bool),
            "sky": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "variances": np.zeros((1, 4, 4, 3), dtype=np.float64),
            "psfs": np.zeros((5, 5, 1, 3), dtype=np.float64),
            "mcgauss": np.zeros((3, 3, 3), dtype=np.float64),
            "exp_times": np.zeros((3,), dtype=np.float64),
        },
        metadata={},
    )
    issues = validate_bundle(bundle)
    assert any("rank-4" in issue for issue in issues)


def test_prepare_reconstruction_inputs_roi_reads_only_requested_pixels() -> (
    None
):
    class SliceOnlyArray:
        def __init__(self, array):
            self.array = array
            self.shape = array.shape
            self.ndim = array.ndim
            self.dtype = array.dtype
            self.nbytes = array.nbytes

        def __array__(self, *args, **kwargs):
            raise AssertionError("Full array materialization is not allowed")

        def __getitem__(self, key):
            assert key[1:3] == (slice(3, 7), slice(3, 7))
            return self.array[key]

    images = np.ones((1, 10, 10, 2), dtype=np.float64)
    masks = np.ones(images.shape, dtype=bool)
    masks[:, 4, 4, 0] = False
    bundle = XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays={
            "images": SliceOnlyArray(images),
            "masks": SliceOnlyArray(masks),
            "variances": SliceOnlyArray(images),
            "psfs": np.ones((3, 3, 1, 2)) / 9,
        },
        metadata={},
    )
    roi_images, roi_masks, roi_variances = prepare_reconstruction_inputs_roi(
        bundle, center_roi=(5, 5), height_roi=4, width_roi=4
    )
    np.testing.assert_array_equal(roi_images, images[:, 3:7, 3:7])
    np.testing.assert_array_equal(roi_masks, masks[:, 3:7, 3:7])
    np.testing.assert_array_equal(roi_variances, images[:, 3:7, 3:7])


@pytest.mark.parametrize(
    "prepare",
    [
        preprocess.prepare_reconstruction_inputs,
        prepare_reconstruction_inputs_roi,
    ],
)
def test_prepare_minimal_bundle_and_provided_good_pixel_mask(prepare):
    images = np.arange(48).reshape(1, 4, 6, 2)
    variances = np.full_like(images, 4)
    bundle = XImageBundle(
        store_format="npy",
        store_path=Path("."),
        metadata={},
        arrays={
            "images": images,
            "variances": variances,
            "psfs": np.ones((3, 3, 1, 2)) / 9,
        },
    )
    actual_images, masks, actual_variances = prepare(bundle)
    np.testing.assert_array_equal(actual_images, images)
    np.testing.assert_array_equal(actual_variances, variances)
    assert masks.dtype == np.dtype(bool)
    assert masks.all()

    bundle.arrays["masks"] = images % 2 == 0
    _, masks, _ = prepare(bundle)
    np.testing.assert_array_equal(masks, bundle.masks)
    assert prepare(bundle, mask_mode="none")[1] is None
    bundle.arrays["masks"] = np.ones(images.shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="Masks must be boolean"):
        prepare(bundle)


@pytest.mark.parametrize("height,width", [(5, 7), (4, 6), (5, 6), (4, 7)])
@pytest.mark.parametrize("srf", [1, 2, 3])
def test_roi_and_padding_preserve_requested_shape_and_pixels(
    height, width, srf
) -> None:
    images = np.arange(20 * 24, dtype=np.float64).reshape(1, 20, 24, 1)
    center = (10, 12)
    start_y = center[0] - height // 2
    start_x = center[1] - width // 2
    expected = images[
        :, start_y : start_y + height, start_x : start_x + width
    ]
    actual, _, _ = preprocess.select_cutout(
        images, center_roi=center, height_roi=height, width_roi=width
    )
    np.testing.assert_array_equal(actual, expected)

    padded, _, _ = preprocess.pad_with_data(
        images,
        center_roi=center,
        height_roi=height,
        width_roi=width,
        srf=srf,
        psf_size=3,
    )
    full_upsampled = np.repeat(np.repeat(images, srf, axis=1), srf, axis=2)
    expected_padded = full_upsampled[
        :,
        start_y * srf - 1 : (start_y + height) * srf + 1,
        start_x * srf - 1 : (start_x + width) * srf + 1,
    ]
    np.testing.assert_array_equal(padded, expected_padded)


@pytest.mark.parametrize("center", [(2, 4), (6, 4), (4, 2), (4, 6)])
def test_padding_rejects_roi_without_surrounding_pixels(center) -> None:
    images = np.ones((1, 8, 8, 1))
    with pytest.raises(ValueError, match="Padded ROI exceeds.*bounds"):
        preprocess.pad_with_data(
            images, center_roi=center, height_roi=4, width_roi=4, psf_size=3
        )


@pytest.mark.parametrize("height,width", [(0, 4), (4, 0), (-1, 4), (4, -1)])
def test_padding_rejects_nonpositive_roi_before_adding_halo(height, width):
    with pytest.raises(
        ValueError, match="ROI height and width must be positive"
    ):
        preprocess.pad_with_data(
            np.ones((1, 16, 16, 1)),
            height_roi=height,
            width_roi=width,
            psf_size=9,
        )


@pytest.mark.parametrize("center", [(1, 4), (7, 4), (4, 1), (4, 7)])
def test_cutout_rejects_roi_beyond_each_image_edge(center) -> None:
    with pytest.raises(ValueError, match="ROI exceeds image bounds"):
        preprocess.select_cutout(
            np.ones((1, 8, 8, 1)),
            center_roi=center,
            height_roi=4,
            width_roi=4,
        )


def test_cutout_accepts_full_image_at_bounds() -> None:
    images = np.arange(8 * 10).reshape(1, 8, 10, 1)
    actual, _, _ = preprocess.select_cutout(images)
    np.testing.assert_array_equal(actual, images)


def test_padding_rejects_full_image_with_nontrivial_psf() -> None:
    with pytest.raises(ValueError, match="require 4 input pixels"):
        preprocess.pad_with_data(np.ones((1, 64, 64, 1)), psf_size=9)


@pytest.mark.parametrize("srf,psf_size", [(1, 1), (1, 9), (3, 9)])
def test_padding_uses_exact_available_source_halo(srf, psf_size) -> None:
    images = np.arange(12 * 14, dtype=np.float64).reshape(1, 12, 14, 1)
    half_psf = psf_size // 2
    halo = (half_psf + srf - 1) // srf
    actual, _, _ = preprocess.pad_with_data(
        images,
        height_roi=12 - 2 * halo,
        width_roi=14 - 2 * halo,
        srf=srf,
        psf_size=psf_size,
    )
    upsampled = np.repeat(np.repeat(images, srf, axis=1), srf, axis=2)
    expected = upsampled[
        :,
        srf * halo - half_psf : srf * (12 - halo) + half_psf,
        srf * halo - half_psf : srf * (14 - halo) + half_psf,
    ]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ["mean", "median"])
def test_coadd_preserves_flux_with_variance_and_missing_exposures(method):
    images = np.full((1, 2, 2, 3), 5.0)
    masks = np.ones_like(images)
    masks[..., 0] = 0
    result = preprocess.coadd(images, masks, variances=25.0, method=method)
    np.testing.assert_allclose(result, 5.0)


@pytest.mark.parametrize("method,expected", [("mean", 4.0), ("median", 2.0)])
def test_coadd_excludes_invalid_samples_and_handles_empty_pixels(
    method, expected
):
    images = np.array([1.0, 2.0, 9.0, 100.0, np.nan]).reshape(1, 1, 1, 5)
    variances = np.array([1.0, 4.0, 9.0, 0.0, 1.0])
    result = preprocess.coadd(images, variances=variances, method=method)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(
        preprocess.coadd(images, masks=0, variances=variances, method=method),
        np.zeros((1, 1, 1, 1)),
    )


@pytest.mark.parametrize("method", ["mean", "median"])
@pytest.mark.parametrize(
    "values,mask", [([0.0, 0.0], 1), ([-2.0, -1.0], 1), ([1.0, 2.0], 0)]
)
def test_positive_coadd_handles_zero_negative_and_masked_pixels(
    method, values, mask
):
    images = np.asarray(values).reshape(1, 1, 2, 1)
    original = preprocess.coadd(images, masks=mask, method=method)
    result = preprocess.coadd(
        images, masks=mask, method=method, positive=True
    )
    assert np.isfinite(result).all()
    assert (result > 0).all()
    np.testing.assert_allclose(
        np.diff(result, axis=2), np.diff(original, axis=2)
    )


@pytest.mark.parametrize("size", [2, 3, 8, np.int64(9)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_psf_is_centered_and_preserves_integrated_mass(size, dtype):
    psf = gaussian_psf(size, dtype=dtype)
    assert psf.dtype == dtype
    np.testing.assert_allclose(psf, psf[::-1, :], rtol=1e-6)
    np.testing.assert_allclose(psf, psf[:, ::-1], rtol=1e-6)
    np.testing.assert_allclose(
        psf.sum(), erf(size / (2 * sqrt(2))) ** 2, rtol=1e-6
    )
    if size % 2:
        np.testing.assert_allclose(
            psf[size // 2, size // 2], erf(0.5 / sqrt(2)) ** 2, rtol=1e-6
        )


@pytest.mark.parametrize("size", [0, -1, 2.5, True])
def test_gaussian_psf_rejects_invalid_size(size):
    with pytest.raises(ValueError, match="size must be a positive integer"):
        gaussian_psf(size)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_gaussian_psf_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        gaussian_psf(5, sigma=sigma)


@pytest.mark.parametrize("offset", [(np.nan, 0.0), (0.0, np.inf)])
def test_gaussian_psf_rejects_nonfinite_offset(offset):
    with pytest.raises(ValueError, match="offset components must be finite"):
        gaussian_psf(5, offset=offset)


@pytest.mark.parametrize("dtype", [np.int32, np.bool_, np.complex128])
def test_gaussian_psf_rejects_nonfloating_dtype(dtype):
    with pytest.raises(ValueError, match="dtype must be floating-point"):
        gaussian_psf(5, dtype=dtype)
