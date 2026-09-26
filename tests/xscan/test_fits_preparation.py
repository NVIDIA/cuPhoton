# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS stamp preparation keeps the bounded-read contract."""

import numpy as np
import pytest
from astropy.io import fits

from cuphoton.xscan import des, lsstcomcam


@pytest.mark.parametrize("compressed", [False, True])
def test_des_fits_stamp_reads_only_requested_section(
    tmp_path, monkeypatch, compressed
):
    path = tmp_path / "image.fits"
    image = np.arange(900, dtype=np.float32).reshape(30, 30)
    hdu = (
        fits.CompImageHDU(image, compression_type="GZIP_2", quantize_level=0)
        if compressed
        else fits.ImageHDU(image)
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    original = des.read_fits_images
    sections = []

    def record(*args, **kwargs):
        sections.append(kwargs["section"])
        return original(*args, **kwargs)

    monkeypatch.setattr(des, "read_fits_images", record)
    receipts = []
    stamp = des.load_stamp_or_image_cutout(
        path,
        stamp_size=5,
        center_x=15,
        center_y=13,
        fits_reader="astropy",
        read_metadata=receipts,
    )
    assert sections == [(slice(11, 16), slice(13, 18))]
    np.testing.assert_array_equal(stamp, image[11:16, 13:18])
    assert receipts[0]["reader"] == "astropy"


def test_lsst_named_hdu_and_stamp_coordinates_are_preserved(tmp_path):
    path = tmp_path / "image.fits"
    image = np.arange(900, dtype=np.float32).reshape(30, 30)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(image, name="IMAGE")]
    ).writeto(path)
    receipts = []
    result = lsstcomcam.read_fits_stamp(
        path,
        hdu="IMAGE",
        stamp_size=5,
        center_x=15,
        center_y=13,
        fits_reader="astropy",
        read_metadata=receipts,
    )
    assert result.image_shape == image.shape
    assert (result.center_x, result.center_y, result.hdu_name) == (
        15,
        13,
        "IMAGE",
    )
    np.testing.assert_array_equal(result.data, image[11:16, 13:18])
    assert receipts[0]["reader"] == "astropy"
