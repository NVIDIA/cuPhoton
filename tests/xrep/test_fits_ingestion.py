# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS geometry inspection must not decode image payloads."""

import numpy as np
from astropy.io import fits

from cuphoton.xrep import Grid, io, make_north_up_wcs, mapping
from cuphoton.xrep.reproject import reproject_fits


def _image(tmp_path):
    path = tmp_path / "image.fits"
    image = np.arange(121, dtype=np.float32).reshape(11, 11)
    wcs = make_north_up_wcs(
        (150, 2), shape=image.shape, pixel_scale_arcsec=0.2
    )
    fits.PrimaryHDU(image, header=wcs.to_header()).writeto(path)
    return path, wcs


def test_wcs_mapping_inspects_headers_without_pixel_read(
    tmp_path, monkeypatch
):
    path, wcs = _image(tmp_path)

    def no_pixels(*args, **kwargs):
        raise AssertionError("mapping must not read pixel data")

    monkeypatch.setattr(io, "read_fits_images", no_pixels)
    transform, bbox = mapping.make_wcs_mapping(path, Grid.from_wcs(wcs))
    assert bbox.width > 0 and bbox.height > 0
    assert np.isfinite(transform(np.asarray([[5.0, 5.0]]))).all()


def test_cpu_reprojection_reads_image_once_and_retains_reader(
    tmp_path, monkeypatch
):
    path, wcs = _image(tmp_path)
    original = io.read_fits_images
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(io, "read_fits_images", record)
    result = reproject_fits(
        path,
        grid=Grid.from_wcs(wcs),
        backend="cpu",
        interpolation="bilinear",
    )
    assert len(calls) == 1
    assert calls[0]["reader"] == "astropy"
    assert calls[0]["device"] is False
    assert result.metadata["fits_reads"][0]["reader"] == "astropy"


def test_mask_host_reader_preserves_unsigned_bits(tmp_path):
    path = tmp_path / "mask.fits"
    mask = np.asarray([[0, 2**32 - 1]], dtype=np.uint32)
    fits.PrimaryHDU(mask).writeto(path)
    loaded, _, _ = io.load_fits_mask(path)
    assert loaded.dtype == mask.dtype
    np.testing.assert_array_equal(loaded, mask)


def test_device_image_loader_retains_reader_array_ownership(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    path, _ = _image(tmp_path)
    sentinel = object()
    requests = []

    def device_read(path, hdus, **kwargs):
        requests.append((hdus, kwargs))
        return SimpleNamespace(
            arrays=(sentinel,), metadata=lambda: {"reader": "xdr"}
        )

    monkeypatch.setattr(io, "read_fits_images", device_read)
    receipts = []
    array, _, _, hdu = io.load_fits_image_with_wcs(
        path, fits_reader="xdr", device=True, read_metadata=receipts
    )
    assert array is sentinel
    assert hdu == 0
    assert requests == [([0], {"reader": "xdr", "device": True})]
    assert receipts == [{"reader": "xdr"}]
