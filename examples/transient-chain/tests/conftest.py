# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic FITS fixtures: a Rubin-layout visit with coadd patches, and a
generic single-HDU pair.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

NY, NX = 96, 128
HSC_PLANES = (
    "BAD",
    "SAT",
    "INTRP",
    "CR",
    "EDGE",
    "SUSPECT",
    "NO_DATA",
    "CROSSTALK",
    "UNMASKEDNAN",
    "DETECTED",
)


def tan_wcs(crval=(10.0, -5.0), crpix=(NX / 2, NY / 2), scale_deg=0.2 / 3600):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = [-scale_deg, scale_deg]
    return w


def scene(rng, shape=(NY, NX), n=25, level=100.0):
    img = rng.normal(level, 3.0, shape)
    yy, xx = np.mgrid[-5:6, -5:6]
    for _ in range(n):
        y, x = rng.integers(8, shape[0] - 8), rng.integers(8, shape[1] - 8)
        img[y - 5 : y + 6, x - 5 : x + 6] += rng.uniform(200, 2000) * np.exp(
            -(xx**2 + yy**2) / (2 * 1.6**2)
        )
    return img.astype(np.float32)


def mask_header_cards(header):
    for bit, name in enumerate(HSC_PLANES):
        header[f"MP_{name}"] = bit
    return header


@pytest.fixture
def rubin_visit(tmp_path):
    """Visit (HDU 1/2/3, GZIP_2 tiles, MP_ cards) plus two coadd patches
    offset by half a frame each.
    """
    rng = np.random.default_rng(1)
    w = tan_wcs()
    img = scene(rng)
    mask = np.zeros((NY, NX), np.int32)
    mask[10:14, 20:24] = 1 << HSC_PLANES.index("SAT")
    mask[50, 60] = 1 << HSC_PLANES.index("DETECTED")
    var = np.full((NY, NX), 9.0, np.float32)
    ih = w.to_header()
    mh = mask_header_cards(fits.Header())
    vh = fits.Header()
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.CompImageHDU(
                img, header=ih, compression_type="GZIP_2", name="IMAGE"
            ),
            fits.CompImageHDU(
                mask, header=mh, compression_type="GZIP_2", name="MASK"
            ),
            fits.CompImageHDU(
                var, header=vh, compression_type="GZIP_2", name="VARIANCE"
            ),
        ]
    )
    visit = tmp_path / "visit.fits"
    hdus.writeto(visit)
    coadds = []
    for k, dx in enumerate((-NX // 3, NX // 3)):
        cw = tan_wcs(
            crpix=(NX / 2 - dx, NY / 2)
        )  # same sky, shifted pixel origin
        cimg = scene(
            np.random.default_rng(1), level=100.0
        )  # same sources as the visit (positions in pixels differ by dx)
        cmask = np.zeros((NY, NX), np.int32)
        cmask[5:8, 5:8] = 1 << HSC_PLANES.index("BAD")
        h = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.CompImageHDU(
                    cimg,
                    header=cw.to_header(),
                    compression_type="GZIP_2",
                    name="IMAGE",
                ),
                fits.CompImageHDU(
                    cmask,
                    header=mask_header_cards(fits.Header()),
                    compression_type="GZIP_2",
                    name="MASK",
                ),
            ]
        )
        p = tmp_path / f"coadd{k}.fits"
        h.writeto(p)
        coadds.append(str(p))
    return {
        "visit": 2024121000367,
        "detector": "R22_S00",
        "band": "r",
        "date": 20241210,
        "tract": 7850,
        "visit_image": str(visit),
        "template_coadds": coadds,
        "_mask": mask,
        "_image": img,
    }


@pytest.fixture
def generic_pair(tmp_path):
    """Science image in the primary HDU, a separate weight (inverse variance)
    file, a separate mask, template on the grid.
    """
    rng = np.random.default_rng(2)
    w = tan_wcs()
    img = scene(rng, level=500.0)
    tmpl = img + rng.normal(0, 3.0, img.shape).astype(np.float32)
    weight = np.full(img.shape, 1 / 25.0, np.float32)
    weight[0, 0] = 0.0
    mask = np.zeros(img.shape, np.int16)
    mask[30:33, 40:43] = 4
    paths = {}
    for name, data, header in (
        ("image", img, w.to_header()),
        ("template", tmpl, w.to_header()),
        ("weight", weight, None),
        ("mask", mask, None),
    ):
        p = tmp_path / f"{name}.fits"
        fits.PrimaryHDU(data, header=header).writeto(p)
        paths[name] = str(p)
    return {
        "expnum": 123456,
        "ccdnum": "N4",
        "band": "g",
        **paths,
        "_mask": mask,
    }
