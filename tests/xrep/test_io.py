# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from cuphoton.xrep import BBox, Grid
from cuphoton.xrep.io import write_reprojected_fits


def _write_mask(tmp_path: Path, mask: np.ndarray) -> Path:
    path = tmp_path / "reprojected.fits"
    write_reprojected_fits(
        path,
        np.zeros(mask.shape, dtype=np.float64),
        grid=Grid(crval=(150.0, 2.0), pixel_scale_arcsec=0.2),
        bbox=BBox(
            min_x=0,
            min_y=0,
            width=mask.shape[1],
            height=mask.shape[0],
        ),
        mask=mask,
    )
    return path


@pytest.mark.parametrize(
    "mask",
    (
        pytest.param(
            np.array([[0, 1], [256, 1824]], dtype=np.int32),
            id="signed-hsc-mask",
        ),
        pytest.param(
            np.array(
                [[0, 1], [256, np.iinfo(np.uint16).max]],
                dtype=np.uint16,
            ),
            id="unsigned-mask",
        ),
    ),
)
def test_write_reprojected_fits_preserves_wide_integer_mask(
    tmp_path: Path,
    mask: np.ndarray,
) -> None:
    path = _write_mask(tmp_path, mask)

    with fits.open(path) as hdul:
        saved = hdul["MASK"].data
        assert saved.dtype.kind == mask.dtype.kind
        assert saved.dtype.itemsize == mask.dtype.itemsize
        np.testing.assert_array_equal(saved, mask)


def test_write_reprojected_fits_serializes_boolean_mask(
    tmp_path: Path,
) -> None:
    mask = np.array([[False, True], [True, False]])

    path = _write_mask(tmp_path, mask)

    with fits.open(path) as hdul:
        saved = hdul["MASK"].data
        assert saved.dtype == np.dtype(np.uint8)
        np.testing.assert_array_equal(saved, mask.astype(np.uint8))
