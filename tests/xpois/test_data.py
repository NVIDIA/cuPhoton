# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cuphoton.xpois.data import (
    apply_rectangular_cutout,
    inspect_hsc_data_tree,
    load_image_with_wcs,
    load_mask_with_planes,
    load_variance_with_wcs,
)


def test_load_image_with_wcs_rejects_hdu_for_npy(tmp_path: Path) -> None:
    image = tmp_path / "image.npy"
    np.save(image, np.ones((8, 8), dtype=np.float64), allow_pickle=False)

    with pytest.raises(ValueError):
        load_image_with_wcs(image, hdu=0)


def test_inspect_hsc_data_tree_always_reports_counts(tmp_path: Path) -> None:
    missing = tmp_path / "missing-hsc"
    summary = inspect_hsc_data_tree(missing)

    assert summary["exists"] is False
    assert summary["counts"] == {
        "bundle_files": 0,
        "coadds": 0,
        "warps": 0,
        "catalogs": 0,
    }


def test_load_mask_with_planes_prefers_named_mask_extension(
    tmp_path: Path,
) -> None:
    from astropy.io import fits

    path = tmp_path / "mask.fits"
    mask_hdu = fits.ImageHDU(
        np.zeros((8, 8), dtype=np.int64),
        name="MASK",
    )
    mask_hdu.header["MP_BAD"] = 0
    mask_hdu.header["MP_DETECTED"] = 5
    fits.HDUList(
        [
            fits.PrimaryHDU(np.ones((8, 8), dtype=np.float64)),
            mask_hdu,
        ]
    ).writeto(path)

    mask, used_hdu, plane_map = load_mask_with_planes(path)

    assert used_hdu == 1
    assert plane_map == {"BAD": 0, "DETECTED": 5}
    assert mask.shape == (8, 8)


def test_apply_rectangular_cutout_rejects_out_of_bounds_request() -> None:
    array = np.zeros((8, 8), dtype=np.float64)

    with pytest.raises(ValueError):
        apply_rectangular_cutout(array, y0=4, x0=4, height=8, width=8)


def test_load_variance_with_wcs_prefers_named_variance_extension(
    tmp_path: Path,
) -> None:
    from astropy.io import fits

    path = tmp_path / "variance.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(np.ones((8, 8), dtype=np.float64) * 9.0),
            fits.ImageHDU(
                np.ones((8, 8), dtype=np.float64) * 2.0,
                name="VARIANCE",
            ),
        ]
    ).writeto(path)

    variance, _, used_hdu = load_variance_with_wcs(path)

    assert used_hdu == 1
    assert np.allclose(variance, 2.0)


def test_load_variance_with_wcs_rejects_ambiguous_multi_hdu_fits(
    tmp_path: Path,
) -> None:
    from astropy.io import fits

    path = tmp_path / "variance.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(np.ones((8, 8), dtype=np.float64) * 9.0),
            fits.ImageHDU(np.ones((8, 8), dtype=np.float64) * 2.0),
        ]
    ).writeto(path)

    with pytest.raises(ValueError):
        load_variance_with_wcs(path)


def test_load_variance_with_wcs_rejects_error_extension_without_hdu(
    tmp_path: Path,
) -> None:
    from astropy.io import fits

    path = tmp_path / "variance.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(np.ones((8, 8), dtype=np.float64) * 9.0),
            fits.ImageHDU(
                np.ones((8, 8), dtype=np.float64) * 2.0,
                name="ERROR",
            ),
        ]
    ).writeto(path)

    with pytest.raises(ValueError):
        load_variance_with_wcs(path)


def test_load_variance_with_wcs_rejects_sigma_extension_without_hdu(
    tmp_path: Path,
) -> None:
    from astropy.io import fits

    path = tmp_path / "variance.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                np.ones((8, 8), dtype=np.float64) * 2.0,
                name="SIGMA",
            ),
        ]
    ).writeto(path)

    with pytest.raises(ValueError):
        load_variance_with_wcs(path)
