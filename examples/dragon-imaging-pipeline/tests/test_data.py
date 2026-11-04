# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline checks of the archive contract and controlled image experiment."""

import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

_SPEC = importlib.util.spec_from_file_location(
    "dragon_imaging_data", Path(__file__).parents[1] / "data.py"
)
assert _SPEC is not None and _SPEC.loader is not None
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def test_verified_local_cache_works_offline_and_rejects_corruption(
    tmp_path, monkeypatch
):
    source = tmp_path / "local.fits"
    content = b"fixed archive bytes"
    source.write_bytes(content)
    monkeypatch.setattr(
        data,
        "SOURCE_INFO",
        {
            **data.SOURCE_INFO,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    )

    def no_network(*args, **kwargs):
        raise AssertionError("An offline cache must not make a request")

    monkeypatch.setattr(data.urllib.request, "urlopen", no_network)
    cached = data.acquire_image(tmp_path / "run", source)
    assert cached.read_bytes() == content
    assert data.acquire_image(tmp_path / "run") == cached
    cached.write_bytes(b"wrong archive bytes")
    with pytest.raises(ValueError, match="SHA-256"):
        data.acquire_image(tmp_path / "run")


def test_image_pairs_truth_splits_compression_alignment_and_resume(
    tmp_path, monkeypatch
):
    # An offline image with the archive's shape/WCS exercises the full writer.
    y, x = np.ogrid[:2242, :2416]
    pixels = (
        0.01
        + x * 0.00001
        + y * 0.00002
        + np.exp(-((x - 1026) ** 2 + (y - 1168) ** 2) / 5000)
    ).astype(np.float32)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1026, 1168]
    wcs.wcs.crval = [202.469625, 47.195172]
    wcs.wcs.cdelt = [-0.75 / 3600, 0.75 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    source = tmp_path / "fixture.fits"
    fits.PrimaryHDU(pixels, header=wcs.to_header()).writeto(source)
    monkeypatch.setattr(
        data,
        "SOURCE_INFO",
        {
            **data.SOURCE_INFO,
            "bytes": source.stat().st_size,
            "sha256": data._sha256(source),
        },
    )
    root = tmp_path / "run"
    metadata = data.prepare_images(root, source, images=1)
    candidates = [c for s in metadata["corpus"] for c in s["candidates"]]
    assert Counter(c["split"] for c in candidates) == {
        "train": 320,
        "val": 96,
        "test": 96,
    }
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["split_group"], set()).add(
            candidate["split"]
        )
    assert all(len(splits) == 1 for splits in groups.values())
    coordinates = {
        split: {(c["x"], c["y"]) for c in candidates if c["split"] == split}
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert all(
            max(abs(ax - bx), abs(ay - by)) > 2 * data.FOOTPRINT_RADIUS
            for ax, ay in coordinates[left]
            for bx, by in coordinates[right]
        )
    pair = metadata["pairs"][0]
    assert {(c["x"], c["y"]) for c in pair["candidates"]} == coordinates[
        "test"
    ]
    assert Counter(c["kind"] for c in pair["candidates"]) == {
        "moving_source": 16,
        "detector_artifact": 8,
        "unchanged_control": 8,
    }
    reference = fits.getdata(pair["reference"], 1)
    target = fits.getdata(pair["target"], 1)
    assert reference.dtype.itemsize == target.dtype.itemsize == 4
    np.testing.assert_array_equal(
        fits.getdata(pair["target_compressed"], 1), target
    )
    residual = target - gaussian_filter(reference, 1.5, radius=7)
    fit_mask = fits.getdata(pair["fit_mask"], 1)
    sigma = pair["target_noise_sigma"]
    for candidate in pair["candidates"]:
        x, y = candidate["x"], candidate["y"]
        assert fit_mask[y, x] == 0
        stamp = residual[y - 25 : y + 26, x - 25 : x + 26]
        if candidate["kind"] == "moving_source":
            assert stamp.max() > 5 * sigma
            assert stamp.min() < -5 * sigma
        elif candidate["kind"] == "unchanged_control":
            assert np.std(stamp) == pytest.approx(sigma, rel=0.1)
    reprojection = metadata["reprojection"]
    raw, header = fits.getdata(reprojection["raw_target"], 1, header=True)
    np.testing.assert_array_equal(raw[:-2, 3:], target[2:, :-3])
    target_wcs = WCS(fits.getheader(pair["target"], 1))
    raw_wcs = WCS(header)
    np.testing.assert_allclose(
        raw_wcs.pixel_to_world_values(103, 198),
        target_wcs.pixel_to_world_values(100, 200),
        atol=1e-10,
    )
    mtimes = {p: Path(p).stat().st_mtime_ns for p in metadata["files"]}
    assert data.prepare_images(root, source, images=1) == metadata
    assert mtimes == {
        p: Path(p).stat().st_mtime_ns for p in metadata["files"]
    }
    with pytest.raises(ValueError, match="configuration differs"):
        data.prepare_images(root, source, seed=7, images=1)
    Path(pair["fit_mask"]).unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        data.prepare_images(root, source, images=1)
