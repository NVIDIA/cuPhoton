# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS stamp preparation keeps the bounded-read contract."""

import json
from dataclasses import replace

import numpy as np
import pytest
from astropy.io import fits

from cuphoton.xscan import des, lsstcomcam, workflows


@pytest.mark.parametrize("hdu_selector", [None, 1, "SCI"])
@pytest.mark.parametrize("compressed", [False, True])
def test_des_fits_stamp_reads_only_requested_section(
    tmp_path, monkeypatch, compressed, hdu_selector
):
    path = tmp_path / "image.fits"
    image = np.arange(900, dtype=np.float32).reshape(30, 30)
    hdu = (
        fits.CompImageHDU(
            image, name="SCI", compression_type="GZIP_2", quantize_level=0
        )
        if compressed
        else fits.ImageHDU(image, name="SCI")
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    original = des.read_fits_images
    sections = []

    def record(*args, **kwargs):
        assert args[1] == [1]
        sections.append(kwargs["section"])
        return original(*args, **kwargs)

    monkeypatch.setattr(des, "read_fits_images", record)
    receipts = []
    stamp = des.load_stamp_or_image_cutout(
        path,
        stamp_size=5,
        hdu=hdu_selector,
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


@pytest.fixture(params=["autoscan", "nodiff", "lsstcomcam"])
def fits_dataset_builder(request, tmp_path, monkeypatch):
    image_path = tmp_path / "image.fits"
    image = np.arange(81, dtype=np.float32).reshape(9, 9)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(image, name="IMAGE")]
    ).writeto(image_path)
    rows_path = tmp_path / "rows.jsonl"
    manifest = {"stamp_size": 5, "balance_classes": False}
    if request.param == "lsstcomcam":
        rows = [
            {
                "path": str(image_path),
                "product": product,
                "visit": 1,
                "detector": 0,
                "band": "r",
            }
            for product in ("visit_image", "difference_image")
        ]
        manifest.update(
            registry=str(rows_path),
            sample_count=1,
            include_coadd_metadata=False,
        )
        module = lsstcomcam
        workflow = workflows.build_lsstcomcam_smoke_workflow
        expected_reads = 2
    else:
        rows = [
            {
                "search_path": str(image_path),
                "template_path": str(image_path),
                "difference_path": str(image_path),
                "exposure_id": 1,
                "ccd_id": 0,
                "label": 0,
                "x": 4,
                "y": 4,
            }
        ]
        manifest.update(
            search_hdu="IMAGE", template_hdu="IMAGE", difference_hdu="IMAGE"
        )
        module = des
        expected_reads = 3
        if request.param == "autoscan":
            rows.append(dict(rows[0], label=1))
            manifest["records_path"] = str(rows_path)
            workflow = workflows.build_raw_autoscan_workflow
            expected_reads = 6
        else:
            catalog = tmp_path / "fakes.jsonl"
            catalog.write_text(
                json.dumps(
                    {
                        "fake_id": "fake-1",
                        "x": 2,
                        "y": 2,
                        "diff_snr": 6,
                        "snr": 6,
                        "autoscan_score": 0.9,
                        "injected_flux": 120,
                    }
                )
            )
            rows[0]["fake_catalog_path"] = str(catalog)
            manifest["exposures_path"] = str(rows_path)
            workflow = workflows.build_raw_nodiff_workflow
            monkeypatch.setattr(
                des,
                "detect_search_sources",
                lambda *args, **kwargs: [{"x": 2, "y": 2}, {"x": 6, "y": 6}],
            )
    rows_path.write_text("\n".join(json.dumps(row) for row in rows))
    return workflow, module, manifest, expected_reads


@pytest.mark.parametrize(
    ("manifest_reader", "override", "expected"),
    [
        (None, None, "auto"),
        ("xdr", None, "xdr"),
        ("xdr", "astropy", "astropy"),
        ("astropy", "xdr", "xdr"),
        ("invalid", "astropy", "astropy"),
    ],
)
def test_dataset_fits_reader_override_reaches_decoding_and_summary(
    tmp_path,
    monkeypatch,
    fits_dataset_builder,
    manifest_reader,
    override,
    expected,
):
    workflow, module, manifest, expected_reads = fits_dataset_builder
    manifest["xdr_options"] = {"postprocess": "separate"}
    if manifest_reader is not None:
        manifest["fits_reader"] = manifest_reader
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    original = module.read_fits_images
    readers = []

    def read_on_cpu(*args, reader, **kwargs):
        readers.append(reader)
        assert kwargs["xdr_options"] == {"postprocess": "separate"}
        result = original(*args, reader="astropy", **kwargs)
        return replace(result, requested_reader=reader)

    monkeypatch.setattr(module, "read_fits_images", read_on_cpu)
    result = workflow(
        manifest_path=path,
        output_dir=tmp_path / "dataset",
        fits_reader=override,
    )
    assert readers == [expected] * expected_reads
    summary = result.summary.get("builder_summary", result.summary)
    assert summary["fits_reader"] == expected
    assert all(
        receipt["requested_reader"] == expected
        for receipt in summary["fits_reads"]
    )
    assert json.loads(path.read_text()) == manifest


@pytest.mark.parametrize("invalid", ["invalid", None, ["astropy"]])
@pytest.mark.parametrize("from_manifest", [False, True])
def test_dataset_rejects_invalid_fits_reader_before_loading_inputs(
    tmp_path, fits_dataset_builder, invalid, from_manifest
):
    workflow, _, _, _ = fits_dataset_builder
    # None means no override; use an invalid explicit string in that case.
    override = None if from_manifest else invalid or "invalid"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"fits_reader": invalid}))
    output_dir = tmp_path / "dataset"
    with pytest.raises(ValueError, match="FITS reader"):
        workflow(
            manifest_path=path,
            output_dir=output_dir,
            fits_reader=override,
        )
    assert not output_dir.exists()


def test_des_load_image_preserves_named_hdu(tmp_path):
    path = tmp_path / "image.fits"
    image = np.arange(30, dtype=np.float32).reshape(5, 6)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(image, name="SCI")]
    ).writeto(path)
    np.testing.assert_array_equal(
        des.load_image_array(path, hdu="SCI", fits_reader="astropy"), image
    )
