# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

import cuphoton.ximage.adapters.rubin.bundle as rubin_module
from cuphoton.ximage.adapters.rubin.bundle import (
    build_rubin_manifest,
    build_rubin_store,
    parse_section,
    probe_fits_products,
    summarize_rubin_coadd_semantics,
)
from cuphoton.ximage.data.bundle import load_ximage_bundle, validate_bundle


def _write_rubin_like_fits(
    path,
    *,
    image=None,
    image_header=None,
    image_bunit="nJy",
    mask=None,
    mask_header=None,
) -> None:
    if image is None:
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
    image = np.asarray(image, dtype=np.float32)
    if mask is None:
        mask = np.zeros(image.shape, dtype=np.int32)
        mask[2, 3] = 1
    if mask_header is None:
        mask_header = fits.Header()
        mask_header["MP_BAD"] = 0
        mask_header["MP_DETECTED"] = 5
    variance = np.full(image.shape, 2.0, dtype=np.float32)
    primary = fits.PrimaryHDU()
    primary.header["EXPTIME"] = 30.0
    image_header = (
        fits.Header() if image_header is None else image_header.copy()
    )
    if image_bunit is not None:
        image_header["BUNIT"] = image_bunit
    variance_header = fits.Header()
    if image_bunit is not None:
        variance_header["BUNIT"] = f"{image_bunit}**2"
    summary_payload = (
        b"expTime: 30.0\n"
        b"psfArea: 6.283185307179586\n"
        b"psfIxx: 1.0\n"
        b"psfIxy: 0.0\n"
        b"psfIyy: 1.0\n"
        b"psfSigma: 1.0\n"
        b"skyBg: 12.5\n"
        b"skyNoise: 1.25\n"
        b"zeroPoint: 32.0\n"
    )
    summary_bytes = np.frombuffer(summary_payload, dtype=np.uint8)
    summary = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="bytes",
                array=[summary_bytes],
                format=f"{summary_bytes.size}B",
            )
        ],
        name="ExposureSummaryStats",
    )
    fits.HDUList(
        [
            primary,
            fits.CompImageHDU(
                image,
                header=image_header,
                name="IMAGE",
                compression_type="GZIP_2",
            ),
            fits.CompImageHDU(
                mask,
                header=mask_header,
                name="MASK",
                compression_type="GZIP_2",
            ),
            fits.CompImageHDU(
                variance,
                header=variance_header,
                name="VARIANCE",
                compression_type="GZIP_2",
            ),
            summary,
        ]
    ).writeto(path)


def _simple_wcs_header(*, crpix1=1.0, crpix2=1.0) -> fits.Header:
    header = fits.Header()
    header["WCSAXES"] = 2
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = crpix1
    header["CRPIX2"] = crpix2
    header["CD1_1"] = 1.0e-3
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0e-3
    return header


def _manifest(root, *relative_paths) -> dict:
    files = []
    for index, relative_path in enumerate(relative_paths):
        path = root / relative_path
        files.append(
            {
                "group": "stack",
                "product": "visit_image",
                "rubin_image_format": "legacy-fits",
                "band": "r",
                "visit": 2000010100001 + index,
                "detector": 0,
                "size_bytes": path.stat().st_size,
                "hdu_count": 4,
                "relative_path": str(relative_path),
            }
        )
    return {"file_count": len(files), "files": files}


def test_probe_fits_products_reports_rubin_hdus(tmp_path) -> None:
    path = tmp_path / "difference_image" / "20000102" / "r" / "sample.fits"
    path.parent.mkdir(parents=True)
    _write_rubin_like_fits(path)

    result = probe_fits_products(
        tmp_path,
        max_files=1,
        read_sample=True,
        sample_section="2:5,3:7",
    )

    assert result["file_count"] == 1
    assert result["product_counts"] == {"difference_image": 1}
    assert result["image_compression_counts"] == {"GZIP_2": 3}
    file_result = result["files"][0]
    assert file_result["hdu_count"] == 5
    assert file_result["hdus"][1]["name"] == "IMAGE"
    assert file_result["hdus"][1]["shape"] == [10, 10]
    assert file_result["hdus"][1]["zcmptype"] == "GZIP_2"
    assert file_result["hdus"][1]["sample"]["shape"] == [3, 4]
    assert file_result["hdus"][1]["sample"]["finite_count"] == 12
    assert file_result["hdus"][1]["sample"]["min"] == 23.0
    assert file_result["hdus"][1]["sample"]["max"] == 46.0


def test_summarize_rubin_coadd_semantics_reports_metadata_requirements(
    tmp_path,
) -> None:
    coadd_path = tmp_path / "deep_coadd.fits"
    image_header = fits.Header()
    image_header["BUNIT"] = "nJy"
    image_header["EXPTIME"] = 1.0
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((16, 16), dtype=np.float32),
                header=image_header,
                name="IMAGE",
            ),
            fits.ImageHDU(
                data=np.zeros((16, 16), dtype=np.int16),
                header=fits.Header({"MP_BAD": 0}),
                name="MASK",
            ),
            fits.ImageHDU(
                data=np.ones((16, 16), dtype=np.float32),
                header=fits.Header({"BUNIT": "nJy2"}),
                name="VARIANCE",
            ),
            fits.ImageHDU(data=np.zeros((1, 1)), name="CoaddInputs"),
            fits.ImageHDU(data=np.zeros((1, 1)), name="VisitInfo"),
        ]
    ).writeto(coadd_path)

    output_path = tmp_path / "semantics.json"

    summary = summarize_rubin_coadd_semantics(
        coadd_path,
        output_path=output_path,
        require_coadd_inputs=True,
        require_visit_info=True,
    )

    assert summary["artifact_kind"] == "rubin_coadd_semantics"
    assert summary["ok"] is True
    assert summary["requirements_met"] is False
    assert summary["image"]["bunit"] == "nJy"
    assert summary["coadd_psf"] is None
    assert (
        summary["reconstruction_contract"]["raw_exposure_stack_allowed"]
        is False
    )
    assert (
        summary["reconstruction_contract"]["reference_product_allowed"]
        is True
    )
    persisted = json.loads(output_path.read_text())
    assert persisted["output_path"] == str(output_path)


def test_summarize_rubin_coadd_semantics_requires_coadd_psf(
    tmp_path,
) -> None:
    coadd_path = tmp_path / "template_coadd.fits"
    image_header = fits.Header()
    image_header["BUNIT"] = "nJy"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((8, 8), dtype=np.float32),
                header=image_header,
                name="IMAGE",
            ),
            fits.ImageHDU(
                data=np.zeros((8, 8), dtype=np.int16),
                header=fits.Header({"MP_BAD": 0}),
                name="MASK",
            ),
            fits.ImageHDU(
                data=np.ones((8, 8), dtype=np.float32),
                header=fits.Header({"BUNIT": "nJy2"}),
                name="VARIANCE",
            ),
        ]
    ).writeto(coadd_path)

    summary = summarize_rubin_coadd_semantics(
        coadd_path,
        require_coadd_psf=True,
    )

    assert summary["ok"] is False
    assert summary["requirements_met"] is False
    checks = {check["name"]: check for check in summary["checks"]}
    assert checks["coadd_psf_present"]["required"] is True
    assert checks["coadd_psf_present"]["ok"] is False
    assert (
        summary["reconstruction_contract"]["raw_exposure_stack_allowed"]
        is False
    )


def test_build_rubin_manifest_samples_products_and_metadata(tmp_path) -> None:
    root = tmp_path / "rubin-products"
    # Independently chosen parser identifiers; every image is generated below.
    visit_paths = [
        (
            "visit_image/visit_image/20000101/r/r_01/"
            f"2000010100{visit}/"
            "visit_image_SYNTHETIC_r_r_01_"
            f"2000010100{visit}_R01_S01_"
            "synthetic.fits"
        )
        for visit in ("001", "002", "003")
    ]
    diff_path = (
        "difference_image/20000102/g/g_01/2000010200001/"
        "difference_image_SYNTHETIC_g_g_01_2000010200001_R02_S02_"
        "synthetic.fits"
    )
    ignored_coadd = (
        "template_coadd/42/7/g/template_coadd_42_7_g_synthetic.fits"
    )
    for relative_path in [*visit_paths, diff_path, ignored_coadd]:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_rubin_like_fits(path)

    manifest_path = tmp_path / "manifest.json"
    result = build_rubin_manifest(
        root,
        output_path=manifest_path,
        products=("visit_image", "difference_image"),
        sample_per_product=2,
        group_prefix="local",
    )

    assert manifest_path.exists()
    persisted = json.loads(manifest_path.read_text())
    assert persisted["file_count"] == 3
    assert result["file_count"] == 3
    assert result["product_counts"] == {
        "difference_image": 1,
        "visit_image": 2,
    }
    assert result["discovered_product_counts"] == {
        "visit_image": 3,
        "difference_image": 1,
    }
    assert result["metadata_coverage"]["visit_image"] == {
        "file_count": 2,
        "has_exptime": 2,
        "has_exposure_summary": 2,
        "has_image_mask_variance": 2,
    }

    visit_rows = [
        row for row in result["files"] if row["product"] == "visit_image"
    ]
    assert len(visit_rows) == 2
    assert visit_rows[0]["group"] == "local_visit_image"
    assert visit_rows[0]["rubin_image_format"] == "legacy-fits"
    assert visit_rows[0]["date"] == 20000101
    assert visit_rows[0]["band"] == "r"
    assert visit_rows[0]["physical_filter"] == "r_01"
    assert visit_rows[0]["visit"] == 2000010100001
    assert visit_rows[0]["detector"] == "R01_S01"
    assert visit_rows[0]["has_exptime"] is True
    assert visit_rows[0]["exp_time"] == 30.0
    assert visit_rows[0]["exp_time_source"] == "PRIMARY.header.EXPTIME"
    assert visit_rows[0]["has_exposure_summary"] is True
    assert "psfSigma" in visit_rows[0]["exposure_summary_keys"]
    assert visit_rows[0]["has_image_mask_variance"] is True
    assert visit_rows[0]["source_path"].startswith(str(root))
    assert not visit_rows[0]["relative_path"].startswith("/")
    assert [row["visit"] for row in visit_rows] == [
        2000010100001,
        2000010100003,
    ]


@pytest.mark.parametrize("relative_paths", [False, True])
def test_build_rubin_manifest_accepts_explicit_inventory(
    tmp_path, relative_paths
) -> None:
    source_root = "/path/to/rubin-products"
    inventory_prefix = "" if relative_paths else f"{source_root}/"
    inventory = tmp_path / "inventory.tsv"
    inventory.write_text(
        "\n".join(
            [
                (
                    "123\t"
                    f"{inventory_prefix}visit_image/visit_image/20000101/r/r_01/"
                    "2000010100001/visit_image_SYNTHETIC_r_r_01_"
                    "2000010100001_R01_S01_"
                    "synthetic.fits"
                ),
                (
                    "456\t"
                    f"{inventory_prefix}difference_image/20000102/r/r_01/"
                    "2000010200001/difference_image_SYNTHETIC_r_r_01_"
                    "2000010200001_R02_S02_"
                    "synthetic.fits"
                ),
            ]
        )
        + "\n"
    )

    result = build_rubin_manifest(
        source_root,
        inventory_path=inventory,
        products=("visit_image", "difference_image"),
        sample_per_product=1,
        inspect_metadata=False,
        group_prefix="local",
    )

    assert result["file_count"] == 2
    assert result["selection"]["inventory_path"] == str(inventory)
    assert result["selection"]["inspect_metadata"] is False
    rows = {row["product"]: row for row in result["files"]}
    assert rows["visit_image"]["group"] == "local_visit_image"
    assert rows["visit_image"]["relative_path"].startswith(
        "visit_image/visit_image/20000101"
    )
    assert rows["visit_image"]["source_path"] == (
        f"{source_root}/{rows['visit_image']['relative_path']}"
    )
    assert rows["visit_image"]["size_bytes"] == 123
    assert rows["visit_image"]["date"] == 20000101
    assert rows["difference_image"]["detector"] == "R02_S02"
    assert result["metadata_coverage"]["visit_image"] == {
        "file_count": 1,
        "has_exptime": 0,
        "has_exposure_summary": 0,
        "has_image_mask_variance": 0,
    }


def test_parse_section_requires_explicit_bounds() -> None:
    y, x = parse_section("1:3,4:8")

    assert (y.start, y.stop) == (1, 3)
    assert (x.start, x.stop) == (4, 8)


def test_build_rubin_store_writes_image_npy_contract(tmp_path) -> None:
    root = tmp_path / "rubin-products"
    rel1 = "visit_image/first.fits"
    rel2 = "visit_image/second.fits"
    (root / "visit_image").mkdir(parents=True)
    _write_rubin_like_fits(root / rel1)
    _write_rubin_like_fits(root / rel2)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(root, rel1, rel2)))

    result = build_rubin_store(
        manifest_path=manifest_path,
        input_root=root,
        output_dir=tmp_path / "rubin_npy",
        group="stack",
        product="visit_image",
        max_files=0,
        roi="2:6,3:8",
        overwrite=False,
    )

    assert result["file_count"] == 2
    assert result["artifacts"]["images"]["shape"] == [1, 4, 5, 2]
    bundle = load_ximage_bundle(tmp_path / "rubin_npy")
    assert validate_bundle(bundle) == []
    assert bundle.images.shape == (1, 4, 5, 2)
    assert bundle.variances.shape == (1, 4, 5, 2)
    assert bundle.masks.shape == (1, 4, 5, 2)
    assert bundle.masks.dtype == np.bool_
    assert bundle.masks[0, 0, 0, :].tolist() == [False, False]
    assert bundle.sky.dtype == np.float32
    assert bundle.sky.shape == (1, 4, 5, 2)
    assert np.count_nonzero(bundle.sky) == 0
    assert result["psf_source_counts"] == {"ExposureSummaryStats": 2}
    assert bundle.psfs.shape == (25, 25, 1, 2)
    assert np.allclose(bundle.psfs[:, :, 0, 0].sum(), 1.0)
    psf_center = bundle.psfs.shape[0] // 2
    assert bundle.psfs[psf_center, psf_center, 0, 0] < 1.0
    assert bundle.psfs[psf_center, psf_center, 0, 0] > bundle.psfs[0, 0, 0, 0]
    assert bundle.mcgauss.shape == (2, 25, 25)
    assert np.allclose(bundle.mcgauss[0].sum(), 1.0)
    assert bundle.metadata["selection"]["psf_size"] == 25
    assert bundle.exp_times.tolist() == [1.0, 1.0]
    assert bundle.metadata["source_kind"] == "rubin-local-fits"
    assert (
        bundle.metadata["selection"]["psf_policy"]
        == "native-cell-psf-or-exposure-summary-gaussian-or-delta"
    )
    assert bundle.metadata["selection"]["rubin_image_formats"] == [
        "legacy-fits"
    ]
    assert (
        bundle.metadata["selection"]["sky_policy"]
        == "zero-placeholder-not-estimated"
    )
    assert (
        bundle.metadata["selection"]["exp_times_policy"]
        == "calibrated-njy-unit-else-fits-header-exptime-or-unit"
    )
    assert bundle.metadata["rubin_files"][0]["image_bunit"] == "nJy"
    assert bundle.metadata["rubin_files"][0]["exp_time"] == 30.0
    assert (
        bundle.metadata["rubin_files"][0]["exp_time_source"]
        == "PRIMARY.header.EXPTIME"
    )
    assert bundle.metadata["rubin_files"][0]["normalization_exp_time"] == 1.0
    assert (
        bundle.metadata["rubin_files"][0]["normalization_exp_time_source"]
        == "IMAGE.header.BUNIT=nJy"
    )
    assert bundle.metadata["rubin_files"][0]["psf_source"] == (
        "ExposureSummaryStats"
    )
    assert bundle.metadata["rubin_files"][0]["psf_sigma"] == 1.0
    assert bundle.metadata["rubin_files"][0]["exposure_summary"]["skyBg"] == (
        12.5
    )
    assert bundle.metadata["selection"]["mask_policy"] == "rubin-bad-planes"
    assert bundle.metadata["rubin_files"][0]["bad_mask_plane_names"] == [
        "BAD"
    ]
    assert bundle.metadata["rubin_files"][0]["bad_mask_fraction"] == 0.05


@pytest.mark.parametrize("failure_stage", ["read", "write", "validation"])
def test_store_overwrite_failure_preserves_existing_bundle(
    tmp_path, monkeypatch, failure_stage
) -> None:
    root = tmp_path / "products"
    root.mkdir()
    _write_rubin_like_fits(root / "visit.fits")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(root, "visit.fits")))
    output = tmp_path / "store"
    kwargs = {
        "manifest_path": manifest,
        "input_root": root,
        "output_dir": output,
        "group": "stack",
        "roi": "2:6,3:8",
    }
    build_rubin_store(**kwargs)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    if failure_stage == "read":

        def failed_read(*args, **kwargs):
            raise OSError("image read failed")

        monkeypatch.setattr(rubin_module, "read_manifest_stack", failed_read)
    elif failure_stage == "write":
        original_save = np.save

        def failed_write(*args, **kwargs):
            original_save(*args, **kwargs)
            raise OSError("array write failed")

        monkeypatch.setattr(rubin_module.np, "save", failed_write)
    else:
        monkeypatch.setattr(
            rubin_module, "validate_bundle", lambda bundle: ["invalid array"]
        )

    with pytest.raises((OSError, ValueError)):
        build_rubin_store(**kwargs, overwrite=True)

    assert {
        path.name: path.read_bytes() for path in output.iterdir()
    } == before
    assert list(tmp_path.glob(".store.tmp-*")) == []


def test_store_successful_overwrite_installs_validated_bundle(
    tmp_path,
) -> None:
    root = tmp_path / "products"
    root.mkdir()
    _write_rubin_like_fits(root / "visit.fits")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(root, "visit.fits")))
    output = tmp_path / "store"
    output.mkdir()
    (output / "previous.txt").write_text("previous store")

    result = build_rubin_store(
        manifest_path=manifest,
        input_root=root,
        output_dir=output,
        group="stack",
        roi="2:6,3:8",
        overwrite=True,
    )

    assert not (output / "previous.txt").exists()
    with load_ximage_bundle(output) as bundle:
        assert validate_bundle(bundle) == []
        assert bundle.images.shape == (1, 4, 5, 1)
        assert bundle.metadata["output_dir"] == str(output.resolve())
    assert result["metadata"] == str(output / "metadata.json")
    assert list(tmp_path.glob(".store.tmp-*")) == []


@pytest.mark.parametrize("destination_kind", ["file", "symlink"])
def test_store_overwrite_rejects_non_directory_destination(
    tmp_path, destination_kind
) -> None:
    root = tmp_path / "products"
    root.mkdir()
    _write_rubin_like_fits(root / "visit.fits")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(root, "visit.fits")))
    output = tmp_path / "store"
    if destination_kind == "file":
        output.write_text("existing file")
    else:
        output.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink or file"):
        build_rubin_store(
            manifest_path=manifest,
            input_root=root,
            output_dir=output,
            group="stack",
            overwrite=True,
        )

    assert (root / "visit.fits").exists()
    if destination_kind == "file":
        assert output.read_text() == "existing file"
    else:
        assert output.is_symlink()
    assert list(tmp_path.glob(".store.tmp-*")) == []


@pytest.mark.parametrize("overlapping_input", ["manifest", "image"])
def test_store_overwrite_rejects_destination_containing_inputs(
    tmp_path, overlapping_input
) -> None:
    root = tmp_path / "products"
    root.mkdir()
    _write_rubin_like_fits(root / "visit.fits")
    output = root if overlapping_input == "image" else tmp_path / "store"
    output.mkdir(exist_ok=True)
    manifest_root = output if overlapping_input == "manifest" else tmp_path
    manifest = manifest_root / "manifest.json"
    manifest.write_text(json.dumps(_manifest(root, "visit.fits")))
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ValueError, match="cannot contain"):
        build_rubin_store(
            manifest_path=manifest,
            input_root=root,
            output_dir=output,
            group="stack",
            overwrite=True,
        )

    assert {
        path.name: path.read_bytes() for path in output.iterdir()
    } == before
    assert (root / "visit.fits").exists()
    assert manifest.exists()


def test_store_reports_and_warns_about_placeholder_psfs(tmp_path) -> None:
    root = tmp_path / "products"
    root.mkdir()
    path = root / "visit.fits"
    _write_rubin_like_fits(path)
    with fits.open(path, mode="update") as hdul:
        del hdul["ExposureSummaryStats"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(root, "visit.fits")))
    output = tmp_path / "store"

    with pytest.warns(RuntimeWarning, match="delta placeholder kernels"):
        result = build_rubin_store(
            manifest_path=manifest,
            input_root=root,
            output_dir=output,
            group="stack",
        )

    assert result["psf_source_counts"] == {"delta_fallback": 1}
    with load_ximage_bundle(output) as bundle:
        psf = np.asarray(bundle.psfs)[:, :, 0, 0]
        assert np.count_nonzero(psf) == 1
        assert psf[psf.shape[0] // 2, psf.shape[1] // 2] == 1.0
        assert np.count_nonzero(bundle.sky) == 0


def test_build_rubin_store_falls_back_to_exptime_for_uncalibrated_image(
    tmp_path,
) -> None:
    root = tmp_path / "rubin-products"
    rel = "visit_image/raw.fits"
    (root / "visit_image").mkdir(parents=True)
    _write_rubin_like_fits(root / rel, image_bunit=None)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(root, rel)))

    build_rubin_store(
        manifest_path=manifest_path,
        input_root=root,
        output_dir=tmp_path / "rubin_npy",
        group="stack",
        product="visit_image",
        roi="2:4,3:5",
        overwrite=False,
    )

    bundle = load_ximage_bundle(tmp_path / "rubin_npy")
    assert bundle.exp_times.tolist() == [30.0]
    assert bundle.metadata["rubin_files"][0]["image_bunit"] is None
    assert bundle.metadata["rubin_files"][0]["normalization_exp_time"] == 30.0
    assert (
        bundle.metadata["rubin_files"][0]["normalization_exp_time_source"]
        == "PRIMARY.header.EXPTIME"
    )


def test_build_rubin_store_ignores_detected_mask_plane_by_default(
    tmp_path,
) -> None:
    root = tmp_path / "rubin-products"
    rel = "visit_image/detected.fits"
    (root / "visit_image").mkdir(parents=True)
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2, 3] = 1 << 5
    mask_header = fits.Header()
    mask_header["MP_BAD"] = 0
    mask_header["MP_DETECTED"] = 5
    _write_rubin_like_fits(root / rel, mask=mask, mask_header=mask_header)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(root, rel)))

    build_rubin_store(
        manifest_path=manifest_path,
        input_root=root,
        output_dir=tmp_path / "rubin_npy",
        group="stack",
        product="visit_image",
        roi="2:4,3:5",
        overwrite=False,
    )

    bundle = load_ximage_bundle(tmp_path / "rubin_npy")
    assert bundle.masks[0, 0, 0, 0]
    assert bundle.sky[0, 0, 0, 0] == 0
    assert bundle.metadata["rubin_files"][0]["bad_mask_fraction"] == 0.0


def test_build_rubin_store_can_align_to_reference_wcs(tmp_path) -> None:
    root = tmp_path / "rubin-products"
    rel1 = "visit_image/reference.fits"
    rel2 = "visit_image/shifted.fits"
    (root / "visit_image").mkdir(parents=True)
    reference = np.arange(49, dtype=np.float32).reshape(7, 7)
    shifted = np.zeros_like(reference)
    shifted[:, 1:] = reference[:, :-1]
    _write_rubin_like_fits(
        root / rel1,
        image=reference,
        image_header=_simple_wcs_header(crpix1=1.0),
    )
    _write_rubin_like_fits(
        root / rel2,
        image=shifted,
        image_header=_simple_wcs_header(crpix1=2.0),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(root, rel1, rel2)))

    result = build_rubin_store(
        manifest_path=manifest_path,
        input_root=root,
        output_dir=tmp_path / "rubin_wcs_npy",
        group="stack",
        product="visit_image",
        roi="1:5,1:5",
        align_wcs=True,
        overwrite=False,
    )

    assert result["alignment_policy"] == "reference-image-wcs"
    bundle = load_ximage_bundle(tmp_path / "rubin_wcs_npy")
    expected = reference[1:5, 1:5]
    assert np.allclose(bundle.images[0, :, :, 0], expected)
    assert np.allclose(bundle.images[0, :, :, 1], expected)
    assert np.allclose(bundle.variances[0, :, :, 0], 2.0)
    assert np.allclose(bundle.variances[0, :, :, 1], 2.0)
    assert (
        bundle.metadata["selection"]["alignment_policy"]
        == "reference-image-wcs"
    )
    assert bundle.metadata["selection"]["reference_index"] == 0
    assert (
        bundle.metadata["rubin_files"][1]["alignment"]["policy"]
        == "reference-image-wcs"
    )
    assert (
        bundle.metadata["rubin_files"][1]["alignment"][
            "target_valid_fraction"
        ]
        == 1.0
    )
    assert (
        bundle.metadata["rubin_files"][1]["alignment"]["variance_resampling"]
        == "bilinear-independent-squared-weights"
    )
    assert (
        bundle.metadata["rubin_files"][1]["alignment"][
            "resampling_covariance"
        ]
        == "not-represented"
    )
    assert (
        bundle.metadata["rubin_files"][1]["alignment"]["mask_resampling"]
        == "bilinear-footprint-bitwise-or"
    )


def test_bilinear_variance_uses_squared_interpolation_weights() -> None:
    variance = np.array([[1.0, 2.0], [3.0, 4.0]])
    target_y = np.array([[0.25, np.nan]])
    target_x = np.array([[0.50, np.nan]])
    valid = np.array([[True, False]])

    result = rubin_module._propagate_bilinear_variance(
        variance,
        target_y=target_y,
        target_x=target_x,
        valid=valid,
    )

    weights = np.array([0.375, 0.375, 0.125, 0.125])
    expected = np.sum(weights**2 * variance.ravel())
    assert result[0, 0] == pytest.approx(expected)
    assert np.isnan(result[0, 1])


def test_bilinear_mask_combines_every_contributing_pixel() -> None:
    mask = np.array([[0, 2], [4, 8]], dtype=np.int32)
    target_y = np.array([[0.25, 0.0, np.nan]])
    target_x = np.array([[0.50, 0.0, np.nan]])
    valid = np.array([[True, True, False]])

    result = rubin_module._propagate_bilinear_mask(
        mask,
        target_y=target_y,
        target_x=target_x,
        valid=valid,
    )

    assert np.array_equal(result, np.array([[14, 0, 1]], dtype=np.int32))


def test_wcs_alignment_propagates_fractional_pixel_variance(tmp_path) -> None:
    root = tmp_path / "rubin-products"
    rel1 = "visit_image/reference.fits"
    rel2 = "visit_image/half-pixel-shift.fits"
    (root / "visit_image").mkdir(parents=True)
    image = np.ones((7, 7), dtype=np.float32)
    _write_rubin_like_fits(
        root / rel1,
        image=image,
        image_header=_simple_wcs_header(crpix1=1.0),
    )
    _write_rubin_like_fits(
        root / rel2,
        image=image,
        image_header=_simple_wcs_header(crpix1=1.5),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(root, rel1, rel2)))

    build_rubin_store(
        manifest_path=manifest_path,
        input_root=root,
        output_dir=tmp_path / "rubin_wcs_npy",
        group="stack",
        product="visit_image",
        roi="1:5,1:5",
        align_wcs=True,
        overwrite=False,
    )

    bundle = load_ximage_bundle(tmp_path / "rubin_wcs_npy")
    assert np.allclose(bundle.variances[0, :, :, 0], 2.0)
    assert np.allclose(bundle.variances[0, :, :, 1], 1.0)
