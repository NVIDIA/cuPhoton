# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from astropy.io import fits
from astropy.wcs import WCS

from cuphoton.ximage.evaluation import (
    _load_catalog_table,
    average_downsample,
    build_catalog_photometry_binned_rows,
    build_catalog_photometry_rows,
    build_rubin_coadd_reference_catalog,
    catalog_photometry_metrics,
    center_crop,
    crop_about_center,
    evaluate_reconstruction_run,
    fourier_sharpness,
    load_pipeline_coadd_roi,
    load_reference_catalog_csv_sources,
    match_detections_to_catalog,
    project_restoration_to_observation_grid,
    psnr,
    resolve_catalog_parquet,
    resolve_pipeline_coadd_fits,
    resolve_reference_catalog_csv,
    resolve_saved_path,
    sky_sigma,
    ssim,
    write_evaluation_artifacts,
)


def test_psnr_and_ssim_identity_cases() -> None:
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert psnr(image, image) == float("inf")
    assert ssim(image, image) == 1.0


def test_fourier_sharpness_prefers_checkerboard() -> None:
    smooth = np.ones((8, 8), dtype=np.float64)
    sharp = np.indices((8, 8)).sum(axis=0) % 2
    assert fourier_sharpness(sharp) > fourier_sharpness(smooth)


def test_sky_sigma_respects_background_mask() -> None:
    image = np.zeros((4, 4), dtype=np.float64)
    image[0, 0] = 100.0
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    assert sky_sigma(image, background_mask=mask) == 0.0


def test_projection_downsamples_and_crops() -> None:
    image = np.arange(64, dtype=np.float64).reshape(1, 8, 8, 1)
    projected = project_restoration_to_observation_grid(
        image,
        target_shape=(2, 2),
        srf=2,
    )
    expected = center_crop(
        average_downsample(image[0, :, :, 0], 2),
        (2, 2),
    )
    np.testing.assert_allclose(projected, expected)


def test_crop_about_center_preserves_odd_sizes() -> None:
    image = np.arange(49, dtype=np.float64).reshape(7, 7)
    cropped = crop_about_center(image, center=(3, 3), height=5, width=5)
    assert cropped.shape == (5, 5)
    np.testing.assert_allclose(cropped, image[1:6, 1:6])


def test_load_pipeline_coadd_roi_aligns_rubin_reference_wcs(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "rubin"
    reference_path = source_dir / "visit.fits"
    reference_path.parent.mkdir(parents=True)

    reference_header = fits.Header()
    reference_header["NAXIS"] = 2
    reference_header["NAXIS1"] = 64
    reference_header["NAXIS2"] = 64
    reference_header["WCSAXES"] = 2
    reference_header["CTYPE1"] = "LINEAR"
    reference_header["CTYPE2"] = "LINEAR"
    reference_header["CRPIX1"] = 1.0
    reference_header["CRPIX2"] = 1.0
    reference_header["CRVAL1"] = 0.0
    reference_header["CRVAL2"] = 0.0
    reference_header["CD1_1"] = 1.0
    reference_header["CD1_2"] = 0.0
    reference_header["CD2_1"] = 0.0
    reference_header["CD2_2"] = 1.0
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((64, 64), dtype=np.float32),
                header=reference_header,
                name="IMAGE",
            ),
        ]
    ).writeto(reference_path)

    coadd_path = tmp_path / "coadd.fits"
    coadd_header = reference_header.copy()
    coadd_header["CRPIX1"] = 6.0
    coadd_header["CRPIX2"] = 6.0
    coadd_data = np.add.outer(
        np.arange(80, dtype=np.float64) * 100.0,
        np.arange(80, dtype=np.float64),
    )
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=coadd_data, header=coadd_header, name="IMAGE"),
        ]
    ).writeto(coadd_path)

    metadata = {
        "source_dir": str(source_dir),
        "selection": {
            "alignment_policy": "reference-image-wcs",
            "reference_index": 0,
            "roi": "10:18,20:28",
        },
        "rubin_files": [{"relative_path": "visit.fits"}],
    }
    aligned = load_pipeline_coadd_roi(
        coadd_path,
        center=(4, 4),
        height=4,
        width=4,
        bundle_metadata=metadata,
    )

    expected = coadd_data[17:21, 27:31]
    np.testing.assert_allclose(aligned, expected)


def test_build_rubin_coadd_reference_catalog_writes_wcs_rows(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "rubin"
    reference_path = source_dir / "visit.fits"
    reference_path.parent.mkdir(parents=True)

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["WCSAXES"] = 2
    header["CTYPE1"] = "LINEAR"
    header["CTYPE2"] = "LINEAR"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CD1_1"] = 1.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((64, 64), dtype=np.float32),
                header=header,
                name="IMAGE",
            ),
        ]
    ).writeto(reference_path)

    coadd_path = tmp_path / "coadd.fits"
    coadd_data = np.zeros((64, 64), dtype=np.float32)
    coadd_data[20:23, 30:33] = 100.0
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=coadd_data, header=header, name="IMAGE"),
        ]
    ).writeto(coadd_path)

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "selection": {
                    "alignment_policy": "reference-image-wcs",
                    "reference_index": 0,
                    "roi": "10:42,20:52",
                },
                "rubin_files": [{"relative_path": "visit.fits"}],
            }
        )
    )
    output_csv = tmp_path / "catalog.csv"

    summary = build_rubin_coadd_reference_catalog(
        coadd_fits_path=coadd_path,
        store_dir=store_dir,
        output_csv=output_csv,
        threshold=5.0,
        minarea=1,
    )

    rows = list(csv.DictReader(output_csv.open()))
    assert summary["source_count"] == 1
    assert len(rows) == 1
    assert float(rows[0]["aperture_flux"]) > 0.0
    assert 9.0 <= float(rows[0]["x"]) <= 13.0
    assert 9.0 <= float(rows[0]["y"]) <= 13.0
    assert 29.0 <= float(rows[0]["reference_x"]) <= 33.0
    assert 19.0 <= float(rows[0]["reference_y"]) <= 23.0
    assert Path(summary["output_summary"]).exists()


def test_resolve_saved_path_handles_relative_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved = resolve_saved_path(run_dir, Path("artifacts") / "image.npy")
    assert resolved == (run_dir / "artifacts" / "image.npy").resolve()


def test_evaluate_reconstruction_run_reports_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import cuphoton.ximage.evaluation as evaluation

    original_load = evaluation.load_ximage_bundle
    closed: list[bool] = []

    def tracked_load(*args, **kwargs):
        bundle = original_load(*args, **kwargs)
        original_close = bundle.close

        def close() -> None:
            closed.append(True)
            original_close()

        bundle.close = close
        return bundle

    monkeypatch.setattr(evaluation, "load_ximage_bundle", tracked_load)
    data_dir = tmp_path / "data"
    npy_dir = data_dir
    npy_dir.mkdir(parents=True)

    images = np.zeros((1, 8, 8, 2), dtype=np.float64)
    images[:, 3:5, 3:5, :] = 2.0
    variances = np.ones((1, 8, 8, 2), dtype=np.float64)
    masks = np.ones((1, 8, 8, 12, 2), dtype=np.uint8)
    sky = np.zeros((1, 8, 8, 2), dtype=np.uint8)
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    exp_times = np.ones((2,), dtype=np.float64)
    mcgauss = np.zeros((2, 3, 3), dtype=np.float64)

    np.save(npy_dir / "images.npy", images, allow_pickle=False)
    np.save(npy_dir / "variances.npy", variances, allow_pickle=False)
    np.save(npy_dir / "masks.npy", masks, allow_pickle=False)
    np.save(npy_dir / "sky.npy", sky, allow_pickle=False)
    np.save(npy_dir / "psfs.npy", psfs, allow_pickle=False)
    np.save(npy_dir / "exp_times.npy", exp_times, allow_pickle=False)
    np.save(npy_dir / "mcgauss.npy", mcgauss, allow_pickle=False)
    (npy_dir / "metadata.json").write_text("{}")

    coadd_dir = data_dir / "coadd"
    coadd_dir.mkdir(parents=True)
    coadd_path = coadd_dir / "coadd.fits"
    coadd_hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.ones((8, 8), dtype=np.float64),
                name="IMAGE",
            ),
        ]
    )
    coadd_hdul.writeto(coadd_path)

    catalog_dir = data_dir / "catalogs"
    catalog_dir.mkdir(parents=True)
    catalog_path = catalog_dir / "catalog.parq"
    wcs = WCS(coadd_hdul["IMAGE"].header)
    ra, dec = wcs.pixel_to_world_values([4.0, 5.0], [4.0, 5.0])
    catalog_table = pa.table(
        {
            "coord_ra": pa.array(ra, type=pa.float64()),
            "coord_dec": pa.array(dec, type=pa.float64()),
            "detect_isPrimary": pa.array([True, True]),
            "xy_flag": pa.array([False, False]),
            "refExtendedness": pa.array([0.0, 1.0], type=pa.float64()),
            "i_extendedness": pa.array([0.0, 1.0], type=pa.float64()),
            "i_psfFlux": pa.array([1000.0, 500.0], type=pa.float64()),
            "i_psfFlux_flag": pa.array([False, False]),
            "i_cModelFlux": pa.array([900.0, 700.0], type=pa.float64()),
            "i_cModel_flag": pa.array([False, False]),
            "i_calibFlux": pa.array([950.0, 650.0], type=pa.float64()),
            "i_calibFlux_flag": pa.array([False, False]),
        }
    )
    pq.write_table(catalog_table, catalog_path)

    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    image_path = artifacts / "image.npy"
    np.save(
        image_path,
        np.ones((1, 4, 4, 1), dtype=np.float64),
        allow_pickle=False,
    )

    config = {
        "backend": "torch",
        "algorithm": "l2",
        "data": {"base": str(data_dir)},
        "output": {
            "root_dir": str(tmp_path / "runs"),
            "name": "example",
        },
        "roi": {"center": [4, 4], "height": 4, "width": 4},
        "mask": {"mode": "none", "dilation_factor": 1},
        "srf": 1,
        "x_initialization": "median",
        "ucf": 2.0,
        "learning_rate": 1.0,
        "max_iter": 2,
        "stop_tol": 1.0e-6,
        "padding": "SAME",
        "verbose": False,
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "backend": "torch",
                "algorithm": "l2",
                "selected_exposures": [0, 1],
                "saved": {"image": "artifacts/image.npy"},
            }
        )
    )

    result = evaluate_reconstruction_run(
        run_dir,
        pipeline_coadd_fits=coadd_path,
        catalog_parquet=catalog_path,
        write_artifacts=True,
    )
    assert np.isfinite(result["restoration"]["sharpness_sf"])
    assert result["synthetic_coadd"]["method"] == "median"
    assert result["exposures"]["count"] == 2
    assert "pipeline_coadd" in result
    assert resolve_pipeline_coadd_fits(explicit_path=coadd_path) == coadd_path
    assert result["photometry"]["available"] is True
    assert result["photometry"]["baseline_name"] == "pipeline_coadd"
    assert "saved" in result
    assert Path(result["saved"]["evaluation_summary"]).exists()
    assert closed == [True]


def test_evaluate_reconstruction_run_closes_bundle_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import cuphoton.ximage.evaluation as evaluation

    run = SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(base=tmp_path, prefer=("npy",))
        )
    )

    class FakeBundle:
        closed = False

        def close(self) -> None:
            self.closed = True

    bundle = FakeBundle()
    monkeypatch.setattr(
        evaluation, "load_reconstruction_run", lambda path: run
    )
    monkeypatch.setattr(
        evaluation,
        "load_ximage_bundle",
        lambda *args, **kwargs: bundle,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(evaluation, "_evaluate_reconstruction_bundle", fail)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate_reconstruction_run(tmp_path / "run")
    assert bundle.closed is True


def test_catalog_photometry_metrics_reports_matches(
    tmp_path: Path,
) -> None:
    coadd_path = tmp_path / "coadd.fits"
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 16
    header["NAXIS2"] = 16
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 8.0
    header["CRPIX2"] = 8.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CD1_1"] = -1.0e-4
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0e-4
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((16, 16), dtype=np.float64),
                header=header,
                name="IMAGE",
            ),
        ]
    ).writeto(coadd_path)

    wcs = WCS(header)
    ra, dec = wcs.pixel_to_world_values([4.0, 12.0], [5.0, 11.0])
    catalog_path = tmp_path / "catalog.parq"
    table = pa.table(
        {
            "objectId": pa.array([1001, 1002], type=pa.int64()),
            "coord_ra": pa.array(ra, type=pa.float64()),
            "coord_dec": pa.array(dec, type=pa.float64()),
            "detect_isPrimary": pa.array([True, True]),
            "xy_flag": pa.array([False, False]),
            "refExtendedness": pa.array([0.0, 1.0], type=pa.float64()),
            "i_extendedness": pa.array([0.0, 1.0], type=pa.float64()),
            "i_psfFlux": pa.array([1000.0, 500.0], type=pa.float64()),
            "i_psfFlux_flag": pa.array([False, False]),
            "i_cModelFlux": pa.array([900.0, 700.0], type=pa.float64()),
            "i_cModel_flag": pa.array([False, False]),
            "i_calibFlux": pa.array([950.0, 650.0], type=pa.float64()),
            "i_calibFlux_flag": pa.array([False, False]),
        }
    )
    pq.write_table(table, catalog_path)

    result = catalog_photometry_metrics(
        baseline_flux=np.array([1100.0, 600.0]),
        restoration_flux=np.array([1200.0, 800.0]),
        valid_mask=np.array([True, True]),
        detection_x=np.array([4.2, 12.1]),
        detection_y=np.array([5.1, 10.9]),
        baseline_name="pipeline_coadd",
        catalog_parquet=catalog_path,
        catalog_band="i",
        catalog_match_radius_px=2.0,
        coadd_fits_path=coadd_path,
        roi_center=(8, 8),
        roi_shape=(16, 16),
    )
    assert result["available"] is True
    assert result["matched_catalog_count"] == 2
    assert result["reference_flux_column"] in {
        "i_calibFlux",
        "i_cModelFlux",
        "i_psfFlux",
        "auto",
    }
    assert result["catalog_id_column"] == "objectId"
    payload = result["_artifact_payload"]
    assert payload["catalog_id"].tolist() == ["1001", "1002"]
    rows = build_catalog_photometry_rows(payload)
    assert [row["catalog_id"] for row in rows] == ["1001", "1002"]
    assert all(row["catalog_id_column"] == "objectId" for row in rows)
    assert np.allclose(
        [row["catalog_x"] for row in rows],
        [4.0, 12.0],
    )
    assert np.allclose(
        [row["catalog_ra_deg"] for row in rows],
        ra,
    )


def test_catalog_photometry_metrics_accepts_reference_csv(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "rubin_reference.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "x", "y", "aperture_flux"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "0",
                    "x": "4.0",
                    "y": "5.0",
                    "aperture_flux": "1000.0",
                },
                {
                    "source_id": "1",
                    "x": "12.0",
                    "y": "11.0",
                    "aperture_flux": "500.0",
                },
            ]
        )

    result = catalog_photometry_metrics(
        baseline_flux=np.array([1100.0, 600.0]),
        restoration_flux=np.array([1200.0, 800.0]),
        valid_mask=np.array([True, True]),
        detection_x=np.array([4.2, 12.1]),
        detection_y=np.array([5.1, 10.9]),
        baseline_name="pipeline_coadd",
        reference_catalog_csv=catalog_path,
        catalog_band="r",
        catalog_match_radius_px=2.0,
        roi_center=None,
        roi_shape=(16, 16),
    )

    assert result["available"] is True
    assert result["catalog_kind"] == "rubin_reference_catalog_csv"
    assert result["catalog_path"] == str(catalog_path.resolve())
    assert result["matched_catalog_count"] == 2
    assert result["reference_flux_column"] == "aperture_flux"
    assert result["catalog_id_column"] is None
    payload = result["_artifact_payload"]
    assert payload["catalog_id"].tolist() == [None, None]
    rows = build_catalog_photometry_rows(payload)
    assert [row["catalog_id"] for row in rows] == [None, None]
    assert [row["catalog_x"] for row in rows] == pytest.approx([4.0, 12.0])
    assert [row["catalog_ra_deg"] for row in rows] == [None, None]


def test_catalog_photometry_metrics_offsets_reference_csv_to_eval_crop(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "rubin_reference.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "x", "y", "aperture_flux"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_id": "0",
                    "x": "52.0",
                    "y": "53.0",
                    "aperture_flux": "1000.0",
                },
                {
                    "source_id": "1",
                    "x": "12.0",
                    "y": "11.0",
                    "aperture_flux": "500.0",
                },
            ]
        )

    result = catalog_photometry_metrics(
        baseline_flux=np.array([1100.0]),
        restoration_flux=np.array([1200.0]),
        valid_mask=np.array([True]),
        detection_x=np.array([10.0]),
        detection_y=np.array([11.0]),
        baseline_name="pipeline_coadd",
        reference_catalog_csv=catalog_path,
        catalog_band="r",
        catalog_match_radius_px=1.0,
        roi_center=(50, 50),
        roi_shape=(16, 16),
    )

    assert result["available"] is True
    assert result["catalog_source_count"] == 1
    assert result["matched_catalog_count"] == 1


def test_catalog_photometry_metrics_accepts_rubin_csv_radec(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "rubin"
    reference_path = source_dir / "visit.fits"
    reference_path.parent.mkdir(parents=True)

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["WCSAXES"] = 2
    header["CTYPE1"] = "LINEAR"
    header["CTYPE2"] = "LINEAR"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CD1_1"] = 1.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((64, 64), dtype=np.float32),
                header=header,
                name="IMAGE",
            ),
        ]
    ).writeto(reference_path)

    world = WCS(header).all_pix2world([[28.0, 20.0], [29.0, 21.0]], 0)
    catalog_path = tmp_path / "rubin_object.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "objectId",
                "coord_ra",
                "coord_dec",
                "detect_isPrimary",
                "r_psfFlux",
                "r_psfFlux_flag",
                "r_extendedness",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "objectId": "1",
                    "coord_ra": str(world[0][0]),
                    "coord_dec": str(world[0][1]),
                    "detect_isPrimary": "true",
                    "r_psfFlux": "1000.0",
                    "r_psfFlux_flag": "false",
                    "r_extendedness": "0.0",
                },
                {
                    "objectId": "2",
                    "coord_ra": str(world[1][0]),
                    "coord_dec": str(world[1][1]),
                    "detect_isPrimary": "false",
                    "r_psfFlux": "500.0",
                    "r_psfFlux_flag": "false",
                    "r_extendedness": "1.0",
                },
            ]
        )

    result = catalog_photometry_metrics(
        baseline_flux=np.array([1100.0]),
        restoration_flux=np.array([1200.0]),
        valid_mask=np.array([True]),
        detection_x=np.array([8.1]),
        detection_y=np.array([10.1]),
        baseline_name="pipeline_coadd",
        reference_catalog_csv=catalog_path,
        catalog_band="r",
        catalog_match_radius_px=1.0,
        roi_center=None,
        roi_shape=(32, 32),
        bundle_metadata={
            "source_dir": str(source_dir),
            "selection": {
                "alignment_policy": "reference-image-wcs",
                "reference_index": 0,
                "roi": "10:42,20:52",
            },
            "rubin_files": [{"relative_path": "visit.fits"}],
        },
    )

    assert result["available"] is True
    assert result["catalog_kind"] == "rubin_catalog_csv"
    assert result["catalog_provenance"] == {
        "source": "csv",
    }
    assert result["catalog_source_count"] == 1
    assert result["matched_catalog_count"] == 1
    assert result["reference_flux_column"] == "r_psfFlux"
    assert result["pointlike_catalog_count"] == 1
    assert result["catalog_id_column"] == "objectId"
    payload = result["_artifact_payload"]
    assert payload["catalog_id"].tolist() == ["1"]
    rows = build_catalog_photometry_rows(payload)
    assert rows[0]["catalog_id"] == "1"
    assert rows[0]["catalog_id_column"] == "objectId"
    assert rows[0]["catalog_x"] == pytest.approx(8.0)
    assert rows[0]["catalog_y"] == pytest.approx(10.0)
    assert rows[0]["catalog_ra_deg"] == pytest.approx(world[0][0])
    assert rows[0]["catalog_dec_deg"] == pytest.approx(world[0][1])


@pytest.mark.parametrize("identity_column", ["objectId", "object_id", "id"])
def test_reference_pixel_catalog_retains_stable_identity(
    tmp_path: Path,
    identity_column: str,
) -> None:
    path = tmp_path / "reference.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                identity_column,
                "x",
                "y",
                "ra_deg",
                "dec_deg",
                "aperture_flux",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                identity_column: "9007199254740993",
                "x": "3.5",
                "y": "4.5",
                "ra_deg": "150.1",
                "dec_deg": "2.1",
                "aperture_flux": "1000",
            }
        )

    catalog = load_reference_catalog_csv_sources(
        path,
        roi_center=None,
        roi_shape=(16, 16),
    )

    assert catalog["catalog_id_column"] == identity_column
    assert catalog["catalog_id"].tolist() == ["9007199254740993"]
    assert catalog["x"].tolist() == [3.5]
    assert catalog["y"].tolist() == [4.5]
    assert catalog["catalog_ra_deg"].tolist() == [150.1]
    assert catalog["catalog_dec_deg"].tolist() == [2.1]


@pytest.mark.parametrize(
    "sidecar_content",
    [
        (
            '{"artifact_kind": "unrelated-note", '
            '"products": {"note": "not-a-catalog"}}'
        ),
        "This adjacent file is not JSON.",
    ],
)
def test_explicit_reference_catalog_ignores_adjacent_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_content: str,
) -> None:
    object_csv = tmp_path / "object.csv"
    object_csv.write_text("objectId,x,y,aperture_flux\n1,4,5,100\n")
    sidecar = tmp_path / "export_manifest.json"
    sidecar.write_text(sidecar_content)
    original_read_text = Path.read_text

    def read_text_without_sidecar(path: Path, *args, **kwargs):
        if path == sidecar:
            pytest.fail(
                "Explicit catalog evaluation read an unrelated sidecar"
            )
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text_without_sidecar)
    result = catalog_photometry_metrics(
        baseline_flux=np.array([100.0]),
        restoration_flux=np.array([110.0]),
        valid_mask=np.array([True]),
        detection_x=np.array([4.0]),
        detection_y=np.array([5.0]),
        baseline_name="synthetic_coadd",
        reference_catalog_csv=object_csv,
        catalog_band="r",
        catalog_match_radius_px=1.0,
        roi_center=None,
        roi_shape=(16, 16),
    )

    assert result["available"] is True
    assert result["matched_catalog_count"] == 1
    assert result["catalog_path"] == str(object_csv)
    assert result["catalog_provenance"] == {"source": "csv"}
    assert "not-a-catalog" not in str(result)


def test_resolve_reference_catalog_csv_accepts_export_directory(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "rubin-export"
    export_dir.mkdir()
    (export_dir / "source.csv").write_text("sourceId\n1\n")
    object_csv = export_dir / "object.csv"
    object_csv.write_text("objectId\n1\n")

    assert resolve_reference_catalog_csv(explicit_path=export_dir) == (
        object_csv.resolve()
    )


def test_resolve_reference_catalog_csv_rejects_ambiguous_directory(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "rubin-export"
    export_dir.mkdir()
    (export_dir / "source.csv").write_text("sourceId\n1\n")
    (export_dir / "forced_source.csv").write_text("objectId\n1\n")

    with pytest.raises(ValueError, match="multiple CSV outputs"):
        resolve_reference_catalog_csv(explicit_path=export_dir)


def test_build_catalog_photometry_binned_rows_reports_subsets() -> None:
    payload = {
        "reference_mag": np.array([20.0, 21.0, 22.0, 23.0]),
        "restoration_minus_baseline_mag": np.array([-0.1, 0.0, 0.2, 0.4]),
        "baseline_residual_mag": np.array([0.1, 0.1, 0.2, 0.3]),
        "restoration_residual_mag": np.array([0.0, 0.1, 0.4, 0.7]),
        "pointlike": np.array([True, True, False, False]),
    }
    rows = build_catalog_photometry_binned_rows(payload)
    assert rows
    assert {row["subset"] for row in rows} >= {
        "all",
        "pointlike",
        "extended",
    }


def test_write_evaluation_artifacts_persists_catalog_outputs(
    tmp_path: Path,
) -> None:
    payload = {
        "reference_mag": np.array([20.0, 21.0]),
        "baseline_mag_cal": np.array([20.1, 21.2]),
        "restoration_mag_cal": np.array([19.9, 20.8]),
        "baseline_residual_mag": np.array([0.1, 0.2]),
        "restoration_residual_mag": np.array([-0.1, -0.2]),
        "restoration_minus_baseline_mag": np.array([-0.2, -0.4]),
        "reference_flux": np.array([1000.0, 500.0]),
        "baseline_flux": np.array([900.0, 450.0]),
        "restoration_flux": np.array([1100.0, 600.0]),
        "pointlike": np.array([True, False]),
        "catalog_id": np.array(["101", "102"], dtype=object),
        "catalog_id_column": np.array(
            ["objectId", "objectId"],
            dtype=object,
        ),
        "catalog_x": np.array([4.0, 12.0]),
        "catalog_y": np.array([5.0, 11.0]),
        "catalog_ra_deg": np.array([150.1, 150.2]),
        "catalog_dec_deg": np.array([2.1, 2.2]),
    }
    evaluation = {
        "photometry": {
            "baseline_name": "pipeline_coadd",
            "catalog_photometry": {"_artifact_payload": payload},
        }
    }
    saved = write_evaluation_artifacts(tmp_path, evaluation)
    assert Path(saved["evaluation_summary"]).exists()
    assert Path(saved["catalog_photometry_matches_csv"]).exists()
    assert Path(saved["catalog_photometry_binned_csv"]).exists()
    rows = build_catalog_photometry_rows(payload)
    assert len(rows) == 2
    with Path(saved["catalog_photometry_matches_csv"]).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["catalog_id"] for row in csv_rows] == ["101", "102"]
    assert [row["catalog_id_column"] for row in csv_rows] == [
        "objectId",
        "objectId",
    ]
    assert [row["catalog_ra_deg"] for row in csv_rows] == ["150.1", "150.2"]


def test_match_detections_to_catalog_uses_all_candidates() -> None:
    result = match_detections_to_catalog(
        detection_x=np.array([0.75, 1.0]),
        detection_y=np.array([0.0, 0.0]),
        catalog_x=np.array([0.8, 0.15]),
        catalog_y=np.array([0.0, 0.0]),
        match_radius_px=0.7,
    )
    assert result["matched_count"] == 2


def test_match_detections_to_catalog_handles_sparse_case() -> None:
    result = match_detections_to_catalog(
        detection_x=np.array([0.0, 2.0]),
        detection_y=np.array([0.0, 0.0]),
        catalog_x=np.array([0.1, 2.1]),
        catalog_y=np.array([0.0, 0.0]),
        match_radius_px=0.5,
    )
    assert result["matched_count"] == 2


def test_resolve_catalog_parquet_accepts_parquet_extension(
    tmp_path: Path,
) -> None:
    catalog_dir = tmp_path / "catalogs"
    catalog_dir.mkdir(parents=True)
    catalog_path = catalog_dir / "catalog.parquet"
    table = pa.table(
        {
            "coord_ra": pa.array([], type=pa.float64()),
            "coord_dec": pa.array([], type=pa.float64()),
        }
    )
    pq.write_table(table, catalog_path)
    assert resolve_catalog_parquet(explicit_path=catalog_path) == catalog_path


def test_load_catalog_table_validates_required_columns(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.parq"
    table = pa.table({"coord_ra": pa.array([1.0], type=pa.float64())})
    pq.write_table(table, catalog_path)
    try:
        _load_catalog_table(catalog_path, band="i")
    except ValueError as exc:
        assert "coord_dec" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_catalog_table_handles_empty_table(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parq"
    table = pa.table(
        {
            "coord_ra": pa.array([], type=pa.float64()),
            "coord_dec": pa.array([], type=pa.float64()),
            "detect_isPrimary": pa.array([], type=pa.bool_()),
            "xy_flag": pa.array([], type=pa.bool_()),
        }
    )
    pq.write_table(table, catalog_path)
    result = _load_catalog_table(catalog_path, band="i")
    assert result["coord_ra"].size == 0
    assert result["reference_flux_name"] == "auto"


def test_resolve_saved_path_supports_relocated_run_dir(
    tmp_path: Path,
) -> None:
    relocated = tmp_path / "relocated" / "example"
    relocated.mkdir(parents=True)
    artifacts = relocated / "artifacts"
    artifacts.mkdir()
    image_path = artifacts / "image.npy"
    np.save(
        image_path,
        np.ones((1, 2, 2, 1), dtype=np.float64),
        allow_pickle=False,
    )

    loaded = np.load(resolve_saved_path(relocated, "artifacts/image.npy"))
    assert loaded.shape == (1, 2, 2, 1)
    loaded_prefixed = np.load(
        resolve_saved_path(relocated, "runs/example/artifacts/image.npy")
    )
    assert loaded_prefixed.shape == (1, 2, 2, 1)


def test_resolve_saved_path_does_not_fall_back_to_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cwd = tmp_path / "cwd"
    (cwd / "artifacts").mkdir(parents=True)
    np.save(
        cwd / "artifacts" / "image.npy",
        np.zeros((1, 1, 1, 1), dtype=np.float64),
        allow_pickle=False,
    )
    monkeypatch.chdir(cwd)

    resolved = resolve_saved_path(run_dir, "artifacts/image.npy")
    assert resolved == (run_dir / "artifacts" / "image.npy").resolve()
    assert not resolved.exists()


def test_photometry_metrics_degrades_on_measurement_error(
    monkeypatch,
) -> None:
    import cuphoton.ximage.evaluation as evaluation

    def boom(image):
        raise ArithmeticError("boom")

    monkeypatch.setattr(evaluation, "subtract_photometry_background", boom)
    result = evaluation.photometry_metrics(
        restoration=np.ones((8, 8), dtype=np.float64),
        baseline=np.ones((8, 8), dtype=np.float64),
        baseline_name="synthetic_coadd",
        threshold=0.1,
        minarea=5,
    )
    assert result["available"] is False
    assert "boom" in result["reason"]


def test_photometry_metrics_reraises_programming_errors(
    monkeypatch,
) -> None:
    import cuphoton.ximage.evaluation as evaluation

    def boom(image):
        raise AttributeError("bad-code")

    monkeypatch.setattr(evaluation, "subtract_photometry_background", boom)
    try:
        evaluation.photometry_metrics(
            restoration=np.ones((8, 8), dtype=np.float64),
            baseline=np.ones((8, 8), dtype=np.float64),
            baseline_name="synthetic_coadd",
            threshold=0.1,
            minarea=5,
        )
    except AttributeError as exc:
        assert "bad-code" in str(exc)
    else:
        raise AssertionError("expected AttributeError")


def test_photometry_metrics_degrades_when_photutils_unavailable(
    monkeypatch,
) -> None:
    import cuphoton.ximage.evaluation as evaluation

    def boom(*args, **kwargs):
        raise RuntimeError("photutils-missing")

    monkeypatch.setattr(evaluation, "detect_sources", boom)
    result = evaluation.photometry_metrics(
        restoration=np.ones((8, 8), dtype=np.float64),
        baseline=np.ones((8, 8), dtype=np.float64),
        baseline_name="synthetic_coadd",
        threshold=0.1,
        minarea=5,
    )
    assert result["available"] is False
    assert "photutils-missing" in result["reason"]
