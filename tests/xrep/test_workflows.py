# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from cuphoton.xrep import make_north_up_wcs
from cuphoton.xrep.workflows import (
    benchmark_backend_variants_reproject_image,
    benchmark_reproject_image,
    compare_backends_reproject_image,
    inspect_image,
    run_reproject_image,
    run_reproject_stack,
)


def _write_test_fits(path: Path, data: np.ndarray) -> Path:
    wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=data.shape,
        pixel_scale_arcsec=0.2,
    )
    header = wcs.to_header()
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(
        path,
        overwrite=True,
    )
    return path


def test_inspect_image_reports_wcs_and_bbox(tmp_path: Path) -> None:
    fits_path = _write_test_fits(
        tmp_path / "image.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5),
    )

    summary = inspect_image(fits_path)

    assert summary["shape"] == [5, 5]
    assert np.isclose(summary["native_pixel_scale_arcsec"], 0.2)
    assert summary["bbox_on_grid"]["width"] > 0
    assert summary["bbox_on_grid"]["height"] > 0


def test_run_reproject_image_persists_summary_and_arrays(
    tmp_path: Path,
) -> None:
    fits_path = _write_test_fits(
        tmp_path / "image.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5),
    )

    result = run_reproject_image(
        input_path=fits_path,
        output_root=tmp_path / "runs",
        name="reproject-image-test",
        hdu=0,
        mask_path=None,
        mask_hdu=None,
        backend="cpu",
        interpolation="bilinear",
        grid_crval_ra=None,
        grid_crval_dec=None,
        pixel_scale_arcsec=None,
        mapping_grid_step=1,
        area_scaling=True,
        write_fits=True,
    )

    assert result.run_dir.name == "reproject-image-test"
    assert result.summary["requested_backend"] == "cpu"
    assert result.summary["backend"] == "cpu"
    assert result.summary["device"] == "cpu"
    assert result.summary["dtype"] == "float64"
    assert result.summary["runtime"]["package_version"]
    assert (result.run_dir / "summary.json").exists()
    assert (result.run_dir / result.summary["saved"]["image"]).exists()
    assert (
        result.run_dir / result.summary["saved"]["reprojected_fits"]
    ).exists()


def test_run_reproject_image_accepts_explicit_auto(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fits_path = _write_test_fits(
        tmp_path / "image.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5),
    )
    monkeypatch.setattr(
        "cuphoton.xrep.backends.default_backend",
        lambda: "cpu",
    )

    result = run_reproject_image(
        input_path=fits_path,
        output_root=tmp_path / "runs",
        name="reproject-image-auto-test",
        hdu=0,
        mask_path=None,
        mask_hdu=None,
        backend="auto",
        interpolation="bilinear",
        grid_crval_ra=None,
        grid_crval_dec=None,
        pixel_scale_arcsec=None,
        mapping_grid_step=1,
        area_scaling=True,
        write_fits=False,
    )

    assert result.summary["requested_backend"] == "auto"
    assert result.summary["backend"] == "cpu"


def test_run_reproject_stack_persists_stack_outputs(tmp_path: Path) -> None:
    path_a = _write_test_fits(
        tmp_path / "a.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5),
    )
    path_b = _write_test_fits(
        tmp_path / "b.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5) + 10.0,
    )

    result = run_reproject_stack(
        input_paths=[path_a, path_b],
        output_root=tmp_path / "runs",
        name="reproject-stack-test",
        hdu=0,
        backend="cpu",
        interpolation="lanczos3",
        grid_crval_ra=None,
        grid_crval_dec=None,
        pixel_scale_arcsec=None,
        mapping_grid_step=1,
        area_scaling=True,
        write_fits=True,
    )

    assert result.summary["stack_shape"][0] == 2
    assert result.summary["runtime"]["backend"] == "cpu"
    assert result.summary["device"] == "cpu"
    assert (result.run_dir / "summary.json").exists()
    assert (result.run_dir / result.summary["saved"]["stack"]).exists()
    assert (result.run_dir / result.summary["saved"]["stack_fits"]).exists()
    payload = json.loads((result.run_dir / "summary.json").read_text())
    assert payload["workflow"] == "reproject-stack"


def test_benchmark_reproject_image_reports_split_timings(
    tmp_path: Path,
) -> None:
    fits_path = _write_test_fits(
        tmp_path / "image.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5),
    )

    result = benchmark_reproject_image(
        input_path=fits_path,
        output_root=tmp_path / "runs",
        name="benchmark-reproject-image-test",
        hdu=0,
        mask_path=None,
        mask_hdu=None,
        backend="cpu",
        interpolation="bilinear",
        grid_crval_ra=None,
        grid_crval_dec=None,
        pixel_scale_arcsec=None,
        mapping_grid_step=1,
        area_scaling=True,
        write_fits=True,
        repeats=2,
        warmup=0,
    )

    assert result.summary["workflow"] == "benchmark-reproject-image"
    timings = result.summary["timings"]
    assert set(timings) == {
        "load_seconds",
        "mapping_seconds",
        "prepare_seconds",
        "backend_seconds",
        "total_seconds",
    }
    for payload in timings.values():
        assert payload["mean"] >= 0.0
        assert payload["max"] >= payload["min"]
    assert (result.run_dir / result.summary["saved"]["image"]).exists()
    assert (result.run_dir / result.summary["saved"]["timings_json"]).exists()


def test_compare_backends_reports_parity_and_timings(
    tmp_path: Path,
) -> None:
    fits_path = _write_test_fits(
        tmp_path / "image.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5),
    )

    result = compare_backends_reproject_image(
        input_path=fits_path,
        output_root=tmp_path / "runs",
        name="compare-backends-test",
        hdu=0,
        mask_path=None,
        mask_hdu=None,
        backends=["cpu"],
        reference_backend="cpu",
        interpolation="lanczos3",
        grid_crval_ra=None,
        grid_crval_dec=None,
        pixel_scale_arcsec=None,
        mapping_grid_step=1,
        area_scaling=True,
        write_fits=True,
        repeats=2,
        warmup=0,
        atol=1e-9,
        rtol=1e-12,
    )

    assert result.summary["workflow"] == "compare-backends"
    assert result.summary["backends"] == ["cpu"]
    assert result.summary["parity"]["ok"] is True
    assert result.summary["parity"]["comparisons"]["cpu"]["ok"] is True
    assert "cpu" in result.summary["timings"]
    assert (
        result.run_dir / result.summary["saved"]["comparisons_json"]
    ).exists()
    assert (result.run_dir / result.summary["saved"]["timings_json"]).exists()


def test_benchmark_backend_variants_reports_masks_and_parity(
    tmp_path: Path,
) -> None:
    fits_path = _write_test_fits(
        tmp_path / "image.fits",
        np.arange(25, dtype=np.float64).reshape(5, 5),
    )

    result = benchmark_backend_variants_reproject_image(
        input_path=fits_path,
        output_root=tmp_path / "runs",
        name="benchmark-backend-variants-test",
        hdu=0,
        mask_path=None,
        mask_hdu=None,
        variants=["cpu"],
        reference_variant="cpu",
        mask_cases=["none", "mask"],
        interpolation="lanczos3",
        grid_crval_ra=None,
        grid_crval_dec=None,
        pixel_scale_arcsec=None,
        mapping_grid_step=1,
        area_scaling=True,
        write_fits=True,
        repeats=2,
        warmup=0,
        atol=1e-9,
        rtol=1e-12,
    )

    assert result.summary["workflow"] == "benchmark-backend-variants"
    assert result.summary["mask_cases"] == ["none", "mask"]
    assert result.summary["parity"]["ok"] is True
    assert "cpu" in result.summary["host_timings"]["none"]
    assert "cpu" in result.summary["host_timings"]["mask"]
    assert result.summary["device_timings"] == {}
    assert (
        result.run_dir / result.summary["saved"]["comparisons_json"]
    ).exists()
    assert (
        result.run_dir / result.summary["saved"]["host_timings_json"]
    ).exists()
