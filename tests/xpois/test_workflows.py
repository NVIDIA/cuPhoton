# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xpois import workflows
from cuphoton.xpois.ois import GaussianBasisComponent


def _compact_source_image(shape: tuple[int, int]) -> np.ndarray:
    y_coords, x_coords = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        indexing="ij",
    )
    image = np.zeros(shape, dtype=np.float64)
    stars = [
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ]
    for cy, cx, sigma, amp in stars:
        image += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    return image


def test_run_constant_kernel_fit_cleans_failed_run_dir(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    np.save(
        reference, np.ones((16, 16), dtype=np.float64), allow_pickle=False
    )
    np.save(target, np.ones((16, 16), dtype=np.float64), allow_pickle=False)

    with pytest.raises(ValueError):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path,
            name="fit-fail",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(15, 15),
            components=[GaussianBasisComponent(sigma=1.5, degree=6)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=False,
        )

    assert not (tmp_path / "fit-fail").exists()


def test_run_constant_kernel_fit_uses_workflow_prefix_and_name(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="subtract-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
        workflow_name="subtract",
        run_prefix="subtract",
    )

    assert result.summary["workflow"] == "subtract"
    assert result.summary["requested_backend"] == "auto"
    assert result.summary["backend"] == "cpu"
    assert result.summary["device"] == "cpu"
    assert result.summary["dtype"] == "float64"
    assert result.summary["runtime"]["package_version"]
    assert result.run_dir.name == "subtract-run"


def test_benchmark_constant_kernel_backends_writes_parity_artifacts(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.benchmark_constant_kernel_backends(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="backend-benchmark",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
        backends=["cpu"],
        reference_backend="cpu",
        repeats=2,
        warmup=0,
    )

    assert result.summary["workflow"] == "benchmark-backends"
    assert result.summary["backends"] == ["cpu"]
    assert result.summary["runtimes"]["cpu"]["device"] == "cpu"
    assert result.summary["parity"]["ok"] is True
    assert result.summary["timings"]["cpu"]["count"] == 2
    assert result.summary["timings"]["cpu"]["best"] >= 0.0
    assert result.summary["parity"]["comparisons"]["cpu"]["ok"] is True
    assert (result.run_dir / result.summary["saved"]["timings_json"]).exists()
    assert (
        result.run_dir / result.summary["saved"]["comparisons_json"]
    ).exists()
    assert (result.run_dir / result.summary["saved"]["cpu_kernel"]).exists()


def test_run_constant_kernel_fit_auto_stamp_mask_records_metadata(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    arr = _compact_source_image((64, 64))
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="auto-stamps-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=variance,
        variance_hdu=None,
        fit_mask_path=None,
        auto_stamp_mask=True,
        auto_stamp_size=15,
        auto_stamp_count=2,
        auto_peak_percentile=98.0,
        background_degree=0,
        flux_conserve=False,
    )

    fit_region = result.summary["fit_region"]
    assert fit_region["kind"] == "auto_stamp_mask"
    assert fit_region["selected_count"] == 2
    assert fit_region["stamp_size"] == 15
    assert "fit_mask_metadata" in result.summary["saved"]
    metadata_path = (
        result.run_dir / result.summary["saved"]["fit_mask_metadata"]
    )
    assert metadata_path.exists()


def test_run_constant_kernel_fit_writes_interactive_and_numeric_review(
    tmp_path: Path,
) -> None:
    has_bokeh = importlib.util.find_spec("bokeh") is not None

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    arr = _compact_source_image((64, 64))
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="review-artifacts-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=variance,
        variance_hdu=None,
        fit_mask_path=None,
        auto_stamp_mask=True,
        auto_stamp_size=15,
        auto_stamp_count=2,
        auto_peak_percentile=98.0,
        background_degree=0,
        flux_conserve=False,
    )

    saved = result.summary["saved"]
    assert "review_hotspots_metadata" in saved
    metadata_path = result.run_dir / saved["review_hotspots_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_name"] == "review-artifacts-run"
    assert "review_metrics" in metadata
    assert isinstance(metadata["hotspots"], list)
    for key in (
        "review_overview",
        "review_stamps",
        "review_kernel",
        "review_sigma",
        "review_hotspots",
        "review_html",
        "review_manifest",
    ):
        assert key not in saved
    assert not list((result.run_dir / "artifacts").glob("review_*.png"))
    if has_bokeh:
        assert "review_bokeh_html" in saved
        assert (result.run_dir / saved["review_bokeh_html"]).exists()
    else:
        assert "review_bokeh_html" not in saved


def test_evaluate_subtraction_run_reports_fit_region_metrics(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="eval-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )
    evaluation = workflows.evaluate_subtraction_run(result.run_dir)

    assert evaluation["fit_region_pixel_count"] == (64 - 8) * (64 - 8)
    assert "fit_region_residual_std" in evaluation


def test_evaluate_subtraction_run_uses_saved_explicit_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    mask = np.zeros_like(arr, dtype=bool)
    mask[10:30, 10:30] = True
    np.save(mask_path, mask, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="masked-eval-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=mask_path,
        background_degree=0,
        flux_conserve=False,
    )
    evaluation = workflows.evaluate_subtraction_run(result.run_dir)

    clipped_mask = mask.copy()
    clipped_mask[:4, :] = False
    clipped_mask[-4:, :] = False
    clipped_mask[:, :4] = False
    clipped_mask[:, -4:] = False
    assert evaluation["fit_region_pixel_count"] == int(clipped_mask.sum())


def test_evaluate_subtraction_run_works_after_run_dir_move(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    mask = np.zeros_like(arr, dtype=bool)
    mask[12:28, 12:28] = True
    np.save(mask_path, mask, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="movable-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=mask_path,
        background_degree=0,
        flux_conserve=False,
    )

    moved = tmp_path / "moved-run"
    shutil.copytree(result.run_dir, moved)
    evaluation = workflows.evaluate_subtraction_run(moved)

    clipped_mask = mask.copy()
    clipped_mask[:4, :] = False
    clipped_mask[-4:, :] = False
    clipped_mask[:, :4] = False
    clipped_mask[:, -4:] = False
    assert evaluation["fit_region_pixel_count"] == int(clipped_mask.sum())
    summary = (moved / "summary.json").read_text(encoding="utf-8")
    assert '"fit_mask": "artifacts/fit_mask.npy"' in summary


def test_evaluate_subtraction_run_rejects_missing_explicit_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    mask = np.zeros_like(arr, dtype=bool)
    mask[10:30, 10:30] = True
    np.save(mask_path, mask, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="missing-mask-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        variance_hdu=None,
        fit_mask_path=mask_path,
        background_degree=0,
        flux_conserve=False,
    )
    (result.run_dir / "artifacts" / "fit_mask.npy").unlink()

    with pytest.raises(FileNotFoundError):
        workflows.evaluate_subtraction_run(result.run_dir)


def test_run_constant_kernel_fit_rejects_malformed_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    mask_path = tmp_path / "fit-mask.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    mask = np.zeros_like(arr, dtype=np.float64)
    mask[10:30, 10:30] = 0.5
    np.save(mask_path, mask, allow_pickle=False)

    with pytest.raises(ValueError):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path,
            name="bad-mask-run",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(9, 9),
            components=[GaussianBasisComponent(sigma=1.5, degree=0)],
            variance_path=None,
            fit_mask_path=mask_path,
            background_degree=0,
            flux_conserve=False,
        )


def test_evaluate_subtraction_run_rejects_missing_interior_fit_mask(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="missing-interior-mask-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )
    (result.run_dir / "artifacts" / "fit_mask.npy").unlink()

    with pytest.raises(FileNotFoundError):
        workflows.evaluate_subtraction_run(result.run_dir)


def test_evaluate_subtraction_run_rejects_malformed_saved_fit_mask(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bad-fit-mask-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    fit_mask = np.zeros((16, 16), dtype=np.float64)
    fit_mask[4:12, 4:12] = 2.0
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        workflows.evaluate_subtraction_run(run_dir)


def test_evaluate_subtraction_run_rejects_mismatched_saved_fit_mask_shape(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bad-fit-mask-shape-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    fit_mask = np.ones((15, 15), dtype=bool)
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        workflows.evaluate_subtraction_run(run_dir)


def test_run_constant_kernel_fit_uses_timestamp_name_by_default(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name=None,
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    assert result.run_dir.name.startswith("fit-kernel-")
    assert result.run_dir.name != "fit-kernel"


def test_run_constant_kernel_fit_rejects_unsafe_run_name(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    with pytest.raises(ValueError):
        workflows.run_constant_kernel_fit(
            reference_path=reference,
            target_path=target,
            output_root=tmp_path,
            name="../escape",
            reference_hdu=None,
            target_hdu=None,
            kernel_shape=(9, 9),
            components=[GaussianBasisComponent(sigma=1.5, degree=0)],
            variance_path=None,
            fit_mask_path=None,
            background_degree=0,
            flux_conserve=False,
        )


def test_non_finite_inputs_are_excluded_from_saved_statistics(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    reference_arr = arr.copy()
    target_arr = arr.copy()
    reference_arr[1, 1] = np.nan
    reference_arr[24, 24] = np.nan
    target_arr[30, 30] = np.nan
    np.save(reference, reference_arr, allow_pickle=False)
    np.save(target, target_arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="non-finite-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    residual = np.load(result.run_dir / "artifacts" / "residual.npy")
    assert np.isnan(residual[0, 0])
    assert np.isnan(residual[24, 24])
    assert np.isnan(residual[30, 30])
    assert np.isfinite(result.summary["residual_mean"])
    assert np.isfinite(result.summary["residual_std"])

    evaluation = workflows.evaluate_subtraction_run(result.run_dir)

    assert np.isfinite(evaluation["residual_mean"])
    assert np.isfinite(evaluation["residual_std"])
    assert np.isfinite(evaluation["abs_residual_max"])


def test_saved_residual_excludes_convolution_margin(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="margin-run",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    residual = np.load(result.run_dir / "artifacts" / "residual.npy")

    assert np.isnan(residual[0, 0])
    assert np.isnan(residual[3, 3])
    assert np.isfinite(residual[4, 4])


def test_run_constant_kernel_fit_records_effective_variance_hdu(
    tmp_path: Path,
) -> None:
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    variance = tmp_path / "variance.fits"
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = 1.0
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                np.ones((64, 64), dtype=np.float64),
                name="VARIANCE",
            ),
        ]
    ).writeto(variance)

    result = workflows.run_constant_kernel_fit(
        reference_path=reference,
        target_path=target,
        output_root=tmp_path,
        name="variance-hdu-run",
        reference_hdu=0,
        target_hdu=0,
        kernel_shape=(9, 9),
        components=[GaussianBasisComponent(sigma=1.5, degree=0)],
        variance_path=variance,
        variance_hdu=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=False,
    )

    assert result.summary["variance_hdu"] == 1


def test_evaluate_subtraction_run_counts_spike_when_mad_is_zero(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "spike-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    residual[8, 8] = 25.0
    fit_mask = np.ones_like(residual, dtype=bool)
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    evaluation = workflows.evaluate_subtraction_run(run_dir)

    assert evaluation["pixels_gt_3sigma"] == 1
    assert evaluation["pixels_gt_5sigma"] == 1


def test_evaluate_subtraction_run_reports_zero_outliers_for_constant_field(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "constant-run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    residual = np.zeros((16, 16), dtype=np.float64)
    fit_mask = np.ones_like(residual, dtype=bool)
    np.save(artifacts_dir / "residual.npy", residual, allow_pickle=False)
    np.save(artifacts_dir / "fit_mask.npy", fit_mask, allow_pickle=False)
    (run_dir / "summary.json").write_text(
        '{"kernel_sum": 1.0}\n',
        encoding="utf-8",
    )

    evaluation = workflows.evaluate_subtraction_run(run_dir)

    assert evaluation["pixels_gt_3sigma"] == 0
    assert evaluation["pixels_gt_5sigma"] == 0
