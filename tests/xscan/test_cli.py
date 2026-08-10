# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cuphoton.core.cli import get_component, run_component
from cuphoton.xscan import commands as xscan_commands
from cuphoton.xscan import dataset as xscan_dataset
from cuphoton.xscan import workflows as xscan_workflows
from cuphoton.xscan.butler import (
    _parse_hsc_path_metadata,
    resolve_butler_registry_context,
)
from cuphoton.xscan.config import load_training_config
from cuphoton.xscan.dataset import (
    load_metadata_rows,
    validate_dataset_dir,
)
from cuphoton.xscan.des import (
    collect_nodiff_release_group_inventory,
    detect_search_sources,
)
from cuphoton.xscan.hsc import (
    HSC_MASK_PLANE_INDEX,
    create_hsc_valid_masks,
    resolve_hsc_npy_dir,
)
from cuphoton.xscan.lsstcomcam import (
    LsstComCamPathRewrite,
    _finalize_stamp,
    _rewrite_path,
    read_fits_stamp,
)
from cuphoton.xscan.metrics import (
    evaluate_predictions,
    select_threshold_for_scores,
)
from cuphoton.xscan.review import (
    _append_annotation,
    _load_entity_source_dataset,
)


def _run_cli(argv: list[str]) -> int:
    return run_component("xscan", argv)


def test_default_output_root_uses_product_state_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    expected = (tmp_path / "cuphoton" / "xscan" / "runs").resolve()

    assert xscan_workflows.default_output_root() == expected
    assert not expected.exists()


def test_reproduction_summary_keeps_undefined_auc_nullable() -> None:
    runs = [
        {"roc_auc": None, "accuracy": 0.50},
        {"roc_auc": None, "accuracy": 0.75},
    ]

    summary = xscan_workflows.summarize_reproduction_runs(runs)

    assert summary["mean_roc_auc"] is None
    assert summary["std_roc_auc"] is None
    assert summary["mean_accuracy"] == pytest.approx(0.625)
    assert summary["std_accuracy"] == pytest.approx(0.125)
    markdown = xscan_workflows.build_reproduction_markdown(
        {"jobs": {"autoscan_pair": summary}}
    )
    assert "| autoscan_pair | - | - | 0.625000 | 0.125000 | 2 |" in markdown


def test_hsc_ranking_separates_jobs_with_undefined_auc() -> None:
    jobs = {
        "undefined_high_accuracy": xscan_workflows.summarize_hsc_job_runs(
            input_mode="pair",
            training_mode="scratch",
            job_runs=[
                {
                    "roc_auc": None,
                    "pr_auc": None,
                    "accuracy": 0.99,
                    "run_dir": "undefined-high",
                }
            ],
        ),
        "defined": xscan_workflows.summarize_hsc_job_runs(
            input_mode="triplet",
            training_mode="scratch",
            job_runs=[
                {
                    "roc_auc": 0.75,
                    "pr_auc": 0.70,
                    "accuracy": 0.60,
                    "run_dir": "defined",
                }
            ],
        ),
        "partial_coverage": xscan_workflows.summarize_hsc_job_runs(
            input_mode="triplet",
            training_mode="scratch",
            job_runs=[
                {
                    "roc_auc": 0.75,
                    "pr_auc": 0.70,
                    "accuracy": 0.50,
                    "run_dir": "partial-defined",
                },
                {
                    "roc_auc": None,
                    "pr_auc": None,
                    "accuracy": 1.0,
                    "run_dir": "partial-undefined",
                },
            ],
        ),
        "undefined_low_accuracy": xscan_workflows.summarize_hsc_job_runs(
            input_mode="triplet",
            training_mode="scratch",
            job_runs=[
                {
                    "roc_auc": None,
                    "pr_auc": None,
                    "accuracy": 0.10,
                    "run_dir": "undefined-low",
                }
            ],
        ),
        "nonfinite_auc": xscan_workflows.summarize_hsc_job_runs(
            input_mode="pair",
            training_mode="scratch",
            job_runs=[
                {
                    "roc_auc": float("nan"),
                    "pr_auc": float("inf"),
                    "accuracy": 1.0,
                    "brier_score": float("nan"),
                    "calibration": {"expected_error": float("inf")},
                    "run_dir": "nonfinite",
                }
            ],
        ),
        "disjoint_auc_runs": xscan_workflows.summarize_hsc_job_runs(
            input_mode="triplet",
            training_mode="scratch",
            job_runs=[
                {
                    "roc_auc": 0.90,
                    "pr_auc": None,
                    "accuracy": 0.80,
                    "run_dir": "roc-only",
                },
                {
                    "roc_auc": None,
                    "pr_auc": 0.95,
                    "accuracy": 0.85,
                    "run_dir": "pr-only",
                },
            ],
        ),
    }

    ranking, unranked = xscan_workflows.rank_hsc_jobs(jobs)

    assert ranking == ["defined", "partial_coverage"]
    assert unranked == [
        "undefined_high_accuracy",
        "undefined_low_accuracy",
        "nonfinite_auc",
        "disjoint_auc_runs",
    ]
    assert jobs["nonfinite_auc"]["runs"][0]["roc_auc"] is None
    assert jobs["nonfinite_auc"]["runs"][0]["pr_auc"] is None
    assert jobs["nonfinite_auc"]["runs"][0]["brier_score"] is None
    assert jobs["nonfinite_auc"]["runs"][0]["calibration"] == {
        "expected_error": None
    }
    assert jobs["nonfinite_auc"]["best_run_dir"] is None
    assert jobs["disjoint_auc_runs"]["defined_run_count"] == 0
    assert jobs["disjoint_auc_runs"]["mean_roc_auc"] is None
    assert jobs["disjoint_auc_runs"]["mean_pr_auc"] is None
    assert jobs["disjoint_auc_runs"]["best_run_dir"] is None
    assert jobs["partial_coverage"]["mean_accuracy"] == pytest.approx(0.75)
    assert jobs["partial_coverage"]["mean_defined_auc_accuracy"] == (
        pytest.approx(0.50)
    )
    json.dumps(jobs, allow_nan=False)
    markdown = xscan_workflows.build_pair_triplet_markdown(
        {
            "dataset_dir": "dataset",
            "pair_config": "pair.yaml",
            "triplet_config": "triplet.yaml",
            "seeds": [0],
            "comparison_controls": {"ok": True},
            "label_provenance": {"placeholder_label_count": 0},
            "jobs": jobs,
            "ranking": ranking,
            "unranked": unranked,
        }
    )
    assert (
        "- Unranked (no run with defined ROC and PR AUC): "
        "undefined_high_accuracy, "
        "undefined_low_accuracy" in markdown
    )
    assert markdown.index("| defined |") < markdown.index(
        "| partial_coverage |"
    )
    assert markdown.index("| partial_coverage |") < markdown.index(
        "| undefined_high_accuracy |"
    )
    assert "| partial_coverage | triplet | scratch |" in markdown
    assert "| 1/2 | partial-defined |" in markdown
    assert "| undefined_high_accuracy | pair | scratch | - | - |" in markdown
    assert "None" not in markdown


def test_compare_inputs_normalizes_nonfinite_auc_values(tmp_path) -> None:
    def write_summary(
        label: str,
        *,
        roc_auc,
        pr_auc,
        accuracy,
    ) -> Path:
        run_dir = tmp_path / label
        evaluation_dir = run_dir / "evaluation" / "test"
        evaluation_dir.mkdir(parents=True)
        (evaluation_dir / "summary.json").write_text(
            json.dumps(
                {
                    "input_mode": label,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "accuracy": accuracy,
                    "sample_count": 4,
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    nonfinite = write_summary(
        "nonfinite",
        roc_auc=float("nan"),
        pr_auc=float("inf"),
        accuracy=0.99,
    )
    partial = write_summary(
        "partial",
        roc_auc=0.95,
        pr_auc=None,
        accuracy=0.90,
    )
    defined = write_summary(
        "defined",
        roc_auc=0.80,
        pr_auc=0.75,
        accuracy=0.70,
    )

    result = xscan_workflows.compare_inputs_workflow(
        [nonfinite, partial, defined]
    )

    assert result["best_run_dir"] == str(defined.resolve())
    assert [row["input_mode"] for row in result["runs"]] == [
        "defined",
        "partial",
        "nonfinite",
    ]
    assert result["runs"][1]["pr_auc"] is None
    assert result["runs"][2]["roc_auc"] is None
    assert result["runs"][2]["pr_auc"] is None
    json.dumps(result, allow_nan=False)


def write_review_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset_dir = root / "review-dataset"
    run_dir = root / "review-run"
    compare_dir = root / "compare-run"
    dataset_dir.mkdir()
    rng = np.random.default_rng(11)
    search = rng.normal(size=(6, 9, 9)).astype(np.float32)
    template = rng.normal(size=(6, 9, 9)).astype(np.float32)
    difference = (search - template).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    split = np.full((6,), 2, dtype=np.int64)
    metadata_rows = []
    for index, label in enumerate(labels.tolist()):
        row = {
            "candidate_id": f"review-{index}",
            "exposure_id": index,
            "ccd_id": None,
            "band": "i",
            "x": 10 + index,
            "y": 20 + index,
            "split_group": f"group-{index}",
            "split": "test",
            "label": label,
            "label_source": "fixture",
            "center_source": "catalog" if index % 2 else "catalog-offset",
            "catalog_pool_role": "positive" if label else "negative",
            "catalog_extendedness": 0.1 if label else 0.9,
            "center_offset_radius": 4.0 + index,
            "search_valid_fraction": 1.0,
            "difference_context_valid_fraction": 1.0,
        }
        metadata_rows.append(row)

    np.save(dataset_dir / "search.npy", search, allow_pickle=False)
    np.save(dataset_dir / "template.npy", template, allow_pickle=False)
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)
    np.save(dataset_dir / "labels.npy", labels, allow_pickle=False)
    np.save(dataset_dir / "split.npy", split, allow_pickle=False)
    (dataset_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
        encoding="utf-8",
    )

    def write_predictions(path: Path, probabilities: list[float]) -> None:
        output_dir = path / "evaluation" / "test"
        output_dir.mkdir(parents=True)
        rows = []
        for row, probability in zip(
            metadata_rows, probabilities, strict=True
        ):
            payload = dict(row)
            payload["sample_index"] = len(rows)
            payload["probability"] = probability
            payload["logit"] = probability - 0.5
            rows.append(json.dumps(payload))
        (output_dir / "predictions.jsonl").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "workflow": "evaluate",
                    "dataset_dir": str(dataset_dir.resolve()),
                    "split": "test",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    write_predictions(run_dir, [0.49, 0.52, 0.95, 0.10, 0.20, 0.85])
    write_predictions(compare_dir, [0.90, 0.10, 0.20, 0.80, 0.20, 0.85])
    return dataset_dir, run_dir, compare_dir


def write_lsstcomcam_placeholder_dataset(root: Path) -> Path:
    dataset_dir = root / "lsstcomcam-smoke"
    dataset_dir.mkdir()
    rng = np.random.default_rng(0)
    search = rng.normal(size=(4, 17, 17)).astype(np.float32)
    template = rng.normal(size=(4, 17, 17)).astype(np.float32)
    difference = (search - template).astype(np.float32)
    labels = np.zeros((4,), dtype=np.int64)
    split = np.array([0, 0, 1, 1], dtype=np.int64)
    metadata_rows = [
        {
            "candidate_id": f"lsst-placeholder-{index}",
            "exposure_id": 1001,
            "ccd_id": 12,
            "band": "r",
            "x": 20 + index,
            "y": 30 + index,
            "split_group": f"visit-1001-detector-12-{index}",
            "split": "train" if index < 2 else "val",
            "label": 0,
            "label_source": "unlabeled_lsstcomcam_smoke_placeholder",
            "target_label_available": False,
            "dataset_source": "LSSTComCam local FITS sample",
            "candidate_label_status": "placeholder",
        }
        for index in range(4)
    ]
    np.save(dataset_dir / "search.npy", search, allow_pickle=False)
    np.save(dataset_dir / "template.npy", template, allow_pickle=False)
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)
    np.save(dataset_dir / "labels.npy", labels, allow_pickle=False)
    np.save(dataset_dir / "split.npy", split, allow_pickle=False)
    (dataset_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
        encoding="utf-8",
    )
    return dataset_dir


def write_fake_hsc_store(root: Path) -> Path:
    store = root / "HSC_npy"
    store.mkdir(parents=True)
    rng = np.random.default_rng(7)
    images = rng.normal(loc=0.0, scale=0.05, size=(1, 64, 64, 4)).astype(
        np.float32
    )
    yy, xx = np.mgrid[:64, :64]
    for exposure in range(4):
        images[0, :, :, exposure] += (
            np.exp(-((yy - 20 - exposure) ** 2 + (xx - 32) ** 2) / 80.0)
            * 0.25
        ).astype(np.float32)
    variances = np.full((1, 64, 64, 4), 0.02, dtype=np.float32)
    masks = np.zeros((1, 64, 64, 12, 4), dtype=np.uint8)
    sky = np.ones((1, 64, 64, 4), dtype=np.uint8)
    yy_psf, xx_psf = np.mgrid[:25, :25]
    psf = np.exp(-((yy_psf - 12) ** 2 + (xx_psf - 12) ** 2) / 16.0).astype(
        np.float32
    )
    psf = psf / psf.sum()
    psfs = np.repeat(psf[:, :, None, None], 4, axis=3)
    exp_times = np.ones((4,), dtype=np.float32)

    np.save(store / "images.npy", images, allow_pickle=False)
    np.save(store / "variances.npy", variances, allow_pickle=False)
    np.save(store / "masks.npy", masks, allow_pickle=False)
    np.save(store / "sky.npy", sky, allow_pickle=False)
    np.save(store / "psfs.npy", psfs, allow_pickle=False)
    np.save(store / "exp_times.npy", exp_times, allow_pickle=False)
    (store / "metadata.json").write_text(
        json.dumps({"artifacts": {"images": {"shape": [1, 64, 64, 4]}}}),
        encoding="utf-8",
    )
    return store


def test_resolve_hsc_npy_dir_accepts_direct_store(tmp_path) -> None:
    store = write_fake_hsc_store(tmp_path)

    assert resolve_hsc_npy_dir(store) == store


def test_resolve_hsc_npy_dir_accepts_base_directory(tmp_path) -> None:
    store = write_fake_hsc_store(tmp_path)

    assert resolve_hsc_npy_dir(tmp_path) == store


def write_fake_hsc_catalog_parquet(root: Path) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    catalog_dir = root / "HSC" / "FITS" / "catalogs" / "0001" / "01"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "objectTable_fake_hsc.parq"
    table = pa.table(
        {
            "objectId": [101, 102, 103, 104],
            "x": [20.2, 42.8, 6.0, 32.1],
            "y": [22.4, 39.7, 7.0, 44.3],
            "tract": [1, 1, 1, 1],
            "patch": ["1,1", "1,1", "1,1", "1,1"],
            "refBand": ["i", "i", "i", "i"],
            "detect_isPrimary": [True, True, False, True],
            "sky_object": [False, False, False, True],
            "merge_peak_sky": [False, False, False, False],
            "i_psfFlux": [500.0, 350.0, 800.0, 200.0],
            "i_extendedness": [0.1, 0.8, 0.2, 0.3],
        }
    )
    pq.write_table(table, path)
    return path


def write_fake_hsc_fits_products(root: Path) -> Path:
    fits_root = root / "HSC" / "FITS"
    warp_dir = fits_root / "warps" / "0001" / "01" / "HSC-I"
    coadd_dir = fits_root / "coadd" / "0001" / "01" / "HSC-I"
    warp_dir.mkdir(parents=True, exist_ok=True)
    coadd_dir.mkdir(parents=True, exist_ok=True)
    for visit in (1001, 1003, 1005, 1007):
        (
            warp_dir
            / (
                "deepCoadd_directWarp_HSC_0001_01_i_HSC-I_"
                f"{visit}_test-run.fits"
            )
        ).write_bytes(b"")
    (coadd_dir / "deepCoadd_0001_01_i_test-run.fits").write_bytes(b"")
    return fits_root


def write_fake_lsstcomcam_fits(
    path: Path,
    values: np.ndarray,
    header=None,
) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    path.parent.mkdir(parents=True, exist_ok=True)
    image_hdu = fits.ImageHDU(
        data=np.asarray(values, dtype=np.float32), name="IMAGE"
    )
    if header is not None:
        image_hdu.header.update(header)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            image_hdu,
        ]
    ).writeto(path)


def write_fake_lsstcomcam_registry(root: Path) -> Path:
    registry_path = root / "lsstcomcam-registry.jsonl"
    visit_path = root / "fits" / "visit-r0.fits"
    difference_path = root / "fits" / "difference-r0.fits"
    other_visit_path = root / "fits" / "visit-g1.fits"
    other_difference_path = root / "fits" / "difference-g1.fits"
    yy, xx = np.mgrid[:7, :7]
    search = (100 + yy * 10 + xx).astype(np.float32)
    difference = np.full((7, 7), 3.0, dtype=np.float32)
    write_fake_lsstcomcam_fits(visit_path, search)
    write_fake_lsstcomcam_fits(difference_path, difference)
    write_fake_lsstcomcam_fits(other_visit_path, search + 1000)
    write_fake_lsstcomcam_fits(other_difference_path, difference + 10)

    rows = [
        {
            "path": str(visit_path),
            "registry_relative_path": "visit_image/r0.fits",
            "product": "visit_image",
            "butler_datasettype": "visit_image",
            "butler_run": "local/test-run",
            "data_id": json.dumps({"visit": 1001, "detector": 0}),
            "instrument": "LSSTComCam",
            "date": "20000102",
            "band": "r",
            "physical_filter": "r_03",
            "visit": 1001,
            "detector": 0,
            "tract": 1,
            "patch": 10,
            "detector_name": "R22_S00",
            "hdu_count": 2,
            "size_bytes": visit_path.stat().st_size,
        },
        {
            "path": str(difference_path),
            "registry_relative_path": "difference_image/r0.fits",
            "product": "difference_image",
            "butler_datasettype": "difference_image",
            "butler_run": "local/test-run",
            "data_id": json.dumps({"visit": 1001, "detector": 0}),
            "instrument": "LSSTComCam",
            "date": "20000102",
            "band": "r",
            "physical_filter": "r_03",
            "visit": 1001,
            "detector": 0,
            "tract": 1,
            "patch": 10,
            "detector_name": "R22_S00",
            "hdu_count": 2,
            "size_bytes": difference_path.stat().st_size,
        },
        {
            "path": str(other_visit_path),
            "registry_relative_path": "visit_image/g1.fits",
            "product": "visit_image",
            "butler_datasettype": "visit_image",
            "butler_run": "local/test-run",
            "data_id": json.dumps({"visit": 1002, "detector": 1}),
            "instrument": "LSSTComCam",
            "date": "20000101",
            "band": "g",
            "physical_filter": "g_01",
            "visit": 1002,
            "detector": 1,
            "tract": 2,
            "patch": 20,
            "detector_name": "R22_S01",
        },
        {
            "path": str(other_difference_path),
            "registry_relative_path": "difference_image/g1.fits",
            "product": "difference_image",
            "butler_datasettype": "difference_image",
            "butler_run": "local/test-run",
            "data_id": json.dumps({"visit": 1002, "detector": 1}),
            "instrument": "LSSTComCam",
            "date": "20000101",
            "band": "g",
            "physical_filter": "g_01",
            "visit": 1002,
            "detector": 1,
            "tract": 2,
            "patch": 20,
            "detector_name": "R22_S01",
        },
        {
            "path": str(root / "fits" / "template-r.fits"),
            "registry_relative_path": "template_coadd/1/10/r.fits",
            "product": "template_coadd",
            "butler_datasettype": "template_coadd",
            "butler_run": "local/test-run",
            "data_id": json.dumps({"band": "r", "tract": 1, "patch": 10}),
            "band": "r",
            "tract": 1,
            "patch": 10,
        },
        {
            "path": str(root / "fits" / "deep-r.fits"),
            "registry_relative_path": "deep_coadd/1/10/r.fits",
            "product": "deep_coadd",
            "butler_datasettype": "deep_coadd",
            "butler_run": "local/test-run",
            "data_id": json.dumps({"band": "r", "tract": 1, "patch": 10}),
            "band": "r",
            "tract": 1,
            "patch": 10,
        },
    ]
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return registry_path


def test_lsstcomcam_path_rewrite_root_prefix_preserves_separator() -> None:
    rewritten, rule = _rewrite_path(
        "/remote/source/file.fits",
        (LsstComCamPathRewrite("/", "/stage"),),
    )

    assert rule is not None
    assert rewritten == "/stage/remote/source/file.fits"


def write_hsc_manifest(
    root: Path,
    *,
    negative_source_mode: str = "random",
    negative_offset_range: tuple[float, float] = (4.0, 6.0),
    difference_mode: str = "simple",
    extra_lines: list[str] | None = None,
) -> Path:
    data_root = root / "hsc-manifest"
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifest.yaml"
    lines = [
        f"base: {root / 'data'}",
        "positive_count: 6",
        "negative_count: 6",
        "stamp_size: 17",
        "template_source: median",
        "include_difference: true",
        "split_fractions: [0.7, 0.15, 0.15]",
        "sample_seed: 5",
        "split_seed: 11",
        "tile_size: 8",
        "band: i",
        "positive_source_mode: catalog",
        f"negative_source_mode: {negative_source_mode}",
        (
            "negative_offset_range: "
            f"[{negative_offset_range[0]}, {negative_offset_range[1]}]"
        ),
        f"difference_mode: {difference_mode}",
        "catalog_primary_only: true",
        "catalog_exclude_sky: true",
        "center_jitter: 0",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    manifest_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_review_split_fixture(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "review-split-dataset"
    run_dir = root / "review-split-run"
    dataset_dir.mkdir()
    rng = np.random.default_rng(31)
    sample_count = 9
    search = rng.normal(size=(sample_count, 9, 9)).astype(np.float32)
    template = rng.normal(size=(sample_count, 9, 9)).astype(np.float32)
    difference = (search - template).astype(np.float32)
    labels = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int64)
    split = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    split_names = ["train", "val", "test"]
    metadata_rows = []
    for index, label in enumerate(labels.tolist()):
        split_name = split_names[int(split[index])]
        metadata_rows.append(
            {
                "candidate_id": f"split-{index}",
                "exposure_id": index,
                "ccd_id": None,
                "band": "i",
                "x": 10 + index,
                "y": 20 + index,
                "split_group": f"group-{index}",
                "split": split_name,
                "label": label,
                "center_source": "catalog" if label else "catalog-offset",
                "catalog_pool_role": "positive" if label else "negative",
                "catalog_extendedness": 0.1 if label else 0.9,
                "center_offset_radius": 0.0 if label else 6.0,
                "positive_quality_stratum": (
                    "positive:weak_snr" if label else None
                ),
                "search_valid_fraction": 1.0,
                "difference_context_valid_fraction": 1.0,
            }
        )

    np.save(dataset_dir / "search.npy", search, allow_pickle=False)
    np.save(dataset_dir / "template.npy", template, allow_pickle=False)
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)
    np.save(dataset_dir / "labels.npy", labels, allow_pickle=False)
    np.save(dataset_dir / "split.npy", split, allow_pickle=False)
    (dataset_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
        encoding="utf-8",
    )

    probabilities = [0.49, 0.51, 0.95, 0.48, 0.52, 0.94, 0.47, 0.53, 0.93]
    for split_name in split_names:
        output_dir = run_dir / "evaluation" / split_name
        output_dir.mkdir(parents=True)
        rows = []
        for sample_index, row in enumerate(metadata_rows):
            if row["split"] != split_name:
                continue
            payload = dict(row)
            payload["sample_index"] = sample_index
            payload["probability"] = probabilities[sample_index]
            payload["logit"] = probabilities[sample_index] - 0.5
            rows.append(json.dumps(payload))
        (output_dir / "predictions.jsonl").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "workflow": "evaluate",
                    "dataset_dir": str(dataset_dir.resolve()),
                    "split": split_name,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return dataset_dir, run_dir


def write_hsc_training_config(
    path: Path,
    *,
    output_root: Path,
    run_name: str,
    input_mode: str,
) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset_dir: placeholder",
                f"output_root: {output_root}",
                f"run_name: {run_name}",
                "epochs: 1",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: auto",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "model:",
                f"  input_mode: {input_mode}",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_hsc_sweep_config(
    path: Path,
    *,
    variants: list[dict[str, object]],
) -> Path:
    path.write_text(
        yaml.safe_dump({"variants": variants}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_prepared_manifest_inputs(
    root: Path,
    *,
    include_difference: bool,
    dataset_kind: str,
) -> Path:
    sample_count = 6
    search = np.arange(sample_count * 17 * 17, dtype=np.float32).reshape(
        sample_count, 17, 17
    )
    template = search + 1.0
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    split = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    difference = search - template

    data_root = root / dataset_kind
    data_root.mkdir(parents=True)
    np.save(data_root / "search.npy", search, allow_pickle=False)
    np.save(data_root / "template.npy", template, allow_pickle=False)
    np.save(data_root / "labels.npy", labels, allow_pickle=False)
    np.save(data_root / "split.npy", split, allow_pickle=False)
    if include_difference:
        np.save(data_root / "difference.npy", difference, allow_pickle=False)

    rows = []
    splits = ["train", "train", "val", "val", "test", "test"]
    for idx in range(sample_count):
        row = {
            "candidate_id": f"cand-{idx:03d}",
            "exposure_id": idx // 2,
            "ccd_id": idx % 2,
            "band": "i",
            "x": 100 + idx,
            "y": 200 + idx,
            "split_group": f"group-{idx // 2}",
            "split": splits[idx],
            "label": int(labels[idx]),
            "label_source": "prepared",
        }
        if dataset_kind == "nodiff":
            row.update(
                {
                    "fake_id": idx if labels[idx] == 1 else None,
                    "autoscan_score": 0.8 if labels[idx] == 1 else 0.1,
                    "diff_snr": 6.0 + idx,
                    "snr": 5.0 + idx,
                    "flux_ratio": 0.1 * (idx + 1),
                }
            )
        rows.append(row)

    metadata_path = data_root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    manifest_path = data_root / "manifest.yaml"
    lines = [
        f"search_path: {data_root / 'search.npy'}",
        f"template_path: {data_root / 'template.npy'}",
        f"labels_path: {data_root / 'labels.npy'}",
        f"split_path: {data_root / 'split.npy'}",
        f"metadata_jsonl: {metadata_path}",
    ]
    if include_difference:
        lines.insert(2, f"difference_path: {data_root / 'difference.npy'}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def write_release_stamp_group(
    source_dir: Path,
    *,
    global_id: int,
    label_name: str,
    stamp_size: int = 17,
    kinds: tuple[str, ...] = ("srch", "tmpl", "diff"),
) -> None:
    group_dir = source_dir / f"gid_{global_id}"
    group_dir.mkdir(parents=True)
    base = np.full(
        (stamp_size, stamp_size),
        float(global_id),
        dtype=np.float32,
    )
    for offset, kind in enumerate(kinds, start=1):
        np.save(
            group_dir
            / f"{offset:05d}_{kind}_{global_id}_{kind}_{label_name}.npy",
            base + float(offset),
            allow_pickle=False,
        )


def write_nodiff_release_manifest(
    root: Path,
    *,
    max_per_class: int | None = 2,
) -> Path:
    source_dir = root / "nodiff-release"
    stamp_size = 17
    for global_id, label_name in (
        (101, "pos"),
        (102, "pos"),
        (103, "pos"),
        (201, "neg"),
        (202, "neg"),
        (203, "neg"),
    ):
        write_release_stamp_group(
            source_dir,
            global_id=global_id,
            label_name=label_name,
            stamp_size=stamp_size,
        )

    manifest_path = root / "nodiff-release.yaml"
    payload = {
        "source_dir": str(source_dir),
        "stamp_size": stamp_size,
        "split_fractions": [0.34, 0.33, 0.33],
        "split_seed": 20260514,
        "release_shard_url": (
            "https://portal.nersc.gov/cfs/dessn/nodiff/all/"
            "nodiff_triplets_07_of_10.tar.gz"
        ),
        "release_byte_range": "bytes=0-20971519",
    }
    if max_per_class is not None:
        payload["max_per_class"] = max_per_class
    manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def write_raw_autoscan_manifest(root: Path) -> Path:
    data_root = root / "raw-autoscan"
    data_root.mkdir(parents=True)
    rows = []
    for idx, label in enumerate([1, 0, 1, 0]):
        search = np.full((17, 17), idx + 1, dtype=np.float32)
        template = np.full((17, 17), idx, dtype=np.float32)
        difference = search - template
        search_path = data_root / f"search-{idx}.npy"
        template_path = data_root / f"template-{idx}.npy"
        difference_path = data_root / f"difference-{idx}.npy"
        np.save(search_path, search, allow_pickle=False)
        np.save(template_path, template, allow_pickle=False)
        np.save(difference_path, difference, allow_pickle=False)
        rows.append(
            {
                "candidate_id": f"raw-{idx}",
                "search_path": str(search_path),
                "template_path": str(template_path),
                "difference_path": str(difference_path),
                "label": label,
                "exposure_id": idx // 2,
                "ccd_id": idx % 2,
                "band": "i",
                "x": 100 + idx,
                "y": 200 + idx,
                "split_group": f"group-{idx // 2}",
                "label_source": "raw-record",
            }
        )
    records_path = data_root / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest_path = data_root / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"records_path: {records_path}",
                "stamp_size: 17",
                "split_fractions: [0.9, 0.0, 0.1]",
                "split_seed: 0",
                "balance_classes: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_raw_autoscan_manifest_with_label_aliases(root: Path) -> Path:
    data_root = root / "raw-autoscan-labels"
    data_root.mkdir(parents=True)
    rows = []
    for idx, label in enumerate(["real", "bogus", "positive", "artifact"]):
        search = np.full((17, 17), idx + 1, dtype=np.float32)
        template = np.full((17, 17), idx, dtype=np.float32)
        difference = search - template
        search_path = data_root / f"search-{idx}.npy"
        template_path = data_root / f"template-{idx}.npy"
        difference_path = data_root / f"difference-{idx}.npy"
        np.save(search_path, search, allow_pickle=False)
        np.save(template_path, template, allow_pickle=False)
        np.save(difference_path, difference, allow_pickle=False)
        rows.append(
            {
                "candidate_id": f"label-{idx}",
                "search_path": str(search_path),
                "template_path": str(template_path),
                "difference_path": str(difference_path),
                "label": label,
                "exposure_id": idx // 2,
                "ccd_id": idx % 2,
                "band": "i",
                "x": 100 + idx,
                "y": 200 + idx,
                "split_group": f"group-{idx // 2}",
                "label_source": "raw-record",
            }
        )
    records_path = data_root / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest_path = data_root / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"records_path: {records_path}",
                "stamp_size: 17",
                "split_fractions: [0.9, 0.0, 0.1]",
                "split_seed: 0",
                "balance_classes: true",
                "positive_labels: [real, positive]",
                "negative_labels: [bogus, artifact]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_raw_autoscan_image_manifest(root: Path) -> Path:
    data_root = root / "raw-autoscan-images"
    data_root.mkdir(parents=True)
    rows = []
    coords = [(20, 20), (20, 40), (40, 20), (40, 40)]
    labels = [1, 0, 1, 0]
    for idx, ((y0, x0), label) in enumerate(zip(coords, labels)):
        search = np.zeros((64, 64), dtype=np.float32)
        template = np.zeros((64, 64), dtype=np.float32)
        difference = np.zeros((64, 64), dtype=np.float32)
        patch = np.full((17, 17), idx + 1, dtype=np.float32)
        search[y0 - 8 : y0 + 9, x0 - 8 : x0 + 9] = patch
        template[y0 - 8 : y0 + 9, x0 - 8 : x0 + 9] = patch - 1
        difference[y0 - 8 : y0 + 9, x0 - 8 : x0 + 9] = 1.0
        search_path = data_root / f"search-image-{idx}.npy"
        template_path = data_root / f"template-image-{idx}.npy"
        difference_path = data_root / f"diff-image-{idx}.npy"
        np.save(search_path, search, allow_pickle=False)
        np.save(template_path, template, allow_pickle=False)
        np.save(difference_path, difference, allow_pickle=False)
        rows.append(
            {
                "cand_id": f"full-{idx}",
                "search_image_path": str(search_path),
                "template_image_path": str(template_path),
                "difference_image_path": str(difference_path),
                "class": label,
                "expnum": idx // 2,
                "ccdnum": idx % 2,
                "filter": "i",
                "xpos": x0,
                "ypos": y0,
                "group_id": f"group-{idx // 2}",
                "label_source": "raw-image-record",
            }
        )
    records_path = data_root / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest_path = data_root / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"records_path: {records_path}",
                "stamp_size: 17",
                "split_fractions: [0.9, 0.0, 0.1]",
                "split_seed: 0",
                "balance_classes: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_raw_nodiff_manifest(
    root: Path, *, subset_fraction: float = 1.0
) -> Path:
    data_root = root / "raw-nodiff"
    data_root.mkdir(parents=True)
    search = np.zeros((64, 64), dtype=np.float32)
    template = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    positive = np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / 6.0) * 50.0
    negative = np.exp(-((yy - 40) ** 2 + (xx - 42) ** 2) / 6.0) * 40.0
    search += positive.astype(np.float32)
    search += negative.astype(np.float32)
    template += (negative * 0.25).astype(np.float32)

    search_path = data_root / "search.npy"
    template_path = data_root / "template.npy"
    np.save(search_path, search, allow_pickle=False)
    np.save(template_path, template, allow_pickle=False)

    fake_rows = [
        {
            "candidate_id": "fake-1",
            "fake_id": "fake-1",
            "x": 20.0,
            "y": 20.0,
            "autoscan_score": 0.9,
            "diff_snr": 6.0,
            "snr": 6.0,
            "injected_flux": 120.0,
        }
    ]
    fake_catalog_path = data_root / "fake-catalog.jsonl"
    with fake_catalog_path.open("w", encoding="utf-8") as handle:
        for row in fake_rows:
            handle.write(json.dumps(row) + "\n")

    exposures_rows = [
        {
            "search_path": str(search_path),
            "template_path": str(template_path),
            "fake_catalog_path": str(fake_catalog_path),
            "exposure_id": 11,
            "ccd_id": 7,
            "band": "i",
            "split_group": "11:7",
        }
    ]
    exposures_path = data_root / "exposures.jsonl"
    with exposures_path.open("w", encoding="utf-8") as handle:
        for row in exposures_rows:
            handle.write(json.dumps(row) + "\n")

    manifest_path = data_root / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"exposures_path: {exposures_path}",
                "stamp_size: 17",
                "detection_threshold_sigma: 5.0",
                "detection_minarea: 3",
                "max_per_class_per_image: 2",
                "diff_snr_min: 3.5",
                "split_fractions: [0.9, 0.0, 0.1]",
                "split_seed: 0",
                "balance_classes: true",
                f"subset_fraction: {subset_fraction}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_raw_nodiff_manifest_with_limits(root: Path) -> Path:
    manifest_path = write_raw_nodiff_manifest(root, subset_fraction=1.0)
    lines = Path(manifest_path).read_text(encoding="utf-8").splitlines()
    lines.append("limit_exposures: 1")
    Path(manifest_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def write_raw_nodiff_manifest_with_aliases(
    root: Path,
    *,
    subset_fraction: float = 1.0,
) -> Path:
    data_root = root / "raw-nodiff-alias"
    data_root.mkdir(parents=True)
    search = np.zeros((64, 64), dtype=np.float32)
    template = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    positive = np.exp(-((yy - 22) ** 2 + (xx - 22) ** 2) / 6.0) * 50.0
    negative = np.exp(-((yy - 42) ** 2 + (xx - 44) ** 2) / 6.0) * 40.0
    search += positive.astype(np.float32)
    search += negative.astype(np.float32)
    template += (negative * 0.25).astype(np.float32)

    search_path = data_root / "search.npy"
    template_path = data_root / "template.npy"
    np.save(search_path, search, allow_pickle=False)
    np.save(template_path, template, allow_pickle=False)

    fake_rows = [
        {
            "candidate_id": "fake-a",
            "fakeid": "fake-a",
            "xpos": 22.0,
            "ypos": 22.0,
            "autoscan": 0.9,
            "diffsnr": 6.0,
            "flux": 120.0,
        }
    ]
    fake_catalog_path = data_root / "fake-catalog.jsonl"
    with fake_catalog_path.open("w", encoding="utf-8") as handle:
        for row in fake_rows:
            handle.write(json.dumps(row) + "\n")

    exposures_rows = [
        {
            "search_image_path": str(search_path),
            "template_image_path": str(template_path),
            "fakes_path": str(fake_catalog_path),
            "expnum": 15,
            "ccdnum": 4,
            "filter": "i",
            "group": "15:4",
        }
    ]
    exposures_path = data_root / "exposures.jsonl"
    with exposures_path.open("w", encoding="utf-8") as handle:
        for row in exposures_rows:
            handle.write(json.dumps(row) + "\n")

    manifest_path = data_root / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"exposures_path: {exposures_path}",
                "stamp_size: 17",
                "detection_threshold_sigma: 5.0",
                "detection_minarea: 3",
                "max_per_class_per_image: 2",
                "diff_snr_min: 3.5",
                "split_fractions: [0.9, 0.0, 0.1]",
                "split_seed: 0",
                "balance_classes: true",
                f"subset_fraction: {subset_fraction}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_component_registry_uses_canonical_namespace() -> None:
    component = get_component("xscan")

    assert component.group == "xscan"
    assert component.import_name == "cuphoton.xscan"


def test_group_runner_uses_umbrella_name_in_help_output(capsys) -> None:
    rc = _run_cli(["help", "data-inspect"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-inspect" in captured.out
    assert "--dataset-dir" in captured.out
    assert "-c FILE, --conf=FILE" not in captured.out


def test_help_for_data_merge_command(capsys) -> None:
    rc = _run_cli(["help", "data-merge"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-merge" in captured.out
    assert "--dataset-dirs" in captured.out
    assert "--output-dir" in captured.out


def test_help_for_train_inada_pair_command(capsys) -> None:
    rc = _run_cli(["help", "train-inada-pair"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan train-inada-pair" in captured.out
    assert "--config" in captured.out


def test_help_for_experimental_build_hsc_command(capsys) -> None:
    rc = _run_cli(["help", "experimental-build-hsc-synthetic"])
    captured = capsys.readouterr()

    assert rc == 0
    assert (
        "Usage: cuphoton xscan experimental-build-hsc-synthetic"
        in captured.out
    )
    assert "--base" in captured.out
    assert "--tile-size" in captured.out


def test_help_for_hsc_build_command(capsys) -> None:
    rc = _run_cli(["help", "data-build-hsc"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-build-hsc" in captured.out
    assert "--manifest" in captured.out
    assert "--output-dir" in captured.out


def test_help_for_nodiff_release_build_command(capsys) -> None:
    rc = _run_cli(["help", "data-build-nodiff-release"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-build-nodiff-release" in captured.out
    assert "--manifest" in captured.out
    assert "--output-dir" in captured.out


def test_help_for_hsc_registry_build_command_case(capsys) -> None:
    rc = _run_cli(["help", "data-build-hsc-registry"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-build-hsc-registry" in captured.out
    assert "--fits-root" in captured.out
    assert "--output-path" in captured.out


def test_help_for_lsstcomcam_smoke_build_command(capsys) -> None:
    rc = _run_cli(["help", "data-build-lsstcomcam-smoke"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-build-lsstcomcam-smoke" in captured.out
    assert "--manifest" in captured.out
    assert "--output-dir" in captured.out


def test_help_for_lsstcomcam_candidate_check_command(capsys) -> None:
    rc = _run_cli(["help", "data-check-lsstcomcam-candidates"])
    captured = capsys.readouterr()

    assert rc == 0
    assert (
        "Usage: cuphoton xscan data-check-lsstcomcam-candidates"
        in captured.out
    )
    assert "--manifest" in captured.out
    assert "--require-ok" in captured.out


def test_help_for_lsstcomcam_staging_plan_command(capsys) -> None:
    rc = _run_cli(["help", "data-plan-lsstcomcam-staging"])
    captured = capsys.readouterr()

    assert rc == 0
    assert (
        "Usage: cuphoton xscan data-plan-lsstcomcam-staging" in captured.out
    )
    assert "--manifest" in captured.out
    assert "--sample-count" in captured.out


def test_help_for_lsstcomcam_fits_stage_command(capsys) -> None:
    rc = _run_cli(["help", "data-stage-lsstcomcam-fits"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-stage-lsstcomcam-fits" in captured.out
    assert "--source-prefix" in captured.out
    assert "--target-prefix" in captured.out
    assert "--search-roots" in captured.out


def test_help_for_training_label_check_command(capsys) -> None:
    rc = _run_cli(["help", "data-check-training-labels"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan data-check-training-labels" in captured.out
    assert "--dataset-dir" in captured.out
    assert "--require-ok" in captured.out


def test_help_for_reproduce_hsc_comparison_command(capsys) -> None:
    rc = _run_cli(["help", "reproduce-hsc-comparison"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan reproduce-hsc-comparison" in captured.out
    assert "--pair-config" in captured.out
    assert "--triplet-config" in captured.out
    assert "--pair-pretrain-checkpoint" in captured.out
    assert "--triplet-pretrain-checkpoint" in captured.out


def test_help_for_reproduce_pair_triplet_command(capsys) -> None:
    rc = _run_cli(["help", "reproduce-pair-triplet"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan reproduce-pair-triplet" in captured.out
    assert "--dataset-dir" in captured.out
    assert "--pair-config" in captured.out
    assert "--triplet-config" in captured.out
    assert "--seeds" in captured.out


def test_help_for_reproduce_hsc_xpois_sweep_command(capsys) -> None:
    rc = _run_cli(["help", "reproduce-hsc-xpois-sweep"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xscan reproduce-hsc-xpois-sweep" in captured.out
    assert "--pair-config" in captured.out
    assert "--triplet-config" in captured.out
    assert "--sweep-config" in captured.out


def test_help_for_review_commands(capsys) -> None:
    rc = _run_cli(["help", "review-queue"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-queue" in captured.out
    assert "--compare-run-dirs" in captured.out

    rc = _run_cli(["help", "review-queue-dataset"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-queue-dataset" in captured.out
    assert "--dataset-dir" in captured.out

    rc = _run_cli(["help", "review-queue-splits"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-queue-splits" in captured.out
    assert "--splits" in captured.out

    rc = _run_cli(["help", "review-bokeh"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-bokeh" in captured.out
    assert "--show-url-only" in captured.out

    rc = _run_cli(["help", "review-raw-compare"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-raw-compare" in captured.out
    assert "--review-dir" in captured.out
    assert "--host" in captured.out
    assert "--port" in captured.out

    rc = _run_cli(["help", "review-alard-lupton"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-alard-lupton" in captured.out
    assert "--review-dir" in captured.out
    assert "--host" in captured.out
    assert "--port" in captured.out

    rc = _run_cli(["help", "review-status"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-status" in captured.out
    assert "--require-ready" in captured.out
    assert "--include-decisions" in captured.out

    rc = _run_cli(["help", "review-contact-sheet"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-contact-sheet" in captured.out
    assert "--items-per-page" in captured.out
    assert "--overwrite" in captured.out

    rc = _run_cli(["help", "review-annotation-template"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-annotation-template" in captured.out
    assert "--output-csv" in captured.out

    rc = _run_cli(["help", "review-import-annotations"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-import-annotations" in captured.out
    assert "--input-csv" in captured.out
    assert "--dry-run" in captured.out

    rc = _run_cli(["help", "review-aggregate"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-aggregate" in captured.out
    assert "--output-report" in captured.out
    assert "--consensus-rule" in captured.out

    rc = _run_cli(["help", "review-apply"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan review-apply" in captured.out
    assert "--review-dir" in captured.out
    assert "--aggregation-report" in captured.out

    rc = _run_cli(["help", "entity-review-queue"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan entity-review-queue" in captured.out
    assert "--source-review-dirs" in captured.out

    rc = _run_cli(["help", "entity-review-bokeh"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan entity-review-bokeh" in captured.out
    assert "--show-url-only" in captured.out
    assert "[default: 5007]" in captured.out

    rc = _run_cli(["help", "entity-review-aggregate"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: cuphoton xscan entity-review-aggregate" in captured.out
    assert "--consensus-rule" in captured.out


@pytest.mark.parametrize(
    ("alias", "runner_name", "host", "port"),
    [
        ("rrc", "run_raw_compare_review_server", "127.0.0.1", 5111),
        ("ral", "run_alard_lupton_review_server", "0.0.0.0", 5112),
    ],
)
def test_review_server_aliases_dispatch_with_parsed_options(
    alias: str,
    runner_name: str,
    host: str,
    port: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        xscan_commands,
        runner_name,
        lambda **kwargs: calls.append(kwargs),
    )

    rc = _run_cli(
        [
            alias,
            "--review-dir",
            str(review_dir),
            "--host",
            host,
            "--port",
            str(port),
        ]
    )

    assert rc == 0
    assert calls == [
        {
            "review_dir": review_dir,
            "host": host,
            "port": port,
        }
    ]


def test_load_training_config_parses_performance_block(tmp_path) -> None:
    config_path = tmp_path / "perf.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_dir: /tmp/dataset",
                "benchmark_regime_name: hard_subtraction",
                "training_mode: fine_tune",
                "pretrain_checkpoint: /tmp/checkpoint.pt",
                "freeze_encoder_stages: [0, 2]",
                "early_stopping_metric: val_pr_auc",
                "early_stopping_patience: 3",
                "early_stopping_min_delta: 0.001",
                "model:",
                "  input_mode: pair",
                "performance:",
                "  amp_dtype: bf16",
                "  allow_tf32: true",
                "  cudnn_benchmark: true",
                "  compile: true",
                "  compile_mode: reduce-overhead",
                "  compile_threads: 1",
                "  compile_worker_start_method: spawn",
                "  worker_start_method: spawn",
                "  worker_cpu_threads: 1",
                "  num_workers: 2",
                "  pin_memory: true",
                "  persistent_workers: true",
                "  non_blocking_transfers: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_training_config(config_path)
    assert config.benchmark_regime_name == "hard_subtraction"
    assert config.training_mode == "fine_tune"
    assert config.pretrain_checkpoint == "/tmp/checkpoint.pt"
    assert config.freeze_encoder_stages == [0, 2]
    assert config.early_stopping_metric == "val_pr_auc"
    assert config.early_stopping_patience == 3
    assert config.early_stopping_min_delta == 0.001
    assert config.performance.amp_dtype == "bf16"
    assert config.performance.allow_tf32 is True
    assert config.performance.cudnn_benchmark is True
    assert config.performance.compile is True
    assert config.performance.compile_mode == "reduce-overhead"
    assert config.performance.compile_threads == 1
    assert config.performance.compile_worker_start_method == "spawn"
    assert config.performance.worker_start_method == "spawn"
    assert config.performance.worker_cpu_threads == 1
    assert config.performance.num_workers == 2
    assert config.performance.pin_memory is True
    assert config.performance.persistent_workers is True
    assert config.performance.non_blocking_transfers is True


@pytest.mark.parametrize(
    ("lines", "section"),
    [
        (["dataset_dir: /tmp/dataset", "learning_rte: 0.001"], "top-level"),
        (
            ["dataset_dir: /tmp/dataset", "model:", "  input_mod: pair"],
            "model",
        ),
        (
            ["dataset_dir: /tmp/dataset", "performance:", "  amp_dtyp: bf16"],
            "performance",
        ),
    ],
)
def test_load_training_config_rejects_unknown_keys(
    tmp_path, lines, section
) -> None:
    config_path = tmp_path / "typo.yaml"
    config_path.write_text("\n".join([*lines, ""]), encoding="utf-8")

    with pytest.raises(ValueError, match=f"unknown {section} option"):
        load_training_config(config_path)


@pytest.mark.parametrize(
    ("input_mode", "channel_count", "expected_parameter_count"),
    [
        ("pair", 2, 65_923_243),
        ("triplet", 3, 67_802_347),
    ],
)
def test_inada_default_models_match_nodiff_shape_contract(
    input_mode: str,
    channel_count: int,
    expected_parameter_count: int,
) -> None:
    from dataclasses import asdict, replace

    import torch

    from cuphoton.xscan.config import ModelConfig
    from cuphoton.xscan.model import build_model

    config = replace(ModelConfig(), input_mode=input_mode)
    model = build_model(**asdict(config)).eval()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert parameter_count == expected_parameter_count

    inputs = torch.randn(2, channel_count, 51, 51)
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_create_hsc_valid_masks_matches_image_modes() -> None:
    masks = np.zeros((1, 7, 7, 12, 1), dtype=np.uint8)
    masks[0, 1, 1, HSC_MASK_PLANE_INDEX["chip_gap"], 0] = 1
    masks[0, 2, 2, HSC_MASK_PLANE_INDEX["stripes_1"], 0] = 1
    masks[0, 3, 3, HSC_MASK_PLANE_INDEX["large_sources"], 0] = 1

    non_conservative = create_hsc_valid_masks(
        masks,
        mode="non_conservative",
    )
    conservative = create_hsc_valid_masks(
        masks,
        mode="conservative",
        dilation_factor=3,
    )

    assert bool(non_conservative[0, 1, 1, 0]) is False
    assert bool(non_conservative[0, 2, 2, 0]) is True
    assert bool(non_conservative[0, 3, 3, 0]) is True
    assert bool(conservative[0, 1, 1, 0]) is False
    assert bool(conservative[0, 2, 2, 0]) is False
    assert bool(conservative[0, 3, 3, 0]) is False
    assert bool(conservative[0, 3, 4, 0]) is False


def test_cli_build_autoscan_from_manifest(tmp_path, capsys) -> None:
    manifest_path = write_prepared_manifest_inputs(
        tmp_path,
        include_difference=True,
        dataset_kind="autoscan",
    )
    output_dir = tmp_path / "packaged-autoscan"
    rc = _run_cli(
        [
            "data-build-autoscan",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "autoscan"
    assert (output_dir / "difference.npy").exists()

    rc = _run_cli(
        [
            "data-validate",
            "--dataset-dir",
            str(output_dir),
            "--dataset-kind",
            "autoscan",
        ]
    )
    captured = capsys.readouterr()
    validation = json.loads(captured.out)
    assert rc == 0
    assert validation["valid"] is True


def test_cli_data_merge_combines_packaged_datasets(tmp_path, capsys) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_dataset = write_lsstcomcam_placeholder_dataset(first_root)
    second_dataset = write_lsstcomcam_placeholder_dataset(second_root)
    output_dir = tmp_path / "merged-lsstcomcam"

    rc = _run_cli(
        [
            "data-merge",
            "--dataset-dirs",
            f"{first_dataset},{second_dataset}",
            "--output-dir",
            str(output_dir),
            "--dataset-kind",
            "lsstcomcam-smoke",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["workflow"] == "data-merge"
    assert payload["sample_count"] == 8
    assert payload["input_mode"] == "triplet"
    assert len(payload["input_summaries"]) == 2
    assert np.load(output_dir / "search.npy", mmap_mode="r").shape == (
        8,
        17,
        17,
    )
    assert (output_dir / "difference.npy").exists()
    rows = load_metadata_rows(output_dir)
    assert len(rows) == 8
    assert rows[0]["source_dataset_order"] == 0
    assert rows[0]["source_sample_index"] == 0
    assert rows[4]["source_dataset_order"] == 1
    assert rows[4]["source_sample_index"] == 0
    assert rows[4]["sample_index"] == 4
    assert all(
        row["label_source"] == "unlabeled_lsstcomcam_smoke_placeholder"
        for row in rows
    )

    rc = _run_cli(
        [
            "data-validate",
            "--dataset-dir",
            str(output_dir),
            "--dataset-kind",
            "lsstcomcam-smoke",
        ]
    )
    captured = capsys.readouterr()
    validation = json.loads(captured.out)
    assert rc == 0
    assert validation["valid"] is True
    assert validation["sample_count"] == 8

    rc = _run_cli(
        [
            "data-merge",
            "--dataset-dirs",
            f"{first_dataset},{second_dataset}",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "not empty" in captured.err


def test_cli_data_merge_accepts_reviewed_label_datasets(
    tmp_path,
    capsys,
) -> None:
    first_root = tmp_path / "first-reviewed"
    second_root = tmp_path / "second-reviewed"
    first_root.mkdir()
    second_root.mkdir()
    first_dataset, _first_run, _first_compare = write_review_fixture(
        first_root
    )
    second_dataset, _second_run, _second_compare = write_review_fixture(
        second_root
    )
    output_dir = tmp_path / "merged-reviewed"

    rc = _run_cli(
        [
            "data-merge",
            "--dataset-dirs",
            f"{first_dataset},{second_dataset}",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] is None
    assert payload["sample_count"] == 12
    assert int(np.load(output_dir / "labels.npy").sum()) == 6
    rows = load_metadata_rows(output_dir)
    assert {row["label_source"] for row in rows} == {"fixture"}
    assert rows[6]["source_dataset_order"] == 1
    assert rows[6]["source_sample_index"] == 0


def test_cli_data_merge_rejects_incompatible_input_modes(
    tmp_path,
    capsys,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_dataset = write_lsstcomcam_placeholder_dataset(first_root)
    second_dataset = write_lsstcomcam_placeholder_dataset(second_root)
    (second_dataset / "difference.npy").unlink()

    rc = _run_cli(
        [
            "data-merge",
            "--dataset-dirs",
            f"{first_dataset},{second_dataset}",
            "--output-dir",
            str(tmp_path / "merged"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "all input datasets must either include difference.npy" in (
        captured.err
    )


def test_cli_build_hsc_from_manifest(tmp_path, capsys) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(tmp_path)
    output_dir = tmp_path / "hsc-packaged"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "hsc-synthetic"
    assert payload["positive_catalog_candidate_count"] == 2
    assert payload["negative_catalog_candidate_count"] == 0
    assert payload["center_source_counts"]["catalog"] == 6
    assert payload["center_source_counts"]["random"] == 6
    assert (output_dir / "difference.npy").exists()

    validation = validate_dataset_dir(
        output_dir, dataset_kind="hsc-synthetic"
    )
    assert validation["valid"] is True
    rows = load_metadata_rows(output_dir)
    assert any(row["center_source"] == "catalog" for row in rows)
    assert any(row["center_source"] == "random" for row in rows)
    assert {
        row["catalog_object_id"]
        for row in rows
        if row["center_source"] == "catalog"
    } <= {101, 102}


def test_cli_build_hsc_with_butler_exposure_filter(tmp_path, capsys) -> None:
    pd = pytest.importorskip("pandas")
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    registry_path = tmp_path / "fits_registry.parquet"
    pd.DataFrame(
        [
            {
                "path": "/fake/exposure-2.fits",
                "butler_run": "test-run",
                "butler_datasettype": "visit_image",
                "data_id": '{"exposure":2}',
                "instrument": "HSC",
                "band": "i",
                "visit": pd.NA,
                "exposure_index": 2,
            }
        ]
    ).to_parquet(registry_path)
    manifest_path = write_hsc_manifest(
        tmp_path,
        extra_lines=[
            f"butler_registry: {registry_path}",
            "butler_datasettype: visit_image",
            "butler_query: \"band == 'i'\"",
            "butler_exposure_index_column: exposure_index",
        ],
    )
    output_dir = tmp_path / "hsc-butler-packaged"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["butler_registry"]["row_count"] == 1
    assert payload["butler_registry"]["allowed_exposure_indices"] == [2]
    rows = load_metadata_rows(output_dir)
    assert {row["exposure_id"] for row in rows} == {2}
    assert {row["butler_path"] for row in rows} == {"/fake/exposure-2.fits"}
    assert {row["butler_visit"] for row in rows} == {None}


def test_cli_build_hsc_registry_and_registry_backed_manifest(
    tmp_path,
    capsys,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    hsc_npy = write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    fits_root = write_fake_hsc_fits_products(tmp_path / "data")
    registry_path = tmp_path / "hsc_registry.parquet"

    rc = _run_cli(
        [
            "data-build-hsc-registry",
            "--fits-root",
            str(fits_root),
            "--hsc-npy-dir",
            str(hsc_npy),
            "--output-path",
            str(registry_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["row_count"] == 6
    assert payload["datasettype_counts"] == {
        "deepCoadd": 1,
        "objectTable": 1,
        "visit_image": 4,
    }
    assert payload["fits_header_status_counts"]["failed"] == 5
    assert payload["fits_header_status_counts"]["not_applicable"] == 1
    assert payload["object_count_status_counts"] == {
        "not_applicable": 5,
        "read": 1,
    }
    collections_path = registry_path.with_name(
        registry_path.name + ".collections.json"
    )
    assert collections_path.exists()
    sidecar_payload = json.loads(collections_path.read_text(encoding="utf-8"))
    sidecar_payload["collections"].append(
        {
            "name": "custom_warps",
            "type": "TAGGED",
            "filter": "butler_datasettype == 'visit_image'",
        }
    )
    collections_path.write_text(
        json.dumps(sidecar_payload) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "data-build-hsc-registry",
            "--fits-root",
            str(fits_root),
            "--hsc-npy-dir",
            str(hsc_npy),
            "--output-path",
            str(registry_path),
        ]
    )
    captured = capsys.readouterr()
    preserved_sidecar = json.loads(
        collections_path.read_text(encoding="utf-8")
    )

    assert rc == 0
    assert any(
        entry["name"] == "custom_warps"
        for entry in preserved_sidecar["collections"]
    )
    registry = pd.read_parquet(registry_path)
    warps = registry[registry["butler_datasettype"] == "visit_image"]
    assert warps["exposure_index"].tolist() == [0, 1, 2, 3]
    assert set(warps["tract"]) == {1}
    assert set(warps["patch"]) == {"01"}
    assert set(warps["band"]) == {"i"}
    assert "fits_header_status" in registry.columns
    assert "object_count_status" in registry.columns
    for coadd_path in (fits_root / "coadd").rglob("*.fits"):
        coadd_path.unlink()

    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        extra_lines=[
            f"butler_registry: {registry_path}",
            "butler_datasettype: visit_image",
            "butler_query: \"band == 'i'\"",
            "butler_exposure_index_column: exposure_index",
            "positive_catalog_object_cap: 3",
            "negative_catalog_object_cap: 3",
            "positive_quality_snr_min: 99.0",
            "positive_quality_flux_ratio_min: 0.95",
        ],
    )
    output_dir = tmp_path / "hsc-registry-backed"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    build_summary = json.loads(captured.out)

    assert rc == 0
    assert build_summary["butler_registry"]["allowed_exposure_indices"] == [
        0,
        1,
        2,
        3,
    ]
    assert build_summary["catalog_object_caps"] == {
        "negative": 3,
        "positive": 3,
    }
    assert (
        build_summary["catalog_object_usage"]["negative"][
            "max_samples_per_object"
        ]
        <= 3
    )
    rows = load_metadata_rows(output_dir)
    assert {row["butler_tract"] for row in rows} == {1}
    assert {row["butler_patch"] for row in rows} == {"01"}
    assert {row["butler_band"] for row in rows} == {"i"}
    positives = [row for row in rows if int(row["label"]) == 1]
    assert positives
    assert all(
        "weak_snr" in row["positive_quality_stratum"] for row in positives
    )

    missing_collection_manifest = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        extra_lines=[
            f"butler_registry: {registry_path}",
            "butler_collection: missing-collection",
            "butler_datasettype: visit_image",
            "butler_exposure_index_column: exposure_index",
        ],
    )
    missing_collection_output = tmp_path / "hsc-missing-collection"
    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(missing_collection_manifest),
            "--output-dir",
            str(missing_collection_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing-collection" in captured.err
    assert "not defined in the sidecar" in captured.err

    sidecar_payload = json.loads(collections_path.read_text(encoding="utf-8"))
    sidecar_payload["collections"].append(
        {
            "name": "broken-chain",
            "type": "CHAINED",
            "members": ["hsc_direct_warps", "missing-member"],
        }
    )
    collections_path.write_text(
        json.dumps(sidecar_payload) + "\n",
        encoding="utf-8",
    )
    missing_member_manifest = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        extra_lines=[
            f"butler_registry: {registry_path}",
            "butler_collection: broken-chain",
            "butler_datasettype: visit_image",
            "butler_exposure_index_column: exposure_index",
        ],
    )
    missing_member_output = tmp_path / "hsc-missing-member"
    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(missing_member_manifest),
            "--output-dir",
            str(missing_member_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing-member" in captured.err
    assert "not defined in the sidecar" in captured.err


def test_cli_build_hsc_registry_includes_uppercase_fits_suffix(
    tmp_path,
    capsys,
) -> None:
    pytest.importorskip("pyarrow")
    hsc_npy = write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    fits_root = write_fake_hsc_fits_products(tmp_path / "data")
    warp_path = next((fits_root / "warps").rglob("*.fits"))
    warp_path.rename(warp_path.with_suffix(".FITS"))
    registry_path = tmp_path / "hsc_uppercase_suffix_registry.parquet"

    rc = _run_cli(
        [
            "data-build-hsc-registry",
            "--fits-root",
            str(fits_root),
            "--hsc-npy-dir",
            str(hsc_npy),
            "--output-path",
            str(registry_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["row_count"] == 6
    assert payload["warp_counts_by_band"] == {"i": 4}


def test_cli_build_hsc_registry_rejects_unexpected_hsc_npy_layout(
    tmp_path,
    capsys,
) -> None:
    fits_root = write_fake_hsc_fits_products(tmp_path / "data")
    hsc_npy = tmp_path / "bad-hsc-npy"
    hsc_npy.mkdir()
    np.save(
        hsc_npy / "images.npy",
        np.zeros((4, 16, 16), dtype=np.float32),
        allow_pickle=False,
    )

    rc = _run_cli(
        [
            "data-build-hsc-registry",
            "--fits-root",
            str(fits_root),
            "--hsc-npy-dir",
            str(hsc_npy),
            "--output-path",
            str(tmp_path / "bad-layout-registry.parquet"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "(filter, y, x, exposure)" in captured.err


def test_cli_build_hsc_registry_assigns_exposure_index_per_band(
    tmp_path,
    capsys,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    hsc_npy = write_fake_hsc_store(tmp_path / "data")
    fits_root = write_fake_hsc_fits_products(tmp_path / "data")
    g_dir = fits_root / "warps" / "0001" / "01" / "HSC-G"
    g_dir.mkdir(parents=True)
    for visit in (2001, 2003, 2005, 2007):
        (
            g_dir
            / (
                "deepCoadd_directWarp_HSC_0001_01_g_HSC-G_"
                f"{visit}_test-run.fits"
            )
        ).write_bytes(b"")
    registry_path = tmp_path / "hsc_multiband_registry.parquet"

    rc = _run_cli(
        [
            "data-build-hsc-registry",
            "--fits-root",
            str(fits_root),
            "--hsc-npy-dir",
            str(hsc_npy),
            "--output-path",
            str(registry_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    registry = pd.read_parquet(registry_path)
    warps = registry[registry["butler_datasettype"] == "visit_image"]

    assert rc == 0
    assert payload["warp_counts_by_band"] == {"g": 4, "i": 4}
    assert warps[warps["band"] == "g"]["exposure_index"].tolist() == [
        0,
        1,
        2,
        3,
    ]
    assert warps[warps["band"] == "i"]["exposure_index"].tolist() == [
        0,
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError, match="duplicate exposure index"):
        resolve_butler_registry_context(
            {
                "butler_registry": str(registry_path),
                "butler_exposure_index_column": "exposure_index",
            },
            exposure_count=4,
        )
    context = resolve_butler_registry_context(
        {
            "butler_registry": str(registry_path),
            "butler_exposure_index_column": "exposure_index",
            "butler_query": "band == 'i'",
        },
        exposure_count=4,
    )
    assert context is not None
    assert context.allowed_exposure_indices == [0, 1, 2, 3]


def test_hsc_path_metadata_parses_underscore_patches(tmp_path) -> None:
    fits_root = tmp_path / "HSC" / "FITS"
    cases = [
        (
            fits_root
            / "warps"
            / "0002"
            / "1_2"
            / "HSC-I"
            / "deepCoadd_directWarp_HSC_0002_1_2_i_HSC-I_2001_run.fits",
            {"tract": 2, "patch": "1_2", "band": "i", "visit": 2001},
        ),
        (
            fits_root
            / "coadd"
            / "0002"
            / "1_2"
            / "HSC-I"
            / "deepCoadd_0002_1_2_i_run.fits",
            {"tract": 2, "patch": "1_2", "band": "i", "visit": None},
        ),
        (
            fits_root
            / "catalogs"
            / "0002"
            / "1_2"
            / "objectTable_0002_1_2_run.parq",
            {"tract": 2, "patch": "1_2", "band": None, "visit": None},
        ),
        (
            fits_root
            / "catalogs"
            / "0002"
            / "0,1"
            / "objectTable_0002_0,1_HSC-G_2024A_run.parq",
            {"tract": 2, "patch": "0,1", "band": None, "visit": None},
        ),
        (
            fits_root
            / "catalogs"
            / "0002"
            / "001_002"
            / "objectTable_0002_001_002_run_hash.parq",
            {
                "tract": 2,
                "patch": "001_002",
                "band": None,
                "visit": None,
            },
        ),
    ]

    for path, expected in cases:
        parsed = _parse_hsc_path_metadata(path, fits_root=fits_root)
        for key, value in expected.items():
            assert parsed[key] == value


def test_cli_build_lsstcomcam_smoke_from_registry_manifest(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in registry_rows:
        if row.get("product") in {"visit_image", "difference_image"}:
            row.pop("tract", None)
            row.pop("patch", None)
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in registry_rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "include_coadd_metadata: true",
                "split_fractions: [0.0, 1.0, 0.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "lsstcomcam-smoke"
    assert payload["registry_row_count"] == 4
    assert payload["compatible_pair_count"] == 1
    assert payload["sample_count"] == 1
    search = np.load(output_dir / "search.npy")
    difference = np.load(output_dir / "difference.npy")
    template = np.load(output_dir / "template.npy")
    assert search.shape == (1, 5, 5)
    assert np.allclose(difference, 3.0)
    assert np.allclose(template, search - difference)

    validation = validate_dataset_dir(
        output_dir,
        dataset_kind="lsstcomcam-smoke",
    )
    assert validation["valid"] is True
    assert validation["semantic_checks"]["labels_are_placeholders"] is True
    assert (
        validation["semantic_checks"]["placeholder_label_source"]
        == "unlabeled_lsstcomcam_smoke_placeholder"
    )
    rows = load_metadata_rows(output_dir)
    assert rows[0]["label_source"] == "unlabeled_lsstcomcam_smoke_placeholder"
    assert rows[0]["visit_image_visit"] == 1001
    assert rows[0]["difference_image_detector"] == 0
    assert (
        rows[0]["template_coadd_lookup_status"] == "visit_missing_tract_patch"
    )
    assert rows[0]["deep_coadd_lookup_status"] == "visit_missing_tract_patch"

    original_labels = np.load(output_dir / "labels.npy")
    bad_labels = original_labels.copy()
    bad_labels[0] = 1
    np.save(output_dir / "labels.npy", bad_labels, allow_pickle=False)
    with pytest.raises(ValueError, match="placeholder zeros"):
        validate_dataset_dir(output_dir, dataset_kind="lsstcomcam-smoke")
    np.save(output_dir / "labels.npy", original_labels, allow_pickle=False)

    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["label_source"] = "reviewed_label"
    (output_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in bad_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metadata label_source"):
        validate_dataset_dir(output_dir, dataset_kind="lsstcomcam-smoke")
    (output_dir / "metadata.jsonl").unlink()
    with pytest.raises(ValueError, match="requires metadata rows"):
        validate_dataset_dir(output_dir, dataset_kind="lsstcomcam-smoke")


def test_cli_build_lsstcomcam_smoke_uses_candidate_catalog_centers(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    candidate_path = tmp_path / "candidates.csv"
    candidate_path.write_text(
        "\n".join(
            [
                "candidate_id,visit,detector,band,x,y",
                "dia-1,1001,0,r,4,2",
                "unmatched,1001,9,r,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-candidates.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {candidate_path}",
                "sample_count: 1",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-candidates"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["candidate_catalog_path"] == str(candidate_path)
    assert payload["candidate_catalog_row_count"] == 2
    assert payload["candidate_matched_sample_count"] == 1
    search = np.load(output_dir / "search.npy")
    difference = np.load(output_dir / "difference.npy")
    yy, xx = np.mgrid[:7, :7]
    expected = (100 + yy * 10 + xx).astype(np.float32)[1:4, 3:6]
    np.testing.assert_allclose(search[0], expected)
    np.testing.assert_allclose(difference[0], 3.0)

    validation = validate_dataset_dir(
        output_dir,
        dataset_kind="lsstcomcam-smoke",
    )
    assert validation["valid"] is True
    rows = load_metadata_rows(output_dir)
    assert rows[0]["candidate_id"] == "lsstcomcam-1001-0-r-dia-1"
    assert rows[0]["center_source"] == "candidate_catalog_pixel"
    assert rows[0]["x"] == 4
    assert rows[0]["y"] == 2
    assert rows[0]["candidate_candidate_id"] == "dia-1"
    assert rows[0]["candidate_catalog_row_index"] == 0


def test_cli_build_lsstcomcam_smoke_projects_candidate_sky_centers(
    tmp_path,
    capsys,
) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    wcs_module = pytest.importorskip("astropy.wcs")
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    difference_path = Path(
        next(
            row["path"]
            for row in registry_rows
            if row["product"] == "difference_image" and row["visit"] == 1001
        )
    )
    wcs = wcs_module.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, -10.0]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    yy, xx = np.mgrid[:7, :7]
    difference_data = np.full((7, 7), 3.0, dtype=np.float32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=difference_data,
                header=wcs.to_header(),
                name="IMAGE",
            ),
        ]
    ).writeto(difference_path, overwrite=True)
    ra_deg, dec_deg = wcs.all_pix2world([[4.0, 2.0]], 0)[0]

    candidate_path = tmp_path / "sky-candidates.csv"
    candidate_path.write_text(
        "\n".join(
            [
                "candidate_id,visit,detector,band,ra_deg,dec_deg",
                f"dia-sky,1001,0,r,{ra_deg},{dec_deg}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-sky-candidates.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {candidate_path}",
                "sample_count: 1",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-sky-candidates"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["candidate_wcs_product"] == "difference_image"
    search = np.load(output_dir / "search.npy")
    expected = (100 + yy * 10 + xx).astype(np.float32)[1:4, 3:6]
    np.testing.assert_allclose(search[0], expected)
    rows = load_metadata_rows(output_dir)
    assert rows[0]["candidate_id"] == "lsstcomcam-1001-0-r-dia-sky"
    assert (
        rows[0]["center_source"]
        == "candidate_catalog_sky_difference_image_wcs"
    )
    assert rows[0]["x"] == 4
    assert rows[0]["y"] == 2


def test_cli_build_lsstcomcam_smoke_defers_unselected_sky_wcs_reads(
    tmp_path,
    capsys,
) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    wcs_module = pytest.importorskip("astropy.wcs")
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    difference_path = Path(
        next(
            row["path"]
            for row in registry_rows
            if row["product"] == "difference_image" and row["visit"] == 1001
        )
    )
    unselected_difference_path = Path(
        next(
            row["path"]
            for row in registry_rows
            if row["product"] == "difference_image" and row["visit"] == 1002
        )
    )
    unselected_difference_path.unlink()
    wcs = wcs_module.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, -10.0]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    yy, xx = np.mgrid[:7, :7]
    difference_data = np.full((7, 7), 3.0, dtype=np.float32)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=difference_data,
                header=wcs.to_header(),
                name="IMAGE",
            ),
        ]
    ).writeto(difference_path, overwrite=True)
    ra_deg, dec_deg = wcs.all_pix2world([[4.0, 2.0]], 0)[0]

    candidate_path = tmp_path / "sky-candidates-with-missing-unselected.csv"
    candidate_path.write_text(
        "\n".join(
            [
                "candidate_id,visit,detector,band,ra_deg,dec_deg",
                f"selected,1001,0,r,{ra_deg},{dec_deg}",
                "unselected,1002,1,g,10.0,-10.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-deferred-sky.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {candidate_path}",
                "sample_count: 1",
                "stamp_size: 3",
                "shuffle: false",
                "bands: [r, g]",
                "visits: [1001, 1002]",
                "detectors: [0, 1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-deferred-sky"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["candidate_catalog_row_count"] == 2
    assert payload["candidate_matched_sample_count"] == 1
    search = np.load(output_dir / "search.npy")
    expected = (100 + yy * 10 + xx).astype(np.float32)[1:4, 3:6]
    np.testing.assert_allclose(search[0], expected)
    rows = load_metadata_rows(output_dir)
    assert rows[0]["candidate_id"] == "lsstcomcam-1001-0-r-selected"
    assert (
        rows[0]["center_source"]
        == "candidate_catalog_sky_difference_image_wcs"
    )


def test_cli_build_lsstcomcam_smoke_accepts_dia_source_catalog_shape(
    tmp_path,
    capsys,
) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    wcs_module = pytest.importorskip("astropy.wcs")
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    difference_path = Path(
        next(
            row["path"]
            for row in registry_rows
            if row["product"] == "difference_image" and row["visit"] == 1001
        )
    )
    wcs = wcs_module.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, -10.0]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.full((7, 7), 3.0, dtype=np.float32),
                header=wcs.to_header(),
                name="IMAGE",
            ),
        ]
    ).writeto(difference_path, overwrite=True)
    coord_ra, coord_dec = wcs.all_pix2world([[4.0, 2.0]], 0)[0]

    dia_source_path = tmp_path / "dia_source.csv"
    dia_source_path.write_text(
        "\n".join(
            [
                (
                    "diaSourceId,diaObjectId,visit,detector,band,"
                    "coord_ra,coord_dec,reliability"
                ),
                (f"991,881,1001,0,r,{coord_ra},{coord_dec},0.97"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-dia-source.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {dia_source_path}",
                "candidate_catalog_format: dia_source_csv",
                "candidate_catalog_provenance: approved_local_catalog",
                "candidate_label_status: placeholder",
                "sample_count: 1",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-dia-source"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["candidate_catalog_path"] == str(dia_source_path)
    assert payload["candidate_catalog_format"] == "dia_source_csv"
    assert payload["candidate_catalog_provenance"] == "approved_local_catalog"
    assert payload["candidate_label_status"] == "placeholder"
    rows = load_metadata_rows(output_dir)
    assert rows[0]["candidate_id"] == "lsstcomcam-1001-0-r-991"
    assert rows[0]["center_source"] == (
        "candidate_catalog_sky_difference_image_wcs"
    )
    assert rows[0]["candidate_diaSourceId"] == 991
    assert rows[0]["candidate_reliability"] == 0.97
    assert rows[0]["label_source"] == "unlabeled_lsstcomcam_smoke_placeholder"
    assert rows[0]["candidate_catalog_format"] == "dia_source_csv"
    assert rows[0]["candidate_catalog_provenance"] == "approved_local_catalog"
    assert rows[0]["candidate_label_status"] == "placeholder"


def test_cli_check_lsstcomcam_candidates_accepts_dia_source_shape(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    dia_source_path = tmp_path / "dia_source.csv"
    dia_source_path.write_text(
        "\n".join(
            [
                (
                    "diaSourceId,diaObjectId,visit,detector,band,"
                    "coord_ra,coord_dec,reliability"
                ),
                "991,881,1001,0,r,10.0,-10.0,0.97",
                "992,882,1001,9,r,10.1,-10.1,0.51",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-dia-source-check.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {dia_source_path}",
                "candidate_catalog_format: dia_source_csv",
                "candidate_catalog_provenance: approved_local_catalog",
                "candidate_label_status: placeholder",
                "sample_count: 1",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-check-lsstcomcam-candidates",
            "--manifest",
            str(manifest_path),
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["candidate_catalog_format"] == "dia_source_csv"
    assert payload["candidate_catalog_provenance"] == "approved_local_catalog"
    assert payload["candidate_label_status"] == "placeholder"
    assert payload["candidate_catalog_row_count"] == 2
    assert payload["candidate_matched_row_count"] == 1
    assert payload["candidate_unmatched_row_count"] == 1
    assert payload["candidate_id_columns"] == ["diaSourceId"]
    assert payload["candidate_center_columns"]["sky_pairs"] == [
        ["coord_ra", "coord_dec"]
    ]


def test_cli_plan_lsstcomcam_staging_with_missing_sky_fits(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in registry_rows:
        if row["product"] in {"visit_image", "difference_image"}:
            Path(row["path"]).unlink()
    dia_source_path = tmp_path / "dia_source.csv"
    dia_source_path.write_text(
        "\n".join(
            [
                (
                    "diaSourceId,diaObjectId,visit,detector,band,"
                    "coord_ra,coord_dec,reliability"
                ),
                "991,881,1001,0,r,10.0,-10.0,0.97",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-staging-plan.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {dia_source_path}",
                "sample_count: 1",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-plan-lsstcomcam-staging",
            "--manifest",
            str(manifest_path),
            "--sample-count",
            "1",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["artifact_kind"] == "lsstcomcam_staging_plan"
    assert payload["sample_count_selected"] == 1
    assert payload["required_file_count"] == 2
    assert payload["missing_required_file_count"] == 2
    assert payload["read_required_products"] == [
        "visit_image",
        "difference_image",
    ]
    assert payload["metadata_only_file_count"] == 2
    assert payload["missing_metadata_only_file_count"] == 2
    assert {row["product"] for row in payload["required_files"]} == {
        "visit_image",
        "difference_image",
    }
    selected = payload["selected_samples"][0]
    assert selected["candidate_id"] == "lsstcomcam-1001-0-r-991"
    assert selected["center_source"] == (
        "candidate_catalog_sky_difference_image_wcs_pending_fits"
    )
    assert selected["sky_center_columns"] == ["coord_ra", "coord_dec"]
    assert selected["visit_image_exists"] is False
    assert selected["difference_image_exists"] is False


def test_cli_build_lsstcomcam_smoke_uses_path_rewrites(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    local_fits_root = tmp_path / "fits"
    remote_root = "/remote/local-products"
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in registry_rows:
        path = Path(row["path"])
        try:
            relative = path.relative_to(local_fits_root)
        except ValueError:
            continue
        row["path"] = f"{remote_root}/{relative.as_posix()}"
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in registry_rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-rewrite.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "path_rewrites:",
                f"  - source_prefix: {remote_root}",
                f"    target_prefix: {local_fits_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-rewrite"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["path_rewrites"] == [
        {
            "source_prefix": remote_root,
            "target_prefix": str(local_fits_root),
        }
    ]
    rows = load_metadata_rows(output_dir)
    assert rows[0]["visit_image_original_path"] == (
        f"{remote_root}/visit-r0.fits"
    )
    assert rows[0]["difference_image_original_path"] == (
        f"{remote_root}/difference-r0.fits"
    )
    assert rows[0]["visit_image_path"] == str(
        local_fits_root / "visit-r0.fits"
    )
    assert rows[0]["difference_image_path"] == str(
        local_fits_root / "difference-r0.fits"
    )

    rc = _run_cli(
        [
            "data-plan-lsstcomcam-staging",
            "--manifest",
            str(manifest_path),
        ]
    )
    captured = capsys.readouterr()
    plan = json.loads(captured.out)

    assert rc == 0
    assert plan["missing_required_file_count"] == 0
    assert {
        tuple(item["original_paths"]) for item in plan["required_files"]
    } == {
        (f"{remote_root}/visit-r0.fits",),
        (f"{remote_root}/difference-r0.fits",),
    }


def test_cli_stage_lsstcomcam_fits_by_remote_prefix(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    local_fits_root = tmp_path / "fits"
    remote_root = "/remote/local-products"
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in registry_rows:
        path = Path(row["path"])
        try:
            relative = path.relative_to(local_fits_root)
        except ValueError:
            continue
        row["path"] = f"{remote_root}/{relative.as_posix()}"
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in registry_rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-remote.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target_root = tmp_path / "staged"

    rc = _run_cli(
        [
            "data-stage-lsstcomcam-fits",
            "--manifest",
            str(manifest_path),
            "--source-prefix",
            remote_root,
            "--target-prefix",
            str(target_root),
            "--search-roots",
            str(local_fits_root),
            "--sample-count",
            "1",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["artifact_kind"] == "lsstcomcam_fits_stage_result"
    assert payload["ok"] is True
    assert payload["required_file_count"] == 2
    assert payload["staged_count"] == 2
    assert payload["missing_count"] == 0
    assert (target_root / "visit-r0.fits").is_symlink()
    assert (target_root / "difference-r0.fits").is_symlink()

    rewritten_manifest_path = tmp_path / "lsstcomcam-smoke-staged.yaml"
    rewritten_manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8")
        + "\n".join(
            [
                "path_rewrites:",
                f"  - source_prefix: {remote_root}",
                f"    target_prefix: {target_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "data-plan-lsstcomcam-staging",
            "--manifest",
            str(rewritten_manifest_path),
            "--sample-count",
            "1",
        ]
    )
    captured = capsys.readouterr()
    plan = json.loads(captured.out)

    assert rc == 0
    assert plan["missing_required_file_count"] == 0


def test_cli_stage_lsstcomcam_fits_reports_stale_existing_target(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    local_fits_root = tmp_path / "fits"
    remote_root = "/remote/local-products"
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in registry_rows:
        path = Path(row["path"])
        try:
            relative = path.relative_to(local_fits_root)
        except ValueError:
            continue
        row["path"] = f"{remote_root}/{relative.as_posix()}"
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in registry_rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-stale-target.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target_root = tmp_path / "staged"
    target_root.mkdir()
    stale_target = target_root / "visit-r0.fits"
    stale_target.symlink_to(stale_target)

    rc = _run_cli(
        [
            "data-stage-lsstcomcam-fits",
            "--manifest",
            str(manifest_path),
            "--source-prefix",
            remote_root,
            "--target-prefix",
            str(target_root),
            "--search-roots",
            str(local_fits_root),
            "--sample-count",
            "1",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ok"] is False
    assert payload["staged_count"] == 1
    assert payload["conflict_count"] == 1
    assert payload["conflicts"][0]["relative_path"] == "visit-r0.fits"
    assert payload["conflicts"][0]["target_status"]["reason"] == (
        "stale_symlink"
    )
    assert (
        payload["conflicts"][0]["target_status"]["target"]["resolved_path"]
        is None
    )


def test_cli_check_lsstcomcam_candidates_strict_accepts_full_match(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    dia_source_path = tmp_path / "dia_source.csv"
    dia_source_path.write_text(
        "\n".join(
            [
                (
                    "diaSourceId,diaObjectId,visit,detector,band,"
                    "coord_ra,coord_dec,reliability"
                ),
                "991,881,1001,0,r,10.0,-10.0,0.97",
                "992,882,1001,0,r,10.1,-10.1,0.51",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-dia-source-check.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {dia_source_path}",
                "sample_count: 2",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-check-lsstcomcam-candidates",
            "--manifest",
            str(manifest_path),
            "--strict",
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["strict"] is True
    assert payload["candidate_full_match"] is True
    assert payload["candidate_identity_column"] == "diaSourceId"
    assert payload["candidate_missing_id_count"] == 0
    assert payload["candidate_duplicate_id_count"] == 0


def test_cli_check_lsstcomcam_candidates_strict_rejects_weak_catalog(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    dia_source_path = tmp_path / "dia_source.csv"
    dia_source_path.write_text(
        "\n".join(
            [
                (
                    "diaSourceId,diaObjectId,visit,detector,band,"
                    "coord_ra,coord_dec,reliability"
                ),
                "991,881,1001,0,r,10.0,-10.0,0.97",
                "991,882,1001,0,r,10.1,-10.1,0.51",
                "993,883,1001,9,r,10.2,-10.2,0.44",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-dia-source-check.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {dia_source_path}",
                "sample_count: 2",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-check-lsstcomcam-candidates",
            "--manifest",
            str(manifest_path),
            "--strict",
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 1
    assert payload["ok"] is False
    assert payload["strict"] is True
    assert payload["candidate_matched_row_count"] == 2
    assert payload["candidate_unmatched_row_count"] == 1
    assert payload["candidate_full_match"] is False
    assert payload["candidate_duplicate_id_count"] == 1
    assert payload["candidate_duplicate_ids"] == ["lsstcomcam-1001-0-r-991"]
    assert "candidate_rows_not_all_matched" in payload["errors"]
    assert "candidate_identity_values_duplicate" in payload["errors"]
    assert "candidate_rows_not_all_matched" in captured.err


def test_cli_check_lsstcomcam_candidates_requires_candidate_id(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    candidate_path = tmp_path / "candidates-no-id.csv"
    candidate_path.write_text(
        "\n".join(
            [
                "visit,detector,band,coord_ra,coord_dec",
                "1001,0,r,10.0,-10.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-candidates-no-id.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {candidate_path}",
                "sample_count: 1",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-check-lsstcomcam-candidates",
            "--manifest",
            str(manifest_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ok"] is False
    assert payload["candidate_id_columns"] == []
    assert "candidate_id_columns_missing" in payload["errors"]

    rc = _run_cli(
        [
            "data-check-lsstcomcam-candidates",
            "--manifest",
            str(manifest_path),
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "candidate_id_columns_missing" in captured.err


def test_cli_check_lsstcomcam_candidates_reports_missing_catalog(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    missing_path = tmp_path / "missing-dia-source.csv"
    manifest_path = tmp_path / "lsstcomcam-smoke-missing-candidates.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                f"candidate_catalog: {missing_path}",
                "sample_count: 1",
                "stamp_size: 3",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-check-lsstcomcam-candidates",
            "--manifest",
            str(manifest_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ok"] is False
    assert payload["candidate_catalog_exists"] is False
    assert payload["errors"] == ["candidate_catalog_not_found"]

    rc = _run_cli(
        [
            "data-check-lsstcomcam-candidates",
            "--manifest",
            str(manifest_path),
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "candidate_catalog_not_found" in captured.err


def test_cli_build_lsstcomcam_smoke_accepts_yaml_date_filters(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    manifest_path = tmp_path / "lsstcomcam-smoke-yaml-date.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date_min: 2000-01-02",
                'date_max: "2000-01-02"',
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "tract: 1",
                "patch: 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-yaml-date"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["compatible_pair_count"] == 1
    assert payload["registry_filters"]["date_min"] == "20000102"
    assert payload["registry_filters"]["date_max"] == "20000102"


def test_cli_build_lsstcomcam_smoke_rejects_boolean_center(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    manifest_path = tmp_path / "lsstcomcam-smoke-bool-center.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "center_x: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "lsstcomcam-smoke-bool-center"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "expected integer value" in captured.err


def test_cli_build_lsstcomcam_smoke_excludes_missing_pair_filters(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    registry_rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in registry_rows:
        if row.get("product") in {"visit_image", "difference_image"}:
            row["date"] = None
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in registry_rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-missing-date.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "tract: 1",
                "patch: 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "missing-date-output"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert (
        "no compatible visit_image + difference_image pairs" in captured.err
    )


def test_cli_build_lsstcomcam_smoke_reports_duplicate_difference_pairs(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    duplicate_path = tmp_path / "fits" / "difference-r0-newer.fits"
    write_fake_lsstcomcam_fits(
        duplicate_path,
        np.full((7, 7), 7.0, dtype=np.float32),
    )
    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    duplicate = dict(rows[1])
    duplicate.update(
        {
            "path": str(duplicate_path),
            "registry_relative_path": "difference_image/r0-newer.fits",
            "mtime_ns": 999,
            "size_bytes": duplicate_path.stat().st_size,
        }
    )
    rows.append(duplicate)
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-duplicate.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "tract: 1",
                "patch: 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-duplicate"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["registry_row_count"] == 5
    assert payload["compatible_pair_count"] == 1
    assert payload["pairing"]["duplicate_difference_key_count"] == 1
    assert (
        payload["pairing"]["duplicate_difference_extra_candidate_count"] == 1
    )
    assert (
        payload["pairing"]["selected_difference_rule"]
        == "highest_mtime_ns_then_path"
    )
    assert payload["pairing"]["duplicate_difference_keys"][0][
        "selected_path"
    ] == str(duplicate_path)
    assert np.allclose(np.load(output_dir / "difference.npy"), 7.0)


def test_cli_build_lsstcomcam_smoke_reports_duplicate_visit_pairs(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    duplicate_path = tmp_path / "fits" / "visit-r0-newer.fits"
    write_fake_lsstcomcam_fits(
        duplicate_path,
        np.full((7, 7), 11.0, dtype=np.float32),
    )
    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    duplicate = dict(rows[0])
    duplicate.update(
        {
            "path": str(duplicate_path),
            "registry_relative_path": "visit_image/r0-newer.fits",
            "mtime_ns": 999,
            "size_bytes": duplicate_path.stat().st_size,
        }
    )
    rows.append(duplicate)
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-duplicate-visit.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "tract: 1",
                "patch: 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "lsstcomcam-smoke-duplicate-visit"

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["registry_row_count"] == 5
    assert payload["compatible_pair_count"] == 1
    assert payload["pairing"]["duplicate_visit_key_count"] == 1
    assert payload["pairing"]["duplicate_visit_extra_candidate_count"] == 1
    assert (
        payload["pairing"]["selected_visit_rule"]
        == "highest_mtime_ns_then_path"
    )
    assert payload["pairing"]["duplicate_visit_keys"][0][
        "selected_path"
    ] == str(duplicate_path)
    assert np.allclose(np.load(output_dir / "search.npy"), 11.0)


def test_cli_build_lsstcomcam_smoke_prefers_normalized_product_filter(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["butler_datasettype"] = f"native_{row['product']}"
    registry_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-native-butler.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "products: [visit_image, difference_image]",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "lsstcomcam-smoke-native-butler"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["registry_row_count"] == 2
    assert payload["compatible_pair_count"] == 1


def test_cli_build_lsstcomcam_smoke_honors_singular_dataset_type(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    manifest_path = tmp_path / "lsstcomcam-smoke-dataset-type.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "dataset_type: not_a_real_product",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "lsstcomcam-smoke-dataset-type"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "registry filters produced no rows" in captured.err


def test_cli_build_lsstcomcam_smoke_rejects_dataset_type_conflict(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    manifest_path = tmp_path / "lsstcomcam-smoke-dataset-type-conflict.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "dataset_types: [visit_image, difference_image]",
                "dataset_type: visit_image",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "lsstcomcam-smoke-dataset-type-conflict"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "conflicting dataset_types and dataset_type" in captured.err


def test_cli_build_lsstcomcam_smoke_rejects_mismatched_pair_shapes(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    difference_path = Path(rows[1]["path"])
    difference_path.unlink()
    write_fake_lsstcomcam_fits(
        difference_path,
        np.full((9, 9), 3.0, dtype=np.float32),
    )
    manifest_path = tmp_path / "lsstcomcam-smoke-mismatched-shapes.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "lsstcomcam-smoke-mismatched-shapes"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "matching image shapes" in captured.err


def test_cli_build_lsstcomcam_smoke_rejects_missing_integer_hdu(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    manifest_path = tmp_path / "lsstcomcam-smoke-missing-hdu.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 5",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
                "image_hdu: 99",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "lsstcomcam-smoke-missing-hdu"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "FITS HDU 99 not found" in captured.err


def test_cli_build_lsstcomcam_smoke_rejects_oversized_stamp_before_output(
    tmp_path,
    capsys,
) -> None:
    registry_path = write_fake_lsstcomcam_registry(tmp_path)
    manifest_path = tmp_path / "lsstcomcam-smoke-oversized.yaml"
    output_dir = tmp_path / "lsstcomcam-smoke-oversized"
    manifest_path.write_text(
        "\n".join(
            [
                f"registry: {registry_path}",
                "sample_count: 1",
                "stamp_size: 11",
                "shuffle: false",
                "date: 20000102",
                "bands: [r]",
                "visits: [1001]",
                "detectors: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-lsstcomcam-smoke",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "stamp_size=11 centered" in captured.err
    assert "does not fit inside" in captured.err
    assert not output_dir.exists()


def test_lsstcomcam_read_fits_stamp_rejects_image_cubes(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "cube.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((2, 7, 7), dtype=np.float32),
                name="IMAGE",
            ),
        ]
    ).writeto(path)

    with pytest.raises(ValueError, match="2D image data"):
        read_fits_stamp(path, hdu="IMAGE", stamp_size=5)


def test_lsstcomcam_finalize_stamp_zeroes_all_nonfinite_values() -> None:
    readonly = np.asarray([[np.nan, 1.0]], dtype=np.float32)
    readonly.setflags(write=False)
    readonly_finalized, readonly_invalid = _finalize_stamp(
        readonly,
        nan_policy="zero",
        name="readonly fixture stamp",
    )
    finalized, invalid = _finalize_stamp(
        np.asarray([[np.nan, np.inf, -np.inf, 2.5]], dtype=np.float32),
        nan_policy="zero",
        name="fixture stamp",
    )

    assert readonly_invalid == 1
    np.testing.assert_allclose(readonly_finalized, [[0.0, 1.0]])
    assert invalid == 3
    np.testing.assert_allclose(finalized, [[0.0, 0.0, 0.0, 2.5]])


def test_cli_build_hsc_with_non_conservative_masks(tmp_path, capsys) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        extra_lines=[
            "mask_mode: non_conservative",
            "mask_min_valid_fraction: 1.0",
            "template_min_valid_exposures: 1",
        ],
    )
    output_dir = tmp_path / "hsc-masked-packaged"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["mask_mode"] == "non_conservative"
    assert payload["masks_available"] is True
    assert payload["sky_available"] is True
    assert payload["mask_min_valid_fraction"] == 1.0
    diagnostics = payload["mask_diagnostics"]
    assert diagnostics["enabled"] is True
    assert diagnostics["valid_mask_global_fraction"] == 1.0
    assert diagnostics["exposure_attempts"] >= 12
    assert diagnostics["exposure_rejections"] == {
        "search_stamp_low_valid_fraction": 0,
        "search_context_low_valid_fraction": 0,
        "centers_without_valid_exposure": 0,
    }
    assert diagnostics["accepted_search_valid_fraction"]["count"] == 12
    assert (
        diagnostics["accepted_difference_context_valid_fraction"]["count"]
        == 12
    )
    assert "chip_gap" in diagnostics["mask_plane_fractions"]
    rows = load_metadata_rows(output_dir)
    assert all(row["search_valid_fraction"] == 1.0 for row in rows)
    assert all(
        row["difference_context_valid_fraction"] == 1.0 for row in rows
    )
    validation = validate_dataset_dir(
        output_dir, dataset_kind="hsc-synthetic"
    )
    assert validation["valid"] is True


def test_cli_build_hsc_from_manifest_with_catalog_negatives(
    tmp_path,
    capsys,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog",
    )
    output_dir = tmp_path / "hsc-catalog-negatives"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["center_source_counts"]["catalog"] == 12
    rows = load_metadata_rows(output_dir)
    negatives = [row for row in rows if int(row["label"]) == 0]
    assert negatives
    assert all(row["center_source"] == "catalog" for row in negatives)


def test_cli_build_hsc_from_manifest_with_catalog_offset_negatives(
    tmp_path,
    capsys,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        negative_offset_range=(4.0, 6.0),
    )
    output_dir = tmp_path / "hsc-catalog-offset-negatives"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["negative_source_mode"] == "catalog-offset"
    assert payload["center_source_counts"]["catalog"] == 6
    assert payload["center_source_counts"]["catalog-offset"] == 6
    rows = load_metadata_rows(output_dir)
    negatives = [row for row in rows if int(row["label"]) == 0]
    assert negatives
    assert all(row["center_source"] == "catalog-offset" for row in negatives)
    assert all(row["catalog_object_id"] in {101, 102} for row in negatives)
    assert all(row["center_offset_radius"] >= 4.0 for row in negatives)
    assert all(row["center_offset_radius"] <= 6.1 for row in negatives)
    assert all(row["center_offset_radius"] > 0.0 for row in negatives)


def test_cli_build_hsc_with_separate_positive_and_negative_catalog_pools(
    tmp_path,
    capsys,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        extra_lines=[
            "positive_catalog_extendedness_max: 0.2",
            "negative_catalog_extendedness_min: 0.5",
        ],
    )
    output_dir = tmp_path / "hsc-separated-pools"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["positive_catalog_candidate_count"] == 1
    assert payload["negative_catalog_candidate_count"] == 1
    rows = load_metadata_rows(output_dir)
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    assert positives and negatives
    assert all(row["catalog_pool_role"] == "positive" for row in positives)
    assert all(row["catalog_pool_role"] == "negative" for row in negatives)
    assert all(row["catalog_object_id"] == 101 for row in positives)
    assert all(row["catalog_object_id"] == 102 for row in negatives)


def test_cli_build_hsc_from_manifest_with_xpois_difference(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    calls: list[tuple[tuple[int, int], tuple[int, int] | None]] = []

    def fake_xpois_difference(
        search_stamp: np.ndarray,
        template_stamp: np.ndarray,
        *,
        variance_stamp: np.ndarray | None = None,
        config=None,
    ) -> np.ndarray:
        del template_stamp, config
        calls.append(
            (
                tuple(search_stamp.shape),
                (
                    tuple(variance_stamp.shape)
                    if variance_stamp is not None
                    else None
                ),
            )
        )
        return np.full(search_stamp.shape, 7.0, dtype=np.float32)

    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        fake_xpois_difference,
    )
    manifest_path = write_hsc_manifest(
        tmp_path,
        difference_mode="xpois",
        extra_lines=["difference_context_size: 25"],
    )
    output_dir = tmp_path / "hsc-xpois-diff"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["difference_mode"] == "xpois_constant"
    assert payload["manifest_difference_mode"] == "xpois"
    assert payload["difference_context_size"] == 25
    assert "xpois" in payload
    assert payload["xpois"]["context_size"] == 25
    assert (output_dir / "difference.npy").exists()
    difference = np.load(output_dir / "difference.npy", allow_pickle=False)
    assert difference.shape == (12, 17, 17)
    assert np.allclose(difference, 7.0)
    assert calls == [((25, 25), (25, 25))] * 12
    rows = load_metadata_rows(output_dir)
    assert rows
    assert all(row["difference_mode"] == "xpois_constant" for row in rows)
    assert all(row["manifest_difference_mode"] == "xpois" for row in rows)
    validation = validate_dataset_dir(
        output_dir, dataset_kind="hsc-synthetic"
    )
    assert validation["valid"] is True


def test_cli_train_triplet_on_hsc_xpois_difference(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        lambda search_stamp, template_stamp, **kwargs: (
            np.asarray(search_stamp, dtype=np.float32)
            - np.asarray(template_stamp, dtype=np.float32)
        ).astype(np.float32),
    )
    manifest_path = write_hsc_manifest(
        tmp_path,
        difference_mode="xpois",
        extra_lines=["difference_context_size: 25"],
    )
    dataset_dir = tmp_path / "hsc-triplet-dataset"
    runs_dir = tmp_path / "runs"

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(dataset_dir),
        ]
    )
    captured = capsys.readouterr()
    build_summary = json.loads(captured.out)
    assert rc == 0
    assert build_summary["difference_mode"] == "xpois_constant"
    assert build_summary["manifest_difference_mode"] == "xpois"

    config_path = tmp_path / "train-triplet.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_dir: {dataset_dir}",
                f"output_root: {runs_dir}",
                "run_name: hsc-triplet-run",
                "epochs: 1",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: auto",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "model:",
                "  input_mode: triplet",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run_cli(["train-inada-triplet", "--config", str(config_path)])
    captured = capsys.readouterr()
    train_summary = json.loads(captured.out)
    assert rc == 0
    run_dir = Path(train_summary["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "checkpoint.pt").exists()


def test_cli_build_autoscan_from_raw_records_case(tmp_path, capsys) -> None:
    manifest_path = write_raw_autoscan_manifest(tmp_path)
    output_dir = tmp_path / "raw-autoscan-packaged"
    rc = _run_cli(
        [
            "data-build-autoscan-raw",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "autoscan"
    assert (output_dir / "difference.npy").exists()
    validation = validate_dataset_dir(output_dir, dataset_kind="autoscan")
    assert validation["valid"] is True
    assert validation["sample_count"] == 4


def test_cli_build_autoscan_from_raw_records_with_label_aliases(
    tmp_path,
    capsys,
) -> None:
    manifest_path = write_raw_autoscan_manifest_with_label_aliases(tmp_path)
    output_dir = tmp_path / "raw-autoscan-labels-packaged"
    rc = _run_cli(
        [
            "data-build-autoscan-raw",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "autoscan"
    validation = validate_dataset_dir(output_dir, dataset_kind="autoscan")
    assert validation["valid"] is True


def test_cli_build_autoscan_from_raw_full_images(tmp_path, capsys) -> None:
    manifest_path = write_raw_autoscan_image_manifest(tmp_path)
    output_dir = tmp_path / "raw-autoscan-image-packaged"
    rc = _run_cli(
        [
            "data-build-autoscan-raw",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "autoscan"
    search = np.load(output_dir / "search.npy", allow_pickle=False)
    assert search.shape == (4, 17, 17)
    validation = validate_dataset_dir(output_dir, dataset_kind="autoscan")
    assert validation["valid"] is True


def test_cli_build_nodiff_from_manifest(tmp_path, capsys) -> None:
    manifest_path = write_prepared_manifest_inputs(
        tmp_path,
        include_difference=False,
        dataset_kind="nodiff",
    )
    output_dir = tmp_path / "packaged-nodiff"
    rc = _run_cli(
        [
            "data-build-nodiff",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "nodiff"
    assert not (output_dir / "difference.npy").exists()

    rc = _run_cli(
        [
            "data-validate",
            "--dataset-dir",
            str(output_dir),
            "--dataset-kind",
            "nodiff",
        ]
    )
    captured = capsys.readouterr()
    validation = json.loads(captured.out)
    assert rc == 0
    assert validation["valid"] is True
    assert (
        validation["semantic_checks"]["diff_snr_threshold_enforced"] is True
    )
    assert validation["semantic_checks"]["autoscan_scores_present"] is True


def test_cli_build_nodiff_from_release_shard_case(tmp_path, capsys) -> None:
    manifest_path = write_nodiff_release_manifest(tmp_path)
    output_dir = tmp_path / "release-nodiff-packaged"
    rc = _run_cli(
        [
            "data-build-nodiff-release",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "nodiff"
    assert payload["input_mode"] == "pair"
    assert payload["builder_summary"]["release_group_count"] == 6
    assert payload["builder_summary"]["release_incomplete_group_count"] == 0
    assert payload["builder_summary"]["positive_count_selected"] == 2
    assert payload["builder_summary"]["negative_count_selected"] == 2
    assert (
        payload["builder_summary"]["autoscan_scores"]
        == "release_provenance_only"
    )
    assert not (output_dir / "difference.npy").exists()

    labels = np.load(output_dir / "labels.npy", allow_pickle=False)
    assert np.bincount(labels, minlength=2).tolist() == [2, 2]
    rows = load_metadata_rows(output_dir)
    assert {row["nodiff_release_cut_provenance"] for row in rows} == {
        "published_release_inclusion"
    }
    assert {row["label_source"] for row in rows} == {
        "nodiff_release_filename"
    }

    validation = validate_dataset_dir(output_dir, dataset_kind="nodiff")
    checks = validation["semantic_checks"]
    assert validation["valid"] is True
    assert validation["input_mode"] == "pair"
    assert checks["required_difference_absent"] is True
    assert checks["release_cut_provenance_count"] == 2
    assert checks["autoscan_scores_present"] is False
    assert checks["autoscan_scores_release_provenance_count"] == 2
    assert checks["diff_snr_values_present"] is False
    assert checks["diff_snr_release_provenance_count"] == 2
    assert checks["diff_snr_threshold_enforced"] is True


def test_cli_build_nodiff_from_release_shard_without_class_cap(
    tmp_path, capsys
) -> None:
    manifest_path = write_nodiff_release_manifest(
        tmp_path,
        max_per_class=None,
    )
    output_dir = tmp_path / "release-nodiff-full-balance"
    rc = _run_cli(
        [
            "data-build-nodiff-release",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["sample_count"] == 6
    assert payload["builder_summary"]["max_per_class"] is None
    assert payload["builder_summary"]["positive_count_selected"] == 3
    assert payload["builder_summary"]["negative_count_selected"] == 3


def test_nodiff_release_inventory_reports_incomplete_groups(
    tmp_path,
) -> None:
    source_dir = tmp_path / "release"
    write_release_stamp_group(
        source_dir,
        global_id=1,
        label_name="pos",
    )
    write_release_stamp_group(
        source_dir,
        global_id=2,
        label_name="neg",
        kinds=("srch", "tmpl"),
    )

    inventory = collect_nodiff_release_group_inventory(source_dir)

    assert len(inventory["complete_groups"]) == 1
    assert inventory["incomplete_group_count"] == 1
    assert inventory["incomplete_group_examples"] == [
        {"global_id": 2, "missing": ["difference_path"]}
    ]


def test_nodiff_release_inventory_rejects_duplicate_stamp_kind(
    tmp_path,
) -> None:
    source_dir = tmp_path / "release"
    write_release_stamp_group(
        source_dir,
        global_id=1,
        label_name="pos",
    )
    duplicate_dir = source_dir / "gid_1"
    np.save(
        duplicate_dir / "99999_srch_1_srch_pos.npy",
        np.ones((17, 17), dtype=np.float32),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="duplicate search stamp"):
        collect_nodiff_release_group_inventory(source_dir)


def test_nodiff_release_inventory_rejects_disagreeing_labels(
    tmp_path,
) -> None:
    source_dir = tmp_path / "release"
    group_dir = source_dir / "gid_1"
    group_dir.mkdir(parents=True)
    stamp = np.ones((17, 17), dtype=np.float32)
    np.save(group_dir / "00001_srch_1_srch_pos.npy", stamp)
    np.save(group_dir / "00002_tmpl_1_tmpl_neg.npy", stamp)
    np.save(group_dir / "00003_diff_1_diff_pos.npy", stamp)

    with pytest.raises(ValueError, match="labels disagree"):
        collect_nodiff_release_group_inventory(source_dir)


def test_cli_build_nodiff_release_rejects_missing_label_class(
    tmp_path, capsys
) -> None:
    source_dir = tmp_path / "release"
    write_release_stamp_group(
        source_dir,
        global_id=1,
        label_name="pos",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "source_dir": str(source_dir),
                "stamp_size": 17,
                "split_fractions": [0.8, 0.1, 0.1],
            }
        ),
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-nodiff-release",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "release slice must contain both labels" in captured.err


def test_cli_build_nodiff_release_rejects_bad_split_fractions(
    tmp_path, capsys
) -> None:
    manifest_path = write_nodiff_release_manifest(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["split_fractions"] = [0.7, 0.2]
    manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "data-build-nodiff-release",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "split_fractions must contain exactly three values" in captured.err


def test_validate_nodiff_accepts_mixed_release_provenance_positive(
    tmp_path,
) -> None:
    manifest_path = write_prepared_manifest_inputs(
        tmp_path,
        include_difference=False,
        dataset_kind="nodiff",
    )
    output_dir = tmp_path / "packaged-nodiff"
    rc = _run_cli(
        [
            "data-build-nodiff",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0

    rows = load_metadata_rows(output_dir)
    first_positive = next(row for row in rows if int(row["label"]) == 1)
    first_positive["autoscan_score"] = None
    first_positive["diff_snr"] = None
    first_positive["nodiff_release_cut_provenance"] = (
        "published_release_inclusion"
    )
    (output_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    validation = validate_dataset_dir(output_dir, dataset_kind="nodiff")
    checks = validation["semantic_checks"]

    assert validation["valid"] is True
    assert checks["release_cut_provenance_count"] == 1
    assert checks["autoscan_scores_present"] is False
    assert checks["autoscan_scores_release_provenance_count"] == 1
    assert checks["diff_snr_values_present"] is False
    assert checks["diff_snr_release_provenance_count"] == 1


def test_detect_search_sources_uses_photometry() -> None:
    pytest.importorskip("photutils")
    rng = np.random.default_rng(90210)
    image = rng.normal(loc=100.0, scale=1.0, size=(64, 64))
    image[29:32, 37:40] += 20.0

    sources = detect_search_sources(
        image,
        threshold_sigma=5.0,
        minarea=3,
    )

    assert len(sources) == 1
    assert sources[0]["x"] == pytest.approx(38.0, abs=0.5)
    assert sources[0]["y"] == pytest.approx(30.0, abs=0.5)
    assert sources[0]["flux"] > 0.0


def test_cli_build_nodiff_from_raw_exposures_case(tmp_path, capsys) -> None:
    pytest.importorskip("photutils")
    manifest_path = write_raw_nodiff_manifest(tmp_path)
    output_dir = tmp_path / "raw-nodiff-packaged"
    rc = _run_cli(
        [
            "data-build-nodiff-raw",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "nodiff"
    assert not (output_dir / "difference.npy").exists()
    validation = validate_dataset_dir(output_dir, dataset_kind="nodiff")
    assert validation["valid"] is True
    assert validation["sample_count"] == 2


def test_cli_build_nodiff_from_raw_exposures_with_subset(
    tmp_path, capsys
) -> None:
    pytest.importorskip("photutils")
    manifest_path = write_raw_nodiff_manifest(tmp_path, subset_fraction=0.5)
    output_dir = tmp_path / "raw-nodiff-subset"
    rc = _run_cli(
        [
            "data-build-nodiff-raw",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "nodiff"
    validation = validate_dataset_dir(output_dir, dataset_kind="nodiff")
    assert validation["valid"] is True
    assert validation["sample_count"] == 2


def test_cli_build_nodiff_from_raw_exposures_with_limits(
    tmp_path, capsys
) -> None:
    pytest.importorskip("photutils")
    manifest_path = write_raw_nodiff_manifest_with_limits(tmp_path)
    output_dir = tmp_path / "raw-nodiff-limits"
    rc = _run_cli(
        [
            "data-build-nodiff-raw",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "nodiff"
    validation = validate_dataset_dir(output_dir, dataset_kind="nodiff")
    assert validation["valid"] is True


def test_cli_build_nodiff_from_raw_exposures_with_aliases(
    tmp_path, capsys
) -> None:
    pytest.importorskip("photutils")
    manifest_path = write_raw_nodiff_manifest_with_aliases(tmp_path)
    output_dir = tmp_path / "raw-nodiff-alias-packaged"
    rc = _run_cli(
        [
            "data-build-nodiff-raw",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["dataset_kind"] == "nodiff"
    validation = validate_dataset_dir(output_dir, dataset_kind="nodiff")
    assert validation["valid"] is True


def test_validate_nodiff_rejects_difference_image(tmp_path) -> None:
    manifest_path = write_prepared_manifest_inputs(
        tmp_path,
        include_difference=True,
        dataset_kind="nodiff",
    )
    output_dir = tmp_path / "invalid-nodiff"
    rc = _run_cli(
        [
            "data-build-autoscan",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    try:
        validate_dataset_dir(output_dir, dataset_kind="nodiff")
    except ValueError as exc:
        assert "must not contain difference.npy" in str(exc)
    else:
        raise AssertionError("expected nodiff validation to fail")


def test_evaluate_predictions_reports_consensus_and_autoscan_baseline(
    tmp_path,
) -> None:
    out_dir = tmp_path / "metrics-out"
    logits = np.array([4.0, 3.0, -3.0, 4.0, 1.0], dtype=np.float64)
    labels = np.array([1, 1, 0, 1, 0], dtype=np.int64)
    rows = [
        {
            "candidate_id": "c1a",
            "exposure_id": 1,
            "ccd_id": 1,
            "band": "i",
            "x": 10,
            "y": 10,
            "split_group": "g1",
            "split": "test",
            "label": 1,
            "center_source": "catalog",
            "catalog_pool_role": "positive",
            "catalog_flux": 110.0,
            "catalog_extendedness": 0.1,
            "fake_id": "f1",
            "autoscan_score": 0.6,
            "diff_snr": 5.0,
            "snr": 5.0,
            "flux_ratio": 0.2,
            "center_offset_radius": 0.0,
            "search_valid_fraction": 1.0,
            "difference_context_valid_fraction": 1.0,
        },
        {
            "candidate_id": "c1b",
            "exposure_id": 1,
            "ccd_id": 1,
            "band": "i",
            "x": 11,
            "y": 11,
            "split_group": "g1",
            "split": "test",
            "label": 1,
            "center_source": "catalog",
            "catalog_pool_role": "positive",
            "catalog_flux": 150.0,
            "catalog_extendedness": 0.2,
            "fake_id": "f1",
            "autoscan_score": 0.4,
            "diff_snr": 6.0,
            "snr": 6.0,
            "flux_ratio": 0.3,
            "center_offset_radius": 0.0,
            "search_valid_fraction": 1.0,
            "difference_context_valid_fraction": 1.0,
        },
        {
            "candidate_id": "c2",
            "exposure_id": 2,
            "ccd_id": 1,
            "band": "i",
            "x": 12,
            "y": 12,
            "split_group": "g2",
            "split": "test",
            "label": 0,
            "center_source": "random",
            "catalog_pool_role": None,
            "catalog_flux": None,
            "catalog_extendedness": None,
            "fake_id": None,
            "autoscan_score": 0.2,
            "diff_snr": 0.0,
            "snr": 0.0,
            "flux_ratio": 0.1,
            "center_offset_radius": None,
            "search_valid_fraction": 0.98,
            "difference_context_valid_fraction": None,
        },
        {
            "candidate_id": "c3",
            "exposure_id": 3,
            "ccd_id": 1,
            "band": "i",
            "x": 13,
            "y": 13,
            "split_group": "g3",
            "split": "test",
            "label": 1,
            "center_source": "catalog-offset",
            "catalog_pool_role": "negative",
            "catalog_flux": 1200.0,
            "catalog_extendedness": 1.0,
            "fake_id": "f2",
            "autoscan_score": 0.9,
            "diff_snr": 7.0,
            "snr": 7.0,
            "flux_ratio": 0.4,
            "center_offset_radius": 5.0,
            "search_valid_fraction": 0.87,
            "difference_context_valid_fraction": 0.84,
        },
        {
            "candidate_id": "c4",
            "exposure_id": 4,
            "ccd_id": 1,
            "band": "i",
            "x": 14,
            "y": 14,
            "split_group": "g4",
            "split": "test",
            "label": 0,
            "center_source": "catalog-offset",
            "catalog_pool_role": "negative",
            "catalog_flux": 900.0,
            "catalog_extendedness": 1.0,
            "fake_id": None,
            "autoscan_score": 0.1,
            "diff_snr": 0.0,
            "snr": 0.0,
            "flux_ratio": 0.8,
            "center_offset_radius": 5.5,
            "search_valid_fraction": 0.99,
            "difference_context_valid_fraction": 0.91,
        },
    ]
    metrics = evaluate_predictions(
        y_true=labels,
        logits=logits,
        metadata_rows=rows,
        output_dir=out_dir,
    )
    assert metrics["consensus"]["group_field"] == "fake_id"
    assert metrics["consensus"]["positive_group_count"] == 2
    assert "autoscan_baseline" in metrics
    assert metrics["autoscan_baseline"]["stamp_recovery_rate"] > 0.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["calibration"]["populated_bin_count"] > 0
    assert (
        metrics["threshold_diagnostics"]["fixed_threshold"]["threshold"]
        == 0.5
    )
    assert "center_source_breakdown" in metrics
    assert metrics["center_source_breakdown"]["field"] == "center_source"
    assert (
        metrics["center_source_breakdown"]["groups"]["catalog"]["count"] == 2
    )
    assert (
        metrics["center_source_breakdown"]["groups"]["random"][
            "negative_false_positive_rate"
        ]
        == 0.0
    )
    assert "catalog_pool_role_breakdown" in metrics
    assert (
        metrics["catalog_pool_role_breakdown"]["groups"]["positive"]["count"]
        == 2
    )
    assert (
        metrics["catalog_morphology_breakdown"]["groups"]["pointlike"][
            "count"
        ]
        == 2
    )
    assert (
        metrics["negative_difficulty_breakdown"]["groups"]["random"][
            "negative_count"
        ]
        == 1
    )
    assert (
        metrics["negative_difficulty_breakdown"]["groups"][
            "catalog-offset:near"
        ]["negative_count"]
        == 1
    )
    assert (
        metrics["mask_pressure_breakdown"]["groups"]["fully-valid"]["count"]
        == 2
    )
    assert (out_dir / "consensus.json").exists()
    assert (out_dir / "autoscan_baseline.json").exists()
    assert (out_dir / "center_source_breakdown.json").exists()
    assert (out_dir / "catalog_pool_role_breakdown.json").exists()
    assert (out_dir / "catalog_morphology_breakdown.json").exists()
    assert (out_dir / "negative_difficulty_breakdown.json").exists()
    assert (out_dir / "mask_pressure_breakdown.json").exists()
    assert (out_dir / "binned_catalog_flux.csv").exists()
    assert (out_dir / "binned_catalog_flux_log10.csv").exists()
    assert (out_dir / "binned_catalog_extendedness.csv").exists()
    assert (out_dir / "binned_center_offset_radius.csv").exists()
    assert (out_dir / "binned_search_valid_fraction.csv").exists()
    assert (out_dir / "binned_difference_context_valid_fraction.csv").exists()
    assert (out_dir / "calibration.json").exists()
    assert (out_dir / "reliability_curve.csv").exists()
    assert (out_dir / "threshold_diagnostics.json").exists()
    assert (out_dir / "threshold_sweep.csv").exists()


def test_evaluate_predictions_reports_validation_selected_threshold(
    tmp_path,
) -> None:
    out_dir = tmp_path / "metrics-out"
    probabilities = np.array([0.60, 0.70, 0.80, 0.90], dtype=np.float64)
    logits = np.log(probabilities / (1.0 - probabilities))
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    rows = [
        {
            "candidate_id": f"c{idx}",
            "exposure_id": idx,
            "ccd_id": 1,
            "band": "i",
            "x": 10 + idx,
            "y": 10 + idx,
            "split_group": f"g{idx}",
            "split": "test",
            "label": int(label),
        }
        for idx, label in enumerate(labels)
    ]
    validation_selection = select_threshold_for_scores(
        labels,
        probabilities,
        metric="accuracy",
        source_split="val",
    )

    metrics = evaluate_predictions(
        y_true=labels,
        logits=logits,
        metadata_rows=rows,
        threshold_selection=validation_selection,
        output_dir=out_dir,
    )

    diagnostics = metrics["threshold_diagnostics"]
    assert metrics["accuracy"] == 0.5
    assert diagnostics["fixed_threshold"]["accuracy"] == 0.5
    assert diagnostics["split_optimal"]["accuracy"] == 1.0
    assert diagnostics["validation_selected"]["source_split"] == "val"
    assert (
        diagnostics["validation_selected"]["evaluated_split_metrics"][
            "accuracy"
        ]
        == 1.0
    )
    assert (out_dir / "threshold_diagnostics.json").exists()


def test_evaluate_predictions_rejects_nonfinite_logits(tmp_path) -> None:
    out_dir = tmp_path / "metrics-out"
    logits = np.array([0.5, np.nan], dtype=np.float64)
    labels = np.array([1, 0], dtype=np.int64)
    rows = [
        {
            "candidate_id": "c1",
            "exposure_id": 1,
            "ccd_id": 1,
            "band": "i",
            "x": 10,
            "y": 10,
            "split_group": "g1",
            "split": "test",
            "label": 1,
            "center_source": "catalog",
        },
        {
            "candidate_id": "c2",
            "exposure_id": 2,
            "ccd_id": 1,
            "band": "i",
            "x": 11,
            "y": 11,
            "split_group": "g2",
            "split": "test",
            "label": 0,
            "center_source": "catalog-offset",
        },
    ]
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_predictions(
            y_true=labels,
            logits=logits,
            metadata_rows=rows,
            output_dir=out_dir,
        )


def test_cli_review_queue_bokeh_and_apply(tmp_path, capsys) -> None:
    dataset_dir, run_dir, compare_dir = write_review_fixture(tmp_path)
    review_dir = tmp_path / "review"

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--compare-run-dirs",
            str(compare_dir),
            "--max-items",
            "4",
        ]
    )
    captured = capsys.readouterr()
    queue_summary = json.loads(captured.out)

    assert rc == 0
    assert queue_summary["queue_count"] == 4
    assert (review_dir / "manifest.json").exists()
    assert (review_dir / "queue.jsonl").exists()
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["rank_reason"] for row in queued}
    assert any(row["known_error"] == 1 for row in queued)

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "not empty" in captured.err

    rc = _run_cli(
        [
            "review-bokeh",
            "--review-dir",
            str(review_dir),
            "--show-url-only",
        ]
    )
    captured = capsys.readouterr()
    bokeh_summary = json.loads(captured.out)
    assert rc == 0
    assert bokeh_summary["url"] == "http://localhost:5006/"

    rc = _run_cli(
        [
            "review-bokeh",
            "--review-dir",
            str(review_dir),
            "--show-url-only",
            "--port",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1

    rc = _run_cli(
        [
            "review-bokeh",
            "--review-dir",
            str(review_dir),
            "--show-url-only",
            "--port",
            "5007",
            "--host",
            "0.0.0.0",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "host must be loopback-only" in captured.err

    rc = _run_cli(
        [
            "review-bokeh",
            "--review-dir",
            str(review_dir),
            "--show-url-only",
            "--port",
            "5007",
        ]
    )
    captured = capsys.readouterr()
    bokeh_summary = json.loads(captured.out)
    assert rc == 0
    assert bokeh_summary["server_started"] is False
    assert bokeh_summary["url"] == "http://localhost:5007/"

    rc = _run_cli(
        [
            "review-bokeh",
            "--review-dir",
            str(review_dir),
            "--show-url-only",
            "--port",
            "5008",
            "--host",
            "::1",
        ]
    )
    captured = capsys.readouterr()
    bokeh_summary = json.loads(captured.out)
    assert rc == 0
    assert bokeh_summary["url"] == "http://[::1]:5008/"

    legacy_target = queued[1]
    legacy_annotation = {
        "queue_id": legacy_target["queue_id"],
        "sample_index": legacy_target["sample_index"],
        "candidate_id": legacy_target["candidate_id"],
        "reviewer_label": "bogus",
        "source_label": legacy_target["label"],
        "source_probability": legacy_target["probability"],
    }
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(legacy_annotation) + "\n",
        encoding="utf-8",
    )
    _append_annotation(
        review_dir,
        {
            "timestamp_utc": "2026-05-10T19:59:00+00:00",
            "queue_id": queued[2]["queue_id"],
            "sample_index": queued[2]["sample_index"],
            "candidate_id": queued[2]["candidate_id"],
            "reviewer_label": "real",
            "reviewer": "fixture-reviewer",
            "morphology_tags": [],
            "notes": "",
            "source_label": queued[2]["label"],
            "source_probability": queued[2]["probability"],
        },
    )
    review_state = json.loads(
        (review_dir / "review_state.json").read_text(encoding="utf-8")
    )
    assert review_state["reviewed_count"] == 2
    assert review_state["reviewer_decision_count"] == 2

    target = queued[0]
    annotation = {
        "timestamp_utc": "2026-05-10T20:00:00+00:00",
        "queue_id": target["queue_id"],
        "sample_index": target["sample_index"],
        "candidate_id": target["candidate_id"],
        "reviewer_label": "real",
        "reviewer": "fixture-reviewer",
        "morphology_tags": ["point_source_round"],
        "notes": "fixture annotation",
        "source_label": target["label"],
        "source_probability": target["probability"],
    }
    unsure_annotation = {
        **annotation,
        "timestamp_utc": "2026-05-10T20:01:00+00:00",
        "reviewer_label": "unsure",
    }
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(annotation) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reviewed-dataset"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    apply_summary = json.loads(captured.out)
    assert rc == 0
    assert apply_summary["applied_label_count"] == 1
    labels = np.load(output_dir / "labels.npy")
    assert labels[int(target["sample_index"])] == 1
    rows = load_metadata_rows(output_dir)
    row = rows[int(target["sample_index"])]
    assert row["label_source"] == "human_review"
    assert row["original_label_source"] == "fixture"
    assert row["target_label_available"] is True
    assert row["review_label"] == "real"
    assert row["review_morphology_tags"] == ["point_source_round"]
    assert (dataset_dir / "search.npy").stat().st_ino != (
        output_dir / "search.npy"
    ).stat().st_ino

    latest_unsure_annotation = {
        **annotation,
        "timestamp_utc": "2026-05-10T20:02:00+00:00",
        "reviewer_label": "unsure",
    }
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(annotation)
        + "\n"
        + json.dumps(latest_unsure_annotation)
        + "\n",
        encoding="utf-8",
    )
    latest_unsure_output = tmp_path / "latest-unsure-reviewed-dataset"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(latest_unsure_output),
        ]
    )
    captured = capsys.readouterr()
    latest_unsure_summary = json.loads(captured.out)
    assert rc == 0
    assert latest_unsure_summary["applied_label_count"] == 0
    assert latest_unsure_summary["skipped_unsure_count"] == 1
    latest_unsure_labels = np.load(latest_unsure_output / "labels.npy")
    original_labels = np.load(dataset_dir / "labels.npy")
    assert (
        latest_unsure_labels[int(target["sample_index"])]
        == (original_labels[int(target["sample_index"])])
    )

    legacy_annotation = dict(annotation)
    legacy_annotation.pop("reviewer")
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(legacy_annotation) + "\n",
        encoding="utf-8",
    )
    legacy_rejected_output = tmp_path / "legacy-review-apply-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(legacy_rejected_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "must include reviewer" in captured.err
    assert "re-recorded or migrated" in captured.err
    assert not legacy_rejected_output.exists()

    empty_reviewer_annotation = {**annotation, "reviewer": "   "}
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(empty_reviewer_annotation) + "\n",
        encoding="utf-8",
    )
    empty_reviewer_output = tmp_path / "empty-reviewer-apply-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(empty_reviewer_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "must include reviewer" in captured.err
    assert not empty_reviewer_output.exists()

    mixed_reviewer_annotations = [
        dict(annotation),
        {
            **annotation,
            "timestamp_utc": "2026-05-10T20:02:00+00:00",
            "reviewer": "bob",
            "reviewer_label": "unsure",
        },
    ]
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in mixed_reviewer_annotations)
        + "\n",
        encoding="utf-8",
    )
    rejected_unsure_output = tmp_path / "unsure-multi-reviewer-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(rejected_unsure_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "multiple reviewers" in captured.err
    assert not rejected_unsure_output.exists()

    (review_dir / "annotations.jsonl").write_text(
        json.dumps(annotation) + "\n" + json.dumps(unsure_annotation) + "\n",
        encoding="utf-8",
    )

    other_root = tmp_path / "other-review-root"
    other_root.mkdir()
    other_dataset_dir, _other_run_dir, _other_compare_dir = (
        write_review_fixture(other_root)
    )
    rejected_output = tmp_path / "rejected-review-apply"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(other_dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(rejected_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "does not match requested dataset_dir" in captured.err
    assert not rejected_output.exists()

    stray_annotation = {
        **annotation,
        "queue_id": "not-in-queue",
        "sample_index": 0,
        "reviewer_label": "bogus",
    }
    with (review_dir / "annotations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(stray_annotation) + "\n")
    rejected_stray_output = tmp_path / "rejected-stray-review-apply"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(rejected_stray_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "not present in queue.jsonl" in captured.err
    assert not rejected_stray_output.exists()


def test_cli_review_queue_dataset_builds_model_free_queue(
    tmp_path,
    capsys,
) -> None:
    dataset_dir = write_lsstcomcam_placeholder_dataset(tmp_path)
    review_dir = tmp_path / "dataset-review"

    rc = _run_cli(
        [
            "review-queue-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(review_dir),
            "--max-items",
            "3",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["workflow"] == "review-queue-dataset"
    assert payload["run_dir"] is None
    assert payload["split"] == "all"
    assert payload["candidate_count"] == 4
    assert payload["queue_count"] == 3
    assert payload["dataset_validation"]["valid"] is True
    assert (review_dir / "manifest.json").exists()
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(queued) == 3
    assert {row["review_input_source"] for row in queued} == {"dataset_only"}
    assert {row["probability"] for row in queued} == {0.5}
    assert {row["rank_reason"] for row in queued} == {"dataset_audit"}
    assert all(row["queue_id"].startswith("sample:") for row in queued)
    assert all(
        row["label_source"] == "unlabeled_lsstcomcam_smoke_placeholder"
        for row in queued
    )

    rc = _run_cli(
        [
            "review-bokeh",
            "--review-dir",
            str(review_dir),
            "--show-url-only",
        ]
    )
    captured = capsys.readouterr()
    bokeh_summary = json.loads(captured.out)
    assert rc == 0
    assert bokeh_summary["queue_count"] == 3

    val_review_dir = tmp_path / "dataset-review-val"
    rc = _run_cli(
        [
            "review-queue-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "val",
            "--output-dir",
            str(val_review_dir),
        ]
    )
    captured = capsys.readouterr()
    val_payload = json.loads(captured.out)
    assert rc == 0
    assert val_payload["queue_count"] == 2
    val_queued = [
        json.loads(line)
        for line in (val_review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["split"] for row in val_queued} == {"val"}

    rc = _run_cli(
        [
            "review-queue-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "not empty" in captured.err


def test_cli_review_contact_sheet_exports_static_pages(
    tmp_path,
    capsys,
) -> None:
    image = pytest.importorskip("PIL.Image")
    dataset_dir = write_lsstcomcam_placeholder_dataset(tmp_path)
    review_dir = tmp_path / "dataset-review"
    rc = _run_cli(
        [
            "review-queue-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(review_dir),
            "--max-items",
            "3",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    output_dir = tmp_path / "contact-sheets"
    rc = _run_cli(
        [
            "review-contact-sheet",
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(output_dir),
            "--max-items",
            "3",
            "--items-per-page",
            "2",
            "--columns",
            "2",
            "--stamp-size",
            "32",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["workflow"] == "review-contact-sheet"
    assert payload["queue_count"] == 3
    assert payload["exported_count"] == 3
    assert payload["page_count"] == 2
    assert payload["items"][0]["candidate_id"].startswith("lsst-placeholder-")
    assert payload["items"][0]["label_source"] == (
        "unlabeled_lsstcomcam_smoke_placeholder"
    )
    assert (output_dir / "index.json").exists()
    for page in payload["saved"]["pages"]:
        page_path = output_dir / page
        assert page_path.exists()
        with image.open(page_path) as opened:
            assert opened.width > 0
            assert opened.height > 0

    rc = _run_cli(
        [
            "review-contact-sheet",
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "not empty" in captured.err

    rc = _run_cli(
        [
            "review-contact-sheet",
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(output_dir),
            "--max-items",
            "1",
            "--overwrite",
        ]
    )
    captured = capsys.readouterr()
    overwrite_payload = json.loads(captured.out)
    assert rc == 0
    assert overwrite_payload["exported_count"] == 1
    assert len(overwrite_payload["saved"]["pages"]) == 1
    assert not (output_dir / "contact-sheet-002.png").exists()


def test_cli_review_annotation_template_imports_offline_csv(
    tmp_path,
    capsys,
) -> None:
    dataset_dir = write_lsstcomcam_placeholder_dataset(tmp_path)
    review_dir = tmp_path / "dataset-review"
    rc = _run_cli(
        [
            "review-queue-dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(review_dir),
            "--max-items",
            "3",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    template_csv = tmp_path / "annotations-template.csv"
    rc = _run_cli(
        [
            "review-annotation-template",
            "--review-dir",
            str(review_dir),
            "--output-csv",
            str(template_csv),
            "--reviewer",
            "alice",
        ]
    )
    captured = capsys.readouterr()
    template_payload = json.loads(captured.out)
    assert rc == 0
    assert template_payload["workflow"] == "review-annotation-template"
    assert template_payload["queue_count"] == 3
    assert template_payload["reviewer"] == "alice"

    with template_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    rows[0]["reviewer_label"] = "real"
    rows[0]["morphology_tags"] = "point_source_round"
    rows[0]["notes"] = "compact centered source"
    rows[1]["reviewer_label"] = "bogus"
    rows[1]["notes"] = "residual artifact"
    rows[2]["reviewer_label"] = "real"
    filled_csv = tmp_path / "annotations-filled.csv"
    with filled_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    rc = _run_cli(
        [
            "review-import-annotations",
            "--review-dir",
            str(review_dir),
            "--input-csv",
            str(filled_csv),
            "--dry-run",
            "--require-all",
        ]
    )
    captured = capsys.readouterr()
    dry_run_payload = json.loads(captured.out)
    assert rc == 0
    assert dry_run_payload["validated_count"] == 3
    assert dry_run_payload["appended_count"] == 0
    assert not (review_dir / "annotations.jsonl").exists()

    rc = _run_cli(
        [
            "review-import-annotations",
            "--review-dir",
            str(review_dir),
            "--input-csv",
            str(filled_csv),
            "--require-all",
        ]
    )
    captured = capsys.readouterr()
    import_payload = json.loads(captured.out)
    assert rc == 0
    assert import_payload["workflow"] == "review-import-annotations"
    assert import_payload["appended_count"] == 3
    assert import_payload["reviewer_label_counts"] == {"real": 2, "bogus": 1}
    annotations = [
        json.loads(line)
        for line in (review_dir / "annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["reviewer"] for row in annotations} == {"alice"}
    assert annotations[0]["morphology_tags"] == ["point_source_round"]

    rc = _run_cli(
        [
            "review-status",
            "--review-dir",
            str(review_dir),
            "--min-reviewers",
            "1",
            "--min-actionable-reviewers",
            "1",
            "--require-ready",
        ]
    )
    captured = capsys.readouterr()
    status_payload = json.loads(captured.out)
    assert rc == 0
    assert status_payload["ready_for_review_apply"] is True


def test_cli_review_queue_splits_preserves_positive_quality_strata(
    tmp_path,
    capsys,
) -> None:
    dataset_dir, run_dir = write_review_split_fixture(tmp_path)
    output_root = tmp_path / "review-splits"

    rc = _run_cli(
        [
            "review-queue-splits",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--output-root",
            str(output_root),
            "--splits",
            "train,val,test",
            "--max-items",
            "2",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["workflow"] == "review-queue-splits"
    assert payload["splits"] == ["train", "val", "test"]
    assert set(payload["split_results"]) == {"train", "val", "test"}
    queued_rows = []
    for split in ("train", "val", "test"):
        review_dir = output_root / f"{split}-hybrid-2"
        assert (review_dir / "manifest.json").exists()
        rows = [
            json.loads(line)
            for line in (review_dir / "queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == 2
        queued_rows.extend(rows)
    assert any(
        "positive:weak_snr" in row["review_stratum"] for row in queued_rows
    )


def test_cli_review_status_reports_missing_and_ready(
    tmp_path,
    capsys,
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    review_dir = tmp_path / "status-review"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "1",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = _run_cli(
        [
            "review-status",
            "--review-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["workflow"] == "review-status"
    assert payload["ready_for_review_apply"] is False
    assert payload["annotation_file"] is None
    assert payload["missing_annotation_count"] == 1
    assert "annotations.jsonl is missing" in payload["blockers"]
    assert "decisions" not in payload["aggregation"]

    rc = _run_cli(
        [
            "review-status",
            "--review-dir",
            str(review_dir),
            "--require-ready",
        ]
    )
    captured = capsys.readouterr()
    require_payload = json.loads(captured.out)
    assert rc == 1
    assert require_payload["ready_for_review_apply"] is False
    assert "not ready for review-apply" in captured.err

    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    item = queued[0]
    annotation = {
        "timestamp_utc": "2026-05-21T19:00:00+00:00",
        "queue_id": item["queue_id"],
        "sample_index": item["sample_index"],
        "candidate_id": item["candidate_id"],
        "reviewer": "alice",
        "reviewer_label": "real",
        "morphology_tags": ["point_source_round"],
        "notes": "centered compact residual",
        "source_label": item["label"],
        "source_probability": item["probability"],
    }
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(annotation) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-status",
            "--review-dir",
            str(review_dir),
            "--min-reviewers",
            "1",
            "--min-actionable-reviewers",
            "1",
            "--include-decisions",
            "--require-ready",
        ]
    )
    captured = capsys.readouterr()
    ready_payload = json.loads(captured.out)
    assert rc == 0
    assert ready_payload["ready_for_review_apply"] is True
    assert ready_payload["blockers"] == []
    assert ready_payload["status_counts"]["actionable"] == 1
    assert ready_payload["actionable_label_counts"]["real"] == 1
    assert ready_payload["aggregation"]["decisions"][0]["status"] == (
        "actionable"
    )


def test_cli_review_aggregate_report_and_apply(tmp_path, capsys) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    review_dir = tmp_path / "aggregate-review"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "4",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    def annotation(
        item: dict[str, object],
        *,
        reviewer: str,
        label: str,
        minute: int,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, object]:
        return {
            "timestamp_utc": f"2026-05-10T20:{minute:02d}:00+00:00",
            "queue_id": item["queue_id"],
            "sample_index": item["sample_index"],
            "candidate_id": item["candidate_id"],
            "reviewer": reviewer,
            "reviewer_label": label,
            "morphology_tags": tags or [],
            "notes": notes,
            "source_label": item["label"],
            "source_probability": item["probability"],
        }

    annotations = [
        annotation(queued[0], reviewer="alice", label="bogus", minute=0),
        annotation(
            queued[0],
            reviewer="alice",
            label="real",
            minute=1,
            tags=["point_source_round"],
            notes="alice latest",
        ),
        annotation(
            queued[0],
            reviewer="bob",
            label="real",
            minute=2,
            tags=["round_extended"],
            notes="bob agrees",
        ),
        annotation(queued[1], reviewer="alice", label="real", minute=3),
        annotation(queued[1], reviewer="bob", label="bogus", minute=4),
        annotation(queued[1], reviewer="carol", label="real", minute=5),
        annotation(queued[2], reviewer="alice", label="unsure", minute=5),
        annotation(queued[2], reviewer="bob", label="unsure", minute=6),
        annotation(queued[3], reviewer="alice", label="real", minute=7),
        annotation(queued[3], reviewer="bob", label="unsure", minute=8),
    ]
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n",
        encoding="utf-8",
    )

    unknown_annotations = [dict(annotations[1])]
    unknown_annotations[0]["queue_id"] = "not-in-queue"
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in unknown_annotations) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "not present in queue.jsonl" in captured.err
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n",
        encoding="utf-8",
    )

    mismatched_annotations = [dict(annotations[1]), dict(annotations[2])]
    mismatched_annotations[0]["sample_index"] = queued[1]["sample_index"]
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in mismatched_annotations) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "sample_index does not match queued item" in captured.err
    assert "annotations.jsonl:1" in captured.err
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n",
        encoding="utf-8",
    )

    report_path = tmp_path / "aggregation-report.json"
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    dry_run_summary = json.loads(captured.out)
    assert rc == 0
    assert not report_path.exists()
    assert dry_run_summary["status_counts"] == {
        "actionable": 1,
        "conflicted": 1,
        "unsure_only": 1,
        "no_actionable": 0,
        "insufficient_review": 1,
    }

    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
            "--output-report",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()
    report_summary = json.loads(captured.out)
    assert rc == 0
    assert report_path.exists()
    assert (
        report_summary["aggregation_rule_version"] == "review-aggregation-v2"
    )
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_report == report_summary
    assert persisted_report["queue_file"]["line_count"] == len(queued)
    actionable = [
        decision
        for decision in report_summary["decisions"]
        if decision["status"] == "actionable"
    ]
    assert len(actionable) == 1
    assert actionable[0]["queue_id"] == queued[0]["queue_id"]
    assert actionable[0]["consensus_label"] == "real"
    assert actionable[0]["supporting_reviewers"] == ["alice", "bob"]
    assert actionable[0]["consensus_morphology_tags"] == [
        "point_source_round",
        "round_extended",
    ]
    alice_latest = [
        row
        for row in actionable[0]["latest_annotations"]
        if row["reviewer"] == "alice"
    ][0]
    assert alice_latest["reviewer_label"] == "real"
    assert alice_latest["notes"] == "alice latest"

    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
            "--consensus-rule",
            "majority",
        ]
    )
    captured = capsys.readouterr()
    majority_summary = json.loads(captured.out)
    assert rc == 0
    assert majority_summary["status_counts"] == {
        "actionable": 2,
        "conflicted": 0,
        "unsure_only": 1,
        "no_actionable": 0,
        "insufficient_review": 1,
    }
    majority_decision = [
        decision
        for decision in majority_summary["decisions"]
        if decision["queue_id"] == queued[1]["queue_id"]
    ][0]
    assert majority_decision["consensus_label"] == "real"
    assert majority_decision["supporting_reviewers"] == ["alice", "carol"]

    rejected_output = tmp_path / "aggregate-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(rejected_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "multiple reviewers" in captured.err
    assert not rejected_output.exists()

    original_labels = np.load(dataset_dir / "labels.npy")
    output_dir = tmp_path / "aggregate-reviewed-dataset"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--aggregation-report",
            str(report_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    apply_summary = json.loads(captured.out)
    assert rc == 0
    assert apply_summary["review_apply_source"] == "aggregation_report"
    assert apply_summary["applied_label_count"] == 1
    assert apply_summary["skipped_conflicted_count"] == 1
    assert apply_summary["skipped_unsure_count"] == 1
    assert apply_summary["skipped_insufficient_review_count"] == 1
    labels = np.load(output_dir / "labels.npy")
    assert labels[int(queued[0]["sample_index"])] == 1
    for item in queued[1:]:
        sample_index = int(item["sample_index"])
        assert labels[sample_index] == original_labels[sample_index]
    rows = load_metadata_rows(output_dir)
    row = rows[int(queued[0]["sample_index"])]
    assert row["label_source"] == "human_review_aggregation"
    assert row["original_label_source"] == "fixture"
    assert row["target_label_available"] is True
    assert row["review_label"] == "real"
    assert row["review_timestamp_utc"] == "2026-05-10T20:02:00+00:00"
    assert row["review_morphology_tags"] == [
        "point_source_round",
        "round_extended",
    ]
    assert row["review_notes"] == "alice: alice latest\nbob: bob agrees"
    assert row["review_aggregation_status"] == "actionable"
    assert row["review_aggregation_consensus_rule"] == "unanimous"
    assert row["review_aggregation_supporting_reviewers"] == [
        "alice",
        "bob",
    ]
    assert row["review_aggregation_label_counts"] == {
        "real": 2,
        "bogus": 0,
        "unsure": 0,
        "invalid": 0,
    }
    assert row["review_aggregation_latest_annotations"][0]["reviewer"] == (
        "alice"
    )

    forward_compatible_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    forward_compatible_report["annotation_file"][
        "future_descriptive_field"
    ] = "ignored"
    forward_compatible_report_path = tmp_path / (
        "aggregation-report-extra-fingerprint.json"
    )
    forward_compatible_report_path.write_text(
        json.dumps(forward_compatible_report) + "\n",
        encoding="utf-8",
    )
    compatible_output = tmp_path / "aggregate-compatible-report"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--aggregation-report",
            str(forward_compatible_report_path),
            "--output-dir",
            str(compatible_output),
        ]
    )
    captured = capsys.readouterr()
    compatible_summary = json.loads(captured.out)
    assert rc == 0
    assert compatible_summary["applied_label_count"] == 1

    tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered_report["decisions"][0]["consensus_label"] = "bogus"
    tampered_report_path = tmp_path / "aggregation-report-tampered.json"
    tampered_report_path.write_text(
        json.dumps(tampered_report) + "\n",
        encoding="utf-8",
    )
    tampered_report_output = tmp_path / "aggregate-tampered-report-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--aggregation-report",
            str(tampered_report_path),
            "--output-dir",
            str(tampered_report_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "does not match current annotations.jsonl" in captured.err
    assert f"queue_id={queued[0]['queue_id']}" in captured.err
    assert not tampered_report_output.exists()

    list_report_path = tmp_path / "aggregation-report-list.json"
    list_report_path.write_text("[]\n", encoding="utf-8")
    list_report_output = tmp_path / "aggregate-list-report-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--aggregation-report",
            str(list_report_path),
            "--output-dir",
            str(list_report_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "aggregation report must be a JSON object" in captured.err

    null_decision_report = json.loads(report_path.read_text(encoding="utf-8"))
    null_decision_report["decisions"] = [None]
    null_decision_report_path = tmp_path / (
        "aggregation-report-null-decision.json"
    )
    null_decision_report_path.write_text(
        json.dumps(null_decision_report) + "\n",
        encoding="utf-8",
    )
    null_decision_output = tmp_path / "aggregate-null-decision-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--aggregation-report",
            str(null_decision_report_path),
            "--output-dir",
            str(null_decision_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "decisions must be JSON objects" in captured.err

    def assert_report_rejected(
        report_payload: dict[str, object],
        *,
        slug: str,
        expected_error: str,
    ) -> None:
        rejected_report_path = tmp_path / f"aggregation-report-{slug}.json"
        rejected_report_path.write_text(
            json.dumps(report_payload) + "\n",
            encoding="utf-8",
        )
        rejected_output = tmp_path / f"aggregate-{slug}-rejected"
        rc = _run_cli(
            [
                "review-apply",
                "--dataset-dir",
                str(dataset_dir),
                "--review-dir",
                str(review_dir),
                "--aggregation-report",
                str(rejected_report_path),
                "--output-dir",
                str(rejected_output),
            ]
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert expected_error in captured.err
        assert not rejected_output.exists()

    stale_queue_report = json.loads(report_path.read_text(encoding="utf-8"))
    queue_path = review_dir / "queue.jsonl"
    original_queue_text = queue_path.read_text(encoding="utf-8")
    try:
        queue_path.write_text(
            original_queue_text.replace(
                str(queued[0]["queue_id"]),
                f"{queued[0]['queue_id']}-tampered",
                1,
            ),
            encoding="utf-8",
        )
        assert_report_rejected(
            stale_queue_report,
            slug="stale-queue",
            expected_error="current queue.jsonl",
        )
    finally:
        queue_path.write_text(original_queue_text, encoding="utf-8")

    bad_min_reviewers = json.loads(report_path.read_text(encoding="utf-8"))
    bad_min_reviewers["min_reviewers"] = 0
    assert_report_rejected(
        bad_min_reviewers,
        slug="bad-min-reviewers",
        expected_error="min_reviewers must be a positive integer",
    )

    bad_min_reviewers_bool = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    bad_min_reviewers_bool["min_reviewers"] = True
    assert_report_rejected(
        bad_min_reviewers_bool,
        slug="bool-min-reviewers",
        expected_error="min_reviewers must be a positive integer",
    )

    bad_min_reviewers_float = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    bad_min_reviewers_float["min_reviewers"] = 1.5
    assert_report_rejected(
        bad_min_reviewers_float,
        slug="float-min-reviewers",
        expected_error="min_reviewers must be a positive integer",
    )

    missing_consensus_rule = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    missing_consensus_rule.pop("consensus_rule")
    assert_report_rejected(
        missing_consensus_rule,
        slug="missing-consensus-rule",
        expected_error="must include consensus_rule",
    )

    stale_rule_version = json.loads(report_path.read_text(encoding="utf-8"))
    stale_rule_version["aggregation_rule_version"] = "review-aggregation-v1"
    assert_report_rejected(
        stale_rule_version,
        slug="stale-rule-version",
        expected_error="aggregation_rule_version is stale",
    )

    stale_latest_rule = json.loads(report_path.read_text(encoding="utf-8"))
    stale_latest_rule["latest_per_reviewer_rule"] = "old-rule"
    assert_report_rejected(
        stale_latest_rule,
        slug="stale-latest-rule",
        expected_error="latest_per_reviewer_rule is stale",
    )

    bad_min_actionable = json.loads(report_path.read_text(encoding="utf-8"))
    bad_min_actionable["min_actionable_reviewers"] = 3
    assert_report_rejected(
        bad_min_actionable,
        slug="bad-min-actionable",
        expected_error="min_actionable_reviewers cannot exceed",
    )

    missing_fingerprint = json.loads(report_path.read_text(encoding="utf-8"))
    missing_fingerprint["annotation_file"].pop("sha256")
    assert_report_rejected(
        missing_fingerprint,
        slug="missing-fingerprint",
        expected_error="annotation fingerprint missing content fields",
    )

    bad_fingerprint_shape = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    bad_fingerprint_shape["annotation_file"] = "bad"
    assert_report_rejected(
        bad_fingerprint_shape,
        slug="bad-fingerprint-shape",
        expected_error="annotation fingerprint must be a JSON object",
    )

    bad_status_counts = json.loads(report_path.read_text(encoding="utf-8"))
    bad_status_counts["status_counts"]["actionable"] = 99
    assert_report_rejected(
        bad_status_counts,
        slug="bad-status-counts",
        expected_error="status_counts do not match",
    )

    bad_actionable_counts = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    bad_actionable_counts["actionable_label_counts"]["real"] = 99
    assert_report_rejected(
        bad_actionable_counts,
        slug="bad-actionable-counts",
        expected_error="actionable_label_counts do not match",
    )

    duplicate_decision = json.loads(report_path.read_text(encoding="utf-8"))
    duplicate_decision["decisions"].append(
        dict(duplicate_decision["decisions"][0])
    )
    assert_report_rejected(
        duplicate_decision,
        slug="duplicate-decision",
        expected_error="duplicate decision",
    )

    missing_queue_id = json.loads(report_path.read_text(encoding="utf-8"))
    missing_queue_id["decisions"][0].pop("queue_id")
    assert_report_rejected(
        missing_queue_id,
        slug="missing-queue-id",
        expected_error="missing required field 'queue_id'",
    )

    missing_status = json.loads(report_path.read_text(encoding="utf-8"))
    missing_status["decisions"][0].pop("status")
    assert_report_rejected(
        missing_status,
        slug="missing-status",
        expected_error="missing required field 'status'",
    )

    reordered_support = json.loads(report_path.read_text(encoding="utf-8"))
    reordered_support["decisions"][0]["supporting_reviewers"] = [
        "bob",
        "alice",
    ]
    assert_report_rejected(
        reordered_support,
        slug="reordered-supporting-reviewers",
        expected_error="does not match current annotations.jsonl",
    )

    with (review_dir / "annotations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                annotation(
                    queued[0],
                    reviewer="dana",
                    label="real",
                    minute=9,
                )
            )
            + "\n"
        )
    stale_output = tmp_path / "aggregate-stale-rejected"
    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--aggregation-report",
            str(report_path),
            "--output-dir",
            str(stale_output),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "rerun review-aggregate" in captured.err
    assert not stale_output.exists()

    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
            "--min-reviewers",
            "1",
            "--min-actionable-reviewers",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "min_actionable_reviewers cannot exceed" in captured.err

    invalid_annotations = [
        annotation(
            queued[0],
            reviewer="alice",
            label="banana",
            minute=10,
        ),
        annotation(
            queued[0],
            reviewer="bob",
            label="waffle",
            minute=11,
        ),
    ]
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in invalid_annotations) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
            "--min-reviewers",
            "2",
            "--min-actionable-reviewers",
            "1",
        ]
    )
    captured = capsys.readouterr()
    invalid_summary = json.loads(captured.out)
    invalid_decision = [
        decision
        for decision in invalid_summary["decisions"]
        if decision["queue_id"] == queued[0]["queue_id"]
    ][0]
    assert rc == 0
    assert invalid_decision["status"] == "insufficient_review"
    assert invalid_decision["reason"] == "not_enough_valid_reviewers"

    mixed_invalid_annotations = [
        annotation(
            queued[0],
            reviewer="alice",
            label="real",
            minute=10,
        ),
        annotation(
            queued[0],
            reviewer="bob",
            label="waffle",
            minute=11,
        ),
    ]
    (review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in mixed_invalid_annotations)
        + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
            "--min-reviewers",
            "2",
            "--min-actionable-reviewers",
            "1",
        ]
    )
    captured = capsys.readouterr()
    mixed_invalid_summary = json.loads(captured.out)
    mixed_invalid_decision = [
        decision
        for decision in mixed_invalid_summary["decisions"]
        if decision["queue_id"] == queued[0]["queue_id"]
    ][0]
    assert rc == 0
    assert mixed_invalid_decision["status"] == "insufficient_review"
    assert mixed_invalid_decision["reason"] == "not_enough_valid_reviewers"

    missing_timestamp = [dict(invalid_annotations[0])]
    missing_timestamp[0].pop("timestamp_utc")
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(missing_timestamp[0]) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "must include timestamp_utc" in captured.err
    assert "annotations.jsonl:1" in captured.err

    malformed_timestamp = [dict(invalid_annotations[0])]
    malformed_timestamp[0]["timestamp_utc"] = "not-a-timestamp"
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(malformed_timestamp[0]) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "timestamp_utc must be parseable" in captured.err
    assert "annotations.jsonl:1" in captured.err

    missing_reviewer = [dict(invalid_annotations[0])]
    missing_reviewer[0].pop("reviewer")
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(missing_reviewer[0]) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "review-aggregate",
            "--review-dir",
            str(review_dir),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "must include reviewer" in captured.err
    assert "annotations.jsonl:1" in captured.err


def test_cli_entity_review_queue_bokeh_and_aggregate(
    tmp_path,
    capsys,
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    source_review_dir = tmp_path / "source-review"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(source_review_dir),
            "--max-items",
            "4",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    queued = [
        json.loads(line)
        for line in (source_review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    source_annotations = [
        {
            "timestamp_utc": "2026-05-13T01:00:00+00:00",
            "reviewer": "alice",
            "queue_id": queued[0]["queue_id"],
            "sample_index": queued[0]["sample_index"],
            "candidate_id": queued[0]["candidate_id"],
            "reviewer_label": "real",
            "morphology_tags": ["point_source_round"],
            "notes": "compact point source",
            "source_label": queued[0]["label"],
            "source_probability": queued[0]["probability"],
        },
        {
            "timestamp_utc": "2026-05-13T01:01:00+00:00",
            "reviewer": "alice",
            "queue_id": queued[1]["queue_id"],
            "sample_index": queued[1]["sample_index"],
            "candidate_id": queued[1]["candidate_id"],
            "reviewer_label": "bogus",
            "morphology_tags": ["noise_artifact"],
            "notes": "fixture bogus",
            "source_label": queued[1]["label"],
            "source_probability": queued[1]["probability"],
        },
        {
            "timestamp_utc": "2026-05-13T01:02:00+00:00",
            "reviewer": "bob",
            "queue_id": queued[2]["queue_id"],
            "sample_index": queued[2]["sample_index"],
            "candidate_id": queued[2]["candidate_id"],
            "reviewer_label": "real",
            "morphology_tags": ["point_source_round", "unclear"],
            "notes": "maybe point source",
            "source_label": queued[2]["label"],
            "source_probability": queued[2]["probability"],
        },
    ]
    (source_review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in source_annotations) + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "entity-review-queue",
            "--source-review-dirs",
            f"{source_review_dir},{source_review_dir}",
            "--output-dir",
            str(tmp_path / "entity-review-duplicate-source"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "source_review_dirs must be unique" in captured.err

    entity_dir = tmp_path / "entity-review"
    rc = _run_cli(
        [
            "entity-review-queue",
            "--source-review-dirs",
            str(source_review_dir),
            "--output-dir",
            str(entity_dir),
        ]
    )
    captured = capsys.readouterr()
    queue_summary = json.loads(captured.out)
    assert rc == 0
    assert queue_summary["queue_count"] == 2
    assert queue_summary["selected_binary_label"] == "real"
    assert queue_summary["queue_identity_rule"] == (
        "one_entity_item_per_latest_binary_real_annotation"
    )
    assert queue_summary["unique_source_queue_count"] == 2
    assert queue_summary["multi_reviewer_fanout_count"] == 0
    assert (entity_dir / "dataset" / "search.npy").exists()
    entity_queued = [
        json.loads(line)
        for line in (entity_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["sample_index"] for row in entity_queued] == [0, 1]
    assert entity_queued[0]["source_queue_id"] == queued[0]["queue_id"]
    assert entity_queued[1]["source_queue_id"] == queued[2]["queue_id"]
    assert [row["label"] for row in entity_queued] == [1, 1]
    assert entity_queued[0]["source_label"] == queued[0]["label"]
    source_search = np.load(dataset_dir / "search.npy")
    entity_search = np.load(entity_dir / "dataset" / "search.npy")
    entity_labels = np.load(entity_dir / "dataset" / "labels.npy")
    assert entity_labels.tolist() == [1, 1]
    np.testing.assert_allclose(
        entity_search[0],
        source_search[int(queued[0]["sample_index"])],
    )
    np.testing.assert_allclose(
        entity_search[1],
        source_search[int(queued[2]["sample_index"])],
    )
    rows = load_metadata_rows(entity_dir / "dataset")
    assert rows[0]["entity_source_queue_id"] == queued[0]["queue_id"]
    assert rows[0]["entity_source_reviewer"] == "alice"
    assert rows[0]["label"] == 1
    assert rows[0]["entity_source_original_label"] == queued[0]["label"]
    assert rows[0]["entity_source_binary_label_value"] == 1
    assert validate_dataset_dir(entity_dir / "dataset")["valid"] is True

    rc = _run_cli(
        [
            "entity-review-bokeh",
            "--review-dir",
            str(entity_dir),
            "--show-url-only",
            "--port",
            "5010",
        ]
    )
    captured = capsys.readouterr()
    bokeh_summary = json.loads(captured.out)
    assert rc == 0
    assert bokeh_summary["workflow"] == "entity-review-bokeh"
    assert bokeh_summary["url"] == "http://localhost:5010/"

    rc = _run_cli(
        [
            "entity-review-bokeh",
            "--review-dir",
            str(entity_dir),
            "--show-url-only",
        ]
    )
    captured = capsys.readouterr()
    bokeh_summary = json.loads(captured.out)
    assert rc == 0
    assert bokeh_summary["url"] == "http://localhost:5007/"

    rc = _run_cli(
        [
            "entity-review-bokeh",
            "--review-dir",
            str(entity_dir),
            "--show-url-only",
            "--port",
            "5011",
            "--host",
            "::1",
        ]
    )
    captured = capsys.readouterr()
    bokeh_summary = json.loads(captured.out)
    assert rc == 0
    assert bokeh_summary["url"] == "http://[::1]:5011/"

    entity_annotations = [
        {
            "timestamp_utc": "2026-05-13T02:00:00+00:00",
            "reviewer": "gpt-5.5-xhigh",
            "queue_id": entity_queued[0]["queue_id"],
            "sample_index": entity_queued[0]["sample_index"],
            "candidate_id": entity_queued[0]["candidate_id"],
            "entity_label": "point_source_star_or_planet",
            "confidence": "high",
            "notes": "compact point-source fixture",
        },
        {
            "timestamp_utc": "2026-05-13T02:01:00+00:00",
            "reviewer": "gpt-5.5-xhigh",
            "queue_id": entity_queued[1]["queue_id"],
            "sample_index": entity_queued[1]["sample_index"],
            "candidate_id": entity_queued[1]["candidate_id"],
            "entity_label": "other_or_unsure",
            "confidence": "low",
            "notes": "fixture unsure",
        },
    ]
    mismatched_entity_annotations = [dict(entity_annotations[0])]
    mismatched_entity_annotations[0]["sample_index"] = entity_queued[1][
        "sample_index"
    ]
    (entity_dir / "entity_annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in mismatched_entity_annotations)
        + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "entity-review-aggregate",
            "--review-dir",
            str(entity_dir),
            "--min-reviewers",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "sample_index does not match queued item" in captured.err

    (entity_dir / "entity_annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in entity_annotations) + "\n",
        encoding="utf-8",
    )
    report_path = entity_dir / "entity-aggregation.json"
    rc = _run_cli(
        [
            "entity-review-aggregate",
            "--review-dir",
            str(entity_dir),
            "--output-report",
            str(report_path),
            "--min-reviewers",
            "1",
        ]
    )
    captured = capsys.readouterr()
    aggregate_summary = json.loads(captured.out)
    assert rc == 0
    assert (
        aggregate_summary["aggregation_rule_version"]
        == "entity-review-aggregation-v2"
    )
    assert aggregate_summary["status_counts"] == {
        "actionable": 1,
        "conflicted": 0,
        "other_or_unsure": 1,
        "insufficient_review": 0,
    }
    assert (
        aggregate_summary["consensus_entity_label_counts"][
            "point_source_star_or_planet"
        ]
        == 1
    )
    assert (
        aggregate_summary["consensus_entity_label_counts"]["other_or_unsure"]
        == 1
    )
    assert report_path.exists()

    plurality_annotations = []
    for minute, reviewer, label in [
        (2, "alice", "point_source_star_or_planet"),
        (3, "bob", "point_source_star_or_planet"),
        (4, "carol", "galaxy_elliptical_oval"),
        (5, "dana", "satellite_or_linear_trail"),
    ]:
        plurality_annotations.append(
            {
                **entity_annotations[0],
                "timestamp_utc": f"2026-05-13T02:0{minute}:00+00:00",
                "reviewer": reviewer,
                "entity_label": label,
            }
        )
    (entity_dir / "entity_annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in plurality_annotations) + "\n",
        encoding="utf-8",
    )
    rc = _run_cli(
        [
            "entity-review-aggregate",
            "--review-dir",
            str(entity_dir),
            "--min-reviewers",
            "4",
            "--consensus-rule",
            "majority",
        ]
    )
    captured = capsys.readouterr()
    plurality_summary = json.loads(captured.out)
    assert rc == 0
    plurality_decision = [
        decision
        for decision in plurality_summary["decisions"]
        if decision["queue_id"] == entity_queued[0]["queue_id"]
    ][0]
    assert plurality_summary["status_counts"]["conflicted"] == 1
    assert plurality_decision["status"] == "conflicted"
    assert plurality_decision["reason"] == "majority_consensus_not_met"
    assert "consensus_entity_label" not in plurality_decision


def test_entity_source_dataset_rejects_mismatched_arrays(tmp_path) -> None:
    dataset_dir, _run_dir, _compare_dir = write_review_fixture(tmp_path)
    template = np.load(dataset_dir / "template.npy")
    np.save(dataset_dir / "template.npy", template[:-1], allow_pickle=False)

    with pytest.raises(ValueError, match="array row count"):
        _load_entity_source_dataset(dataset_dir)


def test_cli_entity_review_queue_rejects_empty_real_selection(
    tmp_path,
    capsys,
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    source_review_dir = tmp_path / "source-review-empty-real"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(source_review_dir),
            "--max-items",
            "2",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    queued = [
        json.loads(line)
        for line in (source_review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    annotations = [
        {
            "timestamp_utc": "2026-05-13T03:00:00+00:00",
            "reviewer": "alice",
            "queue_id": item["queue_id"],
            "sample_index": item["sample_index"],
            "candidate_id": item["candidate_id"],
            "reviewer_label": "bogus",
        }
        for item in queued
    ]
    (source_review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "entity-review-queue",
            "--source-review-dirs",
            str(source_review_dir),
            "--output-dir",
            str(tmp_path / "entity-review-empty-real"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "no latest binary real annotations" in captured.err


def test_cli_entity_review_queue_reports_binary_reviewer_fanout(
    tmp_path,
    capsys,
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    source_review_dir = tmp_path / "source-review-fanout"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(source_review_dir),
            "--max-items",
            "1",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    queued = [
        json.loads(line)
        for line in (source_review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    annotations = [
        {
            "timestamp_utc": f"2026-05-13T01:0{minute}:00+00:00",
            "reviewer": reviewer,
            "queue_id": queued[0]["queue_id"],
            "sample_index": queued[0]["sample_index"],
            "candidate_id": queued[0]["candidate_id"],
            "reviewer_label": "real",
            "morphology_tags": ["point_source_round"],
            "notes": f"{reviewer} real",
            "source_label": queued[0]["label"],
            "source_probability": queued[0]["probability"],
        }
        for minute, reviewer in enumerate(("alice", "bob"))
    ]
    (source_review_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n",
        encoding="utf-8",
    )

    entity_dir = tmp_path / "entity-review-fanout"
    rc = _run_cli(
        [
            "entity-review-queue",
            "--source-review-dirs",
            str(source_review_dir),
            "--output-dir",
            str(entity_dir),
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert rc == 0
    assert summary["queue_count"] == 2
    assert summary["unique_source_queue_count"] == 1
    assert summary["multi_reviewer_fanout_count"] == 1
    queued_entities = [
        json.loads(line)
        for line in (entity_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["source_reviewer"] for row in queued_entities} == {
        "alice",
        "bob",
    }
    assert np.load(entity_dir / "dataset" / "labels.npy").tolist() == [1, 1]


def test_entity_review_bokeh_document_persists_annotations(tmp_path) -> None:
    pytest.importorskip("bokeh")
    from bokeh.document import Document
    from bokeh.models import (
        Button,
        CheckboxGroup,
        ColorBar,
        LabelSet,
        Plot,
        Select,
        TextInput,
    )

    from cuphoton.xscan.review import (
        build_entity_review_document,
        build_entity_review_queue,
        build_review_queue,
    )

    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    source_review_dir = tmp_path / "source-review-doc"
    build_review_queue(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="test",
        output_dir=source_review_dir,
        max_items=2,
    )
    queued = [
        json.loads(line)
        for line in (source_review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    (source_review_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-05-13T02:00:00+00:00",
                "reviewer": "alice",
                "queue_id": queued[0]["queue_id"],
                "sample_index": queued[0]["sample_index"],
                "candidate_id": queued[0]["candidate_id"],
                "reviewer_label": "real",
                "morphology_tags": ["point_source_round"],
                "notes": "entity fixture source",
                "source_label": queued[0]["label"],
                "source_probability": queued[0]["probability"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entity_dir = tmp_path / "entity-review-doc"
    build_entity_review_queue(
        source_review_dirs=[source_review_dir],
        output_dir=entity_dir,
    )
    entity_queued = [
        json.loads(line)
        for line in (entity_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    doc = Document()
    build_entity_review_document(doc, review_dir=entity_dir)
    buttons = {
        button.label: button for button in doc.select({"type": Button})
    }
    assert {"Save Entity Label", "Previous", "Next"} <= set(buttons)
    assert buttons["Save Entity Label"].disabled
    username_inputs = [
        item
        for item in doc.select({"type": TextInput})
        if item.title == "Username:"
    ]
    assert len(username_inputs) == 1
    username_inputs[0].value = "gpt-5.5-xhigh"
    assert not buttons["Save Entity Label"].disabled
    show_others = [
        item
        for item in doc.select({"type": CheckboxGroup})
        if item.labels == ["Show Other Entity Reviews"]
    ]
    assert len(show_others) == 1
    image_plots = [
        item
        for item in doc.select({"type": Plot})
        if item.title.text
        in {"Search", "Template", "Search - Template", "Alard-Lupton"}
    ]
    assert len(image_plots) == 4
    assert len({id(plot.x_range) for plot in image_plots}) == 1
    assert len({id(plot.y_range) for plot in image_plots}) == 1
    color_bars = list(doc.select({"type": ColorBar}))
    assert len(color_bars) == 4
    image_mappers = [
        renderer.glyph.color_mapper
        for plot in image_plots
        for renderer in plot.renderers
        if hasattr(getattr(renderer, "glyph", None), "color_mapper")
    ]
    assert len(image_mappers) == 4
    assert {id(bar.color_mapper) for bar in color_bars} == {
        id(mapper) for mapper in image_mappers
    }
    assert len(list(doc.select({"type": LabelSet}))) == 4
    selects = {item.title: item for item in doc.select({"type": Select})}
    selects["Entity Label:"].value = "point_source_star_or_planet"
    selects["Confidence:"].value = "high"

    callbacks = buttons["Save Entity Label"]._event_callbacks["button_click"]
    assert callbacks
    callbacks[0]()
    annotations = [
        json.loads(line)
        for line in (entity_dir / "entity_annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert annotations[-1]["reviewer"] == "gpt-5.5-xhigh"
    assert annotations[-1]["entity_label"] == "point_source_star_or_planet"
    assert annotations[-1]["confidence"] == "high"
    assert annotations[-1]["sample_index"] == entity_queued[0]["sample_index"]
    assert annotations[-1]["source_queue_id"] == queued[0]["queue_id"]
    state = json.loads((entity_dir / "review_state.json").read_text())
    assert state["reviewed_count"] == 1
    assert state["reviewer_decision_count"] == 1
    assert state["reviewer_count"] == 1


def test_review_bokeh_document_persists_annotations_and_handles_edge_images(
    tmp_path,
) -> None:
    pytest.importorskip("bokeh")
    from bokeh.document import Document
    from bokeh.models import (
        Button,
        ColorBar,
        Div,
        LabelSet,
        Plot,
        Span,
        TextInput,
    )

    from cuphoton.xscan.review import build_review_document

    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    search = np.load(dataset_dir / "search.npy", allow_pickle=False)
    template = np.load(dataset_dir / "template.npy", allow_pickle=False)
    difference = np.load(dataset_dir / "difference.npy", allow_pickle=False)
    search[:] = np.nan
    template[:] = 3.0
    difference[:] = 0.0
    np.save(dataset_dir / "search.npy", search, allow_pickle=False)
    np.save(dataset_dir / "template.npy", template, allow_pickle=False)
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)

    review_dir = tmp_path / "review-doc"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "2",
        ]
    )
    assert rc == 0
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sample_index = int(queued[0]["sample_index"])

    doc = Document()
    build_review_document(doc, review_dir=review_dir)
    buttons = {
        button.label: button for button in doc.select({"type": Button})
    }
    assert {"Real", "Bogus", "Unsure", "Previous", "Next"} <= set(buttons)
    assert buttons["Real"].disabled
    assert buttons["Bogus"].disabled
    assert buttons["Unsure"].disabled
    assert not buttons["Previous"].disabled
    assert not buttons["Next"].disabled
    username_inputs = [
        item
        for item in doc.select({"type": TextInput})
        if item.title == "Username:"
    ]
    assert len(username_inputs) == 1
    username_required = [
        item
        for item in doc.select({"type": Div})
        if item.name == "review-username-required"
    ]
    assert len(username_required) == 1
    assert "Mandatory" in username_required[0].text
    username_inputs[0].value = "alice"
    assert not buttons["Real"].disabled
    assert not buttons["Bogus"].disabled
    assert not buttons["Unsure"].disabled
    assert username_required[0].text == ""
    url_states = [
        item
        for item in doc.select({"type": Div})
        if item.name == "review-url-state"
    ]
    assert len(url_states) == 1
    assert url_states[0].text == str(sample_index)
    center_markers = list(doc.select({"type": Span}))
    assert len(center_markers) == 8
    assert {marker.dimension for marker in center_markers} == {
        "height",
        "width",
    }
    image_plots = [
        item
        for item in doc.select({"type": Plot})
        if item.title.text
        in {"Search", "Template", "Search - Template", "Alard-Lupton"}
    ]
    assert len(image_plots) == 4
    assert len({id(plot.x_range) for plot in image_plots}) == 1
    assert len({id(plot.y_range) for plot in image_plots}) == 1
    color_bars = list(doc.select({"type": ColorBar}))
    assert len(color_bars) == 4
    image_mappers = [
        renderer.glyph.color_mapper
        for plot in image_plots
        for renderer in plot.renderers
        if hasattr(getattr(renderer, "glyph", None), "color_mapper")
    ]
    assert len(image_mappers) == 4
    assert {id(bar.color_mapper) for bar in color_bars} == {
        id(mapper) for mapper in image_mappers
    }
    assert len(list(doc.select({"type": LabelSet}))) == 4
    image_headings = [
        item
        for item in doc.select({"type": Div})
        if item.name == "review-image-heading"
    ]
    assert len(image_headings) == 1
    assert "Image Review" in image_headings[0].text
    assert "Candidate ID" in image_headings[0].text

    callbacks = buttons["Real"]._event_callbacks["button_click"]
    assert callbacks
    callbacks[0]()
    assert url_states[0].text == str(int(queued[1]["sample_index"]))

    annotations = [
        json.loads(line)
        for line in (review_dir / "annotations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert annotations[-1]["reviewer_label"] == "real"
    assert annotations[-1]["reviewer"] == "alice"
    assert annotations[-1]["sample_index"] == sample_index
    state = json.loads((review_dir / "review_state.json").read_text())
    assert state["reviewed_count"] == 1
    assert state["reviewer_decision_count"] == 1
    assert state["reviewer_count"] == 1


def test_raw_compare_bokeh_document_builds_from_package(tmp_path) -> None:
    pytest.importorskip("bokeh")
    from bokeh.document import Document
    from bokeh.models import ColorBar, LabelSet, Plot

    from cuphoton.xscan.raw_compare_review import build_document

    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    review_dir = tmp_path / "raw-compare-doc"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "2",
        ]
    )
    assert rc == 0

    doc = Document()
    build_document(doc, review_dir=review_dir)
    color_bars = list(doc.select({"type": ColorBar}))
    assert len(color_bars) == 10
    distribution_plots = [
        item
        for item in doc.select({"type": Plot})
        if item.name == "normalized-pixel-distribution"
    ]
    assert distribution_plots == []
    assert len(list(doc.select({"type": LabelSet}))) == 10


def test_alard_lupton_experiment_bokeh_document_builds(tmp_path) -> None:
    pytest.importorskip("bokeh")
    from bokeh.document import Document
    from bokeh.models import ColorBar, LabelSet, LinearColorMapper, Plot

    from cuphoton.xscan.alard_lupton_experiment_review import (
        build_document,
    )

    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    difference = np.load(dataset_dir / "difference.npy")
    difference[0] *= 250.0
    np.save(dataset_dir / "difference.npy", difference, allow_pickle=False)
    review_dir = tmp_path / "alard-lupton-experiment-doc"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "2",
        ]
    )
    assert rc == 0

    doc = Document()
    build_document(doc, review_dir=review_dir)
    assert "Alard-Lupton Display Lab" in doc.title
    color_bars = list(doc.select({"type": ColorBar}))
    assert len(color_bars) == 8
    image_plots = [
        item for item in doc.select({"type": Plot}) if item.title.text
    ]
    assert any(
        item.title.text == "Flux: Alard-Lupton" for item in image_plots
    )
    assert any(item.title.text == "z: Alard-Lupton" for item in image_plots)
    assert len(list(doc.select({"type": LabelSet}))) == 8
    mappers = list(doc.select({"type": LinearColorMapper}))
    assert any(
        mapper.low == -3.0 and mapper.high == 3.0 for mapper in mappers
    )


def test_review_float_formatters_use_four_decimal_places() -> None:
    pytest.importorskip("bokeh")
    from cuphoton.xscan.raw_compare_review import _fmt as raw_compare_fmt
    from cuphoton.xscan.review import _fmt_metadata_value

    assert _fmt_metadata_value(0.888758793492457) == "0.8888"
    assert _fmt_metadata_value(1887.8477739538985) == "1.8878e+03"
    assert raw_compare_fmt(0.5651851514027153) == "0.5652"
    assert raw_compare_fmt(19666.08612369407) == "1.9666e+04"


def test_review_bokeh_initial_index_uses_sample_query() -> None:
    from cuphoton.xscan.review import _initial_review_index_from_query

    queue = [
        {"sample_index": 10},
        {"sample_index": 20},
        {"sample_index": 30},
    ]
    doc = SimpleNamespace(
        session_context=SimpleNamespace(
            request=SimpleNamespace(arguments={"s": [b"20"]})
        )
    )
    index, notice = _initial_review_index_from_query(
        doc,
        queue,
        fallback_index=0,
    )
    assert index == 1
    assert notice is None

    doc = SimpleNamespace(
        session_context=SimpleNamespace(
            request=SimpleNamespace(arguments={"sample_index": ["999"]})
        )
    )
    index, notice = _initial_review_index_from_query(
        doc,
        queue,
        fallback_index=2,
    )
    assert index == 2
    assert "not in this review queue" in str(notice)

    doc = SimpleNamespace(
        session_context=SimpleNamespace(
            request=SimpleNamespace(arguments={"s": [b"bad"]})
        )
    )
    index, notice = _initial_review_index_from_query(
        doc,
        queue,
        fallback_index=1,
    )
    assert index == 1
    assert "invalid sample query" in str(notice)


def test_review_other_reviews_html_excludes_current_reviewer() -> None:
    from cuphoton.xscan.review import _other_reviews_html

    item = {"queue_id": "sample:20"}
    latest_by_reviewer = {
        "sample:20": {
            "alice": {
                "reviewer_label": "real",
                "morphology_tags": ["point_source_round"],
                "notes": "my note",
                "timestamp_utc": "2026-05-10T00:00:00Z",
            },
            "bob": {
                "reviewer_label": "unsure",
                "morphology_tags": ["unclear"],
                "notes": "blended center",
                "timestamp_utc": "2026-05-10T00:01:00Z",
            },
        }
    }

    rendered = _other_reviews_html(
        item,
        latest_by_reviewer,
        reviewer="alice",
    )

    assert "bob" in rendered
    assert "unsure" in rendered
    assert "blended center" in rendered
    assert "alice" not in rendered
    assert "my note" not in rendered


def test_review_bokeh_document_rejects_stale_queue_indices(tmp_path) -> None:
    pytest.importorskip("bokeh")
    from bokeh.document import Document

    from cuphoton.xscan.review import build_review_document

    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    review_dir = tmp_path / "stale-review-doc"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "2",
        ]
    )
    assert rc == 0
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    queued[0]["sample_index"] = 999
    (review_dir / "queue.jsonl").write_text(
        "\n".join(json.dumps(row) for row in queued) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the dataset arrays"):
        build_review_document(Document(), review_dir=review_dir)


def test_cli_review_apply_rejects_modified_dataset_fingerprint(
    tmp_path, capsys
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    review_dir = tmp_path / "fingerprint-review"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "2",
        ]
    )
    assert rc == 0
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    target = queued[0]
    (review_dir / "annotations.jsonl").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-05-10T20:02:00+00:00",
                "queue_id": target["queue_id"],
                "sample_index": target["sample_index"],
                "reviewer_label": "real",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    search = np.load(dataset_dir / "search.npy", allow_pickle=False)
    search[0] = search[0] + 1.0
    np.save(dataset_dir / "search.npy", search, allow_pickle=False)

    rc = _run_cli(
        [
            "review-apply",
            "--dataset-dir",
            str(dataset_dir),
            "--review-dir",
            str(review_dir),
            "--output-dir",
            str(tmp_path / "fingerprint-rejected"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "fingerprint does not match" in captured.err


def test_cli_review_queue_rejects_mismatched_prediction_identity(
    tmp_path, capsys
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    prediction_path = run_dir / "evaluation" / "test" / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["candidate_id"] = "wrong-candidate"
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "bad-review"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "identity mismatch" in captured.err


def test_cli_review_queue_rejects_mismatched_prediction_labels(
    tmp_path, capsys
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    prediction_path = run_dir / "evaluation" / "test" / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["label"] = 1 - int(rows[0]["label"])
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "bad-label-review"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "label mismatch" in captured.err


def test_cli_review_queue_requires_prediction_summary(
    tmp_path, capsys
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    (run_dir / "evaluation" / "test" / "summary.json").unlink()

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "missing-summary-review"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing prediction summary" in captured.err


def test_cli_review_queue_requires_prediction_summary_identity_fields(
    tmp_path, capsys
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    summary_path = run_dir / "evaluation" / "test" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["split"]
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "missing-summary-field-review"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "missing required split" in captured.err


def test_cli_review_queue_uses_validation_selected_threshold(
    tmp_path, capsys
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    summary_path = run_dir / "evaluation" / "test" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["threshold_diagnostics"] = {
        "validation_selected": {
            "evaluated_split_metrics": {
                "threshold": 0.8,
            },
        },
    }
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    review_dir = tmp_path / "threshold-review"

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "6",
        ]
    )
    capsys.readouterr()
    assert rc == 0
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    sample_one = next(row for row in queued if row["sample_index"] == 1)
    assert sample_one["decision_threshold"] == 0.8
    assert sample_one["prediction"] == 0
    assert sample_one["known_error"] == 1
    sample_two = next(row for row in queued if row["sample_index"] == 2)
    assert sample_two["probability"] == 0.95
    assert sample_two["uncertainty_score"] < 0.3


def test_cli_review_queue_keeps_duplicate_candidate_ids(
    tmp_path, capsys
) -> None:
    dataset_dir, run_dir, _compare_dir = write_review_fixture(tmp_path)
    metadata_rows = load_metadata_rows(dataset_dir)
    for row in metadata_rows:
        row["candidate_id"] = "duplicate-candidate"
    (dataset_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata_rows) + "\n",
        encoding="utf-8",
    )
    prediction_path = run_dir / "evaluation" / "test" / "predictions.jsonl"
    prediction_rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in prediction_rows:
        row["candidate_id"] = "duplicate-candidate"
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    review_dir = tmp_path / "duplicate-review"

    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(review_dir),
            "--max-items",
            "6",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["queue_count"] == 6
    queued = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    queue_ids = [row["queue_id"] for row in queued]
    assert len(set(queue_ids)) == 6
    assert all(queue_id.startswith("sample:") for queue_id in queue_ids)


def test_cli_smoke_build_train_infer_evaluate_compare(
    tmp_path, capsys
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    dataset_dir = tmp_path / "dataset"
    runs_dir = tmp_path / "runs"

    rc = _run_cli(
        [
            "experimental-build-hsc-synthetic",
            "--base",
            str(tmp_path / "data"),
            "--output-dir",
            str(dataset_dir),
            "--positive-count",
            "18",
            "--negative-count",
            "18",
            "--stamp-size",
            "17",
            "--tile-size",
            "8",
            "--seed",
            "3",
        ]
    )
    captured = capsys.readouterr()
    build_summary = json.loads(captured.out)
    assert rc == 0
    assert build_summary["positive_count"] == 18
    assert (dataset_dir / "search.npy").exists()
    assert (dataset_dir / "metadata.jsonl").exists()

    rc = _run_cli(
        [
            "data-validate",
            "--dataset-dir",
            str(dataset_dir),
        ]
    )
    captured = capsys.readouterr()
    validation = json.loads(captured.out)
    assert rc == 0
    assert validation["valid"] is True

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_dir: {dataset_dir}",
                f"output_root: {runs_dir}",
                "run_name: pair-run",
                "epochs: 1",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: auto",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "performance:",
                "  amp_dtype: off",
                "  allow_tf32: false",
                "  cudnn_benchmark: false",
                "  compile: false",
                "  compile_threads: null",
                "  compile_worker_start_method: none",
                "  worker_start_method: none",
                "  worker_cpu_threads: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "  persistent_workers: false",
                "  non_blocking_transfers: false",
                "model:",
                "  input_mode: pair",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run_cli(["train-inada-pair", "--config", str(config_path)])
    captured = capsys.readouterr()
    train_summary = json.loads(captured.out)
    assert rc == 0
    run_dir = Path(train_summary["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert train_summary["package_version"]
    assert train_summary["backend"] == "torch"
    assert train_summary["dtype"] == "float32"
    assert "performance" in train_summary
    assert train_summary["performance"]["runtime"]["torch_version"]
    assert "cuda_runtime_version" in train_summary["performance"]["runtime"]
    assert train_summary["performance"]["runtime"]["amp_dtype"] == "off"
    assert (
        train_summary["performance"]["runtime"]["compile"]["enabled"] is False
    )
    assert train_summary["performance"]["runtime"]["compile_threads"] is None
    assert (
        train_summary["performance"]["runtime"]["compile_worker_start_method"]
        is None
    )
    assert (
        train_summary["performance"]["runtime"]["requested"][
            "worker_start_method"
        ]
        is None
    )
    assert train_summary["performance"]["runtime"]["worker_cpu_threads"] == 1
    if train_summary["device"] == "cuda":
        assert "gpu_memory" in train_summary

    rc = _run_cli(
        [
            "infer-real-bogus",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--batch-size",
            "8",
        ]
    )
    captured = capsys.readouterr()
    infer_summary = json.loads(captured.out)
    assert rc == 0
    assert infer_summary["dataset_dir"] == str(dataset_dir.resolve())
    assert infer_summary["split"] == "test"
    assert "probabilities" in infer_summary["saved"]

    inference_review_dir = tmp_path / "inference-review"
    rc = _run_cli(
        [
            "review-queue",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--output-dir",
            str(inference_review_dir),
            "--max-items",
            "4",
        ]
    )
    captured = capsys.readouterr()
    inference_review = json.loads(captured.out)
    assert rc == 0
    assert inference_review["queue_count"] > 0

    rc = _run_cli(
        [
            "evaluate-real-bogus",
            "--run-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--split",
            "test",
            "--batch-size",
            "8",
        ]
    )
    captured = capsys.readouterr()
    eval_summary = json.loads(captured.out)
    assert rc == 0
    assert eval_summary["dataset_dir"] == str(dataset_dir.resolve())
    assert eval_summary["split"] == "test"
    assert "roc_auc" in eval_summary
    assert "brier_score" in eval_summary
    assert "calibration" in eval_summary
    assert "threshold_diagnostics" in eval_summary
    assert (run_dir / "evaluation" / "test" / "predictions.jsonl").exists()
    assert (run_dir / "evaluation" / "test" / "summary.md").exists()
    assert (run_dir / "evaluation" / "test" / "roc_curve.csv").exists()
    assert (
        run_dir / "evaluation" / "test" / "precision_recall_curve.csv"
    ).exists()
    assert (
        run_dir / "evaluation" / "test" / "binned_flux_ratio.csv"
    ).exists()
    assert (run_dir / "evaluation" / "test" / "calibration.json").exists()
    assert (
        run_dir / "evaluation" / "test" / "reliability_curve.csv"
    ).exists()
    assert (
        run_dir / "evaluation" / "test" / "threshold_diagnostics.json"
    ).exists()
    assert (run_dir / "evaluation" / "test" / "threshold_sweep.csv").exists()
    assert (run_dir / "evaluation" / "test" / "consensus.json").exists()

    rc = _run_cli(
        [
            "compare-inputs",
            "--run-dirs",
            str(run_dir),
        ]
    )
    captured = capsys.readouterr()
    compare_summary = json.loads(captured.out)
    assert rc == 0
    assert compare_summary["best_run_dir"] == str(run_dir.resolve())
    assert "# XScan Compare Inputs" in compare_summary["leaderboard_markdown"]


def test_cli_train_inada_pair_finetunes_from_checkpoint(
    tmp_path,
    capsys,
) -> None:
    manifest_path = write_prepared_manifest_inputs(
        tmp_path,
        include_difference=False,
        dataset_kind="nodiff",
    )
    dataset_dir = tmp_path / "packaged-nodiff"
    runs_dir = tmp_path / "runs"

    rc = _run_cli(
        [
            "data-build-nodiff",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(dataset_dir),
        ]
    )
    captured = capsys.readouterr()
    build_summary = json.loads(captured.out)
    assert rc == 0
    assert build_summary["dataset_kind"] == "nodiff"

    scratch_config = tmp_path / "scratch.yaml"
    scratch_config.write_text(
        "\n".join(
            [
                f"dataset_dir: {dataset_dir}",
                f"output_root: {runs_dir}",
                "run_name: scratch-run",
                "epochs: 1",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: auto",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "performance:",
                "  amp_dtype: off",
                "  allow_tf32: false",
                "  cudnn_benchmark: false",
                "  compile: false",
                "  worker_start_method: none",
                "  worker_cpu_threads: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "  persistent_workers: false",
                "  non_blocking_transfers: false",
                "model:",
                "  input_mode: pair",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run_cli(["train-inada-pair", "--config", str(scratch_config)])
    captured = capsys.readouterr()
    scratch_summary = json.loads(captured.out)
    assert rc == 0
    scratch_run_dir = Path(scratch_summary["run_dir"])
    checkpoint_path = scratch_run_dir / "checkpoint.pt"
    assert checkpoint_path.exists()

    finetune_config = tmp_path / "finetune.yaml"
    finetune_config.write_text(
        "\n".join(
            [
                f"dataset_dir: {dataset_dir}",
                f"output_root: {runs_dir}",
                "run_name: finetune-run",
                "benchmark_regime_name: baseline_aligned",
                "training_mode: fine_tune",
                f"pretrain_checkpoint: {checkpoint_path}",
                "freeze_encoder_stages: [0]",
                "epochs: 1",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 1",
                "device: auto",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "performance:",
                "  amp_dtype: off",
                "  allow_tf32: false",
                "  cudnn_benchmark: false",
                "  compile: false",
                "  worker_start_method: none",
                "  worker_cpu_threads: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "  persistent_workers: false",
                "  non_blocking_transfers: false",
                "model:",
                "  input_mode: pair",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run_cli(["train-inada-pair", "--config", str(finetune_config)])
    captured = capsys.readouterr()
    finetune_summary = json.loads(captured.out)
    assert rc == 0
    assert finetune_summary["training_mode"] == "fine_tune"
    assert finetune_summary["benchmark_regime_name"] == "baseline_aligned"
    assert finetune_summary["transfer"]["enabled"] is True
    assert finetune_summary["transfer"]["loaded_tensor_count"] > 0
    assert finetune_summary["frozen_encoder_stages"] == [0]


def test_cli_train_stops_early_with_patience_case(tmp_path, capsys) -> None:
    write_fake_hsc_store(tmp_path / "data")
    dataset_dir = tmp_path / "dataset"
    runs_dir = tmp_path / "runs"

    rc = _run_cli(
        [
            "experimental-build-hsc-synthetic",
            "--base",
            str(tmp_path / "data"),
            "--output-dir",
            str(dataset_dir),
            "--positive-count",
            "18",
            "--negative-count",
            "18",
            "--stamp-size",
            "17",
            "--tile-size",
            "8",
            "--seed",
            "3",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    config_path = tmp_path / "early-stop.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_dir: {dataset_dir}",
                f"output_root: {runs_dir}",
                "run_name: early-stop-run",
                "epochs: 5",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: cpu",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "early_stopping_metric: val_roc_auc",
                "early_stopping_patience: 0",
                "early_stopping_min_delta: 1.0",
                "performance:",
                "  amp_dtype: off",
                "  allow_tf32: false",
                "  cudnn_benchmark: false",
                "  compile: false",
                "  worker_start_method: none",
                "  worker_cpu_threads: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "  persistent_workers: false",
                "  non_blocking_transfers: false",
                "model:",
                "  input_mode: pair",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run_cli(["train-inada-pair", "--config", str(config_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    history = json.loads(
        (Path(payload["run_dir"]) / "history.json").read_text()
    )

    assert rc == 0
    assert payload["epochs"] == 5
    assert payload["epochs_completed"] == 2
    assert len(history) == 2
    assert payload["early_stopping"]["enabled"] is True
    assert payload["early_stopping"]["stopped_early"] is True
    assert payload["early_stopping"]["stop_epoch"] == 2
    assert payload["early_stopping"]["best_epoch"] == 1
    assert payload["best_metric"]["name"] == "val_roc_auc"


def test_cli_train_rejects_lsstcomcam_placeholder_labels(
    tmp_path,
    capsys,
) -> None:
    dataset_dir = write_lsstcomcam_placeholder_dataset(tmp_path)

    config_path = tmp_path / "placeholder-train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_dir: {dataset_dir}",
                f"output_root: {tmp_path / 'runs'}",
                "run_name: placeholder-rejected",
                "epochs: 1",
                "batch_size: 2",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: cpu",
                "train_split: train",
                "val_split: val",
                "eval_split: val",
                "performance:",
                "  amp_dtype: off",
                "  allow_tf32: false",
                "  cudnn_benchmark: false",
                "  compile: false",
                "  worker_start_method: none",
                "  worker_cpu_threads: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "  persistent_workers: false",
                "  non_blocking_transfers: false",
                "model:",
                "  input_mode: pair",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run_cli(["train-inada-pair", "--config", str(config_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "refusing to train on LSSTComCam smoke placeholder labels" in (
        captured.err
    )
    assert "review-apply" in captured.err


def test_cli_check_training_labels_reports_placeholder_provenance(
    tmp_path,
    capsys,
) -> None:
    dataset_dir = write_lsstcomcam_placeholder_dataset(tmp_path)

    rc = _run_cli(
        ["data-check-training-labels", "--dataset-dir", str(dataset_dir)]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ok"] is False
    assert payload["placeholder_label_count"] == 4
    assert "lsstcomcam_placeholder_labels_present" in payload["errors"]
    assert payload["label_sources"] == {
        "unlabeled_lsstcomcam_smoke_placeholder": 4
    }

    rc = _run_cli(
        [
            "data-check-training-labels",
            "--dataset-dir",
            str(dataset_dir),
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "lsstcomcam_placeholder_labels_present" in captured.err


def test_cli_check_training_labels_accepts_reviewed_provenance(
    tmp_path,
    capsys,
) -> None:
    dataset_dir, _run_dir, _compare_dir = write_review_fixture(tmp_path)

    rc = _run_cli(
        [
            "data-check-training-labels",
            "--dataset-dir",
            str(dataset_dir),
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["placeholder_label_count"] == 0
    assert payload["label_counts"] == {"negative": 3, "positive": 3}
    assert payload["label_sources"] == {"fixture": 6}


def test_cli_reproduce_pair_triplet_uses_one_reviewed_dataset(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    dataset_dir, _run_dir, _compare_dir = write_review_fixture(tmp_path)
    pair_config = write_hsc_training_config(
        tmp_path / "pair.yaml",
        output_root=tmp_path / "unused",
        run_name="placeholder-pair",
        input_mode="pair",
    )
    triplet_config = write_hsc_training_config(
        tmp_path / "triplet.yaml",
        output_root=tmp_path / "unused",
        run_name="placeholder-triplet",
        input_mode="triplet",
    )
    train_calls = []
    eval_calls = []

    def fake_train_classifier(config, *, run_dir):
        train_calls.append(
            {
                "dataset_dir": config.dataset_dir,
                "input_mode": config.model.input_mode,
                "run_name": config.run_name,
                "seed": config.seed,
                "run_dir": str(run_dir),
            }
        )
        return {
            "dataset_dir": config.dataset_dir,
            "training_mode": config.training_mode,
            "benchmark_regime_name": config.benchmark_regime_name,
            "epochs": config.epochs,
            "epochs_completed": 1,
            "batch_size": config.batch_size,
            "best_val_roc_auc": 0.5,
            "best_metric": {"name": "val_roc_auc", "value": 0.5},
            "saved": {},
        }

    def fake_evaluate_workflow(
        *,
        run_dir,
        dataset_dir,
        split,
        batch_size,
        xfit_feature_dir,
        use_xfit_features,
    ):
        input_mode = (
            "triplet" if Path(run_dir).name.startswith("triplet") else "pair"
        )
        eval_calls.append(
            {
                "run_dir": str(run_dir),
                "dataset_dir": str(dataset_dir),
                "split": split,
                "batch_size": batch_size,
                "input_mode": input_mode,
                "xfit_feature_dir": xfit_feature_dir,
                "use_xfit_features": use_xfit_features,
            }
        )
        score = 0.92 if input_mode == "pair" else 0.94
        return SimpleNamespace(
            summary={
                "roc_auc": score,
                "pr_auc": score - 0.01,
                "accuracy": score - 0.02,
                "threshold": 0.5,
                "tpr_at_fpr_1pct": score - 0.10,
                "tpr_at_fpr_5pct": score - 0.05,
                "brier_score": 1.0 - score,
                "confusion": {"tp": 2, "tn": 3, "fp": 1, "fn": 0},
                "calibration": {
                    "brier_score": 1.0 - score,
                    "expected_calibration_error": 0.04,
                    "max_calibration_error": 0.08,
                },
                "sample_count": 6,
                "input_mode": input_mode,
            }
        )

    monkeypatch.setattr(
        xscan_workflows,
        "train_classifier",
        fake_train_classifier,
    )
    monkeypatch.setattr(
        xscan_workflows,
        "evaluate_workflow",
        fake_evaluate_workflow,
    )

    rc = _run_cli(
        [
            "reproduce-pair-triplet",
            "--dataset-dir",
            str(dataset_dir),
            "--pair-config",
            str(pair_config),
            "--triplet-config",
            str(triplet_config),
            "--seeds",
            "0,1",
            "--output-dir",
            str(tmp_path / "compare"),
            "--name",
            "reviewed-pair-triplet",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    run_dir = Path(payload["run_dir"])
    assert payload["workflow"] == "reproduce-pair-triplet"
    assert payload["dataset_dir"] == str(dataset_dir.resolve())
    assert payload["label_provenance"]["ok"] is True
    assert payload["comparison_controls"]["ok"] is True
    assert payload["comparison_controls"]["pair_input_mode"] == "pair"
    assert payload["comparison_controls"]["triplet_input_mode"] == "triplet"
    assert "train_split" in payload["comparison_controls"]["matched_fields"]
    assert payload["ranking"] == ["triplet", "pair"]
    assert set(payload["jobs"]) == {"pair", "triplet"}
    triplet = payload["jobs"]["triplet"]
    assert triplet["mean_tpr_at_fpr_1pct"] == pytest.approx(0.84)
    assert triplet["mean_tpr_at_fpr_5pct"] == pytest.approx(0.89)
    assert triplet["mean_brier_score"] == pytest.approx(0.06)
    assert triplet["confusion_totals"] == {
        "tp": 4,
        "tn": 6,
        "fp": 2,
        "fn": 0,
        "runs": 2,
    }
    assert triplet["calibration_summary"][
        "mean_expected_calibration_error"
    ] == (pytest.approx(0.04))
    assert {call["input_mode"] for call in train_calls} == {
        "pair",
        "triplet",
    }
    assert {call["dataset_dir"] for call in train_calls} == {
        str(dataset_dir.resolve())
    }
    assert len(train_calls) == 4
    assert len(eval_calls) == 4
    assert all(call["xfit_feature_dir"] is None for call in eval_calls)
    assert all(call["use_xfit_features"] is False for call in eval_calls)
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert payload["saved"]["summary_markdown"] == "summary.md"
    assert (
        "# XScan Pair/Triplet Comparison Summary"
        in (payload["summary_markdown"])
    )
    assert "TPR @ 1% FPR" in payload["summary_markdown"]
    assert "4/6/2/0" in payload["summary_markdown"]


def test_cli_reproduce_pair_triplet_rejects_placeholder_labels(
    tmp_path,
    capsys,
) -> None:
    dataset_dir = write_lsstcomcam_placeholder_dataset(tmp_path)
    pair_config = write_hsc_training_config(
        tmp_path / "pair.yaml",
        output_root=tmp_path / "unused",
        run_name="placeholder-pair",
        input_mode="pair",
    )
    triplet_config = write_hsc_training_config(
        tmp_path / "triplet.yaml",
        output_root=tmp_path / "unused",
        run_name="placeholder-triplet",
        input_mode="triplet",
    )
    output_root = tmp_path / "compare"

    rc = _run_cli(
        [
            "reproduce-pair-triplet",
            "--dataset-dir",
            str(dataset_dir),
            "--pair-config",
            str(pair_config),
            "--triplet-config",
            str(triplet_config),
            "--output-dir",
            str(output_root),
            "--name",
            "should-not-exist",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "lsstcomcam_placeholder_labels_present" in captured.err
    assert not (output_root / "should-not-exist").exists()


def test_cli_reproduce_pair_triplet_rejects_config_control_mismatch(
    tmp_path,
    capsys,
) -> None:
    dataset_dir, _run_dir, _compare_dir = write_review_fixture(tmp_path)
    pair_config = write_hsc_training_config(
        tmp_path / "pair.yaml",
        output_root=tmp_path / "unused",
        run_name="placeholder-pair",
        input_mode="pair",
    )
    triplet_config = write_hsc_training_config(
        tmp_path / "triplet.yaml",
        output_root=tmp_path / "unused",
        run_name="placeholder-triplet",
        input_mode="triplet",
    )
    triplet_config.write_text(
        triplet_config.read_text(encoding="utf-8").replace(
            "epochs: 1",
            "epochs: 2",
        ),
        encoding="utf-8",
    )

    rc = _run_cli(
        [
            "reproduce-pair-triplet",
            "--dataset-dir",
            str(dataset_dir),
            "--pair-config",
            str(pair_config),
            "--triplet-config",
            str(triplet_config),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "pair/triplet comparison controls differ" in captured.err
    assert "epochs" in captured.err


def test_cli_reproduce_hsc_comparison(tmp_path, capsys, monkeypatch) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        lambda search_stamp, template_stamp, **kwargs: (
            np.asarray(search_stamp, dtype=np.float32)
            - np.asarray(template_stamp, dtype=np.float32)
        ).astype(np.float32),
    )
    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        negative_offset_range=(4.0, 6.0),
        difference_mode="xpois",
        extra_lines=["difference_context_size: 25"],
    )
    pair_config = tmp_path / "pair.yaml"
    pair_config.write_text(
        "\n".join(
            [
                "dataset_dir: placeholder",
                f"output_root: {tmp_path / 'runs'}",
                "run_name: placeholder-pair",
                "epochs: 1",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: auto",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "model:",
                "  input_mode: pair",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    triplet_config = tmp_path / "triplet.yaml"
    triplet_config.write_text(
        "\n".join(
            [
                "dataset_dir: placeholder",
                f"output_root: {tmp_path / 'runs'}",
                "run_name: placeholder-triplet",
                "epochs: 1",
                "batch_size: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "seed: 0",
                "device: auto",
                "train_split: train",
                "val_split: val",
                "eval_split: test",
                "model:",
                "  input_mode: triplet",
                "  image_size: 17",
                "  depths: [1, 1]",
                "  num_heads: [1, 1]",
                "  embed_dims: [8, 16]",
                "  decoder_embedding_dim: 8",
                "  pos_dim: 4",
                "  output_nc: 2",
                "  drop_rate: 0.0",
                "  attn_drop: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    compare_root = tmp_path / "compare"

    rc = _run_cli(
        [
            "reproduce-hsc-comparison",
            "--manifest",
            str(manifest_path),
            "--pair-config",
            str(pair_config),
            "--triplet-config",
            str(triplet_config),
            "--seeds",
            "0",
            "--output-dir",
            str(compare_root),
            "--name",
            "hsc-compare",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    run_dir = Path(payload["run_dir"])
    assert run_dir.exists()
    assert payload["alignment"]["all_aligned"] is True
    assert set(payload["jobs"]) == {
        "pair",
        "triplet_simple",
        "triplet_xpois",
    }
    assert (
        run_dir / "datasets" / "triplet_xpois" / "difference.npy"
    ).exists()
    assert (
        payload["datasets"]["triplet_xpois"]["difference_context_size"] == 25
    )
    triplet_run = payload["jobs"]["triplet_xpois"]["runs"][0]
    assert (
        run_dir
        / "model-runs"
        / "triplet_xpois-seed-0"
        / "evaluation"
        / triplet_run["eval_split"]
        / "summary.json"
    ).exists()
    assert (run_dir / "summary.md").exists()
    assert payload["saved"]["summary_markdown"] == "summary.md"
    assert "# XScan HSC Comparison Summary" in payload["summary_markdown"]


def test_cli_reproduce_hsc_xpois_sweep(tmp_path, capsys, monkeypatch) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        negative_offset_range=(4.0, 6.0),
        difference_mode="xpois",
        extra_lines=["difference_context_size: 25"],
    )
    pair_config = write_hsc_training_config(
        tmp_path / "pair.yaml",
        output_root=tmp_path / "runs",
        run_name="placeholder-pair",
        input_mode="pair",
    )
    triplet_config = write_hsc_training_config(
        tmp_path / "triplet.yaml",
        output_root=tmp_path / "runs",
        run_name="placeholder-triplet",
        input_mode="triplet",
    )
    sweep_config = write_hsc_sweep_config(
        tmp_path / "sweep.yaml",
        variants=[
            {
                "name": "ctx25_default",
                "difference_context_size": 25,
                "xpois_kernel_shape": [9, 9],
                "xpois_basis_sigmas": [1.5, 3.0],
                "xpois_basis_degrees": [2, 1],
                "xpois_background_degree": 0,
                "xpois_flux_conserve": False,
                "xpois_use_variance": True,
            },
            {
                "name": "ctx27_bg1",
                "difference_context_size": 27,
                "xpois_kernel_shape": [9, 9],
                "xpois_basis_sigmas": [1.5, 3.0],
                "xpois_basis_degrees": [2, 1],
                "xpois_background_degree": 1,
                "xpois_flux_conserve": False,
                "xpois_use_variance": True,
            },
        ],
    )

    def fake_xpois_difference(
        search_stamp: np.ndarray,
        template_stamp: np.ndarray,
        *,
        variance_stamp: np.ndarray | None = None,
        config=None,
    ) -> np.ndarray:
        del template_stamp, variance_stamp
        fill_value = 3.0 + float(config.background_degree)
        return np.full(search_stamp.shape, fill_value, dtype=np.float32)

    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        fake_xpois_difference,
    )
    compare_root = tmp_path / "sweep"

    rc = _run_cli(
        [
            "reproduce-hsc-xpois-sweep",
            "--manifest",
            str(manifest_path),
            "--pair-config",
            str(pair_config),
            "--triplet-config",
            str(triplet_config),
            "--sweep-config",
            str(sweep_config),
            "--seeds",
            "0",
            "--output-dir",
            str(compare_root),
            "--name",
            "hsc-sweep",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    run_dir = Path(payload["run_dir"])
    assert run_dir.exists()
    assert payload["workflow"] == "reproduce-hsc-xpois-sweep"
    assert payload["alignment"]["all_aligned"] is True
    assert set(payload["jobs"]) == {
        "pair",
        "triplet_simple",
        "triplet_xpois_ctx25_default",
        "triplet_xpois_ctx27_bg1",
    }
    assert payload["ranking"] == []
    assert set(payload["unranked"]) == set(payload["jobs"])
    assert payload["best_run_dir"] is None
    stable = payload["variant_builds"]["triplet_xpois_ctx25_default"]
    assert stable["stability_status"] == "stable"
    assert stable["difference_diagnostics"]["allclose"] is False
    assert stable["difference_diagnostics"]["mean_abs_delta"] > 0.0
    other = payload["variant_builds"]["triplet_xpois_ctx27_bg1"]
    assert other["stability_status"] == "stable"
    assert other["difference_diagnostics"]["mean_abs_delta"] > 0.0
    assert (run_dir / "summary.md").exists()
    assert payload["saved"]["summary_markdown"] == "summary.md"
    assert "# XScan HSC XPOIS Sweep Summary" in payload["summary_markdown"]
    assert (
        "Unranked because no run has defined ROC and PR AUC"
        in payload["summary_markdown"]
    )
    assert "## Mask Diagnostics" in payload["summary_markdown"]


def test_cli_reproduce_hsc_xpois_sweep_marks_unstable_variants(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        negative_offset_range=(4.0, 6.0),
        difference_mode="xpois",
        extra_lines=["difference_context_size: 25"],
    )
    pair_config = write_hsc_training_config(
        tmp_path / "pair.yaml",
        output_root=tmp_path / "runs",
        run_name="placeholder-pair",
        input_mode="pair",
    )
    triplet_config = write_hsc_training_config(
        tmp_path / "triplet.yaml",
        output_root=tmp_path / "runs",
        run_name="placeholder-triplet",
        input_mode="triplet",
    )
    sweep_config = write_hsc_sweep_config(
        tmp_path / "sweep.yaml",
        variants=[
            {
                "name": "stable",
                "difference_context_size": 27,
                "xpois_kernel_shape": [9, 9],
                "xpois_basis_sigmas": [1.5, 3.0],
                "xpois_basis_degrees": [2, 1],
                "xpois_background_degree": 0,
                "xpois_flux_conserve": False,
                "xpois_use_variance": True,
            },
            {
                "name": "unstable",
                "difference_context_size": 25,
                "xpois_kernel_shape": [9, 9],
                "xpois_basis_sigmas": [1.5, 3.0],
                "xpois_basis_degrees": [2, 1],
                "xpois_background_degree": 1,
                "xpois_flux_conserve": False,
                "xpois_use_variance": True,
            },
        ],
    )

    def fake_xpois_difference(
        search_stamp: np.ndarray,
        template_stamp: np.ndarray,
        *,
        variance_stamp: np.ndarray | None = None,
        config=None,
    ) -> np.ndarray:
        del template_stamp, variance_stamp
        if int(config.background_degree) == 1:
            raise RuntimeError("ill-conditioned fake solve")
        return np.full(search_stamp.shape, 2.0, dtype=np.float32)

    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        fake_xpois_difference,
    )
    compare_root = tmp_path / "sweep"

    rc = _run_cli(
        [
            "reproduce-hsc-xpois-sweep",
            "--manifest",
            str(manifest_path),
            "--pair-config",
            str(pair_config),
            "--triplet-config",
            str(triplet_config),
            "--sweep-config",
            str(sweep_config),
            "--seeds",
            "0",
            "--output-dir",
            str(compare_root),
            "--name",
            "hsc-sweep",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    run_dir = Path(payload["run_dir"])
    unstable = payload["variant_builds"]["triplet_xpois_unstable"]
    assert unstable["stability_status"] == "unstable"
    assert unstable["failure"]["message"] == "ill-conditioned fake solve"
    assert "triplet_xpois_unstable" not in payload["jobs"]
    assert "triplet_xpois_unstable" in payload["unstable_variants"]
    assert "triplet_xpois_stable" in payload["jobs"]
    assert not (
        run_dir / "model-runs" / "triplet_xpois_unstable-seed-0"
    ).exists()
    assert "ill-conditioned fake solve" in payload["summary_markdown"]


def test_cli_build_hsc_xpois_rejects_nonfinite_difference(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        difference_mode="xpois",
        extra_lines=["difference_context_size: 25"],
    )
    output_dir = tmp_path / "hsc-nan-diff"

    def fake_xpois_difference(
        search_stamp: np.ndarray,
        template_stamp: np.ndarray,
        *,
        variance_stamp: np.ndarray | None = None,
        config=None,
    ) -> np.ndarray:
        del template_stamp, variance_stamp, config
        return np.full(search_stamp.shape, np.nan, dtype=np.float32)

    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        fake_xpois_difference,
    )

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "non-finite" in captured.err
    assert not (output_dir / "summary.json").exists()


def test_cli_build_hsc_xpois_separable_sets_solver_mode(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        difference_mode="xpois_separable",
        extra_lines=["difference_context_size: 25"],
    )
    output_dir = tmp_path / "hsc-separable"

    def fake_xpois_difference(
        search_stamp: np.ndarray,
        template_stamp: np.ndarray,
        *,
        variance_stamp: np.ndarray | None = None,
        config=None,
    ) -> np.ndarray:
        del template_stamp, variance_stamp
        assert config is not None
        assert config.solver_mode == "separable"
        return np.zeros_like(search_stamp, dtype=np.float32)

    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        fake_xpois_difference,
    )

    rc = _run_cli(
        [
            "data-build-hsc",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["difference_mode"] == "xpois_separable"
    assert payload["manifest_difference_mode"] == "xpois_separable"
    assert payload["xpois"]["solver_mode"] == "separable"


def test_cli_reproduce_hsc_xpois_sweep_marks_nonfinite_variants_unstable(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    write_fake_hsc_store(tmp_path / "data")
    write_fake_hsc_catalog_parquet(tmp_path / "data")
    manifest_path = write_hsc_manifest(
        tmp_path,
        negative_source_mode="catalog-offset",
        negative_offset_range=(4.0, 6.0),
        difference_mode="xpois",
        extra_lines=["difference_context_size: 25"],
    )
    pair_config = write_hsc_training_config(
        tmp_path / "pair.yaml",
        output_root=tmp_path / "runs",
        run_name="placeholder-pair",
        input_mode="pair",
    )
    triplet_config = write_hsc_training_config(
        tmp_path / "triplet.yaml",
        output_root=tmp_path / "runs",
        run_name="placeholder-triplet",
        input_mode="triplet",
    )
    sweep_config = write_hsc_sweep_config(
        tmp_path / "sweep.yaml",
        variants=[
            {
                "name": "stable",
                "difference_context_size": 25,
                "xpois_kernel_shape": [9, 9],
                "xpois_basis_sigmas": [1.5, 3.0],
                "xpois_basis_degrees": [2, 1],
                "xpois_background_degree": 0,
                "xpois_flux_conserve": False,
                "xpois_use_variance": True,
            },
            {
                "name": "nan_variant",
                "difference_context_size": 25,
                "xpois_kernel_shape": [9, 9],
                "xpois_basis_sigmas": [1.5, 3.0],
                "xpois_basis_degrees": [2, 1],
                "xpois_background_degree": 1,
                "xpois_flux_conserve": False,
                "xpois_use_variance": True,
            },
        ],
    )

    def fake_xpois_difference(
        search_stamp: np.ndarray,
        template_stamp: np.ndarray,
        *,
        variance_stamp: np.ndarray | None = None,
        config=None,
    ) -> np.ndarray:
        del template_stamp, variance_stamp
        if int(config.background_degree) == 1:
            return np.full(search_stamp.shape, np.nan, dtype=np.float32)
        return np.full(search_stamp.shape, 2.0, dtype=np.float32)

    monkeypatch.setattr(
        xscan_dataset,
        "xpois_difference",
        fake_xpois_difference,
    )
    compare_root = tmp_path / "sweep"

    rc = _run_cli(
        [
            "reproduce-hsc-xpois-sweep",
            "--manifest",
            str(manifest_path),
            "--pair-config",
            str(pair_config),
            "--triplet-config",
            str(triplet_config),
            "--sweep-config",
            str(sweep_config),
            "--seeds",
            "0",
            "--output-dir",
            str(compare_root),
            "--name",
            "hsc-sweep",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    unstable = payload["variant_builds"]["triplet_xpois_nan_variant"]
    assert unstable["stability_status"] == "unstable"
    assert "non-finite" in unstable["failure"]["message"]
    assert "triplet_xpois_nan_variant" not in payload["jobs"]
    assert "triplet_xpois_nan_variant" in payload["unstable_variants"]
