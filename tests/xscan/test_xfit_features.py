# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cuphoton.xscan.xfit_features as xfit_features_module
from cuphoton.core.artifacts import array_sha256, file_sha256
from cuphoton.xscan.xfit_features import (
    FEATURE_NAMES,
    FEATURE_TRANSFORMS,
    build_xfit_feature_bundle,
    export_xfit_input,
    load_xfit_feature_matrix,
)

_GAUSSIAN_PARAMETERS = (
    "amplitude",
    "sigma_x",
    "sigma_y",
    "theta",
    "x_pos",
    "y_pos",
    "x_neg",
    "y_neg",
)
_STAMP_PARAMETERS = ("x_pos", "y_pos", "x_neg", "y_neg", "flux")


def _difference_stamp(candidate_id: str | int) -> np.ndarray:
    candidate_text = str(candidate_id)
    value = sum(
        (index + 1) * ord(character)
        for index, character in enumerate(candidate_text)
    )
    return np.full((11, 13), value % 997, dtype=np.float32)


def _write_dataset(
    path: Path,
    candidate_ids: list[str | int],
    *,
    differences: np.ndarray | None = None,
    splits: list[str] | None = None,
    split_groups: list[str] | None = None,
) -> None:
    path.mkdir()
    if differences is None:
        differences = np.stack(
            [
                _difference_stamp(candidate_id)
                for candidate_id in candidate_ids
            ]
        )
    if splits is None:
        splits = ["train"] * len(candidate_ids)
    if split_groups is None:
        split_groups = [f"group-{value}" for value in candidate_ids]
    rows = [
        {
            "candidate_id": candidate_id,
            "split": splits[index],
            "split_group": split_groups[index],
            "label": 0,
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]
    (path / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    np.save(
        path / "search.npy",
        np.zeros((len(rows), 11, 13), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(path / "difference.npy", differences, allow_pickle=False)


def _gaussian_row(
    candidate_id: str | int,
    candidate_index: int,
    *,
    converged: bool = True,
    sigma_x: float = 2.0,
    sigma_y: float = 1.0,
    theta: float = 0.0,
    x_pos: float = -2.0,
    y_pos: float = 0.0,
    x_neg: float = 2.0,
    y_neg: float = 0.0,
) -> dict[str, object]:
    values = {
        "amplitude": 10.0,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "theta": theta,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "x_neg": x_neg,
        "y_neg": y_neg,
    }
    row: dict[str, object] = {
        "candidate_index": candidate_index,
        "candidate_id": candidate_id,
        "model": "gaussian",
        "mode": "difference",
        "status": "converged_f_tol" if converged else "max_evaluations",
        "converged": converged,
        "evaluations": 12,
        "residual_norm": 5.0,
        "chi_square": 25.0,
        "degrees_of_freedom": 100,
        "reduced_chi_square": 1.0,
        "valid_pixel_count": 108,
        "valid_pixel_fraction": 108.0 / 143.0,
        "null_chi_square": 100.0,
        "delta_chi_square": 75.0,
        "fractional_null_improvement": 0.75,
        "uncertainty_valid": converged,
        "uncertainty_reason": "" if converged else "fit did not converge",
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        **values,
    }
    for name in _GAUSSIAN_PARAMETERS:
        row[f"{name}_standard_error"] = 2.0 if name == "amplitude" else 1.0
    return row


def _stamp_row(
    candidate_id: str | int, candidate_index: int
) -> dict[str, object]:
    values = {
        "x_pos": -2.0,
        "y_pos": -1.0,
        "x_neg": 2.0,
        "y_neg": 1.0,
        "flux": 20.0,
    }
    row: dict[str, object] = {
        "candidate_index": candidate_index,
        "candidate_id": candidate_id,
        "model": "stamp",
        "mode": "difference",
        "status": "converged_f_tol",
        "converged": True,
        "evaluations": 9,
        "residual_norm": 3.0,
        "chi_square": 9.0,
        "degrees_of_freedom": 120,
        "reduced_chi_square": 0.5,
        "valid_pixel_count": 125,
        "valid_pixel_fraction": 125.0 / 143.0,
        "null_chi_square": 36.0,
        "delta_chi_square": 27.0,
        "fractional_null_improvement": 0.75,
        "uncertainty_valid": True,
        "uncertainty_reason": "",
        "backend": "numpy",
        "device": "cpu",
        "dtype": "float64",
        **values,
    }
    for name in _STAMP_PARAMETERS:
        row[f"{name}_standard_error"] = 2.0 if name == "flux" else 1.0
    return row


def _write_run(
    path: Path,
    rows: list[dict[str, object]],
    *,
    model: str = "gaussian",
    variance_present: bool = True,
    covariance_order: list[int] | None = None,
    omit_fit_column: str | None = None,
    residual_shape: tuple[int, ...] | None = None,
    residual_dtype: np.dtype | type = np.float64,
    mode: str = "difference",
    array_candidate_ids: list[str | int] | None = None,
) -> None:
    path.mkdir()
    rows = [dict(row) for row in rows]
    for row in rows:
        row["mode"] = mode
        input_image = _difference_stamp(str(row["candidate_id"]))
        if mode == "split":
            input_image = np.stack([input_image] * 3)
        row.setdefault("input_image_sha256", array_sha256(input_image))
    parameters = (
        _GAUSSIAN_PARAMETERS if model == "gaussian" else _STAMP_PARAMETERS
    )
    table = pa.Table.from_pylist(rows)
    if omit_fit_column is not None:
        table = table.drop([omit_fit_column])
    pq.write_table(table, path / "fits.parquet")
    if covariance_order is None:
        covariance_order = [int(row["candidate_index"]) for row in rows]
    if array_candidate_ids is None:
        candidate_by_index = {
            int(row["candidate_index"]): row["candidate_id"] for row in rows
        }
        array_candidate_ids = [
            candidate_by_index.get(index, rows[position]["candidate_id"])
            for position, index in enumerate(covariance_order)
        ]
    covariance = np.stack(
        [np.eye(len(parameters), dtype=np.float64) for _ in covariance_order]
    )
    if residual_shape is None:
        residual_shape = (
            (len(rows), 11, 13)
            if mode == "difference"
            else (len(rows), 3, 11, 13)
        )
    residuals = np.zeros(residual_shape, dtype=residual_dtype)
    np.savez_compressed(
        path / "fit-arrays.npz",
        candidate_index=np.asarray(covariance_order, dtype=np.int64),
        candidate_id=np.asarray(array_candidate_ids),
        covariance=covariance,
        residuals=residuals,
    )
    config_path = path / "effective-config.yaml"
    config_path.write_text("backend: numpy\n", encoding="utf-8")
    image_shape = (
        [len(rows), 11, 13]
        if mode == "difference"
        else [len(rows), 3, 11, 13]
    )
    summary = {
        "schema_version": 1,
        "workflow": "fit-dipoles",
        "model": model,
        "mode": mode,
        "dtype": str(np.dtype(residual_dtype)),
        "parameter_names": list(parameters),
        "inputs": {
            "candidate_count": len(rows),
            "images_shape": image_shape,
            "mask_present": True,
            "variance_present": variance_present,
            "input_archive_sha256": "a" * 64,
        },
        "artifacts": {
            "effective_config": "effective-config.yaml",
            "fits": "fits.parquet",
            "fit_arrays": "fit-arrays.npz",
        },
        "artifact_sha256": {
            "effective_config": file_sha256(config_path),
            "fits": file_sha256(path / "fits.parquet"),
            "fit_arrays": file_sha256(path / "fit-arrays.npz"),
        },
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _column(values: np.ndarray, name: str) -> np.ndarray:
    return values[:, FEATURE_NAMES.index(name)]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _refresh_run_artifact_hash(run_dir: Path, key: str) -> None:
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    artifacts = summary["artifacts"]
    hashes = summary["artifact_sha256"]
    assert isinstance(artifacts, dict)
    assert isinstance(hashes, dict)
    artifact_name = artifacts[key]
    assert isinstance(artifact_name, str)
    hashes[key] = file_sha256(run_dir / artifact_name)
    _write_json(summary_path, summary)


def _rewrite_feature_values(
    output_dir: Path, updates: dict[str, float]
) -> None:
    feature_path = output_dir / "features.npy"
    features = np.load(feature_path, allow_pickle=False)
    for name, value in updates.items():
        features[0, FEATURE_NAMES.index(name)] = value
    np.save(feature_path, features, allow_pickle=False)
    schema_path = output_dir / "schema.json"
    schema = _read_json(schema_path)
    artifacts = schema["artifacts"]
    assert isinstance(artifacts, dict)
    feature_artifact = artifacts["features"]
    assert isinstance(feature_artifact, dict)
    feature_artifact["sha256"] = file_sha256(feature_path)
    _write_json(schema_path, schema)


def test_export_xfit_input_preserves_dtype_and_collapses_safe_duplicates(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    first = _difference_stamp("fit-a")
    second = _difference_stamp("fit-b")
    _write_dataset(
        dataset_dir,
        ["fit-a", "fit-b", "fit-a"],
        differences=np.stack([first, second, first]),
        splits=["train", "val", "train"],
        split_groups=["group-a", "group-b", "group-a"],
    )
    output_path = tmp_path / "input.npz"

    summary = export_xfit_input(
        dataset_dir=dataset_dir,
        output_path=output_path,
    )

    assert summary["dataset_row_count"] == 3
    assert summary["candidate_count"] == 2
    assert summary["reused_dataset_row_count"] == 1
    assert summary["images_dtype"] == "float32"
    assert summary["input_archive_sha256"] == file_sha256(output_path)
    with np.load(output_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"candidate_id", "images"}
        assert archive["candidate_id"].tolist() == ["fit-a", "fit-b"]
        assert archive["images"].dtype == np.float32
        assert array_sha256(archive["images"][0]) == array_sha256(first)
        assert array_sha256(archive["images"][1]) == array_sha256(second)


def test_export_xfit_input_reads_difference_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a", "fit-b", "fit-c"])
    output_path = tmp_path / "input.npz"
    advanced_read_sizes: list[int] = []
    original_getitem = np.memmap.__getitem__

    def guarded_getitem(
        self: np.memmap, index: object
    ) -> np.ndarray | np.generic:
        if isinstance(index, list):
            advanced_read_sizes.append(len(index))
            if len(index) > 1:
                raise AssertionError("export materialized multiple stamps")
        return original_getitem(self, index)

    monkeypatch.setattr(xfit_features_module, "_EXPORT_BATCH_BYTES", 1)
    monkeypatch.setattr(np.memmap, "__getitem__", guarded_getitem)

    export_xfit_input(dataset_dir=dataset_dir, output_path=output_path)

    assert advanced_read_sizes == [1, 1, 1]
    with np.load(output_path, allow_pickle=False) as archive:
        assert archive["images"].shape == (3, 11, 13)


def test_export_xfit_input_rejects_nonfinite_or_existing_output(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    difference = _difference_stamp("fit-a")
    difference[0, 0] = np.nan
    _write_dataset(
        dataset_dir,
        ["fit-a"],
        differences=difference[None, ...],
    )
    output_path = tmp_path / "input.npz"

    with pytest.raises(ValueError, match="non-finite"):
        export_xfit_input(
            dataset_dir=dataset_dir,
            output_path=output_path,
        )

    difference[0, 0] = 0.0
    np.save(
        dataset_dir / "difference.npy",
        difference[None, ...],
        allow_pickle=False,
    )
    output_path.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="already exists"):
        export_xfit_input(
            dataset_dir=dataset_dir,
            output_path=output_path,
        )


def test_bundle_joins_in_dataset_order_reuses_duplicates_and_reports_extras(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-b", "fit-a", "fit-b", "missing"])
    rows = [
        _gaussian_row("fit-a", 10),
        _gaussian_row(
            "fit-b",
            20,
            sigma_x=1.0,
            sigma_y=2.0,
            theta=math.pi / 2.0,
            x_pos=2.0,
            x_neg=-2.0,
        ),
        _gaussian_row("extra", 30),
    ]
    run_dir = tmp_path / "fit-run"
    _write_run(run_dir, rows, covariance_order=[20, 10, 30])
    output_dir = tmp_path / "features"

    schema = build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
        missing_policy="indicator",
    )

    diagnostics = schema["join_diagnostics"]
    assert diagnostics == {
        "dataset_row_count": 4,
        "dataset_unique_candidate_count": 3,
        "duplicate_dataset_row_count": 1,
        "fit_row_count": 3,
        "matched_dataset_row_count": 3,
        "missing_dataset_row_count": 1,
        "reused_fit_row_count": 1,
        "extra_fit_row_count": 1,
        "missing_candidate_ids": ["missing"],
        "extra_fit_candidate_ids": ["extra"],
    }
    assert schema["source_artifacts"] == {
        "summary": {
            "name": "summary.json",
            "sha256": file_sha256(run_dir / "summary.json"),
        },
        "fits": {
            "name": "fits.parquet",
            "sha256": file_sha256(run_dir / "fits.parquet"),
        },
        "fit_arrays": {
            "name": "fit-arrays.npz",
            "sha256": file_sha256(run_dir / "fit-arrays.npz"),
        },
    }
    assert schema["artifacts"] == {
        "candidate_id": {
            "name": "candidate-id.npy",
            "sha256": file_sha256(output_dir / "candidate-id.npy"),
        },
        "features": {
            "name": "features.npy",
            "sha256": file_sha256(output_dir / "features.npy"),
        },
        "input_image_sha256": {
            "name": "input-image-sha256.npy",
            "sha256": file_sha256(output_dir / "input-image-sha256.npy"),
        },
        "schema": {"name": "schema.json"},
    }
    assert str(tmp_path) not in json.dumps(schema)
    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir,
        feature_dir=output_dir,
        expected_feature_names=FEATURE_NAMES,
    )
    assert matrix.candidate_id.tolist() == [
        "fit-b",
        "fit-a",
        "fit-b",
        "missing",
    ]
    assert matrix.values.dtype == np.float32
    assert np.isfinite(matrix.values).all()
    assert np.array_equal(matrix.values[0], matrix.values[2])
    assert _column(matrix.values, "fit_present").tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert _column(matrix.values, "variance_weighted").tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    missing = matrix.values[3].copy()
    assert not missing.any()
    assert matrix.values[
        0, FEATURE_NAMES.index("axis_ratio")
    ] == pytest.approx(0.5)
    assert matrix.values[
        0, FEATURE_NAMES.index("aligned_ellipticity")
    ] == pytest.approx(
        matrix.values[1, FEATURE_NAMES.index("aligned_ellipticity")]
    )
    candidate_id = np.load(
        output_dir / "candidate-id.npy", allow_pickle=False
    )
    features = np.load(output_dir / "features.npy", allow_pickle=False)
    input_image_sha256 = np.load(
        output_dir / "input-image-sha256.npy", allow_pickle=False
    )
    assert candidate_id.dtype.kind == "U"
    assert features.dtype == np.float32
    assert input_image_sha256.dtype == np.dtype("S64")


def test_bundle_writes_aligned_features_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(
        dataset_dir,
        ["fit-d", "fit-a", "missing", "fit-c", "fit-b"],
    )
    run_dir = tmp_path / "fit-run"
    _write_run(
        run_dir,
        [
            _gaussian_row("fit-a", 0),
            _gaussian_row("fit-b", 1),
            _gaussian_row("fit-c", 2),
            _gaussian_row("fit-d", 3),
        ],
    )
    take_sizes: list[int] = []
    original_take = np.take

    def guarded_take(
        array: np.ndarray,
        indices: np.ndarray,
        axis: int | None = None,
        out: np.ndarray | None = None,
        mode: str = "raise",
    ) -> np.ndarray:
        take_sizes.append(indices.shape[0])
        if indices.shape[0] > 2:
            raise AssertionError("feature alignment materialized all rows")
        return original_take(array, indices, axis=axis, out=out, mode=mode)

    monkeypatch.setattr(xfit_features_module, "_FEATURE_WRITE_BATCH_ROWS", 2)
    monkeypatch.setattr(np, "take", guarded_take)

    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=tmp_path / "features",
        missing_policy="indicator",
    )

    assert take_sizes == [2, 2, 1]


def test_bundle_publish_failure_cleans_up_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "features"
    original_save = np.save
    save_calls = 0

    def fail_second_save(*args: object, **kwargs: object) -> None:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise OSError("injected artifact write failure")
        original_save(*args, **kwargs)

    monkeypatch.setattr(np, "save", fail_second_save)

    with pytest.raises(OSError, match="injected artifact write failure"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".features.*"))
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )
    assert matrix.values.shape == (1, len(FEATURE_NAMES))


def test_all_seventeen_features_match_independent_calculation(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "features"

    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )

    # For an 11x13 stamp the half axes are 5 and 6, the diagonal is
    # sqrt(10**2 + 12**2), and the lobe separation is 4. The identity
    # covariance gives separation variance 2. These values are calculated
    # directly from the row contract rather than through feature helpers.
    expected = np.asarray(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            108.0 / 143.0,
            0.75,
            math.log1p(75.0) / 10.0,
            0.0,
            math.log1p(5.0) / math.log1p(100.0),
            4.0 / math.sqrt(244.0),
            math.log1p(4.0 / math.sqrt(2.0)) / math.log1p(100.0),
            0.0,
            4.0 / 5.0,
            1.0,
            math.sqrt(2.0) / 5.0,
            0.5,
            1.0 / 3.0,
        ],
        dtype=np.float32,
    )
    assert len(expected) == len(FEATURE_NAMES) == 17
    np.testing.assert_allclose(matrix.values[0], expected, rtol=1e-6)


def test_integer_candidate_ids_round_trip_export_bundle_and_load(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, [20, 10, 20])
    input_path = tmp_path / "input.npz"

    export_xfit_input(dataset_dir=dataset_dir, output_path=input_path)

    with np.load(input_path, allow_pickle=False) as archive:
        assert archive["candidate_id"].dtype == np.dtype(np.int64)
        assert archive["candidate_id"].tolist() == [20, 10]
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row(10, 7), _gaussian_row(20, 9)],
        covariance_order=[9, 7],
    )
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )

    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )

    assert matrix.candidate_id.dtype == np.dtype(np.int64)
    assert matrix.candidate_id.tolist() == [20, 10, 20]
    assert np.array_equal(matrix.values[0], matrix.values[2])
    assert matrix.join_diagnostics["reused_fit_row_count"] == 1


@pytest.mark.parametrize(
    "case",
    ["missing_ids", "reused_and_fit_counts", "fit_count", "extra_ids"],
)
def test_loader_cross_checks_join_diagnostics_against_rows(
    tmp_path: Path, case: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-b", "fit-a", "fit-b", "missing"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            _gaussian_row("fit-a", 10),
            _gaussian_row("fit-b", 20),
            _gaussian_row("extra", 30),
        ],
        covariance_order=[20, 10, 30],
    )
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
        missing_policy="indicator",
    )
    schema_path = output_dir / "schema.json"
    schema = _read_json(schema_path)
    diagnostics = schema["join_diagnostics"]
    assert isinstance(diagnostics, dict)
    if case == "missing_ids":
        diagnostics["missing_candidate_ids"] = ["not-missing"]
    elif case == "reused_and_fit_counts":
        diagnostics["reused_fit_row_count"] = 0
        diagnostics["fit_row_count"] = 4
    elif case == "fit_count":
        diagnostics["fit_row_count"] = 4
    else:
        diagnostics["extra_fit_candidate_ids"] = []
    _write_json(schema_path, schema)

    with pytest.raises(ValueError, match="join_diagnostics"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


def test_bundle_missing_policy_error_rejects_absent_fit(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a", "missing"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])

    with pytest.raises(ValueError, match="missing XScan candidate_id"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_rejects_duplicate_xfit_ids(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 0), _gaussian_row("fit-a", 1)],
    )

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_invalid_fit_zeros_parameter_derived_features(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["failed"])
    run_dir = tmp_path / "run"
    row = _gaussian_row("failed", 0, converged=False)
    _write_run(run_dir, [row])
    output_dir = tmp_path / "output"

    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )

    assert matrix.values[0, FEATURE_NAMES.index("fit_present")] == 1.0
    assert matrix.values[0, FEATURE_NAMES.index("fit_valid")] == 0.0
    assert matrix.values[0, FEATURE_NAMES.index("valid_pixel_fraction")] > 0
    for name in (
        "log_strength_snr",
        "separation_over_stamp",
        "separation_significance",
        "midpoint_offset_over_stamp",
        "edge_margin_over_stamp",
        "gaussian_shape_available",
        "size_over_stamp",
        "axis_ratio",
        "aligned_ellipticity",
    ):
        assert matrix.values[0, FEATURE_NAMES.index(name)] == 0.0


def test_stamp_bundle_marks_gaussian_shape_unavailable(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["stamp"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_stamp_row("stamp", 0)], model="stamp")
    output_dir = tmp_path / "output"

    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )

    assert matrix.values[0, FEATURE_NAMES.index("fit_valid")] == 1.0
    assert matrix.values[0, FEATURE_NAMES.index("log_strength_snr")] > 0
    for name in (
        "gaussian_shape_available",
        "size_over_stamp",
        "axis_ratio",
        "aligned_ellipticity",
    ):
        assert matrix.values[0, FEATURE_NAMES.index(name)] == 0.0


def test_unweighted_bundle_zeros_variance_only_statistics(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 0)],
        variance_present=False,
    )
    output_dir = tmp_path / "output"

    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )

    assert matrix.values[0, FEATURE_NAMES.index("variance_weighted")] == 0.0
    assert (
        matrix.values[0, FEATURE_NAMES.index("log_delta_chi_square")] == 0.0
    )
    assert (
        matrix.values[0, FEATURE_NAMES.index("log_reduced_chi_square")] == 0.0
    )
    assert matrix.values[
        0, FEATURE_NAMES.index("fractional_null_improvement")
    ] == pytest.approx(0.75)


def test_loader_enforces_unweighted_source_feature_contract(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 0)],
        variance_present=False,
    )
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    _rewrite_feature_values(output_dir, {"log_delta_chi_square": 0.25})

    with pytest.raises(ValueError, match="variance-only features"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


def test_loader_enforces_gaussian_source_shape_gate(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    _rewrite_feature_values(
        output_dir,
        {
            "gaussian_shape_available": 0.0,
            "size_over_stamp": 0.0,
            "axis_ratio": 0.0,
            "aligned_ellipticity": 0.0,
        },
    )

    with pytest.raises(ValueError, match="does not match the source model"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


def test_loader_enforces_stamp_source_shape_gate(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_stamp_row("fit-a", 0)], model="stamp")
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    _rewrite_feature_values(output_dir, {"gaussian_shape_available": 1.0})

    with pytest.raises(ValueError, match="does not match the source model"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


def test_weighted_bundle_preserves_worse_than_null_delta(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    row = _gaussian_row("fit-a", 0)
    row["delta_chi_square"] = -75.0
    row["fractional_null_improvement"] = -0.75
    run_dir = tmp_path / "run"
    _write_run(run_dir, [row], variance_present=True)
    output_dir = tmp_path / "output"

    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )

    assert matrix.values[0, FEATURE_NAMES.index("log_delta_chi_square")] < 0
    assert matrix.values[
        0, FEATURE_NAMES.index("fractional_null_improvement")
    ] == pytest.approx(-0.75)


def test_bundle_requests_xfit_rerun_for_legacy_artifacts(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 0)],
        omit_fit_column="null_chi_square",
    )

    with pytest.raises(ValueError, match="rerun xFit"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize("schema_version", [2, 1.0, True, "1"])
def test_bundle_rejects_unsupported_source_schema_version(
    tmp_path: Path, schema_version: object
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    summary["schema_version"] = schema_version
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="schema_version is unsupported"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_requires_residual_shape_from_summary(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 0)],
        residual_shape=(1, 10, 13),
    )

    with pytest.raises(ValueError, match="shape and dtype"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_rejects_split_mode_xfit_run(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)], mode="split")

    with pytest.raises(ValueError, match="difference-mode"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_requires_xscan_difference_array(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    (dataset_dir / "difference.npy").unlink()
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])

    with pytest.raises(FileNotFoundError, match="difference.npy"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize(
    ("key", "artifact_name"),
    [
        ("effective_config", "effective-config.yaml"),
        ("fits", "fits.parquet"),
        ("fit_arrays", "fit-arrays.npz"),
    ],
)
def test_bundle_verifies_every_xfit_artifact_before_use(
    tmp_path: Path, key: str, artifact_name: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    artifact_path = run_dir / artifact_name
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=rf"{key} artifact SHA-256"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_requires_xfit_input_archive_hash(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    inputs = summary["inputs"]
    assert isinstance(inputs, dict)
    del inputs["input_archive_sha256"]
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="input_archive_sha256"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_rejects_mismatched_input_image_hash(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    row = _gaussian_row("fit-a", 0)
    row["input_image_sha256"] = "0" * 64
    _write_run(run_dir, [row])

    with pytest.raises(ValueError, match="difference.npy stamp"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_loader_rebinds_bundle_to_current_difference_stamps(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    difference_path = dataset_dir / "difference.npy"
    difference = np.load(difference_path, allow_pickle=False)
    difference[0, 0, 0] += 1.0
    np.save(difference_path, difference, allow_pickle=False)

    with pytest.raises(ValueError, match="current XScan difference.npy"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir,
            feature_dir=output_dir,
        )


def test_bundle_rejects_duplicate_dataset_id_with_different_stamp(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    first = _difference_stamp("fit-a")
    second = first.copy()
    second[0, 0] += 1.0
    _write_dataset(
        dataset_dir,
        ["fit-a", "fit-a"],
        differences=np.stack([first, second]),
    )
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])

    with pytest.raises(ValueError, match="identical difference.npy"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize(
    ("splits", "split_groups"),
    [
        (["train", "val"], ["same", "same"]),
        (["train", "train"], ["group-a", "group-b"]),
        ([None, None], ["same", "same"]),
        (["train", "train"], [None, None]),
    ],
)
def test_bundle_rejects_duplicate_dataset_id_across_split_identity(
    tmp_path: Path,
    splits: list[str | None],
    split_groups: list[str | None],
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(
        dataset_dir,
        ["fit-a", "fit-a"],
        splits=splits,
        split_groups=split_groups,
    )
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])

    with pytest.raises(ValueError, match="same split and split_group"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_rejects_duplicate_parquet_candidate_index(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a", "fit-b"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 0), _gaussian_row("fit-b", 0)],
        covariance_order=[0, 1],
        array_candidate_ids=["fit-a", "fit-b"],
    )

    with pytest.raises(
        ValueError, match="candidate_index values must be unique"
    ):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_requires_candidate_index_bijection(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a", "fit-b"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 10), _gaussian_row("fit-b", 20)],
        covariance_order=[10, 30],
        array_candidate_ids=["fit-a", "fit-b"],
    )

    with pytest.raises(ValueError, match="exact bijection"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_requires_npz_candidate_id_alignment(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a", "fit-b"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 10), _gaussian_row("fit-b", 20)],
        covariance_order=[20, 10],
        array_candidate_ids=["fit-a", "fit-b"],
    )

    with pytest.raises(ValueError, match="candidate_id is not aligned"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_bundle_does_not_materialize_residual_cube(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    from numpy.lib.npyio import NpzFile

    original_getitem = NpzFile.__getitem__

    def guarded_getitem(self: NpzFile, key: str) -> np.ndarray:
        if key == "residuals":
            raise AssertionError("residual cube was materialized")
        return original_getitem(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", guarded_getitem)

    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=tmp_path / "output",
    )


def test_bundle_uses_columnar_and_contiguous_fit_structures() -> None:
    source = inspect.getsource(xfit_features_module)

    assert ".to_pylist(" not in source
    assert "covariance_by_index" not in source
    assert "candidate_keys" not in source


def test_bundle_rejects_object_residual_header(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_gaussian_row("fit-a", 0)],
        residual_dtype=object,
    )
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    summary["dtype"] = "float64"
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="must not contain objects"):
        build_xfit_feature_bundle(
            dataset_dir=dataset_dir,
            xfit_run_dir=run_dir,
            output_dir=tmp_path / "output",
        )


def test_loader_requires_exact_expected_feature_order(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="expected_feature_names"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir,
            feature_dir=output_dir,
            expected_feature_names=tuple(reversed(FEATURE_NAMES)),
        )


def test_loader_returns_read_only_memory_mapped_features(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )

    matrix = load_xfit_feature_matrix(
        dataset_dir=dataset_dir, feature_dir=output_dir
    )

    assert isinstance(matrix.values, np.memmap)
    assert matrix.values.flags.writeable is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema_version"),
        ("schema_version", 1.0, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("schema_version", "1", "schema_version"),
        ("artifact", "other-artifact", "artifact type"),
    ],
)
def test_loader_requires_v1_bundle_identity(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    schema_path = output_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema[field] = value
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


@pytest.mark.parametrize("field", ["features", "transform_constants"])
def test_loader_requires_exact_v1_feature_semantics(
    tmp_path: Path, field: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    schema_path = output_dir / "schema.json"
    schema = _read_json(schema_path)
    if field == "features":
        definitions = schema["features"]
        assert isinstance(definitions, list)
        definition = definitions[0]
        assert isinstance(definition, dict)
        definition["transform"] = "different transform"
    else:
        constants = schema["transform_constants"]
        assert isinstance(constants, dict)
        constants["log_delta_chi_square_scale"] = 11.0
    _write_json(schema_path, schema)

    with pytest.raises(ValueError, match="canonical v1 contract"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


def test_loader_rejects_tampered_feature_artifact(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    feature_path = output_dir / "features.npy"
    features = np.load(feature_path, allow_pickle=False)
    features[0, 0] = 0.0
    np.save(feature_path, features, allow_pickle=False)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


@pytest.mark.parametrize("feature_name", FEATURE_NAMES)
def test_loader_enforces_every_feature_range_case(
    tmp_path: Path, feature_name: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    upper = float(FEATURE_TRANSFORMS[feature_name]["range"][1])
    _rewrite_feature_values(output_dir, {feature_name: upper + 0.25})

    with pytest.raises(ValueError, match="outside its declared range"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


@pytest.mark.parametrize(
    "feature_name",
    [
        "fit_present",
        "fit_valid",
        "uncertainty_valid",
        "variance_weighted",
        "gaussian_shape_available",
    ],
)
def test_loader_enforces_binary_flags(
    tmp_path: Path, feature_name: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    _rewrite_feature_values(output_dir, {feature_name: 0.5})

    with pytest.raises(ValueError, match=rf"{feature_name} must be binary"):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        (
            {"fit_valid": 0.0, "uncertainty_valid": 1.0},
            "uncertainty_valid must not exceed fit_valid",
        ),
        (
            {"fit_present": 0.0, "fit_valid": 1.0},
            "fit_valid must not exceed fit_present",
        ),
        (
            {
                "fit_valid": 0.0,
                "uncertainty_valid": 0.0,
                "gaussian_shape_available": 1.0,
            },
            "gaussian_shape_available must not exceed fit_valid",
        ),
    ],
)
def test_loader_enforces_validity_hierarchy(
    tmp_path: Path, updates: dict[str, float], match: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    _rewrite_feature_values(output_dir, updates)

    with pytest.raises(ValueError, match=match):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        (
            {
                "fit_present": 0.0,
                "fit_valid": 0.0,
                "uncertainty_valid": 0.0,
                "gaussian_shape_available": 0.0,
                "separation_over_stamp": 0.25,
            },
            "absent xFit rows",
        ),
        (
            {
                "fit_valid": 0.0,
                "uncertainty_valid": 0.0,
                "gaussian_shape_available": 0.0,
                "separation_over_stamp": 0.25,
            },
            "invalid xFit rows",
        ),
        (
            {"uncertainty_valid": 0.0, "log_strength_snr": 0.25},
            "without valid uncertainty",
        ),
        (
            {"gaussian_shape_available": 0.0, "axis_ratio": 0.25},
            "without Gaussian shape",
        ),
    ],
)
def test_loader_enforces_feature_gate_semantics(
    tmp_path: Path, updates: dict[str, float], match: str
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, ["fit-a"])
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_gaussian_row("fit-a", 0)])
    output_dir = tmp_path / "output"
    build_xfit_feature_bundle(
        dataset_dir=dataset_dir,
        xfit_run_dir=run_dir,
        output_dir=output_dir,
    )
    _rewrite_feature_values(output_dir, updates)

    with pytest.raises(ValueError, match=match):
        load_xfit_feature_matrix(
            dataset_dir=dataset_dir, feature_dir=output_dir
        )
