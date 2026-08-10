# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Portable, row-aligned xFit feature bundles for XScan."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cuphoton.core.artifacts import array_sha256, file_sha256

from .types import (
    MissingPolicy,
    XFitBundleIdentity,
    XFitFeatureDefinition,
    XFitFeatureSchema,
    XFitJoinDiagnostics,
    XFitModel,
    XFitSourceArtifacts,
)

FEATURE_SCHEMA_VERSION = 1
SOURCE_XFIT_SCHEMA_VERSION = 1
CANDIDATE_ID_ARTIFACT_NAME = "candidate-id.npy"
FEATURE_ARTIFACT_NAME = "features.npy"
INPUT_IMAGE_SHA256_ARTIFACT_NAME = "input-image-sha256.npy"
SCHEMA_ARTIFACT_NAME = "schema.json"
BUNDLE_ARTIFACT_TYPE = "xfit-xscan-feature-bundle"
_EXPORT_BATCH_BYTES = 16 * 1024 * 1024
_PARQUET_BATCH_ROWS = 65_536
_FEATURE_WRITE_BATCH_ROWS = 65_536

_SUMMARY_ARTIFACT_NAMES = {
    "effective_config": "effective-config.yaml",
    "fits": "fits.parquet",
    "fit_arrays": "fit-arrays.npz",
}
_FIT_ARRAY_MEMBERS = {
    "candidate_index.npy",
    "candidate_id.npy",
    "covariance.npy",
    "residuals.npy",
}
_BINARY_FEATURE_NAMES = (
    "fit_present",
    "fit_valid",
    "uncertainty_valid",
    "variance_weighted",
    "gaussian_shape_available",
)

LOG_DELTA_CHI_SQUARE_SCALE = 10.0
LOG_REDUCED_CHI_SQUARE_SCALE = math.log(10.0)
LOG_SIGNIFICANCE_SCALE = math.log1p(100.0)
FLOAT_EPSILON = float(np.finfo(np.float64).tiny)

FEATURE_NAMES = (
    "fit_present",
    "fit_valid",
    "uncertainty_valid",
    "variance_weighted",
    "valid_pixel_fraction",
    "fractional_null_improvement",
    "log_delta_chi_square",
    "log_reduced_chi_square",
    "log_strength_snr",
    "separation_over_stamp",
    "separation_significance",
    "midpoint_offset_over_stamp",
    "edge_margin_over_stamp",
    "gaussian_shape_available",
    "size_over_stamp",
    "axis_ratio",
    "aligned_ellipticity",
)

TRANSFORM_CONSTANTS: Mapping[str, float] = {
    "log_delta_chi_square_scale": LOG_DELTA_CHI_SQUARE_SCALE,
    "log_reduced_chi_square_scale": LOG_REDUCED_CHI_SQUARE_SCALE,
    "log_significance_scale": LOG_SIGNIFICANCE_SCALE,
}

FEATURE_TRANSFORMS: Mapping[str, Mapping[str, object]] = {
    "fit_present": {
        "range": [0.0, 1.0],
        "transform": "1 when candidate_id occurs in fits.parquet",
    },
    "fit_valid": {
        "range": [0.0, 1.0],
        "transform": (
            "converged with positive degrees of freedom, finite parameters, "
            "in-stamp lobes, and positive in-stamp Gaussian widths"
        ),
    },
    "uncertainty_valid": {
        "range": [0.0, 1.0],
        "transform": "fit_valid AND the xFit uncertainty_valid flag",
    },
    "variance_weighted": {
        "range": [0.0, 1.0],
        "transform": (
            "fit_present AND run-level variance_present flag from "
            "summary.json"
        ),
    },
    "valid_pixel_fraction": {
        "range": [0.0, 1.0],
        "transform": "clip(valid_pixel_fraction, 0, 1)",
    },
    "fractional_null_improvement": {
        "range": [-1.0, 1.0],
        "transform": "clip(1 - chi_square / null_chi_square, -1, 1)",
    },
    "log_delta_chi_square": {
        "range": [-1.0, 1.0],
        "transform": (
            "variance weighted only; clip(sign(delta_chi_square) * "
            "log1p(abs(delta_chi_square)) / 10, -1, 1)"
        ),
    },
    "log_reduced_chi_square": {
        "range": [-1.0, 1.0],
        "transform": (
            "variance weighted only; clip(log(reduced_chi_square) / "
            "log(10), -1, 1), centered at reduced chi-square 1"
        ),
    },
    "log_strength_snr": {
        "range": [0.0, 1.0],
        "transform": (
            "clip(log1p(abs(amplitude_or_flux) / standard_error) / "
            "log(101), 0, 1)"
        ),
    },
    "separation_over_stamp": {
        "range": [0.0, 1.0],
        "transform": "clip(lobe separation / stamp diagonal, 0, 1)",
    },
    "separation_significance": {
        "range": [0.0, 1.0],
        "transform": (
            "clip(log1p(separation / covariance-propagated error) / "
            "log(101), 0, 1)"
        ),
    },
    "midpoint_offset_over_stamp": {
        "range": [0.0, 1.0],
        "transform": "clip(lobe midpoint radius / stamp half-diagonal, 0, 1)",
    },
    "edge_margin_over_stamp": {
        "range": [0.0, 1.0],
        "transform": (
            "clip(minimum lobe-to-edge margin / stamp half-minimum-axis, "
            "0, 1)"
        ),
    },
    "gaussian_shape_available": {
        "range": [0.0, 1.0],
        "transform": "1 for a valid Gaussian fit, otherwise 0",
    },
    "size_over_stamp": {
        "range": [0.0, 1.0],
        "transform": (
            "Gaussian only; clip(sqrt(sigma_major*sigma_minor) / "
            "stamp half-minimum-axis, 0, 1)"
        ),
    },
    "axis_ratio": {
        "range": [0.0, 1.0],
        "transform": "Gaussian only; sigma_minor / sigma_major",
    },
    "aligned_ellipticity": {
        "range": [-1.0, 1.0],
        "transform": (
            "Gaussian only; ((major-minor)/(major+minor)) * "
            "cos(2*(major_axis-separation_axis))"
        ),
    },
}

_REQUIRED_FIT_STATISTICS = (
    "valid_pixel_count",
    "valid_pixel_fraction",
    "null_chi_square",
    "delta_chi_square",
    "fractional_null_improvement",
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
_PARAMETER_FEATURE_NAMES = (
    "log_strength_snr",
    "separation_over_stamp",
    "separation_significance",
    "midpoint_offset_over_stamp",
    "edge_margin_over_stamp",
    "gaussian_shape_available",
    "size_over_stamp",
    "axis_ratio",
    "aligned_ellipticity",
)


@dataclass(frozen=True, slots=True)
class XFitFeatureMatrix:
    """Loaded XScan-row-aligned scalar features."""

    dataset_dir: Path
    candidate_id: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]
    schema: XFitFeatureSchema
    bundle_identity: XFitBundleIdentity
    join_diagnostics: XFitJoinDiagnostics

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> np.ndarray:
        return self.values[index]


@dataclass(frozen=True, slots=True)
class _XFitRunContract:
    model: XFitModel
    parameter_names: tuple[str, ...]
    image_shape: tuple[int, int]
    residual_shape: tuple[int, ...]
    residual_dtype: np.dtype[Any]
    mask_present: bool
    variance_present: bool
    input_archive_sha256: str


@dataclass(frozen=True, slots=True)
class _DatasetContract:
    candidate_id: np.ndarray
    split: np.ndarray
    split_group: np.ndarray
    image_sha256: np.ndarray


@dataclass(frozen=True, slots=True)
class _FitArrayContract:
    sorted_candidate_index: np.ndarray
    array_row_by_sorted_index: np.ndarray
    candidate_id: np.ndarray
    covariance: np.ndarray


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _required_sha256(value: Any, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _ascii_sha256(value: object, *, description: str) -> str:
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{description} must be a lowercase SHA-256 digest"
        ) from exc
    return _required_sha256(decoded, description=description)


def _resolve_run_artifact(
    run_dir: Path,
    summary: Mapping[str, Any],
    key: str,
    expected_name: str,
) -> Path:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("xFit summary.json is missing the artifacts object")
    value = artifacts.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"xFit summary.json is missing artifacts.{key}")
    candidate = (run_dir / value).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(
            f"xFit artifacts.{key} escapes the run directory"
        ) from exc
    if candidate.name != expected_name:
        raise ValueError(
            f"xFit artifacts.{key} must name {expected_name}, got {value!r}"
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"xFit artifact does not exist: {candidate}")
    return candidate


def _verify_run_artifact(
    summary: Mapping[str, Any], key: str, path: Path
) -> str:
    hashes = summary.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise ValueError(
            "xFit run lacks artifact hashes; rerun xFit with a version that "
            "records artifact_sha256"
        )
    expected = _required_sha256(
        hashes.get(key), description=f"xFit artifact_sha256.{key}"
    )
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"xFit {key} artifact SHA-256 mismatch")
    return actual


def _summary_presence_flag(summary: Mapping[str, Any], name: str) -> bool:
    inputs = summary.get("inputs")
    value = inputs.get(name) if isinstance(inputs, dict) else None
    if value is None:
        value = summary.get(name)
    if not isinstance(value, bool):
        raise ValueError(
            "xFit run lacks portable feature statistics; rerun xFit with a "
            f"version that records inputs.{name} and per-fit null statistics"
        )
    return value


def _candidate_key(value: Any) -> tuple[str, int | str]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError("candidate_id values must be integers or strings")
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**63 - 1:
            raise ValueError(
                "integer candidate_id values must fit signed 64-bit"
            )
        return ("integer", value)
    if isinstance(value, str) and value:
        return ("string", value)
    raise ValueError(
        "candidate_id values must be non-empty integers or strings"
    )


def _candidate_array(
    values: Sequence[Any] | np.ndarray[Any, np.dtype[Any]],
    *,
    allow_duplicates: bool,
) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError("candidate_id must be a one-dimensional array")
    if len(values) == 0:
        return np.asarray([], dtype="U1")
    first_kind, first_value = _candidate_key(values[0])
    if first_kind == "integer":
        normalized = np.empty(len(values), dtype=np.int64)
        normalized[0] = cast(int, first_value)
        for index in range(1, len(values)):
            kind, value = _candidate_key(values[index])
            if kind != first_kind:
                raise ValueError(
                    "candidate_id values must not mix integers and strings"
                )
            normalized[index] = cast(int, value)
    else:
        for index in range(1, len(values)):
            kind, _ = _candidate_key(values[index])
            if kind != first_kind:
                raise ValueError(
                    "candidate_id values must not mix integers and strings"
                )
        normalized = np.asarray(values, dtype=str)
    if (
        not allow_duplicates
        and np.unique(normalized).shape[0] != normalized.shape[0]
    ):
        raise ValueError(
            "xFit fits.parquet contains duplicate candidate_id values"
        )
    return normalized


def _arrow_values(column: pa.Array | pa.ChunkedArray) -> np.ndarray:
    array = (
        column.combine_chunks()
        if isinstance(column, pa.ChunkedArray)
        else column
    )
    return np.asarray(array.to_numpy(zero_copy_only=False))


def _fit_candidate_dtype(parquet: pq.ParquetFile) -> np.dtype[Any]:
    candidate_type = parquet.schema_arrow.field("candidate_id").type
    if pa.types.is_integer(candidate_type):
        return np.dtype(np.int64)
    if not (
        pa.types.is_string(candidate_type)
        or pa.types.is_large_string(candidate_type)
    ):
        raise ValueError(
            "fits.parquet candidate_id must contain integers or strings"
        )
    maximum_length = 0
    for batch in parquet.iter_batches(
        batch_size=_PARQUET_BATCH_ROWS, columns=["candidate_id"]
    ):
        candidate_column = batch.column(0)
        if candidate_column.null_count:
            raise ValueError(
                "candidate_id values must be non-empty integers or strings"
            )
        # PyArrow's runtime exports these compute kernels even though its
        # current type information does not enumerate them.
        compute = cast(Any, pc)
        batch_maximum = compute.max(
            compute.utf8_length(candidate_column)
        ).as_py()
        if batch_maximum is not None:
            maximum_length = max(maximum_length, int(batch_maximum))
    return np.dtype(f"U{max(maximum_length, 1)}")


def _load_fit_indices(parquet: pq.ParquetFile, row_count: int) -> np.ndarray:
    indices = np.empty(row_count, dtype=np.int64)
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=_PARQUET_BATCH_ROWS, columns=["candidate_index"]
    ):
        values = _arrow_values(batch.column(0))
        if values.dtype.kind not in "iu" or (
            values.dtype.kind == "u"
            and np.any(values > np.iinfo(np.int64).max)
        ):
            raise ValueError(
                "fits.parquet candidate_index must contain integers"
            )
        indices[offset : offset + batch.num_rows] = values
        offset += batch.num_rows
    if offset != row_count:
        raise ValueError("fits.parquet row count changed while reading")
    if np.unique(indices).shape[0] != row_count:
        raise ValueError("fits.parquet candidate_index values must be unique")
    return indices


def _load_dataset_metadata(
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"XScan dataset directory does not exist: {root}"
        )
    jsonl_path = root / "metadata.jsonl"
    candidate_values: list[Any] | np.ndarray
    split_values: list[Any] | np.ndarray
    split_group_values: list[Any] | np.ndarray
    if jsonl_path.is_file():
        candidate_values = []
        split_values = []
        split_group_values = []
        try:
            with jsonl_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(
                            "metadata.jsonl row "
                            f"{line_number} is not an object"
                        )
                    if "candidate_id" not in row:
                        raise ValueError(
                            "XScan metadata row "
                            f"{len(candidate_values)} is missing candidate_id"
                        )
                    candidate_values.append(row["candidate_id"])
                    split_values.append(row.get("split"))
                    split_group_values.append(row.get("split_group"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read XScan metadata: {jsonl_path}"
            ) from exc
    else:
        parquet_path = root / "metadata.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(
                "XScan dataset lacks metadata.jsonl or metadata.parquet: "
                f"{root}"
            )
        schema_names = set(pq.read_schema(parquet_path).names)
        if "candidate_id" not in schema_names:
            raise ValueError("XScan metadata is missing candidate_id")
        columns = [
            name
            for name in ("candidate_id", "split", "split_group")
            if name in schema_names
        ]
        table = pq.read_table(parquet_path, columns=columns)
        candidate_values = _arrow_values(table.column("candidate_id"))
        row_count = len(candidate_values)
        split_values = (
            _arrow_values(table.column("split"))
            if "split" in columns
            else [None] * row_count
        )
        split_group_values = (
            _arrow_values(table.column("split_group"))
            if "split_group" in columns
            else [None] * row_count
        )
    if len(candidate_values) == 0:
        raise ValueError("XScan metadata contains no rows")
    candidate_id = _candidate_array(candidate_values, allow_duplicates=True)
    split = np.asarray(split_values, dtype=object)
    split_group = np.asarray(split_group_values, dtype=object)
    search_path = root / "search.npy"
    if search_path.is_file():
        search = np.load(search_path, mmap_mode="r", allow_pickle=False)
        if search.shape[0] != candidate_id.shape[0]:
            raise ValueError(
                "XScan metadata row count does not match search.npy"
            )
    return candidate_id, split, split_group


def _load_dataset_candidate_ids(dataset_dir: Path) -> np.ndarray:
    candidate_id, _, _ = _load_dataset_metadata(dataset_dir)
    return candidate_id


def _load_dataset_contract(
    dataset_dir: Path, *, image_shape: tuple[int, int]
) -> _DatasetContract:
    candidate_id, split, split_group = _load_dataset_metadata(dataset_dir)
    difference_path = dataset_dir / "difference.npy"
    if not difference_path.is_file():
        raise FileNotFoundError(
            "XScan xFit features require dataset difference.npy"
        )
    try:
        difference = np.load(
            difference_path, mmap_mode="r", allow_pickle=False
        )
    except ValueError as exc:
        raise ValueError(
            "XScan difference.npy must be a pickle-free numeric array"
        ) from exc
    expected_shape = (candidate_id.shape[0], *image_shape)
    if difference.shape != expected_shape or difference.dtype.kind != "f":
        raise ValueError(
            "XScan difference.npy must be a floating array with shape "
            f"{expected_shape}"
        )
    image_hashes = np.empty(candidate_id.shape[0], dtype="S64")
    for index in range(candidate_id.shape[0]):
        image_hashes[index] = array_sha256(difference[index]).encode("ascii")
    keys = [_candidate_key(value) for value in candidate_id]
    first_row_by_key: dict[tuple[str, int | str], int] = {}
    for row_index, key in enumerate(keys):
        first_row = first_row_by_key.setdefault(key, row_index)
        if first_row == row_index:
            continue
        if image_hashes[first_row] != image_hashes[row_index]:
            raise ValueError(
                "duplicate XScan candidate_id rows must have identical "
                "difference.npy stamps"
            )
        first_identity = (split[first_row], split_group[first_row])
        row_identity = (split[row_index], split_group[row_index])
        if (
            any(value is None for value in (*first_identity, *row_identity))
            or first_identity != row_identity
        ):
            raise ValueError(
                "duplicate XScan candidate_id rows must have the same split "
                "and split_group"
            )
    return _DatasetContract(
        candidate_id=candidate_id,
        split=split,
        split_group=split_group,
        image_sha256=image_hashes,
    )


def _column_value(
    columns: Mapping[str, np.ndarray], row_index: int, name: str
) -> object | None:
    values = columns.get(name)
    if values is None:
        return None
    value = values[row_index]
    return value.item() if isinstance(value, np.generic) else value


def _finite_float(
    columns: Mapping[str, np.ndarray], row_index: int, name: str
) -> float | None:
    value = _column_value(columns, row_index, name)
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _scaled_log(value: float, scale: float) -> float:
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return float(np.clip(math.log1p(value) / scale, 0.0, 1.0))


def _scaled_signed_log(value: float, scale: float) -> float:
    if not math.isfinite(value) or value == 0:
        return 0.0
    transformed = math.copysign(math.log1p(abs(value)) / scale, value)
    return float(np.clip(transformed, -1.0, 1.0))


def _summary_contract(
    summary: Mapping[str, Any], fit_row_count: int
) -> _XFitRunContract:
    schema_version = summary.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SOURCE_XFIT_SCHEMA_VERSION
    ):
        raise ValueError(
            "xFit summary schema_version is unsupported; expected version 1"
        )
    if summary.get("workflow") != "fit-dipoles":
        raise ValueError("xFit summary workflow must be 'fit-dipoles'")
    model = summary.get("model")
    mode = summary.get("mode")
    if model not in {"gaussian", "stamp"}:
        raise ValueError("xFit summary model must be 'gaussian' or 'stamp'")
    if mode != "difference":
        raise ValueError(
            "XScan xFit features require a difference-mode xFit run"
        )
    expected_parameters = (
        _GAUSSIAN_PARAMETERS if model == "gaussian" else _STAMP_PARAMETERS
    )
    parameter_names = summary.get("parameter_names")
    if tuple(parameter_names or ()) != expected_parameters:
        raise ValueError(
            f"xFit summary parameter_names do not match the {model} model"
        )
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("xFit summary is missing inputs")
    if inputs.get("candidate_count") != fit_row_count:
        raise ValueError(
            "xFit summary candidate count does not match fits.parquet"
        )
    input_archive_sha256 = _required_sha256(
        inputs.get("input_archive_sha256"),
        description="xFit inputs.input_archive_sha256",
    )
    images_shape = inputs.get("images_shape")
    if (
        not isinstance(images_shape, list)
        or len(images_shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in images_shape
        )
        or any(value <= 0 for value in images_shape)
    ):
        raise ValueError("xFit summary inputs.images_shape is invalid")
    if images_shape[0] != fit_row_count:
        raise ValueError(
            "xFit summary candidate count does not match fits.parquet"
        )
    height, width = int(images_shape[-2]), int(images_shape[-1])
    dtype_value = summary.get("dtype")
    if not isinstance(dtype_value, str) or not dtype_value:
        raise ValueError("xFit summary dtype is invalid")
    try:
        residual_dtype = np.dtype(dtype_value)
    except TypeError as exc:
        raise ValueError("xFit summary dtype is invalid") from exc
    if residual_dtype.kind != "f" or residual_dtype.hasobject:
        raise ValueError("xFit summary dtype must be floating point")
    mask_present = _summary_presence_flag(summary, "mask_present")
    variance_present = _summary_presence_flag(summary, "variance_present")
    return _XFitRunContract(
        model=model,
        parameter_names=expected_parameters,
        image_shape=(height, width),
        residual_shape=tuple(images_shape),
        residual_dtype=residual_dtype,
        mask_present=mask_present,
        variance_present=variance_present,
        input_archive_sha256=input_archive_sha256,
    )


def _read_npz_array_header(
    path: Path, member_name: str
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(_FIT_ARRAY_MEMBERS) or set(names) != (
                _FIT_ARRAY_MEMBERS
            ):
                raise ValueError(
                    "fit-arrays.npz must contain candidate_index, "
                    "candidate_id, covariance, and residuals"
                )
            with archive.open(member_name) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(
                        member
                    )
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(
                        member
                    )
                else:
                    raise ValueError(
                        f"fit-arrays.npz {member_name} uses unsupported NPY "
                        f"format {version}"
                    )
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError("fit-arrays.npz is not a valid NPZ archive") from exc
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _load_fit_arrays(
    path: Path,
    *,
    row_count: int,
    parameter_count: int,
    residual_shape: tuple[int, ...],
    residual_dtype: np.dtype[Any],
) -> _FitArrayContract:
    header_shape, header_dtype = _read_npz_array_header(path, "residuals.npy")
    if header_dtype.hasobject:
        raise ValueError("fit-arrays.npz residuals must not contain objects")
    if header_shape != residual_shape or header_dtype != residual_dtype:
        raise ValueError(
            "fit-arrays.npz residuals shape and dtype must match summary.json"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            candidate_index = np.ascontiguousarray(archive["candidate_index"])
            candidate_id_raw = np.asarray(archive["candidate_id"])
            covariance = np.ascontiguousarray(archive["covariance"])
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                "fit-arrays.npz contains a pickle-backed array"
            ) from exc
        raise
    if (
        candidate_index.dtype.kind not in "iu"
        or candidate_index.shape != (row_count,)
        or (
            candidate_index.dtype.kind == "u"
            and np.any(candidate_index > np.iinfo(np.int64).max)
        )
    ):
        raise ValueError(
            "fit-arrays.npz candidate_index has an invalid shape or dtype"
        )
    if covariance.dtype.kind not in "fiu" or covariance.shape != (
        row_count,
        parameter_count,
        parameter_count,
    ):
        raise ValueError(
            "fit-arrays.npz covariance has an invalid shape or dtype"
        )
    candidate_id = _candidate_array(candidate_id_raw, allow_duplicates=True)
    if candidate_id.shape != (row_count,):
        raise ValueError(
            "fit-arrays.npz candidate_id has an invalid shape or dtype"
        )
    indices = np.asarray(candidate_index, dtype=np.int64)
    if np.unique(indices).shape[0] != row_count:
        raise ValueError(
            "fit-arrays.npz candidate_index values must be unique"
        )
    index_order = np.argsort(indices, kind="stable")
    return _FitArrayContract(
        sorted_candidate_index=np.ascontiguousarray(indices[index_order]),
        array_row_by_sorted_index=np.ascontiguousarray(index_order),
        candidate_id=np.ascontiguousarray(candidate_id),
        covariance=covariance,
    )


def _fit_array_row(arrays: _FitArrayContract, candidate_index: int) -> int:
    position = int(
        np.searchsorted(arrays.sorted_candidate_index, candidate_index)
    )
    if (
        position == arrays.sorted_candidate_index.shape[0]
        or int(arrays.sorted_candidate_index[position]) != candidate_index
    ):
        raise ValueError(
            "fits.parquet and fit-arrays.npz candidate_index values must "
            "form an exact bijection"
        )
    return int(arrays.array_row_by_sorted_index[position])


def _fit_is_valid(
    columns: Mapping[str, np.ndarray],
    row_index: int,
    *,
    parameter_names: Sequence[str],
    height: int,
    width: int,
) -> bool:
    if _column_value(columns, row_index, "converged") is not True:
        return False
    degrees_of_freedom = _finite_float(
        columns, row_index, "degrees_of_freedom"
    )
    if degrees_of_freedom is None or degrees_of_freedom <= 0:
        return False
    parameters = [
        _finite_float(columns, row_index, name) for name in parameter_names
    ]
    if any(value is None for value in parameters):
        return False
    half_width = (width - 1) / 2.0
    half_height = (height - 1) / 2.0
    for prefix in ("pos", "neg"):
        x_value = _finite_float(columns, row_index, f"x_{prefix}")
        y_value = _finite_float(columns, row_index, f"y_{prefix}")
        if (
            x_value is None
            or y_value is None
            or abs(x_value) > half_width
            or abs(y_value) > half_height
        ):
            return False
    if "sigma_x" in parameter_names:
        sigma_x = _finite_float(columns, row_index, "sigma_x")
        sigma_y = _finite_float(columns, row_index, "sigma_y")
        stamp_radius = max(min(half_width, half_height), 1.0)
        if (
            sigma_x is None
            or sigma_y is None
            or sigma_x <= 0
            or sigma_y <= 0
            or max(sigma_x, sigma_y) > stamp_radius
        ):
            return False
    return True


def _separation_error(
    covariance: np.ndarray,
    *,
    parameter_names: Sequence[str],
    dx: float,
    dy: float,
    separation: float,
) -> float | None:
    if separation <= 0 or not np.isfinite(covariance).all():
        return None
    positions = [
        parameter_names.index(name)
        for name in ("x_pos", "y_pos", "x_neg", "y_neg")
    ]
    submatrix = covariance[np.ix_(positions, positions)]
    gradient = np.asarray(
        [dx / separation, dy / separation, -dx / separation, -dy / separation]
    )
    variance = float(gradient @ submatrix @ gradient)
    if not math.isfinite(variance) or variance <= 0:
        return None
    return math.sqrt(variance)


def _feature_row(
    columns: Mapping[str, np.ndarray],
    row_index: int,
    covariance: np.ndarray,
    *,
    model: str,
    parameter_names: Sequence[str],
    image_shape: tuple[int, int],
    variance_present: bool,
) -> np.ndarray:
    feature = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    height, width = image_shape
    half_width = (width - 1) / 2.0
    half_height = (height - 1) / 2.0
    stamp_diagonal = max(math.hypot(width - 1, height - 1), 1.0)
    stamp_half_diagonal = max(stamp_diagonal / 2.0, 1.0)
    stamp_radius = max(min(half_width, half_height), 1.0)

    feature[index["fit_present"]] = 1.0
    feature[index["variance_weighted"]] = float(variance_present)
    valid_fraction = _finite_float(columns, row_index, "valid_pixel_fraction")
    if valid_fraction is not None:
        feature[index["valid_pixel_fraction"]] = np.clip(
            valid_fraction, 0.0, 1.0
        )
    improvement = _finite_float(
        columns, row_index, "fractional_null_improvement"
    )
    if improvement is not None:
        feature[index["fractional_null_improvement"]] = np.clip(
            improvement, -1.0, 1.0
        )
    if variance_present:
        delta_chi_square = _finite_float(
            columns, row_index, "delta_chi_square"
        )
        if delta_chi_square is not None:
            feature[index["log_delta_chi_square"]] = _scaled_signed_log(
                delta_chi_square, LOG_DELTA_CHI_SQUARE_SCALE
            )
        reduced_chi_square = _finite_float(
            columns, row_index, "reduced_chi_square"
        )
        if reduced_chi_square is not None and reduced_chi_square > 0:
            feature[index["log_reduced_chi_square"]] = np.clip(
                math.log(max(reduced_chi_square, FLOAT_EPSILON))
                / LOG_REDUCED_CHI_SQUARE_SCALE,
                -1.0,
                1.0,
            )

    fit_valid = _fit_is_valid(
        columns,
        row_index,
        parameter_names=parameter_names,
        height=height,
        width=width,
    )
    feature[index["fit_valid"]] = float(fit_valid)
    uncertainty_valid = (
        fit_valid
        and _column_value(columns, row_index, "uncertainty_valid") is True
    )
    feature[index["uncertainty_valid"]] = float(uncertainty_valid)
    if not fit_valid:
        return feature.astype(np.float32)

    strength_name = "amplitude" if model == "gaussian" else "flux"
    strength = _finite_float(columns, row_index, strength_name)
    strength_error = _finite_float(
        columns, row_index, f"{strength_name}_standard_error"
    )
    if (
        uncertainty_valid
        and strength is not None
        and strength_error is not None
        and strength_error > 0
    ):
        feature[index["log_strength_snr"]] = _scaled_log(
            abs(strength) / strength_error, LOG_SIGNIFICANCE_SCALE
        )

    x_pos = float(columns["x_pos"][row_index])
    y_pos = float(columns["y_pos"][row_index])
    x_neg = float(columns["x_neg"][row_index])
    y_neg = float(columns["y_neg"][row_index])
    dx = x_pos - x_neg
    dy = y_pos - y_neg
    separation = math.hypot(dx, dy)
    feature[index["separation_over_stamp"]] = np.clip(
        separation / stamp_diagonal, 0.0, 1.0
    )
    midpoint_x = (x_pos + x_neg) / 2.0
    midpoint_y = (y_pos + y_neg) / 2.0
    feature[index["midpoint_offset_over_stamp"]] = np.clip(
        math.hypot(midpoint_x, midpoint_y) / stamp_half_diagonal,
        0.0,
        1.0,
    )
    edge_margin = min(
        half_width - abs(x_pos),
        half_height - abs(y_pos),
        half_width - abs(x_neg),
        half_height - abs(y_neg),
    )
    feature[index["edge_margin_over_stamp"]] = np.clip(
        edge_margin / stamp_radius, 0.0, 1.0
    )
    if uncertainty_valid:
        separation_error = _separation_error(
            covariance,
            parameter_names=parameter_names,
            dx=dx,
            dy=dy,
            separation=separation,
        )
        if separation_error is not None:
            feature[index["separation_significance"]] = _scaled_log(
                separation / separation_error, LOG_SIGNIFICANCE_SCALE
            )

    if model == "gaussian":
        sigma_x = float(columns["sigma_x"][row_index])
        sigma_y = float(columns["sigma_y"][row_index])
        theta = float(columns["theta"][row_index])
        if sigma_x >= sigma_y:
            major, minor, major_axis = sigma_x, sigma_y, theta
        else:
            major, minor, major_axis = sigma_y, sigma_x, theta + math.pi / 2.0
        feature[index["gaussian_shape_available"]] = 1.0
        feature[index["size_over_stamp"]] = np.clip(
            math.sqrt(major * minor) / stamp_radius, 0.0, 1.0
        )
        feature[index["axis_ratio"]] = np.clip(minor / major, 0.0, 1.0)
        if separation > 0:
            separation_axis = math.atan2(dy, dx)
            ellipticity = (major - minor) / (major + minor)
            feature[index["aligned_ellipticity"]] = np.clip(
                ellipticity * math.cos(2.0 * (major_axis - separation_axis)),
                -1.0,
                1.0,
            )
    return feature.astype(np.float32)


def _json_candidate_ids(
    values: Sequence[Any] | np.ndarray[Any, np.dtype[Any]],
) -> list[int | str]:
    return [
        value.item() if isinstance(value, np.generic) else value
        for value in values
    ]


def _canonical_feature_definitions() -> list[XFitFeatureDefinition]:
    return [
        {
            "index": index,
            "name": name,
            "range": list(
                cast(Sequence[float], FEATURE_TRANSFORMS[name]["range"])
            ),
            "transform": cast(str, FEATURE_TRANSFORMS[name]["transform"]),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]


def _validate_feature_values(
    values: np.ndarray, *, variance_present: bool, model: XFitModel
) -> None:
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    for name in FEATURE_NAMES:
        bounds = cast(Sequence[float], FEATURE_TRANSFORMS[name]["range"])
        lower, upper = float(bounds[0]), float(bounds[1])
        column = values[:, index[name]]
        if np.any((column < lower) | (column > upper)):
            raise ValueError(
                f"xFit feature {name} is outside its declared range "
                f"[{lower}, {upper}]"
            )
    for name in _BINARY_FEATURE_NAMES:
        column = values[:, index[name]]
        if not np.all((column == 0.0) | (column == 1.0)):
            raise ValueError(f"xFit feature {name} must be binary")
    fit_present = values[:, index["fit_present"]]
    fit_valid = values[:, index["fit_valid"]]
    uncertainty_valid = values[:, index["uncertainty_valid"]]
    gaussian_shape = values[:, index["gaussian_shape_available"]]
    if np.any(uncertainty_valid > fit_valid):
        raise ValueError("uncertainty_valid must not exceed fit_valid")
    if np.any(fit_valid > fit_present):
        raise ValueError("fit_valid must not exceed fit_present")
    if np.any(gaussian_shape > fit_valid):
        raise ValueError("gaussian_shape_available must not exceed fit_valid")
    absent = fit_present == 0.0
    if any(np.any(values[absent, index[name]]) for name in FEATURE_NAMES):
        raise ValueError("absent xFit rows must be all-zero feature vectors")
    expected_variance_weighted = fit_present * float(variance_present)
    if not np.array_equal(
        values[:, index["variance_weighted"]], expected_variance_weighted
    ):
        raise ValueError(
            "variance_weighted does not match the source variance contract"
        )
    if not variance_present and any(
        np.any(values[:, index[name]])
        for name in ("log_delta_chi_square", "log_reduced_chi_square")
    ):
        raise ValueError(
            "variance-only features must be zero for an unweighted source"
        )
    invalid = fit_valid == 0.0
    if any(
        np.any(values[invalid, index[name]])
        for name in _PARAMETER_FEATURE_NAMES
    ):
        raise ValueError("invalid xFit rows must zero parameter features")
    uncertainty_missing = uncertainty_valid == 0.0
    uncertainty_names = ("log_strength_snr", "separation_significance")
    if any(
        np.any(values[uncertainty_missing, index[name]])
        for name in uncertainty_names
    ):
        raise ValueError(
            "rows without valid uncertainty must zero uncertainty features"
        )
    shape_missing = gaussian_shape == 0.0
    shape_names = ("size_over_stamp", "axis_ratio", "aligned_ellipticity")
    if any(
        np.any(values[shape_missing, index[name]]) for name in shape_names
    ):
        raise ValueError(
            "rows without Gaussian shape must zero Gaussian shape features"
        )
    expected_gaussian_shape = (
        fit_valid if model == "gaussian" else np.zeros_like(fit_valid)
    )
    if not np.array_equal(gaussian_shape, expected_gaussian_shape):
        raise ValueError(
            "gaussian_shape_available does not match the source model"
        )


def _write_npy_member(
    archive: zipfile.ZipFile, name: str, values: np.ndarray
) -> None:
    with archive.open(name, mode="w", force_zip64=True) as member:
        np.lib.format.write_array(member, values, allow_pickle=False)


def _write_indexed_image_member(
    archive: zipfile.ZipFile,
    difference: np.ndarray,
    row_indices: Sequence[int],
) -> None:
    output_shape = (len(row_indices), *difference.shape[1:])
    header = {
        "descr": np.lib.format.dtype_to_descr(difference.dtype),
        "fortran_order": False,
        "shape": output_shape,
    }
    bytes_per_row = max(
        int(np.prod(difference.shape[1:], dtype=np.int64))
        * difference.dtype.itemsize,
        1,
    )
    batch_rows = max(1, _EXPORT_BATCH_BYTES // bytes_per_row)
    with archive.open("images.npy", mode="w", force_zip64=True) as member:
        np.lib.format.write_array_header_2_0(member, header)
        for start in range(0, len(row_indices), batch_rows):
            selected_rows = row_indices[start : start + batch_rows]
            batch = np.ascontiguousarray(difference[selected_rows])
            member.write(memoryview(batch).cast("B"))


def _write_xfit_input_archive(
    path: Path,
    *,
    candidate_id: np.ndarray,
    difference: np.ndarray,
    row_indices: Sequence[int],
) -> None:
    created = False
    try:
        archive = zipfile.ZipFile(
            path,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        )
        created = True
        with archive:
            _write_npy_member(archive, "candidate_id.npy", candidate_id)
            _write_indexed_image_member(archive, difference, row_indices)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def export_xfit_input(
    *,
    dataset_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export exact, unique stamps in a pickle-free xFit NPZ."""

    dataset_root = Path(dataset_dir).expanduser().resolve()
    difference_path = dataset_root / "difference.npy"
    if not difference_path.is_file():
        raise FileNotFoundError(
            "XScan xFit export requires dataset difference.npy"
        )
    try:
        difference = np.load(
            difference_path, mmap_mode="r", allow_pickle=False
        )
    except ValueError as exc:
        raise ValueError(
            "XScan difference.npy must be a pickle-free numeric array"
        ) from exc
    if difference.ndim != 3 or difference.dtype.kind != "f":
        raise ValueError(
            "XScan difference.npy must be a floating array with shape "
            "(sample, y, x)"
        )
    dataset = _load_dataset_contract(
        dataset_root,
        image_shape=(int(difference.shape[1]), int(difference.shape[2])),
    )
    for row_index in range(difference.shape[0]):
        if not np.isfinite(difference[row_index]).all():
            raise ValueError(
                "XScan difference.npy contains non-finite pixels at row "
                f"{row_index}; construct a masked xFit input explicitly"
            )
    keys = [_candidate_key(value) for value in dataset.candidate_id]
    unique_rows: list[int] = []
    seen: set[tuple[str, int | str]] = set()
    for row_index, key in enumerate(keys):
        if key not in seen:
            unique_rows.append(row_index)
            seen.add(key)
    candidate_id = np.asarray(dataset.candidate_id[unique_rows])

    resolved_output = Path(output_path).expanduser().resolve()
    if resolved_output.suffix.lower() != ".npz":
        raise ValueError("xFit input output path must end in .npz")
    if resolved_output.exists():
        raise FileExistsError(
            f"xFit input output already exists: {resolved_output}"
        )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    _write_xfit_input_archive(
        resolved_output,
        candidate_id=candidate_id,
        difference=difference,
        row_indices=unique_rows,
    )
    images_shape = (len(unique_rows), *difference.shape[1:])
    return {
        "schema_version": 1,
        "workflow": "export-xfit-input",
        "dataset_dir": str(dataset_root),
        "output": str(resolved_output),
        "dataset_row_count": int(dataset.candidate_id.shape[0]),
        "candidate_count": int(candidate_id.shape[0]),
        "reused_dataset_row_count": int(
            dataset.candidate_id.shape[0] - candidate_id.shape[0]
        ),
        "images_shape": list(images_shape),
        "images_dtype": str(difference.dtype),
        "input_archive_sha256": file_sha256(resolved_output),
    }


def build_xfit_feature_bundle(
    *,
    dataset_dir: str | Path,
    xfit_run_dir: str | Path,
    output_dir: str | Path,
    missing_policy: MissingPolicy = "error",
) -> XFitFeatureSchema:
    """Build finite, pickle-free xFit features in XScan row order."""

    if missing_policy not in {"error", "indicator"}:
        raise ValueError("missing_policy must be 'error' or 'indicator'")
    dataset_root = Path(dataset_dir).expanduser().resolve()
    run_root = Path(xfit_run_dir).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(
            f"xFit run directory does not exist: {run_root}"
        )
    summary_path = run_root / "summary.json"
    summary = _load_json_object(summary_path, description="xFit summary.json")
    schema_version = summary.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SOURCE_XFIT_SCHEMA_VERSION
    ):
        raise ValueError(
            "xFit summary schema_version is unsupported; expected version 1"
        )
    run_artifacts = {
        key: _resolve_run_artifact(run_root, summary, key, name)
        for key, name in _SUMMARY_ARTIFACT_NAMES.items()
    }
    verified_artifact_sha256 = {
        key: _verify_run_artifact(summary, key, path)
        for key, path in run_artifacts.items()
    }
    fits_path = run_artifacts["fits"]
    arrays_path = run_artifacts["fit_arrays"]
    parquet = pq.ParquetFile(fits_path)
    fit_row_count = int(parquet.metadata.num_rows)
    if fit_row_count == 0:
        raise ValueError("xFit fits.parquet contains no rows")
    contract = _summary_contract(summary, fit_row_count)
    required_columns = {
        "candidate_index",
        "candidate_id",
        "input_image_sha256",
        "converged",
        "degrees_of_freedom",
        "reduced_chi_square",
        "uncertainty_valid",
        *_REQUIRED_FIT_STATISTICS,
        *contract.parameter_names,
        *(f"{name}_standard_error" for name in contract.parameter_names),
    }
    parquet_columns = set(parquet.schema_arrow.names)
    missing_columns = sorted(required_columns - parquet_columns)
    if missing_columns:
        raise ValueError(
            "xFit run lacks portable feature statistics; rerun xFit with a "
            "version that writes: " + ", ".join(missing_columns)
        )
    fit_arrays = _load_fit_arrays(
        arrays_path,
        row_count=fit_row_count,
        parameter_count=len(contract.parameter_names),
        residual_shape=contract.residual_shape,
        residual_dtype=contract.residual_dtype,
    )
    fit_features = np.zeros(
        (fit_row_count + 1, len(FEATURE_NAMES)), dtype=np.float32
    )
    fit_indices = _load_fit_indices(parquet, fit_row_count)
    if not np.array_equal(
        np.sort(fit_indices), fit_arrays.sorted_candidate_index
    ):
        raise ValueError(
            "fits.parquet and fit-arrays.npz candidate_index values must "
            "form an exact bijection"
        )
    fit_input_sha256 = np.empty(fit_row_count, dtype="S64")
    fit_candidate_id = np.empty(
        fit_row_count, dtype=_fit_candidate_dtype(parquet)
    )
    selected_columns = sorted(
        (required_columns - {"candidate_index"})
        | ({"model", "mode"} & parquet_columns)
    )
    fit_row_offset = 0
    for batch in parquet.iter_batches(
        batch_size=_PARQUET_BATCH_ROWS,
        columns=selected_columns,
    ):
        columns = {
            name: _arrow_values(batch.column(position))
            for position, name in enumerate(batch.schema.names)
        }
        batch_candidate_id = _candidate_array(
            columns["candidate_id"], allow_duplicates=True
        )
        if batch_candidate_id.dtype.kind != fit_candidate_id.dtype.kind:
            raise ValueError(
                "fits.parquet candidate_id values have inconsistent types"
            )
        fit_candidate_id[fit_row_offset : fit_row_offset + batch.num_rows] = (
            batch_candidate_id
        )
        for batch_row in range(batch.num_rows):
            fit_row = fit_row_offset + batch_row
            candidate_index = int(fit_indices[fit_row])
            array_row = _fit_array_row(fit_arrays, candidate_index)
            key = _candidate_key(columns["candidate_id"][batch_row])
            if _candidate_key(fit_arrays.candidate_id[array_row]) != key:
                raise ValueError(
                    "fit-arrays.npz candidate_id is not aligned with "
                    "fits.parquet candidate_index"
                )
            if _column_value(columns, batch_row, "model") not in {
                None,
                contract.model,
            } or _column_value(columns, batch_row, "mode") not in {
                None,
                "difference",
            }:
                raise ValueError(
                    "fits.parquet model or mode conflicts with summary.json"
                )
            fit_features[fit_row] = _feature_row(
                columns,
                batch_row,
                fit_arrays.covariance[array_row],
                model=contract.model,
                parameter_names=contract.parameter_names,
                image_shape=contract.image_shape,
                variance_present=contract.variance_present,
            )
            fit_input_sha256[fit_row] = _required_sha256(
                _column_value(columns, batch_row, "input_image_sha256"),
                description="fits.parquet input_image_sha256",
            ).encode("ascii")
        fit_row_offset += batch.num_rows
    if fit_row_offset != fit_row_count:
        raise ValueError("fits.parquet row count changed while reading")
    if np.unique(fit_candidate_id).shape[0] != fit_row_count:
        raise ValueError(
            "xFit fits.parquet contains duplicate candidate_id values"
        )
    if not np.isfinite(fit_features[:fit_row_count]).all():
        raise ValueError("derived xFit feature values are not finite")

    dataset = _load_dataset_contract(
        dataset_root, image_shape=contract.image_shape
    )
    dataset_candidate_id = dataset.candidate_id
    fit_row_by_dataset = np.full(
        dataset_candidate_id.shape[0], fit_row_count, dtype=np.int64
    )
    matched = np.zeros(dataset_candidate_id.shape[0], dtype=bool)
    if dataset_candidate_id.dtype.kind == fit_candidate_id.dtype.kind:
        fit_order = np.argsort(fit_candidate_id, kind="stable")
        sorted_fit_candidate_id = fit_candidate_id[fit_order]
        positions = np.searchsorted(
            sorted_fit_candidate_id, dataset_candidate_id
        )
        within = positions < fit_row_count
        probe = np.minimum(positions, fit_row_count - 1)
        matched = within & (
            sorted_fit_candidate_id[probe] == dataset_candidate_id
        )
        fit_row_by_dataset[matched] = fit_order[positions[matched]]
    missing_candidate_id = np.unique(dataset_candidate_id[~matched])
    if missing_candidate_id.size and missing_policy == "error":
        raise ValueError(
            "xFit run is missing XScan candidate_id value(s): "
            + ", ".join(str(value) for value in missing_candidate_id[:8])
        )
    matched_dataset_rows = np.flatnonzero(matched)
    for start in range(0, matched_dataset_rows.size, _PARQUET_BATCH_ROWS):
        rows = matched_dataset_rows[start : start + _PARQUET_BATCH_ROWS]
        expected = fit_input_sha256[fit_row_by_dataset[rows]]
        mismatched = rows[dataset.image_sha256[rows] != expected]
        if mismatched.size:
            row_index = int(mismatched[0])
            raise ValueError(
                "xFit input_image_sha256 does not match the XScan "
                "difference.npy stamp for candidate_id "
                f"{dataset_candidate_id[row_index]!r}"
            )
    fit_matched = np.zeros(fit_row_count, dtype=bool)
    fit_matched[fit_row_by_dataset[matched]] = True
    extra_fit_candidate_id = np.sort(fit_candidate_id[~fit_matched])
    dataset_unique_count = int(np.unique(dataset_candidate_id).shape[0])
    matched_row_count = int(np.count_nonzero(matched))
    matched_unique_count = int(
        np.unique(dataset_candidate_id[matched]).shape[0]
    )
    diagnostics: XFitJoinDiagnostics = {
        "dataset_row_count": int(dataset_candidate_id.shape[0]),
        "dataset_unique_candidate_count": dataset_unique_count,
        "duplicate_dataset_row_count": int(dataset_candidate_id.shape[0])
        - dataset_unique_count,
        "fit_row_count": fit_row_count,
        "matched_dataset_row_count": matched_row_count,
        "missing_dataset_row_count": int(dataset_candidate_id.shape[0])
        - matched_row_count,
        "reused_fit_row_count": matched_row_count - matched_unique_count,
        "extra_fit_row_count": int(extra_fit_candidate_id.shape[0]),
        "missing_candidate_ids": _json_candidate_ids(missing_candidate_id),
        "extra_fit_candidate_ids": _json_candidate_ids(
            extra_fit_candidate_id
        ),
    }

    source_artifacts: XFitSourceArtifacts = {
        "summary": {
            "name": "summary.json",
            "sha256": file_sha256(summary_path),
        },
        "fits": {
            "name": "fits.parquet",
            "sha256": verified_artifact_sha256["fits"],
        },
        "fit_arrays": {
            "name": "fit-arrays.npz",
            "sha256": verified_artifact_sha256["fit_arrays"],
        },
    }
    output_root = Path(output_dir).expanduser().resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise FileExistsError(
            f"xFit feature output already exists: {output_root}"
        )
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", dir=output_root.parent
        )
    )
    published = False
    try:
        candidate_id_path = temporary_root / CANDIDATE_ID_ARTIFACT_NAME
        feature_path = temporary_root / FEATURE_ARTIFACT_NAME
        input_image_sha256_path = (
            temporary_root / INPUT_IMAGE_SHA256_ARTIFACT_NAME
        )
        np.save(candidate_id_path, dataset_candidate_id, allow_pickle=False)
        feature_values = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(dataset_candidate_id.shape[0], len(FEATURE_NAMES)),
        )
        for start in range(
            0, fit_row_by_dataset.shape[0], _FEATURE_WRITE_BATCH_ROWS
        ):
            stop = min(
                start + _FEATURE_WRITE_BATCH_ROWS,
                fit_row_by_dataset.shape[0],
            )
            np.take(
                fit_features,
                fit_row_by_dataset[start:stop],
                axis=0,
                out=feature_values[start:stop],
            )
        feature_values.flush()
        del feature_values
        np.save(
            input_image_sha256_path,
            np.asarray(dataset.image_sha256, dtype="S64"),
            allow_pickle=False,
        )
        schema: XFitFeatureSchema = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "artifact": BUNDLE_ARTIFACT_TYPE,
            "feature_names": list(FEATURE_NAMES),
            "feature_dtype": "float32",
            "candidate_id_dtype": str(dataset_candidate_id.dtype),
            "features": _canonical_feature_definitions(),
            "transform_constants": dict(TRANSFORM_CONSTANTS),
            "missing_policy": missing_policy,
            "source": {
                "model": contract.model,
                "mode": "difference",
                "image_shape": list(contract.image_shape),
                "mask_present": contract.mask_present,
                "variance_present": contract.variance_present,
                "input_archive_sha256": contract.input_archive_sha256,
            },
            "join_diagnostics": diagnostics,
            "source_artifacts": source_artifacts,
            "artifacts": {
                "candidate_id": {
                    "name": CANDIDATE_ID_ARTIFACT_NAME,
                    "sha256": file_sha256(candidate_id_path),
                },
                "features": {
                    "name": FEATURE_ARTIFACT_NAME,
                    "sha256": file_sha256(feature_path),
                },
                "input_image_sha256": {
                    "name": INPUT_IMAGE_SHA256_ARTIFACT_NAME,
                    "sha256": file_sha256(input_image_sha256_path),
                },
                "schema": {"name": SCHEMA_ARTIFACT_NAME},
            },
        }
        (temporary_root / SCHEMA_ARTIFACT_NAME).write_text(
            json.dumps(schema, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if output_root.exists():
            raise FileExistsError(
                f"xFit feature output already exists: {output_root}"
            )
        temporary_root.rename(output_root)
        published = True
        return schema
    finally:
        if not published:
            shutil.rmtree(temporary_root, ignore_errors=True)


def _require_exact_keys(
    value: object, expected: set[str], *, description: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{description} has invalid fields")
    return value


def _validate_named_digest(
    value: object, *, expected_name: str, description: str
) -> None:
    artifact = _require_exact_keys(
        value, {"name", "sha256"}, description=description
    )
    if artifact["name"] != expected_name:
        raise ValueError(f"{description} has an invalid name")
    _required_sha256(artifact["sha256"], description=f"{description} SHA-256")


def _validate_join_diagnostics(value: object) -> None:
    count_names = {
        "dataset_row_count",
        "dataset_unique_candidate_count",
        "duplicate_dataset_row_count",
        "fit_row_count",
        "matched_dataset_row_count",
        "missing_dataset_row_count",
        "reused_fit_row_count",
        "extra_fit_row_count",
    }
    diagnostics = _require_exact_keys(
        value,
        count_names | {"missing_candidate_ids", "extra_fit_candidate_ids"},
        description="xFit feature schema join_diagnostics",
    )
    for name in count_names:
        count = diagnostics[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"xFit feature schema join_diagnostics.{name} is invalid"
            )
    dataset_rows = diagnostics["dataset_row_count"]
    dataset_unique = diagnostics["dataset_unique_candidate_count"]
    duplicate_rows = diagnostics["duplicate_dataset_row_count"]
    fit_rows = diagnostics["fit_row_count"]
    matched_rows = diagnostics["matched_dataset_row_count"]
    missing_rows = diagnostics["missing_dataset_row_count"]
    reused_rows = diagnostics["reused_fit_row_count"]
    extra_rows = diagnostics["extra_fit_row_count"]
    if (
        dataset_unique > dataset_rows
        or duplicate_rows != dataset_rows - dataset_unique
        or matched_rows + missing_rows != dataset_rows
        or reused_rows > matched_rows
        or fit_rows != matched_rows - reused_rows + extra_rows
    ):
        raise ValueError(
            "xFit feature schema join_diagnostics counts are inconsistent"
        )
    for name, maximum, exact_count in (
        ("missing_candidate_ids", missing_rows, False),
        ("extra_fit_candidate_ids", extra_rows, True),
    ):
        candidate_ids = diagnostics[name]
        if not isinstance(candidate_ids, list):
            raise ValueError(
                f"xFit feature schema join_diagnostics.{name} is invalid"
            )
        try:
            normalized = _candidate_array(
                candidate_ids, allow_duplicates=False
            )
        except ValueError as exc:
            raise ValueError(
                f"xFit feature schema join_diagnostics.{name} is invalid"
            ) from exc
        if normalized.shape[0] > maximum or (
            exact_count and normalized.shape[0] != maximum
        ):
            raise ValueError(
                f"xFit feature schema join_diagnostics.{name} is invalid"
            )


def _validate_canonical_schema(raw: dict[str, Any]) -> XFitFeatureSchema:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "artifact",
            "feature_names",
            "feature_dtype",
            "candidate_id_dtype",
            "features",
            "transform_constants",
            "missing_policy",
            "source",
            "join_diagnostics",
            "source_artifacts",
            "artifacts",
        },
        description="xFit feature schema",
    )
    schema_version = raw.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != FEATURE_SCHEMA_VERSION
    ):
        raise ValueError("xFit feature schema_version must be 1")
    if raw.get("artifact") != BUNDLE_ARTIFACT_TYPE:
        raise ValueError("xFit feature schema artifact type is invalid")
    if raw.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError(
            "xFit feature schema does not match the v1 feature contract"
        )
    if raw.get("feature_dtype") != "float32":
        raise ValueError("xFit feature schema feature_dtype must be float32")
    if not isinstance(raw.get("candidate_id_dtype"), str):
        raise ValueError("xFit feature schema candidate_id_dtype is invalid")
    canonical_definitions = _canonical_feature_definitions()
    definitions = raw.get("features")
    definitions_normalized = isinstance(definitions, list) and all(
        isinstance(definition, dict)
        and set(definition) == {"index", "name", "range", "transform"}
        and type(definition["index"]) is int
        and isinstance(definition["name"], str)
        and isinstance(definition["range"], list)
        and all(type(value) is float for value in definition["range"])
        and isinstance(definition["transform"], str)
        for definition in definitions
    )
    if not definitions_normalized or definitions != canonical_definitions:
        raise ValueError(
            "xFit feature schema feature definitions do not match "
            "the canonical v1 contract"
        )
    constants = raw.get("transform_constants")
    if (
        not isinstance(constants, dict)
        or constants != dict(TRANSFORM_CONSTANTS)
        or not all(
            isinstance(value, float) and math.isfinite(value)
            for value in constants.values()
        )
    ):
        raise ValueError(
            "xFit feature schema transform_constants do not match "
            "the canonical v1 contract"
        )
    if raw.get("missing_policy") not in {"error", "indicator"}:
        raise ValueError("xFit feature schema missing_policy is invalid")

    source = _require_exact_keys(
        raw.get("source"),
        {
            "model",
            "mode",
            "image_shape",
            "mask_present",
            "variance_present",
            "input_archive_sha256",
        },
        description="xFit feature schema source",
    )
    if source["model"] not in {"gaussian", "stamp"}:
        raise ValueError("xFit feature schema source.model is invalid")
    if source["mode"] != "difference":
        raise ValueError("xFit feature schema source.mode is invalid")
    image_shape = source["image_shape"]
    if (
        not isinstance(image_shape, list)
        or len(image_shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in image_shape
        )
    ):
        raise ValueError("xFit feature schema source.image_shape is invalid")
    for name in ("mask_present", "variance_present"):
        if not isinstance(source[name], bool):
            raise ValueError(f"xFit feature schema source.{name} is invalid")
    _required_sha256(
        source["input_archive_sha256"],
        description="xFit feature schema source.input_archive_sha256",
    )

    _validate_join_diagnostics(raw.get("join_diagnostics"))
    source_artifacts = _require_exact_keys(
        raw.get("source_artifacts"),
        {"summary", "fits", "fit_arrays"},
        description="xFit feature schema source_artifacts",
    )
    for key, expected_name in (
        ("summary", "summary.json"),
        ("fits", "fits.parquet"),
        ("fit_arrays", "fit-arrays.npz"),
    ):
        _validate_named_digest(
            source_artifacts[key],
            expected_name=expected_name,
            description=f"xFit feature schema source_artifacts.{key}",
        )
    artifacts = _require_exact_keys(
        raw.get("artifacts"),
        {"candidate_id", "features", "input_image_sha256", "schema"},
        description="xFit feature schema artifacts",
    )
    for key, expected_name in (
        ("candidate_id", CANDIDATE_ID_ARTIFACT_NAME),
        ("features", FEATURE_ARTIFACT_NAME),
        ("input_image_sha256", INPUT_IMAGE_SHA256_ARTIFACT_NAME),
    ):
        _validate_named_digest(
            artifacts[key],
            expected_name=expected_name,
            description=f"xFit feature schema artifacts.{key}",
        )
    schema_artifact = _require_exact_keys(
        artifacts["schema"],
        {"name"},
        description="xFit feature schema artifacts.schema",
    )
    if schema_artifact["name"] != SCHEMA_ARTIFACT_NAME:
        raise ValueError(
            "xFit feature schema names an invalid schema artifact"
        )
    return cast(XFitFeatureSchema, raw)


def _load_bundle_array(
    feature_root: Path,
    schema: XFitFeatureSchema,
    key: str,
) -> tuple[np.ndarray, str]:
    artifact = schema["artifacts"][key]
    artifact_name = artifact["name"]
    expected_sha256 = artifact["sha256"]
    path = (feature_root / artifact_name).resolve()
    try:
        path.relative_to(feature_root)
    except ValueError as exc:
        raise ValueError(
            f"xFit feature artifact {key} escapes feature_dir"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"xFit feature artifact does not exist: {path}"
        )
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"xFit feature artifact {key} SHA-256 mismatch")
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                f"xFit feature artifact {key} is pickle-backed"
            ) from exc
        raise
    if not isinstance(values, np.ndarray):
        values.close()
        raise ValueError(f"xFit feature artifact {key} must be an NPY array")
    return values, expected_sha256


def load_xfit_feature_matrix(
    *,
    dataset_dir: str | Path,
    feature_dir: str | Path,
    expected_feature_names: Sequence[str] | None = None,
) -> XFitFeatureMatrix:
    """Load and validate one feature bundle against XScan metadata order."""

    feature_root = Path(feature_dir).expanduser().resolve()
    schema_path = feature_root / SCHEMA_ARTIFACT_NAME
    schema = _validate_canonical_schema(
        _load_json_object(
            schema_path,
            description="xFit feature schema.json",
        )
    )
    feature_names = tuple(schema["feature_names"])
    if (
        expected_feature_names is not None
        and tuple(expected_feature_names) != feature_names
    ):
        raise ValueError(
            "xFit feature names do not match expected_feature_names"
        )
    image_shape_value = schema["source"]["image_shape"]
    dataset_root = Path(dataset_dir).expanduser().resolve()
    dataset = _load_dataset_contract(
        dataset_root,
        image_shape=(int(image_shape_value[0]), int(image_shape_value[1])),
    )
    dataset_candidate_id = dataset.candidate_id
    candidate_id_raw, _ = _load_bundle_array(
        feature_root, schema, "candidate_id"
    )
    values, feature_sha256 = _load_bundle_array(
        feature_root, schema, "features"
    )
    input_image_sha256, _ = _load_bundle_array(
        feature_root, schema, "input_image_sha256"
    )
    if str(candidate_id_raw.dtype) != schema["candidate_id_dtype"]:
        raise ValueError(
            "xFit feature candidate_id dtype does not match schema.json"
        )
    candidate_id = _candidate_array(candidate_id_raw, allow_duplicates=True)
    if (
        candidate_id.shape != dataset_candidate_id.shape
        or candidate_id.dtype.kind != dataset_candidate_id.dtype.kind
        or not np.array_equal(candidate_id, dataset_candidate_id)
    ):
        raise ValueError(
            "xFit feature candidate_id order does not match XScan metadata"
        )
    if (
        input_image_sha256.shape != candidate_id.shape
        or input_image_sha256.dtype != np.dtype("S64")
    ):
        raise ValueError(
            "xFit feature input_image_sha256 must be an S64 row vector"
        )
    for row_index in range(input_image_sha256.shape[0]):
        recorded = _ascii_sha256(
            input_image_sha256[row_index],
            description="xFit feature input_image_sha256",
        )
        if recorded.encode("ascii") != dataset.image_sha256[row_index]:
            raise ValueError(
                "xFit feature bundle does not match the current XScan "
                "difference.npy stamp at row "
                f"{row_index}"
            )
    if values.dtype != np.dtype(np.float32):
        raise ValueError("xFit feature matrix must have dtype float32")
    expected_shape = (candidate_id.shape[0], len(feature_names))
    if values.shape != expected_shape:
        raise ValueError(
            "xFit feature matrix must have shape "
            f"{expected_shape}, got {values.shape}"
        )
    if any(
        not np.isfinite(values[:, column]).all()
        for column in range(values.shape[1])
    ):
        raise ValueError(
            "xFit feature matrix must contain only finite values"
        )
    _validate_feature_values(
        values,
        variance_present=schema["source"]["variance_present"],
        model=schema["source"]["model"],
    )
    diagnostics = schema["join_diagnostics"]
    fit_present = values[:, FEATURE_NAMES.index("fit_present")] == 1.0
    dataset_unique_count = int(np.unique(dataset_candidate_id).shape[0])
    matched_row_count = int(np.count_nonzero(fit_present))
    matched_unique_count = int(
        np.unique(dataset_candidate_id[fit_present]).shape[0]
    )
    missing_candidate_id = np.unique(dataset_candidate_id[~fit_present])
    recorded_missing_candidate_id = _candidate_array(
        diagnostics["missing_candidate_ids"], allow_duplicates=False
    )
    missing_ids_match = (
        missing_candidate_id.shape == recorded_missing_candidate_id.shape
        and (
            missing_candidate_id.size == 0
            or (
                missing_candidate_id.dtype.kind
                == recorded_missing_candidate_id.dtype.kind
                and np.array_equal(
                    missing_candidate_id, recorded_missing_candidate_id
                )
            )
        )
    )
    expected_reused_rows = matched_row_count - matched_unique_count
    expected_fit_rows = (
        matched_unique_count + diagnostics["extra_fit_row_count"]
    )
    if (
        diagnostics["dataset_row_count"] != candidate_id.shape[0]
        or diagnostics["dataset_unique_candidate_count"]
        != dataset_unique_count
        or diagnostics["duplicate_dataset_row_count"]
        != candidate_id.shape[0] - dataset_unique_count
        or diagnostics["matched_dataset_row_count"] != matched_row_count
        or diagnostics["missing_dataset_row_count"]
        != candidate_id.shape[0] - matched_row_count
        or diagnostics["reused_fit_row_count"] != expected_reused_rows
        or diagnostics["fit_row_count"] != expected_fit_rows
        or len(diagnostics["extra_fit_candidate_ids"])
        != diagnostics["extra_fit_row_count"]
        or not missing_ids_match
    ):
        raise ValueError(
            "xFit feature schema join_diagnostics do not match feature rows"
        )
    if schema["missing_policy"] == "error" and not fit_present.all():
        raise ValueError(
            "xFit feature schema error missing_policy contains absent fits"
        )
    bundle_identity: XFitBundleIdentity = {
        "schema_sha256": file_sha256(schema_path),
        "feature_sha256": feature_sha256,
        "source_artifacts": schema["source_artifacts"],
    }
    return XFitFeatureMatrix(
        dataset_dir=dataset_root,
        candidate_id=candidate_id,
        values=values,
        feature_names=feature_names,
        schema=schema,
        bundle_identity=bundle_identity,
        join_diagnostics=schema["join_diagnostics"],
    )


__all__ = [
    "BUNDLE_ARTIFACT_TYPE",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_TRANSFORMS",
    "TRANSFORM_CONSTANTS",
    "XFitFeatureMatrix",
    "build_xfit_feature_bundle",
    "export_xfit_input",
    "load_xfit_feature_matrix",
]
