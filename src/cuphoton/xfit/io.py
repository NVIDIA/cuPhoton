# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Safe input validation and artifact persistence for xFit."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cuphoton import __version__
from cuphoton.core.artifacts import array_sha256, file_sha256

from ._types import FIT_MODES, MODEL_NAMES, FitMode, ModelName
from .models import GaussianDipoleModel, StampDipoleModel

if TYPE_CHECKING:
    from .api import DipoleFitResult

REQUIRED_KEYS = frozenset({"candidate_id", "images"})
OPTIONAL_KEYS = frozenset({"initial", "mask", "variance", "stamp_basis"})
ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS
PARAMETER_COUNTS: dict[ModelName, int] = {
    GaussianDipoleModel.name: len(GaussianDipoleModel.parameter_names),
    StampDipoleModel.name: len(StampDipoleModel.parameter_names),
}


@dataclass(frozen=True, slots=True)
class XFitDataset:
    """Validated, pickle-free input arrays for one xFit batch."""

    path: Path
    candidate_id: np.ndarray
    images: np.ndarray
    input_archive_sha256: str
    initial: np.ndarray | None = None
    mask: np.ndarray | None = None
    variance: np.ndarray | None = None
    stamp_basis: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    @property
    def inferred_mode(self) -> FitMode:
        return "split" if self.images.ndim == 4 else "difference"


def _require_npz_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".npz":
        raise ValueError(
            "xFit accepts .npz inputs only; .npy and pickle-backed inputs "
            "are not supported"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"xFit input does not exist: {resolved}")
    return resolved


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = sorted(REQUIRED_KEYS - keys)
            if missing:
                raise ValueError(
                    "xFit input is missing required array(s): "
                    + ", ".join(missing)
                )
            unknown = sorted(keys - ALLOWED_KEYS)
            if unknown:
                raise ValueError(
                    "xFit input contains unsupported array(s): "
                    + ", ".join(unknown)
                )
            # Access every member while allow_pickle=False is active. An
            # object member in an otherwise valid ZIP is rejected here.
            arrays = {key: np.asarray(archive[key]) for key in sorted(keys)}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                "xFit input contains an object or pickle-backed array"
            ) from exc
        raise
    return arrays


def _validate_candidate_id(candidate_id: np.ndarray, batch_size: int) -> None:
    if candidate_id.ndim != 1:
        raise ValueError("candidate_id must be a one-dimensional array")
    if candidate_id.shape[0] != batch_size:
        raise ValueError(
            "candidate_id length must match the images batch dimension"
        )
    if candidate_id.dtype.kind not in "iuU":
        raise ValueError(
            "candidate_id must contain integers or Unicode strings"
        )
    if candidate_id.dtype.kind == "u" and np.any(
        candidate_id > np.iinfo(np.int64).max
    ):
        raise ValueError("integer candidate_id values must fit signed 64-bit")
    if candidate_id.dtype.kind == "U":
        text = candidate_id.astype(str)
        if np.any(np.char.str_len(text) == 0):
            raise ValueError("candidate_id values must not be empty")
    if np.unique(candidate_id).size != candidate_id.size:
        raise ValueError("candidate_id values must be unique")


def _validate_real_array(
    name: str,
    array: np.ndarray,
    *,
    finite: bool = True,
    allow_bool: bool = False,
) -> None:
    valid_kinds = "fiub" if allow_bool else "fiu"
    if array.dtype.kind not in valid_kinds:
        raise ValueError(f"{name} must be a real numeric array")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_images(images: np.ndarray) -> None:
    _validate_real_array("images", images, finite=False)
    if images.ndim == 3:
        if images.shape[1] < 1 or images.shape[2] < 1:
            raise ValueError("images must have non-empty spatial dimensions")
        return
    if images.ndim == 4 and images.shape[1] == 3:
        if images.shape[2] < 1 or images.shape[3] < 1:
            raise ValueError("images must have non-empty spatial dimensions")
        return
    raise ValueError(
        "images must have shape (batch, y, x) for difference mode or "
        "(batch, 3, y, x) for split mode"
    )


def _validate_image_auxiliary(
    name: str,
    array: np.ndarray,
    images: np.ndarray,
    *,
    finite: bool = True,
    allow_bool: bool = False,
) -> None:
    _validate_real_array(
        name,
        array,
        finite=finite,
        allow_bool=allow_bool,
    )
    spatial_shape = images.shape[-2:]
    allowed_shapes = {(), spatial_shape, images.shape}
    if images.ndim == 4:
        allowed_shapes.add((images.shape[0], *spatial_shape))
        allowed_shapes.add((images.shape[1], *spatial_shape))
        allowed_shapes.add((1, images.shape[1], *spatial_shape))
    if array.shape not in allowed_shapes:
        expected = " or ".join(str(shape) for shape in sorted(allowed_shapes))
        raise ValueError(f"{name} must have shape {expected}")


def _broadcast_image_auxiliary(
    array: np.ndarray,
    images: np.ndarray,
) -> np.ndarray:
    # With a three-row split batch, (3, H, W) means per-candidate. Callers
    # that need per-plane values can disambiguate with (1, 3, H, W).
    if images.ndim == 4 and array.shape == (
        images.shape[0],
        *images.shape[-2:],
    ):
        array = array[:, None, :, :]
    return np.broadcast_to(array, images.shape)


def _validate_dataset_contract(
    dataset: XFitDataset,
    *,
    mode: FitMode | None,
    model: ModelName | None,
) -> None:
    _validate_images(dataset.images)
    _validate_candidate_id(dataset.candidate_id, dataset.batch_size)
    if dataset.batch_size == 0:
        raise ValueError("images must contain at least one candidate")

    if mode is not None:
        if mode not in FIT_MODES:
            raise ValueError("mode must be 'difference' or 'split'")
        if mode != dataset.inferred_mode:
            raise ValueError(
                f"mode {mode!r} does not match images with inferred mode "
                f"{dataset.inferred_mode!r}"
            )

    if dataset.mask is not None:
        _validate_image_auxiliary(
            "mask",
            dataset.mask,
            dataset.images,
            allow_bool=True,
        )
    included = (
        np.ones(dataset.images.shape, dtype=bool)
        if dataset.mask is None
        else _broadcast_image_auxiliary(
            dataset.mask,
            dataset.images,
        ).astype(bool)
    )
    if np.any(included & ~np.isfinite(dataset.images)):
        raise ValueError("images must be finite at included pixels")
    if dataset.variance is not None:
        _validate_image_auxiliary(
            "variance",
            dataset.variance,
            dataset.images,
            finite=False,
        )
        variance = _broadcast_image_auxiliary(
            dataset.variance,
            dataset.images,
        )
        invalid = ~np.isfinite(variance) | (variance <= 0)
        if np.any(included & invalid):
            raise ValueError(
                "variance must be finite and strictly positive at included "
                "pixels"
            )

    if dataset.initial is not None:
        _validate_real_array("initial", dataset.initial)
        if dataset.initial.ndim != 2:
            raise ValueError("initial must have shape (batch, parameters)")
        if dataset.initial.shape[0] != dataset.batch_size:
            raise ValueError(
                "initial batch dimension must match the images batch "
                "dimension"
            )

    if dataset.stamp_basis is not None:
        _validate_real_array("stamp_basis", dataset.stamp_basis)
        basis = dataset.stamp_basis
        if basis.ndim == 3 and basis.shape[0] != 1:
            raise ValueError(
                "stamp_basis may contain only one mode in CLI inputs; use "
                "the Python API to provide explicit basis_weights"
            )
        if basis.ndim not in {2, 3}:
            raise ValueError(
                "stamp_basis must have shape (y, x) or (1, y, x)"
            )
        if basis.shape[-2] < 1 or basis.shape[-1] < 1:
            raise ValueError("stamp_basis must have non-empty dimensions")

    if model is None:
        return
    if model not in MODEL_NAMES:
        raise ValueError("model must be 'stamp' or 'gaussian'")
    if dataset.initial is not None:
        expected = PARAMETER_COUNTS[model]
        if dataset.initial.shape[1] != expected:
            raise ValueError(
                f"initial must contain {expected} parameters for "
                f"{model} model"
            )
    if model == "stamp" and dataset.stamp_basis is None:
        raise ValueError("stamp model requires a stamp_basis array")
    if model == "gaussian" and dataset.stamp_basis is not None:
        raise ValueError("stamp_basis is not used by the gaussian model")


def load_xfit_dataset(
    path: str | Path,
    *,
    mode: FitMode | None = None,
    model: ModelName | None = None,
) -> XFitDataset:
    """Load one pickle-free xFit ``.npz`` of numeric or Unicode arrays."""

    resolved = _require_npz_path(path)
    input_archive_sha256 = file_sha256(resolved)
    arrays = _load_arrays(resolved)
    dataset = XFitDataset(
        path=resolved,
        candidate_id=arrays["candidate_id"],
        images=arrays["images"],
        input_archive_sha256=input_archive_sha256,
        initial=arrays.get("initial"),
        mask=arrays.get("mask"),
        variance=arrays.get("variance"),
        stamp_basis=arrays.get("stamp_basis"),
    )
    _validate_dataset_contract(dataset, mode=mode, model=model)
    return dataset


def _array_description(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def inspect_xfit_dataset(dataset: XFitDataset) -> dict[str, Any]:
    """Return a deterministic, JSON-compatible dataset inventory."""

    arrays: dict[str, Any] = {
        "candidate_id": _array_description(dataset.candidate_id),
        "images": _array_description(dataset.images),
    }
    for name in sorted(OPTIONAL_KEYS):
        value = getattr(dataset, name)
        if value is not None:
            arrays[name] = _array_description(value)
    return {
        "schema_version": 1,
        "input": str(dataset.path),
        "batch_size": dataset.batch_size,
        "mode": dataset.inferred_mode,
        "arrays": arrays,
    }


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _status_text(value: Any) -> str:
    value = _python_scalar(value)
    return str(getattr(value, "value", value))


def _finite_float_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _fit_rows(
    candidate_id: np.ndarray,
    images: np.ndarray,
    result: DipoleFitResult,
) -> list[dict[str, Any]]:
    parameter_names = tuple(str(name) for name in result.parameter_names)
    parameters = np.asarray(result.parameters)
    standard_errors = np.asarray(result.standard_errors)
    batch_size = candidate_id.shape[0]
    if images.shape[0] != batch_size:
        raise ValueError(
            "dataset images have an inconsistent batch dimension"
        )
    if parameters.shape != (batch_size, len(parameter_names)):
        raise ValueError("fit result parameters have an inconsistent shape")
    if standard_errors.shape != parameters.shape:
        raise ValueError(
            "fit result standard_errors have an inconsistent shape"
        )

    per_candidate_fields = {
        "status": np.asarray(result.status),
        "converged": np.asarray(result.converged),
        "evaluations": np.asarray(result.evaluations),
        "residual_norm": np.asarray(result.residual_norm),
        "chi_square": np.asarray(result.chi_square),
        "valid_pixel_count": np.asarray(result.valid_pixel_count),
        "valid_pixel_fraction": np.asarray(result.valid_pixel_fraction),
        "null_chi_square": np.asarray(result.null_chi_square),
        "delta_chi_square": np.asarray(result.delta_chi_square),
        "fractional_null_improvement": np.asarray(
            result.fractional_null_improvement
        ),
        "degrees_of_freedom": np.asarray(result.degrees_of_freedom),
        "reduced_chi_square": np.asarray(result.reduced_chi_square),
        "uncertainty_valid": np.asarray(result.uncertainty_valid),
        "uncertainty_reason": np.asarray(
            result.uncertainty_reason,
            dtype=str,
        ),
    }
    for name, values in per_candidate_fields.items():
        if values.shape != (batch_size,):
            raise ValueError(f"fit result {name} has an inconsistent shape")

    rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        row: dict[str, Any] = {
            "candidate_index": index,
            "candidate_id": _python_scalar(candidate_id[index]),
            "input_image_sha256": array_sha256(images[index]),
            "model": str(result.model),
            "mode": str(result.mode),
            "status": _status_text(per_candidate_fields["status"][index]),
            "converged": bool(per_candidate_fields["converged"][index]),
            "evaluations": int(per_candidate_fields["evaluations"][index]),
            "residual_norm": float(
                per_candidate_fields["residual_norm"][index]
            ),
            "chi_square": float(per_candidate_fields["chi_square"][index]),
            "valid_pixel_count": int(
                per_candidate_fields["valid_pixel_count"][index]
            ),
            "valid_pixel_fraction": float(
                per_candidate_fields["valid_pixel_fraction"][index]
            ),
            "null_chi_square": float(
                per_candidate_fields["null_chi_square"][index]
            ),
            "delta_chi_square": float(
                per_candidate_fields["delta_chi_square"][index]
            ),
            "fractional_null_improvement": float(
                per_candidate_fields["fractional_null_improvement"][index]
            ),
            "degrees_of_freedom": int(
                per_candidate_fields["degrees_of_freedom"][index]
            ),
            "reduced_chi_square": float(
                per_candidate_fields["reduced_chi_square"][index]
            ),
            "uncertainty_valid": bool(
                per_candidate_fields["uncertainty_valid"][index]
            ),
            "uncertainty_reason": str(
                per_candidate_fields["uncertainty_reason"][index]
            ),
            "backend": str(result.backend),
            "device": str(result.device),
            "dtype": str(result.dtype),
        }
        for parameter_index, parameter_name in enumerate(parameter_names):
            row[parameter_name] = float(parameters[index, parameter_index])
            row[f"{parameter_name}_standard_error"] = float(
                standard_errors[index, parameter_index]
            )
        rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_fit_artifacts(
    output_dir: str | Path,
    *,
    dataset: XFitDataset,
    result: DipoleFitResult,
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the stable xFit fit table, arrays, config, and summary."""

    resolved = Path(output_dir).expanduser().resolve()
    candidate_id = np.asarray(dataset.candidate_id)
    _validate_candidate_id(candidate_id, dataset.batch_size)
    rows = _fit_rows(candidate_id, dataset.images, result)
    covariance = np.asarray(result.covariance)
    residuals = np.asarray(result.residuals)
    parameter_count = len(result.parameter_names)
    expected_covariance = (
        dataset.batch_size,
        parameter_count,
        parameter_count,
    )
    if covariance.shape != expected_covariance:
        raise ValueError("fit result covariance has an inconsistent shape")
    if residuals.shape != dataset.images.shape:
        raise ValueError("fit result residuals have an inconsistent shape")
    _validate_real_array("covariance", covariance, finite=False)
    _validate_real_array("residuals", residuals, finite=False)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(exist_ok=False)
    fits_path = resolved / "fits.parquet"
    pq.write_table(pa.Table.from_pylist(rows), fits_path)

    arrays_path = resolved / "fit-arrays.npz"
    np.savez_compressed(
        arrays_path,
        candidate_index=np.arange(dataset.batch_size, dtype=np.int64),
        candidate_id=candidate_id,
        covariance=covariance,
        residuals=residuals,
    )

    config_path = resolved / "effective-config.yaml"
    config_path.write_text(
        yaml.safe_dump(dict(effective_config), sort_keys=False),
        encoding="utf-8",
    )
    artifact_sha256 = {
        "effective_config": file_sha256(config_path),
        "fits": file_sha256(fits_path),
        "fit_arrays": file_sha256(arrays_path),
    }

    status_counts = Counter(row["status"] for row in rows)
    reduced = np.asarray(result.reduced_chi_square)
    finite_reduced = reduced[np.isfinite(reduced)]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "fit-dipoles",
        "package_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(dataset.path),
        "model": str(result.model),
        "mode": str(result.mode),
        "requested_backend": str(effective_config["backend"]),
        "backend": str(result.backend),
        "device": str(result.device),
        "dtype": str(result.dtype),
        "parameter_names": [str(name) for name in result.parameter_names],
        "inputs": {
            "candidate_count": dataset.batch_size,
            "images_shape": list(dataset.images.shape),
            "images_dtype": str(dataset.images.dtype),
            "input_archive_sha256": dataset.input_archive_sha256,
            "mask_present": dataset.mask is not None,
            "variance_present": dataset.variance is not None,
        },
        "metrics": {
            "candidate_count": dataset.batch_size,
            "converged_count": int(np.count_nonzero(result.converged)),
            "uncertainty_valid_count": int(
                np.count_nonzero(result.uncertainty_valid)
            ),
            "median_reduced_chi_square": (
                _finite_float_or_none(np.median(finite_reduced))
                if finite_reduced.size
                else None
            ),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "artifacts": {
            "effective_config": config_path.name,
            "fits": fits_path.name,
            "fit_arrays": arrays_path.name,
        },
        "artifact_sha256": artifact_sha256,
    }
    _write_json(resolved / "summary.json", summary)
    return summary


__all__ = [
    "ALLOWED_KEYS",
    "OPTIONAL_KEYS",
    "REQUIRED_KEYS",
    "XFitDataset",
    "inspect_xfit_dataset",
    "load_xfit_dataset",
    "write_fit_artifacts",
]
