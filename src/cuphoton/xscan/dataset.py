# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dataset packaging, validation, and loading for XScan."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .butler import (
    butler_metadata_for_exposure,
    butler_summary,
    resolve_butler_registry_context,
)
from .hsc import (
    HSC_MASK_PLANE_INDEX,
    INDEX_TO_SPLIT,
    SPLIT_TO_INDEX,
    HscCatalogCandidate,
    HscNpyStore,
    HscXPOISConfig,
    assign_split,
    create_hsc_valid_masks,
    extract_stamp,
    extract_valid_mask_stamp,
    inject_transient,
    load_hsc_catalog_candidates,
    local_noise_sigma,
    normalized_psf_patch,
    prediction_group_indices,
    simple_difference,
    template_from_stack,
    tile_group,
    valid_fraction,
    xpois_difference,
)
from .types import (
    DatasetSample,
    InputMode,
    XFitBundleIdentity,
    XFitFeatureSchema,
    XFitJoinDiagnostics,
)

if TYPE_CHECKING:
    from .xfit_features import XFitFeatureMatrix

COMMON_METADATA_FIELDS = {
    "candidate_id",
    "exposure_id",
    "ccd_id",
    "band",
    "x",
    "y",
    "split_group",
    "split",
    "label",
}
AUTOSCAN_REQUIRED_FIELDS = COMMON_METADATA_FIELDS | {
    "label_source",
}
NODIFF_REQUIRED_FIELDS = COMMON_METADATA_FIELDS | {
    "label_source",
    "fake_id",
    "autoscan_score",
    "diff_snr",
    "snr",
    "flux_ratio",
}


@dataclass(slots=True)
class DatasetBuildResult:
    """Location and summary of a newly packaged XScan dataset.

    Attributes
    ----------
    output_dir
        Directory that owns the emitted ``.npy`` arrays and metadata.
    summary
        JSON-compatible build parameters, counts, and validation details.
    """

    output_dir: Path
    summary: dict[str, Any]


def ensure_finite_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    invalid = int(np.size(array) - np.isfinite(array).sum())
    if invalid > 0:
        raise ValueError(f"{name} contains {invalid} non-finite values")
    return array


def crop_center_stamp(stamp: np.ndarray, stamp_size: int) -> np.ndarray:
    image = np.asarray(stamp, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("stamp must be a 2D array")
    height, width = image.shape
    if height != width:
        raise ValueError("stamp must be square")
    if stamp_size <= 0:
        raise ValueError("stamp_size must be positive")
    if stamp_size > height:
        raise ValueError("stamp_size must not exceed the input stamp size")
    if stamp_size == height:
        return image
    offset = (height - stamp_size) // 2
    end = offset + stamp_size
    return np.asarray(image[offset:end, offset:end], dtype=np.float32)


def summarize_float_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


class StampDataset(Dataset[DatasetSample]):
    """Memory-mapped canonical XScan dataset for PyTorch.

    The dataset directory owns ``search.npy`` and ``template.npy`` arrays
    shaped ``(samples, height, width)``, plus one-dimensional ``labels.npy``
    and ``split.npy`` arrays. Triplet mode additionally requires
    ``difference.npy`` with the same shape as the image arrays.

    Parameters
    ----------
    dataset_dir
        Directory containing the canonical arrays.
    input_mode
        ``"pair"`` for search/template channels or ``"triplet"`` to append
        the difference channel.
    split
        Named split resolved through the stored integer ``split.npy`` values.
    xfit_feature_dir
        Optional bundle to validate and load for one dataset instance.
    xfit_feature_names
        Expected ordered feature contract when fusion is enabled.
    xfit_feature_matrix
        Prevalidated matrix shared by multiple split datasets. Mutually
        exclusive with ``xfit_feature_dir`` and made read-only on attachment.

    Notes
    -----
    The instance retains read-only NumPy memory maps. Each indexed feature is
    copied into an owned float32 array before conversion to a Torch tensor, so
    returned tensors do not alias the memory-mapped files. Feature tensors
    have shape ``(2, height, width)`` or ``(3, height, width)``; labels are
    scalar float32 tensors with values zero or one.
    """

    def __init__(
        self,
        dataset_dir: Path,
        *,
        input_mode: InputMode,
        split: str,
        xfit_feature_dir: Path | None = None,
        xfit_feature_names: list[str] | tuple[str, ...] | None = None,
        xfit_feature_matrix: XFitFeatureMatrix | None = None,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.input_mode = input_mode
        self.search = np.load(self.dataset_dir / "search.npy", mmap_mode="r")
        self.template = np.load(
            self.dataset_dir / "template.npy",
            mmap_mode="r",
        )
        self.labels = np.load(self.dataset_dir / "labels.npy", mmap_mode="r")
        self.split = np.load(self.dataset_dir / "split.npy", mmap_mode="r")
        self.indices = prediction_group_indices(self.split, split)
        self.difference = None
        if input_mode == "triplet":
            difference_path = self.dataset_dir / "difference.npy"
            if not difference_path.exists():
                raise FileNotFoundError(
                    f"Triplet input requested but missing {difference_path}"
                )
            self.difference = np.load(difference_path, mmap_mode="r")
        self.xfit_features: np.ndarray | None = None
        self.xfit_feature_names: tuple[str, ...] = ()
        self.xfit_feature_schema: XFitFeatureSchema | None = None
        self.xfit_feature_bundle_identity: XFitBundleIdentity | None = None
        self.xfit_join_diagnostics: XFitJoinDiagnostics | None = None
        if xfit_feature_dir is not None and xfit_feature_matrix is not None:
            raise ValueError(
                "provide xfit_feature_dir or xfit_feature_matrix, not both"
            )
        if xfit_feature_dir is not None:
            from .xfit_features import load_xfit_feature_matrix

            xfit_feature_matrix = load_xfit_feature_matrix(
                dataset_dir=self.dataset_dir,
                feature_dir=xfit_feature_dir,
                expected_feature_names=xfit_feature_names,
            )
        if xfit_feature_matrix is not None:
            if xfit_feature_matrix.dataset_dir != self.dataset_dir:
                raise ValueError(
                    "xFit feature matrix was validated for a different "
                    "XScan dataset directory"
                )
            if len(xfit_feature_matrix) != int(self.labels.shape[0]):
                raise ValueError(
                    "xFit feature row count does not match the XScan dataset"
                )
            if (
                xfit_feature_names is not None
                and tuple(xfit_feature_names)
                != xfit_feature_matrix.feature_names
            ):
                raise ValueError(
                    "xFit feature names do not match xfit_feature_names"
                )
            # A single validated matrix is shared across split views. Marking
            # it read-only prevents either dataset (or a caller retaining the
            # loader result) from mutating the other split's features.
            xfit_feature_matrix.values.setflags(write=False)
            self.xfit_features = xfit_feature_matrix.values
            self.xfit_feature_names = xfit_feature_matrix.feature_names
            self.xfit_feature_schema = xfit_feature_matrix.schema
            self.xfit_feature_bundle_identity = (
                xfit_feature_matrix.bundle_identity
            )
            self.xfit_join_diagnostics = xfit_feature_matrix.join_diagnostics
        elif xfit_feature_names:
            raise ValueError("xfit_feature_names requires xFit feature data")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> DatasetSample:
        sample_index = int(self.indices[idx])
        channels = [
            self.search[sample_index],
            self.template[sample_index],
        ]
        if self.difference is not None:
            channels.append(self.difference[sample_index])
        features = np.stack(channels, axis=0).astype(np.float32)
        label = np.float32(self.labels[sample_index])
        image_tensor = torch.from_numpy(features)
        if self.xfit_features is None:
            model_features = image_tensor
        else:
            scalar_features = np.array(
                self.xfit_features[sample_index],
                dtype=np.float32,
                copy=True,
            )
            model_features = (
                image_tensor,
                torch.from_numpy(scalar_features),
            )
        return model_features, torch.tensor(label)


def inspect_dataset_dir(
    dataset_dir: Path,
    *,
    dataset_kind: str | None = None,
) -> dict[str, Any]:
    """Inspect canonical array shapes, metadata fields, and split counts.

    Parameters
    ----------
    dataset_dir
        Directory containing canonical XScan arrays.
    dataset_kind
        Optional dataset contract name included in the summary.

    Returns
    -------
    dict
        JSON-compatible structural summary. Arrays are opened read-only with
        memory mapping and are not retained after the call.
    """

    root = Path(dataset_dir).expanduser().resolve()
    summary_path = root / "summary.json"
    search = np.load(root / "search.npy", mmap_mode="r")
    template = np.load(root / "template.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    split = np.load(root / "split.npy", mmap_mode="r")
    difference_path = root / "difference.npy"
    metadata_rows = load_metadata_rows(root)
    summary: dict[str, Any] = {
        "dataset_dir": str(root),
        "dataset_kind": dataset_kind,
        "summary_json_exists": summary_path.exists(),
        "arrays": {
            "search_shape": list(search.shape),
            "template_shape": list(template.shape),
            "labels_shape": list(labels.shape),
            "split_shape": list(split.shape),
            "difference_present": difference_path.exists(),
        },
        "metadata": {
            "row_count": len(metadata_rows),
            "fields": sorted(
                {key for row in metadata_rows for key in row.keys()}
            ),
        },
        "splits": (
            summarize_splits(metadata_rows)
            if metadata_rows
            else summarize_split_array(split, labels)
        ),
    }
    if difference_path.exists():
        difference = np.load(difference_path, mmap_mode="r")
        summary["arrays"]["difference_shape"] = list(difference.shape)
    return summary


def validate_dataset_dir(
    dataset_dir: Path,
    *,
    dataset_kind: str | None = None,
) -> dict[str, Any]:
    """Validate a canonical XScan dataset directory.

    Canonical image arrays are rank three with shape
    ``(samples, height, width)``. Labels and split assignments are rank-one,
    share the sample count, and labels contain only zero or one. If present,
    ``difference.npy`` must match the search/template shape and metadata must
    contain one row per sample.

    Parameters
    ----------
    dataset_dir
        Directory containing canonical arrays and optional metadata.
    dataset_kind
        Optional stronger contract such as ``autoscan`` or ``nodiff``.

    Returns
    -------
    dict
        Structural summary augmented with validation results.
    """

    summary = inspect_dataset_dir(dataset_dir, dataset_kind=dataset_kind)
    root = Path(dataset_dir).expanduser().resolve()
    search = np.load(root / "search.npy", mmap_mode="r")
    template = np.load(root / "template.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    split = np.load(root / "split.npy", mmap_mode="r")
    metadata_rows = load_metadata_rows(root)

    if search.ndim != 3 or template.ndim != 3:
        raise ValueError("search.npy and template.npy must be rank-3 arrays")
    if search.shape != template.shape:
        raise ValueError(
            "search.npy and template.npy must share the same shape"
        )
    if labels.ndim != 1 or split.ndim != 1:
        raise ValueError("labels.npy and split.npy must be rank-1 arrays")
    if (
        labels.shape[0] != search.shape[0]
        or split.shape[0] != search.shape[0]
    ):
        raise ValueError(
            "search/template/labels/split sample counts must match"
        )

    difference_path = root / "difference.npy"
    input_mode = "pair"
    if difference_path.exists():
        difference = np.load(difference_path, mmap_mode="r")
        if difference.shape != search.shape:
            raise ValueError(
                "difference.npy must have the same shape as search.npy"
            )
        input_mode = "triplet"
    if dataset_kind == "nodiff" and difference_path.exists():
        raise ValueError("nodiff datasets must not contain difference.npy")

    if metadata_rows and len(metadata_rows) != search.shape[0]:
        raise ValueError("metadata row count must match sample count")

    if metadata_rows:
        required = COMMON_METADATA_FIELDS
        if dataset_kind == "autoscan":
            required = AUTOSCAN_REQUIRED_FIELDS
        elif dataset_kind == "nodiff":
            required = NODIFF_REQUIRED_FIELDS
        present = {key for row in metadata_rows for key in row.keys()}
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                "dataset metadata is missing required fields: "
                + ", ".join(missing)
            )
        row_splits = [row.get("split") for row in metadata_rows]
        if any(split_name not in SPLIT_TO_INDEX for split_name in row_splits):
            raise ValueError(
                "metadata rows must contain valid split names for every "
                "sample"
            )
        expected_split = np.array(
            [SPLIT_TO_INDEX[row["split"]] for row in metadata_rows],
            dtype=np.int64,
        )
        if not np.array_equal(
            expected_split, np.asarray(split, dtype=np.int64)
        ):
            raise ValueError(
                "split.npy does not match metadata split assignments"
            )
        if any(not row.get("split_group") for row in metadata_rows):
            raise ValueError(
                "metadata rows must include non-empty split_group"
            )
        if dataset_kind in {"autoscan", "nodiff"} and any(
            row.get("ccd_id") is None for row in metadata_rows
        ):
            raise ValueError("metadata rows must include non-null ccd_id")

    labels_arr = np.asarray(labels, dtype=np.int64)
    if np.any((labels_arr != 0) & (labels_arr != 1)):
        raise ValueError("labels.npy must contain only 0/1 values")
    if dataset_kind in {"autoscan", "nodiff"}:
        counts = np.bincount(labels_arr, minlength=2)
        if 0 in counts:
            raise ValueError(
                f"{dataset_kind} dataset must contain both positive and "
                "negative labels"
            )
        if counts[0] != counts[1]:
            raise ValueError(
                f"{dataset_kind} dataset must be class balanced for "
                "faithful reproduction"
            )

    semantic_checks = semantic_validation_summary(
        metadata_rows,
        dataset_kind=dataset_kind,
        difference_present=difference_path.exists(),
        labels=labels_arr,
    )

    validation = {
        **summary,
        "valid": True,
        "input_mode": input_mode,
        "sample_count": int(search.shape[0]),
        "semantic_checks": semantic_checks,
    }
    return validation


def build_dataset_from_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    dataset_kind: str,
) -> DatasetBuildResult:
    payload = load_manifest(manifest_path)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    search = load_array_from_manifest(payload, "search_path")
    template = load_array_from_manifest(payload, "template_path")
    labels = load_array_from_manifest(payload, "labels_path")
    split = load_optional_split(payload)

    if search.shape != template.shape:
        raise ValueError("search_path and template_path arrays must match")
    if labels.ndim != 1 or labels.shape[0] != search.shape[0]:
        raise ValueError("labels_path must be a vector matching sample count")
    if split.shape[0] != search.shape[0]:
        raise ValueError("split information must match sample count")

    np.save(root / "search.npy", np.asarray(search), allow_pickle=False)
    np.save(root / "template.npy", np.asarray(template), allow_pickle=False)
    np.save(
        root / "labels.npy",
        np.asarray(labels, dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        root / "split.npy",
        np.asarray(split, dtype=np.int64),
        allow_pickle=False,
    )

    difference = None
    if payload.get("difference_path"):
        difference = load_array_from_manifest(payload, "difference_path")
        if difference.shape != search.shape:
            raise ValueError(
                "difference_path array must match search/template shape"
            )
        np.save(
            root / "difference.npy",
            np.asarray(difference),
            allow_pickle=False,
        )

    metadata_rows = load_manifest_metadata(
        payload, sample_count=search.shape[0]
    )
    write_metadata_jsonl(root / "metadata.jsonl", metadata_rows)
    maybe_write_metadata_parquet(root / "metadata.parquet", metadata_rows)

    summary = {
        "dataset_dir": str(root),
        "dataset_kind": dataset_kind,
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "sample_count": int(search.shape[0]),
        "input_mode": "triplet" if difference is not None else "pair",
        "saved": {
            "search": "search.npy",
            "template": "template.npy",
            "labels": "labels.npy",
            "split": "split.npy",
            "metadata_jsonl": "metadata.jsonl",
        },
    }
    if difference is not None:
        summary["saved"]["difference"] = "difference.npy"
    if (root / "metadata.parquet").exists():
        summary["saved"]["metadata_parquet"] = "metadata.parquet"
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_dataset_dir(root, dataset_kind=dataset_kind)
    return DatasetBuildResult(output_dir=root, summary=summary)


def merge_dataset_dirs(
    *,
    dataset_dirs: list[Path],
    output_dir: Path,
    dataset_kind: str | None = None,
) -> DatasetBuildResult:
    if not dataset_dirs:
        raise ValueError("at least one dataset_dir is required")
    roots = [Path(path).expanduser().resolve() for path in dataset_dirs]
    output_root = output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output_dir already exists and is not empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    search_parts = []
    template_parts = []
    difference_parts = []
    label_parts = []
    split_parts = []
    merged_rows: list[dict[str, Any]] = []
    input_summaries = []
    difference_present: bool | None = None
    stamp_shape: tuple[int, int] | None = None
    offset = 0
    metadata_policy: bool | None = None

    for dataset_order, root in enumerate(roots):
        validation = validate_dataset_dir(root, dataset_kind=dataset_kind)
        search = np.load(root / "search.npy", mmap_mode="r")
        template = np.load(root / "template.npy", mmap_mode="r")
        labels = np.load(root / "labels.npy", mmap_mode="r")
        split = np.load(root / "split.npy", mmap_mode="r")
        has_difference = (root / "difference.npy").exists()
        if difference_present is None:
            difference_present = has_difference
        elif difference_present != has_difference:
            raise ValueError(
                "all input datasets must either include difference.npy or "
                "omit it"
            )
        current_stamp_shape = tuple(int(value) for value in search.shape[1:])
        if stamp_shape is None:
            stamp_shape = current_stamp_shape
        elif stamp_shape != current_stamp_shape:
            raise ValueError(
                "all input datasets must share stamp shape: expected "
                f"{stamp_shape}, got {current_stamp_shape} for {root}"
            )

        metadata_rows = load_metadata_rows(root)
        has_metadata = bool(metadata_rows)
        if metadata_policy is None:
            metadata_policy = has_metadata
        elif metadata_policy != has_metadata:
            raise ValueError(
                "all input datasets must consistently include metadata.jsonl"
            )

        search_parts.append(np.asarray(search))
        template_parts.append(np.asarray(template))
        label_parts.append(np.asarray(labels, dtype=np.int64))
        split_parts.append(np.asarray(split, dtype=np.int64))
        if has_difference:
            difference_parts.append(
                np.asarray(
                    np.load(
                        root / "difference.npy",
                        mmap_mode="r",
                    )
                )
            )

        for local_index in range(int(labels.shape[0])):
            row = dict(metadata_rows[local_index]) if metadata_rows else {}
            row["source_dataset_order"] = int(dataset_order)
            row["source_dataset_dir"] = str(root)
            row["source_sample_index"] = int(local_index)
            row["sample_index"] = int(offset + local_index)
            row["label"] = int(labels[local_index])
            row.setdefault(
                "candidate_id", f"sample-{offset + local_index:06d}"
            )
            row.setdefault(
                "split",
                INDEX_TO_SPLIT.get(int(split[local_index]), "unknown"),
            )
            merged_rows.append(row)

        input_summaries.append(
            {
                "dataset_dir": str(root),
                "sample_count": int(labels.shape[0]),
                "input_mode": validation["input_mode"],
            }
        )
        offset += int(labels.shape[0])

    np.save(
        output_root / "search.npy",
        np.concatenate(search_parts, axis=0),
        allow_pickle=False,
    )
    np.save(
        output_root / "template.npy",
        np.concatenate(template_parts, axis=0),
        allow_pickle=False,
    )
    np.save(
        output_root / "labels.npy",
        np.concatenate(label_parts, axis=0).astype(np.int64),
        allow_pickle=False,
    )
    np.save(
        output_root / "split.npy",
        np.concatenate(split_parts, axis=0).astype(np.int64),
        allow_pickle=False,
    )
    if difference_present:
        np.save(
            output_root / "difference.npy",
            np.concatenate(difference_parts, axis=0),
            allow_pickle=False,
        )
    if metadata_policy:
        write_metadata_jsonl(output_root / "metadata.jsonl", merged_rows)
        maybe_write_metadata_parquet(
            output_root / "metadata.parquet",
            merged_rows,
        )

    validation = validate_dataset_dir(output_root, dataset_kind=dataset_kind)
    summary = {
        "workflow": "data-merge",
        "dataset_dir": str(output_root),
        "dataset_kind": dataset_kind,
        "input_dataset_dirs": [str(root) for root in roots],
        "input_summaries": input_summaries,
        "sample_count": int(validation["sample_count"]),
        "input_mode": validation["input_mode"],
        "saved": {
            "search": "search.npy",
            "template": "template.npy",
            "labels": "labels.npy",
            "split": "split.npy",
        },
    }
    if difference_present:
        summary["saved"]["difference"] = "difference.npy"
    if metadata_policy:
        summary["saved"]["metadata_jsonl"] = "metadata.jsonl"
        if (output_root / "metadata.parquet").exists():
            summary["saved"]["metadata_parquet"] = "metadata.parquet"
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return DatasetBuildResult(output_dir=output_root, summary=summary)


def build_hsc_synthetic_dataset(
    *,
    base: str | Path,
    output_dir: Path,
    positive_count: int,
    negative_count: int,
    stamp_size: int,
    seed: int,
    tile_size: int = 256,
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    template_source: str = "median",
    include_difference: bool = True,
    flux_scale_range: tuple[float, float] = (6.0, 18.0),
) -> DatasetBuildResult:
    if stamp_size % 2 == 0:
        raise ValueError("stamp_size must be odd")
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("positive_count and negative_count must be positive")
    if not math.isclose(sum(split_fractions), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("split_fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    store = HscNpyStore.open(base)
    height, width = store.spatial_shape
    half = stamp_size // 2
    margin = max(half + 1, 16)

    samples = positive_count + negative_count
    search = np.zeros((samples, stamp_size, stamp_size), dtype=np.float32)
    template = np.zeros_like(search)
    difference = np.zeros_like(search) if include_difference else None
    labels = np.zeros((samples,), dtype=np.int64)
    split = np.zeros((samples,), dtype=np.int64)
    rows: list[dict[str, Any]] = []

    def make_template(
        exposure_index: int, center_y: int, center_x: int
    ) -> np.ndarray:
        if template_source != "median":
            raise ValueError(
                "Only 'median' template_source is supported in the "
                "experimental HSC builder"
            )
        return template_from_stack(
            store.images,
            center_y,
            center_x,
            stamp_size,
            exclude_exposure=exposure_index,
        )

    def random_center() -> tuple[int, int]:
        return (
            int(rng.integers(margin, height - margin)),
            int(rng.integers(margin, width - margin)),
        )

    index = 0
    for label, count in ((1, positive_count), (0, negative_count)):
        for _ in range(count):
            exposure_index = int(rng.integers(0, store.exposure_count))
            center_y, center_x = random_center()
            base_search = extract_stamp(
                store.images[0, :, :, exposure_index],
                center_y,
                center_x,
                stamp_size,
            )
            template_stamp = make_template(
                exposure_index,
                center_y,
                center_x,
            )
            psf_patch = normalized_psf_patch(
                store.psfs,
                exposure_index,
                stamp_size,
            )
            noise_sigma = local_noise_sigma(
                store.variances,
                center_y,
                center_x,
                exposure_index,
            )

            injected_flux = 0.0
            if label == 1:
                multiplier = float(
                    rng.uniform(flux_scale_range[0], flux_scale_range[1])
                )
                injected_flux = multiplier * noise_sigma
                search_stamp, injection = inject_transient(
                    base_search,
                    psf_patch,
                    injected_flux,
                )
            else:
                search_stamp = base_search
                injection = np.zeros_like(base_search)

            diff_stamp = simple_difference(search_stamp, template_stamp)
            fd = float(
                diff_stamp[half - 2 : half + 3, half - 2 : half + 3].sum()
            )
            fi = float(
                template_stamp[half - 2 : half + 3, half - 2 : half + 3].sum()
            )
            fsn = float(
                injection[half - 2 : half + 3, half - 2 : half + 3].sum()
            )
            denom = fi + fsn
            flux_ratio = (
                float(np.clip(fd / denom, 0.0, 1.0)) if denom > 0 else 0.0
            )
            snr = injected_flux / noise_sigma if label == 1 else 0.0
            magnitude_proxy = (
                float(-2.5 * np.log10(max(injected_flux, 1e-6)))
                if label == 1
                else None
            )
            search[index] = search_stamp
            template[index] = template_stamp
            if difference is not None:
                difference[index] = diff_stamp
            labels[index] = label
            group = tile_group(center_y, center_x, tile_size)
            split[index] = assign_split(group, seed, split_fractions)
            rows.append(
                {
                    "candidate_id": f"cand-{index:06d}",
                    "exposure_id": exposure_index,
                    "ccd_id": None,
                    "x": center_x,
                    "y": center_y,
                    "band": "i",
                    "template_source": template_source,
                    "label_source": (
                        "synthetic_injection"
                        if label == 1
                        else "background_patch"
                    ),
                    "fake_id": index if label == 1 else None,
                    "autoscan_score": None,
                    "diff_snr": snr,
                    "injected_flux": injected_flux,
                    "snr": snr,
                    "magnitude_proxy": magnitude_proxy,
                    "flux_ratio": flux_ratio,
                    "split_group": group,
                    "split": INDEX_TO_SPLIT[int(split[index])],
                    "label": label,
                }
            )
            index += 1

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "search.npy", search, allow_pickle=False)
    np.save(output_dir / "template.npy", template, allow_pickle=False)
    if difference is not None:
        np.save(output_dir / "difference.npy", difference, allow_pickle=False)
    np.save(output_dir / "labels.npy", labels, allow_pickle=False)
    np.save(output_dir / "split.npy", split, allow_pickle=False)
    write_metadata_jsonl(output_dir / "metadata.jsonl", rows)
    maybe_write_metadata_parquet(output_dir / "metadata.parquet", rows)

    summary = {
        "dataset_dir": str(output_dir),
        "dataset_kind": "experimental-hsc-synthetic",
        "base": str(Path(base).expanduser().resolve()),
        "stamp_size": stamp_size,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "template_source": template_source,
        "include_difference": include_difference,
        "seed": seed,
        "tile_size": tile_size,
        "splits": summarize_splits(rows),
        "saved": {
            "search": "search.npy",
            "template": "template.npy",
            "labels": "labels.npy",
            "split": "split.npy",
            "metadata_jsonl": "metadata.jsonl",
        },
    }
    if include_difference:
        summary["saved"]["difference"] = "difference.npy"
    if (output_dir / "metadata.parquet").exists():
        summary["saved"]["metadata_parquet"] = "metadata.parquet"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return DatasetBuildResult(output_dir=output_dir, summary=summary)


def build_hsc_dataset_from_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> DatasetBuildResult:
    payload = load_manifest(manifest_path)
    base = payload.get("base") or payload.get("hsc_base")
    if not base:
        raise ValueError("manifest is missing required field 'base'")

    positive_count = int(payload.get("positive_count", 0))
    negative_count = int(payload.get("negative_count", 0))
    stamp_size = int(payload.get("stamp_size", 51))
    if stamp_size % 2 == 0:
        raise ValueError("stamp_size must be odd")
    difference_context_size = int(
        payload.get("difference_context_size", stamp_size)
    )
    benchmark_regime_name = payload.get("benchmark_regime_name")
    if difference_context_size % 2 == 0:
        raise ValueError("difference_context_size must be odd")
    if difference_context_size < stamp_size:
        raise ValueError(
            "difference_context_size must be greater than or equal to "
            "stamp_size"
        )
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("positive_count and negative_count must be positive")

    split_fractions = tuple(
        float(value)
        for value in payload.get("split_fractions", [0.7, 0.15, 0.15])
    )
    if len(split_fractions) != 3:
        raise ValueError("split_fractions must contain exactly three values")
    if not math.isclose(sum(split_fractions), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("split_fractions must sum to 1.0")

    sample_seed = int(payload.get("sample_seed", payload.get("seed", 0)))
    split_seed = int(payload.get("split_seed", sample_seed))
    tile_size = int(payload.get("tile_size", 256))
    template_source = str(payload.get("template_source", "median"))
    mask_mode = str(payload.get("mask_mode", "none"))
    mask_dilation_factor = int(payload.get("mask_dilation_factor", 1))
    mask_min_valid_fraction = float(
        payload.get("mask_min_valid_fraction", 1.0)
    )
    template_min_valid_exposures = int(
        payload.get("template_min_valid_exposures", 1)
    )
    if not 0.0 <= mask_min_valid_fraction <= 1.0:
        raise ValueError("mask_min_valid_fraction must satisfy 0 <= f <= 1")
    if template_min_valid_exposures <= 0:
        raise ValueError("template_min_valid_exposures must be positive")
    include_difference = bool(payload.get("include_difference", True))
    difference_mode_raw = str(payload.get("difference_mode", "simple"))
    difference_mode_aliases = {
        "simple": "simple",
        "xpois": "xpois_constant",
        "xpois_constant": "xpois_constant",
        "xpois_separable": "xpois_separable",
    }
    allowed_difference_modes = set(difference_mode_aliases)
    if difference_mode_raw not in allowed_difference_modes:
        raise ValueError(
            "difference_mode must be one of: "
            + ", ".join(sorted(allowed_difference_modes))
        )
    difference_mode = difference_mode_aliases[difference_mode_raw]
    band = str(payload.get("band", "i"))
    center_jitter = int(payload.get("center_jitter", 0))
    if center_jitter < 0:
        raise ValueError("center_jitter must be non-negative")
    positive_quality_snr_min = float(
        payload.get("positive_quality_snr_min", 8.0)
    )
    positive_quality_flux_ratio_min = float(
        payload.get("positive_quality_flux_ratio_min", 0.25)
    )
    positive_quality_center_offset_max = float(
        payload.get("positive_quality_center_offset_max", 1.5)
    )
    if positive_quality_snr_min < 0.0:
        raise ValueError("positive_quality_snr_min must be non-negative")
    if not 0.0 <= positive_quality_flux_ratio_min <= 1.0:
        raise ValueError(
            "positive_quality_flux_ratio_min must satisfy 0 <= f <= 1"
        )
    if positive_quality_center_offset_max < 0.0:
        raise ValueError(
            "positive_quality_center_offset_max must be non-negative"
        )

    positive_source_mode = str(payload.get("positive_source_mode", "catalog"))
    negative_source_mode = str(payload.get("negative_source_mode", "random"))
    allowed_positive_center_modes = {"catalog", "random"}
    allowed_negative_center_modes = {"catalog", "random", "catalog-offset"}
    if positive_source_mode not in allowed_positive_center_modes:
        raise ValueError(
            "positive_source_mode must be one of: "
            + ", ".join(sorted(allowed_positive_center_modes))
        )
    if negative_source_mode not in allowed_negative_center_modes:
        raise ValueError(
            "negative_source_mode must be one of: "
            + ", ".join(sorted(allowed_negative_center_modes))
        )

    negative_offset_range_raw = payload.get(
        "negative_offset_range", [8.0, 24.0]
    )
    if (
        not isinstance(negative_offset_range_raw, (list, tuple))
        or len(negative_offset_range_raw) != 2
    ):
        raise ValueError(
            "negative_offset_range must contain exactly two values"
        )
    negative_offset_range = (
        float(negative_offset_range_raw[0]),
        float(negative_offset_range_raw[1]),
    )
    if negative_offset_range[0] <= 0.0:
        raise ValueError("negative_offset_range lower bound must be positive")
    if negative_offset_range[1] < negative_offset_range[0]:
        raise ValueError("negative_offset_range must be an ascending pair")

    def catalog_object_cap_value(role: str) -> int | None:
        value = payload.get(
            f"{role}_catalog_object_cap",
            payload.get("catalog_object_cap"),
        )
        if value is None:
            return None
        cap = int(value)
        if cap <= 0:
            raise ValueError("catalog object caps must be positive")
        return cap

    catalog_object_caps = {
        "positive": catalog_object_cap_value("positive"),
        "negative": catalog_object_cap_value("negative"),
    }

    flux_scale_range_raw = payload.get("flux_scale_range", [6.0, 18.0])
    if (
        not isinstance(flux_scale_range_raw, (list, tuple))
        or len(flux_scale_range_raw) != 2
    ):
        raise ValueError("flux_scale_range must contain exactly two values")
    flux_scale_range = (
        float(flux_scale_range_raw[0]),
        float(flux_scale_range_raw[1]),
    )
    if (
        flux_scale_range[0] <= 0.0
        or flux_scale_range[1] <= flux_scale_range[0]
    ):
        raise ValueError("flux_scale_range must be a positive ascending pair")

    rng = np.random.default_rng(sample_seed)
    store = HscNpyStore.open(base)
    butler_context = resolve_butler_registry_context(
        payload,
        exposure_count=store.exposure_count,
    )
    valid_masks = create_hsc_valid_masks(
        store.masks,
        mode=mask_mode,
        dilation_factor=mask_dilation_factor,
    )
    mask_rejection_counts = {
        "search_stamp_low_valid_fraction": 0,
        "search_context_low_valid_fraction": 0,
        "centers_without_valid_exposure": 0,
    }
    mask_exposure_attempts = 0
    accepted_search_valid_fractions: list[float] = []
    accepted_context_valid_fractions: list[float] = []
    mask_rejection_plane_sums = {
        "search_stamp": {
            name: {"count": 0, "fraction_sum": 0.0}
            for name in HSC_MASK_PLANE_INDEX
        },
        "search_context": {
            name: {"count": 0, "fraction_sum": 0.0}
            for name in HSC_MASK_PLANE_INDEX
        },
    }
    height, width = store.spatial_shape
    half = stamp_size // 2
    difference_half = difference_context_size // 2
    required_half = max(half, difference_half)
    min_center_y = required_half
    min_center_x = required_half
    max_center_y = height - required_half - 1
    max_center_x = width - required_half - 1
    if max_center_y < min_center_y or max_center_x < min_center_x:
        raise ValueError(
            "stamp_size is too large for the available HSC image dimensions"
        )

    def resolve_optional_float(*keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return float(value)
        return None

    def resolve_optional_int(*keys: str) -> int | None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return int(value)
        return None

    def load_catalog_pool(role: str) -> list[HscCatalogCandidate]:
        flux_min = resolve_optional_float(
            f"{role}_catalog_flux_min",
            "catalog_flux_min",
        )
        flux_max = resolve_optional_float(
            f"{role}_catalog_flux_max",
            "catalog_flux_max",
        )
        extendedness_min = resolve_optional_float(
            f"{role}_catalog_extendedness_min",
            "catalog_extendedness_min",
        )
        extendedness_max = resolve_optional_float(
            f"{role}_catalog_extendedness_max",
            "catalog_extendedness_max",
        )
        catalog_limit = resolve_optional_int(
            f"{role}_catalog_limit",
            "catalog_limit",
        )
        candidates, _catalog_path = load_hsc_catalog_candidates(
            base,
            catalog_parquet=payload.get("catalog_parquet"),
            coadd_fits=payload.get("coadd_fits"),
            band=band,
            flux_column=payload.get("catalog_flux_column"),
            primary_only=bool(payload.get("catalog_primary_only", True)),
            exclude_sky=bool(payload.get("catalog_exclude_sky", True)),
            flux_min=flux_min,
            flux_max=flux_max,
            extendedness_min=extendedness_min,
            extendedness_max=extendedness_max,
            limit=catalog_limit,
            stamp_size=difference_context_size,
        )
        bounded = [
            candidate
            for candidate in candidates
            if (
                min_center_x <= candidate.center_x <= max_center_x
                and min_center_y <= candidate.center_y <= max_center_y
            )
        ]
        if not bounded:
            raise ValueError(
                f"No in-bounds HSC {role} catalog candidates remained after "
                "applying the requested filters"
            )
        return bounded

    positive_catalog_candidates: list[HscCatalogCandidate] = []
    negative_catalog_candidates: list[HscCatalogCandidate] = []
    catalog_path: Path | None = None
    if "catalog" in {positive_source_mode, negative_source_mode} or (
        "catalog-offset" in {positive_source_mode, negative_source_mode}
    ):
        _, catalog_path = load_hsc_catalog_candidates(
            base,
            catalog_parquet=payload.get("catalog_parquet"),
            coadd_fits=payload.get("coadd_fits"),
            band=band,
            flux_column=payload.get("catalog_flux_column"),
            primary_only=bool(payload.get("catalog_primary_only", True)),
            exclude_sky=bool(payload.get("catalog_exclude_sky", True)),
            stamp_size=difference_context_size,
        )
    if positive_source_mode == "catalog":
        positive_catalog_candidates = load_catalog_pool("positive")
    if negative_source_mode in {"catalog", "catalog-offset"}:
        negative_catalog_candidates = load_catalog_pool("negative")

    samples = positive_count + negative_count
    search = np.zeros((samples, stamp_size, stamp_size), dtype=np.float32)
    template = np.zeros_like(search)
    difference = np.zeros_like(search) if include_difference else None
    labels = np.zeros((samples,), dtype=np.int64)
    split = np.zeros((samples,), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    xpois_config = HscXPOISConfig(
        kernel_shape=tuple(
            int(value) for value in payload.get("xpois_kernel_shape", [9, 9])
        ),
        basis_sigmas=tuple(
            float(value)
            for value in payload.get("xpois_basis_sigmas", [1.5, 3.0])
        ),
        basis_degrees=tuple(
            int(value) for value in payload.get("xpois_basis_degrees", [2, 1])
        ),
        background_degree=int(payload.get("xpois_background_degree", 0)),
        flux_conserve=bool(payload.get("xpois_flux_conserve", False)),
        use_variance=bool(payload.get("xpois_use_variance", True)),
        solver_mode=(
            "separable"
            if difference_mode == "xpois_separable"
            else "constant"
        ),
    )
    if include_difference and difference_mode.startswith("xpois_"):
        kernel_margin = max(value // 2 for value in xpois_config.kernel_shape)
        min_context_size = stamp_size + 2 * kernel_margin
        if difference_context_size < min_context_size:
            raise ValueError(
                "difference_context_size is too small for xpois residual "
                "cropping with the requested kernel_shape; expected at least "
                f"{min_context_size}"
            )

    def make_template(
        exposure_index: int,
        center_y: int,
        center_x: int,
        size: int,
    ) -> np.ndarray:
        if template_source != "median":
            raise ValueError(
                "Only 'median' template_source is currently supported "
                "for the HSC builder"
            )
        return template_from_stack(
            store.images,
            center_y,
            center_x,
            size,
            exclude_exposure=exposure_index,
            valid_masks=valid_masks,
            min_valid_exposures=template_min_valid_exposures,
        )

    def require_mask_fraction(
        mask: np.ndarray | None,
        *,
        name: str,
        sample_index: int,
        exposure_index: int,
    ) -> None:
        fraction = valid_fraction(mask)
        if fraction is None:
            return
        if fraction < mask_min_valid_fraction:
            raise ValueError(
                f"{name} valid mask fraction {fraction:.6f} is below "
                f"mask_min_valid_fraction={mask_min_valid_fraction:.6f} "
                f"(sample_index={sample_index}, exposure={exposure_index})"
            )

    def random_center() -> tuple[int, int]:
        return (
            int(rng.integers(min_center_y, max_center_y + 1)),
            int(rng.integers(min_center_x, max_center_x + 1)),
        )

    def record_rejected_mask_planes(
        *,
        region: str,
        center_y: int,
        center_x: int,
        size: int,
        exposure_index: int,
    ) -> None:
        if store.masks is None:
            return
        region_sums = mask_rejection_plane_sums[region]
        for name, plane_index in HSC_MASK_PLANE_INDEX.items():
            plane_stamp = extract_stamp(
                store.masks[0, :, :, plane_index, exposure_index],
                center_y,
                center_x,
                size,
            )
            fraction = float(np.mean(np.asarray(plane_stamp, dtype=bool)))
            if fraction <= 0.0:
                continue
            region_sums[name]["count"] += 1
            region_sums[name]["fraction_sum"] += fraction

    def choose_exposure_index_for_center(
        center_y: int,
        center_x: int,
    ) -> int:
        nonlocal mask_exposure_attempts
        allowed_indices = (
            np.asarray(
                butler_context.allowed_exposure_indices, dtype=np.int64
            )
            if butler_context is not None
            and butler_context.allowed_exposure_indices is not None
            else np.arange(store.exposure_count)
        )
        if valid_masks is None:
            return int(
                allowed_indices[int(rng.integers(0, len(allowed_indices)))]
            )

        exposure_indices = np.array(allowed_indices, copy=True)
        rng.shuffle(exposure_indices)
        for exposure_index_raw in exposure_indices:
            exposure_index = int(exposure_index_raw)
            mask_exposure_attempts += 1
            stamp_mask = extract_valid_mask_stamp(
                valid_masks,
                center_y,
                center_x,
                stamp_size,
                exposure_index,
            )
            stamp_fraction = valid_fraction(stamp_mask)
            if (
                stamp_fraction is not None
                and stamp_fraction < mask_min_valid_fraction
            ):
                mask_rejection_counts["search_stamp_low_valid_fraction"] += 1
                record_rejected_mask_planes(
                    region="search_stamp",
                    center_y=center_y,
                    center_x=center_x,
                    size=stamp_size,
                    exposure_index=exposure_index,
                )
                continue
            context_fraction = None
            if include_difference:
                context_mask = extract_valid_mask_stamp(
                    valid_masks,
                    center_y,
                    center_x,
                    difference_context_size,
                    exposure_index,
                )
                context_fraction = valid_fraction(context_mask)
                if (
                    context_fraction is not None
                    and context_fraction < mask_min_valid_fraction
                ):
                    mask_rejection_counts[
                        "search_context_low_valid_fraction"
                    ] += 1
                    record_rejected_mask_planes(
                        region="search_context",
                        center_y=center_y,
                        center_x=center_x,
                        size=difference_context_size,
                        exposure_index=exposure_index,
                    )
                    continue
            if stamp_fraction is not None:
                accepted_search_valid_fractions.append(stamp_fraction)
            if context_fraction is not None:
                accepted_context_valid_fractions.append(context_fraction)
            return exposure_index
        mask_rejection_counts["centers_without_valid_exposure"] += 1
        raise ValueError(
            "No exposure satisfies mask_min_valid_fraction for sampled HSC "
            f"center y={center_y}, x={center_x}"
        )

    def sample_center(
        mode: str,
        *,
        role: str,
    ) -> tuple[
        int,
        int,
        str,
        HscCatalogCandidate | None,
        float | None,
        float | None,
        float | None,
    ]:
        def candidate_key(candidate: HscCatalogCandidate) -> str:
            if candidate.object_id is not None:
                return f"object:{candidate.object_id}"
            return f"xy:{candidate.center_x}:{candidate.center_y}"

        def mark_candidate_used(candidate: HscCatalogCandidate) -> None:
            role_usage = catalog_usage[role]
            key = candidate_key(candidate)
            role_usage[key] = role_usage.get(key, 0) + 1

        if mode == "random":
            center_y, center_x = random_center()
            return center_y, center_x, "random", None, None, None, None

        if role == "positive":
            candidate_pool = positive_catalog_candidates
        elif role == "negative":
            candidate_pool = negative_catalog_candidates
        else:
            raise ValueError("role must be 'positive' or 'negative'")
        if not candidate_pool:
            raise ValueError(
                f"catalog-based {role} sampling requested but no candidate "
                "pool is available"
            )
        cap = catalog_object_caps[role]
        if cap is not None:
            role_usage = catalog_usage[role]
            candidate_pool = [
                candidate
                for candidate in candidate_pool
                if role_usage.get(candidate_key(candidate), 0) < cap
            ]
            if not candidate_pool:
                raise ValueError(
                    f"{role}_catalog_object_cap exhausted the catalog "
                    "candidate pool before the requested samples were built"
                )
        candidate = candidate_pool[int(rng.integers(0, len(candidate_pool)))]
        if mode == "catalog-offset":
            for _ in range(64):
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                radius = float(
                    rng.uniform(
                        negative_offset_range[0],
                        negative_offset_range[1],
                    )
                )
                offset_dx = float(np.cos(angle) * radius)
                offset_dy = float(np.sin(angle) * radius)
                center_y = int(round(candidate.center_y + offset_dy))
                center_x = int(round(candidate.center_x + offset_dx))
                if (
                    min_center_y <= center_y <= max_center_y
                    and min_center_x <= center_x <= max_center_x
                ):
                    mark_candidate_used(candidate)
                    return (
                        center_y,
                        center_x,
                        "catalog-offset",
                        candidate,
                        float(center_x - candidate.center_x),
                        float(center_y - candidate.center_y),
                        float(
                            np.hypot(
                                center_x - candidate.center_x,
                                center_y - candidate.center_y,
                            )
                        ),
                    )
            raise ValueError(
                "Could not place a valid catalog-offset center within bounds"
            )

        jitter_y = (
            int(rng.integers(-center_jitter, center_jitter + 1))
            if center_jitter > 0
            else 0
        )
        jitter_x = (
            int(rng.integers(-center_jitter, center_jitter + 1))
            if center_jitter > 0
            else 0
        )
        center_y = min(
            max(candidate.center_y + jitter_y, min_center_y),
            max_center_y,
        )
        center_x = min(
            max(candidate.center_x + jitter_x, min_center_x),
            max_center_x,
        )
        mark_candidate_used(candidate)
        return (
            center_y,
            center_x,
            "catalog",
            candidate,
            float(center_x - candidate.center_x),
            float(center_y - candidate.center_y),
            float(
                np.hypot(
                    center_x - candidate.center_x,
                    center_y - candidate.center_y,
                )
            ),
        )

    def build_row(
        *,
        sample_index: int,
        exposure_index: int,
        center_y: int,
        center_x: int,
        split_name: str,
        split_group: str,
        label: int,
        label_source: str,
        center_source: str,
        catalog_pool_role: str | None,
        candidate: HscCatalogCandidate | None,
        center_offset_dx: float | None,
        center_offset_dy: float | None,
        center_offset_radius: float | None,
        difference_mode_name: str,
        injected_flux: float,
        snr: float,
        magnitude_proxy: float | None,
        flux_ratio: float,
        search_valid_fraction: float | None,
        difference_context_valid_fraction: float | None,
        butler_metadata: dict[str, Any],
        positive_quality_stratum: str | None,
    ) -> dict[str, Any]:
        row = {
            "candidate_id": f"hsc-{sample_index:06d}",
            "exposure_id": exposure_index,
            "ccd_id": None,
            "band": band,
            "x": center_x,
            "y": center_y,
            "split_group": split_group,
            "split": split_name,
            "label": label,
            "label_source": label_source,
            "fake_id": sample_index if label == 1 else None,
            "autoscan_score": None,
            "diff_snr": snr,
            "snr": snr,
            "flux_ratio": flux_ratio,
            "magnitude_proxy": magnitude_proxy,
            "injected_flux": injected_flux,
            "template_source": template_source,
            "difference_mode": difference_mode,
            "manifest_difference_mode": difference_mode_name,
            "search_valid_fraction": search_valid_fraction,
            "difference_context_valid_fraction": (
                difference_context_valid_fraction
            ),
            "center_source": center_source,
            "catalog_pool_role": catalog_pool_role,
            "center_offset_dx": center_offset_dx,
            "center_offset_dy": center_offset_dy,
            "center_offset_radius": center_offset_radius,
            "positive_quality_stratum": positive_quality_stratum,
            "catalog_object_id": (
                candidate.object_id if candidate is not None else None
            ),
            "catalog_x": (
                candidate.original_x if candidate is not None else None
            ),
            "catalog_y": (
                candidate.original_y if candidate is not None else None
            ),
            "catalog_flux": candidate.flux if candidate is not None else None,
            "catalog_flux_column": (
                candidate.flux_column if candidate is not None else None
            ),
            "catalog_extendedness": (
                candidate.extendedness if candidate is not None else None
            ),
            "catalog_tract": (
                candidate.tract if candidate is not None else None
            ),
            "catalog_patch": (
                candidate.patch if candidate is not None else None
            ),
            "catalog_ref_band": (
                candidate.ref_band if candidate is not None else None
            ),
            "catalog_is_primary": (
                candidate.is_primary if candidate is not None else None
            ),
            "catalog_is_sky": (
                candidate.is_sky if candidate is not None else None
            ),
        }
        row.update(butler_metadata)
        return row

    def resolve_positive_quality_stratum(
        *,
        label: int,
        snr: float,
        flux_ratio: float,
        center_offset_radius: float | None,
    ) -> str | None:
        if label != 1:
            return None
        flags = []
        if snr < positive_quality_snr_min:
            flags.append("weak_snr")
        if flux_ratio < positive_quality_flux_ratio_min:
            flags.append("weak_flux_ratio")
        if (
            center_offset_radius is not None
            and center_offset_radius > positive_quality_center_offset_max
        ):
            flags.append("shifted_center")
        if not flags:
            return "positive:clean"
        return "positive:" + "+".join(flags)

    index = 0
    center_source_counts = {
        "catalog": 0,
        "catalog-offset": 0,
        "random": 0,
    }
    catalog_usage: dict[str, dict[str, int]] = {
        "positive": {},
        "negative": {},
    }
    for label, count, center_mode in (
        (1, positive_count, positive_source_mode),
        (0, negative_count, negative_source_mode),
    ):
        for _ in range(count):
            (
                center_y,
                center_x,
                center_source,
                candidate,
                center_offset_dx,
                center_offset_dy,
                center_offset_radius,
            ) = sample_center(
                center_mode,
                role="positive" if label == 1 else "negative",
            )
            exposure_index = choose_exposure_index_for_center(
                center_y,
                center_x,
            )
            butler_metadata = butler_metadata_for_exposure(
                butler_context,
                exposure_index,
            )
            center_source_counts[center_source] += 1
            base_search = extract_stamp(
                store.images[0, :, :, exposure_index],
                center_y,
                center_x,
                stamp_size,
            )
            template_stamp = make_template(
                exposure_index,
                center_y,
                center_x,
                stamp_size,
            )
            psf_patch = normalized_psf_patch(
                store.psfs,
                exposure_index,
                stamp_size,
            )
            search_valid_stamp = extract_valid_mask_stamp(
                valid_masks,
                center_y,
                center_x,
                stamp_size,
                exposure_index,
            )
            require_mask_fraction(
                search_valid_stamp,
                name="HSC search stamp",
                sample_index=index,
                exposure_index=exposure_index,
            )
            if include_difference:
                if difference_context_size == stamp_size:
                    base_search_context = base_search
                    template_context = template_stamp
                    search_valid_context = search_valid_stamp
                else:
                    base_search_context = extract_stamp(
                        store.images[0, :, :, exposure_index],
                        center_y,
                        center_x,
                        difference_context_size,
                    )
                    template_context = make_template(
                        exposure_index,
                        center_y,
                        center_x,
                        difference_context_size,
                    )
                    search_valid_context = extract_valid_mask_stamp(
                        valid_masks,
                        center_y,
                        center_x,
                        difference_context_size,
                        exposure_index,
                    )
                psf_patch_context = normalized_psf_patch(
                    store.psfs,
                    exposure_index,
                    difference_context_size,
                )
                require_mask_fraction(
                    search_valid_context,
                    name="HSC search context",
                    sample_index=index,
                    exposure_index=exposure_index,
                )
            else:
                base_search_context = None
                template_context = None
                psf_patch_context = None
                search_valid_context = None
            noise_sigma = local_noise_sigma(
                store.variances,
                center_y,
                center_x,
                exposure_index,
            )

            injected_flux = 0.0
            if label == 1:
                multiplier = float(
                    rng.uniform(flux_scale_range[0], flux_scale_range[1])
                )
                injected_flux = multiplier * noise_sigma
                search_stamp, injection = inject_transient(
                    base_search,
                    psf_patch,
                    injected_flux,
                )
                if include_difference and base_search_context is not None:
                    search_context, _ = inject_transient(
                        base_search_context,
                        psf_patch_context,
                        injected_flux,
                    )
                else:
                    search_context = None
            else:
                search_stamp = base_search
                injection = np.zeros_like(base_search)
                search_context = base_search_context

            search_stamp = ensure_finite_array(
                search_stamp,
                name=(
                    "HSC search stamp "
                    f"(exposure={exposure_index}, sample_index={index})"
                ),
            )
            template_stamp = ensure_finite_array(
                template_stamp,
                name=(
                    "HSC template stamp "
                    f"(exposure={exposure_index}, sample_index={index})"
                ),
            )
            if include_difference:
                if search_context is None or template_context is None:
                    raise ValueError(
                        "difference context must be available when "
                        "include_difference is enabled"
                    )
                search_context = ensure_finite_array(
                    search_context,
                    name=(
                        "HSC search context "
                        f"(mode={difference_mode}, "
                        f"exposure={exposure_index}, "
                        f"sample_index={index})"
                    ),
                )
                template_context = ensure_finite_array(
                    template_context,
                    name=(
                        "HSC template context "
                        f"(mode={difference_mode}, "
                        f"exposure={exposure_index}, "
                        f"sample_index={index})"
                    ),
                )
                if search_valid_context is not None:
                    search_context = np.where(
                        search_valid_context,
                        search_context,
                        np.nan,
                    )
                if difference_mode.startswith("xpois_"):
                    variance_context = extract_stamp(
                        store.variances[0, :, :, exposure_index],
                        center_y,
                        center_x,
                        difference_context_size,
                    )
                    diff_context = xpois_difference(
                        search_context,
                        template_context,
                        variance_stamp=variance_context,
                        config=xpois_config,
                    )
                elif difference_context_size == stamp_size:
                    diff_context = simple_difference(
                        search_stamp, template_stamp
                    )
                else:
                    diff_context = simple_difference(
                        search_context,
                        template_context,
                    )
                diff_stamp = crop_center_stamp(diff_context, stamp_size)
            else:
                diff_stamp = simple_difference(search_stamp, template_stamp)
            diff_stamp = ensure_finite_array(
                diff_stamp,
                name=(
                    "HSC difference stamp "
                    f"(mode={difference_mode}, exposure={exposure_index}, "
                    f"sample_index={index})"
                ),
            )
            fd = float(
                diff_stamp[half - 2 : half + 3, half - 2 : half + 3].sum()
            )
            fi = float(
                template_stamp[half - 2 : half + 3, half - 2 : half + 3].sum()
            )
            fsn = float(
                injection[half - 2 : half + 3, half - 2 : half + 3].sum()
            )
            denom = fi + fsn
            flux_ratio = (
                float(np.clip(fd / denom, 0.0, 1.0)) if denom > 0 else 0.0
            )
            snr = injected_flux / noise_sigma if label == 1 else 0.0
            magnitude_proxy = (
                float(-2.5 * np.log10(max(injected_flux, 1e-6)))
                if label == 1
                else None
            )

            search[index] = search_stamp
            template[index] = template_stamp
            if difference is not None:
                difference[index] = diff_stamp
            labels[index] = label
            group = tile_group(center_y, center_x, tile_size)
            split_index = assign_split(group, split_seed, split_fractions)
            split[index] = split_index
            split_name = INDEX_TO_SPLIT[int(split_index)]
            positive_quality = resolve_positive_quality_stratum(
                label=label,
                snr=snr,
                flux_ratio=flux_ratio,
                center_offset_radius=center_offset_radius,
            )
            rows.append(
                build_row(
                    sample_index=index,
                    exposure_index=exposure_index,
                    center_y=center_y,
                    center_x=center_x,
                    split_name=split_name,
                    split_group=group,
                    label=label,
                    label_source=(
                        "synthetic_injection"
                        if label == 1
                        else "real_hsc_patch"
                    ),
                    center_source=center_source,
                    catalog_pool_role=(
                        "positive"
                        if center_source != "random" and label == 1
                        else "negative"
                        if center_source != "random"
                        else None
                    ),
                    candidate=candidate,
                    center_offset_dx=center_offset_dx,
                    center_offset_dy=center_offset_dy,
                    center_offset_radius=center_offset_radius,
                    difference_mode_name=difference_mode_raw,
                    injected_flux=injected_flux,
                    snr=snr,
                    magnitude_proxy=magnitude_proxy,
                    flux_ratio=flux_ratio,
                    search_valid_fraction=valid_fraction(search_valid_stamp),
                    difference_context_valid_fraction=valid_fraction(
                        search_valid_context
                    ),
                    butler_metadata=butler_metadata,
                    positive_quality_stratum=positive_quality,
                )
            )
            index += 1

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "search.npy", search, allow_pickle=False)
    np.save(output_dir / "template.npy", template, allow_pickle=False)
    if difference is not None:
        np.save(output_dir / "difference.npy", difference, allow_pickle=False)
    np.save(output_dir / "labels.npy", labels, allow_pickle=False)
    np.save(output_dir / "split.npy", split, allow_pickle=False)
    write_metadata_jsonl(output_dir / "metadata.jsonl", rows)
    maybe_write_metadata_parquet(output_dir / "metadata.parquet", rows)

    def summarize_rejection_plane_overlaps() -> dict[str, Any]:
        summary_by_region: dict[str, Any] = {}
        for region, region_sums in mask_rejection_plane_sums.items():
            summary_by_region[region] = {}
            for name, payload_sum in region_sums.items():
                count = int(payload_sum["count"])
                summary_by_region[region][name] = {
                    "rejected_stamps_with_plane": count,
                    "mean_fraction_when_present": (
                        float(payload_sum["fraction_sum"] / count)
                        if count
                        else 0.0
                    ),
                }
        return summary_by_region

    mask_plane_fractions = None
    if store.masks is not None:
        mask_plane_fractions = {
            name: float(
                np.mean(
                    np.asarray(
                        store.masks[:, :, :, plane_index, :],
                        dtype=bool,
                    )
                )
            )
            for name, plane_index in HSC_MASK_PLANE_INDEX.items()
        }

    summary: dict[str, Any] = {
        "dataset_dir": str(output_dir),
        "dataset_kind": "hsc-synthetic",
        "base": str(Path(base).expanduser().resolve()),
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "stamp_size": stamp_size,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "template_source": template_source,
        "mask_mode": mask_mode,
        "mask_dilation_factor": mask_dilation_factor,
        "mask_min_valid_fraction": mask_min_valid_fraction,
        "template_min_valid_exposures": template_min_valid_exposures,
        "masks_available": bool(store.masks is not None),
        "mask_diagnostics": {
            "enabled": bool(valid_masks is not None),
            "valid_mask_global_fraction": (
                float(np.mean(valid_masks))
                if valid_masks is not None
                else None
            ),
            "mask_plane_fractions": mask_plane_fractions,
            "exposure_attempts": int(mask_exposure_attempts),
            "exposure_rejections": mask_rejection_counts,
            "accepted_search_valid_fraction": summarize_float_values(
                accepted_search_valid_fractions
            ),
            "accepted_difference_context_valid_fraction": (
                summarize_float_values(accepted_context_valid_fractions)
            ),
            "rejected_plane_overlaps": summarize_rejection_plane_overlaps(),
        },
        "sky_available": bool(store.sky is not None),
        "include_difference": include_difference,
        "difference_mode": difference_mode,
        "manifest_difference_mode": difference_mode_raw,
        "difference_context_size": difference_context_size,
        "benchmark_regime_name": benchmark_regime_name,
        "sample_seed": sample_seed,
        "split_seed": split_seed,
        "tile_size": tile_size,
        "band": band,
        "positive_source_mode": positive_source_mode,
        "negative_source_mode": negative_source_mode,
        "negative_offset_range": list(negative_offset_range),
        "center_jitter": center_jitter,
        "positive_quality": {
            "snr_min": positive_quality_snr_min,
            "flux_ratio_min": positive_quality_flux_ratio_min,
            "center_offset_max": positive_quality_center_offset_max,
        },
        "catalog_object_caps": catalog_object_caps,
        "catalog_object_usage": {
            role: {
                "unique_object_count": len(usage),
                "max_samples_per_object": max(usage.values()) if usage else 0,
            }
            for role, usage in catalog_usage.items()
        },
        "butler_registry": butler_summary(butler_context),
        "positive_catalog_candidate_count": len(positive_catalog_candidates),
        "negative_catalog_candidate_count": len(negative_catalog_candidates),
        "center_source_counts": center_source_counts,
        "splits": summarize_splits(rows),
        "saved": {
            "search": "search.npy",
            "template": "template.npy",
            "labels": "labels.npy",
            "split": "split.npy",
            "metadata_jsonl": "metadata.jsonl",
        },
    }
    if catalog_path is not None:
        summary["catalog_parquet"] = str(catalog_path)
    if difference_mode.startswith("xpois_"):
        summary["xpois"] = {
            "solver_mode": xpois_config.solver_mode,
            "kernel_shape": list(xpois_config.kernel_shape),
            "basis_sigmas": list(xpois_config.basis_sigmas),
            "basis_degrees": list(xpois_config.basis_degrees),
            "background_degree": xpois_config.background_degree,
            "flux_conserve": xpois_config.flux_conserve,
            "use_variance": xpois_config.use_variance,
            "context_size": difference_context_size,
        }
    if include_difference:
        summary["saved"]["difference"] = "difference.npy"
    if (output_dir / "metadata.parquet").exists():
        summary["saved"]["metadata_parquet"] = "metadata.parquet"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_dataset_dir(output_dir, dataset_kind="hsc-synthetic")
    return DatasetBuildResult(output_dir=output_dir, summary=summary)


def load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def load_array_from_manifest(
    payload: dict[str, Any],
    key: str,
) -> np.ndarray:
    value = payload.get(key)
    if not value:
        raise ValueError(f"manifest is missing required field '{key}'")
    array = np.load(Path(value).expanduser().resolve(), allow_pickle=False)
    return np.asarray(array)


def load_optional_split(payload: dict[str, Any]) -> np.ndarray:
    split_path = payload.get("split_path")
    if split_path:
        split = np.load(
            Path(split_path).expanduser().resolve(), allow_pickle=False
        )
        return np.asarray(split, dtype=np.int64)
    metadata_rows = load_manifest_metadata(
        payload,
        sample_count=None,
        require_metadata=False,
    )
    if metadata_rows:
        if any(
            row.get("split") not in SPLIT_TO_INDEX for row in metadata_rows
        ):
            raise ValueError("metadata split fields are missing or invalid")
        return np.asarray(
            [SPLIT_TO_INDEX[row["split"]] for row in metadata_rows],
            dtype=np.int64,
        )
    raise ValueError(
        "manifest must provide split_path or metadata split values"
    )


def load_manifest_metadata(
    payload: dict[str, Any],
    *,
    sample_count: int | None,
    require_metadata: bool = True,
) -> list[dict[str, Any]]:
    jsonl_path = payload.get("metadata_jsonl")
    parquet_path = payload.get("metadata_parquet")
    if jsonl_path:
        rows = load_metadata_rows(Path(jsonl_path).expanduser().resolve())
    elif parquet_path:
        rows = load_metadata_parquet(
            Path(parquet_path).expanduser().resolve()
        )
    elif require_metadata:
        raise ValueError(
            "manifest must provide metadata_jsonl or metadata_parquet"
        )
    else:
        rows = []
    if sample_count is not None and rows and len(rows) != sample_count:
        raise ValueError("metadata row count must match sample count")
    return rows


def write_metadata_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def maybe_write_metadata_parquet(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def load_metadata_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    path = Path(dataset_dir)
    if path.is_file():
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    jsonl_path = path / "metadata.jsonl"
    if not jsonl_path.exists():
        return []
    with jsonl_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_metadata_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ValueError(
            "pyarrow is required to load metadata_parquet"
        ) from exc
    table = pq.read_table(path)
    return table.to_pylist()


def summarize_splits(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        result[split_name] = {
            "count": len(split_rows),
            "positives": sum(int(row["label"]) for row in split_rows),
            "negatives": sum(1 - int(row["label"]) for row in split_rows),
        }
    return result


def semantic_validation_summary(
    metadata_rows: list[dict[str, Any]],
    *,
    dataset_kind: str | None,
    difference_present: bool,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    if dataset_kind == "lsstcomcam-smoke" and not metadata_rows:
        raise ValueError(
            "lsstcomcam-smoke validation requires metadata rows with "
            "placeholder label_source provenance"
        )
    if not metadata_rows:
        return {"dataset_kind": dataset_kind, "available": False}

    summary: dict[str, Any] = {
        "dataset_kind": dataset_kind,
        "available": True,
        "difference_present": difference_present,
    }

    if dataset_kind == "autoscan":
        if not difference_present:
            raise ValueError("autoscan dataset must include difference.npy")
        summary["required_difference_present"] = True
        return summary

    if dataset_kind == "lsstcomcam-smoke":
        if not difference_present:
            raise ValueError(
                "lsstcomcam-smoke dataset must include difference.npy"
            )
        if labels is None:
            raise ValueError(
                "lsstcomcam-smoke validation requires labels.npy"
            )
        labels_arr = np.asarray(labels, dtype=np.int64)
        if np.any(labels_arr != 0):
            raise ValueError(
                "lsstcomcam-smoke labels must all be placeholder zeros"
            )
        expected_source = "unlabeled_lsstcomcam_smoke_placeholder"
        label_sources = {
            str(row.get("label_source", "")) for row in metadata_rows
        }
        if label_sources != {expected_source}:
            raise ValueError(
                "lsstcomcam-smoke metadata label_source values must all be "
                f"{expected_source!r}"
            )
        summary["required_difference_present"] = True
        summary["labels_are_placeholders"] = True
        summary["placeholder_label_source"] = expected_source
        return summary

    if dataset_kind == "nodiff":
        if difference_present:
            raise ValueError("nodiff dataset must not include difference.npy")

        positives = [row for row in metadata_rows if int(row["label"]) == 1]
        if not positives:
            raise ValueError("nodiff dataset must contain positive samples")

        missing_fake_id = [
            row["candidate_id"]
            for row in positives
            if row.get("fake_id") is None
        ]
        if missing_fake_id:
            raise ValueError(
                "nodiff positive samples must have fake_id values"
            )

        release_cut_provenance = "published_release_inclusion"
        release_provenance_rows = [
            row
            for row in positives
            if row.get("nodiff_release_cut_provenance")
            == release_cut_provenance
        ]
        missing_autoscan = [
            row["candidate_id"]
            for row in positives
            if row.get("autoscan_score") is None
            and row.get("nodiff_release_cut_provenance")
            != release_cut_provenance
        ]
        if missing_autoscan:
            raise ValueError(
                "nodiff positive samples must have autoScan scores"
            )

        low_diff_snr = []
        numeric_diff_snr_count = 0
        release_diff_snr_provenance_count = 0
        for row in positives:
            try:
                diff_snr = float(row["diff_snr"])
            except Exception as exc:
                if (
                    row.get("nodiff_release_cut_provenance")
                    == release_cut_provenance
                ):
                    release_diff_snr_provenance_count += 1
                    continue
                raise ValueError(
                    "nodiff positive samples must have numeric diff_snr"
                ) from exc
            numeric_diff_snr_count += 1
            if diff_snr <= 3.5:
                low_diff_snr.append(row["candidate_id"])
        if low_diff_snr:
            raise ValueError(
                "nodiff positive samples must satisfy diff_snr > 3.5"
            )

        summary["positive_fake_count"] = len(
            {row["fake_id"] for row in positives}
        )
        summary["required_difference_absent"] = True
        summary["release_cut_provenance_count"] = len(release_provenance_rows)
        summary["autoscan_scores_present"] = all(
            row.get("autoscan_score") is not None for row in positives
        )
        summary["autoscan_scores_release_provenance_count"] = sum(
            1
            for row in positives
            if row.get("autoscan_score") is None
            and row.get("nodiff_release_cut_provenance")
            == release_cut_provenance
        )
        summary["diff_snr_values_present"] = numeric_diff_snr_count == len(
            positives
        )
        summary["diff_snr_release_provenance_count"] = (
            release_diff_snr_provenance_count
        )
        summary["diff_snr_threshold_enforced"] = True
        return summary

    return summary


def autoscan_manifest_example() -> dict[str, Any]:
    return {
        "search_path": "/abs/path/to/autoscan_search.npy",
        "template_path": "/abs/path/to/autoscan_template.npy",
        "difference_path": "/abs/path/to/autoscan_difference.npy",
        "labels_path": "/abs/path/to/autoscan_labels.npy",
        "split_path": "/abs/path/to/autoscan_split.npy",
        "metadata_jsonl": "/abs/path/to/autoscan_metadata.jsonl",
    }


def nodiff_manifest_example() -> dict[str, Any]:
    return {
        "search_path": "/abs/path/to/nodiff_search.npy",
        "template_path": "/abs/path/to/nodiff_template.npy",
        "labels_path": "/abs/path/to/nodiff_labels.npy",
        "split_path": "/abs/path/to/nodiff_split.npy",
        "metadata_jsonl": "/abs/path/to/nodiff_metadata.jsonl",
    }


def summarize_split_array(
    split: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split_name, split_index in SPLIT_TO_INDEX.items():
        mask = np.asarray(split, dtype=np.int64) == split_index
        selected = np.asarray(labels, dtype=np.int64)[mask]
        result[split_name] = {
            "count": int(mask.sum()),
            "positives": int(np.sum(selected)),
            "negatives": int(np.sum(1 - selected)),
        }
    return result
