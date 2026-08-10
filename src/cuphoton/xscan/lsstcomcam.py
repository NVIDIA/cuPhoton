# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry-backed LSSTComCam smoke dataset support."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .butler import (
    _is_missing,
    _jsonable,
    _load_registry_dataframe,
    _row_metadata,
    _safe_query,
)
from .dataset import (
    DatasetBuildResult,
    ensure_finite_array,
    load_manifest,
    maybe_write_metadata_parquet,
    validate_dataset_dir,
    write_metadata_jsonl,
)
from .hsc import INDEX_TO_SPLIT, assign_split

LSSTCOMCAM_SMOKE_DATASET_KIND = "lsstcomcam-smoke"
LSSTCOMCAM_PLACEHOLDER_LABEL_SOURCE = "unlabeled_lsstcomcam_smoke_placeholder"
LSSTCOMCAM_DATASET_SOURCE = "LSSTComCam local FITS sample"
LSSTCOMCAM_CANDIDATE_CONTEXT_FIELDS = (
    "candidate_catalog_format",
    "candidate_catalog_provenance",
    "candidate_label_status",
)
DEFAULT_PAIR_KEY_COLUMNS = ("visit", "detector", "band")
DEFAULT_FILTER_PRODUCTS = (
    "visit_image",
    "difference_image",
    "template_coadd",
    "deep_coadd",
)
STAGE_LINK_MODES = ("symlink", "copy", "hardlink")
STAGE_DUPLICATE_POLICIES = ("fail", "first", "same-size")
PROVENANCE_FIELDS = (
    "path",
    "original_path",
    "path_rewrite_source_prefix",
    "path_rewrite_target_prefix",
    "registry_relative_path",
    "product",
    "butler_datasettype",
    "butler_run",
    "data_id",
    "instrument",
    "telescope",
    "object",
    "date",
    "band",
    "physical_filter",
    "visit",
    "detector",
    "detector_name",
    "tract",
    "patch",
    "exptime",
    "ra",
    "dec",
    "hdu_count",
    "size_bytes",
    "mtime_ns",
)


@dataclass(slots=True)
class FitsStamp:
    data: np.ndarray
    center_x: int
    center_y: int
    image_shape: tuple[int, int]
    hdu_name: str


@dataclass(slots=True)
class LsstComCamRegistrySelection:
    registry_path: Path
    rows: list[dict[str, Any]]
    pairs: list[dict[str, dict[str, Any]]]
    filters: dict[str, Any]
    pairing_summary: dict[str, Any]
    path_rewrites: list[dict[str, str]]


@dataclass(slots=True)
class LsstComCamPathRewrite:
    source_prefix: str
    target_prefix: str


@dataclass(slots=True)
class LsstComCamStampSample:
    pair: dict[str, dict[str, Any]]
    center_x: int | None
    center_y: int | None
    center_source: str
    candidate_row: dict[str, Any] | None = None


def select_lsstcomcam_registry_rows(
    *,
    registry_path: Path,
    filters: dict[str, Any],
    path_rewrites: tuple[LsstComCamPathRewrite, ...] = (),
) -> LsstComCamRegistrySelection:
    """Filter an LSSTComCam registry and pair visit/difference rows."""
    registry_path = registry_path.expanduser().resolve()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"LSSTComCam registry not found: {registry_path}"
        )
    df = _load_registry_dataframe(registry_path)
    filtered = _filter_lsstcomcam_dataframe(df, filters)
    if filtered.empty:
        raise ValueError("LSSTComCam registry filters produced no rows")
    rows = [_row_metadata(row.to_dict()) for _, row in filtered.iterrows()]
    rows = _apply_path_rewrites_to_rows(rows, path_rewrites)
    pairs, pairing_summary = _pair_lsstcomcam_visit_difference_rows(rows)
    return LsstComCamRegistrySelection(
        registry_path=registry_path,
        rows=rows,
        pairs=pairs,
        filters=_manifest_filter_summary(filters),
        pairing_summary=pairing_summary,
        path_rewrites=[
            {
                "source_prefix": rule.source_prefix,
                "target_prefix": rule.target_prefix,
            }
            for rule in path_rewrites
        ],
    )


def pair_lsstcomcam_visit_difference_rows(
    rows: list[dict[str, Any]],
    *,
    key_columns: tuple[str, ...] = DEFAULT_PAIR_KEY_COLUMNS,
) -> list[dict[str, dict[str, Any]]]:
    """Return compatible visit_image + difference_image registry pairs."""
    pairs, _summary = _pair_lsstcomcam_visit_difference_rows(
        rows,
        key_columns=key_columns,
    )
    return pairs


def _path_rewrites_from_manifest(
    payload: dict[str, Any],
) -> tuple[LsstComCamPathRewrite, ...]:
    raw = payload.get(
        "path_rewrites", payload.get("path_prefix_rewrites", [])
    )
    if raw in (None, ""):
        return ()
    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raise ValueError(
            "path_rewrites must be a mapping or list of mappings"
        )
    rules = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(
                f"path_rewrites[{index}] must be a mapping, got {type(item)}"
            )
        source = (
            item.get("source_prefix")
            or item.get("from")
            or item.get("source")
            or item.get("old")
        )
        target = (
            item.get("target_prefix")
            or item.get("to")
            or item.get("target")
            or item.get("new")
        )
        if not source or not target:
            raise ValueError(
                "path_rewrites entries require source_prefix/from and "
                "target_prefix/to"
            )
        rules.append(
            LsstComCamPathRewrite(
                source_prefix=_normalize_rewrite_prefix(str(source)),
                target_prefix=_normalize_rewrite_prefix(str(target)),
            )
        )
    return tuple(rules)


def _normalize_rewrite_prefix(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("path rewrite prefixes must be non-empty")
    if normalized == "/":
        return normalized
    return normalized.rstrip("/")


def _apply_path_rewrites_to_rows(
    rows: list[dict[str, Any]],
    path_rewrites: tuple[LsstComCamPathRewrite, ...],
) -> list[dict[str, Any]]:
    if not path_rewrites:
        return rows
    return [_apply_path_rewrites_to_row(row, path_rewrites) for row in rows]


def _apply_path_rewrites_to_row(
    row: dict[str, Any],
    path_rewrites: tuple[LsstComCamPathRewrite, ...],
) -> dict[str, Any]:
    path_value = row.get("path")
    if _is_missing(path_value):
        return row
    original = str(_metadata_value(path_value))
    rewritten, rule = _rewrite_path(original, path_rewrites)
    if rule is None:
        return row
    updated = dict(row)
    updated["original_path"] = original
    updated["path"] = rewritten
    updated["path_rewrite_source_prefix"] = rule.source_prefix
    updated["path_rewrite_target_prefix"] = rule.target_prefix
    return updated


def _rewrite_path(
    path: str,
    path_rewrites: tuple[LsstComCamPathRewrite, ...],
) -> tuple[str, LsstComCamPathRewrite | None]:
    for rule in path_rewrites:
        source = rule.source_prefix
        if path == source:
            return rule.target_prefix, rule
        if source == "/" and path.startswith("/"):
            return _join_rewritten_path(rule.target_prefix, path[1:]), rule
        prefix = source if source.endswith("/") else source + "/"
        if path.startswith(prefix):
            suffix = path[len(prefix) :]
            return _join_rewritten_path(rule.target_prefix, suffix), rule
    return path, None


def _join_rewritten_path(target_prefix: str, suffix: str) -> str:
    if not suffix:
        return target_prefix
    if target_prefix == "/":
        return "/" + suffix.lstrip("/")
    return target_prefix.rstrip("/") + "/" + suffix.lstrip("/")


def _pair_lsstcomcam_visit_difference_rows(
    rows: list[dict[str, Any]],
    *,
    key_columns: tuple[str, ...] = DEFAULT_PAIR_KEY_COLUMNS,
) -> tuple[list[dict[str, dict[str, Any]]], dict[str, Any]]:
    visit_rows = sorted(
        (row for row in rows if _row_product(row) == "visit_image"),
        key=_sort_key,
    )
    difference_rows = sorted(
        (row for row in rows if _row_product(row) == "difference_image"),
        key=_sort_key,
    )
    visit_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in visit_rows:
        key = _pair_key(row, key_columns=key_columns)
        if key is None:
            continue
        visit_by_key.setdefault(key, []).append(row)
    difference_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in difference_rows:
        key = _pair_key(row, key_columns=key_columns)
        if key is None:
            continue
        difference_by_key.setdefault(key, []).append(row)

    duplicate_visit_keys = {
        key: sorted(matches, key=_registry_row_preference_key, reverse=True)
        for key, matches in visit_by_key.items()
        if len(matches) > 1
    }
    duplicate_keys = {
        key: sorted(matches, key=_registry_row_preference_key, reverse=True)
        for key, matches in difference_by_key.items()
        if len(matches) > 1
    }
    duplicate_visit_details = [
        {
            "key": list(key),
            "candidate_count": len(matches),
            "selected_path": _metadata_value(matches[0].get("path")),
            "candidate_paths": [
                _metadata_value(row.get("path")) for row in matches
            ],
        }
        for key, matches in sorted(
            duplicate_visit_keys.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
    ]
    duplicate_details = [
        {
            "key": list(key),
            "candidate_count": len(matches),
            "selected_path": _metadata_value(matches[0].get("path")),
            "candidate_paths": [
                _metadata_value(row.get("path")) for row in matches
            ],
        }
        for key, matches in sorted(
            duplicate_keys.items(),
            key=lambda item: tuple(str(value) for value in item[0]),
        )
    ]
    pairing_summary = {
        "pair_key_columns": list(key_columns),
        "visit_image_count": len(visit_rows),
        "selected_visit_image_count": len(visit_by_key),
        "difference_image_count": len(difference_rows),
        "duplicate_visit_key_count": len(duplicate_visit_details),
        "duplicate_visit_extra_candidate_count": sum(
            detail["candidate_count"] - 1
            for detail in duplicate_visit_details
        ),
        "duplicate_visit_keys": duplicate_visit_details,
        "duplicate_difference_key_count": len(duplicate_details),
        "duplicate_difference_extra_candidate_count": sum(
            detail["candidate_count"] - 1 for detail in duplicate_details
        ),
        "duplicate_difference_keys": duplicate_details,
        "selected_visit_rule": "highest_mtime_ns_then_path",
        "selected_difference_rule": "highest_mtime_ns_then_path",
    }
    selected_visit_rows = sorted(
        (
            sorted(matches, key=_registry_row_preference_key, reverse=True)[0]
            for matches in visit_by_key.values()
        ),
        key=_sort_key,
    )

    pairs: list[dict[str, dict[str, Any]]] = []
    for visit_row in selected_visit_rows:
        key = _pair_key(visit_row, key_columns=key_columns)
        if key is None:
            continue
        matches = difference_by_key.get(key, [])
        if not matches:
            continue
        selected = sorted(
            matches,
            key=_registry_row_preference_key,
            reverse=True,
        )[0]
        pairs.append(
            {
                "visit_image": visit_row,
                "difference_image": selected,
            }
        )
    pairing_summary["compatible_pair_count"] = len(pairs)
    return pairs, pairing_summary


def build_lsstcomcam_smoke_dataset_from_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> DatasetBuildResult:
    payload = load_manifest(manifest_path)
    registry_path = _registry_path_from_manifest(payload)
    sample_count = int(payload.get("sample_count", payload.get("limit", 8)))
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    stamp_size = int(payload.get("stamp_size", 51))
    if stamp_size <= 0 or stamp_size % 2 == 0:
        raise ValueError("stamp_size must be a positive odd integer")
    sample_seed = int(payload.get("sample_seed", payload.get("seed", 0)))
    split_seed = int(payload.get("split_seed", sample_seed))
    split_fractions = _split_fractions_from_manifest(payload)
    shuffle = bool(payload.get("shuffle", True))
    include_coadd_metadata = bool(
        payload.get(
            "include_coadd_metadata", payload.get("include_coadds", True)
        )
    )
    candidate_catalog_path = _candidate_catalog_path_from_manifest(payload)
    candidate_context = _candidate_context_from_manifest(payload)
    candidate_pixel_origin = int(payload.get("candidate_pixel_origin", 0))
    if candidate_pixel_origin not in {0, 1}:
        raise ValueError("candidate_pixel_origin must be 0 or 1")
    candidate_wcs_product = str(
        payload.get("candidate_wcs_product", "difference_image")
    )
    if candidate_wcs_product not in {"visit_image", "difference_image"}:
        raise ValueError(
            "candidate_wcs_product must be visit_image or difference_image"
        )
    image_hdu = payload.get("image_hdu", "IMAGE")
    difference_hdu = payload.get("difference_hdu", image_hdu)
    nan_policy = str(payload.get("nan_policy", "zero"))
    if nan_policy not in {"raise", "zero"}:
        raise ValueError("nan_policy must be 'raise' or 'zero'")

    manifest_template_mode = str(
        payload.get("template_mode", "search_minus_difference")
    )
    template_mode = manifest_template_mode.replace("-", "_")
    if template_mode not in {
        "search_minus_difference",
    }:
        raise ValueError("template_mode must be 'search_minus_difference'")

    filters = _filters_from_manifest(payload)
    path_rewrites = _path_rewrites_from_manifest(payload)
    selection = select_lsstcomcam_registry_rows(
        registry_path=registry_path,
        filters=filters,
        path_rewrites=path_rewrites,
    )
    if not selection.pairs:
        raise ValueError(
            "LSSTComCam registry filters produced no compatible "
            "visit_image + difference_image pairs"
        )

    candidate_rows: list[dict[str, Any]] = []
    pairs = list(selection.pairs)
    if candidate_catalog_path is None:
        center_x = _optional_int(payload.get("center_x"))
        center_y = _optional_int(payload.get("center_y"))
        samples = [
            LsstComCamStampSample(
                pair=pair,
                center_x=center_x,
                center_y=center_y,
                center_source=(
                    "manifest_pixel"
                    if center_x is not None or center_y is not None
                    else "image_center"
                ),
            )
            for pair in pairs
        ]
    else:
        candidate_rows = load_lsstcomcam_candidate_catalog(
            candidate_catalog_path
        )
        samples = _candidate_samples_for_pairs(
            pairs,
            candidate_rows,
            pixel_origin=candidate_pixel_origin,
            wcs_product=candidate_wcs_product,
        )
        if not samples:
            raise ValueError(
                "candidate_catalog produced no rows matching compatible "
                "visit_image + difference_image pairs"
            )
    if shuffle:
        rng = np.random.default_rng(sample_seed)
        order = rng.permutation(len(samples))
        samples = [samples[int(index)] for index in order]
    if sample_count > len(samples):
        raise ValueError(
            "sample_count exceeds compatible LSSTComCam sample count: "
            f"requested={sample_count} available={len(samples)}"
        )
    samples = samples[:sample_count]
    if candidate_catalog_path is not None:
        samples = [
            _resolve_candidate_sample_center(
                sample,
                image_hdu=image_hdu,
                difference_hdu=difference_hdu,
                wcs_product=candidate_wcs_product,
            )
            for sample in samples
        ]
    _validate_stamp_size_against_samples(
        samples,
        image_hdu=image_hdu,
        difference_hdu=difference_hdu,
        stamp_size=stamp_size,
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    search = np.zeros(
        (sample_count, stamp_size, stamp_size), dtype=np.float32
    )
    difference = np.zeros_like(search)
    template = np.zeros_like(search)
    labels = np.zeros((sample_count,), dtype=np.int64)
    split = np.zeros((sample_count,), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    nonfinite_counts = {"search": 0, "difference": 0, "template": 0}

    coadd_rows = _coadd_rows_by_product(selection.rows)
    for index, sample in enumerate(samples):
        visit_row = sample.pair["visit_image"]
        difference_row = sample.pair["difference_image"]
        search_stamp = read_fits_stamp(
            Path(str(visit_row["path"])),
            hdu=image_hdu,
            stamp_size=stamp_size,
            center_x=sample.center_x,
            center_y=sample.center_y,
        )
        difference_stamp = read_fits_stamp(
            Path(str(difference_row["path"])),
            hdu=difference_hdu,
            stamp_size=stamp_size,
            center_x=search_stamp.center_x,
            center_y=search_stamp.center_y,
        )
        if difference_stamp.image_shape != search_stamp.image_shape:
            raise ValueError(
                "visit_image and difference_image FITS HDUs must have "
                "matching image shapes before center reuse: "
                f"visit={search_stamp.image_shape} "
                f"difference={difference_stamp.image_shape}"
            )
        search_data, search_invalid = _finalize_stamp(
            search_stamp.data,
            nan_policy=nan_policy,
            name="search stamp",
        )
        difference_data, difference_invalid = _finalize_stamp(
            difference_stamp.data,
            nan_policy=nan_policy,
            name="difference stamp",
        )
        template_data, template_invalid = _finalize_stamp(
            search_data - difference_data,
            nan_policy=nan_policy,
            name="template stamp",
        )
        search[index] = search_data
        difference[index] = difference_data
        template[index] = template_data
        nonfinite_counts["search"] += search_invalid
        nonfinite_counts["difference"] += difference_invalid
        nonfinite_counts["template"] += template_invalid

        split_group = _split_group_for_pair(visit_row)
        split[index] = assign_split(split_group, split_seed, split_fractions)
        coadd_metadata = (
            _matched_coadd_metadata(visit_row, coadd_rows)
            if include_coadd_metadata
            else {
                "template_coadd_lookup_status": "not_requested",
                "deep_coadd_lookup_status": "not_requested",
            }
        )
        label = int(labels[index])
        row = {
            "candidate_id": _candidate_id(
                index,
                visit_row,
                candidate_row=sample.candidate_row,
            ),
            "exposure_id": _metadata_value(visit_row.get("visit")),
            "ccd_id": _metadata_value(visit_row.get("detector")),
            "band": _metadata_value(visit_row.get("band")),
            "x": int(search_stamp.center_x),
            "y": int(search_stamp.center_y),
            "center_source": sample.center_source,
            "split_group": split_group,
            "split": INDEX_TO_SPLIT[int(split[index])],
            "label": label,
            "label_source": LSSTCOMCAM_PLACEHOLDER_LABEL_SOURCE,
            "target_label_available": False,
            "dataset_source": LSSTCOMCAM_DATASET_SOURCE,
            "template_source": "search_minus_difference_image",
            "butler_registry_path": str(selection.registry_path),
            "search_hdu": str(search_stamp.hdu_name),
            "difference_hdu": str(difference_stamp.hdu_name),
            "stamp_size": stamp_size,
            "search_image_shape": list(search_stamp.image_shape),
            "difference_image_shape": list(difference_stamp.image_shape),
            **_prefixed_row_metadata("visit_image", visit_row),
            **_prefixed_row_metadata("difference_image", difference_row),
            **coadd_metadata,
            **_present_candidate_context(candidate_context),
        }
        if sample.candidate_row is not None:
            row.update(
                _prefixed_candidate_metadata(
                    sample.candidate_row,
                    candidate_catalog_path=candidate_catalog_path,
                )
            )
        rows.append(row)

    np.save(output_dir / "search.npy", search, allow_pickle=False)
    np.save(output_dir / "template.npy", template, allow_pickle=False)
    np.save(output_dir / "difference.npy", difference, allow_pickle=False)
    np.save(output_dir / "labels.npy", labels, allow_pickle=False)
    np.save(output_dir / "split.npy", split, allow_pickle=False)
    write_metadata_jsonl(output_dir / "metadata.jsonl", rows)
    maybe_write_metadata_parquet(output_dir / "metadata.parquet", rows)

    summary = {
        "dataset_dir": str(output_dir),
        "dataset_kind": LSSTCOMCAM_SMOKE_DATASET_KIND,
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "registry_path": str(selection.registry_path),
        "registry_filters": selection.filters,
        "path_rewrites": selection.path_rewrites,
        "registry_row_count": len(selection.rows),
        "compatible_pair_count": len(selection.pairs),
        "candidate_catalog_path": (
            str(candidate_catalog_path) if candidate_catalog_path else None
        ),
        **candidate_context,
        "candidate_catalog_row_count": len(candidate_rows),
        "candidate_matched_sample_count": (
            len(samples) if candidate_catalog_path is not None else 0
        ),
        "candidate_pixel_origin": (
            candidate_pixel_origin if candidate_catalog_path else None
        ),
        "candidate_wcs_product": (
            candidate_wcs_product if candidate_catalog_path else None
        ),
        "pairing": selection.pairing_summary,
        "sample_count": sample_count,
        "stamp_size": stamp_size,
        "sample_seed": sample_seed,
        "split_seed": split_seed,
        "shuffle": shuffle,
        "template_mode": template_mode,
        "manifest_template_mode": manifest_template_mode,
        "include_coadd_metadata": include_coadd_metadata,
        "nan_policy": nan_policy,
        "nonfinite_replaced": nonfinite_counts,
        "splits": _summarize_lsst_splits(rows),
        "saved": {
            "search": "search.npy",
            "template": "template.npy",
            "difference": "difference.npy",
            "labels": "labels.npy",
            "split": "split.npy",
            "metadata_jsonl": "metadata.jsonl",
        },
    }
    if (output_dir / "metadata.parquet").exists():
        summary["saved"]["metadata_parquet"] = "metadata.parquet"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_dataset_dir(
        output_dir,
        dataset_kind=LSSTCOMCAM_SMOKE_DATASET_KIND,
    )
    return DatasetBuildResult(output_dir=output_dir, summary=summary)


def check_lsstcomcam_candidate_catalog_from_manifest(
    *,
    manifest_path: Path,
    strict: bool = False,
) -> dict[str, Any]:
    """Preflight candidate-center catalog compatibility before stamp build."""
    manifest_path = manifest_path.expanduser().resolve()
    payload = load_manifest(manifest_path)
    registry_path = _registry_path_from_manifest(payload)
    filters = _filters_from_manifest(payload)
    path_rewrites = _path_rewrites_from_manifest(payload)
    selection = select_lsstcomcam_registry_rows(
        registry_path=registry_path,
        filters=filters,
        path_rewrites=path_rewrites,
    )
    candidate_catalog_path = _candidate_catalog_path_from_manifest(payload)
    candidate_context = _candidate_context_from_manifest(payload)
    result: dict[str, Any] = {
        "artifact_kind": "lsstcomcam_candidate_catalog_check",
        "manifest_path": str(manifest_path),
        "registry_path": str(selection.registry_path),
        "registry_filters": selection.filters,
        "path_rewrites": selection.path_rewrites,
        "registry_row_count": len(selection.rows),
        "compatible_pair_count": len(selection.pairs),
        "candidate_catalog_path": (
            str(candidate_catalog_path) if candidate_catalog_path else None
        ),
        **candidate_context,
        "candidate_catalog_exists": (
            candidate_catalog_path.exists()
            if candidate_catalog_path is not None
            else False
        ),
        "candidate_catalog_row_count": 0,
        "candidate_matched_row_count": 0,
        "candidate_unmatched_row_count": 0,
        "candidate_full_match": False,
        "candidate_center_columns": {
            "pixel_pairs": [],
            "sky_pairs": [],
        },
        "candidate_id_columns": [],
        "candidate_identity_column": None,
        "candidate_missing_id_count": 0,
        "candidate_duplicate_id_count": 0,
        "candidate_duplicate_ids": [],
        "strict": strict,
        "errors": [],
        "ok": False,
    }
    if candidate_catalog_path is None:
        result["errors"].append("manifest_missing_candidate_catalog")
        return result
    if not candidate_catalog_path.exists():
        result["errors"].append("candidate_catalog_not_found")
        return result

    candidate_rows = load_lsstcomcam_candidate_catalog(candidate_catalog_path)
    result["candidate_catalog_row_count"] = len(candidate_rows)
    columns = sorted({key for row in candidate_rows for key in row})
    result["candidate_catalog_columns"] = columns
    result["candidate_center_columns"] = {
        "pixel_pairs": _available_candidate_pixel_pairs(columns),
        "sky_pairs": _available_candidate_sky_pairs(columns),
    }
    result["candidate_id_columns"] = _available_candidate_id_columns(columns)
    if result["candidate_id_columns"]:
        identity = _candidate_identity_summary(
            candidate_rows,
        )
        result.update(identity)

    pair_keys = {
        _pair_key(pair["visit_image"], key_columns=DEFAULT_PAIR_KEY_COLUMNS)
        for pair in selection.pairs
    }
    matched_rows = [
        row
        for row in candidate_rows
        if _pair_key(row, key_columns=DEFAULT_PAIR_KEY_COLUMNS) in pair_keys
    ]
    result["candidate_matched_row_count"] = len(matched_rows)
    result["candidate_unmatched_row_count"] = len(candidate_rows) - len(
        matched_rows
    )
    result["candidate_full_match"] = len(matched_rows) == len(candidate_rows)
    if (
        not result["candidate_center_columns"]["pixel_pairs"]
        and not result["candidate_center_columns"]["sky_pairs"]
    ):
        result["errors"].append("candidate_center_columns_missing")
    missing_key_columns = [
        column for column in DEFAULT_PAIR_KEY_COLUMNS if column not in columns
    ]
    if missing_key_columns:
        result["errors"].append(
            "candidate_pair_key_columns_missing:"
            + ",".join(missing_key_columns)
        )
    if not result["candidate_id_columns"]:
        result["errors"].append("candidate_id_columns_missing")
    if not matched_rows:
        result["errors"].append("candidate_rows_match_no_compatible_pairs")
    if strict:
        if not result["candidate_full_match"]:
            result["errors"].append("candidate_rows_not_all_matched")
        if result["candidate_missing_id_count"]:
            result["errors"].append("candidate_identity_values_missing")
        if result["candidate_duplicate_id_count"]:
            result["errors"].append("candidate_identity_values_duplicate")
    result["ok"] = not result["errors"]
    return result


def plan_lsstcomcam_staging_from_manifest(
    *,
    manifest_path: Path,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Plan FITS inputs needed by an LSSTComCam smoke manifest."""
    manifest_path = manifest_path.expanduser().resolve()
    payload = load_manifest(manifest_path)
    registry_path = _registry_path_from_manifest(payload)
    sample_count = int(
        sample_count
        if sample_count is not None
        else payload.get("sample_count", payload.get("limit", 8))
    )
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    sample_seed = int(payload.get("sample_seed", payload.get("seed", 0)))
    shuffle = bool(payload.get("shuffle", True))
    candidate_catalog_path = _candidate_catalog_path_from_manifest(payload)
    candidate_pixel_origin = int(payload.get("candidate_pixel_origin", 0))
    if candidate_pixel_origin not in {0, 1}:
        raise ValueError("candidate_pixel_origin must be 0 or 1")
    candidate_wcs_product = str(
        payload.get("candidate_wcs_product", "difference_image")
    )
    if candidate_wcs_product not in {"visit_image", "difference_image"}:
        raise ValueError(
            "candidate_wcs_product must be visit_image or difference_image"
        )
    filters = _filters_from_manifest(payload)
    path_rewrites = _path_rewrites_from_manifest(payload)
    selection = select_lsstcomcam_registry_rows(
        registry_path=registry_path,
        filters=filters,
        path_rewrites=path_rewrites,
    )
    selected = _staging_plan_samples(
        selection.pairs,
        candidate_catalog_path=candidate_catalog_path,
        candidate_pixel_origin=candidate_pixel_origin,
        candidate_wcs_product=candidate_wcs_product,
        sample_count=sample_count,
        sample_seed=sample_seed,
        shuffle=shuffle,
    )
    coadd_rows = _coadd_rows_by_product(selection.rows)
    selected_rows = [
        _staging_plan_sample_row(
            index,
            sample,
            coadd_rows=coadd_rows,
        )
        for index, sample in enumerate(selected)
    ]
    required_files = _staging_plan_required_files(selected_rows)
    coadd_metadata_files = _staging_plan_coadd_metadata_files(selected_rows)
    return {
        "artifact_kind": "lsstcomcam_staging_plan",
        "manifest_path": str(manifest_path),
        "registry_path": str(selection.registry_path),
        "registry_filters": selection.filters,
        "path_rewrites": selection.path_rewrites,
        "registry_row_count": len(selection.rows),
        "compatible_pair_count": len(selection.pairs),
        "pairing_summary": selection.pairing_summary,
        "candidate_catalog_path": (
            str(candidate_catalog_path) if candidate_catalog_path else None
        ),
        "candidate_catalog_exists": (
            candidate_catalog_path.exists()
            if candidate_catalog_path is not None
            else False
        ),
        "sample_count_requested": sample_count,
        "sample_count_available": len(selected.available_samples),
        "sample_count_selected": len(selected_rows),
        "sample_seed": sample_seed,
        "shuffle": shuffle,
        "template_mode": payload.get(
            "template_mode",
            "search_minus_difference",
        ),
        "read_required_products": ["visit_image", "difference_image"],
        "metadata_only_products": ["template_coadd", "deep_coadd"],
        "note": (
            "Current LSSTComCam smoke builds read visit_image and "
            "difference_image FITS. The template stamp is derived as "
            "search_minus_difference; template_coadd and deep_coadd rows are "
            "metadata-only unless a future template mode reads them."
        ),
        "required_file_count": len(required_files),
        "missing_required_file_count": sum(
            1 for row in required_files if not row["exists"]
        ),
        "required_files": required_files,
        "metadata_only_file_count": len(coadd_metadata_files),
        "missing_metadata_only_file_count": sum(
            1 for row in coadd_metadata_files if not row["exists"]
        ),
        "metadata_only_files": coadd_metadata_files,
        "selected_samples": selected_rows,
    }


def stage_lsstcomcam_fits_from_manifest(
    *,
    manifest_path: Path,
    source_prefix: str,
    target_prefix: Path,
    search_roots: list[Path],
    sample_count: int | None = None,
    link_mode: str = "symlink",
    duplicate_policy: str = "same-size",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stage exact FITS files required by an LSSTComCam smoke manifest."""
    if link_mode not in STAGE_LINK_MODES:
        raise ValueError(
            "link_mode must be one of: " + ", ".join(STAGE_LINK_MODES)
        )
    if duplicate_policy not in STAGE_DUPLICATE_POLICIES:
        raise ValueError(
            "duplicate_policy must be one of: "
            + ", ".join(STAGE_DUPLICATE_POLICIES)
        )
    target_prefix = target_prefix.expanduser().resolve(strict=False)
    source_prefix = _normalize_rewrite_prefix(str(source_prefix))
    roots = [root.expanduser().resolve(strict=False) for root in search_roots]
    if not roots:
        raise ValueError("search_roots must contain at least one path")
    missing_roots = [str(root) for root in roots if not root.exists()]
    if missing_roots:
        raise FileNotFoundError(
            "search root not found: " + ", ".join(missing_roots)
        )

    plan = plan_lsstcomcam_staging_from_manifest(
        manifest_path=manifest_path,
        sample_count=sample_count,
    )
    candidates = _stage_candidate_fits(roots)
    by_name: dict[str, list[Path]] = {}
    for candidate in candidates:
        by_name.setdefault(candidate.name, []).append(candidate)

    staged: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []

    for required in plan["required_files"]:
        required_path = str(required["path"])
        relative = _relative_to_stage_prefix(required_path, source_prefix)
        target = target_prefix.joinpath(*relative.parts)
        matches = _matching_stage_candidates(
            by_name=by_name,
            relative_path=relative,
        )
        if not matches:
            missing.append(
                {
                    "product": required["product"],
                    "required_path": required_path,
                    "relative_path": relative.as_posix(),
                    "target_path": str(target),
                    "sample_indices": required["sample_indices"],
                }
            )
            continue
        selected, duplicate_detail = _select_stage_candidate(
            matches,
            duplicate_policy=duplicate_policy,
        )
        if selected is None:
            ambiguous.append(
                {
                    "product": required["product"],
                    "required_path": required_path,
                    "relative_path": relative.as_posix(),
                    "target_path": str(target),
                    "sample_indices": required["sample_indices"],
                    "matches": [
                        _stage_candidate_summary(match) for match in matches
                    ],
                    "reason": duplicate_detail,
                }
            )
            continue

        row = {
            "product": required["product"],
            "required_path": required_path,
            "relative_path": relative.as_posix(),
            "target_path": str(target),
            "source_path": str(selected),
            "link_mode": link_mode,
            "sample_indices": required["sample_indices"],
        }
        if duplicate_detail:
            row["duplicate_resolution"] = duplicate_detail
        if (target.exists() or target.is_symlink()) and not force:
            target_status = _existing_stage_target_status(
                target,
                selected,
                link_mode=link_mode,
            )
            if not target_status["matches"]:
                conflicts.append(
                    {
                        **row,
                        "action": "existing_target_conflict",
                        "target_status": target_status,
                    }
                )
                continue
            existing.append(
                {
                    **row,
                    "action": "already_exists",
                    "target_status": target_status,
                }
            )
            continue
        if not dry_run:
            _stage_one_fits(
                source=selected,
                target=target,
                link_mode=link_mode,
                force=force,
            )
        action = "would_stage" if dry_run else "staged"
        staged.append({**row, "action": action})

    return {
        "artifact_kind": "lsstcomcam_fits_stage_result",
        "manifest_path": plan["manifest_path"],
        "registry_path": plan["registry_path"],
        "source_prefix": source_prefix,
        "target_prefix": str(target_prefix),
        "search_roots": [str(root) for root in roots],
        "sample_count_requested": plan["sample_count_requested"],
        "sample_count_selected": plan["sample_count_selected"],
        "required_file_count": plan["required_file_count"],
        "candidate_file_count": len(candidates),
        "link_mode": link_mode,
        "duplicate_policy": duplicate_policy,
        "force": force,
        "dry_run": dry_run,
        "staged_count": len(staged),
        "existing_count": len(existing),
        "missing_count": len(missing),
        "ambiguous_count": len(ambiguous),
        "conflict_count": len(conflicts),
        "resolved_count": len(staged) + len(existing),
        "ok": not missing and not ambiguous and not conflicts,
        "staged": staged,
        "existing": existing,
        "missing": missing,
        "ambiguous": ambiguous,
        "conflicts": conflicts,
    }


def _stage_candidate_fits(search_roots: list[Path]) -> list[Path]:
    seen: set[str] = set()
    candidates: list[Path] = []
    patterns = ("*.fits", "*.fits.gz", "*.fits.fz")
    for root in search_roots:
        items = [root] if root.is_file() else []
        if root.is_dir():
            for pattern in patterns:
                items.extend(root.rglob(pattern))
        for item in items:
            if not item.is_file():
                continue
            resolved = _safe_resolve_path(item)
            if resolved is None:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(resolved)
    return sorted(candidates, key=lambda path: str(path))


def _relative_to_stage_prefix(
    path_value: str,
    source_prefix: str,
) -> PurePosixPath:
    if path_value == source_prefix:
        raise ValueError(
            f"required file path must be below source_prefix: {path_value}"
        )
    if source_prefix == "/":
        suffix = path_value.lstrip("/")
    else:
        prefix = source_prefix + "/"
        if not path_value.startswith(prefix):
            raise ValueError(
                "required file path is outside source_prefix: "
                f"{path_value} not under {source_prefix}"
            )
        suffix = path_value[len(prefix) :]
    relative = PurePosixPath(suffix)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"unsafe relative stage path: {suffix}")
    return relative


def _matching_stage_candidates(
    *,
    by_name: dict[str, list[Path]],
    relative_path: PurePosixPath,
) -> list[Path]:
    relative_text = relative_path.as_posix()
    suffix = "/" + relative_text
    return [
        candidate
        for candidate in by_name.get(relative_path.name, [])
        if candidate.as_posix().endswith(suffix)
    ]


def _select_stage_candidate(
    matches: list[Path],
    *,
    duplicate_policy: str,
) -> tuple[Path | None, str | None]:
    ordered = sorted(matches, key=lambda path: str(path))
    if len(ordered) == 1:
        return ordered[0], None
    if duplicate_policy == "first":
        return ordered[0], "first"
    if duplicate_policy == "same-size":
        sizes = {_safe_file_size(match) for match in ordered}
        if len(sizes) == 1 and None not in sizes:
            return ordered[0], "same-size"
        return None, "multiple_matches_different_sizes"
    return None, "multiple_matches"


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _safe_file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _stage_candidate_summary(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": _safe_file_size(path)}


def _stage_existing_target_summary(path: Path) -> dict[str, Any]:
    summary = _stage_candidate_summary(path)
    summary["is_symlink"] = path.is_symlink()
    summary["mtime_ns"] = _safe_file_mtime_ns(path)
    if path.is_symlink():
        resolved = _safe_resolve_path(path)
        summary["resolved_path"] = str(resolved) if resolved else None
    return summary


def _safe_resolve_path(path: Path) -> Path | None:
    try:
        # Requiring the target to exist makes dangling and cyclic symlinks
        # behave consistently across supported Python versions.
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def _existing_stage_target_status(
    target: Path,
    selected: Path,
    *,
    link_mode: str,
) -> dict[str, Any]:
    selected_summary = _stage_existing_target_summary(selected)
    target_summary = _stage_existing_target_summary(target)
    if target.is_symlink():
        resolved = _safe_resolve_path(target)
        selected_resolved = _safe_resolve_path(selected)
        matches = (
            resolved is not None
            and selected_resolved is not None
            and str(resolved) == str(selected_resolved)
        )
        return {
            "matches": matches,
            "reason": "symlink_target_match" if matches else "stale_symlink",
            "target": target_summary,
            "selected": selected_summary,
        }
    if not target.is_file():
        return {
            "matches": False,
            "reason": "existing_target_not_file",
            "target": target_summary,
            "selected": selected_summary,
        }
    if _same_existing_file(target, selected):
        return {
            "matches": True,
            "reason": "same_file",
            "target": target_summary,
            "selected": selected_summary,
        }
    if link_mode == "copy":
        size_matches = (
            target_summary["size_bytes"] == selected_summary["size_bytes"]
        )
        mtime_matches = (
            target_summary["mtime_ns"] == selected_summary["mtime_ns"]
        )
        matches = bool(size_matches and mtime_matches)
        return {
            "matches": matches,
            "reason": "copy_stat_match" if matches else "stale_copy",
            "target": target_summary,
            "selected": selected_summary,
        }
    return {
        "matches": False,
        "reason": f"existing_target_not_{link_mode}",
        "target": target_summary,
        "selected": selected_summary,
    }


def _stage_one_fits(
    *,
    source: Path,
    target: Path,
    link_mode: str,
    force: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            raise IsADirectoryError(f"stage target is a directory: {target}")
        if force:
            target.unlink()
    if link_mode == "symlink":
        target.symlink_to(source)
    elif link_mode == "copy":
        shutil.copy2(source, target)
    elif link_mode == "hardlink":
        os.link(source, target)
    else:
        raise ValueError(f"unsupported link_mode: {link_mode}")


@dataclass(slots=True)
class LsstComCamStagingPlanSamples:
    available_samples: list[dict[str, Any]]
    selected_samples: list[dict[str, Any]]

    def __iter__(self):
        return iter(self.selected_samples)


def _staging_plan_samples(
    pairs: list[dict[str, dict[str, Any]]],
    *,
    candidate_catalog_path: Path | None,
    candidate_pixel_origin: int,
    candidate_wcs_product: str,
    sample_count: int,
    sample_seed: int,
    shuffle: bool,
) -> LsstComCamStagingPlanSamples:
    if candidate_catalog_path is None:
        available = [
            {
                "pair": pair,
                "candidate_row": None,
                "candidate_catalog_row_index": None,
                "center_source": "manifest_or_image_center",
                "center_x": None,
                "center_y": None,
                "sky_center_columns": [],
            }
            for pair in pairs
        ]
    else:
        if not candidate_catalog_path.exists():
            raise FileNotFoundError(
                f"candidate catalog not found: {candidate_catalog_path}"
            )
        candidate_rows = load_lsstcomcam_candidate_catalog(
            candidate_catalog_path,
        )
        available = _candidate_staging_plan_samples(
            pairs,
            candidate_rows,
            pixel_origin=candidate_pixel_origin,
            wcs_product=candidate_wcs_product,
        )
        if not available:
            raise ValueError(
                "candidate_catalog produced no rows matching compatible "
                "visit_image + difference_image pairs"
            )
    ordered = list(available)
    if shuffle:
        rng = np.random.default_rng(sample_seed)
        order = rng.permutation(len(ordered))
        ordered = [ordered[int(index)] for index in order]
    if sample_count > len(ordered):
        raise ValueError(
            "sample_count exceeds compatible LSSTComCam sample count: "
            f"requested={sample_count} available={len(ordered)}"
        )
    return LsstComCamStagingPlanSamples(
        available_samples=available,
        selected_samples=ordered[:sample_count],
    )


def _candidate_staging_plan_samples(
    pairs: list[dict[str, dict[str, Any]]],
    candidate_rows: list[dict[str, Any]],
    *,
    pixel_origin: int,
    wcs_product: str,
) -> list[dict[str, Any]]:
    pair_by_key = {
        _pair_key(
            pair["visit_image"],
            key_columns=DEFAULT_PAIR_KEY_COLUMNS,
        ): pair
        for pair in pairs
    }
    samples: list[dict[str, Any]] = []
    for row_index, candidate in enumerate(candidate_rows):
        key = _pair_key(candidate, key_columns=DEFAULT_PAIR_KEY_COLUMNS)
        if key is None:
            continue
        pair = pair_by_key.get(key)
        if pair is None:
            continue
        center = _candidate_staging_center(
            candidate,
            pixel_origin=pixel_origin,
            wcs_product=wcs_product,
        )
        samples.append(
            {
                "pair": pair,
                "candidate_row": dict(candidate),
                "candidate_catalog_row_index": row_index,
                **center,
            }
        )
    return samples


def _candidate_staging_center(
    candidate: dict[str, Any],
    *,
    pixel_origin: int,
    wcs_product: str,
) -> dict[str, Any]:
    pixel = _candidate_pixel_xy(candidate, pixel_origin=pixel_origin)
    if pixel is not None:
        return {
            "center_source": "candidate_catalog_pixel",
            "center_x": pixel[0],
            "center_y": pixel[1],
            "sky_center_columns": [],
        }
    sky_pair = _candidate_sky_column_used(candidate)
    if sky_pair is not None:
        return {
            "center_source": (
                f"candidate_catalog_sky_{wcs_product}_wcs_pending_fits"
            ),
            "center_x": None,
            "center_y": None,
            "sky_center_columns": list(sky_pair),
        }
    raise ValueError(
        "candidate_catalog row must include pixel center columns "
        "(x/y, center_x/center_y, x_image/y_image, coord_x/coord_y) "
        "or sky columns (ra/dec, ra_deg/dec_deg, coord_ra/coord_dec)"
    )


def _candidate_sky_column_used(
    row: dict[str, Any],
) -> tuple[str, str] | None:
    for ra_name, dec_name in _candidate_sky_column_pairs():
        ra = _optional_float(row.get(ra_name))
        dec = _optional_float(row.get(dec_name))
        if ra is None and dec is None:
            continue
        if ra is None or dec is None:
            raise ValueError(
                "candidate_catalog must provide both "
                f"{ra_name} and {dec_name}"
            )
        return ra_name, dec_name
    return None


def _staging_plan_sample_row(
    index: int,
    sample: dict[str, Any],
    *,
    coadd_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pair = sample["pair"]
    visit_row = pair["visit_image"]
    difference_row = pair["difference_image"]
    candidate_row = sample.get("candidate_row")
    candidate_with_index = None
    if candidate_row is not None:
        candidate_with_index = dict(candidate_row)
        candidate_with_index["candidate_catalog_row_index"] = sample[
            "candidate_catalog_row_index"
        ]
    row = {
        "sample_index": index,
        "visit": _metadata_value(visit_row.get("visit")),
        "detector": _metadata_value(visit_row.get("detector")),
        "band": _metadata_value(visit_row.get("band")),
        "candidate_id": _candidate_id(
            index,
            visit_row,
            candidate_row=candidate_with_index,
        ),
        "candidate_catalog_row_index": sample.get(
            "candidate_catalog_row_index"
        ),
        "center_source": sample["center_source"],
        "center_x": sample["center_x"],
        "center_y": sample["center_y"],
        "sky_center_columns": sample["sky_center_columns"],
        "visit_image_path": str(_metadata_value(visit_row.get("path"))),
        "visit_image_original_path": _metadata_value(
            visit_row.get("original_path")
        ),
        "difference_image_path": str(
            _metadata_value(difference_row.get("path"))
        ),
        "difference_image_original_path": _metadata_value(
            difference_row.get("original_path")
        ),
        "visit_image_exists": Path(str(visit_row["path"])).is_file(),
        "difference_image_exists": Path(
            str(difference_row["path"])
        ).is_file(),
    }
    row.update(_staging_plan_matched_coadd_paths(visit_row, coadd_rows))
    return row


def _staging_plan_matched_coadd_paths(
    visit_row: dict[str, Any],
    coadd_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for product in ("template_coadd", "deep_coadd"):
        row = _match_coadd_row(visit_row, coadd_rows.get(product, []))
        if row is None:
            result[f"{product}_path"] = None
            result[f"{product}_original_path"] = None
            result[f"{product}_exists"] = False
            continue
        path = str(_metadata_value(row.get("path")))
        result[f"{product}_path"] = path
        result[f"{product}_original_path"] = _metadata_value(
            row.get("original_path")
        )
        result[f"{product}_exists"] = Path(path).is_file()
    return result


def _staging_plan_required_files(
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected_rows:
        for product, key in (
            ("visit_image", "visit_image_path"),
            ("difference_image", "difference_image_path"),
        ):
            path = row[key]
            entry = by_key.setdefault(
                (product, path),
                {
                    "product": product,
                    "path": path,
                    "exists": Path(path).is_file(),
                    "original_paths": [],
                    "sample_indices": [],
                },
            )
            original = row.get(f"{product}_original_path")
            if original and original not in entry["original_paths"]:
                entry["original_paths"].append(original)
            entry["sample_indices"].append(row["sample_index"])
    return [
        by_key[key]
        for key in sorted(by_key, key=lambda item: (item[0], item[1]))
    ]


def _staging_plan_coadd_metadata_files(
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected_rows:
        for product, key in (
            ("template_coadd", "template_coadd_path"),
            ("deep_coadd", "deep_coadd_path"),
        ):
            path = row.get(key)
            if not path:
                continue
            entry = by_key.setdefault(
                (product, path),
                {
                    "product": product,
                    "path": path,
                    "exists": Path(path).is_file(),
                    "original_paths": [],
                    "sample_indices": [],
                },
            )
            original = row.get(f"{product}_original_path")
            if original and original not in entry["original_paths"]:
                entry["original_paths"].append(original)
            entry["sample_indices"].append(row["sample_index"])
    return [
        by_key[key]
        for key in sorted(by_key, key=lambda item: (item[0], item[1]))
    ]


def _candidate_identity_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    identities = []
    missing = 0
    columns_used = set()
    for row in rows:
        value, column = _candidate_row_identity_value(row)
        if value is None:
            missing += 1
            continue
        if column is not None:
            columns_used.add(column)
        visit = _canonical_filter_value(row.get("visit"))
        detector = _canonical_filter_value(row.get("detector"))
        band = _canonical_filter_value(row.get("band"))
        identities.append(f"lsstcomcam-{visit}-{detector}-{band}-{value}")
    counts = Counter(identities)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    columns_used_sorted = sorted(columns_used)
    return {
        "candidate_identity_column": (
            columns_used_sorted[0] if len(columns_used_sorted) == 1 else None
        ),
        "candidate_identity_columns_used": columns_used_sorted,
        "candidate_missing_id_count": missing,
        "candidate_unique_id_count": len(counts),
        "candidate_duplicate_id_count": len(duplicates),
        "candidate_duplicate_ids": duplicates[:10],
    }


def _candidate_row_identity_value(
    row: dict[str, Any],
) -> tuple[str | None, str | None]:
    for column in _candidate_id_column_names():
        value = _canonical_filter_value(row.get(column))
        if value is not None:
            return value, column
    return None, None


def read_fits_stamp(
    path: Path,
    *,
    hdu: str | int,
    stamp_size: int,
    center_x: int | None = None,
    center_y: int | None = None,
) -> FitsStamp:
    """Read a small centered FITS stamp without materializing full images."""
    try:
        from astropy.io import fits
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "astropy is required to read LSSTComCam FITS smoke stamps"
        ) from exc

    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        image_hdu = _select_fits_image_hdu(hdul, hdu)
        shape = _fits_hdu_image_shape(image_hdu)
        height, width = shape
        center_x = width // 2 if center_x is None else int(center_x)
        center_y = height // 2 if center_y is None else int(center_y)
        half = stamp_size // 2
        y0 = center_y - half
        y1 = y0 + stamp_size
        x0 = center_x - half
        x1 = x0 + stamp_size
        if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
            raise ValueError(
                f"stamp centered at x={center_x}, y={center_y} with "
                f"size={stamp_size} does not fit inside {path}"
            )
        section = getattr(image_hdu, "section", None)
        if section is not None:
            data = section[y0:y1, x0:x1]
        else:
            data = image_hdu.data[y0:y1, x0:x1]
        return FitsStamp(
            data=np.asarray(data, dtype=np.float32),
            center_x=center_x,
            center_y=center_y,
            image_shape=shape,
            hdu_name=str(image_hdu.name),
        )


def load_lsstcomcam_candidate_catalog(path: Path) -> list[dict[str, Any]]:
    """Load optional candidate-center metadata for LSSTComCam stamps."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"candidate_catalog not found: {path}")
    suffix = path.suffix.lower()
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required when candidate_catalog is used"
        ) from exc
    if suffix in {".parq", ".parquet"}:
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(
            "candidate_catalog must be parquet, csv, jsonl, ndjson, or json"
        )
    if df.empty:
        raise ValueError(f"candidate_catalog contains no rows: {path}")
    rows = [_row_metadata(row.to_dict()) for _, row in df.iterrows()]
    return rows


def _filter_lsstcomcam_dataframe(df, filters: dict[str, Any]):
    product_values = _first_filter_values(
        filters,
        ("products", "product", "dataset_types", "dataset_type"),
    )
    butler_datasettype_values = _first_filter_values(
        filters,
        ("butler_datasettypes", "butler_datasettype"),
    )
    if product_values is None and butler_datasettype_values is None:
        product_values = list(DEFAULT_FILTER_PRODUCTS)
    if "product" not in df.columns and "butler_datasettype" not in df.columns:
        raise ValueError(
            "LSSTComCam registry must contain a product or "
            "butler_datasettype column"
        )
    if product_values is not None:
        product_column = (
            "product" if "product" in df.columns else "butler_datasettype"
        )
        df = _filter_dataframe_values(
            df,
            column=product_column,
            values=product_values,
            keep_missing=False,
        )
    if butler_datasettype_values is not None:
        if "butler_datasettype" not in df.columns:
            raise ValueError(
                "butler_datasettype filter requested but registry has no "
                "butler_datasettype column"
            )
        df = _filter_dataframe_values(
            df,
            column="butler_datasettype",
            values=butler_datasettype_values,
            keep_missing=False,
        )
    df = df.copy()

    for column, keys in (
        ("date", ("date", "dates")),
        ("band", ("band", "bands")),
        ("physical_filter", ("physical_filter", "physical_filters")),
        ("visit", ("visit", "visits")),
        ("detector", ("detector", "detectors")),
        ("tract", ("tract", "tracts")),
        ("patch", ("patch", "patches")),
    ):
        values = _first_filter_values(filters, keys)
        if values is None:
            continue
        df = _filter_dataframe_values(
            df,
            column=column,
            values=values,
            keep_missing=True,
        )

    for column, min_key, max_key in (
        ("date", "date_min", "date_max"),
        ("visit", "visit_min", "visit_max"),
        ("detector", "detector_min", "detector_max"),
        ("tract", "tract_min", "tract_max"),
        ("patch", "patch_min", "patch_max"),
    ):
        df = _filter_dataframe_range(
            df,
            column=column,
            lower=filters.get(min_key),
            upper=filters.get(max_key),
            keep_missing=True,
        )

    query = filters.get("query") or filters.get("butler_query")
    if query:
        df = _safe_query(df, str(query), source="LSSTComCam registry query")
    return df


def _filter_dataframe_values(
    df,
    *,
    column: str,
    values: list[Any],
    keep_missing: bool,
):
    if column not in df.columns:
        raise ValueError(
            f"registry filter requested missing column {column!r}"
        )
    series = df[column]
    wanted = {_canonical_filter_value(value) for value in values}
    observed = series.map(_canonical_filter_value)
    mask = observed.isin(wanted)
    if keep_missing:
        mask = mask | _optional_product_missing_mask(df, series)
    return df[mask]


def _filter_dataframe_range(
    df,
    *,
    column: str,
    lower: Any,
    upper: Any,
    keep_missing: bool,
):
    if lower is None and upper is None:
        return df
    if column not in df.columns:
        raise ValueError(
            f"registry range requested missing column {column!r}"
        )
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required when LSSTComCam registry filters are used"
        ) from exc

    series = df[column]
    canonical = series.map(_canonical_filter_value)
    numeric = pd.to_numeric(canonical, errors="coerce")
    invalid = numeric.isna() & ~series.map(_is_missing)
    if bool(invalid.any()):
        raise ValueError(
            "registry range requested non-numeric values in column "
            f"{column!r}"
        )
    mask = pd.Series(True, index=df.index)
    if lower is not None:
        mask = mask & (numeric >= _numeric_range_bound(lower, column=column))
    if upper is not None:
        mask = mask & (numeric <= _numeric_range_bound(upper, column=column))
    if keep_missing:
        mask = mask | _optional_product_missing_mask(df, series)
    return df[mask]


def _optional_product_missing_mask(df, series):
    missing = series.map(_is_missing)
    if not bool(missing.any()):
        return missing
    return missing & ~_pair_forming_product_mask(df)


def _pair_forming_product_mask(df):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required when LSSTComCam registry filters are used"
        ) from exc
    mask = pd.Series(False, index=df.index)
    for column in ("product", "butler_datasettype"):
        if column not in df.columns:
            continue
        mask = mask | df[column].map(
            lambda value: str(value) in {"visit_image", "difference_image"}
        )
    return mask


def _filters_from_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    _validate_manifest_filter_aliases(payload)
    for key in (
        "dataset_types",
        "dataset_type",
        "products",
        "product",
        "butler_datasettypes",
        "butler_datasettype",
        "date",
        "dates",
        "date_min",
        "date_max",
        "band",
        "bands",
        "physical_filter",
        "physical_filters",
        "visit",
        "visits",
        "visit_min",
        "visit_max",
        "detector",
        "detectors",
        "detector_min",
        "detector_max",
        "tract",
        "tracts",
        "tract_min",
        "tract_max",
        "patch",
        "patches",
        "patch_min",
        "patch_max",
        "query",
        "butler_query",
    ):
        if key in payload:
            filters[key] = payload[key]
    return filters


def _validate_manifest_filter_aliases(payload: dict[str, Any]) -> None:
    for plural, singular in (
        ("dataset_types", "dataset_type"),
        ("products", "product"),
        ("butler_datasettypes", "butler_datasettype"),
        ("bands", "band"),
        ("physical_filters", "physical_filter"),
        ("visits", "visit"),
        ("detectors", "detector"),
        ("tracts", "tract"),
        ("patches", "patch"),
        ("dates", "date"),
    ):
        if plural not in payload or singular not in payload:
            continue
        plural_values = _string_list(payload[plural])
        singular_values = _string_list(payload[singular])
        if plural_values != singular_values:
            raise ValueError(
                "manifest cannot specify conflicting "
                f"{plural} and {singular} filters"
            )


def _manifest_filter_summary(filters: dict[str, Any]) -> dict[str, Any]:
    summary = dict(filters)
    dataset_types = _string_list(
        summary.get("products")
        or summary.get("product")
        or summary.get("dataset_types")
        or summary.get("dataset_type")
        or DEFAULT_FILTER_PRODUCTS
    )
    summary["dataset_types"] = dataset_types
    if summary.get("butler_datasettypes") or summary.get(
        "butler_datasettype"
    ):
        summary["butler_datasettypes"] = _string_list(
            summary.get("butler_datasettypes")
            or summary.get("butler_datasettype")
        )
    summary["missing_filter_values_kept"] = True
    summary["missing_filter_values_kept_for_columns"] = [
        "date",
        "band",
        "physical_filter",
        "visit",
        "detector",
        "tract",
        "patch",
    ]
    return {
        key: _manifest_filter_summary_value(value)
        for key, value in summary.items()
    }


def _registry_path_from_manifest(payload: dict[str, Any]) -> Path:
    value = (
        payload.get("registry")
        or payload.get("registry_path")
        or payload.get("lsstcomcam_registry")
        or payload.get("butler_registry")
    )
    if not value:
        raise ValueError(
            "manifest is missing required field 'registry' "
            "(or registry_path / lsstcomcam_registry / butler_registry)"
        )
    return Path(str(value)).expanduser()


def _candidate_catalog_path_from_manifest(
    payload: dict[str, Any],
) -> Path | None:
    value = (
        payload.get("candidate_catalog")
        or payload.get("candidate_catalog_path")
        or payload.get("dia_source_catalog")
        or payload.get("source_catalog")
    )
    if not value:
        return None
    return Path(str(value)).expanduser()


def _candidate_context_from_manifest(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        field: (
            _metadata_value(payload[field])
            if field in payload and payload[field] is not None
            else None
        )
        for field in LSSTCOMCAM_CANDIDATE_CONTEXT_FIELDS
    }


def _present_candidate_context(context: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in context.items() if value is not None}


def _split_fractions_from_manifest(
    payload: dict[str, Any],
) -> tuple[float, float, float]:
    raw = payload.get("split_fractions", [0.7, 0.15, 0.15])
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("split_fractions must contain exactly three values")
    values = tuple(float(value) for value in raw)
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("split_fractions must sum to 1.0")
    return values


def _select_fits_image_hdu(hdul, hdu: str | int):
    if isinstance(hdu, int):
        try:
            image_hdu = hdul[hdu]
        except IndexError as exc:
            raise ValueError(f"FITS HDU {hdu!r} not found") from exc
    else:
        try:
            image_hdu = hdul[str(hdu)]
        except KeyError as exc:
            raise ValueError(f"FITS HDU {hdu!r} not found") from exc
    if len(image_hdu.shape) != 2:
        raise ValueError(
            f"FITS HDU {hdu!r} does not contain 2D image data; "
            "LSSTComCam smoke datasets currently require 2D image HDUs"
        )
    return image_hdu


def _fits_hdu_image_shape(image_hdu) -> tuple[int, int]:
    return tuple(int(value) for value in image_hdu.shape)


def _fits_image_shape(path: Path, *, hdu: str | int) -> tuple[int, int]:
    try:
        from astropy.io import fits
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "astropy is required to read LSSTComCam FITS smoke stamps"
        ) from exc
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        return _fits_hdu_image_shape(_select_fits_image_hdu(hdul, hdu))


def _candidate_samples_for_pairs(
    pairs: list[dict[str, dict[str, Any]]],
    candidate_rows: list[dict[str, Any]],
    *,
    pixel_origin: int,
    wcs_product: str,
) -> list[LsstComCamStampSample]:
    pair_by_key = {
        _pair_key(
            pair["visit_image"],
            key_columns=DEFAULT_PAIR_KEY_COLUMNS,
        ): pair
        for pair in pairs
    }
    samples: list[LsstComCamStampSample] = []
    for row_index, candidate in enumerate(candidate_rows):
        key = _pair_key(candidate, key_columns=DEFAULT_PAIR_KEY_COLUMNS)
        if key is None:
            continue
        pair = pair_by_key.get(key)
        if pair is None:
            continue
        center = _candidate_staging_center(
            candidate,
            pixel_origin=pixel_origin,
            wcs_product=wcs_product,
        )
        candidate_with_index = dict(candidate)
        candidate_with_index["candidate_catalog_row_index"] = row_index
        samples.append(
            LsstComCamStampSample(
                pair=pair,
                center_x=center["center_x"],
                center_y=center["center_y"],
                center_source=center["center_source"],
                candidate_row=candidate_with_index,
            )
        )
    return samples


def _resolve_candidate_sample_center(
    sample: LsstComCamStampSample,
    *,
    image_hdu: str | int,
    difference_hdu: str | int,
    wcs_product: str,
) -> LsstComCamStampSample:
    if sample.center_x is not None and sample.center_y is not None:
        return sample
    if not sample.center_source.endswith("_wcs_pending_fits"):
        return sample
    if sample.candidate_row is None:
        raise ValueError("candidate sky WCS center requires candidate_row")
    sky = _candidate_sky_radec(sample.candidate_row)
    if sky is None:
        raise ValueError(
            "candidate_catalog row must include pixel center columns "
            "(x/y, center_x/center_y, x_image/y_image, coord_x/coord_y) "
            "or sky columns (ra/dec, ra_deg/dec_deg, coord_ra/coord_dec)"
        )
    source_row = sample.pair[wcs_product]
    hdu = image_hdu if wcs_product == "visit_image" else difference_hdu
    x, y = _sky_to_image_xy(
        Path(str(source_row["path"])),
        hdu=hdu,
        ra_deg=sky[0],
        dec_deg=sky[1],
    )
    return replace(
        sample,
        center_x=x,
        center_y=y,
        center_source=f"candidate_catalog_sky_{wcs_product}_wcs",
    )


def _candidate_pixel_xy(
    row: dict[str, Any],
    *,
    pixel_origin: int,
) -> tuple[int, int] | None:
    for x_name, y_name in _candidate_pixel_column_pairs():
        x = _optional_float(row.get(x_name))
        y = _optional_float(row.get(y_name))
        if x is None and y is None:
            continue
        if x is None or y is None:
            raise ValueError(
                f"candidate_catalog must provide both {x_name} and {y_name}"
            )
        return (
            int(np.rint(x - pixel_origin)),
            int(np.rint(y - pixel_origin)),
        )
    return None


def _candidate_pixel_column_pairs() -> tuple[tuple[str, str], ...]:
    return (
        ("x", "y"),
        ("center_x", "center_y"),
        ("x_image", "y_image"),
        ("coord_x", "coord_y"),
        ("slot_Centroid_x", "slot_Centroid_y"),
        ("base_SdssCentroid_x", "base_SdssCentroid_y"),
    )


def _available_candidate_pixel_pairs(columns: list[str]) -> list[list[str]]:
    available = set(columns)
    return [
        [x_name, y_name]
        for x_name, y_name in _candidate_pixel_column_pairs()
        if x_name in available and y_name in available
    ]


def _candidate_sky_radec(row: dict[str, Any]) -> tuple[float, float] | None:
    for ra_name, dec_name in _candidate_sky_column_pairs():
        ra = _optional_float(row.get(ra_name))
        dec = _optional_float(row.get(dec_name))
        if ra is None and dec is None:
            continue
        if ra is None or dec is None:
            raise ValueError(
                "candidate_catalog must provide both "
                f"{ra_name} and {dec_name}"
            )
        return ra, dec
    return None


def _candidate_sky_column_pairs() -> tuple[tuple[str, str], ...]:
    return (
        ("ra_deg", "dec_deg"),
        ("ra", "dec"),
        ("coord_ra", "coord_dec"),
    )


def _available_candidate_sky_pairs(columns: list[str]) -> list[list[str]]:
    available = set(columns)
    return [
        [ra_name, dec_name]
        for ra_name, dec_name in _candidate_sky_column_pairs()
        if ra_name in available and dec_name in available
    ]


def _available_candidate_id_columns(columns: list[str]) -> list[str]:
    available = set(columns)
    return [
        column
        for column in _candidate_id_column_names()
        if column in available
    ]


def _candidate_id_column_names() -> tuple[str, ...]:
    return (
        "candidate_id",
        "diaSourceId",
        "dia_source_id",
        "sourceId",
        "source_id",
        "objectId",
        "object_id",
    )


def _sky_to_image_xy(
    path: Path,
    *,
    hdu: str | int,
    ra_deg: float,
    dec_deg: float,
) -> tuple[int, int]:
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "astropy is required for sky-coordinate candidate centers"
        ) from exc
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        image_hdu = _select_fits_image_hdu(hdul, hdu)
        wcs = WCS(image_hdu.header)
        x, y = wcs.all_world2pix([[ra_deg, dec_deg]], 0)[0]
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(
            "candidate_catalog sky coordinate did not project to finite "
            f"pixels for {path}"
        )
    return int(np.rint(float(x))), int(np.rint(float(y)))


def _validate_stamp_size_against_samples(
    samples: list[LsstComCamStampSample],
    *,
    image_hdu: str | int,
    difference_hdu: str | int,
    stamp_size: int,
) -> None:
    for sample in samples:
        pair = sample.pair
        visit_path = Path(str(pair["visit_image"]["path"]))
        visit_shape = _fits_image_shape(visit_path, hdu=image_hdu)
        visit_center_x, visit_center_y = _validate_stamp_bounds_for_shape(
            visit_path,
            product="visit_image",
            shape=visit_shape,
            stamp_size=stamp_size,
            center_x=sample.center_x,
            center_y=sample.center_y,
        )
        difference_path = Path(str(pair["difference_image"]["path"]))
        difference_shape = _fits_image_shape(
            difference_path,
            hdu=difference_hdu,
        )
        if difference_shape != visit_shape:
            raise ValueError(
                "visit_image and difference_image FITS HDUs must have "
                "matching image shapes before center reuse: "
                f"visit={visit_shape} difference={difference_shape}"
            )
        _validate_stamp_bounds_for_shape(
            difference_path,
            product="difference_image",
            shape=difference_shape,
            stamp_size=stamp_size,
            center_x=visit_center_x,
            center_y=visit_center_y,
        )


def _validate_stamp_size_against_pairs(
    pairs: list[dict[str, dict[str, Any]]],
    *,
    image_hdu: str | int,
    difference_hdu: str | int,
    stamp_size: int,
    center_x: int | None,
    center_y: int | None,
) -> None:
    for pair in pairs:
        visit_path = Path(str(pair["visit_image"]["path"]))
        visit_shape = _fits_image_shape(visit_path, hdu=image_hdu)
        visit_center_x, visit_center_y = _validate_stamp_bounds_for_shape(
            visit_path,
            product="visit_image",
            shape=visit_shape,
            stamp_size=stamp_size,
            center_x=center_x,
            center_y=center_y,
        )
        difference_path = Path(str(pair["difference_image"]["path"]))
        difference_shape = _fits_image_shape(
            difference_path,
            hdu=difference_hdu,
        )
        if difference_shape != visit_shape:
            raise ValueError(
                "visit_image and difference_image FITS HDUs must have "
                "matching image shapes before center reuse: "
                f"visit={visit_shape} difference={difference_shape}"
            )
        _validate_stamp_bounds_for_shape(
            difference_path,
            product="difference_image",
            shape=difference_shape,
            stamp_size=stamp_size,
            center_x=visit_center_x,
            center_y=visit_center_y,
        )


def _validate_stamp_bounds_for_shape(
    path: Path,
    *,
    product: str,
    shape: tuple[int, int],
    stamp_size: int,
    center_x: int | None,
    center_y: int | None,
) -> tuple[int, int]:
    height, width = shape
    resolved_center_x = width // 2 if center_x is None else int(center_x)
    resolved_center_y = height // 2 if center_y is None else int(center_y)
    half = stamp_size // 2
    y0 = resolved_center_y - half
    y1 = y0 + stamp_size
    x0 = resolved_center_x - half
    x1 = x0 + stamp_size
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        raise ValueError(
            f"stamp_size={stamp_size} centered at x={resolved_center_x}, "
            f"y={resolved_center_y} does not fit inside {product} FITS "
            f"image {path}: shape={(height, width)}"
        )
    return resolved_center_x, resolved_center_y


def _finalize_stamp(
    stamp: np.ndarray,
    *,
    nan_policy: str,
    name: str,
) -> tuple[np.ndarray, int]:
    array = np.asarray(stamp, dtype=np.float32)
    invalid = int(np.size(array) - np.isfinite(array).sum())
    if invalid and nan_policy == "raise":
        raise ValueError(f"{name} contains {invalid} non-finite values")
    if invalid:
        array = np.nan_to_num(
            array,
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return ensure_finite_array(array, name=name), invalid


def _matched_coadd_metadata(
    visit_row: dict[str, Any],
    coadd_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    missing_visit_keys = _missing_coadd_visit_keys(visit_row)
    if missing_visit_keys:
        status = "visit_missing_" + "_".join(missing_visit_keys)
        for product in ("template_coadd", "deep_coadd"):
            metadata[f"{product}_lookup_status"] = status
        return metadata
    for product in ("template_coadd", "deep_coadd"):
        row = _match_coadd_row(visit_row, coadd_rows.get(product, []))
        if row is None:
            metadata[f"{product}_lookup_status"] = "no_match"
            continue
        metadata[f"{product}_lookup_status"] = "matched"
        metadata.update(_prefixed_row_metadata(product, row))
    return metadata


def _missing_coadd_visit_keys(visit_row: dict[str, Any]) -> list[str]:
    missing = []
    for key in ("band", "tract", "patch"):
        if _canonical_filter_value(visit_row.get(key)) is None:
            missing.append(key)
    return missing


def _match_coadd_row(
    visit_row: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    visit_band = _canonical_filter_value(visit_row.get("band"))
    tract = _canonical_filter_value(visit_row.get("tract"))
    patch = _canonical_filter_value(visit_row.get("patch"))
    if visit_band is None or tract is None or patch is None:
        return None
    matches = []
    for row in candidates:
        if _canonical_filter_value(row.get("band")) != visit_band:
            continue
        if _canonical_filter_value(row.get("tract")) != tract:
            continue
        if _canonical_filter_value(row.get("patch")) != patch:
            continue
        matches.append(row)
    if not matches:
        return None
    return sorted(matches, key=_sort_key)[0]


def _coadd_rows_by_product(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_product: dict[str, list[dict[str, Any]]] = {
        "template_coadd": [],
        "deep_coadd": [],
    }
    for row in rows:
        product = _row_product(row)
        if product in by_product:
            by_product[product].append(row)
    return by_product


def _prefixed_row_metadata(
    prefix: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_{field}": _metadata_value(row.get(field))
        for field in PROVENANCE_FIELDS
        if field in row
    }


def _prefixed_candidate_metadata(
    row: dict[str, Any],
    *,
    candidate_catalog_path: Path | None,
) -> dict[str, Any]:
    metadata = {
        f"candidate_{key}": _metadata_value(value)
        for key, value in row.items()
        if not str(key).startswith("_")
        and key != "candidate_catalog_row_index"
    }
    if "candidate_catalog_row_index" in row:
        metadata["candidate_catalog_row_index"] = _metadata_value(
            row["candidate_catalog_row_index"]
        )
    if candidate_catalog_path is not None:
        metadata["candidate_catalog_path"] = str(candidate_catalog_path)
    return metadata


def _candidate_id(
    index: int,
    visit_row: dict[str, Any],
    *,
    candidate_row: dict[str, Any] | None = None,
) -> str:
    visit = _canonical_filter_value(visit_row.get("visit"))
    detector = _canonical_filter_value(visit_row.get("detector"))
    band = _canonical_filter_value(visit_row.get("band"))
    if candidate_row is not None:
        for key in _candidate_id_column_names():
            value = _canonical_filter_value(candidate_row.get(key))
            if value is not None:
                return f"lsstcomcam-{visit}-{detector}-{band}-{value}"
    return f"lsstcomcam-{visit}-{detector}-{band}-{index:06d}"


def _split_group_for_pair(row: dict[str, Any]) -> str:
    return "|".join(
        str(_canonical_filter_value(row.get(column)))
        for column in DEFAULT_PAIR_KEY_COLUMNS
    )


def _summarize_lsst_splits(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        result[split_name] = {
            "count": len(split_rows),
            "positives": sum(int(row["label"]) for row in split_rows),
            "negatives": sum(1 - int(row["label"]) for row in split_rows),
        }
    return result


def _row_product(row: dict[str, Any]) -> str:
    return str(row.get("product") or row.get("butler_datasettype") or "")


def _pair_key(
    row: dict[str, Any],
    *,
    key_columns: tuple[str, ...],
) -> tuple[Any, ...] | None:
    values: list[Any] = []
    for column in key_columns:
        value = _canonical_filter_value(row.get(column))
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _sort_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(_canonical_filter_value(row.get(column)) or "")
        for column in (
            "date",
            "visit",
            "detector",
            "band",
            "tract",
            "patch",
            "path",
        )
    )


def _registry_row_preference_key(row: dict[str, Any]) -> tuple[int, int, str]:
    value = row.get("mtime_ns")
    if _is_missing(value):
        mtime_ns = -1
        has_mtime = 0
    else:
        try:
            mtime_ns = int(value)
        except (TypeError, ValueError):
            mtime_ns = -1
            has_mtime = 0
        else:
            has_mtime = 1
    return (
        has_mtime,
        mtime_ns,
        str(_metadata_value(row.get("path")) or ""),
    )


def _first_filter_values(
    filters: dict[str, Any],
    keys: tuple[str, ...],
) -> list[Any] | None:
    for key in keys:
        if key not in filters:
            continue
        return _any_list(filters[key])
    return None


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _any_list(value)]


def _any_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _canonical_filter_value(value: Any) -> str | None:
    if _is_missing(value):
        return None
    timestamp = _timestamp_yyyymmdd(value)
    if timestamp is not None:
        return timestamp
    value = _jsonable(value)
    if (
        isinstance(value, float)
        and value.is_integer()
        and abs(value) <= 2**53
    ):
        return str(int(value))
    return str(value)


def _numeric_range_bound(value: Any, *, column: str) -> float:
    canonical = _canonical_filter_value(value)
    try:
        return float(canonical)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "registry range requested non-numeric bound for column "
            f"{column!r}: {value!r}"
        ) from exc


def _manifest_filter_summary_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return [_manifest_filter_summary_value(item) for item in value]
    return _metadata_value(value)


def _metadata_value(value: Any) -> Any:
    timestamp = _timestamp_yyyymmdd(value)
    if timestamp is not None:
        return timestamp
    value = _jsonable(value)
    if isinstance(value, float) and value.is_integer():
        if abs(value) <= 2**53:
            return int(value)
        return value
    return value


def _timestamp_yyyymmdd(value: Any) -> str | None:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None
    if pd is not None and isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "T" in text or " " in text:
                return datetime.fromisoformat(
                    text.replace("Z", "+00:00")
                ).strftime("%Y%m%d")
            return date.fromisoformat(text).strftime("%Y%m%d")
        except ValueError:
            return None
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"expected integer value, got boolean: {value!r}")
    return int(value)


def _optional_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"expected numeric value, got boolean: {value!r}")
    result = float(value)
    if not np.isfinite(result):
        return None
    return result
