# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Raw DES-oriented dataset builders for XScan."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cuphoton._photometry import detect_sources, estimate_background

from .dataset import (
    INDEX_TO_SPLIT,
    SPLIT_TO_INDEX,
    DatasetBuildResult,
    maybe_write_metadata_parquet,
    validate_dataset_dir,
    write_metadata_jsonl,
)
from .hsc import assign_split, extract_stamp, simple_difference

AUTOSCAN_FIELD_ALIASES = {
    "candidate_id": ("candidate_id", "cand_id", "id"),
    "search_path": ("search_path", "search_stamp_path", "search_image_path"),
    "template_path": (
        "template_path",
        "template_stamp_path",
        "template_image_path",
    ),
    "difference_path": (
        "difference_path",
        "diff_path",
        "difference_image_path",
    ),
    "label": ("label", "target", "class"),
    "exposure_id": ("exposure_id", "expnum", "exposure", "exposure_num"),
    "ccd_id": ("ccd_id", "ccdnum", "ccd", "ccd_num"),
    "band": ("band", "filter"),
    "x": ("x", "xpos", "x_pos", "col"),
    "y": ("y", "ypos", "y_pos", "row"),
    "split_group": ("split_group", "group", "group_id"),
    "split": ("split",),
    "label_source": ("label_source",),
}

NODIFF_EXPOSURE_ALIASES = {
    "search_path": ("search_path", "search_image_path"),
    "template_path": ("template_path", "template_image_path"),
    "difference_path": (
        "difference_path",
        "diff_path",
        "difference_image_path",
    ),
    "fake_catalog_path": ("fake_catalog_path", "fakes_path", "fake_path"),
    "exposure_id": ("exposure_id", "expnum", "exposure", "exposure_num"),
    "ccd_id": ("ccd_id", "ccdnum", "ccd", "ccd_num"),
    "band": ("band", "filter"),
    "split_group": ("split_group", "group", "group_id"),
    "split": ("split",),
}

NODIFF_FAKE_ALIASES = {
    "candidate_id": ("candidate_id", "cand_id", "id"),
    "fake_id": ("fake_id", "fakeid"),
    "label": ("label", "class", "target"),
    "x": ("x", "xpos", "x_pos", "col"),
    "y": ("y", "ypos", "y_pos", "row"),
    "autoscan_score": ("autoscan_score", "autoscan", "autoscanScore"),
    "diff_snr": ("diff_snr", "diffsnr", "snr_diff"),
    "snr": ("snr",),
    "injected_flux": ("injected_flux", "fake_flux", "flux"),
}


def build_autoscan_dataset_from_raw(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> DatasetBuildResult:
    manifest = load_manifest(manifest_path)
    rows = load_table_rows(resolve_required_path(manifest, "records_path"))
    if not rows:
        raise ValueError("records_path did not contain any rows")

    split_fractions = tuple(manifest.get("split_fractions", [0.9, 0.0, 0.1]))
    split_seed = int(manifest.get("split_seed", 0))
    balance_classes = bool(manifest.get("balance_classes", True))
    subset_fraction = float(manifest.get("subset_fraction", 1.0))
    subset_seed = int(manifest.get("subset_seed", split_seed))
    limit_records = manifest.get("limit_records")
    stamp_size = int(manifest.get("stamp_size", 51))
    search_hdu = manifest.get("search_hdu")
    template_hdu = manifest.get("template_hdu")
    difference_hdu = manifest.get("difference_hdu")
    positive_labels, negative_labels = resolve_label_sets(manifest)

    if limit_records is not None:
        rows = rows[: int(limit_records)]

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        search_path = get_required_alias(
            row, AUTOSCAN_FIELD_ALIASES, "search_path"
        )
        template_path = get_required_alias(
            row, AUTOSCAN_FIELD_ALIASES, "template_path"
        )
        difference_path = get_required_alias(
            row, AUTOSCAN_FIELD_ALIASES, "difference_path"
        )
        center_x = get_optional_alias(row, AUTOSCAN_FIELD_ALIASES, "x")
        center_y = get_optional_alias(row, AUTOSCAN_FIELD_ALIASES, "y")
        search = load_stamp_or_image_cutout(
            search_path,
            stamp_size=stamp_size,
            center_x=center_x,
            center_y=center_y,
            hdu=search_hdu,
        )
        template = load_stamp_or_image_cutout(
            template_path,
            stamp_size=stamp_size,
            center_x=center_x,
            center_y=center_y,
            hdu=template_hdu,
        )
        if not difference_path:
            raise ValueError("autoScan raw rows must include difference_path")
        difference = load_stamp_or_image_cutout(
            difference_path,
            stamp_size=stamp_size,
            center_x=center_x,
            center_y=center_y,
            hdu=difference_hdu,
        )
        if search.shape != template.shape or search.shape != difference.shape:
            raise ValueError(
                "search/template/difference stamp shapes must match"
            )
        label = normalize_label(
            get_required_alias(row, AUTOSCAN_FIELD_ALIASES, "label"),
            positive_labels=positive_labels,
            negative_labels=negative_labels,
        )
        exposure_id = int(
            get_required_alias(row, AUTOSCAN_FIELD_ALIASES, "exposure_id")
        )
        ccd_id = int(
            get_required_alias(row, AUTOSCAN_FIELD_ALIASES, "ccd_id")
        )
        sample = {
            "search": search.astype(np.float32),
            "template": template.astype(np.float32),
            "difference": difference.astype(np.float32),
            "label": label,
            "metadata": {
                "candidate_id": str(
                    get_optional_alias(
                        row, AUTOSCAN_FIELD_ALIASES, "candidate_id"
                    )
                    or f"cand-{index:06d}"
                ),
                "exposure_id": exposure_id,
                "ccd_id": ccd_id,
                "band": str(
                    get_optional_alias(row, AUTOSCAN_FIELD_ALIASES, "band")
                    or "i"
                ),
                "x": float(center_x) if center_x is not None else None,
                "y": float(center_y) if center_y is not None else None,
                "split_group": str(
                    get_optional_alias(
                        row, AUTOSCAN_FIELD_ALIASES, "split_group"
                    )
                    or f"{exposure_id}:{ccd_id}"
                ),
                "split": get_optional_alias(
                    row, AUTOSCAN_FIELD_ALIASES, "split"
                ),
                "label": label,
                "label_source": str(
                    get_optional_alias(
                        row, AUTOSCAN_FIELD_ALIASES, "label_source"
                    )
                    or "raw_autoscan_detection"
                ),
            },
        }
        samples.append(sample)

    samples_before_balance = len(samples)
    samples = maybe_balance_samples(
        samples,
        balance_classes=balance_classes,
        seed=split_seed,
    )
    samples_after_balance = len(samples)
    samples = maybe_subsample_samples(
        samples,
        subset_fraction=subset_fraction,
        seed=subset_seed,
    )
    assign_missing_splits(
        samples,
        split_fractions=split_fractions,
        seed=split_seed,
    )
    return write_canonical_dataset(
        output_dir=output_dir,
        dataset_kind="autoscan",
        manifest_path=manifest_path,
        samples=samples,
        builder_summary={
            "input_record_count": len(rows),
            "samples_before_balance": samples_before_balance,
            "samples_after_balance": samples_after_balance,
            "subset_fraction": subset_fraction,
        },
    )


def build_nodiff_dataset_from_raw(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> DatasetBuildResult:
    manifest = load_manifest(manifest_path)
    exposures = load_table_rows(
        resolve_required_path(manifest, "exposures_path")
    )
    if not exposures:
        raise ValueError("exposures_path did not contain any rows")

    stamp_size = int(manifest.get("stamp_size", 51))
    threshold_sigma = float(manifest.get("detection_threshold_sigma", 5.0))
    minarea = int(manifest.get("detection_minarea", 5))
    max_per_class_per_image = int(manifest.get("max_per_class_per_image", 15))
    diff_snr_min = float(manifest.get("diff_snr_min", 3.5))
    split_fractions = tuple(manifest.get("split_fractions", [0.9, 0.0, 0.1]))
    split_seed = int(manifest.get("split_seed", 0))
    balance_classes = bool(manifest.get("balance_classes", True))
    subset_fraction = float(manifest.get("subset_fraction", 1.0))
    subset_seed = int(manifest.get("subset_seed", split_seed))
    limit_exposures = manifest.get("limit_exposures")
    search_hdu = manifest.get("search_hdu")
    template_hdu = manifest.get("template_hdu")
    difference_hdu = manifest.get("difference_hdu")
    positive_labels, negative_labels = resolve_label_sets(
        manifest,
        default_positive_labels=("real", "positive", "fake", "1"),
        default_negative_labels=("bogus", "artifact", "negative", "0"),
    )

    if limit_exposures is not None:
        exposures = exposures[: int(limit_exposures)]

    all_samples: list[dict[str, Any]] = []
    detections_total = 0
    positives_pre_cap = 0
    negatives_pre_cap = 0
    positives_post_cap = 0
    negatives_post_cap = 0
    for exposure_index, exposure_row in enumerate(exposures):
        search = load_image_array(
            get_required_alias(
                exposure_row,
                NODIFF_EXPOSURE_ALIASES,
                "search_path",
            ),
            hdu=search_hdu,
        )
        template = load_image_array(
            get_required_alias(
                exposure_row,
                NODIFF_EXPOSURE_ALIASES,
                "template_path",
            ),
            hdu=template_hdu,
        )
        if search.shape != template.shape:
            raise ValueError("search and template images must share a shape")
        difference = None
        difference_path = get_optional_alias(
            exposure_row,
            NODIFF_EXPOSURE_ALIASES,
            "difference_path",
        )
        if difference_path:
            difference = load_image_array(difference_path, hdu=difference_hdu)
            if difference.shape != search.shape:
                raise ValueError(
                    "difference image must match search/template shape"
                )

        fake_rows = load_table_rows(
            Path(
                get_required_alias(
                    exposure_row,
                    NODIFF_EXPOSURE_ALIASES,
                    "fake_catalog_path",
                )
            )
            .expanduser()
            .resolve()
        )
        detections = detect_search_sources(
            search,
            threshold_sigma=threshold_sigma,
            minarea=minarea,
        )
        detections_total += len(detections)

        positives: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(detections):
            center_y = int(round(float(detection["y"])))
            center_x = int(round(float(detection["x"])))
            try:
                search_stamp = extract_stamp(
                    search, center_y, center_x, stamp_size
                )
                template_stamp = extract_stamp(
                    template, center_y, center_x, stamp_size
                )
            except ValueError:
                continue
            if difference is not None:
                difference_stamp = extract_stamp(
                    difference,
                    center_y,
                    center_x,
                    stamp_size,
                )
                flux_ratio_source = "difference_image"
            else:
                difference_stamp = simple_difference(
                    search_stamp, template_stamp
                )
                flux_ratio_source = "search_minus_template"

            fake = match_fake_to_cutout(
                fake_rows,
                center_y=center_y,
                center_x=center_x,
                stamp_size=stamp_size,
                diff_snr_min=diff_snr_min,
            )
            label = (
                normalize_label(
                    (
                        get_optional_alias(fake, NODIFF_FAKE_ALIASES, "label")
                        if fake is not None
                        else 0
                    ),
                    positive_labels=positive_labels,
                    negative_labels=negative_labels,
                    missing_defaults_to_positive=fake is not None,
                )
                if fake is not None
                else 0
            )
            flux_ratio = compute_flux_ratio(
                difference_stamp,
                template_stamp,
                injected_flux=(
                    float(
                        get_required_alias(
                            fake, NODIFF_FAKE_ALIASES, "injected_flux"
                        )
                    )
                    if fake
                    else 0.0
                ),
            )
            default_exposure_id = get_required_alias(
                exposure_row,
                NODIFF_EXPOSURE_ALIASES,
                "exposure_id",
            )
            default_ccd_id = get_required_alias(
                exposure_row,
                NODIFF_EXPOSURE_ALIASES,
                "ccd_id",
            )
            split_group_default = f"{default_exposure_id}:{default_ccd_id}"

            sample = {
                "search": search_stamp.astype(np.float32),
                "template": template_stamp.astype(np.float32),
                "label": label,
                "metadata": {
                    "candidate_id": (
                        str(
                            get_optional_alias(
                                fake, NODIFF_FAKE_ALIASES, "candidate_id"
                            )
                        )
                        if fake
                        and get_optional_alias(
                            fake, NODIFF_FAKE_ALIASES, "candidate_id"
                        )
                        is not None
                        else (
                            f"cand-{exposure_index:04d}-{detection_index:04d}"
                        )
                    ),
                    "exposure_id": int(
                        get_required_alias(
                            exposure_row,
                            NODIFF_EXPOSURE_ALIASES,
                            "exposure_id",
                        )
                    ),
                    "ccd_id": int(
                        get_required_alias(
                            exposure_row,
                            NODIFF_EXPOSURE_ALIASES,
                            "ccd_id",
                        )
                    ),
                    "band": str(
                        get_optional_alias(
                            exposure_row,
                            NODIFF_EXPOSURE_ALIASES,
                            "band",
                        )
                        or "i"
                    ),
                    "x": center_x,
                    "y": center_y,
                    "split_group": str(
                        get_optional_alias(
                            exposure_row,
                            NODIFF_EXPOSURE_ALIASES,
                            "split_group",
                        )
                        or split_group_default
                    ),
                    "split": get_optional_alias(
                        exposure_row,
                        NODIFF_EXPOSURE_ALIASES,
                        "split",
                    ),
                    "label": label,
                    "label_source": "search_source_extractor",
                    "fake_id": (
                        get_optional_alias(
                            fake, NODIFF_FAKE_ALIASES, "fake_id"
                        )
                        if fake
                        else None
                    ),
                    "autoscan_score": (
                        get_optional_alias(
                            fake,
                            NODIFF_FAKE_ALIASES,
                            "autoscan_score",
                        )
                        if fake
                        else None
                    ),
                    "diff_snr": (
                        float(
                            get_required_alias(
                                fake, NODIFF_FAKE_ALIASES, "diff_snr"
                            )
                        )
                        if fake
                        else 0.0
                    ),
                    "snr": (
                        float(
                            get_optional_alias(
                                fake, NODIFF_FAKE_ALIASES, "snr"
                            )
                            or get_required_alias(
                                fake, NODIFF_FAKE_ALIASES, "diff_snr"
                            )
                        )
                        if fake
                        else 0.0
                    ),
                    "flux_ratio": flux_ratio,
                    "flux_ratio_source": flux_ratio_source,
                    "injected_flux": (
                        float(
                            get_required_alias(
                                fake, NODIFF_FAKE_ALIASES, "injected_flux"
                            )
                        )
                        if fake
                        else 0.0
                    ),
                    "detection_flux": float(detection.get("flux", 0.0)),
                },
            }
            if label == 1:
                positives.append(sample)
            else:
                negatives.append(sample)

        positives_pre_cap += len(positives)
        negatives_pre_cap += len(negatives)
        positives = sort_and_limit_by_flux(
            positives,
            limit=max_per_class_per_image,
        )
        negatives = sort_and_limit_by_flux(
            negatives,
            limit=max_per_class_per_image,
        )
        positives_post_cap += len(positives)
        negatives_post_cap += len(negatives)
        all_samples.extend(positives)
        all_samples.extend(negatives)

    samples_before_balance = len(all_samples)
    all_samples = maybe_balance_samples(
        all_samples,
        balance_classes=balance_classes,
        seed=split_seed,
    )
    samples_after_balance = len(all_samples)
    all_samples = maybe_subsample_samples(
        all_samples,
        subset_fraction=subset_fraction,
        seed=subset_seed,
    )
    assign_missing_splits(
        all_samples,
        split_fractions=split_fractions,
        seed=split_seed,
    )
    return write_canonical_dataset(
        output_dir=output_dir,
        dataset_kind="nodiff",
        manifest_path=manifest_path,
        samples=all_samples,
        builder_summary={
            "exposure_count": len(exposures),
            "detections_total": detections_total,
            "positives_pre_cap": positives_pre_cap,
            "negatives_pre_cap": negatives_pre_cap,
            "positives_post_cap": positives_post_cap,
            "negatives_post_cap": negatives_post_cap,
            "samples_before_balance": samples_before_balance,
            "samples_after_balance": samples_after_balance,
            "subset_fraction": subset_fraction,
        },
    )


def build_nodiff_release_dataset_from_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> DatasetBuildResult:
    manifest = load_manifest(manifest_path)
    source_dir = resolve_required_path(manifest, "source_dir")
    stamp_size = int(manifest.get("stamp_size", 51))
    split_fractions = tuple(
        float(value)
        for value in manifest.get("split_fractions", [0.8, 0.1, 0.1])
    )
    if len(split_fractions) != 3:
        raise ValueError("split_fractions must contain exactly three values")
    if not np.isclose(sum(split_fractions), 1.0):
        raise ValueError("split_fractions must sum to 1.0")
    split_seed = int(manifest.get("split_seed", 0))
    max_per_class_raw = manifest.get("max_per_class")
    max_per_class = (
        int(max_per_class_raw) if max_per_class_raw is not None else None
    )
    if max_per_class is not None and max_per_class <= 0:
        raise ValueError("max_per_class must be positive")

    release_inventory = collect_nodiff_release_group_inventory(source_dir)
    release_groups = release_inventory["complete_groups"]
    if not release_groups:
        raise ValueError("source_dir did not contain complete release stamps")

    positives = [
        group for group in release_groups if int(group["label"]) == 1
    ]
    negatives = [
        group for group in release_groups if int(group["label"]) == 0
    ]
    if not positives or not negatives:
        raise ValueError("release slice must contain both labels")

    rng = np.random.default_rng(split_seed)
    keep = min(len(positives), len(negatives))
    if max_per_class is not None:
        keep = min(keep, max_per_class)

    def select_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        indices = rng.permutation(len(group))[:keep]
        return [group[index] for index in sorted(indices)]

    selected_groups = select_group(positives) + select_group(negatives)

    samples: list[dict[str, Any]] = []
    for group in selected_groups:
        search = load_release_stamp(group["search_path"], stamp_size)
        template = load_release_stamp(group["template_path"], stamp_size)
        if search.shape != template.shape:
            raise ValueError(
                "release search/template stamp shapes must match for "
                f"global_id={group['global_id']}"
            )
        label = int(group["label"])
        global_id = int(group["global_id"])
        samples.append(
            {
                "search": search.astype(np.float32),
                "template": template.astype(np.float32),
                "label": label,
                "metadata": {
                    "candidate_id": (
                        f"nodiff-release-{global_id:08d}-"
                        f"{'pos' if label else 'neg'}"
                    ),
                    "exposure_id": global_id,
                    "ccd_id": 0,
                    "band": str(manifest.get("band", "unknown")),
                    "x": stamp_size // 2,
                    "y": stamp_size // 2,
                    "split_group": str(global_id),
                    "split": None,
                    "label": label,
                    "label_source": "nodiff_release_filename",
                    "fake_id": str(global_id) if label == 1 else None,
                    "autoscan_score": None,
                    "diff_snr": None,
                    "snr": None,
                    "flux_ratio": None,
                    "nodiff_release_cut_provenance": (
                        "published_release_inclusion"
                    ),
                    "nodiff_release_global_id": global_id,
                    "nodiff_release_label": "pos" if label else "neg",
                    "nodiff_release_source_dir": str(source_dir),
                    "nodiff_release_search_path": str(group["search_path"]),
                    "nodiff_release_template_path": str(
                        group["template_path"]
                    ),
                    "nodiff_release_difference_path": str(
                        group["difference_path"]
                    ),
                    "nodiff_release_shard_url": manifest.get(
                        "release_shard_url"
                    ),
                    "nodiff_release_byte_range": manifest.get(
                        "release_byte_range"
                    ),
                },
            }
        )

    if not samples:
        raise ValueError("release builder did not find usable stamp pairs")
    assign_missing_splits(
        samples,
        split_fractions=split_fractions,
        seed=split_seed,
    )
    return write_canonical_dataset(
        output_dir=output_dir,
        dataset_kind="nodiff",
        manifest_path=manifest_path,
        samples=samples,
        builder_summary={
            "release_source_dir": str(source_dir),
            "release_shard_url": manifest.get("release_shard_url"),
            "release_byte_range": manifest.get("release_byte_range"),
            "release_group_count": len(release_groups),
            "release_incomplete_group_count": release_inventory[
                "incomplete_group_count"
            ],
            "release_incomplete_group_examples": release_inventory[
                "incomplete_group_examples"
            ],
            "usable_pair_count": len(positives) + len(negatives),
            "positive_count_available": len(positives),
            "negative_count_available": len(negatives),
            "positive_count_selected": keep,
            "negative_count_selected": keep,
            "max_per_class": max_per_class,
            "split_fractions": list(split_fractions),
            "split_seed": split_seed,
            "autoscan_scores": "release_provenance_only",
            "diff_snr": "release_provenance_only",
        },
    )


def write_canonical_dataset(
    *,
    output_dir: Path,
    dataset_kind: str,
    manifest_path: Path,
    samples: list[dict[str, Any]],
    builder_summary: dict[str, Any] | None = None,
) -> DatasetBuildResult:
    if not samples:
        raise ValueError("builder did not produce any samples")

    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    search = np.stack([sample["search"] for sample in samples], axis=0)
    template = np.stack([sample["template"] for sample in samples], axis=0)
    labels = np.asarray(
        [sample["label"] for sample in samples], dtype=np.int64
    )
    metadata_rows = [sample["metadata"] for sample in samples]
    split = np.asarray(
        [SPLIT_TO_INDEX[row["split"]] for row in metadata_rows],
        dtype=np.int64,
    )

    np.save(root / "search.npy", search, allow_pickle=False)
    np.save(root / "template.npy", template, allow_pickle=False)
    np.save(root / "labels.npy", labels, allow_pickle=False)
    np.save(root / "split.npy", split, allow_pickle=False)

    has_difference = all("difference" in sample for sample in samples)
    if has_difference:
        difference = np.stack(
            [sample["difference"] for sample in samples],
            axis=0,
        )
        np.save(root / "difference.npy", difference, allow_pickle=False)

    write_metadata_jsonl(root / "metadata.jsonl", metadata_rows)
    maybe_write_metadata_parquet(root / "metadata.parquet", metadata_rows)

    summary = {
        "dataset_dir": str(root),
        "dataset_kind": dataset_kind,
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "sample_count": int(search.shape[0]),
        "input_mode": "triplet" if has_difference else "pair",
        "builder_summary": builder_summary or {},
        "saved": {
            "search": "search.npy",
            "template": "template.npy",
            "labels": "labels.npy",
            "split": "split.npy",
            "metadata_jsonl": "metadata.jsonl",
        },
    }
    if has_difference:
        summary["saved"]["difference"] = "difference.npy"
    if (root / "metadata.parquet").exists():
        summary["saved"]["metadata_parquet"] = "metadata.parquet"

    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    validation = validate_dataset_dir(root, dataset_kind=dataset_kind)
    summary["semantic_checks"] = validation.get("semantic_checks")
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return DatasetBuildResult(output_dir=root, summary=summary)


def collect_nodiff_release_groups(source_dir: Path) -> list[dict[str, Any]]:
    inventory = collect_nodiff_release_group_inventory(source_dir)
    return inventory["complete_groups"]


def collect_nodiff_release_group_inventory(
    source_dir: Path,
) -> dict[str, Any]:
    groups: dict[int, dict[str, Any]] = {}
    for path in sorted(source_dir.expanduser().resolve().glob("gid_*/*.npy")):
        parsed = parse_nodiff_release_stamp_path(path)
        if parsed is None:
            continue
        group = groups.setdefault(
            parsed["global_id"],
            {"global_id": parsed["global_id"], "labels": set()},
        )
        path_key = parsed["kind"] + "_path"
        if path_key in group:
            raise ValueError(
                "release shard contains duplicate "
                f"{parsed['kind']} stamp for global_id={parsed['global_id']}"
            )
        group[path_key] = path
        group["labels"].add(parsed["label"])

    complete: list[dict[str, Any]] = []
    incomplete_examples: list[dict[str, Any]] = []
    for group in groups.values():
        required_paths = {
            "search_path",
            "template_path",
            "difference_path",
        }
        missing = sorted(required_paths - set(group))
        if missing:
            if len(incomplete_examples) < 5:
                incomplete_examples.append(
                    {
                        "global_id": group["global_id"],
                        "missing": missing,
                    }
                )
            continue
        labels = group["labels"]
        if len(labels) != 1:
            raise ValueError(
                "release stamp labels disagree for "
                f"global_id={group['global_id']}"
            )
        group["label"] = 1 if next(iter(labels)) == "pos" else 0
        group.pop("labels", None)
        complete.append(group)
    return {
        "complete_groups": sorted(
            complete,
            key=lambda group: int(group["global_id"]),
        ),
        "incomplete_group_count": len(groups) - len(complete),
        "incomplete_group_examples": incomplete_examples,
    }


def parse_nodiff_release_stamp_path(path: Path) -> dict[str, Any] | None:
    parent_name = path.parent.name
    if not parent_name.startswith("gid_"):
        return None
    try:
        global_id = int(parent_name.removeprefix("gid_"))
    except ValueError:
        return None
    name = path.name
    if name.endswith("_pos.npy"):
        label = "pos"
    elif name.endswith("_neg.npy"):
        label = "neg"
    else:
        return None
    if "_srch_" in name:
        kind = "search"
    elif "_tmpl_" in name:
        kind = "template"
    elif "_diff_" in name:
        kind = "difference"
    else:
        return None
    return {"global_id": global_id, "kind": kind, "label": label}


def load_release_stamp(path: Path, stamp_size: int) -> np.ndarray:
    stamp = np.load(path, allow_pickle=False)
    stamp = np.asarray(stamp, dtype=np.float32)
    if stamp.ndim == 3 and 1 in stamp.shape:
        stamp = np.squeeze(stamp)
    if stamp.ndim != 2:
        raise ValueError(f"release stamp must be rank-2: {path}")
    if stamp.shape != (stamp_size, stamp_size):
        raise ValueError(
            f"release stamp shape {stamp.shape} does not match "
            f"stamp_size={stamp_size}: {path}"
        )
    if not np.isfinite(stamp).all():
        raise ValueError(f"release stamp contains non-finite values: {path}")
    return stamp


def maybe_balance_samples(
    samples: list[dict[str, Any]],
    *,
    balance_classes: bool,
    seed: int,
) -> list[dict[str, Any]]:
    if not balance_classes:
        return samples
    positives = [sample for sample in samples if int(sample["label"]) == 1]
    negatives = [sample for sample in samples if int(sample["label"]) == 0]
    if not positives or not negatives:
        raise ValueError("cannot balance classes without both labels present")
    keep = min(len(positives), len(negatives))
    rng = np.random.default_rng(seed)
    positives = [
        positives[index] for index in rng.permutation(len(positives))[:keep]
    ]
    negatives = [
        negatives[index] for index in rng.permutation(len(negatives))[:keep]
    ]
    return positives + negatives


def maybe_subsample_samples(
    samples: list[dict[str, Any]],
    *,
    subset_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not 0.0 < subset_fraction <= 1.0:
        raise ValueError("subset_fraction must be in the interval (0, 1]")
    if subset_fraction >= 1.0:
        return samples

    positives = [sample for sample in samples if int(sample["label"]) == 1]
    negatives = [sample for sample in samples if int(sample["label"]) == 0]
    rng = np.random.default_rng(seed)

    def sample_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keep = max(1, int(np.floor(len(group) * subset_fraction)))
        indices = rng.permutation(len(group))[:keep]
        return [group[index] for index in indices]

    if positives and negatives:
        return sample_group(positives) + sample_group(negatives)
    return sample_group(samples)


def assign_missing_splits(
    samples: list[dict[str, Any]],
    *,
    split_fractions: tuple[float, float, float],
    seed: int,
) -> None:
    if all(sample["metadata"].get("split") for sample in samples):
        return
    for sample in samples:
        group = sample["metadata"]["split_group"]
        split_index = assign_split(group, seed, split_fractions)
        sample["metadata"]["split"] = INDEX_TO_SPLIT[split_index]


def sort_and_limit_by_flux(
    samples: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        samples,
        key=lambda sample: float(
            sample["metadata"].get("detection_flux", 0.0)
        ),
        reverse=True,
    )
    return ordered[:limit]


def match_fake_to_cutout(
    fake_rows: list[dict[str, Any]],
    *,
    center_y: int,
    center_x: int,
    stamp_size: int,
    diff_snr_min: float,
) -> dict[str, Any] | None:
    half = stamp_size // 2
    best: dict[str, Any] | None = None
    best_distance = None
    for row in fake_rows:
        fake_x = float(get_required_alias(row, NODIFF_FAKE_ALIASES, "x"))
        fake_y = float(get_required_alias(row, NODIFF_FAKE_ALIASES, "y"))
        if abs(fake_x - center_x) > half or abs(fake_y - center_y) > half:
            continue
        diff_snr = float(
            get_required_alias(row, NODIFF_FAKE_ALIASES, "diff_snr")
        )
        if diff_snr <= diff_snr_min:
            continue
        if (
            get_optional_alias(row, NODIFF_FAKE_ALIASES, "autoscan_score")
            is None
        ):
            continue
        distance = (fake_x - center_x) ** 2 + (fake_y - center_y) ** 2
        if best is None or distance < best_distance:
            best = row
            best_distance = distance
    return best


def compute_flux_ratio(
    difference_stamp: np.ndarray,
    template_stamp: np.ndarray,
    *,
    injected_flux: float,
) -> float:
    half = difference_stamp.shape[0] // 2
    fd = float(
        difference_stamp[half - 2 : half + 3, half - 2 : half + 3].sum()
    )
    fi = float(template_stamp[half - 2 : half + 3, half - 2 : half + 3].sum())
    denom = fi + float(injected_flux)
    if denom <= 0:
        return 0.0
    return float(np.clip(fd / denom, 0.0, 1.0))


def detect_search_sources(
    image: np.ndarray,
    *,
    threshold_sigma: float,
    minarea: int,
) -> list[dict[str, Any]]:
    array = np.asarray(image, dtype=np.float64)
    background = estimate_background(array)
    finite_rms = background.background_rms[
        np.isfinite(background.background_rms)
    ]
    if finite_rms.size == 0:
        raise ValueError("background RMS estimate contains no finite values")
    background_subtracted = np.where(
        np.isfinite(array),
        array - background.background,
        0.0,
    )
    amplitude = float(np.max(np.abs(background_subtracted)))
    numeric_rms_floor = np.finfo(np.float32).eps * amplitude
    effective_rms = max(float(np.median(finite_rms)), numeric_rms_floor)
    threshold = float(threshold_sigma) * effective_rms
    objects = detect_sources(
        background_subtracted,
        threshold=threshold,
        minarea=minarea,
    )
    return [
        {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "flux": float(row["flux"]),
        }
        for row in objects
    ]


def load_stamp_array(path: str | Path) -> np.ndarray:
    array = load_image_array(path)
    if array.ndim != 2:
        raise ValueError(f"stamp array at {path} must be 2D")
    return np.asarray(array, dtype=np.float32)


def load_stamp_or_image_cutout(
    path: str | Path,
    *,
    stamp_size: int,
    center_x: Any | None,
    center_y: Any | None,
    hdu: int | None = None,
) -> np.ndarray:
    array = load_image_array(path, hdu=hdu)
    if array.ndim != 2:
        raise ValueError(f"array at {path} must be 2D")
    if (
        center_x is not None
        and center_y is not None
        and (array.shape[0] > stamp_size or array.shape[1] > stamp_size)
    ):
        return extract_stamp(
            array,
            int(round(float(center_y))),
            int(round(float(center_x))),
            stamp_size,
        ).astype(np.float32)
    return np.asarray(array, dtype=np.float32)


def load_image_array(path: str | Path, hdu: int | None = None) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(resolved, allow_pickle=False))
    if suffix in {".fits", ".fit", ".fts"}:
        try:
            from astropy.io import fits
        except ModuleNotFoundError as exc:
            raise ValueError(
                "astropy is required for FITS ingestion"
            ) from exc
        with fits.open(resolved, memmap=True) as hdul:
            if hdu is not None:
                data = hdul[hdu].data
                return np.asarray(data)
            for item in hdul:
                if item.data is not None and item.data.ndim == 2:
                    return np.asarray(item.data)
        raise ValueError(f"could not find a 2D image HDU in {resolved}")
    raise ValueError(f"unsupported image format: {resolved}")


def load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def load_table_rows(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected list payload in {resolved}")
        return payload
    if suffix == ".yaml" or suffix == ".yml":
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or []
        if not isinstance(payload, list):
            raise ValueError(f"expected list payload in {resolved}")
        return payload
    if suffix == ".csv":
        with resolved.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet" or suffix == ".parq":
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ValueError(
                "pyarrow is required for parquet ingestion"
            ) from exc
        return pq.read_table(resolved).to_pylist()
    raise ValueError(f"unsupported table format: {resolved}")


def resolve_required_path(payload: dict[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not value:
        raise ValueError(f"manifest is missing required field '{key}'")
    return Path(value).expanduser().resolve()


def resolve_label_sets(
    manifest: dict[str, Any],
    *,
    default_positive_labels: tuple[str, ...] = (
        "1",
        "true",
        "real",
        "positive",
        "signal",
        "transient",
    ),
    default_negative_labels: tuple[str, ...] = (
        "0",
        "false",
        "bogus",
        "negative",
        "artifact",
        "artifacts",
    ),
) -> tuple[set[str], set[str]]:
    positive = manifest.get("positive_labels", default_positive_labels)
    negative = manifest.get("negative_labels", default_negative_labels)
    return (
        {str(item).strip().lower() for item in positive},
        {str(item).strip().lower() for item in negative},
    )


def normalize_label(
    value: Any,
    *,
    positive_labels: set[str],
    negative_labels: set[str],
    missing_defaults_to_positive: bool = False,
) -> int:
    if value is None:
        if missing_defaults_to_positive:
            return 1
        raise ValueError("label value is missing")
    if isinstance(value, (np.integer, int)):
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0
    text = str(value).strip().lower()
    if text in positive_labels:
        return 1
    if text in negative_labels:
        return 0
    raise ValueError(f"unrecognized label value: {value!r}")


def get_optional_alias(
    row: dict[str, Any] | None,
    aliases: dict[str, tuple[str, ...]],
    canonical: str,
) -> Any | None:
    if row is None:
        return None
    for alias in aliases[canonical]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def get_required_alias(
    row: dict[str, Any],
    aliases: dict[str, tuple[str, ...]],
    canonical: str,
) -> Any:
    value = get_optional_alias(row, aliases, canonical)
    if value is None:
        raise ValueError(
            f"missing required field '{canonical}' "
            f"(aliases: {aliases[canonical]!r})"
        )
    return value
