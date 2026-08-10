# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Human review queue and annotation helpers for XScan."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import (
    load_metadata_rows,
    maybe_write_metadata_parquet,
    validate_dataset_dir,
    write_metadata_jsonl,
)
from .hsc import INDEX_TO_SPLIT, prediction_group_indices
from .metrics import (
    resolve_catalog_morphology,
    resolve_mask_pressure,
    resolve_negative_difficulty,
)

MORPHOLOGY_TAGS = (
    "point_source_round",
    "round_extended",
    "elliptical_extended",
    "spiral_structured",
    "streak_linear",
    "saturated_bleed",
    "edge_or_mask",
    "noise_artifact",
    "unclear",
)

REVIEW_LABEL_TO_VALUE = {
    "real": 1,
    "bogus": 0,
}
REVIEW_CONSENSUS_RULES = {"unanimous", "majority"}
REVIEW_AGGREGATION_STATUSES = (
    "actionable",
    "conflicted",
    "unsure_only",
    "no_actionable",
    "insufficient_review",
)
REVIEW_AGGREGATION_RULE_VERSION = "review-aggregation-v2"
REVIEW_LATEST_PER_REVIEWER_RULE = (
    "append_order_last_per_queue_id_and_reviewer"
)

ENTITY_REVIEW_LABELS = (
    "point_source_star_or_planet",
    "galaxy_elliptical_oval",
    "galaxy_spiral_structured",
    "diffuse_nebula_cloud",
    "satellite_or_linear_trail",
    "blend_or_multiple",
    "artifact_or_not_real",
    "other_or_unsure",
)
ENTITY_REVIEW_LABEL_DESCRIPTIONS = {
    "point_source_star_or_planet": "Point source, round star, or planet",
    "galaxy_elliptical_oval": "Galaxy, elliptical or oval",
    "galaxy_spiral_structured": "Galaxy with spiral or resolved structure",
    "diffuse_nebula_cloud": "Diffuse nebula, cloud, or extended emission",
    "satellite_or_linear_trail": "Satellite, streak, or linear trail",
    "blend_or_multiple": "Blend or multiple centered sources",
    "artifact_or_not_real": "Artifact or not actually real",
    "other_or_unsure": "Other, ambiguous, or unsure",
}
ENTITY_REVIEW_CONFIDENCE_LEVELS = ("low", "medium", "high")
ENTITY_REVIEW_AGGREGATION_STATUSES = (
    "actionable",
    "conflicted",
    "other_or_unsure",
    "insufficient_review",
)
ENTITY_REVIEW_AGGREGATION_RULE_VERSION = "entity-review-aggregation-v2"


@dataclass(slots=True)
class ReviewQueueResult:
    review_dir: Path
    summary: dict[str, Any]


@dataclass(slots=True)
class ReviewApplyResult:
    output_dir: Path
    summary: dict[str, Any]


@dataclass(slots=True)
class ReviewAggregationResult:
    review_dir: Path
    report_path: Path | None
    summary: dict[str, Any]


@dataclass(slots=True)
class ReviewStatusResult:
    review_dir: Path
    summary: dict[str, Any]


@dataclass(slots=True)
class ReviewContactSheetResult:
    output_dir: Path
    summary: dict[str, Any]


@dataclass(slots=True)
class ReviewAnnotationTemplateResult:
    output_csv: Path
    summary: dict[str, Any]


@dataclass(slots=True)
class ReviewAnnotationImportResult:
    review_dir: Path
    summary: dict[str, Any]


def build_review_queue(
    *,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    output_dir: Path | None = None,
    compare_run_dirs: list[Path] | None = None,
    max_items: int = 200,
    strategy: str = "hybrid",
) -> ReviewQueueResult:
    """Build and persist a prioritized human-review queue."""

    run_dir = run_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    strategy = strategy.strip().lower()
    if strategy not in {"hybrid", "uncertainty", "known-errors"}:
        raise ValueError(
            "strategy must be one of: hybrid, uncertainty, known-errors"
        )

    predictions = load_prediction_rows(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split=split,
    )
    compare_maps = [
        _compare_probability_map(
            run_dir=path.expanduser().resolve(),
            dataset_dir=dataset_dir,
            split=split,
        )
        for path in (compare_run_dirs or [])
    ]
    prepared = [
        _prepare_review_row(row, compare_maps=compare_maps)
        for row in predictions
    ]
    selected = _select_review_rows(
        prepared,
        max_items=max_items,
        strategy=strategy,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = run_dir / "review" / f"{split}-{timestamp}"
    review_dir = output_dir.expanduser().resolve()
    if review_dir.exists() and any(review_dir.iterdir()):
        raise FileExistsError(
            f"review output_dir already exists and is not empty: {review_dir}"
        )
    review_dir.mkdir(parents=True, exist_ok=True)

    queue_path = review_dir / "queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(_json_line(item) + "\n")

    manifest = {
        "workflow": "review-queue",
        "created_at_utc": _utc_now(),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "dataset_fingerprint": _dataset_fingerprint(dataset_dir),
        "split": split,
        "strategy": strategy,
        "max_items": int(max_items),
        "candidate_count": len(prepared),
        "queue_count": len(selected),
        "compare_run_dirs": [
            str(path.expanduser().resolve())
            for path in (compare_run_dirs or [])
        ],
        "morphology_tags": list(MORPHOLOGY_TAGS),
        "saved": {
            "manifest": "manifest.json",
            "queue": "queue.jsonl",
            "annotations": "annotations.jsonl",
            "state": "review_state.json",
        },
    }
    (review_dir / "manifest.json").write_text(
        _json_dumps(manifest) + "\n",
        encoding="utf-8",
    )
    _write_review_state(review_dir, current_index=0)
    return ReviewQueueResult(
        review_dir=review_dir,
        summary={"review_dir": str(review_dir), **manifest},
    )


def build_dataset_review_queue(
    *,
    dataset_dir: Path,
    split: str = "all",
    output_dir: Path | None = None,
    max_items: int = 200,
) -> ReviewQueueResult:
    """Build and persist a review queue directly from dataset samples."""

    dataset_dir = dataset_dir.expanduser().resolve()
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    split = split.strip().lower()
    if split not in {"all", "train", "val", "test"}:
        raise ValueError("split must be one of: all, train, val, test")

    dataset_validation = validate_dataset_dir(dataset_dir)
    prepared = [
        _prepare_dataset_review_row(row)
        for row in _dataset_review_rows(dataset_dir, split)
    ]
    selected = _select_dataset_review_rows(
        prepared,
        max_items=max_items,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = dataset_dir / "review" / f"{split}-dataset-{timestamp}"
    review_dir = output_dir.expanduser().resolve()
    if review_dir.exists() and any(review_dir.iterdir()):
        raise FileExistsError(
            f"review output_dir already exists and is not empty: {review_dir}"
        )
    review_dir.mkdir(parents=True, exist_ok=True)

    queue_path = review_dir / "queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(_json_line(item) + "\n")

    manifest = {
        "workflow": "review-queue-dataset",
        "created_at_utc": _utc_now(),
        "run_dir": None,
        "dataset_dir": str(dataset_dir),
        "dataset_fingerprint": _dataset_fingerprint(dataset_dir),
        "split": split,
        "strategy": "dataset-audit",
        "max_items": int(max_items),
        "candidate_count": len(prepared),
        "queue_count": len(selected),
        "dataset_validation": dataset_validation,
        "morphology_tags": list(MORPHOLOGY_TAGS),
        "saved": {
            "manifest": "manifest.json",
            "queue": "queue.jsonl",
            "annotations": "annotations.jsonl",
            "state": "review_state.json",
        },
    }
    (review_dir / "manifest.json").write_text(
        _json_dumps(manifest) + "\n",
        encoding="utf-8",
    )
    _write_review_state(review_dir, current_index=0)
    return ReviewQueueResult(
        review_dir=review_dir,
        summary={"review_dir": str(review_dir), **manifest},
    )


def build_entity_review_queue(
    *,
    source_review_dirs: list[Path],
    output_dir: Path,
) -> ReviewQueueResult:
    """Build an entity-class queue from existing binary real reviews."""

    if not source_review_dirs:
        raise ValueError("at least one source review directory is required")
    source_dirs = [path.expanduser().resolve() for path in source_review_dirs]
    seen_source_dirs: set[Path] = set()
    duplicate_source_dirs = []
    for source_dir in source_dirs:
        if source_dir in seen_source_dirs:
            duplicate_source_dirs.append(source_dir)
        seen_source_dirs.add(source_dir)
    if duplicate_source_dirs:
        duplicates = ", ".join(str(path) for path in duplicate_source_dirs)
        raise ValueError(f"source_review_dirs must be unique: {duplicates}")
    review_dir = output_dir.expanduser().resolve()
    if review_dir.exists() and any(review_dir.iterdir()):
        raise FileExistsError(
            "entity review output_dir already exists and is not empty: "
            f"{review_dir}"
        )
    review_dir.mkdir(parents=True, exist_ok=True)
    entity_dataset_dir = review_dir / "dataset"
    entity_dataset_dir.mkdir(exist_ok=True)

    source_caches: dict[Path, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    source_summaries = []
    for source_order, source_review_dir in enumerate(source_dirs):
        manifest, queue = load_review_manifest_and_queue(source_review_dir)
        source_dataset_dir = (
            Path(str(manifest["dataset_dir"])).expanduser().resolve()
        )
        _validate_dataset_fingerprint(manifest, source_dataset_dir)
        if source_dataset_dir not in source_caches:
            source_caches[source_dataset_dir] = _load_entity_source_dataset(
                source_dataset_dir
            )
        source_cache = source_caches[source_dataset_dir]
        _validate_queue_sample_indices(
            queue,
            sample_count=int(source_cache["sample_count"]),
        )
        latest_by_reviewer = latest_annotations_by_reviewer(source_review_dir)
        _validate_latest_annotation_keys(
            latest_by_reviewer,
            queue_keys={_review_item_key(item) for item in queue},
            annotation_kind="source review annotations",
        )
        source_digest = hashlib.sha256(
            str(source_review_dir).encode("utf-8")
        ).hexdigest()[:12]
        selected_count = 0
        for source_item in queue:
            source_queue_id = _review_item_key(source_item)
            for source_reviewer, annotation in sorted(
                latest_by_reviewer.get(source_queue_id, {}).items()
            ):
                reviewer_label = (
                    str(annotation.get("reviewer_label", "")).strip().lower()
                )
                if reviewer_label != "real":
                    continue
                _validate_annotation_identity(
                    source_item,
                    annotation,
                    reviewer=source_reviewer,
                    annotation_kind="source binary annotation",
                )
                source_sample_index = int(source_item["sample_index"])
                selected.append(
                    {
                        "source_order": source_order,
                        "source_review_dir": source_review_dir,
                        "source_review_digest": source_digest,
                        "source_manifest": manifest,
                        "source_dataset_dir": source_dataset_dir,
                        "source_item": source_item,
                        "source_queue_id": source_queue_id,
                        "source_sample_index": source_sample_index,
                        "source_reviewer": source_reviewer,
                        "binary_annotation": annotation,
                    }
                )
                selected_count += 1
        source_summaries.append(
            {
                "review_dir": str(source_review_dir),
                "dataset_dir": str(source_dataset_dir),
                "queue_count": len(queue),
                "selected_real_latest_annotations": selected_count,
                "annotation_file": _annotation_file_fingerprint(
                    source_review_dir
                ),
            }
        )

    if not selected:
        raise ValueError(
            "entity review queue has no latest binary real annotations"
        )
    _validate_entity_source_cache_compatibility(source_caches)

    first_cache = source_caches[selected[0]["source_dataset_dir"]]
    search_items = []
    template_items = []
    difference_items = []
    labels = []
    split_values = []
    metadata_rows = []
    queue_items = []
    for entity_index, record in enumerate(selected):
        source_cache = source_caches[record["source_dataset_dir"]]
        source_sample_index = int(record["source_sample_index"])
        search_image = np.asarray(source_cache["search"][source_sample_index])
        template_image = np.asarray(
            source_cache["template"][source_sample_index]
        )
        difference_array = source_cache["difference"]
        if difference_array is None:
            difference_image = search_image - template_image
        else:
            difference_image = np.asarray(
                difference_array[source_sample_index]
            )
        search_items.append(search_image)
        template_items.append(template_image)
        difference_items.append(difference_image)
        source_label = int(source_cache["labels"][source_sample_index])
        binary_label = (
            str(record["binary_annotation"].get("reviewer_label", ""))
            .strip()
            .lower()
        )
        if binary_label not in REVIEW_LABEL_TO_VALUE:
            raise ValueError(
                "entity source binary annotation has unsupported label: "
                f"{binary_label}"
            )
        entity_label = REVIEW_LABEL_TO_VALUE[binary_label]
        source_split_index = int(source_cache["split"][source_sample_index])
        labels.append(entity_label)
        split_values.append(source_split_index)
        queue_item = _entity_queue_item(
            record,
            entity_index=entity_index,
            entity_label=entity_label,
            source_label=source_label,
        )
        queue_items.append(queue_item)
        metadata_rows.append(
            _entity_metadata_row(
                record,
                queue_item=queue_item,
                source_cache=source_cache,
                entity_index=entity_index,
                entity_label=entity_label,
                source_label=source_label,
                source_split_index=source_split_index,
            )
        )
    if len({item["queue_id"] for item in queue_items}) != len(queue_items):
        raise ValueError("entity review queue_id collision detected")

    search_array = _stack_or_empty(search_items, first_cache["search"])
    template_array = _stack_or_empty(template_items, first_cache["template"])
    difference_array = _stack_or_empty(
        difference_items,
        (
            first_cache["difference"]
            if first_cache["difference"] is not None
            else first_cache["search"]
        ),
    )
    np.save(
        entity_dataset_dir / "search.npy",
        search_array,
        allow_pickle=False,
    )
    np.save(
        entity_dataset_dir / "template.npy",
        template_array,
        allow_pickle=False,
    )
    np.save(
        entity_dataset_dir / "difference.npy",
        difference_array,
        allow_pickle=False,
    )
    np.save(
        entity_dataset_dir / "labels.npy",
        np.asarray(labels, dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        entity_dataset_dir / "split.npy",
        np.asarray(split_values, dtype=np.int64),
        allow_pickle=False,
    )
    write_metadata_jsonl(entity_dataset_dir / "metadata.jsonl", metadata_rows)
    maybe_write_metadata_parquet(
        entity_dataset_dir / "metadata.parquet",
        metadata_rows,
    )
    dataset_validation = validate_dataset_dir(entity_dataset_dir)

    with (review_dir / "queue.jsonl").open("w", encoding="utf-8") as handle:
        for item in queue_items:
            handle.write(_json_line(item) + "\n")

    manifest = {
        "workflow": "entity-review-queue",
        "created_at_utc": _utc_now(),
        "review_task": "entity_classification",
        "dataset_dir": str(entity_dataset_dir),
        "dataset_fingerprint": _dataset_fingerprint(entity_dataset_dir),
        "source_review_dirs": [str(path) for path in source_dirs],
        "source_reviews": source_summaries,
        "queue_count": len(queue_items),
        "selected_binary_label": "real",
        "queue_identity_rule": (
            "one_entity_item_per_latest_binary_real_annotation"
        ),
        "queue_fanout_warning": (
            "multiple binary reviewers can create multiple entity items for "
            "the same source_queue_id"
        ),
        "unique_source_queue_count": len(
            {
                (
                    str(record["source_review_dir"]),
                    str(record["source_queue_id"]),
                )
                for record in selected
            }
        ),
        "multi_reviewer_fanout_count": len(selected)
        - len(
            {
                (
                    str(record["source_review_dir"]),
                    str(record["source_queue_id"]),
                )
                for record in selected
            }
        ),
        "latest_per_reviewer_rule": REVIEW_LATEST_PER_REVIEWER_RULE,
        "entity_labels": list(ENTITY_REVIEW_LABELS),
        "entity_label_descriptions": ENTITY_REVIEW_LABEL_DESCRIPTIONS,
        "entity_confidence_levels": list(ENTITY_REVIEW_CONFIDENCE_LEVELS),
        "dataset_validation": dataset_validation,
        "saved": {
            "manifest": "manifest.json",
            "queue": "queue.jsonl",
            "annotations": "entity_annotations.jsonl",
            "state": "review_state.json",
            "dataset": "dataset",
        },
    }
    (review_dir / "manifest.json").write_text(
        _json_dumps(manifest) + "\n",
        encoding="utf-8",
    )
    _write_review_state(review_dir, current_index=0)
    return ReviewQueueResult(
        review_dir=review_dir,
        summary={"review_dir": str(review_dir), **manifest},
    )


def load_prediction_rows(
    *,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
) -> list[dict[str, Any]]:
    """Load model predictions joined with canonical dataset metadata."""

    dataset_rows = _dataset_split_rows(dataset_dir, split)
    canonical_labels = np.asarray(
        [row["label"] for row in dataset_rows],
        dtype=np.int64,
    )
    probabilities: np.ndarray
    logits: np.ndarray
    prediction_jsonl = run_dir / "evaluation" / split / "predictions.jsonl"
    if prediction_jsonl.exists():
        artifact_summary = _validate_artifact_context(
            prediction_jsonl.parent,
            dataset_dir=dataset_dir,
            split=split,
            require_summary=True,
        )
        decision_threshold = _decision_threshold(artifact_summary)
        payloads = [
            json.loads(line)
            for line in prediction_jsonl.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(payloads) != len(dataset_rows):
            raise ValueError(
                "prediction row count does not match dataset split row count"
            )
        if any("sample_index" not in payload for payload in payloads):
            _validate_unique_prediction_candidate_ids(payloads)
        for index, (meta, payload) in enumerate(
            zip(dataset_rows, payloads, strict=True)
        ):
            _validate_prediction_identity(
                meta,
                payload,
                row_number=index,
            )
            if "label" in payload and int(payload["label"]) != int(
                meta["label"]
            ):
                raise ValueError(
                    "prediction row label mismatch at row "
                    f"{index}: label={payload['label']} "
                    f"expected={meta['label']}"
                )
        probabilities = np.asarray(
            [row["probability"] for row in payloads],
            dtype=np.float64,
        )
        logits = np.asarray(
            [row["logit"] for row in payloads], dtype=np.float64
        )
    else:
        inference_dir = run_dir / "inference" / split
        probabilities_path = inference_dir / "probabilities.npy"
        logits_path = inference_dir / "logits.npy"
        labels_path = inference_dir / "labels.npy"
        if not (
            probabilities_path.exists()
            and logits_path.exists()
            and labels_path.exists()
        ):
            raise FileNotFoundError(
                "missing evaluation predictions.jsonl or inference arrays "
                f"for run {run_dir} split {split}"
            )
        artifact_summary = _validate_artifact_context(
            inference_dir,
            dataset_dir=dataset_dir,
            split=split,
            require_summary=True,
        )
        decision_threshold = _decision_threshold(artifact_summary)
        probabilities = np.load(probabilities_path)
        logits = np.load(logits_path)
        artifact_labels = np.load(labels_path)
        if len(probabilities) != len(dataset_rows):
            raise ValueError(
                "inference row count does not match dataset split row count"
            )
        if not np.array_equal(
            np.asarray(artifact_labels, dtype=np.int64),
            canonical_labels,
        ):
            raise ValueError(
                "inference labels.npy does not match dataset split labels"
            )

    rows = []
    for meta, label, prob, logit in zip(
        dataset_rows,
        canonical_labels,
        probabilities,
        logits,
        strict=True,
    ):
        row = dict(meta)
        row["label"] = int(label)
        row["probability"] = float(prob)
        row["logit"] = float(logit)
        row["decision_threshold"] = float(decision_threshold)
        rows.append(_jsonable(row))
    return rows


def aggregate_review_annotations(
    *,
    review_dir: Path,
    output_report: Path | None = None,
    min_reviewers: int = 2,
    min_actionable_reviewers: int = 2,
    consensus_rule: str = "unanimous",
) -> ReviewAggregationResult:
    """Summarize latest per-reviewer annotations into explicit decisions."""

    review_dir = review_dir.expanduser().resolve()
    output_report = (
        output_report.expanduser().resolve()
        if output_report is not None
        else None
    )
    if min_reviewers <= 0:
        raise ValueError("min_reviewers must be positive")
    if min_actionable_reviewers <= 0:
        raise ValueError("min_actionable_reviewers must be positive")
    if min_actionable_reviewers > min_reviewers:
        raise ValueError(
            "min_actionable_reviewers cannot exceed min_reviewers"
        )
    consensus_rule = consensus_rule.strip().lower()
    if consensus_rule not in REVIEW_CONSENSUS_RULES:
        choices = ", ".join(sorted(REVIEW_CONSENSUS_RULES))
        raise ValueError(f"consensus_rule must be one of: {choices}")

    manifest, queue = load_review_manifest_and_queue(review_dir)
    latest_by_reviewer = latest_annotations_by_reviewer(review_dir)
    _validate_latest_annotation_keys(
        latest_by_reviewer,
        queue_keys={_review_item_key(item) for item in queue},
        annotation_kind="review annotations",
    )
    annotation_file = _annotation_file_fingerprint(review_dir)
    queue_file = _queue_file_fingerprint(review_dir)
    decisions = []
    status_counts = {status: 0 for status in REVIEW_AGGREGATION_STATUSES}
    actionable_label_counts = {
        label: 0 for label in sorted(REVIEW_LABEL_TO_VALUE)
    }
    reviewer_names = sorted(
        {
            reviewer
            for reviewer_payloads in latest_by_reviewer.values()
            for reviewer in reviewer_payloads
        }
    )

    for item in queue:
        decision = _aggregate_review_item(
            item,
            latest_by_reviewer.get(_review_item_key(item), {}),
            min_reviewers=min_reviewers,
            min_actionable_reviewers=min_actionable_reviewers,
            consensus_rule=consensus_rule,
        )
        status_counts[decision["status"]] += 1
        consensus_label = decision.get("consensus_label")
        if decision["status"] == "actionable" and consensus_label:
            if str(consensus_label) not in actionable_label_counts:
                raise ValueError(
                    "aggregation produced unrecognized consensus_label: "
                    f"{consensus_label}"
                )
            actionable_label_counts[str(consensus_label)] += 1
        decisions.append(decision)

    summary = {
        "workflow": "review-aggregate",
        "created_at_utc": _utc_now(),
        "review_dir": str(review_dir),
        "dataset_dir": manifest.get("dataset_dir"),
        "queue_count": len(queue),
        "reviewer_count": len(reviewer_names),
        "reviewers": reviewer_names,
        "latest_review_count": sum(
            len(reviewer_payloads)
            for reviewer_payloads in latest_by_reviewer.values()
        ),
        "annotation_file": annotation_file,
        "queue_file": queue_file,
        "latest_per_reviewer_rule": REVIEW_LATEST_PER_REVIEWER_RULE,
        "aggregation_rule_version": REVIEW_AGGREGATION_RULE_VERSION,
        "consensus_rule": consensus_rule,
        "min_reviewers": int(min_reviewers),
        "min_actionable_reviewers": int(min_actionable_reviewers),
        "status_counts": status_counts,
        "actionable_label_counts": actionable_label_counts,
        "decisions": decisions,
    }
    if output_report is not None:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(
            _json_dumps(summary) + "\n",
            encoding="utf-8",
        )
    return ReviewAggregationResult(
        review_dir=review_dir,
        report_path=output_report,
        summary=summary,
    )


def summarize_review_status(
    *,
    review_dir: Path,
    min_reviewers: int = 2,
    min_actionable_reviewers: int = 2,
    consensus_rule: str = "unanimous",
    include_decisions: bool = False,
) -> ReviewStatusResult:
    """Summarize whether a review queue is ready for label application."""

    review_dir = review_dir.expanduser().resolve()
    manifest, queue = load_review_manifest_and_queue(review_dir)
    queue_count = len(queue)
    queue_keys = {_review_item_key(item) for item in queue}
    latest_by_reviewer = latest_annotations_by_reviewer(review_dir)
    annotated_queue_keys = {
        key
        for key, reviewer_payloads in latest_by_reviewer.items()
        if reviewer_payloads
    }
    aggregation = aggregate_review_annotations(
        review_dir=review_dir,
        min_reviewers=min_reviewers,
        min_actionable_reviewers=min_actionable_reviewers,
        consensus_rule=consensus_rule,
    ).summary
    status_counts = dict(aggregation["status_counts"])
    actionable_count = int(status_counts.get("actionable") or 0)
    non_actionable_counts = {
        status: int(count)
        for status, count in status_counts.items()
        if status != "actionable" and int(count)
    }
    annotation_file = _annotation_file_fingerprint(review_dir)
    annotated_queue_count = len(queue_keys & annotated_queue_keys)
    missing_annotation_count = max(queue_count - annotated_queue_count, 0)
    blockers: list[str] = []
    if queue_count <= 0:
        blockers.append("review queue is empty")
    if annotation_file is None:
        blockers.append("annotations.jsonl is missing")
    elif missing_annotation_count:
        blockers.append(
            "annotations are missing for "
            f"{missing_annotation_count}/{queue_count} queued samples"
        )
    if actionable_count != queue_count:
        blockers.append(
            "actionable consensus covers "
            f"{actionable_count}/{queue_count} queued samples"
        )
    ready = queue_count > 0 and actionable_count == queue_count
    aggregation_summary = {
        key: value
        for key, value in aggregation.items()
        if include_decisions or key != "decisions"
    }
    summary = {
        "workflow": "review-status",
        "created_at_utc": _utc_now(),
        "review_dir": str(review_dir),
        "dataset_dir": manifest.get("dataset_dir"),
        "queue_count": queue_count,
        "annotation_file": annotation_file,
        "annotated_queue_count": annotated_queue_count,
        "missing_annotation_count": missing_annotation_count,
        "reviewer_count": aggregation["reviewer_count"],
        "reviewers": aggregation["reviewers"],
        "latest_review_count": aggregation["latest_review_count"],
        "min_reviewers": int(min_reviewers),
        "min_actionable_reviewers": int(min_actionable_reviewers),
        "consensus_rule": consensus_rule,
        "status_counts": status_counts,
        "actionable_label_counts": aggregation["actionable_label_counts"],
        "non_actionable_counts": non_actionable_counts,
        "ready_for_review_apply": ready,
        "blockers": blockers,
        "aggregation": aggregation_summary,
    }
    return ReviewStatusResult(review_dir=review_dir, summary=summary)


def aggregate_entity_review_annotations(
    *,
    review_dir: Path,
    output_report: Path | None = None,
    min_reviewers: int = 1,
    consensus_rule: str = "unanimous",
) -> ReviewAggregationResult:
    """Summarize entity-class annotations without changing binary labels."""

    review_dir = review_dir.expanduser().resolve()
    output_report = (
        output_report.expanduser().resolve()
        if output_report is not None
        else None
    )
    if min_reviewers <= 0:
        raise ValueError("min_reviewers must be positive")
    consensus_rule = consensus_rule.strip().lower()
    if consensus_rule not in REVIEW_CONSENSUS_RULES:
        choices = ", ".join(sorted(REVIEW_CONSENSUS_RULES))
        raise ValueError(f"consensus_rule must be one of: {choices}")

    manifest, queue = load_review_manifest_and_queue(review_dir)
    latest_by_reviewer = latest_entity_annotations_by_reviewer(review_dir)
    _validate_latest_annotation_keys(
        latest_by_reviewer,
        queue_keys={_review_item_key(item) for item in queue},
        annotation_kind="entity annotations",
    )
    annotation_file = _entity_annotation_file_fingerprint(review_dir)
    decisions = []
    status_counts = {
        status: 0 for status in ENTITY_REVIEW_AGGREGATION_STATUSES
    }
    consensus_label_counts = {label: 0 for label in ENTITY_REVIEW_LABELS}
    reviewer_names = sorted(
        {
            reviewer
            for reviewer_payloads in latest_by_reviewer.values()
            for reviewer in reviewer_payloads
        }
    )

    for item in queue:
        decision = _aggregate_entity_review_item(
            item,
            latest_by_reviewer.get(_review_item_key(item), {}),
            min_reviewers=min_reviewers,
            consensus_rule=consensus_rule,
        )
        status_counts[decision["status"]] += 1
        # Entity aggregation is report-only, so this summary counts resolved
        # labels for both actionable and explicit other_or_unsure outcomes.
        consensus_label = decision.get("resolved_entity_label")
        if consensus_label in consensus_label_counts:
            consensus_label_counts[str(consensus_label)] += 1
        decisions.append(decision)

    summary = {
        "workflow": "entity-review-aggregate",
        "created_at_utc": _utc_now(),
        "review_dir": str(review_dir),
        "dataset_dir": manifest.get("dataset_dir"),
        "queue_count": len(queue),
        "reviewer_count": len(reviewer_names),
        "reviewers": reviewer_names,
        "latest_review_count": sum(
            len(reviewer_payloads)
            for reviewer_payloads in latest_by_reviewer.values()
        ),
        "annotation_file": annotation_file,
        "latest_per_reviewer_rule": REVIEW_LATEST_PER_REVIEWER_RULE,
        "aggregation_rule_version": ENTITY_REVIEW_AGGREGATION_RULE_VERSION,
        "consensus_rule": consensus_rule,
        "min_reviewers": int(min_reviewers),
        "status_counts": status_counts,
        "consensus_entity_label_counts": consensus_label_counts,
        "entity_labels": list(ENTITY_REVIEW_LABELS),
        "entity_label_descriptions": ENTITY_REVIEW_LABEL_DESCRIPTIONS,
        "decisions": decisions,
    }
    if output_report is not None:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(
            _json_dumps(summary) + "\n",
            encoding="utf-8",
        )
    return ReviewAggregationResult(
        review_dir=review_dir,
        report_path=output_report,
        summary=summary,
    )


def apply_review_annotations(
    *,
    dataset_dir: Path,
    review_dir: Path,
    output_dir: Path,
    aggregation_report: Path | None = None,
) -> ReviewApplyResult:
    """Write a reviewed dataset from aggregated or single-reviewer labels."""

    dataset_dir = dataset_dir.expanduser().resolve()
    review_dir = review_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    aggregation_report = (
        aggregation_report.expanduser().resolve()
        if aggregation_report is not None
        else None
    )
    manifest, queue = load_review_manifest_and_queue(review_dir)
    manifest_dataset_dir = (
        Path(manifest["dataset_dir"]).expanduser().resolve()
    )
    if manifest_dataset_dir != dataset_dir:
        raise ValueError(
            "review manifest dataset_dir does not match requested "
            f"dataset_dir: manifest={manifest_dataset_dir} "
            f"requested={dataset_dir}"
        )
    _validate_dataset_fingerprint(manifest, dataset_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output_dir already exists and is not empty: {output_dir}"
        )

    all_latest = latest_annotation_per_queue(review_dir)
    latest_by_reviewer = latest_annotations_by_reviewer(
        review_dir,
        strict=True,
    )
    reviewer_names = {
        reviewer
        for reviewer_payloads in latest_by_reviewer.values()
        for reviewer in reviewer_payloads
    }
    aggregation_payload: dict[str, Any] | None = None
    if aggregation_report is not None:
        aggregation_payload = _load_review_aggregation_report(
            aggregation_report,
            review_dir=review_dir,
        )
        _validate_review_aggregation_report_decisions(
            aggregation_payload,
            queue=queue,
            latest_by_reviewer=latest_by_reviewer,
        )
        annotations = _aggregated_actionable_annotations(
            aggregation_payload,
            report_path=aggregation_report,
        )
    else:
        if len(reviewer_names) > 1:
            raise ValueError(
                "review-apply found annotations from multiple reviewers; "
                "run review-aggregate and pass --aggregation-report before "
                "materializing reviewed labels"
            )
        annotations = {
            key: annotation
            for key, annotation in all_latest.items()
            if str(annotation.get("reviewer_label", "")).strip().lower()
            in REVIEW_LABEL_TO_VALUE
        }
    labels = np.load(dataset_dir / "labels.npy").astype(np.int64)
    _validate_queue_sample_indices(queue, sample_count=int(labels.shape[0]))
    queued_samples = _queue_sample_index_map(queue)
    queued_items = {_review_item_key(item): item for item in queue}
    metadata_rows = load_metadata_rows(dataset_dir)
    if not metadata_rows:
        metadata_rows = [
            {"sample_index": index, "label": int(label)}
            for index, label in enumerate(labels.tolist())
        ]
    if len(metadata_rows) != labels.shape[0]:
        raise ValueError("metadata row count must match labels.npy")

    applied = 0
    status_counts = (
        aggregation_payload.get("status_counts", {})
        if aggregation_payload is not None
        else {}
    )
    if aggregation_payload is not None:
        skipped_unsure = int(status_counts.get("unsure_only", 0))
        skipped_conflicted = int(status_counts.get("conflicted", 0))
        skipped_no_actionable = int(status_counts.get("no_actionable", 0))
        skipped_insufficient_review = int(
            status_counts.get("insufficient_review", 0)
        )
    else:
        skipped_unsure = sum(
            1
            for key, annotation in all_latest.items()
            if key not in annotations
            and str(annotation.get("reviewer_label", "")).strip().lower()
            == "unsure"
        )
        skipped_conflicted = 0
        skipped_no_actionable = 0
        skipped_insufficient_review = 0
    for key, annotation in annotations.items():
        label_text = str(annotation.get("reviewer_label", "")).strip().lower()
        if label_text not in REVIEW_LABEL_TO_VALUE:
            if aggregation_payload is not None:
                raise ValueError(
                    "actionable aggregation decision has unrecognized "
                    f"consensus_label: {label_text}"
                )
            continue
        queue_id = str(annotation.get("queue_id") or key)
        if queue_id not in queued_samples:
            raise ValueError(
                f"review annotation is not present in queue.jsonl: {queue_id}"
            )
        reviewer = (
            _reviewer_name(annotation.get("reviewer"))
            if "reviewer" in annotation
            else "aggregation"
        )
        if not reviewer:
            raise ValueError(
                "review annotation must include reviewer: "
                f"queue_id={queue_id}"
            )
        _validate_annotation_identity(
            queued_items[queue_id],
            annotation,
            reviewer=reviewer,
            annotation_kind="review annotation",
        )
        sample_index = int(annotation["sample_index"])
        if sample_index != queued_samples[queue_id]:
            raise ValueError(
                "review annotation sample_index does not match queue.jsonl "
                f"for {queue_id}: annotation={sample_index} "
                f"queue={queued_samples[queue_id]}"
            )
        if sample_index < 0 or sample_index >= labels.shape[0]:
            raise ValueError(
                f"review sample_index is out of bounds: {sample_index}"
            )
        if "original_label" not in metadata_rows[sample_index]:
            metadata_rows[sample_index]["original_label"] = int(
                labels[sample_index]
            )
        if "original_label_source" not in metadata_rows[sample_index]:
            metadata_rows[sample_index]["original_label_source"] = (
                metadata_rows[sample_index].get("label_source")
            )
        labels[sample_index] = REVIEW_LABEL_TO_VALUE[label_text]
        metadata_update = {
            "label": int(labels[sample_index]),
            "label_source": (
                "human_review_aggregation"
                if aggregation_payload is not None
                else "human_review"
            ),
            "target_label_available": True,
            "review_queue_id": annotation.get("queue_id", key),
            "review_label": label_text,
            "review_timestamp_utc": annotation.get("timestamp_utc"),
            "review_morphology_tags": annotation.get("morphology_tags", []),
            "review_notes": annotation.get("notes", ""),
        }
        if "aggregation" in annotation:
            metadata_update.update(
                _review_aggregation_metadata(annotation["aggregation"])
            )
        metadata_rows[sample_index].update(metadata_update)
        applied += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("search.npy", "template.npy", "split.npy", "difference.npy"):
        source = dataset_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    np.save(output_dir / "labels.npy", labels, allow_pickle=False)
    write_metadata_jsonl(output_dir / "metadata.jsonl", metadata_rows)
    maybe_write_metadata_parquet(
        output_dir / "metadata.parquet", metadata_rows
    )

    validation = validate_dataset_dir(output_dir)
    summary = {
        "workflow": "review-apply",
        "created_at_utc": _utc_now(),
        "source_dataset_dir": str(dataset_dir),
        "review_dir": str(review_dir),
        "dataset_dir": str(output_dir),
        "review_apply_source": (
            "aggregation_report"
            if aggregation_report is not None
            else "legacy_latest_actionable"
        ),
        "aggregation_report": (
            str(aggregation_report)
            if aggregation_report is not None
            else None
        ),
        "annotation_count": len(annotations),
        "applied_label_count": applied,
        "skipped_unsure_count": skipped_unsure,
        "skipped_conflicted_count": skipped_conflicted,
        "skipped_no_actionable_count": skipped_no_actionable,
        "skipped_insufficient_review_count": skipped_insufficient_review,
        "validation": validation,
        "saved": {
            "search": "search.npy",
            "template": "template.npy",
            "labels": "labels.npy",
            "split": "split.npy",
            "metadata_jsonl": "metadata.jsonl",
        },
    }
    if (output_dir / "difference.npy").exists():
        summary["saved"]["difference"] = "difference.npy"
    if (output_dir / "metadata.parquet").exists():
        summary["saved"]["metadata_parquet"] = "metadata.parquet"
    (output_dir / "summary.json").write_text(
        _json_dumps(summary) + "\n",
        encoding="utf-8",
    )
    return ReviewApplyResult(output_dir=output_dir, summary=summary)


def review_bokeh_server_summary(
    *,
    review_dir: Path,
    host: str = "localhost",
    port: int = 0,
    show_url_only: bool = False,
) -> dict[str, Any]:
    """Return server metadata or start a blocking local Bokeh server."""

    review_dir = review_dir.expanduser().resolve()
    _validate_bokeh_host(host)
    manifest, queue = load_review_manifest_and_queue(review_dir)
    requested_port = int(port)
    if requested_port <= 0:
        raise ValueError("review-bokeh requires --port > 0")
    url = f"http://{_bokeh_url_host(host)}:{requested_port}/"
    if show_url_only:
        return {
            "workflow": "review-bokeh",
            "review_dir": str(review_dir),
            "queue_count": len(queue),
            "url": url,
            "server_started": False,
            "saved": manifest.get("saved", {}),
        }

    try:
        from bokeh.server.server import Server
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bokeh is required for review-bokeh; run "
            "'uv sync --extra viz' for development or install "
            "'cuphoton[viz]'"
        ) from exc

    def app(doc):
        build_review_document(doc, review_dir=review_dir)

    origin_port = str(requested_port)
    allow_origin = [
        f"{_bokeh_url_host(host)}:{origin_port}",
        f"localhost:{origin_port}",
        f"127.0.0.1:{origin_port}",
    ]
    server = Server(
        {"/": app},
        address=host,
        port=requested_port,
        allow_websocket_origin=allow_origin,
    )
    server.start()
    print(f"XScan review server: {url}")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        pass
    return {
        "workflow": "review-bokeh",
        "review_dir": str(review_dir),
        "queue_count": len(queue),
        "url": url,
        "server_started": True,
        "saved": manifest.get("saved", {}),
    }


def entity_review_bokeh_server_summary(
    *,
    review_dir: Path,
    host: str = "localhost",
    port: int = 0,
    show_url_only: bool = False,
) -> dict[str, Any]:
    """Return server metadata or start a blocking entity-review server."""

    review_dir = review_dir.expanduser().resolve()
    _validate_bokeh_host(host)
    manifest, queue = load_review_manifest_and_queue(review_dir)
    requested_port = int(port)
    if requested_port <= 0:
        raise ValueError("entity-review-bokeh requires --port > 0")
    url = f"http://{_bokeh_url_host(host)}:{requested_port}/"
    if show_url_only:
        return {
            "workflow": "entity-review-bokeh",
            "review_dir": str(review_dir),
            "queue_count": len(queue),
            "url": url,
            "server_started": False,
            "saved": manifest.get("saved", {}),
        }

    try:
        from bokeh.server.server import Server
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bokeh is required for entity-review-bokeh; run "
            "'uv sync --extra viz' for development or install "
            "'cuphoton[viz]'"
        ) from exc

    def app(doc):
        build_entity_review_document(doc, review_dir=review_dir)

    origin_port = str(requested_port)
    allow_origin = [
        f"{_bokeh_url_host(host)}:{origin_port}",
        f"localhost:{origin_port}",
        f"127.0.0.1:{origin_port}",
    ]
    server = Server(
        {"/": app},
        address=host,
        port=requested_port,
        allow_websocket_origin=allow_origin,
    )
    server.start()
    print(f"XScan entity review server: {url}")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        pass
    return {
        "workflow": "entity-review-bokeh",
        "review_dir": str(review_dir),
        "queue_count": len(queue),
        "url": url,
        "server_started": True,
        "saved": manifest.get("saved", {}),
    }


_XSCAN_REVIEW_CSS = """
:root {
  --review-bg: #f7fbff;
  --review-surface: #ffffff;
  --review-surface-soft: #f3f7fc;
  --review-text: #07145c;
  --review-muted: #7080b5;
  --review-border: #dde6f4;
  --review-teal: #03c7b7;
  --review-indigo: #5d4df2;
  --review-red: #ff2f3f;
  --review-green: #19c98d;
  --review-amber: #ff9f43;
  --review-shadow: 0 14px 38px rgba(42, 55, 105, 0.12);
}
html, body {
  min-height: 100%;
  margin: 0;
  background: var(--review-bg);
  color: var(--review-text);
  font-family: "Inter", "IBM Plex Sans", "Avenir Next", "Segoe UI",
    sans-serif;
}
.review-shell {
  width: min(1440px, calc(100vw - 36px));
  margin: 18px auto 34px;
  gap: 12px !important;
}
.review-header,
.main-review-row,
.review-card {
  width: 100%;
}
.review-header {
  align-items: center;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid var(--review-border);
  border-radius: 8px;
  box-shadow: var(--review-shadow);
  gap: 14px !important;
  padding: 10px 14px;
}
.review-brand {
  min-width: 320px;
}
.brand-title {
  color: var(--review-text);
  font-size: 30px;
  font-weight: 760;
  line-height: 1.06;
  margin: 0;
}
.brand-subtitle {
  color: var(--review-muted);
  font-size: 14px;
  margin-top: 5px;
}
.candidate-pill {
  align-items: center;
  background: #eefaff;
  border: 1px solid #d5edf7;
  border-radius: 999px;
  display: inline-flex;
  gap: 10px;
  margin-top: 0;
  max-width: 100%;
  padding: 6px 11px;
}
.candidate-pill span {
  color: var(--review-muted);
  font-size: 11px;
  font-weight: 700;
}
.candidate-pill strong {
  color: var(--review-text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.review-nav {
  align-items: center;
  background: var(--review-surface);
  border: 1px solid var(--review-border);
  border-radius: 8px;
  box-shadow: 0 8px 22px rgba(42, 55, 105, 0.08);
  gap: 10px !important;
  justify-content: center;
  padding: 8px 10px;
}
.progress-count {
  color: var(--review-text);
  font-size: 18px;
  font-weight: 760;
  min-width: 92px;
  text-align: center;
}
.progress-track {
  background: #e7ebf3;
  border-radius: 999px;
  height: 6px;
  margin: 8px auto 0;
  overflow: hidden;
  width: 88px;
}
.progress-fill {
  background: var(--review-teal);
  border-radius: inherit;
  height: 100%;
}
.header-tools {
  gap: 8px !important;
  min-width: 260px;
}
.main-review-row {
  align-items: stretch;
  gap: 12px !important;
}
.review-card {
  background: var(--review-surface);
  border: 1px solid var(--review-border);
  border-radius: 8px;
  box-shadow: var(--review-shadow);
  padding: 12px;
}
.image-triptych {
  flex: 1 1 auto;
  gap: 10px !important;
}
.triptych-row {
  gap: 12px !important;
  justify-content: space-between;
}
.stamp-card {
  background: #ffffff;
  border: 1px solid #e4ebf5;
  border-radius: 8px;
  overflow: hidden;
  padding: 8px;
}
.card-title h2,
.summary-card h2,
.classification-card h2,
.other-review-card h2 {
  color: var(--review-text);
  font-size: 18px;
  font-weight: 760;
  margin: 0 0 8px;
}
.summary-wrap {
  max-width: none;
  min-width: 0;
  width: 100%;
}
.summary-card {
  background: var(--review-surface);
  border: 1px solid var(--review-border);
  border-radius: 8px;
  box-sizing: border-box;
  box-shadow: var(--review-shadow);
  padding: 10px 12px;
  width: 100%;
}
.summary-grid {
  column-gap: 20px;
  color: var(--review-text);
  display: grid;
  font-size: 12px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  row-gap: 0;
  width: 100%;
}
.summary-field {
  border-top: 1px solid #edf1f7;
  display: grid;
  gap: 8px;
  grid-template-columns: minmax(112px, 42%) minmax(0, 1fr);
  line-height: 1.25;
  padding: 4px 0;
}
.summary-key {
  color: var(--review-muted);
  font-weight: 650;
}
.summary-value {
  color: var(--review-text);
  min-width: 0;
  overflow-wrap: anywhere;
}
.summary-table,
.other-review-table {
  border-collapse: collapse;
  color: var(--review-text);
  font-size: 13px;
  width: 100%;
}
.summary-table th,
.summary-table td,
.other-review-table th,
.other-review-table td {
  border-bottom: 1px solid #edf1f7;
  padding: 8px 0;
  text-align: left;
  vertical-align: top;
}
.summary-table th,
.other-review-table th {
  color: var(--review-muted);
  font-weight: 650;
  padding-right: 16px;
  width: 38%;
}
.summary-table td,
.other-review-table td {
  overflow-wrap: anywhere;
}
.classification-card {
  gap: 10px !important;
}
.field-grid,
.decision-row {
  gap: 12px !important;
}
.decision-row {
  align-items: center;
}
.status-card {
  color: var(--review-muted);
  font-size: 13px;
}
.status-line {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.status-pill {
  background: var(--review-surface-soft);
  border-radius: 999px;
  color: var(--review-text);
  display: inline-block;
  font-weight: 650;
  padding: 6px 10px;
}
.status-warning {
  color: #b00020;
  font-weight: 760;
}
.required-hint {
  color: #b00020;
  font-weight: 760;
}
.other-reviews-wrap {
  width: 100%;
}
.other-review-card {
  background: var(--review-surface);
  border: 1px solid var(--review-border);
  border-radius: 8px;
  box-shadow: var(--review-shadow);
  padding: 14px;
}
.bk-btn-success,
.bk-btn-primary {
  background-color: var(--review-indigo) !important;
  border-color: var(--review-indigo) !important;
}
.bk-btn-danger {
  background-color: var(--review-red) !important;
  border-color: var(--review-red) !important;
}
.bk-btn-warning {
  background-color: var(--review-amber) !important;
  border-color: var(--review-amber) !important;
}
.bk-input,
.bk-input-group,
.bk-select,
.bk-textarea {
  color: var(--review-text);
}
@media (max-width: 1180px) {
  .review-header,
  .main-review-row,
  .triptych-row,
  .field-grid {
    flex-direction: column !important;
  }
  .summary-wrap,
  .header-tools,
  .review-brand {
    max-width: none;
    min-width: 0;
    width: 100%;
  }
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .brand-title {
    font-size: 28px;
  }
}
"""


_ASSET_DATA_URI_CACHE: dict[tuple[str, str], str] = {}


def _asset_data_uri(filename: str, mime_type: str) -> str:
    key = (filename, mime_type)
    cached = _ASSET_DATA_URI_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        payload = (
            resources.files(__package__)
            .joinpath("assets", filename)
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError):
        _ASSET_DATA_URI_CACHE[key] = ""
        return ""
    encoded = base64.b64encode(payload).decode("ascii")
    uri = f"data:{mime_type};base64,{encoded}"
    _ASSET_DATA_URI_CACHE[key] = uri
    return uri


def _favicon_links_html() -> str:
    icon_src = _asset_data_uri("favicon.ico", "image/x-icon")
    if not icon_src:
        return ""
    escaped_icon = html.escape(icon_src, quote=True)
    return (
        f'<link rel="icon" type="image/x-icon" href="{escaped_icon}">\n'
        f'<link rel="shortcut icon" href="{escaped_icon}">\n'
    )


def _review_logo_html(size_px: int = 76) -> str:
    logo_src = _asset_data_uri("cuphoton-logo-128.png", "image/png")
    if not logo_src:
        return ""
    escaped_logo = html.escape(logo_src, quote=True)
    return (
        "<img alt='cuPhoton logo' "
        f"src='{escaped_logo}' "
        "style='display:block;flex:0 0 auto;"
        f"height:{size_px}px;width:{size_px}px;"
        "object-fit:contain;' />"
    )


_REVIEW_HEADER_STYLE = {
    "align-items": "center",
    "background": "rgba(255, 255, 255, 0.94)",
    "border": "1px solid #dde6f4",
    "border-radius": "8px",
    "box-shadow": "0 14px 38px rgba(42, 55, 105, 0.12)",
    "gap": "14px",
    "padding": "10px 14px",
}
_REVIEW_CARD_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #dde6f4",
    "border-radius": "8px",
    "box-shadow": "0 14px 38px rgba(42, 55, 105, 0.12)",
    "padding": "12px",
}
_STAMP_CARD_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #e4ebf5",
    "border-radius": "8px",
    "padding": "8px",
}
_TABLE_CELL_STYLE = "border-top:1px solid #edf1f7;padding:8px 10px 8px 0;"
_TABLE_LAST_CELL_STYLE = "border-top:1px solid #edf1f7;padding:8px 0;"
_TABLE_HEADER_STYLE = "color:#7080b5;text-align:left;padding:0 10px 8px 0;"
_TABLE_LAST_HEADER_STYLE = "color:#7080b5;text-align:left;padding:0 0 8px;"


def _apply_xscan_review_style(doc: Any) -> None:
    from bokeh.core.templates import get_env

    doc.template = get_env().from_string(
        "{% extends base %}\n"
        "{% block preamble %}\n"
        f"{_favicon_links_html()}"
        "<style>\n"
        f"{_XSCAN_REVIEW_CSS}\n"
        "</style>\n"
        "{% endblock %}"
    )


def _short_review_identifier(value: Any, *, keep: int = 12) -> str:
    text = str(value or "")
    if len(text) <= keep * 2 + 3:
        return text
    return f"{text[:keep]}...{text[-keep:]}"


def _review_header_html(
    *,
    title: str,
    subtitle: str,
    id_label: str,
    item: dict[str, Any],
    include_identifier: bool = True,
) -> str:
    identifier = item.get("candidate_id") or item.get("queue_id", "")
    identifier_html = (
        _review_identifier_pill_html(id_label, identifier)
        if include_identifier
        else ""
    )
    return (
        "<div style='align-items:center;display:flex;gap:16px;"
        "min-width:320px;'>"
        f"{_review_logo_html()}"
        "<div style='min-width:0;'>"
        "<div style='color:#07145c;font-size:34px;font-weight:760;"
        "line-height:1.06;'>"
        f"{html.escape(title)}</div>"
        "<div style='color:#7080b5;font-size:15px;margin-top:7px;'>"
        f"{html.escape(subtitle)}</div>"
        f"{identifier_html}"
        "</div>"
        "</div>"
    )


def _review_identifier_pill_html(id_label: str, identifier: Any) -> str:
    return (
        "<div class='candidate-pill' style='align-items:center;"
        "background:#eefaff;border:1px solid #d5edf7;border-radius:999px;"
        "display:inline-flex;gap:10px;max-width:100%;padding:6px 11px;'>"
        "<span style='color:#7080b5;font-size:11px;font-weight:700;'>"
        f"{html.escape(id_label)}</span>"
        "<strong style='color:#07145c;font-family:ui-monospace,"
        "SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;"
        "overflow-wrap:anywhere;'>"
        f"{html.escape(_short_review_identifier(identifier))}"
        "</strong>"
        "</div>"
    )


def _image_review_heading_html(item: dict[str, Any]) -> str:
    identifier = item.get("candidate_id") or item.get("queue_id", "")
    return (
        "<div style='align-items:center;display:flex;gap:14px;"
        "margin:0 0 8px;'>"
        "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
        "line-height:1.1;margin:0;'>Image Review</h2>"
        f"{_review_identifier_pill_html('Candidate ID', identifier)}"
        "</div>"
    )


def _review_progress_html(current: int, total: int) -> str:
    percent = 0 if total <= 0 else int(round((current / total) * 100))
    percent = min(max(percent, 0), 100)
    return (
        "<div style='color:#07145c;font-size:18px;font-weight:760;"
        "min-width:92px;text-align:center;'>"
        f"{current} / {total}"
        "</div>"
        "<div style='background:#e7ebf3;border-radius:999px;height:6px;"
        "margin:8px auto 0;overflow:hidden;width:88px;'>"
        "<div style='background:#03c7b7;border-radius:inherit;height:100%;"
        f"width:{percent}%;'></div>"
        "</div>"
    )


def _review_status_html(
    *,
    notice: str | None,
    reviewer: str,
    reviewed_by_user: int,
    total_items: int,
    total_decisions: int,
    previous: str,
) -> str:
    reviewer_text = html.escape(reviewer or "no username")
    notice_html = (
        "<span style='color:#b00020;font-weight:760;'>"
        f"{html.escape(notice)}</span>"
        if notice
        else ""
    )
    username_notice = (
        ""
        if reviewer
        else (
            "<span style='color:#b00020;font-weight:760;'>"
            "Set Username before saving.</span>"
        )
    )
    previous_html = (
        "<span style='background:#f3f7fc;border-radius:999px;color:#07145c;"
        "display:inline-block;font-weight:650;padding:6px 10px;'>"
        f"{html.escape(previous)}</span>"
        if previous
        else ""
    )
    return (
        "<div style='align-items:center;display:flex;flex-wrap:wrap;"
        "gap:12px;'>"
        f"{notice_html}"
        f"{username_notice}"
        f"<span>Reviewer: <strong>{reviewer_text}</strong></span>"
        f"<span>Reviewed: {reviewed_by_user} / {total_items}</span>"
        f"<span>Total decisions: {total_decisions}</span>"
        f"{previous_html}"
        "</div>"
    )


def build_review_document(doc, *, review_dir: Path) -> None:
    """Populate a Bokeh document for interactive review."""

    from bokeh.events import DocumentReady, Tap
    from bokeh.layouts import column, row
    from bokeh.models import (
        Button,
        CheckboxButtonGroup,
        ColorBar,
        ColumnDataSource,
        CustomJS,
        Div,
        LabelSet,
        LinearColorMapper,
        Range1d,
        Span,
        TextAreaInput,
        TextInput,
    )
    from bokeh.palettes import Greys256, RdBu11
    from bokeh.plotting import figure

    _apply_xscan_review_style(doc)
    manifest, queue = load_review_manifest_and_queue(review_dir)
    dataset_dir = Path(manifest["dataset_dir"]).expanduser().resolve()
    _validate_dataset_fingerprint(manifest, dataset_dir)
    if not queue:
        doc.add_root(Div(text="<h2>No review queue items found.</h2>"))
        return

    search = np.load(dataset_dir / "search.npy", mmap_mode="r")
    template = np.load(dataset_dir / "template.npy", mmap_mode="r")
    difference = (
        np.load(dataset_dir / "difference.npy", mmap_mode="r")
        if (dataset_dir / "difference.npy").exists()
        else None
    )
    _validate_queue_sample_indices(queue, sample_count=int(search.shape[0]))
    state_path = review_dir / "review_state.json"
    state = _load_review_state(review_dir)
    fallback_index = min(
        max(int(state.get("current_index", 0)), 0),
        len(queue) - 1,
    )
    initial_index, query_notice = _initial_review_index_from_query(
        doc,
        queue,
        fallback_index=fallback_index,
    )
    index = {"value": initial_index}
    review_storage_key = hashlib.sha256(
        str(review_dir).encode("utf-8")
    ).hexdigest()[:16]

    title = Div(width=470, css_classes=["review-brand"])
    image_heading = Div(width=1370, name="review-image-heading")
    progress_div = Div(
        sizing_mode="stretch_width",
        css_classes=["review-progress-card"],
    )
    metadata_div = Div(
        width=1400,
        css_classes=["summary-wrap"],
        styles={"margin": "0"},
    )
    status_div = Div(sizing_mode="stretch_width", css_classes=["status-card"])
    username = TextInput(
        title="Username:",
        placeholder="Required before saving reviews",
        sizing_mode="stretch_width",
        css_classes=["review-username"],
    )
    username_required = Div(
        text=_username_required_html(False),
        name="review-username-required",
        sizing_mode="stretch_width",
        css_classes=["required-hint"],
    )
    others_div = Div(
        text="",
        width=1400,
        css_classes=["other-reviews-wrap"],
    )
    notes = TextAreaInput(
        title="Notes",
        rows=4,
        sizing_mode="stretch_width",
        css_classes=["notes-input"],
    )
    tag_control = CheckboxButtonGroup(
        labels=list(MORPHOLOGY_TAGS),
        active=[],
        css_classes=["tag-control"],
    )
    url_state = Div(
        text=str(int(queue[initial_index]["sample_index"])),
        visible=False,
        name="review-url-state",
    )
    source_a = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    source_b = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    source_c = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    source_d = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    label_source_a = _pixel_label_source(ColumnDataSource)
    label_source_b = _pixel_label_source(ColumnDataSource)
    label_source_c = _pixel_label_source(ColumnDataSource)
    label_source_d = _pixel_label_source(ColumnDataSource)
    pixel_selection_source = ColumnDataSource(
        {"x": [], "y": [], "width": [], "height": []}
    )
    mapper_a = LinearColorMapper(palette=list(Greys256), low=0.0, high=1.0)
    mapper_b = LinearColorMapper(palette=list(Greys256), low=0.0, high=1.0)
    mapper_c = LinearColorMapper(
        palette=list(reversed(RdBu11)), low=-1.0, high=1.0
    )
    mapper_d = LinearColorMapper(
        palette=list(reversed(RdBu11)), low=-1.0, high=1.0
    )
    stamp_height, stamp_width = search.shape[1:]
    shared_x_range = Range1d(0, stamp_width)
    shared_y_range = Range1d(0, stamp_height)

    def image_figure(
        label: str,
        source: ColumnDataSource,
        mapper,
        label_source: ColumnDataSource,
    ):
        fig = figure(
            title=label,
            width=320,
            height=320,
            match_aspect=True,
            x_range=shared_x_range,
            y_range=shared_y_range,
            tools="tap,pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
        )
        fig.image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=source,
            color_mapper=mapper,
        )
        fig.add_layout(
            ColorBar(
                color_mapper=mapper,
                width=10,
                label_standoff=5,
                border_line_color=None,
                background_fill_alpha=0.0,
                major_label_text_font_size="10px",
            ),
            "right",
        )
        fig.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=pixel_selection_source,
            fill_alpha=0.0,
            line_color="#ffd400",
            line_width=2,
        )
        fig.add_layout(_pixel_label_set(LabelSet, label_source))
        fig.add_layout(
            Span(
                location=stamp_width / 2.0,
                dimension="height",
                line_color="#03c7b7",
                line_alpha=0.75,
                line_width=1,
            )
        )
        fig.add_layout(
            Span(
                location=stamp_height / 2.0,
                dimension="width",
                line_color="#03c7b7",
                line_alpha=0.75,
                line_width=1,
            )
        )
        fig.grid.visible = False
        fig.axis.visible = False
        fig.outline_line_color = "#d9e2f0"
        fig.border_fill_color = "#ffffff"
        fig.background_fill_color = "#ffffff"
        fig.title.text_color = "#07145c"
        fig.title.text_font_size = "13px"
        fig.title.text_font_style = "bold"
        fig.toolbar.logo = None
        return fig

    def current_item() -> dict[str, Any]:
        return queue[index["value"]]

    def refresh() -> None:
        item = current_item()
        sample_index = int(item["sample_index"])
        url_state.text = str(sample_index)
        raw_diff_image = search[sample_index] - template[sample_index]
        alard_lupton_image = (
            difference[sample_index]
            if difference is not None
            else raw_diff_image
        )
        _update_image(
            source_a, mapper_a, search[sample_index], diverging=False
        )
        _update_image(
            source_b, mapper_b, template[sample_index], diverging=False
        )
        _update_image(source_c, mapper_c, raw_diff_image, diverging=True)
        _update_image(source_d, mapper_d, alard_lupton_image, diverging=True)
        latest_by_reviewer = latest_annotations_by_reviewer(
            review_dir,
            strict=False,
        )
        queue_id = str(item["queue_id"])
        reviewer = _reviewer_name(username.value)
        has_reviewer = bool(reviewer)
        for button in (real, bogus, unsure):
            button.disabled = not has_reviewer
        username_required.text = _username_required_html(has_reviewer)
        annotation = (
            latest_by_reviewer.get(queue_id, {}).get(reviewer)
            if has_reviewer
            else None
        )
        active_tags = []
        notes.value = ""
        if annotation is not None:
            tag_set = set(annotation.get("morphology_tags", []))
            active_tags = [
                idx
                for idx, label in enumerate(MORPHOLOGY_TAGS)
                if label in tag_set
            ]
            notes.value = str(annotation.get("notes", ""))
        tag_control.active = active_tags
        others_div.text = ""
        title.text = _review_header_html(
            title="XScan Review",
            subtitle="Review astronomical detections with human insight",
            id_label="Candidate ID",
            item=item,
            include_identifier=False,
        )
        image_heading.text = _image_review_heading_html(item)
        progress_div.text = _review_progress_html(
            index["value"] + 1,
            len(queue),
        )
        metadata_div.text = _metadata_table(item)
        reviewed_by_user = _reviewer_reviewed_count(
            latest_by_reviewer,
            reviewer,
        )
        total_decisions = sum(
            len(payloads) for payloads in latest_by_reviewer.values()
        )
        previous = (
            f"Last label: {annotation.get('reviewer_label')}"
            if annotation is not None
            else ""
        )
        status_div.text = _review_status_html(
            notice=query_notice,
            reviewer=reviewer,
            reviewed_by_user=reviewed_by_user,
            total_items=len(queue),
            total_decisions=total_decisions,
            previous=previous,
        )
        state = _load_review_state(review_dir)
        state["current_index"] = index["value"]
        state["updated_at_utc"] = _utc_now()
        state_path.write_text(_json_dumps(state) + "\n", encoding="utf-8")

    def persist(label: str) -> None:
        item = current_item()
        reviewer = _reviewer_name(username.value)
        if not reviewer:
            status_div.text = (
                "<p><strong>Set Username before saving a review.</strong></p>"
                + status_div.text
            )
            return
        payload = {
            "timestamp_utc": _utc_now(),
            "reviewer": reviewer,
            "queue_id": item["queue_id"],
            "sample_index": int(item["sample_index"]),
            "candidate_id": item.get("candidate_id"),
            "reviewer_label": label,
            "morphology_tags": [
                MORPHOLOGY_TAGS[idx] for idx in tag_control.active
            ],
            "notes": notes.value,
            "source_label": item.get("label"),
            "source_probability": item.get("probability"),
            "rank_reason": item.get("rank_reason"),
        }
        _append_annotation(review_dir, payload)
        if index["value"] < len(queue) - 1:
            index["value"] += 1
        refresh()

    def move(delta: int) -> None:
        index["value"] = min(max(index["value"] + delta, 0), len(queue) - 1)
        refresh()

    real = Button(
        label="Real",
        button_type="success",
        width=120,
        css_classes=["decision-button"],
    )
    bogus = Button(
        label="Bogus",
        button_type="danger",
        width=120,
        css_classes=["decision-button"],
    )
    unsure = Button(
        label="Unsure",
        button_type="warning",
        width=120,
        css_classes=["decision-button"],
    )
    for button in (real, bogus, unsure):
        button.disabled = True
    previous_button = Button(
        label="Previous",
        width=120,
        css_classes=["nav-button"],
    )
    next_button = Button(
        label="Next",
        width=120,
        css_classes=["nav-button"],
    )
    real.on_click(lambda: persist("real"))
    bogus.on_click(lambda: persist("bogus"))
    unsure.on_click(lambda: persist("unsure"))
    previous_button.on_click(lambda: move(-1))
    next_button.on_click(lambda: move(1))
    username.on_change("value", lambda _attr, _old, _new: refresh())

    update_url_js = CustomJS(
        args={"url_state": url_state},
        code="""
            if (!url_state.text) {
                return;
            }
            const sample = `${url_state.text}`;
            const url = new URL(window.location.href);
            if (
                url.searchParams.get("s") === sample &&
                !url.searchParams.has("sample_index")
            ) {
                return;
            }
            url.searchParams.set("s", sample);
            url.searchParams.delete("sample_index");
            window.history.replaceState(
                null,
                "",
                `${url.pathname}${url.search}${url.hash}`,
            );
        """,
    )
    url_state.js_on_change("text", update_url_js)

    username_key = f"xscan.review.{review_storage_key}.username"
    username.js_on_change(
        "value",
        CustomJS(
            args={
                "review_buttons": [real, bogus, unsure],
                "storage_key": username_key,
                "username": username,
                "username_required": username_required,
            },
            code="""
                window.localStorage.setItem(
                    storage_key,
                    username.value || "",
                );
                const hasUsername = username.value.trim().length > 0;
                for (const button of review_buttons) {
                    button.disabled = !hasUsername;
                }
                username_required.text = hasUsername
                    ? ""
                    : '<span style="color:#b00020; font-weight:700;">'
                        + 'Mandatory</span>';
            """,
        ),
    )
    doc.js_on_event(
        DocumentReady,
        CustomJS(
            args={
                "url_state": url_state,
                "username": username,
                "username_key": username_key,
            },
            code="""
                const storedUsername =
                    window.localStorage.getItem(username_key);
                if (storedUsername !== null) {
                    username.value = storedUsername;
                    username.change.emit();
                }
                if (url_state.text) {
                    const sample = `${url_state.text}`;
                    const url = new URL(window.location.href);
                    url.searchParams.set("s", sample);
                    url.searchParams.delete("sample_index");
                    window.history.replaceState(
                        null,
                        "",
                        `${url.pathname}${url.search}${url.hash}`,
                    );
                }
            """,
        ),
    )

    search_fig = image_figure("Search", source_a, mapper_a, label_source_a)
    template_fig = image_figure(
        "Template", source_b, mapper_b, label_source_b
    )
    raw_difference_fig = image_figure(
        "Search - Template", source_c, mapper_c, label_source_c
    )
    alard_lupton_fig = image_figure(
        "Alard-Lupton", source_d, mapper_d, label_source_d
    )
    pixel_callback = _pixel_inspector_callback(
        CustomJS,
        labels=["Search", "Template", "Search - Template", "Alard-Lupton"],
        sources=[source_a, source_b, source_c, source_d],
        mappers=[mapper_a, mapper_b, mapper_c, mapper_d],
        label_sources=[
            label_source_a,
            label_source_b,
            label_source_c,
            label_source_d,
        ],
        selection_source=pixel_selection_source,
        stamp_width=stamp_width,
        stamp_height=stamp_height,
    )
    for plot in (
        search_fig,
        template_fig,
        raw_difference_fig,
        alard_lupton_fig,
    ):
        plot.js_on_event(Tap, pixel_callback)
    header = row(
        title,
        row(
            previous_button,
            progress_div,
            next_button,
            width=450,
            css_classes=["review-nav"],
            styles={
                "align-items": "center",
                "background": "#ffffff",
                "border": "1px solid #dde6f4",
                "border-radius": "8px",
                "box-shadow": "0 8px 22px rgba(42,55,105,0.08)",
                "gap": "10px",
                "justify-content": "center",
                "padding": "8px 10px",
            },
        ),
        column(
            username,
            username_required,
            width=420,
            css_classes=["header-tools"],
        ),
        width=1400,
        css_classes=["review-header"],
        styles=dict(_REVIEW_HEADER_STYLE),
    )
    image_card = column(
        image_heading,
        row(
            column(
                search_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            column(
                template_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            column(
                raw_difference_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            column(
                alard_lupton_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            width=1370,
            css_classes=["triptych-row"],
        ),
        width=1400,
        css_classes=["review-card", "image-triptych"],
        styles=dict(_REVIEW_CARD_STYLE),
    )
    classification_card = column(
        Div(
            text=(
                "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
                "margin:0 0 8px;'>Your Classification</h2>"
            ),
            width=1370,
        ),
        Div(text="<strong>Morphology Tags</strong>"),
        tag_control,
        notes,
        row(
            real,
            bogus,
            unsure,
            width=1370,
            css_classes=["decision-row"],
        ),
        status_div,
        width=1400,
        css_classes=["review-card", "classification-card"],
        styles=dict(_REVIEW_CARD_STYLE),
    )
    layout = column(
        header,
        image_card,
        metadata_div,
        classification_card,
        others_div,
        url_state,
        width=1400,
        css_classes=["review-shell"],
        styles={"margin": "18px auto 34px", "gap": "12px"},
    )
    doc.title = "XScan Review"
    doc.add_root(layout)
    refresh()


def build_entity_review_document(doc, *, review_dir: Path) -> None:
    """Populate a Bokeh document for entity classification review."""

    from bokeh.events import DocumentReady, Tap
    from bokeh.layouts import column, row
    from bokeh.models import (
        Button,
        CheckboxGroup,
        ColorBar,
        ColumnDataSource,
        CustomJS,
        Div,
        LabelSet,
        LinearColorMapper,
        Range1d,
        Select,
        Span,
        TextAreaInput,
        TextInput,
    )
    from bokeh.palettes import Greys256, RdBu11
    from bokeh.plotting import figure

    _apply_xscan_review_style(doc)
    manifest, queue = load_review_manifest_and_queue(review_dir)
    dataset_dir = Path(manifest["dataset_dir"]).expanduser().resolve()
    _validate_dataset_fingerprint(manifest, dataset_dir)
    if not queue:
        doc.add_root(Div(text="<h2>No entity review queue items found.</h2>"))
        return

    search = np.load(dataset_dir / "search.npy", mmap_mode="r")
    template = np.load(dataset_dir / "template.npy", mmap_mode="r")
    difference = (
        np.load(dataset_dir / "difference.npy", mmap_mode="r")
        if (dataset_dir / "difference.npy").exists()
        else None
    )
    _validate_queue_sample_indices(queue, sample_count=int(search.shape[0]))
    state_path = review_dir / "review_state.json"
    state = _load_review_state(review_dir)
    fallback_index = min(
        max(int(state.get("current_index", 0)), 0),
        len(queue) - 1,
    )
    initial_index, query_notice = _initial_review_index_from_query(
        doc,
        queue,
        fallback_index=fallback_index,
    )
    index = {"value": initial_index}
    review_storage_key = hashlib.sha256(
        str(review_dir).encode("utf-8")
    ).hexdigest()[:16]

    title = Div(width=470, css_classes=["review-brand"])
    progress_div = Div(
        sizing_mode="stretch_width",
        css_classes=["review-progress-card"],
    )
    metadata_div = Div(
        width=1400,
        css_classes=["summary-wrap"],
        styles={"margin": "0"},
    )
    status_div = Div(sizing_mode="stretch_width", css_classes=["status-card"])
    username = TextInput(
        title="Username:",
        placeholder="Required before saving entity reviews",
        sizing_mode="stretch_width",
        css_classes=["review-username"],
    )
    username_required = Div(
        text=_username_required_html(False),
        name="review-username-required",
        sizing_mode="stretch_width",
        css_classes=["required-hint"],
    )
    show_others = CheckboxGroup(
        labels=["Show Other Entity Reviews"],
        active=[0],
        css_classes=["show-others-toggle"],
    )
    others_div = Div(
        text="",
        width=1400,
        css_classes=["other-reviews-wrap"],
    )
    entity_label = Select(
        title="Entity Label:",
        value="",
        options=["", *ENTITY_REVIEW_LABELS],
        sizing_mode="stretch_width",
        css_classes=["entity-label-select"],
    )
    confidence = Select(
        title="Confidence:",
        value="medium",
        options=list(ENTITY_REVIEW_CONFIDENCE_LEVELS),
        sizing_mode="stretch_width",
        css_classes=["confidence-select"],
    )
    notes = TextAreaInput(
        title="Notes",
        rows=4,
        sizing_mode="stretch_width",
        css_classes=["notes-input"],
    )
    url_state = Div(
        text=str(int(queue[initial_index]["sample_index"])),
        visible=False,
        name="review-url-state",
    )
    source_a = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    source_b = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    source_c = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    source_d = ColumnDataSource(_image_payload(np.zeros(search.shape[1:])))
    label_source_a = _pixel_label_source(ColumnDataSource)
    label_source_b = _pixel_label_source(ColumnDataSource)
    label_source_c = _pixel_label_source(ColumnDataSource)
    label_source_d = _pixel_label_source(ColumnDataSource)
    pixel_selection_source = ColumnDataSource(
        {"x": [], "y": [], "width": [], "height": []}
    )
    mapper_a = LinearColorMapper(palette=list(Greys256), low=0.0, high=1.0)
    mapper_b = LinearColorMapper(palette=list(Greys256), low=0.0, high=1.0)
    mapper_c = LinearColorMapper(
        palette=list(reversed(RdBu11)), low=-1.0, high=1.0
    )
    mapper_d = LinearColorMapper(
        palette=list(reversed(RdBu11)), low=-1.0, high=1.0
    )
    stamp_height, stamp_width = search.shape[1:]
    shared_x_range = Range1d(0, stamp_width)
    shared_y_range = Range1d(0, stamp_height)

    def image_figure(
        label: str,
        source: ColumnDataSource,
        mapper,
        label_source: ColumnDataSource,
    ):
        fig = figure(
            title=label,
            width=320,
            height=320,
            match_aspect=True,
            x_range=shared_x_range,
            y_range=shared_y_range,
            tools="tap,pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
        )
        fig.image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=source,
            color_mapper=mapper,
        )
        fig.add_layout(
            ColorBar(
                color_mapper=mapper,
                width=10,
                label_standoff=5,
                border_line_color=None,
                background_fill_alpha=0.0,
                major_label_text_font_size="10px",
            ),
            "right",
        )
        fig.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=pixel_selection_source,
            fill_alpha=0.0,
            line_color="#ffd400",
            line_width=2,
        )
        fig.add_layout(_pixel_label_set(LabelSet, label_source))
        fig.add_layout(
            Span(
                location=stamp_width / 2.0,
                dimension="height",
                line_color="#03c7b7",
                line_alpha=0.75,
                line_width=1,
            )
        )
        fig.add_layout(
            Span(
                location=stamp_height / 2.0,
                dimension="width",
                line_color="#03c7b7",
                line_alpha=0.75,
                line_width=1,
            )
        )
        fig.grid.visible = False
        fig.axis.visible = False
        fig.outline_line_color = "#d9e2f0"
        fig.border_fill_color = "#ffffff"
        fig.background_fill_color = "#ffffff"
        fig.title.text_color = "#07145c"
        fig.title.text_font_size = "13px"
        fig.title.text_font_style = "bold"
        fig.toolbar.logo = None
        return fig

    def current_item() -> dict[str, Any]:
        return queue[index["value"]]

    def refresh() -> None:
        item = current_item()
        sample_index = int(item["sample_index"])
        url_state.text = str(sample_index)
        raw_diff_image = search[sample_index] - template[sample_index]
        alard_lupton_image = (
            difference[sample_index]
            if difference is not None
            else raw_diff_image
        )
        _update_image(
            source_a, mapper_a, search[sample_index], diverging=False
        )
        _update_image(
            source_b, mapper_b, template[sample_index], diverging=False
        )
        _update_image(source_c, mapper_c, raw_diff_image, diverging=True)
        _update_image(source_d, mapper_d, alard_lupton_image, diverging=True)
        latest_by_reviewer = latest_entity_annotations_by_reviewer(review_dir)
        queue_id = str(item["queue_id"])
        reviewer = _reviewer_name(username.value)
        has_reviewer = bool(reviewer)
        save_button.disabled = not has_reviewer
        username_required.text = _username_required_html(has_reviewer)
        annotation = (
            latest_by_reviewer.get(queue_id, {}).get(reviewer)
            if has_reviewer
            else None
        )
        entity_label.value = ""
        confidence.value = "medium"
        notes.value = ""
        if annotation is not None:
            saved_label = str(annotation.get("entity_label", ""))
            if saved_label in ENTITY_REVIEW_LABELS:
                entity_label.value = saved_label
            saved_confidence = str(annotation.get("confidence", ""))
            if saved_confidence in ENTITY_REVIEW_CONFIDENCE_LEVELS:
                confidence.value = saved_confidence
            notes.value = str(annotation.get("notes", ""))
        others_div.text = (
            _entity_other_reviews_html(
                item,
                latest_by_reviewer,
                reviewer=reviewer,
            )
            if show_others.active
            else ""
        )
        title.text = _review_header_html(
            title="XScan Entity Review",
            subtitle="Review astronomical detections with human insight",
            id_label="Entity ID",
            item=item,
        )
        progress_div.text = _review_progress_html(
            index["value"] + 1,
            len(queue),
        )
        metadata_div.text = _metadata_table(item)
        reviewed_by_user = _reviewer_reviewed_count(
            latest_by_reviewer,
            reviewer,
        )
        total_decisions = sum(
            len(payloads) for payloads in latest_by_reviewer.values()
        )
        previous = (
            f"Last label: {annotation.get('entity_label')}"
            if annotation is not None
            else ""
        )
        status_div.text = _review_status_html(
            notice=query_notice,
            reviewer=reviewer,
            reviewed_by_user=reviewed_by_user,
            total_items=len(queue),
            total_decisions=total_decisions,
            previous=previous,
        )
        state = _load_review_state(review_dir)
        state["current_index"] = index["value"]
        state["updated_at_utc"] = _utc_now()
        state_path.write_text(_json_dumps(state) + "\n", encoding="utf-8")

    def persist() -> None:
        item = current_item()
        reviewer = _reviewer_name(username.value)
        if not reviewer:
            status_div.text = (
                "<p><strong>Set Username before saving a review.</strong></p>"
                + status_div.text
            )
            return
        selected_label = str(entity_label.value).strip().lower()
        if selected_label not in ENTITY_REVIEW_LABELS:
            status_div.text = (
                "<p><strong>Select an entity label before saving."
                "</strong></p>" + status_div.text
            )
            return
        selected_confidence = str(confidence.value).strip().lower()
        if selected_confidence not in ENTITY_REVIEW_CONFIDENCE_LEVELS:
            selected_confidence = "medium"
        payload = {
            "timestamp_utc": _utc_now(),
            "reviewer": reviewer,
            "queue_id": item["queue_id"],
            "sample_index": int(item["sample_index"]),
            "candidate_id": item.get("candidate_id"),
            "entity_label": selected_label,
            "confidence": selected_confidence,
            "notes": notes.value,
            "source_review_dir": item.get("source_review_dir"),
            "source_queue_id": item.get("source_queue_id"),
            "source_sample_index": item.get("source_sample_index"),
            "source_candidate_id": item.get("source_candidate_id"),
            "source_reviewer": item.get("source_reviewer"),
            "source_binary_label": item.get("source_binary_label"),
            "source_binary_morphology_tags": item.get(
                "source_binary_morphology_tags", []
            ),
            "source_binary_notes": item.get("source_binary_notes", ""),
            "source_binary_timestamp_utc": item.get(
                "source_binary_timestamp_utc"
            ),
            "source_label": item.get("source_label"),
            "source_probability": item.get("source_probability"),
            "source_rank_reason": item.get("source_rank_reason"),
        }
        _append_entity_annotation(review_dir, payload)
        if index["value"] < len(queue) - 1:
            index["value"] += 1
        refresh()

    def move(delta: int) -> None:
        index["value"] = min(max(index["value"] + delta, 0), len(queue) - 1)
        refresh()

    save_button = Button(
        label="Save Entity Label",
        button_type="primary",
        css_classes=["decision-button"],
    )
    save_button.disabled = True
    previous_button = Button(
        label="Previous",
        width=120,
        css_classes=["nav-button"],
    )
    next_button = Button(
        label="Next",
        width=120,
        css_classes=["nav-button"],
    )
    save_button.on_click(persist)
    previous_button.on_click(lambda: move(-1))
    next_button.on_click(lambda: move(1))
    username.on_change("value", lambda _attr, _old, _new: refresh())
    show_others.on_change("active", lambda _attr, _old, _new: refresh())

    update_url_js = CustomJS(
        args={"url_state": url_state},
        code="""
            if (!url_state.text) {
                return;
            }
            const sample = `${url_state.text}`;
            const url = new URL(window.location.href);
            if (
                url.searchParams.get("s") === sample &&
                !url.searchParams.has("sample_index")
            ) {
                return;
            }
            url.searchParams.set("s", sample);
            url.searchParams.delete("sample_index");
            window.history.replaceState(
                null,
                "",
                `${url.pathname}${url.search}${url.hash}`,
            );
        """,
    )
    url_state.js_on_change("text", update_url_js)

    username_key = f"xscan.entity_review.{review_storage_key}.username"
    show_others_key = f"xscan.entity_review.{review_storage_key}.show_others"
    username.js_on_change(
        "value",
        CustomJS(
            args={
                "save_button": save_button,
                "storage_key": username_key,
                "username": username,
                "username_required": username_required,
            },
            code="""
                window.localStorage.setItem(
                    storage_key,
                    username.value || "",
                );
                const hasUsername = username.value.trim().length > 0;
                save_button.disabled = !hasUsername;
                username_required.text = hasUsername
                    ? ""
                    : '<span style="color:#b00020; font-weight:700;">'
                        + 'Mandatory</span>';
            """,
        ),
    )
    show_others.js_on_change(
        "active",
        CustomJS(
            args={
                "show_others": show_others,
                "storage_key": show_others_key,
            },
            code="""
                window.localStorage.setItem(
                    storage_key,
                    show_others.active.length ? "1" : "0",
                );
            """,
        ),
    )
    doc.js_on_event(
        DocumentReady,
        CustomJS(
            args={
                "url_state": url_state,
                "username": username,
                "show_others": show_others,
                "username_key": username_key,
                "show_others_key": show_others_key,
            },
            code="""
                const storedUsername =
                    window.localStorage.getItem(username_key);
                if (storedUsername !== null) {
                    username.value = storedUsername;
                    username.change.emit();
                }
                const storedShowOthers =
                    window.localStorage.getItem(show_others_key);
                if (storedShowOthers === "0") {
                    show_others.active = [];
                    show_others.change.emit();
                } else if (storedShowOthers === "1") {
                    show_others.active = [0];
                    show_others.change.emit();
                }
                if (url_state.text) {
                    const sample = `${url_state.text}`;
                    const url = new URL(window.location.href);
                    url.searchParams.set("s", sample);
                    url.searchParams.delete("sample_index");
                    window.history.replaceState(
                        null,
                        "",
                        `${url.pathname}${url.search}${url.hash}`,
                    );
                }
            """,
        ),
    )

    search_fig = image_figure("Search", source_a, mapper_a, label_source_a)
    template_fig = image_figure(
        "Template", source_b, mapper_b, label_source_b
    )
    raw_difference_fig = image_figure(
        "Search - Template", source_c, mapper_c, label_source_c
    )
    alard_lupton_fig = image_figure(
        "Alard-Lupton", source_d, mapper_d, label_source_d
    )
    pixel_callback = _pixel_inspector_callback(
        CustomJS,
        labels=["Search", "Template", "Search - Template", "Alard-Lupton"],
        sources=[source_a, source_b, source_c, source_d],
        mappers=[mapper_a, mapper_b, mapper_c, mapper_d],
        label_sources=[
            label_source_a,
            label_source_b,
            label_source_c,
            label_source_d,
        ],
        selection_source=pixel_selection_source,
        stamp_width=stamp_width,
        stamp_height=stamp_height,
    )
    for plot in (
        search_fig,
        template_fig,
        raw_difference_fig,
        alard_lupton_fig,
    ):
        plot.js_on_event(Tap, pixel_callback)
    header = row(
        title,
        row(
            previous_button,
            progress_div,
            next_button,
            width=450,
            css_classes=["review-nav"],
            styles={
                "align-items": "center",
                "background": "#ffffff",
                "border": "1px solid #dde6f4",
                "border-radius": "8px",
                "box-shadow": "0 8px 22px rgba(42,55,105,0.08)",
                "gap": "10px",
                "justify-content": "center",
                "padding": "8px 10px",
            },
        ),
        column(
            show_others,
            username,
            username_required,
            width=420,
            css_classes=["header-tools"],
        ),
        width=1400,
        css_classes=["review-header"],
        styles=dict(_REVIEW_HEADER_STYLE),
    )
    image_card = column(
        Div(
            text=(
                "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
                "margin:0 0 8px;'>Image Review</h2>"
            ),
            width=1370,
        ),
        row(
            column(
                search_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            column(
                template_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            column(
                raw_difference_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            column(
                alard_lupton_fig,
                width=334,
                css_classes=["stamp-card"],
                styles=dict(_STAMP_CARD_STYLE),
            ),
            width=1370,
            css_classes=["triptych-row"],
        ),
        width=1400,
        css_classes=["review-card", "image-triptych"],
        styles=dict(_REVIEW_CARD_STYLE),
    )
    classification_card = column(
        Div(
            text=(
                "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
                "margin:0 0 8px;'>Your Classification</h2>"
            ),
            width=1370,
        ),
        row(
            entity_label,
            confidence,
            width=1370,
            css_classes=["field-grid"],
        ),
        notes,
        row(
            save_button,
            width=1370,
            css_classes=["decision-row"],
        ),
        status_div,
        width=1400,
        css_classes=["review-card", "classification-card"],
        styles=dict(_REVIEW_CARD_STYLE),
    )
    layout = column(
        header,
        image_card,
        metadata_div,
        classification_card,
        others_div,
        url_state,
        width=1400,
        css_classes=["review-shell"],
        styles={"margin": "18px auto 34px", "gap": "12px"},
    )
    doc.title = "XScan Entity Review"
    doc.add_root(layout)
    refresh()


def load_review_manifest_and_queue(
    review_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    review_dir = review_dir.expanduser().resolve()
    manifest = json.loads((review_dir / "manifest.json").read_text())
    queue = [
        json.loads(line)
        for line in (review_dir / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return manifest, queue


def export_review_contact_sheets(
    *,
    review_dir: Path,
    output_dir: Path,
    max_items: int = 64,
    items_per_page: int = 16,
    columns: int = 4,
    stamp_size: int = 96,
    overwrite: bool = False,
) -> ReviewContactSheetResult:
    """Export static PNG contact sheets for a saved review queue."""

    review_dir = review_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if items_per_page <= 0:
        raise ValueError("items_per_page must be positive")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if stamp_size <= 0:
        raise ValueError("stamp_size must be positive")

    manifest, queue = load_review_manifest_and_queue(review_dir)
    dataset_dir = Path(manifest["dataset_dir"]).expanduser().resolve()
    _validate_dataset_fingerprint(manifest, dataset_dir)
    search = np.load(dataset_dir / "search.npy", mmap_mode="r")
    template = np.load(dataset_dir / "template.npy", mmap_mode="r")
    difference = (
        np.load(dataset_dir / "difference.npy", mmap_mode="r")
        if (dataset_dir / "difference.npy").exists()
        else None
    )
    _validate_queue_sample_indices(queue, sample_count=int(search.shape[0]))

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"contact-sheet output_dir already exists and is not empty: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_dir.glob("contact-sheet-*.png"):
            path.unlink()
        index_path = output_dir / "index.json"
        if index_path.exists():
            index_path.unlink()

    selected = queue[: int(max_items)]
    pages: list[dict[str, Any]] = []
    for page_index, start in enumerate(
        range(0, len(selected), int(items_per_page)),
        start=1,
    ):
        page_items = selected[start : start + int(items_per_page)]
        page_name = f"contact-sheet-{page_index:03d}.png"
        page_path = output_dir / page_name
        _render_review_contact_sheet_page(
            page_path=page_path,
            page_index=page_index,
            page_items=page_items,
            search=search,
            template=template,
            difference=difference,
            columns=int(columns),
            stamp_size=int(stamp_size),
        )
        pages.append(
            {
                "path": page_name,
                "start_index": start,
                "item_count": len(page_items),
            }
        )

    summary = {
        "workflow": "review-contact-sheet",
        "review_dir": str(review_dir),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "queue_count": len(queue),
        "exported_count": len(selected),
        "page_count": len(pages),
        "max_items": int(max_items),
        "items_per_page": int(items_per_page),
        "columns": int(columns),
        "stamp_size": int(stamp_size),
        "saved": {
            "index": "index.json",
            "pages": [page["path"] for page in pages],
        },
        "pages": pages,
        "items": [_contact_sheet_item_summary(item) for item in selected],
    }
    (output_dir / "index.json").write_text(
        _json_dumps(summary) + "\n",
        encoding="utf-8",
    )
    return ReviewContactSheetResult(output_dir=output_dir, summary=summary)


_REVIEW_ANNOTATION_TEMPLATE_FIELDS = (
    "queue_id",
    "sample_index",
    "candidate_id",
    "rank",
    "reviewer",
    "reviewer_label",
    "morphology_tags",
    "notes",
    "source_label",
    "source_probability",
    "label_source",
    "rank_reason",
)


def export_review_annotation_template(
    *,
    review_dir: Path,
    output_csv: Path,
    reviewer: str | None = None,
    overwrite: bool = False,
) -> ReviewAnnotationTemplateResult:
    """Write a CSV template for offline review annotations."""

    review_dir = review_dir.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    reviewer_name = _reviewer_name(reviewer)
    manifest, queue = load_review_manifest_and_queue(review_dir)
    dataset_dir = Path(manifest["dataset_dir"]).expanduser().resolve()
    _validate_dataset_fingerprint(manifest, dataset_dir)
    labels = np.load(dataset_dir / "labels.npy", mmap_mode="r")
    _validate_queue_sample_indices(queue, sample_count=int(labels.shape[0]))
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"annotation template already exists: {output_csv}"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(_REVIEW_ANNOTATION_TEMPLATE_FIELDS),
        )
        writer.writeheader()
        for item in queue:
            writer.writerow(
                {
                    "queue_id": item.get("queue_id"),
                    "sample_index": item.get("sample_index"),
                    "candidate_id": item.get("candidate_id"),
                    "rank": item.get("rank"),
                    "reviewer": reviewer_name,
                    "reviewer_label": "",
                    "morphology_tags": "",
                    "notes": "",
                    "source_label": item.get("label"),
                    "source_probability": item.get("probability"),
                    "label_source": item.get("label_source"),
                    "rank_reason": item.get("rank_reason"),
                }
            )
    summary = {
        "workflow": "review-annotation-template",
        "review_dir": str(review_dir),
        "dataset_dir": str(dataset_dir),
        "output_csv": str(output_csv),
        "queue_count": len(queue),
        "reviewer": reviewer_name or None,
        "saved": {"template_csv": str(output_csv)},
        "reviewer_labels": ["real", "bogus", "unsure"],
        "morphology_tags": list(MORPHOLOGY_TAGS),
    }
    return ReviewAnnotationTemplateResult(
        output_csv=output_csv,
        summary=summary,
    )


def import_review_annotations_from_csv(
    *,
    review_dir: Path,
    input_csv: Path,
    reviewer: str | None = None,
    dry_run: bool = False,
    require_all: bool = False,
) -> ReviewAnnotationImportResult:
    """Validate and append offline review annotations from a CSV file."""

    review_dir = review_dir.expanduser().resolve()
    input_csv = input_csv.expanduser().resolve()
    reviewer_override = _reviewer_name(reviewer)
    manifest, queue = load_review_manifest_and_queue(review_dir)
    dataset_dir = Path(manifest["dataset_dir"]).expanduser().resolve()
    _validate_dataset_fingerprint(manifest, dataset_dir)
    labels = np.load(dataset_dir / "labels.npy", mmap_mode="r")
    _validate_queue_sample_indices(queue, sample_count=int(labels.shape[0]))
    queue_by_id = {
        str(item.get("queue_id") or item.get("sample_index")): item
        for item in queue
    }
    rows = _read_review_annotation_csv(input_csv)
    timestamp = _utc_now()
    annotations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped_blank_label_count = 0
    for line_number, row in rows:
        label = str(row.get("reviewer_label", "")).strip().lower()
        if not label:
            skipped_blank_label_count += 1
            continue
        if label not in {"real", "bogus", "unsure"}:
            raise ValueError(
                "reviewer_label must be one of real, bogus, unsure "
                f"({input_csv}:{line_number})"
            )
        queue_id = str(row.get("queue_id", "")).strip()
        if not queue_id:
            raise ValueError(
                f"queue_id is required ({input_csv}:{line_number})"
            )
        item = queue_by_id.get(queue_id)
        if item is None:
            raise ValueError(
                f"queue_id is not present in queue.jsonl "
                f"({input_csv}:{line_number}, queue_id={queue_id})"
            )
        reviewer_name = reviewer_override or _reviewer_name(
            row.get("reviewer")
        )
        if not reviewer_name:
            raise ValueError(
                f"reviewer is required ({input_csv}:{line_number}, "
                f"queue_id={queue_id})"
            )
        key = (queue_id, reviewer_name)
        if key in seen:
            raise ValueError(
                "input CSV contains duplicate rows for "
                f"queue_id={queue_id}, reviewer={reviewer_name}"
            )
        seen.add(key)
        payload = {
            "timestamp_utc": timestamp,
            "queue_id": queue_id,
            "sample_index": _coerce_int(row.get("sample_index")),
            "candidate_id": row.get("candidate_id")
            or item.get("candidate_id"),
            "reviewer": reviewer_name,
            "reviewer_label": label,
            "morphology_tags": _parse_review_morphology_tags(
                row.get("morphology_tags", "")
            ),
            "notes": str(row.get("notes", "")),
            "source_label": _coerce_int(
                row.get("source_label", item.get("label"))
            ),
            "source_probability": _coerce_float(
                row.get("source_probability", item.get("probability"))
            ),
        }
        _validate_annotation_identity(
            item,
            payload,
            reviewer=reviewer_name,
            annotation_kind="review annotation import",
        )
        annotations.append(payload)

    if require_all:
        annotated_queue_ids = {row["queue_id"] for row in annotations}
        missing = sorted(set(queue_by_id) - annotated_queue_ids)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                "review annotation import is missing queued samples: "
                f"{preview}{suffix}"
            )
    if not dry_run:
        for payload in annotations:
            _append_annotation(review_dir, payload)

    label_counts: dict[str, int] = {}
    reviewers = set()
    for payload in annotations:
        label_counts[payload["reviewer_label"]] = (
            label_counts.get(payload["reviewer_label"], 0) + 1
        )
        reviewers.add(payload["reviewer"])
    summary = {
        "workflow": "review-import-annotations",
        "review_dir": str(review_dir),
        "dataset_dir": str(dataset_dir),
        "input_csv": str(input_csv),
        "dry_run": bool(dry_run),
        "row_count": len(rows),
        "validated_count": len(annotations),
        "appended_count": 0 if dry_run else len(annotations),
        "skipped_blank_label_count": skipped_blank_label_count,
        "reviewer_count": len(reviewers),
        "reviewers": sorted(reviewers),
        "reviewer_label_counts": label_counts,
        "saved": {"annotations": "annotations.jsonl"},
    }
    return ReviewAnnotationImportResult(
        review_dir=review_dir,
        summary=summary,
    )


def latest_annotation_per_queue(
    review_dir: Path,
    *,
    actionable_only: bool = False,
) -> dict[str, dict[str, Any]]:
    path = review_dir / "annotations.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("review annotations must be JSON objects")
        if actionable_only:
            label_text = (
                str(payload.get("reviewer_label", "")).strip().lower()
            )
            if label_text not in REVIEW_LABEL_TO_VALUE:
                continue
        key = str(payload.get("queue_id") or payload.get("sample_index"))
        latest[key] = payload
    return latest


def latest_annotations_by_reviewer(
    review_dir: Path,
    *,
    actionable_only: bool = False,
    strict: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    path = review_dir / "annotations.jsonl"
    latest: dict[str, dict[str, dict[str, Any]]] = {}
    if not path.exists():
        return latest
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"review annotations must be JSON objects "
                f"({_annotation_context(path, line_number, {})})"
            )
        payload = _with_annotation_source(payload, path, line_number)
        if strict:
            reviewer = _require_annotation_reviewer(
                payload,
                path=path,
                line_number=line_number,
                kind="review annotations",
            )
            _require_annotation_timestamp(
                payload,
                path=path,
                line_number=line_number,
                kind="review annotations",
            )
        else:
            reviewer = _reviewer_name(payload.get("reviewer"))
            if not reviewer:
                # Legacy review files lacked reviewer provenance; non-strict
                # UI/state bookkeeping keeps rows distinct without inventing
                # reviewer identity.
                path_digest = hashlib.sha256(
                    str(path).encode("utf-8")
                ).hexdigest()[:8]
                reviewer = f"anonymous:{path_digest}:{line_number}"
        if actionable_only:
            label_text = (
                str(payload.get("reviewer_label", "")).strip().lower()
            )
            if label_text not in REVIEW_LABEL_TO_VALUE:
                continue
        key = str(payload.get("queue_id") or payload.get("sample_index"))
        latest.setdefault(key, {})[reviewer] = payload
    return latest


def latest_entity_annotation_per_queue(
    review_dir: Path,
) -> dict[str, dict[str, Any]]:
    path = review_dir / "entity_annotations.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("entity annotations must be JSON objects")
        key = str(payload.get("queue_id") or payload.get("sample_index"))
        latest[key] = payload
    return latest


def latest_entity_annotations_by_reviewer(
    review_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    path = review_dir / "entity_annotations.jsonl"
    latest: dict[str, dict[str, dict[str, Any]]] = {}
    if not path.exists():
        return latest
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"entity annotations must be JSON objects "
                f"({_annotation_context(path, line_number, {})})"
            )
        payload = _with_annotation_source(payload, path, line_number)
        reviewer = _require_annotation_reviewer(
            payload,
            path=path,
            line_number=line_number,
            kind="entity annotations",
        )
        _require_annotation_timestamp(
            payload,
            path=path,
            line_number=line_number,
            kind="entity annotations",
        )
        key = str(payload.get("queue_id") or payload.get("sample_index"))
        latest.setdefault(key, {})[reviewer] = payload
    return latest


class _AnnotationPayload(dict):
    """Dict payload with runtime-only source location attributes."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        source_path: str,
        source_line: int,
    ) -> None:
        super().__init__(payload)
        self.source_path = source_path
        self.source_line = source_line


def _with_annotation_source(
    payload: dict[str, Any],
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    return _AnnotationPayload(
        payload,
        source_path=str(path),
        source_line=int(line_number),
    )


def _annotation_context(
    path: Path | None,
    line_number: int | None,
    payload: dict[str, Any],
) -> str:
    parts = []
    source_path = path or getattr(payload, "source_path", None)
    source_line = line_number or getattr(payload, "source_line", None)
    if source_path is not None:
        location = str(source_path)
        if source_line is not None:
            location = f"{location}:{source_line}"
        parts.append(location)
    queue_id = payload.get("queue_id")
    if queue_id is not None:
        parts.append(f"queue_id={queue_id}")
    sample_index = payload.get("sample_index")
    if sample_index is not None:
        parts.append(f"sample_index={sample_index}")
    return ", ".join(parts) if parts else "unknown annotation"


def _require_annotation_reviewer(
    payload: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    kind: str,
) -> str:
    reviewer = _reviewer_name(payload.get("reviewer"))
    if not reviewer:
        raise ValueError(
            f"{kind} must include reviewer "
            f"({_annotation_context(path, line_number, payload)}); "
            "legacy review files must be re-recorded or migrated with "
            "reviewer and timestamp_utc before review-aggregate or "
            "review-apply"
        )
    return reviewer


def _require_annotation_timestamp(
    payload: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    kind: str,
) -> str:
    timestamp = str(payload.get("timestamp_utc", "")).strip()
    if not timestamp:
        raise ValueError(
            f"{kind} must include timestamp_utc "
            f"({_annotation_context(path, line_number, payload)}); "
            "legacy review files must be re-recorded or migrated with "
            "reviewer and timestamp_utc before review-aggregate or "
            "review-apply"
        )
    try:
        _parse_review_timestamp(timestamp)
    except ValueError as exc:
        raise ValueError(
            f"{kind} timestamp_utc must be parseable as ISO-8601 "
            f"({_annotation_context(path, line_number, payload)}): "
            f"{timestamp!r}"
        ) from exc
    return timestamp


def _aggregate_review_item(
    item: dict[str, Any],
    latest_by_reviewer: dict[str, dict[str, Any]],
    *,
    min_reviewers: int,
    min_actionable_reviewers: int,
    consensus_rule: str,
) -> dict[str, Any]:
    queue_id = _review_item_key(item)
    latest_records = []
    label_counts = {"real": 0, "bogus": 0, "unsure": 0, "invalid": 0}
    actionable_counts = {label: 0 for label in sorted(REVIEW_LABEL_TO_VALUE)}
    supporting_tags: set[str] = set()
    supporting_reviewers: list[str] = []
    ignored_unsure_reviewers: list[str] = []

    for reviewer, payload in sorted(latest_by_reviewer.items()):
        _validate_annotation_identity(
            item,
            payload,
            reviewer=reviewer,
            annotation_kind="review annotation",
        )
        label_text = str(payload.get("reviewer_label", "")).strip().lower()
        if label_text in REVIEW_LABEL_TO_VALUE:
            actionable_counts[label_text] += 1
            label_counts[label_text] += 1
        elif label_text == "unsure":
            label_counts["unsure"] += 1
            ignored_unsure_reviewers.append(reviewer)
        else:
            label_counts["invalid"] += 1
        latest_records.append(
            _review_annotation_provenance(
                reviewer=reviewer,
                payload=payload,
                reviewer_label=label_text,
            )
        )

    reviewer_count = len(latest_by_reviewer)
    valid_reviewer_count = reviewer_count - label_counts["invalid"]
    actionable_reviewer_count = sum(actionable_counts.values())
    status = "insufficient_review"
    reason = (
        "not_enough_valid_reviewers"
        if label_counts["invalid"]
        else "not_enough_reviewers"
    )
    consensus_label = None
    if valid_reviewer_count >= min_reviewers:
        if actionable_reviewer_count == 0:
            if label_counts["unsure"] == valid_reviewer_count:
                status = "unsure_only"
                reason = "all_latest_reviews_are_unsure"
            else:
                status = "no_actionable"
                if label_counts["invalid"] == reviewer_count:
                    reason = "all_latest_reviews_have_unrecognized_labels"
                else:
                    reason = "no_actionable_latest_reviews"
        elif actionable_reviewer_count < min_actionable_reviewers:
            reason = "not_enough_actionable_reviewers"
        else:
            consensus_label = _resolve_consensus_label(
                actionable_counts,
                consensus_rule=consensus_rule,
            )
            if consensus_label is None:
                status = "conflicted"
                reason = f"{consensus_rule}_consensus_not_met"
            else:
                status = "actionable"
                reason = f"{consensus_rule}_consensus"

    if consensus_label is not None:
        for reviewer, payload in sorted(latest_by_reviewer.items()):
            label_text = (
                str(payload.get("reviewer_label", "")).strip().lower()
            )
            if label_text != consensus_label:
                continue
            supporting_reviewers.append(reviewer)
            for tag in payload.get("morphology_tags") or []:
                supporting_tags.add(str(tag))

    sample_index = _coerce_int(item.get("sample_index"))
    decision = {
        "queue_id": queue_id,
        "sample_index": sample_index,
        "candidate_id": item.get("candidate_id"),
        "status": status,
        "reason": reason,
        "consensus_label": consensus_label,
        "reviewer_count": reviewer_count,
        "valid_reviewer_count": valid_reviewer_count,
        "actionable_reviewer_count": actionable_reviewer_count,
        "label_counts": label_counts,
        "actionable_label_counts": actionable_counts,
        "supporting_reviewers": supporting_reviewers,
        "ignored_unsure_reviewers": ignored_unsure_reviewers,
        "consensus_morphology_tags": sorted(supporting_tags),
        "latest_annotations": latest_records,
        "aggregation": {
            "rule_version": REVIEW_AGGREGATION_RULE_VERSION,
            "latest_per_reviewer_rule": REVIEW_LATEST_PER_REVIEWER_RULE,
            "consensus_rule": consensus_rule,
            "min_reviewers": int(min_reviewers),
            "min_actionable_reviewers": int(min_actionable_reviewers),
        },
    }
    return _jsonable(
        {key: value for key, value in decision.items() if value is not None}
    )


def _aggregate_entity_review_item(
    item: dict[str, Any],
    latest_by_reviewer: dict[str, dict[str, Any]],
    *,
    min_reviewers: int,
    consensus_rule: str,
) -> dict[str, Any]:
    queue_id = _review_item_key(item)
    latest_records = []
    label_counts = {label: 0 for label in ENTITY_REVIEW_LABELS}
    label_counts["invalid"] = 0
    confidence_counts = {
        confidence: 0 for confidence in ENTITY_REVIEW_CONFIDENCE_LEVELS
    }
    confidence_counts["invalid"] = 0
    supporting_reviewers: list[str] = []

    for reviewer, payload in sorted(latest_by_reviewer.items()):
        _validate_annotation_identity(
            item,
            payload,
            reviewer=reviewer,
            annotation_kind="entity annotation",
        )
        entity_label = str(payload.get("entity_label", "")).strip().lower()
        confidence = str(payload.get("confidence", "")).strip().lower()
        if entity_label in ENTITY_REVIEW_LABELS:
            label_counts[entity_label] += 1
        else:
            label_counts["invalid"] += 1
        if confidence in ENTITY_REVIEW_CONFIDENCE_LEVELS:
            confidence_counts[confidence] += 1
        else:
            confidence_counts["invalid"] += 1
        latest_records.append(
            _entity_annotation_provenance(
                reviewer=reviewer,
                payload=payload,
                entity_label=entity_label,
                confidence=confidence,
            )
        )

    reviewer_count = len(latest_by_reviewer)
    # other_or_unsure is a valid taxonomy choice here; it resolves to a
    # non-actionable status rather than being dropped as an invalid vote.
    valid_reviewer_count = sum(
        label_counts[label] for label in ENTITY_REVIEW_LABELS
    )
    status = "insufficient_review"
    reason = "not_enough_reviewers"
    consensus_label = None
    if reviewer_count >= min_reviewers:
        if valid_reviewer_count < min_reviewers:
            reason = "not_enough_valid_entity_labels"
        else:
            entity_counts = {
                label: label_counts[label] for label in ENTITY_REVIEW_LABELS
            }
            consensus_label = _resolve_consensus_label(
                entity_counts,
                consensus_rule=consensus_rule,
            )
            if consensus_label is None:
                status = "conflicted"
                reason = f"{consensus_rule}_consensus_not_met"
            elif consensus_label == "other_or_unsure":
                status = "other_or_unsure"
                reason = f"{consensus_rule}_consensus_other_or_unsure"
            else:
                status = "actionable"
                reason = f"{consensus_rule}_consensus"

    if consensus_label is not None:
        for reviewer, payload in sorted(latest_by_reviewer.items()):
            entity_label = (
                str(payload.get("entity_label", "")).strip().lower()
            )
            if entity_label == consensus_label:
                supporting_reviewers.append(reviewer)

    sample_index = _coerce_int(item.get("sample_index"))
    decision = {
        "queue_id": queue_id,
        "sample_index": sample_index,
        "candidate_id": item.get("candidate_id"),
        "source_review_dir": item.get("source_review_dir"),
        "source_queue_id": item.get("source_queue_id"),
        "source_sample_index": _coerce_int(item.get("source_sample_index")),
        "source_reviewer": item.get("source_reviewer"),
        "source_binary_label": item.get("source_binary_label"),
        "source_binary_morphology_tags": item.get(
            "source_binary_morphology_tags", []
        ),
        "status": status,
        "reason": reason,
        "resolved_entity_label": consensus_label,
        "consensus_entity_label": (
            consensus_label if status == "actionable" else None
        ),
        "reviewer_count": reviewer_count,
        "valid_reviewer_count": valid_reviewer_count,
        "label_counts": label_counts,
        "confidence_counts": confidence_counts,
        "supporting_reviewers": supporting_reviewers,
        "latest_annotations": latest_records,
        "aggregation": {
            "rule_version": ENTITY_REVIEW_AGGREGATION_RULE_VERSION,
            "latest_per_reviewer_rule": REVIEW_LATEST_PER_REVIEWER_RULE,
            "consensus_rule": consensus_rule,
            "min_reviewers": int(min_reviewers),
        },
    }
    return _jsonable(
        {key: value for key, value in decision.items() if value is not None}
    )


def _resolve_consensus_label(
    actionable_counts: dict[str, int],
    *,
    consensus_rule: str,
) -> str | None:
    nonzero = {
        label: count
        for label, count in actionable_counts.items()
        if count > 0
    }
    if not nonzero:
        return None
    if consensus_rule == "unanimous":
        if len(nonzero) == 1:
            return next(iter(nonzero))
        return None
    ordered = sorted(nonzero.items(), key=lambda item: (-item[1], item[0]))
    total_votes = sum(nonzero.values())
    # Strict majority is intentional: exact ties remain conflicted.
    if ordered[0][1] > (total_votes / 2):
        return ordered[0][0]
    return None


def _review_annotation_provenance(
    *,
    reviewer: str,
    payload: dict[str, Any],
    reviewer_label: str,
) -> dict[str, Any]:
    record = {
        "reviewer": reviewer,
        "reviewer_label": reviewer_label,
        "timestamp_utc": payload.get("timestamp_utc"),
        "queue_id": payload.get("queue_id"),
        "sample_index": _coerce_int(payload.get("sample_index")),
        "candidate_id": payload.get("candidate_id"),
        "morphology_tags": payload.get("morphology_tags", []),
        "notes": payload.get("notes", ""),
        "source_label": payload.get("source_label"),
        "source_probability": payload.get("source_probability"),
    }
    return _jsonable(
        {key: value for key, value in record.items() if value is not None}
    )


def _entity_annotation_provenance(
    *,
    reviewer: str,
    payload: dict[str, Any],
    entity_label: str,
    confidence: str,
) -> dict[str, Any]:
    record = {
        "reviewer": reviewer,
        "entity_label": entity_label,
        "confidence": confidence,
        "timestamp_utc": payload.get("timestamp_utc"),
        "queue_id": payload.get("queue_id"),
        "sample_index": _coerce_int(payload.get("sample_index")),
        "candidate_id": payload.get("candidate_id"),
        "notes": payload.get("notes", ""),
        "source_review_dir": payload.get("source_review_dir"),
        "source_queue_id": payload.get("source_queue_id"),
        "source_sample_index": _coerce_int(
            payload.get("source_sample_index")
        ),
        "source_reviewer": payload.get("source_reviewer"),
        "source_binary_label": payload.get("source_binary_label"),
        "source_binary_morphology_tags": payload.get(
            "source_binary_morphology_tags", []
        ),
    }
    return _jsonable(
        {key: value for key, value in record.items() if value is not None}
    )


def _validate_annotation_identity(
    item: dict[str, Any],
    payload: dict[str, Any],
    *,
    reviewer: str,
    annotation_kind: str,
) -> None:
    queue_id = _review_item_key(item)
    expected_sample_index = _coerce_int(item.get("sample_index"))
    annotation_sample_index = _coerce_int(payload.get("sample_index"))
    if (
        expected_sample_index is not None
        and annotation_sample_index != expected_sample_index
    ):
        raise ValueError(
            f"{annotation_kind} sample_index does not match queued item "
            f"({_annotation_context(None, None, payload)}, "
            f"reviewer={reviewer}, queue_id={queue_id}, "
            f"expected_sample_index={expected_sample_index}, "
            f"annotation_sample_index={annotation_sample_index})"
        )
    expected_candidate_id = item.get("candidate_id")
    annotation_candidate_id = payload.get("candidate_id")
    if not _empty_annotation_value(expected_candidate_id) and (
        _empty_annotation_value(annotation_candidate_id)
        or str(annotation_candidate_id) != str(expected_candidate_id)
    ):
        raise ValueError(
            f"{annotation_kind} candidate_id does not match queued item "
            f"({_annotation_context(None, None, payload)}, "
            f"reviewer={reviewer}, queue_id={queue_id}, "
            f"expected_candidate_id={expected_candidate_id}, "
            f"annotation_candidate_id={annotation_candidate_id})"
        )


def _validate_latest_annotation_keys(
    latest_by_reviewer: dict[str, dict[str, dict[str, Any]]],
    *,
    queue_keys: set[str],
    annotation_kind: str,
) -> None:
    unknown_keys = sorted(set(latest_by_reviewer) - queue_keys)
    if unknown_keys:
        preview = ", ".join(unknown_keys[:5])
        if len(unknown_keys) > 5:
            preview += f", ... (+{len(unknown_keys) - 5} more)"
        raise ValueError(
            f"{annotation_kind} reference queue_id values not present in "
            f"queue.jsonl: {preview}"
        )


def _empty_annotation_value(value: Any) -> bool:
    return value is None or value == ""


def _load_review_aggregation_report(
    report_path: Path,
    *,
    review_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("aggregation report must be a JSON object")
    if payload.get("workflow") != "review-aggregate":
        raise ValueError(
            "aggregation report workflow must be review-aggregate"
        )
    if payload.get("aggregation_rule_version") != (
        REVIEW_AGGREGATION_RULE_VERSION
    ):
        raise ValueError(
            "aggregation report aggregation_rule_version is stale; "
            "rerun review-aggregate"
        )
    if payload.get("latest_per_reviewer_rule") != (
        REVIEW_LATEST_PER_REVIEWER_RULE
    ):
        raise ValueError(
            "aggregation report latest_per_reviewer_rule is stale; "
            "rerun review-aggregate"
        )
    reported_review_dir = payload.get("review_dir")
    if reported_review_dir:
        reported_path = Path(str(reported_review_dir)).expanduser().resolve()
        if reported_path != review_dir:
            raise ValueError(
                "aggregation report review_dir does not match requested "
                f"review_dir: report={reported_review_dir} "
                f"requested={review_dir}"
            )
    expected_annotation_file = payload.get("annotation_file")
    current_annotation_file = _annotation_file_fingerprint(review_dir)
    if _fingerprint_content(expected_annotation_file) != (
        _fingerprint_content(current_annotation_file)
    ):
        raise ValueError(
            "aggregation report does not match current annotations.jsonl; "
            "rerun review-aggregate"
        )
    expected_queue_file = payload.get("queue_file")
    current_queue_file = _queue_file_fingerprint(review_dir)
    if _fingerprint_content(expected_queue_file, label="queue") != (
        _fingerprint_content(current_queue_file, label="queue")
    ):
        raise ValueError(
            "aggregation report does not match current queue.jsonl; "
            "rerun review-aggregate"
        )
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("aggregation report must contain a decisions list")
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(
                "aggregation report decisions must be JSON objects: "
                f"index={index}"
            )
        for key in ("queue_id", "status"):
            if key not in decision:
                raise ValueError(
                    "aggregation report decision is missing required "
                    f"field {key!r}; rerun review-aggregate: index={index}"
                )
    return payload


def _validate_review_aggregation_report_decisions(
    payload: dict[str, Any],
    *,
    queue: list[dict[str, Any]],
    latest_by_reviewer: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Reject hand-edited reports whose decisions no longer derive locally."""

    queue_keys = {_review_item_key(item) for item in queue}
    _validate_latest_annotation_keys(
        latest_by_reviewer,
        queue_keys=queue_keys,
        annotation_kind="review annotations",
    )
    min_reviewers = _required_report_positive_int(payload, "min_reviewers")
    min_actionable_reviewers = _required_report_positive_int(
        payload,
        "min_actionable_reviewers",
    )
    if min_actionable_reviewers > min_reviewers:
        raise ValueError(
            "aggregation report min_actionable_reviewers cannot exceed "
            "min_reviewers"
        )
    if "consensus_rule" not in payload:
        raise ValueError("aggregation report must include consensus_rule")
    consensus_rule = str(payload["consensus_rule"])
    if consensus_rule not in REVIEW_CONSENSUS_RULES:
        raise ValueError(
            "aggregation report consensus_rule must be one of: "
            + ", ".join(sorted(REVIEW_CONSENSUS_RULES))
        )

    expected_decisions = [
        _aggregate_review_item(
            item,
            latest_by_reviewer.get(_review_item_key(item), {}),
            min_reviewers=min_reviewers,
            min_actionable_reviewers=min_actionable_reviewers,
            consensus_rule=consensus_rule,
        )
        for item in queue
    ]
    expected_by_queue_id = {
        str(decision["queue_id"]): decision for decision in expected_decisions
    }
    reported_by_queue_id: dict[str, dict[str, Any]] = {}
    for decision in payload.get("decisions", []):
        queue_id = str(decision.get("queue_id", ""))
        if not queue_id:
            raise ValueError(
                "aggregation report decision is missing queue_id; "
                "rerun review-aggregate"
            )
        if queue_id in reported_by_queue_id:
            raise ValueError(
                "aggregation report contains duplicate decision "
                f"queue_id={queue_id}; rerun review-aggregate"
            )
        reported_by_queue_id[queue_id] = decision

    if set(reported_by_queue_id) != set(expected_by_queue_id):
        raise ValueError(
            "aggregation report decisions do not match queue.jsonl; "
            "rerun review-aggregate"
        )

    for queue_id, expected in expected_by_queue_id.items():
        reported = reported_by_queue_id[queue_id]
        if _canonical_review_decision(reported) != (
            _canonical_review_decision(expected)
        ):
            raise ValueError(
                "aggregation report decision does not match current "
                f"annotations.jsonl; rerun review-aggregate: "
                f"queue_id={queue_id}"
            )

    status_counts = {status: 0 for status in REVIEW_AGGREGATION_STATUSES}
    actionable_label_counts = {
        label: 0 for label in sorted(REVIEW_LABEL_TO_VALUE)
    }
    for decision in expected_decisions:
        status_counts[decision["status"]] += 1
        consensus_label = decision.get("consensus_label")
        if decision["status"] == "actionable" and consensus_label:
            if str(consensus_label) not in actionable_label_counts:
                raise ValueError(
                    "aggregation report consensus_label is not recognized; "
                    "rerun review-aggregate"
                )
            actionable_label_counts[str(consensus_label)] += 1
    if payload.get("status_counts") != status_counts:
        raise ValueError(
            "aggregation report status_counts do not match current "
            "annotations.jsonl; rerun review-aggregate"
        )
    if payload.get("actionable_label_counts") != actionable_label_counts:
        raise ValueError(
            "aggregation report actionable_label_counts do not match "
            "current annotations.jsonl; rerun review-aggregate"
        )


def _canonical_review_decision(decision: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "queue_id",
        "sample_index",
        "candidate_id",
        "status",
        "reason",
        "consensus_label",
        "reviewer_count",
        "valid_reviewer_count",
        "actionable_reviewer_count",
        "label_counts",
        "actionable_label_counts",
        "supporting_reviewers",
        "ignored_unsure_reviewers",
        "consensus_morphology_tags",
        "latest_annotations",
        "aggregation",
    )
    return _jsonable({key: decision.get(key) for key in keys})


def _required_report_positive_int(
    payload: dict[str, Any],
    key: str,
) -> int:
    if key not in payload:
        raise ValueError(f"aggregation report must include {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"aggregation report {key} must be a positive integer"
        )
    if value <= 0:
        raise ValueError(
            f"aggregation report {key} must be a positive integer"
        )
    return value


def _fingerprint_content(
    fingerprint: dict[str, Any] | None,
    *,
    label: str = "annotation",
) -> dict[str, Any] | None:
    if fingerprint is None:
        return None
    if not isinstance(fingerprint, dict):
        raise ValueError(f"{label} fingerprint must be a JSON object")
    stable_keys = ("sha256", "size_bytes", "line_count")
    missing = [key for key in stable_keys if fingerprint.get(key) is None]
    if missing:
        raise ValueError(
            f"{label} fingerprint missing content fields: "
            + ", ".join(missing)
        )
    # Compare only the stable content identity fields; path and future
    # descriptive fields are intentionally not freshness inputs.
    return {key: fingerprint.get(key) for key in stable_keys}


def _aggregated_actionable_annotations(
    payload: dict[str, Any],
    *,
    report_path: Path,
) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for decision in payload.get("decisions", []):
        if decision.get("status") != "actionable":
            continue
        label_text = str(decision.get("consensus_label", "")).strip().lower()
        if label_text not in REVIEW_LABEL_TO_VALUE:
            continue
        queue_id = str(decision.get("queue_id"))
        timestamp = _aggregation_supporting_timestamp(decision)
        if not timestamp:
            raise ValueError(
                "actionable aggregation decision lacks supporting "
                "timestamp_utc; rerun review-aggregate with timestamped "
                f"annotations: queue_id={queue_id}"
            )
        annotations[queue_id] = {
            "queue_id": queue_id,
            "sample_index": decision.get("sample_index"),
            "candidate_id": decision.get("candidate_id"),
            "reviewer_label": label_text,
            "timestamp_utc": timestamp,
            "morphology_tags": decision.get("consensus_morphology_tags", []),
            "notes": _aggregation_supporting_notes(decision),
            "aggregation": {
                "report_path": str(report_path),
                "report_created_at_utc": payload.get("created_at_utc"),
                "status": decision.get("status"),
                "reason": decision.get("reason"),
                "rule_version": payload.get("aggregation_rule_version"),
                "latest_per_reviewer_rule": payload.get(
                    "latest_per_reviewer_rule"
                ),
                "consensus_rule": payload.get("consensus_rule"),
                "min_reviewers": payload.get("min_reviewers"),
                "min_actionable_reviewers": payload.get(
                    "min_actionable_reviewers"
                ),
                "reviewer_count": decision.get("reviewer_count"),
                "actionable_reviewer_count": decision.get(
                    "actionable_reviewer_count"
                ),
                "label_counts": decision.get("label_counts", {}),
                "actionable_label_counts": decision.get(
                    "actionable_label_counts", {}
                ),
                "supporting_reviewers": decision.get(
                    "supporting_reviewers", []
                ),
                "ignored_unsure_reviewers": decision.get(
                    "ignored_unsure_reviewers", []
                ),
                "latest_annotations": decision.get("latest_annotations", []),
            },
        }
    return annotations


def _aggregation_supporting_notes(decision: dict[str, Any]) -> str:
    supporting = {
        str(reviewer) for reviewer in decision.get("supporting_reviewers", [])
    }
    snippets = []
    for annotation in decision.get("latest_annotations", []):
        reviewer = str(annotation.get("reviewer", ""))
        if reviewer not in supporting:
            continue
        note = str(annotation.get("notes", "")).strip()
        if not note:
            continue
        snippets.append(f"{reviewer}: {note}")
    return "\n".join(snippets)


def _aggregation_supporting_timestamp(decision: dict[str, Any]) -> str | None:
    supporting = {
        str(reviewer) for reviewer in decision.get("supporting_reviewers", [])
    }
    timestamps: list[tuple[datetime, str]] = []
    for annotation in decision.get("latest_annotations", []):
        if str(annotation.get("reviewer", "")) not in supporting:
            continue
        timestamp = str(annotation.get("timestamp_utc", "")).strip()
        if timestamp:
            try:
                parsed_timestamp = _parse_review_timestamp(timestamp)
            except ValueError as exc:
                raise ValueError(
                    "invalid timestamp_utc in supporting review "
                    f"annotation: queue_id={decision.get('queue_id')} "
                    f"reviewer={annotation.get('reviewer')} "
                    f"timestamp={timestamp!r}"
                ) from exc
            timestamps.append((parsed_timestamp, timestamp))
    if not timestamps:
        return None
    return max(timestamps, key=lambda item: item[0])[1]


def _parse_review_timestamp(timestamp: str) -> datetime:
    normalized = timestamp.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _review_aggregation_metadata(
    aggregation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "review_aggregation_report": aggregation.get("report_path"),
        "review_aggregation_status": aggregation.get("status"),
        "review_aggregation_reason": aggregation.get("reason"),
        "review_aggregation_rule_version": aggregation.get("rule_version"),
        "review_aggregation_latest_per_reviewer_rule": aggregation.get(
            "latest_per_reviewer_rule"
        ),
        "review_aggregation_consensus_rule": aggregation.get(
            "consensus_rule"
        ),
        "review_aggregation_min_reviewers": aggregation.get("min_reviewers"),
        "review_aggregation_min_actionable_reviewers": aggregation.get(
            "min_actionable_reviewers"
        ),
        "review_aggregation_reviewer_count": aggregation.get(
            "reviewer_count"
        ),
        "review_aggregation_actionable_reviewer_count": aggregation.get(
            "actionable_reviewer_count"
        ),
        "review_aggregation_label_counts": aggregation.get(
            "label_counts", {}
        ),
        "review_aggregation_actionable_label_counts": aggregation.get(
            "actionable_label_counts", {}
        ),
        "review_aggregation_supporting_reviewers": aggregation.get(
            "supporting_reviewers", []
        ),
        "review_aggregation_ignored_unsure_reviewers": aggregation.get(
            "ignored_unsure_reviewers", []
        ),
        "review_aggregation_latest_annotations": aggregation.get(
            "latest_annotations", []
        ),
        "review_aggregation_report_created_at_utc": aggregation.get(
            "report_created_at_utc"
        ),
    }


def _annotation_file_fingerprint(review_dir: Path) -> dict[str, Any] | None:
    return _jsonl_file_fingerprint(review_dir / "annotations.jsonl")


def _queue_file_fingerprint(review_dir: Path) -> dict[str, Any] | None:
    return _jsonl_file_fingerprint(review_dir / "queue.jsonl")


def _entity_annotation_file_fingerprint(
    review_dir: Path,
) -> dict[str, Any] | None:
    return _jsonl_file_fingerprint(review_dir / "entity_annotations.jsonl")


def _jsonl_file_fingerprint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    line_count = 0
    line_has_content = False
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            for byte in chunk:
                if byte in (10, 13):
                    if line_has_content:
                        line_count += 1
                        line_has_content = False
                elif byte not in (9, 32):
                    line_has_content = True
    if line_has_content:
        line_count += 1
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(path.stat().st_size),
        "line_count": line_count,
    }


def _review_item_key(item: dict[str, Any]) -> str:
    return str(item.get("queue_id") or item.get("sample_index"))


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_review_annotation_csv(
    path: Path,
) -> list[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"annotation CSV is empty: {path}")
        missing = {
            "queue_id",
            "sample_index",
            "candidate_id",
            "reviewer_label",
        } - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "annotation CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = [
            (line_number, dict(row))
            for line_number, row in enumerate(reader, start=2)
        ]
    return rows


def _parse_review_morphology_tags(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    tags = []
    for part in text.replace(",", ";").split(";"):
        tag = part.strip()
        if tag:
            tags.append(tag)
    unknown = sorted(set(tags) - set(MORPHOLOGY_TAGS))
    if unknown:
        raise ValueError(
            "unknown review morphology_tags: " + ", ".join(unknown)
        )
    return tags


def _dataset_review_rows(
    dataset_dir: Path, split: str
) -> list[dict[str, Any]]:
    labels = np.load(dataset_dir / "labels.npy", mmap_mode="r")
    split_array = np.load(dataset_dir / "split.npy", mmap_mode="r")
    clean_split = split.strip().lower()
    if clean_split == "all":
        indices = np.arange(labels.shape[0], dtype=np.int64)
    else:
        indices = prediction_group_indices(split_array, clean_split)
    metadata_rows = load_metadata_rows(dataset_dir)
    rows = []
    for sample_index in indices.tolist():
        index = int(sample_index)
        if metadata_rows:
            row = dict(metadata_rows[index])
        else:
            row = {}
        row.setdefault("sample_index", index)
        row.setdefault("candidate_id", f"sample-{index:06d}")
        row.setdefault(
            "split",
            INDEX_TO_SPLIT.get(int(split_array[index]), "unknown"),
        )
        row["sample_index"] = index
        row["label"] = int(labels[index])
        rows.append(row)
    return rows


def _dataset_split_rows(
    dataset_dir: Path, split: str
) -> list[dict[str, Any]]:
    labels = np.load(dataset_dir / "labels.npy", mmap_mode="r")
    split_array = np.load(dataset_dir / "split.npy", mmap_mode="r")
    indices = prediction_group_indices(split_array, split)
    metadata_rows = load_metadata_rows(dataset_dir)
    rows = []
    for sample_index in indices.tolist():
        if metadata_rows:
            row = dict(metadata_rows[int(sample_index)])
        else:
            row = {}
        row.setdefault("sample_index", int(sample_index))
        row.setdefault("candidate_id", f"sample-{int(sample_index):06d}")
        row["sample_index"] = int(sample_index)
        row["label"] = int(labels[int(sample_index)])
        rows.append(row)
    return rows


def _compare_probability_map(
    *,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
) -> dict[str, float]:
    rows = load_prediction_rows(
        run_dir=run_dir, dataset_dir=dataset_dir, split=split
    )
    mapping = {}
    for row in rows:
        key = _prediction_key(row)
        mapping[key] = float(row["probability"])
    return mapping


def _validate_artifact_context(
    artifact_dir: Path,
    *,
    dataset_dir: Path,
    split: str,
    require_summary: bool,
) -> dict[str, Any]:
    summary_path = artifact_dir / "summary.json"
    if not summary_path.exists():
        if require_summary:
            raise FileNotFoundError(
                f"missing prediction summary for identity validation: "
                f"{summary_path}"
            )
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_dataset = summary.get("dataset_dir")
    if require_summary and summary_dataset is None:
        raise ValueError(
            f"prediction summary missing required dataset_dir: {summary_path}"
        )
    if summary_dataset is not None:
        resolved = Path(str(summary_dataset)).expanduser().resolve()
        if resolved != dataset_dir:
            raise ValueError(
                "prediction artifact dataset_dir does not match requested "
                f"dataset_dir: artifact={resolved} requested={dataset_dir}"
            )
    summary_split = summary.get("split")
    if require_summary and summary_split is None:
        raise ValueError(
            f"prediction summary missing required split: {summary_path}"
        )
    if summary_split is not None and str(summary_split) != split:
        raise ValueError(
            "prediction artifact split does not match requested split: "
            f"artifact={summary_split} requested={split}"
        )
    return summary


def _decision_threshold(summary: dict[str, Any]) -> float:
    diagnostics = summary.get("threshold_diagnostics")
    if isinstance(diagnostics, dict):
        validation_selected = diagnostics.get("validation_selected")
        if isinstance(validation_selected, dict):
            evaluated = validation_selected.get("evaluated_split_metrics")
            if isinstance(evaluated, dict) and "threshold" in evaluated:
                return _validated_threshold(evaluated["threshold"])
        fixed = diagnostics.get("fixed_threshold")
        if isinstance(fixed, dict) and "threshold" in fixed:
            return _validated_threshold(fixed["threshold"])
    if "threshold" in summary:
        return _validated_threshold(summary["threshold"])
    return 0.5


def _validated_threshold(value: Any) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "prediction summary threshold must be finite and satisfy "
            "0.0 <= threshold <= 1.0"
        )
    return threshold


def _validate_prediction_identity(
    meta: dict[str, Any],
    payload: dict[str, Any],
    *,
    row_number: int,
) -> None:
    checked = False
    if "sample_index" in payload:
        checked = True
        if int(payload["sample_index"]) != int(meta["sample_index"]):
            raise ValueError(
                "prediction row identity mismatch at row "
                f"{row_number}: sample_index={payload['sample_index']} "
                f"expected={meta['sample_index']}"
            )
    payload_candidate = payload.get("candidate_id")
    meta_candidate = meta.get("candidate_id")
    if payload_candidate not in {None, ""} and meta_candidate not in {
        None,
        "",
    }:
        checked = True
        if str(payload_candidate) != str(meta_candidate):
            raise ValueError(
                "prediction row identity mismatch at row "
                f"{row_number}: candidate_id={payload_candidate} "
                f"expected={meta_candidate}"
            )
    if not checked:
        raise ValueError(
            "prediction row lacks sample_index or candidate_id for identity "
            f"validation at row {row_number}"
        )


def _validate_unique_prediction_candidate_ids(
    payloads: list[dict[str, Any]],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    missing = 0
    for payload in payloads:
        candidate_id = payload.get("candidate_id")
        if candidate_id in {None, ""}:
            missing += 1
            continue
        key = str(candidate_id)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if missing or duplicates:
        detail = []
        if missing:
            detail.append(f"{missing} missing candidate_id values")
        if duplicates:
            preview = ", ".join(sorted(duplicates)[:5])
            detail.append(f"duplicate candidate_id values: {preview}")
        raise ValueError(
            "prediction rows without sample_index require unique "
            "candidate_id fallback keys: " + "; ".join(detail)
        )


def _prepare_review_row(
    row: dict[str, Any],
    *,
    compare_maps: list[dict[str, float]],
) -> dict[str, Any]:
    probability = float(row["probability"])
    label = int(row["label"])
    decision_threshold = float(row.get("decision_threshold", 0.5))
    key = _prediction_key(row)
    compare_probs = [
        mapping[key] for mapping in compare_maps if key in mapping
    ]
    disagreement = (
        max(abs(probability - value) for value in compare_probs)
        if compare_probs
        else 0.0
    )
    prediction = 1 if probability >= decision_threshold else 0
    known_error = int(prediction != label)
    threshold_distance = abs(probability - decision_threshold)
    threshold_scale = (
        max(decision_threshold, 1e-6)
        if probability < decision_threshold
        else max(1.0 - decision_threshold, 1e-6)
    )
    normalized_margin = min(1.0, threshold_distance / threshold_scale)
    uncertainty = max(0.0, 1.0 - normalized_margin)
    mistake_confidence = normalized_margin if known_error else 0.0
    stratum = _review_stratum(row)
    prepared = dict(row)
    prepared.update(
        {
            "queue_id": f"sample:{int(row['sample_index'])}",
            "prediction": prediction,
            "decision_threshold": decision_threshold,
            "known_error": known_error,
            "uncertainty_score": float(uncertainty),
            "disagreement_score": float(disagreement),
            "mistake_confidence": float(mistake_confidence),
            "audit_score": _stable_audit_score(key),
            "review_stratum": stratum,
            "compare_probabilities": compare_probs,
        }
    )
    return _jsonable(prepared)


def _prepare_dataset_review_row(row: dict[str, Any]) -> dict[str, Any]:
    key = _prediction_key(row)
    prepared = dict(row)
    prepared.update(
        {
            "queue_id": f"sample:{int(row['sample_index'])}",
            "prediction": None,
            "probability": 0.5,
            "decision_threshold": 0.5,
            "known_error": 0,
            "uncertainty_score": 1.0,
            "disagreement_score": 0.0,
            "mistake_confidence": 0.0,
            "audit_score": _stable_audit_score(key),
            "review_stratum": _review_stratum(row),
            "compare_probabilities": [],
            "review_input_source": "dataset_only",
        }
    )
    return _jsonable(prepared)


def _select_dataset_review_rows(
    rows: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    rarity = _rarity_scores(rows)
    for row in rows:
        row["stratum_rarity_score"] = rarity.get(
            str(row["review_stratum"]), 0.0
        )
        row["dataset_review_score"] = 0.70 * float(
            row["stratum_rarity_score"]
        ) + 0.30 * float(row["audit_score"])
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["dataset_review_score"]),
            float(row["audit_score"]),
        ),
        reverse=True,
    )
    return _finalize_queue(ordered[:max_items], "dataset_audit")


def _select_review_rows(
    rows: list[dict[str, Any]],
    *,
    max_items: int,
    strategy: str,
) -> list[dict[str, Any]]:
    if strategy == "uncertainty":
        ordered = sorted(
            rows,
            key=lambda row: float(row["uncertainty_score"]),
            reverse=True,
        )
        return _finalize_queue(ordered[:max_items], "uncertainty")
    if strategy == "known-errors":
        ordered = sorted(
            [row for row in rows if int(row["known_error"])],
            key=lambda row: float(row["mistake_confidence"]),
            reverse=True,
        )
        return _finalize_queue(ordered[:max_items], "known_error")

    rarity = _rarity_scores(rows)
    for row in rows:
        row["stratum_rarity_score"] = rarity.get(
            str(row["review_stratum"]), 0.0
        )
        row["hybrid_score"] = (
            0.45 * float(row["uncertainty_score"])
            + 0.25 * float(row["disagreement_score"])
            + 0.15 * float(row["mistake_confidence"])
            + 0.10 * float(row["stratum_rarity_score"])
            + 0.05 * float(row["audit_score"])
        )

    quotas = [
        (reason, quota, field)
        for reason, quota, field in (
            (
                "uncertainty",
                _hybrid_quota(max_items, "uncertainty"),
                "uncertainty_score",
            ),
            (
                "disagreement",
                _hybrid_quota(max_items, "disagreement"),
                "disagreement_score",
            ),
            (
                "known_error",
                _hybrid_quota(max_items, "known_error"),
                "mistake_confidence",
            ),
            (
                "undercovered_stratum",
                _hybrid_quota(max_items, "undercovered_stratum"),
                "stratum_rarity_score",
            ),
            ("audit", _hybrid_quota(max_items, "audit"), "audit_score"),
        )
    ]
    selected: dict[str, dict[str, Any]] = {}
    for reason, quota, field in quotas:
        if quota <= 0:
            continue
        eligible = rows
        if reason == "known_error":
            eligible = [row for row in rows if int(row["known_error"])]
        if reason == "disagreement":
            eligible = [
                row for row in rows if float(row["disagreement_score"]) > 0.0
            ]
        ordered = sorted(
            eligible,
            key=lambda row: float(row.get(field, 0.0)),
            reverse=True,
        )
        for row in ordered:
            key = str(row["queue_id"])
            if key in selected:
                continue
            item = dict(row)
            item["rank_reason"] = reason
            selected[key] = item
            if (
                sum(
                    1
                    for value in selected.values()
                    if value["rank_reason"] == reason
                )
                >= quota
            ):
                break

    if len(selected) < max_items:
        ordered = sorted(
            rows,
            key=lambda row: float(row["hybrid_score"]),
            reverse=True,
        )
        for row in ordered:
            key = str(row["queue_id"])
            if key in selected:
                continue
            item = dict(row)
            item["rank_reason"] = "hybrid"
            selected[key] = item
            if len(selected) >= max_items:
                break
    final = sorted(
        selected.values(),
        key=lambda row: float(row.get("hybrid_score", 0.0)),
        reverse=True,
    )[:max_items]
    for rank, row in enumerate(final, start=1):
        row["rank"] = rank
    return [_jsonable(row) for row in final]


def _finalize_queue(
    rows: list[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    final = []
    for rank, row in enumerate(rows, start=1):
        item = dict(row)
        item["rank"] = rank
        item["rank_reason"] = reason
        final.append(_jsonable(item))
    return final


def _hybrid_quota(max_items: int, reason: str) -> int:
    order = [
        "uncertainty",
        "disagreement",
        "known_error",
        "undercovered_stratum",
        "audit",
    ]
    weights = {
        "uncertainty": 0.45,
        "disagreement": 0.20,
        "known_error": 0.15,
        "undercovered_stratum": 0.10,
        "audit": 0.10,
    }
    if max_items <= 0:
        return 0
    if max_items < len(order):
        return 1 if reason in order[:max_items] else 0
    quotas = {
        name: max(1, int(round(max_items * weight)))
        for name, weight in weights.items()
    }
    while sum(quotas.values()) > max_items:
        for name in sorted(
            order,
            key=lambda item: quotas[item],
            reverse=True,
        ):
            if quotas[name] > 1:
                quotas[name] -= 1
                break
    while sum(quotas.values()) < max_items:
        quotas["uncertainty"] += 1
    return quotas[reason]


def _rarity_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["review_stratum"])
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {key: 1.0 - (count / max_count) for key, count in counts.items()}


def _review_stratum(row: dict[str, Any]) -> str:
    values = [
        str(row.get("center_source") or "unknown-center"),
        str(row.get("catalog_pool_role") or "unknown-pool"),
        str(row.get("positive_quality_stratum") or "not-positive"),
        str(resolve_catalog_morphology(row) or "unknown-morphology"),
        str(resolve_negative_difficulty(row) or "unknown-difficulty"),
        str(resolve_mask_pressure(row) or "unknown-mask"),
    ]
    return "|".join(values)


def _prediction_key(row: dict[str, Any]) -> str:
    if "sample_index" in row:
        return f"sample:{int(row['sample_index'])}"
    candidate_id = row.get("candidate_id")
    if candidate_id in {None, ""}:
        raise ValueError("prediction row has no sample_index or candidate_id")
    return f"candidate:{candidate_id}"


def _load_entity_source_dataset(dataset_dir: Path) -> dict[str, Any]:
    search = np.load(dataset_dir / "search.npy", mmap_mode="r")
    template = np.load(dataset_dir / "template.npy", mmap_mode="r")
    difference = (
        np.load(dataset_dir / "difference.npy", mmap_mode="r")
        if (dataset_dir / "difference.npy").exists()
        else None
    )
    labels = np.load(dataset_dir / "labels.npy", mmap_mode="r")
    split = np.load(dataset_dir / "split.npy", mmap_mode="r")
    sample_count = int(search.shape[0])
    arrays = {
        "template.npy": template,
        "labels.npy": labels,
        "split.npy": split,
    }
    if difference is not None:
        arrays["difference.npy"] = difference
    for name, array in arrays.items():
        if int(array.shape[0]) != sample_count:
            raise ValueError(
                "entity source dataset array row count does not match "
                f"search.npy: {name}={int(array.shape[0])} "
                f"search={sample_count} dataset_dir={dataset_dir}"
            )
    return {
        "search": search,
        "template": template,
        "difference": difference,
        "labels": labels,
        "split": split,
        "metadata_rows": load_metadata_rows(dataset_dir),
        "sample_count": sample_count,
    }


def _validate_entity_source_cache_compatibility(
    source_caches: dict[Path, dict[str, Any]],
) -> None:
    reference_path: Path | None = None
    reference: dict[str, Any] | None = None
    for dataset_dir, cache in source_caches.items():
        if reference is None:
            reference = cache
            reference_path = dataset_dir
            continue
        for name in ("search", "template"):
            current = np.asarray(cache[name])
            expected = np.asarray(reference[name])
            if (
                current.shape[1:] != expected.shape[1:]
                or current.dtype != expected.dtype
            ):
                raise ValueError(
                    "entity source datasets must use matching image "
                    f"shape and dtype for {name}: "
                    f"{dataset_dir} shape={current.shape[1:]} "
                    f"dtype={current.dtype}, "
                    f"{reference_path} shape={expected.shape[1:]} "
                    f"dtype={expected.dtype}"
                )
        current_difference = cache["difference"]
        expected_difference = reference["difference"]
        if (current_difference is None) != (expected_difference is None):
            raise ValueError(
                "entity source datasets must consistently include "
                "difference.npy"
            )
        if current_difference is not None and expected_difference is not None:
            current = np.asarray(current_difference)
            expected = np.asarray(expected_difference)
            if (
                current.shape[1:] != expected.shape[1:]
                or current.dtype != expected.dtype
            ):
                raise ValueError(
                    "entity source datasets must use matching image "
                    f"shape and dtype for difference: "
                    f"{dataset_dir} shape={current.shape[1:]} "
                    f"dtype={current.dtype}, "
                    f"{reference_path} shape={expected.shape[1:]} "
                    f"dtype={expected.dtype}"
                )


def _stack_or_empty(
    items: list[np.ndarray],
    reference: np.ndarray,
) -> np.ndarray:
    if items:
        return np.stack(items)
    return np.empty(
        (0, *tuple(reference.shape[1:])),
        dtype=np.asarray(reference).dtype,
    )


def _entity_queue_id(record: dict[str, Any]) -> str:
    source_queue_digest = hashlib.sha256(
        str(record["source_queue_id"]).encode("utf-8")
    ).hexdigest()[:12]
    source_reviewer_digest = hashlib.sha256(
        str(record["source_reviewer"]).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"entity:{record['source_review_digest']}:"
        f"{source_queue_digest}:{source_reviewer_digest}"
    )


def _entity_queue_item(
    record: dict[str, Any],
    *,
    entity_index: int,
    entity_label: int,
    source_label: int,
) -> dict[str, Any]:
    source_item = record["source_item"]
    annotation = record["binary_annotation"]
    source_candidate_id = (
        source_item.get("candidate_id")
        or annotation.get("candidate_id")
        or f"source-sample-{int(record['source_sample_index']):06d}"
    )
    source_probability = annotation.get("source_probability")
    if source_probability is None:
        source_probability = source_item.get("probability")
    entity_queue_id = _entity_queue_id(record)
    item = {
        "queue_id": entity_queue_id,
        "sample_index": int(entity_index),
        "candidate_id": entity_queue_id,
        "label": int(entity_label),
        "probability": source_item.get("probability"),
        "rank_reason": "source_binary_real_review",
        "source_review_dir": str(record["source_review_dir"]),
        "source_review_workflow": record["source_manifest"].get("workflow"),
        "source_review_split": record["source_manifest"].get("split"),
        "source_dataset_dir": str(record["source_dataset_dir"]),
        "source_run_dir": record["source_manifest"].get("run_dir"),
        "source_queue_id": str(record["source_queue_id"]),
        "source_sample_index": int(record["source_sample_index"]),
        "source_candidate_id": source_candidate_id,
        "source_reviewer": record["source_reviewer"],
        "source_binary_label": annotation.get("reviewer_label"),
        "source_binary_morphology_tags": annotation.get(
            "morphology_tags", []
        ),
        "source_binary_notes": annotation.get("notes", ""),
        "source_binary_timestamp_utc": annotation.get("timestamp_utc"),
        "source_label": int(source_label),
        "source_probability": source_probability,
        "source_rank_reason": source_item.get("rank_reason"),
    }
    return _jsonable({key: value for key, value in item.items()})


def _entity_metadata_row(
    record: dict[str, Any],
    *,
    queue_item: dict[str, Any],
    source_cache: dict[str, Any],
    entity_index: int,
    entity_label: int,
    source_label: int,
    source_split_index: int,
) -> dict[str, Any]:
    source_sample_index = int(record["source_sample_index"])
    source_rows = source_cache["metadata_rows"]
    if source_rows and source_sample_index < len(source_rows):
        row = dict(source_rows[source_sample_index])
    else:
        row = {}
    source_candidate_id = (
        record["source_item"].get("candidate_id")
        or row.get("candidate_id")
        or queue_item["source_candidate_id"]
    )
    source_split_name = str(
        row.get("split")
        or INDEX_TO_SPLIT.get(
            source_split_index,
            record["source_manifest"].get("split") or "test",
        )
    )
    if source_split_name not in {"train", "val", "test"}:
        source_split_name = "test"
    original_label_source = row.get("label_source")
    row["candidate_id"] = queue_item["candidate_id"]
    row["sample_index"] = int(entity_index)
    row["exposure_id"] = row.get("exposure_id", int(entity_index))
    row.setdefault("ccd_id", None)
    row.setdefault("band", "unknown")
    row.setdefault("x", 0.0)
    row.setdefault("y", 0.0)
    row["split_group"] = queue_item["queue_id"]
    row["split"] = source_split_name
    row["label"] = int(entity_label)
    row["label_source"] = "entity_review_source_binary_label"
    row.update(
        {
            "entity_review_task": "entity_classification",
            "entity_source_review_dir": queue_item["source_review_dir"],
            "entity_source_queue_id": queue_item["source_queue_id"],
            "entity_source_sample_index": queue_item["source_sample_index"],
            "entity_source_candidate_id": source_candidate_id,
            "entity_source_reviewer": queue_item["source_reviewer"],
            "entity_source_binary_label": queue_item["source_binary_label"],
            "entity_source_binary_morphology_tags": queue_item[
                "source_binary_morphology_tags"
            ],
            "entity_source_binary_notes": queue_item["source_binary_notes"],
            "entity_source_binary_timestamp_utc": queue_item[
                "source_binary_timestamp_utc"
            ],
            "entity_source_binary_label_value": int(entity_label),
            "entity_source_dataset_dir": queue_item["source_dataset_dir"],
            "entity_source_run_dir": queue_item["source_run_dir"],
            "entity_source_split": queue_item["source_review_split"],
            "entity_source_original_label": int(source_label),
            "entity_source_original_label_source": original_label_source,
            "entity_source_probability": queue_item["source_probability"],
            "entity_source_rank_reason": queue_item["source_rank_reason"],
        }
    )
    return _jsonable(row)


def _dataset_fingerprint(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    files = {}
    for name in (
        "search.npy",
        "template.npy",
        "difference.npy",
        "labels.npy",
        "split.npy",
        "metadata.jsonl",
    ):
        path = dataset_dir / name
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[name] = {
                "sha256": digest.hexdigest(),
                "size_bytes": int(path.stat().st_size),
            }
        else:
            files[name] = None
    labels = np.load(dataset_dir / "labels.npy", mmap_mode="r")
    return {
        "sample_count": int(labels.shape[0]),
        "files": files,
    }


def _validate_dataset_fingerprint(
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> None:
    expected = manifest.get("dataset_fingerprint")
    if expected is None:
        return
    current = _dataset_fingerprint(dataset_dir)
    if current != expected:
        raise ValueError(
            "review dataset fingerprint does not match the queued dataset"
        )


def _validate_queue_sample_indices(
    queue: list[dict[str, Any]],
    *,
    sample_count: int,
) -> None:
    invalid = []
    for item in queue:
        try:
            sample_index = int(item["sample_index"])
        except (KeyError, TypeError, ValueError):
            invalid.append(f"{item.get('queue_id', '<missing>')}:missing")
            continue
        if sample_index < 0 or sample_index >= sample_count:
            queue_id = item.get("queue_id", sample_index)
            invalid.append(f"{queue_id}:{sample_index}")
    if invalid:
        preview = ", ".join(invalid[:5])
        suffix = "..." if len(invalid) > 5 else ""
        raise ValueError(
            "review queue contains sample_index values outside the dataset "
            f"arrays (sample_count={sample_count}): {preview}{suffix}"
        )


def _queue_sample_index_map(queue: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in queue:
        sample_index = int(item["sample_index"])
        queue_id = str(item.get("queue_id") or sample_index)
        if queue_id in mapping and mapping[queue_id] != sample_index:
            raise ValueError(f"duplicate review queue_id: {queue_id}")
        mapping[queue_id] = sample_index
    return mapping


def _validate_bokeh_host(host: str) -> None:
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            "review-bokeh host must be loopback-only: localhost, "
            "127.0.0.1, or ::1"
        )


def _bokeh_url_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _stable_audit_score(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(value / ((1 << 64) - 1))


def _append_annotation(review_dir: Path, payload: dict[str, Any]) -> None:
    with (review_dir / "annotations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(_json_line(payload) + "\n")
    state = _load_review_state(review_dir)
    by_reviewer = latest_annotations_by_reviewer(review_dir, strict=False)
    state["reviewed_count"] = len(latest_annotation_per_queue(review_dir))
    state["reviewer_decision_count"] = sum(
        len(payloads) for payloads in by_reviewer.values()
    )
    state["reviewer_count"] = len(
        {
            reviewer
            for payloads in by_reviewer.values()
            for reviewer in payloads
        }
    )
    state["updated_at_utc"] = _utc_now()
    (review_dir / "review_state.json").write_text(
        _json_dumps(state) + "\n",
        encoding="utf-8",
    )


def _append_entity_annotation(
    review_dir: Path,
    payload: dict[str, Any],
) -> None:
    with (review_dir / "entity_annotations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(_json_line(payload) + "\n")
    state = _load_review_state(review_dir)
    by_reviewer = latest_entity_annotations_by_reviewer(review_dir)
    state["reviewed_count"] = len(
        latest_entity_annotation_per_queue(review_dir)
    )
    state["reviewer_decision_count"] = sum(
        len(payloads) for payloads in by_reviewer.values()
    )
    state["reviewer_count"] = len(
        {
            reviewer
            for payloads in by_reviewer.values()
            for reviewer in payloads
        }
    )
    state["updated_at_utc"] = _utc_now()
    (review_dir / "review_state.json").write_text(
        _json_dumps(state) + "\n",
        encoding="utf-8",
    )


def _write_review_state(review_dir: Path, *, current_index: int) -> None:
    state = {
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "current_index": current_index,
        "reviewed_count": 0,
    }
    (review_dir / "review_state.json").write_text(
        _json_dumps(state) + "\n",
        encoding="utf-8",
    )


def _load_review_state(review_dir: Path) -> dict[str, Any]:
    path = review_dir / "review_state.json"
    if not path.exists():
        return {"current_index": 0, "reviewed_count": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _reviewer_name(value: Any) -> str:
    return " ".join(str(value or "").split())


def _username_required_html(has_reviewer: bool) -> str:
    if has_reviewer:
        return ""
    return '<span style="color:#b00020; font-weight:700;">Mandatory</span>'


def _reviewer_reviewed_count(
    latest_by_reviewer: dict[str, dict[str, dict[str, Any]]],
    reviewer: str,
) -> int:
    if not reviewer:
        return 0
    return sum(
        1
        for per_reviewer in latest_by_reviewer.values()
        if reviewer in per_reviewer
    )


def _other_reviews_html(
    item: dict[str, Any],
    latest_by_reviewer: dict[str, dict[str, dict[str, Any]]],
    *,
    reviewer: str,
) -> str:
    queue_id = str(item["queue_id"])
    entries = [
        (name, payload)
        for name, payload in sorted(
            latest_by_reviewer.get(queue_id, {}).items()
        )
        if name != reviewer
    ]
    if not entries:
        return (
            "<div style='background:#fff;border:1px solid #dde6f4;"
            "border-radius:8px;box-shadow:0 14px 38px "
            "rgba(42,55,105,0.12);padding:14px;'>"
            "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
            "margin:0 0 8px;'>Other Reviewer Decisions</h2>"
            "<p>No other reviewer decisions for this sample.</p>"
            "</div>"
        )

    rows = []
    for name, payload in entries:
        tags = ", ".join(
            str(tag) for tag in payload.get("morphology_tags", [])
        )
        rows.append(
            "<tr>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(name)}</td>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(str(payload.get('reviewer_label', '')))}</td>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(tags)}</td>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(str(payload.get('notes', '')))}</td>"
            f"<td style='{_TABLE_LAST_CELL_STYLE}'>"
            f"{html.escape(str(payload.get('timestamp_utc', '')))}</td>"
            "</tr>"
        )
    return (
        "<div style='background:#fff;border:1px solid #dde6f4;"
        "border-radius:8px;box-shadow:0 14px 38px "
        "rgba(42,55,105,0.12);padding:14px;'>"
        "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
        "margin:0 0 8px;'>Other Reviewer Decisions</h2>"
        "<table style='border-collapse:collapse;color:#07145c;"
        "font-size:13px;width:100%;'>"
        f"<tr><th style='{_TABLE_HEADER_STYLE}'>Reviewer</th>"
        f"<th style='{_TABLE_HEADER_STYLE}'>Label</th>"
        f"<th style='{_TABLE_HEADER_STYLE}'>Tags</th>"
        f"<th style='{_TABLE_HEADER_STYLE}'>Notes</th>"
        f"<th style='{_TABLE_LAST_HEADER_STYLE}'>Timestamp</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _entity_other_reviews_html(
    item: dict[str, Any],
    latest_by_reviewer: dict[str, dict[str, dict[str, Any]]],
    *,
    reviewer: str,
) -> str:
    queue_id = str(item["queue_id"])
    entries = [
        (name, payload)
        for name, payload in sorted(
            latest_by_reviewer.get(queue_id, {}).items()
        )
        if name != reviewer
    ]
    if not entries:
        return (
            "<div style='background:#fff;border:1px solid #dde6f4;"
            "border-radius:8px;box-shadow:0 14px 38px "
            "rgba(42,55,105,0.12);padding:14px;'>"
            "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
            "margin:0 0 8px;'>Other Reviewer Decisions</h2>"
            "<p>No other entity-review decisions for this sample.</p>"
            "</div>"
        )

    rows = []
    for name, payload in entries:
        rows.append(
            "<tr>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(name)}</td>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(str(payload.get('entity_label', '')))}</td>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(str(payload.get('confidence', '')))}</td>"
            f"<td style='{_TABLE_CELL_STYLE}'>"
            f"{html.escape(str(payload.get('notes', '')))}</td>"
            f"<td style='{_TABLE_LAST_CELL_STYLE}'>"
            f"{html.escape(str(payload.get('timestamp_utc', '')))}</td>"
            "</tr>"
        )
    return (
        "<div style='background:#fff;border:1px solid #dde6f4;"
        "border-radius:8px;box-shadow:0 14px 38px "
        "rgba(42,55,105,0.12);padding:14px;'>"
        "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
        "margin:0 0 8px;'>Other Reviewer Decisions</h2>"
        "<table style='border-collapse:collapse;color:#07145c;"
        "font-size:13px;width:100%;'>"
        f"<tr><th style='{_TABLE_HEADER_STYLE}'>Reviewer</th>"
        f"<th style='{_TABLE_HEADER_STYLE}'>Entity Label</th>"
        f"<th style='{_TABLE_HEADER_STYLE}'>Confidence</th>"
        f"<th style='{_TABLE_HEADER_STYLE}'>Notes</th>"
        f"<th style='{_TABLE_LAST_HEADER_STYLE}'>Timestamp</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _initial_review_index_from_query(
    doc: Any,
    queue: list[dict[str, Any]],
    *,
    fallback_index: int,
) -> tuple[int, str | None]:
    raw = _first_query_argument(doc, "s", "sample_index")
    if raw is None:
        return fallback_index, None
    try:
        sample_index = int(raw)
    except ValueError:
        return (
            fallback_index,
            f"Ignoring invalid sample query value {raw!r}; "
            "showing saved queue position instead.",
        )

    for queue_index, item in enumerate(queue):
        if int(item["sample_index"]) == sample_index:
            return queue_index, None
    return (
        fallback_index,
        f"Requested sample_index {sample_index} is not in this review queue; "
        "showing saved queue position instead.",
    )


def _first_query_argument(doc: Any, *names: str) -> str | None:
    session_context = getattr(doc, "session_context", None)
    request = getattr(session_context, "request", None)
    arguments = getattr(request, "arguments", None)
    if not isinstance(arguments, dict):
        return None
    for name in names:
        if name not in arguments:
            continue
        values = arguments[name]
        if isinstance(values, bytes | str):
            raw = values
        else:
            values = list(values)
            if not values:
                continue
            raw = values[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        value = str(raw).strip()
        if value:
            return value
    return None


def _pixel_inspector_callback(
    custom_js_type: Any,
    *,
    labels: list[str],
    sources: list[Any],
    mappers: list[Any],
    label_sources: list[Any],
    selection_source: Any,
    stamp_width: int,
    stamp_height: int,
) -> Any:
    return custom_js_type(
        args={
            "labels": labels,
            "sources": sources,
            "mappers": mappers,
            "label_sources": label_sources,
            "selection_source": selection_source,
            "stamp_width": stamp_width,
            "stamp_height": stamp_height,
        },
        code="""
const col = Math.floor(cb_obj.x);
const row = Math.floor(cb_obj.y);
if (
  !Number.isFinite(col) || !Number.isFinite(row) ||
  col < 0 || row < 0 || col >= stamp_width || row >= stamp_height
) {
  return;
}

const currentX = selection_source.data.x || [];
const currentY = selection_source.data.y || [];
const samePixel = (
  currentX.length > 0 && currentY.length > 0 &&
  Math.floor(currentX[0]) === col &&
  Math.floor(currentY[0]) === row
);
if (samePixel) {
  selection_source.data = {x: [], y: [], width: [], height: []};
  selection_source.change.emit();
  for (const source of label_sources) {
    source.data = {x: [], y: [], x_offset: [], y_offset: [], text: []};
    source.change.emit();
  }
  return;
}

selection_source.data = {
  x: [col + 0.5],
  y: [row + 0.5],
  width: [1],
  height: [1],
};
selection_source.change.emit();

function formatValue(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "nan";
  }
  const numeric = Number(value);
  const magnitude = Math.abs(numeric);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) {
    return numeric.toExponential(4);
  }
  return numeric.toFixed(4);
}

function scalar(source, name) {
  const values = source.data[name];
  if (!values || !values.length) {
    return NaN;
  }
  return Number(values[0]);
}

function pixelValue(source) {
  const flatColumn = source.data.flat;
  if (!flatColumn || !flatColumn.length) {
    return NaN;
  }
  const flat = flatColumn[0];
  const index = row * stamp_width + col;
  if (flat && typeof flat.get === "function") {
    return flat.get(index);
  }
  return flat[index];
}

function robustSigma(source, value) {
  const center = scalar(source, "robust_center");
  const scale = scalar(source, "robust_sigma");
  if (!Number.isFinite(center) || !Number.isFinite(scale) || scale <= 0) {
    return NaN;
  }
  return (Number(value) - center) / scale;
}

const xOffset = col > stamp_width * 0.58 ? -150 : 10;
const yOffset = row > stamp_height * 0.58 ? -78 : 10;
for (let index = 0; index < labels.length; index += 1) {
  const value = pixelValue(sources[index]);
  const sigma = robustSigma(sources[index], value);
  const mapper = mappers[index];
  label_sources[index].data = {
    x: [col + 0.5],
    y: [row + 0.5],
    x_offset: [xOffset],
    y_offset: [yOffset],
    text: [
      `${labels[index]}\n`
      + `(x,y): ${col}, ${row}\n`
      + `value: ${formatValue(value)}\n`
      + `robust sigma: ${formatValue(sigma)}\n`
      + `stretch: ${formatValue(mapper.low)} .. ${formatValue(mapper.high)}`,
    ],
  };
  label_sources[index].change.emit();
}
""",
    )


def _pixel_label_source(column_data_source_type: Any) -> Any:
    return column_data_source_type(
        {"x": [], "y": [], "x_offset": [], "y_offset": [], "text": []},
        name="review-pixel-label",
    )


def _pixel_label_set(label_set_type: Any, source: Any) -> Any:
    return label_set_type(
        x="x",
        y="y",
        text="text",
        x_offset="x_offset",
        y_offset="y_offset",
        source=source,
        background_fill_color="#172026",
        background_fill_alpha=0.92,
        border_line_color="#394854",
        border_line_alpha=0.95,
        border_line_width=1,
        text_color="#f5fbff",
        text_font_size="10px",
        text_line_height=1.15,
        text_baseline="top",
    )


def _image_payload(image: np.ndarray) -> dict[str, Any]:
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape
    center, sigma = _robust_pixel_stats(array)
    return {
        "image": [array],
        "flat": [array.ravel()],
        "robust_center": [center],
        "robust_sigma": [sigma],
        "x": [0],
        "y": [0],
        "dw": [width],
        "dh": [height],
    }


def _robust_pixel_stats(array: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = math.nan
    return center, sigma


def _update_image(
    source,
    mapper,
    image: np.ndarray,
    *,
    diverging: bool,
) -> tuple[float, float]:
    array = np.asarray(image, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size:
        if diverging:
            limit = float(np.percentile(np.abs(finite), 98))
            limit = max(limit, 1e-6)
            mapper.low = -limit
            mapper.high = limit
        else:
            low, high = np.percentile(finite, [2, 98])
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low = float(np.min(finite))
                high = float(np.max(finite) + 1e-6)
            mapper.low = float(low)
            mapper.high = float(high)
    source.data = _image_payload(array)
    return float(mapper.low), float(mapper.high)


def _render_review_contact_sheet_page(
    *,
    page_path: Path,
    page_index: int,
    page_items: list[dict[str, Any]],
    search: np.ndarray,
    template: np.ndarray,
    difference: np.ndarray | None,
    columns: int,
    stamp_size: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for review-contact-sheet; run "
            "'uv sync --extra viz' for development or install "
            "'cuphoton[viz]'"
        ) from exc

    if not page_items:
        return
    columns = min(columns, len(page_items))
    rows = int(math.ceil(len(page_items) / columns))
    margin = 16
    gap = 12
    pad = 8
    panel_gap = 4
    title_h = 34
    header_h = 46
    panel_label_h = 16
    tile_w = 2 * pad + (3 * stamp_size) + (2 * panel_gap)
    tile_h = 2 * pad + header_h + panel_label_h + stamp_size
    page_w = 2 * margin + columns * tile_w + (columns - 1) * gap
    page_h = 2 * margin + title_h + rows * tile_h + (rows - 1) * gap

    image = Image.new("RGB", (page_w, page_h), color=(248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    strong_font = ImageFont.load_default()
    title = f"XScan Review Contact Sheet - Page {page_index}"
    draw.text((margin, margin), title, fill=(15, 23, 42), font=strong_font)

    y0 = margin + title_h
    for item_index, item in enumerate(page_items):
        row = item_index // columns
        column = item_index % columns
        x = margin + column * (tile_w + gap)
        y = y0 + row * (tile_h + gap)
        _draw_contact_sheet_item(
            image=image,
            draw=draw,
            font=font,
            item=item,
            search=search,
            template=template,
            difference=difference,
            x=x,
            y=y,
            tile_w=tile_w,
            tile_h=tile_h,
            pad=pad,
            panel_gap=panel_gap,
            header_h=header_h,
            panel_label_h=panel_label_h,
            stamp_size=stamp_size,
        )

    image.save(page_path)


def _draw_contact_sheet_item(
    *,
    image: Any,
    draw: Any,
    font: Any,
    item: dict[str, Any],
    search: np.ndarray,
    template: np.ndarray,
    difference: np.ndarray | None,
    x: int,
    y: int,
    tile_w: int,
    tile_h: int,
    pad: int,
    panel_gap: int,
    header_h: int,
    panel_label_h: int,
    stamp_size: int,
) -> None:
    draw.rounded_rectangle(
        (x, y, x + tile_w, y + tile_h),
        radius=6,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
        width=1,
    )
    sample_index = int(item["sample_index"])
    candidate_id = item.get("candidate_id") or item.get("queue_id", "")
    rank = item.get("rank")
    rank_text = f"#{rank}" if rank is not None else f"item {sample_index}"
    header_lines = [
        f"{rank_text}  sample {sample_index}",
        _short_contact_sheet_text(candidate_id, max_chars=42),
        _contact_sheet_meta_line(item),
    ]
    line_y = y + pad
    for line in header_lines:
        draw.text((x + pad, line_y), line, fill=(15, 23, 42), font=font)
        line_y += 13

    diff_image = (
        np.asarray(difference[sample_index])
        if difference is not None
        else np.asarray(search[sample_index])
        - np.asarray(template[sample_index])
    )
    panels = (
        ("Search", np.asarray(search[sample_index]), False),
        ("Template", np.asarray(template[sample_index]), False),
        ("Difference", diff_image, True),
    )
    panel_y = y + pad + header_h
    stamp_y = panel_y + panel_label_h
    panel_x = x + pad
    for title, stamp, diverging in panels:
        draw.text((panel_x, panel_y), title, fill=(71, 85, 105), font=font)
        stamp_image = _contact_sheet_stamp_image(
            stamp,
            diverging=diverging,
            size=stamp_size,
        )
        image.paste(stamp_image, (panel_x, stamp_y))
        draw.rectangle(
            (
                panel_x,
                stamp_y,
                panel_x + stamp_size,
                stamp_y + stamp_size,
            ),
            outline=(148, 163, 184),
            width=1,
        )
        panel_x += stamp_size + panel_gap


def _contact_sheet_stamp_image(
    stamp: np.ndarray,
    *,
    diverging: bool,
    size: int,
) -> Any:
    from PIL import Image

    array = _contact_sheet_2d_array(stamp)
    if diverging:
        rgb = _diverging_contact_sheet_rgb(array)
    else:
        gray = _scaled_contact_sheet_gray(array)
        rgb = np.stack([gray, gray, gray], axis=-1)
    result = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    return result.resize((size, size), resampling)


def _contact_sheet_2d_array(stamp: np.ndarray) -> np.ndarray:
    array = np.asarray(stamp, dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(
            "review-contact-sheet requires 2D stamp arrays after squeeze"
        )
    return array


def _scaled_contact_sheet_gray(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return np.full(array.shape, 128, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _diverging_contact_sheet_rgb(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    limit = float(np.percentile(np.abs(finite), 98))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.max(np.abs(finite)))
    if not np.isfinite(limit) or limit <= 0:
        return np.full((*array.shape, 3), 128, dtype=np.uint8)
    scaled = np.clip(array / limit, -1.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    positive = np.clip(scaled, 0.0, 1.0)
    negative = np.clip(-scaled, 0.0, 1.0)
    red = 128 + 127 * positive - 96 * negative
    green = 128 - 96 * (positive + negative)
    blue = 128 + 127 * negative - 96 * positive
    return np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8)


def _contact_sheet_meta_line(item: dict[str, Any]) -> str:
    parts = []
    label = item.get("label")
    if label is not None:
        parts.append(f"label={label}")
    probability = item.get("probability")
    if probability is not None:
        try:
            parts.append(f"p={float(probability):.3f}")
        except (TypeError, ValueError):
            parts.append(f"p={probability}")
    rank_reason = item.get("rank_reason")
    if rank_reason:
        parts.append(str(rank_reason))
    return _short_contact_sheet_text("  ".join(parts), max_chars=54)


def _short_contact_sheet_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _contact_sheet_item_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "queue_id": item.get("queue_id"),
        "sample_index": item.get("sample_index"),
        "candidate_id": item.get("candidate_id"),
        "rank": item.get("rank"),
        "rank_reason": item.get("rank_reason"),
        "label": item.get("label"),
        "label_source": item.get("label_source"),
        "probability": item.get("probability"),
    }


_METADATA_LABEL_ACRONYMS = {
    "dec",
    "dia",
    "id",
    "psf",
    "ra",
    "snr",
    "utc",
}


def _metadata_label(key: str) -> str:
    words = key.split("_")
    return " ".join(
        (
            word.upper()
            if word.lower() in _METADATA_LABEL_ACRONYMS
            else word.capitalize()
        )
        for word in words
    )


def _metadata_table(item: dict[str, Any]) -> str:
    keys = [
        "candidate_id",
        "sample_index",
        "label",
        "prediction",
        "probability",
        "logit",
        "rank",
        "rank_reason",
        "hybrid_score",
        "known_error",
        "center_source",
        "catalog_pool_role",
        "catalog_extendedness",
        "catalog_flux",
        "positive_quality_stratum",
        "center_offset_radius",
        "search_valid_fraction",
        "difference_context_valid_fraction",
        "review_stratum",
        "source_review_dir",
        "source_queue_id",
        "source_sample_index",
        "source_candidate_id",
        "source_reviewer",
        "source_binary_label",
        "source_binary_morphology_tags",
        "source_binary_notes",
        "source_binary_timestamp_utc",
        "source_label",
        "source_probability",
        "source_rank_reason",
    ]
    card_style = (
        "background:#fff;border:1px solid #dde6f4;border-radius:8px;"
        "box-shadow:0 14px 38px rgba(42,55,105,0.12);box-sizing:border-box;"
        "padding:10px 12px;width:1400px;"
    )
    heading_style = (
        "color:#07145c;font-size:15px;font-weight:760;margin:0 0 6px;"
    )
    grid_style = (
        "column-gap:20px;color:#07145c;display:grid;font-size:12px;"
        "grid-template-columns:repeat(3,minmax(0,1fr));row-gap:0;"
        "width:100%;"
    )
    field_style = (
        "border-top:1px solid #edf1f7;display:grid;gap:8px;"
        "grid-template-columns:minmax(112px,42%) minmax(0,1fr);"
        "line-height:1.25;padding:4px 0;"
    )
    key_style = "color:#7080b5;font-weight:650;"
    value_style = "color:#07145c;min-width:0;overflow-wrap:anywhere;"
    rows = []
    for key in keys:
        if key not in item:
            continue
        rows.append(
            f"<div class='summary-field' style='{field_style}'>"
            f"<div class='summary-key' style='{key_style}'>"
            f"{html.escape(_metadata_label(key))}</div>"
            f"<div class='summary-value' style='{value_style}'>"
            f"{_fmt_metadata_value(item[key])}</div>"
            "</div>"
        )
    return (
        f"<div class='summary-card' style='{card_style}'>"
        f"<h2 style='{heading_style}'>Detection Summary</h2>"
        f"<div class='summary-grid' style='{grid_style}'>"
        + "".join(rows)
        + "</div></div>"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_metadata_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if abs(value) >= 1000 or (0 < abs(value) < 0.0001):
            return html.escape(f"{value:.4e}")
        return html.escape(f"{value:.4f}")
    return html.escape(str(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_dumps(payload: Any) -> str:
    return json.dumps(_jsonable(payload), indent=2, sort_keys=True)


def _json_line(payload: Any) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True)
