# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy binary metrics for XScan runs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


def ensure_finite_scores(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    invalid = int(np.size(array) - np.isfinite(array).sum())
    if invalid > 0:
        raise ValueError(f"{name} contains {invalid} non-finite values")
    return array


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


class UndefinedMetricError(ValueError):
    """Raised when a metric is undefined for the given input.

    Notably for single-class input, where ROC/PR/AUC are not defined and must
    not be reported as a fabricated 1.0/0.0.
    """


def ensure_binary_labels(
    y_true: np.ndarray, *, scores: np.ndarray | None = None
) -> np.ndarray:
    """Validate a binary label array before any metric computation.

    Rejects non-1-D, empty, non-finite, or non-{0,1} labels (and, when given,
    a score array of a different length) so cumulative counts cannot silently
    disagree with the reported class totals (e.g. y=[0,0,2]).
    """
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise UndefinedMetricError("labels must be one-dimensional")
    if labels.size == 0:
        raise UndefinedMetricError("labels must be non-empty")
    if not np.all(np.isfinite(labels)):
        raise UndefinedMetricError("labels must be finite")
    if not (
        np.all(labels == np.round(labels)) and np.all(np.isin(labels, (0, 1)))
    ):
        raise UndefinedMetricError("labels must be binary (0 or 1)")
    if scores is not None:
        scores_shape = np.asarray(scores).shape
        # A (N, 1) score array shares the first axis with an (N,) label array
        # but broadcasts against it to (N, N); reject anything non-1-D so the
        # confusion/curve math cannot silently operate on a broadcast product.
        if len(scores_shape) != 1 or scores_shape[0] != labels.size:
            raise UndefinedMetricError(
                "scores must be one-dimensional and match the label length"
            )
    return labels.astype(np.int64)


def _binary_scalar_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return int(value)
    return None


def ensure_metadata_labels_match(
    metadata_rows: list[dict[str, Any]], y_true: np.ndarray
) -> None:
    """Assert each metadata row's label is binary and equals y_true.

    Grouped/consensus/binned metrics read row["label"] directly, so a mismatch
    with the (already validated) y_true would silently mix two ground truths.
    """
    if len(metadata_rows) != int(np.size(y_true)):
        raise UndefinedMetricError(
            "metadata_rows and labels must have equal length"
        )
    for index, row in enumerate(metadata_rows):
        meta_label = _binary_scalar_or_none(row.get("label"))
        if meta_label is None:
            raise UndefinedMetricError(
                f"metadata row {index} has a missing or non-binary label"
            )
        if meta_label != int(y_true[index]):
            raise UndefinedMetricError(
                f"metadata label at row {index} ({row.get('label')!r}) "
                f"disagrees with the evaluation label ({int(y_true[index])})"
            )


def roc_curve(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = ensure_binary_labels(y_true, scores=scores)
    scores = ensure_finite_scores(scores, name="scores")
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        raise UndefinedMetricError(
            "ROC curve is undefined for single-class input"
        )
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = scores[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    distinct = np.r_[score_sorted[:-1] != score_sorted[1:], True]
    tp = tp[distinct]
    fp = fp[distinct]
    thresholds = score_sorted[distinct]
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]
    return fpr, tpr, thresholds


def precision_recall_curve(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = ensure_binary_labels(y_true, scores=scores)
    scores = ensure_finite_scores(scores, name="scores")
    positives = int(y_true.sum())
    if positives == 0 or positives == y_true.size:
        raise UndefinedMetricError(
            "precision-recall curve is undefined for single-class input"
        )
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = scores[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    distinct = np.r_[score_sorted[:-1] != score_sorted[1:], True]
    tp = tp[distinct]
    fp = fp[distinct]
    thresholds = score_sorted[distinct]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.r_[1.0, precision, y_true.mean() if y_true.size else 0.0]
    recall = np.r_[0.0, recall, 1.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]
    return precision, recall, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(
        np.trapezoid(
            np.asarray(y, dtype=np.float64),
            np.asarray(x, dtype=np.float64),
        )
    )


def tpr_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float
) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = tpr[fpr <= target_fpr]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def confusion_from_scores(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    y_true = ensure_binary_labels(y_true, scores=scores)
    pred = (ensure_finite_scores(scores, name="scores") >= threshold).astype(
        np.int64
    )
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=np.float64)
    probabilities = ensure_finite_scores(scores, name="scores")
    return (
        float(np.mean(np.square(probabilities - labels)))
        if labels.size
        else 0.0
    )


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = ensure_finite_scores(scores, name="scores")
    confusion = confusion_from_scores(
        labels, probabilities, threshold=threshold
    )
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    sample_count = int(labels.size)
    positive_count = int(tp + fn)
    negative_count = int(tn + fp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(positive_count, 1)
    specificity = tn / max(negative_count, 1)
    fpr = fp / max(negative_count, 1)
    fnr = fn / max(positive_count, 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "accuracy": float((tp + tn) / max(sample_count, 1)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    probabilities = ensure_finite_scores(scores, name="scores")
    unique_scores = np.unique(np.clip(probabilities, 0.0, 1.0))
    candidates = [0.0, 0.5, 1.0]
    candidates.extend(float(value) for value in unique_scores)
    if unique_scores.size > 1:
        candidates.extend(
            float(value)
            for value in (unique_scores[:-1] + unique_scores[1:]) / 2.0
        )
    return np.asarray(sorted(set(candidates)), dtype=np.float64)


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    labels = ensure_binary_labels(y_true, scores=scores)
    if np.unique(labels).size < 2:
        raise UndefinedMetricError(
            "threshold sweep requires both positive and negative labels"
        )
    return [
        threshold_metrics(labels, scores, threshold=float(threshold))
        for threshold in candidate_thresholds(scores)
    ]


def select_threshold_for_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str = "accuracy",
    source_split: str | None = None,
) -> dict[str, Any]:
    rows = threshold_sweep(y_true, scores)
    if not rows:
        raise ValueError("cannot select a threshold from an empty sweep")
    if metric not in rows[0]:
        raise ValueError(f"unknown threshold selection metric: {metric}")
    best = max(
        rows,
        key=lambda row: (
            float(row[metric]),
            float(row["balanced_accuracy"]),
            float(row["f1"]),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )
    selected = dict(best)
    selected["selection_metric"] = metric
    selected["selection_metric_value"] = float(best[metric])
    if source_split is not None:
        selected["source_split"] = source_split
    return selected


def reliability_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = ensure_finite_scores(scores, name="scores")
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include_upper = bool(upper == edges[-1])
        mask = (probabilities >= lower) & (
            probabilities <= upper if include_upper else probabilities < upper
        )
        if not np.any(mask):
            continue
        subset_scores = probabilities[mask]
        subset_labels = labels[mask]
        mean_probability = float(np.mean(subset_scores))
        empirical_rate = float(np.mean(subset_labels))
        error = mean_probability - empirical_rate
        rows.append(
            {
                "bin_start": float(lower),
                "bin_end": float(upper),
                "count": int(mask.sum()),
                "mean_probability": mean_probability,
                "empirical_positive_rate": empirical_rate,
                "calibration_error": float(error),
                "absolute_calibration_error": float(abs(error)),
            }
        )
    return rows


def calibration_summary(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    rows = reliability_curve(y_true, scores, bins=bins)
    sample_count = int(np.asarray(y_true).size)
    expected_error = sum(
        (row["count"] / max(sample_count, 1))
        * row["absolute_calibration_error"]
        for row in rows
    )
    max_error = max(
        (row["absolute_calibration_error"] for row in rows),
        default=0.0,
    )
    return {
        "brier_score": brier_score(y_true, scores),
        "expected_calibration_error": float(expected_error),
        "max_calibration_error": float(max_error),
        "bin_count": int(bins),
        "populated_bin_count": int(len(rows)),
    }


def threshold_diagnostics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fixed_threshold: float,
    validation_selection: dict[str, Any] | None = None,
    validation_selection_undefined_reason: str | None = None,
) -> dict[str, Any]:
    labels = ensure_binary_labels(y_true, scores=scores)
    fixed = threshold_metrics(labels, scores, threshold=fixed_threshold)
    result: dict[str, Any] = {"fixed_threshold": fixed}
    if np.unique(labels).size < 2:
        result["split_optimal_undefined_reason"] = "single_class_split"
    else:
        result["split_optimal"] = select_threshold_for_scores(
            labels, scores, metric="accuracy"
        )
    if validation_selection is not None:
        threshold = float(validation_selection["threshold"])
        evaluated_metrics = threshold_metrics(
            labels,
            scores,
            threshold=threshold,
        )
        result["validation_selected"] = {
            "source_split": validation_selection.get("source_split", "val"),
            "selection_metric": validation_selection.get(
                "selection_metric",
                "accuracy",
            ),
            "source_threshold_metrics": validation_selection,
            "evaluated_split_metrics": evaluated_metrics,
        }
    if validation_selection_undefined_reason is not None:
        result["validation_selection_undefined_reason"] = (
            validation_selection_undefined_reason
        )
    return result


def evaluate_predictions(
    *,
    y_true: np.ndarray,
    logits: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    threshold: float = 0.5,
    threshold_selection: dict[str, Any] | None = None,
    threshold_selection_undefined_reason: str | None = None,
    include_diagnostics: bool | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    # Reject empty / non-binary / mislengthed input up front (empty splits
    # must not fabricate "single_class" diagnostics; y=[0,0,2] must not pass).
    y_true = ensure_binary_labels(y_true, scores=logits)
    # Grouped/consensus/binned metrics below read row["label"] independently
    # of y_true; bind them so overall and grouped metrics cannot describe
    # contradictory ground truth (mirrors the train-time provenance check).
    ensure_metadata_labels_match(metadata_rows, y_true)
    logits = ensure_finite_scores(logits, name="logits")
    probs = ensure_finite_scores(sigmoid(logits), name="probabilities")
    confusion = confusion_from_scores(y_true, probs, threshold=threshold)
    if include_diagnostics is None:
        include_diagnostics = (
            output_dir is not None or threshold_selection is not None
        )
    sample_count = int(len(y_true))
    positive_count = int(np.sum(y_true))
    # ROC/PR AUC and TPR@FPR are undefined without both classes present.
    # Report JSON null (None) with a reason and skip the curves, rather than a
    # fabricated 1.0/0.0; callers must exclude undefined runs from ranking.
    both_classes = 0 < positive_count < sample_count
    if both_classes:
        fpr, tpr, _ = roc_curve(y_true, probs)
        precision, recall, _ = precision_recall_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        tpr_1pct = tpr_at_fpr(y_true, probs, 0.01)
        tpr_5pct = tpr_at_fpr(y_true, probs, 0.05)
        metric_undefined_reason = None
    else:
        roc_auc = pr_auc = tpr_1pct = tpr_5pct = None
        metric_undefined_reason = "single_class_split"
    result = {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": int(np.sum(1 - np.asarray(y_true))),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "metric_undefined_reason": metric_undefined_reason,
        "brier_score": brier_score(y_true, probs),
        "threshold": threshold,
        "confusion": confusion,
        "accuracy": float(
            (confusion["tp"] + confusion["tn"]) / max(len(y_true), 1)
        ),
        "tpr_at_fpr_1pct": tpr_1pct,
        "tpr_at_fpr_5pct": tpr_5pct,
        "binned": {
            "magnitude_proxy": binned_positive_rate(
                metadata_rows,
                probs,
                field="magnitude_proxy",
                positive_only=True,
                bins=8,
                threshold=threshold,
            ),
            "snr": binned_positive_rate(
                metadata_rows,
                probs,
                field="snr",
                positive_only=True,
                bins=8,
                threshold=threshold,
            ),
            "flux_ratio": binned_positive_rate(
                metadata_rows,
                probs,
                field="flux_ratio",
                positive_only=False,
                bins=8,
                threshold=threshold,
            ),
            "catalog_flux": binned_positive_rate(
                metadata_rows,
                probs,
                field="catalog_flux",
                positive_only=False,
                bins=8,
                threshold=threshold,
            ),
            "catalog_flux_log10": binned_positive_rate(
                metadata_rows,
                probs,
                field="catalog_flux",
                positive_only=False,
                bins=8,
                threshold=threshold,
                transform=positive_log10_or_none,
            ),
            "catalog_extendedness": binned_positive_rate(
                metadata_rows,
                probs,
                field="catalog_extendedness",
                positive_only=False,
                bins=4,
                threshold=threshold,
            ),
            "center_offset_radius": binned_positive_rate(
                metadata_rows,
                probs,
                field="center_offset_radius",
                positive_only=False,
                bins=6,
                threshold=threshold,
            ),
            "search_valid_fraction": binned_positive_rate(
                metadata_rows,
                probs,
                field="search_valid_fraction",
                positive_only=False,
                bins=5,
                threshold=threshold,
            ),
            "difference_context_valid_fraction": binned_positive_rate(
                metadata_rows,
                probs,
                field="difference_context_valid_fraction",
                positive_only=False,
                bins=5,
                threshold=threshold,
            ),
        },
    }
    if threshold_selection_undefined_reason is not None:
        result["threshold_selection_undefined_reason"] = (
            threshold_selection_undefined_reason
        )
    if include_diagnostics:
        result["calibration"] = calibration_summary(y_true, probs)
        result["threshold_diagnostics"] = threshold_diagnostics(
            y_true,
            probs,
            fixed_threshold=threshold,
            validation_selection=threshold_selection,
            validation_selection_undefined_reason=(
                threshold_selection_undefined_reason
            ),
        )
    consensus = consensus_metrics(
        metadata_rows=metadata_rows,
        probabilities=probs,
        threshold=threshold,
    )
    if consensus is not None:
        result["consensus"] = consensus

    autoscan = autoscan_baseline_metrics(
        metadata_rows=metadata_rows,
        threshold=threshold,
    )
    if autoscan is not None:
        result["autoscan_baseline"] = autoscan

    center_source = grouped_source_metrics(
        metadata_rows=metadata_rows,
        probabilities=probs,
        threshold=threshold,
        field="center_source",
    )
    if center_source is not None:
        result["center_source_breakdown"] = center_source

    catalog_pool_role = grouped_source_metrics(
        metadata_rows=metadata_rows,
        probabilities=probs,
        threshold=threshold,
        field="catalog_pool_role",
    )
    if catalog_pool_role is not None:
        result["catalog_pool_role_breakdown"] = catalog_pool_role

    catalog_morphology = grouped_derived_metrics(
        metadata_rows=metadata_rows,
        probabilities=probs,
        threshold=threshold,
        field="catalog_morphology",
        resolver=resolve_catalog_morphology,
    )
    if catalog_morphology is not None:
        result["catalog_morphology_breakdown"] = catalog_morphology

    negative_difficulty = grouped_derived_metrics(
        metadata_rows=metadata_rows,
        probabilities=probs,
        threshold=threshold,
        field="negative_difficulty",
        resolver=resolve_negative_difficulty,
        label_filter=0,
    )
    if negative_difficulty is not None:
        result["negative_difficulty_breakdown"] = negative_difficulty

    mask_pressure = grouped_derived_metrics(
        metadata_rows=metadata_rows,
        probabilities=probs,
        threshold=threshold,
        field="mask_pressure",
        resolver=resolve_mask_pressure,
    )
    if mask_pressure is not None:
        result["mask_pressure_breakdown"] = mask_pressure

    if output_dir is not None:
        write_prediction_artifacts(
            output_dir,
            metrics=result,
            metadata_rows=metadata_rows,
            y_true=np.asarray(y_true, dtype=np.int64),
            probabilities=probs,
            logits=np.asarray(logits, dtype=np.float64),
        )
    return result


def binned_positive_rate(
    metadata_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    field: str,
    positive_only: bool,
    bins: int,
    threshold: float,
    transform: Callable[[float], float | None] | None = None,
) -> list[dict[str, Any]]:
    values = []
    labels = []
    scores = []
    for row, score in zip(metadata_rows, probabilities, strict=True):
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        label = int(row["label"])
        if positive_only and label != 1:
            continue
        numeric = float(value)
        if transform is not None:
            transformed = transform(numeric)
            if transformed is None or not math.isfinite(transformed):
                continue
            numeric = transformed
        values.append(numeric)
        labels.append(label)
        scores.append(float(score))
    if not values:
        return []
    values_arr = np.asarray(values, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if np.allclose(values_arr.min(), values_arr.max()):
        edges = np.array([values_arr.min(), values_arr.max() + 1e-6])
    else:
        edges = np.linspace(values_arr.min(), values_arr.max(), bins + 1)
    result: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include_upper = upper == edges[-1]
        mask = (values_arr >= lower) & (
            values_arr <= upper if include_upper else values_arr < upper
        )
        if not np.any(mask):
            continue
        subset_scores = scores_arr[mask]
        subset_labels = labels_arr[mask]
        positives = subset_scores[subset_labels == 1]
        negatives = subset_scores[subset_labels == 0]
        row = {
            "bin_start": float(lower),
            "bin_end": float(upper),
            "count": int(mask.sum()),
        }
        if positives.size:
            row["recovery_rate"] = float(np.mean(positives >= threshold))
        if negatives.size:
            row["false_positive_rate"] = float(
                np.mean(negatives >= threshold)
            )
        result.append(row)
    return result


def positive_log10_or_none(value: float) -> float | None:
    if value <= 0.0:
        return None
    return float(math.log10(value))


def consensus_metrics(
    *,
    metadata_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any] | None:
    grouped: dict[Any, list[float]] = {}
    for row, prob in zip(metadata_rows, probabilities, strict=True):
        if int(row["label"]) != 1:
            continue
        fake_id = row.get("fake_id")
        if fake_id is None:
            continue
        grouped.setdefault(fake_id, []).append(float(prob))

    repeated = {
        key: values for key, values in grouped.items() if len(values) >= 1
    }
    if not repeated:
        return None

    n_shot = np.mean(
        [max(values) >= threshold for values in repeated.values()]
    )
    majority = np.mean(
        [
            np.mean(np.asarray(values) >= threshold) >= 0.5
            for values in repeated.values()
        ]
    )
    return {
        "group_field": "fake_id",
        "positive_group_count": int(len(repeated)),
        "n_shot_recovery_rate": float(n_shot),
        "majority_consensus_recovery_rate": float(majority),
    }


def autoscan_baseline_metrics(
    *,
    metadata_rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    positives = [row for row in metadata_rows if int(row["label"]) == 1]
    if not positives or all(
        row.get("autoscan_score") is None for row in positives
    ):
        return None

    positive_scores = np.asarray(
        [
            float(row["autoscan_score"])
            for row in positives
            if row.get("autoscan_score") is not None
        ],
        dtype=np.float64,
    )
    if positive_scores.size == 0:
        return None

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "stamp_recovery_rate": float(np.mean(positive_scores >= threshold)),
    }

    grouped: dict[Any, list[float]] = {}
    for row in positives:
        fake_id = row.get("fake_id")
        score = row.get("autoscan_score")
        if fake_id is None or score is None:
            continue
        grouped.setdefault(fake_id, []).append(float(score))

    if grouped:
        result["positive_group_count"] = int(len(grouped))
        result["n_shot_recovery_rate"] = float(
            np.mean([max(values) >= threshold for values in grouped.values()])
        )
        result["majority_consensus_recovery_rate"] = float(
            np.mean(
                [
                    np.mean(np.asarray(values) >= threshold) >= 0.5
                    for values in grouped.values()
                ]
            )
        )
    return result


def grouped_source_metrics(
    *,
    metadata_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    field: str,
) -> dict[str, Any] | None:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row, prob in zip(metadata_rows, probabilities, strict=True):
        value = row.get(field)
        if value is None:
            continue
        grouped.setdefault(str(value), []).append(
            (int(row["label"]), float(prob))
        )

    if not grouped:
        return None

    result: dict[str, Any] = {
        "field": field,
        "groups": {},
    }
    for value in sorted(grouped):
        entries = grouped[value]
        labels = np.asarray([label for label, _ in entries], dtype=np.int64)
        scores = np.asarray([score for _, score in entries], dtype=np.float64)
        positives = scores[labels == 1]
        negatives = scores[labels == 0]
        predictions = (scores >= threshold).astype(np.int64)
        result["groups"][value] = {
            "count": int(labels.size),
            "positive_count": int(np.sum(labels == 1)),
            "negative_count": int(np.sum(labels == 0)),
            "mean_probability": (
                float(np.mean(scores)) if scores.size else 0.0
            ),
            "accuracy": (
                float(np.mean(predictions == labels)) if labels.size else 0.0
            ),
            "positive_recovery_rate": (
                float(np.mean(positives >= threshold))
                if positives.size
                else None
            ),
            "negative_false_positive_rate": (
                float(np.mean(negatives >= threshold))
                if negatives.size
                else None
            ),
        }
    return result


def grouped_derived_metrics(
    *,
    metadata_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    field: str,
    resolver: Callable[[dict[str, Any]], str | None],
    label_filter: int | None = None,
) -> dict[str, Any] | None:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row, prob in zip(metadata_rows, probabilities, strict=True):
        label = int(row["label"])
        if label_filter is not None and label != label_filter:
            continue
        value = resolver(row)
        if value is None:
            continue
        grouped.setdefault(value, []).append((label, float(prob)))

    if not grouped:
        return None

    result: dict[str, Any] = {
        "field": field,
        "groups": {},
    }
    for value in sorted(grouped):
        entries = grouped[value]
        labels = np.asarray([label for label, _ in entries], dtype=np.int64)
        scores = np.asarray([score for _, score in entries], dtype=np.float64)
        positives = scores[labels == 1]
        negatives = scores[labels == 0]
        predictions = (scores >= threshold).astype(np.int64)
        result["groups"][value] = {
            "count": int(labels.size),
            "positive_count": int(np.sum(labels == 1)),
            "negative_count": int(np.sum(labels == 0)),
            "mean_probability": (
                float(np.mean(scores)) if scores.size else 0.0
            ),
            "accuracy": (
                float(np.mean(predictions == labels)) if labels.size else 0.0
            ),
            "positive_recovery_rate": (
                float(np.mean(positives >= threshold))
                if positives.size
                else None
            ),
            "negative_false_positive_rate": (
                float(np.mean(negatives >= threshold))
                if negatives.size
                else None
            ),
        }
    return result


def resolve_catalog_morphology(row: dict[str, Any]) -> str | None:
    extendedness = finite_float_or_none(row.get("catalog_extendedness"))
    if extendedness is None:
        return None
    if extendedness < 0.5:
        return "pointlike"
    return "extended"


def resolve_negative_difficulty(row: dict[str, Any]) -> str | None:
    center_source = str(row.get("center_source") or "").strip()
    if center_source == "random":
        return "random"
    if center_source == "catalog":
        return "catalog-center"
    if center_source != "catalog-offset":
        return center_source or None
    radius = finite_float_or_none(row.get("center_offset_radius"))
    if radius is None:
        return "catalog-offset:unknown"
    if radius < 8.0:
        return "catalog-offset:near"
    if radius < 16.0:
        return "catalog-offset:mid"
    return "catalog-offset:far"


def resolve_mask_pressure(row: dict[str, Any]) -> str | None:
    fractions = [
        value
        for value in (
            finite_float_or_none(row.get("search_valid_fraction")),
            finite_float_or_none(
                row.get("difference_context_valid_fraction")
            ),
        )
        if value is not None
    ]
    if not fractions:
        return None
    pressure = min(fractions)
    if pressure >= 0.999:
        return "fully-valid"
    if pressure >= 0.95:
        return "light-mask-pressure"
    if pressure >= 0.80:
        return "moderate-mask-pressure"
    return "heavy-mask-pressure"


def finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def sanitize_json_nonfinite(value: Any) -> Any:
    """Replace non-finite floats with None, recursing through dict/list.

    Prediction rows copy passthrough catalog metadata verbatim, and a catalog
    field can legitimately be NaN/Inf. json_dumps stays strict
    (allow_nan=False) for computed metrics, so sanitize the copied metadata
    here rather than failing the whole write on a single non-finite field.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            key: sanitize_json_nonfinite(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_json_nonfinite(item) for item in value]
    return value


def write_prediction_artifacts(
    output_dir: Path,
    *,
    metrics: dict[str, Any],
    metadata_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, label, prob, logit in zip(
            metadata_rows,
            y_true,
            probabilities,
            logits,
            strict=True,
        ):
            payload = sanitize_json_nonfinite(dict(row))
            payload["label"] = int(label)
            payload["probability"] = float(prob)
            payload["logit"] = float(logit)
            handle.write(json_dumps(payload) + "\n")

    # Curves are undefined for single-class input; skip rather than emit a
    # fabricated perfect curve (metrics["roc_auc"] is None in that case).
    if metrics.get("roc_auc") is not None:
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        write_curve_csv(
            output_dir / "roc_curve.csv",
            rows=zip(thresholds, fpr, tpr, strict=True),
            headers=("threshold", "fpr", "tpr"),
        )

        precision, recall, pr_thresholds = precision_recall_curve(
            y_true, probabilities
        )
        write_curve_csv(
            output_dir / "precision_recall_curve.csv",
            rows=zip(pr_thresholds, precision, recall, strict=True),
            headers=("threshold", "precision", "recall"),
        )
    else:
        (output_dir / "roc_curve.csv").unlink(missing_ok=True)
        (output_dir / "precision_recall_curve.csv").unlink(missing_ok=True)

    for field, rows in metrics["binned"].items():
        write_binned_csv(
            output_dir / f"binned_{field}.csv",
            rows,
        )

    calibration = metrics.get("calibration")
    write_optional_json_artifact(
        output_dir / "calibration.json",
        calibration,
    )
    if isinstance(calibration, dict):
        write_dict_csv(
            output_dir / "reliability_curve.csv",
            reliability_curve(y_true, probabilities),
            headers=(
                "bin_start",
                "bin_end",
                "count",
                "mean_probability",
                "empirical_positive_rate",
                "calibration_error",
                "absolute_calibration_error",
            ),
        )
    else:
        (output_dir / "reliability_curve.csv").unlink(missing_ok=True)

    diagnostics = metrics.get("threshold_diagnostics")
    write_optional_json_artifact(
        output_dir / "threshold_diagnostics.json",
        diagnostics,
    )
    if isinstance(diagnostics, dict) and "split_optimal" in diagnostics:
        write_dict_csv(
            output_dir / "threshold_sweep.csv",
            threshold_sweep(y_true, probabilities),
            headers=(
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "fpr",
                "fnr",
                "f1",
                "tp",
                "tn",
                "fp",
                "fn",
            ),
        )
    else:
        (output_dir / "threshold_sweep.csv").unlink(missing_ok=True)

    write_optional_json_artifact(
        output_dir / "consensus.json", metrics.get("consensus")
    )
    write_optional_json_artifact(
        output_dir / "autoscan_baseline.json",
        metrics.get("autoscan_baseline"),
    )
    write_optional_json_artifact(
        output_dir / "center_source_breakdown.json",
        metrics.get("center_source_breakdown"),
    )
    write_optional_json_artifact(
        output_dir / "catalog_pool_role_breakdown.json",
        metrics.get("catalog_pool_role_breakdown"),
    )
    write_optional_json_artifact(
        output_dir / "catalog_morphology_breakdown.json",
        metrics.get("catalog_morphology_breakdown"),
    )
    write_optional_json_artifact(
        output_dir / "negative_difficulty_breakdown.json",
        metrics.get("negative_difficulty_breakdown"),
    )
    write_optional_json_artifact(
        output_dir / "mask_pressure_breakdown.json",
        metrics.get("mask_pressure_breakdown"),
    )

    (output_dir / "summary.md").write_text(
        build_markdown_report(metrics),
        encoding="utf-8",
    )


def write_optional_json_artifact(path: Path, payload: Any) -> None:
    if isinstance(payload, dict):
        path.write_text(
            json_dumps(payload) + "\n",
            encoding="utf-8",
        )
    else:
        path.unlink(missing_ok=True)


def write_curve_csv(
    path: Path,
    *,
    rows,
    headers: tuple[str, str, str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def write_binned_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "bin_start",
        "bin_end",
        "count",
        "recovery_rate",
        "false_positive_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def write_dict_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    headers: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def build_markdown_report(metrics: dict[str, Any]) -> str:
    confusion = metrics["confusion"]

    def fmt(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "n/a"

    lines = [
        "# XScan Evaluation Summary",
        "",
        f"- Samples: {metrics['sample_count']}",
        f"- Positives: {metrics['positive_count']}",
        f"- Negatives: {metrics['negative_count']}",
        f"- Accuracy: {fmt(metrics['accuracy'])}",
        f"- ROC AUC: {fmt(metrics['roc_auc'])}",
        f"- PR AUC: {fmt(metrics['pr_auc'])}",
        f"- Brier score: {fmt(metrics['brier_score'])}",
        f"- TPR @ 1% FPR: {fmt(metrics['tpr_at_fpr_1pct'])}",
        f"- TPR @ 5% FPR: {fmt(metrics['tpr_at_fpr_5pct'])}",
    ]
    if metrics.get("metric_undefined_reason"):
        lines.append(
            f"- Metrics undefined: {metrics['metric_undefined_reason']}"
        )
    lines += [
        "",
        "## Confusion Matrix",
        "",
        "| TP | TN | FP | FN |",
        "|---:|---:|---:|---:|",
        "| {tp} | {tn} | {fp} | {fn} |".format(
            tp=confusion["tp"],
            tn=confusion["tn"],
            fp=confusion["fp"],
            fn=confusion["fn"],
        ),
        "",
        "## Saved Tables",
        "",
    ]
    # The ROC/PR curves are skipped for single-class input; only advertise
    # them when they were actually written.
    if metrics.get("roc_auc") is not None:
        lines.append("- `roc_curve.csv`")
        lines.append("- `precision_recall_curve.csv`")
    diagnostics = metrics.get("threshold_diagnostics")
    if isinstance(diagnostics, dict) and "split_optimal" in diagnostics:
        lines.append("- `threshold_sweep.csv`")
    if isinstance(metrics.get("calibration"), dict):
        lines.append("- `reliability_curve.csv`")
    for field, rows in metrics["binned"].items():
        if rows:
            lines.append(f"- `binned_{field}.csv`")
    lines.extend(
        [
            "- `predictions.jsonl`",
        ]
    )
    calibration = metrics.get("calibration")
    if isinstance(calibration, dict):
        lines.extend(
            [
                "",
                "## Calibration",
                "",
                f"- Brier score: {calibration['brier_score']:.6f}",
                "- Expected calibration error: "
                f"{calibration['expected_calibration_error']:.6f}",
                "- Max calibration error: "
                f"{calibration['max_calibration_error']:.6f}",
                "",
                "- `calibration.json`",
            ]
        )
    if isinstance(diagnostics, dict):
        lines.extend(render_threshold_diagnostics_section(diagnostics))
    else:
        selection_reason = metrics.get("threshold_selection_undefined_reason")
        if selection_reason is not None:
            lines.extend(
                [
                    "",
                    "## Threshold Diagnostics",
                    "",
                    "- Validation-selected threshold unavailable: "
                    f"`{selection_reason}`",
                ]
            )
    consensus = metrics.get("consensus")
    if isinstance(consensus, dict):
        lines.extend(
            render_grouped_json_section(
                title="Consensus Metrics",
                artifact_name="consensus.json",
                payload=consensus,
                description_lines=[
                    f"- Group field: `{consensus['group_field']}`",
                    f"- Positive groups: {consensus['positive_group_count']}",
                    "- N-shot recovery: "
                    f"{consensus['n_shot_recovery_rate']:.6f}",
                    "- Majority-consensus recovery: "
                    f"{consensus['majority_consensus_recovery_rate']:.6f}",
                ],
            )
        )
    autoscan = metrics.get("autoscan_baseline")
    if isinstance(autoscan, dict):
        lines.extend(
            [
                "",
                "## autoScan Baseline",
                "",
                f"- Stamp recovery: {autoscan['stamp_recovery_rate']:.6f}",
            ]
        )
        if "n_shot_recovery_rate" in autoscan:
            lines.append(
                f"- N-shot recovery: {autoscan['n_shot_recovery_rate']:.6f}"
            )
            lines.append(
                "- Majority-consensus recovery: "
                f"{autoscan['majority_consensus_recovery_rate']:.6f}"
            )
        lines.extend(["", "- `autoscan_baseline.json`"])
    center_source = metrics.get("center_source_breakdown")
    if isinstance(center_source, dict):
        lines.extend(
            render_grouped_breakdown_section(
                title="Center Source Breakdown",
                artifact_name="center_source_breakdown.json",
                payload=center_source,
            )
        )
    catalog_pool_role = metrics.get("catalog_pool_role_breakdown")
    if isinstance(catalog_pool_role, dict):
        lines.extend(
            render_grouped_breakdown_section(
                title="Catalog Pool Role Breakdown",
                artifact_name="catalog_pool_role_breakdown.json",
                payload=catalog_pool_role,
            )
        )
    catalog_morphology = metrics.get("catalog_morphology_breakdown")
    if isinstance(catalog_morphology, dict):
        lines.extend(
            render_grouped_breakdown_section(
                title="Catalog Morphology Breakdown",
                artifact_name="catalog_morphology_breakdown.json",
                payload=catalog_morphology,
            )
        )
    negative_difficulty = metrics.get("negative_difficulty_breakdown")
    if isinstance(negative_difficulty, dict):
        lines.extend(
            render_grouped_breakdown_section(
                title="Negative Difficulty Breakdown",
                artifact_name="negative_difficulty_breakdown.json",
                payload=negative_difficulty,
            )
        )
    mask_pressure = metrics.get("mask_pressure_breakdown")
    if isinstance(mask_pressure, dict):
        lines.extend(
            render_grouped_breakdown_section(
                title="Mask Pressure Breakdown",
                artifact_name="mask_pressure_breakdown.json",
                payload=mask_pressure,
            )
        )
    return "\n".join(lines) + "\n"


def render_threshold_diagnostics_section(
    diagnostics: dict[str, Any],
) -> list[str]:
    fixed = diagnostics["fixed_threshold"]
    lines = [
        "",
        "## Threshold Diagnostics",
        "",
        f"- Fixed threshold `{fixed['threshold']:.6f}` accuracy: "
        f"{fixed['accuracy']:.6f}",
    ]
    split_optimal = diagnostics.get("split_optimal")
    if isinstance(split_optimal, dict):
        lines.append(
            f"- Split-optimal threshold "
            f"`{split_optimal['threshold']:.6f}` accuracy: "
            f"{split_optimal['accuracy']:.6f}"
        )
    else:
        split_reason = diagnostics.get("split_optimal_undefined_reason")
        if split_reason is not None:
            lines.append(
                f"- Split-optimal threshold unavailable: `{split_reason}`"
            )
    validation_selected = diagnostics.get("validation_selected")
    if isinstance(validation_selected, dict):
        source = validation_selected["source_threshold_metrics"]
        evaluated = validation_selected["evaluated_split_metrics"]
        lines.append(
            f"- Validation-selected threshold `{evaluated['threshold']:.6f}` "
            f"accuracy on this split: {evaluated['accuracy']:.6f}"
        )
        lines.append(
            f"- Source `{validation_selected['source_split']}` "
            f"{validation_selected['selection_metric']}: "
            f"{source['selection_metric_value']:.6f}"
        )
    selection_reason = diagnostics.get(
        "validation_selection_undefined_reason"
    )
    if selection_reason is not None:
        lines.append(
            "- Validation-selected threshold unavailable: "
            f"`{selection_reason}`"
        )
    lines.extend(["", "- `threshold_diagnostics.json`"])
    return lines


def render_grouped_json_section(
    *,
    title: str,
    artifact_name: str,
    payload: dict[str, Any],
    description_lines: list[str],
) -> list[str]:
    lines = ["", f"## {title}", ""]
    lines.extend(description_lines)
    lines.extend(["", f"- `{artifact_name}`"])
    return lines


def render_grouped_breakdown_section(
    *,
    title: str,
    artifact_name: str,
    payload: dict[str, Any],
) -> list[str]:
    lines = ["", f"## {title}", "", f"- Group field: `{payload['field']}`"]
    for name, group_payload in payload["groups"].items():
        lines.append(
            f"- `{name}`: count={group_payload['count']}, "
            f"positives={group_payload['positive_count']}, "
            f"negatives={group_payload['negative_count']}, "
            f"accuracy={group_payload['accuracy']:.6f}"
        )
    lines.extend(["", f"- `{artifact_name}`"])
    return lines


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    # allow_nan=False raises on NaN/Infinity instead of emitting invalid JSON
    # (bare NaN/Infinity that strict parsers reject).
    return json.dumps(payload, sort_keys=True, allow_nan=False)
