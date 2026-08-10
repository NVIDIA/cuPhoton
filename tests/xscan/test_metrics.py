# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the hand-rolled binary-classification metrics.

Guards two previously-silent bugs: tied scores were mis-credited in the ROC/PR
curves, and single-class input returned a fabricated AUC of 1.0/0.0 (now it
raises at the curve level and reports JSON null at the summary level).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cuphoton.xscan.metrics import (
    UndefinedMetricError,
    auc,
    confusion_from_scores,
    evaluate_predictions,
    precision_recall_curve,
    roc_curve,
    select_threshold_for_scores,
    threshold_sweep,
)


def _row(label: int, index: int) -> dict:
    return {
        "candidate_id": f"c{index}",
        "exposure_id": index,
        "ccd_id": 1,
        "band": "i",
        "x": 10 + index,
        "y": 10 + index,
        "split_group": f"g{index}",
        "split": "test",
        "label": int(label),
        "center_source": "catalog",
        "catalog_pool_role": "positive" if label == 1 else "negative",
        "catalog_flux": 100.0,
        "catalog_extendedness": 0.1,
        "fake_id": f"f{index}" if label == 1 else None,
        "autoscan_score": 0.5,
        "diff_snr": 5.0,
        "snr": 5.0,
        "flux_ratio": 0.2,
        "center_offset_radius": 0.0,
        "search_valid_fraction": 1.0,
        "difference_context_valid_fraction": 1.0,
    }


def test_roc_auc_tied_scores_credit_half() -> None:
    # One positive and one negative with identical scores must score AUC 0.5,
    # not 0.0 (the old tie mask sampled cumulative counts at the first index
    # of the equal-score run).
    fpr, tpr, _ = roc_curve(np.array([0, 1]), np.array([0.5, 0.5]))
    assert auc(fpr, tpr) == 0.5


def test_roc_auc_perfect_separation() -> None:
    fpr, tpr, _ = roc_curve(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert auc(fpr, tpr) == 1.0


def test_roc_auc_partial_tie_matches_rank_definition() -> None:
    # y=[1,0,1,0], s=[0.9,0.5,0.5,0.1]: rank AUC = 3.5/4 = 0.875.
    fpr, tpr, _ = roc_curve(
        np.array([1, 0, 1, 0]), np.array([0.9, 0.5, 0.5, 0.1])
    )
    assert auc(fpr, tpr) == 0.875


def test_precision_recall_tie_auc_is_correct_case() -> None:
    # Fixed tie mask -> PR AUC 0.75 (the old first-of-run mask gave 0.25).
    precision, recall, _ = precision_recall_curve(
        np.array([0, 1]), np.array([0.5, 0.5])
    )
    assert auc(recall, precision) == 0.75


@pytest.mark.parametrize("labels", [np.ones(4), np.zeros(4)])
def test_curves_raise_on_single_class(labels: np.ndarray) -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(UndefinedMetricError):
        roc_curve(labels.astype(np.int64), scores)
    with pytest.raises(UndefinedMetricError):
        precision_recall_curve(labels.astype(np.int64), scores)


def test_threshold_selection_requires_both_classes() -> None:
    labels = np.zeros(4, dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    with pytest.raises(UndefinedMetricError, match="requires both"):
        threshold_sweep(labels, scores)
    with pytest.raises(UndefinedMetricError, match="requires both"):
        select_threshold_for_scores(labels, scores)


def test_single_class_diagnostics_preserve_defined_thresholds() -> None:
    validation_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    validation_scores = np.array([0.2, 0.3, 0.8, 0.9])
    validation_selection = select_threshold_for_scores(
        validation_labels,
        validation_scores,
        source_split="val",
    )
    labels = np.ones(4, dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    logits = np.log(scores / (1.0 - scores))
    rows = [_row(1, index) for index in range(labels.size)]

    result = evaluate_predictions(
        y_true=labels,
        logits=logits,
        metadata_rows=rows,
        threshold_selection=validation_selection,
    )

    diagnostics = result["threshold_diagnostics"]
    assert diagnostics["fixed_threshold"]["threshold"] == 0.5
    assert diagnostics["split_optimal_undefined_reason"] == (
        "single_class_split"
    )
    assert "split_optimal" not in diagnostics
    assert (
        diagnostics["validation_selected"]["evaluated_split_metrics"][
            "threshold"
        ]
        == validation_selection["threshold"]
    )


def test_single_class_reports_null_auc_with_reason() -> None:
    logits = np.array([2.0, -1.0, 0.5, 3.0])
    positive_rows = [_row(1, i) for i in range(4)]
    negative_rows = [_row(0, i) for i in range(4)]
    all_positive = evaluate_predictions(
        y_true=np.ones(4, dtype=np.int64),
        logits=logits,
        metadata_rows=positive_rows,
    )
    all_negative = evaluate_predictions(
        y_true=np.zeros(4, dtype=np.int64),
        logits=logits,
        metadata_rows=negative_rows,
    )
    for result in (all_positive, all_negative):
        # JSON null, not NaN (strict parsers reject NaN) and not a fake 1.0.
        assert result["roc_auc"] is None
        assert result["pr_auc"] is None
        assert result["tpr_at_fpr_1pct"] is None
        assert result["tpr_at_fpr_5pct"] is None
        assert result["metric_undefined_reason"] == "single_class_split"


def test_both_classes_report_finite_auc() -> None:
    logits = np.array([3.0, -2.0, 2.0, -3.0])
    labels = np.array([1, 0, 1, 0])
    rows = [_row(int(label), i) for i, label in enumerate(labels)]
    result = evaluate_predictions(
        y_true=labels, logits=logits, metadata_rows=rows
    )
    assert result["roc_auc"] == 1.0
    assert result["metric_undefined_reason"] is None


def test_non_binary_labels_raise() -> None:
    scores = np.array([0.1, 0.2, 0.9])
    with pytest.raises(UndefinedMetricError):
        roc_curve(np.array([0, 0, 2]), scores)
    with pytest.raises(UndefinedMetricError):
        evaluate_predictions(
            y_true=np.array([0, 0, 2]),
            logits=scores,
            metadata_rows=[_row(0, i) for i in range(3)],
        )


def test_empty_input_raises() -> None:
    with pytest.raises(UndefinedMetricError):
        evaluate_predictions(
            y_true=np.array([], dtype=np.int64),
            logits=np.array([]),
            metadata_rows=[],
        )


def test_two_dimensional_scores_raise() -> None:
    # An (N, 1) score array shares the first axis with the labels but
    # broadcasts to (N, N); reject it rather than silently inflate counts.
    labels = np.array([1, 0, 1, 0])
    scores_2d = np.array([[3.0], [-2.0], [2.0], [-3.0]])
    with pytest.raises(UndefinedMetricError):
        roc_curve(labels, scores_2d)
    with pytest.raises(UndefinedMetricError):
        confusion_from_scores(labels, scores_2d)


def test_metadata_label_mismatch_raises() -> None:
    # y_true from labels.npy vs a disagreeing metadata label must not produce
    # contradictory overall/grouped metrics.
    labels = np.array([1, 0, 1, 0])
    logits = np.array([3.0, -2.0, 2.0, -3.0])
    rows = [_row(int(v), i) for i, v in enumerate(labels)]
    rows[2]["label"] = 0  # metadata disagrees with y_true[2] == 1
    with pytest.raises(UndefinedMetricError):
        evaluate_predictions(y_true=labels, logits=logits, metadata_rows=rows)


def test_metadata_label_missing_or_nonbinary_raises() -> None:
    labels = np.array([1, 0, 1, 0])
    logits = np.array([3.0, -2.0, 2.0, -3.0])
    rows = [_row(int(v), i) for i, v in enumerate(labels)]
    del rows[1]["label"]
    with pytest.raises(UndefinedMetricError):
        evaluate_predictions(y_true=labels, logits=logits, metadata_rows=rows)
    rows2 = [_row(int(v), i) for i, v in enumerate(labels)]
    rows2[0]["label"] = 0.5
    with pytest.raises(UndefinedMetricError):
        evaluate_predictions(
            y_true=labels, logits=logits, metadata_rows=rows2
        )


def test_non_finite_metadata_serializes_as_null(tmp_path) -> None:
    # A NaN catalog field is legitimate passthrough data; predictions.jsonl
    # must be written as valid JSON (null), not fail the write on allow_nan.
    out = tmp_path / "eval"
    labels = np.array([1, 0, 1, 0])
    logits = np.array([3.0, -2.0, 2.0, -3.0])
    rows = [_row(int(v), i) for i, v in enumerate(labels)]
    rows[0]["diff_snr"] = float("nan")
    rows[1]["snr"] = float("inf")
    evaluate_predictions(
        y_true=labels, logits=logits, metadata_rows=rows, output_dir=out
    )
    written = (out / "predictions.jsonl").read_text().splitlines()
    first = json.loads(written[0])  # would raise if NaN were emitted bare
    assert first["diff_snr"] is None
    assert json.loads(written[1])["snr"] is None


def test_single_class_writes_artifacts_without_crash(tmp_path) -> None:
    # The nullable contract must hold end-to-end: writing artifacts + the
    # markdown report on a single-class split must not crash on None, and the
    # undefined ROC/PR curves must not retain stale files from an earlier run.
    out = tmp_path / "eval"
    mixed_labels = np.array([1, 0, 1, 0], dtype=np.int64)
    mixed_logits = np.array([2.0, -1.0, 0.5, -3.0])
    mixed_rows = [
        _row(int(label), index) for index, label in enumerate(mixed_labels)
    ]
    evaluate_predictions(
        y_true=mixed_labels,
        logits=mixed_logits,
        metadata_rows=mixed_rows,
        output_dir=out,
    )
    assert (out / "roc_curve.csv").exists()
    assert (out / "precision_recall_curve.csv").exists()

    logits = np.array([2.0, -1.0, 0.5, 3.0])
    rows = [_row(1, i) for i in range(4)]
    result = evaluate_predictions(
        y_true=np.ones(4, dtype=np.int64),
        logits=logits,
        metadata_rows=rows,
        threshold_selection_undefined_reason=(
            "single_class_validation_split"
        ),
        output_dir=out,
    )
    assert result["roc_auc"] is None
    diagnostics = result["threshold_diagnostics"]
    assert diagnostics["split_optimal_undefined_reason"] == (
        "single_class_split"
    )
    assert result["threshold_selection_undefined_reason"] == (
        "single_class_validation_split"
    )
    assert (out / "predictions.jsonl").exists()
    assert not (out / "roc_curve.csv").exists()
    assert not (out / "precision_recall_curve.csv").exists()
    assert (out / "threshold_diagnostics.json").exists()
    assert not (out / "threshold_sweep.csv").exists()
    assert (
        "- Split-optimal threshold unavailable: `single_class_split`"
        in (out / "summary.md").read_text()
    )
    assert (
        "`single_class_validation_split`" in (out / "summary.md").read_text()
    )


def test_selection_reason_renders_without_diagnostics(tmp_path) -> None:
    out = tmp_path / "eval"
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    logits = np.array([2.0, -1.0, 0.5, -3.0])
    rows = [_row(int(label), index) for index, label in enumerate(labels)]

    result = evaluate_predictions(
        y_true=labels,
        logits=logits,
        metadata_rows=rows,
        threshold_selection_undefined_reason=(
            "single_class_validation_split"
        ),
        include_diagnostics=False,
        output_dir=out,
    )

    assert "threshold_diagnostics" not in result
    assert not (out / "threshold_diagnostics.json").exists()
    assert (
        "Validation-selected threshold unavailable: "
        "`single_class_validation_split`" in (out / "summary.md").read_text()
    )


def test_reused_output_removes_unavailable_optional_artifacts(
    tmp_path,
) -> None:
    out = tmp_path / "eval"
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    logits = np.array([2.0, -1.0, 0.5, -3.0])
    rich_rows = [
        _row(int(label), index) for index, label in enumerate(labels)
    ]
    rich_rows[0]["fake_id"] = "shared"
    rich_rows[2]["fake_id"] = "shared"
    evaluate_predictions(
        y_true=labels,
        logits=logits,
        metadata_rows=rich_rows,
        output_dir=out,
    )
    optional_artifacts = [
        "calibration.json",
        "reliability_curve.csv",
        "threshold_diagnostics.json",
        "threshold_sweep.csv",
        "consensus.json",
        "autoscan_baseline.json",
        "center_source_breakdown.json",
        "catalog_pool_role_breakdown.json",
        "catalog_morphology_breakdown.json",
        "negative_difficulty_breakdown.json",
        "mask_pressure_breakdown.json",
    ]
    assert all((out / name).exists() for name in optional_artifacts)

    minimal_rows = [{"label": int(label)} for label in labels]
    evaluate_predictions(
        y_true=labels,
        logits=logits,
        metadata_rows=minimal_rows,
        include_diagnostics=False,
        output_dir=out,
    )

    assert all(not (out / name).exists() for name in optional_artifacts)
