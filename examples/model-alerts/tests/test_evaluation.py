# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import (
    confusion,
    paired_field_bootstrap,
    queue_projection,
    recall_threshold,
)


def test_threshold_is_highest_feasible_and_includes_ties():
    labels = np.r_[np.ones(20), np.zeros(4)]
    scores = np.r_[np.linspace(0.2, 0.9, 18), 0.1, 0.1, 0.1, 0.3, 0.5, 0.8]
    threshold = recall_threshold(labels, scores)
    assert threshold == 0.1
    assert confusion(labels, scores, threshold)["recall"] == 1
    assert (
        confusion(labels, scores, np.nextafter(threshold, 1))["recall"] < 0.95
    )


def test_paired_bootstrap_retains_field_dependence():
    labels = np.zeros(8)
    fields = np.repeat(["a", "b"], 4)
    scores = np.repeat([1.0, 0.0], 4)
    result = paired_field_bootstrap(labels, scores, scores, fields, 0.5, 0.5)
    assert result["a"] == [0, 1]
    assert result["a_minus_b"] == [0, 0]


def test_empty_queue_has_undefined_purity():
    assert queue_projection(0, 0, 0.01, 100)["purity"] is None
    assert queue_projection(0.9, 0.1, 0.01, 100)["queue"] == pytest.approx(
        10.8
    )


@pytest.mark.parametrize(
    "labels,scores",
    [([1], [float("nan")]), ([2], [0.5]), ([], []), ([1, 0], [0.5])],
)
def test_reject_invalid_evaluation(labels, scores):
    with pytest.raises(ValueError):
        recall_threshold(labels, scores)


def test_field_rate_intervals_keep_perfect_decisions_and_count_undefined():
    from evaluation import field_rate_intervals

    result = field_rate_intervals(
        [1, 1, 0, 0], [0.8, 0.9, 0.1, 0.2], ["a", "a", "b", "b"], 0.5
    )
    assert result["recall"]["interval"] == [1, 1]
    assert result["fpr"]["interval"] == [0, 0]
    assert result["recall"]["undefined_draws"] > 0
    assert result["fpr"]["undefined_draws"] > 0
