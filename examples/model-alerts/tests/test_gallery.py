# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from views import gallery_indices


def test_gallery_is_deterministic_bounded_and_keeps_rare_errors():
    rows = [
        dict(candidate_id=i, snr=8, family="appearance") for i in range(200)
    ]
    labels = np.ones(200, dtype=int)
    scores = np.ones(200)
    scores[150] = 0.01
    first = gallery_indices(rows, labels, scores, 0.5)
    assert len(first) == 96 and len(set(first)) == 96
    assert 150 in first
    np.testing.assert_array_equal(
        first, gallery_indices(rows, labels, scores, 0.5)
    )


def test_saved_gallery_preserves_receipt_order():
    import pytest

    pytest.importorskip("bokeh")
    from views import candidate_explorer

    rows = [
        dict(candidate_id=i, snr=8.0, family="appearance")
        for i in (90, 3, 17, 8)
    ]
    images = np.random.default_rng(318).normal(size=(4, 3, 31, 31))
    labels, scores = np.array([1, 0, 1, 0]), np.array([0.9, 0.6, 0.3, 0.1])
    with pytest.raises(ValueError, match="complete evaluation"):
        candidate_explorer(
            images, rows, labels, scores, 0.5, preselected=True
        )
    _, receipt = candidate_explorer(
        images,
        rows,
        labels,
        scores,
        0.5,
        preselected=True,
        evaluation=dict(
            labels=labels,
            scores=scores,
            snr=[r["snr"] for r in rows],
            candidate_id=[r["candidate_id"] for r in rows],
            family=[r["family"] for r in rows],
        ),
    )
    assert receipt["indices"] == [0, 1, 2, 3]
