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
