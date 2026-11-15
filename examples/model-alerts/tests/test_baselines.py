# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import LogisticBaseline, measured_features


def test_baseline_scaling_is_frozen_and_constant_features_are_safe():
    features = np.column_stack((np.linspace(-2, 2, 100), np.zeros(100)))
    labels = np.r_[np.zeros(50), np.ones(50)]
    model = LogisticBaseline().fit(features, labels)
    saved = model.receipt()
    scores = model.predict(features)
    assert np.mean((scores >= 0.5) == labels) == 1
    model.predict(features + 1000)
    assert model.receipt() == saved
    assert np.isfinite(scores).all()


def test_measured_statistics_have_no_mask_or_clipping_signature():
    images = np.random.default_rng(3).normal(size=(5, 3, 31, 31))
    features = measured_features(images, np.arange(5) + 4)
    assert features.shape == (5, 7)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, -2:], 0)


def test_signature_statistics_detect_clipped_and_missing_pixels():
    images = np.random.default_rng(3).normal(size=(2, 3, 31, 31))
    images[0] = np.clip(images[0], -1, 1)
    images[1, 0, 0, 0] = np.nan
    features = measured_features(images, [4, 5])
    assert features[0, -1] > 0.2
    assert features[1, -2] > 0
    assert features[0, -2] == 0
    assert np.isfinite(features).all()
