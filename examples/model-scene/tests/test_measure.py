# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for failure modes in scientific comparison, rather than snapshots."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import forward_prediction
from measure import paired_summary
from scene import generate


def test_independent_renderer_discrepancy_is_visible():
    physical, p = generate(19)
    discrete, d = generate(19, matched=True)
    # Independent generation must not accidentally reuse solver convolution.
    delta = p["expectation"][12:60, 12:60] - d["expectation"][12:60, 12:60]
    assert np.max(np.abs(delta)) > 1
    # The same Gaussian noise is paired even across renderer controls.
    np.testing.assert_allclose(
        physical["images"][0] - p["expectation"],
        discrete["images"][0] - d["expectation"],
        atol=2e-14,
    )


def test_valid_forward_prediction_uses_real_halo():
    arrays, truth = generate(9, matched=True)
    full = truth["truth"][5:67, 5:67]
    prediction = forward_prediction(full, arrays["psfs"])
    assert prediction.shape == (48, 48, 12)
    np.testing.assert_allclose(
        prediction, truth["expectation"][12:60, 12:60], atol=1e-12
    )


def test_uncertainty_uses_independent_realizations():
    trials = []
    for noise in [-0.02, -0.01, 0.01, 0.02]:
        trials.append(
            {
                m: {
                    "sources": [
                        {
                            "kind": kind,
                            "input_flux": 1000,
                            "fractional_flux_error": noise + offset,
                        }
                        for kind in ("isolated", "blend", "extended")
                    ]
                }
                for m, offset in [("coadd", 0), ("l2", 0.01), ("huber", 0.02)]
            }
        )
    result = paired_summary(trials)
    assert result["l2"]["isolated"]["n_independent_noise_realizations"] == 4
    assert result["l2"]["isolated"]["bias_standard_error"] > 0.005
    assert (
        result["l2"]["paired_difference_vs_coadd"]["standard_error"] < 1e-15
    )
