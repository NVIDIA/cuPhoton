# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator import associate, expose, field_exposures, psf


def test_pixel_integrated_flux_and_centroid():
    y, x = np.indices((71, 71))
    image = psf(x.shape, 34.2, 35.3, 1.5)
    assert image.sum() == pytest.approx(1, abs=1e-12)
    assert (image * x).sum() == pytest.approx(34.2, abs=1e-6)
    assert (image * y).sum() == pytest.approx(35.3, abs=1e-6)
    assert np.all(image >= 0)


def test_poisson_plus_read_variance():
    expected = np.full((800, 800), 100.0)
    values = expose(expected, np.random.default_rng(982), 4)
    assert abs(values.mean() - 100) < 0.06
    assert abs(values.var() - 116) < 0.6


def test_field_owns_template_and_seeds():
    first = list(field_exposures(41))
    again = list(field_exposures(41))
    other = list(field_exposures(42))
    for a, b in zip(first, again, strict=False):
        np.testing.assert_array_equal(a["search"], b["search"])
        np.testing.assert_array_equal(a["template"], first[0]["template"])
    assert not np.array_equal(first[0]["search"], first[1]["search"])
    assert not np.array_equal(first[0]["template"], other[0]["template"])
    assert all(c["family"] != "column_bleed" for c in first[0]["causes"])
    assert any(
        c["family"] == "column_bleed"
        for c in next(field_exposures(42, shifted=True))["causes"]
    )


def test_unmatched_noise_and_mover_ontology():
    assert associate(dict(x=1, y=2), [], 3)["label"] == 0
    mover = dict(
        x=1, y=2, positive=True, family="moving_source", challenge=True
    )
    result = associate(dict(x=1, y=2), [mover], 3)
    assert result["label"] == 1 and result["challenge"]
    artifact = dict(
        x=2, y=2, positive=False, family="cosmic_ray", challenge=False
    )
    assert associate(dict(x=1, y=2), [mover, artifact], 3)["ambiguous"]


def test_detection_uses_images_without_truth():
    from generator import detect

    exposure = next(field_exposures(10))
    stamps, rows, _, _ = detect(exposure)
    exposure["causes"] = []
    repeated_stamps, repeated_rows, _, _ = detect(exposure)
    assert rows == repeated_rows and len(rows) > 0
    for stamp, repeated, row in zip(
        stamps, repeated_stamps, rows, strict=True
    ):
        np.testing.assert_array_equal(stamp, repeated)
        assert stamp.shape == (3, 31, 31)
        assert row["snr"] >= 4
        assert stamp[0, 15, 15] == exposure["search"][row["y"], row["x"]]


def test_independent_calibration_is_estimated_not_generator_truth():
    from generator import calibration_sample

    calibration = calibration_sample(100.0, 4.0, np.random.default_rng(785))
    assert 99 < calibration["background"] < 101
    assert 14 < calibration["read_variance"] < 18
    assert calibration["background"] != 100
    assert calibration["read_variance"] != 16
    assert calibration["pixels"] == 4096


def test_persistent_detector_defect_is_present_in_reference_and_search():
    exposures = list(field_exposures(41))
    for exposure in exposures:
        cause = next(
            c for c in exposure["causes"] if c["family"] == "bad_pixel"
        )
        x, y = int(cause["x"]), int(cause["y"])
        for name in ("template", "search"):
            image = exposure[name]
            assert (
                image[y, x] - np.median(image[y - 2 : y + 3, x - 2 : x + 3])
                > 200
            )


def test_nearest_cause_defines_negative_family_independent_of_input_order():
    far = dict(
        x=2, y=0, positive=False, family="misregistration", challenge=False
    )
    near = dict(
        x=0.1, y=0, positive=False, family="cosmic_ray", challenge=False
    )
    for causes in ([far, near], [near, far]):
        result = associate(dict(x=0, y=0), causes, 3)
        assert result["family"] == "cosmic_ray"
        assert result["overlapping_causes"] and not result["ambiguous"]
