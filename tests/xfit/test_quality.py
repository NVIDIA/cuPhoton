# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xfit import (
    DipoleQualityFlag,
    GaussianDipoleModel,
    dipole_quality_flags,
    fit_dipoles,
)
from cuphoton.xfit.quality import _joint_snr

SHAPE = (51, 51)


def _stamps(parameters, seed=0, noise=1.0):
    model = GaussianDipoleModel(SHAPE)
    clean = np.asarray(
        model.evaluate(np.asarray(parameters, dtype=np.float64))
    )
    return clean + np.random.default_rng(seed).normal(0.0, noise, clean.shape)


def _result(parameters, valid=True):
    p = np.atleast_2d(np.asarray(parameters, dtype=np.float64))
    return SimpleNamespace(
        parameters=p, uncertainty_valid=np.full(len(p), valid)
    )


def test_resolved_dipole_has_no_flags():
    truth = [[60.0, 2.0, 1.8, 0.3, 3.0, 1.0, -3.0, -1.0]]
    images = _stamps(truth)
    fit = fit_dipoles(
        images,
        model="gaussian",
        mode="difference",
        backend="numpy",
        variance=np.ones_like(images),
    )
    assert dipole_quality_flags(
        fit, images, variance=np.ones_like(images)
    ).tolist() == [0]


def test_single_source_second_lobe_is_unsupported():
    clean = [[60.0, 2.0, 1.8, 0.3, 0.5, -0.5, 1e6, 1e6]]
    images = _stamps(clean, seed=1)
    spurious = [60.0, 2.0, 1.8, 0.3, 0.5, -0.5, 15.0, 10.0]
    flags = dipole_quality_flags(
        _result(spurious), images, variance=np.ones_like(images)
    )
    assert flags[0] & DipoleQualityFlag.NEGATIVE_LOBE_UNSUPPORTED
    assert not flags[0] & DipoleQualityFlag.LOBES_OVERLAP


def test_negative_amplitude_reads_the_source_from_the_other_lobe():
    clean = [[60.0, 2.0, 1.8, 0.3, 0.5, -0.5, 1e6, 1e6]]
    images = _stamps(clean, seed=2)
    swapped = [-60.0, 2.0, 1.8, 0.3, 15.0, 10.0, 0.5, -0.5]
    flags = dipole_quality_flags(
        _result(swapped), images, variance=np.ones_like(images)
    )
    assert not flags[0] & DipoleQualityFlag.NO_LOBE_AT_CANDIDATE
    assert flags[0] & DipoleQualityFlag.NEGATIVE_LOBE_UNSUPPORTED


@pytest.mark.parametrize(
    ("parameters", "valid", "expected"),
    [
        (
            [60.0, 40.0, 1.8, 0.3, 3.0, 1.0, -3.0, -1.0],
            True,
            DipoleQualityFlag.WIDTH_OUT_OF_RANGE,
        ),
        (
            [60.0, 2.0, 1.8, 0.3, 3.0, 1.0, -40.0, -1.0],
            True,
            DipoleQualityFlag.LOBE_OUTSIDE_STAMP,
        ),
        (
            [60.0, 2.0, 1.8, 0.3, 12.0, 8.0, 9.0, 12.0],
            True,
            DipoleQualityFlag.NO_LOBE_AT_CANDIDATE,
        ),
        (
            [60.0, 2.0, 1.8, 0.3, 0.3, 0.1, -0.3, -0.1],
            True,
            DipoleQualityFlag.LOBES_OVERLAP,
        ),
        (
            [60.0, 2.0, 1.8, 0.3, 3.0, 1.0, -3.0, -1.0],
            False,
            DipoleQualityFlag.UNCERTAINTY_INVALID,
        ),
    ],
)
def test_each_flag(parameters, valid, expected):
    images = _stamps(
        [
            parameters
            if abs(parameters[6]) < 25
            else parameters[:6] + [-3.0, -1.0]
        ],
        seed=3,
    )
    flags = dipole_quality_flags(
        _result(parameters, valid), images, variance=np.ones_like(images)
    )
    assert flags[0] & expected


def test_noise_from_the_stamp_when_variance_is_absent():
    truth = [[60.0, 2.0, 1.8, 0.3, 3.0, 1.0, -3.0, -1.0]]
    images = _stamps(truth, seed=4)
    assert dipole_quality_flags(_result(truth[0]), images).tolist() == [0]


def test_masked_pixels_are_ignored():
    truth = [[60.0, 2.0, 1.8, 0.3, 3.0, 1.0, -3.0, -1.0]]
    images = _stamps(truth, seed=5)
    mask = np.ones_like(images, dtype=bool)
    images[0, :5, :] = np.nan
    mask[0, :5, :] = False
    assert dipole_quality_flags(
        _result(truth[0]), images, variance=np.ones_like(images), mask=mask
    ).tolist() == [0]


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        dipole_quality_flags(
            _result([60.0, 2.0, 1.8, 0.3, 3.0, 1.0, -3.0, -1.0]),
            np.zeros(SHAPE),
        )


def test_fading_single_source_positive_lobe_is_unsupported():
    clean = [[-60.0, 2.0, 1.8, 0.3, 0.5, -0.5, 1e6, 1e6]]
    images = _stamps(clean, seed=1)
    spurious = [60.0, 2.0, 1.8, 0.3, 15.0, 10.0, 0.5, -0.5]
    flags = dipole_quality_flags(
        _result(spurious), images, variance=np.ones_like(images)
    )
    assert flags[0] & DipoleQualityFlag.POSITIVE_LOBE_UNSUPPORTED
    assert not flags[0] & DipoleQualityFlag.NEGATIVE_LOBE_UNSUPPORTED


def test_correlated_noise_does_not_support_a_spurious_lobe():
    # The fit's unneeded second lobe sits on the deepest noise dip of a stamp
    # whose noise is correlated: significant for independent pixels, not
    # after the per-stamp calibration.
    rng = np.random.default_rng(7)
    white = rng.normal(0.0, 1.0, (55, 55))
    kernel = np.ones(5) / 5.0
    smooth = np.apply_along_axis(np.convolve, 1, white, kernel, "valid")
    smooth = np.apply_along_axis(np.convolve, 0, smooth, kernel, "valid")
    model = GaussianDipoleModel(SHAPE)

    def lobe(x, y):
        params = [[1.0, 2.5, 2.5, 0.0, x, y, 1e6, 1e6]]
        return np.asarray(model.evaluate(np.array(params)))[0]

    image = 60.0 * lobe(0.0, 0.0) + smooth / smooth.std()
    weight = np.ones(SHAPE)
    candidates = [
        (x, y)
        for x in range(-20, 21, 2)
        for y in range(-20, 21, 2)
        if np.hypot(x, y) >= 10
    ]
    snr = [
        -_joint_snr(image, weight, lobe(0, 0), lobe(*c)) for c in candidates
    ]
    x, y = candidates[int(np.argmax(snr))]
    fit = _result([60.0, 2.5, 2.5, 0.0, 0.0, 0.0, x, y])
    images, variance = image[None], weight[None]
    flag = DipoleQualityFlag.NEGATIVE_LOBE_UNSUPPORTED
    naive = dipole_quality_flags(
        fit, images, variance=variance, calibrate_noise=False
    )
    calibrated = dipole_quality_flags(fit, images, variance=variance)
    assert max(snr) > 3.0
    assert not naive[0] & flag
    assert calibrated[0] & flag


def test_cupy_matches_numpy():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("no visible CUDA device")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    truth = [
        [60.0, 2.0, 1.8, 0.3, 3.0, 1.0, -3.0, -1.0],
        [60.0, 2.0, 1.8, 0.3, 0.5, -0.5, 1e6, 1e6],
        [40.0, 2.0, 1.8, 0.0, 0.4, 0.0, -0.4, 0.0],
    ]
    images = _stamps(truth, seed=5)
    mask = np.ones(images.shape, dtype=bool)
    mask[:, :4] = False
    fit = fit_dipoles(
        images,
        model="gaussian",
        mode="difference",
        backend="numpy",
        variance=np.ones_like(images),
        mask=mask,
    )
    host = dipole_quality_flags(fit, images, mask=mask)
    device = dipole_quality_flags(
        SimpleNamespace(
            parameters=cp.asarray(fit.parameters),
            uncertainty_valid=cp.asarray(fit.uncertainty_valid),
        ),
        cp.asarray(images),
        mask=cp.asarray(mask),
    )
    assert isinstance(device, cp.ndarray)
    assert cp.asnumpy(device).tolist() == host.tolist()
