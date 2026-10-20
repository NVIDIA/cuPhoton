# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xfit import (
    DipoleFitUncertaintyReason,
    GaussianDipoleModel,
    LMConfig,
    fit_dipoles,
    fit_single_sources,
)

SHAPE = (51, 51)
FAR = 1.0e6


def _single(amplitude, sigma_x, sigma_y, theta, x, y):
    """A one-lobe stamp from the dipole model with the other lobe far away."""

    model = GaussianDipoleModel(SHAPE)
    return np.asarray(
        model.evaluate(
            np.asarray([[amplitude, sigma_x, sigma_y, theta, x, y, FAR, FAR]])
        )
    )


def _noisy(clean, count, seed, noise=1.0):
    rng = np.random.default_rng(seed)
    images = clean + rng.normal(0.0, noise, (count, *SHAPE))
    return images, np.full(images.shape, noise**2)


def _flux(amplitude, sigma_x, sigma_y):
    return 2.0 * np.pi * abs(amplitude) * sigma_x * sigma_y


def test_recovers_position_flux_and_shape_of_a_bright_source():
    truth = (40.0, 2.0, 1.6, 0.3, 1.3, -0.8)
    images, variance = _noisy(_single(*truth), 3, seed=0, noise=0.5)
    result = fit_single_sources(images, variance, np.zeros((3, 2)), sign=1)
    assert result.parameter_names == (
        "amplitude",
        "sigma_x",
        "sigma_y",
        "theta",
        "x",
        "y",
    )
    assert result.converged.all() and result.uncertainty_valid.all()
    assert (result.uncertainty_reason_codes == 0).all()
    np.testing.assert_allclose(result.x, truth[4], atol=0.05)
    np.testing.assert_allclose(result.y, truth[5], atol=0.05)
    assert np.abs(result.parameters[:, 1:3] - truth[1:3]).max() < 0.05
    np.testing.assert_allclose(result.parameters[:, 3], truth[3], atol=0.05)
    np.testing.assert_allclose(result.flux, _flux(*truth[:3]), rtol=0.03)
    assert (result.flux_error > 0).all()
    assert (result.sign == 1).all()
    assert result.residuals.shape == images.shape
    assert (result.degrees_of_freedom == 51 * 51 - 6).all()
    assert np.allclose(
        result.position_error_x, np.sqrt(result.covariance[:, 4, 4])
    )
    assert np.allclose(
        result.position_xy_covariance, result.covariance[:, 4, 5]
    )
    assert result.dtype == "float64"


def test_fixed_width_fits_a_circular_lobe_with_three_parameters():
    truth = (25.0, 1.8, 1.8, 0.0, -0.7, 0.9)
    images, variance = _noisy(_single(*truth), 4, seed=1)
    result = fit_single_sources(
        images, variance, np.zeros((4, 2)), np.full(4, 1.8)
    )
    assert result.parameter_names == ("amplitude", "x", "y")
    assert result.covariance.shape == (4, 3, 3)
    assert result.uncertainty_valid.all()
    np.testing.assert_allclose(result.x, truth[4], atol=0.08)
    np.testing.assert_allclose(result.y, truth[5], atol=0.08)
    np.testing.assert_allclose(result.flux, _flux(*truth[:3]), rtol=0.05)
    np.testing.assert_allclose(
        result.flux_error,
        2.0 * np.pi * 1.8**2 * result.standard_errors[:, 0],
    )


def test_position_covariance_is_calibrated_on_repeated_noise():
    """One stamp, 400 noise realizations, the one-lobe and two-lobe fits.

    The one-lobe fit's 95% interval covers the truth 95.8% of the time with
    a spread of 0.97 at this seed. The two-lobe dipole fit on the same
    stamps converges on 80% of them and, where it does, covers 98.9% with a
    spread of 0.64: the spare lobe settles on noise and the source lobe's
    position inherits its freedom.
    """

    truth = (12.0, 2.0, 1.7, 0.4, 0.6, -0.4)
    count = 400
    images, variance = _noisy(_single(*truth), count, seed=7)
    positions = np.zeros((count, 2))

    single = fit_single_sources(images, variance, positions)
    assert single.uncertainty_valid.all()
    z = np.concatenate(
        [
            (single.x - truth[4]) / single.position_error_x,
            (single.y - truth[5]) / single.position_error_y,
        ]
    )
    coverage = np.mean(np.abs(z) < 1.96)
    assert 0.93 <= coverage <= 0.97, coverage
    assert 0.9 <= z.std() <= 1.1, z.std()

    dipole = fit_dipoles(
        images, model="gaussian", variance=variance, backend="numpy"
    )
    p, c = dipole.parameters, dipole.covariance
    rows = np.arange(count)
    source = np.where(p[:, 0] > 0, 4, 6)
    near = np.hypot(
        p[rows, source] - truth[4], p[rows, source + 1] - truth[5]
    )
    scored = dipole.uncertainty_valid & (near < 3.0)
    assert 0.6 < scored.mean() < 0.95
    z_dipole = np.concatenate(
        [
            (p[rows, source] - truth[4]) / np.sqrt(c[rows, source, source]),
            (p[rows, source + 1] - truth[5])
            / np.sqrt(c[rows, source + 1, source + 1]),
        ]
    )[np.tile(scored, 2)]
    dipole_coverage = np.mean(np.abs(z_dipole) < 1.96)
    assert dipole_coverage > 0.975, dipole_coverage
    assert z_dipole.std() < 0.8, z_dipole.std()
    assert np.median(single.position_error_x) < np.median(
        np.sqrt(c[rows, source, source][scored])
    )


def test_fading_source_fits_a_negative_lobe():
    truth = (-18.0, 1.9, 1.5, -0.6, 0.4, 1.1)
    images, variance = _noisy(_single(*truth), 3, seed=3)
    inferred = fit_single_sources(images, variance, np.zeros((3, 2)))
    given = fit_single_sources(images, variance, np.zeros((3, 2)), sign=-1)
    for result in (inferred, given):
        assert (result.sign == -1).all()
        assert result.uncertainty_valid.all()
        assert (result.parameters[:, 0] < 0).all()
        np.testing.assert_allclose(result.x, truth[4], atol=0.08)
        np.testing.assert_allclose(result.y, truth[5], atol=0.08)
        np.testing.assert_allclose(result.flux, _flux(*truth[:3]), rtol=0.05)


def test_masked_and_nonfinite_pixels_carry_no_weight():
    truth = (30.0, 2.0, 2.0, 0.0, 0.5, 0.0)
    images, variance = _noisy(_single(*truth), 2, seed=4)
    full = fit_single_sources(images, variance, np.zeros((2, 2)))
    mask = np.ones(images.shape, dtype=bool)
    mask[:, :, 26] = False
    images[:, :, 26] = np.nan
    variance[:, 10, :] = 0.0
    result = fit_single_sources(images, variance, np.zeros((2, 2)), mask=mask)
    assert result.uncertainty_valid.all()
    assert (result.degrees_of_freedom == 51 * 51 - 51 - 50 - 6).all()
    np.testing.assert_allclose(result.x, truth[4], atol=0.08)
    np.testing.assert_allclose(result.y, truth[5], atol=0.08)
    assert np.isfinite(result.chi_square).all()
    assert (result.position_error_x > full.position_error_x).all()


def test_float32_compute_dtype_and_solver_config():
    truth = (30.0, 2.0, 2.0, 0.0, 0.0, 0.0)
    images, variance = _noisy(_single(*truth), 2, seed=5)
    result = fit_single_sources(
        images,
        variance,
        np.zeros((2, 2)),
        compute_dtype="float32",
        solver=LMConfig(max_evaluations=2),
    )
    assert result.dtype == "float32"
    assert result.parameters.dtype == np.float32
    assert not result.converged.any()
    assert (
        result.uncertainty_reason_codes
        == int(DipoleFitUncertaintyReason.FIT_NOT_CONVERGED)
    ).all()
    assert np.isnan(result.position_error_x).all()
    assert np.isnan(result.covariance).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"images": np.zeros((5, 5))}, "images must have shape"),
        ({"variance": np.ones((2, 5, 5))}, "variance must share"),
        ({"positions": np.zeros((2, 2))}, "positions must have shape"),
        ({"positions": [[np.nan, 0.0]]}, "positions must contain"),
        ({"psf_sigma": -1.0}, "psf_sigma must be finite"),
        ({"mask": np.ones((5, 5), dtype=bool)}, "mask must share"),
        ({"sign": 2}, "sign values must be"),
        ({"compute_dtype": "float16"}, "compute_dtype must be"),
    ],
)
def test_argument_validation(kwargs, message):
    arguments = {
        "images": np.zeros((1, 5, 5)),
        "variance": np.ones((1, 5, 5)),
        "positions": np.zeros((1, 2)),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        fit_single_sources(**arguments)
