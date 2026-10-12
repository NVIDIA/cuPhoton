# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xfit import (
    GaussianDipoleModel,
    derived_dipole_quantities,
    fit_dipoles,
)
from cuphoton.xfit.derived import _values_and_gradients

SHAPE = (31, 31)
TRUTH = np.array([20.0, 1.9, 1.6, 0.3, 4.0, 2.0, -4.0, -2.0])


def _result(parameters, covariance=None, valid=True):
    p = np.atleast_2d(np.asarray(parameters, dtype=np.float64))
    c = (
        np.tile(np.eye(8), (len(p), 1, 1))
        if covariance is None
        else covariance
    )
    return SimpleNamespace(
        model="gaussian",
        parameters=p,
        covariance=c,
        uncertainty_valid=np.full(len(p), valid),
    )


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    p = TRUTH + rng.normal(0.0, 0.3, (6, 8))
    p[3, 0] *= -1.0
    values, gradients = _values_and_gradients(p, np)
    step = 1e-6
    for i in range(8):
        shifted = p.copy()
        shifted[:, i] += step
        numeric = (_values_and_gradients(shifted, np)[0] - values) / step
        np.testing.assert_allclose(gradients[:, :, i], numeric, atol=1e-5)


def test_negative_amplitude_selects_the_other_lobe():
    p = TRUTH.copy()
    p[0] = -p[0]
    q = derived_dipole_quantities(_result(p))
    assert (q.positive_x[0], q.positive_y[0]) == (-4.0, -2.0)
    assert q.lobe_flux[0] == pytest.approx(2 * np.pi * 20.0 * 1.9 * 1.6)


def test_uncertainties_match_monte_carlo_scatter():
    rng = np.random.default_rng(1)
    count = 400
    model = GaussianDipoleModel(SHAPE)
    clean = np.asarray(model.evaluate(TRUTH[None, :]))
    images = clean + rng.normal(0.0, 1.0, (count, *SHAPE))
    fit = fit_dipoles(
        images,
        model="gaussian",
        backend="numpy",
        variance=np.ones_like(images),
        initial=np.tile(TRUTH, (count, 1)),
    )
    q = derived_dipole_quantities(fit)
    ok = q.valid
    assert ok.mean() > 0.95
    for name in ("center_x", "center_y", "positive_x", "separation"):
        scatter = np.std(getattr(q, name)[ok])
        predicted = np.median(getattr(q, f"{name}_error")[ok])
        assert predicted == pytest.approx(scatter, rel=0.12), name
    scatter = np.std(q.lobe_flux[ok])
    assert np.median(q.lobe_flux_error[ok]) == pytest.approx(
        scatter, rel=0.12
    )
    independent = np.sqrt(
        np.einsum(
            "bq,bq->b",
            _values_and_gradients(fit.parameters[ok], np)[1][:, 5] ** 2,
            np.diagonal(fit.covariance[ok], axis1=1, axis2=2),
        )
    )
    assert np.median(independent) > 1.2 * np.median(q.lobe_flux_error[ok])


def test_invalid_fits_have_nan_uncertainties():
    q = derived_dipole_quantities(_result([TRUTH, TRUTH], valid=False))
    assert not q.valid.any()
    assert np.isnan(q.center_x_error).all()
    assert np.isnan(q.lobe_flux_error).all()
    assert np.isfinite(q.center_x).all()


def test_coincident_lobes_have_nan_separation_error():
    p = TRUTH.copy()
    p[6:8] = p[4:6]
    q = derived_dipole_quantities(_result(p))
    assert q.separation[0] == 0.0
    assert np.isnan(q.separation_error[0])
    assert np.isfinite(q.center_x_error[0])


def test_rejects_stamp_model():
    with pytest.raises(ValueError, match="Gaussian"):
        derived_dipole_quantities(SimpleNamespace(model="stamp"))


def test_cupy_matches_numpy():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("no visible CUDA device")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    rng = np.random.default_rng(2)
    p = TRUTH + rng.normal(0.0, 0.2, (8, 8))
    a = rng.normal(size=(8, 8, 8))
    c = 0.01 * a @ a.transpose(0, 2, 1)
    host = derived_dipole_quantities(_result(p, c))
    device = derived_dipole_quantities(
        SimpleNamespace(
            model="gaussian",
            parameters=cp.asarray(p),
            covariance=cp.asarray(c),
            uncertainty_valid=cp.ones(8, dtype=bool),
        )
    )
    assert isinstance(device.lobe_flux_error, cp.ndarray)
    for name in ("center_x_error", "separation_error", "lobe_flux_error"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(device, name)), getattr(host, name)
        )
