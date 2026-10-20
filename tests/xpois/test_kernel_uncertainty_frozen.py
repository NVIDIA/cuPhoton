# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen references for the batched kernel-fit and correlation code.

The digests below (shape, sum, sum of squares, first values) were recorded
from the per-coefficient and per-position implementations on three seeds.
The batched implementations must reproduce them to ``rtol=1e-12``, with the
chunk budgets forced small so several chunks run on these small inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import correlate
from scipy.signal import fftconvolve

from cuphoton.xpois import (
    GaussianBasisComponent,
    correlation_factor,
    kernel_fit_covariance,
    kernel_fit_variance,
    psf_weights,
    solve_constant_kernel,
)
from cuphoton.xpois import correlation as correlation_module
from cuphoton.xpois import kernel_uncertainty as kernel_module

SIZE = 96
KERNEL_SHAPE = (9, 9)
COMPONENTS = (
    GaussianBasisComponent(sigma=1.0, degree=1),
    GaussianBasisComponent(sigma=2.0, degree=0),
)
BACKGROUND_DEGREE = 1
N_POSITIONS = 25
WINDOW = 7
SEEDS = (0, 1, 2)
HEAD = 4

# Digests recorded from the per-coefficient and per-position loops.
REFERENCES = {
    "plain": {
        0: {
            "shape": (7, 7),
            "sum": 0.008944378552227728,
            "sumsq": 3.331566958042317e-05,
            "head": [
                3.474726752254163e-09,
                -7.405398857407167e-12,
                -1.2406688318240667e-11,
                -1.125536641426417e-09,
            ],
        },
        1: {
            "shape": (7, 7),
            "sum": 0.008900418136507342,
            "sumsq": 3.24789030743648e-05,
            "head": [
                2.86318538362839e-09,
                -1.3907776609311413e-11,
                8.434946626862118e-12,
                -9.389030757991924e-10,
            ],
        },
        2: {
            "shape": (7, 7),
            "sum": 0.0089155078585706,
            "sumsq": 3.243934633933451e-05,
            "head": [
                2.911950782492047e-09,
                1.5016282833961205e-11,
                4.855975415945293e-12,
                -9.408812333465172e-10,
            ],
        },
    },
    "sandwich": {
        0: {
            "shape": (7, 7),
            "sum": 0.027199218725802728,
            "sumsq": 0.00030599647890609957,
            "head": [
                6.07609766172597e-09,
                -4.519329777650764e-12,
                -2.3529632109063068e-11,
                -2.066691270601429e-09,
            ],
        },
        1: {
            "shape": (7, 7),
            "sum": 0.027083820823675647,
            "sumsq": 0.00029971168754152754,
            "head": [
                5.197629621907343e-09,
                -3.194600736864017e-11,
                1.6074374802060547e-12,
                -1.7955930601554737e-09,
            ],
        },
        2: {
            "shape": (7, 7),
            "sum": 0.027080682498045974,
            "sumsq": 0.00029991860649854565,
            "head": [
                5.232426201476728e-09,
                1.7330138011316347e-11,
                1.4183712743482226e-11,
                -1.7930780963567773e-09,
            ],
        },
    },
    "variance": {
        0: {
            "shape": (25,),
            "sum": 0.009800640796435035,
            "sumsq": 9.80820237146057e-06,
            "head": [
                0.0010886278766617289,
                5.100830812651617e-06,
                3.3555521479633563e-06,
                0.0017308386874634158,
            ],
        },
        1: {
            "shape": (25,),
            "sum": 0.006778761896600701,
            "sumsq": 1.4217631153292737e-05,
            "head": [
                0.00046295539115736244,
                1.4373313095881125e-05,
                3.3483312134536702e-06,
                2.0493338229832e-06,
            ],
        },
        2: {
            "shape": (25,),
            "sum": 0.0010849524677412922,
            "sumsq": 2.3765687678343244e-07,
            "head": [
                1.069538067181868e-05,
                0.00010764235573352007,
                5.7104743432997e-06,
                4.052088957870147e-06,
            ],
        },
    },
    "factor": {
        0: {
            "shape": (25,),
            "sum": 37.454620080088006,
            "sumsq": 56.14260207784114,
            "head": [
                1.504780112222151,
                1.4877896444076137,
                1.4868005934452486,
                1.5394371409990388,
            ],
        },
        1: {
            "shape": (25,),
            "sum": 37.18347335618376,
            "sumsq": 55.31805143951073,
            "head": [
                1.4802190908362907,
                1.486097311551712,
                1.510193970699635,
                1.4975300631714288,
            ],
        },
        2: {
            "shape": (25,),
            "sum": 37.09018981529562,
            "sumsq": 55.04352206616343,
            "head": [
                1.4963515169321016,
                1.5237294841406688,
                1.4059730395770362,
                1.5037407399830267,
            ],
        },
    },
}


def _gaussian(shape, sigma):
    y = np.arange(shape[0]) - shape[0] // 2
    x = np.arange(shape[1]) - shape[1] // 2
    k = np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2) / sigma**2)
    return k / k.sum()


def _resampling_stamp():
    taps = np.array([0.1, 0.4, 0.4, 0.1])
    return taps[:, None] * taps[None, :]


def scene(seed):
    """A fitted scene with noise, variance planes, positions and weights."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    raw = np.full((SIZE, SIZE), 10.0)
    for x0, y0 in rng.uniform(20, SIZE - 20, (12, 2)):
        raw += 3000.0 * np.exp(
            -0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5**2
        )
    stamp = _resampling_stamp()
    reference = correlate(raw, stamp, mode="nearest")
    reference += rng.normal(0.0, 1.0, reference.shape)
    target = fftconvolve(reference, _gaussian(KERNEL_SHAPE, 1.2), "same")
    target += 2.0 + 0.01 * xx + rng.normal(0.0, 2.0, target.shape)
    target_variance = rng.uniform(3.0, 5.0, (SIZE, SIZE))
    reference_variance = rng.uniform(0.8, 1.2, (SIZE, SIZE))
    fit_mask = np.zeros((SIZE, SIZE), dtype=bool)
    fit_mask[12:-12, 12:-12] = True
    fit = solve_constant_kernel(
        reference,
        target,
        COMPONENTS,
        kernel_shape=KERNEL_SHAPE,
        variance=target_variance,
        fit_mask=fit_mask,
        background_degree=BACKGROUND_DEGREE,
        backend="cpu",
    )
    positions = rng.uniform(15, SIZE - 15, (N_POSITIONS, 2))
    weights = psf_weights(
        1.5, WINDOW, offset=positions - np.rint(positions), derivative="x"
    )
    valid = np.ones((SIZE, SIZE), dtype=bool)
    valid[:, 40] = False
    return {
        "fit": fit,
        "reference": reference,
        "target_variance": target_variance,
        "reference_variance": reference_variance,
        "stamp": stamp,
        "positions": positions,
        "weights": weights,
        "valid": valid,
    }


def outputs(seed):
    """The four arrays the references digest, for one seed."""

    s = scene(seed)
    plain = kernel_fit_covariance(
        s["fit"],
        reference=s["reference"],
        target_variance=s["target_variance"],
    )
    sandwich = kernel_fit_covariance(
        s["fit"],
        reference=s["reference"],
        target_variance=s["target_variance"],
        reference_variance=s["reference_variance"],
        resampling_weights=s["stamp"],
    )
    variance = kernel_fit_variance(
        sandwich,
        reference=s["reference"],
        positions=s["positions"],
        weights=s["weights"],
    )
    factor = correlation_factor(
        s["weights"],
        kernel=s["fit"].kernel,
        target_variance=s["target_variance"],
        reference_variance=s["reference_variance"],
        resampling_weights=s["stamp"],
        positions=s["positions"],
        valid=s["valid"],
    ).factor
    return {
        "plain": plain.covariance,
        "sandwich": sandwich.covariance,
        "variance": variance,
        "factor": factor,
    }


def digest(array):
    flat = np.asarray(array, dtype=np.float64).ravel()
    return {
        "shape": tuple(np.shape(array)),
        "sum": float(flat.sum()),
        "sumsq": float((flat**2).sum()),
        "head": [float(v) for v in flat[:HEAD]],
    }


def _assert_digest(array, expected):
    got = digest(array)
    assert got["shape"] == tuple(expected["shape"])
    np.testing.assert_allclose(got["sum"], expected["sum"], rtol=1e-12)
    np.testing.assert_allclose(got["sumsq"], expected["sumsq"], rtol=1e-12)
    np.testing.assert_allclose(got["head"], expected["head"], rtol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_batched_code_reproduces_the_frozen_references(seed):
    for name, array in outputs(seed).items():
        _assert_digest(array, REFERENCES[name][seed])


@pytest.mark.parametrize("seed", SEEDS)
def test_small_chunks_reproduce_the_frozen_references(seed, monkeypatch):
    """Several coefficient and position chunks give the one-chunk result."""

    monkeypatch.setattr(
        kernel_module, "_FOOTPRINT_CHUNK_ELEMENTS", 2 * SIZE**2
    )
    monkeypatch.setattr(kernel_module, "_RESPONSE_CHUNK_ELEMENTS", 1)
    monkeypatch.setattr(correlation_module, "_CHUNK", 7)
    for name, array in outputs(seed).items():
        _assert_digest(array, REFERENCES[name][seed])
