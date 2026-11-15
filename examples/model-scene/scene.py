# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent pixel-integrated Gaussian scene and exposure models.

No dataset, survey adapter, or solver implementation is used by this renderer.
Coordinates denote pixel centers. Flux is the integral on the infinite plane;
finite-stamp losses are retained rather than silently renormalizing sources.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
from scipy.special import ndtr


@dataclass(frozen=True)
class Source:
    x: float
    y: float
    flux: float
    sigma: float
    kind: str


def catalog(development=False):
    """Declare catalogs before noise draws or reconstruction measurements."""
    offset = 2.0 if development else 0.0
    return (
        Source(18 + offset, 20, 1200, 0.22, "isolated"),
        Source(46, 19 + offset, 300, 0.22, "isolated"),
        Source(30, 38, 900, 0.22, "blend"),
        Source(33, 38, 450, 0.22, "blend"),
        Source(20, 53, 2200, 2.5, "extended"),
        Source(49, 50, 3200, 4.0, "extended"),
    )


def pixel_gaussian(size, x, y, sigma):
    """Integrate an axis-aligned unit Gaussian over each square pixel."""
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    edges = np.arange(size + 1, dtype=float) - 0.5
    px = np.diff(ndtr((edges - x) / sigma))
    py = np.diff(ndtr((edges - y) / sigma))
    return py[:, None] * px[None, :]


def render(sources, sigma=0.0, size=72, background=2.0):
    """Continuous Gaussian convolution followed by detector pixel integration.

    This differs deliberately from convolving an already pixel-integrated
    latent grid with a sampled finite kernel. That discretization discrepancy
    is measured separately with the matched-operator control.
    """
    image = np.full((size, size), background, dtype=float)
    for source in sources:
        image += source.flux * pixel_gaussian(
            size, source.x, source.y, np.hypot(source.sigma, sigma)
        )
    return image


def kernels(widths, support=15):
    """Sample normalized finite kernels for the discrete operator."""
    axis = np.arange(support) - support // 2
    radius2 = axis[:, None] ** 2 + axis[None, :] ** 2
    result = np.exp(-radius2[..., None] / (2 * np.asarray(widths) ** 2))
    return result / result.sum(axis=(0, 1), keepdims=True)


def generate(seed, *, development=False, matched=False, case="gaussian"):
    """Return twelve shared-grid frames: eight training and four held out.

    Units are electrons per unit exposure; all exposure times are exactly one.
    Known Gaussian variance is 4 electrons squared in the principal case.
    Noise draws are paired across methods by saving and reusing this bundle.
    """
    sources = catalog(development)
    truth = render(sources)
    widths = np.linspace(0.9, 1.7, 12)
    psfs = kernels(widths)
    expectation = np.stack(
        [
            convolve2d(truth, psfs[..., i], mode="same", boundary="fill")
            if matched
            else render(sources, sigma=width)
            for i, width in enumerate(widths)
        ],
        axis=-1,
    )
    rng = np.random.default_rng(seed)
    images = expectation + rng.normal(0, 2, expectation.shape)
    good = np.ones(images.shape, dtype=bool)
    variances = np.full(images.shape, 4.0)
    if case == "outliers":
        bad = rng.random(images.shape) < 0.003
        images[bad] += 100
        good[bad] = False
    elif case == "photon-read":
        images = rng.poisson(expectation) + rng.normal(
            0, 2, expectation.shape
        )
        variances = expectation + 4
    elif case == "transient":
        transient = 1200 * pixel_gaussian(72, 43, 36, 1.2)
        images[..., [1, 6, 9]] += transient[..., None]
    elif case != "gaussian":
        raise ValueError(f"Unknown case: {case}")
    return {
        "images": images[None],
        "variances": variances[None],
        "masks": good[None],
        "psfs": psfs[:, :, None, :],
        "sky": np.full_like(images[None], 2),
        "exp_times": np.ones(12),
    }, {
        "truth": truth,
        "expectation": expectation,
        "widths": widths,
        "matched": matched,
    }


def write_bundle(destination: Path, arrays):
    """Refuse pre-existing destinations, including symbolic links."""
    images = np.asarray(arrays["images"])
    variance = np.asarray(arrays["variances"])
    psfs = np.asarray(arrays["psfs"])
    if images.ndim != 4 or variance.shape != images.shape:
        raise ValueError("Images and variances need matching rank-four axes")
    if not np.all(np.isfinite(images)):
        raise ValueError("Nonfinite image values")
    if not np.all(np.isfinite(variance) & (variance > 0)):
        raise ValueError("Variances must be finite and positive")
    if not np.all(np.isfinite(psfs) & (psfs >= 0)):
        raise ValueError("Multiplicative reconstruction needs positive PSFs")
    masks = np.asarray(arrays["masks"])
    if masks.dtype != bool or masks.shape != images.shape:
        raise ValueError(
            "Masks must be boolean, True = good, matching images"
        )
    times = np.asarray(arrays["exp_times"])
    if not np.all(np.isfinite(times) & (times > 0)):
        raise ValueError("Exposure times must be finite and positive")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for key, array in arrays.items():
        np.save(destination / f"{key}.npy", array, allow_pickle=False)
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "source": "independent analytic Gaussian mixture",
                "image_units": "electrons per unit exposure",
                "variance_units": "(electrons / unit exposure)^2",
                "mask_convention": "True = good",
                "dither_pixels": 0,
            },
            indent=2,
        )
        + "\n"
    )
