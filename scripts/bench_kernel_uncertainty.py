# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time the kernel-fit covariance and variance on a detector-sized frame.

Builds a synthetic float64 frame, fits a constant kernel of three Gaussian
bases times polynomials with a flat background, then times
``kernel_fit_covariance`` (plain and sandwich) and ``kernel_fit_variance``
at a set of positions, printing seconds per call and peak memory.

    python scripts/bench_kernel_uncertainty.py --backend numpy --size 1024
    python scripts/bench_kernel_uncertainty.py --backend cupy

Peak memory is ``tracemalloc`` for NumPy and the default memory pool's
``used_bytes`` for CuPy, both relative to the start of each call.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from cuphoton.xpois import (
    GaussianBasisComponent,
    kernel_fit_covariance,
    kernel_fit_variance,
    psf_weights,
    solve_constant_kernel,
)

COMPONENTS = (
    GaussianBasisComponent(sigma=1.5, degree=2),
    GaussianBasisComponent(sigma=3.0, degree=1),
    GaussianBasisComponent(sigma=6.0, degree=0),
)
BACKGROUND_DEGREE = 0
KERNEL_SHAPE = (15, 15)
WINDOW = 15
PSF_SIGMA = 2.0


def gaussian(shape, sigma):
    y = np.arange(shape[0]) - shape[0] // 2
    x = np.arange(shape[1]) - shape[1] // 2
    k = np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2) / sigma**2)
    return k / k.sum()


def resampling_stamp():
    taps = np.array([0.1, 0.4, 0.4, 0.1])
    return taps[:, None] * taps[None, :]


def scene(size, n_positions, seed=0):
    """Reference, target, variance planes and positions, in NumPy."""

    rng = np.random.default_rng(seed)
    reference = rng.normal(100.0, 1.0, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for x0, y0 in rng.uniform(30, size - 30, (max(20, size // 20), 2)):
        reference += 5000.0 * np.exp(
            -0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / PSF_SIGMA**2
        )
    from scipy.signal import fftconvolve

    target = fftconvolve(reference, gaussian(KERNEL_SHAPE, 1.2), "same")
    target += 3.0 + rng.normal(0.0, 2.0, target.shape)
    target_variance = np.full((size, size), 4.0)
    reference_variance = np.full((size, size), 1.0)
    fit_mask = np.zeros((size, size), dtype=bool)
    fit_mask[16:-16, 16:-16] = True
    positions = rng.uniform(20, size - 20, (n_positions, 2))
    return (
        reference,
        target,
        target_variance,
        reference_variance,
        fit_mask,
        positions,
    )


class Meter:
    """Seconds and peak bytes of one call, per backend."""

    def __init__(self, xp):
        self.xp = xp
        self.cupy = xp.__name__ == "cupy"

    def run(self, label, fn):
        if self.cupy:
            pool = self.xp.get_default_memory_pool()
            pool.free_all_blocks()
            base = pool.used_bytes()
            self.xp.cuda.Device().synchronize()
        else:
            tracemalloc.start()
        start = time.perf_counter()
        out = fn()
        if self.cupy:
            self.xp.cuda.Device().synchronize()
        seconds = time.perf_counter() - start
        if self.cupy:
            peak = pool.used_bytes() - base
        else:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        print(f"{label:<28s} {seconds:9.3f} s   peak {peak / 2**30:7.3f} GiB")
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--backend", choices=("numpy", "cupy"), default="numpy"
    )
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--positions", type=int, default=2000)
    args = parser.parse_args()
    xp = np if args.backend == "numpy" else __import__("cupy")

    reference, target, target_variance, reference_variance, fit_mask, xy = (
        scene(args.size, args.positions)
    )
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
    print(
        f"{args.backend}: {args.size} x {args.size} frame, "
        f"{fit.basis_kernels.shape[0] + 1} coefficients, "
        f"{args.positions} positions"
    )
    reference_d = xp.asarray(reference)
    target_variance_d = xp.asarray(target_variance)
    reference_variance_d = xp.asarray(reference_variance)
    stamp = xp.asarray(resampling_stamp())
    weights = xp.asarray(
        psf_weights(PSF_SIGMA, WINDOW, offset=xy - np.rint(xy)),
        dtype=xp.float64,
    )
    xy_d = xp.asarray(xy)
    if args.backend == "cupy":
        from dataclasses import replace

        fit = replace(
            fit,
            basis_kernels=xp.asarray(fit.basis_kernels),
            fit_mask=xp.asarray(fit.fit_mask),
            kernel=xp.asarray(fit.kernel),
        )

    meter = Meter(xp)
    meter.run(
        "kernel_fit_covariance plain",
        lambda: kernel_fit_covariance(
            fit, reference=reference_d, target_variance=target_variance_d
        ),
    )
    cov = meter.run(
        "kernel_fit_covariance sandwich",
        lambda: kernel_fit_covariance(
            fit,
            reference=reference_d,
            target_variance=target_variance_d,
            reference_variance=reference_variance_d,
            resampling_weights=stamp,
        ),
    )
    meter.run(
        "kernel_fit_variance",
        lambda: kernel_fit_variance(
            cov, reference=reference_d, positions=xy_d, weights=weights
        ),
    )


if __name__ == "__main__":
    main()
