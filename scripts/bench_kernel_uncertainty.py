# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time the kernel-fit covariance and variance on a detector-sized frame.

Builds a synthetic float64 frame, fits a constant kernel of three Gaussian
bases times polynomials with a flat background (11 coefficients), then
times ``kernel_fit_covariance`` (plain and sandwich) and
``kernel_fit_variance`` at a set of positions, printing seconds per call
and peak memory.

    python scripts/bench_kernel_uncertainty.py --backend numpy --size 1024
    python scripts/bench_kernel_uncertainty.py --backend cupy

Every call runs twice and the second run is timed, so FFT plans and pool
blocks from the first run do not count. Peak memory is the ``tracemalloc``
peak for NumPy. For CuPy it is the high-water mark of the default memory
pool's allocations during the call, recorded by a ``MemoryHook``, after
clearing the FFT plan cache and freeing the pool's blocks; the plan cache
size after the call is printed beside it, since cuFFT work areas live
there and not in the pool.

The resampling stamp is the Lanczos-3 6 x 6 stamp at the frame center with
a half-pixel phase, from ``cuphoton.xrep.resampling_weights_at`` when the
tree exposes it and the outer product of the one-dimensional Lanczos-3 taps
otherwise. The PSF weights cover 5 sigma each way, the window a
PSF-weighted sum uses in the transient chain.
"""

from __future__ import annotations

import argparse
import math
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
PSF_SIGMA = 2.0
LANCZOS_A = 3


def window_side(psf_sigma):
    """Side of the window a PSF-weighted sum covers: 5 sigma each way."""

    return 2 * int(math.ceil(5.0 * float(psf_sigma))) + 1


def gaussian(shape, sigma):
    y = np.arange(shape[0]) - shape[0] // 2
    x = np.arange(shape[1]) - shape[1] // 2
    k = np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2) / sigma**2)
    return k / k.sum()


def lanczos_taps(phase, a=LANCZOS_A):
    """Normalized one-dimensional Lanczos taps at a sub-pixel ``phase``."""

    x = phase - np.arange(1 - a, a + 1)
    taps = np.sinc(x) * np.sinc(x / a)
    return taps / taps.sum()


def resampling_stamp(size):
    """The ``(6, 6)`` Lanczos-3 stamp at the frame center and its origin."""

    try:
        from cuphoton.xrep import resampling_weights_at
    except ImportError:
        taps = lanczos_taps(0.5)
        return taps[:, None] * taps[None, :], "Lanczos-3 outer product"
    center = np.array([size // 2], dtype=np.float64)
    weights = resampling_weights_at(
        lambda points: points + 0.5,
        center,
        center,
        interpolation="lanczos3",
        lanczos_a=LANCZOS_A,
    )
    return (
        np.asarray(weights.weights[0], dtype=np.float64),
        "cuphoton.xrep.resampling_weights_at",
    )


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


def high_water_hook(cp):
    """A ``MemoryHook`` that records the pool's peak allocated bytes."""

    class HighWater(cp.cuda.MemoryHook):
        name = "bench_high_water"

        def __init__(self):
            super().__init__()
            self.used = 0
            self.peak = 0

        def malloc_postprocess(
            self, device_id, size, mem_size, mem_ptr, pmem_id
        ):
            self.used += mem_size
            self.peak = max(self.peak, self.used)

        def free_preprocess(self, device_id, mem_size, mem_ptr, pmem_id):
            self.used -= mem_size

    return HighWater()


class Meter:
    """Seconds and peak bytes of the second of two calls, per backend."""

    def __init__(self, xp):
        self.xp = xp
        self.cupy = xp.__name__ == "cupy"

    def _sync(self):
        if self.cupy:
            self.xp.cuda.Device().synchronize()

    def run(self, label, fn):
        fn()
        self._sync()
        if self.cupy:
            cache = self.xp.fft.config.get_plan_cache()
            cache.clear()
            self.xp.get_default_memory_pool().free_all_blocks()
            hook = high_water_hook(self.xp)
            with hook:
                start = time.perf_counter()
                out = fn()
                self._sync()
                seconds = time.perf_counter() - start
            peak = hook.peak
            plan_bytes = cache.get_curr_memsize()
            extra = f"   plan cache {plan_bytes / 2**30:7.3f} GiB"
        else:
            tracemalloc.start()
            start = time.perf_counter()
            out = fn()
            seconds = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            extra = ""
        print(
            f"{label:<28s} {seconds:9.3f} s   peak {peak / 2**30:7.3f} GiB"
            + extra
        )
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
    stamp_np, stamp_origin = resampling_stamp(args.size)
    window = window_side(PSF_SIGMA)
    print(
        f"{args.backend}: {args.size} x {args.size} frame, "
        f"{fit.basis_kernels.shape[0] + 1} coefficients, "
        f"{args.positions} positions, "
        f"{stamp_np.shape[0]} x {stamp_np.shape[1]} resampling stamp "
        f"({stamp_origin}), {window} x {window} PSF window"
    )
    reference_d = xp.asarray(reference)
    target_variance_d = xp.asarray(target_variance)
    reference_variance_d = xp.asarray(reference_variance)
    stamp = xp.asarray(stamp_np)
    weights = xp.asarray(
        psf_weights(PSF_SIGMA, window, offset=xy - np.rint(xy)),
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
