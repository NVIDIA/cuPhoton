# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched Levenberg-Marquardt refinement of damped modes on NumPy or CuPy.

Many detector-row traces share one time axis and one model shape (K damped
modes plus a constant), so the refinement of :mod:`mode_refinement` can be
run for a whole batch at once: analytic Jacobians for all traces by
broadcasting, normal equations solved as a batch of small (4K+1) systems,
and per-trace damping with accept/reject. The array module is a parameter,
so the same code runs on NumPy and CuPy; the parameters are the same
(amplitude, decay, angular frequency, phase per mode, then the constant)
and, on convergence, the result matches the per-trace SciPy solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


def _sync(xp) -> None:
    dev = getattr(getattr(xp, "cuda", None), "Device", None)
    if dev is not None:
        dev().synchronize()


def model_and_jacobian_batched(xp, theta, time, n_modes: int):
    """theta (B, 4K+1), time (N,) -> model (B, N), jacobian (B, N, 4K+1)."""
    b = theta.shape[0]
    n = time.shape[0]
    p = 4 * n_modes + 1
    t = time[None, :]
    model = xp.broadcast_to(theta[:, 4 * n_modes][:, None], (b, n)).copy()
    jac = xp.empty((b, n, p), dtype=theta.dtype)
    jac[:, :, 4 * n_modes] = 1.0
    for k in range(n_modes):
        a = theta[:, 4 * k][:, None]
        d = theta[:, 4 * k + 1][:, None]
        w = theta[:, 4 * k + 2][:, None]
        ph = theta[:, 4 * k + 3][:, None]
        env = xp.exp(-d * t)
        arg = w * t + ph
        cos = xp.cos(arg)
        sin = xp.sin(arg)
        model += a * env * cos
        jac[:, :, 4 * k] = env * cos
        jac[:, :, 4 * k + 1] = -a * t * env * cos
        jac[:, :, 4 * k + 2] = -a * t * env * sin
        jac[:, :, 4 * k + 3] = -a * env * sin
    return model, jac


@dataclass(frozen=True)
class BatchedRefinement:
    """Refined parameters for a batch: ``theta`` (B, 4K+1) in the same
    layout as the input, per-trace residual rms, convergence flags, the
    number of iterations used and the wall time including any device
    synchronisation."""

    theta: Any
    residual_rms: Any
    converged: Any
    iterations: int
    elapsed_s: float


def refine_modes_batched(
    xp,
    time,
    traces,
    theta0,
    n_modes: int,
    *,
    max_iter: int = 60,
    tol: float = 1e-9,
    lambda0: float = 1e-3,
) -> BatchedRefinement:
    """Levenberg-Marquardt over a batch of traces (B, N) from starts
    ``theta0`` (B, 4K+1). Damping is per trace: a step that lowers the
    cost is accepted and its damping divided by 3, otherwise the damping
    is multiplied by 5 and the trace keeps its parameters. Iteration
    stops when every trace has converged or ``max_iter`` is reached."""
    t = xp.asarray(time, dtype=xp.float64)
    y = xp.asarray(traces, dtype=xp.float64)
    theta = xp.asarray(theta0, dtype=xp.float64).copy()
    if y.ndim != 2 or theta.ndim != 2 or theta.shape[0] != y.shape[0]:
        raise ValueError("traces must be (B, N) and theta0 (B, 4K+1)")
    if theta.shape[1] != 4 * n_modes + 1:
        raise ValueError("theta0 must have 4 * n_modes + 1 columns")
    _sync(xp)
    start = perf_counter()
    b = y.shape[0]
    lam = xp.full(b, lambda0, dtype=xp.float64)
    model, jac = model_and_jacobian_batched(xp, theta, t, n_modes)
    resid = model - y
    cost = xp.sum(resid * resid, axis=1)
    converged = xp.zeros(b, dtype=bool)
    eye = xp.eye(theta.shape[1], dtype=xp.float64)[None, :, :]
    iterations = 0
    for iterations in range(1, max_iter + 1):
        jtj = xp.einsum("bnp,bnq->bpq", jac, jac)
        jtr = xp.einsum("bnp,bn->bp", jac, resid)
        diag = jtj * eye
        step = -xp.linalg.solve(
            jtj + lam[:, None, None] * diag, jtr[:, :, None]
        )[:, :, 0]
        trial = theta + step
        model_t, jac_t = model_and_jacobian_batched(xp, trial, t, n_modes)
        resid_t = model_t - y
        cost_t = xp.sum(resid_t * resid_t, axis=1)
        accept = (cost_t < cost) & ~converged
        scale = xp.sqrt(xp.sum(theta * theta, axis=1)) + 1e-12
        small = xp.sqrt(xp.sum(step * step, axis=1)) < tol * scale
        theta = xp.where(accept[:, None], trial, theta)
        resid = xp.where(accept[:, None], resid_t, resid)
        jac = xp.where(accept[:, None, None], jac_t, jac)
        cost = xp.where(accept, cost_t, cost)
        lam = xp.where(accept, lam / 3.0, lam * 5.0)
        converged = converged | (accept & small)
        if bool(xp.all(converged)):
            break
    _sync(xp)
    rms = xp.sqrt(cost / y.shape[1])
    return BatchedRefinement(
        theta=theta,
        residual_rms=rms,
        converged=converged,
        iterations=iterations,
        elapsed_s=perf_counter() - start,
    )


@dataclass(frozen=True)
class RefinementBenchmark:
    traces: int
    samples: int
    n_modes: int
    repeat: int
    cpu_serial_scipy_s: float
    numpy_batched_s: float
    cupy_batched_s: float | None
    max_abs_theta_diff_numpy: float
    max_abs_theta_diff_cupy: float | None
    gpu_error: str | None


def benchmark_refinement(
    *,
    samples: int = 96,
    traces: int = 256,
    n_modes: int = 2,
    noise_sigma: float = 0.02,
    seed: int = 0,
    run_gpu: bool = True,
    repeat: int = 3,
) -> RefinementBenchmark:
    """Time the per-trace SciPy refinement against the batched solver on
    NumPy and on CuPy, from identical perturbed starts, and report the
    largest parameter difference between the batched and SciPy results.

    Every path is timed ``repeat`` times and the best time is reported.
    The batched timings cover the solver only: the arrays are on the
    device before the clock starts and the device is synchronised before
    and after. The SciPy timing covers the Python loop over traces."""
    if repeat < 1:
        raise ValueError("repeat must be positive")
    from .mode_refinement import refine_modes
    from .synthetic_validation import (
        DEFAULT_MODES,
        modes_to_theta,
        synthetic_modes_trace,
    )

    modes = DEFAULT_MODES[:n_modes]
    fx = synthetic_modes_trace(samples, modes=modes)
    rng = np.random.default_rng(seed)
    y = fx.clean[None, :] + rng.normal(0.0, noise_sigma, (traces, samples))
    truth = modes_to_theta(modes, fx.constant)
    theta0 = truth[None, :] * (1.0 + 0.05 * rng.standard_normal((traces, 1)))
    theta0 = theta0 + 0.02 * rng.standard_normal(theta0.shape)

    # warm the device path once so compile and allocation costs are not
    # charged to the timed run
    if run_gpu:
        try:
            import cupy as cp

            refine_modes_batched(cp, fx.time, y[:16], theta0[:16], n_modes)
        except Exception:  # pragma: no cover - reported below
            pass

    def serial_scipy():
        out = np.empty_like(theta0)
        for i in range(traces):
            init = [
                tuple(theta0[i, 4 * k : 4 * k + 4]) for k in range(n_modes)
            ]
            r = refine_modes(fx.time, y[i], init, float(theta0[i, -1]))
            # refine_modes normalises signs and orders by amplitude; undo
            # the ordering by matching frequencies to the start
            for k in range(n_modes):
                j = int(
                    np.argmin(
                        np.abs(r.angular_frequency - theta0[i, 4 * k + 2])
                    )
                )
                out[i, 4 * k : 4 * k + 4] = (
                    r.amplitude[j],
                    r.decay[j],
                    r.angular_frequency[j],
                    r.phase[j],
                )
            out[i, -1] = r.constant
        return out

    cpu_serial_s = float("inf")
    for _ in range(repeat):
        start = perf_counter()
        serial = serial_scipy()
        cpu_serial_s = min(cpu_serial_s, perf_counter() - start)

    res_np = min(
        (
            refine_modes_batched(np, fx.time, y, theta0, n_modes)
            for _ in range(repeat)
        ),
        key=lambda r: r.elapsed_s,
    )
    diff_np = float(
        np.max(np.abs(_canonical(res_np.theta, n_modes) - serial))
    )

    cupy_s = None
    diff_cp = None
    gpu_error = None
    if run_gpu:
        try:
            import cupy as cp

            res_cp = min(
                (
                    refine_modes_batched(cp, fx.time, y, theta0, n_modes)
                    for _ in range(repeat)
                ),
                key=lambda r: r.elapsed_s,
            )
            theta_cp = cp.asnumpy(res_cp.theta)
            cupy_s = res_cp.elapsed_s
            diff_cp = float(
                np.max(np.abs(_canonical(theta_cp, n_modes) - serial))
            )
        except Exception as exc:  # pragma: no cover - depends on hardware
            gpu_error = f"{type(exc).__name__}: {exc}"
    return RefinementBenchmark(
        traces=traces,
        samples=samples,
        n_modes=n_modes,
        repeat=repeat,
        cpu_serial_scipy_s=cpu_serial_s,
        numpy_batched_s=res_np.elapsed_s,
        cupy_batched_s=cupy_s,
        max_abs_theta_diff_numpy=diff_np,
        max_abs_theta_diff_cupy=diff_cp,
        gpu_error=gpu_error,
    )


def _canonical(theta: np.ndarray, n_modes: int) -> np.ndarray:
    """Same sign and phase conventions as :func:`refine_modes`."""
    out = np.array(theta, dtype=np.float64, copy=True)
    for k in range(n_modes):
        a = out[:, 4 * k]
        w = out[:, 4 * k + 2]
        ph = out[:, 4 * k + 3]
        neg = a < 0
        a[neg] = -a[neg]
        ph[neg] += np.pi
        negf = w < 0
        w[negf] = -w[negf]
        ph[negf] = -ph[negf]
        out[:, 4 * k + 3] = (ph + np.pi) % (2.0 * np.pi) - np.pi
    return out
