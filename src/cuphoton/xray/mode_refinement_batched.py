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
(amplitude, decay, angular frequency, phase per mode, then the constant).
Both solvers find local optima; convergence alone does not establish that
they found the same optimum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np


def _sync(xp) -> None:
    dev = getattr(getattr(xp, "cuda", None), "Device", None)
    if dev is not None:
        dev().synchronize()


def _wrap_phase(xp, phase):
    return xp.where(
        (phase >= -xp.pi) & (phase < xp.pi),
        phase,
        (phase + xp.pi) % (2.0 * xp.pi) - xp.pi,
    )


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
        ph = _wrap_phase(xp, theta[:, 4 * k + 3][:, None])
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
    stops when every trace has converged or ``max_iter`` is reached.
    Convergence requires a small gradient in Jacobian-scaled coordinates;
    a step made tiny by damping does not imply stationarity. Scaling the
    Jacobian columns makes the solve and stopping criterion independent of
    parameter units. Positive damping also handles zero Jacobian columns
    from an initial zero-amplitude mode.
    """
    for name, count in (("n_modes", n_modes), ("max_iter", max_iter)):
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or count < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("tol", tol), ("lambda0", lambda0)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    for name, values in (
        ("time", time),
        ("traces", traces),
        ("theta0", theta0),
    ):
        if xp.iscomplexobj(values):
            raise ValueError(f"{name} must contain only real values")
    t = xp.asarray(time, dtype=xp.float64)
    y = xp.asarray(traces, dtype=xp.float64)
    theta = xp.asarray(theta0, dtype=xp.float64).copy()
    if y.ndim != 2 or theta.ndim != 2 or theta.shape[0] != y.shape[0]:
        raise ValueError("traces must be (B, N) and theta0 (B, 4K+1)")
    if theta.shape[1] != 4 * n_modes + 1:
        raise ValueError("theta0 must have 4 * n_modes + 1 columns")
    if t.ndim != 1 or t.shape[0] != y.shape[1]:
        raise ValueError("time must be (N,) matching traces (B, N)")
    if y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("traces must contain at least one trace and sample")
    for name, values in (("time", t), ("traces", y), ("theta0", theta)):
        if not bool(xp.all(xp.isfinite(values))):
            raise ValueError(f"{name} must contain only finite values")
    theta[:, 3 : 4 * n_modes : 4] = _wrap_phase(
        xp, theta[:, 3 : 4 * n_modes : 4]
    )
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
    for _ in range(max_iter):
        iterations += 1
        scale = xp.sqrt(xp.sum(jac * jac, axis=1))
        scale = xp.where(scale > 0, scale, 1.0)
        scaled_jac = jac / scale[:, None, :]
        jtj = xp.einsum("bnp,bnq->bpq", scaled_jac, scaled_jac)
        jtr = xp.einsum("bnp,bn->bp", scaled_jac, resid)
        stationary = xp.max(xp.abs(jtr), axis=1) <= (
            tol * xp.maximum(xp.sqrt(cost), 1.0)
        )
        converged = converged | stationary
        if bool(xp.all(converged)):
            break
        step = (
            -xp.linalg.solve(jtj + lam[:, None, None] * eye, jtr[:, :, None])[
                :, :, 0
            ]
            / scale
        )
        trial = theta + step
        model_t, jac_t = model_and_jacobian_batched(xp, trial, t, n_modes)
        resid_t = model_t - y
        cost_t = xp.sum(resid_t * resid_t, axis=1)
        accept = (cost_t < cost) & ~converged
        theta = xp.where(accept[:, None], trial, theta)
        resid = xp.where(accept[:, None], resid_t, resid)
        jac = xp.where(accept[:, None, None], jac_t, jac)
        cost = xp.where(accept, cost_t, cost)
        lam = xp.where(accept, lam / 3.0, lam * 5.0)
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
    """Timings and fit quality from each path's fastest repetition.

    Convergence counts and worst residual RMS accompany parameter agreement;
    a fast or matching result does not establish that every fit converged.
    """

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
    cpu_serial_converged: int
    numpy_batched_converged: int
    cupy_batched_converged: int | None
    cpu_serial_max_residual_rms: float
    numpy_batched_max_residual_rms: float
    cupy_batched_max_residual_rms: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"traces={self.traces} samples={self.samples} "
            f"repeat={self.repeat}",
            f"cpu_serial_scipy_s={self.cpu_serial_scipy_s:.6g}",
            f"numpy_batched_s={self.numpy_batched_s:.6g}",
            f"max_abs_theta_diff_numpy={self.max_abs_theta_diff_numpy:.3g}",
            f"cpu_serial_converged={self.cpu_serial_converged}/{self.traces}",
            f"numpy_batched_converged={self.numpy_batched_converged}/{self.traces}",
            f"cpu_serial_max_residual_rms={self.cpu_serial_max_residual_rms:.6g}",
            f"numpy_batched_max_residual_rms={self.numpy_batched_max_residual_rms:.6g}",
        ]
        if self.gpu_error:
            lines.extend(
                ("gpu_status=unavailable", f"gpu_error={self.gpu_error}")
            )
        elif self.cupy_batched_s is None:
            lines.append("gpu_status=skipped")
        else:
            lines.extend(
                (
                    f"cupy_batched_s={self.cupy_batched_s:.6g}",
                    f"max_abs_theta_diff_cupy={self.max_abs_theta_diff_cupy:.3g}",
                    f"cupy_batched_converged={self.cupy_batched_converged}/{self.traces}",
                    f"cupy_batched_max_residual_rms={self.cupy_batched_max_residual_rms:.6g}",
                )
            )
        return "\n".join(lines)


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
    for name, value in (
        ("samples", samples),
        ("traces", traces),
        ("n_modes", n_modes),
        ("repeat", repeat),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("noise_sigma must be finite and nonnegative")
    from .mode_refinement import refine_modes
    from .synthetic_validation import (
        DEFAULT_MODES,
        modes_to_theta,
        synthetic_modes_trace,
    )

    if n_modes > len(DEFAULT_MODES):
        raise ValueError(f"n_modes must be at most {len(DEFAULT_MODES)}")
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
        converged = 0
        max_residual_rms = 0.0
        for i in range(traces):
            init = [
                tuple(theta0[i, 4 * k : 4 * k + 4]) for k in range(n_modes)
            ]
            r = refine_modes(fx.time, y[i], init, float(theta0[i, -1]))
            converged += int(r.converged)
            max_residual_rms = max(max_residual_rms, r.residual_rms)
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
        return out, converged, max_residual_rms

    cpu_serial_s = float("inf")
    for _ in range(repeat):
        start = perf_counter()
        candidate = serial_scipy()
        elapsed = perf_counter() - start
        if elapsed < cpu_serial_s:
            cpu_serial_s = elapsed
            serial, serial_converged, serial_max_residual_rms = candidate

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
    cp_converged = None
    cp_max_residual_rms = None
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
            cp_converged = int(cp.count_nonzero(res_cp.converged))
            cp_max_residual_rms = float(cp.max(res_cp.residual_rms))
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
        cpu_serial_converged=serial_converged,
        numpy_batched_converged=int(np.count_nonzero(res_np.converged)),
        cupy_batched_converged=cp_converged,
        cpu_serial_max_residual_rms=serial_max_residual_rms,
        numpy_batched_max_residual_rms=float(np.max(res_np.residual_rms)),
        cupy_batched_max_residual_rms=cp_max_residual_rms,
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
