# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nonlinear least-squares refinement of damped modes, initialised by
linear prediction.

Linear prediction gives frequencies and decays without an initial guess,
but it is not a maximum-likelihood estimator and its root filter can drop a
lightly damped mode under noise. Refining the same damped-mode model by
nonlinear least squares from the linear-prediction result recovers the
Cramer-Rao bound, and a mode the filter dropped can be seeded from the
peak of the residual spectrum before refinement. The model is

    trace(t) = c + sum_k A_k exp(-d_k t) cos(w_k t + p_k)

with an analytic Jacobian; the solver is ``scipy.optimize.least_squares``.
NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .linear_prediction import linear_prediction_numpy


@dataclass(frozen=True)
class RefinedModes:
    """Refined damped modes and the fit diagnostics.

    Arrays are ordered by descending amplitude. ``sigma_*`` are one-sigma
    uncertainties from the Jacobian at the solution and the residual
    variance; ``initial_mode_count`` is how many oscillating modes the
    linear-prediction step supplied and ``seeded_mode_count`` how many had
    to be seeded from the residual spectrum.
    """

    time: np.ndarray
    amplitude: np.ndarray
    decay: np.ndarray
    angular_frequency: np.ndarray
    phase: np.ndarray
    constant: float
    sigma_amplitude: np.ndarray
    sigma_decay: np.ndarray
    sigma_angular_frequency: np.ndarray
    sigma_phase: np.ndarray
    reconstruction: np.ndarray
    residual_rms: float
    converged: bool
    iterations: int
    initial_mode_count: int
    seeded_mode_count: int


def _model_and_jacobian(theta: np.ndarray, time: np.ndarray, n: int):
    model = np.full(time.shape, float(theta[4 * n]))
    jac = np.empty((time.size, 4 * n + 1))
    jac[:, 4 * n] = 1.0
    for k in range(n):
        a, d, w, p = theta[4 * k : 4 * k + 4]
        env = np.exp(-d * time)
        arg = w * time + p
        cos = np.cos(arg)
        sin = np.sin(arg)
        model += a * env * cos
        jac[:, 4 * k] = env * cos
        jac[:, 4 * k + 1] = -a * time * env * cos
        jac[:, 4 * k + 2] = -a * time * env * sin
        jac[:, 4 * k + 3] = -a * env * sin
    return model, jac


def seed_from_residual(
    time: np.ndarray, residual: np.ndarray, pad: int = 4096
) -> tuple[float, float, float, float]:
    """Seed one damped mode from the strongest peak of the residual
    spectrum: amplitude from the peak height, zero decay, zero phase."""
    dt = float(time[1] - time[0])
    r = residual - residual.mean()
    spec = np.abs(np.fft.rfft(r, pad))
    freq = 2.0 * np.pi * np.fft.rfftfreq(pad, d=dt)
    k = int(np.argmax(spec[1:])) + 1
    return float(2.0 * spec[k] / time.size), 0.0, float(freq[k]), 0.0


def refine_modes(
    time: np.ndarray,
    trace: np.ndarray,
    initial_modes: list[tuple[float, float, float, float]],
    constant: float,
    *,
    max_nfev: int = 2000,
    initial_mode_count: int | None = None,
    seeded_mode_count: int = 0,
) -> RefinedModes:
    """Refine ``initial_modes`` (amplitude, decay, angular frequency, phase)
    and the constant by nonlinear least squares."""
    time = np.asarray(time, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    n = len(initial_modes)
    if n == 0:
        raise ValueError("at least one initial mode is required")
    theta0 = np.array(
        [v for m in initial_modes for v in m] + [constant], dtype=np.float64
    )

    def residual(theta):
        return _model_and_jacobian(theta, time, n)[0] - trace

    def jacobian(theta):
        return _model_and_jacobian(theta, time, n)[1]

    res = least_squares(
        residual,
        theta0,
        jac=jacobian,
        method="lm",
        xtol=1e-12,
        ftol=1e-12,
        max_nfev=max_nfev,
    )
    theta = res.x
    model, jac = _model_and_jacobian(theta, time, n)
    resid = trace - model
    dof = max(time.size - theta.size, 1)
    var = float(resid @ resid) / dof
    try:
        cov = np.linalg.inv(jac.T @ jac) * var
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        sigma = np.full(theta.size, np.nan)
    amp = theta[0 : 4 * n : 4].copy()
    dec = theta[1 : 4 * n : 4].copy()
    freq = theta[2 : 4 * n : 4].copy()
    ph = theta[3 : 4 * n : 4].copy()
    # a negative amplitude or frequency is the same mode with a shifted phase
    neg = amp < 0
    amp[neg] = -amp[neg]
    ph[neg] = ph[neg] + np.pi
    negf = freq < 0
    freq[negf] = -freq[negf]
    ph[negf] = -ph[negf]
    ph = (ph + np.pi) % (2.0 * np.pi) - np.pi
    order = np.argsort(-amp)
    sig = sigma.reshape(-1)
    return RefinedModes(
        time=time,
        amplitude=amp[order],
        decay=dec[order],
        angular_frequency=freq[order],
        phase=ph[order],
        constant=float(theta[4 * n]),
        sigma_amplitude=sig[0 : 4 * n : 4][order],
        sigma_decay=sig[1 : 4 * n : 4][order],
        sigma_angular_frequency=sig[2 : 4 * n : 4][order],
        sigma_phase=sig[3 : 4 * n : 4][order],
        reconstruction=model,
        residual_rms=float(np.sqrt(np.mean(resid**2))),
        converged=bool(res.success),
        iterations=int(res.nfev),
        initial_mode_count=(
            n if initial_mode_count is None else initial_mode_count
        ),
        seeded_mode_count=seeded_mode_count,
    )


def linear_prediction_refined(
    time,
    trace,
    n_components: int,
    n_modes: int,
    *,
    roots_backend: str = "eigvals",
) -> RefinedModes:
    """Linear prediction, then nonlinear refinement of the ``n_modes``
    strongest oscillating modes; modes the linear-prediction step did not
    return are seeded one at a time from the residual spectrum."""
    time = np.asarray(time, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    lp = linear_prediction_numpy(
        time, trace, n_components, roots_backend=roots_backend
    )
    w = np.asarray(lp.angular_frequency, dtype=float)
    d = np.asarray(lp.decay, dtype=float)
    a = np.asarray(lp.amplitude, dtype=float)
    p = np.asarray(lp.phase, dtype=float)
    modes = [
        (float(a[i]), float(d[i]), float(w[i]), float(p[i]))
        for i in range(w.size)
        if w[i] > 0
    ]
    modes.sort(key=lambda m: -m[0])
    modes = modes[:n_modes]
    found = len(modes)
    constant = float(np.mean(trace))
    residual = trace - np.asarray(lp.reconstruction, dtype=float)
    seeded = 0
    while len(modes) < n_modes:
        modes.append(seed_from_residual(time, residual))
        seeded += 1
        partial = refine_modes(time, trace, modes, constant)
        residual = trace - partial.reconstruction
        modes = [
            (
                float(partial.amplitude[i]),
                float(partial.decay[i]),
                float(partial.angular_frequency[i]),
                float(partial.phase[i]),
            )
            for i in range(len(modes))
        ]
        constant = partial.constant
    return refine_modes(
        time,
        trace,
        modes,
        constant,
        initial_mode_count=found,
        seeded_mode_count=seeded,
    )
