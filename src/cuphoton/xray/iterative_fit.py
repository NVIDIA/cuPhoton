# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional damped-cosine fitting with NumPy or CuPy.

The model, analytic Jacobian, frequency transform, and LM damping updates
adapt work by Irina Demeshko and Malte Foerster. This implementation uses
array-library linear solves and deterministic FFT/least-squares
initialization.

The minimized cost is ``0.5 * sum(residual**2) + amplitude_l2 * sum(C**2)``.
Thus amplitude regularization adds ``2 * amplitude_l2`` to the appropriate
normal-matrix diagonal and ``2 * amplitude_l2 * C`` to its gradient.
The reported ``chi2`` remains the unregularized mean squared residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class IterativeFitOptions:
    """Optimizer settings; frequencies are cycles per input time unit.

    ``max_frequency=None`` uses the sampling Nyquist frequency. ``tolerance``
    bounds the gradient divided by Jacobian-column norms, relative to
    ``max(1, residual norm)``. It describes numerical stationarity, not fit
    quality or science acceptance. Regularization defaults to zero. A negative
    ``min_frequency`` moves the lower optimizer boundary below zero; FFT
    scouting still uses nonnegative frequencies and outputs are sign-folded.
    """

    max_iterations: int = 600
    tolerance: float = 1e-8
    amplitude_l2: float = 0.0
    min_frequency: float = 0.0
    max_frequency: float | None = None

    def validate(self) -> None:
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or self.max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        for name in ("tolerance", "amplitude_l2", "min_frequency"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.amplitude_l2 < 0:
            raise ValueError("amplitude_l2 must be nonnegative")
        if self.max_frequency is not None and (
            isinstance(self.max_frequency, bool)
            or not isinstance(self.max_frequency, Real)
            or not np.isfinite(self.max_frequency)
            or self.max_frequency <= max(0.0, self.min_frequency)
        ):
            raise ValueError(
                "max_frequency must be finite, positive, "
                "and exceed min_frequency"
            )

    def to_dict(self) -> dict[str, int | float | None]:
        self.validate()
        return {
            "max_iterations": int(self.max_iterations),
            "tolerance": float(self.tolerance),
            "amplitude_l2": float(self.amplitude_l2),
            "min_frequency": float(self.min_frequency),
            "max_frequency": (
                None
                if self.max_frequency is None
                else float(self.max_frequency)
            ),
        }


@dataclass(frozen=True)
class IterativeFitResult:
    """Host arrays and diagnostics for one fitted trace.

    Modal phases refer to ``time - time[0]``. ``time_components`` has shape
    ``(samples, modes)``; its sum plus ``offset`` equals ``reconstruction``.
    ``angular_frequency`` uses radians per input-time unit; divide by
    ``2*pi`` for cycles per input-time unit. Decay rates are inverse
    input-time units and amplitudes are nonnegative in input-trace units.
    ``gradient_norm`` is the infinity norm of the unscaled objective
    gradient. Status is one of
    ``converged``, ``constant``, ``iteration-limit``, ``stalled``, or
    ``nonfinite``; the first two set ``converged=True``. Unsuccessful fits
    retain their best finite iterate. A constant trace has no modes and
    zero relative residual.
    """

    backend: str
    time: np.ndarray
    time_components: np.ndarray
    reconstruction: np.ndarray
    angular_frequency: np.ndarray
    decay: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    offset: float
    chi2: float
    trace_std: float
    residual_std: float
    relative_residual: float
    converged: bool
    status: str
    iterations: int
    cost: float
    gradient_norm: float


def _evaluate(
    xp, time, parameters, components, omega_min, omega_max, *, jacobian=True
):
    """Evaluate the legacy model and its analytic parameter Jacobian."""
    modes = parameters[: 4 * components].reshape(components, 4)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        decay = xp.exp(modes[:, 0, None])
        amplitude = modes[:, 1, None]
        phase = modes[:, 2, None]
        u = modes[:, 3, None]
        span = omega_max - omega_min
        omega = omega_min + span * (xp.arctan(u) / np.pi + 0.5)
        envelope = xp.exp(-decay * time)
        angle = omega * time + phase
        cosine = envelope * xp.cos(angle)
        values = xp.sum(amplitude * cosine, axis=0) + parameters[-1]
        if not jacobian:
            return values, None
        d_phase = -amplitude * envelope * xp.sin(angle)
        blocks = xp.stack(
            (
                -amplitude * time * decay * cosine,
                cosine,
                d_phase,
                d_phase * time * span / (np.pi * (1.0 + u * u)),
            ),
            axis=1,
        )
    jac = xp.concatenate(
        (
            blocks.reshape(4 * components, len(time)).T,
            xp.ones((len(time), 1), dtype=xp.float64),
        ),
        axis=1,
    )
    return values, jac


def _normal_equations(xp, jacobian, residual, parameters, amplitude_l2):
    normal = jacobian.T @ jacobian
    gradient = jacobian.T @ residual
    c_indices = xp.arange(1, len(parameters) - 1, 4)
    amplitudes = parameters[c_indices]
    normal[c_indices, c_indices] += 2.0 * amplitude_l2
    gradient[c_indices] += 2.0 * amplitude_l2 * amplitudes
    cost = 0.5 * xp.sum(residual**2) + amplitude_l2 * xp.sum(amplitudes**2)
    return normal, gradient, float(cost)


def _wrap_phase(xp, phase):
    """Reduce phases before evaluation, retaining already reduced values."""
    with np.errstate(invalid="ignore"):
        return xp.where(
            (phase >= -np.pi) & (phase < np.pi),
            phase,
            (phase + np.pi) % (2.0 * np.pi) - np.pi,
        )


def _normalize_phases(xp, parameters):
    parameters = parameters.copy()
    parameters[2:-1:4] = _wrap_phase(xp, parameters[2:-1:4])
    return parameters


def _initial_parameters(
    time, trace, components, min_frequency, max_frequency
):
    """Seed separated FFT peaks, then solve linear amplitudes and phases."""
    from scipy.signal import find_peaks

    duration = time[-1]
    spectrum = np.fft.rfft(
        (trace - trace.mean()) * np.hanning(len(time)), n=4 * len(time)
    )
    frequencies = np.fft.rfftfreq(4 * len(time), d=duration / (len(time) - 1))
    magnitude = np.abs(spectrum)
    scout_minimum = max(0.0, min_frequency)
    band = (frequencies > scout_minimum) & (frequencies < max_frequency)
    restricted = np.where(band, magnitude, 0.0)
    peaks, _ = find_peaks(restricted)
    # Stable ties make initialization reproducible across runs.
    peaks = peaks[np.lexsort((peaks, -magnitude[peaks]))]
    selected = list(frequencies[peaks[:components]])
    # Extra requested modes begin away from the detected peaks. The optimizer
    # can reduce their amplitudes; no detector-specific gate is applied.
    candidates = np.linspace(
        scout_minimum, max_frequency, 4 * components + 2
    )[1:-1]
    while len(selected) < components:
        if selected:
            distance = np.min(
                np.abs(candidates[:, None] - np.asarray(selected)), axis=1
            )
            selected.append(float(candidates[np.argmax(distance)]))
        else:
            selected.append(float(candidates[len(candidates) // 2]))
    selected = np.sort(np.asarray(selected))
    decay = 1.0 / duration
    omega = 2.0 * np.pi * selected
    envelope = np.exp(-decay * time)
    design = np.empty((len(time), 2 * components + 1))
    design[:, 0:-1:2] = envelope[:, None] * np.cos(time[:, None] * omega)
    design[:, 1:-1:2] = envelope[:, None] * np.sin(time[:, None] * omega)
    design[:, -1] = 1.0
    coefficients = np.linalg.lstsq(design, trace, rcond=None)[0]
    parameters = np.empty(4 * components + 1)
    parameters[0:-1:4] = np.log(decay)
    parameters[1:-1:4] = np.hypot(coefficients[0:-1:2], coefficients[1:-1:2])
    parameters[2:-1:4] = np.arctan2(
        -coefficients[1:-1:2], coefficients[0:-1:2]
    )
    fraction = (selected - min_frequency) / (max_frequency - min_frequency)
    parameters[3:-1:4] = np.tan(
        np.pi * (np.clip(fraction, 1e-6, 1.0 - 1e-6) - 0.5)
    )
    parameters[-1] = coefficients[-1]
    return parameters


def _array_module(backend):
    if backend == "cpu":
        return np
    if backend != "gpu":
        raise ValueError("backend must be 'cpu' or 'gpu'")
    try:
        import cupy
    except ImportError as exc:
        raise RuntimeError(
            "GPU iterative fitting requires CuPy. Install 'cuphoton[gpu]' "
            "or run 'uv sync --extra gpu'."
        ) from exc
    return cupy


def _host_array(xp, values, name):
    array = xp.asarray(values)
    if xp.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(
        array if xp is np else xp.asnumpy(array), dtype=np.float64
    )
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def iterative_fit(
    time,
    trace,
    components,
    *,
    options: IterativeFitOptions | None = None,
    backend: str = "cpu",
    initial_parameters=None,
) -> IterativeFitResult:
    """Fit one uniformly sampled trace to damped cosines and an offset.

    ``components`` is the number of modes, requiring at least ``4*K+1``
    samples. ``initial_parameters`` optionally supplies ``[log_b, C, phi,
    u, ..., offset]`` with ``4*K+1`` finite entries. Here
    ``omega = 2*pi*min_frequency + 2*pi*(max_frequency-min_frequency) *
    (atan(u)/pi + 0.5)``. Model time starts at the first input sample.
    Initial and trial phases are reduced to ``[-pi, pi)`` before evaluating
    the model, so argument reduction also applies during optimization.

    Initialization runs on the host; the LM iterations use the requested
    backend and float64. GPU requests never fall back to CPU execution.
    Sampling intervals may differ by at most 1e-4 relative to their mean,
    accommodating float32-rounded delay metadata. The model uses the actual
    sample coordinates; FFT initialization uses the mean interval.
    Convergence means a small scaled objective gradient. A small rejected
    step or exhausted iteration budget does not establish convergence.
    """
    options = IterativeFitOptions() if options is None else options
    if not isinstance(options, IterativeFitOptions):
        raise ValueError("options must be IterativeFitOptions")
    options.validate()
    xp = _array_module(backend)
    time_host = _host_array(xp, time, "time")
    trace_host = _host_array(xp, trace, "trace")
    if len(time_host) != len(trace_host) or len(time_host) < 5:
        raise ValueError(
            "time and trace must match and contain at least five samples"
        )
    if (
        isinstance(components, bool)
        or not isinstance(components, Integral)
        or components < 1
        or 4 * components + 1 > len(time_host)
    ):
        raise ValueError(
            "components must be a positive integer "
            "with 4*components+1 <= samples"
        )
    spacing = np.diff(time_host)
    mean_spacing = (time_host[-1] - time_host[0]) / (len(time_host) - 1)
    if np.any(spacing <= 0) or not np.allclose(
        spacing, mean_spacing, rtol=1e-4, atol=abs(mean_spacing) * 1e-12
    ):
        raise ValueError(
            "time must be strictly increasing and uniformly sampled"
        )
    nyquist = 0.5 / mean_spacing
    maximum = (
        nyquist
        if options.max_frequency is None
        else float(options.max_frequency)
    )
    minimum = float(options.min_frequency)
    if not -nyquist <= minimum < maximum <= nyquist or maximum <= 0:
        raise ValueError(
            "frequency bounds require -Nyquist <= min < max <= Nyquist "
            "and max > 0"
        )
    relative_time = time_host - time_host[0]
    if initial_parameters is not None:
        parameters = _host_array(xp, initial_parameters, "initial_parameters")
        if parameters.shape != (4 * components + 1,):
            raise ValueError(
                "initial_parameters must contain 4*components+1 values"
            )
    elif np.all(trace_host == trace_host[0]):
        return _result(
            backend,
            time_host,
            trace_host,
            np.array([trace_host[0]]),
            0,
            2.0 * np.pi * minimum,
            2.0 * np.pi * maximum,
            "constant",
            0,
            0.0,
            0.0,
        )
    else:
        parameters = _initial_parameters(
            relative_time, trace_host, components, minimum, maximum
        )
    device_time = xp.asarray(relative_time)
    device_trace = xp.asarray(trace_host)
    parameters = _normalize_phases(xp, xp.asarray(parameters))
    omega_min, omega_max = 2.0 * np.pi * minimum, 2.0 * np.pi * maximum
    values, jacobian = _evaluate(
        xp, device_time, parameters, components, omega_min, omega_max
    )
    if not bool(xp.all(xp.isfinite(values))) or not bool(
        xp.all(xp.isfinite(jacobian))
    ):
        raise ValueError(
            "initial_parameters produce a nonfinite model or Jacobian"
        )
    normal, gradient, cost = _normal_equations(
        xp, jacobian, values - device_trace, parameters, options.amplitude_l2
    )
    if (
        not np.isfinite(cost)
        or not bool(xp.all(xp.isfinite(normal)))
        or not bool(xp.all(xp.isfinite(gradient)))
    ):
        raise ValueError(
            "initial fit objective or normal equations are nonfinite"
        )
    damping = 1e-3
    status, iterations = "iteration-limit", 0
    for iteration in range(options.max_iterations + 1):
        scaled_gradient = xp.abs(gradient) / xp.maximum(
            xp.sqrt(xp.diag(normal)), np.finfo(float).tiny
        )
        threshold = options.tolerance * max(
            1.0, float(xp.linalg.norm(values - device_trace))
        )
        if float(xp.max(scaled_gradient)) <= threshold:
            status = "converged"
            break
        if iteration == options.max_iterations:
            break
        iterations = iteration + 1
        system = normal + damping * xp.eye(len(parameters), dtype=xp.float64)
        try:
            step = xp.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            step = xp.full_like(parameters, xp.nan)
        candidate = _normalize_phases(xp, parameters + step)
        candidate_finite = bool(xp.all(xp.isfinite(candidate)))
        candidate_values, _ = _evaluate(
            xp,
            device_time,
            candidate,
            components,
            omega_min,
            omega_max,
            jacobian=False,
        )
        candidate_cost = float(
            0.5 * xp.sum((candidate_values - device_trace) ** 2)
            + options.amplitude_l2 * xp.sum(candidate[1:-1:4] ** 2)
        )
        accepted = (
            candidate_finite
            and np.isfinite(candidate_cost)
            and candidate_cost < cost
        )
        if accepted:
            new_values, new_jacobian = _evaluate(
                xp, device_time, candidate, components, omega_min, omega_max
            )
            new_normal, new_gradient, new_cost = _normal_equations(
                xp,
                new_jacobian,
                new_values - device_trace,
                candidate,
                options.amplitude_l2,
            )
            accepted = bool(xp.all(xp.isfinite(new_normal))) and bool(
                xp.all(xp.isfinite(new_gradient))
            )
        if accepted:
            parameters, values = candidate, new_values
            normal, gradient, cost = new_normal, new_gradient, new_cost
            damping = max(1e-12, damping / 10.0)
        elif damping >= 1e10:
            status = (
                "stalled"
                if candidate_finite and np.isfinite(candidate_cost)
                else "nonfinite"
            )
            break
        else:
            damping = min(1e10, damping * 10.0)
    parameters_host = np.asarray(
        parameters if xp is np else xp.asnumpy(parameters)
    )
    return _result(
        backend,
        time_host,
        trace_host,
        parameters_host,
        components,
        omega_min,
        omega_max,
        status,
        iterations,
        options.amplitude_l2,
        float(xp.max(xp.abs(gradient))),
    )


def _result(
    backend,
    time,
    trace,
    parameters,
    components,
    omega_min,
    omega_max,
    status,
    iterations,
    amplitude_l2,
    gradient_norm,
):
    modes = parameters[: 4 * components].reshape(components, 4)
    decay = np.exp(modes[:, 0])
    amplitude = np.abs(modes[:, 1])
    phase = modes[:, 2]
    omega = omega_min + (omega_max - omega_min) * (
        np.arctan(modes[:, 3]) / np.pi + 0.5
    )
    phase = np.where(omega < 0, -phase, phase)
    omega = np.abs(omega)
    phase = phase + np.where(modes[:, 1] < 0, np.pi, 0.0)
    phase = _wrap_phase(np, phase)
    order = np.argsort(omega, kind="stable")
    decay, amplitude, phase, omega = (
        array[order] for array in (decay, amplitude, phase, omega)
    )
    relative_time = time[:, None] - time[0]
    time_components = (
        amplitude
        * np.exp(-relative_time * decay)
        * np.cos(relative_time * omega + phase)
    )
    offset = float(parameters[-1])
    reconstruction = time_components.sum(axis=1) + offset
    residual = reconstruction - trace
    trace_std, residual_std = float(np.std(trace)), float(np.std(residual))
    return IterativeFitResult(
        backend=backend,
        time=time.copy(),
        time_components=time_components,
        reconstruction=reconstruction,
        angular_frequency=omega,
        decay=decay,
        amplitude=amplitude,
        phase=phase,
        offset=offset,
        chi2=float(np.mean(residual**2)),
        trace_std=trace_std,
        residual_std=residual_std,
        relative_residual=(
            residual_std / trace_std
            if trace_std > 0
            else (0.0 if residual_std == 0 else float("inf"))
        ),
        converged=status in ("converged", "constant"),
        status=status,
        iterations=iterations,
        cost=float(
            0.5 * np.sum(residual**2) + amplitude_l2 * np.sum(amplitude**2)
        ),
        gradient_norm=gradient_norm,
    )
