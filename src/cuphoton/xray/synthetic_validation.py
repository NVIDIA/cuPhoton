# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic damped-mode fixtures and Cramer-Rao validation for the
linear-prediction fit.

The fixture generates traces with known modes so the workflow can be
validated without external datasets. The validation sweep compares the
estimator's scatter and bias with the Cramer-Rao lower bound computed from
the exact Fisher information of the same model, together with the rate at
which true modes are lost from the fit. Everything here runs on NumPy; the
estimator under test is :func:`linear_prediction_numpy` unless another
estimator callable is supplied.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from cuphoton.core.runtime import runtime_metadata

from .linear_prediction import linear_prediction_numpy

SUMMARY_SCHEMA_VERSION = 1
PARAMETERS = ("amplitude", "decay", "angular_frequency", "phase")


@dataclass(frozen=True)
class DampedMode:
    """One real damped sinusoid.

    ``amplitude * exp(-decay * t) * cos(angular_frequency * t + phase)``
    """

    amplitude: float
    decay: float
    angular_frequency: float
    phase: float = 0.0


@dataclass(frozen=True)
class SyntheticModesTrace:
    time: np.ndarray
    trace: np.ndarray
    clean: np.ndarray
    modes: tuple[DampedMode, ...]
    constant: float
    noise_sigma: float
    seed: int | None


DEFAULT_MODES = (
    DampedMode(1.25, 0.09, 2.4, 0.30),
    DampedMode(0.45, 0.03, 0.9, -0.75),
)


def modes_model(
    theta: np.ndarray, time: np.ndarray, n_modes: int
) -> np.ndarray:
    """Evaluate the multi-mode model from a flat parameter vector.

    Four parameters per mode, then the constant.
    """
    out = np.full(time.shape, float(theta[4 * n_modes]))
    for k in range(n_modes):
        a, d, w, p = theta[4 * k : 4 * k + 4]
        out = out + a * np.exp(-d * time) * np.cos(w * time + p)
    return out


def modes_to_theta(
    modes: Sequence[DampedMode], constant: float
) -> np.ndarray:
    return np.array(
        [
            v
            for m in modes
            for v in (m.amplitude, m.decay, m.angular_frequency, m.phase)
        ]
        + [constant],
        dtype=np.float64,
    )


def synthetic_modes_trace(
    samples: int = 96,
    *,
    modes: Sequence[DampedMode] = DEFAULT_MODES,
    constant: float = 0.15,
    duration: float = 9.5,
    noise_sigma: float = 0.0,
    seed: int | None = None,
) -> SyntheticModesTrace:
    """Build a trace with known modes plus optional white Gaussian noise.

    With the defaults and ``noise_sigma=0`` this reproduces
    :func:`synthetic_trace` exactly.
    """
    if samples < 16:
        raise ValueError("samples must be at least 16")
    time = np.linspace(0.0, duration, samples, dtype=np.float64)
    clean = modes_model(modes_to_theta(modes, constant), time, len(modes))
    trace = clean
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        trace = clean + rng.normal(0.0, noise_sigma, size=time.shape)
    return SyntheticModesTrace(
        time, trace, clean, tuple(modes), constant, float(noise_sigma), seed
    )


DISTORTIONS = (
    "chirp",
    "gaussian_envelope",
    "baseline_drift",
    "clip",
    "glitch",
)


def distort_trace(
    time: np.ndarray,
    modes: Sequence[DampedMode],
    constant: float,
    kind: str,
    amount: float,
) -> np.ndarray:
    """Return a clean trace that departs from the damped-mode model.

    ``chirp``: every frequency rises by the fraction ``amount`` across the
    record. ``gaussian_envelope``: the first mode decays as a Gaussian with
    1/e time ``amount`` instead of an exponential. ``baseline_drift``: a
    linear ramp of ``amount`` across the record. ``clip``: the trace is
    clipped above ``amount``. ``glitch``: one sample at mid-record is
    offset by ``amount``. These are outside the model class on purpose:
    the sweep reports how the fit responds and the residual-to-noise ratio
    that can help diagnose them.
    """
    if kind not in DISTORTIONS:
        raise ValueError(f"unknown distortion {kind!r}")
    span = float(time[-1] - time[0])
    out = np.full(time.shape, float(constant))
    for k, m in enumerate(modes):
        phase = m.angular_frequency * time + m.phase
        if kind == "chirp":
            phase = phase + (
                0.5
                * m.angular_frequency
                * amount
                * (time - time[0]) ** 2
                / span
            )
        if kind == "gaussian_envelope" and k == 0:
            env = np.exp(-((time / amount) ** 2))
        else:
            env = np.exp(-m.decay * time)
        out = out + m.amplitude * env * np.cos(phase)
    if kind == "baseline_drift":
        out = out + amount * (time - time[0]) / span
    elif kind == "clip":
        out = np.minimum(out, amount)
    elif kind == "glitch":
        out = out.copy()
        out[time.size // 2] += amount
    return out


def cramer_rao_bounds(
    modes: Sequence[DampedMode],
    constant: float,
    time: np.ndarray,
    noise_sigma: float,
    step: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Cramer-Rao lower bound (standard deviation) per parameter for white
    Gaussian noise of ``noise_sigma``.

    The Fisher information is J^T J / sigma^2 with J the numerical Jacobian
    of :func:`modes_model`; the bound is the square root of the diagonal of
    its inverse. Returned per mode as arrays ``amplitude``, ``decay``,
    ``angular_frequency``, ``phase`` plus the scalar ``constant``.
    """
    if noise_sigma <= 0:
        raise ValueError("noise_sigma must be positive")
    theta = modes_to_theta(modes, constant)
    n = len(modes)
    jac = np.empty((time.size, theta.size))
    for k in range(theta.size):
        tp = theta.copy()
        tm = theta.copy()
        tp[k] += step
        tm[k] -= step
        jac[:, k] = (modes_model(tp, time, n) - modes_model(tm, time, n)) / (
            2 * step
        )
    fisher = jac.T @ jac / noise_sigma**2
    sd = np.sqrt(np.diag(np.linalg.inv(fisher)))
    return {
        "amplitude": sd[0 : 4 * n : 4],
        "decay": sd[1 : 4 * n : 4],
        "angular_frequency": sd[2 : 4 * n : 4],
        "phase": sd[3 : 4 * n : 4],
        "constant": sd[4 * n],
    }


def _to_host(values) -> np.ndarray:
    """Return a float NumPy array; CuPy arrays are copied to the host."""
    get = getattr(values, "get", None)
    if get is not None and not isinstance(values, np.ndarray):
        values = get()
    return np.asarray(values, dtype=float)


def match_modes(
    modes: Sequence[DampedMode],
    angular_frequency: np.ndarray,
    decay: np.ndarray,
    *,
    tolerance: float,
) -> list[int | None]:
    """For each true mode, the index of the fitted mode within ``tolerance``
    (rad per time unit) or ``None``.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    out: list[int | None] = [None] * len(modes)
    if not modes or angular_frequency.size == 0:
        return out
    err = np.abs(
        np.array([m.angular_frequency for m in modes])[:, None]
        - angular_frequency[None, :]
    )
    # A missing match costs more than every valid distance combined:
    # maximize recovered modes first, then minimize total frequency error.
    missing_cost = len(modes) + 1
    cost = np.full(
        (len(modes), angular_frequency.size + len(modes)), float(missing_cost)
    )
    cost[:, : angular_frequency.size] = np.where(
        err <= tolerance, err / (tolerance or 1.0), np.inf
    )
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        if column < angular_frequency.size:
            out[int(row)] = int(column)
    return out


@dataclass(frozen=True)
class ParameterStats:
    """Statistics of one estimated parameter over the trials in which the
    mode was recovered, against the unconditional Cramer-Rao bound.

    ``crlb_std`` is the standard-deviation bound and ``crlb_variance`` its
    square; ``std_over_crlb_std`` compares the sample standard deviation
    with the former.
    """

    truth: float
    bias: float
    std: float
    rmse: float
    crlb_std: float
    crlb_variance: float
    std_over_crlb_std: float


@dataclass(frozen=True)
class ModeValidation:
    """One true mode: recovery count, loss rate and per-parameter
    statistics computed over the recovered trials only."""

    angular_frequency: ParameterStats
    decay: ParameterStats
    amplitude: ParameterStats
    phase: ParameterStats
    recovered_trials: int
    loss_rate: float

    @property
    def frequency_ratio(self) -> float:
        return self.angular_frequency.std_over_crlb_std

    @property
    def decay_ratio(self) -> float:
        return self.decay.std_over_crlb_std


@dataclass(frozen=True)
class ValidationLevel:
    """One sweep condition. A trial is successful when the estimator
    returned and every true mode was matched; ``estimator_errors`` counts
    trials in which the estimator raised, which are also failed trials."""

    snr_db: float
    noise_sigma: float
    trials_attempted: int
    trials_successful: int
    trials_failed: int
    estimator_errors: int
    n_components: int
    modes: tuple[ModeValidation, ...]
    any_mode_lost_rate: float
    residual_ratio: float


@dataclass(frozen=True)
class ValidationSweep:
    samples: int
    duration: float
    signal_rms: float
    levels: tuple[ValidationLevel, ...]
    match_tolerance: float
    backend: str
    modes: tuple[DampedMode, ...] = DEFAULT_MODES
    constant: float = 0.15
    seed: int = 0
    distortion: tuple[str, float] | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _wrap_phase(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _canonical(amplitude: np.ndarray, phase: np.ndarray):
    """A negative amplitude is the same mode with the phase shifted by pi."""
    amplitude = np.array(amplitude, dtype=float, copy=True)
    phase = np.array(phase, dtype=float, copy=True)
    neg = amplitude < 0
    amplitude[neg] = -amplitude[neg]
    phase[neg] = phase[neg] + np.pi
    return amplitude, _wrap_phase(phase)


def _stats(
    samples: np.ndarray, truth: float, crlb_std: float, *, wrap: bool
) -> ParameterStats:
    err = samples - truth
    if wrap:
        err = _wrap_phase(err)
    n = err.size
    bias = float(err.mean()) if n else float("nan")
    std = float(err.std(ddof=1)) if n > 1 else float("nan")
    rmse = float(np.sqrt(np.mean(err**2))) if n else float("nan")
    return ParameterStats(
        truth=float(truth),
        bias=bias,
        std=std,
        rmse=rmse,
        crlb_std=float(crlb_std),
        crlb_variance=float(crlb_std) ** 2,
        std_over_crlb_std=std / float(crlb_std),
    )


def validation_sweep(
    *,
    samples: int = 96,
    modes: Sequence[DampedMode] = DEFAULT_MODES,
    constant: float = 0.15,
    duration: float = 9.5,
    snr_db: Sequence[float] = (40.0, 30.0, 20.0, 10.0),
    trials: int = 200,
    n_components: int = 6,
    match_tolerance: float = 0.3,
    seed: int = 20260914,
    estimator: Callable | None = None,
    backend: str = "cpu",
    distortion: tuple[str, float] | None = None,
) -> ValidationSweep:
    """Monte Carlo sweep of the linear-prediction estimator against the
    Cramer-Rao bound.

    ``snr_db`` is defined against the rms of the mean-removed clean trace.
    For each level the per-mode scatter, bias and rmse of every parameter
    are compared with the bound, and ``loss_rate`` is the fraction of
    trials in which the true mode had no fitted mode within
    ``match_tolerance`` (radians per time unit) of its angular frequency;
    each fitted mode can match one true mode.

    The statistics are computed over the recovered trials only and the
    bound is unconditional. The root filter keeps decaying roots, so the
    surviving decay estimates are truncated at zero; for a lightly damped
    mode the reported decay statistics are therefore conditional on
    survival and can move non-monotonically with noise. Read them together
    with ``loss_rate``.

    ``estimator`` may return CuPy arrays; they are moved to the host. A
    trial in which the estimator raises is counted in ``estimator_errors``
    and as a loss of every mode.

    ``distortion`` = (kind, amount) replaces the clean trace with one from
    :func:`distort_trace`, outside the model class; ``residual_ratio`` is
    then the median rms residual of the reconstruction divided by the
    noise sigma. Large values can reflect model mismatch, missed modes or
    estimator error; this ratio alone does not distinguish those causes.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not snr_db:
        raise ValueError("snr_db must contain at least one level")
    est = estimator or (lambda t, y, k: linear_prediction_numpy(t, y, k))
    base = synthetic_modes_trace(
        samples, modes=modes, constant=constant, duration=duration
    )
    clean = base.clean
    if distortion is not None:
        clean = distort_trace(
            base.time, modes, constant, distortion[0], distortion[1]
        )
    signal_rms = float(np.std(clean - clean.mean()))
    rng = np.random.default_rng(seed)
    levels = []
    for snr in snr_db:
        sigma = signal_rms / 10 ** (snr / 20)
        bounds = cramer_rao_bounds(modes, constant, base.time, sigma)
        found: list[dict[str, list[float]]] = [
            {name: [] for name in PARAMETERS} for _ in modes
        ]
        lost = np.zeros(len(modes), dtype=int)
        any_lost = 0
        errors = 0
        ratios = []
        for _ in range(trials):
            y = clean + rng.normal(0.0, sigma, size=base.time.shape)
            try:
                r = est(base.time, y, n_components)
            except Exception:
                errors += 1
                any_lost += 1
                lost += 1
                continue
            w = _to_host(r.angular_frequency)
            d = _to_host(r.decay)
            a, p = _canonical(_to_host(r.amplitude), _to_host(r.phase))
            rec = _to_host(r.reconstruction)
            ratios.append(float(np.sqrt(np.mean((y - rec) ** 2)) / sigma))
            hit = match_modes(modes, w, d, tolerance=match_tolerance)
            if any(h is None for h in hit):
                any_lost += 1
            for k, h in enumerate(hit):
                if h is None:
                    lost[k] += 1
                    continue
                found[k]["amplitude"].append(a[h])
                found[k]["decay"].append(d[h])
                found[k]["angular_frequency"].append(w[h])
                found[k]["phase"].append(p[h])
        per_mode = []
        for k, m in enumerate(modes):
            stats = {
                name: _stats(
                    np.asarray(found[k][name]),
                    getattr(m, name),
                    bounds[name][k],
                    wrap=name == "phase",
                )
                for name in PARAMETERS
            }
            per_mode.append(
                ModeValidation(
                    angular_frequency=stats["angular_frequency"],
                    decay=stats["decay"],
                    amplitude=stats["amplitude"],
                    phase=stats["phase"],
                    recovered_trials=int(trials - lost[k]),
                    loss_rate=float(lost[k] / trials),
                )
            )
        levels.append(
            ValidationLevel(
                snr_db=float(snr),
                noise_sigma=float(sigma),
                trials_attempted=trials,
                trials_successful=trials - any_lost,
                trials_failed=any_lost,
                estimator_errors=errors,
                n_components=n_components,
                modes=tuple(per_mode),
                any_mode_lost_rate=float(any_lost / trials),
                residual_ratio=float(np.median(ratios))
                if ratios
                else float("nan"),
            )
        )
    return ValidationSweep(
        samples=samples,
        duration=duration,
        signal_rms=signal_rms,
        levels=tuple(levels),
        match_tolerance=match_tolerance,
        backend=backend,
        modes=tuple(modes),
        constant=constant,
        seed=seed,
        distortion=distortion,
    )


def source_revision() -> str | None:
    """Git revision of the checkout containing this module, if any."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def build_summary(
    sweeps: Sequence[ValidationSweep],
    *,
    command: str = "linear-prediction-validate",
    artifacts: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Assemble the ``schema_version`` 1 summary of one or more sweeps.

    ``config`` records the truth, the sampling, the noise definition, the
    seed, the trial count, the model order and the mode-matching rule;
    ``runtime`` the package, Python, NumPy and SciPy versions, the source
    revision and the backend; ``results`` one entry per sweep condition
    (estimator by distortion by signal-to-noise level); ``artifacts`` the
    relative filenames written next to the summary.
    """
    if not sweeps:
        raise ValueError("at least one sweep is required")
    first = sweeps[0]
    shared_fields = (
        "samples",
        "duration",
        "modes",
        "constant",
        "seed",
        "match_tolerance",
    )
    first_levels = [
        (level.snr_db, level.trials_attempted, level.n_components)
        for level in first.levels
    ]
    for sweep in sweeps[1:]:
        if (
            any(
                getattr(sweep, name) != getattr(first, name)
                for name in shared_fields
            )
            or [
                (level.snr_db, level.trials_attempted, level.n_components)
                for level in sweep.levels
            ]
            != first_levels
        ):
            raise ValueError(
                "summary sweeps must share truth, sampling, seed, "
                "matching and trial settings"
            )
    dt = first.duration / (first.samples - 1)
    config = {
        "model": (
            "trace(t) = constant + sum_k amplitude_k exp(-decay_k t) "
            "cos(angular_frequency_k t + phase_k)"
        ),
        "units": {
            "time": "input time unit",
            "amplitude": "trace units",
            "decay": "1 / time unit",
            "angular_frequency": "rad / time unit",
            "phase": "rad",
        },
        "truth": {
            "modes": [asdict(m) for m in first.modes],
            "constant": first.constant,
        },
        "sampling": {
            "samples": first.samples,
            "duration": first.duration,
            "dt": dt,
            "start": 0.0,
        },
        "noise": {
            "kind": "white Gaussian, independent per sample",
            "snr_db_definition": (
                "20 log10(rms of the mean-removed clean trace / sigma)"
            ),
            "signal_rms": first.signal_rms,
            "snr_db": [lvl.snr_db for lvl in first.levels],
        },
        "seed": first.seed,
        "trials_per_level": first.levels[0].trials_attempted,
        "n_components": first.levels[0].n_components,
        "mode_matching": {
            "criterion": (
                "a true mode is recovered when a fitted mode lies within "
                "the tolerance of its angular frequency; fitted modes are "
                "assigned to maximize matches, then minimize total frequency "
                "error, and each is used at most once"
            ),
            "tolerance_rad_per_time": first.match_tolerance,
        },
        "statistics": (
            "bias, std (ddof 1) and rmse are computed over the recovered "
            "trials of each mode; crlb_std and crlb_variance are the "
            "unconditional Cramer-Rao bounds from the exact Fisher "
            "information of the model at the true parameters; the phase "
            "error is wrapped to [-pi, pi)"
        ),
        "estimators": sorted({s.backend for s in sweeps}),
        "distortions": sorted(
            {
                f"{s.distortion[0]}:{s.distortion[1]:g}"
                if s.distortion
                else "none"
                for s in sweeps
            }
        ),
    }
    runtime = runtime_metadata(backend="cpu", dtype="float64")
    try:
        from scipy import __version__ as scipy_version
    except ImportError:  # pragma: no cover - scipy is a dependency
        scipy_version = None
    runtime["scipy_version"] = scipy_version
    runtime["source_revision"] = source_revision()
    results = []
    for sweep in sweeps:
        for lvl in sweep.levels:
            results.append(
                {
                    "estimator": sweep.backend,
                    "distortion": (
                        {
                            "kind": sweep.distortion[0],
                            "amount": sweep.distortion[1],
                        }
                        if sweep.distortion
                        else None
                    ),
                    "snr_db": lvl.snr_db,
                    "signal_rms": sweep.signal_rms,
                    "noise_sigma": lvl.noise_sigma,
                    "trials": {
                        "attempted": lvl.trials_attempted,
                        "successful": lvl.trials_successful,
                        "failed": lvl.trials_failed,
                        "estimator_errors": lvl.estimator_errors,
                    },
                    "any_mode_lost_rate": lvl.any_mode_lost_rate,
                    "residual_rms_over_sigma_median": (
                        lvl.residual_ratio
                        if np.isfinite(lvl.residual_ratio)
                        else None
                    ),
                    "modes": [
                        {
                            "recovered_trials": m.recovered_trials,
                            "loss_rate": m.loss_rate,
                            "statistics_over": "recovered_trials",
                            **{
                                name: {
                                    key: value if np.isfinite(value) else None
                                    for key, value in asdict(
                                        getattr(m, name)
                                    ).items()
                                }
                                for name in PARAMETERS
                            },
                        }
                        for m in lvl.modes
                    ],
                }
            )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "command": command,
        "config": config,
        "runtime": runtime,
        "results": results,
        "artifacts": dict(artifacts or {}),
    }


def write_validation_figure(
    sweeps: Sequence[ValidationSweep], path: Path, *, title: str
) -> bool:
    """Standalone Bokeh HTML: per estimator and parameter the sample
    standard deviation against the bound versus signal-to-noise, and the
    loss rate. Returns False when the ``viz`` extra is not installed."""
    try:
        from bokeh.layouts import gridplot
        from bokeh.plotting import figure, output_file, save
    except ImportError:
        return False
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    rows = []
    for sweep in sweeps:
        snr = [lvl.snr_db for lvl in sweep.levels]
        label = sweep.backend + (
            f" {sweep.distortion[0]}:{sweep.distortion[1]:g}"
            if sweep.distortion
            else ""
        )
        row = []
        for name in ("angular_frequency", "decay"):
            fig = figure(
                width=360,
                height=260,
                y_axis_type="log",
                title=f"{label}: {name} std (points) vs bound (dashed)",
                x_axis_label="snr (dB)",
                y_axis_label="std",
            )
            for k, mode in enumerate(sweep.modes):
                std = [
                    getattr(lvl.modes[k], name).std for lvl in sweep.levels
                ]
                crlb = [
                    getattr(lvl.modes[k], name).crlb_std
                    for lvl in sweep.levels
                ]
                color = palette[k % len(palette)]
                fig.line(snr, crlb, color=color, line_dash="dashed")
                fig.scatter(
                    snr,
                    std,
                    color=color,
                    size=7,
                    legend_label=f"mode w={mode.angular_frequency:g}",
                )
            fig.legend.label_text_font_size = "8pt"
            row.append(fig)
        fig = figure(
            width=360,
            height=260,
            title=f"{label}: loss rate",
            x_axis_label="snr (dB)",
            y_axis_label="fraction of trials",
        )
        for k, mode in enumerate(sweep.modes):
            fig.line(
                snr,
                [lvl.modes[k].loss_rate for lvl in sweep.levels],
                color=palette[k % len(palette)],
                legend_label=f"mode w={mode.angular_frequency:g}",
            )
        fig.line(
            snr,
            [lvl.any_mode_lost_rate for lvl in sweep.levels],
            color="black",
            line_dash="dotted",
            legend_label="any mode",
        )
        fig.legend.label_text_font_size = "8pt"
        row.append(fig)
        rows.append(row)
    output_file(str(path), title=title)
    save(gridplot(rows))
    return True


def write_validation_run(
    output_dir: Path | str,
    sweeps: Sequence[ValidationSweep],
    *,
    command: str = "linear-prediction-validate",
    figure_name: str = "validation.html",
) -> dict[str, Any]:
    """Write ``summary.json`` and, when Bokeh is available, the figure
    into ``output_dir`` (created if needed). Returns the summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    have_figure = write_validation_figure(
        sweeps, out / figure_name, title="xray linear prediction validation"
    )
    artifacts = {
        "summary": "summary.json",
        "figure": figure_name if have_figure else None,
    }
    summary = build_summary(sweeps, command=command, artifacts=artifacts)
    with open(out / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return summary
