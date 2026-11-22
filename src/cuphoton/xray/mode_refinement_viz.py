# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visual comparison of linear prediction and experimental mode refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .linear_prediction import linear_prediction_numpy
from .mode_refinement import linear_prediction_refined
from .synthetic_validation import distort_trace, synthetic_modes_trace


@dataclass(frozen=True)
class ModesReviewReport:
    """Location and fit status of a standalone synthetic-mode review."""

    html_path: Path
    trace_count: int
    converged_count: int

    def __str__(self) -> str:
        return (
            f"modes_review={self.html_path}\n"
            f"traces={self.trace_count}\n"
            f"refinement_converged={self.converged_count}/{self.trace_count}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "html_path": str(self.html_path),
            "trace_count": self.trace_count,
            "converged_count": self.converged_count,
        }


def _mode_curve(amplitude, decay, frequency, phase, time):
    return (
        amplitude * np.exp(-decay * time) * np.cos(frequency * time + phase)
    )


def write_modes_review(
    output: Path | str,
    *,
    snr_db: Sequence[float] = (40.0, 20.0, 10.0, 5.0),
    distortion: tuple[str, float] | None = None,
    seed: int = 11,
    samples: int = 96,
) -> ModesReviewReport:
    """Write an offline HTML comparison at each requested noise level.

    SNR uses the standard deviation of the noise-free signal. Distortions
    change that signal; the mode panel still shows the undistorted reference
    modes. Refinement convergence is an optimizer status, not scientific
    acceptance. Requires the ``viz`` extra.
    """
    snr_levels = tuple(float(level) for level in snr_db)
    if not snr_levels:
        raise ValueError("snr_db must contain at least one level")
    if not np.all(np.isfinite(snr_levels)):
        raise ValueError("snr_db levels must be finite")
    fixture = synthetic_modes_trace(samples)
    time = fixture.time
    clean = fixture.clean
    if distortion is not None:
        clean = distort_trace(
            time, fixture.modes, fixture.constant, *distortion
        )
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        signal_rms = np.std(clean - clean.mean())
        sigmas = signal_rms / np.power(10.0, np.asarray(snr_levels) / 20)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("snr_db must produce finite, positive noise levels")

    try:
        from bokeh.embed import file_html
        from bokeh.layouts import gridplot
        from bokeh.plotting import figure
        from bokeh.resources import INLINE
    except ImportError as exc:  # pragma: no cover - depends on the viz extra
        raise RuntimeError(
            "Bokeh is required for modes review; install 'cuphoton[viz]' "
            "or run 'uv sync --extra viz' for development"
        ) from exc

    fine = np.linspace(time[0], time[-1], 2000)
    rng = np.random.default_rng(seed)
    rows = []
    converged_count = 0
    for snr, sigma in zip(snr_levels, sigmas, strict=True):
        trace = clean + rng.normal(0.0, sigma, size=time.shape)
        lp = linear_prediction_numpy(time, trace, 6)
        refined = linear_prediction_refined(
            time, trace, 6, len(fixture.modes)
        )
        converged_count += int(refined.converged)
        lp_ratio = np.sqrt(np.mean((trace - lp.reconstruction) ** 2)) / sigma
        status = "converged" if refined.converged else "not converged"
        left = figure(
            width=760,
            height=280,
            title=(
                f"{snr:g} dB, sigma {sigma:.4g}: residual/noise "
                f"lp {lp_ratio:.1f}, refined "
                f"{refined.residual_rms / sigma:.1f}; {status}"
            ),
            x_axis_label="time",
            y_axis_label="trace",
        )
        left.scatter(time, trace, size=4, color="black", legend_label="trace")
        # Plot the generated samples: a glitch is defined on that sample grid.
        left.line(time, clean, color="gray", legend_label="noise-free signal")
        left.line(
            time,
            np.asarray(lp.reconstruction),
            color="red",
            legend_label="linear prediction",
        )
        left.line(
            time,
            refined.reconstruction,
            color="blue",
            line_dash="dashed",
            legend_label="refined",
        )
        left.legend.location = "top_right"
        left.legend.label_text_font_size = "8pt"
        right = figure(
            width=520,
            height=280,
            title=(
                "modes: reference (solid), LP (dashed), refined (dotted)"
                + (
                    f"; signal distortion {distortion[0]}"
                    if distortion
                    else ""
                )
            ),
            x_axis_label="time",
        )
        colors = ("#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd")
        for index, mode in enumerate(fixture.modes):
            right.line(
                fine,
                _mode_curve(
                    mode.amplitude,
                    mode.decay,
                    mode.angular_frequency,
                    mode.phase,
                    fine,
                ),
                color=colors[index % len(colors)],
                alpha=0.5,
                legend_label=f"reference w={mode.angular_frequency:g}",
            )
        for result, label, color, dash in (
            (lp, "lp", "red", "dashed"),
            (refined, "refined", "blue", "dotted"),
        ):
            for index, frequency in enumerate(result.angular_frequency):
                if frequency == 0:
                    continue
                decay = float(result.decay[index])
                right.line(
                    fine,
                    _mode_curve(
                        float(result.amplitude[index]),
                        decay,
                        float(frequency),
                        float(result.phase[index]),
                        fine,
                    ),
                    color=color,
                    line_dash=dash,
                    legend_label=f"{label} w={frequency:.3f} d={decay:+.3f}",
                )
        right.legend.label_text_font_size = "7pt"
        rows.append([left, right])
    title = "XRay linear prediction and mode refinement"
    if distortion is not None:
        title += f" ({distortion[0]}:{distortion[1]:g})"
    output = Path(output).expanduser()
    html = file_html(gridplot(rows), INLINE, title)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return ModesReviewReport(output, len(snr_levels), converged_count)
