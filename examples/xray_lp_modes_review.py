# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Review view for linear prediction on the synthetic damped-mode fixture.

Writes one standalone Bokeh HTML file with, for a few noise levels and one
optional model mismatch, the trace, the true model, the linear-prediction
reconstruction, the refined reconstruction, and the recovered modes
against the true ones. No external data. Requires the ``viz`` extra.

    uv run python examples/xray_lp_modes_review.py --output review.html
    uv run python examples/xray_lp_modes_review.py --distortion chirp:0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cuphoton.xray.linear_prediction import linear_prediction_numpy
from cuphoton.xray.mode_refinement import linear_prediction_refined
from cuphoton.xray.synthetic_validation import (
    distort_trace,
    modes_model,
    modes_to_theta,
    synthetic_modes_trace,
)


def _mode_curve(a, d, w, p, t):
    return a * np.exp(-d * t) * np.cos(w * t + p)


def build(output: Path, snr_db, distortion, seed: int, samples: int):
    try:
        from bokeh.layouts import gridplot
        from bokeh.plotting import figure, output_file, save
    except ImportError:  # pragma: no cover - depends on the viz extra
        sys.exit("bokeh is required: uv sync --extra viz")

    fx = synthetic_modes_trace(samples)
    t = fx.time
    clean = fx.clean
    if distortion is not None:
        clean = distort_trace(t, fx.modes, fx.constant, *distortion)
    fine = np.linspace(t[0], t[-1], 2000)
    true_fine = modes_model(modes_to_theta(fx.modes, fx.constant), fine, 2)
    sig = float(np.std(clean - clean.mean()))
    rng = np.random.default_rng(seed)
    rows = []
    for snr in snr_db:
        sigma = sig / 10 ** (snr / 20)
        y = clean + rng.normal(0.0, sigma, size=t.shape)
        lp = linear_prediction_numpy(t, y, 6)
        rf = linear_prediction_refined(t, y, 6, len(fx.modes))
        lp_ratio = np.sqrt(np.mean((y - lp.reconstruction) ** 2)) / sigma
        left = figure(
            width=760,
            height=280,
            title=(
                f"{snr:g} dB, sigma {sigma:.4g}: residual/noise "
                f"lp {lp_ratio:.1f}, refined {rf.residual_rms / sigma:.1f}"
            ),
            x_axis_label="time",
            y_axis_label="trace",
        )
        left.scatter(t, y, size=4, color="black", legend_label="trace")
        left.line(fine, true_fine, color="gray", legend_label="true model")
        left.line(
            t,
            np.asarray(lp.reconstruction),
            color="red",
            legend_label="linear prediction",
        )
        left.line(
            t,
            rf.reconstruction,
            color="blue",
            line_dash="dashed",
            legend_label="refined",
        )
        left.legend.location = "top_right"
        left.legend.label_text_font_size = "8pt"
        w = np.asarray(lp.angular_frequency)
        lost = [
            m.angular_frequency
            for m in fx.modes
            if w.size == 0 or np.min(np.abs(w - m.angular_frequency)) > 0.3
        ]
        right = figure(
            width=520,
            height=280,
            title="modes: true (solid), linear prediction (dashed), "
            "refined (dotted)" + (f"; lp lost w={lost[0]:g}" if lost else ""),
            x_axis_label="time",
        )
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        for k, m in enumerate(fx.modes):
            right.line(
                fine,
                _mode_curve(
                    m.amplitude, m.decay, m.angular_frequency, m.phase, fine
                ),
                color=colors[k % 4],
                alpha=0.5,
                legend_label=f"true w={m.angular_frequency:g}",
            )
        for k in range(w.size):
            if w[k] == 0:
                continue
            right.line(
                fine,
                _mode_curve(
                    float(lp.amplitude[k]),
                    float(lp.decay[k]),
                    float(w[k]),
                    float(lp.phase[k]),
                    fine,
                ),
                color="red",
                line_dash="dashed",
                legend_label=f"lp w={w[k]:.3f} d={lp.decay[k]:+.3f}",
            )
        for k in range(rf.angular_frequency.size):
            right.line(
                fine,
                _mode_curve(
                    float(rf.amplitude[k]),
                    float(rf.decay[k]),
                    float(rf.angular_frequency[k]),
                    float(rf.phase[k]),
                    fine,
                ),
                color="blue",
                line_dash="dotted",
                legend_label=(
                    f"refined w={rf.angular_frequency[k]:.3f} "
                    f"d={rf.decay[k]:+.3f}"
                ),
            )
        right.legend.label_text_font_size = "7pt"
        rows.append([left, right])
    output_file(str(output), title="xray linear prediction on the fixture")
    save(gridplot(rows))
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="xray_lp_modes_review.html")
    parser.add_argument("--snr-db", default="40,20,10,5")
    parser.add_argument(
        "--distortion",
        default="",
        help="kind:amount, e.g. chirp:0.05 (see distort_trace)",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args(argv)
    snr = tuple(float(x) for x in args.snr_db.split(",") if x.strip())
    distortion = None
    if args.distortion:
        kind, _, amount = args.distortion.partition(":")
        distortion = (kind.strip(), float(amount))
    out = build(Path(args.output), snr, distortion, args.seed, args.samples)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
