# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Saved-result views and accessible explanatory HTML for Model Scene."""

import html
import json
from pathlib import Path

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Select
from bokeh.plotting import figure
from experiment import forward_prediction
from scene import science_window
from views import linked_panels


def products(root, names):
    cases = {}
    for label, directory in names.items():
        with np.load(Path(root) / directory / "page.npz") as saved:
            cases[label] = [
                saved[key].copy() for key in ("truth", "coadd", "l2", "huber")
            ]
    return linked_panels(cases, ["Target truth", "Coadd", "L2", "Huber"])


def exposures(root, case="trial-000", method="huber"):
    with np.load(Path(root) / case / "page.npz") as saved:
        predicted = forward_prediction(
            saved[f"{method}_native"], saved["kernels"]
        )
        predicted = science_window(predicted)
        labels = [
            str(i + 1).zfill(2)
            + (" (fit)" if i in saved["train_frames"] else " (held out)")
            for i in range(12)
        ]
        cases = {
            labels[i]: [
                saved["observations"][..., i].copy(),
                predicted[..., i],
                saved["observations"][..., i] - predicted[..., i],
            ]
            for i in range(12)
        }
    return linked_panels(
        cases, ["Observation", "Prediction", "Residual"], signed=(2,)
    )


def diagnostics(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    colors = {"coadd": "#3377bb", "l2": "#cc5533", "huber": "#228855"}
    flux = figure(
        title="Aperture flux bias with ±1.96 standard errors",
        x_axis_label="Fixed catalog source index",
        y_axis_label="Fractional aperture flux error",
        width=760,
        height=330,
        tools="pan,wheel_zoom,reset,save",
    )
    centroid = figure(
        title="Centroid error by source",
        x_axis_label="Fixed catalog source index",
        y_axis_label="Mean error (pixel)",
        width=760,
        height=280,
    )
    trials = json.loads((root / "trials.json").read_text())
    for method, color in colors.items():
        rows = summary[method]["by_source"]
        x = np.arange(len(rows))
        y = np.array([r["mean_fractional_bias"] for r in rows])
        se = np.array([r["bias_standard_error"] for r in rows])
        flux.scatter(x, y, color=color, legend_label=method, size=8)
        flux.segment(x, y - 1.96 * se, x, y + 1.96 * se, color=color)
        values = np.array(
            [
                [s["centroid_error_px"] for s in trial[method]["sources"]]
                for trial in trials
            ]
        )
        center = values.mean(axis=0)
        se_centroid = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        centroid.line(x, center, color=color, legend_label=method)
        centroid.segment(
            x,
            center - 1.96 * se_centroid,
            x,
            center + 1.96 * se_centroid,
            color=color,
        )
    convergence = figure(
        title="Fixed-objective loss traces (first trial)",
        x_axis_label="Iteration",
        y_axis_label="Native loss",
        width=760,
        height=280,
        y_axis_type="log",
    )
    with np.load(root / "trial-000/page.npz") as data:
        for method in ("l2", "huber"):
            loss = data[f"{method}_loss"]
            convergence.line(
                np.arange(len(loss)),
                loss,
                color=colors[method],
                legend_label=method,
            )
    profile = figure(
        title="Annular profile error with ±1.96 standard errors",
        x_axis_label="Radius (pixel)",
        y_axis_label="Mean intensity error (electron / exposure)",
        width=760,
        height=300,
    )
    profile_bank = []
    for index in range(len(trials[0]["coadd"]["sources"])):
        data = {"radius": np.arange(5) + 0.5}
        for method in colors:
            values = np.array(
                [
                    trial[method]["sources"][index]["profile_error"]
                    for trial in trials
                ]
            )
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            data[method] = mean
            data[method + "_low"] = mean - 1.96 * se
            data[method + "_high"] = mean + 1.96 * se
        profile_bank.append(ColumnDataSource(data))
    source = ColumnDataSource(dict(profile_bank[0].data))
    for method, color in colors.items():
        profile.line(
            "radius", method, source=source, color=color, legend_label=method
        )
        profile.segment(
            "radius",
            method + "_low",
            "radius",
            method + "_high",
            source=source,
            color=color,
        )
    selector = Select(
        title="Precomputed source profile",
        value="0",
        options=[str(i) for i in range(len(profile_bank))],
    )
    selector.js_on_change(
        "value",
        CustomJS(
            args=dict(source=source, bank=profile_bank),
            code="""
source.data = {...bank[Number(cb_obj.value)].data};
source.change.emit();
""",
        ),
    )
    return column(flux, centroid, selector, profile, convergence)


def calibration_diagnostics(root):
    root = Path(root) / "calibration"
    diagnostics = json.loads((root / "diagnostics.json").read_text())
    plot = figure(
        title="Saved kernel normalization after factorization",
        x_axis_label="Exposure",
        y_axis_label="Kernel sum",
        width=700,
        height=280,
    )
    for method, color in [("mm", "#228855"), ("adam", "#cc5533")]:
        plot.line(
            np.arange(1, 13),
            diagnostics[method]["sums"],
            color=color,
            legend_label=method,
        )
    trials = json.loads((root / "finite-calibration.json").read_text())
    errors = np.array(
        [
            [
                s["calibration_flux_error_increment"]
                for s in trial["l2"]["sources"]
            ]
            for trial in trials
        ]
    )
    uncertainty = figure(
        title="Calibration draws minus noiseless control: L2 ±1 SD",
        x_axis_label="Source",
        y_axis_label="Fractional aperture error",
        width=700,
        height=280,
    )
    x = np.arange(errors.shape[1])
    mean = errors.mean(axis=0)
    scatter = errors.std(axis=0, ddof=1)
    uncertainty.scatter(x, mean, size=8)
    uncertainty.segment(x, mean - scatter, x, mean + scatter)
    return row(plot, uncertainty, sizing_mode="stretch_width")


def case_measurement_table(root):
    """Show quantitative controls from the first noise realization."""
    rows = []
    for case in (
        "trial-000",
        "outliers",
        "explicit-mask",
        "psf-width-0.95",
        "psf-width-1.05",
        "variance-times-four",
        "photon-read",
        "transient",
        "matched-operator",
        "noiseless-physical",
        "calibrated-mm",
        "perfect-factorization",
    ):
        values = json.loads(
            (Path(root) / case / "measurements.json").read_text()
        )
        for method, result in values.items():
            bias = np.mean(
                [s["fractional_flux_error"] for s in result["sources"]]
            )
            centroid = np.mean(
                [s["centroid_error_px"] for s in result["sources"]]
            )
            rows.append(
                f"<tr><td>{html.escape(case)}</td>"
                f"<td>{html.escape(method)}</td><td>{bias:.5f}</td>"
                f"<td>{centroid:.5f}</td></tr>"
            )
    floors = []
    for case in (
        "matched-operator",
        "noiseless-physical",
        "perfect-factorization",
    ):
        values = json.loads(
            (Path(root) / case / "measurements.json").read_text()
        )
        for method, result in values.items():
            for source in result["sources"]:
                floors.append(
                    f"<tr><td>{case}</td><td>{method}</td>"
                    f"<td>{source['kind']}</td>"
                    f"<td>{source['input_flux']}</td>"
                    f"<td>{source['fractional_flux_error']:.6f}</td>"
                    "</tr>"
                )
    floor_table = (
        "<details><summary>Per-source numerical bias floors</summary>"
        "<table><thead><tr><th>Control</th><th>Method</th>"
        "<th>Environment</th><th>Input flux</th>"
        "<th>Fractional aperture error</th></tr></thead><tbody>"
        + "".join(floors)
        + "</tbody></table></details>"
    )
    return (
        "<table><caption>Case controls; "
        "photon/read and compact context use separate draws. "
        "No trial error bars.</caption>"
        "<thead><tr><th>Case</th><th>Method</th>"
        "<th>Mean fractional aperture error</th>"
        "<th>Mean centroid error (pixel)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + floor_table
    )
