# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser-only candidate review using existing xScan image helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def gallery_indices(rows, labels, scores, threshold, maximum=96):
    """Round-robin confusion/SNR/family strata, stable candidate-ID ordering.

    Classification metrics never use this deliberately stratified gallery.
    Empty strata stay empty. The selection is frozen at the validation
    operating point, even when the exploratory threshold slider changes.
    """
    from collections import defaultdict

    strata = defaultdict(list)
    for i, (row, label, score) in enumerate(
        zip(rows, labels, scores, strict=True)
    ):
        category = (
            "T" if bool(label) == bool(score >= threshold) else "F"
        ) + ("P" if score >= threshold else "N")
        snr_bin = int(np.digitize(row["snr"], [5, 10, 20]))
        strata[(category, snr_bin, row["family"])].append(i)
    for values in strata.values():
        values.sort(key=lambda i: str(rows[i]["candidate_id"]))
    chosen = []
    depth = 0
    while len(chosen) < min(maximum, len(rows)):
        for key in sorted(strata):
            if depth < len(strata[key]):
                chosen.append(strata[key][depth])
                if len(chosen) == maximum:
                    break
        depth += 1
    return np.array(chosen, dtype=int)


def candidate_explorer(
    images,
    rows,
    labels,
    scores,
    threshold,
    *,
    evaluation=None,
    preselected=False,
):
    """All statistics from complete evaluation; at most 96 image triplets."""
    from bokeh.events import Tap
    from bokeh.layouts import column, row
    from bokeh.models import (
        Button,
        ColumnDataSource,
        CustomJS,
        Div,
        LinearColorMapper,
        Range1d,
        Select,
        Slider,
        Span,
    )
    from bokeh.palettes import Greys256, RdBu11

    from cuphoton.xscan import alard_lupton_experiment_review as lab
    from cuphoton.xscan import raw_compare_review as raw

    labels, scores = np.asarray(labels), np.asarray(scores)
    if images.shape[:2] != (len(rows), 3) or len(rows) == 0:
        raise ValueError("nonempty candidate triplets required")
    if images.shape[2] != images.shape[3]:
        raise ValueError("square stamps required")
    if preselected and len(rows) > 96:
        raise ValueError("saved gallery exceeds the 96-candidate limit")
    chosen = (
        np.arange(len(rows))
        if preselected
        else gallery_indices(rows, labels, scores, threshold)
    )
    stamps = images[chosen]
    size = images.shape[-1]
    bank = ColumnDataSource(
        dict(
            search=[s[0].ravel() for s in stamps],
            template=[s[1].ravel() for s in stamps],
            difference=[s[2].ravel() for s in stamps],
            identifier=[str(rows[i]["candidate_id"]) for i in chosen],
            family=[rows[i]["family"] for i in chosen],
            snr=[rows[i]["snr"] for i in chosen],
            label=labels[chosen],
            score=scores[chosen],
        )
    )
    if evaluation is None:
        evaluation = dict(
            labels=labels,
            scores=scores,
            snr=[r["snr"] for r in rows],
            candidate_id=[str(r["candidate_id"]) for r in rows],
            family=[r["family"] for r in rows],
        )
    population_labels = np.asarray(evaluation["labels"])
    population_scores = np.asarray(evaluation["scores"])
    from evaluation import checked

    checked(population_labels, population_scores)
    if any(
        len(evaluation[key]) != len(population_labels)
        for key in ("snr", "candidate_id", "family")
    ):
        raise ValueError("complete evaluation columns must be row-aligned")
    metrics = ColumnDataSource(
        dict(
            label=population_labels,
            score=population_scores,
            snr=evaluation["snr"],
            candidate=evaluation["candidate_id"],
            family=evaluation["family"],
            color=[
                "#007f99" if label else "#ab3d75"
                for label in population_labels
            ],
        )
    )
    from bokeh.plotting import figure

    score_plot = figure(
        title="Every evaluation candidate: score versus SNR",
        height=280,
        sizing_mode="stretch_width",
        x_axis_type="log",
        x_axis_label="Measured detection SNR (marginal diagnostic)",
        y_axis_label="Saved classifier score",
        tools="pan,wheel_zoom,reset,hover",
        tooltips=[
            ("Candidate", "@candidate"),
            ("Family", "@family"),
            ("Label", "@label"),
            ("SNR", "@snr"),
            ("Score", "@score"),
        ],
    )
    score_plot.scatter(
        x="snr", y="score", source=metrics, color="color", size=5, alpha=0.55
    )
    threshold_line = Span(
        location=threshold,
        dimension="width",
        line_color="#777777",
        line_dash="dashed",
    )
    score_plot.add_layout(threshold_line)
    index = Slider(
        start=0,
        end=max(1, len(chosen) - 1),
        value=0,
        step=1,
        title="Gallery position",
    )
    threshold_control = Slider(
        start=0,
        end=1,
        value=float(threshold),
        step=0.001,
        title="Exploratory score threshold",
    )
    prevalence = Slider(
        start=0.001,
        end=0.5,
        value=0.01,
        step=0.001,
        title="Assumed positive prevalence",
    )
    family = Select(
        title="Gallery family",
        value="all",
        options=["all"] + sorted(set(bank.data["family"])),
    )
    category = Select(
        title="Gallery outcome at current threshold",
        value="all",
        options=["all", "TP", "FP", "FN", "TN"],
    )
    transform = Select(
        title="Display units",
        value="raw",
        options=["raw", "robust_z", "unit_robust"],
    )
    previous, following = (
        Button(label="Previous candidate"),
        Button(label="Next candidate"),
    )
    state = ColumnDataSource(dict(candidate=[0]))
    restore = Button(label="Restore frozen threshold")
    restore.js_on_click(
        CustomJS(
            args=dict(threshold=threshold_control, frozen=float(threshold)),
            code="threshold.value=frozen;",
        )
    )
    metadata, queue, exact = (
        Div(),
        Div(),
        Div(text="Tap any image for exact raw pixel values."),
    )
    selection = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))
    sources, mappers, label_sources, plots = [], [], [], []
    xr, yr = Range1d(0, size), Range1d(0, size)
    panel_names = ["Search", "Reference", "xPois difference"]
    for p, title in enumerate(panel_names):
        values = stamps[0, p]
        source = ColumnDataSource(
            lab._image_payload(values, values, transform_label="raw")
        )
        low, high = np.percentile(values, [1, 99])
        if p == 2:
            high = max(abs(low), abs(high))
            low = -high
        mapper = LinearColorMapper(
            palette=Greys256 if p < 2 else list(reversed(RdBu11)),
            low=float(low),
            high=float(high),
        )
        label_source = raw._pixel_label_source()
        plot = raw._image_figure(
            title,
            source,
            mapper,
            selection,
            label_source,
            shared_x_range=xr,
            shared_y_range=yr,
            stamp_width=size,
            stamp_height=size,
        )
        sources.append(source)
        mappers.append(mapper)
        label_sources.append(label_source)
        plots.append(plot)
    inspector = lab._pixel_inspector_callback(
        labels=panel_names,
        sources=sources,
        mappers=mappers,
        label_sources=label_sources,
        selection_source=selection,
        stamp_width=size,
        stamp_height=size,
    )
    for plot in plots:
        plot.js_on_event(Tap, inspector)
        plot.js_on_event(
            Tap,
            CustomJS(
                args=dict(
                    sources=sources,
                    exact=exact,
                    size=size,
                    selection=selection,
                    label_sources=label_sources,
                ),
                code=(
                    Path(__file__).with_name("pixel-readout.js")
                ).read_text(),
            ),
        )
    update = CustomJS(
        args=dict(
            bank=bank,
            metrics=metrics,
            state=state,
            inspector=inspector,
            index=index,
            threshold=threshold_control,
            threshold_line=threshold_line,
            prevalence=prevalence,
            family=family,
            category=category,
            transform=transform,
            sources=sources,
            mappers=mappers,
            metadata=metadata,
            queue=queue,
            exact=exact,
            selection=selection,
            label_sources=label_sources,
            size=size,
        ),
        code=(Path(__file__).with_name("review-controls.js")).read_text(),
    )
    for widget, prop in [
        (index, "value"),
        (threshold_control, "value"),
        (prevalence, "value"),
        (family, "value"),
        (category, "value"),
        (transform, "value"),
    ]:
        widget.js_on_change(prop, update)
    for button, delta in [(previous, -1), (following, 1)]:
        button.js_on_click(
            CustomJS(
                args=dict(
                    index=index,
                    delta=delta,
                    bank=bank,
                    threshold=threshold_control,
                    family=family,
                    category=category,
                ),
                code="""
const matches=[];
for(let j=0;j<bank.data.score.length;j++) {
 const keep=bank.data.score[j]>=threshold.value;
 const positive=bank.data.label[j]===1;
 const outcome=(positive===keep?'T':'F')+(keep?'P':'N');
 if((family.value==='all'||family.value===bank.data.family[j]) &&
    (category.value==='all'||category.value===outcome))matches.push(j);
}
if(!matches.length)return;
let position=matches.indexOf(index.value);
if(position<0)position=delta>0?-1:0;
index.value=matches[(position+delta+matches.length)%matches.length];
""",
            )
        )
    # Initial complete statistics are filled without waiting for an event.
    from evaluation import confusion

    summary = confusion(population_labels, population_scores, threshold)
    queue.text = (
        f"Frozen validation threshold {threshold:.8g}; "
        f"complete evaluation: {summary}. "
        "Move a slider to explore the queue."
    )
    metadata.text = (
        f"Candidate {bank.data['identifier'][0]}, gallery 1/{len(chosen)}; "
        f"family {bank.data['family'][0]}; label {labels[chosen][0]}; "
        f"score {scores[chosen][0]}; measured SNR {bank.data['snr'][0]}. "
        f"Gallery from {len(population_labels)} evaluation candidates."
    )
    return column(
        queue,
        score_plot,
        row(threshold_control, prevalence, restore),
        row(family, category, transform),
        row(previous, index, following),
        metadata,
        row(*plots),
        exact,
        sizing_mode="stretch_width",
    ), dict(
        indices=chosen.tolist(),
        rule="round-robin confusion/SNR/family, candidate-ID order",
        maximum=96,
    )


def performance_diagnostics(labels, predictions):
    """ROC/PR/reliability for complete modeled evaluation, each seed separate.

    ``predictions`` maps a display name to a saved score vector. Empty
    reliability bins are omitted rather than presented as zero-frequency
    measurements. Curves do not select or change operating thresholds.
    """
    from bokeh.layouts import column
    from bokeh.models import ColumnDataSource, Div
    from bokeh.palettes import Category10
    from bokeh.plotting import figure

    from cuphoton.xscan.metrics import (
        brier_score,
        precision_recall_curve,
        reliability_curve,
        roc_curve,
    )

    roc = figure(
        title="ROC: complete candidate pool",
        height=300,
        sizing_mode="stretch_width",
        x_axis_label="False-positive rate",
        y_axis_label="Achieved recall",
        tools="pan,wheel_zoom,reset,save",
    )
    pr = figure(
        title="Precision–recall: modeled prevalence",
        height=300,
        sizing_mode="stretch_width",
        x_axis_label="Recall",
        y_axis_label="Precision",
        tools="pan,wheel_zoom,reset,save",
    )
    reliability = figure(
        title="Reliability on modeled data",
        height=300,
        sizing_mode="stretch_width",
        x_axis_label="Mean saved score in bin",
        y_axis_label="Observed positive fraction in bin",
        tools="pan,wheel_zoom,reset,save,hover",
        tooltips=[("Model/seed", "@name"), ("Bin count", "@count")],
    )
    reliability.line([0, 1], [0, 1], line_dash="dashed", color="#777777")
    briers = []
    for index, (name, scores) in enumerate(predictions.items()):
        color = Category10[10][index % 10]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        roc.line(fpr, tpr, legend_label=name, color=color, line_width=2)
        pr.line(
            recall, precision, legend_label=name, color=color, line_width=2
        )
        bins = reliability_curve(labels, scores)
        source = ColumnDataSource(
            dict(
                score=[b["mean_probability"] for b in bins],
                frequency=[b["empirical_positive_rate"] for b in bins],
                count=[b["count"] for b in bins],
                name=[name] * len(bins),
            )
        )
        reliability.scatter(
            "score",
            "frequency",
            source=source,
            color=color,
            legend_label=name,
            size=7,
        )
        briers.append(f"{name}: {brier_score(labels, scores):.5f}")
    for plot in (roc, pr, reliability):
        plot.legend.click_policy = "hide"
    return column(
        Div(
            text="Brier scores (lower is better): "
            + "; ".join(briers)
            + ". Simulated reliability does not establish probabilities "
            "for an observed sky."
        ),
        roc,
        pr,
        reliability,
        sizing_mode="stretch_width",
    )


def training_diagnostics(root, reference):
    """Saved losses and development learning curve, without refitting."""
    import json

    from bokeh.layouts import column
    from bokeh.palettes import Category10
    from bokeh.plotting import figure

    root = Path(root)
    losses = figure(
        title="Training and validation loss, each seed",
        x_axis_label="Epoch",
        y_axis_label="Binary cross entropy",
        height=300,
        sizing_mode="stretch_width",
    )
    for index, result in enumerate(reference["training"]):
        name = f"{result['mode']}-{result['seed']}"
        history = json.loads(
            (
                root
                / reference["development_directory"]
                / name
                / "history.json"
            ).read_text()
        )
        color = Category10[10][index % 10]
        epochs = [r["epoch"] for r in history]
        losses.line(
            epochs,
            [r["train_loss"] for r in history],
            color=color,
            legend_label=name + " train",
            line_dash="dashed",
        )
        losses.line(
            epochs,
            [r["val_loss"] for r in history],
            color=color,
            legend_label=name + " validation",
            line_width=2,
        )
        losses.scatter(
            epochs, [r["train_loss"] for r in history], color=color, size=5
        )
        losses.scatter(
            epochs,
            [r["val_loss"] for r in history],
            color=color,
            marker="square",
            size=5,
        )
    losses.legend.click_policy = "hide"
    curve = figure(
        title="Development learning curve: triplet seed 11",
        x_axis_label="Independent training fields",
        y_axis_label="Validation FPR at validation-selected recall threshold",
        height=300,
        sizing_mode="stretch_width",
    )
    points = [
        (r["training_fields"], r["validation"]["fpr"])
        for r in reference["controls"]["learning_curve"]
    ]
    full = next(
        r
        for r in reference["training"]
        if r["mode"] == "triplet" and r["seed"] == 11
    )
    points.append(
        (reference["design"]["train_fields"], full["validation"]["fpr"])
    )
    points.sort()
    x, y = zip(*points, strict=True)
    curve.line(x, y, color="#007f99")
    curve.scatter(x, y, color="#007f99", size=9)
    return column(losses, curve, sizing_mode="stretch_width")
