# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bokeh view for experimenting with Alard-Lupton display normalization."""

from __future__ import annotations

import html
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    CustomJS,
    Div,
    LinearColorMapper,
    Range1d,
    TextInput,
)
from bokeh.palettes import Greys256, RdBu11
from bokeh.server.server import Server

from cuphoton.xscan import raw_compare_review as raw_compare

DEFAULT_REVIEW_DIR = raw_compare.DEFAULT_REVIEW_DIR
DEFAULT_HOST = os.environ.get("CUPHOTON_XSCAN_ALARD_LUPTON_HOST", "localhost")
DEFAULT_PORT = int(os.environ.get("CUPHOTON_XSCAN_ALARD_LUPTON_PORT", "5012"))


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    source_key: str
    transform: str
    diverging: bool
    fixed_range: tuple[float, float] | None = None


PANEL_ROWS = (
    (
        "Flux units",
        (
            PanelSpec("search_flux", "Flux: Search", "search", "raw", False),
            PanelSpec(
                "template_flux", "Flux: Template", "template", "raw", False
            ),
            PanelSpec(
                "raw_diff_flux",
                "Flux: Search - Template",
                "raw_diff",
                "raw",
                True,
            ),
            PanelSpec(
                "al_flux",
                "Flux: Alard-Lupton",
                "stored_diff",
                "raw",
                True,
            ),
        ),
    ),
    (
        "Empirical normalized display",
        (
            PanelSpec(
                "search_unit",
                "Norm: Search",
                "search",
                "unit_robust",
                False,
                (0.0, 1.0),
            ),
            PanelSpec(
                "template_unit",
                "Norm: Template",
                "template",
                "unit_robust",
                False,
                (0.0, 1.0),
            ),
            PanelSpec(
                "raw_diff_z",
                "z: Search - Template",
                "raw_diff",
                "robust_z",
                True,
                (-3.0, 3.0),
            ),
            PanelSpec(
                "al_z",
                "z: Alard-Lupton",
                "stored_diff",
                "robust_z",
                True,
                (-3.0, 3.0),
            ),
        ),
    ),
)


def run_server(
    *,
    review_dir: Path | None = DEFAULT_REVIEW_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Start the blocking Alard-Lupton display-lab Bokeh server."""

    if review_dir is None:
        raise ValueError(
            "--review-dir or CUPHOTON_XSCAN_REVIEW_DIR is required"
        )
    review_dir = review_dir.expanduser().resolve()

    def app(doc):
        build_document(doc, review_dir=review_dir)

    port = int(port)
    origins = [
        f"localhost:{port}",
        f"127.0.0.1:{port}",
    ]
    server = Server(
        {"/": app},
        address=host,
        port=port,
        allow_websocket_origin=origins,
        session_token_expiration=24 * 60 * 60,
    )
    server.start()
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(
        "XScan Alard-Lupton display lab: "
        f"http://{display_host}:{port}/?s=3510",
        flush=True,
    )
    server.io_loop.start()


def build_document(doc, *, review_dir: Path) -> None:
    manifest, queue = raw_compare._load_review_manifest_and_queue(review_dir)
    dataset_dir = Path(manifest["dataset_dir"]).expanduser().resolve()
    queue_by_sample = {int(item["sample_index"]): item for item in queue}
    queue_order = [int(item["sample_index"]) for item in queue]

    search = np.load(dataset_dir / "search.npy", mmap_mode="r")
    template = np.load(dataset_dir / "template.npy", mmap_mode="r")
    difference_path = dataset_dir / "difference.npy"
    difference = (
        np.load(difference_path, mmap_mode="r")
        if difference_path.exists()
        else None
    )
    metadata_by_sample = raw_compare._load_metadata_by_sample(
        dataset_dir / "metadata.jsonl"
    )

    sample_count = int(search.shape[0])
    if template.shape != search.shape:
        raise ValueError(
            f"template shape {template.shape} does not match "
            f"search {search.shape}"
        )
    if difference is not None and difference.shape != search.shape:
        raise ValueError(
            f"difference shape {difference.shape} does not match "
            f"search {search.shape}"
        )

    stamp_height, stamp_width = search.shape[1:]
    initial_sample, query_notice = raw_compare._initial_sample_from_query(
        doc,
        queue_order=queue_order,
        sample_count=sample_count,
    )
    state = {"sample_index": initial_sample}

    doc.title = f"XScan Alard-Lupton Display Lab {initial_sample}"

    title = Div(width=380, css_classes=["review-brand"])
    progress_div = Div(width=120)
    sample_input = TextInput(
        title="Sample index",
        value=str(initial_sample),
        width=220,
    )
    go_button = Button(label="Go", button_type="primary", width=70)
    previous_button = Button(label="Previous", width=110)
    next_button = Button(label="Next", width=110)
    reset_button = Button(label="Reset View", width=118)
    status = Div(text="", width=360, css_classes=["status-card"])
    metadata = Div(text="", width=1160, css_classes=["summary-wrap"])
    url_state = Div(text=str(initial_sample), visible=False)
    url_state.js_on_change(
        "text",
        CustomJS(
            code="""
const url = new URL(window.location.href);
url.searchParams.set("s", cb_obj.text);
window.history.replaceState({}, "", url.toString());
""",
        ),
    )

    shared_x_range = Range1d(0, stamp_width, bounds=(0, stamp_width))
    shared_y_range = Range1d(0, stamp_height, bounds=(0, stamp_height))
    sources: dict[str, ColumnDataSource] = {}
    label_sources: dict[str, ColumnDataSource] = {}
    mappers: dict[str, LinearColorMapper] = {}
    pixel_selection_source = ColumnDataSource(
        {"x": [], "y": [], "width": [], "height": []}
    )
    image_plots = []
    pixel_labels = []
    pixel_sources = []
    pixel_label_sources = []
    pixel_mappers = []
    figures = []

    for row_label, row_panels in PANEL_ROWS:
        row_figures = []
        for spec in row_panels:
            sources[spec.key] = ColumnDataSource(
                _image_payload(
                    np.zeros((stamp_height, stamp_width)),
                    np.zeros((stamp_height, stamp_width)),
                    transform_label=spec.transform,
                )
            )
            label_sources[spec.key] = raw_compare._pixel_label_source()
            palette = (
                list(reversed(RdBu11)) if spec.diverging else list(Greys256)
            )
            low, high = spec.fixed_range or (
                (-1.0, 1.0) if spec.diverging else (0.0, 1.0)
            )
            mappers[spec.key] = LinearColorMapper(
                palette=palette,
                low=low,
                high=high,
            )
            plot = raw_compare._image_figure(
                spec.title,
                sources[spec.key],
                mappers[spec.key],
                pixel_selection_source,
                label_sources[spec.key],
                shared_x_range=shared_x_range,
                shared_y_range=shared_y_range,
                stamp_width=stamp_width,
                stamp_height=stamp_height,
            )
            image_plots.append(plot)
            pixel_labels.append(f"{row_label}: {spec.title}")
            pixel_sources.append(sources[spec.key])
            pixel_label_sources.append(label_sources[spec.key])
            pixel_mappers.append(mappers[spec.key])
            row_figures.append(
                column(
                    plot,
                    width=260,
                    css_classes=["stamp-card"],
                    styles=dict(raw_compare.STAMP_CARD_STYLE),
                )
            )
        figures.append(
            row(
                *row_figures,
                width=1130,
                css_classes=["triptych-row"],
                styles={"gap": "12px", "justify-content": "space-between"},
            )
        )

    pixel_callback = _pixel_inspector_callback(
        labels=pixel_labels,
        sources=pixel_sources,
        mappers=pixel_mappers,
        label_sources=pixel_label_sources,
        selection_source=pixel_selection_source,
        stamp_width=stamp_width,
        stamp_height=stamp_height,
    )
    for plot in image_plots:
        plot.js_on_event(Tap, pixel_callback)

    def queue_position(sample_index: int) -> int | None:
        try:
            return queue_order.index(sample_index)
        except ValueError:
            return None

    def current_item(sample_index: int) -> dict[str, Any]:
        item = dict(metadata_by_sample.get(sample_index, {}))
        item.update(queue_by_sample.get(sample_index, {}))
        item.setdefault("sample_index", sample_index)
        return item

    def update_sample(
        sample_index: int,
        *,
        reset_ranges: bool = False,
    ) -> None:
        if sample_index < 0 or sample_index >= sample_count:
            status.text = raw_compare._status_html(
                f"Sample index {sample_index} is outside "
                f"0..{sample_count - 1}.",
                error=True,
            )
            return
        state["sample_index"] = sample_index
        doc.title = f"XScan Alard-Lupton Display Lab {sample_index}"
        sample_input.value = str(sample_index)
        url_state.text = str(sample_index)

        base_arrays = raw_compare._sample_arrays(
            sample_index,
            search=search,
            template=template,
            difference=difference,
        )
        scale_ranges: dict[str, tuple[float, float]] = {}
        for _, row_panels in PANEL_ROWS:
            for spec in row_panels:
                source_array = base_arrays[spec.source_key]
                display_array = _display_array(source_array, spec.transform)
                if spec.fixed_range is not None:
                    low, high = spec.fixed_range
                else:
                    low, high = raw_compare._scale_range(
                        display_array,
                        mode="review",
                        diverging=spec.diverging,
                    )
                scale_ranges[spec.key] = (low, high)
                mapper = mappers[spec.key]
                mapper.low = low
                mapper.high = high
                sources[spec.key].data = _image_payload(
                    display_array,
                    source_array,
                    transform_label=spec.transform,
                )

        if reset_ranges:
            shared_x_range.start = 0
            shared_x_range.end = stamp_width
            shared_y_range.start = 0
            shared_y_range.end = stamp_height

        item = current_item(sample_index)
        qpos = queue_position(sample_index)
        title.text = raw_compare._review_header_html(
            title="XScan AL Display Lab",
            subtitle="Raw flux beside empirical normalized display",
            id_label="Candidate ID",
            item=item,
        )
        progress_div.text = raw_compare._progress_html(qpos, len(queue_order))
        metadata.text = _metadata_html(
            sample_index=sample_index,
            item=item,
            queue_position=qpos,
            queue_count=len(queue_order),
            arrays=base_arrays,
            scales=scale_ranges,
            difference_path=(
                difference_path if difference is not None else None
            ),
        )
        if query_notice:
            status.text = raw_compare._status_html(query_notice)
        elif qpos is None:
            status.text = raw_compare._status_html(
                f"Sample {sample_index} is not in this review queue; "
                "showing dataset metadata where available."
            )
        else:
            status.text = raw_compare._status_html(
                f"Queue item {qpos + 1} of {len(queue_order)}."
            )

    def go_to_sample() -> None:
        raw = sample_input.value.strip()
        try:
            sample_index = int(raw)
        except ValueError:
            status.text = raw_compare._status_html(
                f"Invalid sample index: {raw!r}",
                error=True,
            )
            return
        update_sample(sample_index)

    def move_queue(delta: int) -> None:
        current = state["sample_index"]
        qpos = queue_position(current)
        if qpos is None:
            qpos = 0 if delta >= 0 else len(queue_order) - 1
        else:
            qpos = (qpos + delta) % len(queue_order)
        update_sample(queue_order[qpos])

    go_button.on_click(go_to_sample)
    previous_button.on_click(lambda: move_queue(-1))
    next_button.on_click(lambda: move_queue(1))
    reset_button.on_click(
        lambda: update_sample(state["sample_index"], reset_ranges=True)
    )

    header = row(
        title,
        row(
            previous_button,
            progress_div,
            next_button,
            width=390,
            css_classes=["review-nav"],
            styles={
                "align-items": "center",
                "background": "#ffffff",
                "border": "1px solid #dde6f4",
                "border-radius": "8px",
                "box-shadow": "0 8px 22px rgba(42,55,105,0.08)",
                "gap": "10px",
                "justify-content": "center",
                "padding": "8px 10px",
            },
        ),
        column(
            row(
                sample_input,
                go_button,
                styles={"align-items": "end", "gap": "10px"},
            ),
            reset_button,
            status,
            width=360,
            css_classes=["header-tools"],
            styles={"gap": "8px"},
        ),
        width=1160,
        css_classes=["review-header"],
        styles=dict(raw_compare.REVIEW_HEADER_STYLE),
    )
    image_card = column(
        Div(
            text=(
                "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
                "margin:0 0 8px;'>Alard-Lupton Display Normalization</h2>"
            ),
            width=1130,
        ),
        figures[0],
        figures[1],
        width=1160,
        css_classes=["review-card", "image-triptych"],
        styles=dict(raw_compare.REVIEW_CARD_STYLE),
    )
    layout = column(
        header,
        image_card,
        metadata,
        url_state,
        width=1160,
        css_classes=["review-shell"],
        styles={"margin": "18px auto 34px", "gap": "12px"},
    )
    raw_compare._apply_style(doc)
    doc.add_root(layout)
    update_sample(initial_sample, reset_ranges=True)


def _display_array(array: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if transform == "raw":
        return values
    center, sigma = raw_compare._robust_pixel_stats(values)
    if not np.isfinite(center):
        center = 0.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    if transform == "robust_z":
        return (values - center) / sigma
    if transform == "unit_robust":
        return (values - (center - 3.0 * sigma)) / (6.0 * sigma)
    raise ValueError(f"unknown display transform: {transform}")


def _image_payload(
    display: np.ndarray,
    raw: np.ndarray,
    *,
    transform_label: str,
) -> dict[str, Any]:
    display_array = np.asarray(display, dtype=np.float64)
    raw_array = np.asarray(raw, dtype=np.float64)
    height, width = display_array.shape
    raw_center, raw_sigma = raw_compare._robust_pixel_stats(raw_array)
    display_center, display_sigma = raw_compare._robust_pixel_stats(
        display_array
    )
    return {
        "image": [display_array],
        "flat": [display_array.ravel()],
        "raw_flat": [raw_array.ravel()],
        "robust_center": [raw_center],
        "robust_sigma": [raw_sigma],
        "display_center": [display_center],
        "display_sigma": [display_sigma],
        "transform": [transform_label],
        "x": [0],
        "y": [0],
        "dw": [width],
        "dh": [height],
    }


def _pixel_inspector_callback(
    *,
    labels: list[str],
    sources: list[ColumnDataSource],
    mappers: list[LinearColorMapper],
    label_sources: list[ColumnDataSource],
    selection_source: ColumnDataSource,
    stamp_width: int,
    stamp_height: int,
) -> CustomJS:
    return CustomJS(
        args={
            "labels": labels,
            "sources": sources,
            "mappers": mappers,
            "label_sources": label_sources,
            "selection_source": selection_source,
            "stamp_width": stamp_width,
            "stamp_height": stamp_height,
        },
        code="""
const col = Math.floor(cb_obj.x);
const row = Math.floor(cb_obj.y);
if (
  !Number.isFinite(col) || !Number.isFinite(row) ||
  col < 0 || row < 0 || col >= stamp_width || row >= stamp_height
) {
  return;
}

const currentX = selection_source.data.x || [];
const currentY = selection_source.data.y || [];
const samePixel = (
  currentX.length > 0 && currentY.length > 0 &&
  Math.floor(currentX[0]) === col &&
  Math.floor(currentY[0]) === row
);
if (samePixel) {
  selection_source.data = {x: [], y: [], width: [], height: []};
  selection_source.change.emit();
  for (const source of label_sources) {
    source.data = {x: [], y: [], x_offset: [], y_offset: [], text: []};
    source.change.emit();
  }
  return;
}

selection_source.data = {
  x: [col + 0.5],
  y: [row + 0.5],
  width: [1],
  height: [1],
};
selection_source.change.emit();

function formatValue(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "nan";
  }
  const numeric = Number(value);
  const magnitude = Math.abs(numeric);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) {
    return numeric.toExponential(4);
  }
  return numeric.toFixed(4);
}

function scalar(source, name) {
  const values = source.data[name];
  if (!values || !values.length) {
    return NaN;
  }
  return Number(values[0]);
}

function flatValue(source, name) {
  const flatColumn = source.data[name];
  if (!flatColumn || !flatColumn.length) {
    return NaN;
  }
  const flat = flatColumn[0];
  const index = row * stamp_width + col;
  if (flat && typeof flat.get === "function") {
    return flat.get(index);
  }
  return flat[index];
}

function robustSigma(source, rawValue) {
  const center = scalar(source, "robust_center");
  const scale = scalar(source, "robust_sigma");
  if (!Number.isFinite(center) || !Number.isFinite(scale) || scale <= 0) {
    return NaN;
  }
  return (Number(rawValue) - center) / scale;
}

const xOffset = col > stamp_width * 0.58 ? -158 : 10;
const yOffset = row > stamp_height * 0.58 ? -90 : 10;
for (let index = 0; index < labels.length; index += 1) {
  const shown = flatValue(sources[index], "flat");
  const rawValue = flatValue(sources[index], "raw_flat");
  const sigma = robustSigma(sources[index], rawValue);
  const mapper = mappers[index];
  label_sources[index].data = {
    x: [col + 0.5],
    y: [row + 0.5],
    x_offset: [xOffset],
    y_offset: [yOffset],
    text: [
      `${labels[index]}\\n`
      + `(x,y): ${col}, ${row}\\n`
      + `raw: ${formatValue(rawValue)}\\n`
      + `shown: ${formatValue(shown)}\\n`
      + `raw robust sigma: ${formatValue(sigma)}\\n`
      + `colorbar: ${formatValue(mapper.low)} .. ${formatValue(mapper.high)}`,
    ],
  };
  label_sources[index].change.emit();
}
""",
    )


def _metadata_html(
    *,
    sample_index: int,
    item: dict[str, Any],
    queue_position: int | None,
    queue_count: int,
    arrays: dict[str, np.ndarray],
    scales: dict[str, tuple[float, float]],
    difference_path: Path | None,
) -> str:
    stats = _array_stats(arrays)
    queue_text = (
        f"{queue_position + 1} / {queue_count}"
        if queue_position is not None
        else "not in queue"
    )
    fields: list[tuple[str, Any]] = [
        ("Candidate ID", item.get("candidate_id")),
        ("Sample Index", sample_index),
        ("Queue Position", queue_text),
        ("Label", item.get("label")),
        ("Prediction", item.get("prediction")),
        ("Probability", item.get("probability")),
        ("Rank", item.get("rank")),
        ("Alard-Lupton Mode", item.get("difference_mode")),
        ("SNR", item.get("snr")),
        ("Alard-Lupton SNR", item.get("diff_snr")),
        ("raw diff robust sigma", stats["raw_diff_sigma"]),
        ("Alard-Lupton robust sigma", stats["al_sigma"]),
        ("raw diff |p98|", stats["raw_diff_p98_abs"]),
        ("Alard-Lupton |p98|", stats["al_p98_abs"]),
        ("Alard-Lupton |max|", stats["al_max_abs"]),
        ("corr(Alard-Lupton, raw)", stats["corr"]),
        ("rms(Alard-Lupton - raw)", stats["rms_residual"]),
        ("center Alard-Lupton robust z", stats["center_al_z"]),
    ]
    rows = [
        _summary_field_html(label, value)
        for label, value in fields
        if value is not None
    ]

    scale_rows = []
    scale_labels = {
        spec.key: spec.title
        for _, row_specs in PANEL_ROWS
        for spec in row_specs
    }
    for key, (low, high) in scales.items():
        scale_rows.append(
            "<tr>"
            "<th style='border-bottom:1px solid #e3eaf5;color:#51617f;"
            "font-weight:760;padding:5px 7px;text-align:left;'>"
            f"{html.escape(scale_labels.get(key, key))}</th>"
            "<td style='border-bottom:1px solid #e3eaf5;padding:5px 7px;"
            f"text-align:left;'>{raw_compare._fmt(low)} .. "
            f"{raw_compare._fmt(high)}</td>"
            "</tr>"
        )
    diff_source = (
        html.escape(str(difference_path))
        if difference_path is not None
        else "not present; Alard-Lupton falls back to search - template"
    )
    card_style = (
        "background:#fff;border:1px solid #dde6f4;border-radius:8px;"
        "box-shadow:0 14px 38px rgba(42,55,105,0.12);box-sizing:border-box;"
        "padding:14px;width:1160px;"
    )
    heading_style = (
        "color:#07145c;font-size:16px;font-weight:760;margin:0 0 7px;"
    )
    grid_style = (
        "column-gap:22px;color:#07145c;display:grid;font-size:12px;"
        "grid-template-columns:repeat(3,minmax(0,1fr));width:100%;"
    )
    return (
        f"<div class='summary-card' style='{card_style}'>"
        f"<h2 style='{heading_style}'>Detection Summary</h2>"
        f"<div class='summary-grid' style='{grid_style}'>"
        + "".join(rows)
        + "</div>"
        "<h2 style='color:#07145c;font-size:16px;font-weight:760;"
        "margin:14px 0 7px;'>Display Experiment</h2>"
        "<div style='color:#51617f;font-size:13px;line-height:1.35;"
        "margin-bottom:8px;'>Top row uses the saved flux-unit arrays. "
        "Bottom row is an empirical display diagnostic: search/template use "
        "robust center +/- 3 sigma mapped to 0..1, and the difference panes "
        "show robust z values on a fixed -3..3 colorbar. This is not a "
        "propagated Poisson-variance-normalized Alard-Lupton product.</div>"
        "<table style='border-collapse:collapse;color:#07145c;font-size:12px;"
        "margin-bottom:8px;width:100%;'><thead><tr>"
        "<th style='border-bottom:1px solid #e3eaf5;color:#51617f;"
        "font-weight:760;padding:5px 7px;text-align:left;'>Pane</th>"
        "<th style='border-bottom:1px solid #e3eaf5;color:#51617f;"
        "font-weight:760;padding:5px 7px;text-align:left;'>Colorbar</th>"
        "</tr></thead><tbody>" + "".join(scale_rows) + "</tbody></table>"
        "<div style='color:#51617f;font-size:13px;'>Alard-Lupton source: "
        f"{diff_source}</div>"
        "</div>"
    )


def _summary_field_html(label: str, value: Any) -> str:
    field_style = (
        "border-top:1px solid #edf1f7;display:grid;gap:8px;"
        "grid-template-columns:minmax(118px,42%) minmax(0,1fr);"
        "line-height:1.25;padding:4px 0;"
    )
    key_style = "color:#7080b5;font-weight:650;"
    value_style = "color:#07145c;min-width:0;overflow-wrap:anywhere;"
    return (
        f"<div class='summary-field' style='{field_style}'>"
        f"<div class='summary-key' style='{key_style}'>"
        f"{html.escape(label)}</div>"
        f"<div class='summary-value' style='{value_style}'>"
        f"{raw_compare._fmt(value)}</div>"
        "</div>"
    )


def _array_stats(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    raw = np.asarray(arrays["raw_diff"], dtype=np.float64)
    al = np.asarray(arrays["stored_diff"], dtype=np.float64)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    mask = np.isfinite(raw) & np.isfinite(al)
    if np.count_nonzero(mask) >= 2:
        corr = float(np.corrcoef(raw[mask].ravel(), al[mask].ravel())[0, 1])
    else:
        corr = math.nan
    _, raw_sigma = raw_compare._robust_pixel_stats(raw)
    al_center, al_sigma = raw_compare._robust_pixel_stats(al)
    center_y = raw.shape[0] // 2
    center_x = raw.shape[1] // 2
    center_al = float(al[center_y, center_x])
    if np.isfinite(al_sigma) and al_sigma > 0:
        center_al_z = (center_al - al_center) / al_sigma
    else:
        center_al_z = math.nan
    return {
        "raw_diff_sigma": raw_sigma,
        "al_sigma": al_sigma,
        "raw_diff_p98_abs": _percentile_abs(raw, 98),
        "al_p98_abs": _percentile_abs(al, 98),
        "al_max_abs": _max_abs(al),
        "corr": corr,
        "rms_residual": float(np.sqrt(np.nanmean(residual**2))),
        "center_al_z": float(center_al_z),
    }


def _percentile_abs(array: np.ndarray, percentile: float) -> float:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(np.abs(finite), percentile))


def _max_abs(array: np.ndarray) -> float:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    return float(np.max(np.abs(finite)))
