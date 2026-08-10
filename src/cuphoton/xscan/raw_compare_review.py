# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bokeh view for comparing XScan review stamp arrays."""

from __future__ import annotations

import base64
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    LabelSet,
    LinearColorMapper,
    Range1d,
    Span,
    TextInput,
    WheelZoomTool,
)
from bokeh.palettes import Greys256, RdBu11
from bokeh.plotting import figure
from bokeh.server.server import Server

_DEFAULT_REVIEW_DIR = os.environ.get("CUPHOTON_XSCAN_REVIEW_DIR")
DEFAULT_REVIEW_DIR = (
    Path(_DEFAULT_REVIEW_DIR).expanduser() if _DEFAULT_REVIEW_DIR else None
)
DEFAULT_HOST = os.environ.get("CUPHOTON_XSCAN_COMPARE_HOST", "localhost")
DEFAULT_PORT = int(os.environ.get("CUPHOTON_XSCAN_COMPARE_PORT", "5011"))
_LOGO_DEFAULT = Path(__file__).with_name("assets") / "cuphoton-logo-128.png"
LOGO_PATH = Path(
    os.environ.get("CUPHOTON_XSCAN_LOGO_PATH", str(_LOGO_DEFAULT))
)

PANEL_COLUMNS = (
    ("search", "Search", False),
    ("template", "Template", False),
    ("raw_diff", "Search - Template", True),
    ("stored_diff", "Alard-Lupton", True),
    ("residual", "Alard-Lupton - Raw", True),
)
ROW_MODES = (
    ("full", "Native Stretch"),
    ("review", "Percentile Stretch"),
)
SUMMARY_FIELDS = (
    "queue_id",
    "rank",
    "rank_reason",
    "candidate_id",
    "label",
    "label_source",
    "known_error",
    "prediction",
    "probability",
    "band",
    "exposure_id",
    "x",
    "y",
    "center_source",
    "snr",
    "diff_snr",
    "difference_mode",
    "manifest_difference_mode",
)

REVIEW_HEADER_STYLE = {
    "align-items": "center",
    "background": "rgba(255, 255, 255, 0.94)",
    "border": "1px solid #dde6f4",
    "border-radius": "8px",
    "box-shadow": "0 14px 38px rgba(42, 55, 105, 0.12)",
    "gap": "18px",
    "padding": "14px 16px",
}
REVIEW_CARD_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #dde6f4",
    "border-radius": "8px",
    "box-shadow": "0 14px 38px rgba(42, 55, 105, 0.12)",
    "padding": "14px",
}
STAMP_CARD_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #e4ebf5",
    "border-radius": "8px",
    "padding": "8px",
}


def run_server(
    *,
    review_dir: Path | None = DEFAULT_REVIEW_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Start the blocking raw-comparison Bokeh server."""

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
    print(
        f"XScan raw comparison server: http://{host}:{port}/?s=1352",
        flush=True,
    )
    server.io_loop.start()


def build_document(doc, *, review_dir: Path) -> None:
    manifest, queue = _load_review_manifest_and_queue(review_dir)
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
    metadata_by_sample = _load_metadata_by_sample(
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
    initial_sample, query_notice = _initial_sample_from_query(
        doc,
        queue_order=queue_order,
        sample_count=sample_count,
    )
    state = {"sample_index": initial_sample}

    doc.title = f"XScan Raw Comparison {initial_sample}"

    title = Div(width=470, css_classes=["review-brand"])
    progress_div = Div(width=120)
    sample_input = TextInput(
        title="Sample index",
        value=str(initial_sample),
        width=245,
    )
    go_button = Button(label="Go", button_type="primary", width=74)
    previous_button = Button(label="Previous", width=120)
    next_button = Button(label="Next", width=120)
    reset_button = Button(label="Reset View", width=120)
    status = Div(text="", width=420, css_classes=["status-card"])
    metadata = Div(text="", width=1400, css_classes=["summary-wrap"])
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
    sources: dict[tuple[str, str], ColumnDataSource] = {}
    label_sources: dict[tuple[str, str], ColumnDataSource] = {}
    mappers: dict[tuple[str, str], LinearColorMapper] = {}
    pixel_selection_source = ColumnDataSource(
        {"x": [], "y": [], "width": [], "height": []}
    )
    image_plots = []
    pixel_labels = []
    pixel_sources = []
    pixel_label_sources = []
    pixel_mappers = []
    figures = []

    for mode, row_label in ROW_MODES:
        row_figures = []
        for key, label, diverging in PANEL_COLUMNS:
            panel_key = (mode, key)
            sources[panel_key] = ColumnDataSource(
                _image_payload(np.zeros((stamp_height, stamp_width)))
            )
            label_sources[panel_key] = _pixel_label_source()
            palette = list(reversed(RdBu11)) if diverging else list(Greys256)
            mappers[panel_key] = LinearColorMapper(
                palette=palette,
                low=-1.0 if diverging else 0.0,
                high=1.0,
            )
            plot = _image_figure(
                f"{row_label}: {label}",
                sources[panel_key],
                mappers[panel_key],
                pixel_selection_source,
                label_sources[panel_key],
                shared_x_range=shared_x_range,
                shared_y_range=shared_y_range,
                stamp_width=stamp_width,
                stamp_height=stamp_height,
            )
            image_plots.append(plot)
            pixel_labels.append(f"{row_label}: {label}")
            pixel_sources.append(sources[panel_key])
            pixel_label_sources.append(label_sources[panel_key])
            pixel_mappers.append(mappers[panel_key])
            row_figures.append(
                column(
                    plot,
                    width=260,
                    css_classes=["stamp-card"],
                    styles=dict(STAMP_CARD_STYLE),
                )
            )
        figures.append(
            row(
                *row_figures,
                width=1370,
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
            status.text = _status_html(
                f"Sample index {sample_index} is outside "
                f"0..{sample_count - 1}.",
                error=True,
            )
            return
        state["sample_index"] = sample_index
        doc.title = f"XScan Raw Comparison {sample_index}"
        sample_input.value = str(sample_index)
        url_state.text = str(sample_index)

        arrays = _sample_arrays(
            sample_index,
            search=search,
            template=template,
            difference=difference,
        )
        scale_ranges: dict[tuple[str, str], tuple[float, float]] = {}
        for mode, _ in ROW_MODES:
            for key, _, diverging in PANEL_COLUMNS:
                panel_key = (mode, key)
                low, high = _scale_range(
                    arrays[key],
                    mode=mode,
                    diverging=diverging,
                )
                scale_ranges[panel_key] = (low, high)
                mapper = mappers[panel_key]
                mapper.low = low
                mapper.high = high
                sources[panel_key].data = _image_payload(arrays[key])

        if reset_ranges:
            shared_x_range.start = 0
            shared_x_range.end = stamp_width
            shared_y_range.start = 0
            shared_y_range.end = stamp_height

        item = current_item(sample_index)
        qpos = queue_position(sample_index)
        title.text = _review_header_html(
            title="XScan Review",
            subtitle="Review astronomical detections with human insight",
            id_label="Candidate ID",
            item=item,
        )
        progress_div.text = _progress_html(qpos, len(queue_order))
        metadata.text = _metadata_html(
            sample_index=sample_index,
            item=item,
            queue_position=qpos,
            queue_count=len(queue_order),
            arrays=arrays,
            scales=scale_ranges,
            difference_path=(
                difference_path if difference is not None else None
            ),
        )
        if query_notice:
            status.text = _status_html(query_notice)
        elif qpos is None:
            status.text = _status_html(
                f"Sample {sample_index} is not in this review queue; "
                "showing dataset metadata where available."
            )
        else:
            status.text = _status_html(
                f"Queue item {qpos + 1} of {len(queue_order)}."
            )

    def go_to_sample() -> None:
        raw = sample_input.value.strip()
        try:
            sample_index = int(raw)
        except ValueError:
            status.text = _status_html(
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
            width=450,
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
            width=420,
            css_classes=["header-tools"],
            styles={"gap": "8px"},
        ),
        width=1400,
        css_classes=["review-header"],
        styles=dict(REVIEW_HEADER_STYLE),
    )
    image_card = column(
        Div(
            text=(
                "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
                "margin:0 0 8px;'>Image Comparison</h2>"
            ),
            width=1370,
        ),
        figures[0],
        figures[1],
        width=1400,
        css_classes=["review-card", "image-triptych"],
        styles=dict(REVIEW_CARD_STYLE),
    )
    layout = column(
        header,
        image_card,
        metadata,
        url_state,
        width=1400,
        css_classes=["review-shell"],
        styles={"margin": "18px auto 34px", "gap": "12px"},
    )
    _apply_style(doc)
    doc.add_root(layout)
    update_sample(initial_sample, reset_ranges=True)


def _load_review_manifest_and_queue(
    review_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = review_dir / "manifest.json"
    queue_path = review_dir / "queue.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue = []
    with queue_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                queue.append(json.loads(line))
    return manifest, queue


def _load_metadata_by_sample(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "sample_index" in row:
                rows[int(row["sample_index"])] = row
    return rows


def _initial_sample_from_query(
    doc,
    *,
    queue_order: list[int],
    sample_count: int,
) -> tuple[int, str | None]:
    fallback = queue_order[0] if queue_order else 0
    raw = _first_query_argument(doc, "s", "sample_index")
    if raw is None:
        return fallback, None
    try:
        sample_index = int(raw)
    except ValueError:
        return fallback, f"Ignoring invalid sample query value {raw!r}."
    if sample_index < 0 or sample_index >= sample_count:
        return (
            fallback,
            f"Requested sample_index {sample_index} is outside "
            f"0..{sample_count - 1}; showing queue start instead.",
        )
    return sample_index, None


def _first_query_argument(doc, *names: str) -> str | None:
    session_context = getattr(doc, "session_context", None)
    request = getattr(session_context, "request", None)
    arguments = getattr(request, "arguments", None)
    if not isinstance(arguments, dict):
        return None
    for name in names:
        values = arguments.get(name)
        if values is None:
            continue
        if isinstance(values, bytes | str):
            raw = values
        else:
            values = list(values)
            if not values:
                continue
            raw = values[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        value = str(raw).strip()
        if value:
            return value
    return None


def _sample_arrays(
    sample_index: int,
    *,
    search: np.ndarray,
    template: np.ndarray,
    difference: np.ndarray | None,
) -> dict[str, np.ndarray]:
    search_image = np.asarray(search[sample_index], dtype=np.float64)
    template_image = np.asarray(template[sample_index], dtype=np.float64)
    raw_diff = search_image - template_image
    stored_diff = (
        np.asarray(difference[sample_index], dtype=np.float64)
        if difference is not None
        else raw_diff
    )
    return {
        "search": search_image,
        "template": template_image,
        "raw_diff": raw_diff,
        "stored_diff": stored_diff,
        "residual": stored_diff - raw_diff,
    }


def _scale_range(
    array: np.ndarray,
    *,
    mode: str,
    diverging: bool,
) -> tuple[float, float]:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if diverging else (0.0, 1.0)

    if mode == "full":
        if diverging:
            limit = float(np.max(np.abs(finite)))
            limit = max(limit, 1e-6)
            return -limit, limit
        low = float(np.min(finite))
        high = float(np.max(finite))
    elif mode == "review":
        if diverging:
            limit = float(np.percentile(np.abs(finite), 98))
            limit = max(limit, 1e-6)
            return -limit, limit
        low, high = np.percentile(finite, [2, 98])
        low = float(low)
        high = float(high)
    else:
        raise ValueError(f"unknown scale mode: {mode}")

    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        center = float(np.min(finite)) if finite.size else 0.0
        return center, center + 1e-6
    return low, high


def _image_payload(image: np.ndarray) -> dict[str, Any]:
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape
    center, sigma = _robust_pixel_stats(array)
    return {
        "image": [array],
        "flat": [array.ravel()],
        "robust_center": [center],
        "robust_sigma": [sigma],
        "x": [0],
        "y": [0],
        "dw": [width],
        "dh": [height],
    }


def _robust_pixel_stats(array: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = math.nan
    return center, sigma


def _pixel_label_source() -> ColumnDataSource:
    return ColumnDataSource(
        {"x": [], "y": [], "x_offset": [], "y_offset": [], "text": []},
        name="raw-pixel-label",
    )


def _pixel_label_set(source: ColumnDataSource) -> LabelSet:
    return LabelSet(
        x="x",
        y="y",
        text="text",
        x_offset="x_offset",
        y_offset="y_offset",
        source=source,
        background_fill_color="#172026",
        background_fill_alpha=0.92,
        border_line_color="#394854",
        border_line_alpha=0.95,
        border_line_width=1,
        text_color="#f5fbff",
        text_font_size="10px",
        text_line_height=1.15,
        text_baseline="top",
    )


def _image_figure(
    title: str,
    source: ColumnDataSource,
    mapper: LinearColorMapper,
    pixel_selection_source: ColumnDataSource,
    label_source: ColumnDataSource,
    *,
    shared_x_range: Range1d,
    shared_y_range: Range1d,
    stamp_width: int,
    stamp_height: int,
):
    fig = figure(
        title=title,
        width=248,
        height=248,
        match_aspect=True,
        x_range=shared_x_range,
        y_range=shared_y_range,
        tools="tap,pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        min_border=2,
    )
    fig.image(
        image="image",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        source=source,
        color_mapper=mapper,
    )
    fig.add_layout(_color_bar(mapper), "right")
    fig.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=pixel_selection_source,
        fill_alpha=0.0,
        line_color="#ffd400",
        line_width=2,
    )
    fig.add_layout(_pixel_label_set(label_source))
    fig.add_layout(
        Span(
            location=stamp_width / 2.0,
            dimension="height",
            line_color="#03c7b7",
            line_alpha=0.75,
            line_width=1,
        )
    )
    fig.add_layout(
        Span(
            location=stamp_height / 2.0,
            dimension="width",
            line_color="#03c7b7",
            line_alpha=0.75,
            line_width=1,
        )
    )
    wheel = fig.select_one(WheelZoomTool)
    if wheel is not None:
        fig.toolbar.active_scroll = wheel
    fig.grid.visible = False
    fig.axis.visible = False
    fig.outline_line_color = "#d9e2f0"
    fig.border_fill_color = "#ffffff"
    fig.background_fill_color = "#ffffff"
    fig.title.text_color = "#07145c"
    fig.title.text_font_size = "11px"
    fig.title.text_font_style = "bold"
    fig.toolbar.logo = None
    return fig


def _color_bar(mapper: LinearColorMapper) -> ColorBar:
    return ColorBar(
        color_mapper=mapper,
        width=8,
        label_standoff=4,
        border_line_color=None,
        background_fill_alpha=0.0,
        major_label_text_font_size="8px",
    )


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

function pixelValue(source) {
  const flatColumn = source.data.flat;
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

function robustSigma(source, value) {
  const center = scalar(source, "robust_center");
  const scale = scalar(source, "robust_sigma");
  if (!Number.isFinite(center) || !Number.isFinite(scale) || scale <= 0) {
    return NaN;
  }
  return (Number(value) - center) / scale;
}

const xOffset = col > stamp_width * 0.58 ? -150 : 10;
const yOffset = row > stamp_height * 0.58 ? -78 : 10;
for (let index = 0; index < labels.length; index += 1) {
  const value = pixelValue(sources[index]);
  const sigma = robustSigma(sources[index], value);
  const mapper = mappers[index];
  label_sources[index].data = {
    x: [col + 0.5],
    y: [row + 0.5],
    x_offset: [xOffset],
    y_offset: [yOffset],
    text: [
      `${labels[index]}\n`
      + `(x,y): ${col}, ${row}\n`
      + `value: ${formatValue(value)}\n`
      + `robust sigma: ${formatValue(sigma)}\n`
      + `stretch: ${formatValue(mapper.low)} .. ${formatValue(mapper.high)}`,
    ],
  };
  label_sources[index].change.emit();
}
""",
    )


def _asset_data_uri(path: Path, mime_type: str) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _review_logo_html(size_px: int = 76) -> str:
    logo_src = _asset_data_uri(LOGO_PATH, "image/png")
    if not logo_src:
        return ""
    escaped_logo = html.escape(logo_src, quote=True)
    return (
        "<img alt='cuPhoton logo' "
        f"src='{escaped_logo}' "
        "style='display:block;flex:0 0 auto;"
        f"height:{size_px}px;width:{size_px}px;"
        "object-fit:contain;' />"
    )


def _short_review_identifier(value: Any, *, keep: int = 12) -> str:
    text = str(value or "")
    if len(text) <= keep * 2 + 3:
        return text
    return f"{text[:keep]}...{text[-keep:]}"


def _review_header_html(
    *,
    title: str,
    subtitle: str,
    id_label: str,
    item: dict[str, Any],
) -> str:
    identifier = item.get("candidate_id") or item.get("queue_id", "")
    return (
        "<div style='align-items:center;display:flex;gap:16px;"
        "min-width:320px;'>"
        f"{_review_logo_html()}"
        "<div style='min-width:0;'>"
        "<div style='color:#07145c;font-size:34px;font-weight:760;"
        "line-height:1.06;'>"
        f"{html.escape(title)}</div>"
        "<div style='color:#7080b5;font-size:15px;margin-top:7px;'>"
        f"{html.escape(subtitle)}</div>"
        "<div style='align-items:center;background:#eefaff;"
        "border:1px solid #d5edf7;border-radius:999px;display:inline-flex;"
        "gap:10px;margin-top:12px;max-width:100%;padding:8px 14px;'>"
        "<span style='color:#7080b5;font-size:12px;font-weight:700;'>"
        f"{html.escape(id_label)}</span>"
        "<strong style='color:#07145c;font-family:ui-monospace,"
        "SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;"
        "overflow-wrap:anywhere;'>"
        f"{html.escape(_short_review_identifier(identifier))}"
        "</strong>"
        "</div>"
        "</div>"
        "</div>"
    )


def _progress_html(queue_position: int | None, queue_count: int) -> str:
    if queue_position is None or queue_count <= 0:
        label = "-- / --"
        percent = 0
    else:
        current = queue_position + 1
        label = f"{current} / {queue_count}"
        percent = int(round((current / queue_count) * 100))
        percent = min(max(percent, 0), 100)
    return (
        "<div style='color:#07145c;font-size:18px;font-weight:760;"
        "min-width:92px;text-align:center;'>"
        f"{html.escape(label)}</div>"
        "<div style='background:#e7ebf3;border-radius:999px;height:6px;"
        "margin:8px auto 0;overflow:hidden;width:88px;'>"
        "<div style='background:#03c7b7;border-radius:inherit;height:100%;"
        f"width:{percent}%;'></div>"
        "</div>"
    )


def _summary_field_html(label: str, value: Any) -> str:
    field_style = (
        "border-top:1px solid #edf1f7;display:grid;gap:16px;"
        "grid-template-columns:minmax(132px,38%) minmax(0,1fr);"
        "padding:8px 0;"
    )
    key_style = "color:#7080b5;font-weight:650;"
    value_style = "color:#07145c;min-width:0;overflow-wrap:anywhere;"
    return (
        f"<div class='summary-field' style='{field_style}'>"
        f"<div class='summary-key' style='{key_style}'>"
        f"{html.escape(label)}</div>"
        f"<div class='summary-value' style='{value_style}'>"
        f"{_fmt(value)}</div>"
        "</div>"
    )


def _metadata_html(
    *,
    sample_index: int,
    item: dict[str, Any],
    queue_position: int | None,
    queue_count: int,
    arrays: dict[str, np.ndarray],
    scales: dict[tuple[str, str], tuple[float, float]],
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
        ("Rank Reason", item.get("rank_reason")),
        ("Known Error", item.get("known_error")),
        ("Label Source", item.get("label_source")),
        ("Center Source", item.get("center_source")),
        ("Alard-Lupton Mode", item.get("difference_mode")),
        ("SNR", item.get("snr")),
        ("Alard-Lupton SNR", item.get("diff_snr")),
        ("x", item.get("x")),
        ("y", item.get("y")),
    ]
    fields.extend(
        [
            ("corr(Alard-Lupton, raw)", stats["corr"]),
            ("mean |Alard-Lupton - raw|", stats["mean_abs_residual"]),
            ("rms(Alard-Lupton - raw)", stats["rms_residual"]),
            ("center search", stats["center_search"]),
            ("center template", stats["center_template"]),
            ("center raw", stats["center_raw_diff"]),
            ("center Alard-Lupton", stats["center_stored_diff"]),
            ("center Alard-Lupton - raw", stats["center_residual"]),
        ]
    )
    rows = []
    for label, value in fields:
        if value is None:
            continue
        rows.append(_summary_field_html(label, value))

    scale_rows = []
    for key, label, _ in PANEL_COLUMNS:
        full = scales[("full", key)]
        review = scales[("review", key)]
        scale_rows.append(
            "<tr>"
            "<th style='border-bottom:1px solid #e3eaf5;color:#51617f;"
            "font-weight:760;padding:6px 8px;text-align:left;'>"
            f"{html.escape(label)}</th>"
            "<td style='border-bottom:1px solid #e3eaf5;padding:6px 8px;"
            f"text-align:left;'>{_fmt(full[0])} .. {_fmt(full[1])}</td>"
            "<td style='border-bottom:1px solid #e3eaf5;padding:6px 8px;"
            f"text-align:left;'>{_fmt(review[0])} .. {_fmt(review[1])}</td>"
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
        "padding:18px;width:1400px;"
    )
    heading_style = (
        "color:#07145c;font-size:18px;font-weight:760;margin:0 0 8px;"
    )
    grid_style = (
        "column-gap:32px;color:#07145c;display:grid;font-size:13px;"
        "grid-template-columns:repeat(2,minmax(0,1fr));width:100%;"
    )
    return (
        f"<div class='summary-card' style='{card_style}'>"
        f"<h2 style='{heading_style}'>Detection Summary</h2>"
        f"<div class='summary-grid' style='{grid_style}'>"
        + "".join(rows)
        + "</div>"
        "<h2 style='color:#07145c;font-size:18px;font-weight:760;"
        "margin:18px 0 8px;'>Display Stretch Ranges</h2>"
        "<table style='border-collapse:collapse;color:#07145c;font-size:12px;"
        "margin-bottom:8px;width:100%;'><thead><tr>"
        "<th style='border-bottom:1px solid #e3eaf5;color:#51617f;"
        "font-weight:760;padding:6px 8px;text-align:left;'>Pane</th>"
        "<th style='border-bottom:1px solid #e3eaf5;color:#51617f;"
        "font-weight:760;padding:6px 8px;text-align:left;'>"
        "Native Stretch</th>"
        "<th style='border-bottom:1px solid #e3eaf5;color:#51617f;"
        "font-weight:760;padding:6px 8px;text-align:left;'>"
        "Percentile Stretch</th>"
        "</tr></thead><tbody>" + "".join(scale_rows) + "</tbody></table>"
        "<div style='color:#51617f;font-size:13px;'>Alard-Lupton source: "
        f"{diff_source}</div>"
        "</div>"
    )


def _array_stats(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    raw = arrays["raw_diff"]
    stored = arrays["stored_diff"]
    residual = arrays["residual"]
    mask = np.isfinite(raw) & np.isfinite(stored)
    if np.count_nonzero(mask) >= 2:
        corr = float(
            np.corrcoef(raw[mask].ravel(), stored[mask].ravel())[0, 1]
        )
    else:
        corr = math.nan
    center_y = raw.shape[0] // 2
    center_x = raw.shape[1] // 2
    return {
        "corr": corr,
        "mean_abs_residual": float(np.nanmean(np.abs(residual))),
        "rms_residual": float(np.sqrt(np.nanmean(residual**2))),
        "center_search": float(arrays["search"][center_y, center_x]),
        "center_template": float(arrays["template"][center_y, center_x]),
        "center_raw_diff": float(raw[center_y, center_x]),
        "center_stored_diff": float(stored[center_y, center_x]),
        "center_residual": float(residual[center_y, center_x]),
    }


def _status_html(message: str, *, error: bool = False) -> str:
    if error:
        return (
            "<div style='background:#fff1f1;border:1px solid #ffd1d1;"
            "border-radius:6px;color:#8a1f1f;font-size:13px;margin:2px 0;"
            f"padding:8px 10px;'>{html.escape(message)}</div>"
        )
    return (
        "<div style='background:#eef4ff;border:1px solid #d6e5ff;"
        "border-radius:6px;color:#24436f;font-size:13px;margin:2px 0;"
        f"padding:8px 10px;'>{html.escape(message)}</div>"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if abs(value) >= 1000 or (0 < abs(value) < 0.0001):
            return f"{value:.4e}"
        return f"{value:.4f}"
    return html.escape(str(value))


def _apply_style(doc) -> None:
    from bokeh.core.templates import get_env

    css = """
:root {
  --review-bg: #f7fbff;
  --review-surface: #ffffff;
  --review-text: #07145c;
  --review-muted: #7080b5;
  --review-border: #dde6f4;
  --review-teal: #03c7b7;
  --review-indigo: #5d4df2;
  --review-shadow: 0 14px 38px rgba(42, 55, 105, 0.12);
}
html, body {
  min-height: 100%;
  margin: 0;
  background: var(--review-bg);
  color: var(--review-text);
  font-family: "Inter", "IBM Plex Sans", "Avenir Next", "Segoe UI",
    sans-serif;
}
.review-shell {
  width: min(1440px, calc(100vw - 36px));
  margin: 18px auto 34px;
  gap: 12px !important;
}
.review-header,
.review-card,
.summary-wrap {
  width: 100%;
}
.triptych-row {
  gap: 12px !important;
  justify-content: space-between;
}
.status-card {
  color: var(--review-muted);
  font-size: 13px;
}
.bk-btn-primary {
  background-color: var(--review-indigo) !important;
  border-color: var(--review-indigo) !important;
}
.bk-input,
.bk-input-group,
.bk-select,
.bk-textarea {
  color: var(--review-text);
}
"""
    doc.template = get_env().from_string(
        "{% extends base %}\n"
        "{% block preamble %}\n"
        "<style>\n"
        f"{css}\n"
        "</style>\n"
        "{% endblock %}"
    )
