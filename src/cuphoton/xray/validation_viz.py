# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TraceRecord:
    path: Path
    label: str
    row_y: int | None
    time: np.ndarray
    trace: np.ndarray
    summary: dict[str, Any] | str | None


@dataclass(frozen=True)
class TraceFit:
    label: str
    row_y: int | None
    reconstruction: np.ndarray
    residual: np.ndarray
    frequency: np.ndarray
    spectrum_components: np.ndarray
    spectrum_total: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency_centers: np.ndarray
    chi2: float
    rms_residual: float
    selected_model_order: int
    decaying_root_count: int
    mode_count: int


@dataclass(frozen=True)
class ValidationVizResult:
    html_path: Path
    trace_count: int
    fit_count: int
    profile_log_count: int


_HTML_TEMPLATE = """
{% block postamble %}
<style>
  :root {
    --xray-bg: #f6f8fb;
    --xray-surface: #ffffff;
    --xray-soft: #eef4f8;
    --xray-text: #142033;
    --xray-muted: #637083;
    --xray-border: #dce5ee;
    --xray-blue: #3767a3;
    --xray-teal: #0aa6a6;
    --xray-red: #d95f4f;
    --xray-amber: #d18d2f;
  }
  html, body {
    background: var(--xray-bg);
    color: var(--xray-text);
    font-family: Inter, "IBM Plex Sans", "Segoe UI", sans-serif;
    margin: 0;
    padding: 0;
  }
  .xray-shell {
    margin: 0 auto;
    max-width: 1480px;
    padding: 20px 18px 42px;
  }
  .xray-hero,
  .xray-card {
    background: var(--xray-surface);
    border: 1px solid var(--xray-border);
    border-radius: 8px;
    box-shadow: 0 12px 34px rgba(41, 56, 78, 0.10);
    margin-bottom: 16px;
    padding: 18px;
  }
  .xray-hero h1 {
    font-size: 30px;
    line-height: 1.12;
    margin: 0 0 8px;
  }
  .xray-subtitle {
    color: var(--xray-muted);
    font-size: 14px;
    line-height: 1.45;
    margin: 0;
  }
  .xray-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
  }
  .xray-pill {
    background: var(--xray-soft);
    border: 1px solid var(--xray-border);
    border-radius: 999px;
    color: var(--xray-text);
    display: inline-block;
    font-weight: 650;
    padding: 7px 11px;
  }
  .xray-pill strong {
    color: var(--xray-muted);
    font-size: 12px;
    margin-right: 6px;
  }
  .xray-section-title {
    color: var(--xray-text);
    font-size: 18px;
    font-weight: 760;
    margin: 0 0 10px;
  }
  .xray-card table {
    border-collapse: collapse;
    width: 100%;
  }
  .xray-card th,
  .xray-card td {
    border-bottom: 1px solid #edf1f5;
    padding: 7px 10px;
    text-align: left;
    vertical-align: top;
  }
  .xray-card th {
    color: var(--xray-muted);
    font-size: 12px;
    width: 210px;
  }
  .bk-root .bk-Row {
    gap: 12px !important;
  }
  @media (max-width: 1100px) {
    .bk-root .bk-Row {
      flex-direction: column !important;
    }
  }
</style>
{% endblock %}
{% block contents %}
<div class="xray-shell">
  {{ plot_div | indent(2) }}
</div>
{% endblock %}
"""


def build_validation_viz(
    *,
    output: Path | str,
    trace_paths: tuple[Path, ...] = (),
    trace_dir: Path | None = None,
    profile_logs: tuple[Path, ...] = (),
    title: str = "XRay Validation Review",
    components: int = 30,
    roots_backend: str = "eigvals",
    max_traces: int = 16,
    fit: bool = True,
) -> ValidationVizResult:
    """Write a standalone interactive HTML validation dashboard."""

    try:
        from bokeh.embed import file_html
        from bokeh.layouts import column, row
        from bokeh.models import Div
        from bokeh.resources import INLINE
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bokeh is required for validation-viz; run "
            "'uv sync --extra viz' for development or install "
            "'cuphoton[viz]'"
        ) from exc

    records = _load_trace_records(
        trace_paths=trace_paths,
        trace_dir=trace_dir,
        max_traces=max_traces,
    )
    if not records:
        raise ValueError("provide --trace-npz or --trace-dir")

    fits = (
        _fit_traces(
            records,
            components=components,
            roots_backend=roots_backend,
        )
        if fit
        else ()
    )

    profile_log_count = len(profile_logs)
    layout_items = [
        Div(
            text=_hero_html(
                title=title,
                records=records,
                fits=fits,
                components=components if fit else None,
                roots_backend=roots_backend if fit else None,
                profile_log_count=profile_log_count,
            ),
            sizing_mode="stretch_width",
        ),
        Div(
            text=_legacy_plot_coverage_html(records, fits),
            sizing_mode="stretch_width",
        ),
        row(
            _trace_matrix_figure(records),
            _trace_overlay_figure(records, fits),
            sizing_mode="stretch_width",
        ),
        row(
            _waterfall_figure(records),
            _row_activity_figure(records, fits),
            sizing_mode="stretch_width",
        ),
    ]

    if fits:
        layout_items.extend(
            [
                row(
                    _residual_matrix_figure(records, fits),
                    _fit_metrics_figure(fits),
                    sizing_mode="stretch_width",
                ),
                row(
                    _frequency_centers_figure(fits),
                    _amplitude_phase_figure(fits),
                    sizing_mode="stretch_width",
                ),
                _reconstruction_lineouts_figure(records, fits),
                _spectrum_figure(fits),
            ]
        )

    profile_panel = _profile_panel(profile_logs)
    if profile_panel is not None:
        layout_items.append(profile_panel)

    layout_items.append(
        Div(
            text=_metadata_html(records),
            sizing_mode="stretch_width",
        )
    )

    layout = column(*layout_items, sizing_mode="stretch_width")
    html_path = Path(output)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        file_html(layout, INLINE, title, template=_HTML_TEMPLATE),
        encoding="utf-8",
    )
    return ValidationVizResult(
        html_path=html_path,
        trace_count=len(records),
        fit_count=len(fits),
        profile_log_count=profile_log_count,
    )


def _load_trace_records(
    *,
    trace_paths: tuple[Path, ...],
    trace_dir: Path | None,
    max_traces: int,
) -> tuple[TraceRecord, ...]:
    if trace_dir is not None:
        if trace_paths:
            raise ValueError(
                "--trace-dir cannot be combined with --trace-npz"
            )
        if not trace_dir.is_dir():
            raise ValueError("--trace-dir must be a directory")
        trace_paths = tuple(
            sorted(trace_dir.glob("*.npz"), key=_trace_npz_sort_key)
        )
    if max_traces <= 0:
        raise ValueError("--max-traces must be positive")

    records = tuple(_load_trace_record(path) for path in trace_paths)
    records = records[:max_traces]
    if not records:
        return ()

    reference_time = records[0].time
    for record in records[1:]:
        if record.time.shape != reference_time.shape or not np.allclose(
            record.time,
            reference_time,
        ):
            raise ValueError(
                "all trace NPZ files must use the same time array"
            )
    return records


def _load_trace_record(path: Path) -> TraceRecord:
    trace_path = Path(path)
    with np.load(trace_path, allow_pickle=False) as loaded:
        if "time" not in loaded or "trace" not in loaded:
            raise ValueError(
                "trace NPZ must contain 'time' and 'trace' arrays"
            )
        time = np.asarray(loaded["time"], dtype=float)
        trace = np.asarray(loaded["trace"], dtype=float)
        summary = _load_npz_summary(loaded)

    if time.ndim != 1 or trace.ndim != 1:
        raise ValueError("trace NPZ time and trace arrays must be 1D")
    if time.shape != trace.shape:
        raise ValueError("trace NPZ time and trace arrays must match")

    row_y = _summary_row_y(summary)
    label = f"y={row_y}" if row_y is not None else trace_path.stem
    return TraceRecord(
        path=trace_path,
        label=label,
        row_y=row_y,
        time=time,
        trace=trace,
        summary=summary,
    )


def _load_npz_summary(loaded) -> dict[str, Any] | str | None:
    if "summary" not in loaded:
        return None
    summary = np.asarray(loaded["summary"])
    raw = (
        str(summary.item())
        if summary.shape == ()
        else json.dumps(summary.tolist())
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return parsed


def _summary_row_y(summary: dict[str, Any] | str | None) -> int | None:
    if isinstance(summary, dict) and summary.get("row_y") is not None:
        return int(summary["row_y"])
    return None


def _trace_npz_sort_key(path: Path):
    parts = re.split(r"(\d+)", Path(path).name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _public_path_label(
    value: Path | str | None,
    *,
    fallback: str = "local input",
) -> str:
    """Return a non-absolute label suitable for publishable artifacts."""

    if value is None:
        return fallback
    normalized = str(value).strip().replace("\\", "/").rstrip("/")
    label = normalized.rsplit("/", 1)[-1]
    if label in {"", ".", ".."}:
        return fallback
    return label


def _fit_traces(
    records: tuple[TraceRecord, ...],
    *,
    components: int,
    roots_backend: str,
) -> tuple[TraceFit, ...]:
    from .linear_prediction import linear_prediction_numpy

    if components <= 0:
        raise ValueError("--components must be positive")
    fits = []
    for record in records:
        result = linear_prediction_numpy(
            record.time,
            record.trace,
            components,
            roots_backend=roots_backend,
        )
        reconstruction = np.asarray(result.reconstruction, dtype=float)
        residual = reconstruction - record.trace
        fits.append(
            TraceFit(
                label=record.label,
                row_y=record.row_y,
                reconstruction=reconstruction,
                residual=residual,
                frequency=np.asarray(result.frequency, dtype=float),
                spectrum_components=np.asarray(
                    result.spectrum_components,
                    dtype=float,
                ),
                spectrum_total=np.asarray(
                    result.spectrum_total,
                    dtype=float,
                ),
                amplitude=np.asarray(result.amplitude, dtype=float),
                phase=np.asarray(result.phase, dtype=float),
                frequency_centers=_frequency_centers(result),
                chi2=float(result.chi2),
                rms_residual=float(
                    np.sqrt(np.mean(np.asarray(residual) ** 2))
                ),
                selected_model_order=int(result.selected_model_order),
                decaying_root_count=int(result.decaying_root_count),
                mode_count=int(len(result.angular_frequency)),
            )
        )
    return tuple(fits)


def _frequency_centers(result) -> np.ndarray:
    components = np.asarray(result.spectrum_components, dtype=float)
    frequency = np.asarray(result.frequency, dtype=float)
    if components.size == 0 or components.ndim != 2 or frequency.size == 0:
        return np.asarray([], dtype=float)
    return frequency[np.argmax(components, axis=0)]


def _hero_html(
    *,
    title: str,
    records: tuple[TraceRecord, ...],
    fits: tuple[TraceFit, ...],
    components: int | None,
    roots_backend: str | None,
    profile_log_count: int,
) -> str:
    time = records[0].time
    row_values = [
        record.row_y for record in records if record.row_y is not None
    ]
    row_text = "-"
    if row_values:
        row_text = f"{min(row_values)}:{max(row_values)}"
    pills = [
        ("Traces", len(records)),
        ("Samples", len(time)),
        ("Delay", f"{float(np.min(time)):.3g}:{float(np.max(time)):.3g}"),
        ("Rows", row_text),
        ("Fits", len(fits)),
        ("Profile logs", profile_log_count),
    ]
    if components is not None:
        pills.append(("Components", components))
    if roots_backend is not None:
        pills.append(("Roots", roots_backend))

    hero_style = (
        "background:#ffffff;border:1px solid #dce5ee;border-radius:8px;"
        "box-shadow:0 12px 34px rgba(41,56,78,0.10);"
        "margin-bottom:16px;padding:18px;"
    )
    h1_style = "font-size:30px;line-height:1.12;margin:0 0 8px;"
    subtitle_style = "color:#637083;font-size:14px;line-height:1.45;margin:0;"
    pills_style = "display:flex;flex-wrap:wrap;gap:10px;margin-top:14px;"
    pill_style = (
        "background:#eef4f8;border:1px solid #dce5ee;border-radius:999px;"
        "color:#142033;display:inline-flex;align-items:center;"
        "column-gap:6px;font-weight:650;padding:7px 11px;"
    )
    label_style = "color:#637083;font-size:12px;"
    pill_html = "".join(
        f"<span class='xray-pill' style='{pill_style}'>"
        f"<strong style='{label_style}'>{html.escape(str(key))}</strong>"
        f"{html.escape(str(value))}"
        "</span>"
        for key, value in pills
    )
    return (
        f"<div class='xray-hero' style='{hero_style}'>"
        f"<h1 style='{h1_style}'>{html.escape(title)}</h1>"
        f"<p class='xray-subtitle' style='{subtitle_style}'>"
        "Human validation surface for extracted XRay traces, fitted "
        "reconstructions, LPF mode metrics, residuals, spectra, and "
        "profile timing."
        "</p>"
        f"<div class='xray-pills' style='{pills_style}'>{pill_html}</div>"
        "</div>"
    )


def _legacy_plot_coverage_html(
    records: tuple[TraceRecord, ...],
    fits: tuple[TraceFit, ...],
) -> str:
    first_summary = records[0].summary
    has_normalization = any(
        _has_normalization_metadata(record) for record in records
    )
    has_rows = any(record.row_y is not None for record in records)
    fit_status = "represented" if fits else "fit disabled"
    normalization_status = (
        "represented" if has_normalization else "not in NPZ"
    )
    row_status = "represented" if has_rows else "trace index only"
    rows = [
        (
            "I0/on-off normalization",
            normalization_status,
            "on/off normalization dataset, normalized-sum ranges, and shift "
            "metadata from the trace NPZ summaries.",
        ),
        (
            "Full-detector ROI images",
            "artifact gap",
            "current trace NPZs provide row traces only; full detector views "
            "need freq_all, amp_all, fft_all, or equivalent detector arrays.",
        ),
        (
            "LPF frequency/phase/amplitude/chi2",
            fit_status,
            "mode frequency centers, amplitude, phase, chi2, model order, "
            "and mode counts by detector row.",
        ),
        (
            "Reconstruction overlays",
            fit_status,
            "raw, reconstructed, and residual delay lineouts with row "
            "offsets.",
        ),
        (
            "Phonon dispersion x-slices",
            row_status,
            "frequency-center scatter over detector rows, colored and sized "
            "by mode amplitude, plus spectrum overlays.",
        ),
        (
            "Waterfall traces",
            "represented",
            "offset delay traces for fast scanning of row-to-row structure.",
        ),
    ]
    if isinstance(first_summary, dict) and first_summary.get("kind"):
        rows.insert(
            0,
            (
                "Source artifact kind",
                str(first_summary["kind"]),
                "dashboard generated from extracted trace artifacts.",
            ),
        )

    body = "".join(
        "<tr>"
        f"<th>{html.escape(name)}</th>"
        f"<td><span class='xray-status'>{html.escape(status)}</span></td>"
        f"<td>{html.escape(detail)}</td>"
        "</tr>"
        for name, status, detail in rows
    )
    return (
        "<div class='xray-card'>"
        "<h2 class='xray-section-title'>Reference Plot Coverage</h2>"
        f"<table>{body}</table>"
        "</div>"
    )


def _has_normalization_metadata(record: TraceRecord) -> bool:
    if not isinstance(record.summary, dict):
        return False
    for branch in ("on", "off"):
        branch_summary = record.summary.get(branch)
        if isinstance(branch_summary, dict) and (
            "normalization_dataset" in branch_summary
            or "normalized_sum_mean" in branch_summary
        ):
            return True
    return False


def _trace_matrix_figure(records: tuple[TraceRecord, ...]):
    from bokeh.models import ColorBar, LinearColorMapper, Range1d
    from bokeh.palettes import Viridis256
    from bokeh.plotting import figure

    image = np.stack([record.trace for record in records])
    low, high = _finite_bounds(image)
    mapper = LinearColorMapper(palette=list(Viridis256), low=low, high=high)
    time = records[0].time
    plot = figure(
        title="Trace matrix",
        width=700,
        height=360,
        sizing_mode="stretch_width",
        x_axis_label="delay",
        y_axis_label="trace index",
        x_range=Range1d(float(np.min(time)), float(np.max(time))),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.image(
        image=[image],
        x=float(np.min(time)),
        y=-0.5,
        dw=max(float(np.max(time) - np.min(time)), 1.0),
        dh=float(len(records)),
        color_mapper=mapper,
    )
    plot.add_layout(ColorBar(color_mapper=mapper, width=10), "right")
    _style_plot(plot)
    return plot


def _trace_overlay_figure(
    records: tuple[TraceRecord, ...],
    fits: tuple[TraceFit, ...],
):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    time = records[0].time
    source = ColumnDataSource(
        {
            "xs": [time for _record in records],
            "ys": [record.trace for record in records],
            "label": [record.label for record in records],
            "path": [_public_path_label(record.path) for record in records],
        }
    )
    plot = figure(
        title="Trace and reconstruction overlays",
        width=700,
        height=360,
        sizing_mode="stretch_width",
        x_axis_label="delay",
        y_axis_label="ratio minus one",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    raw_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        line_color="#3767a3",
        line_alpha=0.44,
        line_width=1.6,
    )
    renderers = [raw_renderer]
    if fits:
        fit_source = ColumnDataSource(
            {
                "xs": [time for _fit in fits],
                "ys": [fit.reconstruction for fit in fits],
                "label": [fit.label for fit in fits],
                "path": [record.path.name for record in records[: len(fits)]],
            }
        )
        fit_renderer = plot.multi_line(
            xs="xs",
            ys="ys",
            source=fit_source,
            line_color="#d95f4f",
            line_alpha=0.58,
            line_width=1.4,
            line_dash="dashed",
        )
        renderers.append(fit_renderer)
    plot.add_tools(
        HoverTool(
            renderers=renderers,
            tooltips=[("trace", "@label"), ("source", "@path")],
            line_policy="nearest",
        )
    )
    _style_plot(plot)
    return plot


def _waterfall_figure(records: tuple[TraceRecord, ...]):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    time = records[0].time
    selected = records[: min(len(records), 18)]
    ys = []
    for index, record in enumerate(selected):
        scaled, _scale = _lineout_scaled(record.trace)
        ys.append(scaled + index)
    source = ColumnDataSource(
        {
            "xs": [time for _record in selected],
            "ys": ys,
            "label": [record.label for record in selected],
            "path": [record.path.name for record in selected],
        }
    )
    plot = figure(
        title="Delay waterfall lineouts",
        width=700,
        height=340,
        sizing_mode="stretch_width",
        x_axis_label="delay",
        y_axis_label="row offset",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        line_color="#3767a3",
        line_alpha=0.76,
        line_width=1.35,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[("trace", "@label"), ("source", "@path")],
            line_policy="nearest",
        )
    )
    _style_plot(plot)
    return plot


def _row_activity_figure(
    records: tuple[TraceRecord, ...],
    fits: tuple[TraceFit, ...],
):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    x_values = _row_axis_values(records)
    fit_by_label = {fit.label: fit for fit in fits}
    trace_span = [
        float(np.nanmax(record.trace) - np.nanmin(record.trace))
        for record in records
    ]
    ratio_mean = [
        _summary_float(record.summary, "ratio_mean") for record in records
    ]
    rms_residual = [
        (
            fit_by_label[record.label].rms_residual
            if record.label in fit_by_label
            else np.nan
        )
        for record in records
    ]
    source = ColumnDataSource(
        {
            "row": x_values,
            "label": [record.label for record in records],
            "trace_span": trace_span,
            "ratio_mean": ratio_mean,
            "rms_residual": rms_residual,
        }
    )
    plot = figure(
        title="ROI row activity proxy",
        width=700,
        height=340,
        sizing_mode="stretch_width",
        x_axis_label=_row_axis_label(records),
        y_axis_label="trace span",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    bars = plot.vbar(
        x="row",
        top="trace_span",
        width=_row_bar_width(x_values),
        source=source,
        fill_color="#0aa6a6",
        line_color="#087878",
        alpha=0.72,
    )
    residuals = plot.scatter(
        x="row",
        y="rms_residual",
        source=source,
        marker="circle",
        size=8,
        color="#d95f4f",
        alpha=0.82,
    )
    plot.add_tools(
        HoverTool(
            renderers=[bars, residuals],
            tooltips=[
                ("trace", "@label"),
                ("span", "@trace_span{0.000e}"),
                ("ratio mean", "@ratio_mean{0.000e}"),
                ("rms residual", "@rms_residual{0.000e}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _residual_matrix_figure(
    records: tuple[TraceRecord, ...],
    fits: tuple[TraceFit, ...],
):
    from bokeh.models import ColorBar, LinearColorMapper, Range1d
    from bokeh.palettes import RdBu11
    from bokeh.plotting import figure

    residuals = np.stack([fit.residual for fit in fits])
    bound = max(abs(value) for value in _finite_bounds(residuals))
    bound = max(bound, 1.0e-12)
    mapper = LinearColorMapper(
        palette=list(reversed(RdBu11)),
        low=-bound,
        high=bound,
    )
    time = records[0].time
    plot = figure(
        title="Fit residual matrix",
        width=700,
        height=350,
        sizing_mode="stretch_width",
        x_axis_label="delay",
        y_axis_label="trace index",
        x_range=Range1d(float(np.min(time)), float(np.max(time))),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.image(
        image=[residuals],
        x=float(np.min(time)),
        y=-0.5,
        dw=max(float(np.max(time) - np.min(time)), 1.0),
        dh=float(len(fits)),
        color_mapper=mapper,
    )
    plot.add_layout(ColorBar(color_mapper=mapper, width=10), "right")
    _style_plot(plot)
    return plot


def _fit_metrics_figure(fits: tuple[TraceFit, ...]):
    from bokeh.models import (
        ColumnDataSource,
        FactorRange,
        HoverTool,
        LinearAxis,
        Range1d,
    )
    from bokeh.plotting import figure

    labels = [fit.label for fit in fits]
    source = ColumnDataSource(
        {
            "label": labels,
            "rms_residual": [fit.rms_residual for fit in fits],
            "chi2": [fit.chi2 for fit in fits],
            "selected_model_order": [
                fit.selected_model_order for fit in fits
            ],
            "mode_count": [fit.mode_count for fit in fits],
            "decaying_root_count": [fit.decaying_root_count for fit in fits],
        }
    )
    plot = figure(
        title="LPF fit quality by trace",
        width=700,
        height=350,
        sizing_mode="stretch_width",
        x_range=FactorRange(factors=labels),
        y_axis_label="RMS residual",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.vbar(
        x="label",
        top="rms_residual",
        width=0.72,
        source=source,
        fill_color="#0aa6a6",
        line_color="#087878",
        alpha=0.82,
    )
    order_high = max(
        max(fit.selected_model_order for fit in fits) * 1.15,
        1.0,
    )
    plot.extra_y_ranges = {"order": Range1d(start=0, end=order_high)}
    plot.add_layout(
        LinearAxis(y_range_name="order", axis_label="selected model order"),
        "right",
    )
    plot.scatter(
        x="label",
        y="selected_model_order",
        source=source,
        size=8,
        color="#d18d2f",
        y_range_name="order",
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("trace", "@label"),
                ("rms", "@rms_residual{0.000e}"),
                ("chi2", "@chi2{0.000e}"),
                ("model order", "@selected_model_order"),
                ("modes", "@mode_count"),
                ("decaying roots", "@decaying_root_count"),
            ],
        )
    )
    plot.xaxis.major_label_orientation = 0.9
    _style_plot(plot)
    return plot


def _frequency_centers_figure(fits: tuple[TraceFit, ...]):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    source_data = _mode_source_data(fits)
    source = ColumnDataSource(source_data)
    plot = figure(
        title="Frequency centers by row",
        width=700,
        height=350,
        sizing_mode="stretch_width",
        x_axis_label=_fit_row_axis_label(fits),
        y_axis_label="frequency center",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.scatter(
        x="row",
        y="frequency_center",
        source=source,
        marker="circle",
        size="size",
        fill_color="color",
        line_color="#24364d",
        line_alpha=0.55,
        fill_alpha=0.76,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("trace", "@label"),
                ("mode", "@mode"),
                ("frequency", "@frequency_center{0.00000}"),
                ("amplitude", "@amplitude{0.000e}"),
                ("phase", "@phase{0.000}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _amplitude_phase_figure(fits: tuple[TraceFit, ...]):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    source_data = _mode_source_data(fits)
    source = ColumnDataSource(source_data)
    plot = figure(
        title="LPF amplitude and phase by row",
        width=700,
        height=350,
        sizing_mode="stretch_width",
        x_axis_label=_fit_row_axis_label(fits),
        y_axis_label="amplitude",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.scatter(
        x="row",
        y="amplitude",
        source=source,
        marker="circle",
        size=8,
        fill_color="color",
        line_color="#24364d",
        line_alpha=0.55,
        fill_alpha=0.78,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("trace", "@label"),
                ("mode", "@mode"),
                ("amplitude", "@amplitude{0.000e}"),
                ("phase", "@phase{0.000}"),
                ("frequency", "@frequency_center{0.00000}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _reconstruction_lineouts_figure(
    records: tuple[TraceRecord, ...],
    fits: tuple[TraceFit, ...],
):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    time = records[0].time
    selected_pairs = list(zip(records, fits, strict=False))[:18]
    raw_ys = []
    fit_ys = []
    residual_ys = []
    for index, (record, fit) in enumerate(selected_pairs):
        raw_scaled, raw_scale = _lineout_scaled(record.trace)
        reconstruction = np.asarray(fit.reconstruction, dtype=float)
        reconstruction_scaled = (
            (reconstruction - np.nanmedian(record.trace)) / raw_scale * 0.42
        )
        residual_scaled = np.asarray(fit.residual, dtype=float) / raw_scale
        raw_ys.append(raw_scaled + index)
        fit_ys.append(reconstruction_scaled + index)
        residual_ys.append((residual_scaled * 0.42) + index)
    base_source = {
        "xs": [time for _pair in selected_pairs],
        "label": [record.label for record, _fit in selected_pairs],
        "path": [record.path.name for record, _fit in selected_pairs],
    }
    raw_source = ColumnDataSource({**base_source, "ys": raw_ys})
    fit_source = ColumnDataSource({**base_source, "ys": fit_ys})
    residual_source = ColumnDataSource({**base_source, "ys": residual_ys})
    plot = figure(
        title="Offset reconstruction lineouts",
        width=1400,
        height=390,
        sizing_mode="stretch_width",
        x_axis_label="delay",
        y_axis_label="row offset",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    raw_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=raw_source,
        line_color="#3767a3",
        line_alpha=0.55,
        line_width=1.2,
    )
    fit_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=fit_source,
        line_color="#142033",
        line_alpha=0.72,
        line_width=1.25,
        line_dash="dashed",
    )
    residual_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=residual_source,
        line_color="#d95f4f",
        line_alpha=0.78,
        line_width=1.05,
    )
    plot.add_tools(
        HoverTool(
            renderers=[raw_renderer, fit_renderer, residual_renderer],
            tooltips=[("trace", "@label"), ("source", "@path")],
            line_policy="nearest",
        )
    )
    _style_plot(plot)
    return plot


def _spectrum_figure(fits: tuple[TraceFit, ...]):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    selected = fits[: min(len(fits), 12)]
    source = ColumnDataSource(
        {
            "xs": [fit.frequency for fit in selected],
            "ys": [fit.spectrum_total for fit in selected],
            "label": [fit.label for fit in selected],
            "mode_count": [fit.mode_count for fit in selected],
        }
    )
    plot = figure(
        title="Spectrum totals",
        width=1400,
        height=340,
        sizing_mode="stretch_width",
        x_axis_label="frequency",
        y_axis_label="spectrum total",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        line_color="#3767a3",
        line_alpha=0.55,
        line_width=1.5,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[("trace", "@label"), ("modes", "@mode_count")],
            line_policy="nearest",
        )
    )
    _style_plot(plot)
    return plot


def _profile_panel(profile_logs: tuple[Path, ...]):
    if not profile_logs:
        return None

    from bokeh.models import ColumnDataSource, FactorRange, HoverTool
    from bokeh.plotting import figure

    from .profile_summary import summarize_linear_prediction_profile_files

    summary = summarize_linear_prediction_profile_files(profile_logs)
    stages = tuple(reversed(summary.stages))
    labels = [stage.stage for stage in stages]
    source = ColumnDataSource(
        {
            "stage": labels,
            "seconds": [stage.seconds for stage in stages],
            "count": [stage.count for stage in stages],
            "mean_s": [stage.mean_s for stage in stages],
        }
    )
    plot = figure(
        title="Profile stage time",
        width=1400,
        height=max(300, 32 * max(len(labels), 1)),
        sizing_mode="stretch_width",
        x_axis_label="seconds",
        y_range=FactorRange(factors=labels),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.hbar(
        y="stage",
        right="seconds",
        height=0.62,
        source=source,
        fill_color="#3767a3",
        line_color="#254a75",
        alpha=0.84,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("stage", "@stage"),
                ("seconds", "@seconds{0.000}"),
                ("count", "@count"),
                ("mean", "@mean_s{0.000}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _metadata_html(records: tuple[TraceRecord, ...]) -> str:
    first = records[0]
    rows = [
        ("First source", _public_path_label(first.path)),
        ("Trace count", len(records)),
        ("Samples", len(first.time)),
        ("Delay min", f"{float(np.min(first.time)):.6g}"),
        ("Delay max", f"{float(np.max(first.time)):.6g}"),
    ]
    if isinstance(first.summary, dict):
        for key, label in (
            ("h5dir", "Input directory"),
            ("fon", "On file"),
            ("foff", "Off file"),
        ):
            if key in first.summary:
                rows.append((label, _public_path_label(first.summary[key])))
        for key in (
            "roi_lower",
            "roi_dim",
            "exclude_y",
            "shift",
            "reference_shift",
            "ratio_min",
            "ratio_mean",
            "ratio_max",
        ):
            if key in first.summary:
                rows.append((key, first.summary[key]))
        for branch in ("on", "off"):
            branch_summary = first.summary.get(branch)
            if not isinstance(branch_summary, dict):
                continue
            for key in (
                "normalization_dataset",
                "normalized_sum_min",
                "normalized_sum_mean",
                "normalized_sum_max",
                "pixel_count",
                "schema",
            ):
                if key in branch_summary:
                    rows.append((f"{branch}.{key}", branch_summary[key]))

    body = "".join(
        "<tr>"
        f"<th>{html.escape(str(key))}</th>"
        f"<td>{html.escape(_display_value(value))}</td>"
        "</tr>"
        for key, value in rows
    )
    return (
        "<div class='xray-card'>"
        "<h2 class='xray-section-title'>Source Metadata</h2>"
        f"<table>{body}</table>"
        "</div>"
    )


def _display_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _row_axis_values(records: tuple[TraceRecord, ...]) -> list[float]:
    if all(record.row_y is not None for record in records):
        return [float(record.row_y) for record in records]
    return [float(index) for index, _record in enumerate(records)]


def _row_axis_label(records: tuple[TraceRecord, ...]) -> str:
    if all(record.row_y is not None for record in records):
        return "detector row y"
    return "trace index"


def _fit_row_axis_label(fits: tuple[TraceFit, ...]) -> str:
    if all(fit.row_y is not None for fit in fits):
        return "detector row y"
    return "trace index"


def _fit_row_value(fit: TraceFit, index: int) -> float:
    if fit.row_y is not None:
        return float(fit.row_y)
    return float(index)


def _row_bar_width(x_values: list[float]) -> float:
    if len(x_values) < 2:
        return 0.72
    unique_values = sorted(set(x_values))
    if len(unique_values) < 2:
        return 0.72
    spacing = min(
        right - left
        for left, right in zip(unique_values, unique_values[1:], strict=False)
    )
    return max(float(spacing) * 0.72, 0.72)


def _summary_float(summary: dict[str, Any] | str | None, key: str) -> float:
    if not isinstance(summary, dict) or key not in summary:
        return float("nan")
    try:
        return float(summary[key])
    except (TypeError, ValueError):
        return float("nan")


def _lineout_scaled(values: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=float)
    median = float(np.nanmedian(array))
    centered = array - median
    finite = centered[np.isfinite(centered)]
    scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    scale = max(scale, 1.0e-12)
    return centered / scale * 0.42, scale


def _mode_source_data(fits: tuple[TraceFit, ...]) -> dict[str, list[Any]]:
    rows: list[float] = []
    labels: list[str] = []
    modes: list[int] = []
    frequency_centers: list[float] = []
    amplitudes: list[float] = []
    phases: list[float] = []
    for fit_index, fit in enumerate(fits):
        count = min(
            len(fit.frequency_centers),
            len(fit.amplitude),
            len(fit.phase),
        )
        for mode_index in range(count):
            rows.append(_fit_row_value(fit, fit_index))
            labels.append(fit.label)
            modes.append(mode_index)
            frequency_centers.append(float(fit.frequency_centers[mode_index]))
            amplitudes.append(float(fit.amplitude[mode_index]))
            phases.append(float(fit.phase[mode_index]))

    colors = _palette_colors(amplitudes)
    sizes = _amplitude_sizes(amplitudes)
    return {
        "row": rows,
        "label": labels,
        "mode": modes,
        "frequency_center": frequency_centers,
        "amplitude": amplitudes,
        "phase": phases,
        "color": colors,
        "size": sizes,
    }


def _palette_colors(values: list[float]) -> list[str]:
    from bokeh.palettes import Viridis256

    if not values:
        return []
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [Viridis256[0] for _value in values]
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return [Viridis256[len(Viridis256) // 2] for _value in values]
    colors = []
    for value in values:
        if not np.isfinite(value):
            colors.append("#9aa5b1")
            continue
        fraction = (value - low) / (high - low)
        index = int(np.clip(round(fraction * (len(Viridis256) - 1)), 0, 255))
        colors.append(Viridis256[index])
    return colors


def _amplitude_sizes(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [7.0 for _value in values]
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return [8.0 for _value in values]
    sizes = []
    for value in values:
        if not np.isfinite(value):
            sizes.append(6.0)
            continue
        fraction = (value - low) / (high - low)
        sizes.append(float(6.0 + 10.0 * np.clip(fraction, 0.0, 1.0)))
    return sizes


def _finite_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high > low:
        return low, high
    pad = max(abs(high), 1.0) * 0.05
    return low - pad, high + pad


def _style_plot(plot) -> None:
    plot.background_fill_color = "#ffffff"
    plot.border_fill_color = "#ffffff"
    plot.outline_line_color = "#dce5ee"
    plot.grid.grid_line_color = "#e9eef5"
    plot.axis.axis_label_text_color = "#465468"
    plot.axis.major_label_text_color = "#465468"
    plot.title.text_color = "#142033"
    plot.title.text_font_size = "13px"
    plot.title.text_font_style = "bold"
    plot.toolbar.logo = None
    _prefer_box_zoom(plot)


def _prefer_box_zoom(plot) -> None:
    for tool in plot.tools:
        if tool.__class__.__name__ == "BoxZoomTool":
            plot.toolbar.active_drag = tool
            break
