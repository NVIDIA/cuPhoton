# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detector_artifacts import (
    detector_artifact_origin,
    detector_artifact_x_index,
    detector_artifact_y_slice,
)
from .validation_viz import (
    _fit_traces,
    _load_trace_records,
    _public_path_label,
)


@dataclass(frozen=True)
class PhononVizResult:
    html_path: Path
    source_kind: str
    trace_count: int
    mode_count: int


_HTML_TEMPLATE = """
{% block postamble %}
<style>
  :root {
    --xray-bg: #f6f8fb;
    --xray-surface: #ffffff;
    --xray-text: #142033;
    --xray-muted: #637083;
    --xray-border: #dce5ee;
    --xray-soft: #eef4f8;
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
  .xray-hero {
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
</style>
{% endblock %}
{% block contents %}
<div class="xray-shell">
  {{ plot_div | indent(2) }}
</div>
{% endblock %}
"""


def build_phonon_viz(
    *,
    output: Path | str,
    workflow_bundle: Path | None = None,
    trace_paths: tuple[Path, ...] = (),
    trace_dir: Path | None = None,
    detector_artifact_dir: Path | None = None,
    x_value: int | None = None,
    y_start: int | None = None,
    y_end: int | None = None,
    title: str = "XRay Phonon Dispersion",
    components: int = 30,
    roots_backend: str = "eigvals",
    max_traces: int = 256,
    max_points: int = 60_000,
    amp_threshold: float | None = None,
) -> PhononVizResult:
    try:
        from bokeh.embed import file_html
        from bokeh.layouts import column
        from bokeh.models import Div
        from bokeh.resources import INLINE
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bokeh is required for phonon-viz; run "
            "'uv sync --extra viz' for development or install "
            "'cuphoton[viz]'"
        ) from exc

    html_path = Path(output)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    if workflow_bundle is not None:
        if trace_paths or trace_dir is not None or detector_artifact_dir:
            raise ValueError(
                "--workflow-bundle cannot be combined with trace or "
                "detector artifact input"
            )
        source = _workflow_source(
            workflow_bundle,
            max_points=max_points,
        )
    elif detector_artifact_dir is not None:
        if trace_paths or trace_dir is not None:
            raise ValueError(
                "--detector-artifact-dir cannot be combined with trace input"
            )
        source = _detector_source(
            detector_artifact_dir,
            x_value=x_value,
            y_start=y_start,
            y_end=y_end,
            max_points=max_points,
            amp_threshold=amp_threshold,
        )
    else:
        if x_value is not None or y_start is not None or y_end is not None:
            raise ValueError(
                "--x-value/--y-start/--y-end require --detector-artifact-dir"
            )
        source = _trace_source(
            trace_paths=trace_paths,
            trace_dir=trace_dir,
            components=components,
            roots_backend=roots_backend,
            max_traces=max_traces,
        )

    layout = column(
        Div(
            text=_hero_html(
                title=title,
                source_kind=source["source_kind"],
                source_label=source["source_label"],
                row_count=source["row_count"],
                mode_count=source["mode_count"],
                x_value=source.get("x_value"),
                y_start=source.get("y_start"),
                y_end=source.get("y_end"),
                amp_threshold=source.get("amp_threshold"),
            ),
            sizing_mode="stretch_width",
        ),
        source["plot"],
        sizing_mode="stretch_width",
    )
    html_path.write_text(
        file_html(layout, INLINE, title, template=_HTML_TEMPLATE),
        encoding="utf-8",
    )
    return PhononVizResult(
        html_path=html_path,
        source_kind=str(source["source_kind"]),
        trace_count=int(source["row_count"]),
        mode_count=int(source["mode_count"]),
    )


def _workflow_source(
    workflow_bundle: Path,
    *,
    max_points: int,
) -> dict[str, Any]:
    from .workflow_viz import load_workflow_bundle

    workflow = load_workflow_bundle(workflow_bundle)
    return {
        "source_kind": "workflow bundle",
        "source_label": _public_path_label(workflow_bundle),
        "row_count": int(workflow.row_y.shape[0]),
        "mode_count": int(np.sum(workflow.mode_count)),
        "x_value": None,
        "y_start": None,
        "y_end": None,
        "plot": _workflow_phonon_figure(
            workflow,
            max_points=max_points,
        ),
    }


def _trace_source(
    *,
    trace_paths: tuple[Path, ...],
    trace_dir: Path | None,
    components: int,
    roots_backend: str,
    max_traces: int,
) -> dict[str, Any]:
    records = _load_trace_records(
        trace_paths=trace_paths,
        trace_dir=trace_dir,
        max_traces=max_traces,
    )
    if not records:
        raise ValueError(
            "provide --trace-npz, --trace-dir, or detector arrays"
        )
    fits = _fit_traces(
        records,
        components=components,
        roots_backend=roots_backend,
    )
    source_path = trace_dir if trace_dir is not None else records[0].path
    return {
        "source_kind": "trace-derived phonon proxy",
        "source_label": _public_path_label(source_path),
        "row_count": len(records),
        "mode_count": sum(fit.mode_count for fit in fits),
        "plot": _trace_phonon_figure(fits),
    }


def _detector_source(
    detector_artifact_dir: Path,
    *,
    x_value: int | None,
    y_start: int | None,
    y_end: int | None,
    max_points: int,
    amp_threshold: float | None,
) -> dict[str, Any]:
    root = Path(detector_artifact_dir)
    freq_all = _load_detector_array(root / "freq_all.npy")
    amp_all = _load_detector_array(root / "amp_all.npy")
    fft_all = _load_detector_array(root / "fft_all.npy")
    fft_freq_all = _load_detector_array(root / "fft_freq_all.npy")
    arrays = (freq_all, amp_all, fft_all, fft_freq_all)
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("detector artifact arrays must have matching shapes")
    if freq_all.ndim != 3:
        raise ValueError("detector artifact arrays must be 3D")
    height, width, _depth = freq_all.shape
    origin_x, origin_y = detector_artifact_origin(root)
    local_x, display_x = detector_artifact_x_index(
        width=width,
        x_value=x_value,
        origin_x=origin_x,
    )
    local_y_start, local_y_end, display_y_start, display_y_end = (
        detector_artifact_y_slice(
            height=height,
            y_start=y_start,
            y_end=y_end,
            origin_y=origin_y,
        )
    )
    freq_slice = freq_all[local_y_start:local_y_end, local_x, :]
    amp_slice = amp_all[local_y_start:local_y_end, local_x, :]
    if amp_threshold is None:
        mode_mask = amp_slice > 0
    else:
        mode_mask = (amp_slice > float(amp_threshold)) & (amp_slice < 1.0e6)
    mode_mask &= (
        np.isfinite(freq_slice) & np.isfinite(amp_slice) & (freq_slice > 0.0)
    )
    nonzero_modes = np.count_nonzero(mode_mask)
    source_kind = (
        "detector-wide filtered artifact"
        if amp_threshold is not None
        else "detector-wide artifact"
    )
    return {
        "source_kind": source_kind,
        "source_label": _public_path_label(root),
        "row_count": display_y_end - display_y_start,
        "mode_count": int(nonzero_modes),
        "x_value": display_x,
        "y_start": display_y_start,
        "y_end": display_y_end,
        "amp_threshold": amp_threshold,
        "plot": _detector_phonon_figure(
            freq_all=freq_all,
            amp_all=amp_all,
            fft_all=fft_all,
            fft_freq_all=fft_freq_all,
            local_x=local_x,
            local_y_start=local_y_start,
            local_y_end=local_y_end,
            display_x=display_x,
            display_y_start=display_y_start,
            display_y_end=display_y_end,
            max_points=max_points,
            amp_threshold=amp_threshold,
        ),
    }


def _load_detector_array(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _trace_phonon_figure(fits):
    from bokeh.models import (
        ColorBar,
        ColumnDataSource,
        HoverTool,
        LinearColorMapper,
    )
    from bokeh.palettes import Magma256
    from bokeh.plotting import figure

    ordered = sorted(
        enumerate(fits),
        key=lambda item: _row_value(item[1], item[0]),
    )
    row_values = np.asarray(
        [_row_value(fit, index) for index, fit in ordered],
        dtype=float,
    )
    frequency_max = max(
        (
            float(np.max(fit.frequency))
            for _index, fit in ordered
            if fit.frequency.size
        ),
        default=1.0,
    )
    common_frequency = np.linspace(0.0, frequency_max, 300)
    spectrum_rows = []
    for _index, fit in ordered:
        if fit.frequency.size and fit.spectrum_total.size:
            spectrum_rows.append(
                np.interp(
                    common_frequency,
                    fit.frequency,
                    fit.spectrum_total,
                    left=0.0,
                    right=0.0,
                )
            )
        else:
            spectrum_rows.append(np.zeros_like(common_frequency))
    spectrum_image = np.log1p(np.stack(spectrum_rows, axis=1))
    low, high = _finite_bounds(spectrum_image)
    mapper = LinearColorMapper(palette=list(Magma256), low=low, high=high)
    x0, dw = _row_extent(row_values)

    plot = figure(
        title="Phonon dispersion proxy: frequency centers by detector row",
        width=1400,
        height=620,
        sizing_mode="stretch_width",
        x_axis_label="detector row y",
        y_axis_label="frequency center",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.image(
        image=[spectrum_image],
        x=x0,
        y=float(common_frequency[0]),
        dw=dw,
        dh=max(float(common_frequency[-1] - common_frequency[0]), 1.0e-12),
        color_mapper=mapper,
        alpha=0.78,
    )
    plot.add_layout(ColorBar(color_mapper=mapper, width=10), "right")

    mode_source = ColumnDataSource(_trace_mode_data(ordered))
    renderer = plot.scatter(
        x="row",
        y="frequency",
        source=mode_source,
        size="size",
        fill_color="color",
        line_color="#142033",
        line_alpha=0.55,
        fill_alpha=0.84,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("trace", "@label"),
                ("mode", "@mode"),
                ("row", "@row{0}"),
                ("frequency", "@frequency{0.00000}"),
                ("amplitude", "@amplitude{0.000e}"),
                ("phase", "@phase{0.000}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _workflow_phonon_figure(workflow, *, max_points: int):
    from bokeh.models import (
        ColorBar,
        ColumnDataSource,
        HoverTool,
        LinearColorMapper,
    )
    from bokeh.palettes import Magma256
    from bokeh.plotting import figure

    row_values = np.asarray(workflow.row_y, dtype=float)
    frequency = np.asarray(workflow.dispersion_frequency, dtype=float)
    image = np.asarray(workflow.dispersion_image, dtype=float)
    low, high = _finite_bounds(image)
    mapper = LinearColorMapper(palette=list(Magma256), low=low, high=high)
    x0, dw = _row_extent(row_values)
    y0, dh = _frequency_extent(frequency)

    plot = figure(
        title="Phonon dispersion from workflow bundle",
        width=1400,
        height=620,
        sizing_mode="stretch_width",
        x_axis_label="detector row y",
        y_axis_label="frequency",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.image(
        image=[image],
        x=x0,
        y=y0,
        dw=dw,
        dh=dh,
        color_mapper=mapper,
        alpha=0.78,
    )
    plot.add_layout(ColorBar(color_mapper=mapper, width=10), "right")

    source = ColumnDataSource(
        _workflow_mode_data(workflow, max_points=max_points)
    )
    renderer = plot.scatter(
        x="row",
        y="frequency",
        source=source,
        size="size",
        fill_color="color",
        line_color="#142033",
        line_alpha=0.55,
        fill_alpha=0.84,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("row", "@row{0}"),
                ("mode", "@mode"),
                ("frequency", "@frequency{0.00000}"),
                ("amplitude", "@amplitude{0.000e}"),
                ("chi2", "@chi2{0.000e}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _detector_phonon_figure(
    *,
    freq_all: np.ndarray,
    amp_all: np.ndarray,
    fft_all: np.ndarray,
    fft_freq_all: np.ndarray,
    local_x: int,
    local_y_start: int,
    local_y_end: int,
    display_x: int,
    display_y_start: int,
    display_y_end: int,
    max_points: int,
    amp_threshold: float | None,
):
    from bokeh.models import (
        ColorBar,
        ColumnDataSource,
        HoverTool,
        LinearColorMapper,
    )
    from bokeh.palettes import Magma256
    from bokeh.plotting import figure

    row_values = np.arange(display_y_start, display_y_end, dtype=float)
    freq_slice = np.asarray(
        freq_all[local_y_start:local_y_end, local_x, :],
        dtype=float,
    )
    amp_slice = np.asarray(
        amp_all[local_y_start:local_y_end, local_x, :],
        dtype=float,
    )
    fft_slice = np.asarray(
        fft_all[local_y_start:local_y_end, local_x, :],
        dtype=float,
    )
    fft_freq_slice = np.asarray(
        fft_freq_all[local_y_start:local_y_end, local_x, :],
        dtype=float,
    )
    frequency_axis = _detector_frequency_axis(fft_freq_slice)
    max_bins = min(30, fft_slice.shape[1], frequency_axis.shape[0])
    image = np.log1p(np.maximum(fft_slice[:, :max_bins], 0.0)).T
    low, high = _finite_bounds(image)
    mapper = LinearColorMapper(palette=list(Magma256), low=low, high=high)
    x0, dw = _row_extent(row_values)

    title = (
        f"Phonon dispersion x={display_x}, "
        f"y=[{display_y_start},{display_y_end})"
    )
    if amp_threshold is not None:
        title += f", filtered amp>{amp_threshold:g}"
    plot = figure(
        title=title,
        width=1400,
        height=620,
        sizing_mode="stretch_width",
        x_axis_label="pixel y",
        y_axis_label="frequency (THz)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.image(
        image=[image],
        x=x0,
        y=float(frequency_axis[0]),
        dw=dw,
        dh=max(
            float(frequency_axis[max_bins - 1] - frequency_axis[0]), 1.0e-12
        ),
        color_mapper=mapper,
        alpha=0.78,
    )
    plot.add_layout(ColorBar(color_mapper=mapper, width=10), "right")

    source = ColumnDataSource(
        _detector_mode_data(
            rows=row_values,
            freq_slice=freq_slice,
            amp_slice=amp_slice,
            max_points=max_points,
            amp_threshold=amp_threshold,
        )
    )
    renderer = plot.scatter(
        x="row",
        y="frequency",
        source=source,
        size="size",
        fill_color="color",
        line_color="#142033",
        line_alpha=0.55,
        fill_alpha=0.84,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("row", "@row{0}"),
                ("mode", "@mode"),
                ("frequency", "@frequency{0.00000}"),
                ("amplitude", "@amplitude{0.000e}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _hero_html(
    *,
    title: str,
    source_kind: str,
    source_label: str,
    row_count: int,
    mode_count: int,
    x_value: int | None,
    y_start: int | None,
    y_end: int | None,
    amp_threshold: float | None,
) -> str:
    pills = [
        ("Source", source_kind),
        ("Rows", row_count),
        ("Modes", mode_count),
    ]
    if x_value is not None:
        pills.append(("x", x_value))
    if y_start is not None and y_end is not None:
        pills.append(("y", f"{y_start}:{y_end}"))
    if amp_threshold is not None:
        pills.append(("Amp", f">{amp_threshold:g}"))
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
        f"{html.escape(source_label)}"
        "</p>"
        f"<div class='xray-pills' style='{pills_style}'>{pill_html}</div>"
        "</div>"
    )


def _trace_mode_data(ordered) -> dict[str, list[Any]]:
    rows: list[float] = []
    labels: list[str] = []
    modes: list[int] = []
    frequencies: list[float] = []
    amplitudes: list[float] = []
    phases: list[float] = []
    for index, fit in ordered:
        count = min(
            len(fit.frequency_centers),
            len(fit.amplitude),
            len(fit.phase),
        )
        row = _row_value(fit, index)
        for mode in range(count):
            rows.append(row)
            labels.append(fit.label)
            modes.append(mode)
            frequencies.append(float(fit.frequency_centers[mode]))
            amplitudes.append(float(fit.amplitude[mode]))
            phases.append(float(fit.phase[mode]))
    return {
        "row": rows,
        "label": labels,
        "mode": modes,
        "frequency": frequencies,
        "amplitude": amplitudes,
        "phase": phases,
        "color": _palette_colors(amplitudes),
        "size": _amplitude_sizes(amplitudes),
    }


def _detector_mode_data(
    *,
    rows: np.ndarray,
    freq_slice: np.ndarray,
    amp_slice: np.ndarray,
    max_points: int,
    amp_threshold: float | None,
) -> dict[str, list[Any]]:
    row_grid = np.repeat(rows[:, None], freq_slice.shape[1], axis=1)
    mode_grid = np.repeat(
        np.arange(freq_slice.shape[1], dtype=int)[None, :],
        freq_slice.shape[0],
        axis=0,
    )
    if amp_threshold is None:
        amp_mask = amp_slice > 0.0
    else:
        amp_mask = (amp_slice > float(amp_threshold)) & (amp_slice < 1.0e6)
    mask = np.isfinite(freq_slice) & np.isfinite(amp_slice)
    mask &= (freq_slice > 0.0) & amp_mask
    row_values = row_grid[mask]
    mode_values = mode_grid[mask]
    frequencies = freq_slice[mask]
    amplitudes = amp_slice[mask]
    if max_points > 0 and amplitudes.size > max_points:
        keep = np.argpartition(amplitudes, -max_points)[-max_points:]
        row_values = row_values[keep]
        mode_values = mode_values[keep]
        frequencies = frequencies[keep]
        amplitudes = amplitudes[keep]
    amplitude_list = [float(value) for value in amplitudes]
    return {
        "row": [float(value) for value in row_values],
        "mode": [int(value) for value in mode_values],
        "frequency": [float(value) for value in frequencies],
        "amplitude": amplitude_list,
        "color": _palette_colors(amplitude_list),
        "size": _amplitude_sizes(amplitude_list),
    }


def _workflow_mode_data(workflow, *, max_points: int) -> dict[str, list[Any]]:
    row_values = np.repeat(
        np.asarray(workflow.row_y, dtype=float)[:, None],
        workflow.frequency_centers.shape[1],
        axis=1,
    )
    mode_values = np.repeat(
        np.arange(workflow.frequency_centers.shape[1], dtype=int)[None, :],
        workflow.frequency_centers.shape[0],
        axis=0,
    )
    frequencies = np.asarray(workflow.frequency_centers, dtype=float)
    amplitudes = np.asarray(workflow.amplitudes, dtype=float)
    chi2_values = np.repeat(
        np.asarray(workflow.chi2, dtype=float)[:, None],
        workflow.frequency_centers.shape[1],
        axis=1,
    )
    mask = (
        np.isfinite(frequencies)
        & np.isfinite(amplitudes)
        & (amplitudes > 0.0)
    )
    rows = row_values[mask]
    modes = mode_values[mask]
    frequency_values = frequencies[mask]
    amplitude_values = amplitudes[mask]
    chi2_flat = chi2_values[mask]
    if max_points > 0 and amplitude_values.size > max_points:
        keep = np.argpartition(amplitude_values, -max_points)[-max_points:]
        rows = rows[keep]
        modes = modes[keep]
        frequency_values = frequency_values[keep]
        amplitude_values = amplitude_values[keep]
        chi2_flat = chi2_flat[keep]
    amplitude_list = [float(value) for value in amplitude_values]
    return {
        "row": [float(value) for value in rows],
        "mode": [int(value) for value in modes],
        "frequency": [float(value) for value in frequency_values],
        "amplitude": amplitude_list,
        "chi2": [float(value) for value in chi2_flat],
        "color": _palette_colors(amplitude_list),
        "size": _amplitude_sizes(amplitude_list),
    }


def _detector_frequency_axis(fft_freq_slice: np.ndarray) -> np.ndarray:
    if fft_freq_slice.ndim != 2 or fft_freq_slice.shape[0] == 0:
        return np.arange(max(fft_freq_slice.shape[-1], 1), dtype=float)
    middle = fft_freq_slice.shape[0] // 2
    axis = np.asarray(fft_freq_slice[middle], dtype=float)
    if axis.size == 0 or not np.any(np.isfinite(axis)):
        return np.arange(max(fft_freq_slice.shape[-1], 1), dtype=float)
    return axis


def _frequency_extent(frequency: np.ndarray) -> tuple[float, float]:
    values = np.asarray(frequency, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    if values.size == 1:
        return float(values[0] - 0.5), 1.0
    return float(np.min(values)), max(float(np.ptp(values)), 1.0e-12)


def _row_value(fit, index: int) -> float:
    if fit.row_y is not None:
        return float(fit.row_y)
    return float(index)


def _row_extent(rows: np.ndarray) -> tuple[float, float]:
    if rows.size == 0:
        return -0.5, 1.0
    if rows.size == 1:
        return float(rows[0] - 0.5), 1.0
    unique_rows = np.unique(rows)
    spacing = (
        float(np.min(np.diff(unique_rows))) if unique_rows.size > 1 else 1.0
    )
    return float(np.min(rows) - spacing / 2), float(
        np.max(rows) - np.min(rows) + spacing
    )


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
