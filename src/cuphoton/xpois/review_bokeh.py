# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive Bokeh review report generation for subtraction runs."""

from __future__ import annotations

import base64
import html
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

_REVIEW_TEMPLATE = """
{% block preamble %}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {% if favicon_href %}
    <link rel="icon" type="image/x-icon" href="{{ favicon_href | safe }}">
    <link rel="shortcut icon" href="{{ favicon_href | safe }}">
    {% endif %}
{% endblock %}
{% block postamble %}
    <style>
      :root {
        --review-bg: #f7fbff;
        --review-surface: #ffffff;
        --review-surface-soft: #f3f7fc;
        --review-text: #07145c;
        --review-muted: #7080b5;
        --review-border: #dde6f4;
        --review-teal: #03c7b7;
        --review-indigo: #5d4df2;
        --review-green: #19c98d;
        --review-amber: #ff9f43;
        --review-shadow: 0 14px 38px rgba(42, 55, 105, 0.12);
      }
      html, body {
        min-height: 100%;
        margin: 0;
        padding: 0;
        background: var(--review-bg);
        color: var(--review-text);
        font-family: "Inter", "IBM Plex Sans", "Avenir Next", "Segoe UI",
          sans-serif;
      }
      .page-shell {
        width: min(1440px, calc(100vw - 36px));
        margin: 0 auto;
        padding: 18px 0 42px;
      }
      .page-shell h1, .page-shell h2, .page-shell h3 {
        margin: 0 0 12px;
      }
      .page-shell p, .page-shell li {
        line-height: 1.5;
      }
      .page-shell details {
        margin-top: 12px;
        border: 1px solid var(--review-border);
        border-radius: 8px;
        background: var(--review-surface-soft);
        padding: 10px 12px;
      }
      .page-shell summary {
        cursor: pointer;
        color: var(--review-indigo);
        font-weight: 600;
      }
      .page-shell pre {
        margin: 12px 0 0;
        padding: 14px;
        border-radius: 8px;
        background: #f7f9fd;
        color: var(--review-text);
        overflow-x: auto;
        font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      }
      .page-shell table.meta {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
      }
      .page-shell table.meta th,
      .page-shell table.meta td {
        text-align: left;
        vertical-align: top;
        padding: 8px 10px;
        border-bottom: 1px solid #edf1f7;
      }
      .page-shell table.meta th {
        width: 220px;
        color: var(--review-muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      .review-hero,
      .page-shell .section-card {
        margin-bottom: 18px;
        padding: 18px;
        border: 1px solid var(--review-border);
        border-radius: 8px;
        background: var(--review-surface);
        box-shadow: var(--review-shadow);
      }
      .review-hero h1 {
        color: var(--review-text);
        font-size: 34px;
        font-weight: 760;
        line-height: 1.08;
      }
      .review-subtitle {
        color: var(--review-muted);
        font-size: 15px;
        max-width: 980px;
      }
      .review-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
      }
      .review-pill {
        background: #eefaff;
        border: 1px solid #d5edf7;
        border-radius: 999px;
        color: var(--review-text);
        display: inline-block;
        font-weight: 650;
        padding: 7px 11px;
      }
      .review-pill strong {
        color: var(--review-muted);
        font-size: 12px;
        margin-right: 6px;
      }
      .bk-root .bk-Column {
        width: 100% !important;
      }
      .bk-root .bk-Row {
        width: 100% !important;
      }
      .panel-row {
        gap: 12px !important;
        margin-bottom: 12px;
      }
      @media (max-width: 1180px) {
        .panel-row {
          flex-direction: column !important;
        }
        .review-hero h1 {
          font-size: 28px;
        }
      }
    </style>
{% endblock %}
{% block contents %}
    <div class="page-shell">
      {{ plot_div | indent(6) }}
    </div>
{% endblock %}
"""


_ASSET_DATA_URI_CACHE: dict[tuple[str, str], str] = {}


def _asset_data_uri(filename: str, mime_type: str) -> str:
    key = (filename, mime_type)
    cached = _ASSET_DATA_URI_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        payload = (
            resources.files(__package__)
            .joinpath("assets", filename)
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError):
        _ASSET_DATA_URI_CACHE[key] = ""
        return ""
    encoded = base64.b64encode(payload).decode("ascii")
    uri = f"data:{mime_type};base64,{encoded}"
    _ASSET_DATA_URI_CACHE[key] = uri
    return uri


def _review_logo_html(size_px: int = 76) -> str:
    logo_src = _asset_data_uri("cuphoton-logo-128.png", "image/png")
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


def identify_mask_components(
    mask_values: np.ndarray | None,
    *,
    raw_image: np.ndarray,
    plane_map: dict[str, int] | None,
    masked_plane_names: list[str] | None,
    max_regions: int = 32,
) -> list[dict[str, object]]:
    """Summarize connected masked regions for interactive hover overlays."""

    from scipy import ndimage

    if mask_values is None or plane_map is None or not masked_plane_names:
        return []

    missing = [name for name in masked_plane_names if name not in plane_map]
    if missing:
        return []

    bitmask = 0
    for name in masked_plane_names:
        bitmask |= 1 << int(plane_map[name])

    mask = (np.asarray(mask_values, dtype=np.int64) & bitmask) != 0
    if not np.any(mask):
        return []

    labels, count = ndimage.label(mask)
    components: list[dict[str, object]] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        values = np.unique(np.asarray(mask_values)[ys, xs])
        planes: set[str] = set()
        for value in values.tolist():
            for name, bit in plane_map.items():
                if int(value) & (1 << int(bit)):
                    planes.add(name)

        bbox = [
            int(ys.min()),
            int(ys.max()) + 1,
            int(xs.min()),
            int(xs.max()) + 1,
        ]
        local_image = np.asarray(raw_image)[ys, xs]
        finite = local_image[np.isfinite(local_image)]
        components.append(
            {
                "bbox_y0y1x0x1": bbox,
                "pixel_count": int(ys.size),
                "planes": sorted(planes),
                "values": [int(v) for v in values.tolist()],
                "centroid_yx": [float(np.mean(ys)), float(np.mean(xs))],
                "max_signal": float(np.max(finite)) if finite.size else None,
            }
        )

    components.sort(key=lambda item: int(item["pixel_count"]), reverse=True)
    return components[:max_regions]


def write_interactive_review_artifact(
    artifacts_dir: Path,
    *,
    run_name: str,
    raw_reference: np.ndarray,
    raw_target: np.ndarray,
    matched: np.ndarray,
    residual: np.ndarray,
    fit_mask_metadata: dict[str, Any] | None,
    input_mask_metadata: dict[str, Any] | None,
    reference_mask_values: np.ndarray | None,
    target_mask_values: np.ndarray | None,
    reference_plane_map: dict[str, int] | None,
    target_plane_map: dict[str, int] | None,
    hotspots: list[dict[str, object]],
    raw_gray_lo: float,
    raw_gray_hi: float,
    matched_lo: float,
    matched_hi: float,
    residual_limit: float,
) -> dict[str, str]:
    """Persist a standalone interactive Bokeh HTML report."""

    try:
        from bokeh.embed import file_html
        from bokeh.layouts import column, row
        from bokeh.models import (
            ColorBar,
            ColumnDataSource,
            CustomJS,
            Div,
            HoverTool,
            LabelSet,
            LinearColorMapper,
            Range1d,
        )
        from bokeh.palettes import Greys256, RdBu11
        from bokeh.plotting import figure
        from bokeh.resources import INLINE
    except ModuleNotFoundError:
        return {}

    height, width = raw_reference.shape

    reference_components = identify_mask_components(
        reference_mask_values,
        raw_image=raw_reference,
        plane_map=reference_plane_map,
        masked_plane_names=(
            input_mask_metadata.get("reference_mask", {}).get(
                "masked_plane_names"
            )
            if input_mask_metadata is not None
            else None
        ),
    )
    target_components = identify_mask_components(
        target_mask_values,
        raw_image=raw_target,
        plane_map=target_plane_map,
        masked_plane_names=(
            input_mask_metadata.get("target_mask", {}).get(
                "masked_plane_names"
            )
            if input_mask_metadata is not None
            else None
        ),
    )

    def image_figure(
        title: str,
        image: np.ndarray,
        *,
        low: float,
        high: float,
        palette: list[str],
        width_px: int = 640,
        height_px: int = 640,
    ):
        mapper = LinearColorMapper(palette=palette, low=low, high=high)
        fig = figure(
            title=title,
            width=width_px,
            height=height_px,
            sizing_mode="scale_width",
            match_aspect=True,
            x_range=Range1d(0, width),
            y_range=Range1d(0, height),
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
        )
        fig.image(
            image=[np.asarray(image, dtype=np.float64)],
            x=0,
            y=0,
            dw=width,
            dh=height,
            color_mapper=mapper,
        )
        fig.grid.visible = False
        fig.axis.visible = False
        fig.outline_line_color = "#d9e2f0"
        fig.border_fill_color = "#ffffff"
        fig.background_fill_color = "#ffffff"
        fig.title.text_color = "#07145c"
        fig.title.text_font_size = "13px"
        fig.title.text_font_style = "bold"
        fig.toolbar.logo = None
        fig.add_layout(ColorBar(color_mapper=mapper, width=10), "right")
        return fig

    def overlay_source(
        items: list[dict[str, object]], *, mode: str
    ) -> ColumnDataSource:
        return ColumnDataSource(
            {
                "label": [str(index) for index in range(1, len(items) + 1)],
                "x": [
                    (item["bbox_y0y1x0x1"][2] + item["bbox_y0y1x0x1"][3]) / 2
                    for item in items
                ],
                "y": [
                    (item["bbox_y0y1x0x1"][0] + item["bbox_y0y1x0x1"][1]) / 2
                    for item in items
                ],
                "width": [
                    item["bbox_y0y1x0x1"][3] - item["bbox_y0y1x0x1"][2]
                    for item in items
                ],
                "height": [
                    item["bbox_y0y1x0x1"][1] - item["bbox_y0y1x0x1"][0]
                    for item in items
                ],
                "bbox": [str(item["bbox_y0y1x0x1"]) for item in items],
                "planes": [
                    ", ".join(item.get("planes", [])) for item in items
                ],
                "pixel_count": [
                    int(item.get("pixel_count", 0)) for item in items
                ],
                "max_signal": [item.get("max_signal") for item in items],
                "peak_yx": [str(item.get("peak_yx")) for item in items],
                "peak_abs_sigma": [
                    item.get("peak_abs_sigma") for item in items
                ],
                "peak_residual": [
                    item.get("peak_residual") for item in items
                ],
                "center_yx": [str(item.get("center_yx")) for item in items],
                "score": [item.get("score") for item in items],
                "compact_fraction": [
                    item.get("compact_fraction") for item in items
                ],
                "tooltip_title": [mode for _ in items],
            }
        )

    def add_overlay(
        fig,
        source: ColumnDataSource,
        *,
        line_color: str,
        fill_color: str,
        tooltip_rows: list[tuple[str, str]],
    ) -> None:
        renderer = fig.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=source,
            fill_alpha=0.05,
            fill_color=fill_color,
            line_color=line_color,
            line_width=2.0,
            selection_line_color="white",
            nonselection_alpha=0.12,
        )
        fig.add_layout(
            LabelSet(
                x="x",
                y="y",
                text="label",
                source=source,
                text_color=line_color,
                text_font_size="9pt",
                text_align="center",
                text_baseline="middle",
            )
        )
        fig.add_tools(HoverTool(renderers=[renderer], tooltips=tooltip_rows))

    reference_fig = image_figure(
        "Raw reference with mask overlays",
        raw_reference,
        low=raw_gray_lo,
        high=raw_gray_hi,
        palette=list(Greys256),
    )
    target_fig = image_figure(
        "Raw target with mask overlays",
        raw_target,
        low=raw_gray_lo,
        high=raw_gray_hi,
        palette=list(Greys256),
    )
    stamps_fig = image_figure(
        "Raw target with fit-stamp overlays",
        raw_target,
        low=raw_gray_lo,
        high=raw_gray_hi,
        palette=list(Greys256),
    )
    matched_fig = image_figure(
        "Matched model",
        matched,
        low=matched_lo,
        high=matched_hi,
        palette=list(Greys256),
    )
    residual_fig = image_figure(
        "Residual with hotspot overlays",
        residual,
        low=-residual_limit,
        high=residual_limit,
        palette=list(reversed(RdBu11)),
    )
    hotspot_target_fig = image_figure(
        "Raw target with hotspot overlays",
        raw_target,
        low=raw_gray_lo,
        high=raw_gray_hi,
        palette=list(Greys256),
    )

    reference_source = overlay_source(reference_components, mode="mask")
    target_source = overlay_source(target_components, mode="mask")
    add_overlay(
        reference_fig,
        reference_source,
        line_color="#ff9f43",
        fill_color="#ff9f43",
        tooltip_rows=[
            ("planes", "@planes"),
            ("pixels", "@pixel_count"),
            ("bbox", "@bbox"),
            ("max signal", "@max_signal{0.00}"),
        ],
    )
    add_overlay(
        target_fig,
        target_source,
        line_color="#03c7b7",
        fill_color="#03c7b7",
        tooltip_rows=[
            ("planes", "@planes"),
            ("pixels", "@pixel_count"),
            ("bbox", "@bbox"),
            ("max signal", "@max_signal{0.00}"),
        ],
    )

    stamp_items: list[dict[str, object]] = []
    if fit_mask_metadata is not None:
        for center, bbox, score, compact_fraction in zip(
            fit_mask_metadata.get("centers_yx", []),
            fit_mask_metadata.get("stamps_y0y1x0x1", []),
            fit_mask_metadata.get("scores", []),
            fit_mask_metadata.get("compact_fractions", []),
        ):
            stamp_items.append(
                {
                    "center_yx": center,
                    "bbox_y0y1x0x1": bbox,
                    "score": score,
                    "compact_fraction": compact_fraction,
                }
            )
    stamp_source = overlay_source(stamp_items, mode="stamp")
    add_overlay(
        stamps_fig,
        stamp_source,
        line_color="#19c98d",
        fill_color="#19c98d",
        tooltip_rows=[
            ("center", "@center_yx"),
            ("bbox", "@bbox"),
            ("score", "@score{0.00}"),
            ("compact fraction", "@compact_fraction{0.000}"),
        ],
    )

    hotspot_source = overlay_source(hotspots, mode="hotspot")
    add_overlay(
        residual_fig,
        hotspot_source,
        line_color="#ff9f43",
        fill_color="#ff9f43",
        tooltip_rows=[
            ("peak (y, x)", "@peak_yx"),
            ("peak |sigma|", "@peak_abs_sigma{0.00}"),
            ("peak residual", "@peak_residual{0.00}"),
            ("pixels", "@pixel_count"),
            ("bbox", "@bbox"),
        ],
    )
    add_overlay(
        hotspot_target_fig,
        hotspot_source,
        line_color="#ff9f43",
        fill_color="#ff9f43",
        tooltip_rows=[
            ("peak (y, x)", "@peak_yx"),
            ("peak |sigma|", "@peak_abs_sigma{0.00}"),
            ("peak residual", "@peak_residual{0.00}"),
            ("pixels", "@pixel_count"),
            ("bbox", "@bbox"),
        ],
    )

    mask_policy = html.escape(
        str((input_mask_metadata or {}).get("mask_policy", "none"))
    )
    overview_text = Div(
        text=(
            "<div style='background:#fff;border:1px solid #dde6f4;"
            "border-radius:8px;box-shadow:0 14px 38px "
            "rgba(42,55,105,0.12);margin-bottom:18px;padding:18px;'>"
            "<div style='align-items:center;display:flex;gap:16px;"
            "margin-bottom:12px;'>"
            f"{_review_logo_html()}"
            "<div style='min-width:0;'>"
            "<h1 style='color:#07145c;font-size:34px;font-weight:760;"
            "line-height:1.08;margin:0 0 12px;'>"
            "XPOIS Subtraction Review</h1>"
            "</div>"
            "</div>"
            "<p style='color:#7080b5;font-size:15px;line-height:1.5;"
            "max-width:980px;'>"
            f"{html.escape(run_name)}. Hover overlay boxes to inspect mask "
            "planes, fit-stamp metadata, and hotspot sigma. Scroll the page "
            "to inspect each panel independently."
            "</p>"
            "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-top:14px;'>"
            "<span style='background:#eefaff;border:1px solid #d5edf7;"
            "border-radius:999px;color:#07145c;display:inline-block;"
            "font-weight:650;padding:7px 11px;'>"
            "<strong style='color:#7080b5;font-size:12px;margin-right:6px;'>"
            f"Mask policy</strong>{mask_policy}"
            "</span>"
            "<span style='background:#eefaff;border:1px solid #d5edf7;"
            "border-radius:999px;color:#07145c;display:inline-block;"
            "font-weight:650;padding:7px 11px;'>"
            "<strong style='color:#7080b5;font-size:12px;margin-right:6px;'>"
            f"Selected stamps</strong>{len(stamp_items)}"
            "</span>"
            "<span style='background:#eefaff;border:1px solid #d5edf7;"
            "border-radius:999px;color:#07145c;display:inline-block;"
            "font-weight:650;padding:7px 11px;'>"
            "<strong style='color:#7080b5;font-size:12px;margin-right:6px;'>"
            f"Hotspots</strong>{len(hotspots)}"
            "</span>"
            "</div>"
            "</div>"
        ),
        sizing_mode="stretch_width",
    )

    hotspot_items = "".join(
        (
            "<li>"
            f"<strong>{index}.</strong> peak {item['peak_yx']}, "
            f"|sigma|={float(item['peak_abs_sigma']):.2f}, "
            f"residual={float(item['peak_residual']):.2f}, "
            f"pixels={int(item['pixel_count'])}"
            "</li>"
        )
        for index, item in enumerate(hotspots, start=1)
    )
    fit_summary_html = _render_metadata_table(
        [
            ("Mask kind", (fit_mask_metadata or {}).get("kind")),
            ("Selected stamps", len(stamp_items)),
            ("Stamp size", (fit_mask_metadata or {}).get("stamp_size")),
            (
                "Peak percentile",
                (fit_mask_metadata or {}).get("peak_percentile"),
            ),
            (
                "Background filter size",
                (fit_mask_metadata or {}).get("background_filter_size"),
            ),
            (
                "Peak filter size",
                (fit_mask_metadata or {}).get("peak_filter_size"),
            ),
            (
                "Min separation",
                (fit_mask_metadata or {}).get("min_separation"),
            ),
        ]
    )
    mask_summary_html = _render_metadata_table(
        [
            ("Mask policy", (input_mask_metadata or {}).get("mask_policy")),
            (
                "Reference mask fraction",
                (input_mask_metadata or {}).get("reference_mask_fraction"),
            ),
            (
                "Target mask fraction",
                (input_mask_metadata or {}).get("target_mask_fraction"),
            ),
            (
                "Masked planes",
                ", ".join(
                    (input_mask_metadata or {})
                    .get("reference_mask", {})
                    .get("masked_plane_names", [])
                ),
            ),
            (
                "Reference mask source",
                (input_mask_metadata or {}).get("reference_mask_source"),
            ),
            (
                "Target mask source",
                (input_mask_metadata or {}).get("target_mask_source"),
            ),
        ]
    )
    section_card_style = (
        "background:#fff;border:1px solid #dde6f4;border-radius:8px;"
        "box-shadow:0 14px 38px rgba(42,55,105,0.12);"
        "margin-bottom:18px;padding:18px;"
    )
    section_title_style = (
        "color:#07145c;font-size:18px;font-weight:760;margin:0 0 8px;"
    )
    section_copy_style = "color:#07145c;line-height:1.5;"
    details_style = (
        "background:#f3f7fc;border:1px solid #dde6f4;border-radius:8px;"
        "margin-top:12px;padding:10px 12px;"
    )
    summary_style = "color:#5d4df2;cursor:pointer;font-weight:650;"
    pre_style = (
        "background:#f7f9fd;border-radius:8px;color:#07145c;"
        "font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,"
        "monospace;margin:12px 0 0;overflow-x:auto;padding:14px;"
    )
    hotspot_summary = Div(
        text=(
            f"<div style='{section_card_style}'>"
            f"<h2 style='{section_title_style}'>Residual Hotspots</h2>"
            f"<p style='{section_copy_style}'>"
            "These are the strongest connected residual excursions above the "
            "review threshold. Match the numbered boxes in the target and "
            "residual panels.</p>"
            f"<ul>{hotspot_items or '<li>No hotspots above threshold.</li>'}</ul>"
            "</div>"
        ),
        sizing_mode="stretch_width",
    )

    fit_summary = Div(
        text=(
            f"<div style='{section_card_style}'>"
            f"<h2 style='{section_title_style}'>Fit Stamp Summary</h2>"
            f"<p style='{section_copy_style}'>"
            "The green boxes are the compact-source regions used to fit the "
            "kernel. Hover to inspect score and compactness.</p>"
            f"{fit_summary_html}"
            f"<details style='{details_style}'>"
            f"<summary style='{summary_style}'>"
            "Show full compact-source selection metadata</summary>"
            f"<pre style='{pre_style}'>"
            f"{html.escape(json.dumps(fit_mask_metadata or {}, indent=2))}</pre>"
            "</details>"
            "</div>"
        ),
        sizing_mode="stretch_width",
    )
    mask_summary = Div(
        text=(
            f"<div style='{section_card_style}'>"
            f"<h2 style='{section_title_style}'>Input Mask Summary</h2>"
            f"<p style='{section_copy_style}'>"
            "The blue/yellow boxes correspond to connected components "
            "selected by the input mask policy.</p>"
            f"{mask_summary_html}"
            f"<details style='{details_style}'>"
            f"<summary style='{summary_style}'>"
            "Show full input mask metadata</summary>"
            f"<pre style='{pre_style}'>"
            f"{html.escape(json.dumps(input_mask_metadata or {}, indent=2))}</pre>"
            "</details>"
            "</div>"
        ),
        sizing_mode="stretch_width",
    )

    def attach_selection_zoom(source: ColumnDataSource, *figures) -> None:
        args = {f"fig{i}": fig for i, fig in enumerate(figures)}
        fig_list = ", ".join(args.keys())
        callback = CustomJS(
            args=args,
            code=f"""
            const idx = cb_obj.indices[0]
            if (idx == null) {{
              return
            }}
            const x = cb_obj.data.x[idx]
            const y = cb_obj.data.y[idx]
            const w = cb_obj.data.width[idx]
            const h = cb_obj.data.height[idx]
            const pad = 10
            const figs = [{fig_list}].filter((value) => value && value.x_range)
            for (const fig of figs) {{
              fig.x_range.start = x - (w / 2) - pad
              fig.x_range.end = x + (w / 2) + pad
              fig.y_range.start = y - (h / 2) - pad
              fig.y_range.end = y + (h / 2) + pad
            }}
            """,
        )
        source.selected.js_on_change("indices", callback)

    attach_selection_zoom(target_source, target_fig)
    attach_selection_zoom(reference_source, reference_fig)
    attach_selection_zoom(stamp_source, stamps_fig, matched_fig)
    attach_selection_zoom(hotspot_source, hotspot_target_fig, residual_fig)

    layout = column(
        overview_text,
        row(
            reference_fig,
            target_fig,
            sizing_mode="stretch_width",
            css_classes=["panel-row"],
            styles={"gap": "12px", "margin-bottom": "12px"},
        ),
        mask_summary,
        row(
            stamps_fig,
            matched_fig,
            sizing_mode="stretch_width",
            css_classes=["panel-row"],
            styles={"gap": "12px", "margin-bottom": "12px"},
        ),
        fit_summary,
        row(
            hotspot_target_fig,
            residual_fig,
            sizing_mode="stretch_width",
            css_classes=["panel-row"],
            styles={"gap": "12px", "margin-bottom": "12px"},
        ),
        hotspot_summary,
        sizing_mode="stretch_width",
    )

    html_path = artifacts_dir / "review_bokeh.html"
    html_path.write_text(
        file_html(
            layout,
            INLINE,
            f"XPOIS interactive review: {run_name}",
            template=_REVIEW_TEMPLATE,
            template_variables={
                "favicon_href": _asset_data_uri("favicon.ico", "image/x-icon")
            },
        ),
        encoding="utf-8",
    )
    return {
        "review_bokeh_html": str(html_path.relative_to(artifacts_dir.parent))
    }


def _render_metadata_table(rows: list[tuple[str, object]]) -> str:
    rendered = []
    for key, value in rows:
        if value is None or value == "":
            display = "n/a"
        else:
            display = str(value)
        rendered.append(
            "<tr>"
            "<th style='border-bottom:1px solid #edf1f7;color:#7080b5;"
            "font-size:12px;font-weight:650;padding:8px 16px 8px 0;"
            "text-align:left;vertical-align:top;width:220px;'>"
            f"{html.escape(str(key))}</th>"
            "<td style='border-bottom:1px solid #edf1f7;color:#07145c;"
            "padding:8px 10px;text-align:left;vertical-align:top;'>"
            f"{html.escape(display)}</td>"
            "</tr>"
        )
    return (
        "<table style='border-collapse:collapse;margin-top:8px;width:100%;'>"
        + "".join(rendered)
        + "</table>"
    )
