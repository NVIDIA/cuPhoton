# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline, linked scientific views for the Model Sky notebook.

Images retain their native pixels. Pixel (0, 0) is at the lower left, as in
FITS/WCS; a supplied WCS must describe the supplied cutout, not its parent.
All factories return fresh Bokeh models. JavaScript changes only the display;
scientific fits and learner parameter changes run in Python or the CLI.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    Arrow,
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Label,
    LinearColorMapper,
    NormalHead,
    RadioButtonGroup,
    Range1d,
    Slider,
    Span,
)
from bokeh.palettes import Greys256, RdBu11
from bokeh.plotting import figure
from bokeh.resources import INLINE

_INK = "#172b45"
_MUTED = "#52647c"
_BLUE = "#007f99"
_AMBER = "#d97a00"
_SURFACE = "#f7f9fc"
_DIVERGING = list(RdBu11)  # Negative blue; positive red.
_GRAY = list(Greys256)  # Low dark; high light.


def _text(title, body="", *, eyebrow=""):
    return Div(
        text=(
            '<div style="font-family:Arial,sans-serif;padding:12px 16px;'
            "border-left:4px solid #007f99;background:#f0f6fa;"
            'border-radius:0 8px 8px 0;color:#172b45">'
            f'<div style="font-size:11px;letter-spacing:1.5px;'
            f'color:#52647c">{escape(eyebrow.upper())}</div>'
            f'<div style="font-size:22px;line-height:1.4;'
            f'font-weight:600">{escape(title)}</div>'
            '<div style="font-size:13px;line-height:1.6">'
            f"{escape(body)}</div>"
            "</div>"
        ),
        sizing_mode="stretch_width",
    )


def _note(body):
    return Div(
        text=f'<div style="color:{_MUTED};font:13px/1.6 Arial;'
        f'padding:4px 12px">{escape(body)}</div>',
        sizing_mode="stretch_width",
    )


def _style(plot):
    plot.background_fill_color = "#ffffff"
    plot.border_fill_color = "#ffffff"
    plot.outline_line_color = "#d8e1ec"
    plot.title.text_color = _INK
    plot.title.text_font = "Arial"
    plot.title.text_font_size = "14px"
    plot.title.text_font_style = "bold"
    plot.axis.axis_label_text_font = "Arial"
    plot.axis.axis_label_text_font_style = "normal"
    plot.axis.axis_label_text_font_size = "12px"
    plot.axis.major_label_text_font_size = "11px"
    plot.axis.axis_label_text_color = _MUTED
    plot.axis.major_label_text_color = _MUTED
    plot.axis.axis_line_color = "#bdcad9"
    plot.axis.major_tick_line_color = "#bdcad9"
    plot.axis.minor_tick_line_color = None
    plot.grid.grid_line_color = "#e9eef4"
    plot.toolbar.logo = None
    plot.min_border = 12
    plot.hidpi = True
    return plot


def _image(array):
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError("An image must be a nonempty two-dimensional array.")
    return result


def _bounds(*arrays):
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if not finite.size:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1, 99.7])
    if low == high:
        high = low + max(abs(low) * 0.01, 1.0)
    return float(low), float(high)


def _limit(array):
    finite = np.abs(array[np.isfinite(array)])
    limit = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    return limit if limit > 0 else 1.0


def _sky(shape, wcs):
    if wcs is None:
        return {}
    yy, xx = np.indices(shape, dtype=float)
    ra, dec = wcs.celestial.pixel_to_world_values(xx, yy)
    return {"ra": [np.asarray(ra)], "dec": [np.asarray(dec)]}


def _scale_bar(plot, shape, pixel_scale):
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive arcseconds per pixel.")
    height, width = shape
    target = width * pixel_scale / 4
    decade = 10 ** np.floor(np.log10(target))
    length = max(v for v in (1, 2, 5, 10) if v <= target / decade) * decade
    x, y = width * 0.07, height * 0.08
    plot.line(
        [x, x + length / pixel_scale],
        [y, y],
        line_width=5,
        line_color="#152436",
        line_alpha=0.7,
    )
    plot.line(
        [x, x + length / pixel_scale],
        [y, y],
        line_width=2,
        line_color="white",
    )
    plot.add_layout(
        Label(
            x=x,
            y=y,
            y_offset=5,
            text=f"{length:g} arcsec",
            text_font_size="11px",
            text_color="white",
            background_fill_color="#152436",
            background_fill_alpha=0.8,
            border_line_color=None,
        )
    )


def _image_panel(
    array,
    title,
    *,
    mapper,
    pixel_scale=0.2,
    unit="nJy",
    mask=None,
    wcs=None,
    ranges=None,
    width=350,
):
    array = _image(array)
    height, image_width = array.shape
    mask_array = (
        np.zeros_like(array, dtype=np.uint8)
        if mask is None
        else np.asarray(mask, dtype=bool).astype(np.uint8)
    )
    if mask_array.shape != array.shape:
        raise ValueError(
            "mask must have the image shape; True means excluded."
        )
    source = ColumnDataSource(
        dict(image=[array], mask=[mask_array], **_sky(array.shape, wcs))
    )
    if ranges is None:
        ranges = (
            Range1d(-0.5, image_width - 0.5),
            Range1d(-0.5, height - 0.5),
        )
    plot = _style(
        figure(
            title=title,
            width=width + 40,
            height=round((width - 85) * height / image_width) + 85,
            sizing_mode="fixed",
            x_range=ranges[0],
            y_range=ranges[1],
            match_aspect=True,
            x_axis_label="Cutout x [pixel]",
            y_axis_label="Cutout y [pixel]",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
            name=title,
        )
    )
    plot.min_border_left = 50
    plot.min_border_right = 75
    plot.min_border_top = 35
    plot.min_border_bottom = 50
    plot.grid.visible = False
    renderer = plot.image(
        image="image",
        source=source,
        x=-0.5,
        y=-0.5,
        dw=image_width,
        dh=height,
        color_mapper=mapper,
    )
    tooltips = [
        ("Pixel (x, y)", "($x{0.0}, $y{0.0})"),
        (f"Signal [{unit}]", "@image{0.0000}"),
    ]
    if mask is not None:
        tooltips.append(("Excluded pixel", "@mask"))
    if wcs is not None:
        tooltips += [
            ("RA [deg]", "@ra{0.000000}"),
            ("Dec [deg]", "@dec{0.000000}"),
        ]
    plot.add_tools(HoverTool(renderers=[renderer], tooltips=tooltips))
    plot.add_layout(
        ColorBar(
            color_mapper=mapper,
            title=unit,
            width=8,
            padding=7,
            title_text_font_size="11px",
            title_text_font_style="normal",
            major_label_text_font_size="10px",
            label_standoff=4,
        ),
        "right",
    )
    _scale_bar(plot, array.shape, pixel_scale)
    overlay = None
    if mask is not None:
        rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
        rgba[mask_array != 0] = [245, 173, 50, 100]
        overlay = plot.image_rgba(
            image=[rgba.view(np.uint32).reshape(array.shape)],
            x=-0.5,
            y=-0.5,
            dw=image_width,
            dh=height,
        )
    return plot, source, overlay


def _triptych(
    science, reference, difference, *, mask, pixel_scale, unit, wcs
):
    science, reference, difference = map(
        _image, (science, reference, difference)
    )
    if reference.shape != science.shape or difference.shape != science.shape:
        raise ValueError(
            "The three registered images must have equal shapes."
        )
    low, high = _bounds(science, reference)
    raw_mapper = LinearColorMapper(palette=_GRAY, low=low, high=high)
    limit = _limit(difference)
    residual_mapper = LinearColorMapper(
        palette=_DIVERGING, low=-limit, high=limit, nan_color="#b8c3d1"
    )
    ranges = (
        Range1d(-0.5, science.shape[1] - 0.5),
        Range1d(-0.5, science.shape[0] - 0.5),
    )
    panels, overlays = [], []
    for array, title, mapper in (
        (reference, "Reference", raw_mapper),
        (science, "Science", raw_mapper),
        (difference, "Science − matched reference", residual_mapper),
    ):
        panel, _, overlay = _image_panel(
            array,
            title,
            mapper=mapper,
            mask=mask,
            ranges=ranges,
            pixel_scale=pixel_scale,
            unit=unit,
            wcs=wcs,
        )
        panels.append(panel)
        if overlay is not None:
            overlays.append(overlay)
    items = [row(*panels, sizing_mode="stretch_width")]
    if overlays:
        toggle = CheckboxGroup(
            labels=["Show excluded pixels in amber"], active=[0]
        )
        toggle.js_on_change(
            "active",
            CustomJS(
                args=dict(overlays=overlays),
                code="""
            for (const overlay of overlays) {
                overlay.visible = cb_obj.active.includes(0);
            }
        """,
            ),
        )
        items.insert(0, toggle)
    items.append(
        _note(
            "Pan and zoom are linked. Raw panels share a linear color scale "
            "(1st–99.7th percentile); residuals use a symmetric scale "
            "(99.5th percentile of absolute signal). Blue is negative; red "
            "is "
            "positive. Color clipping changes only the display, not the data."
        )
    )
    return items


def observed_image(
    image,
    *,
    title="Model Sky · modeled field",
    eyebrow="Model Sky imagery",
    pixel_scale=0.2,
    wcs=None,
    unit="nJy",
    context="",
):
    """Display one exposure without invented comparison planes."""
    image = _image(image)
    low, high = _bounds(image)
    plot, _, _ = _image_panel(
        image,
        title,
        mapper=LinearColorMapper(palette=_GRAY, low=low, high=high),
        pixel_scale=pixel_scale,
        wcs=wcs,
        unit=unit,
        width=760,
    )
    return column(
        _text(title, context, eyebrow=eyebrow),
        plot,
        _note(
            "Native image pixels. The linear display clips at the 1st "
            "and 99.7th percentiles; hover reports original values."
        ),
        sizing_mode="stretch_width",
        max_width=1200,
    )


def image_triptych(
    science,
    reference,
    difference,
    *,
    mask=None,
    title="",
    pixel_scale=0.2,
    wcs=None,
    unit="nJy",
    context="",
):
    """Compare registered images; ``mask=True`` identifies excluded pixels."""
    return column(
        _text(
            title or "What changed in the sky?",
            context,
            eyebrow="Image evidence",
        ),
        *_triptych(
            science,
            reference,
            difference,
            mask=mask,
            pixel_scale=pixel_scale,
            unit=unit,
            wcs=wcs,
        ),
        sizing_mode="stretch_width",
        max_width=1200,
    )


def trail_view(
    science,
    reference,
    difference,
    mask,
    *,
    profile=None,
    pixel_scale=0.2,
    wcs=None,
    unit="nJy",
    context="",
):
    """Inspect a trail, its exclusion mask and an optional supplied profile.

    A profile mapping contains ``x``, ``science``, ``reference`` and optional
    ``difference`` arrays; ``x_label`` describes the supplied coordinate.
    Without a profile, show the literal mean of each complete image row.
    """
    science, reference, difference = map(
        _image, (science, reference, difference)
    )
    count = np.count_nonzero(mask)
    body = (
        f"{count:,} of {science.size:,} pixels marked for exclusion. "
        + context
    )
    if profile is None:

        def row_mean(values):
            finite = np.isfinite(values)
            count = finite.sum(axis=1)
            return np.divide(
                np.nansum(values, axis=1),
                count,
                out=np.full(len(count), np.nan),
                where=count > 0,
            )

        profile = dict(
            x=np.arange(science.shape[0]),
            science=row_mean(science),
            reference=row_mean(reference),
            difference=row_mean(difference),
            x_label="Cutout y [pixel] · mean over every column",
            y_label=f"Mean signal [{unit}]",
        )
    p = _style(
        figure(
            title="Where does the trail contribute signal?",
            height=240,
            width=1050,
            sizing_mode="stretch_width",
            x_axis_label=str(profile.get("x_label", "Profile coordinate")),
            y_axis_label=str(profile.get("y_label", f"Signal [{unit}]")),
            tools="pan,wheel_zoom,reset,save",
        )
    )
    for key, color in (
        ("science", _AMBER),
        ("reference", _BLUE),
        ("difference", "#9a497c"),
    ):
        if key in profile:
            p.line(
                profile["x"],
                profile[key],
                line_width=2,
                line_color=color,
                legend_label=key.capitalize(),
            )
    p.legend.click_policy = "hide"
    p.legend.location = "top_right"
    return column(
        _text(
            "A satellite crosses the field",
            body,
            eyebrow="01 · Satellite trail",
        ),
        *_triptych(
            science,
            reference,
            difference,
            mask=mask,
            pixel_scale=pixel_scale,
            unit=unit,
            wcs=wcs,
        ),
        p,
        sizing_mode="stretch_width",
        max_width=1200,
    )


def motion_view(
    old,
    new,
    difference,
    *,
    old_xy,
    new_xy,
    uncertainty=None,
    baseline_years=10,
    pixel_scale=0.2,
    wcs=None,
    unit="nJy",
    context="",
    epoch_labels=("Earlier epoch", "Later epoch"),
):
    """Blink epochs and show measured centroids in cutout pixel coordinates.

    ``uncertainty`` optionally maps ``old``/``new`` to (sigma_x, sigma_y),
    in pixels. Ellipses mark one standard deviation along each coordinate;
    they do not imply a fitted covariance or confidence region.
    """
    old, new, difference = map(_image, (old, new, difference))
    if old.shape != new.shape or old.shape != difference.shape:
        raise ValueError("Motion images must have equal registered shapes.")
    if baseline_years <= 0:
        raise ValueError("baseline_years must be positive.")
    old_xy, new_xy = np.asarray(old_xy, float), np.asarray(new_xy, float)
    displacement = np.linalg.norm(new_xy - old_xy) * pixel_scale
    # Keep the small motion legible without resampling or smoothing pixels.
    if max(old.shape) > 120:
        center = np.rint((old_xy + new_xy) / 2).astype(int)
        x0 = max(0, min(int(center[0]) - 48, old.shape[1] - 96))
        y0 = max(0, min(int(center[1]) - 48, old.shape[0] - 96))
        region = (slice(y0, y0 + 96), slice(x0, x0 + 96))
        old, new, difference = (a[region] for a in (old, new, difference))
        old_xy, new_xy = old_xy - (x0, y0), new_xy - (x0, y0)
        if wcs is not None:
            wcs = wcs.slice(region)
        context = (
            context + f" Native target cutout; parent pixel origin "
            f"({x0}, {y0})."
        ).strip()
    low, high = _bounds(old, new)
    mapper = LinearColorMapper(palette=_GRAY, low=low, high=high)
    ranges = (
        Range1d(-0.5, old.shape[1] - 0.5),
        Range1d(-0.5, old.shape[0] - 0.5),
    )
    blink, display, _ = _image_panel(
        old,
        str(epoch_labels[0]),
        mapper=mapper,
        ranges=ranges,
        pixel_scale=pixel_scale,
        wcs=wcs,
        unit=unit,
        width=490,
    )
    limit = _limit(difference)
    residual, _, _ = _image_panel(
        difference,
        "Later − matched earlier",
        ranges=ranges,
        mapper=LinearColorMapper(palette=_DIVERGING, low=-limit, high=limit),
        pixel_scale=pixel_scale,
        wcs=wcs,
        unit=unit,
        width=490,
    )
    for plot in (blink, residual):
        for key, point, color, label in (
            ("old", old_xy, _AMBER, str(epoch_labels[0])),
            ("new", new_xy, "#00bed6", str(epoch_labels[1])),
        ):
            plot.scatter(
                [point[0]],
                [point[1]],
                marker="circle",
                size=17,
                fill_alpha=0,
                line_width=2,
                line_color=color,
                legend_label=label,
            )
            if uncertainty is not None and key in uncertainty:
                sx, sy = uncertainty[key]
                plot.ellipse(
                    [point[0]],
                    [point[1]],
                    width=[2 * sx],
                    height=[2 * sy],
                    fill_alpha=0.15,
                    fill_color=color,
                    line_color=color,
                )
        plot.add_layout(
            Arrow(
                x_start=old_xy[0],
                y_start=old_xy[1],
                x_end=new_xy[0],
                y_end=new_xy[1],
                line_color="#00bed6",
                end=NormalHead(
                    size=8, fill_color="#00bed6", line_color="#00bed6"
                ),
            )
        )
        plot.legend.label_text_font_size = "11px"
        plot.legend.background_fill_alpha = 0.85
        plot.legend.location = "top_left"
    frames = ColumnDataSource(dict(image=[old, new]))
    epochs = RadioButtonGroup(labels=list(map(str, epoch_labels)), active=0)
    epochs.js_on_change(
        "active",
        CustomJS(
            args=dict(
                display=display,
                frames=frames,
                plot=blink,
                labels=list(epoch_labels),
            ),
            code="""
            const i = cb_obj.active;
            display.data.image = [frames.data.image[i]];
            display.change.emit();
            plot.title.text = labels[i];
        """,
        ),
    )
    play = Button(
        label="Blink for 8 seconds", button_type="primary", width=180
    )
    play.js_on_click(
        CustomJS(
            args=dict(epochs=epochs, button=play),
            code="""
        if (button.disabled) return;
        button.disabled = true;
        let steps = 0;
        const timer = setInterval(() => {
            epochs.active = 1 - epochs.active;
            if (++steps >= 16) {
                clearInterval(timer); button.disabled = false;
            }
        }, 500);
    """,
        )
    )
    return column(
        _text(
            "The star moves; the field stays put",
            context,
            eyebrow="02 · Controlled proper motion",
        ),
        _note(
            f"Centroid displacement: {displacement:.3f} arcsec "
            f"over {baseline_years:g} years "
            f"→ {1000 * displacement / baseline_years:.1f} mas/year. "
            "Positions are supplied measurements. The time baseline "
            "belongs to the injection experiment."
        ),
        row(epochs, play, sizing_mode="stretch_width"),
        row(blink, residual, sizing_mode="stretch_width"),
        _note(
            "Raw epochs share one linear scale; the difference has a "
            "symmetric scale about zero. The blue/red dipole records "
            "displaced light. Amber/cyan rings mark the two centroids. "
            "Pan and zoom are linked; the blink uses the native image pixels."
        ),
        sizing_mode="stretch_width",
        max_width=1200,
    )


def transit_view(
    times_minutes,
    flux,
    flux_error,
    model,
    *,
    transit_model,
    stamps,
    difference_stamps,
    valid=None,
    check_flux=None,
    reference_stamp=None,
    summary=None,
    unit="Relative flux",
    pixel_scale=0.2,
    image_unit="nJy",
    wcs=None,
):
    """Link light curve, model residual, stamps and in-minus-baseline mean.

    Arrays share cadence order and units. Excluded points remain visible
    but are excluded from both means. ``model`` is the supplied fit,
    not a fit performed here. ``transit_model`` is its baseline-normalized
    transit component: membership uses half-depth and the baseline uses
    less than 5% of depth. The full model remains the light-curve line.
    """
    times, flux, error, model = map(
        lambda a: np.asarray(a, dtype=float),
        (times_minutes, flux, flux_error, model),
    )
    n = len(times)
    transit_model = np.asarray(transit_model, dtype=float)
    stamps = np.asarray(stamps, dtype=np.float32)
    differences = np.asarray(difference_stamps, dtype=np.float32)
    if n < 2 or any(
        a.shape != (n,) for a in (times, flux, error, model, transit_model)
    ):
        raise ValueError(
            "Light-curve arrays must be 1D with at least two cadences."
        )
    if (
        stamps.ndim != 3
        or differences.shape != stamps.shape
        or len(stamps) != n
    ):
        raise ValueError(
            "Stamp arrays must have equal (cadence, y, x) shapes."
        )
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(
            "times_minutes must be finite and strictly increasing."
        )
    if np.any(error[np.isfinite(error)] < 0):
        raise ValueError("Flux uncertainties must be nonnegative.")
    valid = (
        np.ones(n, bool) if valid is None else np.asarray(valid, dtype=bool)
    )
    if valid.shape != (n,):
        raise ValueError("valid must have one Boolean per cadence.")
    usable = (
        valid
        & np.isfinite(flux)
        & np.isfinite(error)
        & np.isfinite(model)
        & np.isfinite(transit_model)
    )
    model_finite = transit_model[np.isfinite(transit_model)]
    if not model_finite.size:
        raise ValueError(
            "The supplied model needs at least one finite value."
        )
    floor, ceiling = np.min(model_finite), np.max(model_finite)
    in_transit = usable & (transit_model < (floor + ceiling) / 2)
    baseline = usable & (transit_model > ceiling - 0.05 * (ceiling - floor))
    coadd = (
        np.nanmean(differences[in_transit], axis=0)
        - np.nanmean(differences[baseline], axis=0)
        if in_transit.any() and baseline.any()
        else np.full(stamps.shape[1:], np.nan, dtype=np.float32)
    )
    first = (
        int(np.flatnonzero(in_transit)[len(np.flatnonzero(in_transit)) // 2])
        if in_transit.any()
        else 0
    )
    source = ColumnDataSource(
        dict(
            time=times,
            flux=flux,
            error=error,
            model=model,
            lower=flux - error,
            upper=flux + error,
            residual=flux - model,
            res_lower=flux - model - error,
            res_upper=flux - model + error,
            cadence=np.arange(n),
            status=np.where(usable, "Used", "Excluded"),
            color=np.where(usable, _BLUE, _AMBER),
            marker=np.where(usable, "circle", "x"),
        )
    )
    lc = _style(
        figure(
            title="How much light is missing?",
            height=310,
            width=1050,
            sizing_mode="stretch_width",
            y_axis_label=unit,
            tools="pan,wheel_zoom,box_zoom,tap,reset,save",
        )
    )
    lc.segment(
        "time",
        "lower",
        "time",
        "upper",
        source=source,
        line_color="color",
        line_alpha=0.55,
    )
    lc.line(
        times,
        model,
        line_color="#ab3d75",
        line_width=2,
        legend_label="Fitted transit model",
    )
    points = lc.scatter(
        "time",
        "flux",
        source=source,
        size=7,
        marker="marker",
        color="color",
        alpha=0.85,
        selection_color="#ec8c18",
        nonselection_alpha=0.6,
        legend_label="Measured flux (amber × = excluded)",
    )
    if check_flux is not None:
        check = np.asarray(check_flux, float)
        if check.shape != (n,):
            raise ValueError(
                "check_flux needs one value per cadence in the same units."
            )
        lc.line(
            times,
            check,
            line_color="#657385",
            line_alpha=0.75,
            line_width=1.5,
            legend_label="Check star",
        )
    lc.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=[
                ("Cadence", "@cadence"),
                ("Time [min]", "@time{0.00}"),
                (unit, "@flux{0.00000}"),
                ("1σ uncertainty", "@error{0.00000}"),
                ("Use", "@status"),
            ],
        )
    )
    lc.legend.location = "top_right"
    lc.legend.label_text_font_size = "11px"
    lc.legend.click_policy = "hide"
    residual = _style(
        figure(
            title="Measured − model",
            height=180,
            width=1050,
            sizing_mode="stretch_width",
            x_range=lc.x_range,
            x_axis_label="Time from mid-transit [minutes]",
            y_axis_label=f"Residual [{unit}]",
            tools="pan,wheel_zoom,tap,reset,save",
        )
    )
    residual.segment(
        "time",
        "res_lower",
        "time",
        "res_upper",
        source=source,
        line_color="color",
        line_alpha=0.5,
    )
    residual.scatter(
        "time",
        "residual",
        source=source,
        size=6,
        marker="marker",
        color="color",
        nonselection_alpha=0.6,
    )
    residual.add_layout(
        Span(
            location=0,
            dimension="width",
            line_color="#8190a3",
            line_dash="dashed",
        )
    )
    cursors = []
    for plot in (lc, residual):
        cursor = Span(
            location=times[first],
            dimension="height",
            line_color=_AMBER,
            line_width=2,
            line_alpha=0.65,
        )
        plot.add_layout(cursor)
        cursors.append(cursor)
    low, high = _bounds(stamps)
    raw_mapper = LinearColorMapper(palette=_GRAY, low=low, high=high)
    limit = _limit(differences)
    diff_mapper = LinearColorMapper(
        palette=_DIVERGING, low=-limit, high=limit, nan_color="#b8c3d1"
    )
    ranges = (
        Range1d(-0.5, stamps.shape[2] - 0.5),
        Range1d(-0.5, stamps.shape[1] - 0.5),
    )
    image_panels, image_sources = [], []
    for image, title, mapper in (
        (stamps[first], "Selected cadence", raw_mapper),
        (differences[first], "Selected difference", diff_mapper),
        (
            coadd,
            "In-transit − baseline mean",
            diff_mapper,
        ),
    ):
        panel, image_source, _ = _image_panel(
            image,
            title,
            mapper=mapper,
            ranges=ranges,
            pixel_scale=pixel_scale,
            unit=image_unit,
            wcs=wcs,
        )
        image_panels.append(panel)
        image_sources.append(image_source)
    slider = Slider(
        start=0,
        end=n - 1,
        value=first,
        step=1,
        title="Selected cadence · tap a point or move the slider",
        sizing_mode="stretch_width",
        name="cadence_slider",
    )
    frames = ColumnDataSource(
        dict(image=list(stamps), difference=list(differences))
    )
    readout = Div(sizing_mode="stretch_width", name="cadence_readout")

    def caption(i):
        measurement = (
            f"flux = {flux[i]:.5f} ± {error[i]:.5f}"
            if np.isfinite(flux[i]) and np.isfinite(error[i])
            else "flux unavailable after masking"
        )
        return (
            f"Cadence {i} · t = {times[i]:.2f} min · "
            f"{measurement} · "
            f"{'Used' if usable[i] else 'Excluded'}"
        )

    readout.text = caption(first)
    slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                source=source,
                frames=frames,
                raw=image_sources[0],
                diff=image_sources[1],
                cursors=cursors,
                readout=readout,
            ),
            code="""
        const i = Math.round(cb_obj.value), d = source.data;
        raw.data.image = [frames.data.image[i]];
        diff.data.image = [frames.data.difference[i]];
        raw.change.emit(); diff.change.emit();
        for (const cursor of cursors) cursor.location = d.time[i];
        const measurement = Number.isFinite(d.flux[i]) &&
            Number.isFinite(d.error[i]) ?
            `flux = ${d.flux[i].toFixed(5)} ± ${d.error[i].toFixed(5)}` :
            'flux unavailable after masking';
        readout.text = `Cadence ${i} · t = ${d.time[i].toFixed(2)} min · ` +
            `${measurement} · ${d.status[i]}`;
    """,
        ),
    )
    source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(slider=slider),
            code="""
        const indices = cb_obj.indices;
        if (indices.length) slider.value = indices[indices.length - 1];
    """,
        ),
    )
    details = (
        ""
        if summary is None
        else (
            " · ".join(f"{k}: {v}" for k, v in summary.items())
            if isinstance(summary, Mapping)
            else str(summary)
        )
    )
    extras = []
    if reference_stamp is not None:
        ref, _, _ = _image_panel(
            reference_stamp,
            "Out-of-transit reference",
            mapper=raw_mapper,
            pixel_scale=pixel_scale,
            unit=image_unit,
            wcs=wcs,
            width=350,
        )
        extras = [
            row(
                ref,
                _note(
                    "The supplied reference uses the cadence stamps' scale. "
                    "It is not reconstructed from a difference image."
                ),
                sizing_mode="stretch_width",
            )
        ]
    return column(
        _text(
            "A smaller shadow, on the same star",
            details,
            eyebrow="03 · Controlled transit",
        ),
        lc,
        residual,
        slider,
        readout,
        row(*image_panels, sizing_mode="stretch_width"),
        _note(
            "Image scales stay fixed across cadences. The difference and "
            "stack share a symmetric scale: dimming is blue. "
            f"Stack: {in_transit.sum()} in-transit minus "
            f"{baseline.sum()} baseline cadences, all valid. The baseline "
            "mean removes persistent residual structure. Model-selected "
            "intervals (below half-depth / within 5% of full light) "
            "illustrate the fit; this is not an independent detection or "
            "significance map. Error "
            "bars show supplied 1σ uncertainties. The light curve is a "
            "controlled injection."
        ),
        *extras,
        sizing_mode="stretch_width",
        max_width=1200,
    )


def export_html(
    layout, path, *, title="cuPhoton · A changing sky", sources=None
):
    """Write a self-contained HTML document with inline BokehJS and data."""
    import mistune

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    references = Path(
        sources or Path(__file__).with_name("sources.md")
    ).read_text(encoding="utf-8")
    footer = Div(
        text=(
            "<details><summary>Sources and further reading</summary>"
            + mistune.html(references)
            + "</details>"
        ),
        styles={"color": _INK, "font-size": "13px", "padding": "16px"},
        sizing_mode="stretch_width",
        max_width=1200,
    )
    document = column(layout, footer, sizing_mode="stretch_width")
    try:
        path.write_text(
            file_html(document, INLINE, title=title), encoding="utf-8"
        )
    finally:
        # file_html gives an unowned layout a persistent Document. Release
        # this export-only wrapper so the caller can show its layout inline.
        if document.document is not None:
            document.document.remove_root(document)
    return path


def export_png(layout, path, *, scale_factor=2):
    """Export a native high-DPI PNG using preparation-only Chrome tooling.

    Selenium Manager resolves a matching driver for installed Chrome.
    Its cache can be relocated with ``SE_CACHE_PATH``; the first export
    may download the driver. HTML and notebook viewing do not use it.
    """
    from bokeh.io import export_png as bokeh_export_png
    from selenium import webdriver

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--hide-scrollbars")
    options.add_argument(f"--force-device-scale-factor={scale_factor}")
    options.add_argument("--force-color-profile=srgb")
    # A fixed outer width keeps responsive children independent of the
    # initial headless browser viewport, which can be only 400 CSS pixels.
    document = column(layout, width=1200)
    with webdriver.Chrome(options=options) as driver:
        bokeh_export_png(
            document,
            filename=str(path),
            scale_factor=scale_factor,
            webdriver=driver,
            timeout=30,
        )
    return path
