# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presentation-only styling for the Model Sky notebook figures."""

from bokeh.io import show
from bokeh.models import (
    Axis,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    Glyph,
    Grid,
    Legend,
    Plot,
    Title,
    Toolbar,
    Widget,
)

_BACKGROUND = "#10180d"
_TEXT = "#f0f4eb"
_MUTED = "#b7c9a9"
_BORDER = "#34472b"
_TEXT_COLORS = {
    "#172b45": _TEXT,
    "#52647c": _MUTED,
    "#f0f6fa": "#182310",
    "#f7f9fc": _BACKGROUND,
}
_GLYPH_COLORS = {
    "#007f99": "#67d5e5",
    "#ab3d75": "#ed91bd",
    "#657385": "#82938a",
    "#8190a3": "#93a6b8",
}
_WIDGET_CSS = """
:host { color: #f0f4eb; color-scheme: dark; }
.bk-btn { color: #f0f4eb; background: #1c2815; border-color: #465d35; }
.bk-btn:hover { background: #2b3d1c; }
.bk-btn.bk-active, .bk-btn-primary {
  color: #0b1008; background: #b5ed39; border-color: #b5ed39;
}
.bk-input { color: #f0f4eb; background: #121a0e; border-color: #465d35; }
.bk-slider-title, .bk-input-group { color: #f0f4eb; }
.bk-toolbar-button { background-color: #1c2815; }
"""


def _text_colors(text):
    for before, after in _TEXT_COLORS.items():
        text = text.replace(before, after)
    return text


def show_dark(layout):
    """Show a dark Bokeh layout, preserving data and scientific palettes."""
    for model in layout.references():
        if isinstance(model, Plot):
            model.background_fill_color = _BACKGROUND
            model.border_fill_color = _BACKGROUND
            model.outline_line_color = _BORDER
        elif isinstance(model, Axis):
            model.axis_label_text_color = _MUTED
            model.major_label_text_color = _MUTED
            model.axis_line_color = _BORDER
            model.major_tick_line_color = _BORDER
        elif isinstance(model, Grid):
            model.grid_line_color = _BORDER
            model.grid_line_alpha = 0.5
        elif isinstance(model, Title):
            model.text_color = _TEXT
        elif isinstance(model, ColorBar):
            model.background_fill_color = _BACKGROUND
            model.major_label_text_color = _MUTED
            model.title_text_color = _TEXT
            model.major_tick_line_color = _MUTED
        elif isinstance(model, Legend):
            model.background_fill_color = _BACKGROUND
            model.label_text_color = _TEXT
            model.border_line_color = _BORDER
        elif isinstance(model, Div):
            model.text = _text_colors(model.text)
            model.styles = {
                key: _text_colors(value) if isinstance(value, str) else value
                for key, value in model.styles.items()
            }
        elif isinstance(model, CustomJS):
            # Dynamic readouts use the same explicit CSS colors as the Divs.
            model.code = _text_colors(model.code)
        if isinstance(model, ColumnDataSource) and "color" in model.data:
            colors = model.data["color"]
            if all(isinstance(color, str) for color in colors):
                model.data = {
                    **model.data,
                    "color": [
                        _GLYPH_COLORS.get(color, color) for color in colors
                    ],
                }
        if isinstance(model, Glyph):
            for attr in ("line_color", "fill_color"):
                if attr in model.properties():
                    color = getattr(model, attr)
                    if isinstance(color, str) and color in _GLYPH_COLORS:
                        setattr(model, attr, _GLYPH_COLORS[color])
        if isinstance(model, (Widget, Toolbar)):
            model.stylesheets = [*model.stylesheets, _WIDGET_CSS]
    return show(layout)
