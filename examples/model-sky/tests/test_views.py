# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone export must leave figures available for notebook display."""

import sys
from pathlib import Path

import numpy as np
from bokeh.io import notebook, showing
from bokeh.io.state import State
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.plotting import figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_style import show_dark  # noqa: E402
from views import export_html, transit_view  # noqa: E402


def test_export_then_inline_display(tmp_path, monkeypatch):
    plot = figure()
    plot.scatter([1, 2], [3, 4])
    layout = column(Div(text="Fading-light diagnostic"), plot)
    path = export_html(layout, tmp_path / "view.html")
    assert "Fading-light diagnostic" in path.read_text()
    assert "Bokeh" in path.read_text()

    displayed = []
    monkeypatch.setattr(
        notebook,
        "publish_display_data",
        lambda data, **kwargs: displayed.append(data),
    )
    state = State()
    state.output_notebook(notebook_type="jupyter")
    monkeypatch.setattr(showing, "curstate", lambda: state)
    # Exercise the actual blend export/display sequence. Only the frontend
    # publishing step and notebook output state are supplied by the test.
    show_dark(layout)
    assert any("text/html" in output for output in displayed)
    assert any("application/javascript" in output for output in displayed)


def test_transit_image_selection_ignores_the_fitted_linear_baseline():
    times = np.array([-60, -30, 0, 30, 60])
    component = np.array([1, 0.99, 0.98, 0.99, 1])
    model = (1 + 0.03 * times / 60) * component
    differences = np.broadcast_to(
        np.array([0, 2, -8, 5, 6])[:, None, None], (5, 3, 3)
    )
    layout = transit_view(
        times,
        model,
        np.full(5, 0.001),
        model,
        transit_model=component,
        stamps=np.ones((5, 3, 3)),
        difference_stamps=differences,
    )
    coadd = next(
        p
        for p in layout.select({"type": figure})
        if p.title.text == "In-transit − baseline mean"
    )
    image = next(
        r.data_source.data["image"][0]
        for r in coadd.renderers
        if "image" in getattr(r.data_source, "data", {})
    )
    np.testing.assert_array_equal(image, np.full((3, 3), -11))
