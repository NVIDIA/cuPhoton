# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone export must leave figures available for notebook display."""

import sys
from pathlib import Path

from bokeh.io import notebook, showing
from bokeh.io.state import State
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.plotting import figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_style import show_dark  # noqa: E402
from views import export_html  # noqa: E402


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
