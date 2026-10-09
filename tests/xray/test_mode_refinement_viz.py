# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import replace

import numpy as np
import pytest

from cuphoton.xray import mode_refinement_viz as viz
from cuphoton.xray.synthetic_validation import (
    distort_trace,
    synthetic_modes_trace,
)


def test_modes_review_writes_offline_html_and_reports_fit_status(tmp_path):
    pytest.importorskip("bokeh")
    output = tmp_path / "review" / "modes.html"

    report = viz.write_modes_review(output, snr_db=(40.0, 20.0), samples=48)

    assert report.html_path == output
    assert report.trace_count == 2
    assert report.converged_count == 2
    assert report.to_dict() == {
        "html_path": str(output),
        "trace_count": 2,
        "converged_count": 2,
    }
    assert "refinement_converged=2/2" in str(report)
    html = output.read_text(encoding="utf-8")
    assert "Bokeh" in html
    assert "noise-free signal" in html
    assert "residual/noise" in html
    assert re.search(r"<script\b[^>]*\bsrc\s*=", html) is None


@pytest.mark.parametrize("distortion", [("chirp", 0.3), ("glitch", 1.0)])
def test_modes_review_plots_distorted_truth_and_failed_convergence(
    tmp_path, monkeypatch, distortion
):
    pytest.importorskip("bokeh")
    from bokeh.models import Plot

    captured = {}

    def capture_html(layout, resources, title):
        captured["layout"] = layout
        return "<html></html>"

    original_refine = viz.linear_prediction_refined

    def unconverged(*args, **kwargs):
        return replace(original_refine(*args, **kwargs), converged=False)

    monkeypatch.setattr("bokeh.embed.file_html", capture_html)
    monkeypatch.setattr(viz, "linear_prediction_refined", unconverged)
    report = viz.write_modes_review(
        tmp_path / "modes.html",
        snr_db=(20.0,),
        distortion=distortion,
        samples=48,
    )

    plots = list(captured["layout"].select({"type": Plot}))
    trace_plot = next(p for p in plots if "residual/noise" in p.title.text)
    assert "not converged" in trace_plot.title.text
    assert report.converged_count == 0
    truth = next(
        item.renderers[0]
        for legend in trace_plot.legend
        for item in legend.items
        if item.label.value == "noise-free signal"
    )
    fixture = synthetic_modes_trace(48)
    expected = distort_trace(
        fixture.time, fixture.modes, fixture.constant, *distortion
    )
    np.testing.assert_array_equal(truth.data_source.data["x"], fixture.time)
    np.testing.assert_array_equal(truth.data_source.data["y"], expected)
    assert not np.array_equal(expected, fixture.clean)
    assert any("signal distortion" in p.title.text for p in plots)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"snr_db": ()}, "at least one"),
        ({"snr_db": (np.nan,)}, "must be finite"),
        ({"snr_db": (10000.0,)}, "positive noise"),
        ({"snr_db": (-10000.0,)}, "positive noise"),
        ({"samples": 8}, "at least 16"),
    ],
)
def test_modes_review_rejects_invalid_fixture_before_writing(
    tmp_path, kwargs, message
):
    output = tmp_path / "modes.html"

    with pytest.raises(ValueError, match=message):
        viz.write_modes_review(output, **kwargs)

    assert not output.exists()
