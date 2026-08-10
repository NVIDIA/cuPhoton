# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xray.linear_prediction import synthetic_trace_batch
from cuphoton.xray.validation_viz import build_validation_viz


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def test_build_validation_viz_writes_bokeh_html(tmp_path):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    output = tmp_path / "validation.html"

    result = build_validation_viz(
        output=output,
        trace_dir=trace_dir,
        components=4,
    )

    assert result.html_path == output
    assert result.trace_count == 2
    assert result.fit_count == 2
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "XRay Validation Review" in text
    assert "Reference Plot Coverage" in text
    assert "Trace matrix" in text
    assert "Delay waterfall lineouts" in text
    assert "ROI row activity proxy" in text
    assert "LPF fit quality by trace" in text
    assert "Frequency centers by row" in text
    assert "LPF amplitude and phase by row" in text
    assert "Offset reconstruction lineouts" in text
    assert "Full-detector ROI images" in text
    assert str(tmp_path) not in text
    assert "caller-local-hdf5" in text
    assert "on.h5" in text
    assert "off.h5" in text


def test_validation_viz_cli(tmp_path, capsys):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    log_path = tmp_path / "profile.log"
    log_path.write_text(
        "linearpred_profile: ROI [0:16,0:16] "
        "linearpred_total=2.000000s/1, eigvals_cupy=0.700000s/16\n",
        encoding="utf-8",
    )
    output = tmp_path / "viz.html"

    assert (
        main(
            [
                "validation-viz",
                "--trace-dir",
                str(trace_dir),
                "--profile-log",
                str(log_path),
                "--components",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"validation_viz={output}" in captured.out
    assert "traces=2" in captured.out
    assert "profile_logs=1" in captured.out
    assert output.exists()


def _write_trace_batch(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, traces = synthetic_trace_batch(samples=48, traces=2)
    for row_y, trace in zip((0, 16), traces, strict=True):
        h5dir = tmp_path / "caller-local-hdf5"
        summary = {
            "row_y": row_y,
            "roi_lower": [256, 0],
            "h5dir": str(h5dir),
            "fon": str(h5dir / "on.h5"),
            "foff": str(h5dir / "off.h5"),
        }
        np.savez(
            trace_dir / f"trace-y{row_y}.npz",
            time=time,
            trace=trace,
            summary=json.dumps(summary),
        )
    return trace_dir
