# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xray.linear_prediction import synthetic_trace_batch
from cuphoton.xray.phonon_viz import _load_detector_array, build_phonon_viz


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def test_build_phonon_viz_from_trace_dir(tmp_path):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    output = tmp_path / "phonon.html"

    result = build_phonon_viz(
        output=output,
        trace_dir=trace_dir,
        components=4,
    )

    assert result.html_path == output
    assert result.source_kind == "trace-derived phonon proxy"
    assert result.trace_count == 3
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "XRay Phonon Dispersion" in text
    assert "trace-derived phonon proxy" in text
    assert "Phonon dispersion proxy" in text
    assert str(tmp_path) not in text
    assert trace_dir.name in text


def test_build_phonon_viz_from_detector_artifacts(tmp_path):
    pytest.importorskip("bokeh")
    artifact_dir = _write_detector_artifacts(tmp_path)
    output = tmp_path / "detector-phonon.html"

    result = build_phonon_viz(
        output=output,
        detector_artifact_dir=artifact_dir,
        x_value=1,
        y_start=1,
        y_end=4,
    )

    assert result.source_kind == "detector-wide artifact"
    assert result.trace_count == 3
    assert result.mode_count == 3
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "detector-wide artifact" in text
    assert "Phonon dispersion x=1" in text
    assert "frequency (THz)" in text
    assert str(tmp_path) not in text
    assert artifact_dir.name in text


def test_build_phonon_viz_filters_detector_artifacts(tmp_path):
    pytest.importorskip("bokeh")
    artifact_dir = _write_detector_artifacts(tmp_path)
    output = tmp_path / "detector-phonon-filtered.html"

    result = build_phonon_viz(
        output=output,
        detector_artifact_dir=artifact_dir,
        x_value=1,
        y_start=1,
        y_end=4,
        amp_threshold=4.0,
    )

    assert result.source_kind == "detector-wide filtered artifact"
    assert result.trace_count == 3
    assert result.mode_count == 1
    text = output.read_text(encoding="utf-8")
    assert "detector-wide filtered artifact" in text
    assert "Phonon dispersion x=1, y=[1,4), filtered amp&gt;4" in text
    assert "&gt;4" in text


def test_build_phonon_viz_uses_detector_artifact_origin(tmp_path):
    pytest.importorskip("bokeh")
    artifact_dir = _write_detector_artifacts(tmp_path, roi_lower=(10, 20))
    output = tmp_path / "detector-phonon-cropped.html"

    result = build_phonon_viz(
        output=output,
        detector_artifact_dir=artifact_dir,
        x_value=11,
        y_start=21,
        y_end=24,
    )

    assert result.trace_count == 3
    assert result.mode_count == 3
    text = output.read_text(encoding="utf-8")
    assert "Phonon dispersion x=11, y=[21,24)" in text
    assert "21:24" in text


def test_load_detector_array_uses_memmap(tmp_path):
    artifact_dir = _write_detector_artifacts(tmp_path)

    array = _load_detector_array(artifact_dir / "freq_all.npy")

    assert isinstance(array, np.memmap)
    assert array.shape == (4, 3, 6)


def test_phonon_viz_cli_trace_dir(tmp_path, capsys):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    output = tmp_path / "cli-phonon.html"

    assert (
        main(
            [
                "phonon-viz",
                "--trace-dir",
                str(trace_dir),
                "--components",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"phonon_viz={output}" in captured.out
    assert "source=trace-derived phonon proxy" in captured.out
    assert "rows=3" in captured.out
    assert output.exists()


def _write_trace_batch(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, traces = synthetic_trace_batch(samples=48, traces=3)
    for row_y, trace in zip((0, 16, 32), traces, strict=True):
        summary = {"row_y": row_y, "roi_lower": [256, 0]}
        np.savez(
            trace_dir / f"trace-y{row_y}.npz",
            time=time,
            trace=trace,
            summary=json.dumps(summary),
        )
    return trace_dir


def _write_detector_artifacts(tmp_path, *, roi_lower=None):
    artifact_dir = tmp_path / "detector"
    artifact_dir.mkdir()
    shape = (4, 3, 6)
    freq_all = np.zeros(shape, dtype=float)
    amp_all = np.zeros(shape, dtype=float)
    fft_all = np.zeros(shape, dtype=float)
    fft_freq_all = np.zeros(shape, dtype=float)
    for y in range(shape[0]):
        for x in range(shape[1]):
            freq_all[y, x, 1] = 0.1 + 0.01 * y + 0.001 * x
            amp_all[y, x, 1] = 2.0 + y
            fft_all[y, x, :] = np.linspace(0.1, 1.0, shape[2]) + y * 0.05
            fft_freq_all[y, x, :] = np.linspace(0.0, 0.5, shape[2])
    amp_all[2, 1, 2] = 9.0
    np.save(artifact_dir / "freq_all.npy", freq_all)
    np.save(artifact_dir / "amp_all.npy", amp_all)
    np.save(artifact_dir / "fft_all.npy", fft_all)
    np.save(artifact_dir / "fft_freq_all.npy", fft_freq_all)
    if roi_lower is not None:
        (artifact_dir / "manifest.json").write_text(
            json.dumps({"roi_lower": list(roi_lower)}),
            encoding="utf-8",
        )
    return artifact_dir
