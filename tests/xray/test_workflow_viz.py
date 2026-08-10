# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from cuphoton.core.cli import run_component
from cuphoton.xray.linear_prediction import synthetic_trace_batch
from cuphoton.xray.workflow_viz import (
    _publishable_manifest,
    build_workflow_viz,
    load_workflow_bundle,
)


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def test_build_workflow_viz_from_trace_dir_writes_bundle(tmp_path):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    output = tmp_path / "workflow.html"
    bundle = tmp_path / "workflow-bundle.npz"

    result = build_workflow_viz(
        output=output,
        trace_dir=trace_dir,
        bundle_output=bundle,
        components=4,
    )

    assert result.html_path == output
    assert result.bundle_path == bundle
    assert result.source_kind == "trace-derived workflow review"
    assert result.row_count == 3
    assert output.exists()
    assert bundle.exists()

    loaded = load_workflow_bundle(bundle)
    assert loaded.manifest["kind"] == "xray-workflow-viz-bundle"
    assert loaded.manifest["version"] == 3
    assert loaded.manifest["trace_source_label"] == trace_dir.name
    assert loaded.manifest["trace_files"] == [
        "trace-y0.npz",
        "trace-y16.npz",
        "trace-y32.npz",
    ]
    assert "trace_dir" not in loaded.manifest
    assert str(tmp_path) not in json.dumps(loaded.manifest)
    assert loaded.detector_image.shape == (3, 72)
    assert loaded.trace_matrix.shape == loaded.reconstruction_matrix.shape
    assert loaded.raw_fft_matrix.shape[0] == 3

    text = output.read_text(encoding="utf-8")
    assert "Detector Image: Laser On-Off" in text
    assert "Time Profiles and Fits" in text
    assert "FFT Spectra of Time Profiles" in text
    assert "FFT Spectra of Fits" in text
    assert "Phonon Dispersion" in text
    assert "Subset of Selected Frequencies" in text
    assert "Subset of Fitted Frequencies" in text
    assert str(tmp_path) not in text


def test_load_workflow_bundle_rejects_unknown_version(tmp_path):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    bundle = tmp_path / "workflow-bundle.npz"
    build_workflow_viz(
        output=tmp_path / "workflow.html",
        trace_dir=trace_dir,
        bundle_output=bundle,
        components=4,
    )
    with np.load(bundle, allow_pickle=False) as loaded:
        payload = {name: np.asarray(loaded[name]) for name in loaded.files}
    manifest = json.loads(str(payload["manifest"].item()))
    manifest["version"] = 999
    payload["manifest"] = json.dumps(manifest)
    np.savez(bundle, **payload)

    with pytest.raises(ValueError, match="unsupported.*version"):
        load_workflow_bundle(bundle)


def test_publishable_manifest_reduces_source_paths_to_labels(tmp_path):
    manifest = {
        "h5dir": str(tmp_path / "hdf5"),
        "fon": str(tmp_path / "hdf5" / "on.h5"),
        "foff": str(tmp_path / "hdf5" / "off.h5"),
        "trace_dir": str(tmp_path / "traces"),
        "detector_artifact_dir": str(tmp_path / "activity"),
        "phonon_detector_artifact_dir": str(tmp_path / "phonon"),
        "row_count": 3,
    }

    published = _publishable_manifest(manifest)

    assert published == {
        "input_directory_label": "hdf5",
        "on_file": "on.h5",
        "off_file": "off.h5",
        "trace_source_label": "traces",
        "detector_artifact_label": "activity",
        "phonon_detector_artifact_label": "phonon",
        "row_count": 3,
    }
    assert str(tmp_path) not in json.dumps(published)


def test_workflow_viz_cli_renders_existing_bundle(tmp_path, capsys):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    bundle = tmp_path / "workflow-bundle.npz"
    first_output = tmp_path / "first.html"
    second_output = tmp_path / "second.html"
    build_workflow_viz(
        output=first_output,
        trace_dir=trace_dir,
        bundle_output=bundle,
        components=4,
    )

    assert (
        main(
            [
                "workflow-viz",
                "--bundle",
                str(bundle),
                "--output",
                str(second_output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"workflow_viz={second_output}" in captured.out
    assert f"bundle={bundle}" in captured.out
    assert "source=trace-derived workflow review" in captured.out
    assert "rows=3" in captured.out
    assert second_output.exists()
    assert str(tmp_path) not in second_output.read_text(encoding="utf-8")


def test_workflow_viz_trace_dir_with_detector_activity_image(tmp_path):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    artifact_dir = _write_detector_artifacts(tmp_path)
    output = tmp_path / "workflow.html"
    bundle = tmp_path / "workflow-bundle.npz"

    result = build_workflow_viz(
        output=output,
        trace_dir=trace_dir,
        detector_artifact_dir=artifact_dir,
        x_value=1,
        bundle_output=bundle,
        components=4,
        phonon_amp_threshold=4.0,
    )

    assert result.source_kind == "detector workflow review"
    loaded = load_workflow_bundle(bundle)
    assert loaded.detector_image.shape == (4, 3)
    assert loaded.manifest["detector_panel_title"] == (
        "Detector Image: Fitted Activity"
    )
    assert loaded.manifest["detector_x_axis_label"] == "pixel x"
    assert loaded.manifest["detector_marker_x"] == 1.0
    assert loaded.manifest["source_note"] == (
        "detector artifact filtered amplitude sum; x=1"
    )
    assert loaded.manifest["phonon_amp_threshold"] == 4.0
    assert loaded.manifest["filtered_phonon_x_value"] == 1
    assert loaded.manifest["filtered_phonon_count"] == 1
    assert loaded.manifest["trace_source_label"] == trace_dir.name
    assert loaded.manifest["detector_artifact_label"] == artifact_dir.name
    assert loaded.manifest["phonon_detector_artifact_label"] == (
        artifact_dir.name
    )
    assert str(tmp_path) not in json.dumps(loaded.manifest)
    assert loaded.filtered_phonon_row.tolist() == [3.0]
    np.testing.assert_allclose(loaded.filtered_phonon_frequency, [0.131])
    assert loaded.filtered_phonon_mode.tolist() == [1]

    text = output.read_text(encoding="utf-8")
    assert "Detector Image: Fitted Activity" in text
    assert "Filtered Phonon Dispersion x=1, amp&gt;4" in text


def test_workflow_viz_uses_detector_artifact_origin(tmp_path):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    artifact_dir = _write_detector_artifacts(tmp_path, roi_lower=(10, 20))
    output = tmp_path / "workflow-cropped.html"
    bundle = tmp_path / "workflow-cropped-bundle.npz"

    build_workflow_viz(
        output=output,
        trace_dir=trace_dir,
        detector_artifact_dir=artifact_dir,
        x_value=11,
        bundle_output=bundle,
        components=4,
        phonon_amp_threshold=4.0,
    )

    loaded = load_workflow_bundle(bundle)
    assert loaded.manifest["detector_x0"] == 10.0
    assert loaded.manifest["detector_y0"] == 20.0
    assert loaded.manifest["detector_marker_x"] == 11.0
    assert loaded.manifest["source_note"] == (
        "detector artifact filtered amplitude sum; x=11"
    )
    assert loaded.manifest["filtered_phonon_x_value"] == 11
    assert loaded.filtered_phonon_row.tolist() == [23.0]
    np.testing.assert_allclose(loaded.filtered_phonon_frequency, [0.131])

    text = output.read_text(encoding="utf-8")
    assert "Filtered Phonon Dispersion x=11, amp&gt;4" in text


def test_workflow_viz_can_use_separate_phonon_artifact(tmp_path):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    activity_dir = _write_detector_artifacts(tmp_path, name="activity")
    phonon_dir = _write_detector_artifacts(
        tmp_path,
        name="phonon",
        freq_offset=1.0,
    )
    output = tmp_path / "workflow-separate-phonon.html"
    bundle = tmp_path / "workflow-separate-phonon-bundle.npz"

    build_workflow_viz(
        output=output,
        trace_dir=trace_dir,
        detector_artifact_dir=activity_dir,
        phonon_detector_artifact_dir=phonon_dir,
        x_value=1,
        bundle_output=bundle,
        components=4,
        phonon_amp_threshold=4.0,
    )

    loaded = load_workflow_bundle(bundle)
    assert loaded.detector_image.shape == (4, 3)
    assert loaded.manifest["detector_artifact_label"] == activity_dir.name
    assert loaded.manifest["phonon_detector_artifact_label"] == (
        phonon_dir.name
    )
    assert "detector_artifact_dir" not in loaded.manifest
    assert "phonon_detector_artifact_dir" not in loaded.manifest
    assert str(tmp_path) not in json.dumps(loaded.manifest)
    assert loaded.manifest["filtered_phonon_count"] == 1
    np.testing.assert_allclose(loaded.filtered_phonon_frequency, [1.131])


def test_phonon_viz_cli_renders_workflow_bundle(tmp_path, capsys):
    pytest.importorskip("bokeh")
    trace_dir = _write_trace_batch(tmp_path)
    bundle = tmp_path / "workflow-bundle.npz"
    workflow_output = tmp_path / "workflow.html"
    phonon_output = tmp_path / "phonon.html"
    build_workflow_viz(
        output=workflow_output,
        trace_dir=trace_dir,
        bundle_output=bundle,
        components=4,
    )

    assert (
        main(
            [
                "phonon-viz",
                "--workflow-bundle",
                str(bundle),
                "--output",
                str(phonon_output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert f"phonon_viz={phonon_output}" in captured.out
    assert "source=workflow bundle" in captured.out
    assert "rows=3" in captured.out
    assert phonon_output.exists()
    phonon_html = phonon_output.read_text(encoding="utf-8")
    assert "Phonon dispersion from workflow bundle" in phonon_html
    assert bundle.name in phonon_html
    assert str(tmp_path) not in phonon_html


def test_workflow_viz_hdf5_detector_workbench(tmp_path):
    pytest.importorskip("bokeh")
    _write_hdf5_pair(tmp_path, samples=72)
    output = tmp_path / "hdf5-workflow.html"
    bundle = tmp_path / "hdf5-workflow-bundle.npz"

    result = build_workflow_viz(
        output=output,
        bundle_output=bundle,
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        roi_lower=(1, 1),
        roi_dim=(3, 2),
        drop_leading=0,
        chunk_frames=9,
        components=4,
        max_traces=2,
    )

    assert result.source_kind == "hdf5 detector workflow"
    assert result.row_count == 2
    assert output.exists()
    loaded = load_workflow_bundle(bundle)
    assert loaded.manifest["input_directory_label"] == tmp_path.name
    assert loaded.manifest["on_file"] == "on.h5"
    assert loaded.manifest["off_file"] == "off.h5"
    assert "h5dir" not in loaded.manifest
    assert "fon" not in loaded.manifest
    assert "foff" not in loaded.manifest
    assert str(tmp_path) not in json.dumps(loaded.manifest)
    text = output.read_text(encoding="utf-8")
    assert "Detector Image: Laser On-Off" in text
    assert "hdf5 detector workflow" in text
    assert str(tmp_path) not in text


def _write_trace_batch(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    time, traces = synthetic_trace_batch(samples=72, traces=3)
    for row_y, trace in zip((0, 16, 32), traces, strict=True):
        summary = {"row_y": row_y, "roi_lower": [256, 0]}
        np.savez(
            trace_dir / f"trace-y{row_y}.npz",
            time=time,
            trace=trace,
            summary=json.dumps(summary),
        )
    return trace_dir


def _write_hdf5_pair(tmp_path, *, samples: int):
    time, traces = synthetic_trace_batch(samples=samples, traces=4)
    off = np.ones((samples, 4, 5), dtype=np.float64)
    on = np.ones((samples, 4, 5), dtype=np.float64)
    for row_index in range(4):
        scale = 0.02 + row_index * 0.01
        on[:, row_index, :] = 1.0 + scale * traces[row_index][:, None]
    for name, image in (("on.h5", on), ("off.h5", off)):
        with h5py.File(tmp_path / name, "w") as h5:
            h5.create_dataset("ROI", data=np.ones((2, 2), dtype=np.float64))
            h5.create_dataset("bin_count", data=np.ones(samples))
            h5.create_dataset("i0", data=np.ones(samples))
            h5.create_dataset("i0_ipm3", data=np.ones(samples))
            h5.create_dataset("imgs", data=image)
            h5.create_dataset("scan_var", data=time)


def _write_detector_artifacts(
    tmp_path,
    *,
    roi_lower=None,
    name="detector",
    freq_offset=0.0,
):
    artifact_dir = tmp_path / name
    artifact_dir.mkdir()
    amp_sum = np.arange(12, dtype=float).reshape(4, 3)
    shape = (4, 3, 6)
    freq_all = np.zeros(shape, dtype=float)
    amp_all = np.zeros(shape, dtype=float)
    for y in range(shape[0]):
        for x in range(shape[1]):
            freq_all[y, x, 1] = freq_offset + 0.1 + 0.01 * y + 0.001 * x
            amp_all[y, x, 1] = 2.0 + y
    np.save(artifact_dir / "amp_all_sum_filtered.npy", amp_sum)
    np.save(artifact_dir / "freq_all.npy", freq_all)
    np.save(artifact_dir / "amp_all.npy", amp_all)
    if roi_lower is not None:
        (artifact_dir / "manifest.json").write_text(
            json.dumps({"roi_lower": list(roi_lower)}),
            encoding="utf-8",
        )
    return artifact_dir
