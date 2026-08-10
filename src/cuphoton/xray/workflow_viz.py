# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detector_artifacts import (
    detector_artifact_origin,
    detector_artifact_x_index,
)
from .detector_mask import (
    AxisRange,
)
from .detector_mask import (
    excluded_row_mask as detector_excluded_row_mask,
)
from .validation_viz import (
    TraceRecord,
    _fit_traces,
    _load_trace_records,
    _public_path_label,
)

WORKFLOW_BUNDLE_VERSION = 3

_MANIFEST_PATH_LABELS = {
    "h5dir": "input_directory_label",
    "fon": "on_file",
    "foff": "off_file",
    "trace_dir": "trace_source_label",
    "detector_artifact_dir": "detector_artifact_label",
    "phonon_detector_artifact_dir": "phonon_detector_artifact_label",
}


@dataclass(frozen=True)
class WorkflowVizBundle:
    manifest: dict[str, Any]
    detector_image: np.ndarray
    row_y: np.ndarray
    time: np.ndarray
    trace_matrix: np.ndarray
    reconstruction_matrix: np.ndarray
    residual_matrix: np.ndarray
    raw_fft_frequency: np.ndarray
    raw_fft_matrix: np.ndarray
    fit_fft_frequency: np.ndarray
    fit_fft_matrix: np.ndarray
    frequency_centers: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    chi2: np.ndarray
    selected_model_order: np.ndarray
    mode_count: np.ndarray
    dispersion_frequency: np.ndarray
    dispersion_image: np.ndarray
    filtered_phonon_row: np.ndarray
    filtered_phonon_frequency: np.ndarray
    filtered_phonon_amplitude: np.ndarray
    filtered_phonon_mode: np.ndarray


@dataclass(frozen=True)
class WorkflowVizResult:
    html_path: Path
    bundle_path: Path | None
    source_kind: str
    row_count: int
    mode_count: int


_HTML_TEMPLATE = """
{% block postamble %}
<style>
  :root {
    --xray-bg: #f7f8fa;
    --xray-text: #152032;
    --xray-muted: #667085;
    --xray-border: #d9e2ec;
    --xray-blue: #315f9f;
    --xray-green: #14866d;
    --xray-orange: #c87519;
    --xray-red: #c64740;
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
    max-width: 1560px;
    padding: 12px 14px 28px;
  }
  .xray-workflow-header {
    border-bottom: 1px solid var(--xray-border);
    margin-bottom: 10px;
    padding: 2px 0 10px;
  }
  .xray-workflow-header h1 {
    font-size: 22px;
    line-height: 1.16;
    margin: 0 0 8px;
  }
  .xray-workflow-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .xray-workflow-pill {
    background: #ffffff;
    border: 1px solid var(--xray-border);
    border-radius: 6px;
    color: var(--xray-text);
    display: inline-block;
    font-size: 12px;
    font-weight: 640;
    padding: 5px 8px;
  }
  .xray-workflow-pill strong {
    color: var(--xray-muted);
    font-weight: 620;
    margin-right: 5px;
  }
  .bk-root .bk-Row {
    gap: 8px !important;
  }
  .bk-root .bk-Column {
    gap: 8px !important;
  }
  @media (max-width: 1120px) {
    .bk-root .bk-Row {
      flex-direction: column !important;
    }
  }
</style>
{% endblock %}
{% block contents %}
<div class="xray-shell">
  {{ plot_div | indent(2) }}
</div>
{% endblock %}
"""


def build_workflow_viz(
    *,
    output: Path | str,
    bundle: Path | None = None,
    bundle_output: Path | None = None,
    trace_paths: tuple[Path, ...] = (),
    trace_dir: Path | None = None,
    h5dir: Path | None = None,
    fon: str | None = None,
    foff: str | None = None,
    roi_lower: tuple[int, int] | None = None,
    roi_dim: tuple[int, int] | None = None,
    detector_artifact_dir: Path | None = None,
    phonon_detector_artifact_dir: Path | None = None,
    x_value: int | None = None,
    exclude_y: tuple[AxisRange, ...] = (),
    drop_leading: int = 1,
    chunk_frames: int = 16,
    reference_shift: bool = True,
    title: str = "XRay Workflow Workbench",
    components: int = 30,
    roots_backend: str = "eigvals",
    max_traces: int = 48,
    max_points: int = 60_000,
    phonon_amp_threshold: float | None = None,
) -> WorkflowVizResult:
    """Write a standalone Figure-3-style workflow dashboard."""

    try:
        from bokeh.embed import file_html
        from bokeh.resources import INLINE
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bokeh is required for workflow-viz; run "
            "'uv sync --extra viz' for development or install "
            "'cuphoton[viz]'"
        ) from exc

    if bundle is not None:
        if _has_source_inputs(
            trace_paths=trace_paths,
            trace_dir=trace_dir,
            h5dir=h5dir,
            fon=fon,
            foff=foff,
        ):
            raise ValueError("--bundle cannot be combined with data inputs")
        workflow = load_workflow_bundle(bundle)
        written_bundle = Path(bundle)
    else:
        workflow = build_workflow_bundle(
            trace_paths=trace_paths,
            trace_dir=trace_dir,
            h5dir=h5dir,
            fon=fon,
            foff=foff,
            roi_lower=roi_lower,
            roi_dim=roi_dim,
            detector_artifact_dir=detector_artifact_dir,
            phonon_detector_artifact_dir=phonon_detector_artifact_dir,
            x_value=x_value,
            exclude_y=exclude_y,
            drop_leading=drop_leading,
            chunk_frames=chunk_frames,
            reference_shift=reference_shift,
            components=components,
            roots_backend=roots_backend,
            max_traces=max_traces,
            max_points=max_points,
            phonon_amp_threshold=phonon_amp_threshold,
        )
        written_bundle = None
        if bundle_output is not None:
            written_bundle = Path(bundle_output)
            write_workflow_bundle(workflow, written_bundle)

    layout = _workflow_layout(workflow, title=title)
    html_path = Path(output)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        file_html(layout, INLINE, title, template=_HTML_TEMPLATE),
        encoding="utf-8",
    )
    return WorkflowVizResult(
        html_path=html_path,
        bundle_path=written_bundle,
        source_kind=str(workflow.manifest["source_kind"]),
        row_count=int(workflow.row_y.shape[0]),
        mode_count=int(np.sum(workflow.mode_count)),
    )


def build_workflow_bundle(
    *,
    trace_paths: tuple[Path, ...] = (),
    trace_dir: Path | None = None,
    h5dir: Path | None = None,
    fon: str | None = None,
    foff: str | None = None,
    roi_lower: tuple[int, int] | None = None,
    roi_dim: tuple[int, int] | None = None,
    detector_artifact_dir: Path | None = None,
    phonon_detector_artifact_dir: Path | None = None,
    x_value: int | None = None,
    exclude_y: tuple[AxisRange, ...] = (),
    drop_leading: int = 1,
    chunk_frames: int = 16,
    reference_shift: bool = True,
    components: int = 30,
    roots_backend: str = "eigvals",
    max_traces: int = 48,
    max_points: int = 60_000,
    phonon_amp_threshold: float | None = None,
) -> WorkflowVizBundle:
    source_count = sum(
        (
            bool(trace_paths) or trace_dir is not None,
            h5dir is not None or fon is not None or foff is not None,
        )
    )
    if source_count != 1:
        raise ValueError(
            "provide exactly one workflow-viz source: --bundle, trace "
            "input, or --h5dir/--fon/--foff"
        )
    if (
        phonon_detector_artifact_dir is not None
        and detector_artifact_dir is None
    ):
        raise ValueError(
            "--phonon-detector-artifact-dir requires --detector-artifact-dir"
        )
    if h5dir is not None or fon is not None or foff is not None:
        return _build_hdf5_bundle(
            h5dir=h5dir,
            fon=fon,
            foff=foff,
            roi_lower=roi_lower,
            roi_dim=roi_dim,
            exclude_y=exclude_y,
            drop_leading=drop_leading,
            chunk_frames=chunk_frames,
            reference_shift=reference_shift,
            components=components,
            roots_backend=roots_backend,
            max_traces=max_traces,
        )
    return _build_trace_bundle(
        trace_paths=trace_paths,
        trace_dir=trace_dir,
        detector_artifact_dir=detector_artifact_dir,
        phonon_detector_artifact_dir=phonon_detector_artifact_dir,
        x_value=x_value,
        phonon_amp_threshold=phonon_amp_threshold,
        components=components,
        roots_backend=roots_backend,
        max_traces=max_traces,
        max_points=max_points,
    )


def write_workflow_bundle(
    workflow: WorkflowVizBundle,
    output: Path | str,
) -> Path:
    path = Path(output)
    if path.suffix != ".npz":
        raise ValueError("--bundle-output must end with .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        manifest=json.dumps(
            _publishable_manifest(workflow.manifest),
            sort_keys=True,
        ),
        detector_image=workflow.detector_image,
        row_y=workflow.row_y,
        time=workflow.time,
        trace_matrix=workflow.trace_matrix,
        reconstruction_matrix=workflow.reconstruction_matrix,
        residual_matrix=workflow.residual_matrix,
        raw_fft_frequency=workflow.raw_fft_frequency,
        raw_fft_matrix=workflow.raw_fft_matrix,
        fit_fft_frequency=workflow.fit_fft_frequency,
        fit_fft_matrix=workflow.fit_fft_matrix,
        frequency_centers=workflow.frequency_centers,
        amplitudes=workflow.amplitudes,
        phases=workflow.phases,
        chi2=workflow.chi2,
        selected_model_order=workflow.selected_model_order,
        mode_count=workflow.mode_count,
        dispersion_frequency=workflow.dispersion_frequency,
        dispersion_image=workflow.dispersion_image,
        filtered_phonon_row=workflow.filtered_phonon_row,
        filtered_phonon_frequency=workflow.filtered_phonon_frequency,
        filtered_phonon_amplitude=workflow.filtered_phonon_amplitude,
        filtered_phonon_mode=workflow.filtered_phonon_mode,
    )
    return path


def load_workflow_bundle(path: Path | str) -> WorkflowVizBundle:
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as loaded:
        required = {
            "manifest",
            "detector_image",
            "row_y",
            "time",
            "trace_matrix",
            "reconstruction_matrix",
            "residual_matrix",
            "raw_fft_frequency",
            "raw_fft_matrix",
            "fit_fft_frequency",
            "fit_fft_matrix",
            "frequency_centers",
            "amplitudes",
            "phases",
            "chi2",
            "selected_model_order",
            "mode_count",
            "dispersion_frequency",
            "dispersion_image",
        }
        missing = sorted(required.difference(loaded.files))
        if missing:
            raise ValueError(
                "workflow bundle is missing arrays: " + ",".join(missing)
            )
        manifest = _publishable_manifest(_loaded_manifest(loaded["manifest"]))
        if manifest.get("kind") != "xray-workflow-viz-bundle":
            raise ValueError("not an XRay workflow-viz bundle")
        if manifest.get("version") != WORKFLOW_BUNDLE_VERSION:
            raise ValueError(
                "unsupported XRay workflow-viz bundle version: "
                f"{manifest.get('version')!r}; expected "
                f"{WORKFLOW_BUNDLE_VERSION}"
            )
        return WorkflowVizBundle(
            manifest=manifest,
            detector_image=np.asarray(loaded["detector_image"], dtype=float),
            row_y=np.asarray(loaded["row_y"], dtype=int),
            time=np.asarray(loaded["time"], dtype=float),
            trace_matrix=np.asarray(loaded["trace_matrix"], dtype=float),
            reconstruction_matrix=np.asarray(
                loaded["reconstruction_matrix"],
                dtype=float,
            ),
            residual_matrix=np.asarray(
                loaded["residual_matrix"],
                dtype=float,
            ),
            raw_fft_frequency=np.asarray(
                loaded["raw_fft_frequency"],
                dtype=float,
            ),
            raw_fft_matrix=np.asarray(loaded["raw_fft_matrix"], dtype=float),
            fit_fft_frequency=np.asarray(
                loaded["fit_fft_frequency"],
                dtype=float,
            ),
            fit_fft_matrix=np.asarray(loaded["fit_fft_matrix"], dtype=float),
            frequency_centers=np.asarray(
                loaded["frequency_centers"],
                dtype=float,
            ),
            amplitudes=np.asarray(loaded["amplitudes"], dtype=float),
            phases=np.asarray(loaded["phases"], dtype=float),
            chi2=np.asarray(loaded["chi2"], dtype=float),
            selected_model_order=np.asarray(
                loaded["selected_model_order"],
                dtype=int,
            ),
            mode_count=np.asarray(loaded["mode_count"], dtype=int),
            dispersion_frequency=np.asarray(
                loaded["dispersion_frequency"],
                dtype=float,
            ),
            dispersion_image=np.asarray(
                loaded["dispersion_image"],
                dtype=float,
            ),
            filtered_phonon_row=np.asarray(
                (
                    loaded["filtered_phonon_row"]
                    if "filtered_phonon_row" in loaded.files
                    else np.asarray([], dtype=float)
                ),
                dtype=float,
            ),
            filtered_phonon_frequency=np.asarray(
                (
                    loaded["filtered_phonon_frequency"]
                    if "filtered_phonon_frequency" in loaded.files
                    else np.asarray([], dtype=float)
                ),
                dtype=float,
            ),
            filtered_phonon_amplitude=np.asarray(
                (
                    loaded["filtered_phonon_amplitude"]
                    if "filtered_phonon_amplitude" in loaded.files
                    else np.asarray([], dtype=float)
                ),
                dtype=float,
            ),
            filtered_phonon_mode=np.asarray(
                (
                    loaded["filtered_phonon_mode"]
                    if "filtered_phonon_mode" in loaded.files
                    else np.asarray([], dtype=int)
                ),
                dtype=int,
            ),
        )


def _has_source_inputs(
    *,
    trace_paths: tuple[Path, ...],
    trace_dir: Path | None,
    h5dir: Path | None,
    fon: str | None,
    foff: str | None,
) -> bool:
    return bool(trace_paths) or any(
        item is not None for item in (trace_dir, h5dir, fon, foff)
    )


def _build_hdf5_bundle(
    *,
    h5dir: Path | None,
    fon: str | None,
    foff: str | None,
    roi_lower: tuple[int, int] | None,
    roi_dim: tuple[int, int] | None,
    exclude_y: tuple[AxisRange, ...],
    drop_leading: int,
    chunk_frames: int,
    reference_shift: bool,
    components: int,
    roots_backend: str,
    max_traces: int,
) -> WorkflowVizBundle:
    if h5dir is None or fon is None or foff is None:
        raise ValueError("--h5dir, --fon, and --foff are required together")

    from .hdf5 import load_hdf5_pair_roi_trace

    roi_x, roi_y, roi_width, roi_height = _resolve_hdf5_roi(
        h5dir=Path(h5dir),
        fon=fon,
        roi_lower=roi_lower,
        roi_dim=roi_dim,
    )
    detector_image = _load_hdf5_detector_image(
        h5dir=Path(h5dir),
        fon=fon,
        foff=foff,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
        drop_leading=drop_leading,
        chunk_frames=chunk_frames,
        reference_shift=reference_shift,
    )
    rows = _sample_workflow_rows(
        roi_y=roi_y,
        roi_height=roi_height,
        max_traces=max_traces,
        exclude_y=exclude_y,
    )
    records = []
    for row_value in rows:
        trace = load_hdf5_pair_roi_trace(
            h5dir=h5dir,
            fon=fon,
            foff=foff,
            roi_x=roi_x,
            roi_y=roi_y,
            roi_width=roi_width,
            roi_height=roi_height,
            row_y=int(row_value),
            exclude_y=exclude_y,
            drop_leading=drop_leading,
            chunk_frames=chunk_frames,
            reference_shift=reference_shift,
        )
        summary = {
            "kind": "hdf5-workflow-row",
            "input_directory_label": _public_path_label(h5dir),
            "on_file": _public_path_label(fon),
            "off_file": _public_path_label(foff),
            "roi_lower": [roi_x, roi_y],
            "roi_dim": [roi_width, roi_height],
            "row_y": int(row_value),
            **trace.summary(),
        }
        records.append(
            TraceRecord(
                path=Path(f"workflow-row-y{int(row_value)}.npz"),
                label=f"y={int(row_value)}",
                row_y=int(row_value),
                time=trace.delay,
                trace=trace.ratio_minus_one,
                summary=summary,
            )
        )

    manifest = _base_manifest(
        source_kind="hdf5 detector workflow",
        components=components,
        roots_backend=roots_backend,
        detector_panel_title="Detector Image: Laser On-Off",
        detector_x_axis_label="pixel x",
        detector_y_axis_label="pixel y",
    )
    manifest.update(
        {
            "input_directory_label": _public_path_label(h5dir),
            "on_file": _public_path_label(fon),
            "off_file": _public_path_label(foff),
            "roi_lower": [roi_x, roi_y],
            "roi_dim": [roi_width, roi_height],
            "drop_leading": int(drop_leading),
            "reference_shift": bool(reference_shift),
        }
    )
    return _records_to_bundle(
        records=tuple(records),
        detector_image=detector_image,
        detector_x0=float(roi_x),
        detector_y0=float(roi_y),
        detector_dw=float(roi_width),
        detector_dh=float(roi_height),
        manifest=manifest,
        components=components,
        roots_backend=roots_backend,
    )


def _build_trace_bundle(
    *,
    trace_paths: tuple[Path, ...],
    trace_dir: Path | None,
    detector_artifact_dir: Path | None,
    phonon_detector_artifact_dir: Path | None,
    x_value: int | None,
    phonon_amp_threshold: float | None,
    components: int,
    roots_backend: str,
    max_traces: int,
    max_points: int,
) -> WorkflowVizBundle:
    records = _load_trace_records(
        trace_paths=trace_paths,
        trace_dir=trace_dir,
        max_traces=max_traces,
    )
    if not records:
        raise ValueError("provide --trace-npz or --trace-dir")

    row_y = _record_row_values(records)
    if detector_artifact_dir is not None:
        filtered_artifact_dir = (
            detector_artifact_dir
            if phonon_detector_artifact_dir is None
            else phonon_detector_artifact_dir
        )
        (
            detector_image,
            detector_extent,
            source_note,
            detector_title,
            x_label,
            marker_x,
        ) = _detector_artifact_view(
            detector_artifact_dir,
            row_y=row_y,
            x_value=x_value,
            max_points=max_points,
        )
        filtered_phonon = _detector_filtered_phonon_data(
            filtered_artifact_dir,
            x_value=x_value,
            amp_threshold=phonon_amp_threshold,
            max_points=max_points,
        )
    else:
        detector_image = np.stack([record.trace for record in records])
        detector_extent = (
            float(np.min(records[0].time)),
            float(np.min(row_y)),
            max(float(np.ptp(records[0].time)), 1.0e-12),
            max(float(np.ptp(row_y)), 1.0),
        )
        detector_title = "Detector Image: Laser On-Off (trace proxy)"
        x_label = "delay"
        source_note = "trace-only fallback"
        filtered_phonon = _empty_filtered_phonon_data()

    source_kind = (
        "detector workflow review"
        if detector_artifact_dir is not None
        else "trace-derived workflow review"
    )
    manifest = _base_manifest(
        source_kind=source_kind,
        components=components,
        roots_backend=roots_backend,
        detector_panel_title=detector_title,
        detector_x_axis_label=x_label,
        detector_y_axis_label="detector row y",
    )
    manifest.update(
        {
            "trace_source_label": _public_path_label(
                trace_dir,
                fallback="explicit trace files",
            ),
            "trace_files": [
                _public_path_label(record.path) for record in records
            ],
            "detector_artifact_label": (
                _public_path_label(detector_artifact_dir)
                if detector_artifact_dir is not None
                else None
            ),
            "phonon_detector_artifact_label": (
                None
                if detector_artifact_dir is None
                else _public_path_label(filtered_artifact_dir)
            ),
            "detector_marker_x": (
                marker_x if detector_artifact_dir is not None else None
            ),
            "source_note": source_note,
            "phonon_amp_threshold": (
                float(phonon_amp_threshold)
                if phonon_amp_threshold is not None
                else None
            ),
            "filtered_phonon_x_value": filtered_phonon["x_value"],
        }
    )
    return _records_to_bundle(
        records=records,
        detector_image=detector_image,
        detector_x0=detector_extent[0],
        detector_y0=detector_extent[1],
        detector_dw=detector_extent[2],
        detector_dh=detector_extent[3],
        manifest=manifest,
        components=components,
        roots_backend=roots_backend,
        filtered_phonon=filtered_phonon,
    )


def _records_to_bundle(
    *,
    records: tuple[TraceRecord, ...],
    detector_image: np.ndarray,
    detector_x0: float,
    detector_y0: float,
    detector_dw: float,
    detector_dh: float,
    manifest: dict[str, Any],
    components: int,
    roots_backend: str,
    filtered_phonon: dict[str, np.ndarray] | None = None,
) -> WorkflowVizBundle:
    if not records:
        raise ValueError("workflow-viz needs at least one trace")
    fits = _fit_traces(
        records,
        components=components,
        roots_backend=roots_backend,
    )
    time = np.asarray(records[0].time, dtype=float)
    trace_matrix = np.stack([record.trace for record in records])
    reconstruction_matrix = np.stack([fit.reconstruction for fit in fits])
    residual_matrix = np.stack([fit.residual for fit in fits])
    raw_fft_frequency, raw_fft_matrix = _fft_rows(time, trace_matrix)
    fit_fft_frequency, fit_fft_matrix = _fft_rows(
        time,
        reconstruction_matrix,
    )
    frequency_centers = _padded_modes(
        tuple(fit.frequency_centers for fit in fits)
    )
    amplitudes = _padded_modes(tuple(fit.amplitude for fit in fits))
    phases = _padded_modes(tuple(fit.phase for fit in fits))
    row_y = _record_row_values(records)
    chi2 = np.asarray([fit.chi2 for fit in fits], dtype=float)
    selected_model_order = np.asarray(
        [fit.selected_model_order for fit in fits],
        dtype=int,
    )
    mode_count = np.asarray([fit.mode_count for fit in fits], dtype=int)
    dispersion_image = np.log1p(np.maximum(fit_fft_matrix, 0.0)).T
    if filtered_phonon is None:
        filtered_phonon = _empty_filtered_phonon_data()

    manifest.update(
        {
            "kind": "xray-workflow-viz-bundle",
            "version": WORKFLOW_BUNDLE_VERSION,
            "row_count": int(row_y.shape[0]),
            "mode_count": int(np.sum(mode_count)),
            "samples": int(time.shape[0]),
            "detector_x0": float(detector_x0),
            "detector_y0": float(detector_y0),
            "detector_dw": float(detector_dw),
            "detector_dh": float(detector_dh),
            "detector_shape": [
                int(detector_image.shape[0]),
                int(detector_image.shape[1]),
            ],
            "filtered_phonon_count": int(filtered_phonon["row"].shape[0]),
        }
    )
    return WorkflowVizBundle(
        manifest=manifest,
        detector_image=np.asarray(detector_image, dtype=float),
        row_y=row_y,
        time=time,
        trace_matrix=trace_matrix,
        reconstruction_matrix=reconstruction_matrix,
        residual_matrix=residual_matrix,
        raw_fft_frequency=raw_fft_frequency,
        raw_fft_matrix=raw_fft_matrix,
        fit_fft_frequency=fit_fft_frequency,
        fit_fft_matrix=fit_fft_matrix,
        frequency_centers=frequency_centers,
        amplitudes=amplitudes,
        phases=phases,
        chi2=chi2,
        selected_model_order=selected_model_order,
        mode_count=mode_count,
        dispersion_frequency=fit_fft_frequency,
        dispersion_image=dispersion_image,
        filtered_phonon_row=np.asarray(filtered_phonon["row"], dtype=float),
        filtered_phonon_frequency=np.asarray(
            filtered_phonon["frequency"],
            dtype=float,
        ),
        filtered_phonon_amplitude=np.asarray(
            filtered_phonon["amplitude"],
            dtype=float,
        ),
        filtered_phonon_mode=np.asarray(filtered_phonon["mode"], dtype=int),
    )


def _base_manifest(
    *,
    source_kind: str,
    components: int,
    roots_backend: str,
    detector_panel_title: str,
    detector_x_axis_label: str,
    detector_y_axis_label: str,
) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "components": int(components),
        "roots_backend": roots_backend,
        "detector_panel_title": detector_panel_title,
        "detector_x_axis_label": detector_x_axis_label,
        "detector_y_axis_label": detector_y_axis_label,
    }


def _resolve_hdf5_roi(
    *,
    h5dir: Path,
    fon: str,
    roi_lower: tuple[int, int] | None,
    roi_dim: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    import h5py

    from .hdf5 import probe_hdf5_file

    path = _resolve_input_path(h5dir, fon)
    probe = probe_hdf5_file(path)
    if probe.load_plan is None:
        raise ValueError(f"unsupported HDF5 schema: {path}")
    with h5py.File(path, "r") as h5:
        shape = h5[probe.load_plan.image_dataset].shape
    if len(shape) != 3:
        raise ValueError("workflow-viz HDF5 input must be a 3D image cube")
    _frame_count, height, width = (
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
    )
    roi_x, roi_y = (0, 0) if roi_lower is None else roi_lower
    roi_width, roi_height = (
        (width - roi_x, height - roi_y) if roi_dim is None else roi_dim
    )
    if roi_x < 0 or roi_y < 0:
        raise ValueError("ROI origin must be non-negative")
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    if roi_x + roi_width > width or roi_y + roi_height > height:
        raise ValueError("ROI exceeds HDF5 detector image bounds")
    return int(roi_x), int(roi_y), int(roi_width), int(roi_height)


def _load_hdf5_detector_image(
    *,
    h5dir: Path,
    fon: str,
    foff: str,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    drop_leading: int,
    chunk_frames: int,
    reference_shift: bool,
) -> np.ndarray:
    on_mean = _normalized_mean_roi_image(
        _resolve_input_path(h5dir, fon),
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
        drop_leading=drop_leading,
        chunk_frames=chunk_frames,
    )
    off_mean = _normalized_mean_roi_image(
        _resolve_input_path(h5dir, foff),
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
        drop_leading=drop_leading,
        chunk_frames=chunk_frames,
    )
    shift = float(np.mean(off_mean)) if reference_shift else 0.0
    denominator = off_mean + shift
    detector = np.divide(
        on_mean + shift,
        denominator,
        out=np.zeros_like(on_mean, dtype=np.float64),
        where=denominator != 0,
    )
    return detector - 1.0


def _normalized_mean_roi_image(
    path: Path,
    *,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    drop_leading: int,
    chunk_frames: int,
) -> np.ndarray:
    import h5py

    from .hdf5 import probe_hdf5_file

    if drop_leading < 0:
        raise ValueError("drop_leading must be non-negative")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    probe = probe_hdf5_file(path)
    if probe.load_plan is None:
        raise ValueError(f"unsupported HDF5 schema: {path}")
    plan = probe.load_plan
    with h5py.File(path, "r") as h5:
        image = h5[plan.image_dataset]
        normalization = np.asarray(
            h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=float,
        )
        frame_count = int(image.shape[0])
        if drop_leading >= frame_count:
            raise ValueError("drop_leading removes all HDF5 frames")
        total = np.zeros((roi_height, roi_width), dtype=np.float64)
        out_index = 0
        row_slice = slice(roi_y, roi_y + roi_height)
        col_slice = slice(roi_x, roi_x + roi_width)
        for start in range(drop_leading, frame_count, chunk_frames):
            stop = min(frame_count, start + chunk_frames)
            frames = np.asarray(
                image[start:stop, row_slice, col_slice],
                dtype=np.float64,
            )
            frames = np.nan_to_num(
                frames,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            norm = normalization[out_index : out_index + frames.shape[0]]
            frames = np.divide(
                frames,
                norm[:, None, None],
                out=np.zeros_like(frames, dtype=np.float64),
                where=norm[:, None, None] != 0,
            )
            total += np.sum(frames, axis=0)
            out_index += frames.shape[0]
    return total / max(out_index, 1)


def _sample_workflow_rows(
    *,
    roi_y: int,
    roi_height: int,
    max_traces: int,
    exclude_y: tuple[AxisRange, ...],
) -> np.ndarray:
    if max_traces <= 0:
        raise ValueError("--max-traces must be positive")
    rows = np.arange(roi_y, roi_y + roi_height, dtype=int)
    excluded = np.asarray(
        detector_excluded_row_mask(roi_y, roi_y + roi_height, exclude_y),
        dtype=bool,
    )
    usable = rows[np.logical_not(excluded)]
    if usable.size == 0:
        raise ValueError("workflow-viz ROI has no usable detector rows")
    if usable.size <= max_traces:
        return usable
    positions = np.linspace(0, usable.size - 1, max_traces)
    return usable[np.unique(np.round(positions).astype(int))]


def _detector_artifact_view(
    detector_artifact_dir: Path,
    *,
    row_y: np.ndarray,
    x_value: int | None,
    max_points: int,
) -> tuple[
    np.ndarray,
    tuple[float, float, float, float],
    str,
    str,
    str,
    float,
]:
    root = Path(detector_artifact_dir)
    origin_x, origin_y = detector_artifact_origin(root)
    filtered_sum = root / "amp_all_sum_filtered.npy"
    if filtered_sum.exists():
        image = np.asarray(np.load(filtered_sum, mmap_mode="r"), dtype=float)
        if image.ndim != 2:
            raise ValueError("amp_all_sum_filtered.npy must be a 2D image")
        height, width = image.shape
        _local_x, display_x = detector_artifact_x_index(
            width=width,
            x_value=x_value,
            origin_x=origin_x,
        )
        extent = (
            float(origin_x),
            float(origin_y),
            float(width),
            float(height),
        )
        note = f"detector artifact filtered amplitude sum; x={display_x}"
        return (
            np.log1p(np.maximum(image, 0.0)),
            extent,
            note,
            "Detector Image: Fitted Activity",
            "pixel x",
            float(display_x),
        )

    freq_all = np.load(root / "freq_all.npy", mmap_mode="r")
    amp_all = np.load(root / "amp_all.npy", mmap_mode="r")
    if freq_all.shape != amp_all.shape or freq_all.ndim != 3:
        raise ValueError("detector artifact freq_all/amp_all shapes differ")
    height, width, depth = freq_all.shape
    local_x, display_x = detector_artifact_x_index(
        width=width,
        x_value=x_value,
        origin_x=origin_x,
    )
    local_y_start = max(int(np.min(row_y)) - origin_y, 0)
    local_y_end = min(int(np.max(row_y)) + 1 - origin_y, height)
    if local_y_end <= local_y_start:
        local_y_start, local_y_end = 0, height
    display_y_start = origin_y + local_y_start
    display_y_end = origin_y + local_y_end
    image = np.log1p(
        np.maximum(
            amp_all[local_y_start:local_y_end, local_x, :],
            0.0,
        )
    )
    if image.shape[1] > max_points:
        image = image[:, :max_points]
    extent = (
        0.0,
        float(display_y_start),
        float(min(depth, image.shape[1])),
        max(float(display_y_end - display_y_start), 1.0),
    )
    note = f"detector artifact x={display_x}"
    return (
        image,
        extent,
        note,
        "Detector Image: Artifact-Derived Activity",
        f"mode slot, capped at {max_points} points",
        float(extent[0] + extent[2] / 2.0),
    )


def _detector_filtered_phonon_data(
    detector_artifact_dir: Path,
    *,
    x_value: int | None,
    amp_threshold: float | None,
    max_points: int,
) -> dict[str, Any]:
    if amp_threshold is None:
        return _empty_filtered_phonon_data()
    root = Path(detector_artifact_dir)
    freq_all = np.load(root / "freq_all.npy", mmap_mode="r")
    amp_all = np.load(root / "amp_all.npy", mmap_mode="r")
    if freq_all.shape != amp_all.shape or freq_all.ndim != 3:
        raise ValueError("detector artifact freq_all/amp_all shapes differ")
    height, width, depth = freq_all.shape
    origin_x, origin_y = detector_artifact_origin(root)
    local_x, display_x = detector_artifact_x_index(
        width=width,
        x_value=x_value,
        origin_x=origin_x,
    )
    freq_slice = np.asarray(freq_all[:, local_x, :], dtype=float)
    amp_slice = np.asarray(amp_all[:, local_x, :], dtype=float)
    row_grid = np.repeat(
        (origin_y + np.arange(height, dtype=float))[:, None],
        depth,
        axis=1,
    )
    mode_grid = np.repeat(
        np.arange(depth, dtype=int)[None, :],
        height,
        axis=0,
    )
    mask = (
        np.isfinite(freq_slice)
        & np.isfinite(amp_slice)
        & (freq_slice > 0.0)
        & (amp_slice > float(amp_threshold))
        & (amp_slice < 1.0e6)
    )
    rows = row_grid[mask]
    modes = mode_grid[mask]
    frequencies = freq_slice[mask]
    amplitudes = amp_slice[mask]
    if max_points > 0 and amplitudes.size > max_points:
        keep = np.argpartition(amplitudes, -max_points)[-max_points:]
        rows = rows[keep]
        modes = modes[keep]
        frequencies = frequencies[keep]
        amplitudes = amplitudes[keep]
    return {
        "row": np.asarray(rows, dtype=float),
        "frequency": np.asarray(frequencies, dtype=float),
        "amplitude": np.asarray(amplitudes, dtype=float),
        "mode": np.asarray(modes, dtype=int),
        "x_value": int(display_x),
    }


def _empty_filtered_phonon_data() -> dict[str, Any]:
    return {
        "row": np.asarray([], dtype=float),
        "frequency": np.asarray([], dtype=float),
        "amplitude": np.asarray([], dtype=float),
        "mode": np.asarray([], dtype=int),
        "x_value": None,
    }


def _record_row_values(records: tuple[TraceRecord, ...]) -> np.ndarray:
    values = []
    for index, record in enumerate(records):
        values.append(index if record.row_y is None else int(record.row_y))
    return np.asarray(values, dtype=int)


def _fft_rows(
    time: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if time.ndim != 1 or rows.ndim != 2:
        raise ValueError("FFT rows need 1D time and 2D row matrix")
    if rows.shape[1] != time.shape[0]:
        raise ValueError("FFT rows must match the time sample count")
    if time.shape[0] < 2:
        return np.asarray([0.0]), np.zeros((rows.shape[0], 1), dtype=float)
    diffs = np.diff(time)
    finite = diffs[np.isfinite(diffs) & (diffs != 0)]
    spacing = float(np.median(np.abs(finite))) if finite.size else 1.0
    frequency = np.fft.rfftfreq(time.shape[0], d=spacing)
    centered = rows - np.mean(rows, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    return frequency, spectrum


def _padded_modes(items: tuple[np.ndarray, ...]) -> np.ndarray:
    max_modes = max((len(item) for item in items), default=0)
    result = np.full((len(items), max_modes), np.nan, dtype=float)
    for row_index, item in enumerate(items):
        values = np.asarray(item, dtype=float)
        if values.size:
            result[row_index, : values.shape[0]] = values
    return result


def _workflow_layout(workflow: WorkflowVizBundle, *, title: str):
    from bokeh.layouts import column, row
    from bokeh.models import Div, Select, Slider

    row_source = _row_marker_source(workflow)
    mode_source = _mode_source(workflow)
    selected_trace = _selected_trace_source(workflow, 0)
    selected_raw_fft = _selected_fft_source(
        workflow.raw_fft_frequency,
        workflow.raw_fft_matrix,
        0,
    )
    selected_fit_fft = _selected_fft_source(
        workflow.fit_fft_frequency,
        workflow.fit_fft_matrix,
        0,
    )
    selected_raw_modes = _selected_raw_frequency_source(workflow, 0)
    selected_modes = _selected_mode_source(workflow, 0, threshold=0.0)

    row_select = Select(
        title="Row",
        value=str(int(workflow.row_y[0])),
        options=[str(int(value)) for value in workflow.row_y],
        width=140,
    )
    max_amp = _finite_max(workflow.amplitudes)
    amp_slider = Slider(
        title="Amplitude threshold",
        start=0.0,
        end=max(max_amp, 1.0e-12),
        value=0.0,
        step=max(max_amp / 100.0, 1.0e-12),
        width=260,
    )
    full_source = _full_source(workflow)

    update_callback = _selection_callback(
        row_source=row_source,
        row_select=row_select,
        amp_slider=amp_slider,
        full_source=full_source,
        selected_trace=selected_trace,
        selected_raw_fft=selected_raw_fft,
        selected_fit_fft=selected_fit_fft,
        selected_raw_modes=selected_raw_modes,
        selected_modes=selected_modes,
        mode_source=mode_source,
    )
    row_source.selected.js_on_change("indices", update_callback)
    amp_slider.js_on_change("value", update_callback)
    row_select.js_on_change(
        "value",
        _row_select_callback(row_source=row_source),
    )

    detector = _detector_figure(workflow, row_source)
    trace = _trace_figure(workflow, selected_trace)
    raw_fft = _fft_figure(
        selected_raw_fft,
        title="FFT Spectra of Time Profiles",
    )
    fit_fft = _fft_figure(
        selected_fit_fft,
        title="FFT Spectra of Fits",
    )
    dispersion = _dispersion_figure(workflow, mode_source)
    selected_raw = _selected_raw_frequency_figure(selected_raw_modes)
    selected_fit = _selected_fit_frequency_figure(selected_modes)

    controls = row(row_select, amp_slider, sizing_mode="stretch_width")
    signal_stack = column(
        trace,
        raw_fft,
        fit_fft,
        sizing_mode="stretch_width",
    )
    frequency_stack = row(
        selected_raw,
        selected_fit,
        sizing_mode="stretch_width",
    )
    panels = [
        Div(
            text=_header_html(workflow, title=title),
            sizing_mode="stretch_width",
        ),
        controls,
        row(detector, signal_stack, sizing_mode="stretch_width"),
        row(dispersion, frequency_stack, sizing_mode="stretch_width"),
    ]
    if workflow.filtered_phonon_row.size:
        panels.append(_filtered_phonon_figure(workflow))
    return column(
        *panels,
        sizing_mode="stretch_width",
    )


def _detector_figure(workflow: WorkflowVizBundle, row_source):
    from bokeh.models import ColorBar, HoverTool, LinearColorMapper
    from bokeh.palettes import Viridis256
    from bokeh.plotting import figure

    manifest = workflow.manifest
    low, high = _finite_bounds(workflow.detector_image)
    mapper = LinearColorMapper(palette=list(Viridis256), low=low, high=high)
    plot = figure(
        title=str(manifest["detector_panel_title"]),
        width=930,
        height=690,
        sizing_mode="stretch_width",
        x_axis_label=str(manifest["detector_x_axis_label"]),
        y_axis_label=str(manifest["detector_y_axis_label"]),
        tools="pan,wheel_zoom,box_zoom,tap,reset,save",
    )
    plot.image(
        image=[workflow.detector_image],
        x=float(manifest["detector_x0"]),
        y=float(manifest["detector_y0"]),
        dw=float(manifest["detector_dw"]),
        dh=float(manifest["detector_dh"]),
        color_mapper=mapper,
    )
    plot.add_layout(ColorBar(color_mapper=mapper, width=10), "right")
    renderer = plot.scatter(
        x="detector_x",
        y="row_y",
        source=row_source,
        size=9,
        marker="circle",
        fill_color="#d95f4f",
        line_color="#142033",
        fill_alpha=0.88,
        line_alpha=0.75,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("row", "@row_y"),
                ("chi2", "@chi2{0.000e}"),
                ("modes", "@mode_count"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _trace_figure(workflow: WorkflowVizBundle, selected_trace):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    stacked = _stacked_trace_source(workflow)
    plot = figure(
        title="Time Profiles and Fits",
        width=600,
        height=220,
        sizing_mode="stretch_width",
        x_axis_label="time delay",
        y_axis_label="ratio minus one",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    stacked_source = ColumnDataSource(stacked)
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=stacked_source,
        line_color="#315f9f",
        line_alpha=0.25,
        line_width=1.0,
    )
    raw = plot.line(
        x="time",
        y="raw",
        source=selected_trace,
        line_color="#315f9f",
        line_width=2.0,
        legend_label="raw",
    )
    fit = plot.line(
        x="time",
        y="fit",
        source=selected_trace,
        line_color="#c87519",
        line_width=1.8,
        line_dash="dashed",
        legend_label="fit",
    )
    residual = plot.line(
        x="time",
        y="residual",
        source=selected_trace,
        line_color="#c64740",
        line_width=1.1,
        line_alpha=0.72,
        legend_label="residual",
    )
    plot.add_tools(
        HoverTool(
            renderers=[raw, fit, residual],
            tooltips=[("time", "@time{0.000}"), ("value", "$y{0.000e}")],
            line_policy="nearest",
        )
    )
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    _style_plot(plot)
    return plot


def _fft_figure(source, *, title: str):
    from bokeh.models import HoverTool
    from bokeh.plotting import figure

    plot = figure(
        title=title,
        width=600,
        height=220,
        sizing_mode="stretch_width",
        x_axis_label="frequency",
        y_axis_label="magnitude",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.line(
        x="frequency",
        y="spectrum",
        source=source,
        line_color="#14866d",
        line_width=1.8,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("frequency", "@frequency{0.00000}"),
                ("magnitude", "@spectrum{0.000e}"),
            ],
            line_policy="nearest",
        )
    )
    _style_plot(plot)
    return plot


def _dispersion_figure(workflow: WorkflowVizBundle, mode_source):
    from bokeh.models import ColorBar, HoverTool, LinearColorMapper
    from bokeh.palettes import Magma256
    from bokeh.plotting import figure

    row_values = workflow.row_y.astype(float)
    freq = workflow.dispersion_frequency
    image = workflow.dispersion_image
    low, high = _finite_bounds(image)
    mapper = LinearColorMapper(palette=list(Magma256), low=low, high=high)
    plot = figure(
        title="Phonon Dispersion",
        width=770,
        height=390,
        sizing_mode="stretch_width",
        x_axis_label="detector row y",
        y_axis_label="frequency",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    x0, dw = _axis_extent(row_values)
    y0, dh = _axis_extent(freq)
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
    renderer = plot.scatter(
        x="row_y",
        y="frequency",
        source=mode_source,
        size="size",
        fill_color="color",
        line_color="#142033",
        fill_alpha="alpha",
        line_alpha=0.55,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("row", "@row_y"),
                ("mode", "@mode"),
                ("frequency", "@frequency{0.00000}"),
                ("amplitude", "@amplitude{0.000e}"),
                ("chi2", "@chi2{0.000e}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _filtered_phonon_figure(workflow: WorkflowVizBundle):
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    manifest = workflow.manifest
    threshold = manifest.get("phonon_amp_threshold")
    x_value = manifest.get("filtered_phonon_x_value")
    title = "Filtered Phonon Dispersion"
    if x_value is not None:
        title += f" x={int(x_value)}"
    if threshold is not None:
        title += f", amp>{float(threshold):g}"
    source = ColumnDataSource(_filtered_phonon_source(workflow))
    plot = figure(
        title=title,
        width=1540,
        height=420,
        sizing_mode="stretch_width",
        x_axis_label="pixel y",
        y_axis_label="frequency (THz)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.scatter(
        x="row_y",
        y="frequency",
        source=source,
        size="size",
        fill_color="color",
        line_color="#142033",
        fill_alpha=0.84,
        line_alpha=0.45,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("row", "@row_y{0}"),
                ("mode", "@mode"),
                ("frequency", "@frequency{0.00000}"),
                ("amplitude", "@amplitude{0.000e}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _selected_raw_frequency_figure(selected_modes):
    from bokeh.models import HoverTool
    from bokeh.plotting import figure

    plot = figure(
        title="Subset of Selected Frequencies",
        width=380,
        height=390,
        sizing_mode="stretch_width",
        x_axis_label="rank",
        y_axis_label="frequency",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.scatter(
        x="rank",
        y="frequency",
        source=selected_modes,
        size="size",
        fill_color="#315f9f",
        line_color="#142033",
        fill_alpha=0.82,
        line_alpha=0.58,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("row", "@row_y"),
                ("rank", "@rank"),
                ("frequency", "@frequency{0.00000}"),
                ("magnitude", "@magnitude{0.000e}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _selected_fit_frequency_figure(selected_modes):
    from bokeh.models import HoverTool
    from bokeh.plotting import figure

    plot = figure(
        title="Subset of Fitted Frequencies",
        width=380,
        height=390,
        sizing_mode="stretch_width",
        x_axis_label="mode",
        y_axis_label="frequency",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    renderer = plot.scatter(
        x="mode",
        y="frequency",
        source=selected_modes,
        size="size",
        fill_color="color",
        line_color="#142033",
        fill_alpha=0.82,
        line_alpha=0.58,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=[
                ("row", "@row_y"),
                ("mode", "@mode"),
                ("frequency", "@frequency{0.00000}"),
                ("amplitude", "@amplitude{0.000e}"),
                ("phase", "@phase{0.000}"),
            ],
        )
    )
    _style_plot(plot)
    return plot


def _row_marker_source(workflow: WorkflowVizBundle):
    from bokeh.models import ColumnDataSource

    manifest = workflow.manifest
    marker_x = manifest.get("detector_marker_x")
    if marker_x is None:
        x = (
            float(manifest["detector_x0"])
            + float(manifest["detector_dw"]) / 2.0
        )
    else:
        x = float(marker_x)
    source = ColumnDataSource(
        {
            "index": list(range(workflow.row_y.shape[0])),
            "row_y": workflow.row_y.tolist(),
            "detector_x": [x] * workflow.row_y.shape[0],
            "chi2": workflow.chi2.tolist(),
            "mode_count": workflow.mode_count.tolist(),
        }
    )
    source.selected.indices = [0]
    return source


def _mode_source(workflow: WorkflowVizBundle):
    from bokeh.models import ColumnDataSource

    data = _flat_mode_data(workflow, threshold=0.0)
    return ColumnDataSource(data)


def _selected_trace_source(workflow: WorkflowVizBundle, index: int):
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(
        {
            "time": workflow.time.tolist(),
            "raw": workflow.trace_matrix[index].tolist(),
            "fit": workflow.reconstruction_matrix[index].tolist(),
            "residual": workflow.residual_matrix[index].tolist(),
        }
    )


def _selected_fft_source(
    frequency: np.ndarray,
    matrix: np.ndarray,
    index: int,
):
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(
        {
            "frequency": frequency.tolist(),
            "spectrum": matrix[index].tolist(),
        }
    )


def _selected_raw_frequency_source(workflow: WorkflowVizBundle, index: int):
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(_raw_frequency_data(workflow, index))


def _selected_mode_source(
    workflow: WorkflowVizBundle,
    index: int,
    *,
    threshold: float,
):
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(_row_mode_data(workflow, index, threshold))


def _stacked_trace_source(workflow: WorkflowVizBundle) -> dict[str, Any]:
    row_count = min(workflow.trace_matrix.shape[0], 18)
    traces = workflow.trace_matrix[:row_count]
    span = max(_finite_max(np.abs(traces)), 1.0e-12)
    ys = []
    for index, trace in enumerate(traces):
        centered = trace - np.nanmedian(trace)
        ys.append((centered / span * 0.7 + index).tolist())
    return {
        "xs": [workflow.time.tolist() for _index in range(row_count)],
        "ys": ys,
        "row_y": workflow.row_y[:row_count].tolist(),
    }


def _full_source(workflow: WorkflowVizBundle):
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(
        {
            "row_y": [workflow.row_y.tolist()],
            "time": [workflow.time.tolist()],
            "trace_matrix": [workflow.trace_matrix.tolist()],
            "reconstruction_matrix": [
                workflow.reconstruction_matrix.tolist()
            ],
            "residual_matrix": [workflow.residual_matrix.tolist()],
            "raw_fft_frequency": [workflow.raw_fft_frequency.tolist()],
            "raw_fft_matrix": [workflow.raw_fft_matrix.tolist()],
            "fit_fft_frequency": [workflow.fit_fft_frequency.tolist()],
            "fit_fft_matrix": [workflow.fit_fft_matrix.tolist()],
            "frequency_centers": [workflow.frequency_centers.tolist()],
            "amplitudes": [workflow.amplitudes.tolist()],
            "phases": [workflow.phases.tolist()],
            "chi2": [workflow.chi2.tolist()],
        }
    )


def _selection_callback(
    *,
    row_source,
    row_select,
    amp_slider,
    full_source,
    selected_trace,
    selected_raw_fft,
    selected_fit_fft,
    selected_raw_modes,
    selected_modes,
    mode_source,
):
    from bokeh.models import CustomJS

    return CustomJS(
        args={
            "row_source": row_source,
            "row_select": row_select,
            "amp_slider": amp_slider,
            "full_source": full_source,
            "selected_trace": selected_trace,
            "selected_raw_fft": selected_raw_fft,
            "selected_fit_fft": selected_fit_fft,
            "selected_raw_modes": selected_raw_modes,
            "selected_modes": selected_modes,
            "mode_source": mode_source,
        },
        code="""
const selected = row_source.selected.indices;
const index = selected.length ? selected[0] : 0;
const threshold = amp_slider.value;
const full = full_source.data;
const rows = full.row_y[0];
row_select.value = String(rows[index]);

selected_trace.data = {
  time: full.time[0],
  raw: full.trace_matrix[0][index],
  fit: full.reconstruction_matrix[0][index],
  residual: full.residual_matrix[0][index],
};
selected_raw_fft.data = {
  frequency: full.raw_fft_frequency[0],
  spectrum: full.raw_fft_matrix[0][index],
};
selected_fit_fft.data = {
  frequency: full.fit_fft_frequency[0],
  spectrum: full.fit_fft_matrix[0][index],
};

const rawFrequency = full.raw_fft_frequency[0] || [];
const rawSpectrum = full.raw_fft_matrix[0][index] || [];
const rawPeaks = [];
for (let i = 0; i < rawFrequency.length; i++) {
  const frequency = rawFrequency[i];
  const magnitude = rawSpectrum[i];
  if (!Number.isFinite(frequency) || !Number.isFinite(magnitude)) {
    continue;
  }
  if (frequency <= 0) {
    continue;
  }
  rawPeaks.push({frequency: frequency, magnitude: magnitude});
}
rawPeaks.sort((a, b) => Math.abs(b.magnitude) - Math.abs(a.magnitude));
const rawModeLimit = Math.min(12, rawPeaks.length);
const rawRank = [];
const rawFreq = [];
const rawMagnitude = [];
const rawRow = [];
const rawSize = [];
for (let i = 0; i < rawModeLimit; i++) {
  const peak = rawPeaks[i];
  rawRank.push(i);
  rawFreq.push(peak.frequency);
  rawMagnitude.push(peak.magnitude);
  rawRow.push(rows[index]);
  const peakSize = 6 + Math.log1p(Math.abs(peak.magnitude)) * 2;
  rawSize.push(Math.max(6, Math.min(17, peakSize)));
}
selected_raw_modes.data = {
  rank: rawRank,
  frequency: rawFreq,
  magnitude: rawMagnitude,
  row_y: rawRow,
  size: rawSize,
};

const centers = full.frequency_centers[0][index] || [];
const amplitudes = full.amplitudes[0][index] || [];
const phases = full.phases[0][index] || [];
const modes = [];
const freq = [];
const amps = [];
const phase = [];
const row = [];
const size = [];
const color = [];
for (let i = 0; i < centers.length; i++) {
  const center = centers[i];
  const amp = amplitudes[i];
  if (!Number.isFinite(center) || !Number.isFinite(amp)) {
    continue;
  }
  if (amp < threshold) {
    continue;
  }
  modes.push(i);
  freq.push(center);
  amps.push(amp);
  phase.push(phases[i]);
  row.push(rows[index]);
  size.push(Math.max(6, Math.min(17, 6 + Math.log1p(Math.abs(amp)) * 2)));
  color.push(["#315f9f", "#14866d", "#c87519", "#c64740"][i % 4]);
}
selected_modes.data = {
  mode: modes,
  frequency: freq,
  amplitude: amps,
  phase: phase,
  row_y: row,
  size: size,
  color: color,
};

const all_amp = mode_source.data.amplitude;
const alpha = [];
for (let i = 0; i < all_amp.length; i++) {
  alpha.push(all_amp[i] >= threshold ? 0.82 : 0.08);
}
mode_source.data.alpha = alpha;
selected_trace.change.emit();
selected_raw_fft.change.emit();
selected_fit_fft.change.emit();
selected_raw_modes.change.emit();
selected_modes.change.emit();
mode_source.change.emit();
""",
    )


def _row_select_callback(*, row_source):
    from bokeh.models import CustomJS

    return CustomJS(
        args={"row_source": row_source},
        code="""
const rows = row_source.data.row_y;
const target = Number(cb_obj.value);
let index = rows.indexOf(target);
if (index < 0) {
  index = 0;
}
row_source.selected.indices = [index];
row_source.selected.change.emit();
row_source.change.emit();
""",
    )


def _flat_mode_data(
    workflow: WorkflowVizBundle,
    *,
    threshold: float,
) -> dict[str, list[Any]]:
    rows: list[int] = []
    modes: list[int] = []
    frequencies: list[float] = []
    amplitudes: list[float] = []
    chi2: list[float] = []
    sizes: list[float] = []
    colors: list[str] = []
    alpha: list[float] = []
    palette = ["#315f9f", "#14866d", "#c87519", "#c64740"]
    for row_index, row_y in enumerate(workflow.row_y):
        for mode_index, frequency in enumerate(
            workflow.frequency_centers[row_index]
        ):
            amplitude = workflow.amplitudes[row_index, mode_index]
            if not np.isfinite(frequency) or not np.isfinite(amplitude):
                continue
            rows.append(int(row_y))
            modes.append(int(mode_index))
            frequencies.append(float(frequency))
            amplitudes.append(float(amplitude))
            chi2.append(float(workflow.chi2[row_index]))
            sizes.append(_mode_size(amplitude))
            colors.append(palette[mode_index % len(palette)])
            alpha.append(0.82 if amplitude >= threshold else 0.08)
    return {
        "row_y": rows,
        "mode": modes,
        "frequency": frequencies,
        "amplitude": amplitudes,
        "chi2": chi2,
        "size": sizes,
        "color": colors,
        "alpha": alpha,
    }


def _row_mode_data(
    workflow: WorkflowVizBundle,
    row_index: int,
    threshold: float,
) -> dict[str, list[Any]]:
    palette = ["#315f9f", "#14866d", "#c87519", "#c64740"]
    modes: list[int] = []
    frequencies: list[float] = []
    amplitudes: list[float] = []
    phases: list[float] = []
    rows: list[int] = []
    sizes: list[float] = []
    colors: list[str] = []
    for mode_index, frequency in enumerate(
        workflow.frequency_centers[row_index]
    ):
        amplitude = workflow.amplitudes[row_index, mode_index]
        phase = workflow.phases[row_index, mode_index]
        if not np.isfinite(frequency) or not np.isfinite(amplitude):
            continue
        if amplitude < threshold:
            continue
        modes.append(int(mode_index))
        frequencies.append(float(frequency))
        amplitudes.append(float(amplitude))
        phases.append(float(phase))
        rows.append(int(workflow.row_y[row_index]))
        sizes.append(_mode_size(amplitude))
        colors.append(palette[mode_index % len(palette)])
    return {
        "mode": modes,
        "frequency": frequencies,
        "amplitude": amplitudes,
        "phase": phases,
        "row_y": rows,
        "size": sizes,
        "color": colors,
    }


def _raw_frequency_data(
    workflow: WorkflowVizBundle,
    row_index: int,
    *,
    limit: int = 12,
) -> dict[str, list[Any]]:
    frequency = np.asarray(workflow.raw_fft_frequency, dtype=float)
    spectrum = np.asarray(workflow.raw_fft_matrix[row_index], dtype=float)
    valid = np.isfinite(frequency) & np.isfinite(spectrum) & (frequency > 0.0)
    valid_indices = np.nonzero(valid)[0]
    if valid_indices.shape[0] == 0:
        return {
            "rank": [],
            "frequency": [],
            "magnitude": [],
            "row_y": [],
            "size": [],
        }
    order = valid_indices[
        np.argsort(np.abs(spectrum[valid_indices]))[::-1][:limit]
    ]
    magnitudes = spectrum[order]
    return {
        "rank": list(range(order.shape[0])),
        "frequency": [float(value) for value in frequency[order]],
        "magnitude": [float(value) for value in magnitudes],
        "row_y": [int(workflow.row_y[row_index])] * int(order.shape[0]),
        "size": [_mode_size(value) for value in magnitudes],
    }


def _filtered_phonon_source(workflow: WorkflowVizBundle) -> dict[str, Any]:
    rows = np.asarray(workflow.filtered_phonon_row, dtype=float)
    frequency = np.asarray(workflow.filtered_phonon_frequency, dtype=float)
    amplitude = np.asarray(workflow.filtered_phonon_amplitude, dtype=float)
    mode = np.asarray(workflow.filtered_phonon_mode, dtype=int)
    return {
        "row_y": [float(value) for value in rows],
        "frequency": [float(value) for value in frequency],
        "amplitude": [float(value) for value in amplitude],
        "mode": [int(value) for value in mode],
        "color": _palette_colors(amplitude),
        "size": [_mode_size(value) for value in amplitude],
    }


def _palette_colors(values: np.ndarray) -> list[str]:
    from bokeh.palettes import Viridis256

    if values.size == 0:
        return []
    finite = values[np.isfinite(values)]
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
        fraction = (float(value) - low) / (high - low)
        index = int(np.clip(round(fraction * (len(Viridis256) - 1)), 0, 255))
        colors.append(Viridis256[index])
    return colors


def _mode_size(amplitude: float) -> float:
    return float(max(6.0, min(17.0, 6.0 + np.log1p(abs(amplitude)) * 2.0)))


def _header_html(workflow: WorkflowVizBundle, *, title: str) -> str:
    manifest = workflow.manifest
    fields = [
        ("Source", str(manifest["source_kind"])),
        ("Rows", str(workflow.row_y.shape[0])),
        ("Samples", str(workflow.time.shape[0])),
        ("Modes", str(int(np.sum(workflow.mode_count)))),
        ("Components", str(manifest["components"])),
        ("Roots", str(manifest["roots_backend"])),
    ]
    if "roi_lower" in manifest:
        fields.append(
            ("ROI", f"{manifest['roi_lower']} {manifest['roi_dim']}")
        )
    filtered_count = int(manifest.get("filtered_phonon_count", 0) or 0)
    if filtered_count:
        fields.append(("Filtered Modes", str(filtered_count)))
    header_style = (
        "border-bottom:1px solid #d9e2ec;margin-bottom:10px;"
        "padding:2px 0 10px;"
    )
    h1_style = "font-size:22px;line-height:1.16;margin:0 0 8px;"
    meta_style = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;"
    pill_style = (
        "background:#ffffff;border:1px solid #d9e2ec;border-radius:6px;"
        "color:#152032;display:inline-flex;align-items:center;"
        "column-gap:5px;font-size:12px;font-weight:640;padding:5px 8px;"
    )
    label_style = "color:#667085;font-weight:620;"
    pills = "".join(
        f"<span class='xray-workflow-pill' style='{pill_style}'>"
        f"<strong style='{label_style}'>{html.escape(name)}</strong>"
        f"{html.escape(value)}"
        "</span>"
        for name, value in fields
    )
    return (
        f"<div class='xray-workflow-header' style='{header_style}'>"
        f"<h1 style='{h1_style}'>{html.escape(title)}</h1>"
        f"<div class='xray-workflow-meta' style='{meta_style}'>{pills}</div>"
        "</div>"
    )


def _style_plot(plot) -> None:
    plot.background_fill_color = "#ffffff"
    plot.border_fill_color = "#ffffff"
    plot.outline_line_color = "#d9e2ec"
    plot.grid.grid_line_color = "#e8edf3"
    plot.axis.axis_label_text_color = "#344054"
    plot.axis.major_label_text_color = "#344054"
    plot.title.text_color = "#152032"
    plot.title.text_font_size = "13pt"
    _prefer_box_zoom(plot)


def _prefer_box_zoom(plot) -> None:
    for tool in plot.tools:
        if tool.__class__.__name__ == "BoxZoomTool":
            plot.toolbar.active_drag = tool
            break


def _axis_extent(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = high - low
    if span <= 0:
        return low - 0.5, 1.0
    return low, span


def _finite_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        return low - 0.5, high + 0.5
    return low, high


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def _loaded_manifest(value: np.ndarray) -> dict[str, Any]:
    raw = str(value.item()) if value.shape == () else str(value.tolist())
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("workflow bundle manifest must be a JSON object")
    return parsed


def _publishable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return manifest metadata with local locations reduced to labels."""

    published = dict(manifest)
    for source_key, label_key in _MANIFEST_PATH_LABELS.items():
        if source_key not in published:
            continue
        value = published.pop(source_key)
        if value is not None and label_key not in published:
            published[label_key] = _public_path_label(value)
    return published


def _resolve_input_path(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return root / path
