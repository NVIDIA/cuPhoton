# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from cuphoton.core.cli import run_component
from cuphoton.xray.detector_mask import parse_y_ranges
from cuphoton.xray.hdf5 import (
    build_hdf5_load_plan,
    classify_hdf5_schema,
    detect_ipm_pairs,
    load_hdf5_pair_roi_trace,
    load_hdf5_pair_trace,
    probe_hdf5_pair,
    scan_roi_candidates,
    select_ipm_pair,
)
from cuphoton.xray.linear_prediction import synthetic_trace


def _assert_cli_error(capsys, args: list[str], message: str) -> None:
    assert main(args) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def test_schema_and_ipm_detection():
    assert (
        classify_hdf5_schema({"imgs", "scan_var", "i0", "bin_count", "ROI"})
        == "cropped-cube"
    )
    assert (
        classify_hdf5_schema({"jungfrau1M_data", "binVar_bins", "nEntries"})
        == "legate-cube"
    )
    assert detect_ipm_pairs({"ipm3__sum", "ipm2__sum"}) == ("ipm3/ipm2",)
    assert detect_ipm_pairs({"i0", "i0_ipm3"}) == ("i0/i0_ipm3",)


def test_select_ipm_pair_prefers_scipy_fix_order_for_legate_cube():
    keys = {
        "jungfrau1M_data",
        "binVar_bins",
        "nEntries",
        "ipm5__sum",
        "ipm4__sum",
        "ipm3__sum",
        "ipm2__sum",
    }

    pair = select_ipm_pair(keys, schema="legate-cube")

    assert pair.label == "ipm3/ipm2"
    assert pair.normalization == "ipm2__sum"


def test_build_load_plan_for_cropped_fixture_case():
    plan = build_hdf5_load_plan(
        {"imgs", "scan_var", "i0", "i0_ipm3", "bin_count", "ROI"}
    )

    assert plan is not None
    assert plan.image_dataset == "imgs"
    assert plan.delay_dataset == "scan_var"
    assert plan.entries_dataset == "bin_count"
    assert plan.ipm_pair.normalization == "i0"


def test_probe_hdf5_pair(tmp_path):
    _write_fixture(tmp_path / "on.h5")
    _write_fixture(tmp_path / "off.h5")

    result = probe_hdf5_pair(h5dir=tmp_path, fon="on.h5", foff="off.h5")

    assert result.on.schema == "cropped-cube"
    assert result.off.schema == "cropped-cube"
    assert result.on.ipm_pairs == ("i0/i0_ipm3",)
    assert result.on.datasets[0].name == "ROI"
    assert result.on.load_plan is not None
    assert result.on.load_plan.image_dataset == "imgs"


def test_data_probe_cli_json(tmp_path, capsys):
    _write_fixture(tmp_path / "on.h5")
    _write_fixture(tmp_path / "off.h5")

    assert (
        main(
            [
                "data-probe",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "on.h5",
                "--foff",
                "off.h5",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["on"]["schema"] == "cropped-cube"
    assert payload["off"]["ipm_pairs"] == ["i0/i0_ipm3"]
    assert payload["on"]["load_plan"]["ipm_pair"]["normalization"] == "i0"


def test_load_hdf5_pair_trace_matches_reference_shift(tmp_path):
    _write_fixture(
        tmp_path / "on.h5",
        imgs=np.asarray(
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ]
        ),
        i0=np.asarray([2.0, 4.0]),
    )
    _write_fixture(
        tmp_path / "off.h5",
        imgs=np.asarray(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        ),
        i0=np.asarray([2.0, 2.0]),
    )

    trace = load_hdf5_pair_trace(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        drop_leading=0,
        chunk_frames=1,
    )

    np.testing.assert_allclose(trace.on.normalized_sum, [4.0, 4.0])
    np.testing.assert_allclose(trace.off.normalized_sum, [2.0, 4.0])
    assert trace.shift == 0.75
    np.testing.assert_allclose(trace.ratio_minus_one, [0.4, 0.0])


def test_load_hdf5_pair_roi_trace_extracts_detector_row(tmp_path):
    _write_legate_fixture_pair(tmp_path)

    trace = load_hdf5_pair_roi_trace(
        h5dir=tmp_path,
        fon="legate-on.h5",
        foff="legate-off.h5",
        roi_x=2,
        roi_y=0,
        roi_width=2,
        roi_height=2,
        row_y=0,
        drop_leading=0,
        chunk_frames=1,
    )

    assert trace.on.pixel_count == 2
    np.testing.assert_allclose(trace.on.normalized_sum, [2.0, 4.0, 8.0])
    np.testing.assert_allclose(trace.off.normalized_sum, [2.0, 2.0, 2.0])
    assert trace.shift == 1.0
    np.testing.assert_allclose(trace.ratio_minus_one, [0.0, 0.5, 1.5])


def test_load_hdf5_pair_roi_trace_honors_excluded_rows(tmp_path):
    _write_legate_fixture_pair(tmp_path)

    try:
        load_hdf5_pair_roi_trace(
            h5dir=tmp_path,
            fon="legate-on.h5",
            foff="legate-off.h5",
            roi_x=2,
            roi_y=0,
            roi_width=2,
            roi_height=2,
            row_y=0,
            exclude_y=parse_y_ranges("0:1"),
            drop_leading=0,
        )
    except ValueError as exc:
        assert "excluded" in str(exc)
    else:
        raise AssertionError("expected excluded row to fail")


def test_load_hdf5_pair_roi_trace_reuses_excluded_row_iterable(tmp_path):
    _write_legate_fixture_pair(tmp_path)

    exclude_y = (item for item in parse_y_ranges("0:1"))
    trace = load_hdf5_pair_roi_trace(
        h5dir=tmp_path,
        fon="legate-on.h5",
        foff="legate-off.h5",
        roi_x=0,
        roi_y=0,
        roi_width=2,
        roi_height=2,
        exclude_y=exclude_y,
        drop_leading=0,
        chunk_frames=1,
    )

    assert trace.on.pixel_count == 2
    assert trace.off.pixel_count == 2


def test_trace_smoke_cli_json(tmp_path, capsys):
    _write_fixture(tmp_path / "on.h5")
    _write_fixture(tmp_path / "off.h5")

    assert (
        main(
            [
                "trace-smoke",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "on.h5",
                "--foff",
                "off.h5",
                "--drop-leading",
                "0",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["trace"]["samples"] == 2
    assert payload["trace"]["on"]["normalization_dataset"] == "i0"
    assert payload["zero_offset"]["status"] == "insufficient-extrema"


def test_extract_trace_cli_writes_trace_npz(tmp_path, capsys):
    _write_legate_fixture_pair(tmp_path)
    output = tmp_path / "row-y0.npz"

    assert (
        main(
            [
                "extract-trace",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "legate-on.h5",
                "--foff",
                "legate-off.h5",
                "--output",
                str(output),
                "--roi-lower",
                "2",
                "0",
                "--roi-dim",
                "2",
                "2",
                "--row-y",
                "0",
                "--drop-leading",
                "0",
                "--chunk-frames",
                "1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifact"] == str(output)
    assert payload["kind"] == "hdf5-roi"
    assert payload["roi_lower"] == [2, 0]
    assert payload["roi_dim"] == [2, 2]
    assert payload["row_y"] == 0
    assert payload["samples"] == 3
    assert output.exists()
    with np.load(output, allow_pickle=False) as loaded:
        np.testing.assert_allclose(loaded["time"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(loaded["trace"], [0.0, 0.5, 1.5])
        summary = json.loads(str(np.asarray(loaded["summary"]).item()))
    assert summary["artifact"] == str(output)
    assert summary["row_y"] == 0
    assert summary["on"]["pixel_count"] == 2


def test_extract_trace_cli_writes_row_batch(tmp_path, capsys):
    _write_legate_fixture_pair(tmp_path)
    output_dir = tmp_path / "traces"

    assert (
        main(
            [
                "extract-trace",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "legate-on.h5",
                "--foff",
                "legate-off.h5",
                "--output-dir",
                str(output_dir),
                "--output-prefix",
                "synthetic-row",
                "--roi-lower",
                "2",
                "0",
                "--roi-dim",
                "2",
                "2",
                "--row-y",
                "0",
                "--row-y",
                "1",
                "--drop-leading",
                "0",
                "--chunk-frames",
                "1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["kind"] == "trace-npz-batch"
    assert payload["trace_count"] == 2
    assert payload["artifacts"] == [
        str(output_dir / "synthetic-row-y0.npz"),
        str(output_dir / "synthetic-row-y1.npz"),
    ]
    for index, artifact in enumerate(payload["artifacts"]):
        path = Path(artifact)
        assert path.exists()
        with np.load(path, allow_pickle=False) as loaded:
            np.testing.assert_allclose(loaded["time"], [0.0, 1.0, 2.0])
            np.testing.assert_allclose(loaded["trace"], [0.0, 0.5, 1.5])
            summary = json.loads(str(np.asarray(loaded["summary"]).item()))
        assert summary["row_y"] == index
        assert summary["batch_index"] == index
        assert summary["batch_count"] == 2


def test_extract_trace_cli_rejects_duplicate_batch_rows(tmp_path, capsys):
    _write_legate_fixture_pair(tmp_path)

    _assert_cli_error(
        capsys,
        [
            "extract-trace",
            "--h5dir",
            str(tmp_path),
            "--fon",
            "legate-on.h5",
            "--foff",
            "legate-off.h5",
            "--output-dir",
            str(tmp_path / "traces"),
            "--roi-lower",
            "2",
            "0",
            "--roi-dim",
            "2",
            "2",
            "--row-y",
            "0",
            "--row-y",
            "0",
        ],
        "--output-dir requires distinct --row-y values",
    )


def test_extract_trace_cli_requires_npz_output(tmp_path, capsys):
    _write_legate_fixture_pair(tmp_path)

    _assert_cli_error(
        capsys,
        [
            "extract-trace",
            "--h5dir",
            str(tmp_path),
            "--fon",
            "legate-on.h5",
            "--foff",
            "legate-off.h5",
            "--output",
            str(tmp_path / "row.txt"),
        ],
        "--output must end with .npz",
    )


def test_extract_trace_cli_rejects_multiple_rows_with_single_output(
    tmp_path, capsys
):
    _write_legate_fixture_pair(tmp_path)

    _assert_cli_error(
        capsys,
        [
            "extract-trace",
            "--h5dir",
            str(tmp_path),
            "--fon",
            "legate-on.h5",
            "--foff",
            "legate-off.h5",
            "--output",
            str(tmp_path / "row.npz"),
            "--roi-lower",
            "2",
            "0",
            "--roi-dim",
            "2",
            "2",
            "--row-y",
            "0",
            "--row-y",
            "1",
        ],
        "multiple --row-y values require --output-dir",
    )


def test_scan_roi_candidates_scores_variable_tile(tmp_path):
    _write_legate_fixture_pair(tmp_path)

    candidates = scan_roi_candidates(
        h5dir=tmp_path,
        fon="legate-on.h5",
        foff="legate-off.h5",
        tile_width=2,
        tile_height=2,
        stride_x=2,
        stride_y=2,
        drop_leading=0,
        max_candidates=2,
    )

    assert candidates[0].x == 2
    assert candidates[0].y == 0
    assert candidates[0].usable_rows == 2
    assert candidates[0].score > candidates[1].score


def test_scan_roi_candidates_honors_excluded_y_rows(tmp_path):
    _write_legate_fixture_pair(tmp_path)

    candidates = scan_roi_candidates(
        h5dir=tmp_path,
        fon="legate-on.h5",
        foff="legate-off.h5",
        tile_width=2,
        tile_height=2,
        stride_x=2,
        stride_y=2,
        drop_leading=0,
        max_candidates=1,
        exclude_y=parse_y_ranges("0:2"),
    )

    assert candidates[0].y == 2
    assert candidates[0].usable_rows == 2


def test_roi_candidates_cli_json(tmp_path, capsys):
    _write_legate_fixture_pair(tmp_path)

    assert (
        main(
            [
                "roi-candidates",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "legate-on.h5",
                "--foff",
                "legate-off.h5",
                "--tile-width",
                "2",
                "--tile-height",
                "2",
                "--stride-x",
                "2",
                "--stride-y",
                "2",
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["x"] == 2
    assert payload[0]["y"] == 0


def test_model_order_sweep_cli_uses_full_detector_hdf5_chunking(
    tmp_path,
    capsys,
):
    _write_workbench_pair(tmp_path, samples=32)

    assert (
        main(
            [
                "model-order-sweep",
                "--h5dir",
                str(tmp_path),
                "--fon",
                "workbench-on.h5",
                "--foff",
                "workbench-off.h5",
                "--drop-leading",
                "0",
                "--chunk-frames",
                "5",
                "--no-reference-shift",
                "--component",
                "2",
                "--component",
                "4",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"]["kind"] == "hdf5-pair"
    assert payload["samples"] == 32
    assert [entry["components"] for entry in payload["entries"]] == [2, 4]


def _write_fixture(path, *, imgs=None, i0=None):
    if imgs is None:
        imgs = [[[1.0]], [[2.0]]]
    if i0 is None:
        i0 = [3.0, 4.0]
    with h5py.File(path, "w") as h5:
        h5.create_dataset("ROI", data=[[1, 2], [3, 4]])
        h5.create_dataset("bin_count", data=[1.0, 2.0])
        h5.create_dataset("i0", data=i0)
        h5.create_dataset("i0_ipm3", data=[5.0, 6.0])
        h5.create_dataset("imgs", data=imgs)
        h5.create_dataset("scan_var", data=[0.0, 1.0])


def _write_legate_fixture_pair(tmp_path):
    off = np.ones((3, 4, 4), dtype=np.float64)
    on = np.ones((3, 4, 4), dtype=np.float64)
    on[:, 0:2, 2:4] = np.asarray([1.0, 2.0, 4.0])[:, None, None]
    on[:, 2:4, 0:2] = np.asarray([1.0, 1.5, 2.0])[:, None, None]
    for name, image in (("legate-on.h5", on), ("legate-off.h5", off)):
        with h5py.File(tmp_path / name, "w") as h5:
            h5.create_dataset("jungfrau1M_data", data=image)
            h5.create_dataset("binVar_bins", data=[0.0, 1.0, 2.0])
            h5.create_dataset("nEntries", data=[1, 1, 1])
            h5.create_dataset("ipm3__sum", data=[1.0, 1.0, 1.0])
            h5.create_dataset("ipm2__sum", data=[1.0, 1.0, 1.0])


def _write_workbench_pair(tmp_path, *, samples: int):
    delay, trace = synthetic_trace(samples)
    trace = 0.05 * trace
    off = np.ones((samples, 2, 2), dtype=np.float64)
    on = (1.0 + trace)[:, None, None] * off
    for name, image in (("workbench-on.h5", on), ("workbench-off.h5", off)):
        with h5py.File(tmp_path / name, "w") as h5:
            h5.create_dataset("ROI", data=np.ones((2, 2), dtype=np.float64))
            h5.create_dataset("bin_count", data=np.ones(samples))
            h5.create_dataset("i0", data=np.ones(samples))
            h5.create_dataset("i0_ipm3", data=np.ones(samples))
            h5.create_dataset("imgs", data=image)
            h5.create_dataset("scan_var", data=delay)
