# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from cuphoton.core.cli import run_component
from cuphoton.xray.report import build_report


def test_report_tolerates_missing_optional_fft_plot(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    figures_dir = input_dir / "analysis_figures"
    figures_dir.mkdir(parents=True)
    _touch_png(input_dir / "fulldetector_image_run_7_roifull.png")
    _touch_png(
        figures_dir / "fulldetector_image_run_7_amp_all_sum_filtered.png"
    )

    result = build_report(input_dir, output_dir, run=7)

    assert result.report_html.exists()
    assert "fulldetector_image_run_7_fft_all_sum.png" in (
        result.missing_optional_plots
    )
    assert (output_dir / "fulldetector_image_run_7_roifull.png").exists()
    assert (
        output_dir / "fulldetector_image_run_7_amp_all_sum_filtered.png"
    ).exists()


def test_report_discovers_phonon_plots_in_analysis_figures(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    figures_dir = input_dir / "analysis_figures"
    figures_dir.mkdir(parents=True)
    phonon = "Phonon_Dispersion_x_slice_[12,34:56]_Run7.png"
    _touch_png(figures_dir / phonon)

    result = build_report(input_dir, output_dir, run=7)

    assert result.phonon_plots == (phonon,)
    assert (output_dir / phonon).exists()
    assert phonon in result.report_html.read_text(encoding="utf-8")


def test_report_copies_tile_prediction_and_reconstruction(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    figures_dir = input_dir / "analysis_figures"
    figures_dir.mkdir(parents=True)
    prediction = "linear_prediction_[1:2,3:4]_run_7.png"
    reconstruction = "reconst_[1:2,3:4]_run_7.png"
    _touch_png(figures_dir / prediction)
    _touch_png(figures_dir / reconstruction)

    result = build_report(input_dir, output_dir, run=7)

    assert len(result.tile_plots) == 1
    assert result.tile_plots[0].tile == "1:2,3:4"
    assert (output_dir / prediction).exists()
    assert (output_dir / reconstruction).exists()


def test_cli_report_command(tmp_path, capsys):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    figures_dir = input_dir / "analysis_figures"
    figures_dir.mkdir(parents=True)
    _touch_png(input_dir / "fulldetector_image_run_7_roifull.png")

    assert (
        run_component(
            "xray",
            [
                "report",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--run",
                "7",
            ],
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "report=" in captured.out
    assert (output_dir / "report.html").exists()


def _touch_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a png")
