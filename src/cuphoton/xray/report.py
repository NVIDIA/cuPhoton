# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import re
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

SUMMARY_PLOT_NAMES = (
    "fulldetector_image_run_{run}_roifull.png",
    "fulldetector_image_run_{run}_amp_all_sum_filtered.png",
    "fulldetector_image_run_{run}_fft_all_sum.png",
    "fulldetector_image_run_{run}_roifiltered frequencies.png",
)


@dataclass(frozen=True)
class TilePlot:
    tile: str
    prediction: str
    reconstruction: str | None


@dataclass(frozen=True)
class ReportResult:
    report_html: Path
    copied_plots: tuple[str, ...]
    missing_optional_plots: tuple[str, ...]
    tile_plots: tuple[TilePlot, ...]
    summary_plots: tuple[str, ...]
    phonon_plots: tuple[str, ...]


def build_report(input_path: Path | str, output_path: Path | str, run: int):
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    figures_dir = input_dir / "analysis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing_optional: list[str] = []
    summary_plots: list[str] = []

    for template in SUMMARY_PLOT_NAMES:
        name = template.format(run=run)
        source = _find_plot(input_dir, figures_dir, name)
        if source is None:
            missing_optional.append(name)
            continue
        _copy_plot(source, output_dir / name)
        copied.append(name)
        summary_plots.append(name)

    tile_plots = _discover_tile_plots(figures_dir, run)
    for tile in tile_plots:
        for name in (tile.prediction, tile.reconstruction):
            if name is None:
                continue
            source = figures_dir / name
            if source.exists():
                _copy_plot(source, output_dir / name)
                copied.append(name)

    phonon_plots = _discover_phonon_plots(figures_dir, run)
    for name in phonon_plots:
        source = figures_dir / name
        _copy_plot(source, output_dir / name)
        copied.append(name)

    report_html = output_dir / "report.html"
    report_html.write_text(
        _render_report(
            run=run,
            summary_plots=summary_plots,
            tile_plots=tile_plots,
            phonon_plots=phonon_plots,
            missing_optional=missing_optional,
        ),
        encoding="utf-8",
    )

    return ReportResult(
        report_html=report_html,
        copied_plots=tuple(copied),
        missing_optional_plots=tuple(missing_optional),
        tile_plots=tuple(tile_plots),
        summary_plots=tuple(summary_plots),
        phonon_plots=tuple(phonon_plots),
    )


def _discover_tile_plots(figures_dir: Path, run: int):
    if not figures_dir.is_dir():
        return ()

    pattern = re.compile(
        r"linear_prediction_\[(?P<tile>[^\]]+)\]_run_" rf"{run}\.png$"
    )
    tile_plots: list[TilePlot] = []
    for path in sorted(figures_dir.iterdir()):
        match = pattern.match(path.name)
        if match is None:
            continue
        tile = match.group("tile")
        reconstruction = f"reconst_[{tile}]_run_{run}.png"
        if not (figures_dir / reconstruction).exists():
            reconstruction = None
        tile_plots.append(
            TilePlot(
                tile=tile,
                prediction=path.name,
                reconstruction=reconstruction,
            )
        )
    return tuple(tile_plots)


def _discover_phonon_plots(figures_dir: Path, run: int):
    if not figures_dir.is_dir():
        return ()

    pattern = re.compile(
        r"Phonon_Dispersion_x_slice_\[[^\]]+\]_Run" rf"{run}\.png$"
    )
    return tuple(
        path.name
        for path in sorted(figures_dir.iterdir())
        if pattern.match(path.name)
    )


def _find_plot(input_dir: Path, figures_dir: Path, name: str):
    for source in (input_dir / name, figures_dir / name):
        if source.exists():
            return source
    return None


def _copy_plot(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _render_report(
    *,
    run: int,
    summary_plots: list[str],
    tile_plots: tuple[TilePlot, ...],
    phonon_plots: tuple[str, ...],
    missing_optional: list[str],
):
    template = (
        resources.files("cuphoton.xray.templates")
        .joinpath("report.html")
        .read_text(encoding="utf-8")
    )
    return template.format(
        run=run,
        summary_plots=_render_image_list(summary_plots),
        tile_plots=_render_tile_plots(tile_plots),
        phonon_plots=_render_image_list(phonon_plots),
        missing_optional=_render_missing(missing_optional),
    )


def _render_image_list(names):
    return "\n".join(
        f'<div class="row"><a href="{_e(name)}">'
        f'<img src="{_e(name)}" alt="{_e(name)}" /></a></div>'
        for name in names
    )


def _render_tile_plots(tile_plots):
    chunks = []
    for tile in tile_plots:
        prediction = (
            f'<div class="column"><a href="{_e(tile.prediction)}">'
            f'<img src="{_e(tile.prediction)}" '
            f'alt="linear prediction {_e(tile.tile)}" /></a></div>'
        )
        reconstruction = ""
        if tile.reconstruction is not None:
            reconstruction = (
                f'<div class="column"><a href="{_e(tile.reconstruction)}">'
                f'<img src="{_e(tile.reconstruction)}" '
                f'alt="reconstruction {_e(tile.tile)}" /></a></div>'
            )
        chunks.append(
            f"<h2>Tile {_e(tile.tile)}</h2>"
            f'<div class="row">{prediction}{reconstruction}</div>'
        )
    return "\n".join(chunks)


def _render_missing(missing_optional):
    if not missing_optional:
        return ""
    items = "\n".join(
        f"<li>{_e(name)}</li>" for name in sorted(missing_optional)
    )
    return f"<h2>Missing Optional Plots</h2><ul>{items}</ul>"


def _e(value):
    return html.escape(str(value), quote=True)
