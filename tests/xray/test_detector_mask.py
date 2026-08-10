# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from cuphoton.core.cli import run_component
from cuphoton.xray.detector_mask import (
    AxisRange,
    excluded_row_mask,
    parse_y_ranges,
    summarize_tiled_roi,
)


def main(argv=None, *, program_name=None):
    return run_component("xray", argv, program_name=program_name)


def test_parse_y_ranges_merges_and_accepts_axis_prefix():
    assert parse_y_ranges(["y=10:20", "20:24,30:32"]) == (
        AxisRange(10, 24),
        AxisRange(30, 32),
    )


def test_excluded_row_mask_uses_global_detector_rows():
    mask = excluded_row_mask(508, 524, parse_y_ranges("512:520"))

    assert mask == (
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    )


def test_summarize_tiled_roi_counts_full_2048_case():
    summary = summarize_tiled_roi(
        roi_x=256,
        roi_y=0,
        roi_width=512,
        roi_height=1024,
        tile_width=16,
        tile_height=16,
        exclude_y=parse_y_ranges("512:560"),
    )

    assert summary.total_tiles == 2048
    assert summary.active_tiles == 1952
    assert summary.fully_excluded_tiles == 96
    assert summary.partially_excluded_tiles == 0
    assert summary.total_row_fits == 32768
    assert summary.skipped_row_fits == 1536
    assert summary.active_row_fits == 31232


def test_detector_mask_cli_json(capsys):
    assert (
        main(
            [
                "detector-mask",
                "--roi-lower",
                "256",
                "0",
                "--roi-dim",
                "512",
                "1024",
                "--tile-width",
                "16",
                "--tile-height",
                "16",
                "--exclude-y",
                "512:560",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["total_tiles"] == 2048
    assert payload["fully_excluded_tiles"] == 96
    assert payload["skipped_row_fits"] == 1536
