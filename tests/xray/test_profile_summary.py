# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict

import pytest

from cuphoton.xray.profile_summary import (
    summarize_linear_prediction_profile_files,
    summarize_linear_prediction_profile_lines,
    summarize_linear_prediction_steady_state_profile_lines,
)

PROFILE_SUMMARY_KEYS = {
    "profile_lines",
    "fit_lines",
    "roi_task_lines",
    "processing_tiles",
    "tile_processing_seconds",
    "command_real_seconds",
    "raw_fits",
    "failures",
    "skipped_fits",
    "filtered_fits",
    "amp_thresholds",
    "stages",
}

PROFILE_STAGE_KEYS = {
    "stage",
    "seconds",
    "count",
    "mean_s",
}


def test_profile_summary_schema_is_stable():
    payload = asdict(
        summarize_linear_prediction_profile_lines(
            [
                "linearpred_profile: ROI [0:16,0:16] "
                "linearpred_total=2.000000s/1, "
                "savgol_cupy=0.400000s/16",
            ]
        )
    )

    assert set(payload) == PROFILE_SUMMARY_KEYS
    assert set(payload["stages"][0]) == PROFILE_STAGE_KEYS


def test_profile_summary_aggregates_raw_profile_lines():
    summary = summarize_linear_prediction_profile_lines(
        [
            "DEBUG:\t20.000s: Processing 2 tiles took   12.500s",
            "linearpred_cupy: ROI [0:16,0:16] raw_fits = 16, "
            "failures = 0, skipped_fits = 0, filtered_fits = 1, "
            "amp_threshold = 1.6",
            "linearpred_profile: ROI [0:16,0:16] "
            "linearpred_total=2.000000s/1, "
            "savgol_cupy=0.400000s/16, p1_total=1.000000s/16, "
            "eigvals_cupy=0.700000s/16, p2_total=0.300000s/16",
            "linearpred_cupy: ROI [0:16,16:32] raw_fits = 12, "
            "failures = 4, skipped_fits = 0, filtered_fits = 0, "
            "amp_threshold = 1.6",
            "linearpred_profile: ROI [0:16,16:32] "
            "linearpred_total=3.000000s/1, "
            "savgol_cupy=0.600000s/16, p1_total=1.500000s/12, "
            "eigvals_cupy=1.000000s/12, p2_total=0.500000s/12",
            "real 19.75",
        ]
    )

    assert summary.profile_lines == 2
    assert summary.fit_lines == 2
    assert summary.processing_tiles == 2
    assert summary.tile_processing_seconds == 12.5
    assert summary.command_real_seconds == 19.75
    assert summary.raw_fits == 28
    assert summary.failures == 4
    assert summary.skipped_fits == 0
    assert summary.filtered_fits == 1
    assert summary.amp_thresholds == (1.6,)

    stages = {stage.stage: stage for stage in summary.stages}
    assert stages["linearpred_total"].seconds == 5.0
    assert stages["linearpred_total"].count == 2
    assert stages["linearpred_total"].mean_s == 2.5
    assert stages["eigvals_cupy"].seconds == pytest.approx(1.7)
    assert stages["eigvals_cupy"].count == 28


def test_profile_summary_reports_steady_state_after_warmup():
    summary = summarize_linear_prediction_steady_state_profile_lines(
        [
            "linearpred_cupy: ROI [0:16,0:16] raw_fits = 16, "
            "failures = 0, skipped_fits = 0, filtered_fits = 1",
            "linearpred_profile: ROI [0:16,0:16] "
            "linearpred_total=10.000000s/1, eigvals_cupy=8.000000s/16",
            "linearpred_profile: ROI [0:16,16:32] "
            "linearpred_total=2.000000s/1, eigvals_cupy=0.700000s/16",
            "linearpred_profile: ROI [16:32,0:16] "
            "linearpred_total=3.000000s/1, eigvals_cupy=1.000000s/16",
        ],
        skip_profile_lines=1,
    )

    assert summary.profile_lines == 3
    assert summary.skipped_profile_lines == 1
    assert summary.included_profile_lines == 2

    stages = {stage.stage: stage for stage in summary.stages}
    assert stages["linearpred_total"].seconds == 5.0
    assert stages["linearpred_total"].count == 2
    assert stages["linearpred_total"].mean_s == 2.5
    assert stages["eigvals_cupy"].seconds == pytest.approx(1.7)
    assert stages["eigvals_cupy"].count == 32


def test_profile_summary_rejects_negative_steady_state_skip():
    with pytest.raises(ValueError, match="non-negative"):
        summarize_linear_prediction_steady_state_profile_lines(
            [],
            skip_profile_lines=-1,
        )


def test_profile_summary_accepts_unprofiled_runtime_lines():
    summary = summarize_linear_prediction_profile_lines(
        [
            "DEBUG:\t90.771s: Processing 32 tiles took   70.564s",
            "linearpred_cupy: ROI [256:272,0:16] raw_fits = 16, "
            "failures = 0, skipped_fits = 0, filtered_fits = 1, "
            "amp_threshold = 1.6",
            "roi_task_cp: ROI [256:272,0:16] done with "
            "filtered_fits = 1, failures = 0",
            "linearpred_cupy: ROI [272:288,0:16] raw_fits = 16, "
            "failures = 0, skipped_fits = 0, filtered_fits = 0, "
            "amp_threshold = 1.6",
            "roi_task_cp: ROI [272:288,0:16] done with "
            "filtered_fits = 0, failures = 0",
            "real 103.17",
        ]
    )

    assert summary.profile_lines == 0
    assert summary.fit_lines == 2
    assert summary.roi_task_lines == 2
    assert summary.processing_tiles == 32
    assert summary.tile_processing_seconds == 70.564
    assert summary.command_real_seconds == 103.17
    assert summary.raw_fits == 32
    assert summary.failures == 0
    assert summary.skipped_fits == 0
    assert summary.filtered_fits == 1
    assert summary.amp_thresholds == (1.6,)
    assert summary.stages == ()


def test_profile_summary_accepts_aggregated_counter_blocks():
    summary = summarize_linear_prediction_profile_lines(
        [
            "tiles=32",
            "profile_lines=32",
            "tile_processing_seconds=70.461",
            "command_real_seconds=102.02",
            "raw_fits=512",
            "failures=0",
            "skipped_fits=0",
            "filtered_fits=12",
            "linearpred_total_seconds=257.493072 count=32 mean=8.046658",
            "eigvals_cupy_seconds=101.675803 count=512 mean=0.198586",
            "p2_total_seconds=57.821188 count=512 mean=0.112932",
        ]
    )

    assert summary.profile_lines == 32
    assert summary.processing_tiles == 32
    assert summary.tile_processing_seconds == 70.461
    assert summary.command_real_seconds == 102.02
    assert summary.raw_fits == 512
    assert summary.failures == 0
    assert summary.skipped_fits == 0
    assert summary.filtered_fits == 12

    stages = {stage.stage: stage for stage in summary.stages}
    assert stages["linearpred_total"].seconds == 257.493072
    assert stages["linearpred_total"].count == 32
    assert stages["eigvals_cupy"].mean_s == pytest.approx(101.675803 / 512)


def test_profile_summary_sums_repeated_scalar_blocks():
    summary = summarize_linear_prediction_profile_lines(
        [
            "tiles=2",
            "tiles=3",
            "profile_lines=2",
            "profile_lines=3",
            "tile_processing_seconds=1.5",
            "tile_processing_seconds=2.5",
            "command_real_seconds=4.0",
            "command_real_seconds=6.0",
            "raw_fits=32",
            "raw_fits=48",
            "failures=1",
            "failures=2",
            "skipped_fits=4",
            "skipped_fits=6",
            "filtered_fits=7",
            "filtered_fits=8",
        ]
    )

    assert summary.processing_tiles == 5
    assert summary.profile_lines == 5
    assert summary.tile_processing_seconds == 4.0
    assert summary.command_real_seconds == 10.0
    assert summary.raw_fits == 80
    assert summary.failures == 3
    assert summary.skipped_fits == 10
    assert summary.filtered_fits == 15


def test_profile_summary_combines_raw_and_aggregated_files(tmp_path):
    raw_log = tmp_path / "raw.log"
    raw_log.write_text(
        "\n".join(
            [
                "DEBUG:\t20.000s: Processing 1 tiles took   8.500s",
                "linearpred_cupy: ROI [0:16,0:16] raw_fits = 16, "
                "failures = 0, skipped_fits = 0, filtered_fits = 1, "
                "amp_threshold = 1.6",
                "linearpred_profile: ROI [0:16,0:16] "
                "linearpred_total=2.000000s/1, "
                "eigvals_cupy=0.700000s/16",
                "real 12.25",
            ]
        ),
        encoding="utf-8",
    )
    aggregate_log = tmp_path / "aggregate.log"
    aggregate_log.write_text(
        "\n".join(
            [
                "tiles=2",
                "profile_lines=2",
                "tile_processing_seconds=70.461",
                "command_real_seconds=102.02",
                "raw_fits=32",
                "failures=1",
                "skipped_fits=2",
                "filtered_fits=3",
                "linearpred_total_seconds=20.000000 count=2 mean=10.0",
                "eigvals_cupy_seconds=10.000000 count=32 mean=0.3125",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_linear_prediction_profile_files(
        [raw_log, aggregate_log]
    )

    assert summary.profile_lines == 3
    assert summary.processing_tiles == 3
    assert summary.tile_processing_seconds == pytest.approx(78.961)
    assert summary.command_real_seconds == pytest.approx(114.27)
    assert summary.raw_fits == 48
    assert summary.failures == 1
    assert summary.skipped_fits == 2
    assert summary.filtered_fits == 4
    assert summary.amp_thresholds == (1.6,)

    stages = {stage.stage: stage for stage in summary.stages}
    assert stages["linearpred_total"].seconds == 22.0
    assert stages["linearpred_total"].count == 3
    assert stages["eigvals_cupy"].seconds == pytest.approx(10.7)
    assert stages["eigvals_cupy"].count == 48
