# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_PROFILE_STAGE_ORDER = (
    "linearpred_total",
    "savgol_cupy",
    "p1_total",
    "trace_to_numpy",
    "hankel_alloc_numpy",
    "hankel_build_numpy",
    "hankel_to_cupy",
    "svd_cupy",
    "component_cap_cupy",
    "coefficients_cupy",
    "eigvals_cupy",
    "eigvals_cusolver_xgeev",
    "eigvals_numpy_fallback",
    "p2_total",
    "tdsfft_cupy",
)

_PROFILE_LINE_RE = re.compile(
    r"linearpred_profile:\s+ROI\s+\S+\s+(?P<payload>.*)$"
)
_PROFILE_COUNTER_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_]+)=" r"(?P<seconds>[-+0-9.eE]+)s/(?P<count>\d+)"
)
_AGGREGATE_STAGE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_]+)_seconds="
    r"(?P<seconds>[-+0-9.eE]+)\s+count=(?P<count>\d+)"
)
_LINEARPRED_FIT_RE = re.compile(
    r"linearpred_cupy:\s+ROI\s+.+?\s+raw_fits\s*=\s*"
    r"(?P<raw_fits>\d+),\s+failures\s*=\s*(?P<failures>\d+)"
    r"(?:,\s+skipped_fits\s*=\s*(?P<skipped_fits>\d+))?"
    r",\s+filtered_fits\s*=\s*(?P<filtered_fits>\d+)"
    r"(?:,\s+amp_threshold\s*=\s*(?P<amp_threshold>[-+0-9.eE]+))?"
)
_ROI_TASK_RE = re.compile(
    r"roi_task_cp:\s+ROI\s+.+?\s+done with\s+"
    r"(?:(?:fits)|(?:filtered_fits))\s*=\s*(?P<filtered_fits>\d+),\s+"
    r"(?:(?:fails)|(?:failures))\s*=\s*(?P<failures>\d+)"
)
_PROCESSING_RE = re.compile(
    r"Processing\s+(?P<tiles>\d+)\s+tiles\s+took\s+"
    r"(?P<seconds>[-+0-9.eE]+)s"
)
_TIME_REAL_RE = re.compile(r"^\s*real\s+(?P<seconds>[-+0-9.eE]+)\s*$")
_SCALAR_RE = re.compile(
    r"^\s*(?P<key>tiles|profile_lines|raw_fits|failures|"
    r"skipped_fits|filtered_fits|tile_processing_seconds|"
    r"gpu_tile_processing_seconds|command_real_seconds)\s*=\s*"
    r"(?P<value>[-+0-9.eE]+)\s*$"
)


@dataclass(frozen=True)
class LinearPredictionProfileStage:
    stage: str
    seconds: float
    count: int
    mean_s: float


@dataclass(frozen=True)
class LinearPredictionLogSummary:
    profile_lines: int
    fit_lines: int
    roi_task_lines: int
    processing_tiles: int | None
    tile_processing_seconds: float | None
    command_real_seconds: float | None
    raw_fits: int
    failures: int
    skipped_fits: int
    filtered_fits: int
    amp_thresholds: tuple[float, ...]
    stages: tuple[LinearPredictionProfileStage, ...]


@dataclass(frozen=True)
class LinearPredictionSteadyStateProfileSummary:
    profile_lines: int
    skipped_profile_lines: int
    included_profile_lines: int
    stages: tuple[LinearPredictionProfileStage, ...]


def summarize_linear_prediction_profile_file(
    path: Path,
) -> LinearPredictionLogSummary:
    return summarize_linear_prediction_profile_lines(_iter_profile_file(path))


def summarize_linear_prediction_profile_files(
    paths: Iterable[Path],
) -> LinearPredictionLogSummary:
    return merge_linear_prediction_profile_summaries(
        summarize_linear_prediction_profile_file(path) for path in paths
    )


def merge_linear_prediction_profile_summaries(
    summaries: Iterable[LinearPredictionLogSummary],
) -> LinearPredictionLogSummary:
    profile_lines = 0
    fit_lines = 0
    roi_task_lines = 0
    processing_tiles = 0
    tile_processing_seconds = 0.0
    command_real_seconds = 0.0
    raw_fits = 0
    failures = 0
    skipped_fits = 0
    filtered_fits = 0
    has_processing_tiles = False
    has_tile_processing_seconds = False
    has_command_real_seconds = False
    amp_thresholds: set[float] = set()
    stage_totals: dict[str, list[float | int]] = {}

    for summary in summaries:
        profile_lines += summary.profile_lines
        fit_lines += summary.fit_lines
        roi_task_lines += summary.roi_task_lines
        raw_fits += summary.raw_fits
        failures += summary.failures
        skipped_fits += summary.skipped_fits
        filtered_fits += summary.filtered_fits
        amp_thresholds.update(summary.amp_thresholds)
        if summary.processing_tiles is not None:
            has_processing_tiles = True
            processing_tiles += summary.processing_tiles
        if summary.tile_processing_seconds is not None:
            has_tile_processing_seconds = True
            tile_processing_seconds += summary.tile_processing_seconds
        if summary.command_real_seconds is not None:
            has_command_real_seconds = True
            command_real_seconds += summary.command_real_seconds
        for stage in summary.stages:
            _add_stage_total(
                stage_totals,
                stage.stage,
                stage.seconds,
                stage.count,
            )

    return LinearPredictionLogSummary(
        profile_lines=profile_lines,
        fit_lines=fit_lines,
        roi_task_lines=roi_task_lines,
        processing_tiles=processing_tiles if has_processing_tiles else None,
        tile_processing_seconds=(
            tile_processing_seconds if has_tile_processing_seconds else None
        ),
        command_real_seconds=(
            command_real_seconds if has_command_real_seconds else None
        ),
        raw_fits=raw_fits,
        failures=failures,
        skipped_fits=skipped_fits,
        filtered_fits=filtered_fits,
        amp_thresholds=tuple(sorted(amp_thresholds)),
        stages=_stage_payloads(stage_totals),
    )


def summarize_linear_prediction_steady_state_profile_file(
    path: Path,
    *,
    skip_profile_lines: int,
) -> LinearPredictionSteadyStateProfileSummary:
    return summarize_linear_prediction_steady_state_profile_lines(
        _iter_profile_file(path),
        skip_profile_lines=skip_profile_lines,
    )


def summarize_linear_prediction_steady_state_profile_files(
    paths: Iterable[Path],
    *,
    skip_profile_lines: int,
) -> LinearPredictionSteadyStateProfileSummary:
    return merge_linear_prediction_steady_state_profile_summaries(
        summarize_linear_prediction_steady_state_profile_file(
            path,
            skip_profile_lines=skip_profile_lines,
        )
        for path in paths
    )


def merge_linear_prediction_steady_state_profile_summaries(
    summaries: Iterable[LinearPredictionSteadyStateProfileSummary],
) -> LinearPredictionSteadyStateProfileSummary:
    profile_lines = 0
    skipped_profile_lines = 0
    included_profile_lines = 0
    stage_totals: dict[str, list[float | int]] = {}

    for summary in summaries:
        profile_lines += summary.profile_lines
        skipped_profile_lines += summary.skipped_profile_lines
        included_profile_lines += summary.included_profile_lines
        for stage in summary.stages:
            _add_stage_total(
                stage_totals,
                stage.stage,
                stage.seconds,
                stage.count,
            )

    return LinearPredictionSteadyStateProfileSummary(
        profile_lines=profile_lines,
        skipped_profile_lines=skipped_profile_lines,
        included_profile_lines=included_profile_lines,
        stages=_stage_payloads(stage_totals),
    )


def summarize_linear_prediction_profile_lines(
    lines: Iterable[str],
) -> LinearPredictionLogSummary:
    profile_stage_totals: dict[str, list[float | int]] = {}
    aggregate_stage_totals: dict[str, list[float | int]] = {}
    scalar_totals: dict[str, float] = {}
    amp_thresholds: set[float] = set()

    profile_lines = 0
    fit_lines = 0
    roi_task_lines = 0
    fit_raw_fits = 0
    fit_failures = 0
    fit_skipped_fits = 0
    fit_filtered_fits = 0
    roi_task_failures = 0
    roi_task_filtered_fits = 0
    processing_lines = 0
    processing_tiles_total = 0
    tile_processing_seconds_total = 0.0
    command_real_lines = 0
    command_real_seconds_total = 0.0

    for line in lines:
        profile_match = _PROFILE_LINE_RE.search(line)
        if profile_match:
            profile_lines += 1
            _add_profile_counters(
                profile_stage_totals,
                profile_match.group("payload"),
            )
            continue

        aggregate_match = _AGGREGATE_STAGE_RE.match(line)
        if aggregate_match:
            _add_stage_total(
                aggregate_stage_totals,
                aggregate_match.group("name"),
                float(aggregate_match.group("seconds")),
                int(aggregate_match.group("count")),
            )
            continue

        fit_match = _LINEARPRED_FIT_RE.search(line)
        if fit_match:
            fit_lines += 1
            fit_raw_fits += int(fit_match.group("raw_fits"))
            fit_failures += int(fit_match.group("failures"))
            fit_skipped_fits += int(fit_match.group("skipped_fits") or 0)
            fit_filtered_fits += int(fit_match.group("filtered_fits"))
            threshold = fit_match.group("amp_threshold")
            if threshold is not None:
                amp_thresholds.add(float(threshold))
            continue

        roi_task_match = _ROI_TASK_RE.search(line)
        if roi_task_match:
            roi_task_lines += 1
            roi_task_failures += int(roi_task_match.group("failures"))
            roi_task_filtered_fits += int(
                roi_task_match.group("filtered_fits")
            )
            continue

        processing_match = _PROCESSING_RE.search(line)
        if processing_match:
            processing_lines += 1
            processing_tiles_total += int(processing_match.group("tiles"))
            tile_processing_seconds_total += float(
                processing_match.group("seconds")
            )
            continue

        real_match = _TIME_REAL_RE.match(line)
        if real_match:
            command_real_lines += 1
            command_real_seconds_total += float(real_match.group("seconds"))
            continue

        scalar_match = _SCALAR_RE.match(line)
        if scalar_match:
            key = scalar_match.group("key")
            scalar_totals[key] = scalar_totals.get(key, 0.0) + float(
                scalar_match.group("value")
            )

    if processing_lines:
        processing_tiles = processing_tiles_total
        tile_processing_seconds = tile_processing_seconds_total
    else:
        processing_tiles = _optional_int_scalar(scalar_totals, "tiles")
        tile_processing_seconds = _first_scalar(
            scalar_totals,
            "tile_processing_seconds",
            "gpu_tile_processing_seconds",
        )
    if command_real_lines:
        command_real_seconds = command_real_seconds_total
    else:
        command_real_seconds = scalar_totals.get("command_real_seconds")

    profile_lines = profile_lines or int(
        scalar_totals.get("profile_lines", 0)
    )
    raw_fits = fit_raw_fits or int(scalar_totals.get("raw_fits", 0))
    skipped_fits = fit_skipped_fits or int(
        scalar_totals.get("skipped_fits", 0)
    )
    failures = _prefer_detailed_count(
        fit_lines,
        fit_failures,
        scalar_totals.get("failures"),
        roi_task_lines,
        roi_task_failures,
    )
    filtered_fits = _prefer_detailed_count(
        fit_lines,
        fit_filtered_fits,
        scalar_totals.get("filtered_fits"),
        roi_task_lines,
        roi_task_filtered_fits,
    )
    stage_totals = profile_stage_totals or aggregate_stage_totals

    return LinearPredictionLogSummary(
        profile_lines=profile_lines,
        fit_lines=fit_lines,
        roi_task_lines=roi_task_lines,
        processing_tiles=processing_tiles,
        tile_processing_seconds=tile_processing_seconds,
        command_real_seconds=command_real_seconds,
        raw_fits=raw_fits,
        failures=failures,
        skipped_fits=skipped_fits,
        filtered_fits=filtered_fits,
        amp_thresholds=tuple(sorted(amp_thresholds)),
        stages=_stage_payloads(stage_totals),
    )


def summarize_linear_prediction_steady_state_profile_lines(
    lines: Iterable[str],
    *,
    skip_profile_lines: int,
) -> LinearPredictionSteadyStateProfileSummary:
    if skip_profile_lines < 0:
        raise ValueError("skip_profile_lines must be non-negative")

    profile_stage_totals: dict[str, list[float | int]] = {}
    profile_lines = 0
    skipped_profile_lines = 0
    included_profile_lines = 0

    for line in lines:
        profile_match = _PROFILE_LINE_RE.search(line)
        if not profile_match:
            continue

        profile_lines += 1
        if skipped_profile_lines < skip_profile_lines:
            skipped_profile_lines += 1
            continue

        included_profile_lines += 1
        _add_profile_counters(
            profile_stage_totals,
            profile_match.group("payload"),
        )

    return LinearPredictionSteadyStateProfileSummary(
        profile_lines=profile_lines,
        skipped_profile_lines=skipped_profile_lines,
        included_profile_lines=included_profile_lines,
        stages=_stage_payloads(profile_stage_totals),
    )


def _iter_profile_file(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n")


def _add_profile_counters(
    totals: dict[str, list[float | int]],
    payload: str,
) -> None:
    for match in _PROFILE_COUNTER_RE.finditer(payload):
        _add_stage_total(
            totals,
            match.group("name"),
            float(match.group("seconds")),
            int(match.group("count")),
        )


def _add_stage_total(
    totals: dict[str, list[float | int]],
    stage: str,
    seconds: float,
    count: int,
) -> None:
    current = totals.setdefault(stage, [0.0, 0])
    current[0] = float(current[0]) + seconds
    current[1] = int(current[1]) + count


def _stage_payloads(
    totals: dict[str, list[float | int]],
) -> tuple[LinearPredictionProfileStage, ...]:
    stages = []
    for stage, value in sorted(totals.items(), key=_stage_sort_key):
        seconds = float(value[0])
        count = int(value[1])
        stages.append(
            LinearPredictionProfileStage(
                stage=stage,
                seconds=seconds,
                count=count,
                mean_s=seconds / count if count else 0.0,
            )
        )
    return tuple(stages)


def _stage_sort_key(item: tuple[str, object]) -> tuple[int, str]:
    stage = item[0]
    try:
        return (_PROFILE_STAGE_ORDER.index(stage), stage)
    except ValueError:
        return (len(_PROFILE_STAGE_ORDER), stage)


def _first_scalar(
    scalar_totals: dict[str, float],
    *keys: str,
) -> float | None:
    for key in keys:
        if key in scalar_totals:
            return scalar_totals[key]
    return None


def _optional_int_scalar(
    scalar_totals: dict[str, float],
    key: str,
) -> int | None:
    if key not in scalar_totals:
        return None
    return int(scalar_totals[key])


def _prefer_detailed_count(
    detail_lines: int,
    detail_count: int,
    scalar_count: float | None,
    fallback_lines: int,
    fallback_count: int,
) -> int:
    if detail_lines:
        return detail_count
    if scalar_count is not None:
        return int(scalar_count)
    if fallback_lines:
        return fallback_count
    return 0


__all__ = [
    "LinearPredictionLogSummary",
    "LinearPredictionProfileStage",
    "LinearPredictionSteadyStateProfileSummary",
    "merge_linear_prediction_steady_state_profile_summaries",
    "merge_linear_prediction_profile_summaries",
    "summarize_linear_prediction_steady_state_profile_file",
    "summarize_linear_prediction_steady_state_profile_files",
    "summarize_linear_prediction_steady_state_profile_lines",
    "summarize_linear_prediction_profile_file",
    "summarize_linear_prediction_profile_files",
    "summarize_linear_prediction_profile_lines",
]
