# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True)
class AxisRange:
    """Half-open detector-row interval ``[start, end)`` in pixels."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("range start must be non-negative")
        if self.end <= self.start:
            raise ValueError("range end must be greater than start")

    def overlaps(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end

    def clipped(self, start: int, end: int) -> AxisRange | None:
        clipped_start = max(self.start, start)
        clipped_end = min(self.end, end)
        if clipped_end <= clipped_start:
            return None
        return AxisRange(clipped_start, clipped_end)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.start}:{self.end}"


@dataclass(frozen=True)
class DetectorMaskSummary:
    """Tile and row-fit counts for a masked detector ROI.

    ROI origins and dimensions use detector pixels; ``exclude_y`` contains
    global half-open row intervals rather than ROI-local offsets.
    """

    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    tile_width: int
    tile_height: int
    total_tiles: int
    active_tiles: int
    fully_excluded_tiles: int
    partially_excluded_tiles: int
    total_row_fits: int
    active_row_fits: int
    skipped_row_fits: int
    exclude_y: tuple[AxisRange, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exclude_y"] = [item.to_dict() for item in self.exclude_y]
        return payload


def parse_axis_range(value: str) -> AxisRange:
    text = value.strip()
    if not text:
        raise ValueError("empty detector mask range")
    if "=" in text:
        axis, text = text.split("=", 1)
        if axis.strip().lower() not in {"y", "row", "rows"}:
            raise ValueError(f"unsupported detector mask axis: {axis!r}")
    text = text.strip().strip("[]")
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"detector mask range must be start:end: {value!r}")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"detector mask range must use integers: {value!r}"
        ) from exc
    return AxisRange(start, end)


def parse_y_ranges(
    values: str | Iterable[str] | None,
) -> tuple[AxisRange, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)

    ranges: list[AxisRange] = []
    for raw_value in raw_values:
        for part in str(raw_value).split(","):
            part = part.strip()
            if part:
                ranges.append(parse_axis_range(part))
    return _merge_ranges(ranges)


def format_y_ranges(ranges: Iterable[AxisRange]) -> str:
    return ",".join(str(item) for item in ranges)


def excluded_row_mask(
    tile_y_start: int,
    tile_y_end: int,
    exclude_y: Iterable[AxisRange],
) -> tuple[bool, ...]:
    if tile_y_end <= tile_y_start:
        raise ValueError("tile y end must be greater than y start")
    excluded = [False] * (tile_y_end - tile_y_start)
    for item in exclude_y:
        clipped = item.clipped(tile_y_start, tile_y_end)
        if clipped is None:
            continue
        for row in range(clipped.start, clipped.end):
            excluded[row - tile_y_start] = True
    return tuple(excluded)


def summarize_tiled_roi(
    *,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    tile_width: int,
    tile_height: int,
    exclude_y: Iterable[AxisRange],
) -> DetectorMaskSummary:
    if roi_x < 0 or roi_y < 0:
        raise ValueError("ROI origin must be non-negative")
    for name, value in (
        ("roi_width", roi_width),
        ("roi_height", roi_height),
        ("tile_width", tile_width),
        ("tile_height", tile_height),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    ranges = _merge_ranges(exclude_y)
    active_tiles = 0
    fully_excluded_tiles = 0
    partially_excluded_tiles = 0
    total_row_fits = 0
    skipped_row_fits = 0

    x_starts = range(roi_x, roi_x + roi_width, tile_width)
    y_starts = range(roi_y, roi_y + roi_height, tile_height)
    total_tiles = 0
    for _x in x_starts:
        for y_start in y_starts:
            y_end = min(y_start + tile_height, roi_y + roi_height)
            mask = excluded_row_mask(y_start, y_end, ranges)
            skipped = sum(mask)
            rows = len(mask)
            total_tiles += 1
            total_row_fits += rows
            skipped_row_fits += skipped
            if skipped == rows:
                fully_excluded_tiles += 1
            elif skipped:
                partially_excluded_tiles += 1
                active_tiles += 1
            else:
                active_tiles += 1

    return DetectorMaskSummary(
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
        tile_width=tile_width,
        tile_height=tile_height,
        total_tiles=total_tiles,
        active_tiles=active_tiles,
        fully_excluded_tiles=fully_excluded_tiles,
        partially_excluded_tiles=partially_excluded_tiles,
        total_row_fits=total_row_fits,
        active_row_fits=total_row_fits - skipped_row_fits,
        skipped_row_fits=skipped_row_fits,
        exclude_y=ranges,
    )


def _merge_ranges(ranges: Iterable[AxisRange]) -> tuple[AxisRange, ...]:
    sorted_ranges = sorted(ranges)
    if not sorted_ranges:
        return ()

    merged: list[AxisRange] = [sorted_ranges[0]]
    for item in sorted_ranges[1:]:
        last = merged[-1]
        if item.start <= last.end:
            merged[-1] = AxisRange(last.start, max(last.end, item.end))
        else:
            merged.append(item)
    return tuple(merged)
