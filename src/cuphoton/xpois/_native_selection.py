# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native compact-source filtering with the shared ordered host selector."""

from __future__ import annotations

from typing import Any

import numpy as np

from cuphoton.core._native_imaging import asarray, boolean_mask, extension


def build_mask(
    image: Any,
    *,
    variance: Any,
    valid_mask: Any,
    stamp_size: int,
    max_stamps: int,
    peak_percentile: float,
    background_filter_size: int,
    peak_filter_size: int,
    min_separation: int | None,
    compact_core_size: int,
) -> Any:
    import cupy as cp

    from .ois import AutoStampMaskResult, _select_separated_sources

    array = asarray(image, np.float64)
    if array.ndim != 2:
        raise ValueError("build_compact_source_stamp_mask expects a 2D image")
    variance_array = (
        None if variance is None else asarray(variance, np.float64)
    )
    mask = None if valid_mask is None else boolean_mask(valid_mask)
    records, threshold = extension().selection_candidates(
        array,
        variance_array,
        mask,
        background_filter_size,
        peak_filter_size,
        stamp_size,
        compact_core_size,
        peak_percentile,
        cp.cuda.get_current_stream().ptr,
    )
    dtype = np.dtype(
        [
            ("score", np.float64),
            ("compact_fraction", np.float64),
            ("negative_radial", np.float64),
            ("y", np.int64),
            ("x", np.int64),
        ]
    )
    candidates = np.frombuffer(records, dtype=dtype).view(np.recarray)
    ordered = candidates[
        np.lexsort(
            (
                -candidates.x,
                -candidates.y,
                -np.round(candidates.negative_radial, 6),
                -np.round(candidates.compact_fraction, 9),
                -candidates.score,
            )
        )
    ]
    half = stamp_size // 2
    distance = max(1, half) if min_separation is None else min_separation
    if distance <= 0:
        raise ValueError("min_separation must be positive")
    selected = _select_separated_sources(
        ordered, array.shape, distance, max_stamps
    )
    if not selected:
        raise ValueError(
            "compact-source selection produced no separated stamps"
        )
    stamps = tuple(
        (
            int(row.y) - half,
            int(row.y) + half + 1,
            int(row.x) - half,
            int(row.x) + half + 1,
        )
        for row in selected
    )
    output = cp.empty(array.shape, dtype=cp.bool_)
    boxes = cp.asarray(np.asarray(stamps, dtype=np.int64))
    extension().stamp_mask(boxes, output, cp.cuda.get_current_stream().ptr)
    return AutoStampMaskResult(
        mask=output,
        centers=tuple((int(row.y), int(row.x)) for row in selected),
        stamps=stamps,
        scores=tuple(float(row.score) for row in selected),
        compact_fractions=tuple(
            float(row.compact_fraction) for row in selected
        ),
        radial_moments=tuple(float(-row.negative_radial) for row in selected),
        threshold=threshold,
        peak_percentile=float(peak_percentile),
        stamp_size=stamp_size,
        max_stamps=max_stamps,
        background_filter_size=background_filter_size,
        peak_filter_size=peak_filter_size,
        min_separation=distance,
    )
