# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nonvisual review metadata for subtraction runs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from cuphoton.core.arrays import array_module, ndimage_module, to_host


class ResidualHotspot(TypedDict):
    """Numeric summary of one connected residual excursion."""

    bbox_y0y1x0x1: list[int]
    pixel_count: int
    peak_yx: list[int]
    centroid_yx: list[float]
    peak_abs_sigma: float
    peak_residual: float
    mean_residual: float


def _component_coordinates(
    labels: np.ndarray, count: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Group labeled pixels while retaining raster order within each group."""

    ys, xs = np.nonzero(labels)
    if count == 1:
        yield ys, xs
        return
    foreground_labels = labels[ys, xs]
    order = np.argsort(foreground_labels, kind="stable")
    ys, xs = ys[order], xs[order]
    stops = np.cumsum(np.bincount(foreground_labels, minlength=count + 1)[1:])
    start = 0
    for stop in stops:
        yield ys[start:stop], xs[start:stop]
        start = stop


def write_review_metadata(
    artifacts_dir: Path,
    *,
    run_name: str,
    residual: np.ndarray,
    review_metrics: dict[str, float | int],
) -> tuple[dict[str, str], list[ResidualHotspot]]:
    """Persist numeric residual-hotspot metadata without static plots."""

    robust_sigma = max(float(review_metrics["robust_sigma"]), 1.0e-12)
    hotspots = identify_residual_hotspots(
        residual,
        robust_sigma=robust_sigma,
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artifacts_dir / "review_hotspots.json"
    metadata_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "review_metrics": review_metrics,
                "robust_sigma": robust_sigma,
                "threshold_sigma": 5.0,
                "max_regions": 8,
                "hotspots": hotspots,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        {
            "review_hotspots_metadata": str(
                metadata_path.relative_to(artifacts_dir.parent)
            )
        },
        hotspots,
    )


_HOTSPOT_COLUMNS = (
    "pixel_count",
    "peak_y",
    "peak_x",
    "centroid_y",
    "centroid_x",
    "peak_abs_sigma",
    "peak_residual",
    "mean_residual",
    "y0",
    "y1",
    "x0",
    "x1",
)


def _segment_bounds(xp: Any, labels: Any, values: Any, count: int) -> Any:
    """Return ``(count, 2)`` min and max of ``values`` per label."""

    order = xp.lexsort(xp.stack((values, labels)))
    sorted_labels = labels[order]
    ids = xp.arange(1, count + 1)
    first = xp.searchsorted(sorted_labels, ids, side="left")
    last = xp.searchsorted(sorted_labels, ids, side="right") - 1
    return values[order][first], values[order][last]


def _segment_means(xp: Any, labels: Any, values: Any, count: int) -> Any:
    """Mean of ``values`` per label; ``values`` sorted by label, raster order.

    On the host each segment is reduced with :func:`numpy.mean` in the dtype
    of ``values``, which keeps the result identical to a per-component scan
    of the residual. On the device the means accumulate in float64.
    """

    if xp is np:
        stops = np.cumsum(np.bincount(labels, minlength=count + 1)[1:])
        starts = stops - np.diff(stops, prepend=0)
        return np.array(
            [
                float(np.mean(values[start:stop]))
                for start, stop in zip(starts, stops, strict=True)
            ]
        )
    weights = values.astype(xp.float64, copy=False)
    return (
        xp.bincount(labels, weights=weights, minlength=count + 1)[1:]
        / xp.bincount(labels, minlength=count + 1)[1:]
    )


def residual_hotspot_table(
    residual: Any,
    *,
    robust_sigma: float,
    threshold_sigma: float = 5.0,
    max_regions: int | None = None,
) -> dict[str, np.ndarray]:
    """Tabulate the connected residual excursions above a sigma threshold.

    Parameters
    ----------
    residual
        Two-dimensional NumPy or CuPy residual image; non-finite pixels are
        ignored.
    robust_sigma
        Positive noise scale that converts the residual to sigma units.
    threshold_sigma
        Pixels with ``|residual| / robust_sigma`` at or above this value
        form the excursions.
    max_regions
        Keep only ``[:max_regions]`` of the sorted regions; ``None`` keeps
        all of them.

    Returns
    -------
    dict[str, numpy.ndarray]
        Host columns ``pixel_count``, ``peak_y``, ``peak_x``,
        ``centroid_y``, ``centroid_x``, ``peak_abs_sigma``,
        ``peak_residual``, ``mean_residual`` and the half-open bounding box
        ``y0``, ``y1``, ``x0``, ``x1``, one row per region sorted by
        ``peak_abs_sigma`` then ``pixel_count`` descending, ties in label
        order. The peak is the first pixel in raster order that attains the
        region's maximum sigma. Every column is empty when nothing crosses
        the threshold.
    """

    if robust_sigma <= 0.0:
        raise ValueError("robust_sigma must be positive")
    xp = array_module(residual)
    residual = xp.asarray(residual)
    empty = {name: np.empty(0) for name in _HOTSPOT_COLUMNS}
    finite = xp.isfinite(residual)
    if residual.ndim != 2 or not bool(finite.any()):
        return empty
    sigma = xp.abs(residual.astype(xp.float64)) / robust_sigma
    mask = finite & (sigma >= threshold_sigma)
    if not bool(mask.any()):
        return empty
    labels, label_count = ndimage_module(xp).label(mask)
    count = int(label_count)
    flat = xp.flatnonzero(mask)
    label = labels.ravel()[flat]
    local_sigma = sigma.ravel()[flat]
    local_residual = residual.ravel()[flat]
    ys, xs = xp.divmod(flat, residual.shape[1])
    ids = xp.arange(1, count + 1)
    size = xp.bincount(label, minlength=count + 1)[1:]
    centroid_y = (
        xp.bincount(label, weights=ys, minlength=count + 1)[1:] / size
    )
    centroid_x = (
        xp.bincount(label, weights=xs, minlength=count + 1)[1:] / size
    )
    by_label = xp.lexsort(xp.stack((flat, label)))
    mean_residual = _segment_means(
        xp, label[by_label], local_residual[by_label], count
    )
    by_sigma = xp.lexsort(xp.stack((flat, -local_sigma, label)))
    peak = by_sigma[xp.searchsorted(label[by_sigma], ids)]
    y0, y1 = _segment_bounds(xp, label, ys, count)
    x0, x1 = _segment_bounds(xp, label, xs, count)
    order = xp.lexsort(xp.stack((ids, -size, -local_sigma[peak])))
    if max_regions is not None:
        order = order[:max_regions]
    columns = (
        size,
        ys[peak],
        xs[peak],
        centroid_y,
        centroid_x,
        local_sigma[peak],
        local_residual[peak],
        mean_residual,
        y0,
        y1 + 1,
        x0,
        x1 + 1,
    )
    return {
        name: to_host(column[order])
        for name, column in zip(_HOTSPOT_COLUMNS, columns, strict=True)
    }


def identify_residual_hotspots(
    residual: Any,
    *,
    robust_sigma: float,
    threshold_sigma: float = 5.0,
    max_regions: int = 8,
) -> list[ResidualHotspot]:
    """Identify the strongest connected residual excursions.

    Parameters
    ----------
    residual
        Two-dimensional NumPy or CuPy residual image.
    robust_sigma, threshold_sigma, max_regions
        As in :func:`residual_hotspot_table`.

    Returns
    -------
    list[ResidualHotspot]
        The rows of :func:`residual_hotspot_table` as host records.
    """

    table = residual_hotspot_table(
        residual,
        robust_sigma=robust_sigma,
        threshold_sigma=threshold_sigma,
        max_regions=max_regions,
    )
    return [
        {
            "bbox_y0y1x0x1": [int(y0), int(y1), int(x0), int(x1)],
            "pixel_count": int(pixel_count),
            "peak_yx": [int(peak_y), int(peak_x)],
            "centroid_yx": [float(centroid_y), float(centroid_x)],
            "peak_abs_sigma": float(peak_abs_sigma),
            "peak_residual": float(peak_residual),
            "mean_residual": float(mean_residual),
        }
        for (
            pixel_count,
            peak_y,
            peak_x,
            centroid_y,
            centroid_x,
            peak_abs_sigma,
            peak_residual,
            mean_residual,
            y0,
            y1,
            x0,
            x1,
        ) in zip(*(table[name] for name in _HOTSPOT_COLUMNS), strict=True)
    ]
