# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nonvisual review metadata for subtraction runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_review_metadata(
    artifacts_dir: Path,
    *,
    run_name: str,
    residual: np.ndarray,
    review_metrics: dict[str, float | int],
) -> tuple[dict[str, str], list[dict[str, object]]]:
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


def identify_residual_hotspots(
    residual: np.ndarray,
    *,
    robust_sigma: float,
    threshold_sigma: float = 5.0,
    max_regions: int = 8,
) -> list[dict[str, object]]:
    """Identify the strongest connected residual excursions."""

    from scipy import ndimage

    if robust_sigma <= 0.0:
        raise ValueError("robust_sigma must be positive")
    finite = np.isfinite(residual)
    if not np.any(finite):
        return []
    sigma = np.abs(np.asarray(residual, dtype=np.float64)) / robust_sigma
    mask = finite & (sigma >= threshold_sigma)
    if not np.any(mask):
        return []
    labels, count = ndimage.label(mask)
    hotspots: list[dict[str, object]] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        local_sigma = sigma[ys, xs]
        local_residual = residual[ys, xs]
        peak_index = int(np.argmax(local_sigma))
        peak_y = int(ys[peak_index])
        peak_x = int(xs[peak_index])
        hotspots.append(
            {
                "bbox_y0y1x0x1": [
                    int(ys.min()),
                    int(ys.max()) + 1,
                    int(xs.min()),
                    int(xs.max()) + 1,
                ],
                "pixel_count": int(ys.size),
                "peak_yx": [peak_y, peak_x],
                "centroid_yx": [
                    float(np.mean(ys)),
                    float(np.mean(xs)),
                ],
                "peak_abs_sigma": float(local_sigma[peak_index]),
                "peak_residual": float(local_residual[peak_index]),
                "mean_residual": float(np.mean(local_residual)),
            }
        )
    hotspots.sort(
        key=lambda item: (
            float(item["peak_abs_sigma"]),
            int(item["pixel_count"]),
        ),
        reverse=True,
    )
    return hotspots[:max_regions]
