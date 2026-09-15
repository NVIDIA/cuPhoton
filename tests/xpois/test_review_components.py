# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare complete summaries against the original full-image scans."""

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xpois.review import identify_residual_hotspots
from cuphoton.xpois.review_bokeh import identify_mask_components


def _reference_identify_residual_hotspots(
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


def _reference_identify_mask_components(
    mask_values: np.ndarray | None,
    *,
    raw_image: np.ndarray,
    plane_map: dict[str, int] | None,
    masked_plane_names: list[str] | None,
    max_regions: int = 32,
) -> list[dict[str, object]]:
    """Summarize connected masked regions for interactive hover overlays."""

    from scipy import ndimage

    if mask_values is None or plane_map is None or not masked_plane_names:
        return []

    missing = [name for name in masked_plane_names if name not in plane_map]
    if missing:
        return []

    bitmask = 0
    for name in masked_plane_names:
        bitmask |= 1 << int(plane_map[name])

    mask = (np.asarray(mask_values, dtype=np.int64) & bitmask) != 0
    if not np.any(mask):
        return []

    labels, count = ndimage.label(mask)
    components: list[dict[str, object]] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        values = np.unique(np.asarray(mask_values)[ys, xs])
        planes: set[str] = set()
        for value in values.tolist():
            for name, bit in plane_map.items():
                if int(value) & (1 << int(bit)):
                    planes.add(name)

        bbox = [
            int(ys.min()),
            int(ys.max()) + 1,
            int(xs.min()),
            int(xs.max()) + 1,
        ]
        local_image = np.asarray(raw_image)[ys, xs]
        finite = local_image[np.isfinite(local_image)]
        components.append(
            {
                "bbox_y0y1x0x1": bbox,
                "pixel_count": int(ys.size),
                "planes": sorted(planes),
                "values": [int(v) for v in values.tolist()],
                "centroid_yx": [float(np.mean(ys)), float(np.mean(xs))],
                "max_signal": float(np.max(finite)) if finite.size else None,
            }
        )

    components.sort(key=lambda item: int(item["pixel_count"]), reverse=True)
    return components[:max_regions]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_regions", [0, 1, 8, -1, 1000])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_hotspots_match_full_scan(dtype, max_regions, layout):
    residual = np.random.default_rng(42).normal(0, 3, (32, 48)).astype(dtype)
    residual[0, :4] = [np.nan, np.inf, -np.inf, 5.0]
    if layout == "F":
        residual = np.asfortranarray(residual)
    elif layout == "reversed":
        residual = residual[::-1, ::-1]
    options = dict(robust_sigma=1.0, max_regions=max_regions)
    assert identify_residual_hotspots(
        residual, **options
    ) == _reference_identify_residual_hotspots(residual, **options)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hotspot_mean_preserves_dtype_and_raster_reduction(dtype):
    residual = (
        np.random.default_rng(42).uniform(6, 1000, (16, 16)).astype(dtype)
    )
    result = identify_residual_hotspots(residual, robust_sigma=1.0)
    assert result == _reference_identify_residual_hotspots(
        residual, robust_sigma=1.0
    )
    assert result[0]["mean_residual"] == float(np.mean(residual.ravel()))


@pytest.mark.parametrize(
    "residual",
    [
        np.empty((0, 0)),
        np.full((4, 4), np.nan),
        np.array([[np.inf, -np.inf]]),
        np.zeros((4, 4)),
        np.array([[0.0, 4.99, 5.0, -5.0, 0.0, 5.01]]),
    ],
)
def test_hotspot_empty_nonfinite_and_threshold_boundaries(residual):
    assert identify_residual_hotspots(
        residual, robust_sigma=1.0
    ) == _reference_identify_residual_hotspots(residual, robust_sigma=1.0)


def test_hotspot_peak_and_component_ties_use_raster_order():
    residual = np.array([[5.0, -5.0, 0.0, -5.0, 5.0], [0, 0, 5, 0, 0]])
    result = identify_residual_hotspots(residual, robust_sigma=1.0)
    assert result == _reference_identify_residual_hotspots(
        residual, robust_sigma=1.0
    )
    assert [item["peak_yx"] for item in result] == [[0, 0], [0, 3], [1, 2]]
    assert [item["pixel_count"] for item in result] == [2, 2, 1]


@pytest.mark.parametrize("robust_sigma", [0.0, -1.0])
def test_hotspots_reject_nonpositive_sigma(robust_sigma):
    with pytest.raises(ValueError, match="robust_sigma must be positive"):
        identify_residual_hotspots(np.ones((2, 2)), robust_sigma=robust_sigma)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_regions", [0, 1, 32, -1, 1000])
def test_mask_components_match_full_scan(dtype, max_regions):
    rng = np.random.default_rng(7)
    mask = rng.choice([0, 0, 0, 0, 0, 1, 2, 17, -1], size=(32, 48))
    raw = rng.normal(size=mask.shape).astype(dtype)
    raw[:3, :3] = np.nan
    raw[-3:, -3:] = np.inf
    options = dict(
        raw_image=raw,
        plane_map={"BAD": 0, "SAT": 1, "OTHER": 4},
        masked_plane_names=["BAD", "SAT"],
        max_regions=max_regions,
    )
    assert identify_mask_components(
        mask, **options
    ) == _reference_identify_mask_components(mask, **options)


def test_mask_components_preserve_all_bits_ties_and_nonfinite_signal():
    mask = np.array([[1, 17, 0, 2, 2], [0, 0, 1, 0, 0]], dtype=np.uint32)
    raw = np.array([[np.nan, np.inf, 0, 3, 4], [0, 0, -np.inf, 0, 0]])
    options = dict(
        raw_image=raw,
        plane_map={"BAD": 0, "SAT": 1, "OTHER": 4},
        masked_plane_names=["BAD", "SAT"],
    )
    result = identify_mask_components(mask, **options)
    assert result == _reference_identify_mask_components(mask, **options)
    assert result[0] == {
        "bbox_y0y1x0x1": [0, 1, 0, 2],
        "pixel_count": 2,
        "planes": ["BAD", "OTHER"],
        "values": [1, 17],
        "centroid_yx": [0.0, 0.5],
        "max_signal": None,
    }
    assert result[1]["max_signal"] == 4.0
    assert result[2]["max_signal"] is None


@pytest.mark.parametrize(
    ("mask", "plane_map", "names"),
    [
        (None, {"BAD": 0}, ["BAD"]),
        (np.ones((2, 2)), None, ["BAD"]),
        (np.ones((2, 2)), {"BAD": 0}, []),
        (np.ones((2, 2)), {"BAD": 0}, ["MISSING"]),
        (np.zeros((2, 2)), {"BAD": 0}, ["BAD"]),
    ],
)
def test_mask_components_without_selected_pixels(mask, plane_map, names):
    assert (
        identify_mask_components(
            mask,
            raw_image=np.ones((2, 2)),
            plane_map=plane_map,
            masked_plane_names=names,
        )
        == []
    )
