# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare the array-namespace compact-source selector to a per-peak scan."""

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.xpois import build_compact_source_stamp_mask


def _reference_compact_source_stamp_mask(
    image: np.ndarray,
    *,
    variance: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    stamp_size: int = 31,
    max_stamps: int = 5,
    peak_percentile: float = 99.5,
    background_filter_size: int = 15,
    peak_filter_size: int = 9,
    min_separation: int | None = None,
    compact_core_size: int = 3,
) -> dict[str, object]:
    """Per-peak Python scan with the same selection rule."""

    from scipy.ndimage import maximum_filter, median_filter

    array = np.asarray(image, dtype=np.float64)
    base_valid = np.isfinite(array)
    if variance is not None:
        variance = np.asarray(variance, dtype=np.float64)
        base_valid &= np.isfinite(variance) & (variance > 0)
    if valid_mask is not None:
        base_valid &= np.asarray(valid_mask, dtype=bool)
    fill_value = float(np.median(array[base_valid]))
    background = median_filter(
        np.where(base_valid, array, fill_value),
        size=background_filter_size,
        mode="nearest",
    )
    highpass = np.where(base_valid, array - background, -np.inf)
    threshold = float(np.percentile(highpass[base_valid], peak_percentile))
    maxima = maximum_filter(highpass, size=peak_filter_size, mode="nearest")
    peaks = np.argwhere(
        (highpass == maxima) & base_valid & (highpass > threshold)
    )
    half = stamp_size // 2
    min_distance = max(1, half) if min_separation is None else min_separation
    center_half = compact_core_size // 2
    candidates = []
    for y, x in peaks:
        y, x = int(y), int(x)
        y0, y1, x0, x1 = y - half, y + half + 1, x - half, x + half + 1
        if y0 < 0 or x0 < 0 or y1 > array.shape[0] or x1 > array.shape[1]:
            continue
        if not np.all(base_valid[y0:y1, x0:x1]):
            continue
        stamp = array[y0:y1, x0:x1]
        local_background = float(np.median(stamp))
        weights = np.clip(stamp - local_background, 0.0, None)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            continue
        center_weights = weights[
            half - center_half : half + center_half + 1,
            half - center_half : half + center_half + 1,
        ]
        compact_fraction = float(np.sum(center_weights) / total_weight)
        yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
        radial_moment = float(
            np.sum(weights * (xx * xx + yy * yy)) / total_weight
        )
        local_signal = float(stamp[half, half] - local_background)
        score = local_signal
        if variance is not None:
            score = float(local_signal / np.sqrt(variance[y, x]))
        candidates.append(
            (score, compact_fraction, -radial_moment, y, x, (y0, y1, x0, x1))
        )
    candidates.sort(reverse=True)
    selected: list[tuple] = []
    for candidate in candidates:
        _, _, _, y, x, _ = candidate
        if any(
            (y - y2) ** 2 + (x - x2) ** 2 < min_distance**2
            for *_, y2, x2, _ in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_stamps:
            break
    mask = np.zeros(array.shape, dtype=bool)
    for *_, (y0, y1, x0, x1) in selected:
        mask[y0:y1, x0:x1] = True
    return {
        "mask": mask,
        "centers": tuple((y, x) for _, _, _, y, x, _ in selected),
        "stamps": tuple(stamp for *_, stamp in selected),
        "scores": tuple(score for score, *_ in selected),
        "compact_fractions": tuple(c for _, c, *_ in selected),
        "radial_moments": tuple(-r for _, _, r, *_ in selected),
        "threshold": threshold,
    }


def _scene(
    seed: int = 1, shape: tuple[int, int] = (700, 900), count: int = 120
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(100.0, 3.0, shape)
    yy, xx = np.mgrid[-6:7, -6:7]
    for _ in range(count):
        y = rng.integers(30, shape[0] - 30)
        x = rng.integers(30, shape[1] - 30)
        amplitude, width = rng.uniform(50, 3000), rng.uniform(1.4, 2.2)
        image[y - 6 : y + 7, x - 6 : x + 7] += amplitude * np.exp(
            -(xx**2 + yy**2) / (2 * width * width)
        )
    image[10:20, 10:20] = np.nan
    return image


def _assert_matches_reference(result, reference) -> None:
    assert result.centers == reference["centers"]
    assert result.stamps == reference["stamps"]
    assert result.threshold == reference["threshold"]
    assert np.array_equal(result.mask, reference["mask"])
    for name in ("scores", "compact_fractions", "radial_moments"):
        assert getattr(result, name) == pytest.approx(
            reference[name], rel=1e-12
        )


@pytest.mark.parametrize("with_variance", [False, True])
@pytest.mark.parametrize("shape", [(700, 900), (257, 1024)])
def test_scene_selection_matches_reference_scan(shape, with_variance):
    image = _scene(shape=shape)
    variance = np.full(image.shape, 9.0) if with_variance else None
    options = dict(
        variance=variance, stamp_size=31, max_stamps=5, peak_percentile=99.5
    )
    result = build_compact_source_stamp_mask(image, **options)
    reference = _reference_compact_source_stamp_mask(image, **options)
    assert len(result.centers) == 5
    _assert_matches_reference(result, reference)


def test_valid_mask_and_small_stamps_match_reference_scan():
    image = _scene(seed=3, shape=(300, 400), count=60)
    valid_mask = np.ones(image.shape, dtype=bool)
    valid_mask[120:180, :] = False
    options = dict(
        valid_mask=valid_mask,
        stamp_size=11,
        max_stamps=8,
        peak_percentile=98.0,
        min_separation=20,
        compact_core_size=5,
    )
    result = build_compact_source_stamp_mask(image, **options)
    reference = _reference_compact_source_stamp_mask(image, **options)
    assert len(result.centers) == 8
    assert all(not 120 - 5 <= y < 180 + 5 for y, _ in result.centers)
    _assert_matches_reference(result, reference)


def test_cupy_input_returns_device_mask_and_host_diagnostics():
    cupy = pytest.importorskip("cupy")
    image = _scene(shape=(300, 400), count=60)
    variance = np.full(image.shape, 9.0)
    host = build_compact_source_stamp_mask(
        image, variance=variance, stamp_size=21, max_stamps=4
    )
    device = build_compact_source_stamp_mask(
        cupy.asarray(image), variance=cupy.asarray(variance), stamp_size=21
    )
    assert type(device.mask).__module__.startswith("cupy")
    assert device.centers == host.centers
    assert device.threshold == pytest.approx(host.threshold)
    assert device.scores == pytest.approx(host.scores, rel=1e-10)
    assert np.array_equal(device.mask.get(), host.mask)
