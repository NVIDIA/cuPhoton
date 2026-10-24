# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from cuphoton._photometry import (
    FLAG_TRUNCATED,
    detect_sources,
    elliptical_aperture_flux,
    estimate_background,
    kron_radius,
    subtract_background,
)


def gaussian_image(
    *,
    shape: tuple[int, int] = (64, 64),
    sources: tuple[tuple[float, float, float], ...],
    sigma: float = 2.0,
    background: float = 10.0,
) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = np.full(shape, background, dtype=np.float64)
    for x, y, amplitude in sources:
        radius2 = (xx - x) ** 2 + (yy - y) ** 2
        image += amplitude * np.exp(-radius2 / (2.0 * sigma**2))
    return image


def test_detect_sources_preserves_counts_centroids_and_shapes() -> None:
    image = gaussian_image(sources=((18.0, 20.0, 100.0), (45.0, 42.0, 80.0)))
    data = subtract_background(image)

    detections = detect_sources(data, threshold=5.0, minarea=5)

    assert len(detections) == 2
    order = np.argsort(detections["x"])
    np.testing.assert_allclose(detections["x"][order], [18.0, 45.0], atol=0.1)
    np.testing.assert_allclose(detections["y"][order], [20.0, 42.0], atol=0.1)
    assert np.all(detections["flux"] > 0)
    assert np.all(detections["a"] >= detections["b"])


def test_empty_image_and_invalid_pixels_are_handled() -> None:
    empty = np.zeros((32, 32), dtype=np.float64)
    assert len(detect_sources(empty, threshold=1.0, minarea=2)) == 0

    image = gaussian_image(sources=((16.0, 16.0, 50.0),))
    image[0, 0] = np.nan
    estimate = estimate_background(image)
    assert np.all(np.isfinite(estimate.background))
    assert np.all(np.isfinite(estimate.background_rms))
    assert (
        len(
            detect_sources(
                subtract_background(image), threshold=5.0, minarea=3
            )
        )
        == 1
    )


def test_kron_apertures_preserve_matched_flux_ratio() -> None:
    baseline = subtract_background(
        gaussian_image(sources=((30.0, 28.0, 100.0),))
    )
    restoration = baseline * 1.75
    objects = detect_sources(baseline, threshold=5.0, minarea=5)
    radii, kron_flags = kron_radius(
        baseline,
        objects["x"],
        objects["y"],
        objects["a"],
        objects["b"],
        objects["theta"],
    )
    scale = 2.5 * radii
    baseline_flux, _, baseline_flags = elliptical_aperture_flux(
        baseline,
        objects["x"],
        objects["y"],
        objects["a"],
        objects["b"],
        objects["theta"],
        scale=scale,
    )
    restoration_flux, _, restoration_flags = elliptical_aperture_flux(
        restoration,
        objects["x"],
        objects["y"],
        objects["a"],
        objects["b"],
        objects["theta"],
        scale=scale,
    )

    assert np.all(kron_flags == 0)
    assert np.all(baseline_flags == 0)
    assert np.all(restoration_flags == 0)
    np.testing.assert_allclose(restoration_flux / baseline_flux, 1.75)


def test_edge_aperture_is_flagged_as_truncated() -> None:
    image = gaussian_image(sources=((1.5, 1.5, 100.0),), background=0.0)
    flux, _, flags = elliptical_aperture_flux(
        image,
        np.array([1.5]),
        np.array([1.5]),
        np.array([2.0]),
        np.array([1.5]),
        np.array([0.0]),
        scale=2.5,
    )

    assert np.isfinite(flux[0])
    assert flags[0] & FLAG_TRUNCATED
