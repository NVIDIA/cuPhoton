# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared source-detection and aperture-photometry helpers.

The public helpers use NumPy arrays and plain scalar thresholds so callers do
not need to depend on Photutils-specific table or quantity types.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

DETECTION_DTYPE = np.dtype(
    [
        ("x", np.float64),
        ("y", np.float64),
        ("flux", np.float64),
        ("a", np.float64),
        ("b", np.float64),
        ("theta", np.float64),
        ("peak", np.float64),
        ("npix", np.int64),
        ("tnpix", np.int64),
        ("flag", np.int16),
    ]
)

FLAG_INVALID = np.int16(1)
FLAG_TRUNCATED = np.int16(2)
FLAG_PHOTOMETRY_ERROR = np.int16(4)


@dataclass(frozen=True)
class BackgroundEstimate:
    """Per-pixel background and background-RMS estimates."""

    background: np.ndarray
    background_rms: np.ndarray


def estimate_background(image: np.ndarray) -> BackgroundEstimate:
    """Estimate a smooth background and RMS with Photutils.

    Small images cannot support a useful tiled model, so they use a robust
    constant estimate. This is also the fallback when Photutils rejects a
    mostly masked image.
    """

    data, mask = _finite_image(image)
    if min(data.shape) < 32:
        return _constant_background(data, mask)

    Background2D, MedianBackground = _load_background_api()
    box_size = tuple(max(16, min(64, size // 4)) for size in data.shape)
    try:
        model = Background2D(
            data,
            box_size,
            mask=mask,
            filter_size=(3, 3),
            bkg_estimator=MedianBackground(),
        )
    except (TypeError, ValueError):
        return _constant_background(data, mask)
    return BackgroundEstimate(
        background=np.asarray(model.background, dtype=np.float64),
        background_rms=np.asarray(model.background_rms, dtype=np.float64),
    )


def subtract_background(image: np.ndarray) -> np.ndarray:
    """Return a finite, contiguous image with its background removed."""

    data, _ = _finite_image(image)
    estimate = estimate_background(data)
    return np.ascontiguousarray(data - estimate.background, dtype=np.float64)


def detect_sources(
    data: np.ndarray,
    *,
    threshold: float,
    minarea: int,
) -> np.ndarray:
    """Detect connected sources above an absolute pixel threshold.

    The returned structured array has stable ``x``, ``y``, ``flux``, ``a``,
    ``b``, and ``theta`` fields. Coordinates are zero-based pixel centers,
    axes are one-sigma ellipse sizes in pixels, and theta is in radians.
    """

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if minarea < 1:
        raise ValueError("minarea must be positive")

    image, mask = _finite_image(data)
    photutils_detect, SourceCatalog, NoDetectionsWarning = (
        _load_segmentation_api()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDetectionsWarning)
        segmentation = photutils_detect(
            image,
            float(threshold),
            n_pixels=int(minarea),
            mask=mask,
        )
    if segmentation is None:
        return np.empty(0, dtype=DETECTION_DTYPE)

    catalog = SourceCatalog(image, segmentation, mask=mask)
    count = len(catalog)
    detections = np.empty(count, dtype=DETECTION_DTYPE)
    detections["x"] = _plain_array(catalog.x_centroid)
    detections["y"] = _plain_array(catalog.y_centroid)
    detections["flux"] = _plain_array(catalog.segment_flux)
    detections["a"] = np.maximum(_plain_array(catalog.semimajor_axis), 1.0e-3)
    detections["b"] = np.maximum(_plain_array(catalog.semiminor_axis), 1.0e-3)
    orientation = catalog.orientation
    if hasattr(orientation, "to_value"):
        orientation = orientation.to_value("rad")
    detections["theta"] = np.asarray(orientation, dtype=np.float64)
    detections["peak"] = _plain_array(catalog.max_value)
    pixel_count = np.rint(_plain_array(catalog.segment_area)).astype(np.int64)
    detections["npix"] = pixel_count
    detections["tnpix"] = pixel_count
    detections["flag"] = 0
    return detections


def kron_radius(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    *,
    max_radius: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure first-moment elliptical radii for matched source arrays."""

    if not np.isfinite(max_radius) or max_radius <= 0:
        raise ValueError("max_radius must be positive and finite")
    image, _ = _finite_image(data)
    arrays = _broadcast_source_arrays(x, y, a, b, theta)
    radii = np.full(arrays[0].shape, np.nan, dtype=np.float64)
    flags = np.zeros(arrays[0].shape, dtype=np.int16)

    for index, values in enumerate(zip(*arrays, strict=True)):
        cx, cy, semimajor, semiminor, angle = values
        if not _valid_ellipse(cx, cy, semimajor, semiminor, angle):
            flags[index] |= FLAG_INVALID
            continue
        bounds, truncated = _ellipse_bounds(
            image.shape,
            cx,
            cy,
            semimajor * max_radius,
            semiminor * max_radius,
        )
        if truncated:
            flags[index] |= FLAG_TRUNCATED
        y_slice, x_slice = bounds
        yy, xx = np.mgrid[y_slice, x_slice]
        radius = _elliptical_radius(
            xx, yy, cx, cy, semimajor, semiminor, angle
        )
        inside = radius <= max_radius
        weights = image[y_slice, x_slice][inside]
        denominator = float(np.sum(weights))
        if not np.isfinite(denominator) or denominator <= 0:
            flags[index] |= FLAG_INVALID
            continue
        value = float(np.sum(radius[inside] * weights) / denominator)
        if not np.isfinite(value) or value <= 0:
            flags[index] |= FLAG_INVALID
            continue
        radii[index] = value
    return radii, flags


def elliptical_aperture_flux(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    *,
    scale: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure variable elliptical apertures with Photutils.

    Flags are zero for complete, valid measurements. Nonzero bits indicate
    invalid geometry, image-edge truncation, or a Photutils measurement error.
    """

    image, mask = _finite_image(data)
    arrays = _broadcast_source_arrays(x, y, a, b, theta, scale)
    flux = np.full(arrays[0].shape, np.nan, dtype=np.float64)
    flux_error = np.full(arrays[0].shape, np.nan, dtype=np.float64)
    flags = np.zeros(arrays[0].shape, dtype=np.int16)
    EllipticalAperture, aperture_photometry = _load_aperture_api()

    for index, values in enumerate(zip(*arrays, strict=True)):
        cx, cy, semimajor, semiminor, angle, radius_scale = values
        if (
            not _valid_ellipse(cx, cy, semimajor, semiminor, angle)
            or not np.isfinite(radius_scale)
            or radius_scale <= 0
        ):
            flags[index] |= FLAG_INVALID
            continue
        aperture_a = semimajor * radius_scale
        aperture_b = semiminor * radius_scale
        _, truncated = _ellipse_bounds(
            image.shape, cx, cy, aperture_a, aperture_b
        )
        if truncated:
            flags[index] |= FLAG_TRUNCATED
        try:
            aperture = EllipticalAperture(
                (cx, cy), aperture_a, aperture_b, theta=angle
            )
            table = aperture_photometry(
                image,
                aperture,
                mask=mask,
                method="center",
            )
            flux[index] = float(table["aperture_sum"][0])
        except (ArithmeticError, TypeError, ValueError):
            flags[index] |= FLAG_PHOTOMETRY_ERROR
    return flux, flux_error, flags


def _finite_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {data.shape}")
    mask = ~np.isfinite(data)
    if np.all(mask):
        raise ValueError("image contains no finite pixels")
    fill = float(np.median(data[~mask]))
    return np.ascontiguousarray(np.where(mask, fill, data)), mask


def _constant_background(
    data: np.ndarray, mask: np.ndarray
) -> BackgroundEstimate:
    values = data[~mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    rms = max(1.4826 * mad, float(np.std(values)), np.finfo(float).eps)
    return BackgroundEstimate(
        background=np.full(data.shape, median, dtype=np.float64),
        background_rms=np.full(data.shape, rms, dtype=np.float64),
    )


def _plain_array(value: Any) -> np.ndarray:
    if hasattr(value, "value"):
        value = value.value
    return np.asarray(value, dtype=np.float64)


def _broadcast_source_arrays(*values: Any) -> tuple[np.ndarray, ...]:
    arrays = tuple(
        np.atleast_1d(np.asarray(value, dtype=np.float64)) for value in values
    )
    broadcast = tuple(np.broadcast_arrays(*arrays))
    return tuple(np.ravel(value) for value in broadcast)


def _valid_ellipse(
    x: float, y: float, a: float, b: float, theta: float
) -> bool:
    return bool(np.all(np.isfinite((x, y, a, b, theta))) and a > 0 and b > 0)


def _ellipse_bounds(
    shape: tuple[int, int],
    x: float,
    y: float,
    a: float,
    b: float,
) -> tuple[tuple[slice, slice], bool]:
    radius = max(a, b)
    raw_x0 = int(np.floor(x - radius))
    raw_x1 = int(np.ceil(x + radius)) + 1
    raw_y0 = int(np.floor(y - radius))
    raw_y1 = int(np.ceil(y + radius)) + 1
    x0 = max(0, raw_x0)
    x1 = min(shape[1], raw_x1)
    y0 = max(0, raw_y0)
    y1 = min(shape[0], raw_y1)
    truncated = (x0, x1, y0, y1) != (raw_x0, raw_x1, raw_y0, raw_y1)
    return (slice(y0, y1), slice(x0, x1)), truncated


def _elliptical_radius(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: float,
    y: float,
    a: float,
    b: float,
    theta: float,
) -> np.ndarray:
    dx = x_grid - x
    dy = y_grid - y
    cosine = np.cos(theta)
    sine = np.sin(theta)
    major = cosine * dx + sine * dy
    minor = -sine * dx + cosine * dy
    return np.sqrt((major / a) ** 2 + (minor / b) ** 2)


def _load_background_api():
    try:
        from photutils.background import Background2D, MedianBackground
    except ImportError as exc:
        raise RuntimeError(
            "Photometry requires the `photutils` package."
        ) from exc
    return Background2D, MedianBackground


def _load_segmentation_api():
    try:
        from photutils.segmentation import SourceCatalog
        from photutils.segmentation import detect_sources as photutils_detect
        from photutils.utils.exceptions import NoDetectionsWarning
    except ImportError as exc:
        raise RuntimeError(
            "Source detection requires the `photutils` package."
        ) from exc
    return photutils_detect, SourceCatalog, NoDetectionsWarning


def _load_aperture_api():
    try:
        from photutils.aperture import (
            EllipticalAperture,
            aperture_photometry,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Aperture photometry requires the `photutils` package."
        ) from exc
    return EllipticalAperture, aperture_photometry
