# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resampling weights and sub-pixel phase at destination positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import MappingFn
from .interpolation import _lanczos_weight
from .mapping import PreparedReprojection

__all__ = ["ResamplingWeights", "resampling_weights_at"]


@dataclass(frozen=True, slots=True)
class ResamplingWeights:
    """Interpolation weights the reprojection applies at destination pixels.

    Attributes
    ----------
    weights
        ``(N, k, k)`` normalized weights, rows along source y and columns
        along source x. Entry ``[n, j, i]`` multiplies the source pixel in
        row ``floor(source[n, 1]) + offsets[j]`` and column
        ``floor(source[n, 0]) + offsets[i]``.
        All-zero rows mark positions the reprojection filled instead of
        resampling (no valid source tap).
    phase
        ``(N, 2)`` sub-pixel fractions ``(x, y)`` of the source coordinate,
        ``source - floor(source)``, NaN where the mapping is not finite.
    is_translation
        ``(N,)`` boolean, true where the local mapping is a pure translation
        within ``translation_tolerance``: the phase is the same at every pixel
        of a window around the position and the weights describe the whole
        window. False where the mapping rotates or scales the grid.
    interpolation
        Interpolation kernel name, ``"lanczos3"`` or ``"bilinear"``.
    offsets
        ``(k,)`` integer tap offsets relative to the floor of the source
        coordinate along each axis.
    source
        ``(N, 2)`` source ``(x, y)`` coordinates of the destination pixels,
        local to the source array.
    destination
        ``(N, 2)`` integer destination ``(x, y)`` pixels the weights belong
        to, the pixels nearest the requested positions.
    """

    weights: Any
    phase: Any
    is_translation: Any
    interpolation: str
    offsets: Any
    source: Any
    destination: Any


def resampling_weights_at(
    mapping: MappingFn | PreparedReprojection,
    x: Any,
    y: Any,
    *,
    interpolation: str,
    lanczos_a: int = 3,
    two_a_footprint: bool = True,
    source_origin: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    translation_tolerance: float = 1e-3,
    seam_tolerance: float = 0.5,
) -> ResamplingWeights:
    """Return the resampling weights at the destination pixels nearest (x, y).

    The weights are the normalized interpolation weights that
    :func:`reproject_array_device` and :func:`reproject_masked_array` apply
    at each destination pixel, built from the same mapping and kernel
    conventions. The sub-pixel phase and the local Jacobian of the mapping
    come with them, so callers can tell where one set of weights describes a
    whole window (a translation) and where it changes from pixel to pixel.

    Parameters
    ----------
    mapping
        Either the destination-to-source callable of a ``ReprojectionSpec``,
        or the ``PreparedReprojection`` used for the reprojection. With a
        prepared mapping, ``x`` and ``y`` are pixels of its output grid.
    x, y
        Destination positions, NumPy or CuPy arrays of the same shape. They
        are rounded to the nearest destination pixel.
    interpolation
        ``"lanczos3"`` or ``"bilinear"``.
    lanczos_a, two_a_footprint
        Lanczos kernel radius and footprint convention, as in
        ``ReprojectionSpec``.
    source_origin
        Zero-based ``(x, y)`` origin subtracted from mapped source pixels, as
        in ``ReprojectionSpec``. Ignored for a prepared mapping, whose
        coordinates are already local.
    source_shape
        Optional source ``(height, width)``. When given, taps outside the
        source get zero weight and the rest are renormalized, as the
        reprojection does at the source edges. Without it the weights are
        the interior weights.
    translation_tolerance
        Largest ``max|J - I|`` of the local Jacobian ``J`` still reported as
        a translation.
    seam_tolerance
        Where the forward and backward one-pixel differences of the mapping
        disagree by more than this many source pixels, the Jacobian uses the
        shorter one. This keeps a seam between tiled reference frames from
        being read as a rotation.

    Returns
    -------
    ResamplingWeights
        Weights, phase, translation flag, and tap layout per position, in
        the array namespace of ``x``.
    """

    if interpolation not in {"bilinear", "lanczos3"}:
        raise ValueError("interpolation must be 'bilinear' or 'lanczos3'")
    xp = _array_module(x)
    x_arr = xp.asarray(x, dtype=xp.float64).ravel()
    y_arr = xp.asarray(y, dtype=xp.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    destination = xp.stack([xp.rint(x_arr), xp.rint(y_arr)], axis=1)

    if isinstance(mapping, PreparedReprojection):
        source, jacobian = _local_mapping_prepared(
            mapping, destination, xp, seam_tolerance
        )
    elif callable(mapping):
        source, jacobian = _local_mapping_callable(
            mapping, destination, xp, source_origin, seam_tolerance
        )
    else:
        raise TypeError("mapping must be callable or a PreparedReprojection")

    if interpolation == "bilinear":
        offsets = np.array([0, 1])
    else:
        start = -(lanczos_a - 1) if two_a_footprint else -lanczos_a
        window = 2 * lanczos_a if two_a_footprint else 2 * lanczos_a + 1
        offsets = np.arange(start, start + window)

    finite = xp.isfinite(source).all(axis=1)
    safe = xp.where(finite[:, None], source, 0.0)
    base = xp.floor(safe)
    phase = xp.where(finite[:, None], safe - base, xp.nan)
    limits = (None, None) if source_shape is None else source_shape[::-1]
    weights_x = _weights_1d(
        phase[:, 0], base[:, 0], offsets, limits[0], interpolation, lanczos_a
    )
    weights_y = _weights_1d(
        phase[:, 1], base[:, 1], offsets, limits[1], interpolation, lanczos_a
    )
    weights = weights_y[:, :, None] * weights_x[:, None, :]
    norm = weights.sum(axis=(1, 2))
    valid = finite & (norm > 1e-12)
    weights = xp.where(
        valid[:, None, None],
        weights / xp.where(valid, norm, 1.0)[:, None, None],
        0.0,
    )

    identity = xp.asarray(np.eye(2))
    deviation = xp.abs(jacobian - identity).max(axis=(1, 2))
    is_translation = finite & (deviation < translation_tolerance)

    return ResamplingWeights(
        weights=weights,
        phase=phase,
        is_translation=is_translation,
        interpolation=interpolation,
        offsets=xp.asarray(offsets),
        source=source,
        destination=destination.astype(xp.int64),
    )


def _weights_1d(phase, base, offsets, limit, interpolation, lanczos_a):
    """Unnormalized ``(N, k)`` tap weights, out-of-source taps zeroed."""

    xp = _array_module(phase)
    taps = xp.asarray(offsets, dtype=xp.float64)[None, :]
    if interpolation == "bilinear":
        weights = xp.stack([1.0 - phase, phase], axis=1)
    else:
        weights = _lanczos_weight(taps - phase[:, None], lanczos_a)
    if limit is not None:
        pixel = base[:, None] + taps
        weights = xp.where((pixel >= 0) & (pixel < limit), weights, 0.0)
    return weights


def _local_mapping_callable(mapping, destination, xp, source_origin, seam):
    """Source coordinates and Jacobian of a callable mapping at pixels."""

    center = _to_numpy(destination)
    unit_x = np.array([[1.0, 0.0]])
    unit_y = np.array([[0.0, 1.0]])
    points = np.concatenate(
        [
            center,
            center + unit_x,
            center - unit_x,
            center + unit_y,
            center - unit_y,
        ]
    )
    mapped = np.asarray(mapping(points), dtype=np.float64)
    if mapped.shape != points.shape:
        raise ValueError("mapping must return an (N, 2) array")
    origin = np.asarray(source_origin, dtype=np.float64)
    mapped = mapped.reshape(5, center.shape[0], 2) - origin
    jacobian = np.stack(
        [
            _derivative(mapped[0], mapped[1], mapped[2], seam),
            _derivative(mapped[0], mapped[3], mapped[4], seam),
        ],
        axis=-1,
    )
    return xp.asarray(mapped[0]), xp.asarray(jacobian)


def _local_mapping_prepared(prepared, destination, xp, seam):
    """Source coordinates and Jacobian of a prepared mapping at pixels."""

    height, width = prepared.output_shape
    column = destination[:, 0].astype(xp.int64)
    row = destination[:, 1].astype(xp.int64)
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    if not bool(inside.all()):
        raise ValueError("positions must lie inside the prepared output grid")
    x_local = xp.asarray(prepared.x_local, dtype=xp.float64)
    y_local = xp.asarray(prepared.y_local, dtype=xp.float64)

    def at(r, c):
        r = xp.clip(r, 0, height - 1)
        c = xp.clip(c, 0, width - 1)
        return xp.stack([x_local[r, c], y_local[r, c]], axis=1)

    center = at(row, column)
    jacobian = xp.stack(
        [
            _derivative(
                center, at(row, column + 1), at(row, column - 1), seam
            ),
            _derivative(
                center, at(row + 1, column), at(row - 1, column), seam
            ),
        ],
        axis=-1,
    )
    return center, jacobian


def _derivative(center, plus, minus, seam):
    """d(source)/d(destination) along one axis from the neighbors of center.

    The central difference, or, where the two one-sided differences disagree
    by more than ``seam`` source pixels, the shorter one. At the border of a
    dense grid the clamped neighbor repeats the center and the one-sided
    difference that remains is used.
    """

    xp = _array_module(center)
    forward = plus - center
    backward = center - minus
    forward_norm = xp.sqrt((forward**2).sum(axis=1))
    backward_norm = xp.sqrt((backward**2).sum(axis=1))
    zero = xp.asarray(0.0)
    forward = xp.where((forward_norm == zero)[:, None], backward, forward)
    backward = xp.where((backward_norm == zero)[:, None], forward, backward)
    nearer = xp.where(
        (forward_norm <= backward_norm)[:, None], forward, backward
    )
    smooth = xp.abs(forward - backward).max(axis=1) < seam
    return xp.where(smooth[:, None], (forward + backward) / 2.0, nearer)


def _array_module(array):
    if type(array).__module__.split(".", maxsplit=1)[0] == "cupy":
        import cupy

        return cupy
    return np


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "get"):
        return np.asarray(array.get(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)
