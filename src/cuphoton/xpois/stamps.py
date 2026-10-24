# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small stamp and difference helpers shared with downstream tooling."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_centered_stamp(
    image: np.ndarray,
    *,
    center: tuple[int, int],
    size: int,
) -> np.ndarray:
    """Return an odd-sized centered cutout from a two-dimensional image.

    Parameters
    ----------
    image
        Source image.
    center
        Integer ``(y, x)`` center pixel.
    size
        Positive odd side length.

    Returns
    -------
    numpy.ndarray
        Floating-point square cutout.
    """

    if size % 2 == 0 or size <= 0:
        raise ValueError("size must be a positive odd integer")
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("extract_centered_stamp expects a 2D image")
    center_y, center_x = center
    half = size // 2
    y0 = center_y - half
    y1 = center_y + half + 1
    x0 = center_x - half
    x1 = center_x + half + 1
    if y0 < 0 or x0 < 0 or y1 > array.shape[0] or x1 > array.shape[1]:
        raise ValueError("requested stamp exceeds image bounds")
    return np.asarray(array[y0:y1, x0:x1], dtype=np.float64)


def simple_difference_stamp(
    search_stamp: np.ndarray,
    template_stamp: np.ndarray,
) -> np.ndarray:
    """Compute a same-grid search-minus-template difference stamp.

    Parameters
    ----------
    search_stamp, template_stamp
        Equal-shaped input stamps.

    Returns
    -------
    numpy.ndarray
        Floating-point difference with the shared input shape.
    """

    search_arr = np.asarray(search_stamp, dtype=np.float64)
    template_arr = np.asarray(template_stamp, dtype=np.float64)
    if search_arr.shape != template_arr.shape:
        raise ValueError("search_stamp and template_stamp must share a shape")
    return search_arr - template_arr


def window_stamps(
    plane: Any,
    positions: Any,
    size: int | tuple[int, int],
    *,
    fill: float = np.nan,
) -> Any:
    """Cut one window of ``plane`` around each position.

    Parameters
    ----------
    plane
        Two-dimensional image, NumPy or CuPy.
    positions
        ``(N, 2)`` array of ``(x, y)`` pixel coordinates. Each window is
        centered on the nearest integer pixel.
    size
        Odd side length of the square window, or odd ``(height, width)``.
    fill
        Value of the window pixels that fall outside the image.

    Returns
    -------
    ndarray
        ``(N, height, width)`` float64 windows in the namespace of ``plane``.
    """

    xp = _array_module(plane)
    image = xp.asarray(plane)
    if image.ndim != 2:
        raise ValueError("plane must be a two-dimensional array")
    height, width = (size, size) if isinstance(size, int) else size
    if min(height, width) <= 0 or height % 2 == 0 or width % 2 == 0:
        raise ValueError("size must be positive and odd")
    xy = xp.asarray(positions, dtype=xp.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if 0 in image.shape:
        return xp.full((len(xy), height, width), fill, dtype=xp.float64)
    rows = (
        xp.rint(xy[:, 1]).astype(xp.int64)[:, None, None]
        + xp.arange(-(height // 2), height // 2 + 1)[None, :, None]
    )
    columns = (
        xp.rint(xy[:, 0]).astype(xp.int64)[:, None, None]
        + xp.arange(-(width // 2), width // 2 + 1)[None, None, :]
    )
    inside = (
        (rows >= 0)
        & (rows < image.shape[0])
        & (columns >= 0)
        & (columns < image.shape[1])
    )
    values = image[
        xp.clip(rows, 0, image.shape[0] - 1),
        xp.clip(columns, 0, image.shape[1] - 1),
    ]
    return xp.where(inside, values.astype(xp.float64), fill)


def _array_module(*arrays: Any) -> Any:
    """NumPy, or CuPy when any of ``arrays`` is a CuPy array."""

    for array in arrays:
        if type(array).__module__.split(".", maxsplit=1)[0] == "cupy":
            import cupy

            return cupy
    return np
