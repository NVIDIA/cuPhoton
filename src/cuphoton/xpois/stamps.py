# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small stamp and difference helpers shared with downstream tooling."""

from __future__ import annotations

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
