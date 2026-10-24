# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Point-spread-function construction utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike
from scipy.special import ndtr


def gaussian_psf(
    size: int,
    *,
    sigma: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    dtype: DTypeLike = np.float64,
) -> np.ndarray:
    """Return a pixel-integrated circular Gaussian PSF.

    Parameters
    ----------
    size
        Positive integer height and width of the square output array.
    sigma
        Standard deviation in pixels. Must be finite and positive.
    offset
        Finite subpixel ``(row, column)`` displacement from the array center.
    dtype
        Floating-point NumPy dtype for the returned array.

    Returns
    -------
    numpy.ndarray
        A ``(size, size)`` array containing the Gaussian probability mass in
        each pixel. The finite grid is not renormalized, so its sum can be
        slightly less than one when the Gaussian extends beyond the grid.
    """

    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
    ):
        raise ValueError("size must be a positive integer")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if len(offset) != 2:
        raise ValueError("offset must contain row and column displacements")
    if not np.all(np.isfinite(offset)):
        raise ValueError("offset components must be finite")
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError("dtype must be floating-point")

    edges = np.arange(size + 1, dtype=np.float64) - size / 2
    row_mass = np.diff(ndtr((edges - float(offset[0])) / sigma))
    column_mass = np.diff(ndtr((edges - float(offset[1])) / sigma))
    return np.asarray(np.outer(row_mass, column_mass), dtype=dtype)
