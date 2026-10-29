# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Array-namespace helpers shared by the NumPy and CuPy code paths.

Library functions that accept either NumPy or CuPy arrays pick the array
module from their inputs with :func:`array_module`, reach the matching
``ndimage`` and ``fft`` subpackages with :func:`ndimage_module` and
:func:`fft_module`, and move small results back with :func:`to_host`.
None of these import CuPy unless a CuPy array is present.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_cupy(array: Any) -> bool:
    return type(array).__module__.split(".", maxsplit=1)[0] == "cupy"


def array_module(*arrays: Any) -> Any:
    """Return the array module that owns ``arrays``.

    Parameters
    ----------
    *arrays
        Arrays or scalars; non-array arguments are ignored.

    Returns
    -------
    module
        ``cupy`` when any argument is a CuPy array, otherwise ``numpy``.
    """

    for array in arrays:
        if _is_cupy(array):
            import cupy

            return cupy
    return np


def to_host(array: Any) -> np.ndarray:
    """Return ``array`` as a NumPy array, copied from the device when CuPy.

    Parameters
    ----------
    array
        NumPy array, CuPy array, or anything :func:`numpy.asarray` accepts.

    Returns
    -------
    numpy.ndarray
        Host array; a NumPy input is returned without a copy.
    """

    if _is_cupy(array):
        return np.asarray(array.get())
    return np.asarray(array)


def ndimage_module(xp: Any) -> Any:
    """Return ``scipy.ndimage`` for NumPy or ``cupyx.scipy.ndimage`` for CuPy.

    Parameters
    ----------
    xp
        Array module as returned by :func:`array_module`.

    Returns
    -------
    module
        The ``ndimage`` subpackage matching ``xp``.
    """

    if xp is np:
        import scipy.ndimage

        return scipy.ndimage
    import cupyx.scipy.ndimage

    return cupyx.scipy.ndimage


def fft_module(xp: Any) -> Any:
    """Return ``scipy.fft`` for NumPy or ``cupyx.scipy.fft`` for CuPy.

    Parameters
    ----------
    xp
        Array module as returned by :func:`array_module`.

    Returns
    -------
    module
        The ``fft`` subpackage matching ``xp``.
    """

    if xp is np:
        import scipy.fft

        return scipy.fft
    import cupyx.scipy.fft

    return cupyx.scipy.fft


__all__ = ["array_module", "fft_module", "ndimage_module", "to_host"]
