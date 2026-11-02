# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffer conversion and loading for ahead-of-time CUDA imaging operations."""

from __future__ import annotations

import functools
import importlib
from typing import Any

import numpy as np


@functools.cache
def extension() -> Any:
    """Load the optional native imaging implementation."""
    from cuda.pathfinder import load_nvidia_dynamic_lib

    for name in ("cudart", "cublas", "cusolver", "cufft"):
        load_nvidia_dynamic_lib(name)
    try:
        return importlib.import_module("cuphoton.core._imaging_ext")
    except ImportError as exc:
        raise RuntimeError(
            "Native imaging requires an installation built with "
            "CUPHOTON_IMAGING_BUILD_EXT=1 and a CUDA 13 toolkit."
        ) from exc


def asarray(value: Any, dtype: Any = None) -> Any:
    """Return a contiguous CuPy buffer using native conversion as needed."""
    import cupy as cp

    if not isinstance(value, cp.ndarray):
        return cp.asarray(np.asarray(value, dtype=dtype, order="C"))
    if value.device.id != cp.cuda.runtime.getDevice():
        raise ValueError("buffer must reside on the active CUDA device")
    target = np.dtype(value.dtype if dtype is None else dtype)
    if value.dtype == target and value.flags.c_contiguous:
        return value
    result = cp.empty(value.shape, dtype=target)
    extension().copy_array(value, result, cp.cuda.get_current_stream().ptr)
    return result


def boolean_mask(value: Any) -> Any:
    """Validate a binary mask before converting it on the device."""
    import cupy as cp

    value = asarray(value)
    if value.dtype == np.dtype(bool):
        return value
    if value.dtype.kind not in "iuf":
        raise ValueError("fit_mask must contain only boolean or 0/1 values")
    invalid = extension().validate_mask(
        value, cp.cuda.get_current_stream().ptr
    )
    if invalid & 1:
        raise ValueError("fit_mask must contain only finite values")
    if invalid & 2:
        raise ValueError("fit_mask must contain only boolean or 0/1 values")
    return asarray(value, np.bool_)


def selection_image(
    reference: Any, target: Any, valid_mask: Any = None
) -> Any:
    """Prepare the finite mean of two images for native source selection."""
    import cupy as cp

    reference = asarray(reference, np.float64)
    target = asarray(target, np.float64)
    mask = None if valid_mask is None else boolean_mask(valid_mask)
    output = cp.empty(reference.shape, dtype=cp.float64)
    extension().selection_image(
        reference, target, mask, output, cp.cuda.get_current_stream().ptr
    )
    return output
