# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Array ownership for native structured roots and mandatory DD refinement."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray


def require_native() -> ModuleType:
    """Load the installed extension without creating a CUDA context."""
    try:
        return importlib.import_module("cuphoton.xray._structured_ext")
    except ImportError as error:
        raise RuntimeError(
            "The structured xRay backend requires the native extension. "
            "Build cuPhoton with CUPHOTON_XRAY_BUILD_EXT=1."
        ) from error


def solve_roots(
    descending: NDArray[np.float64],
    *,
    workers: int = 1,
    max_sweeps: int = -1,
) -> tuple[NDArray[np.complex128], NDArray[np.int32], NDArray[np.uint8]]:
    """Solve contiguous FP64 rows; every row's status must be checked.

    Status columns are status, root count, unconverged count, sweeps,
    deflations, exceptional shifts, reserved, and effective degree.
    Success requires status zero and a complete finite root vector. Failed
    rows retain their native status and diagnostic message. The input must
    remain unchanged until this synchronous call returns.
    """
    native = require_native()
    _require_array(descending, 2)
    batch, columns = descending.shape
    if not 1 <= batch <= 2147483647 or not 2 <= columns <= 4097:
        raise ValueError(
            "Require a positive batch and polynomial degree 1..4096"
        )
    roots = np.empty((batch, columns - 1, 2), dtype=np.float64)
    stats = np.empty((batch, 8), dtype=np.int32)
    errors = np.empty((batch, 256), dtype=np.uint8)
    native.solve_roots(descending, roots, stats, errors, workers, max_sweeps)
    return (
        roots.view(np.complex128).reshape(batch, columns - 1),
        stats,
        errors,
    )


def refine_selected(
    descending: NDArray[np.float64],
    seeds: NDArray[np.float64],
    separation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    """Refine 1..36 selected roots of one degree-597 polynomial.

    Seeds and outputs use real/imaginary columns. Each status row contains
    the numerical status and completed iteration count. History contains
    real high/low, imaginary high/low, relative correction, residual,
    derivative ratio and stable-rounding count for each of eight iterations.
    A failed status never authorizes replacing a root or retaining its decay.
    Inputs must remain unchanged until this synchronous call returns.
    """
    native = require_native()
    _require_array(descending, 1)
    _require_array(seeds, 2)
    _require_array(separation, 1)
    count = seeds.shape[0]
    if (
        descending.shape != (598,)
        or not 1 <= count <= 36
        or seeds.shape != (count, 2)
        or separation.shape != (count,)
    ):
        raise ValueError("Require degree 597 and 1..36 complex seed pairs")
    output = np.empty((count, 2), dtype=np.float64)
    history = np.empty((count, 8, 8), dtype=np.float64)
    status = np.empty((count, 2), dtype=np.int32)
    native.refine_selected(
        descending, seeds, separation, output, history, status
    )
    return output, history, status


def _require_array(value: Any, dimensions: int) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != dimensions
        or value.dtype != np.float64
        or not value.flags.c_contiguous
        or not value.flags.aligned
    ):
        raise ValueError("Require aligned contiguous FP64 arrays")
