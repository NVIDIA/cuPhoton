# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional coarse CUDA C++ Gaussian fitting, with thread-owned workspaces."""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Any

import numpy as np

from ._types import BackendArray, FitMode
from .solver import (
    BatchedLeastSquaresProblem,
    LMConfig,
    LMResult,
    _finalize_lm,
    _settings,
)

_worker = threading.local()


def load_extension() -> ModuleType:
    """Load the optional extension only for an explicitly selected backend."""

    try:
        return importlib.import_module("cuphoton.xfit._native_ext")
    except ImportError as exc:
        raise RuntimeError(
            "the native xFit extension is unavailable; build cuphoton with "
            "CUPHOTON_XFIT_BUILD_EXT=1 and a CUDA 13 toolkit"
        ) from exc


def release_workspace() -> None:
    """Release this worker's native workspace after its synchronous calls."""

    if hasattr(_worker, "workspace"):
        del _worker.workspace


def fit_gaussian(
    problem: BatchedLeastSquaresProblem,
    initial: BackendArray,
    images: BackendArray,
    weights: BackendArray,
    *,
    image_shape: tuple[int, int],
    mode: FitMode,
    config: LMConfig | None,
) -> LMResult:
    """Run native LM iterations and retain the common diagnostic contract.

    Input preparation and final diagnostics use CuPy on its current stream.
    The extension records that stream's dependency, uses an independent
    reusable stream, and drains it before returning or propagating an error.
    Strong references to all buffers and the producer stream remain live
    throughout the call. The detached native loop accesses no Python objects.
    """

    import cupy as cp

    extension = load_extension()
    dtype = np.dtype(initial.dtype)
    settings = _settings(LMConfig() if config is None else config, dtype, 8)
    if settings.use_finite_difference:
        raise ValueError("native xFit requires an analytic Gaussian Jacobian")
    device = int(cp.cuda.Device().id)
    cached = getattr(_worker, "workspace", None)
    if cached is None or cached[0] != device:
        cached = (device, extension.create_workspace(device))
        _worker.workspace = cached
    producer = cp.cuda.get_current_stream()
    x = cp.array(initial, order="C", copy=True)
    batch = int(x.shape[0])
    prepared_images = cp.ascontiguousarray(images.reshape(batch, -1))
    prepared_weights = cp.ascontiguousarray(weights.reshape(batch, -1))
    residuals = cp.empty_like(prepared_images)
    status = cp.empty(batch, dtype=cp.int8)
    evaluations = cp.empty(batch, dtype=cp.int32)
    diagnostics_current = cp.empty(batch, dtype=cp.bool_)
    owners = (
        x,
        prepared_images,
        prepared_weights,
        residuals,
        status,
        evaluations,
        diagnostics_current,
    )
    _worker.timings = extension.run(
        cached[1],
        owners,
        (
            settings.f_tol,
            settings.x_tol,
            settings.g_tol,
            settings.initial_damping,
            settings.damping_increase,
            settings.damping_decrease,
            settings.max_evaluations,
        ),
        (*image_shape, 1 if mode == "difference" else 3),
        producer.ptr,
    )
    # The native workspace keeps its intermediate Jacobians private.
    # Refresh only eligible rows through the original analytic callback.
    jacobians = cp.full((batch, 8, residuals.shape[1]), cp.nan, dtype=dtype)
    jacobian_current = cp.zeros(batch, dtype=cp.bool_)
    return _finalize_lm(
        problem,
        x,
        residuals,
        status,
        evaluations,
        jacobians,
        jacobian_current,
        diagnostics_current,
        settings,
        cp,
    )


def last_timings() -> dict[str, Any]:
    """Return native iteration timings from the last call in this worker."""

    return dict(getattr(_worker, "timings", {}))
