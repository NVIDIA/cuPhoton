# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Array-backend selection for xFit."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np

from ._types import ArrayLike, BackendRequest, ResolvedBackend


@dataclass(frozen=True)
class Backend:
    """Resolved array backend and execution device."""

    name: ResolvedBackend
    module: ModuleType
    device: str


def _cupy_backend(*, require_device: bool) -> Backend:
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is not installed; install cuphoton with the gpu extra"
        ) from exc

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError("CuPy cannot access a CUDA device") from exc
    if require_device and count < 1:
        raise RuntimeError("CuPy cannot access a CUDA device")
    device_id = int(cp.cuda.Device().id) if count else 0
    return Backend("cupy", cp, f"cuda:{device_id}")


def resolve_backend(name: BackendRequest = "auto") -> Backend:
    """Resolve an xFit backend without importing CuPy on CPU-only paths."""

    if name == "numpy":
        return Backend("numpy", np, "cpu")
    if name == "cupy":
        return _cupy_backend(require_device=True)
    if name != "auto":
        raise ValueError(
            f"unsupported xFit backend {name!r}; expected auto, "
            "numpy, or cupy"
        )
    try:
        return _cupy_backend(require_device=True)
    except RuntimeError:
        return Backend("numpy", np, "cpu")


def as_numpy(value: ArrayLike) -> np.ndarray:
    """Copy an array-backend value into a portable NumPy array."""

    if type(value).__module__.split(".", maxsplit=1)[0] == "cupy":
        return np.asarray(value.get())
    return np.asarray(value)


__all__ = ["Backend", "as_numpy", "resolve_backend"]
