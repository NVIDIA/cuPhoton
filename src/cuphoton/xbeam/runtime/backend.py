# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Array backends without CUDA imports at package import time."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from ..errors import BackendUnavailableError


def get_backend(name: str) -> ModuleType:
    """Return NumPy or CuPy for *name* with actionable backend errors."""

    normalized = name.strip().lower()
    if normalized == "numpy":
        import numpy

        return numpy
    if normalized != "cupy":
        raise BackendUnavailableError(f"unknown array backend {name!r}")
    try:
        import cupy
    except (
        ImportError,
        OSError,
    ) as exc:  # driver/runtime load can raise either
        raise BackendUnavailableError(
            "CuPy is unavailable; install cupy-cuda13x and verify the "
            "NVIDIA driver"
        ) from exc
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except (
        Exception
    ) as exc:  # CuPy exposes runtime-specific exception classes
        raise BackendUnavailableError(
            "the CUDA runtime could not enumerate a device; verify "
            "driver/toolkit compatibility"
        ) from exc
    if count < 1:
        raise BackendUnavailableError("no CUDA-capable device is available")
    return cupy


def to_numpy(value: Any) -> Any:
    """Copy a CuPy value to NumPy, leaving NumPy/Python values unchanged."""

    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy.asnumpy(value)
    return value


def free_device_bytes() -> int:
    """Return free CUDA memory without retaining a memory-pool block."""

    cupy = get_backend("cupy")
    free, _total = cupy.cuda.runtime.memGetInfo()
    return int(free)
