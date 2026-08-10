# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small, optional-dependency-aware runtime metadata helpers."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from cuphoton import __version__


def runtime_metadata(
    *,
    backend: str,
    dtype: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Describe a resolved backend without importing unrelated GPU stacks."""

    resolved_device = _device_type(backend, device)
    metadata: dict[str, Any] = {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "backend": backend,
        "device": resolved_device,
        "dtype": dtype,
        "numpy_version": _package_version("numpy"),
    }
    if device and device != resolved_device:
        metadata["device_name"] = device

    if backend == "torch":
        _add_torch_metadata(metadata, resolved_device)
    elif backend in {"cupy", "cutile"}:
        _add_cupy_metadata(metadata)
    elif backend == "numba-cuda":
        _add_numba_metadata(metadata)
    return metadata


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _device_type(backend: str, device: str | None) -> str:
    if device:
        normalized = str(device).lower()
        if normalized.startswith("cuda"):
            return normalized
        if normalized.startswith("cpu"):
            return "cpu"
    if backend == "cpu":
        return "cpu"
    if backend in {"cupy", "cutile", "numba-cuda"}:
        return "cuda"
    return device or "unknown"


def _add_torch_metadata(metadata: dict[str, Any], device: str) -> None:
    try:
        import torch

        metadata["torch_version"] = str(torch.__version__)
        metadata["cuda_runtime_version"] = torch.version.cuda
        if device.startswith("cuda") and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(device)
            metadata["device_name"] = properties.name
            metadata["compute_capability"] = (
                f"{properties.major}.{properties.minor}"
            )
            metadata["total_memory_bytes"] = int(properties.total_memory)
    except Exception as exc:  # pragma: no cover - runtime-specific
        metadata["metadata_error"] = f"{type(exc).__name__}: {exc}"


def _add_cupy_metadata(metadata: dict[str, Any]) -> None:
    try:
        import cupy as cp

        device_id = int(cp.cuda.runtime.getDevice())
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        metadata.update(
            {
                "cupy_version": str(cp.__version__),
                "device_index": device_id,
                "device_name": str(name),
                "cuda_driver_version": int(
                    cp.cuda.runtime.driverGetVersion()
                ),
                "cuda_runtime_version": int(
                    cp.cuda.runtime.runtimeGetVersion()
                ),
            }
        )
        major = properties.get("major")
        minor = properties.get("minor")
        if major is not None and minor is not None:
            metadata["compute_capability"] = f"{major}.{minor}"
    except Exception as exc:  # pragma: no cover - runtime-specific
        metadata["metadata_error"] = f"{type(exc).__name__}: {exc}"


def _add_numba_metadata(metadata: dict[str, Any]) -> None:
    try:
        import numba
        from numba import cuda

        device = cuda.get_current_device()
        name = device.name
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        metadata.update(
            {
                "numba_version": str(numba.__version__),
                "device_name": str(name),
                "compute_capability": ".".join(
                    str(value) for value in device.compute_capability
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime-specific
        metadata["metadata_error"] = f"{type(exc).__name__}: {exc}"


__all__ = ["runtime_metadata"]
