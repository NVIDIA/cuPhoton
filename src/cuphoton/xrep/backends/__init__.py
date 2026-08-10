# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend registry for xRep."""

from __future__ import annotations

SUPPORTED_BACKENDS = {"cpu", "torch", "cupy"}

# Order in which the "auto" default backend is resolved: prefer the fused
# CuPy GPU path, then CUDA-backed Torch, then the NumPy reference.
BACKEND_PREFERENCE = ("cupy", "torch", "cpu")


def default_backend() -> str:
    """Return the first available GPU backend, otherwise ``cpu``.

    Auto selection tries CuPy, then CUDA-backed Torch, then the NumPy CPU
    reference. Explicit ``backend="torch"`` remains valid on CPU-only hosts.
    """

    for name in BACKEND_PREFERENCE:
        if _backend_available_for_auto(name):
            return name
    return "cpu"


def resolve_backend(name: str | None) -> str:
    """Resolve ``None``/``auto`` and validate an explicit backend name."""

    if name is None or name == "auto":
        return default_backend()
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {name}")
    return name


def _backend_available_for_auto(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "torch":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False
    try:
        return bool(get_backend(name).is_available())
    except Exception:
        return False


def get_backend(name: str | None):
    name = resolve_backend(name)
    if name == "cpu":
        from .cpu_backend import CpuBackend

        return CpuBackend()
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend()
    if name == "cupy":
        from .cupy_backend import CupyBackend

        return CupyBackend()
    raise AssertionError(f"Unregistered backend: {name}")
