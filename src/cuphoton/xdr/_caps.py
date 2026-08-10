# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capability probes for the GPU-native FITS reader.

All expensive/optional imports (`cupy`, `kvikio`, `nvidia.nvcomp`) are done
lazily so that importing `cuphoton.xdr` on a machine without them does
not raise.
"""

from __future__ import annotations

import functools
import warnings


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return True iff GPU dependencies and a CUDA device are present."""
    try:
        import cupy
        import kvikio  # noqa: F401
        import nvidia.nvcomp  # noqa: F401
    except ImportError:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def is_gds_active() -> bool:
    """Return True only if real GDS is working.

    kvikio's ``is_compat_mode_preferred()`` only reports the Python-side
    configuration, not whether cufile at runtime is actually using GDS.
    We additionally require the nvidia-fs kernel module to be loaded
    (``/proc/driver/nvidia-fs/stats`` must exist) — without it every read
    falls back to POSIX + pinned bounce buffer even though the probe says
    GDS is preferred. See `gdscheck -p` to diagnose.
    """
    import os

    if not gpu_available():
        return False
    if not os.path.exists("/proc/driver/nvidia-fs/stats"):
        return False
    try:
        import kvikio.defaults as d

        return not d.is_compat_mode_preferred()
    except Exception:
        return False


def warn_if_gds_fallback() -> None:
    """Emit a one-time warning if GDS is not active. Idempotent."""
    if gpu_available() and not is_gds_active():
        warnings.warn(
            "kvikio is running in POSIX-compat fallback "
            "(no GPUDirect Storage). "
            "The xdr path will still work but will be slower; "
            "expected mode is CompatMode.OFF with nvidia-fs loaded.",
            RuntimeWarning,
            stacklevel=2,
        )
