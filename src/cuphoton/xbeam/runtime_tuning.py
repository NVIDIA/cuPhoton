# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime tuning implementation."""

from __future__ import annotations

from typing import Any

from cuphoton.xbeam.runtime.policy import current_policy

_VDIF_IO_BACKENDS = frozenset({"posix", "cufile"})


def xmac_packs_per_launch() -> int:
    """Maximum independent packs per launch; zero is unlimited."""
    return current_policy().xmac_packs_per_launch


def cufile_high_priority(io_backend: str) -> bool:
    return (
        current_policy(io_backend).cufile_high_priority
        and io_backend == "cufile"
    )


def cufile_strip_stream(xp: Any) -> Any:
    # CUDA clamps out-of-range priorities to the device-supported range.
    if cufile_high_priority("cufile"):
        return xp.cuda.Stream(non_blocking=True, priority=-(2**31))
    return xp.cuda.Stream(non_blocking=True)


def vdif_upload_staging_slots(io_backend: str) -> int:
    """Return the number of raw transfer slots used by one uploader."""

    if io_backend not in _VDIF_IO_BACKENDS:
        raise ValueError(f"unsupported VDIF I/O backend {io_backend!r}")
    policy = current_policy(io_backend)
    return policy.cufile_slots if io_backend == "cufile" else 2
