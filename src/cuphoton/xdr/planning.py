# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native FITS planning helpers for the xDataReader GPU frontend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

_UNSUPPORTED_GPU_COMPRESSION_MARKERS = (
    "not supported by the native streaming path",
    "RICE_1",
    "HCOMPRESS_1",
    "PLIO_1",
    "NOCOMPRESS",
)


def normalize_planning_error(exc: BaseException):
    """Raise a public exception for native FITS planning failures.

    CFITSIO/native planning is the dependency-free replacement for Astropy HDU
    introspection. Unsupported compression formats are expected user-facing
    limitations of the GPU path, so expose them as ``NotImplementedError``.
    Other failures are re-raised unchanged.
    """
    message = str(exc)
    if any(
        marker in message for marker in _UNSUPPORTED_GPU_COMPRESSION_MARKERS
    ):
        raise NotImplementedError(
            f"{message}; this compression format is not supported on the "
            "GPU path"
        ) from exc
    raise exc


def get_native_plan_files(required: bool = True):
    """Return the native CFITSIO planner callable."""
    from .nvcomp_batch import (
        get_native_plan_files as _get_native_plan_files,
    )

    return _get_native_plan_files(required=required)


def plan_native_files(
    paths: Sequence[str | Path],
    file_indices: Sequence[int],
    hdu_indices: Sequence[int],
    native_plan_threads: int,
    section=None,
):
    """Plan FITS HDU byte ranges with the native CFITSIO extension."""
    planner = get_native_plan_files(required=True)
    try:
        return planner(
            [str(path) for path in paths],
            [int(index) for index in file_indices],
            [int(index) for index in hdu_indices],
            int(native_plan_threads),
            section,
        )
    except BaseException as exc:
        normalize_planning_error(exc)
