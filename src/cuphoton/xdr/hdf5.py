# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional Legate-backed HDF5 loading for xDataReader."""

from __future__ import annotations

from os import PathLike
from typing import Any, Callable


class LegateHdf5Unavailable(RuntimeError):
    """Raised when the optional Legate HDF5 backend cannot be imported."""


def _load_legate_from_file() -> Callable[[str | PathLike[str], str], Any]:
    try:
        from legate.io.hdf5 import from_file
    except ImportError as exc:
        raise LegateHdf5Unavailable(
            "Legate with HDF5 support is required for load_hdf5(). "
            "Install the cuphoton 'hdf5' extra or build Legate with "
            "HDF5 enabled."
        ) from exc
    return from_file


def legate_hdf5_available() -> bool:
    """Return whether the optional Legate HDF5 API can be imported."""
    try:
        _load_legate_from_file()
    except LegateHdf5Unavailable:
        return False
    return True


def load_hdf5(path: str | PathLike[str], dataset_name: str) -> Any:
    """Load one HDF5 dataset as a Legate ``LogicalArray``.

    The read is submitted through :func:`legate.io.hdf5.from_file`. Runtime
    placement, partitioning, GDS selection, and completion remain controlled
    by the installed Legate build and its configuration.

    Parameters
    ----------
    path
        Path to an HDF5 file.
    dataset_name
        Full name or path of one array dataset inside the file.
    """
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string")
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty")
    return _load_legate_from_file()(path, dataset_name)
