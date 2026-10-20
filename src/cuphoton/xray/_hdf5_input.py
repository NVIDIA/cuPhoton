# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply the optional self-contained policy to HDF5 input access."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from typing import Any

SELF_CONTAINED_ENV = "CUPHOTON_XRAY_HDF5_SELF_CONTAINED"


def self_contained_hdf5_input() -> bool:
    """Read the process-wide, opt-in HDF5 input policy."""
    value = os.environ.get(SELF_CONTAINED_ENV, "0")
    if value not in ("", "0", "1"):
        raise ValueError(f"{SELF_CONTAINED_ENV} must be 0 or 1")
    return value == "1"


def hdf5_input_reader(
    h5: Any, *, self_contained: bool | None = None
) -> Callable[[str], Any]:
    """Select input access once for an open HDF5 handle.

    The default returns h5py's ordinary lookup directly. The opt-in reader
    checks each requested object, including intermediate path components.
    Create a new reader for each newly opened handle; dataset validation is
    never cached across handles.
    """
    if self_contained is None:
        self_contained = self_contained_hdf5_input()
    if not self_contained:
        return h5.__getitem__
    return partial(_self_contained_hdf5_object, h5)


def _self_contained_hdf5_object(h5: Any, name: str) -> Any:
    """Resolve in-file hard links and datasets without indirection."""
    import h5py

    obj = h5
    for part in name.split("/"):
        if not part or part == ".":
            continue
        if not isinstance(obj, h5py.Group):
            raise KeyError(f"HDF5 input object is not a group: {name!r}")
        link = obj.get(part, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise ValueError(
                f"HDF5 input {name!r} does not support external links"
            )
        if isinstance(link, h5py.SoftLink):
            raise ValueError(
                f"HDF5 input {name!r} does not support soft links"
            )
        if link is None:
            raise KeyError(f"HDF5 input object not found: {name!r}")
        if not isinstance(link, h5py.HardLink):
            raise ValueError(
                f"HDF5 input {name!r} requires in-file hard links"
            )
        obj = obj[part]
        if isinstance(obj, h5py.Dataset):
            if obj.external:
                raise ValueError(
                    f"HDF5 input {name!r} does not support "
                    "external raw storage"
                )
            if obj.is_virtual:
                raise ValueError(
                    f"HDF5 input {name!r} does not support virtual datasets"
                )
    return obj
