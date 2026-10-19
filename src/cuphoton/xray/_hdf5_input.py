# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply the optional self-contained policy to HDF5 input access."""

from __future__ import annotations

import os
from typing import Any

SELF_CONTAINED_ENV = "CUPHOTON_XRAY_HDF5_SELF_CONTAINED"


def self_contained_hdf5_input() -> bool:
    """Read the process-wide, opt-in HDF5 input policy."""
    value = os.environ.get(SELF_CONTAINED_ENV, "0")
    if value not in ("", "0", "1"):
        raise ValueError(f"{SELF_CONTAINED_ENV} must be 0 or 1")
    return value == "1"


def hdf5_input_object(h5: Any, name: str) -> Any:
    """Resolve an object using the configured HDF5 input policy.

    Normal h5py lookup is the default. When self-contained input is requested,
    check each path component before dereferencing it, including intermediate
    groups. That mode supports hard links and ordinary datasets; soft links
    and virtual datasets are unsupported even within the same file.
    Call this on each newly opened read handle.
    """
    import h5py

    if not self_contained_hdf5_input():
        return h5[name]

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
