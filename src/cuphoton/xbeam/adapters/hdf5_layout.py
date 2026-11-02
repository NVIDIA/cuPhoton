# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared HDF5 extent validation and window assembly."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Extent:
    offset: int
    dtype: np.dtype
    samples: int


def extent(dataset, *, file_id):
    """Return an exact raw representation, or None for the HDF5 decoder."""
    import h5py

    dtype = dataset.dtype
    if (
        dataset.file.id != file_id
        or dataset.chunks is not None
        or dataset.compression is not None
        or dataset.is_virtual
        or dataset.external
        or dtype.kind not in "biufc"
        or not dtype.isnative
        or dtype.itemsize not in (1, 2, 4, 8, 16)
        or (dtype.kind in "biuf" and dtype.itemsize > 8)
        or dataset.id.get_type() != h5py.h5t.py_create(dtype, logical=True)
    ):
        return None
    offset = dataset.id.get_offset()
    if offset is None or dataset.id.get_storage_size() != dataset.nbytes:
        return None
    return Extent(int(offset), dtype, len(dataset))


def segments(starts, size, info, *, alignment=4096):
    """Coalesce windows with transport-selected byte alignment."""
    groups = []
    for index, start in enumerate(starts):
        if not 0 <= start <= info.samples - size:
            raise ValueError("read outside source sample bounds")
        if groups and groups[-1][2][-1] <= start <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], start + size)
            groups[-1][2].append(start)
        else:
            groups.append([start, start + size, [start], index])
    planned, cursor = [], 0
    for lo, hi, windows, first_row in groups:
        byte_start = info.offset + lo * info.dtype.itemsize
        file_offset = byte_start // alignment * alignment
        end = (
            (info.offset + hi * info.dtype.itemsize + alignment - 1)
            // alignment
            * alignment
        )
        length = end - file_offset
        planned.append(
            (
                file_offset,
                length,
                cursor,
                byte_start - file_offset,
                lo,
                hi,
                windows,
                first_row,
            )
        )
        cursor += length
    return planned, cursor


def assemble(buffer, plan, starts, size, info, xp):
    """Build the same window views/gathers for host and device buffers."""
    output = (
        None
        if len(plan) == 1
        else xp.empty((len(starts), size), dtype=info.dtype)
    )
    for _, _, dest, head, lo, hi, windows, row in plan:
        values = buffer[
            dest + head : dest + head + (hi - lo) * info.dtype.itemsize
        ].view(info.dtype)
        if len(windows) == 1:
            batch = values.reshape(1, size)
        else:
            hops = np.diff(windows)
            if np.all(hops == hops[0]):
                batch = xp.lib.stride_tricks.as_strided(
                    values,
                    shape=(len(windows), size),
                    strides=(
                        int(hops[0]) * info.dtype.itemsize,
                        info.dtype.itemsize,
                    ),
                )
            else:
                batch = xp.stack(
                    [
                        values[start - lo : start - lo + size]
                        for start in windows
                    ]
                )
        if output is None:
            output = batch
        else:
            output[row : row + len(windows)] = batch
    return output


def plan_batch(request, names, layouts, *, alignment):
    signals, starts, size = request
    plans = {}
    for signal in signals:
        for name in names[signal]:
            if name is not None and name not in plans:
                plans[name] = segments(
                    starts, size, layouts[name], alignment=alignment
                )
    return plans


def assemble_batch(buffers, plans, request, names, layouts, xp):
    signals, starts, size = request
    datasets = {
        name: assemble(buffers[name], plan, starts, size, layouts[name], xp)
        for name, (plan, _) in plans.items()
    }
    return {
        signal: (datasets[names[signal][0]], datasets.get(names[signal][1]))
        for signal in signals
    }
