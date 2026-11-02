# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous cuFile reads for explicitly validated HDF5 raw extents."""

import os

from ..runtime.transports import CuFileTransport
from .hdf5_batches import (
    DeviceBatches,
)
from .hdf5_batches import (
    DeviceLease as _Lease,
)
from .hdf5_batches import (
    DeviceSlot as _Slot,
)
from .hdf5_layout import Extent, segments
from .hdf5_layout import extent as raw_extent

__all__ = ["CuFileBatches", "Extent", "segments", "_Lease", "_Slot"]


def extent(dataset, *, file_id):
    """Apply cuFile alignment requirements to the shared raw layout check."""
    if dataset.file.id != file_id:
        raise ValueError(
            "cuFile requires datasets owned by the opened HDF5 file; "
            "external links to another file are unsupported"
        )
    info = raw_extent(dataset, file_id=file_id)
    if info is None:
        raise ValueError(
            "cuFile requires allocated, native, contiguous, "
            "uncompressed HDF5 datasets without external storage"
        )
    if info.offset % 4096 or dataset.nbytes % 4096:
        raise ValueError(
            "cuFile HDF5 dataset offsets and storage sizes must "
            "be 4096-byte aligned"
        )
    return info


class CuFileBatches(CuFileTransport, DeviceBatches):
    """HDF5 byte plans using the common cuFile device transport."""

    def __init__(self, *args, source_identity=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_identity = source_identity

    def _open(self):
        super()._open()
        # The filename can change after HDF5 supplies its raw extents.
        # Validate the registered descriptor before submitting any reads.
        if self.source_identity is not None:
            stat = os.fstat(self.file.fd)
            if (stat.st_dev, stat.st_ino) != self.source_identity:
                raise OSError("HDF5 source changed before cuFile ingest")
