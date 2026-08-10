# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPU-native FITS and HDF5 loading.

Pipeline: NVMe -> GPU memory via kvikio (GPUDirect Storage) -> nvCOMP batched
decompression on device -> CuPy kernels for GZIP_2 unshuffle, dequantize, and
tile scatter -> cupy.ndarray. Headers remain parsed on the CPU.

Scope: GZIP_1 / GZIP_2 compressed images and uncompressed ImageHDU /
PrimaryHDU pixel data. RICE_1 and HCOMPRESS_1 are explicitly out of scope
(no GPU decompressor available).
"""

from __future__ import annotations

from cuphoton import __version__

from ._caps import gpu_available, is_gds_active, warn_if_gds_fallback
from .convenience import batch_to_device, open_gpu
from .hdf5 import (
    LegateHdf5Unavailable,
    legate_hdf5_available,
    load_hdf5,
)
from .mock_storage import mock_storage, storage_cache
from .prefetch import batch_to_device_stream
from .reader import GpuCompImageReader, GpuImageReader

__all__ = [
    "__version__",
    "gpu_available",
    "is_gds_active",
    "warn_if_gds_fallback",
    "GpuImageReader",
    "GpuCompImageReader",
    "open_gpu",
    "batch_to_device",
    "batch_to_device_stream",
    "LegateHdf5Unavailable",
    "legate_hdf5_available",
    "load_hdf5",
    "mock_storage",
    "storage_cache",
]
