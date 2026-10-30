# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HDF5 sample views over the shared positioned byte-range reader."""

import numpy as np

from ..runtime.posix_io import RangeReader
from .hdf5_layout import Extent, extent

__all__ = ["PosixReader", "Extent", "extent"]


class PosixReader(RangeReader):
    def read(self, info, start, stop):
        if not 0 <= start <= stop <= info.samples:
            raise ValueError("read outside source sample bounds")
        output = np.empty(stop - start, dtype=info.dtype)
        offset = info.offset + start * info.dtype.itemsize
        self.submit(
            self.chunks(output, offset, fd=self._stream_fd(info.offset))
        ).result()
        return output
