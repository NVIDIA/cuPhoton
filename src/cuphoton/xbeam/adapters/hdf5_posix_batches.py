# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""POSIX transport for the shared host/device batch pipeline."""

import numpy as np

from ..runtime.batches import OrderedBatches
from ..runtime.transports import PosixTransport
from .hdf5_batches import DeviceBatches
from .hdf5_layout import assemble_batch, plan_batch


class HostLease(dict):
    def __init__(
        self, future, buffers, plans, request, names, layouts, owner
    ):
        super().__init__()
        self.future = future
        self.context = buffers, plans, request, names, layouts
        self.owner = owner

    def wait(self):
        self.future.result()
        if self.context is not None:
            self.update(assemble_batch(*self.context, np))
            self.context = None

    def release(self):
        if self.owner is not None:
            self.owner.active.pop(id(self), None)
            self.owner = None
        self.context = None


class PosixHostBatches(OrderedBatches):
    def __init__(self, reader, names, layouts, requests, depth):
        super().__init__(requests, depth)
        self.reader, self.names, self.layouts = reader, names, layouts
        self.active = {}

    def __enter__(self):
        try:
            self._prime()
        except BaseException:
            self.close()
            raise
        return self

    def read_request(self, request):
        plans = plan_batch(request, self.names, self.layouts, alignment=1)
        buffers = {
            name: np.empty(needed, dtype=np.uint8)
            for name, (_, needed) in plans.items()
        }
        future = self.reader.submit_batch(buffers, plans)
        lease = HostLease(
            future, buffers, plans, request, self.names, self.layouts, self
        )
        self.active[id(lease)] = lease
        return lease

    def close(self):
        if self.closed:
            return
        self.closed = True
        first_error = None
        for lease in tuple(self.active.values()):
            try:
                lease.future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                lease.release()
        self._discard_queue()
        if first_error is not None:
            raise first_error


class PosixDeviceBatches(PosixTransport, DeviceBatches):
    """HDF5 byte plans using the common POSIX device transport."""
