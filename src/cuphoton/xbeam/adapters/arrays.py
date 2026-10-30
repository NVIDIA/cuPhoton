# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory uniformly sampled series adapter (NumPy or CuPy arrays)."""

from __future__ import annotations


class ArraySource:
    def __init__(
        self, signals, *, sample_rate_hz, origin_seconds=0.0, validity=None
    ):
        if not signals:
            raise ValueError("at least one signal is required")
        if any(not isinstance(s, str) or not s for s in signals):
            raise ValueError("signal IDs must be nonempty strings")
        self._signals = dict(signals)
        self._validity = dict(validity or {})
        if set(self._validity) - set(signals):
            raise ValueError("validity names an unknown signal")
        self.signal_ids = tuple(signals)
        self.sample_count = len(next(iter(signals.values())))
        self.sample_rate_hz = float(sample_rate_hz)
        self.origin_seconds = float(origin_seconds)
        self.dtypes = {}
        for name, values in signals.items():
            if values.ndim != 1 or len(values) != self.sample_count:
                raise ValueError("signals must be vectors of equal length")
            self.dtypes[name] = values.dtype
            mask = self._validity.get(name)
            if mask is not None and (
                mask.shape != values.shape or mask.dtype.kind != "b"
            ):
                raise ValueError("validity must be a matching Boolean vector")

    def read(self, signal_id, start, stop):
        if not 0 <= start <= stop <= self.sample_count:
            raise ValueError("read outside source sample bounds")
        mask = self._validity.get(signal_id)
        return (
            self._signals[signal_id][start:stop],
            None if mask is None else mask[start:stop],
        )
