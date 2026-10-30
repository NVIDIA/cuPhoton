# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional HDF5 time-series input and generic result serialization."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, contextmanager
from itertools import chain

import numpy as np

from ..runtime.prefetch import BatchPrefetch


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "HDF5 support requires h5py; reinstall cuphoton "
            "with its dependencies"
        ) from exc
    return h5py


class Hdf5Source(AbstractContextManager):
    """One vector dataset per signal, with a common explicit sample grid.

    Each dataset must have sample_rate_hz and origin_seconds attributes.
    A Boolean dataset may be supplied per signal through validity_paths.
    Paths may name chunked/compressed datasets; reads are bounded slices.
    """

    def __init__(self, path, dataset_paths, *, validity_paths=None):
        self._readers = set()
        self._file = _h5py().File(path, "r")
        try:
            if not dataset_paths:
                raise ValueError("at least one dataset path is required")
            self._datasets = {
                name: self._file[p] for name, p in dataset_paths.items()
            }
            self._validity = {
                name: self._file[p]
                for name, p in (validity_paths or {}).items()
            }
            if any(not isinstance(s, str) or not s for s in dataset_paths):
                raise ValueError("signal IDs must be nonempty strings")
            if set(self._validity) - set(self._datasets):
                raise ValueError("validity names an unknown signal")
            self.signal_ids = tuple(self._datasets)
            first = next(iter(self._datasets.values()))
            self.sample_count = len(first)
            self.sample_rate_hz = float(first.attrs["sample_rate_hz"])
            self.origin_seconds = float(first.attrs["origin_seconds"])
            self.dtypes = {}
            for name, dataset in self._datasets.items():
                if dataset.ndim != 1 or len(dataset) != self.sample_count:
                    raise ValueError(
                        "signals must be vectors of equal length"
                    )
                if (
                    float(dataset.attrs["sample_rate_hz"])
                    != self.sample_rate_hz
                    or float(dataset.attrs["origin_seconds"])
                    != self.origin_seconds
                ):
                    raise ValueError(
                        "signals do not share a common sample grid"
                    )
                self.dtypes[name] = dataset.dtype
                mask = self._validity.get(name)
                if mask is not None and (
                    mask.shape != dataset.shape or mask.dtype.kind != "b"
                ):
                    raise ValueError(
                        "validity must be a matching Boolean vector"
                    )
        except BaseException:
            self.close()
            raise

    def read(self, signal_id, start, stop):
        if not 0 <= start <= stop <= self.sample_count:
            raise ValueError("read outside source sample bounds")
        mask = self._validity.get(signal_id)
        return self._datasets[signal_id][
            start:stop
        ], None if mask is None else mask[start:stop]

    def read_windows(self, signal_id, starts, size):
        """Read a bounded batch, coalescing connected sample intervals.

        Adjacent windows use one HDF5 slice and a reshape. Overlapping
        windows share one bounded slice; disjoint windows never read gaps.
        Works with compressed datasets through the normal HDF5 decoder.
        """
        starts = list(starts)
        if (
            not starts
            or size < 1
            or any(
                not 0 <= start <= self.sample_count - size for start in starts
            )
        ):
            raise ValueError("read outside source sample bounds")
        connected = all(
            0 <= b - a <= size
            for a, b in zip(starts, starts[1:], strict=False)
        )
        if connected:
            values, mask = self.read(signal_id, starts[0], starts[-1] + size)
            contiguous = all(
                b - a == size
                for a, b in zip(starts, starts[1:], strict=False)
            )

            def windows(v):
                if v is None:
                    return None
                if contiguous:
                    return v.reshape(len(starts), size)
                return np.stack(
                    [
                        v[start - starts[0] : start - starts[0] + size]
                        for start in starts
                    ]
                )

            return windows(values), windows(mask)
        batches = [
            self.read(signal_id, start, start + size) for start in starts
        ]
        return np.stack([v for v, _ in batches]), (
            np.stack([m for _, m in batches])
            if signal_id in self._validity
            else None
        )

    def _read_batch(self, request):
        signals, starts, size = request
        return {
            signal: self.read_windows(signal, starts, size)
            for signal in signals
        }

    @contextmanager
    def prefetch_batches(self, requests):
        """Own the worker until reads finish, including early source close."""
        reader = BatchPrefetch(self._read_batch, requests)
        self._readers.add(reader)
        try:
            with reader:
                yield reader
        finally:
            self._readers.discard(reader)

    def close(self):
        for reader in tuple(self._readers):
            reader.close()
        self._file.close()

    def __exit__(self, *args):
        self.close()


def write_series_hdf5(path, results):
    """Stream result groups into a new file; refuse to overwrite a result.

    Validate and obtain the first result before creating the output.
    Root complete=false remains if a later write fails. No astronomy
    metadata is required. Each group has its global result ordinal.
    """
    results = iter(results)
    first = next(results, None)
    with _h5py().File(path, "x") as output:
        output.attrs["schema"] = "cuphoton-series-v1"
        output.attrs["complete"] = False
        output.attrs["lag_convention"] = (
            "positive means second signal arrives later"
        )
        output.attrs["normalization"] = (
            "spectrum and lag are raw sums averaged over scheduled windows; "
            "exposure counts are totals across those windows; where "
            "lag_valid_samples > 0, the valid-pair mean is "
            "len(window_starts) * lag / lag_valid_samples"
        )
        groups = output.create_group("results")
        for result in chain(() if first is None else (first,), results):
            group = groups.create_group(str(result.result_index))
            group.attrs["pairs_json"] = json.dumps(result.pairs)
            for name in (
                "sample_start",
                "sample_stop",
                "sample_rate_hz",
                "origin_seconds",
                "fft_samples",
                "lag_mode",
            ):
                group.attrs[name] = getattr(result, name)
            for name in (
                "window_starts",
                "contributing_windows",
                "simultaneous_valid_samples",
                "valid",
                "spectrum",
                "frequencies_hz",
                "lag",
                "lag_samples",
                "lag_valid_samples",
            ):
                value = getattr(result, name)
                if value is not None:
                    group.create_dataset(name, data=np.asarray(value))
        output.attrs["complete"] = True
