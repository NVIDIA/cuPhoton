# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional HDF5 time-series input and generic result serialization."""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager, contextmanager
from itertools import chain

import numpy as np

from ..runtime.prefetch import BatchPrefetch
from ..runtime.uploads import (
    MAX_UPLOAD_WORKERS,
    bounded_upload_workers,
    normalize_upload_workers,
)


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
    POSIX supports chunked/compressed datasets. cuFile reads aligned native
    contiguous extents into GPU buffers; unsupported layouts are rejected.
    """

    def __init__(
        self,
        path,
        dataset_paths,
        *,
        validity_paths=None,
        io_backend="posix",
        read_buffer_bytes=64 * 1024**2,
        io_queue_depth=2,
        upload_workers="auto",
    ):
        if io_backend not in {"posix", "cufile"}:
            raise ValueError("io_backend must be posix or cufile")
        if (
            isinstance(io_queue_depth, bool)
            or not isinstance(io_queue_depth, int)
            or not 2 <= io_queue_depth <= 4
        ):
            raise ValueError(
                "io_queue_depth must be an integer from 2 through 4"
            )
        self.io_queue_depth = io_queue_depth
        self.upload_workers = normalize_upload_workers(upload_workers)
        self.upload_worker_limit = bounded_upload_workers(
            self.upload_workers,
            automatic=4 if io_backend == "cufile" else 1,
            available=io_queue_depth
            if io_backend == "cufile"
            else io_queue_depth - 1,
        )
        if (
            isinstance(read_buffer_bytes, bool)
            or not isinstance(read_buffer_bytes, int)
            or read_buffer_bytes < 4096
            or read_buffer_bytes % 4096
        ):
            raise ValueError(
                "read_buffer_bytes must be a positive multiple of 4096"
            )
        self.io_backend, self.read_buffer_bytes = (
            io_backend,
            read_buffer_bytes,
        )
        self.required_backend = "cupy" if io_backend == "cufile" else None
        self._readers = set()
        self._sync_reader = None
        self._posix_reader = None
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
            self._names = {
                signal: (
                    dataset.name,
                    self._validity[signal].name
                    if signal in self._validity
                    else None,
                )
                for signal, dataset in self._datasets.items()
            }
            self._layouts = {}
            if self.io_backend == "cufile":
                from .hdf5_cufile import extent

                if self._file.driver != "sec2":
                    raise ValueError(
                        "cuFile requires the HDF5 sec2 file driver"
                    )
                stat = os.fstat(self._file.id.get_vfd_handle())
                self._cufile_identity = stat.st_dev, stat.st_ino
                for dataset in [
                    *self._datasets.values(),
                    *self._validity.values(),
                ]:
                    self._layouts[dataset.name] = extent(
                        dataset, file_id=self._file.id
                    )
            elif self._file.driver == "sec2" and hasattr(os, "preadv"):
                from .hdf5_posix import PosixReader, extent

                candidates = [
                    (dataset.name, extent(dataset, file_id=self._file.id))
                    for dataset in [
                        *self._datasets.values(),
                        *self._validity.values(),
                    ]
                ]
                # External files may contain the same dataset names. Check
                # every owner before coalescing same-file dataset reads.
                if all(info is not None for _, info in candidates):
                    layouts = dict(candidates)
                    fd = self._file.id.get_vfd_handle()
                    file_bytes = os.fstat(fd).st_size
                    if all(
                        info.offset + info.samples * info.dtype.itemsize
                        <= file_bytes
                        for info in layouts.values()
                    ):
                        self._layouts = layouts
                        self.upload_worker_limit = bounded_upload_workers(
                            self.upload_workers,
                            automatic=4,
                            available=MAX_UPLOAD_WORKERS,
                        )
                        self._posix_reader = PosixReader(
                            fd,
                            workers=self.upload_worker_limit,
                            chunk_bytes=min(read_buffer_bytes, 16 * 1024**2),
                        )
        except BaseException:
            self.close()
            raise

    def read(self, signal_id, start, stop):
        if not 0 <= start <= stop <= self.sample_count:
            raise ValueError("read outside source sample bounds")
        if self.io_backend == "cufile":
            if start == stop:
                import cupy

                return (
                    cupy.empty(0, dtype=self.dtypes[signal_id]),
                    cupy.empty(0, dtype=cupy.bool_)
                    if signal_id in self._validity
                    else None,
                )
            values, mask = self.read_windows(signal_id, [start], stop - start)
            return values[0], None if mask is None else mask[0]
        mask = self._validity.get(signal_id)
        if self._posix_reader is not None:
            value_name, mask_name = self._names[signal_id]
            return self._posix_reader.read(
                self._layouts[value_name], start, stop
            ), (
                None
                if mask_name is None
                else self._posix_reader.read(
                    self._layouts[mask_name], start, stop
                )
            )
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
        if self.io_backend == "cufile":
            if self._sync_reader is None:
                from .hdf5_cufile import CuFileBatches

                reader = CuFileBatches(
                    self._file.filename,
                    self._layouts,
                    self._names,
                    (),
                    self.read_buffer_bytes,
                    upload_workers=self.upload_workers,
                    source_identity=self._cufile_identity,
                )
                self._sync_reader = reader.__enter__()
                self._readers.add(reader)
            batch = self._sync_reader.read_request(
                ((signal_id,), starts, size)
            )
            try:
                batch.wait()
                values, mask = batch[signal_id]
                return values.copy(), None if mask is None else mask.copy()
            finally:
                batch.release()
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
        """Own the host worker or GPU slots until all reads finish."""
        if self.io_backend == "cufile":
            from .hdf5_cufile import CuFileBatches

            reader = CuFileBatches(
                self._file.filename,
                self._layouts,
                self._names,
                requests,
                self.read_buffer_bytes,
                self.io_queue_depth,
                upload_workers=self.upload_workers,
                source_identity=self._cufile_identity,
            )
        elif self._posix_reader is not None and self._can_plan():
            from .hdf5_posix_batches import PosixHostBatches

            reader = PosixHostBatches(
                self._posix_reader,
                self._names,
                self._layouts,
                requests,
                self.io_queue_depth,
            )
        else:
            reader = BatchPrefetch(
                self._read_batch,
                requests,
                workers=1
                if self._posix_reader is not None
                else self.upload_worker_limit,
                ahead=self.io_queue_depth - 1,
            )
        self._readers.add(reader)
        try:
            with reader:
                yield reader
        finally:
            self._readers.discard(reader)

    def _can_plan(self):
        # Preserve explicit customization of either public read hook.
        return all(
            name not in self.__dict__
            and getattr(type(self), name) is getattr(Hdf5Source, name)
            for name in ("read", "read_windows")
        )

    @contextmanager
    def prefetch_device_batches(self, requests):
        """Use the common device queue for either raw transport."""
        if self._posix_reader is None or not self._can_plan():
            with self.prefetch_batches(requests) as reader:
                yield reader
            return
        from .hdf5_posix_batches import PosixDeviceBatches

        reader = PosixDeviceBatches(
            self._posix_reader,
            self._file.filename,
            self._layouts,
            self._names,
            requests,
            self.read_buffer_bytes,
            self.io_queue_depth,
            upload_workers=self.upload_workers,
        )
        self._readers.add(reader)
        try:
            with reader:
                yield reader
        finally:
            self._readers.discard(reader)

    def close(self):
        failure = None
        for reader in tuple(self._readers):
            try:
                reader.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        self._readers.clear()
        self._sync_reader = None
        if self._posix_reader is not None:
            try:
                self._posix_reader.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        try:
            self._file.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure

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
            result = result.to_host()
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
