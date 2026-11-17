# SPDX-License-Identifier: Apache-2.0
"""HDF5 metadata adapter for native time-series execution.

Only metadata and owned host results cross Python. File reads, masks, FFTs,
products, and lag reconstruction execute in the native CUDA library.
"""

import math
import os
from importlib import import_module

import numpy as np

from ..adapters.hdf5 import Hdf5Source
from ..adapters.hdf5_layout import extent
from ..core.series import SeriesResult, plan_series_shards
from ..runtime.uploads import bounded_upload_workers


def _configuration(source, spec):
    if not isinstance(source, Hdf5Source):
        raise TypeError("native series execution requires an Hdf5Source")
    if not math.isfinite(source.sample_rate_hz) or source.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if not math.isfinite(source.origin_seconds):
        raise ValueError("origin_seconds must be finite")
    if source.io_queue_depth != 2:
        raise ValueError("native series currently requires io_queue_depth=2")
    file = source._file
    if not file or file.driver != "sec2":
        raise ValueError("native series requires an open HDF5 sec2 file")
    signals = tuple(dict.fromkeys(s for pair in spec.pairs for s in pair))
    if any(s not in source.signal_ids for s in signals):
        raise ValueError("requested pair contains an unavailable signal")
    kinds = {
        np.dtype("float32"): 0,
        np.dtype("float64"): 1,
        np.dtype("complex64"): 2,
        np.dtype("complex128"): 3,
    }
    descriptors = []
    dtypes = []
    for signal in signals:
        dataset = source._datasets[signal]
        info = extent(dataset, file_id=file.id)
        if info is None or info.dtype not in kinds:
            raise ValueError(
                "native series requires allocated, native contiguous "
                "float32/float64/complex64/complex128 HDF5 arrays"
            )
        mask = source._validity.get(signal)
        mask_info = None if mask is None else extent(mask, file_id=file.id)
        if mask is not None and mask_info is None:
            raise ValueError(
                "native validity must be allocated and contiguous"
            )
        descriptors.append(
            dict(
                offset=info.offset,
                dtype=kinds[info.dtype],
                mask_offset=-1 if mask is None else mask_info.offset,
            )
        )
        dtypes.append(info.dtype)
    wide = spec.precision == "float64" or (
        spec.precision is None
        and any(d.itemsize > (8 if d.kind == "c" else 4) for d in dtypes)
    )
    stat = os.fstat(file.id.get_vfd_handle())
    workers = bounded_upload_workers(
        source.upload_workers, automatic=2, available=64
    )
    return dict(
        path=os.path.abspath(file.filename),
        device=stat.st_dev,
        inode=stat.st_ino,
        io=source.io_backend,
        gpu=-1,
        window_samples=spec.window_samples,
        fft_samples=spec.fft_samples,
        windows_per_result=spec.windows_per_result,
        batch_windows=spec.batch_windows,
        sample_count=source.sample_count,
        hop_samples=spec.hop_samples,
        wide=wide,
        want_lag=spec.output != "spectrum",
        want_spectrum=spec.output != "lag",
        linear=spec.lag_mode == "linear",
        inputs=descriptors,
        pairs=[(signals.index(a), signals.index(b)) for a, b in spec.pairs],
        upload_workers=workers,
        read_buffer_bytes=source.read_buffer_bytes,
        prefetch=bool(spec.prefetch_batches),
    )


def correlate_source(source, spec):
    """Stream host-owned results using the optional native CUDA executor."""
    config = _configuration(source, spec)
    shard = plan_series_shards(source.sample_count, spec)[spec.shard_index]
    if shard.first_result == shard.stop_result:
        return
    from . import cuda_available

    if not cuda_available():
        raise RuntimeError(
            "native CUDA execution was not built; enable "
            "CUPHOTON_XBEAM_BUILD_EXT=1 and CUPHOTON_XBEAM_NATIVE_CUDA=1"
        )
    executor = import_module("cuphoton.xbeam.native._binding").SeriesExecutor(
        config
    )
    n, fft_n = spec.window_samples, spec.fft_samples
    count, group_n = len(spec.pairs), spec.windows_per_result
    dtype = np.complex128 if config["wide"] else np.complex64
    lags = (
        np.arange(-(n - 1), n, dtype=np.int64)
        if spec.lag_mode == "linear"
        else np.arange(-(n // 2), n - n // 2, dtype=np.int64)
    )
    frequencies = np.fft.fftfreq(fft_n, 1 / source.sample_rate_hz)
    try:
        for ordinal in range(shard.first_result, shard.stop_result):
            starts = (
                ordinal * group_n + np.arange(group_n, dtype=np.int64)
            ) * spec.hop_samples
            spectrum = (
                np.empty((count, fft_n), dtype)
                if config["want_spectrum"]
                else np.empty(0, dtype)
            )
            lag = (
                np.empty((count, len(lags)), dtype)
                if config["want_lag"]
                else np.empty(0, dtype)
            )
            coverage = (
                np.empty((count, len(lags)), np.int64)
                if config["want_lag"]
                else np.empty(0, np.int64)
            )
            statistics = np.empty((count, 2), np.int64)
            executor.run(
                int(starts[0]),
                group_n,
                spectrum.reshape(-1).view(np.uint8),
                lag.reshape(-1).view(np.uint8),
                statistics.reshape(-1),
                coverage.reshape(-1),
            )
            yield SeriesResult(
                result_index=ordinal,
                pairs=spec.pairs,
                sample_start=int(starts[0]),
                sample_stop=int(starts[-1]) + n,
                window_starts=starts,
                sample_rate_hz=source.sample_rate_hz,
                origin_seconds=source.origin_seconds,
                contributing_windows=statistics[:, 0].copy(),
                simultaneous_valid_samples=statistics[:, 1].copy(),
                valid=statistics[:, 0] > 0,
                spectrum=spectrum if config["want_spectrum"] else None,
                frequencies_hz=frequencies.copy()
                if config["want_spectrum"]
                else None,
                lag=lag if config["want_lag"] else None,
                lag_samples=lags.copy() if config["want_lag"] else None,
                lag_valid_samples=coverage if config["want_lag"] else None,
                fft_samples=fft_n,
                lag_mode=spec.lag_mode,
            )
    finally:
        executor.close()
