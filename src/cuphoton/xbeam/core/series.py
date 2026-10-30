# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Windowed cross-spectra and lag correlations on a uniform sample grid."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..runtime.backend import get_backend, to_numpy
from .products import cross_power


class SeriesSource(Protocol):
    """Bounded random access on one common time grid.

    Reads return one-dimensional samples and an optional Boolean validity
    array. Adapters own file handles; the caller owns their lifetime.
    """

    signal_ids: tuple[str, ...]
    sample_count: int
    sample_rate_hz: float
    origin_seconds: float
    dtypes: Mapping[str, Any]

    def read(
        self, signal_id: str, start: int, stop: int
    ) -> tuple[Any, Any | None]: ...


@dataclass(frozen=True)
class CorrelationSpec:
    pairs: tuple[tuple[str, str], ...]
    window_samples: int
    hop_samples: int | None = None
    windows_per_result: int = 1
    output: str = "spectrum"
    lag_mode: str = "linear"
    precision: str | None = None
    batch_windows: int = 16
    backend: str = "numpy"
    shard_index: int = 0
    shard_count: int = 1
    prefetch_batches: int = 1

    def __post_init__(self):
        object.__setattr__(self, "pairs", tuple(tuple(p) for p in self.pairs))
        for name in (
            "window_samples",
            "windows_per_result",
            "batch_windows",
            "shard_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.prefetch_batches, bool)
            or not isinstance(self.prefetch_batches, int)
            or self.prefetch_batches not in (0, 1)
        ):
            raise ValueError("prefetch_batches must be 0 or 1")
        if self.hop_samples is None:
            object.__setattr__(self, "hop_samples", self.window_samples)
        if (
            isinstance(self.hop_samples, bool)
            or not isinstance(self.hop_samples, int)
            or self.hop_samples < 1
        ):
            raise ValueError("hop_samples must be a positive integer")
        if not self.pairs or any(
            len(p) != 2 or any(not isinstance(s, str) or not s for s in p)
            for p in self.pairs
        ):
            raise ValueError("pairs must contain nonempty signal ID pairs")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("duplicate pairs are not allowed")
        if self.output not in {"spectrum", "lag", "both"}:
            raise ValueError("output must be spectrum, lag, or both")
        if self.lag_mode not in {"linear", "circular"}:
            raise ValueError("lag_mode must be linear or circular")
        if self.precision not in {None, "float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        if (
            isinstance(self.shard_index, bool)
            or not isinstance(self.shard_index, int)
            or not 0 <= self.shard_index < self.shard_count
        ):
            raise ValueError("shard_index must be in [0, shard_count)")

    @property
    def fft_samples(self) -> int:
        if self.lag_mode == "circular":
            return self.window_samples
        return 1 << (2 * self.window_samples - 2).bit_length()


@dataclass(frozen=True)
class SeriesShard:
    index: int
    first_result: int
    stop_result: int
    sample_start: int
    sample_stop: int


def plan_series_shards(
    sample_count: int, spec: CorrelationSpec
) -> tuple[SeriesShard, ...]:
    """Partition result ordinals, including each read's overlap halo."""
    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a nonnegative integer")
    n = spec.window_samples
    hop = spec.hop_samples
    assert hop is not None
    windows = max(0, 1 + (sample_count - n) // hop)
    results = windows // spec.windows_per_result
    shards = []
    for index in range(spec.shard_count):
        first = results * index // spec.shard_count
        stop = results * (index + 1) // spec.shard_count
        start_sample = first * spec.windows_per_result * hop
        stop_sample = (
            (stop * spec.windows_per_result - 1) * hop + n
            if stop > first
            else start_sample
        )
        shards.append(
            SeriesShard(index, first, stop, start_sample, stop_sample)
        )
    return tuple(shards)


@dataclass(frozen=True)
class SeriesResult:
    """Host arrays; products are raw sums averaged over scheduled windows.

    Axes are [pair, bin]. Positive lag means the second signal arrives later.
    Spectrum uses A * conj(B); delaying B by tau gives phase +2*pi*f*tau.
    lag[k] is ifft(spectrum)[-k mod fft_samples]. Invalid samples contribute
    zero, and all-invalid pairs are explicitly marked invalid.

    contributing_windows, simultaneous_valid_samples and lag_valid_samples
    are totals across the scheduled windows, not averages. They do not
    renormalize the products. Where lag_valid_samples > 0, the valid-pair
    mean is len(window_starts) * lag / lag_valid_samples.
    """

    result_index: int
    pairs: tuple[tuple[str, str], ...]
    sample_start: int
    sample_stop: int
    window_starts: np.ndarray
    sample_rate_hz: float
    origin_seconds: float
    contributing_windows: np.ndarray
    simultaneous_valid_samples: np.ndarray
    valid: np.ndarray
    spectrum: np.ndarray | None
    frequencies_hz: np.ndarray | None
    lag: np.ndarray | None
    lag_samples: np.ndarray | None
    lag_valid_samples: np.ndarray | None
    fft_samples: int
    lag_mode: str


def _read_windows(source, signal, starts, size, xp, dtype, prepared=None):
    # Adapters may coalesce adjacent windows, without reading large gaps.
    read_batch = getattr(source, "read_windows", None)
    if prepared is not None or read_batch is not None:
        values, mask = (
            prepared
            if prepared is not None
            else read_batch(signal, starts, size)
        )
        values = xp.asarray(values, dtype=dtype)
        mask = None if mask is None else xp.asarray(mask)
    else:
        samples, masks = [], []
        for start in starts:
            v, m = source.read(signal, start, start + size)
            v = xp.asarray(v, dtype=dtype)
            if v.shape != (size,):
                raise ValueError(f"short or non-vector read for {signal!r}")
            if m is not None:
                m = xp.asarray(m)
                if m.dtype != xp.bool_ or m.shape != (size,):
                    raise ValueError(
                        "validity must be a Boolean vector matching samples"
                    )
            samples.append(v)
            masks.append(m)
        values = xp.stack(samples)
        mask = (
            xp.stack(
                [
                    xp.ones(size, dtype=xp.bool_) if m is None else m
                    for m in masks
                ]
            )
            if any(m is not None for m in masks)
            else None
        )
    if values.shape != (len(starts), size):
        raise ValueError(f"short or non-vector read for {signal!r}")
    if mask is not None and (
        mask.dtype != xp.bool_ or mask.shape != values.shape
    ):
        raise ValueError("validity must be a Boolean vector matching samples")
    invalid = ~xp.isfinite(values)
    if bool(xp.any(invalid if mask is None else mask & invalid)):
        raise ValueError(f"nonfinite valid samples in {signal!r}")
    return values if mask is None else xp.where(mask, values, 0), mask


def _batch_requests(signals, shard, spec):
    for result in range(shard.first_result, shard.stop_result):
        first = result * spec.windows_per_result
        for offset in range(0, spec.windows_per_result, spec.batch_windows):
            stop = min(offset + spec.batch_windows, spec.windows_per_result)
            starts = [
                (first + w) * spec.hop_samples for w in range(offset, stop)
            ]
            yield signals, starts, spec.window_samples


def correlate_series(
    source: SeriesSource, spec: CorrelationSpec
) -> Iterator[SeriesResult]:
    """Stream bounded correlation results, dropping incomplete result groups.

    Linear mode uses a zero-padded FFT grid for all output selections so
    requesting both shares forward transforms and gives identical spectra
    to spectrum-only. Circular mode uses exactly window_samples FFT bins.
    """
    if not math.isfinite(source.sample_rate_hz) or source.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if not math.isfinite(source.origin_seconds):
        raise ValueError("origin_seconds must be finite")
    signals = tuple(dict.fromkeys(s for p in spec.pairs for s in p))
    if len(set(source.signal_ids)) != len(source.signal_ids):
        raise ValueError("source signal IDs must be unique")
    if any(s not in source.signal_ids for s in signals):
        raise ValueError("requested pair contains an unavailable signal")
    dtypes = [np.dtype(source.dtypes[s]) for s in signals]
    if any(d.kind not in "fciub" for d in dtypes):
        raise ValueError("samples must have numeric dtypes")
    if spec.precision is None and any(
        d.kind in "fc" and d.itemsize > (16 if d.kind == "c" else 8)
        for d in dtypes
    ):
        raise ValueError(
            "extended precision requires an explicit precision override"
        )
    precision = spec.precision or (
        "float64"
        if any(
            d.kind in "iub" or d.itemsize > (8 if d.kind == "c" else 4)
            for d in dtypes
        )
        else "float32"
    )
    complex_dtype = np.dtype(
        "complex64" if precision == "float32" else "complex128"
    )
    input_dtype = (
        complex_dtype
        if any(d.kind == "c" for d in dtypes)
        else np.dtype(precision)
    )
    xp = get_backend(spec.backend)
    n, fft_n, group_n = (
        spec.window_samples,
        spec.fft_samples,
        spec.windows_per_result,
    )
    shard = plan_series_shards(source.sample_count, spec)[spec.shard_index]
    want_lag = spec.output != "spectrum"
    want_spectrum = spec.output != "lag"
    lags = (
        (
            np.arange(-(n - 1), n, dtype=np.int64)
            if spec.lag_mode == "linear"
            else np.arange(-(n // 2), n - n // 2, dtype=np.int64)
        )
        if want_lag
        else None
    )
    lag_indices = xp.asarray((-lags) % fft_n) if lags is not None else None
    frequencies = (
        np.fft.fftfreq(fft_n, 1 / source.sample_rate_hz)
        if want_spectrum
        else None
    )
    ones_spectrum = None
    full_coverage = (
        (
            n - np.abs(lags)
            if spec.lag_mode == "linear"
            else np.full(n, n, dtype=np.int64)
        )
        if lags is not None
        else None
    )
    prefetch = getattr(source, "prefetch_batches", None)
    reader = (
        prefetch(_batch_requests(signals, shard, spec))
        if spec.prefetch_batches and prefetch is not None
        else nullcontext(None)
    )
    assert spec.hop_samples is not None
    with reader as pending:
        for result_index in range(shard.first_result, shard.stop_result):
            first_window = result_index * group_n
            starts = [
                (first_window + w) * spec.hop_samples for w in range(group_n)
            ]
            products = xp.zeros((len(spec.pairs), fft_n), dtype=complex_dtype)
            exposure = xp.zeros(len(spec.pairs), dtype=xp.int64)
            contributing = xp.zeros(len(spec.pairs), dtype=xp.int64)
            mask_products = None
            unmasked_windows = np.zeros(len(spec.pairs), dtype=np.int64)
            for offset in range(0, group_n, spec.batch_windows):
                batch_starts = starts[offset : offset + spec.batch_windows]
                prepared = None
                if pending is not None:
                    try:
                        prepared = next(pending)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "prefetched source closed before completion"
                        ) from exc
                spectra, masks, mask_spectra = {}, {}, {}
                for signal in signals:
                    values, mask = _read_windows(
                        source,
                        signal,
                        batch_starts,
                        n,
                        xp,
                        input_dtype,
                        prepared=prepared[signal]
                        if prepared is not None
                        else None,
                    )
                    spectra[signal] = xp.fft.fft(
                        values, n=fft_n, axis=-1
                    ).astype(complex_dtype, copy=False)
                    masks[signal] = mask
                    if want_lag and mask is not None:
                        mask_spectra[signal] = xp.fft.fft(
                            mask.astype(xp.float64), n=fft_n, axis=-1
                        )
                for pair_index, (a, b) in enumerate(spec.pairs):
                    products[pair_index] += xp.sum(
                        cross_power(spectra[a], spectra[b], xp), axis=0
                    )
                    ma, mb = masks[a], masks[b]
                    if ma is None and mb is None:
                        exposure[pair_index] += len(batch_starts) * n
                        contributing[pair_index] += len(batch_starts)
                        unmasked_windows[pair_index] += len(batch_starts)
                        continue
                    simultaneous = (
                        mb if ma is None else ma if mb is None else ma & mb
                    )
                    exposure[pair_index] += xp.sum(
                        simultaneous, dtype=xp.int64
                    )
                    active = (
                        xp.any(mb, axis=-1)
                        if ma is None
                        else xp.any(ma, axis=-1)
                        if mb is None
                        else xp.any(ma, axis=-1) & xp.any(mb, axis=-1)
                    )
                    contributing[pair_index] += xp.sum(active)
                    if want_lag:
                        if mask_products is None:
                            mask_products = xp.zeros(
                                (len(spec.pairs), fft_n), dtype=xp.complex128
                            )
                        if (
                            ma is None or mb is None
                        ) and ones_spectrum is None:
                            ones_spectrum = xp.fft.fft(
                                xp.ones(n, dtype=xp.float64), n=fft_n
                            )
                        sa = ones_spectrum if ma is None else mask_spectra[a]
                        sb = ones_spectrum if mb is None else mask_spectra[b]
                        mask_products[pair_index] += xp.sum(
                            cross_power(sa, sb, xp),
                            axis=0,
                        )
            products /= group_n
            lag = (
                xp.fft.ifft(products, axis=-1)[:, lag_indices]
                if want_lag
                else None
            )
            lag_coverage = None
            if want_lag:
                assert full_coverage is not None
                lag_coverage = (
                    unmasked_windows[:, None] * full_coverage[None, :]
                )
                if mask_products is not None:
                    lag_coverage = xp.asarray(lag_coverage) + xp.rint(
                        xp.fft.ifft(mask_products, axis=-1).real[
                            :, lag_indices
                        ]
                    ).astype(xp.int64)
            count = np.asarray(to_numpy(contributing))
            yield SeriesResult(
                result_index=result_index,
                pairs=spec.pairs,
                sample_start=starts[0],
                sample_stop=starts[-1] + n,
                window_starts=np.asarray(starts, dtype=np.int64),
                sample_rate_hz=source.sample_rate_hz,
                origin_seconds=source.origin_seconds,
                contributing_windows=count,
                simultaneous_valid_samples=np.asarray(to_numpy(exposure)),
                valid=count > 0,
                spectrum=np.asarray(to_numpy(products))
                if want_spectrum
                else None,
                frequencies_hz=frequencies,
                lag=np.asarray(to_numpy(lag), dtype=complex_dtype)
                if want_lag
                else None,
                lag_samples=lags,
                lag_valid_samples=np.asarray(to_numpy(lag_coverage))
                if want_lag
                else None,
                fft_samples=fft_n,
                lag_mode=spec.lag_mode,
            )
