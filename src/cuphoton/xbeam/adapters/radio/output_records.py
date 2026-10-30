# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Output records implementation."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.radio.compute_plan import (
    _coordinates,
    _fft_length,
    _logical_band,
    _model,
)
from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    Frequency,
    Job,
    VisibilityRecord,
)
from cuphoton.xbeam.profiling import nvtx_range
from cuphoton.xbeam.runtime.backend import to_numpy
from cuphoton.xbeam.runtime_types import _BandSamples


@dataclass(slots=True)
class _PendingGpuRecord:
    integration_index: int
    scan: Any
    station_a: Any
    station_b: Any
    output_frequency_id: int
    polarization_a: str
    polarization_b: str
    coordinates: tuple[float, float, float]
    weight_index: int
    block_duration_s: float
    spectrum_offset: int
    channel_count: int


@dataclass(slots=True)
class _GpuOutputArena:
    raw: Any
    spectra: Any
    weights: Any
    weight_adjustments: Any
    spectrum_capacity: int
    record_capacity: int
    spectrum_cursor: int = 0
    record_cursor: int = 0
    retained: list[Any] | None = None

    def reserve(self, channel_count: int) -> tuple[Any, Any, int, int]:
        spectrum_offset = self.spectrum_cursor
        spectrum_stop = spectrum_offset + channel_count
        weight_index = self.record_cursor
        if (
            spectrum_stop > self.spectrum_capacity
            or weight_index >= self.record_capacity
        ):
            raise RuntimeError("GPU output arena capacity was underestimated")
        self.spectrum_cursor = spectrum_stop
        self.record_cursor += 1
        return (
            self.spectra[spectrum_offset:spectrum_stop],
            self.weights[weight_index : weight_index + 1],
            spectrum_offset,
            weight_index,
        )


def _make_record(
    job: Job,
    integration_index: int,
    scan: Any,
    station_a: Any,
    station_b: Any,
    output_frequency_id: int,
    polarization_a: str,
    polarization_b: str,
    weight: float,
    coordinates: tuple[float, float, float],
    spectrum: Any,
) -> VisibilityRecord:
    integration_origin = max(Decimal(0), scan.start_offset_s)
    timestamp_offset = (
        integration_origin
        + (Decimal(integration_index) + Decimal("0.5"))
        * job.integration_duration_s
    )
    seconds = job.start_time.seconds_of_day + timestamp_offset
    day_offset = int(seconds // Decimal(86400))
    day = job.start_time.mjd + day_offset
    seconds -= Decimal(day_offset) * Decimal(86400)
    return VisibilityRecord(
        antenna_a=station_a.name,
        antenna_b=station_b.name,
        mjd=day,
        seconds_of_day=seconds,
        source_id=scan.source_id,
        configuration_id=scan.configuration_id,
        output_frequency_id=output_frequency_id,
        polarization_a=polarization_a,
        polarization_b=polarization_b,
        weight=weight,
        u_m=coordinates[0],
        v_m=coordinates[1],
        w_m=coordinates[2],
        spectrum=np.ascontiguousarray(to_numpy(spectrum), dtype=np.complex64),
    )


def _gpu_output_capacity(
    job: Job,
    timing_group_count: int,
    frequencies: dict[int, Frequency],
    requested_auto_bands: dict[int, dict[int, set[int]]],
) -> tuple[int, int]:
    """Return record and complex-element counts for deferred GPU output."""

    records_per_group = 0
    channels_per_group = 0
    for baseline in job.baselines:
        for product in baseline.products:
            frequency = frequencies[product.output_frequency_id]
            records_per_group += 1
            channels_per_group += (
                frequency.channel_count // frequency.channels_to_average
            )
    if job.write_autocorrelations:
        for datastream in job.datastreams:
            for (
                output_frequency_id,
                requested_bands,
            ) in requested_auto_bands.get(datastream.identifier, {}).items():
                frequency = frequencies[output_frequency_id]
                record_count = len(requested_bands) ** 2
                records_per_group += record_count
                channels_per_group += record_count * (
                    frequency.channel_count // frequency.channels_to_average
                )
    return (
        records_per_group * timing_group_count,
        channels_per_group * timing_group_count,
    )


def _new_gpu_output_arena(
    xp: Any,
    record_capacity: int,
    spectrum_capacity: int,
) -> _GpuOutputArena:
    spectrum_bytes = spectrum_capacity * np.dtype(np.complex64).itemsize
    weight_bytes = record_capacity * np.dtype(np.float64).itemsize
    raw = xp.empty(spectrum_bytes + 2 * weight_bytes, dtype=xp.uint8)
    weight_start = spectrum_bytes
    adjustment_start = weight_start + weight_bytes
    weight_adjustments = raw[adjustment_start:].view(xp.float64)
    weight_adjustments.fill(0)
    return _GpuOutputArena(
        raw=raw,
        spectra=raw[:spectrum_bytes].view(xp.complex64),
        weights=raw[weight_start:adjustment_start].view(xp.float64),
        weight_adjustments=weight_adjustments,
        spectrum_capacity=spectrum_capacity,
        record_capacity=record_capacity,
    )


def _gpu_output_group_limit(
    total_groups: int,
    bytes_per_group: int,
    available_bytes: int,
) -> int:
    if total_groups == 0:
        return 0
    if bytes_per_group == 0:
        return total_groups
    budget = available_bytes // 4
    return min(total_groups, max(1, budget // bytes_per_group))


def _reserve_gpu_records(
    job: Job,
    timing_groups: list[tuple[int, Any, Sequence[int]]],
    band_data: dict[tuple[int, int], _BandSamples],
    frequencies: dict[int, Frequency],
    datastreams: dict[int, Datastream],
    stations: dict[int, Any],
    requested_auto_bands: dict[int, dict[int, set[int]]],
    arena: _GpuOutputArena,
) -> tuple[list[_PendingGpuRecord], dict[tuple[Any, ...], tuple[int, int]]]:
    """Reserve group-major locations before product-major GPU launches."""

    pending: list[_PendingGpuRecord] = []
    reservations: dict[tuple[Any, ...], tuple[int, int]] = {}
    integration_s = float(job.integration_duration_s)
    baseline_station_ids = {
        datastreams[datastream_id].station_id
        for baseline in job.baselines
        for datastream_id in (
            baseline.datastream_a_id,
            baseline.datastream_b_id,
        )
    }
    for group_index, (integration_index, scan, _subintegrations) in enumerate(
        timing_groups
    ):
        integration_origin = max(0.0, float(scan.start_offset_s))
        midpoint = (
            integration_origin + (integration_index + 0.5) * integration_s
        )
        active_models = {
            station_id: _model(scan, station_id, midpoint)
            for station_id in baseline_station_ids
        }
        for baseline_index, baseline in enumerate(job.baselines):
            stream_a = datastreams[baseline.datastream_a_id]
            stream_b = datastreams[baseline.datastream_b_id]
            station_a = stations[stream_a.station_id]
            station_b = stations[stream_b.station_id]
            coordinates = _coordinates(
                active_models[stream_a.station_id],
                active_models[stream_b.station_id],
                midpoint,
            )
            for product_index, product in enumerate(baseline.products):
                data_a, polarization_a, _frequency_a = _logical_band(
                    stream_a, product.band_a_id, band_data, frequencies
                )
                data_b, polarization_b, _frequency_b = _logical_band(
                    stream_b, product.band_b_id, band_data, frequencies
                )
                output_frequency = frequencies[product.output_frequency_id]
                output_channels = (
                    output_frequency.channel_count
                    // output_frequency.channels_to_average
                )
                _output, _weight, spectrum_offset, weight_index = (
                    arena.reserve(output_channels)
                )
                reservations[
                    ("cross", group_index, baseline_index, product_index)
                ] = (spectrum_offset, weight_index)
                pending.append(
                    _PendingGpuRecord(
                        integration_index=integration_index,
                        scan=scan,
                        station_a=station_a,
                        station_b=station_b,
                        output_frequency_id=product.output_frequency_id,
                        polarization_a=polarization_a,
                        polarization_b=polarization_b,
                        coordinates=coordinates,
                        weight_index=weight_index,
                        block_duration_s=math.sqrt(
                            (_fft_length(data_a) / data_a.sample_rate_hz)
                            * (_fft_length(data_b) / data_b.sample_rate_hz)
                        ),
                        spectrum_offset=spectrum_offset,
                        channel_count=output_channels,
                    )
                )
        if job.write_autocorrelations:
            for datastream in job.datastreams:
                station = stations[datastream.station_id]
                for output_frequency_id, requested_bands in sorted(
                    requested_auto_bands.get(
                        datastream.identifier, {}
                    ).items()
                ):
                    output_frequency = frequencies[output_frequency_id]
                    output_channels = (
                        output_frequency.channel_count
                        // output_frequency.channels_to_average
                    )
                    for band_a_id in sorted(requested_bands):
                        for band_b_id in sorted(requested_bands):
                            data_a, polarization_a, _frequency_a = (
                                _logical_band(
                                    datastream,
                                    band_a_id,
                                    band_data,
                                    frequencies,
                                )
                            )
                            data_b, polarization_b, _frequency_b = (
                                _logical_band(
                                    datastream,
                                    band_b_id,
                                    band_data,
                                    frequencies,
                                )
                            )
                            (
                                _output,
                                _weight,
                                spectrum_offset,
                                weight_index,
                            ) = arena.reserve(output_channels)
                            reservations[
                                (
                                    "auto",
                                    group_index,
                                    datastream.identifier,
                                    output_frequency_id,
                                    band_a_id,
                                    band_b_id,
                                )
                            ] = (spectrum_offset, weight_index)
                            pending.append(
                                _PendingGpuRecord(
                                    integration_index=integration_index,
                                    scan=scan,
                                    station_a=station,
                                    station_b=station,
                                    output_frequency_id=output_frequency_id,
                                    polarization_a=polarization_a,
                                    polarization_b=polarization_b,
                                    coordinates=(0.0, 0.0, 0.0),
                                    weight_index=weight_index,
                                    block_duration_s=math.sqrt(
                                        (
                                            _fft_length(data_a)
                                            / data_a.sample_rate_hz
                                        )
                                        * (
                                            _fft_length(data_b)
                                            / data_b.sample_rate_hz
                                        )
                                    ),
                                    spectrum_offset=spectrum_offset,
                                    channel_count=output_channels,
                                )
                            )
    if (
        arena.record_cursor != arena.record_capacity
        or arena.spectrum_cursor != arena.spectrum_capacity
    ):
        raise RuntimeError(
            "GPU output arena capacity did not match emitted records"
        )
    return pending, reservations


def _flush_gpu_records(
    job: Job,
    pending: list[_PendingGpuRecord],
    arena: _GpuOutputArena,
) -> Iterator[VisibilityRecord]:
    """Transfer one device arena and materialize semantic records."""

    if not pending:
        return
    with nvtx_range("xbeam:output-device-to-host"):
        host_raw = np.asarray(to_numpy(arena.raw))
    yield from _host_gpu_records(job, pending, arena, host_raw)


def _host_gpu_records(job, pending, arena, host_raw, *, copy_spectra=False):
    spectrum_bytes = arena.spectrum_capacity * np.dtype(np.complex64).itemsize
    weight_bytes = arena.record_capacity * np.dtype(np.float64).itemsize
    adjustment_start = spectrum_bytes + weight_bytes
    host_spectra = host_raw[:spectrum_bytes].view(np.complex64)
    host_weights = host_raw[spectrum_bytes:adjustment_start].view(np.float64)
    host_weight_adjustments = host_raw[adjustment_start:].view(np.float64)
    if host_weight_adjustments.size == 0:
        # Compatibility for custom/test arenas that predate the separate
        # reported-weight channel.
        host_weight_adjustments = np.zeros(
            arena.record_capacity, dtype=np.float64
        )
    integration_s = float(job.integration_duration_s)
    for record in pending:
        weight_sum = host_weights[record.weight_index]
        reported_weight_sum = (
            weight_sum + host_weight_adjustments[record.weight_index]
        )
        spectrum = host_spectra[
            record.spectrum_offset : record.spectrum_offset
            + record.channel_count
        ]
        if weight_sum <= 0:
            continue
        with nvtx_range("xbeam:output-host-materialize"):
            if copy_spectra:
                # A caller may retain records after the pinned slot is reused.
                spectrum = spectrum.copy()
            spectrum /= weight_sum
            result = _make_record(
                job,
                record.integration_index,
                record.scan,
                record.station_a,
                record.station_b,
                record.output_frequency_id,
                record.polarization_a,
                record.polarization_b,
                reported_weight_sum * record.block_duration_s / integration_s,
                record.coordinates,
                spectrum,
            )
        yield result


def _flush_pipeline_records(job, pipeline):
    with pipeline.completed() as (pending, arena, host):
        yield from _host_gpu_records(
            job, pending, arena, host, copy_spectra=True
        )
