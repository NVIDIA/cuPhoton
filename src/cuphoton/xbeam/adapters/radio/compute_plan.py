# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute plan implementation."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from decimal import ROUND_CEILING, Decimal
from typing import Any

import numpy as np

from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    RecordedBand,
    SamplingKind,
    Scan,
)
from cuphoton.xbeam.adapters.vdif.ingest import SampleInterval
from cuphoton.xbeam.dsp import (
    evaluate_polynomial,
)
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.profiles import CompatibilityProfile
from cuphoton.xbeam.runtime_types import (
    _BandSamples,
    _BlockSchedule,
    _GpuGroupChunk,
)

_MAX_BLOCK_SCHEDULE_BYTES = 256 * 1024 * 1024


def _logical_band(
    datastream: Datastream,
    identifier: int,
    band_data: dict[tuple[int, int], _BandSamples],
    frequencies: dict[int, Frequency],
) -> tuple[_BandSamples, str, Frequency]:
    for band in datastream.recorded_bands:
        if band.identifier == identifier:
            data = band_data[(datastream.identifier, identifier)]
            return data, band.polarization, data.frequency
    for zoom in datastream.zoom_bands:
        if zoom.identifier == identifier:
            parent = band_data[(datastream.identifier, zoom.parent_band_id)]
            return (
                parent,
                zoom.polarization,
                frequencies[zoom.output_frequency_id],
            )
    raise ConfigurationError(
        f"datastream[{datastream.identifier}]",
        "band",
        f"unknown logical band {identifier}",
    )


def _mjd_absolute_seconds(
    mjd: int, seconds_of_day: Decimal | float
) -> Decimal:
    return Decimal(mjd) * Decimal(86400) + Decimal(str(seconds_of_day))


def _sample_rate(datastream: Datastream, frequency: Frequency) -> float:
    if datastream.sample_rate_hz is not None:
        return datastream.sample_rate_hz
    factor = 2.0 if datastream.sampling is SamplingKind.REAL else 1.0
    return frequency.bandwidth_hz * factor


def _sample_span_limit(
    job: Job, sample_rate_hz: float, observed_sample_count: int
) -> int:
    declared_samples = math.ceil(
        job.execution_duration_s * Decimal(str(sample_rate_hz))
    )
    return max(declared_samples, observed_sample_count * 4, 1)


def _timing_groups(
    job: Job, profile: CompatibilityProfile
) -> list[tuple[int, Scan, Sequence[int]]]:
    """Assign midpoint-selected subintegrations to integration windows.

    Both execution and scan intervals are right-open.  A subintegration may
    extend beyond either endpoint; its physical midpoint alone determines
    inclusion, as required by the normalized contract.
    """

    subintegration_s = job.subintegration_duration_s
    integration_s = job.integration_duration_s
    execution_end_s = job.execution_duration_s
    if profile.execution_end_on_integer_second:
        execution_end_s -= job.start_time.seconds_of_day % Decimal(1)

    result: list[tuple[int, Scan, Sequence[int]]] = []
    for scan in job.scans:
        integration_origin = max(Decimal(0), scan.start_offset_s)
        scan_end_s = min(
            execution_end_s, scan.start_offset_s + scan.duration_s
        )
        midpoint_span = (scan_end_s - integration_origin) / subintegration_s
        subintegration_count = max(
            0,
            int(
                (midpoint_span - Decimal("0.5")).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        )
        index = 0
        while index < subintegration_count:
            midpoint = (Decimal(index) + Decimal("0.5")) * subintegration_s
            integration_index = int(midpoint // integration_s)
            next_boundary = Decimal(integration_index + 1) * integration_s
            stop = int(
                (
                    next_boundary / subintegration_s - Decimal("0.5")
                ).to_integral_value(rounding=ROUND_CEILING)
            )
            stop = min(subintegration_count, max(index + 1, stop))
            result.append((integration_index, scan, range(index, stop)))
            index = stop
    return sorted(
        result,
        key=lambda item: (
            max(Decimal(0), item[1].start_offset_s)
            + (Decimal(item[0]) + Decimal("0.5")) * integration_s
        ),
    )


def _station_models(
    scan: Any, station_id: int
) -> tuple[DelayCoordinateModel, ...]:
    models = tuple(
        sorted(
            (
                candidate
                for candidate in scan.station_models
                if candidate.station_id == station_id
            ),
            key=lambda candidate: candidate.epoch_offset_s,
        )
    )
    if models:
        return models
    raise ConfigurationError(
        "scan", "station_models", f"missing station {station_id}"
    )


def _model(scan: Any, station_id: int, time_s: float) -> DelayCoordinateModel:
    models = _station_models(scan, station_id)
    if len(models) == 1:
        return models[0]
    index = (
        bisect_right(
            tuple(float(model.epoch_offset_s) for model in models), time_s
        )
        - 1
    )
    if index < 0:
        raise ConfigurationError(
            "scan",
            "station_models",
            f"no station {station_id} model covers time {time_s}",
        )
    return models[index]


def _ordered_model_runs(
    scan: Any, station_id: int, times_s: np.ndarray
) -> tuple[tuple[DelayCoordinateModel, slice], ...]:
    """Return contiguous model selections for nondecreasing timestamps."""

    if not times_s.size:
        return ()
    models = _station_models(scan, station_id)
    if len(models) == 1:
        return ((models[0], slice(0, times_s.size)),)
    epochs = np.fromiter(
        (float(model.epoch_offset_s) for model in models),
        dtype=np.float64,
        count=len(models),
    )
    first = int(np.searchsorted(epochs, times_s[0], side="right") - 1)
    last = int(np.searchsorted(epochs, times_s[-1], side="right") - 1)
    if first < 0:
        raise ConfigurationError(
            "scan",
            "station_models",
            f"no station {station_id} model covers time {times_s[0]}",
        )
    if first == last:
        return ((models[first], slice(0, times_s.size)),)

    # The timestamps can contain millions of FFT-block centres while a scan
    # normally has only a handful of polynomial epochs.  Locate each epoch in
    # the ordered timestamps instead of classifying every timestamp against
    # the epoch array.  This keeps segment selection O(model_count * log N)
    # in both work and temporary storage rather than O(N).
    runs: list[tuple[DelayCoordinateModel, slice]] = []
    start = 0
    active = first
    for next_model in range(first + 1, last + 1):
        stop = int(np.searchsorted(times_s, epochs[next_model], side="left"))
        if stop > start:
            runs.append((models[active], slice(start, stop)))
        start = stop
        active = next_model
    if start < times_s.size:
        runs.append((models[active], slice(start, times_s.size)))
    return tuple(runs)


def _clock_delay(job: Job, station_id: int, elapsed_s: Any, xp: Any) -> Any:
    station = next(
        item for item in job.stations if item.identifier == station_id
    )
    job_abs = _mjd_absolute_seconds(
        job.start_time.mjd, job.start_time.seconds_of_day
    )
    epoch_abs = _mjd_absolute_seconds(
        station.clock.epoch.mjd, station.clock.epoch.seconds_of_day
    )
    offset = xp.asarray(elapsed_s, dtype=xp.float64) + xp.float64(
        float(job_abs - epoch_abs)
    )
    return evaluate_polynomial(station.clock.coefficients_s, offset, xp)


def _delay_values(
    job: Job,
    model: DelayCoordinateModel,
    station_id: int,
    elapsed_s: Any,
    band: RecordedBand,
    xp: Any,
) -> Any:
    offset = xp.asarray(elapsed_s, dtype=xp.float64) - xp.float64(
        float(model.epoch_offset_s)
    )
    delay = evaluate_polynomial(model.delay_coefficients_s, offset, xp)
    delay += _clock_delay(job, station_id, elapsed_s, xp)
    delay += xp.float64(band.clock_offset_s)
    return delay


def _approximated_delay(
    job: Job,
    model: DelayCoordinateModel,
    station_id: int,
    band: RecordedBand,
    block_starts_s: Any,
    fft_length: int,
    sample_rate_hz: float,
    order: int,
    xp: Any,
) -> Any:
    starts = xp.asarray(block_starts_s, dtype=xp.float64)
    offsets = xp.arange(fft_length, dtype=xp.float64) / xp.float64(
        sample_rate_hz
    )
    if order == 0:
        centre = starts + xp.float64(
            (fft_length - 1) / (2.0 * sample_rate_hz)
        )
        values = _delay_values(job, model, station_id, centre, band, xp)
        return values[:, None] + xp.zeros((1, fft_length), dtype=xp.float64)
    if order == 1:
        left = _delay_values(job, model, station_id, starts, band, xp)
        right_t = starts + xp.float64((fft_length - 1) / sample_rate_hz)
        right = _delay_values(job, model, station_id, right_t, band, xp)
        fraction = offsets / xp.float64((fft_length - 1) / sample_rate_hz)
        return left[:, None] + (right - left)[:, None] * fraction[None, :]
    sample_times = starts[:, None] + offsets[None, :]
    return _delay_values(job, model, station_id, sample_times, band, xp)


def _station_payload_sample_bounds(
    job: Job,
    data: _BandSamples,
    scan: Any,
    starts: np.ndarray,
) -> tuple[int, int]:
    """Return the exact packed-payload interval needed by station work."""

    fft_length = _fft_length(data)
    centres = starts + (fft_length - 1) / (2.0 * data.sample_rate_hz)
    first_sample: int | None = None
    stop_sample: int | None = None
    for model, selected in _ordered_model_runs(
        scan, data.datastream.station_id, centres
    ):
        (
            current_fft_length,
            _centre_delay,
            coarse_samples,
            output_indices,
        ) = _source_block_indices(
            job,
            data.datastream,
            data.band,
            data.frequency,
            model,
            starts[selected],
        )
        source_starts = output_indices - coarse_samples - data.start_sample
        current_first = int(source_starts.min())
        current_stop = int(source_starts.max()) + current_fft_length
        first_sample = (
            current_first
            if first_sample is None
            else min(first_sample, current_first)
        )
        stop_sample = (
            current_stop
            if stop_sample is None
            else max(stop_sample, current_stop)
        )
    if first_sample is None or stop_sample is None:
        raise RuntimeError("station payload selection produced no model runs")
    return first_sample, stop_sample


def _coordinates(
    model_a: DelayCoordinateModel,
    model_b: DelayCoordinateModel,
    time_s: float,
) -> tuple[float, float, float]:
    def value(
        coefficients: tuple[float, ...], model: DelayCoordinateModel
    ) -> float:
        return float(
            evaluate_polynomial(
                coefficients, time_s - float(model.epoch_offset_s), np
            )
        )

    return (
        value(model_a.u_coefficients_m, model_a)
        - value(model_b.u_coefficients_m, model_b),
        value(model_a.v_coefficients_m, model_a)
        - value(model_b.v_coefficients_m, model_b),
        value(model_a.w_coefficients_m, model_a)
        - value(model_b.w_coefficients_m, model_b),
    )


def _fft_length(data: _BandSamples) -> int:
    return data.frequency.channel_count * (
        2 if data.datastream.sampling is SamplingKind.REAL else 1
    )


def _spectral_normalization(data: _BandSamples) -> int:
    """Return the per-station FFT scale, including complex I/Q power."""

    component_scale = (
        1 if data.datastream.sampling is SamplingKind.REAL else 2
    )
    return component_scale * _fft_length(data)


def _block_starts(
    job: Job,
    sample_rate_hz: float,
    fft_length: int,
    scan: Any,
    subintegrations: Iterable[int],
) -> np.ndarray:
    subintegration_s = float(job.subintegration_duration_s)
    subintegration_origin = max(0.0, float(scan.start_offset_s))
    block_count = int(round(subintegration_s * sample_rate_hz / fft_length))
    offsets = (
        np.arange(block_count, dtype=np.float64) * fft_length / sample_rate_hz
    )
    indices = np.fromiter(subintegrations, dtype=np.float64)
    if not indices.size:
        return np.empty(0, dtype=np.float64)
    return (
        subintegration_origin
        + indices[:, None] * subintegration_s
        + offsets[None, :]
    ).reshape(-1)


def _source_block_indices(
    job: Job,
    datastream: Datastream,
    band: RecordedBand,
    frequency: Frequency,
    model: DelayCoordinateModel,
    starts: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return FFT length, delays, rounded delay, and output sample indices."""

    sample_rate_hz = _sample_rate(datastream, frequency)
    fft_length = frequency.channel_count * (
        2 if datastream.sampling is SamplingKind.REAL else 1
    )
    centres = starts + (fft_length - 1) / (2.0 * sample_rate_hz)
    centre_delay = np.asarray(
        _delay_values(
            job,
            model,
            datastream.station_id,
            centres,
            band,
            np,
        )
    )
    coarse_samples = np.rint(centre_delay * sample_rate_hz).astype(np.int64)
    output_samples = np.rint(starts * sample_rate_hz).astype(np.int64)
    return fft_length, centre_delay, coarse_samples, output_samples


def _source_block_indices_for_scan(
    job: Job,
    datastream: Datastream,
    band: RecordedBand,
    frequency: Frequency,
    scan: Any,
    starts: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve source indices with the model active at each block centre."""

    sample_rate_hz = _sample_rate(datastream, frequency)
    fft_length = frequency.channel_count * (
        2 if datastream.sampling is SamplingKind.REAL else 1
    )
    centres = starts + (fft_length - 1) / (2.0 * sample_rate_hz)
    centre_delay = np.empty(centres.shape, dtype=np.float64)
    for model, selected in _ordered_model_runs(
        scan, datastream.station_id, centres
    ):
        centre_delay[selected] = _delay_values(
            job,
            model,
            datastream.station_id,
            centres[selected],
            band,
            np,
        )
    coarse_samples = np.rint(centre_delay * sample_rate_hz).astype(np.int64)
    output_samples = np.rint(starts * sample_rate_hz).astype(np.int64)
    return fft_length, centre_delay, coarse_samples, output_samples


def _group_block_starts(
    job: Job,
    data: _BandSamples,
    scan: Any,
    subintegrations: Iterable[int],
) -> np.ndarray:
    return _block_starts(
        job,
        data.sample_rate_hz,
        _fft_length(data),
        scan,
        subintegrations,
    )


def _polynomial_interval(
    coefficients: Sequence[float], lower: float, upper: float
) -> tuple[float, float]:
    """Bound binary64 Horner evaluation of a polynomial on an interval."""

    result_lower = float(coefficients[-1])
    result_upper = result_lower
    for coefficient in reversed(coefficients[:-1]):
        products = (
            result_lower * lower,
            result_lower * upper,
            result_upper * lower,
            result_upper * upper,
        )
        result_lower = float(
            np.nextafter(min(products) + coefficient, -np.inf)
        )
        result_upper = float(
            np.nextafter(max(products) + coefficient, np.inf)
        )
    return result_lower, result_upper


def _constant_polynomial_value(
    coefficients: Sequence[float],
) -> float | None:
    if all(coefficient == 0.0 for coefficient in coefficients[1:]):
        return float(coefficients[0])
    return None


def _model_delay_interval(
    job: Job,
    model: DelayCoordinateModel,
    station_id: int,
    band: RecordedBand,
    lower_time_s: float,
    upper_time_s: float,
) -> tuple[float, float]:
    station = next(
        item for item in job.stations if item.identifier == station_id
    )
    model_constant = _constant_polynomial_value(model.delay_coefficients_s)
    clock_constant = _constant_polynomial_value(station.clock.coefficients_s)
    if model_constant is not None and clock_constant is not None:
        value = model_constant + clock_constant + band.clock_offset_s
        return value, value

    model_lower, model_upper = _polynomial_interval(
        model.delay_coefficients_s,
        lower_time_s - float(model.epoch_offset_s),
        upper_time_s - float(model.epoch_offset_s),
    )
    job_abs = _mjd_absolute_seconds(
        job.start_time.mjd, job.start_time.seconds_of_day
    )
    epoch_abs = _mjd_absolute_seconds(
        station.clock.epoch.mjd, station.clock.epoch.seconds_of_day
    )
    clock_origin = float(job_abs - epoch_abs)
    clock_lower, clock_upper = _polynomial_interval(
        station.clock.coefficients_s,
        lower_time_s + clock_origin,
        upper_time_s + clock_origin,
    )
    lower = np.nextafter(
        model_lower + clock_lower + band.clock_offset_s, -np.inf
    )
    upper = np.nextafter(
        model_upper + clock_upper + band.clock_offset_s, np.inf
    )
    return float(lower), float(upper)


def _coarse_delay_bounds(
    job: Job,
    scan: Any,
    datastream: Datastream,
    band: RecordedBand,
    sample_rate_hz: float,
    lower_time_s: float,
    upper_time_s: float,
) -> tuple[int, int]:
    models = _station_models(scan, datastream.station_id)
    epochs = tuple(float(model.epoch_offset_s) for model in models)
    first_model = bisect_right(epochs, lower_time_s) - 1
    last_model = bisect_right(epochs, upper_time_s) - 1
    if first_model < 0:
        raise ConfigurationError(
            "scan",
            "station_models",
            f"no station {datastream.station_id} model covers time "
            f"{lower_time_s}",
        )
    delay_lower = math.inf
    delay_upper = -math.inf
    for model_index in range(first_model, last_model + 1):
        segment_lower = max(lower_time_s, epochs[model_index])
        segment_upper = upper_time_s
        if model_index + 1 < len(models):
            segment_upper = min(segment_upper, epochs[model_index + 1])
        if segment_upper < segment_lower:
            continue
        current_lower, current_upper = _model_delay_interval(
            job,
            models[model_index],
            datastream.station_id,
            band,
            segment_lower,
            segment_upper,
        )
        delay_lower = min(delay_lower, current_lower)
        delay_upper = max(delay_upper, current_upper)
    if delay_lower == delay_upper:
        rounded = int(np.rint(delay_lower * sample_rate_hz))
        return rounded, rounded
    scaled_lower = float(np.nextafter(delay_lower * sample_rate_hz, -np.inf))
    scaled_upper = float(np.nextafter(delay_upper * sample_rate_hz, np.inf))
    return math.floor(scaled_lower), math.ceil(scaled_upper)


def _required_scan_sample_interval(
    job: Job,
    datastream: Datastream,
    band: RecordedBand,
    frequency: Frequency,
    scan: Any,
    subintegration_parts: Sequence[Sequence[int]],
) -> SampleInterval | None:
    selected_parts = [part for part in subintegration_parts if part]
    if not selected_parts:
        return None
    first_subintegration = min(min(part) for part in selected_parts)
    last_subintegration = max(max(part) for part in selected_parts)
    sample_rate_hz = _sample_rate(datastream, frequency)
    fft_length = frequency.channel_count * (
        2 if datastream.sampling is SamplingKind.REAL else 1
    )
    subintegration_s = float(job.subintegration_duration_s)
    block_count = int(round(subintegration_s * sample_rate_hz / fft_length))
    origin = max(0.0, float(scan.start_offset_s))
    first_start = origin + first_subintegration * subintegration_s
    last_start = (
        origin
        + last_subintegration * subintegration_s
        + (block_count - 1) * fft_length / sample_rate_hz
    )
    centre_offset = (fft_length - 1) / (2.0 * sample_rate_hz)
    coarse_lower, coarse_upper = _coarse_delay_bounds(
        job,
        scan,
        datastream,
        band,
        sample_rate_hz,
        first_start + centre_offset,
        last_start + centre_offset,
    )
    first_output = int(np.rint(first_start * sample_rate_hz))
    last_output = int(np.rint(last_start * sample_rate_hz))
    return SampleInterval(
        first_output - coarse_upper,
        last_output - coarse_lower + fft_length,
    )


def _required_sample_intervals(
    job: Job,
    timing_groups: Sequence[tuple[int, Scan, Sequence[int]]],
) -> dict[tuple[int, int], SampleInterval]:
    """Return conservative source bounds for the selected FFT blocks."""

    frequencies = {item.identifier: item for item in job.frequencies}
    groups_by_scan: dict[str, tuple[Scan, list[Sequence[int]]]] = {}
    for _integration_index, scan, subintegrations in timing_groups:
        grouped = groups_by_scan.get(scan.identifier)
        if grouped is None:
            groups_by_scan[scan.identifier] = (scan, [subintegrations])
        else:
            grouped[1].append(subintegrations)
    required: dict[tuple[int, int], SampleInterval] = {}
    for datastream in job.datastreams:
        for band in datastream.recorded_bands:
            frequency = frequencies[band.frequency_id]
            interval: SampleInterval | None = None
            for scan, subintegration_parts in groups_by_scan.values():
                current = _required_scan_sample_interval(
                    job,
                    datastream,
                    band,
                    frequency,
                    scan,
                    subintegration_parts,
                )
                if current is None:
                    continue
                interval = (
                    current if interval is None else interval.union(current)
                )
            if interval is not None:
                required[(datastream.identifier, band.identifier)] = interval
    return required


def _scan_block_schedule(
    job: Job,
    data: _BandSamples,
    group_indices: list[int],
    timing_groups: Sequence[tuple[int, Scan, Sequence[int]]],
) -> _BlockSchedule:
    """Build every block start for one band/scan with one NumPy operation."""

    fft_length = _fft_length(data)
    subintegration_s = float(job.subintegration_duration_s)
    block_count = int(
        round(subintegration_s * data.sample_rate_hz / fft_length)
    )
    subintegration_counts = np.fromiter(
        (len(timing_groups[index][2]) for index in group_indices),
        dtype=np.int64,
        count=len(group_indices),
    )
    total_subintegrations = int(np.sum(subintegration_counts, dtype=np.int64))
    group_lengths_wide = subintegration_counts * block_count
    if np.any(group_lengths_wide > np.iinfo(np.int32).max):
        raise ConfigurationError(
            f"datastream[{data.datastream.identifier}]",
            "subintegration_duration_s",
            "integration block count exceeds the GPU index range",
        )
    total_blocks = int(np.sum(group_lengths_wide, dtype=np.int64))
    if (
        total_blocks
        > _MAX_BLOCK_SCHEDULE_BYTES // np.dtype(np.float64).itemsize
    ):
        raise ConfigurationError(
            f"datastream[{data.datastream.identifier}]",
            "subintegration_duration_s",
            "block schedule exceeds the supported host-memory limit; "
            "split the job into shorter scans",
        )
    flat_subintegrations = np.empty(0, dtype=np.float64)
    parts = [timing_groups[index][2] for index in group_indices]
    if parts and all(isinstance(part, range) for part in parts):
        ranges = [part for part in parts if isinstance(part, range)]
        contiguous = all(
            left.step == right.step == 1 and left.stop == right.start
            for left, right in zip(ranges, ranges[1:], strict=False)
        )
        if ranges[0].step == 1 and contiguous:
            flat_subintegrations = np.arange(
                ranges[0].start, ranges[-1].stop, dtype=np.float64
            )
    if flat_subintegrations.size != total_subintegrations:
        flat_subintegrations = np.fromiter(
            (
                subintegration
                for index in group_indices
                for subintegration in timing_groups[index][2]
            ),
            dtype=np.float64,
            count=total_subintegrations,
        )
    scan = timing_groups[group_indices[0]][1]
    origin = max(0.0, float(scan.start_offset_s))
    block_offsets = (
        np.arange(block_count, dtype=np.float64)
        * fft_length
        / data.sample_rate_hz
    )
    block_starts = (
        origin
        + flat_subintegrations[:, None] * subintegration_s
        + block_offsets[None, :]
    ).reshape(-1)
    group_lengths = np.ascontiguousarray(group_lengths_wide, dtype=np.int32)
    group_starts = np.empty(len(group_indices), dtype=np.int64)
    group_starts[0] = 0
    if len(group_indices) > 1:
        np.cumsum(group_lengths[:-1], dtype=np.int64, out=group_starts[1:])
    return _BlockSchedule(
        np.ascontiguousarray(block_starts),
        group_starts,
        group_lengths,
    )


def _gpu_group_chunks(
    job: Job,
    band_data: dict[tuple[int, int], _BandSamples],
    groups_by_scan: dict[str, list[int]],
    schedules: dict[tuple[str, tuple[int, int]], _BlockSchedule],
    free_bytes: int,
    requested_blocks: int | None,
) -> list[_GpuGroupChunk]:
    """Plan conservative complete-group chunks from peak live device bytes."""

    budget = max(1, int(free_bytes * 0.60))
    pipeline_group_limit = 65_535
    zoom_channels_by_parent: dict[tuple[int, int], int] = {}
    for datastream in job.datastreams:
        for zoom in datastream.zoom_bands:
            parent = (datastream.identifier, zoom.parent_band_id)
            zoom_channels_by_parent[parent] = (
                zoom_channels_by_parent.get(parent, 0) + zoom.channel_count
            )

    chunks: list[_GpuGroupChunk] = []
    for scan_identifier, group_indices in groups_by_scan.items():
        group_count = len(group_indices)
        cursor = 0
        while cursor < group_count:
            stop = cursor
            retained_bytes = 0
            transient_by_band = {key: 0 for key in band_data}
            blocks_by_band = {key: 0 for key in band_data}
            while stop < group_count and stop - cursor < pipeline_group_limit:
                next_retained = retained_bytes
                next_transient = dict(transient_by_band)
                exceeds_requested = False
                for key, data in band_data.items():
                    length = int(
                        schedules[(scan_identifier, key)].group_lengths[stop]
                    )
                    blocks = blocks_by_band[key] + length
                    if (
                        requested_blocks is not None
                        and blocks > requested_blocks
                    ):
                        exceeds_requested = True
                    channels = data.frequency.channel_count
                    zoom_channels = zoom_channels_by_parent.get(key, 0)
                    next_retained += length * (
                        (channels + zoom_channels)
                        * np.dtype(np.complex64).itemsize
                        + np.dtype(np.float64).itemsize
                    )
                    # This deliberately overestimates the live in-place FFT,
                    # channelization output, and cuFFT workspace.  Inputs and
                    # the output arena were allocated before ``free_bytes``.
                    next_transient[key] += length * _fft_length(data) * 40
                estimated_peak = next_retained + max(next_transient.values())
                if stop > cursor and (
                    estimated_peak > budget or exceeds_requested
                ):
                    break
                retained_bytes = next_retained
                transient_by_band = next_transient
                for key in band_data:
                    blocks_by_band[key] += int(
                        schedules[(scan_identifier, key)].group_lengths[stop]
                    )
                stop += 1
                if estimated_peak > budget:
                    break
            chunks.append(_GpuGroupChunk(scan_identifier, cursor, stop))
            cursor = stop
    return chunks


def _automatic_subintegration_tile_limit(
    job: Job,
    band_data: dict[tuple[int, int], _BandSamples],
    timing_groups: Sequence[tuple[int, Any, Sequence[int]]],
    available_bytes: int,
    requested_limit: int | None,
) -> int:
    """Choose a conservative VRAM-sized subintegration tile.

    Payload and output arenas are already accounted for by the caller.  This
    estimate mirrors ``_gpu_group_chunks`` for the retained station spectra
    and largest transient FFT working set, but permits splitting inside one
    logical integration instead of accepting an oversized first group.
    """

    maximum_count = max((len(group[2]) for group in timing_groups), default=1)
    if requested_limit is not None:
        maximum_count = min(maximum_count, requested_limit)
    zoom_channels_by_parent: dict[tuple[int, int], int] = {}
    for datastream in job.datastreams:
        for zoom in datastream.zoom_bands:
            parent = (datastream.identifier, zoom.parent_band_id)
            zoom_channels_by_parent[parent] = (
                zoom_channels_by_parent.get(parent, 0) + zoom.channel_count
            )

    retained_per_subintegration = 0
    transient_per_subintegration = 0
    subintegration_s = float(job.subintegration_duration_s)
    for key, data in band_data.items():
        fft_length = _fft_length(data)
        block_count = max(
            1,
            int(round(subintegration_s * data.sample_rate_hz / fft_length)),
        )
        channels = data.frequency.channel_count
        zoom_channels = zoom_channels_by_parent.get(key, 0)
        retained_per_subintegration += block_count * (
            (channels + zoom_channels) * np.dtype(np.complex64).itemsize
            + np.dtype(np.float64).itemsize
        )
        transient_per_subintegration = max(
            transient_per_subintegration,
            block_count * fft_length * 40,
        )
    estimated_per_subintegration = max(
        1, retained_per_subintegration + transient_per_subintegration
    )
    budget = max(1, int(available_bytes * 0.60))
    automatic_limit = max(1, budget // estimated_per_subintegration)
    return max(1, min(maximum_count, automatic_limit))


def _partition_subintegrations(
    subintegrations: Sequence[int], limit: int
) -> list[Sequence[int]]:
    """Return contiguous, non-overlapping subintegration tiles."""

    if limit <= 0:
        raise ValueError("subintegration tile limit must be positive")
    return [
        subintegrations[start : start + limit]
        for start in range(0, len(subintegrations), limit)
    ]
