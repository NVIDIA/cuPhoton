# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static validation for normalized jobs."""

from __future__ import annotations

import math
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from cuphoton.xbeam.adapters.radio.models import (
    SCHEMA_VERSION,
    DataSourceKind,
    Frequency,
    Job,
    MJDTime,
    RecordedBand,
    SamplingKind,
    Sideband,
    ZoomBand,
)

from .errors import ConfigurationError, UnsupportedFeatureError

_FREQUENCY_ABS_TOL_HZ = 1e-6
_FREQUENCY_REL_TOL = 1e-12


def _error(entity: str, field: str, reason: str) -> None:
    raise ConfigurationError(entity, field, reason)


def _finite(value: float, entity: str, field: str) -> None:
    if not math.isfinite(value):
        _error(entity, field, "must be finite")


def _decimal_finite(value: Decimal, entity: str, field: str) -> None:
    if not value.is_finite():
        _error(entity, field, "must be finite")


def _timestamp(value: MJDTime, entity: str, field: str) -> None:
    if isinstance(value.mjd, bool) or value.mjd < 0:
        _error(entity, f"{field}.mjd", "must be a nonnegative integer")
    _decimal_finite(value.seconds_of_day, entity, f"{field}.seconds_of_day")
    if value.seconds_of_day < 0 or value.seconds_of_day >= Decimal(86400):
        _error(entity, f"{field}.seconds_of_day", "must be in [0, 86400)")


def _unique_nonnegative(items: Iterable[Any], kind: str) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for index, item in enumerate(items):
        entity = f"{kind}[{index}]"
        identifier = item.identifier
        if isinstance(identifier, bool) or identifier < 0:
            _error(entity, "identifier", "must be a nonnegative integer")
        if identifier in result:
            _error(
                entity,
                "identifier",
                f"duplicate {kind} identifier {identifier}",
            )
        result[identifier] = item
    return result


def _coefficients(values: tuple[float, ...], entity: str, field: str) -> None:
    if not values:
        _error(entity, field, "must contain at least one coefficient")
    for index, value in enumerate(values):
        _finite(value, entity, f"{field}[{index}]")


def _frequency_grid(
    frequency: Frequency,
    sampling: SamplingKind = SamplingKind.COMPLEX,
) -> tuple[int, float, float]:
    spacing = frequency.bandwidth_hz / frequency.channel_count
    if sampling is SamplingKind.COMPLEX_DSB:
        lower_hz = frequency.band_edge_hz
    elif frequency.sideband is Sideband.LSB:
        lower_hz = frequency.band_edge_hz - frequency.bandwidth_hz
    else:
        lower_hz = frequency.band_edge_hz
    return frequency.channel_count, lower_hz, spacing


def _same_frequency_value(first: float, second: float) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=_FREQUENCY_REL_TOL,
        abs_tol=_FREQUENCY_ABS_TOL_HZ,
    )


def _same_grid(
    first: tuple[int, float, float],
    second: tuple[int, float, float],
) -> bool:
    return (
        first[0] == second[0]
        and _same_frequency_value(first[1], second[1])
        and _same_frequency_value(first[2], second[2])
    )


def _reject_unsupported_sideband(
    sampling: SamplingKind,
    frequency: Frequency,
    entity: str,
    field: str,
) -> None:
    if (
        sampling is SamplingKind.COMPLEX_DSB
        and frequency.sideband is Sideband.LSB
    ):
        raise UnsupportedFeatureError(
            entity,
            field,
            "complex DSB lower-sideband ordering is not supported",
        )


def _logical_grid(
    datastream: Any,
    band: RecordedBand | ZoomBand,
    frequencies: dict[int, Frequency],
) -> tuple[int, float, float]:
    frequency_id = (
        band.frequency_id
        if isinstance(band, RecordedBand)
        else band.output_frequency_id
    )
    return _frequency_grid(frequencies[frequency_id], datastream.sampling)


def validate_job(job: Job) -> None:
    """Reject invalid and unsupported metadata before recording I/O begins."""

    if job.schema_version != SCHEMA_VERSION:
        _error(
            "job",
            "schema_version",
            f"unsupported version {job.schema_version}; expected "
            f"{SCHEMA_VERSION}",
        )
    if job.buffered_reader is not None:
        for name in ("buffer_factor", "data_segments", "buffered_ffts"):
            value = getattr(job.buffered_reader, name)
            if type(value) is not int or value <= 0:
                _error(
                    "job",
                    f"buffered_reader.{name}",
                    "must be a positive integer",
                )
    _timestamp(job.start_time, "job", "start_time")
    for field_name in (
        "execution_duration_s",
        "integration_duration_s",
        "subintegration_duration_s",
    ):
        value = getattr(job, field_name)
        _decimal_finite(value, "job", field_name)
        if value <= 0:
            _error("job", field_name, "must be greater than zero")
    if job.fringe_approximation_order not in (0, 1, 2):
        _error("job", "fringe_approximation_order", "must be 0, 1, or 2")
    if job.fringe_approximation_order == 2 and (
        any(any(station.clock.coefficients_s[3:]) for station in job.stations)
        or any(
            any(model.delay_coefficients_s[3:])
            for scan in job.scans
            for model in scan.station_models
        )
    ):
        raise UnsupportedFeatureError(
            "job",
            "fringe_approximation_order",
            "order 2 with nonzero cubic or higher delay/clock terms "
            "is unsupported; "
            "CPU and GPU approximations are not equivalent",
        )

    for feature, enabled in vars(job.unsupported_features).items():
        if enabled:
            raise UnsupportedFeatureError(
                "job.unsupported_features", feature, "is not supported"
            )
    if job.output.spectral_placement != 0:
        raise UnsupportedFeatureError(
            "job.output",
            "spectral_placement",
            "nonzero spectral placement is not supported",
        )

    stations = _unique_nonnegative(job.stations, "station")
    if not stations:
        _error("job", "stations", "must contain at least one station")
    station_names: set[str] = set()
    for index, station in enumerate(job.stations):
        entity = f"station[{index}]"
        if not station.name.strip():
            _error(entity, "name", "must not be empty")
        if station.name in station_names:
            _error(entity, "name", f"duplicate station name {station.name!r}")
        station_names.add(station.name)
        _timestamp(station.clock.epoch, entity, "clock.epoch")
        _coefficients(
            station.clock.coefficients_s, entity, "clock.coefficients_s"
        )

    frequencies = _unique_nonnegative(job.frequencies, "frequency")
    if not frequencies:
        _error("job", "frequencies", "must contain at least one frequency")
    for index, frequency in enumerate(job.frequencies):
        entity = f"frequency[{index}]"
        _finite(frequency.band_edge_hz, entity, "band_edge_hz")
        _finite(frequency.bandwidth_hz, entity, "bandwidth_hz")
        if frequency.band_edge_hz < 0:
            _error(entity, "band_edge_hz", "must be nonnegative")
        if frequency.bandwidth_hz <= 0:
            _error(entity, "bandwidth_hz", "must be greater than zero")
        if frequency.channel_count <= 0:
            _error(entity, "channel_count", "must be greater than zero")
        if frequency.channels_to_average <= 0:
            _error(entity, "channels_to_average", "must be greater than zero")
        if frequency.channel_count % frequency.channels_to_average:
            _error(
                entity,
                "channels_to_average",
                "must divide channel_count exactly",
            )
        if frequency.oversampling != 1:
            raise UnsupportedFeatureError(
                entity,
                "oversampling",
                "only an oversampling factor of 1 is supported",
            )
        if frequency.decimation != 1:
            raise UnsupportedFeatureError(
                entity,
                "decimation",
                "only a decimation factor of 1 is supported",
            )

    datastreams = _unique_nonnegative(job.datastreams, "datastream")
    if not datastreams:
        _error("job", "datastreams", "must contain at least one datastream")
    for index, datastream in enumerate(job.datastreams):
        entity = f"datastream[{index}]"
        if datastream.station_id not in stations:
            _error(
                entity,
                "station_id",
                f"unknown station identifier {datastream.station_id}",
            )
        if datastream.data_format.upper() != "VDIF":
            raise UnsupportedFeatureError(
                entity, "data_format", "only non-legacy VDIF is supported"
            )
        if datastream.data_source is not DataSourceKind.FILE:
            raise UnsupportedFeatureError(
                entity,
                "data_source",
                "only file recording sources are supported by normalized "
                "jobs",
            )
        if datastream.quantization_bits != 2:
            raise UnsupportedFeatureError(
                entity, "quantization_bits", "only two-bit input is supported"
            )
        if (
            datastream.frame_size_bytes <= 32
            or datastream.frame_size_bytes % 8
        ):
            _error(
                entity,
                "frame_size_bytes",
                "must be greater than 32 and a multiple of 8",
            )
        if not datastream.source_files:
            _error(entity, "source_files", "must contain at least one path")
        for file_index, source_file in enumerate(datastream.source_files):
            if not source_file.strip():
                _error(
                    entity, f"source_files[{file_index}]", "must not be empty"
                )
        if datastream.sample_rate_hz is not None:
            _finite(datastream.sample_rate_hz, entity, "sample_rate_hz")
            if datastream.sample_rate_hz <= 0:
                _error(entity, "sample_rate_hz", "must be greater than zero")
        bands = _unique_nonnegative(
            datastream.recorded_bands, f"{entity}.recorded_band"
        )
        if not bands:
            _error(entity, "recorded_bands", "must contain at least one band")
        for band_index, band in enumerate(datastream.recorded_bands):
            band_entity = f"{entity}.recorded_band[{band_index}]"
            if band.frequency_id not in frequencies:
                _error(
                    band_entity,
                    "frequency_id",
                    f"unknown frequency identifier {band.frequency_id}",
                )
            if not band.polarization.strip():
                _error(band_entity, "polarization", "must not be empty")
            if band.thread_id < 0:
                _error(band_entity, "thread_id", "must be nonnegative")
            _finite(band.clock_offset_s, band_entity, "clock_offset_s")
            _finite(band.lo_offset_hz, band_entity, "lo_offset_hz")
            frequency = frequencies[band.frequency_id]
            _reject_unsupported_sideband(
                datastream.sampling,
                frequency,
                band_entity,
                "frequency_id",
            )
            sample_rate = datastream.sample_rate_hz
            expected_rate = frequency.bandwidth_hz * (
                2.0 if datastream.sampling is SamplingKind.REAL else 1.0
            )
            if sample_rate is None:
                sample_rate = expected_rate
            elif not _same_frequency_value(sample_rate, expected_rate):
                raise UnsupportedFeatureError(
                    band_entity,
                    "sample_rate_hz",
                    "the supported FFT grid requires sample rate equal to "
                    "twice bandwidth for real input or bandwidth for "
                    "complex input",
                )
            fft_length = frequency.channel_count * (
                2 if datastream.sampling.value == "real" else 1
            )
            samples_per_subintegration = (
                Decimal(str(sample_rate)) * job.subintegration_duration_s
            )
            if (
                samples_per_subintegration
                != samples_per_subintegration.to_integral_value()
            ):
                _error(
                    band_entity,
                    "frequency_id",
                    "subintegration does not contain an integral number of "
                    "samples",
                )
            if int(samples_per_subintegration) % fft_length:
                _error(
                    band_entity,
                    "frequency_id",
                    "subintegration sample count must be divisible by the "
                    "transform length",
                )
        rates_by_thread: dict[int, list[float]] = {}
        for band in datastream.recorded_bands:
            frequency = frequencies[band.frequency_id]
            sample_rate = datastream.sample_rate_hz
            if sample_rate is None:
                sample_rate = frequency.bandwidth_hz * (
                    2.0 if datastream.sampling is SamplingKind.REAL else 1.0
                )
            rates_by_thread.setdefault(band.thread_id, []).append(sample_rate)
        for thread_id, rates in rates_by_thread.items():
            if any(
                not _same_frequency_value(rates[0], rate)
                for rate in rates[1:]
            ):
                _error(
                    entity,
                    "recorded_bands",
                    f"bands sharing thread {thread_id} must use one "
                    "sample rate",
                )
        zooms = _unique_nonnegative(
            datastream.zoom_bands, f"{entity}.zoom_band"
        )
        overlapping_band_ids = sorted(set(bands) & set(zooms))
        if overlapping_band_ids:
            _error(
                entity,
                "zoom_bands",
                "zoom and recorded band identifiers overlap: "
                f"{overlapping_band_ids}",
            )
        for zoom_index, zoom in enumerate(datastream.zoom_bands):
            zoom_entity = f"{entity}.zoom_band[{zoom_index}]"
            if zoom.parent_band_id not in bands:
                _error(
                    zoom_entity,
                    "parent_band_id",
                    f"unknown recorded band identifier {zoom.parent_band_id}",
                )
            if zoom.output_frequency_id not in frequencies:
                _error(
                    zoom_entity,
                    "output_frequency_id",
                    "unknown frequency identifier "
                    f"{zoom.output_frequency_id}",
                )
            if zoom.start_channel < 0:
                _error(zoom_entity, "start_channel", "must be nonnegative")
            if zoom.channel_count <= 0:
                _error(
                    zoom_entity, "channel_count", "must be greater than zero"
                )
            parent_frequency = frequencies[
                bands[zoom.parent_band_id].frequency_id
            ]
            if (
                zoom.start_channel + zoom.channel_count
                > parent_frequency.channel_count
            ):
                _error(
                    zoom_entity,
                    "channel_count",
                    "zoom extends beyond its parent frequency",
                )
            output_frequency = frequencies[zoom.output_frequency_id]
            _reject_unsupported_sideband(
                datastream.sampling,
                output_frequency,
                zoom_entity,
                "output_frequency_id",
            )
            if zoom.channel_count != output_frequency.channel_count:
                _error(
                    zoom_entity,
                    "channel_count",
                    "must equal the output frequency channel_count",
                )
            if not zoom.polarization.strip():
                _error(zoom_entity, "polarization", "must not be empty")
            parent_grid = _frequency_grid(
                parent_frequency, datastream.sampling
            )
            expected_grid = (
                zoom.channel_count,
                parent_grid[1] + zoom.start_channel * parent_grid[2],
                parent_grid[2],
            )
            output_grid = _frequency_grid(
                output_frequency, datastream.sampling
            )
            if not _same_grid(expected_grid, output_grid):
                raise UnsupportedFeatureError(
                    zoom_entity,
                    "output_frequency_id",
                    "zoom output does not describe its channel-aligned "
                    "parent slice",
                )

    baselines = _unique_nonnegative(job.baselines, "baseline")
    if not baselines and not job.write_autocorrelations:
        _error(
            "job",
            "baselines",
            "must contain a baseline when autocorrelations are disabled",
        )
    for index, baseline in enumerate(job.baselines):
        entity = f"baseline[{index}]"
        if baseline.datastream_a_id not in datastreams:
            _error(
                entity,
                "datastream_a_id",
                f"unknown datastream identifier {baseline.datastream_a_id}",
            )
        if baseline.datastream_b_id not in datastreams:
            _error(
                entity,
                "datastream_b_id",
                f"unknown datastream identifier {baseline.datastream_b_id}",
            )
        if not baseline.products:
            _error(entity, "products", "must contain at least one product")
        if (
            baseline.datastream_a_id not in datastreams
            or baseline.datastream_b_id not in datastreams
        ):
            continue
        stream_a = datastreams[baseline.datastream_a_id]
        stream_b = datastreams[baseline.datastream_b_id]
        bands_a = {
            item.identifier: item
            for item in (*stream_a.recorded_bands, *stream_a.zoom_bands)
        }
        bands_b = {
            item.identifier: item
            for item in (*stream_b.recorded_bands, *stream_b.zoom_bands)
        }
        product_keys: set[tuple[int, int, int]] = set()
        for product_index, product in enumerate(baseline.products):
            product_entity = f"{entity}.product[{product_index}]"
            if product.band_a_id not in bands_a:
                _error(
                    product_entity,
                    "band_a_id",
                    f"unknown band identifier {product.band_a_id} for "
                    "datastream A",
                )
            if product.band_b_id not in bands_b:
                _error(
                    product_entity,
                    "band_b_id",
                    f"unknown band identifier {product.band_b_id} for "
                    "datastream B",
                )
            if product.output_frequency_id not in frequencies:
                _error(
                    product_entity,
                    "output_frequency_id",
                    "unknown frequency identifier "
                    f"{product.output_frequency_id}",
                )
            if (
                product.band_a_id in bands_a
                and product.band_b_id in bands_b
                and product.output_frequency_id in frequencies
            ):
                grid_a = _logical_grid(
                    stream_a, bands_a[product.band_a_id], frequencies
                )
                grid_b = _logical_grid(
                    stream_b, bands_b[product.band_b_id], frequencies
                )
                output_frequency = frequencies[product.output_frequency_id]
                _reject_unsupported_sideband(
                    stream_a.sampling,
                    output_frequency,
                    product_entity,
                    "output_frequency_id",
                )
                _reject_unsupported_sideband(
                    stream_b.sampling,
                    output_frequency,
                    product_entity,
                    "output_frequency_id",
                )
                output_grids = (
                    _frequency_grid(output_frequency, stream_a.sampling),
                    _frequency_grid(output_frequency, stream_b.sampling),
                )
                if not _same_grid(grid_a, grid_b):
                    raise UnsupportedFeatureError(
                        product_entity,
                        "band_b_id",
                        "input bands require unsupported spectral placement, "
                        "regridding, or stitching",
                    )
                if not any(
                    _same_grid(grid_a, output_grid)
                    for output_grid in output_grids
                ):
                    raise UnsupportedFeatureError(
                        product_entity,
                        "output_frequency_id",
                        "output frequency requires unsupported spectral "
                        "placement, regridding, or stitching",
                    )
            key = (
                product.band_a_id,
                product.band_b_id,
                product.output_frequency_id,
            )
            if key in product_keys:
                _error(
                    product_entity, "band_a_id", "duplicate baseline product"
                )
            product_keys.add(key)

    if not job.scans:
        _error("job", "scans", "must contain at least one scan")
    scan_names: set[str] = set()
    used_station_ids = {
        datastream.station_id for datastream in job.datastreams
    }
    for index, scan in enumerate(job.scans):
        entity = f"scan[{index}]"
        if not scan.identifier.strip():
            _error(entity, "identifier", "must not be empty")
        if scan.identifier in scan_names:
            _error(
                entity,
                "identifier",
                f"duplicate scan identifier {scan.identifier!r}",
            )
        scan_names.add(scan.identifier)
        _decimal_finite(scan.start_offset_s, entity, "start_offset_s")
        _decimal_finite(scan.duration_s, entity, "duration_s")
        if scan.duration_s <= 0:
            _error(entity, "duration_s", "must be greater than zero")
        if (
            scan.start_offset_s >= job.execution_duration_s
            or scan.start_offset_s + scan.duration_s <= 0
        ):
            _error(
                entity,
                "start_offset_s",
                "scan does not overlap the execution interval",
            )
        if not scan.source_id.strip():
            _error(entity, "source_id", "must not be empty")
        if not scan.configuration_id.strip():
            _error(entity, "configuration_id", "must not be empty")
        models_by_station: dict[int, list[Decimal]] = {}
        for model_index, model in enumerate(scan.station_models):
            model_entity = f"{entity}.station_model[{model_index}]"
            if model.station_id not in stations:
                _error(
                    model_entity,
                    "station_id",
                    f"unknown station identifier {model.station_id}",
                )
            _decimal_finite(
                model.epoch_offset_s, model_entity, "epoch_offset_s"
            )
            epochs = models_by_station.setdefault(model.station_id, [])
            if model.epoch_offset_s in epochs:
                _error(
                    model_entity,
                    "epoch_offset_s",
                    f"duplicate model epoch for station {model.station_id}",
                )
            epochs.append(model.epoch_offset_s)
            _coefficients(
                model.delay_coefficients_s,
                model_entity,
                "delay_coefficients_s",
            )
            _coefficients(
                model.u_coefficients_m, model_entity, "u_coefficients_m"
            )
            _coefficients(
                model.v_coefficients_m, model_entity, "v_coefficients_m"
            )
            _coefficients(
                model.w_coefficients_m, model_entity, "w_coefficients_m"
            )
        missing_models = sorted(used_station_ids - set(models_by_station))
        if missing_models:
            _error(
                entity,
                "station_models",
                f"missing models for stations {missing_models}",
            )
        expected_epochs: tuple[Decimal, ...] | None = None
        effective_start = max(Decimal(0), scan.start_offset_s)
        for station_id in sorted(used_station_ids):
            station_epochs = tuple(sorted(models_by_station[station_id]))
            runtime_epochs = tuple(float(epoch) for epoch in station_epochs)
            if any(
                not math.isfinite(epoch) for epoch in runtime_epochs
            ) or any(
                left >= right
                for left, right in zip(
                    runtime_epochs, runtime_epochs[1:], strict=False
                )
            ):
                _error(
                    entity,
                    "station_models",
                    f"model epochs for station {station_id} must remain "
                    "strictly increasing at runtime precision",
                )
            if station_epochs[0] > effective_start:
                _error(
                    entity,
                    "station_models",
                    f"first model for station {station_id} starts after "
                    "the scan execution interval",
                )
            if expected_epochs is None:
                expected_epochs = station_epochs
            elif station_epochs != expected_epochs:
                _error(
                    entity,
                    "station_models",
                    "used stations must have identical model epoch grids",
                )
