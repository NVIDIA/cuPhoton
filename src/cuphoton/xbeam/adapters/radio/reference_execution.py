# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Portable reference accumulation over the shared station-processing path."""

from __future__ import annotations

import math
from collections.abc import Mapping

from cuphoton.xbeam.adapters.radio.compute_plan import (
    _coordinates,
    _fft_length,
    _group_block_starts,
    _logical_band,
    _model,
    _spectral_normalization,
)
from cuphoton.xbeam.adapters.radio.gpu_execution import (
    _apply_fft_eligibility,
    _bounded_station_blocks,
    _reported_station_weights,
)
from cuphoton.xbeam.adapters.radio.models import AutocorrelationMode
from cuphoton.xbeam.adapters.radio.output_records import _make_record
from cuphoton.xbeam.dsp import average_channels
from cuphoton.xbeam.missing_timestamp_recovery import (
    GlobalFFTEligibilityPlan,
    RecoveryPlanKey,
)
from cuphoton.xbeam.runtime.backend import to_numpy
from cuphoton.xbeam.runtime_types import _StationBlocks


def _run_reference(
    job,
    xp,
    timing_groups,
    band_data,
    frequencies,
    datastreams,
    stations,
    requested_auto_bands,
    batch_blocks,
    autocorrelation_mode,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
):
    baseline_station_ids = {
        datastreams[datastream_id].station_id
        for baseline in job.baselines
        for datastream_id in (
            baseline.datastream_a_id,
            baseline.datastream_b_id,
        )
    }

    integration_s = float(job.integration_duration_s)
    for _group_index, (integration_index, scan, subintegrations) in enumerate(
        timing_groups
    ):
        integration_origin = max(0.0, float(scan.start_offset_s))
        nominal_midpoint = (
            integration_origin + (integration_index + 0.5) * integration_s
        )
        active_models = {
            station_id: _model(scan, station_id, nominal_midpoint)
            for station_id in baseline_station_ids
        }
        spectra_cache: dict[tuple[int, int], _StationBlocks] = {}
        reported_weights_cache = {}
        for key, data in band_data.items():
            all_starts = _group_block_starts(job, data, scan, subintegrations)
            plan = (
                None
                if recovery_plans is None
                else recovery_plans.get((scan.identifier, key[0], key[1]))
            )
            spectra_cache[key] = _apply_fft_eligibility(
                _bounded_station_blocks(
                    job,
                    data,
                    scan,
                    all_starts,
                    xp,
                    batch_blocks,
                    plan,
                ),
                plan=plan,
                scan=scan,
                data=data,
                block_starts=all_starts,
                xp=xp,
            )
            reported_weights_cache[key] = _reported_station_weights(
                spectra_cache[key],
                plan=plan,
                scan=scan,
                data=data,
                block_starts=all_starts,
                xp=xp,
            )

        # Zooms are channel-aligned views of already corrected parent blocks;
        # they deliberately do not repeat station processing or change scale.
        for datastream in job.datastreams:
            for zoom in datastream.zoom_bands:
                parent = spectra_cache[
                    (datastream.identifier, zoom.parent_band_id)
                ]
                start = zoom.start_channel
                stop = start + zoom.channel_count
                spectra_cache[(datastream.identifier, zoom.identifier)] = (
                    _StationBlocks(
                        xp.ascontiguousarray(
                            parent.spectrum[..., start:stop]
                        ),
                        parent.weight,
                        parent.baseband_hz[start:stop],
                        parent.reported_weight,
                    )
                )
                reported_weights_cache[
                    (datastream.identifier, zoom.identifier)
                ] = reported_weights_cache[
                    (datastream.identifier, zoom.parent_band_id)
                ]

        for baseline in job.baselines:
            stream_a = datastreams[baseline.datastream_a_id]
            stream_b = datastreams[baseline.datastream_b_id]
            station_a = stations[stream_a.station_id]
            station_b = stations[stream_b.station_id]
            model_a = active_models[stream_a.station_id]
            model_b = active_models[stream_b.station_id]
            coordinates = _coordinates(model_a, model_b, nominal_midpoint)
            for product in baseline.products:
                data_a, polarization_a, _logical_frequency_a = _logical_band(
                    stream_a, product.band_a_id, band_data, frequencies
                )
                data_b, polarization_b, _logical_frequency_b = _logical_band(
                    stream_b, product.band_b_id, band_data, frequencies
                )
                blocks_a = spectra_cache[
                    (stream_a.identifier, product.band_a_id)
                ]
                blocks_b = spectra_cache[
                    (stream_b.identifier, product.band_b_id)
                ]
                fft_length_a = _fft_length(data_a)
                fft_length_b = _fft_length(data_b)
                normalization = math.sqrt(
                    _spectral_normalization(data_a)
                    * _spectral_normalization(data_b)
                )
                output_frequency = frequencies[product.output_frequency_id]
                block_weights = blocks_a.weight * blocks_b.weight
                weight_sum = float(
                    to_numpy(xp.sum(block_weights, dtype=xp.float64))
                )
                if weight_sum <= 0:
                    continue
                product_blocks = blocks_a.spectrum * xp.conj(
                    blocks_b.spectrum
                )
                accumulator = xp.sum(product_blocks, axis=0)
                visibility = accumulator / xp.float64(
                    normalization * weight_sum
                )
                visibility = average_channels(
                    visibility, output_frequency.channels_to_average, xp
                )
                block_duration = math.sqrt(
                    (fft_length_a / data_a.sample_rate_hz)
                    * (fft_length_b / data_b.sample_rate_hz)
                )
                reported_weights = (
                    reported_weights_cache[
                        (stream_a.identifier, product.band_a_id)
                    ]
                    * reported_weights_cache[
                        (stream_b.identifier, product.band_b_id)
                    ]
                )
                reported_weight_sum = float(
                    to_numpy(xp.sum(reported_weights, dtype=xp.float64))
                )
                record_weight = (
                    reported_weight_sum * block_duration / integration_s
                )
                yield _make_record(
                    job,
                    integration_index,
                    scan,
                    station_a,
                    station_b,
                    product.output_frequency_id,
                    polarization_a,
                    polarization_b,
                    record_weight,
                    coordinates,
                    visibility,
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
                            blocks_a = spectra_cache[
                                (datastream.identifier, band_a_id)
                            ]
                            blocks_b = spectra_cache[
                                (datastream.identifier, band_b_id)
                            ]
                            normalization = math.sqrt(
                                _spectral_normalization(data_a)
                                * _spectral_normalization(data_b)
                            )
                            if band_a_id == band_b_id:
                                block_weights = blocks_a.weight
                            else:
                                block_weights = (
                                    blocks_a.weight * blocks_b.weight
                                )
                            weight_sum = float(
                                to_numpy(
                                    xp.sum(block_weights, dtype=xp.float64)
                                )
                            )
                            if weight_sum <= 0:
                                continue
                            power = xp.sum(
                                blocks_a.spectrum
                                * xp.conj(blocks_b.spectrum),
                                axis=0,
                            )
                            visibility = power / xp.float64(
                                normalization * weight_sum
                            )
                            if (
                                autocorrelation_mode
                                is AutocorrelationMode.TRUNCATION
                                and band_a_id == band_b_id
                            ):
                                output_channels = (
                                    output_frequency.channel_count
                                    // output_frequency.channels_to_average
                                )
                                visibility = visibility[:output_channels]
                            else:
                                visibility = average_channels(
                                    visibility,
                                    output_frequency.channels_to_average,
                                    xp,
                                )
                            block_duration = math.sqrt(
                                (_fft_length(data_a) / data_a.sample_rate_hz)
                                * (
                                    _fft_length(data_b)
                                    / data_b.sample_rate_hz
                                )
                            )
                            reported_weights = reported_weights_cache[
                                (datastream.identifier, band_a_id)
                            ]
                            if band_a_id != band_b_id:
                                reported_weights = (
                                    reported_weights
                                    * (
                                        reported_weights_cache[
                                            (datastream.identifier, band_b_id)
                                        ]
                                    )
                                )
                            reported_weight_sum = float(
                                to_numpy(
                                    xp.sum(reported_weights, dtype=xp.float64)
                                )
                            )
                            record_weight = (
                                reported_weight_sum
                                * block_duration
                                / integration_s
                            )
                            yield _make_record(
                                job,
                                integration_index,
                                scan,
                                station,
                                station,
                                output_frequency_id,
                                polarization_a,
                                polarization_b,
                                record_weight,
                                (0.0, 0.0, 0.0),
                                visibility,
                            )
