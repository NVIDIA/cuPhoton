# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gpu execution implementation."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from cuphoton.xbeam.adapters.radio.compute_plan import (
    _approximated_delay,
    _delay_values,
    _fft_length,
    _gpu_group_chunks,
    _logical_band,
    _ordered_model_runs,
    _partition_subintegrations,
    _scan_block_schedule,
    _source_block_indices,
    _spectral_normalization,
    _station_payload_sample_bounds,
)
from cuphoton.xbeam.adapters.radio.models import (
    AutocorrelationMode,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    SamplingKind,
    Sideband,
)
from cuphoton.xbeam.adapters.radio.output_records import (
    _GpuOutputArena,
    _new_gpu_output_arena,
    _PendingGpuRecord,
    _reserve_gpu_records,
)
from cuphoton.xbeam.dsp import (
    TWO_PI,
    complex_lsb_channels,
    reduce_cycles,
)
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.gpu_kernels import (
    batched_visibility_product,
    batched_visibility_product_packs,
    channelize_and_apply_residual,
    clear_rejected_fft_rows,
    front_rotate,
    front_rotate_packed_real_bands,
    prepare_block_parameters,
    prepare_visibility_index,
    visibility_index_view,
)
from cuphoton.xbeam.ingest_scheduler import _PAYLOAD_PROGRESS_WATCHDOG_SECONDS
from cuphoton.xbeam.missing_timestamp_recovery import (
    GlobalFFTEligibilityPlan,
    RecoveryPlanKey,
    select_fft_authoritative_validity,
    select_fft_eligibility,
    select_fft_source_replay,
    select_fft_source_segments,
    select_fft_source_timing,
    select_fft_source_validity,
    select_fft_weight_exclusions,
    select_fft_weight_overrides,
)
from cuphoton.xbeam.profiling import nvtx_range
from cuphoton.xbeam.runtime_types import (
    _BandSamples,
    _BlockSchedule,
    _GpuGroupChunk,
    _StationBlocks,
)


def _check_source_coverage(
    data: _BandSamples, source_starts: np.ndarray, fft_length: int
) -> tuple[int, int]:
    """Check planned allocations; custom sources may omit boundary samples."""
    if not source_starts.size:
        return 0, 0
    first = int(source_starts.min())
    stop = int(source_starts.max()) + fft_length
    if not data.planned_sample_coverage:
        # RecordingSource batches do not promise the planner's delay margins.
        # Both front ends already zero and invalidate out-of-range samples.
        return first, stop
    if data.device_payload is not None:
        count = data.source_sample_count
        if count is None:
            raise RuntimeError("packed sample buffer has no sample count")
    elif data.device_samples is not None:
        count = data.device_samples.size
    else:
        count = data.samples.size
    if first < 0 or stop > count:
        raise RuntimeError(
            "planned sample buffer does not cover execution request: "
            f"datastream={data.datastream.identifier} "
            f"band={data.band.identifier} "
            f"required=[{first + data.start_sample}, "
            f"{stop + data.start_sample}) "
            f"available=[{data.start_sample}, "
            f"{data.start_sample + count})"
        )
    return first, stop


def _apply_fft_eligibility(
    blocks: _StationBlocks,
    *,
    plan: GlobalFFTEligibilityPlan | None,
    scan: Any,
    data: _BandSamples,
    block_starts: np.ndarray,
    xp: Any,
    _owns_buffers: bool = False,
) -> _StationBlocks:
    """Mask station FFTs; owned GPU buffers allow sparse in-place clearing."""

    if plan is None:
        return blocks
    eligibility = select_fft_eligibility(
        plan,
        scan_origin_s=max(0.0, float(scan.start_offset_s)),
        fft_seconds=_fft_length(data) / data.sample_rate_hz,
        block_starts_s=tuple(float(value) for value in block_starts),
    )
    if all(eligibility):
        return blocks
    # Internal callers pass newly produced buffers before caching them.
    # Public/test callers retain non-mutating semantics by default.
    if _owns_buffers and xp.__name__ == "cupy":
        with nvtx_range("xbeam:clear-rejected-fft-rows"):
            if clear_rejected_fft_rows(
                blocks.spectrum,
                blocks.weight,
                blocks.reported_weight,
                np.flatnonzero(~np.asarray(eligibility, dtype=bool)),
                xp=xp,
            ):
                return blocks
    mask = xp.asarray(eligibility, dtype=xp.bool_)
    return _StationBlocks(
        blocks.spectrum * mask[:, None],
        blocks.weight * mask,
        blocks.baseband_hz,
        (
            None
            if blocks.reported_weight is None
            else blocks.reported_weight * mask
        ),
    )


def _reported_station_weights(
    blocks: _StationBlocks,
    *,
    plan: GlobalFFTEligibilityPlan | None,
    scan: Any,
    data: _BandSamples,
    block_starts: np.ndarray,
    xp: Any,
) -> Any:
    """Apply compatibility-only overrides to the reported station weight."""

    weights = (
        blocks.weight
        if blocks.reported_weight is None
        else blocks.reported_weight
    )
    if plan is None or not plan.has_source_weight_override:
        return weights
    selected = select_fft_weight_overrides(
        plan,
        scan_origin_s=max(0.0, float(scan.start_offset_s)),
        fft_seconds=_fft_length(data) / data.sample_rate_hz,
        block_starts_s=tuple(float(value) for value in block_starts),
        fft_samples=_fft_length(data),
    )
    override = np.asarray(
        [-1 if value is None else value for value in selected],
        dtype=np.int64,
    )
    mask = override >= 0
    if not np.any(mask):
        return weights
    reported = weights.copy()
    device_override = xp.asarray(override, dtype=xp.float64) / xp.float64(
        _fft_length(data)
    )
    reported = xp.where(xp.asarray(mask), device_override, reported)
    return reported


def _covered_interval_lengths(
    starts: np.ndarray,
    stops: np.ndarray,
    intervals: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Return covered samples for parallel right-open query intervals."""

    if starts.shape != stops.shape:
        raise ValueError("coverage query bounds must have matching shapes")
    if not intervals:
        return np.zeros(starts.shape, dtype=np.int64)
    interval_starts = np.fromiter(
        (start for start, _stop in intervals), dtype=np.int64
    )
    interval_stops = np.fromiter(
        (stop for _start, stop in intervals), dtype=np.int64
    )
    lengths = interval_stops - interval_starts
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
    )

    def cumulative(values: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(interval_stops, values, side="right")
        result = prefix[positions].copy()
        partial = positions < interval_starts.size
        selected = positions[partial]
        result[partial] += np.maximum(
            0, values[partial] - interval_starts[selected]
        )
        return result

    return cumulative(stops) - cumulative(starts)


def _timestamp_gap_weight_credit(
    source_starts: np.ndarray,
    source_offsets: np.ndarray,
    source_cutovers: np.ndarray,
    *,
    fft_length: int,
    intervals: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Measure timestamp-hole duration selected by each ordinary output FFT.

    A split replay gets its validity and weight from the physical samples on
    both sides of its cutover.  Applying timestamp-hole credit as well would
    count the missing interval a second time and promote the hybrid FFT toward
    full weight.  A full replay has no ordinary prefix, so its selected source
    interval remains eligible for timestamp-hole credit.
    """

    if not intervals:
        return np.zeros(source_starts.shape, dtype=np.float64)
    first_stops = source_starts + source_cutovers
    second_starts = first_stops + source_offsets
    second_stops = source_starts + fft_length + source_offsets
    covered = _covered_interval_lengths(source_starts, first_stops, intervals)
    covered += _covered_interval_lengths(
        second_starts, second_stops, intervals
    )
    split_replay = (source_offsets != 0) & (source_cutovers != 0)
    covered[split_replay] = 0
    return covered.astype(np.float64) / fft_length


def _apply_station_weight_credits(
    raw_weight: Any,
    credits: np.ndarray | None,
    reported_credits: np.ndarray | None,
    xp: Any,
) -> tuple[Any, Any | None]:
    """Correct normalized frontend weights only where a credit is present."""

    weight = raw_weight
    if credits is not None and np.any(credits):
        device_credits = xp.asarray(credits, dtype=xp.float64)
        weight = xp.where(
            weight > 0,
            xp.minimum(xp.float64(1.0), weight + device_credits),
            weight,
        )
    reported_weight = None
    if reported_credits is not None:
        if not np.any(reported_credits):
            if weight is not raw_weight:
                reported_weight = raw_weight
        elif credits is not None and np.array_equal(
            credits, reported_credits
        ):
            # Equal corrections share normalization and reported weights.
            reported_weight = None
        else:
            device_credits = xp.asarray(reported_credits, dtype=xp.float64)
            reported_weight = xp.where(
                raw_weight > 0,
                xp.minimum(xp.float64(1.0), raw_weight + device_credits),
                raw_weight,
            )
    return weight, reported_weight


def _compat_fft_batch_rows(active_rows: int, enclosing_rows: int) -> int:
    """Reuse bounded FFT shapes without oversized padding for small pieces."""
    if not 1 <= active_rows <= enclosing_rows:
        raise ValueError("active FFT rows must fit the enclosing batch")
    if active_rows == 1:
        return 1
    if 2 * active_rows >= enclosing_rows:
        return enclosing_rows
    return min(enclosing_rows, 1 << (active_rows - 1).bit_length())


def _use_dense_compat_fft_rows(
    xp: Any,
    fringe_order: int,
    authoritative: np.ndarray,
    segments: tuple[Any, ...],
) -> bool:
    """Keep regular GPU batches when one validity mode covers every row."""
    return bool(
        xp.__name__ == "cupy"
        and fringe_order in {0, 1, 2}
        and np.all(authoritative == authoritative[0])
        and not any(item is not None for item in segments)
    )


class _StationBlockOptions(TypedDict, total=False):
    _fft_batch_rows: int
    _fft_eligibility: np.ndarray | None


def _station_blocks(
    job: Job,
    data: _BandSamples,
    model: DelayCoordinateModel,
    output_block_starts: np.ndarray,
    xp: Any,
    source_offsets: np.ndarray | None = None,
    source_cutovers: np.ndarray | None = None,
    valid_starts: np.ndarray | None = None,
    valid_stops: np.ndarray | None = None,
    weight_credit_intervals: tuple[tuple[int, int], ...] = (),
    reported_weight_credit_intervals: (
        tuple[tuple[int, int], ...] | None
    ) = None,
    authoritative_validity: bool = False,
    _fft_batch_rows: int | None = None,
    _fft_eligibility: np.ndarray | None = None,
    source_timing_offsets: np.ndarray | None = None,
) -> _StationBlocks:
    frequency = data.frequency
    (
        fft_length,
        centre_delay,
        coarse_samples,
        output_indices,
    ) = _source_block_indices(
        job,
        data.datastream,
        data.band,
        frequency,
        model,
        output_block_starts,
    )
    source_starts = output_indices - coarse_samples - data.start_sample
    has_replay = source_offsets is not None
    if source_timing_offsets is not None and (
        source_timing_offsets.shape != source_starts.shape
        or source_timing_offsets.dtype != np.dtype(np.int64)
    ):
        raise ValueError("source timing offsets must be int64 per FFT")
    if source_offsets is None:
        source_offsets = np.zeros(source_starts.size, dtype=np.int64)
        source_cutovers = np.zeros(source_starts.size, dtype=np.int64)
    assert source_cutovers is not None
    if (
        source_offsets.shape != source_starts.shape
        or source_cutovers.shape != source_starts.shape
    ):
        raise ValueError("source replay shape does not match FFT blocks")
    if np.any(source_cutovers < 0) or np.any(source_cutovers >= fft_length):
        raise ValueError("source replay cutover lies outside an FFT")
    if valid_starts is None:
        valid_starts = np.zeros(source_starts.size, dtype=np.int64)
        valid_stops = np.full(source_starts.size, fft_length, dtype=np.int64)
    assert valid_stops is not None
    if (
        valid_starts.shape != source_starts.shape
        or valid_stops.shape != source_starts.shape
        or np.any(valid_starts < 0)
        or np.any(valid_stops > fft_length)
        or np.any(valid_stops < valid_starts)
    ):
        raise ValueError("source validity range lies outside an FFT")
    active = valid_stops > valid_starts
    if _fft_eligibility is not None:
        eligible = np.asarray(_fft_eligibility, dtype=bool)
        if eligible.shape != source_starts.shape or not np.any(eligible):
            raise ValueError(
                "FFT eligibility must select active station rows"
            )
        # Rejected rows are zero FFT inputs, never source reads. Their
        # offsets may be outside the uploaded extent and must not widen it.
        valid_stops = np.where(eligible, valid_stops, valid_starts)
        active &= eligible
    replay_starts = source_starts + source_offsets
    split_replay = (
        (source_offsets != 0) & (source_cutovers != 0)
        if has_replay
        else np.zeros(source_starts.size, dtype=bool)
    )
    # Empty validity ranges produce zero FFT inputs without reading source
    # samples, even when their FFT rows remain admitted for reported exposure.
    active_split_replay = split_replay & active
    source_first, source_stop = _check_source_coverage(
        data,
        np.concatenate(
            (replay_starts[active], source_starts[active_split_replay])
        )
        if np.any(active_split_replay)
        else replay_starts[active],
        fft_length,
    )
    if data.payload_progress is not None and source_stop > source_first:
        if data.payload_frame_validity is None:
            raise RuntimeError(
                "packed VDIF upload progress requires frame validity"
            )
        with nvtx_range("xbeam:vdif-upload-wait"):
            data.payload_progress.wait_samples(
                source_first,
                source_stop,
                data.packed_samples_per_frame,
                data.payload_frame_validity,
                xp,
            )
    # Inverted complex LSB samples are conjugated during channelization;
    # their entire front-end phase must therefore use the opposite sign.
    fringe_sign = (
        -1.0
        if data.datastream.sampling is SamplingKind.COMPLEX
        and frequency.sideband is Sideband.LSB
        else 1.0
    )
    fringe_reference_hz = frequency.band_edge_hz
    if data.datastream.sampling is SamplingKind.COMPLEX_DSB:
        fringe_reference_hz += 0.5 * frequency.bandwidth_hz
    raw_fast_path = (
        xp.__name__ == "cupy" and job.fringe_approximation_order in {0, 1, 2}
    )
    if raw_fast_path:
        if data.device_samples is None and data.device_payload is None:
            data.device_samples = xp.ascontiguousarray(
                xp.asarray(data.samples)
            )
            data.device_validity = xp.ascontiguousarray(
                xp.asarray(data.validity, dtype=xp.uint8)
            )
        if job.fringe_approximation_order == 0:
            delay_zero = centre_delay
            delay_step = np.zeros_like(centre_delay)
            delay_quadratic = np.zeros_like(centre_delay)
        else:
            delay_zero = np.asarray(
                _delay_values(
                    job,
                    model,
                    data.datastream.station_id,
                    output_block_starts,
                    data.band,
                    np,
                )
            )
            right = np.asarray(
                _delay_values(
                    job,
                    model,
                    data.datastream.station_id,
                    output_block_starts
                    + (fft_length - 1) / data.sample_rate_hz,
                    data.band,
                    np,
                )
            )
            if job.fringe_approximation_order == 1:
                delay_step = (right - delay_zero) / (fft_length - 1)
                delay_quadratic = np.zeros_like(delay_zero)
            else:
                last_index = float(fft_length - 1)
                middle_index = last_index / 2.0
                middle = np.asarray(
                    _delay_values(
                        job,
                        model,
                        data.datastream.station_id,
                        output_block_starts
                        + middle_index / data.sample_rate_hz,
                        data.band,
                        np,
                    )
                )
                delay_quadratic = (
                    2.0
                    * (right + delay_zero - 2.0 * middle)
                    / (last_index * last_index)
                )
                delay_step = (
                    middle
                    - delay_zero
                    - delay_quadratic * middle_index * middle_index
                ) / middle_index
        block_parameters = prepare_block_parameters(
            output_indices=output_indices,
            coarse_samples=coarse_samples,
            block_starts=output_block_starts,
            delay_zero=delay_zero,
            delay_step=delay_step,
            delay_quadratic=delay_quadratic,
            sample_rate_hz=data.sample_rate_hz,
            centre_delay=centre_delay,
            source_timing_offsets=source_timing_offsets,
            source_offsets=source_offsets,
            source_cutovers=source_cutovers,
            valid_starts=valid_starts,
            valid_stops=valid_stops,
            xp=xp,
        )
        with nvtx_range("xbeam:front-rotate"):
            rotated, weight = front_rotate(
                source=(
                    data.device_payload
                    if data.device_payload is not None
                    else data.device_samples
                ),
                source_validity=data.device_validity,
                validity_stride=data.validity_stride,
                authoritative_validity=authoritative_validity,
                source_start_sample=data.start_sample,
                block_parameters=block_parameters,
                fft_length=fft_length,
                sample_rate_hz=data.sample_rate_hz,
                fringe_reference_hz=fringe_sign * fringe_reference_hz,
                lo_offset_hz=fringe_sign * data.band.lo_offset_hz,
                wall_lo_cycles=0.0,
                complex_input=(
                    data.datastream.sampling is not SamplingKind.REAL
                ),
                source_sample_count=data.source_sample_count,
                packed_samples_per_frame=data.packed_samples_per_frame,
                packed_payload_bytes=data.packed_payload_bytes,
                packed_channel_count=data.packed_channel_count,
                packed_channel_index=data.packed_channel_index,
                xp=xp,
                _validate_geometry=False,
                _fft_batch_rows=_fft_batch_rows,
            )
    else:
        indices = (
            output_indices[:, None]
            + np.arange(fft_length, dtype=np.int64)[None, :]
        )
        indices -= coarse_samples[:, None]
        if has_replay and np.any(source_offsets):
            indices += np.where(
                np.arange(fft_length, dtype=np.int64)[None, :]
                >= source_cutovers[:, None],
                source_offsets[:, None],
                0,
            )
        source_indices = indices - data.start_sample
        in_range = (source_indices >= 0) & (
            source_indices < data.samples.size
        )
        clipped = np.clip(source_indices, 0, max(0, data.samples.size - 1))
        host_samples = data.samples[clipped]
        host_validity = (
            in_range.copy()
            if authoritative_validity
            else data.validity[clipped] & in_range
        )
        sample_indices = np.arange(fft_length, dtype=np.int64)[None, :]
        host_validity &= sample_indices >= valid_starts[:, None]
        host_validity &= sample_indices < valid_stops[:, None]
        host_samples = np.where(host_validity, host_samples, 0)

        samples = xp.asarray(host_samples)
        validity = xp.asarray(host_validity)
        starts = xp.asarray(output_block_starts, dtype=xp.float64)
        delays = _approximated_delay(
            job,
            model,
            data.datastream.station_id,
            data.band,
            starts,
            fft_length,
            data.sample_rate_hz,
            job.fringe_approximation_order,
            xp,
        )
        sample_offsets = xp.arange(fft_length, dtype=xp.float64) / xp.float64(
            data.sample_rate_hz
        )
        times = starts[:, None] + sample_offsets[None, :]
        cycles = -xp.float64(fringe_reference_hz) * delays
        cycles += xp.float64(data.band.lo_offset_hz) * times
        cycles *= xp.float64(fringe_sign)
        phasor = xp.exp(
            xp.asarray(1j, dtype=xp.complex128)
            * xp.float64(TWO_PI)
            * reduce_cycles(cycles, xp)
        )
        rotated = samples * phasor
        weight = xp.mean(validity, axis=-1, dtype=xp.float64)
    credits = None
    if weight_credit_intervals:
        credits = _timestamp_gap_weight_credit(
            output_indices - coarse_samples,
            source_offsets,
            source_cutovers,
            fft_length=fft_length,
            intervals=weight_credit_intervals,
        )
    reported_credits = None
    if reported_weight_credit_intervals is not None:
        reported_credits = _timestamp_gap_weight_credit(
            output_indices - coarse_samples,
            source_offsets,
            source_cutovers,
            fft_length=fft_length,
            intervals=reported_weight_credit_intervals,
        )
    weight, reported_weight = _apply_station_weight_credits(
        weight, credits, reported_credits, xp
    )
    if xp.__name__ == "cupy":
        # The front-end buffer is dead after channelization.  A C2C transform
        # may therefore reuse it, avoiding a second full-size complex array
        # and allowing substantially larger cuFFT batches on 8 GB devices.
        from cupyx.scipy.fft import fft as cupy_fft

        with nvtx_range("xbeam:cufft"):
            transformed = cupy_fft(rotated, axis=-1, overwrite_x=True)
        transformed = transformed[: output_block_starts.size]
    else:
        transformed = xp.fft.fft(rotated, axis=-1)
    if xp.__name__ == "cupy":
        channel_spacing_hz = data.sample_rate_hz / fft_length
        if data.datastream.sampling is SamplingKind.REAL:
            mapping = (
                "identity"
                if frequency.sideband is Sideband.USB
                else "real_lsb"
            )
            baseband_bin_offset = (
                0
                if frequency.sideband is Sideband.USB
                else -frequency.channel_count
            )
        elif data.datastream.sampling is SamplingKind.COMPLEX:
            mapping = (
                "identity"
                if frequency.sideband is Sideband.USB
                else "complex_lsb"
            )
            baseband_bin_offset = (
                0
                if frequency.sideband is Sideband.USB
                else -frequency.channel_count
            )
        else:
            mapping = "complex_dsb"
            baseband_bin_offset = -(fft_length // 2)
        with nvtx_range("xbeam:channelize-residual"):
            spectrum = channelize_and_apply_residual(
                transformed,
                block_parameters,
                mapping=mapping,
                channel_count=frequency.channel_count,
                baseband_bin_offset=baseband_bin_offset,
                channel_spacing_hz=channel_spacing_hz,
                xp=xp,
            )
        baseband = np.arange(
            baseband_bin_offset,
            baseband_bin_offset + frequency.channel_count,
            dtype=np.float64,
        )
        baseband *= channel_spacing_hz
        return _StationBlocks(spectrum, weight, baseband, reported_weight)

    if data.datastream.sampling is SamplingKind.REAL:
        if frequency.sideband is Sideband.USB:
            spectrum = transformed[..., : frequency.channel_count]
            baseband = xp.arange(frequency.channel_count, dtype=xp.float64)
        else:
            spectrum = transformed[..., -frequency.channel_count :]
            baseband = -xp.arange(
                frequency.channel_count, 0, -1, dtype=xp.float64
            )
        baseband *= xp.float64(data.sample_rate_hz / fft_length)
    elif data.datastream.sampling is SamplingKind.COMPLEX:
        if frequency.sideband is Sideband.USB:
            spectrum = transformed
            baseband = xp.arange(frequency.channel_count, dtype=xp.float64)
            baseband *= xp.float64(data.sample_rate_hz / fft_length)
        else:
            spectrum, baseband = complex_lsb_channels(
                transformed, data.sample_rate_hz, xp
            )
    else:
        baseband = xp.fft.fftfreq(fft_length, d=1.0 / data.sample_rate_hz)
        ordering = xp.argsort(baseband)
        baseband = baseband[ordering]
        spectrum = transformed[..., ordering]

    # ``indices`` selected t - coarse_delay.  The remaining error used by the
    # +i2πfε convention is therefore coarse_delay - requested_delay.
    residual = (
        coarse_samples
        - (0 if source_timing_offsets is None else source_timing_offsets)
    ) / data.sample_rate_hz - centre_delay
    phase = xp.float64(TWO_PI) * reduce_cycles(
        residual[:, None] * baseband[None, :], xp
    )
    spectrum *= xp.exp(xp.asarray(1j, dtype=xp.complex128) * phase)
    return _StationBlocks(spectrum, weight, baseband, reported_weight)


def _packed_real_group_compatible(
    left: _BandSamples, right: _BandSamples
) -> bool:
    """Return whether two packed bands can share one front/FFT launch."""

    return (
        left.datastream.identifier == right.datastream.identifier
        and left.datastream.sampling is SamplingKind.REAL
        and right.datastream.sampling is SamplingKind.REAL
        and left.payload_plan is not None
        and left.payload_plan is right.payload_plan
        and left.device_payload is right.device_payload
        and left.device_validity is right.device_validity
        and left.packed_channel_count > 1
        and left.packed_channel_count == right.packed_channel_count
        and left.packed_samples_per_frame == right.packed_samples_per_frame
        and left.packed_payload_bytes == right.packed_payload_bytes
        and left.source_sample_count == right.source_sample_count
        and left.start_sample == right.start_sample
        and left.validity_stride == right.validity_stride
        and left.sample_rate_hz == right.sample_rate_hz
        and left.frequency.channel_count == right.frequency.channel_count
        and left.frequency.sideband is right.frequency.sideband
    )


def _packed_real_block_parameters(
    job: Job,
    data: _BandSamples,
    model: DelayCoordinateModel,
    output_block_starts: np.ndarray,
    fft_length: int,
    centre_delay: np.ndarray,
    coarse_samples: np.ndarray,
    output_indices: np.ndarray,
    xp: Any,
    eligibility: np.ndarray | None = None,
    source_offsets: np.ndarray | None = None,
) -> Any:
    if job.fringe_approximation_order == 0:
        delay_zero = centre_delay
        delay_step = np.zeros_like(centre_delay)
        delay_quadratic = np.zeros_like(centre_delay)
    else:
        delay_zero = np.asarray(
            _delay_values(
                job,
                model,
                data.datastream.station_id,
                output_block_starts,
                data.band,
                np,
            )
        )
        right = np.asarray(
            _delay_values(
                job,
                model,
                data.datastream.station_id,
                output_block_starts + (fft_length - 1) / data.sample_rate_hz,
                data.band,
                np,
            )
        )
        if job.fringe_approximation_order == 1:
            delay_step = (right - delay_zero) / (fft_length - 1)
            delay_quadratic = np.zeros_like(delay_zero)
        else:
            last_index = float(fft_length - 1)
            middle_index = last_index / 2.0
            middle = np.asarray(
                _delay_values(
                    job,
                    model,
                    data.datastream.station_id,
                    output_block_starts + middle_index / data.sample_rate_hz,
                    data.band,
                    np,
                )
            )
            delay_quadratic = (
                2.0
                * (right + delay_zero - 2.0 * middle)
                / (last_index * last_index)
            )
            delay_step = (
                middle
                - delay_zero
                - delay_quadratic * middle_index * middle_index
            ) / middle_index
    return prepare_block_parameters(
        output_indices=output_indices,
        coarse_samples=coarse_samples,
        block_starts=output_block_starts,
        delay_zero=delay_zero,
        delay_step=delay_step,
        delay_quadratic=delay_quadratic,
        sample_rate_hz=data.sample_rate_hz,
        centre_delay=centre_delay,
        source_offsets=source_offsets,
        source_cutovers=(
            None if source_offsets is None else np.zeros_like(source_offsets)
        ),
        valid_starts=(
            None
            if eligibility is None
            else np.zeros(eligibility.size, dtype=np.int64)
        ),
        valid_stops=(
            None
            if eligibility is None
            else np.where(eligibility, fft_length, 0)
        ),
        xp=xp,
    )


def _station_blocks_packed_real_group(
    job: Job,
    group: Sequence[tuple[tuple[int, int], _BandSamples]],
    model: DelayCoordinateModel,
    output_block_starts: np.ndarray,
    xp: Any,
    eligibility_by_band: Mapping[tuple[int, int], np.ndarray] | None = None,
    source_offsets_by_band: Mapping[tuple[int, int], np.ndarray]
    | None = None,
) -> dict[tuple[int, int], _StationBlocks]:
    """Process compatible packed real bands in one front and cuFFT batch."""

    first = group[0][1]
    if not all(
        _packed_real_group_compatible(first, data) for _, data in group
    ):
        raise ValueError("packed real station group is not compatible")
    if first.source_sample_count is None:
        raise ValueError("packed real inputs require a source sample count")
    per_band_indices = []
    wait_start: int | None = None
    wait_stop: int | None = None
    for key, data in group:
        values = _source_block_indices(
            job,
            data.datastream,
            data.band,
            data.frequency,
            model,
            output_block_starts,
        )
        fft_length, centre_delay, coarse_samples, output_indices = values
        source_starts = output_indices - coarse_samples - data.start_sample
        source_offsets = (
            None
            if source_offsets_by_band is None
            else source_offsets_by_band.get(key)
        )
        if source_offsets is not None:
            source_starts = source_starts + source_offsets
        eligibility = (
            None
            if eligibility_by_band is None
            else eligibility_by_band.get(key)
        )
        current_start, current_stop = _check_source_coverage(
            data,
            source_starts
            if eligibility is None
            else source_starts[eligibility],
            fft_length,
        )
        wait_start = (
            current_start
            if wait_start is None
            else min(wait_start, current_start)
        )
        wait_stop = (
            current_stop
            if wait_stop is None
            else max(wait_stop, current_stop)
        )
        per_band_indices.append((data, values, eligibility, source_offsets))
    fft_length = per_band_indices[0][1][0]
    if any(values[0] != fft_length for _, values, _, _ in per_band_indices):
        raise ValueError("packed real station group FFT lengths do not match")
    per_band = []
    for data, values, eligibility, source_offsets in per_band_indices:
        fft_length, centre_delay, coarse_samples, output_indices = values
        parameters = _packed_real_block_parameters(
            job,
            data,
            model,
            output_block_starts,
            fft_length,
            centre_delay,
            coarse_samples,
            output_indices,
            xp,
            eligibility,
            source_offsets,
        )
        per_band.append((data, fft_length, parameters))
    assert wait_start is not None and wait_stop is not None
    if first.payload_progress is not None and wait_stop > wait_start:
        if first.payload_frame_validity is None:
            raise RuntimeError(
                "packed VDIF upload progress requires validity"
            )
        with nvtx_range("xbeam:vdif-upload-wait"):
            first.payload_progress.wait_samples(
                int(wait_start),
                int(wait_stop),
                first.packed_samples_per_frame,
                first.payload_frame_validity,
                xp,
            )
    block_parameters = xp.concatenate([value[2] for value in per_band])
    fringe_references = []
    for data, _length, _parameters in per_band:
        fringe_references.append(data.frequency.band_edge_hz)
    with nvtx_range("xbeam:front-rotate-packed-bands"):
        rotated, weights = front_rotate_packed_real_bands(
            source=first.device_payload,
            source_validity=first.device_validity,
            validity_stride=first.validity_stride,
            source_sample_count=int(first.source_sample_count),
            block_parameters=block_parameters,
            source_start_samples=[data.start_sample for data, *_ in per_band],
            fringe_reference_hz=fringe_references,
            lo_offset_hz=[data.band.lo_offset_hz for data, *_ in per_band],
            wall_lo_cycles=[0.0] * len(per_band),
            packed_channel_indices=[
                data.packed_channel_index for data, *_ in per_band
            ],
            fft_length=fft_length,
            sample_rate_hz=first.sample_rate_hz,
            packed_samples_per_frame=first.packed_samples_per_frame,
            packed_payload_bytes=first.packed_payload_bytes,
            packed_channel_count=first.packed_channel_count,
            xp=xp,
        )
    from cupyx.scipy.fft import fft as cupy_fft

    with nvtx_range("xbeam:cufft-packed-bands"):
        transformed = cupy_fft(rotated, axis=-1, overwrite_x=True)
    band_count = len(group)
    block_count = output_block_starts.size
    flattened = transformed.reshape(band_count * block_count, fft_length)
    channel_spacing_hz = first.sample_rate_hz / fft_length
    mapping = (
        "identity" if first.frequency.sideband is Sideband.USB else "real_lsb"
    )
    baseband_bin_offset = (
        0
        if first.frequency.sideband is Sideband.USB
        else -first.frequency.channel_count
    )
    with nvtx_range("xbeam:channelize-residual-packed-bands"):
        spectra = channelize_and_apply_residual(
            flattened,
            block_parameters,
            mapping=mapping,
            channel_count=first.frequency.channel_count,
            baseband_bin_offset=baseband_bin_offset,
            channel_spacing_hz=channel_spacing_hz,
            xp=xp,
        ).reshape(band_count, block_count, first.frequency.channel_count)
    baseband = np.arange(
        baseband_bin_offset,
        baseband_bin_offset + first.frequency.channel_count,
        dtype=np.float64,
    )
    baseband *= channel_spacing_hz
    return {
        key: _StationBlocks(spectra[index], weights[index], baseband)
        for index, (key, _data) in enumerate(group)
    }


def _station_blocks_packed_real_group_for_scan(
    job: Job,
    group: Sequence[tuple[tuple[int, int], _BandSamples]],
    scan: Any,
    starts: np.ndarray,
    xp: Any,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
) -> dict[tuple[int, int], _StationBlocks]:
    first = group[0][1]
    centres = starts + (_fft_length(first) - 1) / (2.0 * first.sample_rate_hz)
    runs = _ordered_model_runs(scan, first.datastream.station_id, centres)

    def plan_for(key):
        return (
            None
            if recovery_plans is None
            else recovery_plans.get((scan.identifier, *key))
        )

    def single(key, data, model, selected_starts):
        if recovery_plans is not None:
            return _station_blocks_for_scan(
                job, data, scan, selected_starts, xp, plan_for(key)
            )
        return _station_blocks(job, data, model, selected_starts, xp)

    def process(
        current: Sequence[tuple[tuple[int, int], _BandSamples]],
        model: DelayCoordinateModel,
        selected_starts: np.ndarray,
    ) -> dict[tuple[int, int], _StationBlocks]:
        if len(current) == 1:
            key, data = current[0]
            return {key: single(key, data, model, selected_starts)}
        eligibility = {}
        source_offsets = {}
        for key, data in current:
            plan = plan_for(key)
            if plan is not None and plan.source_shifted_fft_count:
                offsets, cutovers = select_fft_source_replay(
                    plan,
                    scan_origin_s=max(0.0, float(scan.start_offset_s)),
                    fft_seconds=_fft_length(data) / data.sample_rate_hz,
                    block_starts_s=tuple(
                        float(value) for value in selected_starts
                    ),
                    fft_samples=_fft_length(data),
                )
                if any(cutovers):
                    raise ValueError(
                        "packed replay requires whole-FFT offsets"
                    )
                source_offsets[key] = np.asarray(offsets, dtype=np.int64)
            if plan is not None and plan.ineligible_fft_count:
                eligibility[key] = np.asarray(
                    select_fft_eligibility(
                        plan,
                        scan_origin_s=max(0.0, float(scan.start_offset_s)),
                        fft_seconds=_fft_length(data) / data.sample_rate_hz,
                        block_starts_s=tuple(
                            float(value) for value in selected_starts
                        ),
                    ),
                    dtype=bool,
                )
        try:
            if not eligibility and not source_offsets:
                return _station_blocks_packed_real_group(
                    job, current, model, selected_starts, xp
                )
            return _station_blocks_packed_real_group(
                job,
                current,
                model,
                selected_starts,
                xp,
                eligibility,
                source_offsets,
            )
        except Exception as exc:
            is_oom = xp.__name__ == "cupy" and isinstance(
                exc, xp.cuda.memory.OutOfMemoryError
            )
            if not is_oom:
                raise
        # Leave the handler before retrying so the traceback releases any
        # front/FFT buffers allocated before the failed allocation.
        xp.get_default_memory_pool().free_all_blocks()
        if len(current) == 2:
            with nvtx_range("xbeam:packed-band-oom-single-fallback"):
                return {
                    key: single(key, data, model, selected_starts)
                    for key, data in current
                }
        midpoint = len(current) // 2
        with nvtx_range("xbeam:packed-band-oom-split"):
            result = process(current[:midpoint], model, selected_starts)
            result.update(process(current[midpoint:], model, selected_starts))
            return result

    pieces = [
        process(group, model, starts[selected]) for model, selected in runs
    ]
    if len(pieces) == 1:
        return pieces[0]
    return {
        key: _StationBlocks(
            xp.concatenate([piece[key].spectrum for piece in pieces]),
            xp.concatenate([piece[key].weight for piece in pieces]),
            pieces[0][key].baseband_hz,
        )
        for key, _data in group
    }


def _packed_group_requires_source_replay(
    group: Sequence[tuple[tuple[int, int], _BandSamples]],
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan] | None,
    scan: Any,
    starts: np.ndarray,
) -> bool:
    """Whether packed processing would miss a replay in this window."""

    if recovery_plans is None:
        return False
    block_starts = tuple(float(value) for value in starts)
    scan_origin_s = max(0.0, float(scan.start_offset_s))
    for key, data in group:
        plan = recovery_plans.get((scan.identifier, key[0], key[1]))
        if plan is None or (
            plan.source_shifted_fft_count == 0
            and plan.source_masked_fft_count == 0
            and not plan.has_source_segments
            and not plan.has_source_weight_credit
            and not plan.has_source_weight_exclusion
            and not plan.has_source_weight_override
            and not plan.has_authoritative_source_validity
            and not plan.source_authoritative_validity_events
            and not plan.source_timing_events
        ):
            continue
        if plan.source_timing_events and np.any(
            select_fft_source_timing(
                plan,
                scan_origin_s=scan_origin_s,
                fft_seconds=_fft_length(data) / data.sample_rate_hz,
                block_starts_s=block_starts,
            )
        ):
            return True
        if plan.has_source_segments:
            segments = select_fft_source_segments(
                plan,
                scan_origin_s=scan_origin_s,
                fft_seconds=_fft_length(data) / data.sample_rate_hz,
                block_starts_s=block_starts,
                fft_samples=_fft_length(data),
            )
            if any(item is not None for item in segments):
                return True
        if plan.has_source_weight_credit:
            return True
        if plan.has_source_weight_exclusion:
            return True
        if plan.has_source_weight_override:
            return True
        if plan.has_authoritative_source_validity or (
            plan.source_authoritative_validity_events
            and np.any(
                select_fft_authoritative_validity(
                    plan,
                    scan_origin_s=scan_origin_s,
                    fft_seconds=_fft_length(data) / data.sample_rate_hz,
                    block_starts_s=block_starts,
                )
            )
        ):
            return True
        _offsets, cutovers = select_fft_source_replay(
            plan,
            scan_origin_s=scan_origin_s,
            fft_seconds=_fft_length(data) / data.sample_rate_hz,
            block_starts_s=block_starts,
            fft_samples=_fft_length(data),
        )
        if any(cutovers):
            return True
        valid_starts, valid_stops = select_fft_source_validity(
            plan,
            scan_origin_s=scan_origin_s,
            fft_seconds=_fft_length(data) / data.sample_rate_hz,
            block_starts_s=block_starts,
            fft_samples=_fft_length(data),
        )
        if any(
            start != 0 or stop != _fft_length(data)
            for start, stop in zip(valid_starts, valid_stops, strict=True)
        ):
            return True
    return False


def _baseband_frequencies(data: _BandSamples) -> np.ndarray:
    """Return the channel-frequency axis without materializing an FFT."""

    fft_length = _fft_length(data)
    channel_count = data.frequency.channel_count
    spacing = data.sample_rate_hz / fft_length
    if data.datastream.sampling in {SamplingKind.REAL, SamplingKind.COMPLEX}:
        if data.frequency.sideband is Sideband.USB:
            bins = np.arange(channel_count, dtype=np.float64)
        else:
            bins = -np.arange(channel_count, 0, -1, dtype=np.float64)
        return bins * spacing
    baseband = np.fft.fftfreq(fft_length, d=1.0 / data.sample_rate_hz)
    return np.sort(baseband)


def _station_blocks_for_scan(
    job: Job,
    data: _BandSamples,
    scan: Any,
    starts: np.ndarray,
    xp: Any,
    replay_plan: GlobalFFTEligibilityPlan | None = None,
) -> _StationBlocks:
    if not starts.size:
        raise ValueError("station block selection must not be empty")
    centres = starts + (_fft_length(data) - 1) / (2.0 * data.sample_rate_hz)
    runs = _ordered_model_runs(scan, data.datastream.station_id, centres)
    if replay_plan is None and len(runs) == 1:
        model, selected = runs[0]
        return _station_blocks(job, data, model, starts[selected], xp)
    if replay_plan is None:
        eligibility = np.ones(starts.size, dtype=bool)
        offsets = np.zeros(starts.size, dtype=np.int64)
        cutovers = np.zeros(starts.size, dtype=np.int64)
        timing_offsets = None
        valid_starts = np.zeros(starts.size, dtype=np.int64)
        valid_stops = np.full(starts.size, _fft_length(data), dtype=np.int64)
        source_segments: tuple[
            tuple[tuple[int, int, int], ...] | None, ...
        ] = (None,) * starts.size
        weight_exclusions = np.zeros(starts.size, dtype=np.int64)
        authoritative = np.zeros(starts.size, dtype=bool)
    else:
        block_starts = tuple(float(value) for value in starts)
        eligibility = np.asarray(
            select_fft_eligibility(
                replay_plan,
                scan_origin_s=max(0.0, float(scan.start_offset_s)),
                fft_seconds=_fft_length(data) / data.sample_rate_hz,
                block_starts_s=block_starts,
            ),
            dtype=bool,
        )
        timing_offsets = (
            select_fft_source_timing(
                replay_plan,
                scan_origin_s=max(0.0, float(scan.start_offset_s)),
                fft_seconds=_fft_length(data) / data.sample_rate_hz,
                block_starts_s=block_starts,
            )
            if replay_plan.source_timing_events
            else None
        )
        selected_offsets, selected_cutovers = select_fft_source_replay(
            replay_plan,
            scan_origin_s=max(0.0, float(scan.start_offset_s)),
            fft_seconds=_fft_length(data) / data.sample_rate_hz,
            block_starts_s=block_starts,
            fft_samples=_fft_length(data),
        )
        offsets = np.asarray(selected_offsets, dtype=np.int64)
        cutovers = np.asarray(selected_cutovers, dtype=np.int64)
        if replay_plan.source_masked_fft_count:
            selected_starts, selected_stops = select_fft_source_validity(
                replay_plan,
                scan_origin_s=max(0.0, float(scan.start_offset_s)),
                fft_seconds=_fft_length(data) / data.sample_rate_hz,
                block_starts_s=block_starts,
                fft_samples=_fft_length(data),
            )
            valid_starts = np.asarray(selected_starts, dtype=np.int64)
            valid_stops = np.asarray(selected_stops, dtype=np.int64)
        else:
            valid_starts = np.zeros(starts.size, dtype=np.int64)
            valid_stops = np.full(
                starts.size, _fft_length(data), dtype=np.int64
            )
        source_segments = select_fft_source_segments(
            replay_plan,
            scan_origin_s=max(0.0, float(scan.start_offset_s)),
            fft_seconds=_fft_length(data) / data.sample_rate_hz,
            block_starts_s=block_starts,
            fft_samples=_fft_length(data),
        )
        weight_exclusions = np.asarray(
            select_fft_weight_exclusions(
                replay_plan,
                scan_origin_s=max(0.0, float(scan.start_offset_s)),
                fft_seconds=_fft_length(data) / data.sample_rate_hz,
                block_starts_s=block_starts,
                fft_samples=_fft_length(data),
            ),
            dtype=np.int64,
        )
        scan_origin = max(0.0, float(scan.start_offset_s))
        fft_seconds = _fft_length(data) / data.sample_rate_hz
        authoritative = np.asarray(
            select_fft_authoritative_validity(
                replay_plan,
                scan_origin_s=scan_origin,
                fft_seconds=fft_seconds,
                block_starts_s=block_starts,
            ),
            dtype=bool,
        )

    execute: (
        Callable[
            [DelayCoordinateModel, slice | np.ndarray, int | None],
            _StationBlocks,
        ]
        | None
    ) = None

    def execute_piece(
        model: DelayCoordinateModel,
        selected: slice | np.ndarray,
        enclosing_rows: int | None = None,
    ) -> _StationBlocks:
        selected_indices = np.arange(starts.size, dtype=np.int64)[selected]
        if enclosing_rows is None:
            enclosing_rows = selected_indices.size
        selected_eligibility = eligibility[selected]
        all_selected = np.all(selected_eligibility)
        dense_rows = (
            not all_selected
            and np.any(selected_eligibility)
            and _use_dense_compat_fft_rows(
                xp,
                job.fringe_approximation_order,
                authoritative[selected],
                tuple(source_segments[int(i)] for i in selected_indices),
            )
            if xp.__name__ == "cupy"
            else False
        )
        if not all_selected and not dense_rows:
            spectrum = xp.zeros(
                (selected_indices.size, data.frequency.channel_count),
                dtype=(
                    np.complex64 if xp.__name__ == "cupy" else np.complex128
                ),
            )
            weight = xp.zeros(selected_indices.size, dtype=np.float64)
            reported_weight = None
            if np.any(selected_eligibility):
                eligible_indices = selected_indices[selected_eligibility]
                assert execute is not None
                piece = execute(model, eligible_indices, enclosing_rows)
                spectrum[selected_eligibility] = piece.spectrum
                weight[selected_eligibility] = piece.weight
                if piece.reported_weight is not None:
                    reported_weight = xp.zeros(
                        selected_indices.size, dtype=np.float64
                    )
                    reported_weight[selected_eligibility] = (
                        piece.reported_weight
                    )
                baseband = piece.baseband_hz
            else:
                baseband = _baseband_frequencies(data)
            return _StationBlocks(spectrum, weight, baseband, reported_weight)

        selected_starts = starts[selected]
        selected_offsets = offsets[selected]
        selected_cutovers = cutovers[selected]
        selected_valid_starts = valid_starts[selected]
        selected_valid_stops = valid_stops[selected]
        selected_authoritative = authoritative[selected]
        if isinstance(selected, slice):
            first = int(selected.start or 0)
            stop = int(
                selected.stop if selected.stop is not None else starts.size
            )
            selected_segments = source_segments[first:stop]
        else:
            selected_segments = tuple(
                source_segments[int(i)] for i in selected
            )

        def run_piece(mask: np.ndarray, is_authoritative: bool):
            options: _StationBlockOptions = {}
            if xp.__name__ == "cupy" and replay_plan is not None:
                options = {
                    "_fft_batch_rows": _compat_fft_batch_rows(
                        int(np.count_nonzero(mask)), enclosing_rows
                    ),
                    "_fft_eligibility": (
                        None
                        if np.all(selected_eligibility[mask])
                        else selected_eligibility[mask]
                    ),
                }
            return _station_blocks(
                job,
                data,
                model,
                selected_starts[mask],
                xp,
                selected_offsets[mask],
                selected_cutovers[mask],
                selected_valid_starts[mask],
                selected_valid_stops[mask],
                (
                    ()
                    if replay_plan is None
                    else replay_plan.source_weight_credit_intervals
                ),
                (
                    None
                    if replay_plan is None
                    else (replay_plan.source_reported_weight_credit_intervals)
                ),
                is_authoritative,
                source_timing_offsets=(
                    None
                    if timing_offsets is None
                    else timing_offsets[selected][mask]
                ),
                **options,
            )

        if not any(item is not None for item in selected_segments):
            if np.all(selected_authoritative == selected_authoritative[0]):
                result = run_piece(
                    np.ones(selected_starts.size, dtype=bool),
                    bool(selected_authoritative[0]),
                )
            else:
                spectrum = xp.empty(
                    (selected_starts.size, data.frequency.channel_count),
                    dtype=(
                        np.complex64
                        if xp.__name__ == "cupy"
                        else np.complex128
                    ),
                )
                weight = xp.empty(selected_starts.size, dtype=np.float64)
                reported_weight = None
                baseband = None
                for is_authoritative in (False, True):
                    mask = selected_authoritative == is_authoritative
                    if not np.any(mask):
                        continue
                    piece = run_piece(mask, is_authoritative)
                    spectrum[mask] = piece.spectrum
                    weight[mask] = piece.weight
                    if piece.reported_weight is not None:
                        if reported_weight is None:
                            reported_weight = weight.copy()
                    if reported_weight is not None:
                        reported_weight[mask] = (
                            piece.weight
                            if piece.reported_weight is None
                            else piece.reported_weight
                        )
                    baseband = piece.baseband_hz
                assert baseband is not None
                result = _StationBlocks(
                    spectrum, weight, baseband, reported_weight
                )
            selected_exclusions = weight_exclusions[selected]
            if np.any(selected_exclusions):
                result.weight -= xp.asarray(
                    selected_exclusions, dtype=xp.float64
                ) / xp.float64(_fft_length(data))
                result.weight = xp.maximum(result.weight, xp.float64(0.0))
                if result.reported_weight is not None:
                    result.reported_weight -= xp.asarray(
                        selected_exclusions, dtype=xp.float64
                    ) / xp.float64(_fft_length(data))
                    result.reported_weight = xp.maximum(
                        result.reported_weight, xp.float64(0.0)
                    )
            return result

        spectrum = xp.empty(
            (selected_starts.size, data.frequency.channel_count),
            dtype=np.complex64 if xp.__name__ == "cupy" else np.complex128,
        )
        weight = xp.empty(selected_starts.size, dtype=np.float64)
        reported_weight = (
            xp.empty(selected_starts.size, dtype=np.float64)
            if replay_plan is not None
            and replay_plan.has_separate_reported_weight_credit
            else None
        )
        ordinary = np.asarray(
            [item is None for item in selected_segments], dtype=bool
        )
        baseband = None
        for is_authoritative in (False, True):
            mask = ordinary & (selected_authoritative == is_authoritative)
            if not np.any(mask):
                continue
            piece = run_piece(mask, is_authoritative)
            spectrum[mask] = piece.spectrum
            weight[mask] = piece.weight
            if reported_weight is not None:
                reported_weight[mask] = (
                    piece.weight
                    if piece.reported_weight is None
                    else piece.reported_weight
                )
            baseband = piece.baseband_hz

        segment_counts = np.fromiter(
            (
                0 if parts is None else len(parts)
                for parts in selected_segments
            ),
            dtype=np.int64,
            count=len(selected_segments),
        )
        if any(parts == () for parts in selected_segments):
            raise RuntimeError("source segment event contains no segments")
        # FFTs are independent. Batch the same segment ordinal across rows,
        # retaining each row's segment addition order and validity mode.
        for ordinal in range(int(segment_counts.max(initial=0))):
            present = segment_counts > ordinal
            for is_authoritative in (False, True):
                indices = np.flatnonzero(
                    present & (selected_authoritative == is_authoritative)
                )
                if not indices.size:
                    continue
                selected_pieces = []
                for index in indices:
                    row_segments = selected_segments[int(index)]
                    assert row_segments is not None
                    selected_pieces.append(row_segments[ordinal])
                pieces = np.asarray(selected_pieces, dtype=np.int64)
                piece = _station_blocks(
                    job,
                    data,
                    model,
                    selected_starts[indices],
                    xp,
                    pieces[:, 2],
                    np.zeros(indices.size, dtype=np.int64),
                    pieces[:, 0],
                    pieces[:, 1],
                    (),
                    None,
                    is_authoritative,
                    source_timing_offsets=(
                        None
                        if timing_offsets is None
                        else timing_offsets[selected][indices]
                    ),
                )
                if ordinal == 0:
                    spectrum[indices] = piece.spectrum
                    weight[indices] = piece.weight
                else:
                    spectrum[indices] += piece.spectrum
                    weight[indices] += piece.weight
                if baseband is None:
                    baseband = piece.baseband_hz
        if reported_weight is not None:
            segmented = segment_counts > 0
            reported_weight[segmented] = weight[segmented]
        assert baseband is not None
        selected_exclusions = weight_exclusions[selected]
        if np.any(selected_exclusions):
            weight -= xp.asarray(
                selected_exclusions, dtype=xp.float64
            ) / xp.float64(_fft_length(data))
            weight = xp.maximum(weight, xp.float64(0.0))
            if reported_weight is not None:
                reported_weight -= xp.asarray(
                    selected_exclusions, dtype=xp.float64
                ) / xp.float64(_fft_length(data))
                reported_weight = xp.maximum(reported_weight, xp.float64(0.0))
        return _StationBlocks(spectrum, weight, baseband, reported_weight)

    execute = execute_piece
    try:
        if len(runs) == 1:
            model, selected = runs[0]
            return execute(model, selected, None)
        # Keep only the final arrays and one bounded model-run batch live.  A
        # boundary inside one timing group can then recover without relying on
        # the outer whole-group OOM splitter.
        spectrum = xp.empty(
            (starts.size, data.frequency.channel_count),
            dtype=np.complex64 if xp.__name__ == "cupy" else np.complex128,
        )
        weight = xp.empty(starts.size, dtype=np.float64)
        reported_weight = (
            xp.empty(starts.size, dtype=np.float64)
            if replay_plan is not None
            and replay_plan.has_separate_reported_weight_credit
            else None
        )
        baseband = None
        for model, selected in runs:
            cursor = int(selected.start or 0)
            stop = int(
                selected.stop if selected.stop is not None else starts.size
            )
            chunk_size = stop - cursor
            while cursor < stop:
                chunk_stop = min(stop, cursor + chunk_size)
                retry_piece = False
                try:
                    if replay_plan is None:
                        piece = _station_blocks(
                            job, data, model, starts[cursor:chunk_stop], xp
                        )
                    else:
                        piece = execute(
                            model, slice(cursor, chunk_stop), None
                        )
                except Exception as exc:
                    is_oom = xp.__name__ == "cupy" and isinstance(
                        exc, xp.cuda.memory.OutOfMemoryError
                    )
                    if not is_oom or chunk_size == 1:
                        raise
                    chunk_size = max(1, chunk_size // 2)
                    retry_piece = True
                if retry_piece:
                    xp.get_default_memory_pool().free_all_blocks()
                    continue
                spectrum[cursor:chunk_stop] = piece.spectrum
                weight[cursor:chunk_stop] = piece.weight
                if reported_weight is not None:
                    reported_weight[cursor:chunk_stop] = (
                        piece.weight
                        if piece.reported_weight is None
                        else piece.reported_weight
                    )
                if baseband is None:
                    baseband = piece.baseband_hz
                del piece
                cursor = chunk_stop
        assert baseband is not None
        return _StationBlocks(spectrum, weight, baseband, reported_weight)
    finally:
        # The recursive helper otherwise keeps its captured source buffers
        # alive until cyclic collection, even after this station work ends.
        execute = None


def _station_ready_to_attempt(
    job: Job,
    group: Sequence[tuple[tuple[int, int], _BandSamples]],
    scan: Any,
    starts: np.ndarray,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
) -> bool:
    """Return whether ordinary packed inputs have been published.

    Replay inputs use different coordinates and possibly segmented or rejected
    rows. Their execution path enforces the exact dependencies; an ordinary
    coordinate precheck can unnecessarily wait for data they never consume.
    """

    intervals: dict[int, tuple[Any, _BandSamples, int, int]] = {}
    for _key, data in group:
        plan = (
            recovery_plans.get((scan.identifier, *_key))
            if recovery_plans is not None
            else None
        )
        if plan is not None and (
            plan.source_offsets
            or plan.source_replay_events
            or plan.source_replay_intervals
            or plan.has_source_segments
            or plan.ineligible_intervals
            or not all(plan.eligible_ffts)
        ):
            continue
        progress = data.payload_progress
        if progress is None:
            continue
        if data.payload_frame_validity is None:
            raise RuntimeError(
                "packed VDIF upload progress requires validity"
            )
        first_sample, stop_sample = _station_payload_sample_bounds(
            job, data, scan, starts
        )
        identifier = id(progress)
        previous = intervals.get(identifier)
        if previous is None:
            intervals[identifier] = (
                progress,
                data,
                first_sample,
                stop_sample,
            )
        else:
            intervals[identifier] = (
                progress,
                previous[1],
                min(previous[2], first_sample),
                max(previous[3], stop_sample),
            )
    for progress, data, first_sample, stop_sample in intervals.values():
        assert data.payload_frame_validity is not None
        if not progress.samples_published(
            first_sample,
            stop_sample,
            data.packed_samples_per_frame,
            data.payload_frame_validity,
        ):
            return False
    return True


def _bounded_station_blocks(
    job: Job,
    data: _BandSamples,
    scan: Any,
    starts: np.ndarray,
    xp: Any,
    requested_blocks: int | None,
    replay_plan: GlobalFFTEligibilityPlan | None = None,
) -> _StationBlocks:
    fft_length = (
        data.frequency.channel_count * 2
        if data.datastream.sampling is SamplingKind.REAL
        else data.frequency.channel_count
    )
    if requested_blocks is None:
        if xp.__name__ == "cupy":
            free_bytes, _total_bytes = xp.cuda.runtime.memGetInfo()
            # The GPU path performs its C2C FFT in place. This estimate
            # covers
            # the input/output buffer, retained spectra, phase vectors, and
            # cuFFT workspace while leaving half of free memory to the driver.
            estimated_per_block = max(1, fft_length * 64)
            requested_blocks = max(
                1, int((free_bytes * 0.5) // estimated_per_block)
            )
        else:
            requested_blocks = starts.size
    chunk_size = max(1, min(int(requested_blocks), starts.size))
    pieces: list[_StationBlocks] = []
    cursor = 0
    while cursor < starts.size:
        stop = min(starts.size, cursor + chunk_size)
        try:
            piece = _station_blocks_for_scan(
                job,
                data,
                scan,
                starts[cursor:stop],
                xp,
                replay_plan,
            )
        except Exception as exc:
            is_oom = xp.__name__ == "cupy" and isinstance(
                exc, xp.cuda.memory.OutOfMemoryError
            )
            if not is_oom or chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            xp.get_default_memory_pool().free_all_blocks()
            continue
        pieces.append(piece)
        cursor = stop
    if len(pieces) == 1:
        return pieces[0]
    return _StationBlocks(
        xp.concatenate([piece.spectrum for piece in pieces]),
        xp.concatenate([piece.weight for piece in pieces]),
        pieces[0].baseband_hz,
        (
            xp.concatenate(
                [
                    piece.weight
                    if piece.reported_weight is None
                    else piece.reported_weight
                    for piece in pieces
                ]
            )
            if any(piece.reported_weight is not None for piece in pieces)
            else None
        ),
    )


def _parent_band_key(datastream: Datastream, band_id: int) -> tuple[int, int]:
    for band in datastream.recorded_bands:
        if band.identifier == band_id:
            return datastream.identifier, band_id
    for zoom in datastream.zoom_bands:
        if zoom.identifier == band_id:
            return datastream.identifier, zoom.parent_band_id
    raise ConfigurationError(
        f"datastream[{datastream.identifier}]",
        "band",
        f"unknown band {band_id}",
    )


def _parallel_auto_reduction(
    blocks: _StationBlocks,
    channels_to_average: int,
    mode: AutocorrelationMode,
    xp: Any,
) -> tuple[_StationBlocks, int]:
    """Select the configured channel reduction for a parallel-hand auto."""

    if mode is AutocorrelationMode.AVERAGE or channels_to_average == 1:
        return blocks, channels_to_average
    output_channels = blocks.spectrum.shape[1] // channels_to_average
    return (
        _StationBlocks(
            blocks.spectrum[..., :output_channels],
            blocks.weight,
            blocks.baseband_hz[:output_channels],
            blocks.reported_weight,
        ),
        1,
    )


def _automatic_xmac_fusion_width(xp: Any) -> int:
    """Use the wider fused-product kernel on Hopper and newer GPUs."""

    properties = xp.cuda.runtime.getDeviceProperties(int(xp.cuda.Device().id))
    major = int(properties.get("major", properties.get(b"major", 0)))
    return 8 if major >= 9 else 4


@dataclass(frozen=True, slots=True)
class _VisibilityProductSpec:
    """Pointer-free product description shared by an execution's GPU tiles."""

    reservation: tuple[Any, ...]
    band_a: tuple[int, int]
    band_b: tuple[int, int]
    boundary_a: tuple[int, int]
    boundary_b: tuple[int, int]
    normalization: float
    averaging: int
    parallel_auto: bool
    multiply_weights: bool


def _build_visibility_layout(
    job: Job,
    band_data: dict[tuple[int, int], _BandSamples],
    frequencies: dict[int, Frequency],
    datastreams: dict[int, Datastream],
    requested_auto_bands: dict[int, dict[int, set[int]]],
) -> tuple[_VisibilityProductSpec, ...]:
    products: list[_VisibilityProductSpec] = []

    def append(
        reservation: tuple[Any, ...],
        stream_a: Datastream,
        band_a: int,
        stream_b: Datastream,
        band_b: int,
        output_frequency: int,
        parallel_auto: bool,
    ) -> None:
        data_a, _, _ = _logical_band(stream_a, band_a, band_data, frequencies)
        data_b, _, _ = _logical_band(stream_b, band_b, band_data, frequencies)
        products.append(
            _VisibilityProductSpec(
                reservation,
                (stream_a.identifier, band_a),
                (stream_b.identifier, band_b),
                _parent_band_key(stream_a, band_a),
                _parent_band_key(stream_b, band_b),
                math.sqrt(
                    _spectral_normalization(data_a)
                    * _spectral_normalization(data_b)
                ),
                frequencies[output_frequency].channels_to_average,
                parallel_auto,
                not parallel_auto,
            )
        )

    for baseline_index, baseline in enumerate(job.baselines):
        for product_index, product in enumerate(baseline.products):
            append(
                ("cross", baseline_index, product_index),
                datastreams[baseline.datastream_a_id],
                product.band_a_id,
                datastreams[baseline.datastream_b_id],
                product.band_b_id,
                product.output_frequency_id,
                False,
            )
    if job.write_autocorrelations:
        for stream in job.datastreams:
            for frequency_id, bands in sorted(
                requested_auto_bands.get(stream.identifier, {}).items()
            ):
                for band_a in sorted(bands):
                    for band_b in sorted(bands):
                        append(
                            (
                                "auto",
                                stream.identifier,
                                frequency_id,
                                band_a,
                                band_b,
                            ),
                            stream,
                            band_a,
                            stream,
                            band_b,
                            frequency_id,
                            band_a == band_b,
                        )
    return tuple(products)


def _spectrum_view_key(spectrum: Any) -> tuple[Any, ...]:
    return (int(spectrum.data.ptr), spectrum.shape, spectrum.strides)


@dataclass(frozen=True, slots=True)
class _VisibilityProduct:
    """Spectra and output reservations retained until submission."""

    blocks_a: _StationBlocks
    blocks_b: _StationBlocks
    reported_weight_a: Any
    reported_weight_b: Any
    boundary: tuple[int, int]
    output_offsets: list[int]
    weight_offsets: list[int]
    normalization: float
    averaging: int
    multiply_weights: bool


def _accumulate_reported_weight_adjustments(
    product_groups: dict[tuple[Any, ...], list[_VisibilityProduct]],
    host_boundaries: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    arena: _GpuOutputArena,
    xp: Any,
) -> None:
    """Separate output weight accounting from visibility normalization."""

    grouped_adjustments: dict[tuple[Any, ...], Any] = {}
    adjustment_outputs: dict[tuple[Any, ...], list[Any]] = {}
    for products in product_groups.values():
        for item in products:
            if (
                item.reported_weight_a is item.blocks_a.weight
                and item.reported_weight_b is item.blocks_b.weight
            ):
                continue
            key = (
                id(item.blocks_a.weight),
                id(item.reported_weight_a),
                id(item.blocks_b.weight) if item.multiply_weights else None,
                id(item.reported_weight_b) if item.multiply_weights else None,
                item.boundary,
            )
            grouped = grouped_adjustments.get(key)
            if grouped is None:
                ordinary = item.blocks_a.weight
                reported = item.reported_weight_a
                if item.multiply_weights:
                    ordinary = ordinary * item.blocks_b.weight
                    reported = reported * item.reported_weight_b
                delta = reported - ordinary
                starts, lengths = host_boundaries[item.boundary]
                device_starts = xp.asarray(starts, dtype=xp.int64)
                device_stops = xp.asarray(starts + lengths, dtype=xp.int64)
                prefix = xp.concatenate(
                    (
                        xp.zeros(1, dtype=xp.float64),
                        xp.cumsum(delta, dtype=xp.float64),
                    )
                )
                grouped = prefix[device_stops] - prefix[device_starts]
                grouped_adjustments[key] = grouped
            adjustment_outputs.setdefault(key, []).append(item.weight_offsets)

    for key, offsets in adjustment_outputs.items():
        host_indices = np.asarray(offsets, dtype=np.int64)
        grouped = grouped_adjustments[key]
        if np.unique(host_indices).size != host_indices.size:
            # Preserve sequential accumulation for overlapping output slots.
            for row in offsets:
                indices = xp.asarray(row, dtype=xp.int64)
                arena.weight_adjustments[indices] += grouped
            continue
        indices = xp.asarray(host_indices, dtype=xp.int64)
        arena.weight_adjustments[indices] += grouped[None, :]


def _launch_visibility_groups(
    product_groups: dict[tuple[Any, ...], list[_VisibilityProduct]],
    group_count: int,
    prepared_boundaries: dict[tuple[int, int], tuple[Any, Any]],
    arena: _GpuOutputArena,
    xp: Any,
    xmac_fusion_width: int,
    launch_visibility: Any,
    product_kind: str = "cross",
) -> None:
    """Batch compatible products with a single-product fallback."""
    for compatible_products in product_groups.values():
        host_offsets = np.empty(
            (2, len(compatible_products), group_count),
            dtype=np.int64,
        )
        for item_index, item in enumerate(compatible_products):
            host_offsets[0, item_index] = item.output_offsets
            host_offsets[1, item_index] = item.weight_offsets
        prepared_offsets = prepare_visibility_index(
            host_offsets, device_dtype=xp.int64, xp=xp
        )
        fusion_supported = compatible_products[0].averaging in (
            1,
            4,
            32,
        )
        packs: list[dict[str, Any]] = []
        singles: list[tuple[int, _VisibilityProduct]] = []
        cursor = 0
        while cursor < len(compatible_products):
            rows: list[int] = []
            items: list[_VisibilityProduct] = []
            unique_spectra: list[Any] = []
            spectrum_indices: dict[tuple[Any, ...], int] = {}
            product_bands: list[tuple[int, int]] = []
            while (
                fusion_supported
                and cursor < len(compatible_products)
                and len(items) < xmac_fusion_width
            ):
                candidate = compatible_products[cursor]
                pointers = {
                    _spectrum_view_key(getattr(candidate, name).spectrum)
                    for name in ("blocks_a", "blocks_b")
                }
                if items and len(set(spectrum_indices).union(pointers)) > 8:
                    break
                indices: list[int] = []
                for blocks_name in ("blocks_a", "blocks_b"):
                    spectrum = getattr(candidate, blocks_name).spectrum
                    pointer = _spectrum_view_key(spectrum)
                    if pointer not in spectrum_indices:
                        spectrum_indices[pointer] = len(unique_spectra)
                        unique_spectra.append(spectrum)
                    indices.append(spectrum_indices[pointer])
                rows.append(cursor)
                items.append(candidate)
                product_bands.append((indices[0], indices[1]))
                cursor += 1
            if len(items) >= 2:
                packs.append(
                    {
                        "spectra": unique_spectra,
                        "product_bands": product_bands,
                        "normalizations": [
                            item.normalization for item in items
                        ],
                        "weight_pairs": [
                            (
                                item.blocks_a.weight,
                                item.blocks_b.weight,
                            )
                            for item in items
                        ],
                        "multiply_weights": [
                            item.multiply_weights for item in items
                        ],
                        "product_rows": rows,
                    }
                )
                continue
            if items:
                singles.append((rows[0], items[0]))
                continue
            singles.append((cursor, compatible_products[cursor]))
            cursor += 1

        if packs:
            boundary = compatible_products[0].boundary
            prepared_starts, prepared_lengths = prepared_boundaries[boundary]
            with nvtx_range(f"xbeam:xmac-fused-{product_kind}"):
                batched_visibility_product_packs(
                    packs,
                    channels_to_average=compatible_products[0].averaging,
                    group_starts=prepared_starts,
                    group_lengths=prepared_lengths,
                    output_offsets=visibility_index_view(prepared_offsets, 0),
                    weight_offsets=visibility_index_view(prepared_offsets, 1),
                    output=arena.spectra,
                    weight_sum=arena.weights,
                    xp=xp,
                    _validate_ranges=False,
                )
        for item_index, item in singles:
            launch_visibility(
                item.blocks_a,
                item.blocks_b,
                item.boundary,
                item.boundary,
                visibility_index_view(prepared_offsets, (0, item_index)),
                visibility_index_view(prepared_offsets, (1, item_index)),
                item.normalization,
                item.averaging,
                item.multiply_weights,
            )


def _run_chunked_gpu(
    job: Job,
    xp: Any,
    timing_groups: list[tuple[int, Any, Sequence[int]]],
    band_data: dict[tuple[int, int], _BandSamples],
    frequencies: dict[int, Frequency],
    datastreams: dict[int, Datastream],
    stations: dict[int, Any],
    requested_auto_bands: dict[int, dict[int, set[int]]],
    arena: _GpuOutputArena,
    requested_blocks: int | None,
    available_bytes: int,
    autocorrelation_mode: AutocorrelationMode,
    payload_futures: dict[int, Future[list[Any]]] | None = None,
    *,
    visibility_layout: tuple[_VisibilityProductSpec, ...] | None = None,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
) -> list[_PendingGpuRecord]:
    """Run station and visibility work in complete-group VRAM-sized chunks."""

    if visibility_layout is None:
        visibility_layout = _build_visibility_layout(
            job, band_data, frequencies, datastreams, requested_auto_bands
        )
    xmac_fusion_width = _automatic_xmac_fusion_width(xp)
    pending, reservations = _reserve_gpu_records(
        job,
        timing_groups,
        band_data,
        frequencies,
        datastreams,
        stations,
        requested_auto_bands,
        arena,
    )
    arena.weight_adjustments.fill(0)
    groups_by_scan: dict[str, list[int]] = {}
    for group_index, (_integration, scan, _subintegrations) in enumerate(
        timing_groups
    ):
        groups_by_scan.setdefault(scan.identifier, []).append(group_index)
    schedules: dict[tuple[str, tuple[int, int]], _BlockSchedule] = {}
    shared_schedules: dict[tuple[str, float, int], _BlockSchedule] = {}
    for scan_identifier, group_indices in groups_by_scan.items():
        for key, data in band_data.items():
            schedule_key = (
                scan_identifier,
                data.sample_rate_hz,
                _fft_length(data),
            )
            schedule = shared_schedules.get(schedule_key)
            if schedule is None:
                schedule = _scan_block_schedule(
                    job, data, group_indices, timing_groups
                )
                shared_schedules[schedule_key] = schedule
            schedules[(scan_identifier, key)] = schedule
    chunk_queue = deque(
        _gpu_group_chunks(
            job,
            band_data,
            groups_by_scan,
            schedules,
            available_bytes,
            requested_blocks,
        )
    )
    payload_progress_notifier = next(
        (
            data.payload_progress._notifier
            for data in band_data.values()
            if data.payload_progress is not None
        ),
        None,
    )
    readiness_scheduling = payload_progress_notifier is not None

    while chunk_queue:
        chunk = chunk_queue.popleft()
        scan_group_indices = groups_by_scan[chunk.scan_identifier]
        group_indices = scan_group_indices[
            chunk.first_group : chunk.stop_group
        ]
        scan = timing_groups[group_indices[0]][1]
        spectra_cache: dict[tuple[int, int], _StationBlocks] = {}
        reported_weights_cache: dict[tuple[int, int], Any] = {}
        host_boundaries: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        completed: dict[tuple[int, int], _StationBlocks] = {}
        parent: _StationBlocks | None = None
        retry_chunk = False
        try:
            processed_band_keys: set[tuple[int, int]] = set()
            band_items = list(band_data.items())
            remaining_indices = list(range(len(band_items)))
            while remaining_indices:
                observed_generation: int | None = None
                selected_position: int | None = (
                    None if readiness_scheduling else 0
                )
                if readiness_scheduling:
                    assert payload_progress_notifier is not None
                    with payload_progress_notifier.condition:
                        observed_generation = (
                            payload_progress_notifier.generation
                        )
                    for position, candidate_index in enumerate(
                        remaining_indices
                    ):
                        candidate_key, candidate = band_items[candidate_index]
                        candidate_schedule = schedules[
                            (chunk.scan_identifier, candidate_key)
                        ]
                        candidate_block_start = int(
                            candidate_schedule.group_starts[chunk.first_group]
                        )
                        candidate_last_group = chunk.stop_group - 1
                        candidate_block_stop = int(
                            candidate_schedule.group_starts[
                                candidate_last_group
                            ]
                            + candidate_schedule.group_lengths[
                                candidate_last_group
                            ]
                        )
                        if _station_ready_to_attempt(
                            job,
                            [(candidate_key, candidate)],
                            scan,
                            candidate_schedule.block_starts[
                                candidate_block_start:candidate_block_stop
                            ],
                            recovery_plans,
                        ):
                            selected_position = position
                            break
                if selected_position is None:
                    if payload_progress_notifier is None:
                        raise RuntimeError(
                            "station work has no published payload producer"
                        )
                    assert observed_generation is not None
                    with nvtx_range("xbeam:vdif-ready-queue-wait"):
                        with payload_progress_notifier.condition:
                            payload_progress_notifier.condition.wait_for(
                                lambda generation=observed_generation: (
                                    payload_progress_notifier.generation
                                    != generation
                                ),
                                timeout=_PAYLOAD_PROGRESS_WATCHDOG_SECONDS,
                            )
                    continue
                item_index = remaining_indices.pop(selected_position)
                key, data = band_items[item_index]
                if (
                    data.payload_plan is not None
                    and payload_futures is not None
                    and data.payload_progress is None
                ):
                    with nvtx_range("xbeam:vdif-upload-wait"):
                        payload_futures[id(data.payload_plan)].result()
                schedule = schedules[(chunk.scan_identifier, key)]
                block_start = int(schedule.group_starts[chunk.first_group])
                last_group = chunk.stop_group - 1
                block_stop = int(
                    schedule.group_starts[last_group]
                    + schedule.group_lengths[last_group]
                )
                packed_group = [(key, data)] + [
                    band_items[candidate_index]
                    for candidate_index in remaining_indices
                    for candidate_key, candidate in [
                        band_items[candidate_index]
                    ]
                    if candidate_key not in processed_band_keys
                    and candidate.device_payload is not None
                    and candidate.source_sample_count is not None
                    and _packed_real_group_compatible(data, candidate)
                    and schedules[(chunk.scan_identifier, candidate_key)]
                    is schedule
                    and (
                        not readiness_scheduling
                        or _station_ready_to_attempt(
                            job,
                            [(candidate_key, candidate)],
                            scan,
                            schedule.block_starts[block_start:block_stop],
                            recovery_plans,
                        )
                    )
                ]
                selected_starts = schedule.block_starts[
                    block_start:block_stop
                ]
                if len(packed_group) >= 2 and not (
                    _packed_group_requires_source_replay(
                        packed_group,
                        recovery_plans,
                        scan,
                        selected_starts,
                    )
                ):
                    completed = _station_blocks_packed_real_group_for_scan(
                        job,
                        packed_group,
                        scan,
                        selected_starts,
                        xp,
                        recovery_plans,
                    )
                    for completed_key, completed_data in packed_group:
                        completed[completed_key] = _apply_fft_eligibility(
                            completed[completed_key],
                            plan=(
                                None
                                if recovery_plans is None
                                else recovery_plans.get(
                                    (
                                        scan.identifier,
                                        completed_key[0],
                                        completed_key[1],
                                    )
                                )
                            ),
                            scan=scan,
                            data=completed_data,
                            block_starts=selected_starts,
                            xp=xp,
                            _owns_buffers=True,
                        )
                    spectra_cache.update(completed)
                    completed_keys = [item[0] for item in packed_group]
                else:
                    spectra_cache[key] = _apply_fft_eligibility(
                        _station_blocks_for_scan(
                            job,
                            data,
                            scan,
                            selected_starts,
                            xp,
                            None
                            if recovery_plans is None
                            else recovery_plans.get(
                                (scan.identifier, key[0], key[1])
                            ),
                        ),
                        plan=(
                            None
                            if recovery_plans is None
                            else recovery_plans.get(
                                (scan.identifier, key[0], key[1])
                            )
                        ),
                        scan=scan,
                        data=data,
                        block_starts=selected_starts,
                        xp=xp,
                        _owns_buffers=True,
                    )
                    completed_keys = [key]
                boundaries = (
                    np.ascontiguousarray(
                        schedule.group_starts[
                            chunk.first_group : chunk.stop_group
                        ]
                        - block_start,
                        dtype=np.int32,
                    ),
                    np.ascontiguousarray(
                        schedule.group_lengths[
                            chunk.first_group : chunk.stop_group
                        ],
                        dtype=np.int32,
                    ),
                )
                for completed_key in completed_keys:
                    completed_data = band_data[completed_key]
                    completed_plan = (
                        None
                        if recovery_plans is None
                        else recovery_plans.get(
                            (
                                scan.identifier,
                                completed_key[0],
                                completed_key[1],
                            )
                        )
                    )
                    reported_weights_cache[completed_key] = (
                        _reported_station_weights(
                            spectra_cache[completed_key],
                            plan=completed_plan,
                            scan=scan,
                            data=completed_data,
                            block_starts=selected_starts,
                            xp=xp,
                        )
                    )
                    host_boundaries[completed_key] = boundaries
                    processed_band_keys.add(completed_key)
                completed_set = set(completed_keys)
                remaining_indices = [
                    candidate_index
                    for candidate_index in remaining_indices
                    if band_items[candidate_index][0] not in completed_set
                ]
            for datastream in job.datastreams:
                for zoom in datastream.zoom_bands:
                    parent_key = (datastream.identifier, zoom.parent_band_id)
                    parent = spectra_cache[parent_key]
                    start = zoom.start_channel
                    stop = start + zoom.channel_count
                    zoom_key = (datastream.identifier, zoom.identifier)
                    spectra_cache[zoom_key] = _StationBlocks(
                        parent.spectrum[..., start:stop],
                        parent.weight,
                        parent.baseband_hz[start:stop],
                    )
                    reported_weights_cache[zoom_key] = reported_weights_cache[
                        parent_key
                    ]
        except xp.cuda.memory.OutOfMemoryError:
            if chunk.stop_group - chunk.first_group <= 1:
                raise
            retry_chunk = True
        finally:
            # These aliases can own spectra even after the cache is cleared.
            completed.clear()
            parent = None
        if retry_chunk:
            spectra_cache.clear()
            reported_weights_cache.clear()
            host_boundaries.clear()
            # Exit the handler so failed-call tracebacks release buffers.
            xp.get_default_memory_pool().free_all_blocks()
            midpoint = (chunk.first_group + chunk.stop_group) // 2
            chunk_queue.appendleft(
                _GpuGroupChunk(
                    chunk.scan_identifier, midpoint, chunk.stop_group
                )
            )
            chunk_queue.appendleft(
                _GpuGroupChunk(
                    chunk.scan_identifier, chunk.first_group, midpoint
                )
            )
            continue

        boundary_signatures = {
            key: (starts.tobytes(), lengths.tobytes())
            for key, (starts, lengths) in host_boundaries.items()
        }
        prepared_boundaries: dict[tuple[int, int], tuple[Any, Any]] = {}
        prepared_by_signature: dict[tuple[bytes, bytes], tuple[Any, Any]] = {}
        for key, signature in boundary_signatures.items():
            prepared = prepared_by_signature.get(signature)
            if prepared is None:
                starts, lengths = host_boundaries[key]
                packed = prepare_visibility_index(
                    np.asarray((starts, lengths), dtype=np.int32),
                    device_dtype=xp.int32,
                    xp=xp,
                )
                prepared = (
                    visibility_index_view(packed, 0),
                    visibility_index_view(packed, 1),
                )
                prepared_by_signature[signature] = prepared
            prepared_boundaries[key] = prepared

        def launch_visibility(
            blocks_a: _StationBlocks,
            blocks_b: _StationBlocks,
            boundary_key_a: tuple[int, int],
            boundary_key_b: tuple[int, int],
            output_offsets: Any,
            weight_offsets: Any,
            normalization: float,
            channels_to_average: int,
            multiply_weights: bool,
            *,
            host_boundaries: Mapping[
                tuple[int, int], tuple[np.ndarray, np.ndarray]
            ] = host_boundaries,
            prepared_boundaries: Mapping[
                tuple[int, int], tuple[Any, Any]
            ] = prepared_boundaries,
        ) -> None:
            starts_a, lengths_a = host_boundaries[boundary_key_a]
            starts_b, lengths_b = host_boundaries[boundary_key_b]
            if not np.array_equal(starts_a, starts_b) or not np.array_equal(
                lengths_a, lengths_b
            ):
                raise ValueError(
                    "visibility bands do not share integration block "
                    "boundaries"
                )
            if not hasattr(output_offsets, "host"):
                packed_offsets = prepare_visibility_index(
                    np.asarray(
                        (output_offsets, weight_offsets), dtype=np.int64
                    ),
                    device_dtype=xp.int64,
                    xp=xp,
                )
                output_offsets = visibility_index_view(packed_offsets, 0)
                weight_offsets = visibility_index_view(packed_offsets, 1)
            prepared_starts, prepared_lengths = prepared_boundaries[
                boundary_key_a
            ]
            with nvtx_range("xbeam:xmac-single"):
                batched_visibility_product(
                    blocks_a.spectrum,
                    blocks_b.spectrum,
                    normalization=normalization,
                    channels_to_average=channels_to_average,
                    weight_a=blocks_a.weight,
                    weight_b=(blocks_b.weight if multiply_weights else None),
                    group_starts=prepared_starts,
                    group_lengths=prepared_lengths,
                    output_offsets=output_offsets,
                    weight_offsets=weight_offsets,
                    output=arena.spectra,
                    weight_sum=arena.weights,
                    xp=xp,
                    _validate_ranges=False,
                )

        # Resolve addresses and boundaries anew, including after OOM splits.
        cross_groups: dict[tuple[Any, ...], list[_VisibilityProduct]] = {}
        auto_groups: dict[tuple[Any, ...], list[_VisibilityProduct]] = {}
        for spec in visibility_layout:
            starts_a, lengths_a = host_boundaries[spec.boundary_a]
            starts_b, lengths_b = host_boundaries[spec.boundary_b]
            if not np.array_equal(starts_a, starts_b) or not np.array_equal(
                lengths_a, lengths_b
            ):
                raise ValueError(
                    "visibility bands do not share integration block "
                    "boundaries"
                )
            blocks_a = spectra_cache[spec.band_a]
            blocks_b = spectra_cache[spec.band_b]
            averaging = spec.averaging
            if spec.parallel_auto:
                blocks_a, averaging = _parallel_auto_reduction(
                    blocks_a, averaging, autocorrelation_mode, xp
                )
                blocks_b = blocks_a
            kind, *reservation_suffix = spec.reservation
            locations = [
                reservations[(kind, group_index, *reservation_suffix)]
                for group_index in group_indices
            ]
            compatibility = (
                averaging,
                blocks_a.spectrum.shape,
                blocks_b.spectrum.shape,
                *boundary_signatures[spec.boundary_a],
            )
            groups = cross_groups if kind == "cross" else auto_groups
            groups.setdefault(compatibility, []).append(
                _VisibilityProduct(
                    blocks_a=blocks_a,
                    blocks_b=blocks_b,
                    reported_weight_a=reported_weights_cache[spec.band_a],
                    reported_weight_b=reported_weights_cache[spec.band_b],
                    boundary=spec.boundary_a,
                    output_offsets=[location[0] for location in locations],
                    weight_offsets=[location[1] for location in locations],
                    normalization=spec.normalization,
                    averaging=averaging,
                    multiply_weights=spec.multiply_weights,
                )
            )
        for kind, groups in (("cross", cross_groups), ("auto", auto_groups)):
            _launch_visibility_groups(
                groups,
                len(group_indices),
                prepared_boundaries,
                arena,
                xp,
                xmac_fusion_width,
                launch_visibility,
                product_kind=kind,
            )
            _accumulate_reported_weight_adjustments(
                groups, host_boundaries, arena, xp
            )
        spectra_cache.clear()
        reported_weights_cache.clear()
        cross_groups.clear()
        auto_groups.clear()
        if visibility_layout:
            del blocks_a, blocks_b
    return pending


def _run_subintegration_tiled_gpu(
    job: Job,
    xp: Any,
    timing_group: tuple[int, Any, Sequence[int]],
    band_data: dict[tuple[int, int], _BandSamples],
    frequencies: dict[int, Frequency],
    datastreams: dict[int, Datastream],
    stations: dict[int, Any],
    requested_auto_bands: dict[int, dict[int, set[int]]],
    records_per_group: int,
    spectra_per_group: int,
    requested_blocks: int | None,
    available_bytes: int,
    autocorrelation_mode: AutocorrelationMode,
    payload_futures: dict[int, Future[list[Any]]] | None,
    initial_limit: int,
    *,
    accumulator: _GpuOutputArena | None = None,
    scratch: _GpuOutputArena | None = None,
    visibility_layout: tuple[_VisibilityProductSpec, ...] | None = None,
    recovery_plans: Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]
    | None = None,
) -> tuple[list[_PendingGpuRecord], _GpuOutputArena, list[int], int]:
    """Accumulate one logical integration from VRAM-sized time tiles."""

    integration_index, scan, subintegrations = timing_group
    queue = deque(_partition_subintegrations(subintegrations, initial_limit))
    if accumulator is None:
        try:
            accumulator = _new_gpu_output_arena(
                xp, records_per_group, spectra_per_group
            )
        except xp.cuda.memory.OutOfMemoryError as error:
            raise MemoryError(
                "the integration accumulator cannot fit in GPU memory"
            ) from error
    accumulator.spectra.fill(0)
    accumulator.weights.fill(0)
    accumulator.weight_adjustments.fill(0)
    pending: list[_PendingGpuRecord] | None = None
    completed_sizes: list[int] = []
    retry_count = 0
    if scratch is None:
        try:
            scratch = _new_gpu_output_arena(
                xp, records_per_group, spectra_per_group
            )
        except xp.cuda.memory.OutOfMemoryError as error:
            raise MemoryError(
                "the integration accumulator and tile scratch "
                "cannot fit in GPU memory"
            ) from error

    while queue:
        selected = queue.popleft()
        tile_arena = scratch
        # Previous additions and failed writes precede this same-stream reset.
        tile_arena.spectrum_cursor = tile_arena.record_cursor = 0
        tile_arena.spectra.fill(0)
        tile_arena.weights.fill(0)
        tile_arena.weight_adjustments.fill(0)
        retry_tile = False
        try:
            recovery_arguments = (
                {}
                if recovery_plans is None
                else {"recovery_plans": recovery_plans}
            )
            tile_pending = _run_chunked_gpu(
                job,
                xp,
                [(integration_index, scan, selected)],
                band_data,
                frequencies,
                datastreams,
                stations,
                requested_auto_bands,
                tile_arena,
                requested_blocks,
                available_bytes,
                autocorrelation_mode,
                payload_futures,
                visibility_layout=visibility_layout,
                **recovery_arguments,
            )
        except xp.cuda.memory.OutOfMemoryError:
            if len(selected) <= 1:
                raise MemoryError(
                    "one subintegration cannot fit in available GPU memory; "
                    "reduce the channel configuration or use a larger GPU"
                ) from None
            retry_tile = True
        if retry_tile:
            del tile_arena
            xp.get_default_memory_pool().free_all_blocks()
            midpoint = len(selected) // 2
            queue.appendleft(selected[midpoint:])
            queue.appendleft(selected[:midpoint])
            retry_count += 1
            continue

        if pending is None:
            pending = tile_pending
        elif len(tile_pending) != len(pending):
            raise RuntimeError(
                "subintegration tiles produced different visibility layouts"
            )
        accumulator.spectra += tile_arena.spectra
        accumulator.weights += tile_arena.weights
        accumulator.weight_adjustments += tile_arena.weight_adjustments
        completed_sizes.append(len(selected))

    if pending is None:
        raise RuntimeError("logical integration contained no subintegrations")
    # Keep the one scratch allocation alive through the final host copy.
    accumulator.retained = [scratch]
    return pending, accumulator, completed_sizes, retry_count
