# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader adapter: input-derived compatibility planning."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import numpy as np

from cuphoton.xbeam.adapters.radio.compute_plan import _sample_rate
from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    Frequency,
    Job,
    RecordedBand,
    SamplingKind,
    Scan,
)
from cuphoton.xbeam.adapters.vdif.ingest import VdifIngestPlan
from cuphoton.xbeam.adapters.vdif.recording import VdifThreadWindow
from cuphoton.xbeam.adapters.vdif.vdif import (
    VdifHeader,
    absolute_frame_slot,
    reference_epoch_datetime,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
    VdifBinaryIndexSummary,
    _summary_physical_timestamp_runs,
    validate_vdif_binary_index,
)

from ..errors import ConfigurationError
from ..missing_timestamp_recovery import (
    GlobalFFTEligibilityPlan,
    RecoveryPlanKey,
)
from ._checks import _integer
from .addressing import LatchedRawSourceCompilation
from .clocks import LeadingRawHeadContract
from .geometry import (
    BufferedNormalFullExtent,
    BufferedRequestGeometry,
    BufferedWindowGeometry,
    RawBufferedRequestState,
    RawResidualOwner,
)
from .plans import (
    RawLinearStorage,
    RawReplayPlanWindow,
    compile_raw_replay_window,
)
from .replay import (
    RawReplayWindowFacts,
    RetainedRawWindowContract,
    bind_known_source_end,
)
from .requests import (
    RawJobRequestInputs,
    ReadClockConvention,
    bind_initial_retained_window_contract,
    evaluate_raw_job_requests,
)
from .timeline import PhysicalAllocationTimeline
from .validity import RawValidityCompilation, RawValidityState


def packed_timestamp_slots(
    packed: np.ndarray,
    *,
    frames_per_second: int,
    template: VdifHeader,
    epoch_cycle: int = 0,
) -> np.ndarray:
    """Decode sparse packed timestamps on the ordinary VDIF slot grid.

    Work scales with run count. Each reference epoch is decoded separately;
    epoch-cycle selection is explicit rather than inferred from a recording
    name. Packed keys and frame numbers are checked before arithmetic.
    """
    rate = _integer(frames_per_second, "frames per second", minimum=1)
    cycle = _integer(epoch_cycle, "epoch cycle")
    if rate > 1 << 24:
        raise ValueError("frames per second exceeds VDIF frame-number range")
    if (
        not isinstance(packed, np.ndarray)
        or packed.ndim != 1
        or packed.dtype.kind not in "iu"
        or not packed.size
    ):
        raise ValueError(
            "packed timestamps must be a nonempty integer vector"
        )
    if np.any(packed < 0) or np.any(packed >= 1 << 60):
        raise ValueError("packed VDIF timestamp exceeds its 60-bit range")
    keys = packed.astype(np.uint64, copy=False)
    epochs = keys >> np.uint64(54)
    seconds = (keys >> np.uint64(24)) & np.uint64(0x3FFF_FFFF)
    frames = keys & np.uint64(0x00FF_FFFF)
    if np.any(frames >= rate):
        raise ValueError("VDIF frame number exceeds frames per second")
    slots = np.empty(keys.shape, dtype=np.int64)
    maximum = int(np.iinfo(np.int64).max)
    for epoch in np.unique(epochs):
        selected = epochs == epoch
        origin = absolute_frame_slot(
            replace(
                template,
                reference_epoch=int(epoch),
                seconds_from_epoch=0,
                frame_number=0,
            ),
            rate,
            epoch_cycle=cycle,
        )
        last = origin + int(seconds[selected].max()) * rate
        if last + int(frames[selected].max()) > maximum:
            raise OverflowError("absolute VDIF slots exceed int64")
        slots[selected] = (
            origin
            + seconds[selected].astype(np.int64) * rate
            + frames[selected].astype(np.int64)
        )
    return slots


def job_frame_slot(
    job: Job, window: VdifThreadWindow, *, epoch_cycle: int = 0
) -> int:
    """Convert job MJD to the same leap-aware grid as the ingest window."""
    cycle = _integer(epoch_cycle, "epoch cycle")
    rate = _integer(window.frames_per_second, "frames per second", minimum=1)
    header = window.template
    epoch = reference_epoch_datetime(
        header.reference_epoch, epoch_cycle=cycle
    )
    epoch_unix_seconds = int(epoch.timestamp())
    nominal_header = (
        epoch_unix_seconds + 40587 * 86400 + header.seconds_from_epoch
    ) * rate + header.frame_number
    native_header = absolute_frame_slot(header, rate, epoch_cycle=cycle)
    nominal_job = (
        Decimal(job.start_time.mjd) * 86400 + job.start_time.seconds_of_day
    ) * rate
    if nominal_job != nominal_job.to_integral_value():
        raise ValueError("job start must be VDIF-frame aligned")
    return int(nominal_job) - (nominal_header - native_header)


@dataclass(frozen=True, slots=True)
class RawSidecarContext:
    """One indexed physical source, reusable across bounded model requests."""

    validity_state: RawValidityState
    request_geometry: BufferedRequestGeometry
    reader_geometry: BufferedWindowGeometry
    job_start_slot: int
    sample_rate: int
    frames_per_second: int
    frame_bytes: int
    control_word_bits: int


def raw_sidecar_context(
    job: Job,
    datastream: Datastream,
    frequency: Frequency,
    window: VdifThreadWindow,
    summary: VdifBinaryIndexSummary,
    *,
    alignment_bytes: int,
    word_bytes: int,
    control_word_bits: int,
    epoch_cycle: int = 0,
    residual_owner: RawResidualOwner = RawResidualOwner.PHYSICAL_FRAME,
) -> RawSidecarContext:
    """Index trustworthy single-thread raw runs once, without reading payload.

    Caller owns sidecar validation and cache lifetime. Header/format rejection
    counters are always errors. This adapter accepts schema-v4 compact runs
    and interleaved real sampling; it does not synthesize missing clocks or
    reset physical allocation phase for a source beginning with filler.
    Residual ownership is an explicit caller contract; the existing default
    attributes it to the preceding physical frame. No source identity selects
    a different owner.
    """
    if job.buffered_reader is None:
        raise ValueError(
            "raw replay requires explicit buffered-reader settings"
        )
    if not isinstance(summary, VdifBinaryIndexSummary):
        raise TypeError("summary must be a validated binary sidecar")
    if summary.schema_version != 4 or not summary.has_physical_timestamp_runs:
        raise ConfigurationError(
            "correlate",
            "profile",
            "raw replay requires compact schema-v4 clock runs; "
            "rebuild VDIF sidecars with --force",
        )
    if (
        summary.rejected_header_frame_count
        or summary.rejected_format_frame_count
    ):
        raise ValueError(
            "raw replay does not accept rejected header/format frames"
        )
    if len(summary.thread_metadata) != 1:
        raise ValueError("raw replay requires one physical VDIF thread")
    metadata = summary.thread_metadata[0]
    if metadata.thread_id != window.template.thread_id:
        raise ValueError("physical sidecar thread differs from ingest window")
    if datastream.sampling is not SamplingKind.REAL or metadata.complex_data:
        raise ValueError("raw replay requires real-sampled VDIF")
    if (
        metadata.frame_size_bytes != datastream.frame_size_bytes
        or summary.frame_size_bytes != datastream.frame_size_bytes
        or metadata.bits_per_sample != datastream.quantization_bits
    ):
        raise ValueError("sidecar and datastream sample geometry differ")
    # Coordinates count time samples per channel; a byte spans all channels.
    samples_per_byte = Fraction(
        8, metadata.bits_per_sample * metadata.channel_count
    )
    rate = float(_sample_rate(datastream, frequency))
    if not np.isfinite(rate) or rate <= 0 or rate != int(rate):
        raise ValueError(
            "raw replay requires a positive integral sample rate"
        )
    if rate != metadata.samples_per_frame * window.frames_per_second:
        raise ValueError("sidecar sample rate differs from ingest window")
    fft_samples = frequency.channel_count * 2
    reader = BufferedWindowGeometry.from_reader(
        job.buffered_reader,
        samples_per_frame=metadata.samples_per_frame,
        fft_samples=fft_samples,
        sample_rate=int(rate),
        subintegration_s=job.subintegration_duration_s,
    )
    request = BufferedRequestGeometry(
        samples_per_byte,
        fft_samples,
        reader.ffts_per_request,
        alignment_bytes,
        word_bytes,
    )
    bits = _integer(control_word_bits, "control word bits", minimum=1)
    if bits > 64:
        raise ValueError("control word bits exceeds integer protocol range")
    runs = _summary_physical_timestamp_runs(summary)
    if runs.ndim != 2 or runs.shape[1] != 3 or not len(runs):
        raise ValueError("physical timestamp runs must be nonempty triples")
    if int(runs[-1, 1]) > summary.frame_count:
        raise ValueError(
            "physical timestamp runs exceed recording frame count"
        )
    slots = packed_timestamp_slots(
        runs[:, 2],
        frames_per_second=window.frames_per_second,
        template=window.template,
        epoch_cycle=epoch_cycle,
    )
    timeline = PhysicalAllocationTimeline.from_runs(
        runs[:, 0],
        runs[:, 1],
        slots,
        allocation_frames=reader.allocation_frames,
    )
    origin = job_frame_slot(job, window, epoch_cycle=epoch_cycle)
    state = RawBufferedRequestState.from_timeline(
        timeline,
        samples_per_frame=metadata.samples_per_frame,
        timestamp_origin_frame=origin,
        residual_owner=residual_owner,
    )
    validity = RawValidityState.from_timeline(timeline, state)
    validity = replace(
        validity,
        source_state=replace(
            validity.source_state,
            recorded_source_stop=summary.frame_count
            * metadata.samples_per_frame,
        ),
    )
    return RawSidecarContext(
        validity,
        request,
        reader,
        origin,
        int(rate),
        window.frames_per_second,
        datastream.frame_size_bytes,
        bits,
    )


@dataclass(frozen=True, slots=True)
class RawJobReplayScope:
    key: Hashable
    band: RecordedBand
    frequency: Frequency
    scan: Scan
    first_fft: int
    stop_fft: int
    timeline_fft_count: int
    geometry: BufferedRequestGeometry


@dataclass(frozen=True, slots=True)
class RawReplayScopePlans:
    """Validated windows sharing storage; facts retain their global scopes.

    If every scope is discarded, storage is None and every window has empty
    physical presentation: no fabricated source extent/read is requested.
    A discarded scope alongside populated scopes shares their common metadata.
    Returned windows require plan_for_scope before execution consumption.
    """

    storage: RawLinearStorage | None
    windows: Mapping[Hashable, RawReplayPlanWindow]
    facts: Mapping[Hashable, RawReplayWindowFacts]


def _merge_affine(spans, sample_name, fft_samples):
    merged = []
    for span in spans:
        if (
            merged
            and merged[-1].stop_fft == span.first_fft
            and getattr(merged[-1], sample_name)
            + (span.first_fft - merged[-1].first_fft) * fft_samples
            == getattr(span, sample_name)
        ):
            merged[-1] = replace(merged[-1], stop_fft=span.stop_fft)
        else:
            merged.append(span)
    return tuple(merged)


def _combine_tiles(tiles):
    first, stop, size = (
        tiles[0].first_fft,
        tiles[-1].stop_fft,
        tiles[0].fft_samples,
    )
    spans = _merge_affine(
        (span for tile in tiles for span in tile.addresses.spans),
        "physical_sample_start",
        size,
    )
    unresolved = tuple(
        span for tile in tiles for span in tile.addresses.unresolved
    )
    addresses = LatchedRawSourceCompilation(spans, unresolved)
    # Preserve independently clipped decoder validity. Re-intersecting only
    # the physical inventory here would restore unavailable decoder samples.
    empty = []
    for tile in tiles:
        for left, right in tile.validity.empty_intervals:
            if empty and empty[-1][1] == left:
                empty[-1] = (empty[-1][0], right)
            else:
                empty.append((left, right))
    validity = RawValidityCompilation(
        _merge_affine(
            (span for tile in tiles for span in tile.validity.full_spans),
            "physical_sample_start",
            size,
        ),
        tuple(
            event for tile in tiles for event in tile.validity.partial_events
        ),
        tuple(empty),
        tuple(span for tile in tiles for span in tile.validity.unresolved),
    )
    rejected = []
    for tile in tiles:
        for left, right in tile.ineligible_intervals:
            if rejected and rejected[-1][1] == left:
                rejected[-1] = (rejected[-1][0], right)
            else:
                rejected.append((left, right))
    ordinary = _merge_affine(
        (span for tile in tiles for span in tile.ordinary_gpu_source_spans),
        "ordinary_sample_start",
        size,
    )
    extents = [
        tile.physical_sample_extent
        for tile in tiles
        if tile.physical_sample_extent is not None
    ]
    physical = (
        (min(x[0] for x in extents), max(x[1] for x in extents))
        if extents
        else None
    )
    return RawReplayWindowFacts(
        first,
        stop,
        size,
        addresses,
        validity,
        tuple(rejected),
        ordinary,
        physical,
        (
            min(tile.ordinary_sample_extent[0] for tile in tiles),
            max(tile.ordinary_sample_extent[1] for tile in tiles),
        ),
        tuple(event for tile in tiles for event in tile.source_timing_events),
    )


def compile_raw_job_replay_scopes(
    validity_state: RawValidityState,
    job: Job,
    datastream: Datastream,
    scopes: tuple[RawJobReplayScope, ...],
    *,
    job_start_slot: int,
    source_index: int,
    control_word_bits: int,
    maximum_requests_per_tile: int = 4096,
    maximum_empty_fft_events: int = 4096,
    window_clock_contract_factory: Callable[
        [RawJobReplayScope, RawJobRequestInputs, RawValidityState],
        RetainedRawWindowContract,
    ],
    read_clock_convention: ReadClockConvention | None = None,
) -> RawReplayScopePlans:
    """Compile bounded tiles and choose storage across all supplied scopes.

    Caller must supply ALL bands/scans/windows sharing the eventual storage
    presentation, not call independently per band or tile. Cached timeline
    coverage retains explicit-invalid frames; ordinary frame validity remains
    required. Request clocks and GPU starts share job_start_slot's basis.
    The source index is the caller's validated global file index.
    A caller-owned factory binds each bounded tile from its evaluated input
    clocks and cached source state. It must return a retained contract for
    exactly that tile, with actual aligned heads and framed/decoded extents;
    this coordinator cannot infer them after gaps. No scan-sized clock array
    or fallback is created.
    """
    limit = _integer(
        maximum_requests_per_tile, "requests per tile", minimum=1
    )
    if limit > 4096:
        raise ValueError("requests per tile cannot exceed 4096")
    if not isinstance(validity_state, RawValidityState):
        raise TypeError("validity state must be cached RawValidityState")
    source = _integer(source_index, "source file index")
    if source >= len(datastream.source_files):
        raise ValueError("source file index lies outside datastream files")
    if not isinstance(scopes, tuple) or not scopes:
        raise ValueError("at least one explicit replay scope is required")
    if not callable(window_clock_contract_factory):
        raise TypeError("window binding requires a callable contract factory")
    facts = {}
    shared_rate = None
    for scope in scopes:
        if not isinstance(scope, RawJobReplayScope):
            raise TypeError("scopes must contain RawJobReplayScope")
        if scope.key in facts:
            raise ValueError("scope keys must be unique")
        if scope.band.frequency_id != scope.frequency.identifier:
            raise ValueError("scope band and frequency do not agree")
        if not isinstance(scope.geometry, BufferedRequestGeometry):
            raise TypeError("scope geometry must be BufferedRequestGeometry")
        first = _integer(scope.first_fft, "scope first FFT")
        stop = _integer(scope.stop_fft, "scope stop FFT", minimum=1)
        count = _integer(
            scope.timeline_fft_count, "timeline FFT count", minimum=1
        )
        if not first < stop <= count:
            raise ValueError("scope must lie within its FFT timeline")
        tiles = []
        while first < stop:
            width = scope.geometry.ffts_per_request
            tile_stop = min(stop, (first // width + limit) * width)
            inputs = evaluate_raw_job_requests(
                job,
                datastream,
                scope.band,
                scope.frequency,
                scope.scan,
                scope.geometry,
                first_fft=first,
                stop_fft=tile_stop,
                timeline_fft_count=count,
                job_start_slot=job_start_slot,
                timestamp_origin_frame=job_start_slot,
                maximum_requests=limit,
                read_clock_convention=read_clock_convention,
            )
            if shared_rate is None:
                shared_rate = inputs.sample_rate
            elif shared_rate != inputs.sample_rate:
                raise ValueError("sharing scopes require one raw sample rate")
            tile_contract = window_clock_contract_factory(
                scope, inputs, validity_state
            )
            if not isinstance(tile_contract, RetainedRawWindowContract):
                raise TypeError(
                    "factory must return RetainedRawWindowContract"
                )
            if (tile_contract.first_fft, tile_contract.stop_fft) != (
                first,
                tile_stop,
            ):
                raise ValueError("factory contract must match tile scope")
            tile = inputs.compile(
                validity_state,
                frame_bytes=datastream.frame_size_bytes,
                control_word_bits=control_word_bits,
                window_clock_contract=tile_contract,
            )
            tiles.append(tile)
            first = tile_stop
        combined = _combine_tiles(tiles)
        if not combined.complete:
            reasons = sorted(
                {span.reason for span in combined.addresses.unresolved}
            )
            raise ValueError("unresolved raw scope: " + "; ".join(reasons))
        facts[scope.key] = combined
    extents = [
        result.physical_sample_extent
        for result in facts.values()
        if result.physical_sample_extent is not None
    ]
    storage = None
    if extents:
        size = validity_state.source_state.request_state.samples_per_frame
        storage = RawLinearStorage.align_to_ordinary_window(
            datastream_id=datastream.identifier,
            source_index=source,
            job_start_slot=job_start_slot,
            samples_per_frame=size,
            raw_first_frame=min(x[0] for x in extents) // size,
            raw_stop_frame=-(-max(x[1] for x in extents) // size),
            ordinary_min_sample=min(
                result.ordinary_sample_extent[0] for result in facts.values()
            ),
        )
    windows = {}
    for key, result in facts.items():
        if storage is not None:
            window = result.compile(
                storage, maximum_empty_fft_events=maximum_empty_fft_events
            )
        else:
            window = compile_raw_replay_window(
                result.addresses,
                result.validity,
                first_fft=result.first_fft,
                stop_fft=result.stop_fft,
                fft_samples=result.fft_samples,
                ordinary_source_spans=result.ordinary_gpu_source_spans,
                ineligible_intervals=result.ineligible_intervals,
                maximum_empty_fft_events=maximum_empty_fft_events,
            )
        window.plan_for_scope(result.first_fft, result.stop_fft)
        windows[key] = window
    return RawReplayScopePlans(
        storage, MappingProxyType(windows), MappingProxyType(facts)
    )


def _scan_scopes(timing_groups):
    selected = {}
    for _integration, scan, subintegrations in timing_groups:
        minimum, maximum = None, None
        if isinstance(subintegrations, range) and subintegrations:
            ends = (subintegrations[0], subintegrations[-1])
            minimum = _integer(min(ends), "selected subintegration")
            maximum = _integer(max(ends), "selected subintegration")
        else:
            for index in subintegrations:
                index = _integer(index, "selected subintegration")
                minimum = index if minimum is None else min(minimum, index)
                maximum = index if maximum is None else max(maximum, index)
        if minimum is None:
            continue
        existing = selected.get(scan.identifier)
        if existing is not None and existing[0] != scan:
            raise ValueError(
                "one scan identifier describes conflicting models"
            )
        if existing is not None:
            minimum, maximum = (
                min(minimum, existing[1]),
                max(maximum, existing[2]),
            )
        selected[scan.identifier] = (scan, minimum, maximum)
    if not selected:
        raise ValueError("raw file replay requires selected subintegrations")
    return selected


def build_raw_file_replay_plans(
    job: Job,
    timing_groups: Sequence[tuple[int, object, Sequence[int]]],
    ingest_plan: VdifIngestPlan,
    *,
    index_dir: str | None,
    alignment_bytes: int,
    word_bytes: int,
    control_word_bits: int,
    maximum_requests_per_tile: int = 4096,
    epoch_cycle: int = 0,
    window_clock_contract_factory: Callable[
        [RawJobReplayScope, RawJobRequestInputs, RawValidityState],
        RetainedRawWindowContract,
    ],
    read_clock_convention: ReadClockConvention | None = None,
    residual_owner: RawResidualOwner = RawResidualOwner.PHYSICAL_FRAME,
) -> Mapping[RecoveryPlanKey, GlobalFFTEligibilityPlan]:
    """Build scoped plans without choosing production activation or exposure.

    Strictly one source file and one physical real VDIF thread per datastream.
    All sharing bands must have identical sample/FFT/allocation geometry.
    Protocol alignment, word size and control bits are explicit caller facts.
    Interior phase compilation is an explicit opt-in, not a default fallback.

    Scan clocks use exact Decimal FFT counts. The enclosing selected scope
    includes every complete selected subintegration: own execution admits a
    subintegration by its midpoint and executes all its FFTs, including a tail
    beyond the scan endpoint. A sparse selection's enclosing gaps are compiled
    too. Returned plans retain global coordinates; caller must consume
    only the timing_groups provided here, never their unknown leading prefix.
    The caller's factory binds each bounded tile to its evaluated input
    clocks and cached raw state, avoiding a whole-scan contract allocation.
    It must establish actual heads and extents in a retained-window contract;
    no clock is inferred after gaps or selected from recording identity.
    """
    if ingest_plan.thread_windows is None:
        raise ValueError("raw file replay requires bounded ingest windows")
    selected = _scan_scopes(timing_groups)
    if not callable(window_clock_contract_factory):
        raise TypeError("window binding requires a callable contract factory")
    frequencies = {
        frequency.identifier: frequency for frequency in job.frequencies
    }
    plans: dict[RecoveryPlanKey, GlobalFFTEligibilityPlan] = {}
    for datastream in job.datastreams:
        if datastream.sampling is not SamplingKind.REAL:
            raise ValueError("raw file replay requires real sampling")
        if len(datastream.source_files) != 1:
            raise ValueError(
                "raw file replay requires exactly one source file"
            )
        if not datastream.recorded_bands:
            raise ValueError("raw file replay requires recorded bands")
        windows = ingest_plan.thread_windows.get(datastream.identifier)
        if windows is None or len(windows) != 1:
            raise ValueError(
                "raw file replay requires one physical ingest thread"
            )
        window = next(iter(windows.values()))
        first_band = datastream.recorded_bands[0]
        if first_band.frequency_id not in frequencies:
            raise ValueError("recorded frequency is absent from job")
        summary = validate_vdif_binary_index(
            datastream.source_files[0],
            frame_size_bytes=datastream.frame_size_bytes,
            index_dir=index_dir,
        )
        context = raw_sidecar_context(
            job,
            datastream,
            frequencies[first_band.frequency_id],
            window,
            summary,
            alignment_bytes=alignment_bytes,
            word_bytes=word_bytes,
            control_word_bits=control_word_bits,
            epoch_cycle=epoch_cycle,
            residual_owner=residual_owner,
        )
        scopes = []
        for band in datastream.recorded_bands:
            frequency = frequencies.get(band.frequency_id)
            if frequency is None:
                raise ValueError("recorded frequency is absent from job")
            # Ingest has already validated logical-to-physical thread aliases.
            if windows.get(band.thread_id) != window:
                raise ValueError(
                    "recorded band differs from physical ingest thread"
                )
            rate = Decimal(str(_sample_rate(datastream, frequency)))
            fft_samples = frequency.channel_count * 2
            if (
                rate != context.sample_rate
                or fft_samples != context.request_geometry.fft_samples
            ):
                raise ValueError(
                    "sharing bands have incompatible raw geometry"
                )
            block_count = job.subintegration_duration_s * rate / fft_samples
            if (
                not block_count.is_finite()
                or block_count <= 0
                or block_count != block_count.to_integral_value()
            ):
                raise ValueError(
                    "subintegration must contain integral FFT blocks"
                )
            blocks = int(block_count)
            for scan_identifier, (scan, minimum, maximum) in selected.items():
                origin = max(Decimal(0), scan.start_offset_s)
                end = min(
                    job.execution_duration_s,
                    scan.start_offset_s + scan.duration_s,
                )
                span = end - origin
                if not span.is_finite() or span <= 0:
                    raise ValueError(
                        "selected scan has no executable duration"
                    )
                # Complete executable subintegrations use midpoint inclusion.
                admitted = max(
                    0,
                    int(
                        (
                            span / job.subintegration_duration_s
                            - Decimal("0.5")
                        ).to_integral_value(rounding=ROUND_CEILING)
                    ),
                )
                if maximum >= admitted:
                    raise ValueError(
                        "selected subintegration lies beyond scan execution"
                    )
                first = minimum * blocks
                stop = (maximum + 1) * blocks
                nominal_count = int(
                    (span * rate / fft_samples).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                )
                timeline_count = max(nominal_count, stop)
                scopes.append(
                    RawJobReplayScope(
                        (
                            scan_identifier,
                            datastream.identifier,
                            band.identifier,
                        ),
                        band,
                        frequency,
                        scan,
                        first,
                        stop,
                        timeline_count,
                        context.request_geometry,
                    )
                )
        coordinated = compile_raw_job_replay_scopes(
            context.validity_state,
            job,
            datastream,
            tuple(scopes),
            job_start_slot=context.job_start_slot,
            source_index=0,
            control_word_bits=context.control_word_bits,
            maximum_requests_per_tile=maximum_requests_per_tile,
            window_clock_contract_factory=window_clock_contract_factory,
            read_clock_convention=read_clock_convention,
        )
        for key, window_plan in coordinated.windows.items():
            facts = coordinated.facts[key]
            plans[cast(RecoveryPlanKey, key)] = replace(
                window_plan.plan_for_scope(facts.first_fft, facts.stop_fft),
                known_fft_scope=(facts.first_fft, facts.stop_fft),
            )
    return MappingProxyType(plans)


def build_returned_window_replay_plans(
    job,
    timing_groups,
    ingest_plan,
    *,
    index_dir,
    alignment_bytes,
    word_bytes,
    control_word_bits,
    timestamp_quantum_nanoseconds,
    replay_identifier,
    maximum_requests_per_tile=256,
):
    """Build sparse plans with a declared retained-window publication policy.

    Reader allocation sizes derive from the job and validated sample geometry.
    Normal returns retain their allocation, with explicit successor transfer
    and uncovered-head publication rules. Initial returns use the origin-zero
    first allocation and endpoint-derived whole-FFT skips. Source validity,
    decoded coverage and logical read phase remain independent of eligibility.

    The adapter supports single-thread real VDIF, including multiple
    interleaved channels. Multiplexed files, rejected headers, uncertain
    initial history and unknown returned payloads raise errors. The
    identifier is output provenance only and never selects source behavior.
    Calling this function does not change any production execution profile.
    """
    if not isinstance(replay_identifier, str) or not replay_identifier:
        raise ValueError("replay identifier must be a nonempty string")
    quantum = _integer(
        timestamp_quantum_nanoseconds, "timestamp quantum", minimum=1
    )
    bits = _integer(control_word_bits, "control word bits", minimum=1)
    if job.buffered_reader is None:
        raise ValueError("returned replay requires declared reader settings")
    reader = job.buffered_reader
    ratio, remainder = divmod(reader.buffer_factor, reader.data_segments)
    if remainder or ratio < 1:
        raise ValueError("reader buffers must contain whole data segments")
    datastreams = {item.identifier: item for item in job.datastreams}

    def bind(scope, inputs, state):
        datastream = datastreams[scope.key[1]]
        allocation = state.source_state.request_state.allocation_frames
        frames, remainder = divmod(allocation, ratio)
        if remainder:
            raise ValueError("allocation must contain whole framed requests")
        extent = BufferedNormalFullExtent(
            datastream.frame_size_bytes,
            datastream.frame_size_bytes - 32,
            frames,
            ratio,
            scope.geometry.samples_per_byte,
        )
        head = LeadingRawHeadContract(int(inputs.sample_rate), quantum)
        contract = bind_initial_retained_window_contract(
            inputs,
            state,
            extent,
            normal_contract_factory=lambda suffix: (
                suffix.published_normal_full_window_contract(
                    state,
                    extent,
                    control_word_bits=bits,
                    ordinary_head_contract=head,
                )
            ),
        )
        source_stop = state.source_state.recorded_source_stop
        # A finite file can end with recorded filler after its final trusted
        # timestamp run. Carry that extent even when this tile stays below
        # EOF, so known absence does not depend on other requests in the tile.
        if source_stop is not None:
            contract = bind_known_source_end(contract, source_stop)
        return contract

    plans = build_raw_file_replay_plans(
        job,
        timing_groups,
        ingest_plan,
        index_dir=index_dir,
        alignment_bytes=alignment_bytes,
        word_bytes=word_bytes,
        control_word_bits=bits,
        maximum_requests_per_tile=maximum_requests_per_tile,
        window_clock_contract_factory=bind,
        read_clock_convention=ReadClockConvention.ENDPOINT_MEAN,
        residual_owner=RawResidualOwner.CLOCK_FRAME,
    )
    return MappingProxyType(
        {
            key: replace(plan, calibration_identifier=replay_identifier)
            for key, plan in plans.items()
        }
    )
