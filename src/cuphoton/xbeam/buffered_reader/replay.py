# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader replay: input-derived compatibility planning."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import cast

import numpy as np

from ._checks import _integer
from .addressing import (
    LatchedRawSourceCompilation,
    LatchedRawSourceSpan,
    UnresolvedRawSourceSpan,
)
from .clocks import (
    RawWindowPayloadSpans,
    compile_raw_window_bounds,
    compile_raw_window_payload_spans,
)
from .geometry import (
    BufferedNormalFullExtent,
    BufferedRequestGeometry,
    _cursor_add,
    _cursor_multiply,
    _cursor_subtract,
    compile_buffered_request_discards,
)
from .plans import (
    OrdinaryRawSourceSpan,
    RawLinearStorage,
    RawReplayPlanWindow,
    compile_translated_raw_replay_window,
)
from .validity import (
    RawValidityCompilation,
    RawValidityState,
    intersect_raw_decoder_validity,
    split_latched_raw_source_validity,
)


@dataclass(frozen=True, slots=True)
class RawWindowClockContract:
    """Explicit bounded operands, independent of source clock inference.

    All vectors cover [first_fft, stop_fft) in global FFT order. Output and
    returned aligned head use the SAME unshifted logical sample clock; storage
    translations must not enter these operands. Delays are actual discrete
    FFT-centre delays, including any initial skip. Framed admission and
    decoded extents are independently supplied SAMPLE counts, not byte counts.
    The caller must establish actual returned heads, especially after gaps.
    No buffer sizing, head reconstruction or exposure policy is implied.
    """

    first_fft: int
    output_samples: np.ndarray
    aligned_head_samples: np.ndarray
    centre_delays_seconds: np.ndarray
    framed_extent_samples: np.ndarray
    decoded_extent_samples: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self, "first_fft", _integer(self.first_fft, "window first FFT")
        )
        shape = None
        for name, dtype in (
            ("output_samples", np.int64),
            ("aligned_head_samples", np.int64),
            ("centre_delays_seconds", np.float64),
            ("framed_extent_samples", np.int64),
            ("decoded_extent_samples", np.int64),
        ):
            array = getattr(self, name)
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 1
                or array.dtype != np.dtype(dtype)
            ):
                raise TypeError(f"{name} must be a {np.dtype(dtype)} vector")
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("window clock vectors must have equal shape")
            frozen = np.frombuffer(array.tobytes(), dtype=array.dtype)
            object.__setattr__(self, name, frozen)
        if not self.output_samples.size:
            raise ValueError("window clock scope must be nonempty")
        if self.stop_fft > np.iinfo(np.int64).max:
            raise OverflowError("window FFT scope exceeds int64")
        if not np.all(np.isfinite(self.centre_delays_seconds)):
            raise ValueError("window centre delays must be finite")
        if np.any(self.framed_extent_samples < 0) or np.any(
            self.decoded_extent_samples < 0
        ):
            raise ValueError("window extents must be nonnegative")

    @property
    def stop_fft(self) -> int:
        return self.first_fft + self.output_samples.size

    def for_scope(
        self, first_fft: int, stop_fft: int
    ) -> RawWindowClockContract:
        """Select explicit operands without extrapolating unknown scope."""
        first = _integer(first_fft, "window first FFT")
        stop = _integer(stop_fft, "window stop FFT", minimum=1)
        if not self.first_fft <= first < stop <= self.stop_fft:
            raise ValueError(
                "requested scope lies outside window clock contract"
            )
        if first == self.first_fft and stop == self.stop_fft:
            return self
        selected = slice(first - self.first_fft, stop - self.first_fft)
        return RawWindowClockContract(
            first,
            self.output_samples[selected],
            self.aligned_head_samples[selected],
            self.centre_delays_seconds[selected],
            self.framed_extent_samples[selected],
            self.decoded_extent_samples[selected],
        )


@dataclass(frozen=True, slots=True)
class RetainedRawWindowContract(RawWindowClockContract):
    """Explicit returned physical heads and request publication decisions.

    This additionally asserts a retained physical window: each accepted
    FFT reads at its head plus the nonnegative signed clock offset, without
    relatching its body to later timestamp anchors. Publication is supplied
    independently of decoded coverage. Extents alone never select this rule.
    An explicitly supplied decoder-clamp vector establishes that accepted
    negative offsets decode from the first payload sample. Clock admission
    still rejects offsets below -1; this never grants source validity.
    Vectors describe bounded input requests, not measured output corrections.
    """

    physical_head_samples: np.ndarray
    publication_eligible: np.ndarray
    decoder_clamps_negative_offsets: np.ndarray | None = None
    known_source_stop: int | None = None

    def __post_init__(self):
        RawWindowClockContract.__post_init__(self)
        if self.known_source_stop is not None:
            stop = _integer(
                self.known_source_stop, "known source stop", minimum=0
            )
            if stop > np.iinfo(np.int64).max:
                raise OverflowError("known source stop exceeds int64")
            object.__setattr__(self, "known_source_stop", stop)
        for name, dtype in (
            ("physical_head_samples", np.int64),
            ("publication_eligible", np.bool_),
        ):
            array = getattr(self, name)
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.dtype(dtype)
                or array.shape != self.output_samples.shape
            ):
                raise TypeError(f"{name} must be a matching vector")
            object.__setattr__(
                self, name, np.frombuffer(array.tobytes(), dtype=array.dtype)
            )

        clamp = self.decoder_clamps_negative_offsets
        if clamp is not None:
            if (
                not isinstance(clamp, np.ndarray)
                or clamp.dtype != np.dtype(np.bool_)
                or clamp.shape != self.output_samples.shape
            ):
                raise TypeError(
                    "decoder clamp policy must be a matching bool vector"
                )
            object.__setattr__(
                self,
                "decoder_clamps_negative_offsets",
                np.frombuffer(clamp.tobytes(), dtype=clamp.dtype),
            )

    def for_scope(self, first_fft: int, stop_fft: int):
        clock = RawWindowClockContract.for_scope(self, first_fft, stop_fft)
        if clock is self:
            return self
        selected = slice(
            first_fft - self.first_fft, stop_fft - self.first_fft
        )
        return RetainedRawWindowContract(
            clock.first_fft,
            clock.output_samples,
            clock.aligned_head_samples,
            clock.centre_delays_seconds,
            clock.framed_extent_samples,
            clock.decoded_extent_samples,
            self.physical_head_samples[selected],
            self.publication_eligible[selected],
            None
            if self.decoder_clamps_negative_offsets is None
            else self.decoder_clamps_negative_offsets[selected],
            self.known_source_stop,
        )


def bind_known_source_end(
    contract: RetainedRawWindowContract, source_stop: int
) -> RetainedRawWindowContract:
    """Declare a finite file end separately from retained control admission.

    The caller must establish the actual physical file extent. This clips
    decoded payload; it neither invents header/control bytes nor changes the
    supplied publication policy. Without this explicit fact, uncertain
    physical returns continue to be unresolved.
    """
    stop = _integer(source_stop, "known source stop", minimum=0)
    if stop > np.iinfo(np.int64).max:
        raise OverflowError("known source stop exceeds int64")
    if not isinstance(contract, RetainedRawWindowContract):
        raise TypeError("source-end binding requires retained raw windows")
    heads = contract.physical_head_samples
    if np.any(heads < 0):
        raise ValueError(
            "source-end binding cannot establish negative history"
        )
    remaining = np.maximum(stop - heads, 0)
    return replace(
        contract,
        decoded_extent_samples=np.minimum(
            contract.decoded_extent_samples, remaining
        ),
        known_source_stop=stop,
    )


def apply_uncovered_head_publication(
    contract: RetainedRawWindowContract,
    state: RawValidityState,
    *,
    extent: BufferedNormalFullExtent,
    sample_rate: int,
    fft_samples: int,
    publication_timestamp_quantum_nanoseconds: int | None = None,
) -> RetainedRawWindowContract:
    """Opt in to allocation-head coverage gating for a retained raw return.

    For allocations starting inside recorded filler, framed control bytes
    and decoded payload samples have distinct allocation extents. A preceding
    returned view can retain a leading FFT whose physical start lies in that
    difference, provided its actual decoded view still contains trustworthy
    payload. This can preserve a partial suffix or a full FFT after the hole.
    A wholly uncovered FFT is not published even when its start precedes the
    unavailable allocation head. Partial coverage remains sample-granular;
    explicit-invalid timestamps are not holes in this inventory.
    A fresh return at the unavailable head cannot retain the previous view.
    A preceding view whose own allocation head is unavailable cannot supply
    that exception either: trustworthy bytes later in that allocation do not
    establish a published predecessor return.
    The margin is derived from the
    supplied frame/header packing, never from an observed loss count.
    A preceding FFT keeps its ordinary partial payload validity.

    The caller must establish this publication contract for its reader. This
    does not classify returned windows, change source addresses or decoded
    masks, infer exposure, or enable a production profile. Initial/EOF and
    adjacent timestamp jumps, changing delay, or stale-ring combinations need
    separate acceptance.
    Timestamp-only jumps and explicit-invalid frames create no coverage hole.
    """
    if not isinstance(contract, RetainedRawWindowContract):
        raise TypeError("publication gating requires retained raw windows")
    if not isinstance(state, RawValidityState):
        raise TypeError("publication gating requires cached raw validity")
    if not isinstance(extent, BufferedNormalFullExtent):
        raise TypeError("publication gating requires a normal-full extent")
    size = _integer(fft_samples, "FFT samples", minimum=1)
    timeline = state.source_state.timeline
    request_state = state.source_state.request_state
    if (
        extent.request_frames * extent.read_buffer_ratio
        != request_state.allocation_frames
        or extent.samples_per_frame != request_state.samples_per_frame
    ):
        raise ValueError("publication extent differs from cached geometry")
    header_span = (
        request_state.allocation_frames
        * (extent.frame_bytes - extent.payload_bytes)
        * cast(Fraction, extent.samples_per_byte)
    )
    # Physical sample starts are integral. Comparing them strictly with a
    # rational framed end is equivalent to comparing with its ceiling.
    prefix_samples = -(-header_span.numerator // header_span.denominator)
    bounds = compile_raw_window_bounds(
        contract.output_samples,
        contract.aligned_head_samples,
        contract.centre_delays_seconds,
        contract.framed_extent_samples,
        sample_rate=sample_rate,
        fft_samples=size,
    )
    physical = _cursor_add(
        contract.physical_head_samples,
        np.maximum(bounds.nearest_samples, 0),
        "allocation-head physical FFT start",
    )
    ends = _cursor_add(physical, size, "allocation-head physical FFT stop")
    available_stop = (
        int(timeline.physical_stops[-1]) * request_state.samples_per_frame
    )
    published = contract.publication_eligible & ~bounds.missing
    file_stop = state.source_state.recorded_source_stop
    if file_stop is not None and file_stop < available_stop:
        raise ValueError("known file end excludes supplied physical history")
    # A finite physical file end establishes absence beyond EOF, including
    # partial FFTs. Sparse coverage below can withhold empty rows; the final
    # source-end binder clips decoded payload. Without that fact, retain the
    # strict rejection of requests extending outside known raw history.
    if np.any(physical < 0) or (
        file_stop is None and np.any(published & (ends > available_stop))
    ):
        raise ValueError("publication source lies outside cached raw history")
    # The start-owned allocation gate alone misses an FFT contained in a
    # physical hole that begins before the next allocation head. Find the
    # next trustworthy run with a sparse interval lookup, not a frame bitmap.
    physical_frames = physical // request_state.samples_per_frame
    next_valid = np.searchsorted(
        timeline.physical_stops, physical_frames, side="right"
    )
    covered = next_valid < timeline.physical_starts.size
    covered[covered] = (
        _cursor_multiply(
            timeline.physical_starts[next_valid[covered]],
            request_state.samples_per_frame,
            "next trustworthy sample start",
        )
        < ends[covered]
    )
    empty = published & ~covered
    intervals = timeline.uncovered_head_allocations()
    if not intervals:
        if not np.any(empty):
            return contract
        return replace(
            contract,
            publication_eligible=contract.publication_eligible & ~empty,
        )
    ownership_frames = physical_frames
    if publication_timestamp_quantum_nanoseconds is not None:
        quantum = _integer(
            publication_timestamp_quantum_nanoseconds,
            "publication timestamp quantum",
            minimum=1,
        )
        units = np.trunc(
            contract.centre_delays_seconds * (1_000_000_000 / quantum)
        )
        publication_bounds = compile_raw_window_bounds(
            contract.output_samples,
            contract.aligned_head_samples,
            units * (quantum / 1_000_000_000),
            contract.framed_extent_samples,
            sample_rate=sample_rate,
            fft_samples=size,
        )
        ownership = _cursor_add(
            contract.physical_head_samples,
            np.maximum(publication_bounds.nearest_samples, 0),
            "quantized publication ownership",
        )
        ownership_frames = ownership // request_state.samples_per_frame
    owners = ownership_frames // request_state.allocation_frames
    starts = np.asarray([first for first, _ in intervals], dtype=np.int64)
    stops = np.asarray([stop for _, stop in intervals], dtype=np.int64)
    positions = np.searchsorted(stops, owners, side="right")
    selected = positions < len(intervals)
    withheld = np.zeros(owners.shape, dtype=np.bool_)
    withheld[selected] = owners[selected] >= starts[positions[selected]]
    # A preceding view can extend beyond the decoded allocation head in its
    # framed control coordinates. Keep that boundary distinct from the
    # actual payload end; an eligibility margin cannot invent valid samples.
    candidates = np.flatnonzero(withheld & published)
    if candidates.size:
        head_frames = _cursor_multiply(
            owners[candidates],
            request_state.allocation_frames,
            "uncovered allocation frame head",
        )
        run_positions = np.searchsorted(
            timeline.physical_stops, head_frames, side="right"
        )
        head_samples = _cursor_multiply(
            head_frames,
            request_state.samples_per_frame,
            "uncovered allocation sample head",
        )
        resume_samples = _cursor_multiply(
            timeline.physical_starts[run_positions],
            request_state.samples_per_frame,
            "trustworthy head-hole resume sample",
        )
        framed_prefix_stop = _cursor_add(
            head_samples,
            prefix_samples,
            "framed head publication stop",
        )
        decoded_stop = _cursor_add(
            contract.physical_head_samples[candidates],
            contract.decoded_extent_samples[candidates],
            "retained publication decoded stop",
        )
        predecessor_owners = (
            contract.physical_head_samples[candidates]
            // request_state.samples_per_frame
            // request_state.allocation_frames
        )
        predecessor_positions = np.searchsorted(
            stops, predecessor_owners, side="right"
        )
        predecessor_uncovered = predecessor_positions < len(intervals)
        predecessor_uncovered[predecessor_uncovered] = (
            predecessor_owners[predecessor_uncovered]
            >= starts[predecessor_positions[predecessor_uncovered]]
        )
        retained_prefix = (
            (timeline.residuals[run_positions - 1] == 0)
            & ~predecessor_uncovered
            & (contract.physical_head_samples[candidates] < head_samples)
            & (physical[candidates] < framed_prefix_stop)
            & (
                np.maximum(physical[candidates], resume_samples)
                < np.minimum(ends[candidates], decoded_stop)
            )
        )
        withheld[candidates[retained_prefix]] = False
    return replace(
        contract,
        publication_eligible=contract.publication_eligible
        & ~withheld
        & ~empty,
    )


@dataclass(frozen=True, slots=True)
class RawReplayWindowFacts:
    """Sparse scoped facts and input extents for a common storage second pass.

    Physical extent is None when no accepted full FFT address exists. Unknown
    source decisions are preserved; compiling a plan rejects unresolved scope.
    Ordinary extents include all requested FFTs, including discarded ones.
    """

    first_fft: int
    stop_fft: int
    fft_samples: int
    addresses: LatchedRawSourceCompilation
    validity: RawValidityCompilation
    ineligible_intervals: tuple[tuple[int, int], ...]
    ordinary_gpu_source_spans: tuple[OrdinaryRawSourceSpan, ...]
    physical_sample_extent: tuple[int, int] | None
    ordinary_sample_extent: tuple[int, int]
    source_timing_events: tuple[tuple[int, int], ...] = ()

    @property
    def complete(self) -> bool:
        return self.addresses.complete and self.validity.complete

    def compile(
        self,
        storage: RawLinearStorage,
        *,
        maximum_empty_fft_events: int = 4096,
    ) -> RawReplayPlanWindow:
        window = compile_translated_raw_replay_window(
            storage,
            self.addresses,
            self.validity,
            first_fft=self.first_fft,
            stop_fft=self.stop_fft,
            fft_samples=self.fft_samples,
            ordinary_gpu_source_spans=self.ordinary_gpu_source_spans,
            ineligible_intervals=self.ineligible_intervals,
            maximum_empty_fft_events=maximum_empty_fft_events,
        )
        return RawReplayPlanWindow(
            window.first_fft,
            window.stop_fft,
            replace(
                window.plan_for_scope(self.first_fft, self.stop_fft),
                source_timing_events=self.source_timing_events,
            ),
        )


def compile_raw_replay_window_facts(
    validity_state: RawValidityState,
    request_fft_starts: np.ndarray,
    desired_sample_starts: np.ndarray,
    endpoint_drifts: np.ndarray,
    ordinary_source_starts: np.ndarray,
    geometry: BufferedRequestGeometry,
    *,
    first_fft: int,
    stop_fft: int,
    sample_rate: float,
    frame_bytes: int,
    control_word_bits: int,
    request_fft_counts: np.ndarray | None = None,
    window_clock_contract: RetainedRawWindowContract,
) -> RawReplayWindowFacts:
    """Compose cached raw state and actual complete requests, then clip scope.

    Supplied requests must cover the selected interval exactly once. Their
    leading FFT may precede first_fft; trimming inputs before compilation
    would alter request cursor semantics. No unknown prefix is synthesized.
    An explicit retained-window contract supplies addresses and admission.
    """
    if not isinstance(window_clock_contract, RetainedRawWindowContract):
        raise TypeError(
            "replay requires an explicit retained-window contract"
        )
    first = _integer(first_fft, "first FFT")
    stop = _integer(stop_fft, "stop FFT", minimum=1)
    if stop <= first:
        raise ValueError("selected FFT scope must be nonempty")
    if not isinstance(validity_state, RawValidityState):
        raise TypeError("validity state must be cached RawValidityState")
    discards = compile_buffered_request_discards(
        validity_state.source_state.request_state,
        request_fft_starts,
        desired_sample_starts,
        endpoint_drifts,
        ordinary_source_starts,
        geometry,
        sample_rate=sample_rate,
        frame_bytes=frame_bytes,
        control_word_bits=control_word_bits,
        request_fft_counts=request_fft_counts,
    )
    full = LatchedRawSourceCompilation((), ())
    n = geometry.fft_samples
    merged = []
    pending_first = None
    pending_stop = first
    pending_source = 0
    covered = first
    for row, start in enumerate(request_fft_starts):
        start = int(start)
        count = (
            ordinary_source_starts.shape[1]
            if request_fft_counts is None
            else int(request_fft_counts[row])
        )
        left, right = max(first, start), min(stop, start + count)
        if left >= right:
            continue
        if left != covered:
            raise ValueError(
                "requests must cover selected scope exactly once"
            )
        values = ordinary_source_starts[row, left - start : right - start]
        # Vectorize only the supplied bounded row. The monotonicity guard
        # prevents an int64 wrap from masquerading as a positive FFT step.
        edges = (
            np.flatnonzero(
                (np.diff(values) != n) | (values[1:] < values[:-1])
            )
            + 1
        )
        run_first = 0
        for run_stop in (*edges, values.size):
            run_stop = int(run_stop)
            run_first_fft = left + run_first
            run_stop_fft = left + run_stop
            run_source = int(values[run_first])
            # Accumulate affine runs across rows before constructing spans.
            # Python integers keep the cross-row comparison free of wrap.
            if (
                pending_first is not None
                and pending_stop == run_first_fft
                and pending_source + (run_first_fft - pending_first) * n
                == run_source
            ):
                pending_stop = run_stop_fft
            else:
                if pending_first is not None:
                    merged.append(
                        OrdinaryRawSourceSpan(
                            pending_first, pending_stop, pending_source
                        )
                    )
                pending_first = run_first_fft
                pending_stop = run_stop_fft
                pending_source = run_source
            run_first = run_stop
        covered = right
    if covered != stop:
        raise ValueError("requests must cover selected scope exactly once")
    if pending_first is not None:
        merged.append(
            OrdinaryRawSourceSpan(pending_first, pending_stop, pending_source)
        )
    spans = tuple(
        replace(
            span,
            first_fft=max(first, span.first_fft),
            stop_fft=min(stop, span.stop_fft),
            physical_sample_start=span.physical_sample_start
            + (max(first, span.first_fft) - span.first_fft) * n,
        )
        for span in full.spans
        if span.first_fft < stop and span.stop_fft > first
    )
    unresolved = tuple(
        replace(
            span,
            first_fft=max(first, span.first_fft),
            stop_fft=min(stop, span.stop_fft),
        )
        for span in full.unresolved
        if span.first_fft < stop and span.stop_fft > first
    )
    addresses = LatchedRawSourceCompilation(spans, unresolved)
    validity = split_latched_raw_source_validity(
        validity_state, addresses, fft_samples=n
    )
    all_rejected: list[tuple[int, int]] = []
    for left, right in sorted(discards.ineligible_intervals):
        if all_rejected and left <= all_rejected[-1][1]:
            previous, end = all_rejected[-1]
            all_rejected[-1] = (previous, max(end, right))
        else:
            all_rejected.append((left, right))
    rejected = tuple(
        (max(first, left), min(stop, right))
        for left, right in all_rejected
        if left < stop and right > first
    )
    physical_extent = (
        (
            min(span.physical_sample_start for span in spans),
            max(
                span.physical_sample_start
                + (span.stop_fft - span.first_fft) * n
                for span in spans
            ),
        )
        if spans
        else None
    )
    ordinary_extent = (
        min(span.ordinary_sample_start for span in merged),
        max(
            span.ordinary_sample_start + (span.stop_fft - span.first_fft) * n
            for span in merged
        ),
    )
    result = RawReplayWindowFacts(
        first,
        stop,
        n,
        addresses,
        validity,
        rejected,
        tuple(merged),
        physical_extent,
        ordinary_extent,
    )
    return _compile_retained_window_facts(
        result, validity_state, window_clock_contract, sample_rate=sample_rate
    )


def _compile_retained_window_facts(
    facts, validity_state, contract, *, sample_rate
):
    """Read only explicitly declared retained windows; keep EOF unknown."""
    if (facts.first_fft, facts.stop_fft) != (
        contract.first_fft,
        contract.stop_fft,
    ):
        raise ValueError("retained window contract must match facts scope")
    size = facts.fft_samples
    if (
        isinstance(sample_rate, bool)
        or not np.isfinite(sample_rate)
        or int(sample_rate) != sample_rate
    ):
        raise ValueError("window clocks require an integral sample rate")
    bounds = compile_raw_window_bounds(
        contract.output_samples,
        contract.aligned_head_samples,
        contract.centre_delays_seconds,
        contract.framed_extent_samples,
        sample_rate=int(sample_rate),
        fft_samples=size,
    )
    physical = _cursor_add(
        contract.physical_head_samples,
        np.maximum(bounds.nearest_samples, 0),
        "retained physical FFT start",
    )
    _cursor_add(physical, size, "retained physical FFT end")
    ends = _cursor_add(
        contract.physical_head_samples,
        contract.decoded_extent_samples,
        "retained physical window end",
    )
    source = validity_state.source_state
    source_stop = contract.known_source_stop
    if source_stop is None:
        unknown = (
            (contract.physical_head_samples < 0)
            | (ends > source.raw_source_stop)
            | (physical > source.raw_source_stop - size)
        )
        outside_source = None
    else:
        if source_stop < source.raw_source_stop:
            raise ValueError(
                "known file end excludes supplied physical history"
            )
        unknown = (contract.physical_head_samples < 0) | (
            (ends > source_stop) & (contract.decoded_extent_samples > 0)
        )
        outside_source = physical >= source_stop
    reason = source.unsupported_reason
    if reason == "noncontiguous raw input is unsupported":
        # Retained physical windows support earlier holes; a hole elsewhere
        # does not explain this window extending past the recorded payload.
        reason = None
    if reason:
        unknown[:] = True
    accepted = contract.publication_eligible & ~bounds.missing & ~unknown
    if outside_source is not None:
        accepted &= ~outside_source

    def intervals(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        return tuple(
            (facts.first_fft + int(a), facts.first_fft + int(b))
            for a, b in zip(edges[::2], edges[1::2], strict=True)
        )

    # The decoder's logical sample start determines the residual phase.
    # Raw physical heads may carry file/ring translations; they cannot be
    # used to infer a logical timing shift. Signed nearest rounding and
    # accepted negative-offset clamping both affect this logical coordinate.
    logical = _cursor_add(
        contract.aligned_head_samples,
        np.maximum(bounds.nearest_samples, 0),
        "retained logical FFT start",
    )
    coarse = np.rint(contract.centre_delays_seconds * sample_rate).astype(
        np.int64
    )
    ordinary = _cursor_subtract(
        contract.output_samples, coarse, "ordinary logical FFT start"
    )
    displacement = _cursor_subtract(
        logical, ordinary, "retained logical timing displacement"
    )
    timing_events = tuple(
        (facts.first_fft + int(k), int(displacement[k]))
        for k in np.flatnonzero(accepted & (displacement != 0))
    )
    indices = np.flatnonzero(accepted)
    spans = []
    if indices.size:
        values = physical[indices]
        edges = np.flatnonzero(
            (np.diff(indices) != 1)
            | (np.diff(values) != size)
            | (values[1:] < values[:-1])
        )
        firsts = np.r_[0, edges + 1]
        lasts = np.r_[edges, indices.size - 1]
        spans = [
            LatchedRawSourceSpan(
                facts.first_fft + int(indices[a]),
                facts.first_fft + int(indices[b]) + 1,
                int(values[a]),
            )
            for a, b in zip(firsts, lasts, strict=True)
        ]
    addresses = LatchedRawSourceCompilation(
        tuple(spans),
        tuple(
            UnresolvedRawSourceSpan(
                a, b, reason or "retained window exceeds recorded payload"
            )
            for a, b in intervals(unknown)
        ),
    )
    validity = split_latched_raw_source_validity(
        validity_state, addresses, fft_samples=size
    )
    result = replace(
        facts,
        addresses=addresses,
        validity=validity,
        ineligible_intervals=intervals(~accepted & ~unknown),
        source_timing_events=timing_events,
        physical_sample_extent=(
            (
                int(physical[indices].min()),
                min(int(physical[indices].max()) + size, source_stop)
                if source_stop is not None
                else int(physical[indices].max()) + size,
            )
            if indices.size
            else None
        ),
    )
    return apply_raw_window_clock_contract(
        result, contract, sample_rate=sample_rate
    )


def _subtract_fft_intervals(first, stop, rejected, rejected_stops):
    """Subtract sparse ordered intervals without expanding a source run."""
    cursor = first
    for position in range(bisect_right(rejected_stops, first), len(rejected)):
        left, right = rejected[position]
        if right <= cursor:
            continue
        if left >= stop:
            break
        if cursor < left:
            yield cursor, min(left, stop)
        cursor = max(cursor, right)
    if cursor < stop:
        yield cursor, stop


def _merge_fft_intervals(intervals):
    merged = []
    for first, stop in sorted(intervals):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((first, stop))
    return tuple(merged)


def apply_raw_window_clock_contract(
    facts: RawReplayWindowFacts,
    contract: RawWindowClockContract,
    *,
    sample_rate: int,
) -> RawReplayWindowFacts:
    """Compose explicit framed admission and decoded masks with scoped facts.

    Rejected known rows lose addresses and validity and become ineligible.
    Unknown source rows remain unresolved, even if window admission rejects
    them; window operands cannot establish physical source publication.
    Masks intersect supplied physical validity, never replace its provenance.
    """
    if not isinstance(facts, RawReplayWindowFacts):
        raise TypeError("facts must be RawReplayWindowFacts")
    if not isinstance(contract, RawWindowClockContract):
        raise TypeError("window contract must be RawWindowClockContract")
    if (facts.first_fft, facts.stop_fft) != (
        contract.first_fft,
        contract.stop_fft,
    ):
        raise ValueError("window clock contract must match the facts scope")
    size = facts.fft_samples
    if (
        isinstance(sample_rate, bool)
        or not np.isfinite(sample_rate)
        or int(sample_rate) != sample_rate
    ):
        raise ValueError("window clocks require an integral sample rate")
    bounds = compile_raw_window_bounds(
        contract.output_samples,
        contract.aligned_head_samples,
        contract.centre_delays_seconds,
        contract.framed_extent_samples,
        sample_rate=int(sample_rate),
        fft_samples=size,
    )
    decoder_offsets = bounds.nearest_samples
    if (
        isinstance(contract, RetainedRawWindowContract)
        and contract.decoder_clamps_negative_offsets is not None
    ):
        decoder_offsets = np.where(
            contract.decoder_clamps_negative_offsets,
            np.maximum(decoder_offsets, 0),
            decoder_offsets,
        )
    payload = compile_raw_window_payload_spans(
        decoder_offsets,
        contract.decoded_extent_samples,
        fft_samples=size,
    )
    if not np.any(bounds.missing) and not np.any(
        (payload.first_samples != 0) | (payload.stop_samples != size)
    ):
        return facts
    # Only the explicitly bounded contract is dense. Runs remain sparse.
    edges = np.flatnonzero(np.diff(np.r_[False, bounds.missing, False]))
    rejected = tuple(
        (facts.first_fft + int(first), facts.first_fft + int(stop))
        for first, stop in zip(edges[::2], edges[1::2], strict=True)
    )
    rejected_stops = tuple(stop for _, stop in rejected)
    spans = tuple(
        replace(
            span,
            first_fft=first,
            stop_fft=stop,
            physical_sample_start=span.physical_sample_start
            + (first - span.first_fft) * size,
        )
        for span in facts.addresses.spans
        for first, stop in _subtract_fft_intervals(
            span.first_fft, span.stop_fft, rejected, rejected_stops
        )
    )
    full = tuple(
        replace(
            span,
            first_fft=first,
            stop_fft=stop,
            physical_sample_start=span.physical_sample_start
            + (first - span.first_fft) * size,
        )
        for span in facts.validity.full_spans
        for first, stop in _subtract_fft_intervals(
            span.first_fft, span.stop_fft, rejected, rejected_stops
        )
    )
    partial = tuple(
        event
        for event in facts.validity.partial_events
        if not bounds.missing[event.fft_index - facts.first_fft]
    )
    empty = tuple(
        interval
        for first, stop in facts.validity.empty_intervals
        for interval in _subtract_fft_intervals(
            first, stop, rejected, rejected_stops
        )
    )
    validity = RawValidityCompilation(
        full, partial, empty, facts.validity.unresolved
    )
    mask_rows = (
        (payload.first_samples != 0) | (payload.stop_samples != size)
    ) & ~bounds.missing
    # Only known accepted addresses can receive decoder validity. Omitted
    # unresolved and already discarded rows preserve their original status.
    resolved = np.zeros(mask_rows.shape, dtype=np.bool_)
    for span in spans:
        resolved[
            span.first_fft - facts.first_fft : span.stop_fft - facts.first_fft
        ] = True
    selected = np.flatnonzero(mask_rows & resolved)
    validity = intersect_raw_decoder_validity(
        validity,
        selected.astype(np.int64) + facts.first_fft,
        RawWindowPayloadSpans(
            payload.first_samples[selected], payload.stop_samples[selected]
        ),
        fft_samples=size,
    )
    physical = (
        (
            min(span.physical_sample_start for span in spans),
            max(
                span.physical_sample_start
                + (span.stop_fft - span.first_fft) * size
                for span in spans
            ),
        )
        if spans
        else None
    )
    if (
        physical is not None
        and isinstance(contract, RetainedRawWindowContract)
        and contract.known_source_stop is not None
    ):
        physical = (physical[0], min(physical[1], contract.known_source_stop))
    return replace(
        facts,
        addresses=LatchedRawSourceCompilation(
            spans, facts.addresses.unresolved
        ),
        validity=validity,
        ineligible_intervals=_merge_fft_intervals(
            (*facts.ineligible_intervals, *rejected)
        ),
        physical_sample_extent=physical,
    )
