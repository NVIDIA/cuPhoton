# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader validity: input-derived compatibility planning."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from ._checks import _integer
from .addressing import (
    LatchedRawSourceCompilation,
    LatchedRawSourceSpan,
    LatchedRawSourceState,
    UnresolvedRawSourceSpan,
)
from .clocks import RawWindowPayloadSpans
from .geometry import (
    RawBufferedRequestState,
)
from .timeline import PhysicalAllocationTimeline, PhysicalSampleSpan


@dataclass(frozen=True, slots=True)
class RawPartialSourceEvent:
    """An accepted FFT's valid local intervals and raw physical identities."""

    fft_index: int
    spans: tuple[PhysicalSampleSpan, ...]


@dataclass(frozen=True, slots=True)
class RawValidityCompilation:
    full_spans: tuple[LatchedRawSourceSpan, ...]
    partial_events: tuple[RawPartialSourceEvent, ...]
    empty_intervals: tuple[tuple[int, int], ...]
    unresolved: tuple[UnresolvedRawSourceSpan, ...]

    @property
    def complete(self) -> bool:
        return not self.unresolved


@dataclass(frozen=True, slots=True)
class RawValidityState:
    """Supplied physical coverage indexed once for many bounded batches.

    Timeline coverage may include explicit-invalid frames whose payload must
    still be suppressed by the ordinary input frame-validity channel. This
    inventory is not an authorization to bypass that channel.
    """

    source_state: LatchedRawSourceState
    physical_valid_spans: tuple[tuple[int, int], ...]
    _valid_stops: tuple[int, ...]

    @classmethod
    def from_timeline(
        cls,
        timeline: PhysicalAllocationTimeline,
        request_state: RawBufferedRequestState,
    ) -> RawValidityState:
        """Cache the supplied timeline coverage without scanning frame states.

        Schema-v4 physical timestamp runs omit recorded filler and retain
        explicit-invalid timestamps. Their intervals can therefore cover
        invalid payload. The resulting masks identify holes in that coverage;
        ordinary ingest/execution frame validity remains required separately.
        """
        source = LatchedRawSourceState.from_timeline(timeline, request_state)
        samples = request_state.samples_per_frame
        spans: list[tuple[int, int]] = []
        for first, last in zip(
            timeline.physical_starts, timeline.physical_stops, strict=True
        ):
            first, last = int(first) * samples, int(last) * samples
            if spans and spans[-1][1] == first:
                spans[-1] = (spans[-1][0], last)
            else:
                spans.append((first, last))
        return cls(source, tuple(spans), tuple(last for _, last in spans))


def _merge_intervals(
    rows: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for first, last in sorted(rows):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    return tuple(merged)


def split_latched_raw_source_validity(
    state: RawValidityState,
    addresses: LatchedRawSourceCompilation,
    *,
    fft_samples: int,
) -> RawValidityCompilation:
    """Intersect compact latched addresses with valid physical raw intervals.

    Address runs are explicit ownership facts, not an authorization to read.
    Each source interval emits one compact full-FFT run plus at most two
    partial boundary events. Multiple holes can yield disjoint spans within
    one FFT. Whole empty FFTs stay intervals; no per-sample or per-FFT scan
    bitmap is created. Unresolved address decisions survive unchanged.
    """
    if not isinstance(state, RawValidityState):
        raise TypeError("state must be a RawValidityState")
    if not isinstance(addresses, LatchedRawSourceCompilation):
        raise TypeError("addresses must be a LatchedRawSourceCompilation")
    size = _integer(fft_samples, "FFT samples", minimum=1)
    full: list[LatchedRawSourceSpan] = []
    partial: dict[int, list[PhysicalSampleSpan]] = {}
    address_intervals: list[tuple[int, int]] = []
    previous_stop = 0
    for address in addresses.spans:
        if (
            address.first_fft < previous_stop
            or address.stop_fft <= address.first_fft
        ):
            raise ValueError(
                "address runs must be ordered and nonoverlapping"
            )
        previous_stop = address.stop_fft
        address_intervals.append((address.first_fft, address.stop_fft))
        first_sample = address.physical_sample_start
        if first_sample < 0:
            raise ValueError("raw physical address must be nonnegative")
        last_sample = (
            first_sample + (address.stop_fft - address.first_fft) * size
        )
        index = bisect_right(state._valid_stops, first_sample)
        while index < len(state.physical_valid_spans):
            source_first, source_last = state.physical_valid_spans[index]
            if source_first >= last_sample:
                break
            left = max(first_sample, source_first)
            right = min(last_sample, source_last)
            if left < right:
                local_left = left - first_sample
                local_right = right - first_sample
                first_full = -(-local_left // size)
                stop_full = local_right // size
                if first_full < stop_full:
                    full.append(
                        LatchedRawSourceSpan(
                            address.first_fft + first_full,
                            address.first_fft + stop_full,
                            first_sample + first_full * size,
                        )
                    )
                boundary_ffts = {
                    local_left // size,
                    (local_right - 1) // size,
                }
                for local_fft in boundary_ffts:
                    if first_full <= local_fft < stop_full:
                        continue
                    fft_start = local_fft * size
                    start = max(local_left, fft_start) - fft_start
                    stop = min(local_right, fft_start + size) - fft_start
                    event_index = address.first_fft + local_fft
                    partial.setdefault(event_index, []).append(
                        PhysicalSampleSpan(
                            start, stop, first_sample + fft_start + start
                        )
                    )
            index += 1
    merged_full: list[LatchedRawSourceSpan] = []
    for span in sorted(full, key=lambda span: span.first_fft):
        if merged_full and (
            merged_full[-1].stop_fft == span.first_fft
            and merged_full[-1].physical_sample_start
            + (span.first_fft - merged_full[-1].first_fft) * size
            == span.physical_sample_start
        ):
            previous = merged_full[-1]
            merged_full[-1] = LatchedRawSourceSpan(
                previous.first_fft,
                span.stop_fft,
                previous.physical_sample_start,
            )
        else:
            merged_full.append(span)
    events = tuple(
        RawPartialSourceEvent(
            index, tuple(sorted(spans, key=lambda span: span.request_start))
        )
        for index, spans in sorted(partial.items())
    )
    occupied = _merge_intervals(
        [(span.first_fft, span.stop_fft) for span in merged_full]
        + [(event.fft_index, event.fft_index + 1) for event in events]
    )
    empty: list[tuple[int, int]] = []
    position = 0
    for first, last in address_intervals:
        cursor = first
        while position < len(occupied) and occupied[position][1] <= cursor:
            position += 1
        selected = position
        while selected < len(occupied) and occupied[selected][0] < last:
            start, stop = occupied[selected]
            if cursor < start:
                empty.append((cursor, min(start, last)))
            cursor = max(cursor, stop)
            selected += 1
        if cursor < last:
            empty.append((cursor, last))
    return RawValidityCompilation(
        tuple(merged_full),
        events,
        _merge_intervals(empty),
        addresses.unresolved,
    )


def intersect_raw_decoder_validity(
    validity: RawValidityCompilation,
    fft_indices: np.ndarray,
    decoder_spans: RawWindowPayloadSpans,
    *,
    fft_samples: int,
) -> RawValidityCompilation:
    """Intersect sparse explicit decoder spans with existing raw validity.

    Indices are sorted, unique global FFT indices. Omitted rows retain their
    validity; supplied rows must belong to known full, partial or empty scope.
    Supplied unresolved rows reject; omitted unresolved rows survive
    unchanged. Physical identities and ordinary frame-validity obligations
    survive clipping. This supplies neither eligibility nor exposure policy.

    Work scales with validity runs/events and supplied decoder rows, never the
    number of FFTs in a full run. Callers can omit all full decoder spans.
    """
    if not isinstance(validity, RawValidityCompilation):
        raise TypeError("validity must be a RawValidityCompilation")
    if not isinstance(decoder_spans, RawWindowPayloadSpans):
        raise TypeError("decoder spans must be RawWindowPayloadSpans")
    size = _integer(fft_samples, "FFT samples", minimum=1)
    for array, name in (
        (fft_indices, "FFT indices"),
        (decoder_spans.first_samples, "decoder first samples"),
        (decoder_spans.stop_samples, "decoder stop samples"),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.ndim != 1
            or array.dtype != np.dtype(np.int64)
        ):
            raise TypeError(f"{name} must be an int64 vector")
        if array.shape != fft_indices.shape:
            raise ValueError("decoder spans must match FFT indices")
    if np.any(fft_indices < 0) or np.any(fft_indices[1:] <= fft_indices[:-1]):
        raise ValueError("FFT indices must be nonnegative, sorted and unique")
    firsts, stops = decoder_spans.first_samples, decoder_spans.stop_samples
    if np.any(firsts < 0) or np.any(stops < firsts) or np.any(stops > size):
        raise ValueError("decoder spans must lie within the FFT")
    partitions = sorted(
        [(span.first_fft, span.stop_fft) for span in validity.full_spans]
        + [
            (event.fft_index, event.fft_index + 1)
            for event in validity.partial_events
        ]
        + list(validity.empty_intervals)
        + [(span.first_fft, span.stop_fft) for span in validity.unresolved]
    )
    previous_stop = 0
    for first, stop in partitions:
        if first < previous_stop or stop <= first:
            raise ValueError(
                "validity partitions must be nonnegative and nonoverlapping"
            )
        previous_stop = stop
    coverage = _merge_intervals(
        [(span.first_fft, span.stop_fft) for span in validity.full_spans]
        + [
            (event.fft_index, event.fft_index + 1)
            for event in validity.partial_events
        ]
        + list(validity.empty_intervals)
    )
    coverage_stops = tuple(stop for _, stop in coverage)
    masks: dict[int, tuple[int, int]] = {}
    for index, first, stop in zip(fft_indices, firsts, stops, strict=True):
        index, first, stop = int(index), int(first), int(stop)
        position = bisect_right(coverage_stops, index)
        if position == len(coverage) or index < coverage[position][0]:
            raise ValueError(
                "decoder FFT lies outside resolved validity scope"
            )
        if first != 0 or stop != size:
            masks[index] = (first, stop)
    if not masks:
        return validity
    indices = tuple(masks)
    full: list[LatchedRawSourceSpan] = []
    partial: list[RawPartialSourceEvent] = []
    empty = list(validity.empty_intervals)
    for span in validity.full_spans:
        cursor = span.first_fft
        position = bisect_right(indices, cursor - 1)
        while position < len(indices) and indices[position] < span.stop_fft:
            index = indices[position]
            if cursor < index:
                full.append(
                    LatchedRawSourceSpan(
                        cursor,
                        index,
                        span.physical_sample_start
                        + (cursor - span.first_fft) * size,
                    )
                )
            first, stop = masks[index]
            if first < stop:
                partial.append(
                    RawPartialSourceEvent(
                        index,
                        (
                            PhysicalSampleSpan(
                                first,
                                stop,
                                span.physical_sample_start
                                + (index - span.first_fft) * size
                                + first,
                            ),
                        ),
                    )
                )
            else:
                empty.append((index, index + 1))
            cursor = index + 1
            position += 1
        if cursor < span.stop_fft:
            full.append(
                LatchedRawSourceSpan(
                    cursor,
                    span.stop_fft,
                    span.physical_sample_start
                    + (cursor - span.first_fft) * size,
                )
            )
    for event in validity.partial_events:
        mask = masks.get(event.fft_index)
        if mask is None:
            partial.append(event)
            continue
        first, stop = mask
        selected = []
        for partial_span in event.spans:
            left, right = (
                max(first, partial_span.request_start),
                min(stop, partial_span.request_stop),
            )
            if left < right:
                selected.append(
                    PhysicalSampleSpan(
                        left,
                        right,
                        partial_span.physical_start
                        + left
                        - partial_span.request_start,
                    )
                )
        if selected:
            partial.append(
                RawPartialSourceEvent(event.fft_index, tuple(selected))
            )
        else:
            empty.append((event.fft_index, event.fft_index + 1))
    return RawValidityCompilation(
        tuple(full),
        tuple(sorted(partial, key=lambda event: event.fft_index)),
        _merge_intervals(empty),
        validity.unresolved,
    )
