# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader timeline: input-derived compatibility planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._checks import BufferedWindowError, _integer


@dataclass(frozen=True, slots=True)
class PhysicalSampleSpan:
    """A request-local valid span with its physical source sample identity."""

    request_start: int
    request_stop: int
    physical_start: int


@dataclass(frozen=True, slots=True)
class BufferedAnchorGap:
    """Timestamp interval exposed by a deferred allocation-anchor change.

    This is a reader-coordinate transition, not a raw source hole. Samples
    read under the preceding anchor can therefore survive a timestamp jump
    before this interval begins. Source validity must be handled separately.
    """

    allocation_index: int
    physical_boundary_frame: int
    timestamp_start: int
    timestamp_stop: int

    def __post_init__(self) -> None:
        for name in (
            "allocation_index",
            "physical_boundary_frame",
            "timestamp_start",
            "timestamp_stop",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        if self.timestamp_stop <= self.timestamp_start:
            raise ValueError("anchor gap must be a forward nonempty interval")

    def fft_window(
        self,
        *,
        timestamp_origin: int,
        samples_per_frame: int,
        fft_samples: int,
        source_sample_offset: int = 0,
    ) -> tuple[int, int]:
        """Project the start-owned interval onto an integer FFT grid.

        The first excluded FFT starts at or beyond the old anchor's stop;
        the first retained FFT starts at or beyond the new anchor. This does
        not reject a preceding FFT merely because its tail crosses the edge.
        Offset is an independently supplied integer decoder displacement.
        A varying delay must be projected per request, not approximated here.
        """
        timestamp_origin = _integer(timestamp_origin, "timestamp origin")
        samples_per_frame = _integer(
            samples_per_frame, "samples per frame", minimum=1
        )
        fft_samples = _integer(fft_samples, "FFT samples", minimum=1)
        if not isinstance(
            source_sample_offset, (int, np.integer)
        ) or isinstance(source_sample_offset, (bool, np.bool_)):
            raise TypeError("source sample offset must be an integer")
        offset = int(source_sample_offset)
        start = (
            self.timestamp_start - timestamp_origin
        ) * samples_per_frame - offset
        stop = (
            self.timestamp_stop - timestamp_origin
        ) * samples_per_frame - offset
        return -(-start // fft_samples), -(-stop // fft_samples)


@dataclass(frozen=True, slots=True)
class PhysicalDiscontinuity:
    physical_start: int
    physical_gap_frames: int
    timestamp_gap_frames: int
    residual_frames: int
    allocation_index: int
    allocation_phase: int


@dataclass(frozen=True, slots=True)
class PhysicalAllocationTimeline:
    allocation_starts: np.ndarray
    anchors: np.ndarray
    timestamp_stops: np.ndarray
    physical_gaps: np.ndarray
    timestamp_gaps: np.ndarray
    residuals: np.ndarray
    physical_starts: np.ndarray
    physical_stops: np.ndarray
    timestamp_starts: np.ndarray
    _allocation_frames: int

    @property
    def discontinuity_count(self) -> int:
        """Count sparse transitions without allocating per-event objects."""
        return int(
            np.count_nonzero(
                (self.physical_gaps != 0) | (self.residuals != 0)
            )
        )

    @property
    def discontinuities(self) -> tuple[PhysicalDiscontinuity, ...]:
        """Materialize transition records only when a policy needs them."""
        # The allocation width is fixed even for a short final allocation.
        capacity = int(self._allocation_frames)
        return tuple(
            PhysicalDiscontinuity(
                int(self.physical_starts[i + 1]),
                int(self.physical_gaps[i]),
                int(self.timestamp_gaps[i]),
                int(self.residuals[i]),
                int(self.physical_starts[i + 1]) // capacity,
                int(self.physical_starts[i + 1]) % capacity,
            )
            for i in np.flatnonzero(
                (self.physical_gaps != 0) | (self.residuals != 0)
            )
        )

    def uncovered_head_allocations(self) -> tuple[tuple[int, int], ...]:
        """Index allocation heads lying in uncovered physical frame slots.

        This is input coverage, not timestamp residuals or an automatic
        publication policy. Schema-v4 clock coverage retains explicit-invalid
        frames; their payload validity is still handled separately. A hole
        ending exactly at a head leaves that head covered. Results are compact
        allocation-index intervals, with no frame or allocation bitmap.
        """
        capacity = int(self._allocation_frames)
        intervals: list[tuple[int, int]] = []
        for position in np.flatnonzero(self.physical_gaps):
            first = -(-int(self.physical_stops[position]) // capacity)
            stop = -(-int(self.physical_starts[position + 1]) // capacity)
            if first >= stop:
                continue
            if intervals and first <= intervals[-1][1]:
                intervals[-1] = (
                    intervals[-1][0],
                    max(stop, intervals[-1][1]),
                )
            else:
                intervals.append((first, stop))
        return tuple(intervals)

    def deferred_anchor_gaps(self) -> tuple[BufferedAnchorGap, ...]:
        """Derive forward anchor jumps at physical refill boundaries.

        Allocation anchors are established from source runs. Comparing each
        successor with the preceding anchor's expected stop defers changes
        inside a read to the next physical refill. Multiple raw timestamp
        events in one read collapse into its net anchor change. No per-FFT or
        per-frame table is materialized, and a filler-only hole creates no
        timestamp-anchor gap.
        """
        if self.allocation_starts.size < 2:
            return ()
        widths = self.allocation_starts[1:] - self.allocation_starts[:-1]
        expected = self.anchors[:-1] + widths
        if np.any(expected < self.anchors[:-1]):
            raise BufferedWindowError(
                "allocation anchor stop exceeds signed range"
            )
        changed = np.flatnonzero(self.anchors[1:] > expected)
        return tuple(
            BufferedAnchorGap(
                int(index) + 1,
                int(self.allocation_starts[index + 1]),
                int(expected[index]),
                int(self.anchors[index + 1]),
            )
            for index in changed
        )

    def sample_source_spans(
        self,
        timestamp_sample_start: int,
        sample_count: int,
        *,
        samples_per_frame: int,
    ) -> tuple[PhysicalSampleSpan, ...]:
        """Split a sample request at every physical/timestamp discontinuity.

        Returned offsets preserve valid samples on either side of a hole;
        absent timestamps and filler slots produce no source span. This is
        physical provenance only, not reader eligibility or reported weight.
        Work is bounded by the number of overlapping compact runs, not by
        the number of frames or samples in the request.
        """
        timestamp_sample_start = _integer(
            timestamp_sample_start, "timestamp sample start"
        )
        sample_count = _integer(sample_count, "sample count", minimum=1)
        samples_per_frame = _integer(
            samples_per_frame, "samples per frame", minimum=1
        )
        sample_stop = int(timestamp_sample_start) + int(sample_count)
        position = int(
            np.searchsorted(
                self.timestamp_stops,
                timestamp_sample_start // samples_per_frame,
                side="right",
            )
        )
        spans: list[PhysicalSampleSpan] = []
        while position < len(self.timestamp_starts):
            timestamp_start = (
                int(self.timestamp_starts[position]) * samples_per_frame
            )
            if timestamp_start >= sample_stop:
                break
            timestamp_stop = (
                int(self.timestamp_stops[position]) * samples_per_frame
            )
            start = max(int(timestamp_sample_start), timestamp_start)
            stop = min(sample_stop, timestamp_stop)
            if start < stop:
                spans.append(
                    PhysicalSampleSpan(
                        start - timestamp_sample_start,
                        stop - timestamp_sample_start,
                        int(self.physical_starts[position])
                        * samples_per_frame
                        + start
                        - timestamp_start,
                    )
                )
            position += 1
        return tuple(spans)

    def latched_source_spans(
        self,
        allocation_index: int,
        timestamp_sample_start: int,
        sample_count: int,
        *,
        samples_per_frame: int,
    ) -> tuple[PhysicalSampleSpan, ...]:
        """Resolve raw physical payload under one latched allocation anchor.

        A timestamp change inside an already returned allocation does not
        move its payload. The successor's anchor establishes the subsequent
        displacement. Physical filler holes remain absent source spans.
        This method describes provenance only: it neither authorizes reads
        outside a returned buffer nor assigns reported exposure. The caller
        must resolve reader eligibility and ring generation independently.
        """
        allocation_index = _integer(allocation_index, "allocation index")
        if allocation_index >= self.allocation_starts.size:
            raise ValueError("allocation index exceeds the physical timeline")
        timestamp_sample_start = _integer(
            timestamp_sample_start, "timestamp sample start"
        )
        sample_count = _integer(sample_count, "sample count", minimum=1)
        samples_per_frame = _integer(
            samples_per_frame, "samples per frame", minimum=1
        )
        physical_start = (
            int(self.allocation_starts[allocation_index]) * samples_per_frame
            + timestamp_sample_start
            - int(self.anchors[allocation_index]) * samples_per_frame
        )
        physical_stop = physical_start + sample_count
        position = int(
            np.searchsorted(
                self.physical_stops,
                physical_start // samples_per_frame,
                side="right",
            )
        )
        spans: list[PhysicalSampleSpan] = []
        while position < self.physical_starts.size:
            run_start = (
                int(self.physical_starts[position]) * samples_per_frame
            )
            if run_start >= physical_stop:
                break
            run_stop = int(self.physical_stops[position]) * samples_per_frame
            start = max(physical_start, run_start)
            stop = min(physical_stop, run_stop)
            if start < stop:
                span = PhysicalSampleSpan(
                    start - physical_start, stop - physical_start, start
                )
                if (
                    spans
                    and spans[-1].request_stop == span.request_start
                    and spans[-1].physical_start
                    + spans[-1].request_stop
                    - spans[-1].request_start
                    == span.physical_start
                ):
                    previous = spans[-1]
                    spans[-1] = PhysicalSampleSpan(
                        previous.request_start,
                        span.request_stop,
                        previous.physical_start,
                    )
                else:
                    spans.append(span)
            position += 1
        return tuple(spans)

    @classmethod
    def from_runs(
        cls,
        physical_starts: np.ndarray,
        physical_stops: np.ndarray,
        timestamp_starts: np.ndarray,
        *,
        allocation_frames: int,
    ) -> PhysicalAllocationTimeline:
        allocation_frames = _integer(
            allocation_frames, "allocation frame count", minimum=1
        )
        arrays = (physical_starts, physical_stops, timestamp_starts)
        if any(
            not isinstance(a, np.ndarray)
            or a.ndim != 1
            or a.dtype.kind not in "iu"
            for a in arrays
        ):
            raise BufferedWindowError(
                "timestamp coordinates must be integer vectors"
            )
        if not len(physical_starts) or any(
            a.shape != physical_starts.shape for a in arrays
        ):
            raise BufferedWindowError(
                "timestamp vectors must be nonempty and equal length"
            )
        if any(np.any(a < 0) for a in arrays):
            raise BufferedWindowError(
                "timestamp coordinates must be nonnegative"
            )
        if int(physical_starts[0]) != 0:
            raise BufferedWindowError(
                "physical allocation phase requires a zero-based source"
            )
        if np.any(physical_stops <= physical_starts):
            raise BufferedWindowError(
                "physical timestamp runs must be nonempty"
            )
        if np.any(physical_starts[1:] < physical_stops[:-1]):
            raise BufferedWindowError(
                "physical timestamp runs overlap or are unordered"
            )
        if any(np.any(a > np.iinfo(np.int64).max) for a in arrays):
            raise BufferedWindowError(
                "timestamp coordinates exceed signed range"
            )
        # Arithmetic is signed: a backwards timestamp must not wrap to a gap.
        starts, stops, timestamps = (
            a.astype(np.int64, copy=False) for a in arrays
        )
        timestamp_stops = timestamps + (stops - starts)
        if np.any(timestamp_stops < timestamps):
            raise BufferedWindowError("timestamp span exceeds signed range")
        physical_gaps = starts[1:] - stops[:-1]
        timestamp_gaps = timestamps[1:] - timestamp_stops[:-1]
        residuals = timestamp_gaps - physical_gaps
        if np.any(residuals < 0):
            raise BufferedWindowError(
                "physical timeline has a backwards timestamp residual"
            )
        allocation_starts = np.arange(
            0, int(stops[-1]), allocation_frames, dtype=np.int64
        )
        positions = np.searchsorted(stops, allocation_starts, side="right")
        positions = np.minimum(positions, len(starts) - 1)
        # Inside a run this is its ordinary affine timestamp. In a physical
        # hole it extrapolates back from the successor run, preserving its
        # residual instead of silently filling the timestamp discontinuity.
        anchors = (
            timestamps[positions] + allocation_starts - starts[positions]
        )
        return cls(
            allocation_starts,
            anchors,
            timestamp_stops,
            physical_gaps,
            timestamp_gaps,
            residuals,
            starts.copy(),
            stops.copy(),
            timestamps.copy(),
            allocation_frames,
        )
