# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable host-side plans for calibrated missing-timestamp recovery.

Loss counts are useful calibration evidence, but they do not identify which
FFTs remain eligible.  This module therefore keeps observations separate from
exact calibration artifacts and refuses to construct a recovery plan unless
the complete missing-event set has an exact mask or rule.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


class MissingTimestampRecoveryError(ValueError):
    """Base class for invalid or unavailable recovery plans."""


class UncalibratedRecoveryError(MissingTimestampRecoveryError):
    """Raised when no exact calibration covers the requested event set."""


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _events(values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError("missing_timestamps must be a tuple")
    for value in values:
        _integer(value, "missing timestamp")
    if tuple(sorted(values)) != values:
        raise ValueError("missing_timestamps must be strictly increasing")
    if len(set(values)) != len(values):
        raise ValueError("missing_timestamps must not contain duplicates")
    return values


@dataclass(frozen=True, slots=True)
class FFTLossObservation:
    """Non-exact evidence from one calibration run.

    These counts intentionally cannot be converted into an eligibility plan:
    neither the rejected FFT identities nor interaction behavior are known.
    """

    missing_timestamps: tuple[int, ...]
    lost_fft_count: int

    def __post_init__(self) -> None:
        _events(self.missing_timestamps)
        _integer(self.lost_fft_count, "lost FFT count")


@dataclass(frozen=True, slots=True)
class IntegrationFFTLossObservation:
    """Observed per-integration losses without FFT identity information."""

    missing_timestamps: tuple[int, ...]
    lost_ffts_by_integration: tuple[int, ...]

    def __post_init__(self) -> None:
        _events(self.missing_timestamps)
        if not isinstance(self.lost_ffts_by_integration, tuple):
            raise TypeError("lost_ffts_by_integration must be a tuple")
        if not self.lost_ffts_by_integration:
            raise ValueError("lost_ffts_by_integration must not be empty")
        for loss in self.lost_ffts_by_integration:
            _integer(loss, "lost FFT count")


@dataclass(frozen=True, slots=True)
class ExactFFTEligibilityMask:
    """An exact global eligibility mask for one complete event set."""

    missing_timestamps: tuple[int, ...]
    eligible_ffts: tuple[bool, ...]

    def __post_init__(self) -> None:
        _events(self.missing_timestamps)
        if not isinstance(self.eligible_ffts, tuple):
            raise TypeError("eligible_ffts must be a tuple")
        if not all(isinstance(value, bool) for value in self.eligible_ffts):
            raise TypeError("eligible_ffts entries must be bool")


@dataclass(frozen=True, slots=True)
class ExactFFTEligibilityRule:
    """A compact exact rule rejecting named global FFT indices.

    A rule applies only to the complete, exact event tuple it names.  It is
    not combined with rules for subsets, so an uncalibrated transition cannot
    accidentally inherit additive behavior from isolated events.
    """

    missing_timestamps: tuple[int, ...]
    ineligible_fft_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _events(self.missing_timestamps)
        if not isinstance(self.ineligible_fft_indices, tuple):
            raise TypeError("ineligible_fft_indices must be a tuple")
        for index in self.ineligible_fft_indices:
            _integer(index, "ineligible FFT index")
        if tuple(sorted(self.ineligible_fft_indices)) != (
            self.ineligible_fft_indices
        ):
            raise ValueError("ineligible_fft_indices must be increasing")
        if len(set(self.ineligible_fft_indices)) != len(
            self.ineligible_fft_indices
        ):
            raise ValueError(
                "ineligible_fft_indices must not contain duplicates"
            )


type ExactCalibrationArtifact = (
    ExactFFTEligibilityMask | ExactFFTEligibilityRule
)

type RecoveryPlanKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class MissingTimestampCalibration:
    """Versioned observations and exact artifacts for one FFT geometry."""

    identifier: str
    fft_count: int
    checkpoint_stops: tuple[int, ...]
    observations: tuple[
        FFTLossObservation | IntegrationFFTLossObservation, ...
    ] = ()
    exact_artifacts: tuple[ExactCalibrationArtifact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str):
            raise TypeError("identifier must be a string")
        if not self.identifier.strip():
            raise ValueError("identifier must not be empty")
        _integer(self.fft_count, "fft_count", minimum=1)
        if not isinstance(self.checkpoint_stops, tuple):
            raise TypeError("checkpoint_stops must be a tuple")
        previous = 0
        for stop in self.checkpoint_stops:
            _integer(stop, "checkpoint stop", minimum=1)
            if stop <= previous:
                raise ValueError(
                    "checkpoint_stops must be strictly increasing"
                )
            if stop > self.fft_count:
                raise ValueError("checkpoint stop exceeds fft_count")
            previous = stop
        if not self.checkpoint_stops:
            raise ValueError("checkpoint_stops must not be empty")
        if self.checkpoint_stops[-1] != self.fft_count:
            raise ValueError("the final checkpoint must stop at fft_count")
        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")
        if not all(
            isinstance(
                item, (FFTLossObservation, IntegrationFFTLossObservation)
            )
            for item in self.observations
        ):
            raise TypeError("observations contain an unsupported artifact")
        if not isinstance(self.exact_artifacts, tuple):
            raise TypeError("exact_artifacts must be a tuple")
        seen: set[tuple[int, ...]] = set()
        for artifact in self.exact_artifacts:
            if not isinstance(
                artifact, (ExactFFTEligibilityMask, ExactFFTEligibilityRule)
            ):
                raise TypeError(
                    "exact_artifacts contain an unsupported artifact"
                )
            if artifact.missing_timestamps in seen:
                raise ValueError("duplicate exact calibration for event set")
            seen.add(artifact.missing_timestamps)
            if isinstance(artifact, ExactFFTEligibilityMask):
                if len(artifact.eligible_ffts) != self.fft_count:
                    raise ValueError(
                        "exact eligibility mask length must equal fft_count"
                    )
            elif artifact.ineligible_fft_indices and (
                artifact.ineligible_fft_indices[-1] >= self.fft_count
            ):
                raise ValueError("ineligible FFT index exceeds fft_count")


@dataclass(frozen=True, slots=True)
class FFTEligibilityCheckpoint:
    """Eligibility counts for a clipped part of a global checkpoint."""

    index: int
    start_fft: int
    stop_fft: int
    eligible_fft_count: int
    ineligible_fft_count: int

    @property
    def fft_count(self) -> int:
        return self.stop_fft - self.start_fft


def _checkpoints(
    eligibility: tuple[bool, ...],
    checkpoint_stops: tuple[int, ...],
    start: int,
    stop: int,
) -> tuple[FFTEligibilityCheckpoint, ...]:
    result: list[FFTEligibilityCheckpoint] = []
    checkpoint_start = 0
    for index, checkpoint_stop in enumerate(checkpoint_stops):
        clipped_start = max(start, checkpoint_start)
        clipped_stop = min(stop, checkpoint_stop)
        if clipped_start < clipped_stop:
            eligible = sum(eligibility[clipped_start:clipped_stop])
            result.append(
                FFTEligibilityCheckpoint(
                    index,
                    clipped_start,
                    clipped_stop,
                    eligible,
                    clipped_stop - clipped_start - eligible,
                )
            )
        checkpoint_start = checkpoint_stop
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FFTEligibilityWindow:
    """A deterministic view copied from a global eligibility plan."""

    global_start_fft: int
    eligible_ffts: tuple[bool, ...]
    checkpoints: tuple[FFTEligibilityCheckpoint, ...]

    @property
    def global_stop_fft(self) -> int:
        return self.global_start_fft + len(self.eligible_ffts)

    @property
    def fft_count(self) -> int:
        return len(self.eligible_ffts)


@dataclass(frozen=True, slots=True)
class GlobalFFTEligibilityPlan:
    """One immutable eligibility decision over the complete FFT timeline."""

    missing_timestamps: tuple[int, ...]
    eligible_ffts: tuple[bool, ...]
    checkpoint_stops: tuple[int, ...]
    checkpoints: tuple[FFTEligibilityCheckpoint, ...]
    calibration_identifier: str | None
    # Source offsets are in samples, relative to the ordinary timestamp-based
    # selection. A cutover permits a gap to begin partway through one FFT.
    # Empty tuples preserve the historical mask-only plan representation.
    source_offsets: tuple[int, ...] = ()
    source_cutovers: tuple[int, ...] = ()
    # Large file-backed jobs use a sparse representation. Intervals are
    # right-open global FFT coordinates; replay entries are
    # (global FFT, source offset samples, cutover samples).
    timeline_fft_count: int | None = None
    ineligible_intervals: tuple[tuple[int, int], ...] = ()
    source_replay_events: tuple[tuple[int, int, int], ...] = ()
    # Sparse complete piecewise source selections for FFTs that cross more
    # than one reader provenance boundary.  Each event is
    # (global FFT, ((local start, local stop, source offset), ...)).  Omitted
    # local samples are invalid.  This representation is intentionally
    # reserved for rare compatibility boundaries; ordinary FFTs keep the
    # compact one-cutover representation above.
    source_segment_events: tuple[
        tuple[int, tuple[tuple[int, int, int], ...]], ...
    ] = ()
    # Sparse per-FFT valid sample ranges. Samples outside [start, stop) are
    # suppressed after source selection. This represents measured fractional
    # recovery boundaries without rejecting a complete FFT.
    source_validity_events: tuple[tuple[int, int, int], ...] = ()
    # Sparse FFTs whose calibrated local validity replaces the raw frame
    # validity channel. This keeps ordinary malformed-frame handling strict
    # while allowing a measured reader boundary to expose selected payload.
    source_authoritative_validity_events: tuple[int, ...] = ()
    # When true, the calibrated local ranges are the complete source-validity
    # decision. This is reserved for an exact compatibility calibration whose
    # measured reader path had no independent sidecar-validity channel.
    source_validity_is_authoritative: bool = False
    # Persistent source offsets over right-open global FFT intervals.  This
    # compact form is used when a timestamp reanchor changes every later
    # source selection.  Per-FFT replay events take precedence at a boundary.
    source_replay_intervals: tuple[tuple[int, int, int], ...] = ()
    # Right-open source-sample intervals, relative to job start, that a
    # calibrated file-backed recovery state machine consumed without
    # publishing.  These intervals mask the ordinary timestamp-indexed
    # source; they do not retime any remaining samples.
    source_invalid_intervals: tuple[tuple[int, int], ...] = ()
    # Calibrated physical-frame presentation for file-backed VDIF.  Each
    # interval is (physical source frame start, physical source frame stop,
    # presented destination frame start), in absolute VDIF frame slots.
    # Frames absent from these intervals are not published.  Ingest offset
    # intervals are (FFT start, FFT stop, source offset samples) and exist
    # only to select the bounded physical source extent; execution sees the
    # already-remapped ordinary timestamp grid.
    source_frame_intervals: tuple[tuple[int, int, int], ...] = ()
    # Physical-file-order presentation for behaviorally calibrated readers.
    # Entries are (source-file index, physical frame start, physical frame
    # stop, presented destination frame start).  Unlike
    # ``source_frame_intervals``, the source coordinates do not come from
    # VDIF headers.  This representation is reserved for exact calibrations
    # where headers control eligibility but accepted payload advances in
    # physical file order.
    source_physical_frame_intervals: tuple[
        tuple[int, int, int, int], ...
    ] = ()
    # Sparse physical-file-order overrides layered on the ordinary timestamp
    # presentation. Entries use the same source/range/destination shape as
    # ``source_physical_frame_intervals``. A destination of -1 suppresses the
    # selected physical frames. This keeps a measured local permutation from
    # replacing otherwise-correct timestamp placement for the full window.
    source_physical_frame_overrides: tuple[
        tuple[int, int, int, int], ...
    ] = ()
    source_ingest_offset_intervals: tuple[tuple[int, int, int, int], ...] = ()
    # Right-open source-sample intervals in the unfilled tail of an incomplete
    # reader output segment.  Their overlap contributes to the compatibility
    # weight for partially materialized FFTs while their missing samples
    # remain zero.  A fully empty FFT remains zero weight.  This is separate
    # from explicit-invalid input, whose duration still reduces the weight
    # fractionally.
    source_weight_credit_intervals: tuple[tuple[int, int], ...] = ()
    # Optional trustworthy subset of ``source_weight_credit_intervals`` used
    # only for the published station weight.  Visibility normalization keeps
    # the full compatibility credit.  ``None`` means both use the same
    # intervals; an empty tuple intentionally reports no credit.
    source_reported_weight_credit_intervals: (
        tuple[tuple[int, int], ...] | None
    ) = None
    # Sparse compatibility accounting exclusions applied after timestamp-gap
    # credit. Entries are (global FFT index, source samples). Payload and
    # payload validity are unchanged; the values describe measured reader
    # accounting for samples already absent from the accepted payload.
    source_weight_exclusion_events: tuple[tuple[int, int], ...] = ()
    # Sparse exact station-weight overrides applied after payload decoding.
    # Entries are (global FFT index, valid source samples).  They are used
    # only when a calibrated reader counts an FFT toward normalization while
    # publishing no payload for that FFT.
    source_weight_override_events: tuple[tuple[int, int], ...] = ()
    # Logical sample displacement for residual-delay phase only. Physical
    # storage replay offsets are independent and must never enter this clock.
    source_timing_events: tuple[tuple[int, int], ...] = ()
    # File-backed compilers may know only a selected portion of the timeline.
    # None preserves caller-authored complete plans, including identity.
    known_fft_scope: tuple[int, int] | None = None
    _ineligible_interval_starts: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_timing_event_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_replay_event_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_replay_interval_starts: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_segment_event_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_validity_event_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_weight_exclusion_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_weight_override_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_frame_interval_starts: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )
    _source_ingest_offset_starts: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=()
    )

    def __post_init__(self) -> None:
        if self.known_fft_scope is not None:
            if (
                not isinstance(self.known_fft_scope, tuple)
                or len(self.known_fft_scope) != 2
            ):
                raise TypeError("known FFT scope must be a start/stop tuple")
            first, stop = self.known_fft_scope
            _integer(first, "known first FFT")
            _integer(stop, "known stop FFT", minimum=1)
            if not first < stop <= self.fft_count:
                raise ValueError("known FFT scope must lie within timeline")
        if not isinstance(self.source_timing_events, tuple):
            raise TypeError("source timing events must be an immutable tuple")
        previous = -1
        for event in self.source_timing_events:
            if not isinstance(event, tuple) or len(event) != 2:
                raise TypeError(
                    "source timing events must be index/offset tuples"
                )
            index, offset = event
            _integer(index, "source timing FFT")
            if index <= previous or index >= self.fft_count:
                raise ValueError(
                    "source timing FFTs must be ordered and unique "
                    "within timeline"
                )
            if not isinstance(offset, int) or isinstance(offset, bool):
                raise TypeError(
                    "source timing offsets must be signed integers"
                )
            if not -(2**63) <= offset < 2**63:
                raise OverflowError("source timing offsets must fit int64")
            if offset == 0:
                raise ValueError(
                    "source timing events must have nonzero offsets"
                )
            previous = index
        object.__setattr__(
            self,
            "_source_timing_event_indices",
            tuple(index for index, _ in self.source_timing_events),
        )
        if not isinstance(self.source_validity_is_authoritative, bool):
            raise TypeError("authoritative source validity must be boolean")
        object.__setattr__(
            self,
            "_ineligible_interval_starts",
            tuple(start for start, _stop in self.ineligible_intervals),
        )
        object.__setattr__(
            self,
            "_source_replay_event_indices",
            tuple(
                index
                for index, _offset, _cutover in self.source_replay_events
            ),
        )
        object.__setattr__(
            self,
            "_source_segment_event_indices",
            tuple(index for index, _segments in self.source_segment_events),
        )
        previous_stop: int | None = None
        for start, stop in self.source_invalid_intervals:
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (start, stop)
            ):
                raise TypeError(
                    "source invalid interval bounds must be integers"
                )
            if stop <= start:
                raise ValueError("source invalid intervals must be nonempty")
            if previous_stop is not None and start < previous_stop:
                raise ValueError(
                    "source invalid intervals must be ordered and disjoint"
                )
            previous_stop = stop
        object.__setattr__(
            self,
            "_source_replay_interval_starts",
            tuple(
                start
                for start, _stop, _offset in self.source_replay_intervals
            ),
        )
        object.__setattr__(
            self,
            "_source_validity_event_indices",
            tuple(
                index for index, _start, _stop in self.source_validity_events
            ),
        )
        object.__setattr__(
            self,
            "_source_weight_exclusion_indices",
            tuple(
                index
                for index, _samples in self.source_weight_exclusion_events
            ),
        )
        object.__setattr__(
            self,
            "_source_weight_override_indices",
            tuple(
                index
                for index, _samples in self.source_weight_override_events
            ),
        )
        object.__setattr__(
            self,
            "_source_frame_interval_starts",
            tuple(
                start
                for start, _stop, _destination in self.source_frame_intervals
            ),
        )
        object.__setattr__(
            self,
            "_source_ingest_offset_starts",
            tuple(
                start
                for start, _stop, _minimum, _maximum in (
                    self.source_ingest_offset_intervals
                )
            ),
        )
        previous_source_stop: int | None = None
        previous_destination_stop: int | None = None
        for (
            source_start,
            source_stop,
            destination_start,
        ) in self.source_frame_intervals:
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (source_start, source_stop, destination_start)
            ):
                raise TypeError(
                    "source frame presentation values must be integers"
                )
            if source_stop <= source_start:
                raise ValueError(
                    "source frame presentation intervals must be nonempty"
                )
            destination_stop = destination_start + source_stop - source_start
            if (
                previous_source_stop is not None
                and source_start < previous_source_stop
            ):
                raise ValueError(
                    "source frame presentation intervals must be source "
                    "ordered"
                )
            if (
                previous_destination_stop is not None
                and destination_start < previous_destination_stop
            ):
                raise ValueError(
                    "source frame presentation intervals must be destination "
                    "ordered"
                )
            previous_source_stop = source_stop
            previous_destination_stop = destination_stop
        previous_physical_destination_stop: int | None = None
        for (
            source_index,
            source_start,
            source_stop,
            destination_start,
        ) in self.source_physical_frame_intervals:
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (
                    source_index,
                    source_start,
                    source_stop,
                    destination_start,
                )
            ):
                raise TypeError(
                    "physical frame presentation values must be integers"
                )
            if source_index < 0:
                raise ValueError(
                    "physical frame presentation source index must be "
                    "nonnegative"
                )
            if source_start < 0 or source_stop <= source_start:
                raise ValueError(
                    "physical frame presentation intervals must be "
                    "nonempty and nonnegative"
                )
            destination_stop = destination_start + source_stop - source_start
            if (
                previous_physical_destination_stop is not None
                and destination_start < previous_physical_destination_stop
            ):
                raise ValueError(
                    "physical frame presentation intervals must be "
                    "destination ordered"
                )
            previous_physical_destination_stop = destination_stop
        previous_override_stop: int | None = None
        for (
            source_index,
            source_start,
            source_stop,
            destination_start,
        ) in self.source_physical_frame_overrides:
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (
                    source_index,
                    source_start,
                    source_stop,
                    destination_start,
                )
            ):
                raise TypeError(
                    "physical frame override values must be integers"
                )
            if source_index < 0:
                raise ValueError(
                    "physical frame override source index must be nonnegative"
                )
            if source_start < 0 or source_stop <= source_start:
                raise ValueError(
                    "physical frame override intervals must be nonempty and "
                    "nonnegative"
                )
            if destination_start < -1:
                raise ValueError(
                    "physical frame override destination must be nonnegative "
                    "or -1"
                )
            if (
                previous_override_stop is not None
                and source_start < previous_override_stop
            ):
                raise ValueError(
                    "physical overrides must be source ordered and disjoint"
                )
            previous_override_stop = source_stop
        previous_credit_stop: int | None = None
        for start, stop in self.source_weight_credit_intervals:
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (start, stop)
            ):
                raise TypeError(
                    "source weight-credit interval bounds must be integers"
                )
            if stop <= start:
                raise ValueError(
                    "source weight-credit intervals must be nonempty"
                )
            if (
                previous_credit_stop is not None
                and start < previous_credit_stop
            ):
                raise ValueError(
                    "source weight-credit intervals must be ordered and "
                    "disjoint"
                )
            previous_credit_stop = stop
        previous_reported_credit_stop: int | None = None
        for start, stop in self.source_reported_weight_credit_intervals or ():
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (start, stop)
            ):
                raise TypeError(
                    "reported source weight-credit interval bounds must be "
                    "integers"
                )
            if stop <= start:
                raise ValueError(
                    "reported source weight-credit intervals must be nonempty"
                )
            if (
                previous_reported_credit_stop is not None
                and start < previous_reported_credit_stop
            ):
                raise ValueError(
                    "reported source weight-credit intervals must be ordered "
                    "and disjoint"
                )
            previous_reported_credit_stop = stop
        if self.timeline_fft_count is not None:
            _integer(self.timeline_fft_count, "timeline FFT count", minimum=1)
            if (
                self.eligible_ffts
                or self.source_offsets
                or self.source_cutovers
            ):
                raise ValueError(
                    "sparse recovery plans cannot contain dense timelines"
                )
            previous_stop = 0
            for start, stop in self.ineligible_intervals:
                _integer(start, "ineligible interval start")
                _integer(stop, "ineligible interval stop", minimum=1)
                if start < previous_stop or stop <= start:
                    raise ValueError(
                        "ineligible intervals must be ordered and disjoint"
                    )
                if stop > self.timeline_fft_count:
                    raise ValueError(
                        "ineligible interval exceeds the FFT timeline"
                    )
                previous_stop = stop
            previous_index = -1
            for index, offset, cutover in self.source_replay_events:
                _integer(index, "source replay FFT")
                if (
                    index <= previous_index
                    or index >= self.timeline_fft_count
                ):
                    raise ValueError(
                        "source replay FFTs must be ordered and unique"
                    )
                if not isinstance(offset, int) or isinstance(offset, bool):
                    raise TypeError("source replay offsets must be integers")
                _integer(cutover, "source replay cutover")
                if offset == 0 and cutover != 0:
                    raise ValueError(
                        "zero source offsets require zero cutovers"
                    )
                previous_index = index
            previous_index = -1
            for index, segments in self.source_segment_events:
                _integer(index, "source segment FFT")
                if (
                    index <= previous_index
                    or index >= self.timeline_fft_count
                ):
                    raise ValueError(
                        "source segment FFTs must be ordered and unique"
                    )
                if not segments:
                    raise ValueError(
                        "source segment events require at least one segment"
                    )
                previous_segment_stop = 0
                for start, stop, offset in segments:
                    _integer(start, "source segment start")
                    _integer(stop, "source segment stop", minimum=1)
                    if start < previous_segment_stop or stop <= start:
                        raise ValueError(
                            "source segments must be ordered, nonempty, and "
                            "disjoint"
                        )
                    if not isinstance(offset, int) or isinstance(
                        offset, bool
                    ):
                        raise TypeError(
                            "source segment offsets must be integers"
                        )
                    previous_segment_stop = stop
                previous_index = index
            previous_stop = 0
            for start, stop, offset in self.source_replay_intervals:
                _integer(start, "source replay interval start")
                _integer(stop, "source replay interval stop", minimum=1)
                if start < previous_stop or stop <= start:
                    raise ValueError(
                        "source replay intervals must be ordered and disjoint"
                    )
                if stop > self.timeline_fft_count:
                    raise ValueError(
                        "source replay interval exceeds the FFT timeline"
                    )
                if not isinstance(offset, int) or isinstance(offset, bool):
                    raise TypeError("source replay offsets must be integers")
                if offset == 0:
                    raise ValueError(
                        "source replay intervals require nonzero offsets"
                    )
                previous_stop = stop
            previous_index = -1
            for index, start, stop in self.source_validity_events:
                _integer(index, "source validity FFT")
                _integer(start, "source validity start")
                _integer(stop, "source validity stop")
                if (
                    index <= previous_index
                    or index >= self.timeline_fft_count
                ):
                    raise ValueError(
                        "source validity FFTs must be ordered and unique"
                    )
                if stop < start:
                    raise ValueError(
                        "source validity range bounds are reversed"
                    )
                previous_index = index
            previous_index = -1
            for index, samples in self.source_weight_exclusion_events:
                _integer(index, "source weight-exclusion FFT")
                _integer(
                    samples, "source weight-exclusion samples", minimum=1
                )
                if (
                    index <= previous_index
                    or index >= self.timeline_fft_count
                ):
                    raise ValueError(
                        "source weight-exclusion FFTs must be ordered and "
                        "unique"
                    )
                previous_index = index
            previous_index = -1
            for index in self.source_authoritative_validity_events:
                _integer(index, "authoritative source-validity FFT")
                if (
                    index <= previous_index
                    or index >= self.timeline_fft_count
                ):
                    raise ValueError(
                        "authoritative source-validity FFTs must be ordered "
                        "and unique"
                    )
                if self.source_validity(index) is None:
                    raise ValueError(
                        "authoritative source validity requires a local "
                        "validity event"
                    )
                previous_index = index
            previous_index = -1
            for index, samples in self.source_weight_override_events:
                _integer(index, "source weight-override FFT")
                _integer(samples, "source weight-override samples")
                if (
                    index <= previous_index
                    or index >= self.timeline_fft_count
                ):
                    raise ValueError(
                        "source weight-override FFTs must be ordered and "
                        "unique"
                    )
                previous_index = index
            previous_stop = 0
            for (
                start,
                stop,
                minimum,
                maximum,
            ) in self.source_ingest_offset_intervals:
                _integer(start, "source ingest offset interval start")
                _integer(
                    stop, "source ingest offset interval stop", minimum=1
                )
                if start < previous_stop or stop <= start:
                    raise ValueError(
                        "source ingest offset intervals must be ordered and "
                        "disjoint"
                    )
                if stop > self.timeline_fft_count:
                    raise ValueError(
                        "source ingest offset interval exceeds the FFT "
                        "timeline"
                    )
                if any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in (minimum, maximum)
                ):
                    raise TypeError("source ingest offsets must be integers")
                if minimum > maximum:
                    raise ValueError(
                        "source ingest offset bounds are reversed"
                    )
                previous_stop = stop
        elif (
            self.ineligible_intervals
            or self.source_replay_events
            or self.source_segment_events
            or self.source_replay_intervals
            or self.source_validity_events
            or self.source_authoritative_validity_events
            or self.source_invalid_intervals
            or self.source_frame_intervals
            or self.source_physical_frame_intervals
            or self.source_physical_frame_overrides
            or self.source_ingest_offset_intervals
            or self.source_weight_credit_intervals
            or self.source_reported_weight_credit_intervals is not None
            or self.source_weight_exclusion_events
            or self.source_weight_override_events
        ):
            raise ValueError(
                "sparse recovery data requires a timeline length"
            )
        if bool(self.source_offsets) != bool(self.source_cutovers):
            raise ValueError(
                "source offsets and cutovers must be specified together"
            )
        if self.source_offsets and len(self.source_offsets) != self.fft_count:
            raise ValueError("source offset count must equal FFT count")
        if (
            self.source_cutovers
            and len(self.source_cutovers) != self.fft_count
        ):
            raise ValueError("source cutover count must equal FFT count")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in self.source_offsets
        ):
            raise TypeError("source offsets must be integers")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.source_cutovers
        ):
            raise ValueError("source cutovers must be nonnegative integers")
        if any(
            offset == 0 and cutover != 0
            for offset, cutover in zip(
                self.source_offsets, self.source_cutovers, strict=True
            )
        ):
            raise ValueError("zero source offsets require zero cutovers")

    @property
    def fft_count(self) -> int:
        if self.timeline_fft_count is not None:
            return self.timeline_fft_count
        return len(self.eligible_ffts)

    @property
    def has_complete_source_replay(self) -> bool:
        """Whether every FFT has an exact ordinary or overridden source."""

        return self.timeline_fft_count is not None or bool(
            self.source_offsets
        )

    @property
    def ineligible_fft_count(self) -> int:
        if self.timeline_fft_count is None:
            return self.fft_count - sum(self.eligible_ffts)
        return sum(stop - start for start, stop in self.ineligible_intervals)

    @property
    def source_shifted_fft_count(self) -> int:
        if self.timeline_fft_count is None:
            return sum(offset != 0 for offset in self.source_offsets)
        shifted = sum(
            stop - start
            for start, stop, _offset in self.source_replay_intervals
        )
        for index, offset, _cutover in self.source_replay_events:
            interval_offset = self._interval_source_offset(index)
            if interval_offset == 0 and offset != 0:
                shifted += 1
            elif interval_offset != 0 and offset == 0:
                shifted -= 1
        return shifted

    @property
    def source_masked_fft_count(self) -> int:
        """Number of FFTs with a calibrated local sample-validity range."""

        return len(self.source_validity_events)

    @property
    def source_invalid_sample_count(self) -> int:
        """Number of timestamp-grid samples consumed during recovery."""

        return sum(
            stop - start for start, stop in self.source_invalid_intervals
        )

    @property
    def has_source_frame_presentation(self) -> bool:
        """Whether physical VDIF frames require calibrated placement."""

        return bool(
            self.source_frame_intervals
            or self.source_physical_frame_intervals
        )

    @property
    def has_source_weight_credit(self) -> bool:
        """Whether timestamp holes require partial-FFT weight credit."""

        return bool(self.source_weight_credit_intervals)

    @property
    def has_separate_reported_weight_credit(self) -> bool:
        """Whether published weight uses a distinct trusted credit set."""

        return self.source_reported_weight_credit_intervals is not None

    @property
    def has_source_weight_exclusion(self) -> bool:
        """Whether calibrated FFTs exclude samples from reported weight."""

        return bool(self.source_weight_exclusion_events)

    @property
    def has_source_weight_override(self) -> bool:
        """Whether calibrated FFTs replace their decoded station weight."""

        return bool(self.source_weight_override_events)

    def source_weight_exclusion(self, index: int) -> int:
        """Return the measured accounting exclusion for one FFT."""

        self.require_scope(index, index + 1)
        position = bisect_left(self._source_weight_exclusion_indices, index)
        if (
            position < len(self.source_weight_exclusion_events)
            and self.source_weight_exclusion_events[position][0] == index
        ):
            return self.source_weight_exclusion_events[position][1]
        return 0

    def source_weight_override(self, index: int) -> int | None:
        """Return the measured station-weight sample count for one FFT."""

        self.require_scope(index, index + 1)
        position = bisect_left(self._source_weight_override_indices, index)
        if (
            position < len(self.source_weight_override_events)
            and self.source_weight_override_events[position][0] == index
        ):
            return self.source_weight_override_events[position][1]
        return None

    @property
    def has_source_segments(self) -> bool:
        """Whether rare FFTs contain complete piecewise source selections."""

        return bool(self.source_segment_events)

    @property
    def has_authoritative_source_validity(self) -> bool:
        """Whether calibrated ranges replace the input validity channel."""

        return self.source_validity_is_authoritative

    def source_validity_is_authoritative_at(self, index: int) -> bool:
        """Return whether one FFT's calibrated range replaces raw validity."""

        self.require_scope(index, index + 1)
        if self.source_validity_is_authoritative:
            return True
        position = bisect_left(
            self.source_authoritative_validity_events, index
        )
        return (
            position < len(self.source_authoritative_validity_events)
            and self.source_authoritative_validity_events[position] == index
        )

    @property
    def source_offset_bounds(self) -> tuple[int, int]:
        if self.timeline_fft_count is None:
            values = self.source_offsets or (0,)
        else:
            values = tuple(
                offset for _, offset, _ in self.source_replay_events
            ) + tuple(offset for _, _, offset in self.source_replay_intervals)
            values += tuple(
                offset
                for _index, segments in self.source_segment_events
                for _start, _stop, offset in segments
            )
            values += tuple(
                offset
                for _, _, minimum, maximum in (
                    self.source_ingest_offset_intervals
                )
                for offset in (minimum, maximum)
            )
            values = values or (0,)
        return min(values), max(values)

    def source_offset_bounds_for(
        self, start_fft: int, stop_fft: int
    ) -> tuple[int, int]:
        """Return conservative replay offsets for one FFT subrange."""

        _integer(start_fft, "start_fft")
        _integer(stop_fft, "stop_fft")
        if start_fft > stop_fft:
            raise ValueError("start_fft must not exceed stop_fft")
        if stop_fft > self.fft_count:
            raise ValueError("window stop exceeds fft_count")
        self.require_scope(start_fft, stop_fft)
        if start_fft == stop_fft:
            return 0, 0
        if self.timeline_fft_count is None:
            offsets = self.source_offsets[start_fft:stop_fft] or (0,)
            return min(offsets), max(offsets)

        values: list[int] = []
        covered_stop = start_fft
        position = max(
            0,
            bisect_right(self._source_replay_interval_starts, start_fft) - 1,
        )
        while position < len(self.source_replay_intervals):
            interval_start, interval_stop, offset = (
                self.source_replay_intervals[position]
            )
            if interval_start >= stop_fft:
                break
            if interval_stop > start_fft:
                gap_stop = min(interval_start, stop_fft)
                if gap_stop > covered_stop and self._has_eligible_fft(
                    covered_stop, gap_stop
                ):
                    values.append(0)
                values.append(offset)
                covered_stop = max(covered_stop, min(stop_fft, interval_stop))
            position += 1
        if covered_stop < stop_fft and self._has_eligible_fft(
            covered_stop, stop_fft
        ):
            values.append(0)
        event_start = bisect_left(
            self._source_replay_event_indices, start_fft
        )
        event_stop = bisect_left(self._source_replay_event_indices, stop_fft)
        events = self.source_replay_events[event_start:event_stop]
        values.extend(offset for _index, offset, _cutover in events)
        if any(cutover for _index, _offset, cutover in events):
            values.append(0)
        segment_start = bisect_left(
            self._source_segment_event_indices, start_fft
        )
        segment_stop = bisect_left(
            self._source_segment_event_indices, stop_fft
        )
        for _index, segments in self.source_segment_events[
            segment_start:segment_stop
        ]:
            values.extend(offset for _start, _stop, offset in segments)
        if (
            (
                self.source_frame_intervals
                or self.source_physical_frame_intervals
            )
            and not self.source_replay_events
            and not self.source_replay_intervals
            and not self.source_segment_events
        ):
            # File-backed presentation remaps the bounded source before
            # execution.  Its local physical offsets replace, rather than
            # extend, the ordinary timestamp query.
            values.clear()
        ingest_position = max(
            0,
            bisect_right(self._source_ingest_offset_starts, start_fft) - 1,
        )
        while ingest_position < len(self.source_ingest_offset_intervals):
            interval_start, interval_stop, minimum, maximum = (
                self.source_ingest_offset_intervals[ingest_position]
            )
            if interval_start >= stop_fft:
                break
            if interval_stop > start_fft:
                values.extend((minimum, maximum))
            ingest_position += 1
        if not values:
            values.append(0)
        return min(values), max(values)

    def _has_eligible_fft(self, start_fft: int, stop_fft: int) -> bool:
        """Return whether a right-open range contains an eligible FFT."""

        if start_fft >= stop_fft:
            return False
        position = max(
            0,
            bisect_right(self._ineligible_interval_starts, start_fft) - 1,
        )
        cursor = start_fft
        while position < len(self.ineligible_intervals):
            interval_start, interval_stop = self.ineligible_intervals[
                position
            ]
            if interval_stop <= cursor:
                position += 1
                continue
            if interval_start >= stop_fft:
                break
            if interval_start > cursor:
                return True
            cursor = max(cursor, interval_stop)
            if cursor >= stop_fft:
                return False
            position += 1
        return cursor < stop_fft

    def _interval_source_offset(self, index: int) -> int:
        position = (
            bisect_right(self._source_replay_interval_starts, index) - 1
        )
        if position >= 0:
            start, stop, offset = self.source_replay_intervals[position]
            if start <= index < stop:
                return offset
        return 0

    def require_scope(self, first_fft: int, stop_fft: int) -> None:
        """Reject execution outside a compiled plan's known decisions."""

        first, stop = self.known_fft_scope or (0, self.fft_count)
        if first_fft < first or stop_fft > stop:
            raise MissingTimestampRecoveryError(
                "requested decisions lie outside the known FFT scope"
            )

    def is_eligible(self, index: int) -> bool:
        _integer(index, "FFT index")
        if index >= self.fft_count:
            raise ValueError("FFT index exceeds the calibrated timeline")
        self.require_scope(index, index + 1)
        if self.timeline_fft_count is None:
            return self.eligible_ffts[index]
        position = bisect_right(self._ineligible_interval_starts, index) - 1
        if position < 0:
            return True
        start, stop = self.ineligible_intervals[position]
        return not (start <= index < stop)

    def source_replay(self, index: int) -> tuple[int, int]:
        _integer(index, "FFT index")
        if index >= self.fft_count:
            raise ValueError("FFT index exceeds the calibrated timeline")
        self.require_scope(index, index + 1)
        if self.timeline_fft_count is None:
            if not self.source_offsets:
                return 0, 0
            return self.source_offsets[index], self.source_cutovers[index]
        position = bisect_left(self._source_replay_event_indices, index)
        if (
            position < len(self.source_replay_events)
            and self.source_replay_events[position][0] == index
        ):
            _replay_index, offset, cutover = self.source_replay_events[
                position
            ]
            return offset, cutover
        return self._interval_source_offset(index), 0

    def source_validity(self, index: int) -> tuple[int, int] | None:
        """Return a sparse local valid sample range for one FFT."""

        _integer(index, "FFT index")
        if index >= self.fft_count:
            raise ValueError("FFT index exceeds the calibrated timeline")
        self.require_scope(index, index + 1)
        position = bisect_left(self._source_validity_event_indices, index)
        if (
            position < len(self.source_validity_events)
            and self.source_validity_events[position][0] == index
        ):
            _validity_index, start, stop = self.source_validity_events[
                position
            ]
            return start, stop
        return None

    def source_segments(
        self, index: int
    ) -> tuple[tuple[int, int, int], ...] | None:
        """Return a complete piecewise source selection for one FFT."""

        _integer(index, "FFT index")
        if index >= self.fft_count:
            raise ValueError("FFT index exceeds the calibrated timeline")
        self.require_scope(index, index + 1)
        position = bisect_left(self._source_segment_event_indices, index)
        if (
            position < len(self.source_segment_events)
            and self.source_segment_events[position][0] == index
        ):
            return self.source_segment_events[position][1]
        return None

    def window(self, start_fft: int, stop_fft: int) -> FFTEligibilityWindow:
        """Slice by global right-open FFT indices."""

        _integer(start_fft, "start_fft")
        _integer(stop_fft, "stop_fft")
        if start_fft > stop_fft:
            raise ValueError("start_fft must not exceed stop_fft")
        if stop_fft > self.fft_count:
            raise ValueError("window stop exceeds fft_count")
        eligibility = tuple(
            self.is_eligible(index) for index in range(start_fft, stop_fft)
        )
        checkpoints: list[FFTEligibilityCheckpoint] = []
        checkpoint_start = 0
        for index, checkpoint_stop in enumerate(self.checkpoint_stops):
            clipped_start = max(start_fft, checkpoint_start)
            clipped_stop = min(stop_fft, checkpoint_stop)
            if clipped_start < clipped_stop:
                local_start = clipped_start - start_fft
                local_stop = clipped_stop - start_fft
                eligible = sum(eligibility[local_start:local_stop])
                checkpoints.append(
                    FFTEligibilityCheckpoint(
                        index,
                        clipped_start,
                        clipped_stop,
                        eligible,
                        clipped_stop - clipped_start - eligible,
                    )
                )
            checkpoint_start = checkpoint_stop
        return FFTEligibilityWindow(
            start_fft, eligibility, tuple(checkpoints)
        )


@lru_cache(maxsize=2)
def _validated_fft_indices(
    fft_count: int,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
    known_fft_scope: tuple[int, int] | None = None,
) -> np.ndarray:
    """Validate an immutable block grid once across independent selectors."""

    if not math.isfinite(scan_origin_s):
        raise ValueError("scan_origin_s must be finite")
    if not math.isfinite(fft_seconds) or fft_seconds <= 0:
        raise ValueError("fft_seconds must be finite and positive")
    starts = np.asarray(block_starts_s, dtype=np.float64)
    if not np.all(np.isfinite(starts)):
        raise ValueError("block start must be finite")
    # Block starts and reconstructed starts group their additions differently.
    # Allow the resulting rounding at the absolute clock magnitude, while
    # refusing clocks whose precision cannot distinguish a small FFT fraction.
    clock_ulp = np.maximum(
        np.abs(np.spacing(starts)), abs(np.spacing(scan_origin_s))
    )
    if np.any(clock_ulp > fft_seconds / 32):
        raise MissingTimestampRecoveryError(
            "clock precision cannot resolve the calibrated FFT timeline"
        )
    tolerance = np.maximum(max(1e-12, fft_seconds * 1e-7), 2 * clock_ulp)
    rounded = np.rint((starts - scan_origin_s) / fft_seconds)
    reconstructed = scan_origin_s + rounded * fft_seconds
    if np.any(np.abs(reconstructed - starts) > tolerance):
        raise MissingTimestampRecoveryError(
            "block start does not lie on the calibrated FFT timeline"
        )
    if fft_count > 1 << 53:
        # Preserve exact Python integer bounds for unusually large timelines.
        indices = [
            round((start - scan_origin_s) / fft_seconds)
            for start in block_starts_s
        ]
        outside = any(index < 0 or index >= fft_count for index in indices)
        result = np.asarray(indices, dtype=object)
    else:
        outside = bool(np.any((rounded < 0) | (rounded >= fft_count)))
        result = rounded.astype(np.int64) if not outside else rounded
    if outside:
        raise MissingTimestampRecoveryError(
            "block start lies outside the calibrated FFT timeline"
        )
    if known_fft_scope is not None:
        first, stop = known_fft_scope
        if np.any((result < first) | (result >= stop)):
            raise MissingTimestampRecoveryError(
                "requested decisions lie outside the known FFT scope"
            )
    result.setflags(write=False)
    return result


def _selected_sparse_events(
    indices: np.ndarray, event_indices: tuple[int, ...]
) -> tuple[list[int], list[int]]:
    """Match exact events, retaining input order and duplicate blocks."""

    if not event_indices or not indices.size:
        return [], []
    keys = np.asarray(event_indices)
    positions = np.searchsorted(keys, indices)
    candidates = np.flatnonzero(positions < keys.size)
    matches = candidates[keys[positions[candidates]] == indices[candidates]]
    return matches.tolist(), positions[matches].tolist()


def select_fft_authoritative_validity(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
) -> tuple[bool, ...]:
    """Select FFTs whose calibrated validity replaces raw validity."""

    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    if plan.source_validity_is_authoritative:
        return (True,) * indices.size
    result = np.zeros(indices.size, dtype=bool)
    outputs, _events = _selected_sparse_events(
        indices, plan.source_authoritative_validity_events
    )
    result[outputs] = True
    return tuple(result.tolist())


def select_fft_eligibility(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
) -> tuple[bool, ...]:
    """Select immutable global decisions for exact output block starts."""

    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    if plan.timeline_fft_count is None:
        return tuple(plan.eligible_ffts[index] for index in indices.tolist())
    result = np.ones(indices.size, dtype=bool)
    if plan.ineligible_intervals:
        positions = (
            np.searchsorted(
                np.asarray(plan._ineligible_interval_starts),
                indices,
                side="right",
            )
            - 1
        )
        candidates = np.flatnonzero(positions >= 0)
        stops = np.asarray(
            [stop for _start, stop in plan.ineligible_intervals]
        )
        result[candidates] = (
            indices[candidates] >= stops[positions[candidates]]
        )
    return tuple(result.tolist())


def select_fft_source_timing(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
) -> np.ndarray:
    """Select logical phase displacements independently of storage replay."""
    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    result = np.zeros(indices.size, dtype=np.int64)
    outputs, events = _selected_sparse_events(
        indices, plan._source_timing_event_indices
    )
    for output, event in zip(outputs, events, strict=True):
        result[output] = plan.source_timing_events[event][1]
    return result


def select_fft_source_replay(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
    fft_samples: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select exact sample offsets and cutovers for output blocks."""

    if not plan.has_complete_source_replay:
        return ((0,) * len(block_starts_s), (0,) * len(block_starts_s))
    _integer(fft_samples, "fft_samples", minimum=1)
    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    if plan.timeline_fft_count is None:
        offsets = [plan.source_offsets[index] for index in indices.tolist()]
        cutovers = [plan.source_cutovers[index] for index in indices.tolist()]
    else:
        offsets = [0] * indices.size
        cutovers = [0] * indices.size
        if plan.source_replay_intervals:
            starts = np.asarray(
                [
                    start
                    for start, _stop, _offset in plan.source_replay_intervals
                ]
            )
            stops = np.asarray(
                [
                    stop
                    for _start, stop, _offset in plan.source_replay_intervals
                ]
            )
            positions = np.searchsorted(starts, indices, side="right") - 1
            candidates = np.flatnonzero(positions >= 0)
            matches = candidates[
                indices[candidates] < stops[positions[candidates]]
            ]
            for output, interval in zip(
                matches.tolist(), positions[matches].tolist(), strict=True
            ):
                offsets[output] = plan.source_replay_intervals[interval][2]
        outputs, events = _selected_sparse_events(
            indices, plan._source_replay_event_indices
        )
        for output, event in zip(outputs, events, strict=True):
            _index, offsets[output], cutovers[output] = (
                plan.source_replay_events[event]
            )
    for cutover in cutovers:
        if cutover >= fft_samples:
            raise MissingTimestampRecoveryError(
                "source cutover must lie inside the selected FFT"
            )
    return tuple(offsets), tuple(cutovers)


def select_fft_source_validity(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
    fft_samples: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select exact local valid sample ranges for output blocks."""

    _integer(fft_samples, "fft_samples", minimum=1)
    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    starts = [0] * indices.size
    stops = [fft_samples] * indices.size
    outputs, events = _selected_sparse_events(
        indices, plan._source_validity_event_indices
    )
    for output, event in zip(outputs, events, strict=True):
        _index, start, stop = plan.source_validity_events[event]
        if not 0 <= start <= stop <= fft_samples:
            raise MissingTimestampRecoveryError(
                "source validity range lies outside the selected FFT"
            )
        starts[output] = start
        stops[output] = stop
    return tuple(starts), tuple(stops)


def select_fft_weight_exclusions(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
    fft_samples: int,
) -> tuple[int, ...]:
    """Select measured post-credit weight exclusions for output blocks."""

    _integer(fft_samples, "fft_samples", minimum=1)
    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    result = [0] * indices.size
    outputs, events = _selected_sparse_events(
        indices, plan._source_weight_exclusion_indices
    )
    for output, event in zip(outputs, events, strict=True):
        _index, samples = plan.source_weight_exclusion_events[event]
        if samples > fft_samples:
            raise MissingTimestampRecoveryError(
                "source weight exclusion exceeds the selected FFT"
            )
        result[output] = samples
    return tuple(result)


def select_fft_weight_overrides(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
    fft_samples: int,
) -> tuple[int | None, ...]:
    """Select exact station-weight sample counts for output blocks."""

    _integer(fft_samples, "fft_samples", minimum=1)
    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    result: list[int | None] = [None] * indices.size
    outputs, events = _selected_sparse_events(
        indices, plan._source_weight_override_indices
    )
    for output, event in zip(outputs, events, strict=True):
        _index, samples = plan.source_weight_override_events[event]
        if samples is not None and samples > fft_samples:
            raise MissingTimestampRecoveryError(
                "source weight override exceeds the selected FFT"
            )
        result[output] = samples
    return tuple(result)


def select_fft_source_segments(
    plan: GlobalFFTEligibilityPlan,
    *,
    scan_origin_s: float,
    fft_seconds: float,
    block_starts_s: tuple[float, ...],
    fft_samples: int,
) -> tuple[tuple[tuple[int, int, int], ...] | None, ...]:
    """Select complete piecewise source mappings for output blocks."""

    _integer(fft_samples, "fft_samples", minimum=1)
    indices = _validated_fft_indices(
        plan.fft_count,
        scan_origin_s,
        fft_seconds,
        tuple(block_starts_s),
        plan.known_fft_scope,
    )
    result: list[tuple[tuple[int, int, int], ...] | None] = [
        None
    ] * indices.size
    outputs, events = _selected_sparse_events(
        indices, plan._source_segment_event_indices
    )
    for output, event in zip(outputs, events, strict=True):
        _index, segments = plan.source_segment_events[event]
        if segments is not None and any(
            start < 0 or stop > fft_samples
            for start, stop, _offset in segments
        ):
            raise MissingTimestampRecoveryError(
                "source segment lies outside the selected FFT"
            )
        result[output] = segments
    return tuple(result)


def with_exact_source_replay(
    plan: GlobalFFTEligibilityPlan,
    *,
    source_offsets: tuple[int, ...],
    source_cutovers: tuple[int, ...],
) -> GlobalFFTEligibilityPlan:
    """Attach an explicit, complete source selection to an exact mask plan."""

    if plan.calibration_identifier is None:
        raise UncalibratedRecoveryError(
            "source replay requires an exact calibration"
        )
    if not source_offsets or len(source_offsets) != plan.fft_count:
        raise ValueError("source replay must cover the complete FFT timeline")
    return GlobalFFTEligibilityPlan(
        plan.missing_timestamps,
        plan.eligible_ffts,
        plan.checkpoint_stops,
        plan.checkpoints,
        plan.calibration_identifier,
        source_offsets,
        source_cutovers,
    )


def build_global_fft_eligibility_plan(
    *,
    fft_count: int,
    checkpoint_stops: tuple[int, ...],
    missing_timestamps: tuple[int, ...],
    calibration: MissingTimestampCalibration | None = None,
) -> GlobalFFTEligibilityPlan:
    """Build an exact global plan, failing closed for uncalibrated events."""

    # Reuse the calibration container's validation for standalone geometry.
    geometry = MissingTimestampCalibration(
        "requested-geometry", fft_count, checkpoint_stops
    )
    events = _events(missing_timestamps)
    if calibration is not None and (
        calibration.fft_count != geometry.fft_count
        or calibration.checkpoint_stops != geometry.checkpoint_stops
    ):
        raise MissingTimestampRecoveryError(
            "calibration geometry does not match the requested plan"
        )
    if not events:
        eligibility = (True,) * fft_count
        identifier = (
            calibration.identifier if calibration is not None else None
        )
    else:
        if calibration is None:
            raise UncalibratedRecoveryError(
                "missing timestamps require an exact calibration artifact"
            )
        artifact = next(
            (
                item
                for item in calibration.exact_artifacts
                if item.missing_timestamps == events
            ),
            None,
        )
        if artifact is None:
            raise UncalibratedRecoveryError(
                "no exact calibration artifact covers the complete "
                f"missing-event set {events!r}"
            )
        if isinstance(artifact, ExactFFTEligibilityMask):
            eligibility = artifact.eligible_ffts
        else:
            rejected = set(artifact.ineligible_fft_indices)
            eligibility = tuple(
                index not in rejected for index in range(fft_count)
            )
        identifier = calibration.identifier
    checkpoints = _checkpoints(eligibility, checkpoint_stops, 0, fft_count)
    return GlobalFFTEligibilityPlan(
        events,
        eligibility,
        checkpoint_stops,
        checkpoints,
        identifier,
    )


__all__ = [
    "ExactCalibrationArtifact",
    "ExactFFTEligibilityMask",
    "ExactFFTEligibilityRule",
    "FFTEligibilityCheckpoint",
    "FFTEligibilityWindow",
    "FFTLossObservation",
    "GlobalFFTEligibilityPlan",
    "IntegrationFFTLossObservation",
    "MissingTimestampCalibration",
    "MissingTimestampRecoveryError",
    "UncalibratedRecoveryError",
    "build_global_fft_eligibility_plan",
    "select_fft_source_replay",
    "select_fft_source_segments",
    "select_fft_source_validity",
    "select_fft_weight_exclusions",
    "select_fft_weight_overrides",
    "with_exact_source_replay",
]
