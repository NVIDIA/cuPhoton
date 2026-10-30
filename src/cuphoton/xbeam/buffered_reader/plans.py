# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader plans: input-derived compatibility planning."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import overload

from ..missing_timestamp_recovery import GlobalFFTEligibilityPlan
from ._checks import _integer
from .addressing import LatchedRawSourceCompilation
from .validity import RawValidityCompilation


@dataclass(frozen=True, slots=True)
class OrdinaryRawSourceSpan:
    """Ordinary execution reads advancing one FFT per index within a run.

    ``ordinary_sample_start`` is signed and identifies the first FFT's read
    in the same raw coordinate system as compiled physical source addresses.
    A new run records a caller-derived coarse-delay change; this adapter does
    not derive or calibrate the ordinary read positions.
    """

    first_fft: int
    stop_fft: int
    ordinary_sample_start: int


def _ordered_intervals(rows, first_fft, count, name):
    previous = first_fft
    result = []
    for first, stop in rows:
        first = _integer(first, name + " start")
        stop = _integer(stop, name + " stop", minimum=1)
        if first < previous or stop <= first or stop > count:
            raise ValueError(
                name + " must be ordered disjoint timeline intervals"
            )
        result.append((first, stop))
        previous = stop
    return tuple(result)


def _partition(rows, first_fft, count, name):
    cursor = first_fft
    for first, stop in sorted(rows):
        if first != cursor:
            raise ValueError(
                name + " must cover the timeline without overlap"
            )
        cursor = stop
    if cursor != count:
        raise ValueError(name + " must cover the complete timeline")


def _compile_raw_replay_plan(
    addresses: LatchedRawSourceCompilation,
    validity: RawValidityCompilation,
    *,
    timeline_fft_count: int,
    fft_samples: int,
    ordinary_sample_origin: int | None = None,
    ordinary_source_spans: tuple[OrdinaryRawSourceSpan, ...] | None = None,
    ineligible_intervals: tuple[tuple[int, int], ...],
    maximum_empty_fft_events: int = 4096,
    first_fft: int = 0,
) -> GlobalFFTEligibilityPlan:
    """Encode complete supplied raw facts into the existing sparse plan.

    Supply exactly one ordinary source basis: an affine global sample origin,
    or explicit piecewise-affine ``ordinary_source_spans`` covering the known
    scope, including discarded FFTs. These must identify actual ordinary
    execution reads in the SAME raw sample coordinate system as ``addresses``.
    This function derives neither coarse delay nor raw-to-ingest presentation.
    Timestamp-normalized storage coordinates violate this contract.

    Address runs and ineligible intervals must partition the requested scope.
    Validity must partition all accepted addresses into full, partial or empty
    rows. Unresolved inputs fail closed instead of becoming excluded FFTs.
    Physical validity still uses the input's ordinary validity channel.
    No authoritative validity or exposure credit/override is asserted.

    Full runs and discards stay intervals; only partial FFTs become events.
    Execution plans lack compact empty-validity runs, so accepted
    empty FFTs require individual (0, 0) events. Their bounded count avoids
    expanding a large empty interval across a whole scan.
    """
    if not isinstance(addresses, LatchedRawSourceCompilation):
        raise TypeError("addresses must be a LatchedRawSourceCompilation")
    if not isinstance(validity, RawValidityCompilation):
        raise TypeError("validity must be a RawValidityCompilation")
    count = _integer(timeline_fft_count, "timeline FFT count", minimum=1)
    first_fft = _integer(first_fft, "first FFT")
    if first_fft >= count:
        raise ValueError("requested FFT window must be nonempty")
    size = _integer(fft_samples, "FFT samples", minimum=1)
    limit = _integer(maximum_empty_fft_events, "empty FFT event limit")
    if (ordinary_sample_origin is None) == (ordinary_source_spans is None):
        raise ValueError(
            "supply exactly one ordinary source coordinate basis"
        )
    if ordinary_source_spans is None:
        if not isinstance(ordinary_sample_origin, int) or isinstance(
            ordinary_sample_origin, bool
        ):
            raise TypeError("ordinary sample origin must be an integer")
        ordinary_source_spans = (
            OrdinaryRawSourceSpan(
                first_fft, count, ordinary_sample_origin + first_fft * size
            ),
        )
    if not isinstance(ordinary_source_spans, tuple) or not all(
        isinstance(span, OrdinaryRawSourceSpan)
        for span in ordinary_source_spans
    ):
        raise TypeError(
            "ordinary source spans must be an OrdinaryRawSourceSpan tuple"
        )
    ordinary_rows = _ordered_intervals(
        [(span.first_fft, span.stop_fft) for span in ordinary_source_spans],
        first_fft,
        count,
        "ordinary source",
    )
    _partition(ordinary_rows, first_fft, count, "ordinary source")
    for ordinary_span in ordinary_source_spans:
        if not isinstance(
            ordinary_span.ordinary_sample_start, int
        ) or isinstance(ordinary_span.ordinary_sample_start, bool):
            raise TypeError("ordinary source starts must be signed integers")
    ordinary_firsts = tuple(span.first_fft for span in ordinary_source_spans)

    def ordinary_at(index):
        span = ordinary_source_spans[bisect_right(ordinary_firsts, index) - 1]
        return span.ordinary_sample_start + (index - span.first_fft) * size

    if addresses.unresolved or validity.unresolved:
        raise ValueError("unresolved raw source decisions cannot form a plan")
    source_rows = _ordered_intervals(
        [(s.first_fft, s.stop_fft) for s in addresses.spans],
        first_fft,
        count,
        "addresses",
    )
    rejected = _ordered_intervals(
        ineligible_intervals, first_fft, count, "discards"
    )
    _partition(
        source_rows + rejected, first_fft, count, "source/discard decisions"
    )
    for span in addresses.spans:
        _integer(span.physical_sample_start, "raw physical sample")
    full_rows = _ordered_intervals(
        [(s.first_fft, s.stop_fft) for s in validity.full_spans],
        first_fft,
        count,
        "full validity",
    )
    partial_rows = _ordered_intervals(
        [(e.fft_index, e.fft_index + 1) for e in validity.partial_events],
        first_fft,
        count,
        "partial validity",
    )
    empty_rows = _ordered_intervals(
        validity.empty_intervals, first_fft, count, "empty validity"
    )
    _partition(
        full_rows + partial_rows + empty_rows + rejected,
        first_fft,
        count,
        "validity",
    )
    if sum(stop - first for first, stop in empty_rows) > limit:
        raise ValueError(
            "accepted empty FFTs exceed the explicit event limit"
        )
    firsts = tuple(s.first_fft for s in addresses.spans)

    def address_at(index):
        position = bisect_right(firsts, index) - 1
        if position < 0 or index >= addresses.spans[position].stop_fft:
            raise ValueError("validity row has no accepted raw address")
        span = addresses.spans[position]
        return span.physical_sample_start + (index - span.first_fft) * size

    # Check provenance across address boundaries without FFT expansion.
    for full in validity.full_spans:
        _integer(full.physical_sample_start, "full raw physical sample")
        index = full.first_fft
        while index < full.stop_fft:
            actual = address_at(index)
            expected = (
                full.physical_sample_start + (index - full.first_fft) * size
            )
            if actual != expected:
                raise ValueError(
                    "full validity does not match raw source identity"
                )
            source = addresses.spans[bisect_right(firsts, index) - 1]
            index = min(source.stop_fft, full.stop_fft)
    replay: list[tuple[int, int, int]] = []
    for span in addresses.spans:
        index = span.first_fft
        while index < span.stop_fft:
            basis = ordinary_source_spans[
                bisect_right(ordinary_firsts, index) - 1
            ]
            stop = min(span.stop_fft, basis.stop_fft)
            offset = (
                int(span.physical_sample_start)
                + (index - span.first_fft) * size
                - ordinary_at(index)
            )
            if offset:
                if (
                    replay
                    and replay[-1][1] == index
                    and replay[-1][2] == offset
                ):
                    first, _, _ = replay.pop()
                    replay.append((first, stop, offset))
                else:
                    replay.append((index, stop, offset))
            index = stop
    local_validity = []
    segments = []
    for event in validity.partial_events:
        index = event.fft_index
        base = address_at(index)
        ordinary = ordinary_at(index)
        previous = 0
        pieces = []
        if not event.spans:
            raise ValueError(
                "partial validity needs at least one nonempty span"
            )
        for partial_span in event.spans:
            first = _integer(partial_span.request_start, "partial start")
            stop = _integer(
                partial_span.request_stop, "partial stop", minimum=1
            )
            if first < previous or stop <= first or stop > size:
                raise ValueError(
                    "partial spans must be ordered disjoint FFT ranges"
                )
            physical = _integer(
                partial_span.physical_start, "partial raw physical sample"
            )
            if physical != base + first:
                raise ValueError(
                    "partial validity does not match raw source identity"
                )
            pieces.append((first, stop, physical - ordinary - first))
            previous = stop
        if len(pieces) == 1:
            first, stop, _ = pieces[0]
            local_validity.append((index, first, stop))
        else:
            segments.append((index, tuple(pieces)))
    for first, stop in empty_rows:
        address_at(first)
        local_validity.extend((index, 0, 0) for index in range(first, stop))
    return GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(),
        checkpoints=(),
        calibration_identifier=None,
        timeline_fft_count=count,
        ineligible_intervals=rejected,
        source_replay_intervals=tuple(replay),
        source_segment_events=tuple(segments),
        source_validity_events=tuple(sorted(local_validity)),
    )


@dataclass(frozen=True, slots=True)
class RawReplayPlanWindow:
    """Explicit known global scope; decisions elsewhere remain unresolved.

    Sparse execution selectors know no scope. Obtain their plan only after
    checking the consumer's requested extent with ``plan_for_scope``; passing
    it to a whole-timeline consumer would violate this unwired contract.
    """

    first_fft: int
    stop_fft: int
    _plan: GlobalFFTEligibilityPlan

    def plan_for_scope(
        self, first_fft: int, stop_fft: int
    ) -> GlobalFFTEligibilityPlan:
        first = _integer(first_fft, "requested first FFT")
        stop = _integer(stop_fft, "requested stop FFT", minimum=1)
        if first < self.first_fft or stop > self.stop_fft or stop <= first:
            raise ValueError(
                "requested decisions lie outside the known FFT scope"
            )
        return self._plan

    def is_eligible(self, index: int) -> bool:
        return self.plan_for_scope(index, index + 1).is_eligible(index)

    def source_replay(self, index: int) -> tuple[int, int]:
        return self.plan_for_scope(index, index + 1).source_replay(index)

    def source_validity(self, index: int) -> tuple[int, int] | None:
        return self.plan_for_scope(index, index + 1).source_validity(index)

    def source_segments(
        self, index: int
    ) -> tuple[tuple[int, int, int], ...] | None:
        return self.plan_for_scope(index, index + 1).source_segments(index)


def compile_raw_replay_plan(
    addresses: LatchedRawSourceCompilation,
    validity: RawValidityCompilation,
    *,
    timeline_fft_count: int,
    fft_samples: int,
    ordinary_sample_origin: int | None = None,
    ordinary_source_spans: tuple[OrdinaryRawSourceSpan, ...] | None = None,
    ineligible_intervals: tuple[tuple[int, int], ...],
    maximum_empty_fft_events: int = 4096,
) -> GlobalFFTEligibilityPlan:
    """Encode a complete timeline under the raw-linear coordinate contract.

    Choose either a global affine ordinary sample origin or complete sparse
    ordinary read spans. Their starts must match actual execution reads in the
    same raw sample coordinate system as the supplied physical addresses.
    This adapter does not infer coarse delay or source presentation.
    For bounded nonzero global starts, use ``compile_raw_replay_window``.
    """
    return _compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=timeline_fft_count,
        fft_samples=fft_samples,
        ordinary_sample_origin=ordinary_sample_origin,
        ordinary_source_spans=ordinary_source_spans,
        ineligible_intervals=ineligible_intervals,
        maximum_empty_fft_events=maximum_empty_fft_events,
    )


def compile_raw_replay_window(
    addresses: LatchedRawSourceCompilation,
    validity: RawValidityCompilation,
    *,
    first_fft: int,
    stop_fft: int,
    fft_samples: int,
    ordinary_sample_origin: int | None = None,
    ordinary_source_spans: tuple[OrdinaryRawSourceSpan, ...] | None = None,
    ineligible_intervals: tuple[tuple[int, int], ...],
    maximum_empty_fft_events: int = 4096,
) -> RawReplayPlanWindow:
    """Encode only [first_fft, stop_fft), retaining global plan coordinates.

    Inputs must be explicitly clipped to this known scope. Prefix/suffix
    eligibility remains unknown, never invented as accepted or discarded.
    An affine origin remains relative to global FFT zero; explicit ordinary
    read spans must instead be clipped to this same known global scope. Their
    starts are caller-derived source facts, never output-coordinate selectors.
    Both bases must use the actual execution storage's raw sample coordinates;
    no conversion from timestamp-normalized storage is inferred.
    """
    plan = _compile_raw_replay_plan(
        addresses,
        validity,
        first_fft=first_fft,
        timeline_fft_count=stop_fft,
        fft_samples=fft_samples,
        ordinary_sample_origin=ordinary_sample_origin,
        ordinary_source_spans=ordinary_source_spans,
        ineligible_intervals=ineligible_intervals,
        maximum_empty_fft_events=maximum_empty_fft_events,
    )
    return RawReplayPlanWindow(first_fft, stop_fft, plan)


@overload
def with_canonical_raw_linear_presentation(
    plan: GlobalFFTEligibilityPlan,
    addresses: LatchedRawSourceCompilation,
    *,
    fft_samples: int,
    samples_per_frame: int,
    job_start_slot: int,
    source_index: int,
) -> GlobalFFTEligibilityPlan: ...


@overload
def with_canonical_raw_linear_presentation(
    plan: RawReplayPlanWindow,
    addresses: LatchedRawSourceCompilation,
    *,
    fft_samples: int,
    samples_per_frame: int,
    job_start_slot: int,
    source_index: int,
) -> RawReplayPlanWindow: ...


def with_canonical_raw_linear_presentation(
    plan: GlobalFFTEligibilityPlan | RawReplayPlanWindow,
    addresses: LatchedRawSourceCompilation,
    *,
    fft_samples: int,
    samples_per_frame: int,
    job_start_slot: int,
    source_index: int,
) -> GlobalFFTEligibilityPlan | RawReplayPlanWindow:
    """Attach the smallest enclosing raw frame extent to an existing plan.

    Compile ``plan`` with actual job-relative ordinary GPU read starts first:
    under this canonical storage those starts are also raw sample positions.
    Caller supplies both geometries, the absolute job-start frame slot and
    the datastream source-file index; no header-derived normalization occurs.

    Every accepted source address contributes its full FFT read extent,
    including locally invalid samples. Frame rounding retains edge payload.
    Addresses and supplied plan discards must cover the entire known scope.
    The plan's offsets, masks, ordinary frame validity and exposure policy are
    preserved. Unresolved input and conflicting presentations fail closed.

    The existing recording interpreter must receive a destination thread
    window enclosing these presented slots and a bounded physical source
    range covering the extent. This helper does not plan those inputs. A
    Execution consumers must call ``plan_for_scope`` on the scoped result.
    """
    if not isinstance(addresses, LatchedRawSourceCompilation):
        raise TypeError("addresses must be a LatchedRawSourceCompilation")
    if addresses.unresolved:
        raise ValueError("unresolved raw addresses cannot be presented")
    size = _integer(fft_samples, "FFT samples", minimum=1)
    frame_samples = _integer(
        samples_per_frame, "samples per frame", minimum=1
    )
    source = _integer(source_index, "source file index")
    if not isinstance(job_start_slot, int) or isinstance(
        job_start_slot, bool
    ):
        raise TypeError("job start slot must be an integer")
    if isinstance(plan, RawReplayPlanWindow):
        first, stop = plan.first_fft, plan.stop_fft
        global_plan = plan.plan_for_scope(first, stop)
    elif isinstance(plan, GlobalFFTEligibilityPlan):
        first, stop = 0, plan.fft_count
        global_plan = plan
    else:
        raise TypeError("plan must be a raw replay plan or scoped raw plan")
    if global_plan.timeline_fft_count is None:
        raise ValueError("canonical raw presentation requires a sparse plan")
    if (
        global_plan.source_frame_intervals
        or global_plan.source_physical_frame_intervals
        or global_plan.source_physical_frame_overrides
    ):
        raise ValueError(
            "canonical raw presentation conflicts with presentation"
        )
    rows = []
    raw_starts = []
    raw_stops = []
    previous = first
    for span in addresses.spans:
        index = _integer(span.first_fft, "source first FFT")
        end = _integer(span.stop_fft, "source stop FFT", minimum=1)
        start = _integer(span.physical_sample_start, "raw physical sample")
        if index < previous or end <= index or end > stop:
            raise ValueError(
                "raw address spans must be ordered within plan scope"
            )
        previous = end
        rows.append((index, end))
        raw_starts.append(start)
        raw_stops.append(start + (end - index) * size)
    cursor = first
    for index, end in sorted(rows + list(global_plan.ineligible_intervals)):
        if index != cursor:
            raise ValueError(
                "raw addresses and discards must partition plan scope"
            )
        cursor = end
    if cursor != stop:
        raise ValueError(
            "raw addresses must cover every accepted FFT in scope"
        )
    intervals: tuple[tuple[int, int, int, int], ...] = ()
    if raw_starts:
        min_frame = min(raw_starts) // frame_samples
        max_frame = -(-max(raw_stops) // frame_samples)
        intervals = (
            (source, min_frame, max_frame, job_start_slot + min_frame),
        )
    presented = replace(
        global_plan, source_physical_frame_intervals=intervals
    )
    if isinstance(plan, RawReplayPlanWindow):
        return RawReplayPlanWindow(first, stop, presented)
    return presented


@dataclass(frozen=True, slots=True)
class RawLinearStorage:
    """One caller-chosen storage contract shared by a datastream's consumers.

    Physical frame p is stored at job_start_slot+p+translation_frames. Caller
    must aggregate physical/ordinary extents across all sharing bands, scans
    and windows, then reuse this SAME object; helpers do not choose a separate
    translation for each band/window. Bounds are input source facts, never
    observed output calibration coordinates.
    """

    datastream_id: int
    source_index: int
    job_start_slot: int
    samples_per_frame: int
    raw_first_frame: int
    raw_stop_frame: int
    translation_frames: int

    def __post_init__(self):
        _integer(self.datastream_id, "datastream ID")
        _integer(self.source_index, "source file index")
        _integer(self.samples_per_frame, "samples per frame", minimum=1)
        _integer(self.raw_first_frame, "first raw frame")
        _integer(self.raw_stop_frame, "stop raw frame", minimum=1)
        if self.raw_stop_frame <= self.raw_first_frame:
            raise ValueError("common raw storage extent must be nonempty")
        for name in ("job_start_slot", "translation_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(name + " must be a signed integer")

    @classmethod
    def align_to_ordinary_window(
        cls,
        *,
        datastream_id: int,
        source_index: int,
        job_start_slot: int,
        samples_per_frame: int,
        raw_first_frame: int,
        raw_stop_frame: int,
        ordinary_min_sample: int,
    ) -> RawLinearStorage:
        """Align common minimum physical frame to the ordinary window floor.

        All extents must already be aggregated for this datastream's chosen
        shared scope. This removes a constant displacement but cannot reduce
        the width of an intrinsically wide/disjoint physical source inventory.
        """
        if not isinstance(ordinary_min_sample, int) or isinstance(
            ordinary_min_sample, bool
        ):
            raise TypeError(
                "ordinary minimum sample must be a signed integer"
            )
        size = _integer(samples_per_frame, "samples per frame", minimum=1)
        return cls(
            datastream_id,
            source_index,
            job_start_slot,
            size,
            raw_first_frame,
            raw_stop_frame,
            ordinary_min_sample // size - raw_first_frame,
        )

    @property
    def translation_samples(self) -> int:
        return self.translation_frames * self.samples_per_frame

    @property
    def physical_presentation(self) -> tuple[tuple[int, int, int, int], ...]:
        return (
            (
                self.source_index,
                self.raw_first_frame,
                self.raw_stop_frame,
                self.job_start_slot
                + self.raw_first_frame
                + self.translation_frames,
            ),
        )


def compile_translated_raw_replay_window(
    storage: RawLinearStorage,
    addresses: LatchedRawSourceCompilation,
    validity: RawValidityCompilation,
    *,
    first_fft: int,
    stop_fft: int,
    fft_samples: int,
    ordinary_gpu_source_spans: tuple[OrdinaryRawSourceSpan, ...],
    ineligible_intervals: tuple[tuple[int, int], ...],
    maximum_empty_fft_events: int = 4096,
) -> RawReplayPlanWindow:
    """Compile replay into a supplied common translated raw storage layout.

    ``ordinary_gpu_source_spans`` must be the actual job-relative ordinary
    reads evaluated by the frontend. ONLY the replay adapter's coordinate
    basis is shifted: raw_basis = ordinary_GPU_start - translation_samples.
    Raw clock/source compilers must still receive unshifted ordinary starts.
    Source physical p then replays at p+translation_samples using the existing
    GPU offsets. No storage/reference translation is inferred from outputs.

    Every returned window carries identical common presentation metadata,
    ensuring bands/scans share one datastream storage grid. Its accepted raw
    extent must fit storage bounds; a conflicting/incomplete scope fails.
    """
    if not isinstance(storage, RawLinearStorage):
        raise TypeError("storage must be a RawLinearStorage")
    if not isinstance(ordinary_gpu_source_spans, tuple) or not all(
        isinstance(span, OrdinaryRawSourceSpan)
        for span in ordinary_gpu_source_spans
    ):
        raise TypeError(
            "ordinary GPU spans must be an OrdinaryRawSourceSpan tuple"
        )
    for span in ordinary_gpu_source_spans:
        if not isinstance(span.ordinary_sample_start, int) or isinstance(
            span.ordinary_sample_start, bool
        ):
            raise TypeError("ordinary GPU starts must be signed integers")
    basis = tuple(
        OrdinaryRawSourceSpan(
            span.first_fft,
            span.stop_fft,
            span.ordinary_sample_start - storage.translation_samples,
        )
        for span in ordinary_gpu_source_spans
    )
    window = compile_raw_replay_window(
        addresses,
        validity,
        first_fft=first_fft,
        stop_fft=stop_fft,
        fft_samples=fft_samples,
        ordinary_source_spans=basis,
        ineligible_intervals=ineligible_intervals,
        maximum_empty_fft_events=maximum_empty_fft_events,
    )
    canonical = with_canonical_raw_linear_presentation(
        window,
        addresses,
        fft_samples=fft_samples,
        samples_per_frame=storage.samples_per_frame,
        job_start_slot=storage.job_start_slot,
        source_index=storage.source_index,
    )
    plan = canonical.plan_for_scope(first_fft, stop_fft)
    for _, first, stop, _ in plan.source_physical_frame_intervals:
        if first < storage.raw_first_frame or stop > storage.raw_stop_frame:
            # The address of a partial FFT describes its whole coordinate
            # range, while authoritative validity can require only a prefix
            # at EOF. Validate the voltages actually consumed; invalid
            # samples are masked and must not demand nonexistent frames.
            lower = storage.raw_first_frame * storage.samples_per_frame
            upper = storage.raw_stop_frame * storage.samples_per_frame
            required = [
                (
                    span.physical_sample_start,
                    span.physical_sample_start
                    + (span.stop_fft - span.first_fft) * fft_samples,
                )
                for span in validity.full_spans
            ]
            required.extend(
                (
                    span.physical_start,
                    span.physical_start
                    + span.request_stop
                    - span.request_start,
                )
                for event in validity.partial_events
                for span in event.spans
            )
            if any(start < lower or end > upper for start, end in required):
                raise ValueError(
                    "accepted raw source lies outside common storage extent"
                )
    translated = replace(
        plan, source_physical_frame_intervals=storage.physical_presentation
    )
    return RawReplayPlanWindow(first_fft, stop_fft, translated)
