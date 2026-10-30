# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.addressing import (
    LatchedRawSourceCompilation,
    LatchedRawSourceSpan,
    UnresolvedRawSourceSpan,
)
from cuphoton.xbeam.buffered_reader.geometry import (
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.plans import (
    OrdinaryRawSourceSpan,
    compile_raw_replay_plan,
    compile_raw_replay_window,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import (
    RawValidityCompilation,
    RawValidityState,
    split_latched_raw_source_validity,
)


def inventory(runs, *, origin=0):
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([first for first, _ in runs], dtype=np.int64),
        np.array([stop for _, stop in runs], dtype=np.int64),
        np.array([origin + first for first, _ in runs], dtype=np.int64),
        allocation_frames=1206,
    )
    requests = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=1, timestamp_origin_frame=origin
    )
    return RawValidityState.from_timeline(timeline, requests)


def adapter(
    state, addresses, *, count, size=8, origin=0, rejects=(), **kwargs
):
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=size
    )
    return compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=count,
        fft_samples=size,
        ordinary_sample_origin=origin,
        ineligible_intervals=rejects,
        **kwargs,
    )


def samples(plan, index, size, origin):
    if not plan.is_eligible(index):
        return None
    segments = plan.source_segments(index)
    if segments is not None:
        return {
            local: origin + index * size + local + offset
            for first, stop, offset in segments
            for local in range(first, stop)
        }
    offset, cutover = plan.source_replay(index)
    first, stop = plan.source_validity(index) or (0, size)
    return {
        local: origin
        + index * size
        + local
        + (offset if local >= cutover else 0)
        for local in range(first, stop)
    }


@pytest.mark.parametrize("origin", [0, -16384, 2**55 + 3])
def test_holes_disjoint_partial_spans_and_source_basis(origin):
    runs = [(0, 3), (5, 9), (11, 40)]
    state = inventory(runs, origin=2**55 + 3)
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 5, 0),), ()
    )
    plan = adapter(state, addresses, count=5, origin=origin)
    valid = {sample for first, stop in runs for sample in range(first, stop)}
    for fft in range(5):
        assert samples(plan, fft, 8, origin) == {
            local: fft * 8 + local
            for local in range(8)
            if fft * 8 + local in valid
        }
        assert plan.is_eligible(fft)
    assert plan.source_segments(0) is not None
    assert not plan.has_authoritative_source_validity
    assert plan.source_weight_override_events == ()
    assert plan.source_weight_exclusion_events == ()
    assert plan.source_weight_credit_intervals == ()
    assert plan.source_physical_frame_intervals == ()
    assert plan.calibration_identifier is None


def test_partial_hole_at_discard_boundary_preserves_eligibility():
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 2, 0), LatchedRawSourceSpan(3, 5, 24)), ()
    )
    plan = adapter(
        inventory([(0, 6), (14, 40)]), addresses, count=5, rejects=((2, 3),)
    )
    assert plan.ineligible_intervals == ((2, 3),)
    assert samples(plan, 0, 8, 0) == {i: i for i in range(6)}
    assert samples(plan, 1, 8, 0) == {6: 14, 7: 15}
    assert samples(plan, 2, 8, 0) is None
    assert samples(plan, 3, 8, 0) == {i: 24 + i for i in range(8)}


def test_empty_source_rows_are_zero_validity_not_discarded():
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 5, 0),), ()
    )
    state = inventory([(0, 8), (32, 40)])
    plan = adapter(state, addresses, count=5)
    assert plan.ineligible_intervals == ()
    assert plan.source_validity_events == ((1, 0, 0), (2, 0, 0), (3, 0, 0))
    for index in range(1, 4):
        assert plan.is_eligible(index)
        assert samples(plan, index, 8, 0) == {}
    with pytest.raises(ValueError, match="event limit"):
        adapter(state, addresses, count=5, maximum_empty_fft_events=2)


def test_trillion_full_rows_keep_one_sparse_offset_interval():
    count = 10**12
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, count, 0),), ()
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    plan = compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=count,
        fft_samples=8,
        ordinary_sample_origin=-17,
        ineligible_intervals=(),
    )
    assert plan.eligible_ffts == ()
    assert plan.source_replay_intervals == ((0, count, 17),)
    assert plan.source_validity_events == ()
    assert plan.source_replay(count - 1) == (17, 0)


def test_unresolved_and_incomplete_decisions_fail_closed():
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 4, 0),),
        (UnresolvedRawSourceSpan(4, 5, "EOF unknown"),),
    )
    state = inventory([(0, 40)])
    with pytest.raises(ValueError, match="unresolved"):
        adapter(state, addresses, count=5)
    with pytest.raises(ValueError, match="complete timeline"):
        adapter(state, replace(addresses, unresolved=()), count=5)


def test_wrong_validity_physical_identity_rejected():
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 5, 0),), ()
    )
    wrong = RawValidityCompilation(
        (LatchedRawSourceSpan(0, 5, 1),), (), (), ()
    )
    with pytest.raises(ValueError, match="source identity"):
        compile_raw_replay_plan(
            addresses,
            wrong,
            timeline_fft_count=5,
            fft_samples=8,
            ordinary_sample_origin=0,
            ineligible_intervals=(),
        )


def test_source_offset_change_and_backward_physical_read_are_retained():
    state = inventory([(0, 80)])
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 2, 16), LatchedRawSourceSpan(2, 5, 0)), ()
    )
    plan = adapter(state, addresses, count=5)
    assert plan.source_replay_intervals == ((0, 2, 16), (2, 5, -16))
    assert samples(plan, 1, 8, 0) == {local: 24 + local for local in range(8)}
    assert samples(plan, 2, 8, 0) == {local: local for local in range(8)}


def test_inconsistent_partial_provenance_and_overlapping_discards_rejected():
    state = inventory([(0, 3), (5, 40)])
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 5, 0),), ()
    )
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=8
    )
    first = validity.partial_events[0]
    wrong_span = replace(first.spans[0], physical_start=1)
    wrong = replace(
        validity, partial_events=(replace(first, spans=(wrong_span,)),)
    )
    with pytest.raises(ValueError, match="source identity"):
        compile_raw_replay_plan(
            addresses,
            wrong,
            timeline_fft_count=5,
            fft_samples=8,
            ordinary_sample_origin=0,
            ineligible_intervals=(),
        )
    with pytest.raises(ValueError, match="without overlap"):
        adapter(state, addresses, count=5, rejects=((2, 3),))


@pytest.mark.parametrize("origin", [-(2**55 + 3), 2**55 + 3])
def test_nonzero_window_matches_whole_global_decisions_after_clipping(origin):
    state = inventory([(0, 18), (22, 34), (38, 80)])
    whole_addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 5, 0), LatchedRawSourceSpan(6, 10, 48)), ()
    )
    whole = adapter(
        state, whole_addresses, count=10, origin=origin, rejects=((5, 6),)
    )
    sliced_addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(2, 5, 16), LatchedRawSourceSpan(6, 8, 48)), ()
    )
    sliced_validity = split_latched_raw_source_validity(
        state, sliced_addresses, fft_samples=8
    )
    window = compile_raw_replay_window(
        sliced_addresses,
        sliced_validity,
        first_fft=2,
        stop_fft=8,
        fft_samples=8,
        ordinary_sample_origin=origin,
        ineligible_intervals=((5, 6),),
    )
    plan = window.plan_for_scope(2, 8)
    assert plan.fft_count == 8
    assert plan.ineligible_intervals == whole.ineligible_intervals
    assert plan.source_replay_intervals == ((2, 5, -origin), (6, 8, -origin))
    assert plan.source_validity_events == tuple(
        row for row in whole.source_validity_events if 2 <= row[0] < 8
    )
    assert plan.source_segment_events == tuple(
        row for row in whole.source_segment_events if 2 <= row[0] < 8
    )
    for index in range(2, 8):
        assert window.is_eligible(index) == whole.is_eligible(index)
        assert samples(window, index, 8, origin) == samples(
            whole, index, 8, origin
        )
    for index in (0, 1, 8, 9):
        for accessor in (
            window.is_eligible,
            window.source_replay,
            window.source_validity,
            window.source_segments,
        ):
            with pytest.raises(ValueError, match="outside"):
                accessor(index)
    with pytest.raises(ValueError, match="outside"):
        window.plan_for_scope(1, 8)
    with pytest.raises(ValueError, match="outside"):
        window.plan_for_scope(2, 9)


def test_window_requires_explicit_clipping_and_complete_known_scope():
    state = inventory([(0, 80)])
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 10, 0),), ()
    )
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=8
    )
    with pytest.raises(ValueError, match="ordered disjoint"):
        compile_raw_replay_window(
            addresses,
            validity,
            first_fft=2,
            stop_fft=8,
            fft_samples=8,
            ordinary_sample_origin=0,
            ineligible_intervals=(),
        )


@pytest.mark.parametrize("origin", [-(2**55 + 3), 2**55 + 3])
def test_piecewise_ordinary_coarse_steps_preserve_whole_and_window_sources(
    origin,
):
    runs = [(0, 18), (22, 34), (38, 80)]
    state = inventory(runs)
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 5, 0), LatchedRawSourceSpan(6, 10, 48)), ()
    )
    validity = split_latched_raw_source_validity(
        state, addresses, fft_samples=8
    )
    ordinary = (
        OrdinaryRawSourceSpan(0, 3, origin),
        OrdinaryRawSourceSpan(3, 7, origin + 3 * 8 - 1),
        OrdinaryRawSourceSpan(7, 10, origin + 7 * 8 + 2),
    )
    whole = compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=10,
        fft_samples=8,
        ordinary_source_spans=ordinary,
        ineligible_intervals=((5, 6),),
    )
    assert whole.source_replay_intervals == (
        (0, 3, -origin),
        (3, 5, 1 - origin),
        (6, 7, 1 - origin),
        (7, 10, -2 - origin),
    )
    window_addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(2, 5, 16), LatchedRawSourceSpan(6, 8, 48)), ()
    )
    window_validity = split_latched_raw_source_validity(
        state, window_addresses, fft_samples=8
    )
    window_ordinary = tuple(
        OrdinaryRawSourceSpan(
            max(2, span.first_fft),
            min(8, span.stop_fft),
            span.ordinary_sample_start
            + (max(2, span.first_fft) - span.first_fft) * 8,
        )
        for span in ordinary
    )
    window = compile_raw_replay_window(
        window_addresses,
        window_validity,
        first_fft=2,
        stop_fft=8,
        fft_samples=8,
        ordinary_source_spans=window_ordinary,
        ineligible_intervals=((5, 6),),
    )
    selected = window.plan_for_scope(2, 8)
    assert selected.source_replay_intervals == tuple(
        (max(2, first), min(8, stop), offset)
        for first, stop, offset in whole.source_replay_intervals
    )
    assert selected.source_segment_events == tuple(
        event for event in whole.source_segment_events if 2 <= event[0] < 8
    )
    assert selected.source_validity_events == tuple(
        event for event in whole.source_validity_events if 2 <= event[0] < 8
    )
    valid = {sample for first, stop in runs for sample in range(first, stop)}
    for index in range(10):
        coarse = 0 if index < 3 else (1 if index < 7 else -2)
        # Independent ordinary read: output coordinate minus coarse delay.
        effective_origin = origin - coarse
        expected = (
            None
            if index == 5
            else {
                local: index * 8 + local
                for local in range(8)
                if index * 8 + local in valid
            }
        )
        assert samples(whole, index, 8, effective_origin) == expected
        if 2 <= index < 8:
            assert samples(window, index, 8, effective_origin) == expected


def test_piecewise_trillion_fft_basis_stays_three_sparse_intersections():
    count = 10**12
    first, second = count // 3, 2 * count // 3
    origin = -(2**55 + 3)
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, count, 0),), ()
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    ordinary = (
        OrdinaryRawSourceSpan(0, first, origin),
        OrdinaryRawSourceSpan(first, second, origin + first * 8 - 11),
        OrdinaryRawSourceSpan(second, count, origin + second * 8 + 7),
    )
    plan = compile_raw_replay_plan(
        addresses,
        validity,
        timeline_fft_count=count,
        fft_samples=8,
        ordinary_source_spans=ordinary,
        ineligible_intervals=(),
    )
    assert plan.source_replay_intervals == (
        (0, first, -origin),
        (first, second, 11 - origin),
        (second, count, -7 - origin),
    )
    assert plan.source_replay_events == ()
    assert plan.source_validity_events == ()


@pytest.mark.parametrize(
    "ranges", [((0, 4), (5, 10)), ((0, 6), (5, 10)), ((1, 10),)]
)
def test_piecewise_ordinary_basis_requires_complete_nonoverlapping_scope(
    ranges,
):
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 10, 0),), ()
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    with pytest.raises(ValueError, match="ordinary source"):
        compile_raw_replay_plan(
            addresses,
            validity,
            timeline_fft_count=10,
            fft_samples=8,
            ordinary_source_spans=tuple(
                OrdinaryRawSourceSpan(first, stop, first * 8)
                for first, stop in ranges
            ),
            ineligible_intervals=(),
        )


def test_ordinary_basis_is_explicit_and_affine_convenience_equivalent():
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(0, 10, 0),), ()
    )
    validity = RawValidityCompilation(addresses.spans, (), (), ())
    common = dict(
        timeline_fft_count=10, fft_samples=8, ineligible_intervals=()
    )
    spans = (OrdinaryRawSourceSpan(0, 10, -17),)
    assert compile_raw_replay_plan(
        addresses, validity, ordinary_sample_origin=-17, **common
    ) == compile_raw_replay_plan(
        addresses, validity, ordinary_source_spans=spans, **common
    )
    with pytest.raises(ValueError, match="exactly one"):
        compile_raw_replay_plan(addresses, validity, **common)
    with pytest.raises(ValueError, match="exactly one"):
        compile_raw_replay_plan(
            addresses,
            validity,
            ordinary_sample_origin=-17,
            ordinary_source_spans=spans,
            **common,
        )
