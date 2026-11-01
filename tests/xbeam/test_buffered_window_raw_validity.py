# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.addressing import (
    LatchedRawSourceCompilation,
    LatchedRawSourceSpan,
    UnresolvedRawSourceSpan,
)
from cuphoton.xbeam.buffered_reader.clocks import (
    RawWindowPayloadSpans,
    compile_raw_window_payload_spans,
)
from cuphoton.xbeam.buffered_reader.geometry import (
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.plans import compile_raw_replay_window
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import (
    RawValidityCompilation,
    RawValidityState,
    intersect_raw_decoder_validity,
    split_latched_raw_source_validity,
)


def state(runs):
    runs = np.asarray(runs, dtype=np.int64)
    timeline = PhysicalAllocationTimeline.from_runs(
        runs[:, 0], runs[:, 1], runs[:, 0], allocation_frames=1206
    )
    return RawValidityState.from_timeline(
        timeline,
        RawBufferedRequestState.from_timeline(timeline, samples_per_frame=1),
    )


def addresses(first, stop, physical=0, unresolved=()):
    return LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(first, stop, physical),), unresolved
    )


def selections(result):
    return {
        event.fft_index: tuple(
            (span.request_start, span.request_stop, span.physical_start)
            for span in event.spans
        )
        for event in result.partial_events
    }


def test_multiple_holes_preserve_disjoint_prefix_middle_tail():
    source = state([(0, 3), (5, 9), (11, 20)])
    result = split_latched_raw_source_validity(
        source, addresses(17, 19), fft_samples=8
    )
    assert selections(result) == {
        17: ((0, 3, 0), (5, 8, 5)),
        18: ((0, 1, 8), (3, 8, 11)),
    }
    assert result.full_spans == ()
    assert result.empty_intervals == ()
    assert result.complete


def test_frame_hole_crossing_fft_edge_retains_prefix_and_tail():
    source = state([(0, 6), (14, 40)])
    result = split_latched_raw_source_validity(
        source, addresses(0, 5), fft_samples=8
    )
    assert selections(result) == {0: ((0, 6, 0),), 1: ((6, 8, 14),)}
    assert result.full_spans == (LatchedRawSourceSpan(2, 5, 16),)
    assert result.empty_intervals == ()


def test_two_million_empty_ffts_remain_a_compact_interval():
    count = 2_000_000
    source = state([(0, 8), (count * 8 - 8, count * 8)])
    result = split_latched_raw_source_validity(
        source, addresses(0, count), fft_samples=8
    )
    assert result.full_spans == (
        LatchedRawSourceSpan(0, 1, 0),
        LatchedRawSourceSpan(count - 1, count, count * 8 - 8),
    )
    assert result.empty_intervals == ((1, count - 1),)
    assert result.partial_events == ()


def test_backward_source_addresses_and_unresolved_scope_survive():
    source = state([(0, 40)])
    unresolved = (UnresolvedRawSourceSpan(12, 13, "unobserved EOF"),)
    compilation = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(8, 10, 16), LatchedRawSourceSpan(10, 12, 0)),
        unresolved,
    )
    result = split_latched_raw_source_validity(
        source, compilation, fft_samples=8
    )
    assert result.full_spans == compilation.spans
    assert result.unresolved == unresolved
    assert not result.complete


def test_invalid_address_order_fails():
    source = state([(0, 40)])
    compilation = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(8, 10, 0), LatchedRawSourceSpan(9, 11, 16)), ()
    )
    with pytest.raises(ValueError, match="ordered"):
        split_latched_raw_source_validity(source, compilation, fft_samples=8)


@pytest.mark.parametrize("delay", [0, 16384])
def test_explicit_source_span_half_frame_hole_and_exact_origin(delay):
    samples, size = 32768, 262144
    origin = 2**55 + 3
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0, 414]),
        np.array([406, 2000]),
        np.array([origin, origin + 414]),
        allocation_frames=1206,
    )
    requests = RawBufferedRequestState.from_timeline(
        timeline,
        samples_per_frame=samples,
        timestamp_origin_frame=origin,
    )
    cached = RawValidityState.from_timeline(timeline, requests)
    addresses = LatchedRawSourceCompilation(
        (LatchedRawSourceSpan(50, 100, 400 * samples - delay),), ()
    )
    result = split_latched_raw_source_validity(
        cached, addresses, fft_samples=size
    )
    edge = 6 * samples + delay
    assert selections(result) == {
        50: ((0, edge, 400 * samples - delay),),
        51: ((edge, size, 414 * samples),),
    }
    assert result.full_spans == (
        LatchedRawSourceSpan(52, 100, 416 * samples - delay),
    )
    assert result.empty_intervals == ()
    assert result.complete


def decoder_intersection(validity, indices, spans, *, size=8):
    pairs = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    return intersect_raw_decoder_validity(
        validity,
        np.asarray(indices, dtype=np.int64),
        RawWindowPayloadSpans(pairs[:, 0], pairs[:, 1]),
        fft_samples=size,
    )


def test_decoder_clips_disjoint_holes_and_roundtrips_plan_provenance():
    # FFT 17 starts at physical 2: two holes cross its edges and a third
    # is strictly inside FFT 18. The decoder independently clips both ends.
    source = state([(0, 3), (5, 9), (11, 13), (14, 40)])
    raw = addresses(17, 21, physical=2)
    validity = split_latched_raw_source_validity(source, raw, fft_samples=8)
    result = decoder_intersection(
        validity, [17, 18, 19, 20], [(2, 7), (0, 7), (4, 4), (0, 8)]
    )
    assert selections(result) == {
        17: ((3, 7, 5),),
        18: ((1, 3, 11), (4, 7, 14)),
    }
    assert result.full_spans == (LatchedRawSourceSpan(20, 21, 26),)
    assert result.empty_intervals == ((19, 20),)
    plan = compile_raw_replay_window(
        raw,
        result,
        first_fft=17,
        stop_fft=21,
        fft_samples=8,
        ordinary_sample_origin=2**55 + 3,
        ineligible_intervals=(),
    )
    inventory = {
        p
        for first, stop in source.physical_valid_spans
        for p in range(first, stop)
    }
    for index, (left, right) in zip(
        range(17, 21), [(2, 7), (0, 7), (4, 4), (0, 8)], strict=True
    ):
        expected = {
            local: 2 + (index - 17) * 8 + local
            for local in range(left, right)
            if 2 + (index - 17) * 8 + local in inventory
        }
        segments = plan.source_segments(index)
        ordinary = 2**55 + 3 + index * 8
        if segments is not None:
            observed = {
                local: ordinary + local + offset
                for first, stop, offset in segments
                for local in range(first, stop)
            }
        else:
            first, stop = plan.source_validity(index) or (0, 8)
            offset, cutover = plan.source_replay(index)
            observed = {
                local: ordinary + local + (offset if local >= cutover else 0)
                for local in range(first, stop)
            }
        assert observed == expected
        assert plan.is_eligible(index)
    encoded = plan.plan_for_scope(17, 21)
    assert not encoded.has_authoritative_source_validity
    assert encoded.source_weight_override_events == ()
    assert encoded.source_weight_credit_intervals == ()
    assert encoded.source_weight_exclusion_events == ()


def test_decoder_sparse_runs_large_physical_origin_and_noop_identity():
    count, origin = 10**12, 2**55 + 7
    validity = RawValidityCompilation(
        (LatchedRawSourceSpan(9, 9 + count, origin),), (), (), ()
    )
    assert decoder_intersection(validity, [10], [(0, 8)]) is validity
    assert decoder_intersection(validity, [], []) is validity
    result = decoder_intersection(
        validity, [11, 9 + count - 1], [(2, 6), (8, 8)]
    )
    assert result.full_spans == (
        LatchedRawSourceSpan(9, 11, origin),
        LatchedRawSourceSpan(12, 9 + count - 1, origin + 24),
    )
    assert selections(result) == {11: ((2, 6, origin + 18),)}
    assert result.empty_intervals == ((9 + count - 1, 9 + count),)


def test_decoder_empty_rows_merge_but_unresolved_scope_survives():
    unresolved = (UnresolvedRawSourceSpan(24, 26, "unknown publication"),)
    validity = RawValidityCompilation(
        (LatchedRawSourceSpan(20, 22, 80),), (), ((22, 24),), unresolved
    )
    result = decoder_intersection(validity, [21, 22], [(5, 5), (0, 8)])
    assert result.empty_intervals == ((21, 24),)
    assert result.unresolved is unresolved
    assert not result.complete
    for index in (19, 24, 26):
        with pytest.raises(ValueError, match="resolved validity scope"):
            decoder_intersection(validity, [index], [(0, 0)])


def test_decoder_generated_window_spans_preserve_intersection_laws():
    raw = addresses(0, 6)
    source = state([(0, 5), (7, 19), (21, 25), (30, 48)])
    validity = split_latched_raw_source_validity(source, raw, fft_samples=8)
    spans = compile_raw_window_payload_spans(
        np.array([-3, 0, 6, 9, 2, 0], dtype=np.int64),
        np.array([20, 8, 10, 3, 7, 8], dtype=np.int64),
        fft_samples=8,
    )
    result = intersect_raw_decoder_validity(
        validity, np.arange(6, dtype=np.int64), spans, fft_samples=8
    )
    assert decoder_intersection(result, [], []) is result
    assert (
        intersect_raw_decoder_validity(
            result, np.arange(6, dtype=np.int64), spans, fft_samples=8
        )
        == result
    )
    assert selections(result) == {
        0: ((3, 5, 3), (7, 8, 7)),
        2: ((0, 3, 16),),
        4: ((0, 5, 32),),
    }
    assert result.empty_intervals == ((3, 4),)
    assert result.full_spans == (
        LatchedRawSourceSpan(1, 2, 8),
        LatchedRawSourceSpan(5, 6, 40),
    )


@pytest.mark.parametrize(
    "indices,spans",
    [
        ([1, 1], [(0, 8), (0, 8)]),
        ([2, 1], [(0, 8), (0, 8)]),
        ([-1], [(0, 8)]),
        ([1], [(-1, 8)]),
        ([1], [(0, 9)]),
        ([1], [(4, 3)]),
    ],
)
def test_decoder_rejects_invalid_sparse_contract(indices, spans):
    validity = RawValidityCompilation(
        (LatchedRawSourceSpan(0, 3, 0),), (), (), ()
    )
    with pytest.raises(ValueError):
        decoder_intersection(validity, indices, spans)


def test_decoder_rejects_overlapping_validity_partitions():
    validity = RawValidityCompilation(
        (LatchedRawSourceSpan(0, 3, 0),), (), ((2, 3),), ()
    )
    with pytest.raises(ValueError, match="partitions"):
        decoder_intersection(validity, [], [])
