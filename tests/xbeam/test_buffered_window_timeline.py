# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader._checks import BufferedWindowError
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline


def timeline(starts, stops, timestamps):
    return PhysicalAllocationTimeline.from_runs(
        np.asarray(starts, dtype=np.int64),
        np.asarray(stops, dtype=np.int64),
        np.asarray(timestamps, dtype=np.int64),
        allocation_frames=1206,
    )


def test_physical_hole_and_timestamp_residual_are_distinct():
    model = timeline([0, 1210], [1202, 5000], [10000, 12425])
    assert model.physical_gaps.tolist() == [8]
    assert model.timestamp_gaps.tolist() == [1223]
    assert model.residuals.tolist() == [1215]
    assert model.allocation_starts.tolist() == [0, 1206, 2412, 3618, 4824]
    assert model.anchors.tolist() == [10000, 12421, 13627, 14833, 16039]
    (event,) = model.discontinuities
    assert event.physical_start == 1210
    assert event.allocation_index == 1
    assert event.allocation_phase == 4


def test_physical_hole_without_residual_preserves_linear_anchors():
    model = timeline([0, 1210], [1202, 5000], [10000, 11210])
    assert model.residuals.tolist() == [0]
    assert model.anchors.tolist() == [10000, 11206, 12412, 13618, 14824]


def test_timestamp_origin_does_not_change_physical_reader_events():
    first = timeline([0, 1210], [1202, 5000], [10000, 12425])
    second = timeline([0, 1210], [1202, 5000], [1000000, 1002425])
    assert first.discontinuities == second.discontinuities
    np.testing.assert_array_equal(second.anchors - first.anchors, 990000)


@pytest.mark.parametrize("integer_type", [int, np.int64, np.uint64])
def test_allocation_geometry_preserves_integer_event_coordinates(
    integer_type,
):
    model = PhysicalAllocationTimeline.from_runs(
        np.array([0, 1210]),
        np.array([1202, 5000]),
        np.array([10000, 12425]),
        allocation_frames=integer_type(1206),
    )
    event = model.discontinuities[0]
    assert type(event.allocation_index) is int
    assert type(event.allocation_phase) is int
    assert (event.allocation_index, event.allocation_phase) == (1, 4)


@pytest.mark.parametrize(
    "starts,stops,timestamps",
    [
        ([0, 12], [13, 20], [0, 12]),
        ([0, 12], [10, 20], [0, 11]),
        ([1], [10], [1]),
        ([0], [0], [0]),
    ],
)
def test_ambiguous_or_backwards_physical_coordinates_fail(
    starts, stops, timestamps
):
    with pytest.raises(BufferedWindowError):
        timeline(starts, stops, timestamps)


@pytest.mark.parametrize("coordinate", [2**63, 2**64 - 1])
def test_unsigned_coordinate_overflow_is_rejected(coordinate):
    with pytest.raises(BufferedWindowError, match="signed range"):
        PhysicalAllocationTimeline.from_runs(
            np.array([0], dtype=np.uint64),
            np.array([10], dtype=np.uint64),
            np.array([coordinate], dtype=np.uint64),
            allocation_frames=1206,
        )


def test_timestamp_span_overflow_is_rejected():
    with pytest.raises(BufferedWindowError, match="span exceeds"):
        timeline([0], [10], [2**63 - 5])


@pytest.mark.parametrize("samples_per_frame", [1, 3, 32768])
def test_request_crossing_residual_and_filler_preserves_sample_identity(
    samples_per_frame,
):
    # Physical frames 4..6 are filler; timestamp frames 104..110 are absent.
    # The request starts halfway through frame 103 and ends in frame 111.
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0, 6]),
        np.array([4, 10]),
        np.array([100, 110]),
        allocation_frames=7,
    )
    start = 103 * samples_per_frame + samples_per_frame // 2
    count = 9 * samples_per_frame
    spans = timeline.sample_source_spans(
        start, count, samples_per_frame=samples_per_frame
    )
    actual = [None] * count
    for span in spans:
        for local in range(span.request_start, span.request_stop):
            actual[local] = span.physical_start + local - span.request_start
    expected = []
    for sample in range(start, start + count):
        frame, phase = divmod(sample, samples_per_frame)
        if 100 <= frame < 104:
            expected.append((frame - 100) * samples_per_frame + phase)
        elif 110 <= frame < 114:
            expected.append((frame - 110 + 6) * samples_per_frame + phase)
        else:
            expected.append(None)
    assert actual == expected
    # Translating all timestamps leaves request-local and physical identities
    # unchanged, including large absolute times beyond float exactness.
    translation = 2**48
    translated = PhysicalAllocationTimeline.from_runs(
        np.array([0, 6]),
        np.array([4, 10]),
        np.array([100 + translation, 110 + translation]),
        allocation_frames=7,
    )
    assert (
        translated.sample_source_spans(
            start + translation * samples_per_frame,
            count,
            samples_per_frame=samples_per_frame,
        )
        == spans
    )


def test_source_request_at_run_stop_does_not_reuse_previous_run():
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0, 6]),
        np.array([4, 10]),
        np.array([100, 110]),
        allocation_frames=7,
    )
    assert timeline.sample_source_spans(1040, 60, samples_per_frame=10) == ()
    (span,) = timeline.sample_source_spans(1100, 1, samples_per_frame=10)
    assert (span.request_start, span.request_stop, span.physical_start) == (
        0,
        1,
        60,
    )


@pytest.mark.parametrize("integer_type", [int, np.int64, np.uint64])
def test_source_coordinates_above_float_precision_stay_integer(integer_type):
    origin = 2**55
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0]),
        np.array([4]),
        np.array([origin]),
        allocation_frames=7,
    )
    (span,) = timeline.sample_source_spans(
        integer_type(origin * 3 + 1),
        integer_type(2),
        samples_per_frame=integer_type(3),
    )
    assert (span.request_start, span.request_stop, span.physical_start) == (
        0,
        2,
        1,
    )


def test_source_mapping_is_independent_of_query_windows():
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0, 6, 15]),
        np.array([4, 10, 19]),
        np.array([100, 110, 130]),
        allocation_frames=7,
    )
    start, stop = 302, 400

    def identities(query_start, query_stop):
        result = {}
        for span in timeline.sample_source_spans(
            query_start, query_stop - query_start, samples_per_frame=3
        ):
            for local in range(span.request_start, span.request_stop):
                result[query_start + local] = (
                    span.physical_start + local - span.request_start
                )
        return result

    whole = identities(start, stop)
    tiled = {}
    for boundary, end in ((302, 311), (311, 333), (333, 397), (397, 400)):
        tiled.update(identities(boundary, end))
    assert tiled == whole
    assert len(whole) == 10 + 12 + 10


def test_timeline_source_history_does_not_alias_caller_arrays():
    starts, stops, timestamps = (
        np.array([0]),
        np.array([4]),
        np.array([100]),
    )
    timeline = PhysicalAllocationTimeline.from_runs(
        starts, stops, timestamps, allocation_frames=7
    )
    starts[0], stops[0], timestamps[0] = 20, 24, 300
    (span,) = timeline.sample_source_spans(1000, 4, samples_per_frame=10)
    assert span.physical_start == 0
