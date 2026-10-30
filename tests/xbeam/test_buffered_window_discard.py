# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from bisect import bisect_right
from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    RawBufferedRequestState,
    buffered_request_cursor,
    compile_buffered_request_discards,
    request_control_mask,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline

SMALL = BufferedRequestGeometry(4, 8, 16, 2, 2)


def timeline(runs, capacity=10):
    rows = np.asarray(runs, dtype=np.int64)
    return PhysicalAllocationTimeline.from_runs(
        rows[:, 0], rows[:, 1], rows[:, 2], allocation_frames=capacity
    )


def state(runs=((0, 13, 0), (13, 80, 33)), *, origin=0):
    return RawBufferedRequestState.from_timeline(
        timeline(runs), samples_per_frame=4, timestamp_origin_frame=origin
    )


def compile_(
    model, starts, desired, ordinary, *, counts=None, geometry=SMALL
):
    return compile_buffered_request_discards(
        model,
        np.asarray(starts, dtype=np.int64),
        np.asarray(desired, dtype=np.int64),
        np.zeros(len(starts)),
        np.asarray(ordinary, dtype=np.int64),
        geometry,
        sample_rate=128.0,
        frame_bytes=2,
        control_word_bits=3,
        request_fft_counts=None
        if counts is None
        else np.asarray(counts, dtype=np.int64),
    )


def test_exact_framed_discard_and_successor_reentry():
    model = state()
    assert model.event_allocations == frozenset({1})
    assert model.anchor_samples[:4].tolist() == [0, 40, 160, 200]
    batch = compile_(model, [100], [48], [48 + np.arange(16) * 8])
    assert batch.selected_allocations.tolist() == [1]
    assert batch.cursors.origin_bytes.tolist() == [2]
    assert batch.eligible[0].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert batch.ineligible_intervals == ((109, 114),)


def test_multiple_source_jumps_in_one_allocation_share_net_anchor_change():
    model = state(((0, 13, 0), (13, 17, 23), (17, 80, 37)))
    assert model.event_allocations == frozenset({1})
    batch = compile_(model, [100], [48], [48 + np.arange(16) * 8])
    assert batch.ineligible_intervals == ((109, 114),)


def test_event_exactly_at_boundary_belongs_to_the_completed_read():
    model = state(((0, 20, 0), (20, 80, 40)))
    assert model.event_allocations == frozenset({1})
    batch = compile_(model, [100], [48], [48 + np.arange(16) * 8])
    assert batch.ineligible_intervals == ((109, 114),)


@pytest.mark.parametrize(
    "boundary,owner", [(1, 0), (10, 0), (19, 1), (20, 1), (21, 2)]
)
def test_positive_transition_ownership_on_both_sides_of_read_end(
    boundary, owner
):
    model = state(((0, boundary, 0), (boundary, 80, boundary + 20)))
    assert model.event_allocations == frozenset({owner})


def test_zero_residual_at_boundary_does_not_mark_a_completed_read():
    model = state(((0, 20, 0), (20, 80, 20)))
    assert model.event_allocations == frozenset()
    assert not model.event_flags.any()


def test_negative_residual_is_rejected_before_event_selection():
    with pytest.raises(ValueError, match="backwards timestamp residual"):
        state(((0, 20, 0), (20, 80, 19)))


@pytest.mark.parametrize("phase,event_owner", [(0, 50), (4, 51)])
def test_generated_boundary_and_interior_controls_use_the_input_read_clock(
    phase, event_owner
):
    capacity, origin, residual = 1206, 2000, 1215
    boundary = 51 * capacity + phase
    model = RawBufferedRequestState.from_timeline(
        timeline(
            [(0, boundary, 0), (boundary, 129000, boundary + residual)],
            capacity,
        ),
        samples_per_frame=32768,
        timestamp_origin_frame=origin,
    )
    assert model.event_allocations == frozenset({event_owner})
    geometry = BufferedRequestGeometry(4, 262144, 50, 8192, 4)
    starts = np.arange(0, 15600, 50, dtype=np.int64)
    desired = starts * 262144
    ordinary = desired[:, None] + np.arange(50) * 262144
    result = compile_buffered_request_discards(
        model,
        starts,
        desired,
        np.zeros(starts.size),
        ordinary,
        geometry,
        sample_rate=4096000000,
        frame_bytes=8224,
        control_word_bits=30,
    )
    # Derive the exclusion from generated input anchors and byte extent,
    # before checking any separately executed comparator observation.
    expected = []
    anchors = [
        i * capacity + (residual if i * capacity >= boundary else 0) - origin
        for i in range(107)
    ]
    for first in map(int, starts):
        owner = bisect_right(anchors, first * 8) - 1
        if owner != event_owner:
            continue
        byte_origin = (first * 8 - anchors[owner]) * 8192
        for position in range(50):
            framed = byte_origin + position * 65536 < capacity * 8224
            reentry = (first + position) * 8 >= anchors[owner + 1]
            if not framed and not reentry:
                expected.append(first + position)
    assert result.ineligible_intervals == ((expected[0], expected[-1] + 1),)
    assert len(expected) == (152 if phase == 0 else 151)


def test_filler_without_timestamp_residual_never_creates_discard_event():
    model = state(((0, 13, 0), (15, 80, 15)))
    assert model.event_allocations == frozenset()
    batch = compile_(model, [100], [48], [48 + np.arange(16) * 8])
    assert batch.eligible.all()
    assert batch.ineligible_intervals == ()


def test_dense_allocation_has_no_framed_control_gate():
    batch = compile_(state(), [0], [0], [np.arange(16) * 8])
    assert not batch.event_requests[0]
    assert batch.eligible.all()


def test_reentry_restores_after_control_word_gate():
    batch = compile_(state(), [100], [156], [156 + np.arange(16) * 8])
    assert batch.cursors.origin_bytes[0] == 28
    assert batch.eligible[0].tolist() == [False] + [True] * 15
    assert batch.ineligible_intervals == ((100, 101),)


def test_reentry_uses_each_ordinary_source_start_independently():
    ordinary = 48 + np.arange(16) * 8
    ordinary[[10, 12]] = 160
    batch = compile_(state(), [100], [48], [ordinary])
    assert batch.ineligible_intervals == ((109, 110), (111, 112), (113, 114))


def test_partial_request_ignores_padded_columns():
    batch = compile_(
        state(), [100], [48], [48 + np.arange(16) * 8], counts=[10]
    )
    assert batch.ineligible_intervals == ((109, 110),)
    assert not batch.eligible[0, 10:].any()


def test_overlapping_request_discards_merge_in_global_coordinates():
    batch = compile_(
        state(),
        [100, 104],
        [48, 48],
        [48 + np.arange(16) * 8, 48 + np.arange(16) * 8],
    )
    assert batch.ineligible_intervals == ((109, 118),)


def test_invalid_allocations_leave_initial_source_policy_open():
    batch = compile_(state(), [0], [-1], [np.arange(16) * 8])
    assert not batch.valid_requests[0]
    assert not batch.cursors.valid[0]
    assert not batch.eligible.any()
    assert batch.ineligible_intervals == ()
    assert batch.cursors.origin_bytes[0] == 0


def test_empty_bounded_batch():
    result = compile_(state(), [], [], np.empty((0, 16), dtype=np.int64))
    assert result.eligible.shape == (0, 16)
    assert result.ineligible_intervals == ()


def test_signed_timestamp_origin_and_readonly_cached_arrays():
    model = state(origin=-7)
    assert model.anchor_samples[0] == 28
    assert not model.anchor_samples.flags.writeable
    assert not model.event_flags.flags.writeable
    batch = compile_(model, [100], [48 + 28], [48 + 28 + np.arange(16) * 8])
    assert batch.ineligible_intervals == ((109, 114),)


def test_large_sample_origins_keep_exact_low_bits():
    model = state(origin=-(2**58))
    desired = 2**60 + 48
    batch = compile_(model, [2**60], [desired], [desired + np.arange(16) * 8])
    assert batch.cursors.origin_bytes.tolist() == [2]
    assert batch.ineligible_intervals == ((2**60 + 9, 2**60 + 14),)


def test_origin_outside_int64_can_have_representable_relative_samples():
    model = RawBufferedRequestState.from_timeline(
        timeline([(0, 80, 0)]),
        samples_per_frame=1,
        timestamp_origin_frame=2**63,
    )
    assert model.anchor_samples[0] == -(2**63)
    assert model.anchor_samples[-1] == -(2**63) + 70


@pytest.mark.parametrize("ratio", [Fraction(1, 8), Fraction(3, 2)])
def test_rational_payload_bytes_match_scalar_framed_masks(ratio):
    geometry = BufferedRequestGeometry(ratio, 12, 16, 8, 2)
    model = state()
    desired = np.asarray([48], dtype=np.int64)
    ordinary = desired[:, None] + np.arange(16) * 12
    batch = compile_(model, [100], desired, ordinary, geometry=geometry)
    cursor = buffered_request_cursor(8, 0.0, geometry, sample_rate=128.0)
    expected = request_control_mask(
        origin=cursor.origin_bytes,
        skipped=cursor.skipped_ffts,
        bit_count=16,
        extent_frames=10,
        frame_bytes=2,
        payload_bytes_per_fft=geometry.payload_bytes_per_fft,
        control_word_bits=3,
    )
    expected[ordinary[0] >= model.anchor_samples[2]] = True
    expected[: cursor.skipped_ffts] = False
    assert np.array_equal(batch.eligible[0], expected)


@pytest.mark.parametrize(
    "count,spacing,residual",
    [(4, 2, 1215), (12, 2, 1215), (6, 3, 1215), (8, 2, 997)],
)
def test_generated_repeated_histories_match_independent_request_arithmetic(
    count, spacing, residual
):
    capacity, frame_count = 1206, 120000
    boundaries = [
        8 * capacity + 53 + i * spacing * capacity for i in range(count)
    ]
    points = [0, *boundaries, frame_count]
    runs = [
        (points[i], points[i + 1], points[i] + i * residual)
        for i in range(len(points) - 1)
    ]
    model = RawBufferedRequestState.from_timeline(
        timeline(runs, capacity), samples_per_frame=32768
    )
    geometry = BufferedRequestGeometry(4, 262144, 50, 8192, 4)
    starts = np.arange(0, 12500, 50, dtype=np.int64)
    desired = starts * 262144
    ordinary = desired[:, None] + np.arange(50, dtype=np.int64) * 262144
    result = compile_buffered_request_discards(
        model,
        starts,
        desired,
        np.zeros(starts.shape),
        ordinary,
        geometry,
        sample_rate=4096000000.0,
        frame_bytes=8224,
        control_word_bits=30,
    )
    # Independently evaluate the generated read grid with Python integer
    # arithmetic and scalar cursor/control helpers, not observed loss tables.
    anchors = [
        slot * capacity
        + sum(
            residual for boundary in boundaries if boundary <= slot * capacity
        )
        for slot in range((frame_count + capacity - 1) // capacity)
    ]
    events = {boundary // capacity for boundary in boundaries}
    for row, first in enumerate(starts):
        owner = bisect_right(anchors, int(first) * 8) - 1
        cursor = buffered_request_cursor(
            (int(first) * 8 - anchors[owner]) * 32768,
            0.0,
            geometry,
            sample_rate=4096000000.0,
        )
        mask = np.ones(50, dtype=np.bool_)
        if owner in events:
            mask = request_control_mask(
                origin=cursor.origin_bytes,
                skipped=cursor.skipped_ffts,
                bit_count=50,
                extent_frames=capacity,
            )
            mask[
                np.arange(int(first), int(first) + 50) * 8
                >= anchors[owner + 1]
            ] = True
        mask[: cursor.skipped_ffts] = False
        assert np.array_equal(result.eligible[row], mask)


@pytest.mark.parametrize("field", ["frame_bytes", "control_word_bits"])
@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_control_geometry_must_be_positive_integers(field, value):
    kwargs = dict(sample_rate=128, frame_bytes=2, control_word_bits=3)
    kwargs[field] = value
    with pytest.raises(
        (ValueError, TypeError), match=field.replace("_", " ")
    ):
        compile_buffered_request_discards(
            state(),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.zeros(1),
            np.zeros((1, 16), dtype=np.int64),
            SMALL,
            **kwargs,
        )


def test_request_and_framed_byte_overflow_fail():
    with pytest.raises(OverflowError, match="global FFT"):
        compile_(state(), [2**63 - 1], [48], [48 + np.arange(16) * 8])
    with pytest.raises(OverflowError, match="framed control"):
        compile_buffered_request_discards(
            state(),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.zeros(1),
            np.zeros((1, 16), dtype=np.int64),
            SMALL,
            sample_rate=128,
            frame_bytes=2**63 - 1,
            control_word_bits=3,
        )


def test_state_anchor_and_request_subtraction_overflow_fail():
    with pytest.raises(OverflowError):
        RawBufferedRequestState.from_timeline(
            timeline([(0, 80, 0)]), samples_per_frame=2**62
        )
    model = RawBufferedRequestState.from_timeline(
        timeline([(0, 80, 0)]),
        samples_per_frame=4,
        timestamp_origin_frame=100,
    )
    with pytest.raises(OverflowError, match="local origin"):
        compile_(model, [100], [2**63 - 1], [np.zeros(16, dtype=np.int64)])


def test_latched_head_keeps_allocation_clock_after_intraread_jump():
    from cuphoton.xbeam.buffered_reader.geometry import (
        compile_latched_raw_window_heads,
    )

    # Independent coordinates: physical allocation 45 retains its old
    # clock although its aligned cursor is beyond the forward jump.
    capacity, padding, boundary = 1206, 2000, 45 * 1206 + 1206 // 3
    cache = RawBufferedRequestState.from_timeline(
        timeline(
            [(0, boundary, 0), (boundary, 129000, boundary + 2)],
            capacity,
        ),
        samples_per_frame=32768,
        timestamp_origin_frame=padding,
    )
    geometry = BufferedRequestGeometry(4, 262144, 50, 8192, 4)
    starts = np.array([6600], dtype=np.int64)
    ordinary = (starts[:, None] + np.arange(50)) * 262144 - 16384
    batch = compile_buffered_request_discards(
        cache,
        starts,
        ordinary[:, 0],
        np.zeros(1),
        ordinary,
        geometry,
        sample_rate=4096000000,
        frame_bytes=8224,
        control_word_bits=30,
    )
    heads = compile_latched_raw_window_heads(cache, batch, geometry)
    physical = 45 * capacity * 32768 + int(batch.cursors.origin_bytes[0]) * 4
    assert physical > boundary * 32768
    assert heads.tolist() == [physical - padding * 32768]
    assert heads[0] != physical - padding * 32768 + 2 * 32768
    with pytest.raises(ValueError):
        heads.setflags(write=True)


def test_latched_head_excludes_decoder_offset_and_skipped_ffts():
    from dataclasses import replace

    from cuphoton.xbeam.buffered_reader.geometry import (
        compile_latched_raw_window_heads,
    )

    cache = state()
    batch = compile_(cache, [100], [48], [48 + np.arange(16) * 8])
    changed = replace(
        batch,
        cursors=replace(
            batch.cursors,
            decoder_delta_samples=np.array([7], dtype=np.int64),
            skipped_ffts=np.array([5], dtype=np.int64),
        ),
    )
    assert compile_latched_raw_window_heads(cache, batch, SMALL).tolist() == [
        48
    ]
    np.testing.assert_array_equal(
        compile_latched_raw_window_heads(cache, changed, SMALL),
        compile_latched_raw_window_heads(cache, batch, SMALL),
    )


def test_latched_head_exact_packing_and_signed_large_origins():
    from dataclasses import replace

    from cuphoton.xbeam.buffered_reader.geometry import (
        compile_latched_raw_window_heads,
    )

    cache = state()
    batch = compile_(cache, [100], [48], [48 + np.arange(16) * 8])
    batch = replace(
        batch,
        cursors=replace(
            batch.cursors, origin_bytes=np.array([6], dtype=np.int64)
        ),
    )
    geometry = BufferedRequestGeometry(Fraction(2, 3), 8, 16, 6, 3)
    for anchor in (2**53 + 1, -(2**53 + 1)):
        cache = replace(
            cache,
            anchor_samples=np.full(
                cache.anchor_samples.shape, anchor, dtype=np.int64
            ),
        )
        assert compile_latched_raw_window_heads(
            cache, batch, geometry
        ).tolist() == [anchor + 4]
    bad = replace(
        batch,
        cursors=replace(
            batch.cursors, origin_bytes=np.array([5], dtype=np.int64)
        ),
    )
    with pytest.raises(ValueError, match="whole sample"):
        compile_latched_raw_window_heads(cache, bad, geometry)
    cache = replace(
        cache,
        anchor_samples=np.full(
            cache.anchor_samples.shape, np.iinfo(np.int64).max, dtype=np.int64
        ),
    )
    with pytest.raises(OverflowError):
        compile_latched_raw_window_heads(cache, batch, geometry)


def test_latched_head_rejects_unknown_return_instead_of_default_clock():
    from dataclasses import replace

    from cuphoton.xbeam.buffered_reader.geometry import (
        compile_latched_raw_window_heads,
    )

    cache = state()
    batch = compile_(cache, [100], [48], [48 + np.arange(16) * 8])
    for unknown in (
        replace(batch, valid_requests=np.array([False])),
        replace(
            batch, cursors=replace(batch.cursors, valid=np.array([False]))
        ),
    ):
        with pytest.raises(ValueError, match="unresolved"):
            compile_latched_raw_window_heads(cache, unknown, SMALL)
    with pytest.raises(ValueError, match="outside raw history"):
        compile_latched_raw_window_heads(
            cache,
            replace(
                batch, selected_allocations=np.array([-1], dtype=np.int64)
            ),
            SMALL,
        )
