# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.timeline import (
    BufferedAnchorGap,
    PhysicalAllocationTimeline,
    PhysicalSampleSpan,
)


def timeline(runs, allocation):
    values = np.asarray(runs, dtype=np.int64)
    return PhysicalAllocationTimeline.from_runs(
        *values.T, allocation_frames=allocation
    )


def test_timestamp_change_waits_for_successor_refill():
    # Raw jump at physical slot 13 is inside allocation [10, 20).
    model = timeline(((0, 13, 100), (13, 40, 120)), 10)
    assert model.deferred_anchor_gaps() == (
        BufferedAnchorGap(2, 20, 120, 127),
    )
    # The old grid's FFT starting at sample 118 survives its crossing tail.
    assert model.deferred_anchor_gaps()[0].fft_window(
        timestamp_origin=100, samples_per_frame=3, fft_samples=8
    ) == (8, 11)


def test_same_read_changes_collapse_to_net_anchor_jump():
    model = timeline(((0, 12, 100), (12, 18, 115), (18, 40, 125)), 10)
    assert model.deferred_anchor_gaps() == (
        BufferedAnchorGap(2, 20, 120, 127),
    )


def test_filler_alone_is_not_timestamp_anchor_loss():
    model = timeline(((0, 8, 100), (12, 40, 112)), 10)
    assert model.deferred_anchor_gaps() == ()


@pytest.mark.parametrize("shift", [0, 2**55])
def test_frame_and_sample_projection_keeps_large_integer_origins(shift):
    model = timeline(((0, 13, shift + 100), (13, 40, shift + 120)), 10)
    gap = model.deferred_anchor_gaps()[0]
    assert gap.fft_window(
        timestamp_origin=shift + 100,
        samples_per_frame=32768,
        fft_samples=262144,
        source_sample_offset=32767,
    ) == (3, 4)


@pytest.mark.parametrize("offset", [-9, -1, 0, 1, 9])
def test_start_owned_projection_matches_direct_grid_membership(offset):
    gap = BufferedAnchorGap(1, 10, 18, 25)
    start, stop = gap.fft_window(
        timestamp_origin=20,
        samples_per_frame=3,
        fft_samples=8,
        source_sample_offset=offset,
    )
    observed = tuple(
        index
        for index in range(-20, 21)
        if (18 - 20) * 3 <= index * 8 + offset < (25 - 20) * 3
    )
    assert tuple(range(start, stop)) == observed


def test_empty_timeline_anchor_change_is_not_invented():
    assert timeline(((0, 9, 100),), 10).deferred_anchor_gaps() == ()


def test_timestamp_jump_on_refill_boundary_is_not_deferred_again():
    model = timeline(((0, 10, 100), (10, 40, 117)), 10)
    assert model.deferred_anchor_gaps() == (
        BufferedAnchorGap(1, 10, 110, 117),
    )


def test_latched_payload_crosses_raw_timestamp_jump_before_anchor_changes():
    model = timeline(((0, 13, 100), (13, 40, 120)), 10)
    # Allocation 1 is latched at timestamp 110, even though its frame 13
    # introduces a jump. Raw payload stays contiguous through that jump.
    assert model.latched_source_spans(
        1, 111 * 3 + 1, 24, samples_per_frame=3
    ) == (PhysicalSampleSpan(0, 24, 11 * 3 + 1),)
    assert model.sample_source_spans(
        111 * 3 + 1, 24, samples_per_frame=3
    ) != model.latched_source_spans(1, 111 * 3 + 1, 24, samples_per_frame=3)
    # The successor anchor is 127, hence timestamp 128 reads physical 21.
    assert model.latched_source_spans(
        2, 128 * 3 + 1, 8, samples_per_frame=3
    ) == (PhysicalSampleSpan(0, 8, 21 * 3 + 1),)


def test_latched_filler_holes_keep_samples_on_both_sides():
    model = timeline(((0, 12, 100), (15, 40, 115)), 10)
    assert model.latched_source_spans(
        1, 111 * 3 + 1, 20, samples_per_frame=3
    ) == (
        PhysicalSampleSpan(0, 2, 34),
        PhysicalSampleSpan(11, 20, 45),
    )


def test_latched_payload_preserves_large_sample_origins_and_offsets():
    origin = 2**55
    model = timeline(((0, 13, origin), (13, 40, origin + 20)), 10)
    assert model.latched_source_spans(
        np.uint64(2),
        np.uint64((origin + 28) * 3 + 1),
        8,
        samples_per_frame=np.uint64(3),
    ) == (PhysicalSampleSpan(0, 8, 64),)


def test_latched_payload_does_not_invent_source_before_or_after_file():
    model = timeline(((0, 20, 100),), 10)
    assert model.latched_source_spans(
        0, 98 * 3 + 1, 8, samples_per_frame=3
    ) == (PhysicalSampleSpan(5, 8, 0),)
    assert model.latched_source_spans(
        1, 119 * 3 + 1, 8, samples_per_frame=3
    ) == (PhysicalSampleSpan(0, 2, 58),)
    with pytest.raises(ValueError, match="allocation index"):
        model.latched_source_spans(2, 120 * 3, 1, samples_per_frame=3)


@pytest.mark.parametrize("offset", [0.5, True, np.bool_(True)])
def test_fractional_or_boolean_offsets_fail_instead_of_rounding(offset):
    with pytest.raises(TypeError, match="offset"):
        BufferedAnchorGap(1, 10, 18, 25).fft_window(
            timestamp_origin=0,
            samples_per_frame=3,
            fft_samples=8,
            source_sample_offset=offset,
        )
