# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedNormalFullExtent,
    RawBufferedRequestState,
    RawResidualOwner,
)
from cuphoton.xbeam.buffered_reader.replay import (
    RawWindowClockContract,
    RetainedRawWindowContract,
    apply_uncovered_head_publication,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import RawValidityState


def state(starts=(0, 13), stops=(9, 60), timestamps=None, capacity=10, spf=4):
    timeline = PhysicalAllocationTimeline.from_runs(
        np.asarray(starts),
        np.asarray(stops),
        np.asarray(starts if timestamps is None else timestamps),
        allocation_frames=capacity,
    )
    return RawValidityState.from_timeline(
        timeline,
        RawBufferedRequestState.from_timeline(
            timeline, samples_per_frame=spf
        ),
    )


def normal_extent(cache):
    reader = cache.source_state.request_state
    return BufferedNormalFullExtent(
        reader.samples_per_frame + 1,
        reader.samples_per_frame,
        reader.allocation_frames,
        1,
        1,
    )


@pytest.mark.parametrize(
    "output,delay,decoded_published,quantized_published",
    [(88, 2.4e-9, False, True), (70, -2.4e-9, True, False)],
)
def test_publication_clock_quantum_does_not_move_decoder(
    output, delay, decoded_published, quantized_published
):
    # Allocation [40,80) has a missing head [40,44), but its tail and the
    # successor are recorded. The decoder keeps fractional-ns precision;
    # ownership uses the explicitly declared integer-ns control clock.
    cache = state(starts=(0, 11), stops=(10, 100), capacity=10, spf=4)
    supplied = RetainedRawWindowContract(
        0,
        np.array([output], dtype=np.int64),
        np.array([64], dtype=np.int64),
        np.array([delay], dtype=np.float64),
        np.array([32], dtype=np.int64),
        np.array([32], dtype=np.int64),
        np.array([64], dtype=np.int64),
        np.array([True]),
    )
    kwargs = dict(
        extent=normal_extent(cache), sample_rate=4_000_000_000, fft_samples=8
    )
    decoded = apply_uncovered_head_publication(supplied, cache, **kwargs)
    quantized = apply_uncovered_head_publication(
        supplied,
        cache,
        publication_timestamp_quantum_nanoseconds=1,
        **kwargs,
    )
    assert decoded.publication_eligible.tolist() == [decoded_published]
    assert quantized.publication_eligible.tolist() == [quantized_published]
    assert np.array_equal(
        quantized.physical_head_samples, supplied.physical_head_samples
    )
    assert np.array_equal(
        quantized.aligned_head_samples, supplied.aligned_head_samples
    )
    assert np.array_equal(
        quantized.centre_delays_seconds, supplied.centre_delays_seconds
    )


def contract(*, size=8, first_sample=0, count=25):
    return RetainedRawWindowContract(
        0,
        first_sample + np.arange(count, dtype=np.int64) * size,
        np.zeros(count, dtype=np.int64),
        np.zeros(count, dtype=np.float64),
        np.full(count, 240, dtype=np.int64),
        np.full(count, 240, dtype=np.int64),
        np.zeros(count, dtype=np.int64),
        np.ones(count, dtype=np.bool_),
    )


@pytest.mark.parametrize(
    "hole,expected",
    [
        ((8, 10), ()),
        ((10, 13), ((1, 2),)),
        ((11, 13), ()),
        ((9, 32), ((1, 4),)),
        ((9, 10), ()),
    ],
)
def test_half_open_physical_holes_classify_heads_only(hole, expected):
    cache = state(starts=(0, hole[1]), stops=(hole[0], 60))
    assert (
        cache.source_state.timeline.uncovered_head_allocations() == expected
    )


def test_multiple_uncovered_heads_coalesce_without_expanding_frames():
    cache = state(starts=(0, 11, 21), stops=(10, 20, 60))
    assert cache.source_state.timeline.uncovered_head_allocations() == (
        (1, 3),
    )


def test_start_owned_gate_preserves_crossing_fft_and_other_contract_fields():
    cache = state()
    original = contract()
    gated = apply_uncovered_head_publication(
        original,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    )
    # Allocation 1 covers samples 40..80. FFT4 starts at32 and crosses the
    # hole at36: retain it; FFT6 starts at48 inside that hole and reaches a
    # valid suffix at52 in the preceding return. FFT10 starts at80: retain it.
    assert np.flatnonzero(~gated.publication_eligible).tolist() == [
        5,
        7,
        8,
        9,
    ]
    for field in (
        "output_samples",
        "aligned_head_samples",
        "centre_delays_seconds",
        "framed_extent_samples",
        "decoded_extent_samples",
        "physical_head_samples",
    ):
        np.testing.assert_array_equal(
            getattr(gated, field), getattr(original, field)
        )
    assert original.publication_eligible.all()
    with pytest.raises(ValueError):
        gated.publication_eligible.setflags(write=True)


def test_unaligned_geometry_keeps_exact_sample_boundaries():
    cache = state(starts=(0, 9), stops=(7, 60), capacity=7, spf=3)
    original = contract(size=5, first_sample=17, count=10)
    gated = apply_uncovered_head_publication(
        original,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=5,
    )
    # Missing head7 belongs to sample interval21..42. Starts17 and42 survive.
    assert gated.publication_eligible.tolist() == [
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize(
    "first,size,expected",
    [(35, 4, True), (36, 4, False), (36, 16, False), (36, 17, True)],
)
def test_wholly_uncovered_fft_before_allocation_head_is_not_published(
    first, size, expected
):
    cache = state()
    original = contract(size=size, first_sample=first, count=1)
    gated = apply_uncovered_head_publication(
        original,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=size,
    )
    assert gated.publication_eligible.tolist() == [expected]
    assert original.publication_eligible.tolist() == [True]


def test_empty_inventory_lookup_does_not_require_an_uncovered_head():
    cache = state(starts=(0, 13), stops=(11, 60))
    assert cache.source_state.timeline.uncovered_head_allocations() == ()
    original = contract(size=8, first_sample=44, count=1)
    gated = apply_uncovered_head_publication(
        original,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    )
    assert gated.publication_eligible.tolist() == [False]


def test_timestamp_jump_without_physical_hole_does_not_change_publication():
    cache = state(starts=(0, 13), stops=(13, 60), timestamps=(1000, 1100))
    original = contract()
    assert cache.source_state.timeline.uncovered_head_allocations() == ()
    assert (
        apply_uncovered_head_publication(
            original,
            cache,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        )
        is original
    )


def test_preexisting_publication_and_scope_clipping_are_preserved():
    cache = state()
    original = contract()
    publication = original.publication_eligible.copy()
    publication[2] = False
    original = replace(original, publication_eligible=publication)
    whole = apply_uncovered_head_publication(
        original,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    )
    assert not whole.publication_eligible[2]
    for first, stop in ((0, 7), (7, 11), (11, 25)):
        clipped = original.for_scope(first, stop)
        actual = apply_uncovered_head_publication(
            clipped,
            cache,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        )
        np.testing.assert_array_equal(
            actual.publication_eligible,
            whole.for_scope(first, stop).publication_eligible,
        )


def test_unknown_physical_extent_and_missing_retained_contract_fail():
    cache = state()
    original = contract(first_sample=238, count=1)
    original = replace(
        original, framed_extent_samples=np.array([300], dtype=np.int64)
    )
    with pytest.raises(ValueError, match="outside cached raw history"):
        apply_uncovered_head_publication(
            original,
            cache,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        )
    original = contract()
    plain = RawWindowClockContract(
        original.first_fft,
        original.output_samples,
        original.aligned_head_samples,
        original.centre_delays_seconds,
        original.framed_extent_samples,
        original.decoded_extent_samples,
    )
    with pytest.raises(TypeError, match="retained"):
        apply_uncovered_head_publication(
            plain,
            cache,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        )
    with pytest.raises(TypeError, match="cached raw validity"):
        apply_uncovered_head_publication(
            original,
            None,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        )


def test_clock_frame_owner_skips_filler_before_timestamp_transition():
    timeline = state(timestamps=(0, 30)).source_state.timeline
    default = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=4
    )
    supplied = RawBufferedRequestState.from_timeline(
        timeline,
        samples_per_frame=4,
        residual_owner=RawResidualOwner.CLOCK_FRAME,
    )
    # The new trustworthy run starts at physical13, but frames9..12 are
    # filler. Slot12 belongs to allocation1; trustworthy slot8 belongs to0.
    assert default.event_allocations == frozenset({1})
    assert supplied.event_allocations == frozenset({0})
    np.testing.assert_array_equal(
        default.anchor_samples, supplied.anchor_samples
    )
    assert np.flatnonzero(supplied.event_flags).tolist() == [0]


@pytest.mark.parametrize("boundary", [13, 20])
def test_both_owner_contracts_agree_without_a_physical_hole(boundary):
    timeline = state(
        starts=(0, boundary),
        stops=(boundary, 60),
        timestamps=(0, boundary + 17),
    ).source_state.timeline
    default = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=4
    )
    supplied = RawBufferedRequestState.from_timeline(
        timeline,
        samples_per_frame=4,
        residual_owner=RawResidualOwner.CLOCK_FRAME,
    )
    assert (
        supplied.event_allocations
        == default.event_allocations
        == frozenset({(boundary - 1) // 10})
    )
    np.testing.assert_array_equal(default.event_flags, supplied.event_flags)


def test_zero_residual_and_unknown_owner_are_separate():
    timeline = state().source_state.timeline
    supplied = RawBufferedRequestState.from_timeline(
        timeline,
        samples_per_frame=4,
        residual_owner=RawResidualOwner.CLOCK_FRAME,
    )
    assert not supplied.event_allocations
    with pytest.raises(TypeError, match="residual owner"):
        RawBufferedRequestState.from_timeline(
            timeline,
            samples_per_frame=4,
            residual_owner="preceding-clock-frame",
        )


def test_retained_suffix_requires_a_preceding_return():
    cache = state()
    preceding = contract(first_sample=48, count=1)
    assert apply_uncovered_head_publication(
        preceding,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    ).publication_eligible.tolist() == [True]
    # Keep the same physical FFT and clocks, but start its return at the
    # unavailable allocation head. That view cannot retain an earlier suffix.
    fresh = replace(
        preceding,
        aligned_head_samples=np.array([40], dtype=np.int64),
        physical_head_samples=np.array([40], dtype=np.int64),
    )
    assert apply_uncovered_head_publication(
        fresh,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    ).publication_eligible.tolist() == [False]


def test_retained_suffix_must_fit_decoded_extent_and_existing_publication():
    cache = state()
    preceding = contract(first_sample=48, count=1)
    for extent, published, expected in (
        (52, True, False),
        (53, True, True),
        (53, False, False),
    ):
        clipped = replace(
            preceding,
            decoded_extent_samples=np.array([extent], dtype=np.int64),
            publication_eligible=np.array([published], dtype=np.bool_),
        )
        assert apply_uncovered_head_publication(
            clipped,
            cache,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        ).publication_eligible.tolist() == [expected]


def test_empty_and_post_resume_ffts_gain_no_suffix_publication():
    cache = state()
    for sample in (40, 52, 56):
        assert apply_uncovered_head_publication(
            contract(first_sample=sample, count=1),
            cache,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        ).publication_eligible.tolist() == [False]


def test_previous_framed_prefix_can_retain_suffix_when_hole_begins_at_head():
    cache = state(starts=(0, 13), stops=(10, 60))
    assert apply_uncovered_head_publication(
        contract(first_sample=48, count=1),
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    ).publication_eligible.tolist() == [True]


def test_adjacent_timestamp_jump_does_not_assert_retained_suffix():
    cache = state(timestamps=(0, 100))
    assert apply_uncovered_head_publication(
        contract(first_sample=48, count=1),
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    ).publication_eligible.tolist() == [False]


def test_retained_suffix_requires_a_publishable_predecessor_allocation():
    cache = state(starts=(0, 13, 23), stops=(9, 19, 60))
    ordinary = contract(first_sample=88, count=1)
    # The same FFT crosses the second hole and has a valid suffix at 92.
    # A return from covered allocation zero may retain it.
    assert apply_uncovered_head_publication(
        ordinary,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    ).publication_eligible.tolist() == [True]
    # The preceding request starts in valid bytes at 70, but its allocation
    # head at 40 was uncovered. Those bytes do not prove a published return.
    unavailable = replace(
        ordinary,
        aligned_head_samples=np.array([70], dtype=np.int64),
        physical_head_samples=np.array([70], dtype=np.int64),
    )
    assert apply_uncovered_head_publication(
        unavailable,
        cache,
        extent=normal_extent(cache),
        sample_rate=100,
        fft_samples=8,
    ).publication_eligible.tolist() == [False]


def test_framed_header_prefix_is_strict_and_uses_actual_packing():
    cache = state(starts=(0, 11), stops=(9, 60))
    for source, expected in ((49, True), (50, False)):
        gated = apply_uncovered_head_publication(
            contract(first_sample=source, count=1),
            cache,
            extent=normal_extent(cache),
            sample_rate=100,
            fft_samples=8,
        )
        assert gated.publication_eligible.tolist() == [expected]
    # Seven half-byte headers span 3.5 samples, rather than the integer
    # ten-sample margin of the preceding geometry. Compare without floats.
    cache = state(starts=(0, 8), stops=(6, 60), capacity=7, spf=3)
    packed = BufferedNormalFullExtent(7, 6, 7, 1, Fraction(1, 2))
    for source, expected in ((24, True), (25, False)):
        gated = apply_uncovered_head_publication(
            contract(first_sample=source, count=1),
            cache,
            extent=packed,
            sample_rate=100,
            fft_samples=8,
        )
        assert gated.publication_eligible.tolist() == [expected]


def test_publication_extent_must_match_cached_payload_and_allocation():
    cache = state()
    original = contract()
    for bad in (
        BufferedNormalFullExtent(5, 4, 9, 1, 1),
        BufferedNormalFullExtent(6, 5, 10, 1, 1),
    ):
        with pytest.raises(ValueError, match="cached geometry"):
            apply_uncovered_head_publication(
                original,
                cache,
                extent=bad,
                sample_rate=100,
                fft_samples=8,
            )
    with pytest.raises(TypeError, match="normal-full extent"):
        apply_uncovered_head_publication(
            original,
            cache,
            extent=None,
            sample_rate=100,
            fft_samples=8,
        )
