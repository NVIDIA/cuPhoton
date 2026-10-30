# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.replay import (
    RetainedRawWindowContract,
    bind_known_source_end,
    compile_raw_replay_window_facts,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import RawValidityState


def state(runs=((0, 100, 0),)):
    runs = np.asarray(runs, dtype=np.int64)
    timeline = PhysicalAllocationTimeline.from_runs(
        runs[:, 0], runs[:, 1], runs[:, 2], allocation_frames=10
    )
    request = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=4
    )
    return RawValidityState.from_timeline(timeline, request)


def contract(*, physical=120, publication=(True, True, True, True)):
    # An independently supplied four-FFT window. Framed admission permits
    # all four; its decoder contains only two full FFTs plus four samples.
    return RetainedRawWindowContract(
        10,
        np.arange(80, 112, 8, dtype=np.int64),
        np.full(4, 80, dtype=np.int64),
        np.zeros(4),
        np.full(4, 32, dtype=np.int64),
        np.full(4, 20, dtype=np.int64),
        np.full(4, physical, dtype=np.int64),
        np.asarray(publication, dtype=np.bool_),
    )


def facts(c, *, cache=None, first=10, stop=14, **kwargs):
    return compile_raw_replay_window_facts(
        state() if cache is None else cache,
        np.asarray([10], dtype=np.int64),
        np.asarray([80], dtype=np.int64),
        np.zeros(1),
        np.arange(80, 112, 8, dtype=np.int64)[None, :],
        BufferedRequestGeometry(1, 8, 4, 2, 1),
        first_fft=first,
        stop_fft=stop,
        sample_rate=128.0,
        frame_bytes=3,
        control_word_bits=2,
        window_clock_contract=c,
        **kwargs,
    )


def addresses(result):
    return {
        i: s.physical_sample_start + (i - s.first_fft) * 8
        for s in result.addresses.spans
        for i in range(s.first_fft, s.stop_fft)
    }


def test_source_publication_and_decoder_extent_are_independent():
    result = facts(contract(publication=(True, False, True, True)))
    assert result.complete
    assert addresses(result) == {10: 120, 12: 136, 13: 144}
    assert result.ineligible_intervals == ((11, 12),)
    assert [
        (s.first_fft, s.stop_fft) for s in result.validity.full_spans
    ] == [(10, 11)]
    partial = result.validity.partial_events
    assert len(partial) == 1 and partial[0].fft_index == 12
    assert [
        (s.request_start, s.request_stop, s.physical_start)
        for s in partial[0].spans
    ] == [(0, 4, 136)]
    assert result.validity.empty_intervals == ((13, 14),)


def test_retained_physical_cursor_does_not_relatch_at_raw_timestamp_jump():
    cache = state(((0, 33, 0), (33, 100, 38)))
    result = facts(contract(), cache=cache)
    assert result.complete
    assert addresses(result) == {10: 120, 11: 128, 12: 136, 13: 144}


def test_physical_hole_intersects_decoder_coverage_without_exposure_credit():
    cache = state(((0, 34, 0), (35, 100, 35)))
    result = facts(contract(), cache=cache)
    assert result.complete
    assert addresses(result)[12] == 136
    assert result.validity.empty_intervals == ((12, 14),)
    assert not result.validity.partial_events


def test_scope_retains_source_contract_and_complete_enclosing_request():
    whole = contract()
    clipped = whole.for_scope(12, 14)
    assert isinstance(clipped, RetainedRawWindowContract)
    assert clipped.physical_head_samples.tolist() == [120, 120]
    assert clipped.aligned_head_samples.tolist() == [80, 80]
    assert addresses(facts(clipped, first=12)) == {12: 136, 13: 144}
    assert clipped.for_scope(12, 14) is clipped
    with pytest.raises(ValueError, match="outside"):
        clipped.for_scope(11, 14)


@pytest.mark.parametrize(
    "field", ["physical_head_samples", "publication_eligible"]
)
def test_retained_vectors_are_immutable_and_shape_checked(field):
    c = contract()
    with pytest.raises(ValueError):
        getattr(c, field).setflags(write=True)
    with pytest.raises(TypeError, match="matching vector"):
        replace(c, **{field: getattr(c, field)[:1]})


@pytest.mark.parametrize("publication", [True, False])
@pytest.mark.parametrize("earlier_hole", [True, False])
def test_unknown_eof_is_never_silently_discarded(publication, earlier_hole):
    cache = state(((0, 10, 0), (11, 100, 11))) if earlier_hole else state()
    result = facts(
        contract(physical=390, publication=(publication,) * 4), cache=cache
    )
    assert not result.complete
    assert not result.ineligible_intervals
    assert not result.addresses.spans
    assert [
        (s.first_fft, s.stop_fft) for s in result.addresses.unresolved
    ] == [(10, 14)]
    assert all(
        span.reason == "retained window exceeds recorded payload"
        for span in result.addresses.unresolved
    )


def test_retained_source_arithmetic_does_not_wrap():
    with pytest.raises(OverflowError, match="retained physical"):
        facts(contract(physical=2**63 - 10))


def test_known_file_end_preserves_full_and_partial_payload():
    # Physical input has exactly 400 samples. Its retained control window
    # permits four FFTs at 390, 398, 406, 414. Only the first FFT and two
    # samples of the second are backed by the finite file.
    bounded = bind_known_source_end(contract(physical=390), 400)
    result = facts(bounded)
    assert result.complete
    assert addresses(result) == {10: 390, 11: 398}
    assert result.physical_sample_extent == (390, 400)
    assert result.ineligible_intervals == ((12, 14),)
    assert [
        (s.first_fft, s.stop_fft) for s in result.validity.full_spans
    ] == [(10, 11)]
    assert len(result.validity.partial_events) == 1
    partial = result.validity.partial_events[0]
    assert partial.fft_index == 11
    assert [
        (s.request_start, s.request_stop, s.physical_start)
        for s in partial.spans
    ] == [(0, 2, 398)]
    assert bounded.for_scope(11, 14).known_source_stop == 400


def test_known_file_end_cannot_hide_existing_source_history():
    with pytest.raises(
        ValueError, match="excludes supplied physical history"
    ):
        facts(bind_known_source_end(contract(), 399))


def test_known_file_end_binds_an_empty_return_after_eof():
    bounded = bind_known_source_end(contract(physical=450), 400)
    result = facts(bounded)
    assert result.complete
    assert not result.addresses.spans
    assert not result.validity.full_spans
    assert not result.validity.partial_events
    assert result.ineligible_intervals == ((10, 14),)


def test_source_end_binding_cannot_guess_negative_prefix():
    with pytest.raises(ValueError, match="negative history"):
        bind_known_source_end(contract(physical=-1), 400)


def test_nonintegral_clock_rate_is_rejected():
    # The public job path also checks this, but explicit supplied clocks
    # must not silently cast a fractional sampling rate to an integer.
    from cuphoton.xbeam.buffered_reader.replay import (
        _compile_retained_window_facts,
    )

    with pytest.raises(ValueError, match="integral sample rate"):
        _compile_retained_window_facts(
            facts(contract()), state(), contract(), sample_rate=128.5
        )
