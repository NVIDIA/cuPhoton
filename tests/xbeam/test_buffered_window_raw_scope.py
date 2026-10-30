# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from cuphoton.xbeam.buffered_reader.adapter import (
    RawJobReplayScope,
    compile_raw_job_replay_scopes,
)
from cuphoton.xbeam.buffered_reader.clocks import LeadingRawHeadContract
from cuphoton.xbeam.buffered_reader.geometry import BufferedNormalFullExtent
from cuphoton.xbeam.buffered_reader.requests import ReadClockConvention
from tests.xbeam.test_buffered_window_job_requests import model_job
from tests.xbeam.test_buffered_window_raw_replay import state

RATE, FFT, SPF = 4096000000, 262144, 32768


def inputs(delay=0):
    job, ds, band, freq, scan, geometry = model_job(
        (delay / RATE,), rate=RATE, fft=FFT
    )
    return (
        job,
        ds,
        RawJobReplayScope(
            "scope", band, freq, scan, 1193, 1571, 2000, geometry
        ),
    )


def compile_scopes(cache, job, ds, scopes, **kwargs):
    def returned(scope, inputs, state):
        extent = BufferedNormalFullExtent(8224, 8192, 402, 3, 4)
        return inputs.published_normal_full_window_contract(
            state,
            extent,
            control_word_bits=30,
            ordinary_head_contract=LeadingRawHeadContract(
                int(inputs.sample_rate), 1
            ),
        )

    return compile_raw_job_replay_scopes(
        cache,
        job,
        ds,
        scopes,
        job_start_slot=cache.source_state.request_state.timestamp_origin_frame,
        source_index=0,
        control_word_bits=30,
        window_clock_contract_factory=returned,
        read_clock_convention=ReadClockConvention.ENDPOINT_MEAN,
        **kwargs,
    )


def test_tiled_and_whole_scopes_share_exact_sparse_facts_and_common_storage():
    job, ds, scope = inputs()
    scopes = (
        scope,
        replace(scope, key="discarded", first_fft=1401, stop_fft=1451),
    )
    cache = state()
    whole = compile_scopes(cache, job, ds, scopes)
    tiled = compile_scopes(
        cache, job, ds, scopes, maximum_requests_per_tile=1
    )
    assert whole.storage == tiled.storage
    assert whole.facts == tiled.facts
    for key in whole.windows:
        expected = whole.windows[key].plan_for_scope(
            whole.facts[key].first_fft, whole.facts[key].stop_fft
        )
        actual = tiled.windows[key].plan_for_scope(
            tiled.facts[key].first_fft, tiled.facts[key].stop_fft
        )
        assert expected == actual
        assert (
            actual.source_physical_frame_intervals
            == whole.storage.physical_presentation
        )
    assert whole.facts["discarded"].physical_sample_extent is None
    with pytest.raises(TypeError):
        whole.windows["new"] = whole.windows["scope"]
    with pytest.raises(ValueError):
        whole.windows["scope"].plan_for_scope(1192, 1571)


def test_all_discarded_keeps_empty_presentation_without_fabricated_reads():
    job, ds, scope = inputs()
    scope = replace(scope, first_fft=1401, stop_fft=1451)
    result = compile_scopes(
        state(), job, ds, (scope,), maximum_requests_per_tile=1
    )
    assert result.storage is None
    plan = result.windows["scope"].plan_for_scope(1401, 1451)
    assert not plan.source_physical_frame_intervals
    assert plan.ineligible_intervals == ((1401, 1451),)


def test_unknown_scope_fails_without_unsupported_fallback():
    job, ds, scope = inputs()
    with pytest.raises(ValueError, match="outside cached raw history"):
        compile_scopes(state(eof=11000), job, ds, (scope,))
    with pytest.raises(ValueError, match="cannot exceed 4096"):
        compile_scopes(
            state(), job, ds, (scope,), maximum_requests_per_tile=4097
        )
    with pytest.raises(ValueError, match="unique"):
        compile_scopes(state(), job, ds, (scope, scope))


def test_differing_sample_bases_cannot_share_storage():
    job, ds, scope = inputs()
    ds = replace(ds, sample_rate_hz=None)
    second = replace(
        scope,
        key="other",
        frequency=replace(scope.frequency, bandwidth_hz=RATE),
    )
    with pytest.raises(ValueError, match="one raw sample rate"):
        compile_scopes(state(), job, ds, (scope, second))
