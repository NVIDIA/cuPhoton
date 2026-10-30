# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent logical-clock and Fourier tests; no comparator expectations."""

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio import gpu_execution
from cuphoton.xbeam.buffered_reader.replay import (
    RetainedRawWindowContract,
)
from cuphoton.xbeam.missing_timestamp_recovery import (
    GlobalFFTEligibilityPlan,
    select_fft_source_replay,
    select_fft_source_timing,
)
from tests.xbeam.test_buffered_window_raw_replay import (
    FFT,
    facts,
    state,
    storage_for,
)
from tests.xbeam.test_sample_coverage import fixture


def test_returned_logical_timing_is_independent_of_raw_storage_translation():
    output = np.arange(100, 104, dtype=np.int64) * FFT
    delay_samples = np.array([3.5, 2.25, 3.25, -2.5])
    delay = delay_samples / 4096000000
    np.testing.assert_array_equal(delay * 4096000000, delay_samples)
    # First/fourth positive decoder offsets; second accepted -1 clamped to
    # the head; third offset -2 fails admission. Heads are explicit inputs.
    heads = output - np.array([9, 0, 0, 8], dtype=np.int64)
    outputs = []
    for translation in (0, 17 * 32768):
        contract = RetainedRawWindowContract(
            100,
            output,
            heads,
            delay,
            np.full(4, 4 * FFT, dtype=np.int64),
            np.full(4, 4 * FFT, dtype=np.int64),
            heads + translation,
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
        )
        result = facts(state(), 100, 104, window=contract)
        assert result.complete
        assert result.ineligible_intervals == ((102, 103),)
        assert result.source_timing_events == ((100, 1), (101, 2), (103, 1))
        plan = result.compile(storage_for(result)).plan_for_scope(100, 104)
        assert plan.source_timing_events == result.source_timing_events
        outputs.append(result.addresses.spans[0].physical_sample_start)
    assert outputs[1] - outputs[0] == 17 * 32768


def timing_plan(events):
    return GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        None,
        timeline_fft_count=20,
        source_replay_events=((3, 123456, 0),),
        source_timing_events=events,
    )


def test_timing_selector_preserves_order_and_storage_independence():
    plan = timing_plan(((3, 2), (7, -1)))
    args = dict(
        scan_origin_s=0.5,
        fft_seconds=0.125,
        block_starts_s=(1.375, 0.875, 0.5, 0.875),
    )
    np.testing.assert_array_equal(
        select_fft_source_timing(plan, **args), [-1, 2, 0, 2]
    )
    assert select_fft_source_replay(plan, **args, fft_samples=8)[0] == (
        0,
        123456,
        0,
        123456,
    )


@pytest.mark.parametrize(
    "events,error",
    [
        (((3, 0),), ValueError),
        (((3, True),), TypeError),
        (((3, 2.5),), TypeError),
        (((3, 2), (3, 1)), ValueError),
        (((20, 1),), ValueError),
        ([(3, 1)], TypeError),
        (([3, 1],), TypeError),
        (((3, 2**63),), OverflowError),
        (((7, 1), (3, 1)), ValueError),
    ],
)
def test_timing_events_reject_ambiguous_or_out_of_scope_inputs(events, error):
    with pytest.raises(error):
        timing_plan(events)


@pytest.mark.parametrize("timing_offset", [-2, 1, 3])
@pytest.mark.parametrize(
    "array_module",
    ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)],
)
def test_logical_phase_preserves_decoded_samples(timing_offset, array_module):
    job, model, data = fixture()
    # A nonconstant real voltage exercises every retained Fourier bin.
    data.samples = np.array([1, 3, -2, 1, -3, 2, 2, -1], dtype=np.float32)
    starts = np.array([0.0])
    if array_module == "cupy":
        from tests.xbeam.test_gpu_backend import _cupy

        xp = _cupy()
        to_host = xp.asnumpy
    else:
        xp = np
        to_host = np.asarray
    original = gpu_execution._station_blocks(job, data, model, starts, xp)
    shifted = gpu_execution._station_blocks(
        job,
        data,
        model,
        starts,
        xp,
        source_timing_offsets=np.array([timing_offset], dtype=np.int64),
    )
    expected = to_host(original.spectrum) * np.exp(
        -2j
        * np.pi
        * timing_offset
        * to_host(original.baseband_hz)
        / data.sample_rate_hz
    )
    np.testing.assert_allclose(
        to_host(shifted.spectrum), expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(
        to_host(shifted.weight), to_host(original.weight)
    )


def test_phase_only_event_disables_fusion_only_in_its_execution_window():
    from types import SimpleNamespace

    job, model, data = fixture()
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        None,
        timeline_fft_count=20,
        source_timing_events=((3, 1),),
    )
    scan = SimpleNamespace(identifier="generated", start_offset_s=0.0)
    groups = (((0, 0), data),)
    plans = {("generated", 0, 0): plan}
    assert gpu_execution._packed_group_requires_source_replay(
        groups, plans, scan, np.array([0.75])
    )
    assert not gpu_execution._packed_group_requires_source_replay(
        groups, plans, scan, np.array([1.0])
    )
