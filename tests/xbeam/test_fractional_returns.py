# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Own packed-buffer examples separate return selection from validity."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedNormalFullExtent,
    BufferedRequestGeometry,
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.replay import (
    compile_raw_replay_window_facts,
)
from cuphoton.xbeam.buffered_reader.requests import (
    RawJobRequestInputs,
    ReadClockConvention,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import RawValidityState


def inputs_and_state(*, gap=2, transfer=False, first=None, stop=None):
    # Four samples in two payload bytes, one header byte, ten-frame physical
    # allocations and known five-frame returns. No output-derived operands.
    timeline = PhysicalAllocationTimeline.from_runs(
        np.array([0, 33], dtype=np.int64),
        np.array([33, 100], dtype=np.int64),
        np.array([0, 33 + gap], dtype=np.int64),
        allocation_frames=10,
    )
    state = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=4
    )
    validity = RawValidityState.from_timeline(timeline, state)
    request = 24 if transfer else 20
    delay_samples = 8 if transfer else 4
    output = np.arange(request, request + 4, dtype=np.int64)[None, :] * 8
    ordinary = output - delay_samples
    inputs = RawJobRequestInputs(
        request if first is None else first,
        request + 4 if stop is None else stop,
        100,
        0,
        BufferedRequestGeometry(2, 8, 4, 2, 1),
        np.array([request], dtype=np.int64),
        np.array([4], dtype=np.int64),
        ordinary[:, 0],
        np.zeros(1),
        ordinary,
        128.0,
        read_clock_convention=ReadClockConvention.ENDPOINT_MEAN,
        centre_delays_seconds=np.full((1, 4), delay_samples / 128),
        output_sample_starts=output,
        read_average_delays_seconds=np.full((1, 4), delay_samples / 128),
    )
    return inputs, validity, BufferedNormalFullExtent(3, 2, 5, 2, 2)


def compile_facts(inputs, validity, extent, contract):
    return compile_raw_replay_window_facts(
        validity,
        inputs.request_fft_starts,
        inputs.desired_sample_starts,
        inputs.endpoint_drifts,
        inputs.ordinary_source_starts,
        inputs.geometry,
        first_fft=inputs.first_fft,
        stop_fft=inputs.stop_fft,
        sample_rate=inputs.sample_rate,
        frame_bytes=extent.frame_bytes,
        control_word_bits=2,
        request_fft_counts=inputs.request_fft_counts,
        window_clock_contract=contract,
    )


def sources(facts):
    return {
        i: span.physical_sample_start
        + (i - span.first_fft) * facts.fft_samples
        for span in facts.addresses.spans
        for i in range(span.first_fft, span.stop_fft)
    }


@pytest.mark.parametrize(
    "centre_samples,admitted", [(-1, True), (-2, False), (-3, False)]
)
@pytest.mark.parametrize("clipped", [False, True])
@pytest.mark.parametrize("packing_denominator", [1, 4, 8, 16])
def test_lane_tail_uses_actual_framed_extent(
    centre_samples, admitted, clipped, packing_denominator
):
    inputs, validity, _ = inputs_and_state(gap=2)
    # Own small buffer: four lane samples per frame, with one header sample
    # allowance per frame across integral and fractional packing ratios.
    # Allocation 3 spans samples 120..160; its framed
    # admission reaches 170. The next anchor advances by eight samples.
    # Keep endpoint selection fixed while independently supplying the FFT
    # centre clocks. The nonleading FFT starts at 169, 170, or 171.
    inputs = replace(
        inputs,
        geometry=BufferedRequestGeometry(
            Fraction(1, packing_denominator),
            8,
            4,
            4 * packing_denominator,
            packing_denominator,
        ),
        centre_delays_seconds=np.full((1, 4), centre_samples / 128),
        first_fft=21 if clipped else 20,
        stop_fft=22 if clipped else 24,
    )
    extent = BufferedNormalFullExtent(
        5 * packing_denominator,
        4 * packing_denominator,
        5,
        2,
        Fraction(1, packing_denominator),
    )
    returned = inputs.published_normal_full_window_contract(
        validity, extent, control_word_bits=2
    )
    assert (
        bool(returned.publication_eligible[0 if clipped else 1]) is admitted
    )
    # Address and clock operands retain their identity; publication alone
    # cannot relocate source samples or change the declared decoder extent.
    np.testing.assert_array_equal(returned.physical_head_samples, 156)
    np.testing.assert_array_equal(returned.aligned_head_samples, 156)
    np.testing.assert_array_equal(returned.framed_extent_samples, 25)
