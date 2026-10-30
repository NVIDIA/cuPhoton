# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.clocks import (
    LeadingRawHeadContract,
    LeadingRawHeadInputs,
    compile_leading_raw_head_coordinates,
)
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.requests import (
    evaluate_raw_job_requests,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import RawValidityState
from tests.xbeam.test_buffered_window_job_requests import model_job

RATE, FFT, SPF = 4_096_000_000, 262144, 32768
GEOMETRY = BufferedRequestGeometry(4, FFT, 50, 8192, 4)
CONTRACT = LeadingRawHeadContract(RATE)


def state(runs=((0, 120000, 0),)):
    values = np.asarray(runs, dtype=np.int64)
    timeline = PhysicalAllocationTimeline.from_runs(
        values[:, 0], values[:, 1], values[:, 2], allocation_frames=1206
    )
    return RawValidityState.from_timeline(
        timeline,
        RawBufferedRequestState.from_timeline(
            timeline, samples_per_frame=SPF, timestamp_origin_frame=0
        ),
    )


def evaluate(coefficients, *, first=1250, stop=1300):
    args = model_job(coefficients, rate=RATE, fft=FFT)
    return evaluate_raw_job_requests(
        *args,
        first_fft=first,
        stop_fft=stop,
        timeline_fft_count=2000,
        job_start_slot=0,
        timestamp_origin_frame=0,
    )


def expand(addresses):
    return {
        index: span.physical_sample_start + (index - span.first_fft) * FFT
        for span in addresses.spans
        for index in range(span.first_fft, span.stop_fft)
    }


def test_large_integer_origins_retain_low_bits():
    output = np.asarray([2**60 + 3], dtype=np.int64)
    inputs = LeadingRawHeadInputs(
        output, np.asarray([-1.49 / RATE]), np.asarray([-1.49 / RATE])
    )
    result = compile_leading_raw_head_coordinates(
        inputs, CONTRACT, samples_per_frame=SPF
    )
    assert result.head_sample_starts.tolist() == [2**60]
    assert result.source_sample_starts.tolist() == [2**60 + 4]
    assert result.nearest_samples.tolist() == [4]
    output[0] = 0
    assert inputs.request_output_samples[0] == 2**60 + 3
    for value in (
        inputs.request_output_samples,
        inputs.endpoint_delays_seconds,
        result.source_sample_starts,
    ):
        with pytest.raises(ValueError):
            value.setflags(write=True)


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "integer_type", [int, np.int64, np.uint64, _IntSubclass]
)
def test_head_contract_normalizes_integral_inputs(integer_type):
    contract = LeadingRawHeadContract(
        integer_type(2**60 + 3), integer_type(8)
    )
    assert type(contract.sample_rate) is int
    assert contract.sample_rate == 2**60 + 3
    assert type(contract.timestamp_quantum_nanoseconds) is int
    assert contract.timestamp_quantum_nanoseconds == 8


@pytest.mark.parametrize("rate", [True, np.bool_(True), 0, -1, 1.0])
def test_head_contract_rate_requires_positive_integer(rate):
    with pytest.raises((ValueError, TypeError)):
        LeadingRawHeadContract(rate)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_delays_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        LeadingRawHeadInputs(
            np.asarray([0], dtype=np.int64),
            np.asarray([value]),
            np.asarray([0.0]),
        )


def test_head_input_shape_and_dtype_are_explicit():
    with pytest.raises(TypeError):
        LeadingRawHeadInputs(
            np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0])
        )
    with pytest.raises(ValueError, match="equal shape"):
        LeadingRawHeadInputs(
            np.asarray([0], dtype=np.int64),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0]),
        )


@pytest.mark.parametrize(
    "endpoint, centre",
    [
        (1e308, 0.0),
        (0.0, 1e308),
        (0.0, 2**53 / RATE),
    ],
)
def test_delay_multiplications_and_phase_range_fail_closed(endpoint, centre):
    inputs = LeadingRawHeadInputs(
        np.asarray([0], dtype=np.int64),
        np.asarray([endpoint]),
        np.asarray([centre]),
    )
    with pytest.raises(OverflowError):
        compile_leading_raw_head_coordinates(
            inputs, CONTRACT, samples_per_frame=SPF
        )


def test_frame_head_overflow_fails_without_origin_float_conversion():
    inputs = LeadingRawHeadInputs(
        np.asarray([np.iinfo(np.int64).min], dtype=np.int64),
        np.asarray([1.0]),
        np.asarray([1.0]),
    )
    with pytest.raises(OverflowError, match="frame head"):
        compile_leading_raw_head_coordinates(
            inputs, CONTRACT, samples_per_frame=SPF
        )


def test_job_inputs_expose_distinct_immutable_endpoint_and_centre_delays():
    result = evaluate((-0.49 / RATE, 100000 / RATE))
    head = result.leading_head_inputs
    assert head is not None
    assert head.request_output_samples.tolist() == [1250 * FFT]
    assert (
        head.first_centre_delays_seconds[0] != head.endpoint_delays_seconds[0]
    )
    for array in (
        head.request_output_samples,
        head.endpoint_delays_seconds,
        head.first_centre_delays_seconds,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
