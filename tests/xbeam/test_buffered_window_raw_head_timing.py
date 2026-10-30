# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clock/residual identities from independent signals, with no comparator."""

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.clocks import (
    LeadingRawHeadContract,
    LeadingRawHeadInputs,
    compile_leading_raw_head_coordinates,
    compile_leading_raw_timing_offsets,
)


def head_inputs(output, delay, rate=4_096_000_000):
    output = np.asarray(output, dtype=np.int64)
    centre = np.full(output.shape, delay / rate, dtype=np.float64)
    return LeadingRawHeadInputs(output, centre, centre)


@pytest.mark.parametrize("delay", [0.51, 1.51, -1.49, 4.51, -9.51])
@pytest.mark.parametrize("origin", [262144 * 1250, 2**60])
def test_clock_translation_and_effective_delay_identity(delay, origin):
    rate = 4_096_000_000
    inputs = head_inputs([origin], delay)
    contract = LeadingRawHeadContract(rate)
    coarse = np.rint(inputs.first_centre_delays_seconds * rate).astype(
        np.int64
    )
    timing = compile_leading_raw_timing_offsets(
        inputs, contract, coarse, samples_per_frame=32768
    )
    coordinates = compile_leading_raw_head_coordinates(
        inputs, contract, samples_per_frame=32768
    )
    assert int(coarse[0]) - int(timing[0]) == (
        origin - int(coordinates.source_sample_starts[0])
    )
    assert not timing.flags.writeable


@pytest.mark.parametrize("clock_delta", [-3, 1, 2, 4])
@pytest.mark.parametrize("storage_shift", [-1206 * 32768, 0, 997 * 32768])
def test_clock_correction_removes_signal_phase_without_storage_operand(
    clock_delta, storage_shift
):
    # Independently sampled complex sinusoid. A change of logical sample
    # clock supplies +omega*delta. Its residual correction must cancel it.
    count, bin_index = 256, 17
    omega = 2 * np.pi * bin_index / count
    samples = np.arange(count, dtype=np.float64)
    nominal = np.exp(1j * omega * samples)
    shifted = np.exp(1j * omega * (samples + clock_delta))
    expected = np.fft.fft(nominal)[bin_index]
    corrected = np.fft.fft(shifted)[bin_index] * np.exp(
        -1j * omega * clock_delta
    )
    assert corrected == pytest.approx(expected, abs=1e-11)
    # Storage movement preserves the logical clock; it never belongs in the
    # residual correction. Moving the same bytes to another allocation keeps
    # the signal and its phase correction identical.
    storage = {storage_shift + i: value for i, value in enumerate(shifted)}
    recovered = np.asarray([storage[storage_shift + i] for i in range(count)])
    assert np.array_equal(recovered, shifted)


def test_mismatched_coarse_delay_is_rejected():
    with pytest.raises(ValueError, match="first-centre"):
        compile_leading_raw_timing_offsets(
            head_inputs([262144 * 1250], 1.51),
            LeadingRawHeadContract(4_096_000_000),
            np.asarray([1], dtype=np.int64),
            samples_per_frame=32768,
        )


def test_empty_heads_and_coarse_shape_validation():
    inputs = head_inputs([], 0)
    contract = LeadingRawHeadContract(4_096_000_000)
    result = compile_leading_raw_timing_offsets(
        inputs, contract, np.empty(0, dtype=np.int64), samples_per_frame=32768
    )
    assert result.shape == (0,)
    with pytest.raises(TypeError, match="int64 vector"):
        compile_leading_raw_timing_offsets(
            inputs,
            contract,
            np.asarray([0], dtype=np.int64),
            samples_per_frame=32768,
        )
