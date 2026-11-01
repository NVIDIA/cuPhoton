# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.plans import (
    OrdinaryRawSourceSpan,
    RawLinearStorage,
)
from cuphoton.xbeam.buffered_reader.replay import (
    RawWindowClockContract,
    RetainedRawWindowContract,
    compile_raw_replay_window_facts,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline
from cuphoton.xbeam.buffered_reader.validity import RawValidityState

SPF = 32768
FFT = 262144
GEOMETRY = BufferedRequestGeometry(4, FFT, 50, 8192, 4)


def state(hole=False, eof=120000):
    runs = (
        [(0, 9711, 0), (9712, eof, 10709)]
        if hole
        else [(0, 9701, 0), (9701, eof, 10698)]
    )
    runs = np.asarray(runs, dtype=np.int64)
    timeline = PhysicalAllocationTimeline.from_runs(
        runs[:, 0], runs[:, 1], runs[:, 2], allocation_frames=1206
    )
    request = RawBufferedRequestState.from_timeline(
        timeline, samples_per_frame=SPF
    )
    return RawValidityState.from_timeline(timeline, request)


def expand_spans(spans):
    return {
        index: span.physical_sample_start + (index - span.first_fft) * FFT
        for span in spans
        for index in range(span.first_fft, span.stop_fft)
    }


def storage_for(result):
    first, stop = result.physical_sample_extent
    return RawLinearStorage.align_to_ordinary_window(
        datastream_id=0,
        source_index=1,
        job_start_slot=0,
        samples_per_frame=SPF,
        raw_first_frame=first // SPF,
        raw_stop_frame=-(-stop // SPF),
        ordinary_min_sample=result.ordinary_sample_extent[0],
    )


def window_contract(first, offsets, framed, decoded, *, origin=2**55):
    offsets = np.asarray(offsets, dtype=np.int64)
    output = origin + np.arange(offsets.size, dtype=np.int64) * FFT
    return RawWindowClockContract(
        first,
        output,
        output - offsets,
        np.zeros(offsets.size, dtype=np.float64),
        np.broadcast_to(
            np.asarray(framed, dtype=np.int64), offsets.shape
        ).copy(),
        np.broadcast_to(
            np.asarray(decoded, dtype=np.int64), offsets.shape
        ).copy(),
    )


def test_window_scope_normalizes_integer_origin_and_rejects_overflow():
    contract = window_contract(np.int64(1200), [0, 0], FFT, FFT)
    assert type(contract.first_fft) is int
    assert contract.stop_fft == 1202
    with pytest.raises(OverflowError, match="FFT scope"):
        window_contract(np.int64(2**63 - 1), [0, 0], FFT, FFT)


@pytest.mark.parametrize(
    "ordinary, expected",
    [
        (
            [
                [-2 * FFT, -FFT, 0],
                [FFT, 2 * FFT, 3 * FFT],
                [4 * FFT, 5 * FFT, 0],
            ],
            ((11, 18, -FFT),),
        ),
        (
            [
                [-2 * FFT, -FFT, 0],
                [FFT, 2 * FFT, 4 * FFT],
                [5 * FFT, 6 * FFT, 0],
            ],
            ((11, 15, -FFT), (15, 18, 4 * FFT)),
        ),
        (
            [
                [-2 * FFT, -FFT, 0],
                [2**63 - FFT, -(2**63), -(2**63) + FFT],
                [-(2**63) + 2 * FFT, -(2**63) + 3 * FFT, 0],
            ],
            (
                (11, 13, -FFT),
                (13, 14, 2**63 - FFT),
                (14, 18, -(2**63)),
            ),
        ),
    ],
)
def test_ordinary_runs_merge_across_clipped_requests_without_int64_wrap(
    ordinary, expected
):
    output = np.arange(11, 18, dtype=np.int64) * FFT
    extent = np.full(output.shape, FFT, dtype=np.int64)
    window = RetainedRawWindowContract(
        11,
        output,
        output,
        np.zeros(output.shape),
        extent,
        extent,
        output,
        np.ones(output.shape, dtype=np.bool_),
    )
    result = compile_raw_replay_window_facts(
        state(),
        np.asarray([10, 13, 16], dtype=np.int64),
        np.asarray([10, 13, 16], dtype=np.int64) * FFT,
        np.zeros(3),
        np.asarray(ordinary, dtype=np.int64),
        GEOMETRY,
        first_fft=11,
        stop_fft=18,
        sample_rate=4096000000,
        frame_bytes=8224,
        control_word_bits=30,
        request_fft_counts=np.asarray([3, 3, 2], dtype=np.int64),
        window_clock_contract=window,
    )
    assert result.complete
    assert result.ordinary_gpu_source_spans == tuple(
        OrdinaryRawSourceSpan(*span) for span in expected
    )
    selected = np.asarray(ordinary, dtype=np.int64).ravel()[1:8]
    assert result.ordinary_sample_extent == (
        int(selected.min()),
        int(selected.max()) + FFT,
    )


def facts(cache, first, stop, *, delay=0, counts=None, window):
    """Independent enclosing request inputs with an explicit returned view."""
    starts = np.arange(first // 50 * 50, stop, 50, dtype=np.int64)
    desired = starts * FFT - delay
    ordinary = desired[:, None] + np.arange(50, dtype=np.int64) * FFT
    return compile_raw_replay_window_facts(
        cache,
        starts,
        desired,
        np.zeros(starts.size),
        ordinary,
        GEOMETRY,
        first_fft=first,
        stop_fft=stop,
        sample_rate=4096000000,
        frame_bytes=8224,
        control_word_bits=30,
        request_fft_counts=counts,
        window_clock_contract=window,
    )
