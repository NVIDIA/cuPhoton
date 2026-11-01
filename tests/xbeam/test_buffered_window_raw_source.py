# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.addressing import LatchedRawSourceState
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    RawBufferedRequestState,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline

GEOMETRY = BufferedRequestGeometry(4, 262144, 50, 8192, 4)
SAMPLES = 32768
FFT = 262144


def expand(source):
    return {
        index: span.physical_sample_start + (index - span.first_fft) * FFT
        for span in source.spans
        for index in range(span.first_fft, span.stop_fft)
    }


def test_prepared_source_state_rejects_a_different_cached_history():
    starts = np.asarray([0, 9701], dtype=np.int64)
    stops = np.asarray([9701, 120000], dtype=np.int64)
    first = PhysicalAllocationTimeline.from_runs(
        starts,
        stops,
        np.asarray([0, 10698], dtype=np.int64),
        allocation_frames=1206,
    )
    second = PhysicalAllocationTimeline.from_runs(
        starts,
        stops,
        np.asarray([0, 10916], dtype=np.int64),
        allocation_frames=1206,
    )
    state = RawBufferedRequestState.from_timeline(
        first, samples_per_frame=SAMPLES
    )
    with pytest.raises(ValueError, match="anchors differ"):
        LatchedRawSourceState.from_timeline(second, state)
