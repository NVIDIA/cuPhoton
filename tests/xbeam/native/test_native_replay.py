# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.buffered_reader.clocks import (
    compile_raw_window_bounds,
    compile_raw_window_payload_spans,
)
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedRequestGeometry,
    RawBufferedRequestState,
    RawResidualOwner,
    compile_buffered_request_discards,
)
from cuphoton.xbeam.buffered_reader.timeline import PhysicalAllocationTimeline


@pytest.fixture(scope="module")
def binding():
    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    assert len(files) == 1
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("clock_owner", [False, True])
def test_request_discards(binding, seed, clock_owner):
    rng = np.random.default_rng(seed)
    runs = [(0, 1000, 10000)]
    for _ in range(5):
        a, b, t = runs[-1]
        first = b + int(rng.integers(0, 1500))
        timestamp = t + (b - a) + (first - b) + int(rng.integers(1, 500))
        runs.append((first, first + 1000, timestamp))
    capacity, samples, origin = 1206, 32768, 10000
    arrays = [
        np.asarray([r[i] for r in runs], dtype=np.int64) for i in range(3)
    ]
    timeline = PhysicalAllocationTimeline.from_runs(
        *arrays, allocation_frames=capacity
    )
    owner = (
        RawResidualOwner.CLOCK_FRAME
        if clock_owner
        else RawResidualOwner.PHYSICAL_FRAME
    )
    state = RawBufferedRequestState.from_timeline(
        timeline,
        samples_per_frame=samples,
        timestamp_origin_frame=origin,
        residual_owner=owner,
    )
    native = binding.NativeRawState(
        binding.NativeTimeline(runs, capacity), samples, origin, clock_owner
    )
    snapshot = native.snapshot()
    assert snapshot["anchors"] == state.anchor_samples.tolist()
    assert snapshot["events"] == state.event_flags.tolist()
    geometry = BufferedRequestGeometry(Fraction(4), 262144, 50, 8192, 4)
    first = np.arange(64, dtype=np.int64) * 50
    desired = np.linspace(
        -262144, int(state.anchor_samples[-1]) + capacity * samples, 64
    ).astype(np.int64)
    drifts = rng.uniform(-1e-6, 1e-6, 64)
    ordinary = np.ascontiguousarray(
        desired[:, None] + np.arange(50, dtype=np.int64)[None, :] * 262144
    )
    counts = rng.integers(1, 51, 64, dtype=np.int64)
    reference = compile_buffered_request_discards(
        state,
        first,
        desired,
        drifts,
        ordinary,
        geometry,
        sample_rate=4096000000.0,
        frame_bytes=8224,
        control_word_bits=30,
        request_fft_counts=counts,
    )
    actual = native.discards(
        first,
        desired,
        drifts,
        ordinary,
        counts,
        4,
        1,
        262144,
        50,
        8192,
        4,
        4096000000.0,
        8224,
        30,
    )
    assert actual["allocations"] == reference.selected_allocations.tolist()
    assert actual["valid"] == reference.valid_requests.tolist()
    assert actual["events"] == reference.event_requests.tolist()
    assert actual["eligible"] == [
        reference.eligible[i, : counts[i]].tolist() for i in range(64)
    ]
    assert actual["excluded"] == list(reference.ineligible_intervals)
    assert actual["cursors"] == list(
        zip(
            reference.cursors.origin_bytes.tolist(),
            reference.cursors.skipped_ffts.tolist(),
            reference.cursors.decoder_delta_samples.tolist(),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "delay",
    [
        0.0,
        -0.5,
        0.5,
        -1.0,
        1.0,
        np.nextafter(0.5, 1.0),
        np.nextafter(0.5, 0.0),
        -5e-324,
    ],
)
@pytest.mark.parametrize(
    "difference", [-100, -9, -8, -2, -1, 0, 1, 2, 8, 9, 100]
)
def test_window_clocks_and_payload(binding, delay, difference):
    head = 2**60
    output = head + difference
    a = np.asarray([output], np.int64)
    b = np.asarray([head], np.int64)
    delays = np.asarray([delay], np.float64)
    extents = np.asarray([16], np.int64)
    bounds = compile_raw_window_bounds(
        a, b, delays, extents, sample_rate=1, fft_samples=8
    )
    spans = compile_raw_window_payload_spans(
        bounds.nearest_samples, extents, fft_samples=8
    )
    assert binding.returned_window_bounds(
        output, head, delay, 1, 16, 16, 8
    ) == (
        int(bounds.nearest_samples[0]),
        bool(bounds.before_window[0]),
        bool(bounds.after_window[0]),
        int(spans.first_samples[0]),
        int(spans.stop_samples[0]),
    )
