# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.parametrize("seed", range(20))
def test_timeline_matches_python(binding, seed):
    rng = random.Random(seed)
    rows = [(0, 10, 100)]
    for _ in range(5):
        a, b, t = rows[-1]
        gap = rng.randrange(5)
        first = b + gap
        timestamp = t + (b - a) + gap + rng.randrange(8)
        rows.append((first, first + rng.randrange(1, 15), timestamp))
    capacity = rng.randrange(2, 20)
    arrays = [
        np.asarray([r[i] for r in rows], dtype=np.int64) for i in range(3)
    ]
    reference = PhysicalAllocationTimeline.from_runs(
        *arrays, allocation_frames=capacity
    )
    native = binding.NativeTimeline(rows, capacity)
    snapshot = native.snapshot()
    assert (
        snapshot["allocation_starts"] == reference.allocation_starts.tolist()
    )
    assert snapshot["anchors"] == reference.anchors.tolist()
    assert snapshot["uncovered_heads"] == list(
        reference.uncovered_head_allocations()
    )
    assert snapshot["deferred_gaps"] == [
        (
            x.allocation_index,
            x.physical_boundary_frame,
            x.timestamp_start,
            x.timestamp_stop,
        )
        for x in reference.deferred_anchor_gaps()
    ]
    for first in range(380, 650, 7):
        expected = reference.sample_source_spans(
            first, 61, samples_per_frame=4
        )
        assert native.source_spans(first, 61, 4) == [
            (x.request_start, x.request_stop, x.physical_start)
            for x in expected
        ]
    for allocation in range(len(reference.allocation_starts)):
        first = int(reference.anchors[allocation]) * 4 - 3
        expected = reference.latched_source_spans(
            allocation, first, 63, samples_per_frame=4
        )
        assert native.latched_spans(allocation, first, 63, 4) == [
            (x.request_start, x.request_stop, x.physical_start)
            for x in expected
        ]
