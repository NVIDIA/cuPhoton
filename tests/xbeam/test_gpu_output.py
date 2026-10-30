# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.engine import _output_pipeline_requested
from cuphoton.xbeam.adapters.radio.output_records import (
    _host_gpu_records,
    _new_gpu_output_arena,
    _PendingGpuRecord,
)
from cuphoton.xbeam.gpu_output import OutputPipeline


class Stream:
    def __init__(self, **kwargs):
        self.operations = []
        self.cursor = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def wait_event(self, event):
        self.operations.append(event.synchronize)

    def run(self, stop):
        while self.cursor < stop:
            operation = self.operations[self.cursor]
            operation()
            self.cursor += 1

    def synchronize(self):
        self.run(len(self.operations))


class Event:
    def __init__(self, **kwargs):
        self.stream = None

    def record(self, stream):
        self.stream = stream
        self.stop = len(stream.operations)

    def synchronize(self):
        self.stream.run(self.stop)


class Raw:
    def __init__(self):
        self.values = np.zeros(24, np.uint8)
        self.nbytes = self.values.nbytes

    def get(self, *, out, stream, blocking):
        assert not blocking
        stream.operations.append(lambda: np.copyto(out, self.values))


def fake_pipeline(*, pinned_factory=np.empty, tiled=False):
    current = Stream()
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=Stream,
            Event=Event,
            get_current_stream=lambda: current,
        )
    )

    def arena():
        return SimpleNamespace(
            raw=Raw(), spectrum_capacity=2, record_capacity=1
        )

    return OutputPipeline(
        xp, arena, pinned_factory=pinned_factory, tiled=tiled
    )


def test_tiled_pipeline_owns_exactly_two_scratch_arenas():
    pipeline = fake_pipeline(tiled=True)
    first, second = pipeline.acquire(), pipeline.acquire()
    scratch = pipeline.scratch_for(first)
    assert scratch is not first
    assert pipeline.scratch_for(second) is not scratch
    pipeline.enqueue(first, [0])
    pipeline.enqueue(second, [1])
    with pipeline.completed():
        pass
    assert pipeline.acquire() is first
    assert pipeline.scratch_for(first) is scratch
    pipeline.close()
    assert not pipeline.slots


def test_pipeline_bounds_ownership_and_drains_in_order():
    pipeline = fake_pipeline()
    try:
        first = pipeline.acquire()
        first.raw.values.fill(1)
        pipeline.enqueue(first, ["first"])
        second = pipeline.acquire()
        second.raw.values.fill(2)
        pipeline.enqueue(second, ["second"])
        with pytest.raises(RuntimeError, match="both output slots"):
            pipeline.acquire()
        with pipeline.completed() as (records, arena, host):
            assert records == ["first"]
            assert arena is first
            np.testing.assert_array_equal(host, np.ones(24, np.uint8))
            with pytest.raises(RuntimeError, match="both output slots"):
                pipeline.acquire()
        recycled = pipeline.acquire()
        assert recycled is first
        assert recycled.spectrum_cursor == recycled.record_cursor == 0
        recycled.raw.values.fill(3)
        pipeline.enqueue(recycled, ["third"])
        for expected in (2, 3):
            with pipeline.completed() as (_, _, host):
                np.testing.assert_array_equal(
                    host, np.full(24, expected, np.uint8)
                )
    finally:
        pipeline.close()


def test_pipeline_records_do_not_alias_reused_host_slot():
    pipeline = fake_pipeline()
    job = SimpleNamespace(
        integration_duration_s=Decimal(1),
        start_time=SimpleNamespace(seconds_of_day=Decimal(0), mjd=60000),
    )
    scan = SimpleNamespace(
        start_offset_s=Decimal(0), source_id=0, configuration_id=0
    )
    station = SimpleNamespace(name="A")
    record = _PendingGpuRecord(
        0, scan, station, station, 0, "R", "R", (0, 0, 0), 0, 1.0, 0, 2
    )
    try:
        arena = pipeline.acquire()
        arena.raw.values[:16].view(np.complex64)[:] = [2 + 4j, 6 + 8j]
        arena.raw.values[16:].view(np.float64)[:] = 2
        pipeline.enqueue(arena, [record])
        with pipeline.completed() as (pending, arena, host):
            results = list(
                _host_gpu_records(
                    job, pending, arena, host, copy_spectra=True
                )
            )
            assert not np.shares_memory(results[0].spectrum, host)
        arena = pipeline.acquire()
        arena.raw.values.fill(0)
        pipeline.enqueue(arena, [record])
        with pipeline.completed() as (pending, arena, host):
            assert (
                list(
                    _host_gpu_records(
                        job, pending, arena, host, copy_spectra=True
                    )
                )
                == []
            )
        np.testing.assert_array_equal(results[0].spectrum, [1 + 2j, 3 + 4j])
    finally:
        pipeline.close()


def test_pipeline_close_drains_unconsumed_copies_and_partial_compute():
    pipeline = fake_pipeline()
    arena = pipeline.acquire()
    arena.raw.values.fill(9)
    pipeline.enqueue(arena, [1])
    host = pipeline.slots[0].host
    second = pipeline.acquire()
    completed = []
    pipeline.slots[1].compute_stream.operations.append(
        lambda: completed.append(second)
    )
    pipeline.close()
    np.testing.assert_array_equal(host, np.full(24, 9, np.uint8))
    assert completed == [second]
    assert not pipeline.slots


def test_pipeline_partial_allocation_failure():
    calls = []

    def pinned(size, dtype):
        calls.append(size)
        if len(calls) == 2:
            raise MemoryError("pinned allocation failed")
        return np.empty(size, dtype)

    with pytest.raises(MemoryError, match="pinned allocation"):
        fake_pipeline(pinned_factory=pinned)
    assert len(calls) == 2


def test_output_pipeline_default_and_override(monkeypatch):
    name = "CUPHOTON_XBEAM_OUTPUT_PIPELINE"
    monkeypatch.delenv(name, raising=False)
    assert _output_pipeline_requested()
    monkeypatch.setenv(name, "0")
    assert not _output_pipeline_requested()
    monkeypatch.setenv(name, "invalid")
    with pytest.raises(ValueError, match="boolean"):
        _output_pipeline_requested()


@pytest.mark.gpu
@pytest.mark.parametrize("tiled", [False, True])
def test_output_pipeline_real_cuda_delayed_events_and_reuse(tiled):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).use()
    except Exception as error:
        pytest.skip(str(error))
    pipeline = OutputPipeline(
        cp, lambda: _new_gpu_output_arena(cp, 1, 128), tiled=tiled
    )
    results = []
    try:
        # Copies depend on non-default compute work; no host sync at enqueue.
        with cp.cuda.Stream(non_blocking=True):
            for value in range(5):
                arena = pipeline.acquire()
                arena.spectra.fill(value + 1j)
                arena.weights.fill(value + 1)
                if tiled:
                    arena.spectra.fill(0)
                    scratch = pipeline.scratch_for(arena)
                    for _ in range(3):
                        scratch.spectra.fill(value + 1j)
                        arena.spectra += scratch.spectra
                    arena.spectra /= 3
                pipeline.enqueue(arena, [value])
                if len(pipeline.queue) == 2:
                    with pipeline.completed() as (pending, _, host):
                        results.append((pending[0], host.copy()))
            while pipeline.queue:
                with pipeline.completed() as (pending, _, host):
                    results.append((pending[0], host.copy()))
        for value, raw in results:
            np.testing.assert_array_equal(
                raw[:1024].view(np.complex64), value + 1j
            )
            np.testing.assert_array_equal(
                raw[1024:1032].view(np.float64), value + 1
            )
            np.testing.assert_array_equal(raw[1032:].view(np.float64), 0)
    finally:
        pipeline.close()
