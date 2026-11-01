# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from cuphoton.xbeam import upload_resources
from cuphoton.xbeam.adapters.radio import engine
from cuphoton.xbeam.adapters.vdif.recording import VdifFileRecordingSource


class ObservedWarmup(Future):
    def __init__(self):
        super().__init__()
        self.waiting = Event()

    def result(self, timeout=5):
        self.waiting.set()
        return super().result(timeout=timeout)


def make_run(monkeypatch, warmup, *, source_type=VdifFileRecordingSource):
    # These tests exercise warmup ordering without a CUDA memory allocator.
    monkeypatch.setattr(
        engine, "_gpu_memory_headroom", lambda _xp: nullcontext()
    )
    source = object.__new__(source_type)
    band = SimpleNamespace(identifier=0, frequency_id=0)
    streams = [
        SimpleNamespace(identifier=i, recorded_bands=[band]) for i in range(2)
    ]
    source.batch_cache_key = lambda stream: (stream.identifier,)
    run = engine.CorrelationRun(
        job=SimpleNamespace(
            frequencies=[SimpleNamespace(identifier=0)],
            datastreams=streams,
            stations=[],
        ),
        backend="cupy",
        profile="analytic-v1",
        batch_blocks=None,
        vdif_io="posix",
        source=source,
        timing_groups=(),
        xp=SimpleNamespace(
            __name__="cupy",
            cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=987)),
        ),
        gpu_warmup=warmup,
    )
    gpu_called = Event()

    def first_gpu_operation(*args, **kwargs):
        gpu_called.set()
        raise RuntimeError("GPU boundary reached")

    monkeypatch.setattr(engine, "_load_band_samples", first_gpu_operation)
    return run, gpu_called


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_builtin_planning_overlaps_but_gpu_waits(monkeypatch, warmup_fails):
    warmup = ObservedWarmup()
    run, gpu_called = make_run(monkeypatch, warmup)
    planned = Event()
    barrier = Barrier(2)

    def batches(stream):
        barrier.wait(timeout=5)
        planned.set()
        return []

    run.source.batches = batches
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda: list(run.records()))
        try:
            assert planned.wait(5)
            assert warmup.waiting.wait(5)
            assert not gpu_called.is_set()
        finally:
            if warmup_fails:
                warmup.set_exception(ValueError("warmup failed"))
            else:
                warmup.set_result(None)
        error = ValueError if warmup_fails else RuntimeError
        message = "warmup failed" if warmup_fails else "GPU boundary reached"
        with pytest.raises(error, match=message):
            result.result(timeout=5)
    assert gpu_called.is_set() != warmup_fails
    assert 987 not in upload_resources._vdif_ingest_reservations


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_host_failure_drains_workers_and_warmup(monkeypatch, warmup_fails):
    warmup = ObservedWarmup()
    run, gpu_called = make_run(monkeypatch, warmup)
    worker_done = Event()
    barrier = Barrier(2)

    def batches(stream):
        barrier.wait(timeout=5)
        if stream.identifier == 0:
            raise LookupError("host failed")
        worker_done.set()
        return []

    run.source.batches = batches
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda: list(run.records()))
        try:
            assert warmup.waiting.wait(5)
            assert worker_done.is_set()
            assert not result.done()
            assert 987 in upload_resources._vdif_ingest_reservations
        finally:
            if warmup_fails:
                warmup.set_exception(ValueError("warmup failed"))
            else:
                warmup.set_result(None)
        with pytest.raises(LookupError, match="host failed") as caught:
            result.result(timeout=5)
    if warmup_fails:
        assert isinstance(caught.value.__cause__, ValueError)
    assert not gpu_called.is_set()
    assert 987 not in upload_resources._vdif_ingest_reservations


def test_custom_subclass_waits_before_source_access(monkeypatch):
    class CustomSource(VdifFileRecordingSource):
        pass

    warmup = ObservedWarmup()
    run, _ = make_run(monkeypatch, warmup, source_type=CustomSource)
    planned = Event()
    run.source.batches = lambda stream: planned.set() or []
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda: list(run.records()))
        try:
            assert warmup.waiting.wait(5)
            assert not planned.is_set()
        finally:
            warmup.set_result(None)
        with pytest.raises(RuntimeError, match="GPU boundary reached"):
            result.result(timeout=5)
    assert planned.is_set()


def test_disabled_warmup(monkeypatch):
    run, gpu_called = make_run(monkeypatch, None)
    run.source.batches = lambda stream: []
    with pytest.raises(RuntimeError, match="GPU boundary reached"):
        list(run.records())
    assert gpu_called.is_set()
    assert 987 not in upload_resources._vdif_ingest_reservations


def test_close_releases_reservation_after_deferred_warmup(monkeypatch):
    warmup = ObservedWarmup()
    warmup.set_result(None)
    run, _ = make_run(monkeypatch, warmup)

    def execute(*args, gpu_warmup=None, **kwargs):
        with engine._host_batch_preparation(gpu_warmup):
            pass
        yield "record"

    monkeypatch.setattr(engine, "_execute_correlation", execute)
    records = run.records()
    assert next(records) == "record"
    assert warmup.waiting.is_set()
    assert 987 in upload_resources._vdif_ingest_reservations
    records.close()
    assert 987 not in upload_resources._vdif_ingest_reservations
