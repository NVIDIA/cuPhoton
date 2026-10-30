# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for bounded GDS ownership and aggregate memory planning."""

import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam import ingest_scheduler, upload_resources
from cuphoton.xbeam.adapters.radio import compute_plan, engine, gpu_execution
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifPayloadFilePlan,
    VdifPayloadPlan,
)


def test_streaming_cufile_rejected_before_device_allocation():
    xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0))
    )
    with pytest.raises(ValueError, match="requires mapped VDIF"):
        upload_resources.load_vdif_payload_plans(
            [SimpleNamespace(streaming=object())],
            xp=xp,
            io_backend="cufile",
        )


@pytest.mark.parametrize("frame_count", [4, 12])
def test_trusted_mapping_budget_matches_file_length(
    monkeypatch, tmp_path, frame_count
):
    class Array:
        dtype = np.dtype(np.uint8)
        ndim = 1
        flags = SimpleNamespace(c_contiguous=True)
        data = SimpleNamespace(ptr=65536)

        def __init__(self, size):
            self.size = self.nbytes = size

    xp = SimpleNamespace(
        ndarray=Array,
        uint8=np.uint8,
        cuda=SimpleNamespace(
            get_current_stream=lambda: object(),
            Event=lambda: SimpleNamespace(record=lambda stream: None),
        ),
    )
    monkeypatch.setattr(
        upload_resources, "vdif_upload_staging_slots", lambda _: 2
    )
    calls = []
    monkeypatch.setattr(
        upload_resources,
        "_load_vdif_payload_plans_cufile",
        lambda **kwargs: calls.append(kwargs) or kwargs["outputs"],
    )
    plan = VdifPayloadPlan(
        40,
        8,
        frame_count,
        (
            VdifPayloadFilePlan(
                tmp_path / "source.vdif",
                frame_count,
                np.arange(frame_count, dtype=np.int32),
            ),
        ),
    )
    # Two raw slots each retain 4 MiB plus alignment padding, and each
    # mapping slot holds the complete file, whether shorter or longer than
    # the ten-frame read chunk.
    budget = 2 * ((4 << 20) + 65535 + frame_count * 4)
    arguments = dict(
        xp=xp,
        target_chunk_bytes=440,
        outputs=[Array(plan.byte_count)],
        io_backend="cufile",
        _validated_mapped_token=upload_resources._TRUSTED_MAPPED_VDIF_PLANS,
    )
    upload_resources._load_vdif_payload_plans(
        [plan], available_device_bytes=budget, **arguments
    )
    assert len(calls) == 1
    with pytest.raises(MemoryError, match="exceed available device memory"):
        upload_resources._load_vdif_payload_plans(
            [plan], available_device_bytes=budget - 1, **arguments
        )
    assert len(calls) == 1


@pytest.mark.parametrize("stage", ["prepare", "kernel"])
def test_owned_cufile_state_is_cleaned_on_initialization_failure(
    monkeypatch, stage
):
    xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0))
    )
    calls = []

    def prepare(state, *args):
        state.workspace = object()
        if stage == "prepare":
            raise MemoryError("initialization failure")

    def fail():
        raise RuntimeError("initialization failure")

    monkeypatch.setattr(
        upload_resources, "_prepare_cufile_upload_state", prepare
    )
    monkeypatch.setattr(
        upload_resources,
        "_cuda_launch_policy",
        lambda *args: SimpleNamespace(ingest_threads=256),
    )
    monkeypatch.setattr(upload_resources, "_vdif_ingest_kernel", fail)
    monkeypatch.setattr(
        upload_resources,
        "_close_vdif_cufile_upload_state",
        lambda state, **kw: calls.append((state, kw)),
    )
    with pytest.raises(
        (MemoryError, RuntimeError), match="initialization failure"
    ):
        upload_resources._load_vdif_payload_plans_cufile(
            plans=[object()],
            outputs=[object()],
            reference_files=(),
            validated_mappings=[],
            frame_size=40,
            payload_bytes=8,
            chunk_frames=1,
            ready_event=None,
            progresses=None,
            xp=xp,
        )
    assert len(calls) == 1
    assert calls[0][1] == {"reusable": False, "best_effort": True}


def test_submission_join_waits_for_all_even_after_failure():
    first, second = Future(), Future()
    done = upload_resources._join_cufile_submissions([first, second])
    first.set_exception(ValueError("submit failed"))
    assert not done.done()
    second.set_result(None)
    with pytest.raises(ValueError, match="submit failed"):
        done.result()


def test_submission_join_handles_already_completed_and_empty():
    first = Future()
    first.set_result(None)
    assert upload_resources._join_cufile_submissions([first]).result() is None
    assert upload_resources._join_cufile_submissions([]).result() is None


def test_frontier_budget_uses_independent_maxima(monkeypatch):
    monkeypatch.setattr(
        ingest_scheduler, "vdif_upload_staging_slots", lambda _: 4
    )
    monkeypatch.setattr(
        ingest_scheduler,
        "_cufile_upload_state_requirements",
        lambda plans, **kwargs: plans,
    )
    tasks = [
        SimpleNamespace(plans=capacities, target_chunk_bytes=64)
        for capacities in [(1024, 16), (64, 2048)]
    ]
    device, pinned = upload_resources._cufile_retained_staging_bytes(
        1024, 2048, 4
    )
    assert ingest_scheduler._frontier_staging_plan(tasks, 2, device * 20) == (
        2,
        1024,
        2048,
        device,
        pinned,
    )
    assert (
        ingest_scheduler._frontier_staging_plan(tasks, 2, device * 10)[0] == 1
    )
    with pytest.raises(MemoryError, match="one worker"):
        ingest_scheduler._frontier_staging_plan(tasks, 2, device * 10 - 1)


@pytest.mark.parametrize("abort_after", [None, 0, 1])
def test_posix_frontier_worker_stops_between_tasks_after_abort(
    monkeypatch, abort_after
):
    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Progress:
        def __init__(self):
            self.errors = []

        def finish(self, error):
            self.errors.append(error)

    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: Context()))
    tasks = [
        ingest_scheduler._FrontierUploadTask(i, [], [], [], [Progress()], 64)
        for i in range(3)
    ]
    abort = Event()
    if abort_after == 0:
        abort.set()
    uploaded = []

    def upload(*args, **kwargs):
        uploaded.append(args[6])
        if len(uploaded) == abort_after:
            abort.set()

    monkeypatch.setattr(ingest_scheduler, "_upload_vdif_plan_group", upload)
    monkeypatch.setattr(
        ingest_scheduler, "_close_vdif_posix_upload_state", lambda state: None
    )
    if abort_after is not None:
        with pytest.raises(RuntimeError, match="stopped before VDIF upload"):
            ingest_scheduler._run_frontier_upload_worker(
                tasks, xp, 0, None, 4096, None, abort
            )
        assert len(uploaded) == abort_after
        assert all(
            task.progresses[0].errors == [] for task in tasks[:abort_after]
        )
        for task in tasks[abort_after:]:
            (error,) = task.progresses[0].errors
            assert isinstance(error, RuntimeError)
    else:
        ingest_scheduler._run_frontier_upload_worker(
            tasks, xp, 0, None, 4096, None, abort
        )
        assert len(uploaded) == 3
        assert all(task.progresses[0].errors == [] for task in tasks)


@pytest.mark.parametrize("fail", [False, True])
def test_frontier_worker_bounds_tasks_and_drains_before_close(
    monkeypatch, fail
):
    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: Context()))
    tasks = [
        ingest_scheduler._FrontierUploadTask(i, [], [], [], [], 64)
        for i in range(3)
    ]
    queue = ingest_scheduler._FrontierUploadQueue(tasks)
    submitted = []
    two = Event()
    third = Event()
    closed = Event()

    def upload(*args, cufile_worker_state, **kwargs):
        future = Future()
        submitted.append(future)
        cufile_worker_state.task_completion = future
        if len(submitted) == 2:
            two.set()
        if len(submitted) == 3:
            third.set()

    monkeypatch.setattr(ingest_scheduler, "_upload_vdif_plan_group", upload)
    monkeypatch.setattr(
        ingest_scheduler,
        "_close_vdif_cufile_upload_state",
        lambda *a, **k: closed.set(),
    )
    with ThreadPoolExecutor(1) as pool:
        worker = pool.submit(
            ingest_scheduler._run_cufile_frontier_upload_worker,
            queue,
            xp,
            0,
            None,
            4096,
            None,
            None,
            64,
            64,
            None,
        )
        try:
            assert two.wait(2)
            assert not third.wait(0.05)
            assert not closed.is_set()
            if fail:
                submitted[0].set_exception(ValueError("submission failed"))
                assert not closed.wait(0.05)
                submitted[1].set_result(None)
                with pytest.raises(ValueError, match="submission failed"):
                    worker.result(timeout=2)
                assert not third.is_set()
            else:
                submitted[0].set_result(None)
                assert third.wait(2)
                submitted[1].set_result(None)
                assert not closed.is_set()
                submitted[2].set_result(None)
                worker.result(timeout=2)
            assert closed.is_set()
        finally:
            queue.abort(RuntimeError("test cleanup"))
            for future in submitted:
                if not future.done():
                    future.set_result(None)


@pytest.mark.parametrize("short", [False, True])
def test_read_replay_matches_production_strip_requests(monkeypatch, short):
    class Raw:
        data = SimpleNamespace(ptr=65536)

        def __getitem__(self, key):
            return self

    event = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(
        upload_resources, "_prepare_cufile_upload_state", lambda *a: None
    )
    monkeypatch.setattr(
        upload_resources,
        "_cuda_launch_policy",
        lambda *a: SimpleNamespace(ingest_threads=256),
    )
    monkeypatch.setattr(upload_resources, "_vdif_ingest_kernel", lambda: None)
    monkeypatch.setattr(
        upload_resources,
        "_lease_validated_cufile_source",
        lambda *a: nullcontext("file"),
    )
    monkeypatch.setattr(
        upload_resources, "_copy_cufile_mapping_to_device", lambda *a: event
    )
    monkeypatch.setattr(
        upload_resources, "_submit_cufile_strip", lambda *a: event
    )
    monkeypatch.setattr(
        upload_resources, "vdif_upload_staging_slots", lambda _: 2
    )
    xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0)),
        int32=np.int32,
    )
    requests = []

    def read(file, pointer, size, offset):
        requests.append((size, offset))
        # EOF can truncate alignment padding, but not actual payload bytes.
        return 519 if short else 520

    descriptors = []
    for read_only in (False, True):
        requests.clear()
        state = upload_resources._VdifCuFileUploadState(
            0,
            workspace=SimpleNamespace(
                session=SimpleNamespace(read=read), buffers=[Raw(), Raw()]
            ),
            stream=object(),
            pinned_mappings=[np.empty(32, np.uint8) for _ in range(2)],
            device_mappings=[np.empty(32, np.uint8) for _ in range(2)],
            slot_events=[None, None],
            mapping_copy_events=[None, None],
            slot_submissions=[None, None],
            read_only=read_only,
        )
        stats = upload_resources.CuFileReadStatistics(capture_reads=True)
        arguments = dict(
            plans=[object()],
            outputs=[object()],
            reference_files=(("file", 5, 8, 520, None, None, None),),
            validated_mappings=[(np.arange(8, dtype=np.int32),)],
            frame_size=40,
            payload_bytes=8,
            chunk_frames=3,
            ready_event=None,
            progresses=None,
            xp=xp,
            worker_state=state,
            statistics=stats,
            trusted_mapped_plans=True,
        )
        if short:
            with pytest.raises(OSError, match="expected at least 520"):
                upload_resources._load_vdif_payload_plans_cufile(**arguments)
        else:
            upload_resources._load_vdif_payload_plans_cufile(**arguments)
        descriptors.append(stats.read_descriptors)
        assert requests == [(65536, 0)] * 3
    assert descriptors[0] == descriptors[1]
    assert [item[3] for item in descriptors[0]] == [120, 120, 80]


def test_exact_cache_reservation_preserves_matching_workspaces(monkeypatch):
    def workspace(capacity, slots):
        return SimpleNamespace(
            capacity=capacity,
            buffers=[None] * slots,
            retained_bytes=capacity * slots,
        )

    exact = workspace(64, 2)
    oversized = workspace(128, 2)
    wrong_slots = workspace(64, 4)
    other_gpu = workspace(64, 2)
    monkeypatch.setattr(
        upload_resources,
        "_cufile_workspaces",
        {0: [exact, oversized, wrong_slots], 1: [other_gpu]},
    )
    monkeypatch.setattr(
        upload_resources, "_cached_cufile_workspace_bytes", 768
    )
    closed = []
    monkeypatch.setattr(
        upload_resources,
        "_close_cufile_workspace",
        lambda w, **kw: closed.append(w),
    )
    assert (
        upload_resources._retain_exact_cufile_workspaces(0, 64, 2, 1) == 128
    )
    assert closed == [oversized, wrong_slots]
    assert upload_resources._cached_cufile_workspace_bytes == 256
    assert upload_resources._cufile_workspaces == {0: [exact], 1: [other_gpu]}


@pytest.fixture
def simulated_storage(monkeypatch):
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
        cupy.cuda.Device(0).use()
    except Exception as exc:
        pytest.skip(str(exc))

    class Storage:
        def __init__(self):
            self.files = {}

        def __enter__(self):
            return self

        def register_buffer(self, pointer, size):
            pass

        @contextmanager
        def lease_file(self, path, *, fingerprint=None, validate=None):
            if path not in self.files:
                self.files[path] = SimpleNamespace(
                    fd=os.open(path, os.O_RDONLY)
                )
            if validate:
                validate(self.files[path])
            yield self.files[path]

        def finish_cached_reads(self):
            pass

        def read(self, file, pointer, size, offset):
            data = np.frombuffer(os.pread(file.fd, size, offset), np.uint8)
            cupy.cuda.runtime.memcpy(
                pointer,
                data.ctypes.data,
                data.size,
                cupy.cuda.runtime.memcpyHostToDevice,
            )
            return data.size

        def close(self):
            for file in self.files.values():
                os.close(file.fd)
            self.files.clear()

    upload_resources.clear_cached_cufile_workspaces(0)
    monkeypatch.setattr(upload_resources, "CuFileSession", Storage)
    yield cupy
    upload_resources.clear_cached_cufile_workspaces(0)


@pytest.mark.gpu
def test_posix_generator_close_stops_frontier_uploads_before_pool_join(
    tmp_path, monkeypatch, simulated_storage
):
    # This test replaces synchronous compute/materialization with sentinels.
    monkeypatch.setenv("CUPHOTON_XBEAM_OUTPUT_PIPELINE", "0")
    from dataclasses import replace
    from decimal import Decimal

    from tests.xbeam.test_gpu_backend import _packed_two_station_source

    job, source = _packed_two_station_source(tmp_path)
    job = replace(
        job,
        integration_duration_s=Decimal(2),
        subintegration_duration_s=Decimal("0.5"),
        frequencies=(replace(job.frequencies[0], channel_count=16),),
    )
    started = Event()
    release = Event()
    uploaded = []
    producers = []
    record = object()
    real_executor = engine.ThreadPoolExecutor

    class UploadPool(real_executor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            producers.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures):
            # The engine must signal cancellation before joining its pool.
            # Releasing here also lets a broken implementation finish, then
            # fail the task-count assertion instead of hanging the test.
            release.set()
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    def executor(*args, **kwargs):
        pool = (
            UploadPool
            if kwargs.get("thread_name_prefix", "").endswith("vdif-upload")
            else real_executor
        )
        return pool(*args, **kwargs)

    def upload(*args, **kwargs):
        uploaded.append(args[0])
        started.set()
        assert release.wait(timeout=10), "upload pool did not shut down"

    def compute(*args, **kwargs):
        assert started.wait(timeout=10), "POSIX producer did not start"
        return []

    monkeypatch.setattr(engine, "ThreadPoolExecutor", executor)
    monkeypatch.setattr(ingest_scheduler, "_upload_vdif_plan_group", upload)
    monkeypatch.setattr(engine, "_run_chunked_gpu", compute)
    monkeypatch.setattr(
        engine, "_flush_gpu_records", lambda *args: iter([record])
    )
    run = engine.prepare_correlation(
        job,
        backend="cupy",
        recording_source=source,
        vdif_io="posix",
        vdif_upload_workers=1,
        subintegration_tiling="off",
    )
    records = run.records()
    try:
        assert next(records) is record
        upload_info = run.ingest_manifest["vdif_upload"]
        assert upload_info["frontier_scheduling"]
        assert upload_info["frontier_tasks"] > 1
        assert len(uploaded) == 1
    finally:
        records.close()
        release.set()
    assert len(uploaded) == 1
    assert len(producers) == 1
    with pytest.raises(RuntimeError, match="stopped before VDIF upload"):
        producers[0].result(timeout=10)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "fallback",
    [None, "allocation", "early-exit", "budget", "retry", "producer"],
)
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("unordered", [False, True])
def test_output_pipeline_correlation_matches_sync(
    tmp_path, monkeypatch, simulated_storage, fallback, tiled, unordered
):
    from dataclasses import replace
    from decimal import Decimal

    from tests.xbeam.test_gpu_backend import _packed_two_station_source

    base, source = _packed_two_station_source(tmp_path)
    if unordered:
        monkeypatch.setenv("CUPHOTON_XBEAM_CUFILE_FRONTIER_PLANNER", "gpu")
    job = replace(
        base,
        integration_duration_s=Decimal(2),
        subintegration_duration_s=Decimal("0.5"),
        frequencies=(replace(base.frequencies[0], channel_count=16),),
        write_autocorrelations=True,
    )
    monkeypatch.setenv("CUPHOTON_XBEAM_CUFILE_FRONTIER", "1")
    flag = "CUPHOTON_XBEAM_OUTPUT_PIPELINE"
    monkeypatch.setenv(flag, "0")
    options = dict(
        backend="cupy",
        recording_source=source,
        vdif_io="cufile",
        vdif_upload_workers=1,
    )
    if tiled:
        options["max_tile_subintegrations"] = 3
    if unordered:
        rng = np.random.default_rng(728)
        for datastream in job.datastreams:
            plan = next(source.batches(datastream)).samples
            for file in plan.files:
                rows = np.fromfile(file.path, np.uint8).reshape(-1, 40)
                rows[:, 32:] = rng.integers(
                    0, 256, (len(rows), 8), dtype=np.uint8
                )
                file.path.write_bytes(rows.tobytes())
    expected = list(engine.correlate(job, **options))
    if unordered:
        original_source = source
        changed = {}
        for datastream in job.datastreams:
            batch = next(original_source.batches(datastream))
            plan = batch.samples
            files = []
            for file in plan.files:
                order = np.arange(file.frame_count)
                order[1], order[2] = order[2], order[1]
                order[15], order[16] = order[16], order[15]
                rows = np.fromfile(file.path, np.uint8).reshape(-1, 40)
                file.path.write_bytes(rows[order].tobytes())
                files.append(
                    replace(
                        file, destination_slots=file.destination_slots[order]
                    )
                )
            changed[datastream.identifier] = replace(
                batch, samples=replace(plan, files=tuple(files))
            )

        class UnorderedSource:
            def batches(self, datastream):
                yield changed[datastream.identifier]

        options["recording_source"] = UnorderedSource()
        if fallback is None:
            monkeypatch.setenv("CUPHOTON_XBEAM_CUFILE_FRONTIER", "0")
            fallback_records = list(engine.correlate(job, **options))
            assert len(fallback_records) == len(expected)
            for reference, record in zip(
                expected, fallback_records, strict=True
            ):
                np.testing.assert_allclose(
                    record.spectrum, reference.spectrum
                )
                assert record.weight == reference.weight
            monkeypatch.setenv("CUPHOTON_XBEAM_CUFILE_FRONTIER", "1")
    monkeypatch.setenv(flag, "1")
    if fallback == "allocation":

        def fail(*args, **kwargs):
            raise MemoryError("injected pinned allocation failure")

        monkeypatch.setattr(engine, "OutputPipeline", fail)
    if fallback == "budget":
        original_limit = compute_plan._automatic_subintegration_tile_limit
        calls = []

        def limit(*args):
            calls.append(args)
            return original_limit(*args) if len(calls) == 1 else 1

        monkeypatch.setattr(
            engine, "_automatic_subintegration_tile_limit", limit
        )
    retried = []
    if fallback == "producer":

        def fail_read(*args):
            raise OSError("injected producer failure")

        monkeypatch.setattr(upload_resources.CuFileSession, "read", fail_read)
    if fallback == "retry" and tiled:
        original_compute = gpu_execution._run_chunked_gpu

        def compute(*args, **kwargs):
            result = original_compute(*args, **kwargs)
            if not retried:
                retried.append(True)
                raise simulated_storage.cuda.memory.OutOfMemoryError(1, 1, 1)
            return result

        monkeypatch.setattr(gpu_execution, "_run_chunked_gpu", compute)
    run = engine.prepare_correlation(job, **options)
    records = run.records()
    if fallback == "producer":
        with pytest.raises(OSError, match="injected producer failure"):
            list(records)
        simulated_storage.cuda.get_current_stream().synchronize()
        return
    if fallback == "early-exit":
        first = next(records)
        records.close()
        simulated_storage.cuda.get_current_stream().synchronize()
        np.testing.assert_allclose(first.spectrum, expected[0].spectrum)
        assert run.ingest_manifest["output_pipeline"]["effective"]
        return
    actual = list(records)
    info = run.ingest_manifest["output_pipeline"]
    if unordered:
        assert (
            run.ingest_manifest["vdif_upload"]["frontier_plan"][
                "gpu_planned_mappings"
            ]
            > 0
        )
    assert run.ingest_manifest["vdif_upload"]["frontier_plan"][
        "mapping_order"
    ] == ("unordered" if unordered else "ordered")
    assert info["effective"] is (fallback not in ("allocation", "budget"))
    assert (
        info["reason"]
        == {
            None: "enabled",
            "allocation": "allocation-failed",
            "budget": "insufficient-device-budget",
            "retry": "enabled",
        }[fallback]
    )
    assert info["layout"] == ("tiled" if tiled else "untiled")
    if info["effective"]:
        assert (
            info["device_bytes"]
            == info["accumulator_bytes"] + info["scratch_bytes"]
        )
        assert info["scratch_bytes"] == (
            info["accumulator_bytes"] if tiled else 0
        )
    if fallback == "retry" and tiled:
        assert (
            run.ingest_manifest["subintegration_tiling"]["oom_retries"] == 1
        )
    assert len(actual) == len(expected) == 12
    for got, want in zip(actual, expected, strict=True):
        assert replace(got, spectrum=None) == replace(want, spectrum=None)
        np.testing.assert_allclose(
            got.spectrum, want.spectrum, rtol=3e-5, atol=3e-5
        )


@pytest.mark.gpu
@pytest.mark.parametrize("slots", [2, 3, 4])
@pytest.mark.parametrize("unordered", [False, True])
def test_frontier_overlap_real_gpu_buffers_with_simulated_storage(
    tmp_path, monkeypatch, slots, simulated_storage, unordered
):
    """Exercise real CUDA buffer/event ownership without requiring GDS."""
    cupy = simulated_storage
    from cuphoton.xbeam.adapters.vdif.recording import (
        VdifPayloadFilePlan,
        VdifPayloadPlan,
    )

    monkeypatch.setenv("CUPHOTON_XBEAM_CUFILE_SLOTS", str(slots))
    plans, outputs, validity_outputs, progresses, expected = (
        [],
        {},
        {},
        {},
        {},
    )
    frame_count = 240 if unordered else 24
    for stream in range(3):
        path = tmp_path / f"stream-{stream}.vdif"
        frames = np.zeros((frame_count, 40), np.uint8)
        frames[:, 32:] = (
            np.arange(frame_count, dtype=np.uint8)[:, None] + stream * 24
        )
        order = np.arange(frame_count, dtype=np.int32)
        if unordered:
            order[39], order[40] = order[40], order[39]
            order[119], order[120] = order[120], order[119]
        path.write_bytes(frames[order].tobytes())
        plan = VdifPayloadPlan(
            40,
            8,
            frame_count,
            (VdifPayloadFilePlan(path, frame_count, order),),
        )
        plans.append([plan])
        outputs[id(plan)] = cupy.zeros(plan.byte_count, cupy.uint8)
        validity_outputs[id(plan)] = cupy.ones(frame_count, cupy.uint8)
        progresses[id(plan)] = ingest_scheduler._PayloadProgress(
            frame_count, static_validity=np.ones(frame_count, np.uint8)
        )
        expected[id(plan)] = frames[:, 32:].reshape(-1)
    ready = cupy.cuda.Event()
    ready.record()
    tasks, futures = ingest_scheduler._mapped_frontier_upload_tasks(
        plans,
        outputs,
        validity_outputs,
        progresses,
        [44] * 3,
        6,
        trust_slices=True,
        xp=cupy,
        planner="gpu" if unordered else "cpu",
    )
    workers, raw, mapping, device_bytes, _ = (
        ingest_scheduler._frontier_staging_plan(tasks, 2, 1 << 30)
    )
    statistics = upload_resources.CuFileReadStatistics(capture_reads=True)
    queue = ingest_scheduler._FrontierUploadQueue(tasks)
    try:
        with upload_resources.reserve_vdif_ingest(0) as token:
            with (
                ThreadPoolExecutor(1) as submitter,
                ThreadPoolExecutor(workers) as readers,
            ):
                producers = [
                    readers.submit(
                        ingest_scheduler._run_cufile_frontier_upload_worker,
                        queue,
                        cupy,
                        0,
                        ready,
                        device_bytes,
                        token,
                        statistics,
                        raw,
                        mapping,
                        submitter,
                    )
                    for _ in range(workers)
                ]
                for producer in producers:
                    producer.result(timeout=20)
        for identifier, output in outputs.items():
            futures[identifier].result(timeout=2)
            np.testing.assert_array_equal(
                cupy.asnumpy(output), expected[identifier]
            )
        if unordered:
            replay = upload_resources.CuFileReadStatistics(capture_reads=True)
            replay_tasks, _ = ingest_scheduler._mapped_frontier_upload_tasks(
                plans,
                outputs,
                validity_outputs,
                progresses,
                [44] * 3,
                6,
                trust_slices=True,
                xp=cupy,
                planner="gpu" if unordered else "cpu",
            )
            with upload_resources.reserve_vdif_ingest(0) as token:
                with ThreadPoolExecutor(1) as submitter:
                    ingest_scheduler._run_cufile_frontier_upload_worker(
                        ingest_scheduler._FrontierUploadQueue(replay_tasks),
                        cupy,
                        0,
                        ready,
                        device_bytes,
                        token,
                        replay,
                        raw,
                        mapping,
                        submitter,
                        read_only=True,
                    )
            assert sorted(replay.read_descriptors) == sorted(
                statistics.read_descriptors
            )
    finally:
        upload_resources.clear_cached_cufile_workspaces(0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("io_backend", "failure"),
    [
        ("cufile", "constructor"),
        ("cufile", "submit"),
        ("posix", "second-submit"),
    ],
)
def test_upload_startup_failure_closes_created_pools(
    tmp_path, monkeypatch, simulated_storage, io_backend, failure
):
    from dataclasses import replace
    from decimal import Decimal

    from tests.xbeam.test_gpu_backend import _packed_two_station_source

    job, source = _packed_two_station_source(tmp_path)
    job = replace(
        job,
        integration_duration_s=Decimal(2),
        subintegration_duration_s=Decimal("0.5"),
        frequencies=(replace(job.frequencies[0], channel_count=16),),
    )
    pools = []
    queues = []
    real_executor = engine.ThreadPoolExecutor
    real_queue = engine._FrontierUploadQueue

    class DeferredPool:
        def __init__(self, name):
            self.name = name
            self.futures = []
            self.closed = False
            pools.append(self)

        def submit(self, *args, **kwargs):
            if failure == "submit" or (
                failure == "second-submit" and self.futures
            ):
                raise RuntimeError("injected startup failure")
            future = Future()
            self.futures.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures):
            assert wait and cancel_futures
            self.closed = True
            for future in self.futures:
                future.cancel()

    def executor(*args, **kwargs):
        name = kwargs.get("thread_name_prefix", "")
        if name.endswith("vdif-plan"):
            return real_executor(*args, **kwargs)
        if failure == "constructor" and name.endswith("cuda-submit"):
            raise RuntimeError("injected startup failure")
        return DeferredPool(name)

    def queue(tasks):
        result = real_queue(tasks)
        queues.append(result)
        return result

    monkeypatch.setattr(engine, "ThreadPoolExecutor", executor)
    monkeypatch.setattr(engine, "_FrontierUploadQueue", queue)
    with pytest.raises(RuntimeError, match="injected startup failure"):
        list(
            engine.correlate(
                job,
                backend="cupy",
                recording_source=source,
                vdif_io=io_backend,
                vdif_upload_workers=1 if io_backend == "cufile" else 2,
            )
        )
    assert pools
    assert all(pool.closed for pool in pools)
    assert all(
        future.cancelled() for pool in pools for future in pool.futures
    )
    if io_backend == "cufile":
        assert queues
        assert all(queue.take() is None for queue in queues)
    else:
        assert sum(len(pool.futures) for pool in pools) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_posix_setup_failure_drains_owned_state_and_preserves_error(
    tmp_path, monkeypatch, simulated_storage, cleanup_fails
):
    from tests.xbeam.test_gpu_backend import _packed_two_station_source

    job, source = _packed_two_station_source(tmp_path)
    plan = next(source.batches(job.datastreams[0])).samples
    states = []
    closed = []

    def prepare(state, *args):
        states.append(state)
        raise MemoryError("injected POSIX setup failure")

    def close(state):
        closed.append(state)
        if cleanup_fails:
            raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(
        upload_resources, "_prepare_posix_upload_session", prepare
    )
    monkeypatch.setattr(
        upload_resources, "_close_vdif_posix_upload_state", close
    )
    with pytest.raises(MemoryError, match="injected POSIX setup failure"):
        upload_resources.load_vdif_payload_plans(
            [plan], xp=simulated_storage, io_backend="posix"
        )
    assert len(states) == 1
    assert closed == states
