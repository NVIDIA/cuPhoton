# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cuphoton.xbeam.cupy_kernels as cupy_kernels
from cuphoton.xbeam import (
    gpu_kernels,
    ingest_scheduler,
    runtime_tuning,
    runtime_types,
    upload_resources,
)
from cuphoton.xbeam.adapters.radio import (
    compute_plan,
    engine,
    gpu_execution,
    output_records,
    reference_execution,
)
from cuphoton.xbeam.adapters.radio.engine import correlate
from cuphoton.xbeam.adapters.radio.models import (
    AutocorrelationMode,
    Baseline,
    BaselineProduct,
    ClockPolynomial,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    MJDTime,
    PackedVoltageLayout,
    RecordedBand,
    SamplingKind,
    Scan,
    Sideband,
    Station,
    VoltageBatch,
)
from cuphoton.xbeam.adapters.vdif.ingest import (
    SampleInterval,
    plan_vdif_ingest,
)
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifFileRecordingSource,
    VdifPayloadFilePlan,
    VdifPayloadPlan,
    VdifStreamingPlan,
)
from cuphoton.xbeam.adapters.vdif.vdif import (
    DuplicatePolicy,
    VdifError,
    VdifHeader,
    absolute_frame_slot,
    decode_2bit,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.cufile_io import (
    CuFileSession,
    CuFileUnavailable,
)
from cuphoton.xbeam.cupy_kernels import (
    BLOCK_PARAMETERS_DTYPE,
    apply_residual_phase,
    automatic_vdif_upload_chunk_bytes,
    batched_visibility_product,
    batched_visibility_products,
    channelize_and_apply_residual,
    decode_2bit_band,
    front_rotate,
    front_rotate_packed_real_bands,
    load_vdif_payload_plans,
    prepare_block_parameters,
    prepare_visibility_index,
    vdif_upload_staging_bytes,
    visibility_product,
)
from cuphoton.xbeam.gpu_kernels import _validate_visibility_ranges
from cuphoton.xbeam.kernel_compilation import _launch_policy_for_compute_major
from cuphoton.xbeam.missing_timestamp_recovery import (
    GlobalFFTEligibilityPlan,
)
from cuphoton.xbeam.profiles import ANALYTIC_V1


@pytest.mark.parametrize(
    ("credits", "reported", "expected", "expected_reported"),
    [
        (None, None, [1.0, 0.5, 0.0], None),
        ([0, 0, 0], [0, 0, 0], [1.0, 0.5, 0.0], None),
        ([0.1, 0.25, 0.1], [0.1, 0.25, 0.1], [1.0, 0.75, 0.0], None),
        ([0.1, 0.25, 0.1], [0, 0, 0], [1.0, 0.75, 0.0], [1.0, 0.5, 0.0]),
        (
            [0.1, 0.25, 0.1],
            [0.1, 0.1, 0.1],
            [1.0, 0.75, 0.0],
            [1.0, 0.6, 0.0],
        ),
    ],
)
def test_station_weight_credits_preserve_zero_and_saturated_weights(
    credits,
    reported,
    expected,
    expected_reported,
):
    raw = np.array([1.0, 0.5, 0.0])
    corrected, reported_weight = gpu_execution._apply_station_weight_credits(
        raw,
        None if credits is None else np.asarray(credits),
        None if reported is None else np.asarray(reported),
        np,
    )
    np.testing.assert_array_equal(corrected, expected)
    np.testing.assert_array_equal(raw, [1.0, 0.5, 0.0])
    if expected_reported is None:
        assert reported_weight is None
    else:
        np.testing.assert_array_equal(reported_weight, expected_reported)
    if credits is None or not any(credits):
        assert corrected is raw
    if reported is not None and not any(reported) and any(credits or ()):
        assert reported_weight is raw


@pytest.mark.parametrize("repeat_output", [False, True])
def test_shared_reported_weights_preserve_each_output_and_boundary(
    repeat_output,
):
    ordinary = np.array([1.0, 0.5, 0.75, 1.0])
    reported = np.array([0.75, 0.5, 1.0, 0.5])
    other = np.array([0.5, 1.0, 0.5, 1.0])
    products = []
    for offsets, boundary, multiply in (
        ((0, 1), (0, 0), False),
        ((2, 3), (0, 0), False),
        ((4, 5), (1, 0), False),
        ((6, 7), (0, 0), True),
    ):
        products.append(
            SimpleNamespace(
                blocks_a=SimpleNamespace(weight=ordinary),
                blocks_b=SimpleNamespace(weight=other),
                reported_weight_a=reported,
                reported_weight_b=other,
                multiply_weights=multiply,
                boundary=boundary,
                weight_offsets=offsets,
            )
        )
    arena = SimpleNamespace(weight_adjustments=np.ones(8))
    if repeat_output:
        products.append(products[0])
    gpu_execution._accumulate_reported_weight_adjustments(
        {("shared",): products},
        {
            (0, 0): (np.array([0, 2]), np.array([2, 2])),
            (1, 0): (np.array([0, 1]), np.array([1, 3])),
        },
        arena,
        np,
    )
    expected = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.875, 0.625]
    if repeat_output:
        expected[:2] = [0.5, 0.5]
    np.testing.assert_array_equal(arena.weight_adjustments, expected)


def test_cached_upload_staging_reports_device_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = 17
    workspaces = [
        type("Workspace", (), {"capacity": capacity})()
        for capacity in (32, 64)
    ]
    monkeypatch.setitem(
        upload_resources._upload_workspaces, device, workspaces
    )

    assert upload_resources.cached_vdif_upload_staging_bytes(device) == 192
    assert upload_resources.cached_vdif_upload_staging_covers(
        device, [120, 60, 20], 2
    )
    assert not upload_resources.cached_vdif_upload_staging_covers(
        device, [130, 60, 20], 2
    )


def test_global_cufile_read_token_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = []
    writes = []
    monkeypatch.setenv(upload_resources._CUFILE_READ_TOKEN_FD, "17")
    monkeypatch.setenv(upload_resources._CUFILE_RETURN_TOKEN_FD, "19")
    monkeypatch.setattr(
        upload_resources.os,
        "read",
        lambda descriptor, size: reads.append((descriptor, size)) or b"x",
    )
    monkeypatch.setattr(
        upload_resources.os,
        "write",
        lambda descriptor, value: (
            writes.append((descriptor, value)) or len(value)
        ),
    )

    with upload_resources._global_cufile_read_token():
        assert reads == [(17, 1)]
        assert writes == []

    assert writes == [(19, b"x")]


def test_cached_upload_workspace_eviction_is_device_scoped(
    monkeypatch,
) -> None:
    first = SimpleNamespace(retained_bytes=40)
    second = SimpleNamespace(retained_bytes=80)
    monkeypatch.setattr(
        upload_resources, "_upload_workspaces", {1: [first], 2: [second]}
    )
    monkeypatch.setattr(
        upload_resources, "_cached_upload_workspace_bytes", 120
    )

    upload_resources.clear_cached_vdif_upload_workspaces(1)

    assert upload_resources._upload_workspaces == {2: [second]}
    assert upload_resources._cached_upload_workspace_bytes == 80


def test_cached_cufile_staging_reports_device_capacity(monkeypatch) -> None:
    device = 17
    workspaces = [
        SimpleNamespace(
            retained_bytes=capacity,
            buffers=tuple(object() for _ in range(4)),
        )
        for capacity in (32, 64)
    ]
    monkeypatch.setitem(
        upload_resources._cufile_workspaces, device, workspaces
    )

    assert upload_resources.cached_cufile_staging_bytes(device) == 96
    assert upload_resources.cached_cufile_staging_covers(
        device, [60, 30, 20], 2
    )
    assert not upload_resources.cached_cufile_staging_covers(
        device, [70, 30, 20], 2
    )


def test_cached_cufile_workspace_eviction_is_device_scoped(
    monkeypatch,
) -> None:
    first = SimpleNamespace(retained_bytes=40)
    second = SimpleNamespace(retained_bytes=80)
    closed = []
    monkeypatch.setattr(
        upload_resources, "_cufile_workspaces", {1: [first], 2: [second]}
    )
    monkeypatch.setattr(
        upload_resources, "_cached_cufile_workspace_bytes", 120
    )
    monkeypatch.setattr(
        upload_resources, "_close_cufile_workspace", closed.append
    )

    upload_resources.clear_cached_cufile_workspaces(1)

    assert closed == [first]
    assert upload_resources._cufile_workspaces == {2: [second]}
    assert upload_resources._cached_cufile_workspace_bytes == 80


def test_cufile_read_statistics_report_concurrent_span() -> None:
    statistics = upload_resources.CuFileReadStatistics()

    statistics.record(100, 300, 40, 64)
    statistics.record(200, 500, 80, 128)
    snapshot = statistics.snapshot()

    assert snapshot.logical_bytes == 120
    assert snapshot.requested_bytes == 192
    assert snapshot.calls == 2
    assert snapshot.span_seconds == 400e-9
    assert snapshot.active_read_seconds == 500e-9
    assert snapshot.mean_read_concurrency == pytest.approx(1.25)
    assert snapshot.logical_gb_per_second == pytest.approx(0.3)


def test_cufile_staging_slot_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(upload_resources._CUFILE_SLOTS_ENV, raising=False)
    assert runtime_tuning.vdif_upload_staging_slots("cufile") == 4
    assert runtime_tuning.vdif_upload_staging_slots("posix") == 2

    monkeypatch.setenv(upload_resources._CUFILE_SLOTS_ENV, "3")
    assert runtime_tuning.vdif_upload_staging_slots("cufile") == 3
    with pytest.raises(ValueError, match="requires the cuFile backend"):
        runtime_tuning.vdif_upload_staging_slots("posix")

    monkeypatch.setenv(upload_resources._CUFILE_SLOTS_ENV, "5")
    with pytest.raises(ValueError, match="must be 2, 3, or 4"):
        runtime_tuning.vdif_upload_staging_slots("cufile")


def test_cufile_staging_footprint_tracks_slot_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = VdifPayloadPlan(
        40,
        8,
        1,
        (
            VdifPayloadFilePlan(
                Path("source.vdif"),
                2,
                np.asarray([0, -1], dtype=np.int32),
            ),
        ),
    )
    monkeypatch.setenv(upload_resources._CUFILE_SLOTS_ENV, "3")
    alignment = 64 * 1024
    registered = 4 * 1024 * 1024

    assert (
        upload_resources.vdif_upload_staging_bytes(
            [plan], target_chunk_bytes=88, io_backend="cufile"
        )
        == 3 * (registered + alignment - 1) + 3 * 2 * 4
    )
    assert upload_resources.reusable_vdif_upload_staging_bytes(
        [plan], target_chunk_bytes=88, io_backend="cufile"
    ) == 3 * (registered + alignment - 1)


def test_cufile_frontier_workspace_requirements_cover_raw_and_mappings() -> (
    None
):
    plan = SimpleNamespace(
        frame_size_bytes=40,
        files=(SimpleNamespace(frame_count=7),),
    )

    raw_bytes, mapping_bytes = (
        upload_resources._cufile_upload_state_requirements(
            [plan, plan], target_chunk_bytes=96
        )
    )

    assert raw_bytes == upload_resources._PINNED_WORKSPACE_GRANULARITY
    assert mapping_bytes == 16
    _, full_mapping_bytes = (
        upload_resources._cufile_upload_state_requirements(
            [plan, plan], target_chunk_bytes=96, full_mappings=True
        )
    )
    assert full_mapping_bytes == 56


def test_cufile_worker_cleanup_drains_and_releases_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    stream = SimpleNamespace(synchronize=lambda: events.append("strip-sync"))
    mapping_stream = SimpleNamespace(
        synchronize=lambda: events.append("mapping-sync")
    )
    workspace = SimpleNamespace(
        session=SimpleNamespace(finish_cached_reads=lambda: None)
    )
    state = upload_resources._VdifCuFileUploadState(
        0,
        workspace=workspace,
        pinned_mappings=(object(), object()),
        device_mappings=(object(), object()),
        mapping_capacity=64,
        stream=stream,
        mapping_stream=mapping_stream,
        slot_events=[object(), object()],
        chunk_index=7,
    )
    monkeypatch.setattr(
        upload_resources,
        "_release_cufile_workspace",
        lambda selected, **options: events.append((selected, options)),
    )

    upload_resources._close_vdif_cufile_upload_state(state)

    assert events == [
        "strip-sync",
        "mapping-sync",
        (workspace, {"reusable": True, "best_effort": False}),
    ]
    assert state.workspace is None
    assert state.stream is None
    assert state.mapping_stream is None
    assert state.chunk_index == 0
    assert state.slot_submissions is None


def test_cufile_strip_submission_records_event_and_publishes() -> None:
    calls = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    event = SimpleNamespace(
        record=lambda stream: calls.append(("event", stream))
    )
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda identifier: Context(),
            Event=lambda **options: event,
        )
    )
    stream = Context()

    def kernel(grid, block, arguments):
        calls.append(("kernel", grid, block, arguments))

    progress = SimpleNamespace(
        publish=lambda destinations, completed: calls.append(
            ("publish", destinations, completed)
        )
    )
    source = object()
    mapping = object()
    output = object()
    destinations = np.asarray([0, 1], dtype=np.int32)

    completed = upload_resources._submit_cufile_strip(
        xp,
        0,
        stream,
        kernel,
        source,
        [mapping],
        [output],
        2,
        5,
        1,
        32,
        [progress],
        [destinations],
    )

    assert completed is event
    assert calls[0][0:3] == ("kernel", (1,), (32,))
    assert calls[0][3] == (
        source,
        mapping,
        np.int32(2),
        np.int32(5),
        np.int32(1),
        output,
    )
    assert calls[1] == ("event", stream)
    assert calls[2][0] == "publish"
    np.testing.assert_array_equal(calls[2][1], destinations)
    assert calls[2][2] is event


def test_cufile_strip_waits_for_mapping_copy_before_launch() -> None:
    calls = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait_event(self, event):
            calls.append(("wait", event))

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    completed = SimpleNamespace(
        record=lambda stream: calls.append(("record", stream))
    )
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda identifier: Context(),
            Event=lambda **options: completed,
        )
    )
    stream = Stream()
    mapping_ready = SimpleNamespace(
        synchronize=lambda: pytest.fail("mapping copy synchronized on host")
    )

    upload_resources._submit_cufile_strip(
        xp,
        0,
        stream,
        lambda *args: calls.append(("kernel", args)),
        object(),
        [object()],
        [object()],
        1,
        5,
        1,
        32,
        None,
        [np.asarray([0], dtype=np.int32)],
        mapping_ready=mapping_ready,
    )

    assert calls[0] == ("wait", mapping_ready)
    assert calls[1][0] == "kernel"
    assert calls[2] == ("record", stream)


def test_cufile_strip_falls_back_to_host_wait_without_stream_events() -> None:
    calls = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    stream = Context()
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda identifier: Context(),
            Event=lambda **options: SimpleNamespace(
                record=lambda selected: calls.append("record")
            ),
        )
    )
    mapping_ready = SimpleNamespace(
        synchronize=lambda: calls.append("host-wait")
    )

    upload_resources._submit_cufile_strip(
        xp,
        0,
        stream,
        lambda *args: calls.append("kernel"),
        object(),
        [object()],
        [object()],
        1,
        5,
        1,
        32,
        None,
        [np.asarray([0], dtype=np.int32)],
        mapping_ready=mapping_ready,
    )

    assert calls == ["host-wait", "kernel", "record"]


def test_cufile_strip_skips_empty_sparse_mapping() -> None:
    calls = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda identifier: Context(),
            Event=lambda **options: SimpleNamespace(
                record=lambda stream: None
            ),
        )
    )

    upload_resources._submit_cufile_strip(
        xp,
        0,
        Context(),
        lambda *args: calls.append("dense"),
        object(),
        [np.empty(0, dtype=np.int32)],
        [object()],
        12,
        5,
        1,
        32,
        None,
        [np.empty(0, dtype=np.int32)],
        [np.empty(0, dtype=np.int32)],
        lambda *args: calls.append("sparse"),
    )

    assert calls == []


def test_cufile_sparse_strip_passes_source_and_destination_bounds() -> None:
    calls = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda identifier: Context(),
            Event=lambda **options: SimpleNamespace(
                record=lambda stream: None
            ),
        )
    )
    sources = np.asarray([11, 0], dtype=np.int32)
    destinations = np.asarray([3, 1], dtype=np.int32)

    upload_resources._submit_cufile_strip(
        xp,
        0,
        Context(),
        lambda *args: calls.append("dense"),
        object(),
        [destinations],
        [np.empty(32, dtype=np.uint8)],
        12,
        5,
        1,
        32,
        None,
        [destinations],
        [sources],
        lambda grid, block, args: calls.append((grid, block, args)),
    )

    assert len(calls) == 1
    grid, block, arguments = calls[0]
    assert grid == (1,)
    assert block == (32,)
    assert arguments[1] is sources
    assert arguments[2] is destinations
    assert arguments[3:8] == (
        np.int32(2),
        np.int32(12),
        np.int32(4),
        np.int32(5),
        np.int32(1),
    )


def test_cufile_sparse_chunk_passes_copy_event_without_host_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Raw:
        data = SimpleNamespace(ptr=65536)

        def __getitem__(self, key):
            return self

    mapping_stream = object()
    mapping_ready = SimpleNamespace(
        synchronize=lambda: pytest.fail("mapping copy synchronized on host")
    )
    copied = []
    submitted = []
    completed = object()
    monkeypatch.setattr(
        upload_resources, "_prepare_cufile_upload_state", lambda *args: None
    )
    monkeypatch.setattr(
        upload_resources,
        "_cuda_launch_policy",
        lambda *args: SimpleNamespace(ingest_threads=32),
    )
    monkeypatch.setattr(upload_resources, "_vdif_ingest_kernel", object)
    monkeypatch.setattr(
        upload_resources, "_sparse_vdif_ingest_kernel", object
    )
    monkeypatch.setattr(
        upload_resources,
        "_lease_validated_cufile_source",
        lambda *args: nullcontext("file"),
    )
    monkeypatch.setattr(
        upload_resources,
        "_copy_cufile_mapping_to_device",
        lambda *args: copied.append(args) or mapping_ready,
    )
    monkeypatch.setattr(
        upload_resources,
        "_submit_cufile_strip",
        lambda *args: submitted.append(args) or completed,
    )
    monkeypatch.setattr(
        upload_resources, "vdif_upload_staging_slots", lambda backend: 2
    )
    state = upload_resources._VdifCuFileUploadState(
        0,
        workspace=SimpleNamespace(
            session=SimpleNamespace(read=lambda *args: 65536),
            buffers=[Raw(), Raw()],
        ),
        pinned_mappings=[np.empty(64, np.uint8) for _ in range(2)],
        device_mappings=[np.empty(64, np.uint8) for _ in range(2)],
        mapping_capacity=64,
        stream=object(),
        mapping_stream=mapping_stream,
        slot_events=[None, None],
        mapping_copy_events=[None, None],
        slot_submissions=[None, None],
    )
    xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda *args: SimpleNamespace(id=0)),
        int32=np.int32,
    )

    upload_resources._load_vdif_payload_plans_cufile(
        plans=[object()],
        outputs=[object()],
        reference_files=(("file", 0, 4, None, None, None, None),),
        validated_mappings=[(np.arange(4, dtype=np.int32),)],
        validated_sources=[(np.asarray([3, 1, 3, 0], dtype=np.int64),)],
        frame_size=40,
        payload_bytes=8,
        chunk_frames=4,
        ready_event=None,
        progresses=None,
        xp=xp,
        worker_state=state,
    )

    assert len(copied) == 1
    assert copied[0][2] is mapping_stream
    assert len(submitted) == 1
    assert submitted[0][-1] is mapping_ready
    assert state.slot_events[0] is completed


def test_cufile_worker_validates_each_source_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    registered_file = SimpleNamespace(fd=17)
    opened = []

    @contextmanager
    def lease(path, *, fingerprint, validate):
        opened.append(path)
        validate(registered_file)
        yield registered_file

    session = SimpleNamespace(lease_file=lease)
    path = tmp_path / "source.vdif"
    path.write_bytes(bytes(100))
    source_stat = path.stat()
    stat_calls = []
    monkeypatch.setattr(
        upload_resources.os,
        "fstat",
        lambda fd: stat_calls.append(fd) or source_stat,
    )
    reference = (
        str(path),
        0,
        10,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_dev,
        source_stat.st_ino,
    )
    for _ in range(2):
        with upload_resources._lease_validated_cufile_source(
            session, reference
        ) as file:
            assert file is registered_file
    assert opened == [str(path), str(path)]
    assert stat_calls == [17, 17]


def test_cufile_worker_rejects_conflicting_source_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    registered_file = SimpleNamespace(fd=17)

    @contextmanager
    def lease(path, *, fingerprint, validate):
        validate(registered_file)
        yield registered_file

    session = SimpleNamespace(lease_file=lease)
    path = tmp_path / "source.vdif"
    path.write_bytes(bytes(100))
    source_stat = path.stat()
    monkeypatch.setattr(upload_resources.os, "fstat", lambda fd: source_stat)
    with pytest.raises(OSError, match="source changed"):
        with upload_resources._lease_validated_cufile_source(
            session,
            (
                str(path),
                1,
                9,
                source_stat.st_size + 1,
                source_stat.st_mtime_ns,
                source_stat.st_dev,
                source_stat.st_ino,
            ),
        ):
            pytest.fail("changed source accepted")


def test_cufile_worker_final_source_validation_disables_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_validation():
        raise OSError("source changed")

    workspace = SimpleNamespace(
        session=SimpleNamespace(finish_cached_reads=fail_validation)
    )
    stream = SimpleNamespace(synchronize=lambda: None)
    state = upload_resources._VdifCuFileUploadState(
        0,
        workspace=workspace,
        stream=stream,
    )
    changed_stat = SimpleNamespace(
        st_size=101,
        st_mtime_ns=200,
        st_dev=300,
        st_ino=400,
    )
    releases = []
    monkeypatch.setattr(upload_resources.os, "fstat", lambda fd: changed_stat)
    monkeypatch.setattr(
        upload_resources,
        "_release_cufile_workspace",
        lambda selected, **options: releases.append((selected, options)),
    )

    with pytest.raises(OSError, match="source changed"):
        upload_resources._close_vdif_cufile_upload_state(state)

    assert releases == [(workspace, {"reusable": False, "best_effort": True})]
    assert state.workspace is None


def test_upload_workspace_cleanup_drains_streams_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    failure = RuntimeError("copy synchronization failed")

    class Stream:
        def __init__(self, name: str, error: BaseException | None = None):
            self.name = name
            self.error = error

        def synchronize(self) -> None:
            events.append(self.name)
            if self.error is not None:
                raise self.error

    workspace = object()
    monkeypatch.setattr(
        upload_resources,
        "_release_upload_workspace",
        lambda actual: events.append(("release", actual)),
    )

    with pytest.raises(RuntimeError, match="copy synchronization failed"):
        upload_resources._drain_and_release_upload_workspace(
            workspace,
            Stream("copy", failure),
            Stream("strip"),
        )

    assert events == ["copy", "strip", ("release", workspace)]


def test_posix_workspace_growth_failure_releases_its_lease_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stream:
        calls = 0

        def synchronize(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("copy synchronization failed")

    stream = Stream()
    workspace = upload_resources.PinnedWorkspace(
        0, 8, (None, None), (None, None), stream
    )
    spare = upload_resources.PinnedWorkspace(
        0, 16, (None, None), (None, None), stream
    )
    monkeypatch.setattr(upload_resources, "_upload_workspaces", {0: [spare]})
    monkeypatch.setattr(
        upload_resources,
        "_cached_upload_workspace_bytes",
        spare.retained_bytes,
    )
    session = upload_resources._VdifPosixUploadState(
        0, workspace=workspace, strip_events=[object(), None], chunk_index=3
    )
    xp = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0))
    )

    with pytest.raises(RuntimeError, match="copy synchronization failed"):
        try:
            upload_resources._prepare_posix_upload_session(
                session, xp, 16, None
            )
        finally:
            upload_resources._close_vdif_posix_upload_state(session)

    cached = tuple(upload_resources._upload_workspaces[0])
    cached_bytes = upload_resources._cached_upload_workspace_bytes
    first = upload_resources._acquire_upload_workspace(xp, 0, 8)
    second = upload_resources._acquire_upload_workspace(xp, 0, 8)
    assert first is workspace
    assert second is not first
    assert sum(item is workspace for item in cached) == 1
    assert cached_bytes == workspace.retained_bytes + spare.retained_bytes
    assert session.workspace is None
    assert session.strip_events is None
    assert session.chunk_index == 0


def test_vdif_ingest_reservation_blocks_independent_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = 71
    wait_entered = threading.Event()
    acquired = threading.Event()
    original_wait = upload_resources._vdif_ingest_condition.wait

    def tracked_wait(*args, **kwargs):
        wait_entered.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(
        upload_resources._vdif_ingest_condition, "wait", tracked_wait
    )

    def independent_loader() -> None:
        with upload_resources._access_vdif_ingest(device, None):
            acquired.set()

    with upload_resources.reserve_vdif_ingest(device) as token:
        with upload_resources._access_vdif_ingest(device, token):
            thread = threading.Thread(target=independent_loader)
            thread.start()
            assert wait_entered.wait(timeout=1)
            assert not acquired.wait(timeout=0.05)

    assert acquired.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_hopper_launch_policy_expands_only_bandwidth_ingest() -> None:
    hopper = _launch_policy_for_compute_major(9)
    ampere = _launch_policy_for_compute_major(8)

    assert hopper.ingest_threads == 512
    assert hopper.front_threads == hopper.visibility_threads == 256
    assert ampere.ingest_threads == 256


def test_automatic_upload_chunk_respects_memory_and_concurrency() -> None:
    integration_bytes = 32 * 1024 * 1024

    assert automatic_vdif_upload_chunk_bytes(4 << 20, 80 << 30, 2) == 32 << 20
    assert (
        automatic_vdif_upload_chunk_bytes(integration_bytes, 80 << 30, 2)
        == 2 * integration_bytes
    )
    assert automatic_vdif_upload_chunk_bytes(
        integration_bytes, 8 << 30, 32
    ) == ((8 << 30) // 10) // (2 * 32)
    large_integration_bytes = 4096 * (8224 + 4)
    assert (
        automatic_vdif_upload_chunk_bytes(
            large_integration_bytes, 80 << 30, 2
        )
        == large_integration_bytes
    )
    assert (
        automatic_vdif_upload_chunk_bytes(80 << 20, 80 << 30, 2) == 64 << 20
    )
    assert (
        automatic_vdif_upload_chunk_bytes(
            large_integration_bytes,
            80 << 30,
            2,
            direct_storage=True,
        )
        == large_integration_bytes
    )
    assert (
        automatic_vdif_upload_chunk_bytes(
            large_integration_bytes,
            80 << 30,
            2,
            direct_storage=True,
            staging_slots=2,
            slot_limit_bytes=160 << 20,
        )
        == 4 * large_integration_bytes
    )
    assert (
        automatic_vdif_upload_chunk_bytes(
            large_integration_bytes,
            80 << 30,
            2,
            direct_storage=True,
            staging_slots=2,
            slot_limit_bytes=320 << 20,
        )
        == 8 * large_integration_bytes
    )


def _cupy():
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
        return cupy
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")


def _four_station_job_and_source():
    epoch = MJDTime(58849, Decimal(0))
    stations = tuple(
        Station(index, f"S{index}", ClockPolynomial(epoch, (0.0,)))
        for index in range(4)
    )
    frequency = Frequency(0, 0.0, 128.0, Sideband.USB, 128)
    datastreams = tuple(
        Datastream(
            index,
            index,
            SamplingKind.REAL,
            2,
            40,
            ("unused.vdif",),
            (RecordedBand(0, 0, "R"),),
            sample_rate_hz=256.0,
        )
        for index in range(4)
    )
    baselines = []
    for first in range(4):
        for second in range(first + 1, 4):
            baselines.append(
                Baseline(
                    len(baselines),
                    first,
                    second,
                    (BaselineProduct(0, 0, 0),),
                )
            )
    models = tuple(
        DelayCoordinateModel(
            index, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,)
        )
        for index in range(4)
    )
    job = Job(
        start_time=epoch,
        execution_duration_s=Decimal(8),
        integration_duration_s=Decimal(8),
        subintegration_duration_s=Decimal(8),
        fringe_approximation_order=1,
        stations=stations,
        frequencies=(frequency,),
        datastreams=datastreams,
        baselines=tuple(baselines),
        scans=(Scan("scan", Decimal(0), Decimal(8), "0", "0", models),),
    )

    class Source:
        def batches(self, datastream):
            samples = np.full(
                2048, datastream.identifier + 1, dtype=np.float32
            )
            yield VoltageBatch(
                samples,
                epoch,
                256.0,
                0,
                "R",
                np.ones(samples.size, dtype=np.bool_),
            )

    return job, Source()


def _packed_two_station_source(tmp_path: Path):
    epoch = MJDTime(58849, Decimal(0))
    stations = tuple(
        Station(index, f"S{index}", ClockPolynomial(epoch, (0.0,)))
        for index in range(2)
    )
    frequency = Frequency(0, 0.0, 128.0, Sideband.USB, 128)
    datastreams = tuple(
        Datastream(
            index,
            index,
            SamplingKind.REAL,
            2,
            40,
            (str(tmp_path / f"station-{index}.vdif"),),
            (RecordedBand(0, 0, "R"),),
            sample_rate_hz=256.0,
        )
        for index in range(2)
    )
    models = tuple(
        DelayCoordinateModel(
            index, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,)
        )
        for index in range(2)
    )
    job = Job(
        start_time=epoch,
        execution_duration_s=Decimal(8),
        integration_duration_s=Decimal(8),
        subintegration_duration_s=Decimal(8),
        fringe_approximation_order=1,
        stations=stations,
        frequencies=(frequency,),
        datastreams=datastreams,
        baselines=(Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),)),),
        scans=(Scan("scan", Decimal(0), Decimal(8), "0", "0", models),),
    )
    plans = {}
    for datastream in datastreams:
        path = Path(datastream.source_files[0])
        path.write_bytes((bytes(32) + bytes([0xAA]) * 8) * 64)
        plans[datastream.identifier] = VdifPayloadPlan(
            40,
            8,
            64,
            (
                VdifPayloadFilePlan(
                    path,
                    64,
                    np.arange(64, dtype=np.int32),
                ),
            ),
        )

    class Source:
        def batches(self, datastream):
            yield VoltageBatch(
                plans[datastream.identifier],
                epoch,
                256.0,
                0,
                "R",
                np.ones(64, dtype=np.bool_),
                PackedVoltageLayout(8, 32, 1, 0, False),
            )

    return job, Source()


def _packed_multichannel_two_station_source(tmp_path: Path):
    epoch = MJDTime(58849, Decimal(0))
    stations = tuple(
        Station(index, f"S{index}", ClockPolynomial(epoch, (0.0,)))
        for index in range(2)
    )
    frequencies = (
        Frequency(0, 0.0, 128.0, Sideband.USB, 128),
        Frequency(1, 128.0, 128.0, Sideband.USB, 128),
    )
    bands = (
        RecordedBand(0, 0, "R"),
        RecordedBand(1, 1, "L"),
    )
    datastreams = tuple(
        Datastream(
            index,
            index,
            SamplingKind.REAL,
            2,
            40,
            (str(tmp_path / f"station-multi-{index}.vdif"),),
            bands,
            sample_rate_hz=256.0,
        )
        for index in range(2)
    )
    models = tuple(
        DelayCoordinateModel(
            index, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,)
        )
        for index in range(2)
    )
    job = Job(
        start_time=epoch,
        execution_duration_s=Decimal(4),
        integration_duration_s=Decimal(4),
        subintegration_duration_s=Decimal(4),
        fringe_approximation_order=1,
        stations=stations,
        frequencies=frequencies,
        datastreams=datastreams,
        baselines=(
            Baseline(
                0,
                0,
                1,
                (
                    BaselineProduct(0, 0, 0),
                    BaselineProduct(1, 1, 1),
                ),
            ),
        ),
        scans=(Scan("scan", Decimal(0), Decimal(4), "0", "0", models),),
    )
    plans = {}
    for datastream in datastreams:
        path = Path(datastream.source_files[0])
        frame_payloads = bytes(
            (frame + byte) & 0xFF for frame in range(64) for byte in range(8)
        )
        raw = bytearray()
        for frame in range(64):
            raw.extend(bytes(32))
            raw.extend(frame_payloads[frame * 8 : (frame + 1) * 8])
        path.write_bytes(raw)
        plans[datastream.identifier] = VdifPayloadPlan(
            40,
            8,
            64,
            (
                VdifPayloadFilePlan(
                    path,
                    64,
                    np.arange(64, dtype=np.int32),
                ),
            ),
        )

    validities = {
        datastream.identifier: np.ones(64, dtype=np.bool_)
        for datastream in datastreams
    }

    class Source:
        def batches(self, datastream):
            for channel, band in enumerate(bands):
                yield VoltageBatch(
                    plans[datastream.identifier],
                    epoch,
                    256.0,
                    band.identifier,
                    band.polarization,
                    validities[datastream.identifier],
                    PackedVoltageLayout(8, 16, 2, channel, False),
                )

    return job, Source()


@pytest.mark.gpu
def test_cupy_fft_smoke_matches_numpy():
    cupy = _cupy()
    values = (
        np.random.default_rng(42).normal(size=(4, 128)).astype(np.float32)
    )
    actual = cupy.asnumpy(cupy.fft.rfft(cupy.asarray(values), axis=-1))
    expected = np.fft.rfft(values, axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("complex_input", [False, True])
def test_raw_front_kernel_matches_vectorized_reference(complex_input):
    cupy = _cupy()
    if complex_input:
        source_host = np.array(
            [1 + 2j, 2 - 1j, -3 + 0.5j, 4 + 3j, -2j], np.complex64
        )
    else:
        source_host = np.array([1, 2, -3, 4, -2], np.float32)
    validity_host = np.array([1, 0, 1, 1, 1], np.uint8)
    output_indices_host = np.array([1], np.int64)
    coarse_host = np.array([0], np.int64)
    starts_host = np.array([0.25], np.float64)
    delay_zero_host = np.array([0.125], np.float64)
    delay_step_host = np.array([0.01], np.float64)
    delay_quadratic_host = np.array([0.002], np.float64)

    block_parameters = prepare_block_parameters(
        output_indices=output_indices_host,
        coarse_samples=coarse_host,
        block_starts=starts_host,
        delay_zero=delay_zero_host,
        delay_step=delay_step_host,
        delay_quadratic=delay_quadratic_host,
        sample_rate_hz=4.0,
        centre_delay=np.zeros(1),
        xp=cupy,
    )
    rotated, weight = front_rotate(
        source=cupy.asarray(source_host),
        source_validity=cupy.asarray(validity_host),
        source_start_sample=0,
        block_parameters=block_parameters,
        fft_length=4,
        sample_rate_hz=4.0,
        fringe_reference_hz=2.0,
        lo_offset_hz=0.5,
        wall_lo_cycles=0.125,
        complex_input=complex_input,
        xp=cupy,
    )

    source_indices = np.arange(1, 5)
    values = source_host[source_indices].astype(np.complex128)
    valid = validity_host[source_indices].astype(bool)
    values[~valid] = 0
    sample_indices = np.arange(4)
    delays = (
        delay_zero_host[0]
        + delay_step_host[0] * sample_indices
        + delay_quadratic_host[0] * sample_indices**2
    )
    times = starts_host[0] + sample_indices / 4.0
    cycles = -2.0 * delays + 0.125 + 0.5 * times
    cycles -= np.floor(cycles + 0.5)
    expected = values * np.exp(2j * np.pi * cycles)
    np.testing.assert_allclose(
        cupy.asnumpy(rotated[0]), expected, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(cupy.asnumpy(weight), [0.75], atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("complex_input", "channel_count", "channel_index"),
    [(False, 2, 1), (True, 1, 0)],
)
def test_raw_bulk_vdif_decode_matches_scalar_decoder(
    complex_input, channel_count, channel_index
):
    cupy = _cupy()
    payload_bytes = 8
    frame_count = 2
    payloads = np.array(
        [0xE4, 0x1B, 0x9C, 0x72, 0x35, 0xCA, 0x87, 0x4D] * frame_count,
        dtype=np.uint8,
    )
    components = channel_count * (2 if complex_input else 1)
    samples_per_frame = payload_bytes * 4 // components
    actual = decode_2bit_band(
        cupy.asarray(payloads),
        frame_count=frame_count,
        samples_per_frame=samples_per_frame,
        payload_bytes=payload_bytes,
        channel_count=channel_count,
        channel_index=channel_index,
        complex_input=complex_input,
        xp=cupy,
    )
    expected_frames = []
    for frame in range(frame_count):
        start = frame * payload_bytes
        decoded, _validity = decode_2bit(
            payloads[start : start + payload_bytes],
            complex_data=complex_input,
            channels=channel_count,
        )
        expected_frames.append(decoded[:, channel_index])
    expected = np.concatenate(expected_frames)
    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)


def _test_vdif_frame(payload: bytes, frame_number: int) -> bytes:
    header = np.zeros(8, dtype="<u4")
    header[0] = frame_number
    return header.tobytes() + payload


def _streaming_vdif_frame(
    payload: bytes,
    frame_number: int,
    *,
    invalid: bool = False,
    thread_id: int = 5,
    reference_epoch: int = 40,
    seconds_from_epoch: int = 0,
    extended_data_version: int = 0,
) -> bytes:
    header = np.zeros(8, dtype="<u4")
    header[0] = np.uint32(seconds_from_epoch | (int(invalid) << 31))
    header[1] = np.uint32(frame_number | (reference_epoch << 24))
    header[2] = np.uint32(5 | (1 << 24))
    header[3] = np.uint32(
        (ord("A") << 8) | ord("A") | (thread_id << 16) | ((2 - 1) << 26)
    )
    header[4] = np.uint32(extended_data_version << 24)
    return header.tobytes() + payload


def _streaming_plan(
    path: Path,
    raw_frame_count: int,
    slot_count: int,
) -> VdifPayloadPlan:
    template = VdifHeader.from_bytes(_streaming_vdif_frame(bytes(8), 0)[:32])
    validity = np.zeros(slot_count, dtype=np.bool_)
    metadata = VdifStreamingPlan(
        5,
        4,
        absolute_frame_slot(template, 4),
        template,
        DuplicatePolicy.FIRST_VALID,
        validity,
    )
    return VdifPayloadPlan(
        40,
        8,
        slot_count,
        (VdifPayloadFilePlan(path, raw_frame_count, None),),
        metadata,
    )


def test_streaming_vdif_mapping_preserves_gaps_and_invalidity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream.vdif"
    raw = b"".join(
        (
            _streaming_vdif_frame(bytes((0x11,)) * 8, 0),
            _streaming_vdif_frame(bytes((0x22,)) * 8, 2, invalid=True),
            _streaming_vdif_frame(bytes((0x33,)) * 8, 3),
        )
    )
    plan = _streaming_plan(path, 3, 4)
    state = upload_resources._StreamingMappingState(
        np.zeros(4, dtype=np.bool_)
    )

    mappings, coverage = upload_resources._streaming_vdif_mappings(
        [plan], np.frombuffer(raw, dtype=np.uint8), 3, 40, [state]
    )

    np.testing.assert_array_equal(mappings[0], [0, 2, 3])
    np.testing.assert_array_equal(
        plan.streaming.frame_validity, [True, False, False, True]
    )
    assert coverage == [3]


def test_streaming_vdif_mapping_resolves_duplicate_across_chunks(
    tmp_path: Path,
) -> None:
    plan = _streaming_plan(tmp_path / "stream.vdif", 2, 1)
    state = upload_resources._StreamingMappingState(
        np.zeros(1, dtype=np.bool_)
    )
    invalid = _streaming_vdif_frame(bytes((0x11,)) * 8, 0, invalid=True)
    valid = _streaming_vdif_frame(bytes((0x22,)) * 8, 0)

    first, _coverage = upload_resources._streaming_vdif_mappings(
        [plan], np.frombuffer(invalid, dtype=np.uint8), 1, 40, [state]
    )
    second, _coverage = upload_resources._streaming_vdif_mappings(
        [plan], np.frombuffer(valid, dtype=np.uint8), 1, 40, [state]
    )

    np.testing.assert_array_equal(first[0], [0])
    np.testing.assert_array_equal(second[0], [0])
    np.testing.assert_array_equal(plan.streaming.frame_validity, [True])


def test_streaming_vdif_mapping_accepts_reference_epoch_transition(
    tmp_path: Path,
) -> None:
    final_second = 182 * 24 * 60 * 60 - 1
    first = _streaming_vdif_frame(
        bytes((0x11,)) * 8,
        3,
        reference_epoch=40,
        seconds_from_epoch=final_second,
    )
    second = _streaming_vdif_frame(
        bytes((0x22,)) * 8,
        0,
        reference_epoch=41,
    )
    template = VdifHeader.from_bytes(first[:32])
    streaming = VdifStreamingPlan(
        5,
        4,
        absolute_frame_slot(template, 4),
        template,
        DuplicatePolicy.FIRST_VALID,
        np.zeros(2, dtype=np.bool_),
    )
    plan = VdifPayloadPlan(
        40,
        8,
        2,
        (VdifPayloadFilePlan(tmp_path / "epochs.vdif", 2, None),),
        streaming,
    )
    state = upload_resources._StreamingMappingState(
        np.zeros(2, dtype=np.bool_)
    )

    mappings, coverage = upload_resources._streaming_vdif_mappings(
        [plan], np.frombuffer(first + second, dtype=np.uint8), 2, 40, [state]
    )

    np.testing.assert_array_equal(mappings[0], [0, 1])
    np.testing.assert_array_equal(streaming.frame_validity, [True, True])
    assert coverage == [1]


def test_streaming_vdif_mapping_rejects_extended_format_change(
    tmp_path: Path,
) -> None:
    plan = _streaming_plan(tmp_path / "stream.vdif", 2, 2)
    raw = b"".join(
        (
            _streaming_vdif_frame(bytes((0x11,)) * 8, 0),
            _streaming_vdif_frame(
                bytes((0x22,)) * 8,
                1,
                extended_data_version=1,
            ),
        )
    )
    state = upload_resources._StreamingMappingState(
        np.zeros(2, dtype=np.bool_)
    )

    with pytest.raises(VdifError, match="format changed"):
        upload_resources._streaming_vdif_mappings(
            [plan], np.frombuffer(raw, dtype=np.uint8), 2, 40, [state]
        )


@pytest.mark.gpu
def test_gpu_streaming_plan_reuse_resets_host_validity(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    path = tmp_path / "reused-stream.vdif"
    path.write_bytes(_streaming_vdif_frame(bytes((0x11,)) * 8, 0))
    plan = _streaming_plan(path, 1, 1)
    assert plan.streaming is not None
    plan = replace(
        plan,
        streaming=replace(
            plan.streaming,
            duplicate_policy=DuplicatePolicy.LAST_VALID,
        ),
    )

    load_vdif_payload_plans(
        [plan],
        xp=cupy,
        validity_outputs=[cupy.zeros(1, dtype=cupy.uint8)],
        target_chunk_bytes=40,
    )
    np.testing.assert_array_equal(plan.streaming.frame_validity, [True])

    path.write_bytes(
        _streaming_vdif_frame(bytes((0x22,)) * 8, 0, invalid=True)
    )
    load_vdif_payload_plans(
        [plan],
        xp=cupy,
        validity_outputs=[cupy.zeros(1, dtype=cupy.uint8)],
        target_chunk_bytes=40,
    )

    np.testing.assert_array_equal(plan.streaming.frame_validity, [False])


@pytest.mark.gpu
def test_gpu_streaming_vdif_loader_matches_mapped_layout(tmp_path: Path):
    cupy = _cupy()
    payloads = [bytes((value,)) * 8 for value in (0x11, 0x22, 0x33)]
    raw = b"".join(
        (
            _streaming_vdif_frame(payloads[0], 0),
            _streaming_vdif_frame(payloads[1], 2, invalid=True),
            _streaming_vdif_frame(payloads[2], 3),
        )
    )
    path = tmp_path / "stream.vdif"
    path.write_bytes(raw)
    plan = _streaming_plan(path, 3, 4)
    output = cupy.zeros(plan.byte_count, dtype=cupy.uint8)
    validity = cupy.full(plan.slot_count, 0xFF, dtype=cupy.uint8)

    load_vdif_payload_plans(
        [plan],
        xp=cupy,
        outputs=[output],
        validity_outputs=[validity],
        target_chunk_bytes=80,
    )

    expected = np.zeros((4, 8), dtype=np.uint8)
    expected[0] = np.frombuffer(payloads[0], dtype=np.uint8)
    expected[2] = np.frombuffer(payloads[1], dtype=np.uint8)
    expected[3] = np.frombuffer(payloads[2], dtype=np.uint8)
    np.testing.assert_array_equal(
        cupy.asnumpy(output).reshape(4, 8), expected
    )
    np.testing.assert_array_equal(cupy.asnumpy(validity), [1, 0, 0, 1])


@pytest.mark.gpu
def test_gpu_streaming_loader_resolves_overlapping_ordered_files(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    first_path = tmp_path / "first.vdif"
    second_path = tmp_path / "second.vdif"
    first_path.write_bytes(
        _streaming_vdif_frame(bytes((0x11,)) * 8, 0, invalid=True)
        + _streaming_vdif_frame(bytes((0x22,)) * 8, 1)
    )
    second_path.write_bytes(
        _streaming_vdif_frame(bytes((0x33,)) * 8, 0)
        + _streaming_vdif_frame(bytes((0x44,)) * 8, 1)
    )
    template = VdifHeader.from_bytes(first_path.read_bytes()[:32])
    streaming = VdifStreamingPlan(
        5,
        4,
        absolute_frame_slot(template, 4),
        template,
        DuplicatePolicy.FIRST_VALID,
        np.zeros(2, dtype=np.bool_),
    )
    plan = VdifPayloadPlan(
        40,
        8,
        2,
        (
            VdifPayloadFilePlan(first_path, 2, None),
            VdifPayloadFilePlan(second_path, 2, None),
        ),
        streaming,
    )
    validity = cupy.full(2, 0xFF, dtype=cupy.uint8)
    publications = []

    class Progress:
        def publish_through(self, stop, event, frame_validity):
            publications.append((stop, frame_validity.copy()))

    (output,) = load_vdif_payload_plans(
        [plan],
        xp=cupy,
        validity_outputs=[validity],
        progresses=[Progress()],
        target_chunk_bytes=80,
    )

    np.testing.assert_array_equal(
        cupy.asnumpy(output).reshape(2, 8)[:, 0], [0x33, 0x22]
    )
    np.testing.assert_array_equal(cupy.asnumpy(validity), [1, 1])
    assert len(publications) == 1
    assert publications[0][0] == 2
    np.testing.assert_array_equal(publications[0][1], [True, True])


@pytest.mark.parametrize(
    "destination_slots",
    [
        np.asarray([-2], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    ],
)
def test_gpu_vdif_loader_rejects_invalid_destination_slots(
    tmp_path: Path, destination_slots: np.ndarray
) -> None:
    file_plan = VdifPayloadFilePlan(
        tmp_path / "input.vdif", 1, destination_slots
    )
    plan = VdifPayloadPlan(40, 8, 2, (file_plan,))

    with pytest.raises((TypeError, ValueError), match="destination slot"):
        load_vdif_payload_plans([plan], xp=None)


@pytest.mark.parametrize(
    "destination_slots",
    [
        (np.asarray([0, 0], dtype=np.int32),),
        (
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        ),
    ],
)
def test_gpu_vdif_loader_rejects_duplicate_destination_slots(
    tmp_path: Path, destination_slots: tuple[np.ndarray, ...]
) -> None:
    files = tuple(
        VdifPayloadFilePlan(
            tmp_path / f"input-{index}.vdif", mapping.size, mapping
        )
        for index, mapping in enumerate(destination_slots)
    )
    plan = VdifPayloadPlan(40, 8, 2, files)

    with pytest.raises(ValueError, match="duplicate VDIF destination slot"):
        load_vdif_payload_plans([plan], xp=None)


def test_gpu_vdif_loader_rejects_unknown_io_backend(tmp_path: Path) -> None:
    plan = VdifPayloadPlan(
        40,
        8,
        1,
        (
            VdifPayloadFilePlan(
                tmp_path / "input.vdif",
                1,
                np.asarray([0], dtype=np.int32),
            ),
        ),
    )

    with pytest.raises(ValueError, match="I/O backend"):
        load_vdif_payload_plans([plan], xp=None, io_backend="unknown")


def test_visibility_ranges_reject_overlapping_output_intervals() -> None:
    with pytest.raises(ValueError, match="output intervals overlap"):
        _validate_visibility_ranges(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            block_count=2,
            output_channels=4,
            output_size=8,
            weight_size=2,
        )


def test_visibility_ranges_reject_duplicate_weight_offsets() -> None:
    with pytest.raises(ValueError, match="weight offsets must be unique"):
        _validate_visibility_ranges(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([0, 4], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
            block_count=2,
            output_channels=4,
            output_size=8,
            weight_size=2,
        )


@pytest.mark.parametrize(
    ("complex_input", "source_dtype"),
    [(False, np.float64), (True, np.complex128)],
)
def test_front_rotate_rejects_non_kernel_sample_dtype(
    complex_input: bool, source_dtype: np.dtype
) -> None:
    block_parameters = np.zeros(
        BLOCK_PARAMETERS_DTYPE.itemsize, dtype=np.uint8
    )

    with pytest.raises(TypeError, match="decoded voltage samples"):
        front_rotate(
            source=np.ones(4, dtype=source_dtype),
            source_validity=np.ones(4, dtype=np.uint8),
            source_start_sample=0,
            block_parameters=block_parameters,
            fft_length=4,
            sample_rate_hz=4.0,
            fringe_reference_hz=2.0,
            lo_offset_hz=0.0,
            wall_lo_cycles=0.0,
            complex_input=complex_input,
            xp=np,
        )


@pytest.mark.parametrize(
    ("validity", "stride", "error", "message"),
    [
        (np.ones(9, dtype=np.uint8), 0, ValueError, "validity_stride"),
        (np.ones(9, dtype=np.uint8), -1, ValueError, "validity_stride"),
        (np.ones(9, dtype=np.uint8), 2**31, ValueError, "validity_stride"),
        (np.ones(9, dtype=bool), 1, TypeError, "contiguous uint8"),
        (np.ones(18, dtype=np.uint8)[::2], 1, TypeError, "contiguous uint8"),
        (np.ones(2, dtype=np.uint8), 4, ValueError, "too short"),
    ],
)
def test_front_rotate_validates_source_validity_before_launch(
    validity, stride, error, message, monkeypatch
):
    def forbidden(*args):
        pytest.fail("invalid source validity reached kernel lookup")

    monkeypatch.setattr(gpu_kernels, "_front_kernel", forbidden)
    with pytest.raises(error, match=message):
        front_rotate(
            source=np.ones(9, dtype=np.float32),
            source_validity=validity,
            validity_stride=stride,
            source_start_sample=0,
            block_parameters=np.zeros(
                BLOCK_PARAMETERS_DTYPE.itemsize, dtype=np.uint8
            ),
            fft_length=4,
            sample_rate_hz=4.0,
            fringe_reference_hz=0.0,
            lo_offset_hz=0.0,
            wall_lo_cycles=0.0,
            complex_input=False,
            xp=np,
        )


@pytest.mark.parametrize("authoritative", [False, True])
def test_front_rotate_accepts_compressed_or_authoritative_validity(
    authoritative, monkeypatch
):
    launches = []
    monkeypatch.setattr(
        gpu_kernels,
        "_front_kernel",
        lambda *args: lambda *args: launches.append(args),
    )
    front_rotate(
        source=np.ones(9, dtype=np.float32),
        source_validity=np.ones(0 if authoritative else 3, dtype=np.uint8),
        validity_stride=0 if authoritative else 4,
        authoritative_validity=authoritative,
        source_start_sample=0,
        block_parameters=np.zeros(
            BLOCK_PARAMETERS_DTYPE.itemsize, dtype=np.uint8
        ),
        fft_length=4,
        sample_rate_hz=4.0,
        fringe_reference_hz=0.0,
        lo_offset_hz=0.0,
        wall_lo_cycles=0.0,
        complex_input=False,
        xp=np,
    )
    assert len(launches) == 1


@pytest.mark.parametrize("bands", [False, True])
def test_packed_front_rotate_requires_one_validity_byte_per_frame(bands):
    arguments = dict(
        source=np.ones(16, dtype=np.uint8),
        source_validity=np.ones(1, dtype=np.uint8),
        validity_stride=64,
        source_sample_count=64,
        block_parameters=np.zeros(
            (2 if bands else 1) * BLOCK_PARAMETERS_DTYPE.itemsize,
            dtype=np.uint8,
        ),
        fft_length=4,
        sample_rate_hz=4.0,
        packed_samples_per_frame=32,
        packed_payload_bytes=8,
        packed_channel_count=1,
        xp=np,
    )
    with pytest.raises(ValueError, match="too short"):
        if bands:
            front_rotate_packed_real_bands(
                **arguments,
                source_start_samples=[0, 0],
                fringe_reference_hz=[0.0, 0.0],
                lo_offset_hz=[0.0, 0.0],
                wall_lo_cycles=[0.0, 0.0],
                packed_channel_indices=[0, 0],
            )
        else:
            front_rotate(
                **arguments,
                source_start_sample=0,
                fringe_reference_hz=0.0,
                lo_offset_hz=0.0,
                wall_lo_cycles=0.0,
                complex_input=False,
            )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"packed_samples_per_frame": 0}, "samples_per_frame"),
        (
            {"packed_payload_bytes": int(np.iinfo(np.int32).max) + 1},
            "payload_bytes",
        ),
        ({"packed_samples_per_frame": 31}, "exactly fill"),
        ({"source_sample_count": 31}, "source size"),
    ],
)
def test_packed_front_rotate_rejects_malformed_geometry(
    overrides: dict[str, int], message: str
) -> None:
    block_parameters = np.zeros(
        BLOCK_PARAMETERS_DTYPE.itemsize, dtype=np.uint8
    )
    arguments = {
        "source": np.ones(8, dtype=np.uint8),
        "source_validity": np.ones(1, dtype=np.uint8),
        "source_start_sample": 0,
        "block_parameters": block_parameters,
        "fft_length": 4,
        "sample_rate_hz": 4.0,
        "fringe_reference_hz": 2.0,
        "lo_offset_hz": 0.0,
        "wall_lo_cycles": 0.0,
        "complex_input": False,
        "source_sample_count": 32,
        "packed_samples_per_frame": 32,
        "packed_payload_bytes": 8,
        "packed_channel_count": 1,
        "packed_channel_index": 0,
        "xp": np,
    }
    arguments.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        front_rotate(**arguments)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frame_count": 0}, "frame_count"),
        (
            {"channel_count": int(np.iinfo(np.int32).max) + 1},
            "channel_count",
        ),
        ({"samples_per_frame": 31}, "exactly fill"),
        ({"frame_count": 2}, "payload size"),
    ],
)
def test_decode_2bit_band_rejects_malformed_geometry(
    overrides: dict[str, int], message: str
) -> None:
    arguments = {
        "payloads": np.ones(8, dtype=np.uint8),
        "frame_count": 1,
        "samples_per_frame": 32,
        "payload_bytes": 8,
        "channel_count": 1,
        "channel_index": 0,
        "complex_input": False,
        "xp": np,
    }
    arguments.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        decode_2bit_band(**arguments)


@pytest.mark.gpu
def test_gpu_vdif_loader_strips_headers_and_scatters_multiple_plans(
    tmp_path: Path,
):
    cupy = _cupy()
    payload_bytes = 8
    frame_size = 32 + payload_bytes
    first_path = tmp_path / "first.vdif"
    second_path = tmp_path / "second.vdif"
    payload_a = bytes(range(8))
    payload_b = bytes(range(8, 16))
    payload_c = bytes(range(16, 24))
    first_path.write_bytes(
        _test_vdif_frame(payload_a, 0) + _test_vdif_frame(payload_b, 1)
    )
    second_path.write_bytes(_test_vdif_frame(payload_c, 2))
    files_a = (
        VdifPayloadFilePlan(first_path, 2, np.asarray([2, -1], np.int32)),
        VdifPayloadFilePlan(second_path, 1, np.asarray([0], np.int32)),
    )
    files_b = (
        VdifPayloadFilePlan(first_path, 2, np.asarray([-1, 1], np.int32)),
        VdifPayloadFilePlan(second_path, 1, np.asarray([-1], np.int32)),
    )
    plans = [
        VdifPayloadPlan(frame_size, payload_bytes, 4, files_a),
        VdifPayloadPlan(frame_size, payload_bytes, 3, files_b),
    ]

    outputs = [
        cupy.zeros(plan.byte_count, dtype=cupy.uint8) for plan in plans
    ]
    ready = cupy.cuda.Event()
    ready.record(cupy.cuda.get_current_stream())
    actual_a, actual_b = load_vdif_payload_plans(
        plans,
        xp=cupy,
        target_chunk_bytes=frame_size,
        outputs=outputs,
        ready_event=ready,
    )
    assert actual_a is outputs[0]
    assert actual_b is outputs[1]

    np.testing.assert_array_equal(
        cupy.asnumpy(actual_a).reshape(4, payload_bytes),
        np.asarray(
            [list(payload_c), [0] * 8, list(payload_a), [0] * 8], np.uint8
        ),
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(actual_b).reshape(3, payload_bytes),
        np.asarray([[0] * 8, list(payload_b), [0] * 8], np.uint8),
    )


@pytest.mark.gpu
def test_gpu_vdif_loader_reads_only_selected_frame_window(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    payloads = [bytes((index,)) * 8 for index in range(5)]
    path = tmp_path / "window.vdif"
    path.write_bytes(
        b"".join(
            _test_vdif_frame(payload, index)
            for index, payload in enumerate(payloads)
        )
    )
    mapping = np.asarray([-1, -1, 0, 1, -1], dtype=np.int32)
    plan = VdifPayloadPlan(
        40,
        8,
        2,
        (VdifPayloadFilePlan(path, 5, mapping),),
    )

    (actual,) = load_vdif_payload_plans(
        [plan], xp=cupy, target_chunk_bytes=40
    )

    np.testing.assert_array_equal(
        cupy.asnumpy(actual).reshape(2, 8),
        np.asarray([list(payloads[2]), list(payloads[3])], dtype=np.uint8),
    )


@pytest.mark.gpu
def test_gpu_vdif_loader_supports_sparse_repeated_source_frames(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    payloads = [bytes((index,)) * 8 for index in range(4)]
    path = tmp_path / "sparse-repeat.vdif"
    path.write_bytes(
        b"".join(
            _test_vdif_frame(payload, index)
            for index, payload in enumerate(payloads)
        )
    )
    plan = VdifPayloadPlan(
        40,
        8,
        4,
        (
            VdifPayloadFilePlan(
                path,
                4,
                np.arange(4, dtype=np.int32),
                source_frame_indices=np.asarray([3, 1, 3, 0], dtype=np.int64),
            ),
        ),
    )

    (actual,) = load_vdif_payload_plans(
        [plan], xp=cupy, target_chunk_bytes=40
    )

    np.testing.assert_array_equal(
        cupy.asnumpy(actual).reshape(4, 8),
        np.asarray(
            [
                list(payloads[3]),
                list(payloads[1]),
                list(payloads[3]),
                list(payloads[0]),
            ],
            dtype=np.uint8,
        ),
    )


def test_vdif_upload_staging_bytes_covers_aligned_transfer_buffers() -> None:
    plan = VdifPayloadPlan(
        40,
        8,
        1,
        (
            VdifPayloadFilePlan(
                Path("source.vdif"),
                2,
                np.asarray([0, -1], dtype=np.int32),
            ),
        ),
    )

    assert (
        vdif_upload_staging_bytes([plan], target_chunk_bytes=88)
        == 2 * 4 * 1024 * 1024
    )


def test_cufile_staging_bytes_include_all_device_allocations() -> None:
    plan = VdifPayloadPlan(
        40,
        8,
        1,
        (
            VdifPayloadFilePlan(
                Path("source.vdif"),
                2,
                np.asarray([0, -1], dtype=np.int32),
            ),
        ),
    )
    alignment = 64 * 1024
    registered = 4 * 1024 * 1024
    expected = 4 * (registered + alignment - 1) + 4 * 2 * 4

    assert (
        vdif_upload_staging_bytes(
            [plan], target_chunk_bytes=88, io_backend="cufile"
        )
        == expected
    )
    assert (
        upload_resources.reusable_vdif_upload_staging_bytes(
            [plan], target_chunk_bytes=88, io_backend="cufile"
        )
        == expected - 4 * 2 * 4
    )


def test_cufile_raw_registration_excludes_per_plan_mappings() -> None:
    plan_count = 64
    target_chunk_bytes = 64 * 1024 * 1024
    plans = [
        VdifPayloadPlan(
            40,
            8,
            1,
            (
                VdifPayloadFilePlan(
                    Path("source.vdif"),
                    1,
                    np.asarray([0], dtype=np.int32),
                ),
            ),
        )
        for _index in range(plan_count)
    ]
    chunk_frames = target_chunk_bytes // (40 + plan_count * 4)
    alignment = 64 * 1024
    granularity = 4 * 1024 * 1024
    aligned_request = (
        (chunk_frames * 40 + 2 * (alignment - 1) + alignment - 1)
        // alignment
        * alignment
    )
    registered = (
        (aligned_request + granularity - 1) // granularity * granularity
    )
    expected_raw = 4 * (registered + alignment - 1)
    expected_mappings = 4 * plan_count * chunk_frames * 4

    assert (
        vdif_upload_staging_bytes(
            plans,
            target_chunk_bytes=target_chunk_bytes,
            io_backend="cufile",
        )
        == expected_raw + expected_mappings
    )
    assert (
        upload_resources.reusable_vdif_upload_staging_bytes(
            plans,
            target_chunk_bytes=target_chunk_bytes,
            io_backend="cufile",
        )
        == expected_raw
    )


@pytest.mark.parametrize(
    "helper_name", ["vdif_upload_staging_slots", "vdif_upload_slot_limit"]
)
def test_cufile_tuning_rejects_unknown_io_backend(helper_name: str) -> None:
    helper = getattr(cupy_kernels, helper_name)

    with pytest.raises(ValueError, match="unsupported VDIF I/O backend"):
        helper("unknown")


def test_vdif_upload_staging_bytes_rejects_invalid_frame_geometry() -> None:
    plan = VdifPayloadPlan(
        0,
        0,
        1,
        (
            VdifPayloadFilePlan(
                Path("source.vdif"),
                1,
                np.asarray([0], dtype=np.int32),
            ),
        ),
    )

    with pytest.raises(ValueError, match="aligned standard headers"):
        vdif_upload_staging_bytes([plan])


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("streaming", "io_backend"),
    [(False, "posix"), (False, "cufile"), (True, "posix")],
)
def test_gpu_upload_preserves_planned_physical_thread_alias(
    tmp_path: Path, streaming: bool, io_backend: str
) -> None:
    from tests.xbeam.test_ingest_planning import _frame, _job

    cupy = _cupy()
    if io_backend == "cufile":
        pytest.importorskip("cuda.bindings.cufile")
    job = _job(tmp_path)
    index_dir = tmp_path / "indexes"
    for datastream in job.datastreams:
        path = Path(datastream.source_files[0])
        path.write_bytes(
            b"".join(
                _frame(
                    job.stations[datastream.station_id].name,
                    second,
                    thread_id=7,
                    payload_value=second + 1,
                )
                for second in range(5)
            )
        )
        build_vdif_binary_index(path, index_dir=index_dir, backend="posix")
    plan = plan_vdif_ingest(
        job,
        intervals={
            (stream.identifier, 0): SampleInterval(64, 96)
            for stream in job.datastreams
        },
        max_read_bytes=1 << 20,
        index_dir=index_dir,
        assume_ordered=False,
    )
    assert plan.manifest["mode"] == "binary-sidecar"
    assert plan.manifest["thread_aliases"][0] == {
        "datastream_id": 0,
        "logical_thread_id": 0,
        "physical_thread_id": 7,
    }
    source = VdifFileRecordingSource(
        job,
        ANALYTIC_V1,
        source_ranges=plan.source_ranges,
        thread_windows=plan.thread_windows,
        max_materialized_bytes=1 << 20,
        binary_sidecars=True,
        binary_index_dir=index_dir,
        defer_ordered_headers=streaming,
    )
    (batch,) = source.batches(job.datastreams[0])
    payload = batch.samples
    assert isinstance(payload, VdifPayloadPlan)
    if streaming:
        assert payload.streaming.thread_id == 7
    validity = cupy.asarray(batch.validity, dtype=cupy.uint8)
    statistics = upload_resources.CuFileReadStatistics()
    try:
        (actual,) = load_vdif_payload_plans(
            [payload],
            xp=cupy,
            target_chunk_bytes=80,
            validity_outputs=[validity],
            io_backend=io_backend,
            cufile_statistics=statistics,
        )
    except CuFileUnavailable as error:
        pytest.skip(str(error))
    # The selected second plus one frame on each side, independently encoded.
    np.testing.assert_array_equal(
        cupy.asnumpy(actual), np.repeat(np.asarray([2, 3, 4], np.uint8), 8)
    )
    np.testing.assert_array_equal(cupy.asnumpy(validity), [1, 1, 1])
    np.testing.assert_array_equal(batch.validity, [True, True, True])
    if io_backend == "cufile":
        assert statistics.snapshot().calls > 0


@pytest.mark.gpu
def test_gpu_cufile_loader_matches_posix_loader(tmp_path: Path) -> None:
    cupy = _cupy()
    pytest.importorskip("cuda.bindings.cufile")
    payloads = [bytes((index,)) * 8 for index in range(8)]
    path = tmp_path / "cufile.vdif"
    path.write_bytes(
        b"".join(
            _test_vdif_frame(payload, index)
            for index, payload in enumerate(payloads)
        )
    )
    mapping = np.asarray([-1, 2, 0, -1, 3, 1, -1, -1], np.int32)
    plan = VdifPayloadPlan(
        40,
        8,
        4,
        (VdifPayloadFilePlan(path, mapping.size, mapping),),
    )
    (expected,) = load_vdif_payload_plans(
        [plan], xp=cupy, target_chunk_bytes=80, io_backend="posix"
    )
    try:
        (actual,) = load_vdif_payload_plans(
            [plan],
            xp=cupy,
            target_chunk_bytes=80,
            io_backend="cufile",
        )
    except CuFileUnavailable as error:
        pytest.skip(str(error))

    np.testing.assert_array_equal(
        cupy.asnumpy(actual), cupy.asnumpy(expected)
    )


@pytest.mark.gpu
def test_gpu_cufile_sparse_repeated_mapping_matches_posix(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    pytest.importorskip("cuda.bindings.cufile")
    payloads = [bytes((index,)) * 8 for index in range(4)]
    path = tmp_path / "cufile-sparse-repeat.vdif"
    path.write_bytes(
        b"".join(
            _test_vdif_frame(payload, index)
            for index, payload in enumerate(payloads)
        )
    )
    plan = VdifPayloadPlan(
        40,
        8,
        4,
        (
            VdifPayloadFilePlan(
                path,
                4,
                np.arange(4, dtype=np.int32),
                source_frame_indices=np.asarray([3, 1, 3, 0], dtype=np.int64),
            ),
        ),
    )
    (expected,) = load_vdif_payload_plans(
        [plan], xp=cupy, target_chunk_bytes=40, io_backend="posix"
    )
    try:
        (actual,) = load_vdif_payload_plans(
            [plan],
            xp=cupy,
            target_chunk_bytes=40,
            io_backend="cufile",
        )
    except CuFileUnavailable as error:
        pytest.skip(str(error))

    np.testing.assert_array_equal(
        cupy.asnumpy(actual), cupy.asnumpy(expected)
    )


@pytest.mark.gpu
def test_gpu_cufile_reuses_registered_workspace_and_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cupy = _cupy()
    pytest.importorskip("cuda.bindings.cufile")
    payloads = [bytes((index,)) * 8 for index in range(4)]
    path = tmp_path / "cached-cufile.vdif"
    path.write_bytes(
        b"".join(
            _test_vdif_frame(payload, index)
            for index, payload in enumerate(payloads)
        )
    )
    mapping = np.arange(4, dtype=np.int32)
    plan = VdifPayloadPlan(
        40,
        8,
        4,
        (VdifPayloadFilePlan(path, mapping.size, mapping),),
    )
    registrations = []
    opened = []
    original_register = CuFileSession.register_buffer
    original_open = CuFileSession.open

    def tracked_register(session, pointer, size):
        registrations.append((pointer, size))
        return original_register(session, pointer, size)

    def tracked_open(session, selected):
        opened.append(Path(selected))
        return original_open(session, selected)

    monkeypatch.setattr(CuFileSession, "register_buffer", tracked_register)
    monkeypatch.setattr(CuFileSession, "open", tracked_open)
    upload_resources._clear_cached_cufile_workspaces()
    state = upload_resources._VdifCuFileUploadState(
        int(cupy.cuda.Device().id)
    )
    try:
        outputs = []
        for _ in range(2):
            try:
                (output,) = load_vdif_payload_plans(
                    [plan],
                    xp=cupy,
                    target_chunk_bytes=160,
                    io_backend="cufile",
                    cufile_worker_state=state,
                )
            except CuFileUnavailable as error:
                pytest.skip(str(error))
            # Caller-owned state is asynchronous; inspect after its stream
            # finishes, while retaining the workspace for the next call.
            state.stream.synchronize()
            outputs.append(cupy.asnumpy(output))

        assert len(registrations) == 4
        assert opened == [path]
        np.testing.assert_array_equal(outputs[1], outputs[0])
        for output in outputs:
            np.testing.assert_array_equal(
                output, np.frombuffer(b"".join(payloads), dtype=np.uint8)
            )
    finally:
        upload_resources._close_vdif_cufile_upload_state(
            state, best_effort=True
        )
        upload_resources._clear_cached_cufile_workspaces()


@pytest.mark.gpu
def test_gpu_vdif_loader_synchronizes_before_short_read_failure(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    payload = bytes(range(8))
    path = tmp_path / "short.vdif"
    path.write_bytes(_test_vdif_frame(payload, 0))
    file_plan = VdifPayloadFilePlan(
        path, 2, np.asarray([0, 1], dtype=np.int32)
    )
    plan = VdifPayloadPlan(40, 8, 2, (file_plan,))

    with pytest.raises(OSError, match="changed during GPU ingest"):
        load_vdif_payload_plans([plan], xp=cupy, target_chunk_bytes=44)

    probe = cupy.arange(16, dtype=cupy.float32)
    np.testing.assert_array_equal(cupy.asnumpy(probe), np.arange(16))


@pytest.mark.gpu
def test_gpu_vdif_loader_rejects_overlapping_outputs(tmp_path: Path) -> None:
    cupy = _cupy()
    path = tmp_path / "input.vdif"
    path.write_bytes(_test_vdif_frame(bytes(range(8)), 0))
    file_plan = VdifPayloadFilePlan(path, 1, np.asarray([0], dtype=np.int32))
    plans = [
        VdifPayloadPlan(40, 8, 1, (file_plan,)),
        VdifPayloadPlan(40, 8, 1, (file_plan,)),
    ]
    output = cupy.empty(8, dtype=cupy.uint8)

    with pytest.raises(ValueError, match="must not overlap"):
        load_vdif_payload_plans(plans, xp=cupy, outputs=[output, output])


@pytest.mark.gpu
def test_gpu_streaming_loader_rejects_payload_validity_overlap(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    path = tmp_path / "stream.vdif"
    path.write_bytes(_streaming_vdif_frame(bytes(range(8)), 0))
    plan = _streaming_plan(path, 1, 1)
    shared = cupy.empty(plan.byte_count + 1, dtype=cupy.uint8)

    with pytest.raises(ValueError, match="must not overlap"):
        load_vdif_payload_plans(
            [plan],
            xp=cupy,
            outputs=[shared[: plan.byte_count]],
            validity_outputs=[shared[:1]],
        )


@pytest.mark.gpu
def test_gpu_streaming_loader_rejects_validity_overlap(
    tmp_path: Path,
) -> None:
    cupy = _cupy()
    path = tmp_path / "stream.vdif"
    path.write_bytes(_streaming_vdif_frame(bytes(range(8)), 0))
    plans = [_streaming_plan(path, 1, 1) for _ in range(2)]
    outputs = [
        cupy.empty(plan.byte_count, dtype=cupy.uint8) for plan in plans
    ]
    validity = cupy.empty(1, dtype=cupy.uint8)

    with pytest.raises(ValueError, match="must not overlap"):
        load_vdif_payload_plans(
            plans,
            xp=cupy,
            outputs=outputs,
            validity_outputs=[validity, validity],
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("complex_input", "channel_count", "channel_index"),
    [(False, 2, 1), (True, 1, 0)],
)
def test_raw_packed_front_kernel_matches_decode_then_front(
    complex_input, channel_count, channel_index
):
    cupy = _cupy()
    payload_bytes = 8
    frame_count = 3
    payloads = np.array(
        [0xE4, 0x1B, 0x9C, 0x72, 0x35, 0xCA, 0x87, 0x4D] * frame_count,
        dtype=np.uint8,
    )
    components = channel_count * (2 if complex_input else 1)
    samples_per_frame = payload_bytes * 4 // components
    decoded = decode_2bit_band(
        cupy.asarray(payloads),
        frame_count=frame_count,
        samples_per_frame=samples_per_frame,
        payload_bytes=payload_bytes,
        channel_count=channel_count,
        channel_index=channel_index,
        complex_input=complex_input,
        xp=cupy,
    )
    block_parameters = prepare_block_parameters(
        output_indices=np.array([1, samples_per_frame + 1], dtype=np.int64),
        coarse_samples=np.array([0, 1], dtype=np.int64),
        block_starts=np.array([0.25, 0.5], dtype=np.float64),
        delay_zero=np.array([0.125, -0.25], dtype=np.float64),
        delay_step=np.array([0.01, -0.02], dtype=np.float64),
        delay_quadratic=np.array([0.002, 0.001], dtype=np.float64),
        sample_rate_hz=16.0,
        centre_delay=np.zeros(2),
        xp=cupy,
    )
    common = dict(
        source_validity=cupy.asarray([1, 0, 1], dtype=cupy.uint8),
        validity_stride=samples_per_frame,
        source_start_sample=-2,
        block_parameters=block_parameters,
        fft_length=min(8, samples_per_frame),
        sample_rate_hz=16.0,
        fringe_reference_hz=2.0,
        lo_offset_hz=0.5,
        wall_lo_cycles=0.125,
        complex_input=complex_input,
        xp=cupy,
    )
    expected_samples, expected_weight = front_rotate(source=decoded, **common)
    actual_samples, actual_weight = front_rotate(
        source=cupy.asarray(payloads),
        source_sample_count=frame_count * samples_per_frame,
        packed_samples_per_frame=samples_per_frame,
        packed_payload_bytes=payload_bytes,
        packed_channel_count=channel_count,
        packed_channel_index=channel_index,
        **common,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(actual_samples),
        cupy.asnumpy(expected_samples),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(actual_weight), cupy.asnumpy(expected_weight)
    )


@pytest.mark.gpu
def test_front_kernel_applies_local_sample_validity_range():
    cupy = _cupy()
    source = cupy.arange(1, 9, dtype=cupy.float32)
    parameters = prepare_block_parameters(
        output_indices=np.array([0], dtype=np.int64),
        coarse_samples=np.array([0], dtype=np.int64),
        block_starts=np.array([0.0]),
        delay_zero=np.array([0.0]),
        delay_step=np.array([0.0]),
        delay_quadratic=np.array([0.0]),
        sample_rate_hz=8.0,
        valid_starts=np.array([2], dtype=np.int64),
        valid_stops=np.array([6], dtype=np.int64),
        xp=cupy,
    )
    rotated, weight = front_rotate(
        source=source,
        source_validity=cupy.ones(8, dtype=cupy.uint8),
        validity_stride=1,
        source_start_sample=0,
        block_parameters=parameters,
        fft_length=8,
        sample_rate_hz=8.0,
        fringe_reference_hz=0.0,
        lo_offset_hz=0.0,
        wall_lo_cycles=0.0,
        complex_input=False,
        xp=cupy,
    )

    np.testing.assert_array_equal(
        cupy.asnumpy(rotated.real), [[0, 0, 3, 4, 5, 6, 0, 0]]
    )
    np.testing.assert_array_equal(cupy.asnumpy(rotated.imag), 0)
    np.testing.assert_allclose(cupy.asnumpy(weight), [0.5])


@pytest.mark.gpu
def test_front_kernel_authoritative_range_replaces_source_validity():
    cupy = _cupy()
    source = cupy.arange(1, 9, dtype=cupy.float32)
    parameters = prepare_block_parameters(
        output_indices=np.array([0], dtype=np.int64),
        coarse_samples=np.array([0], dtype=np.int64),
        block_starts=np.array([0.0]),
        delay_zero=np.array([0.0]),
        delay_step=np.array([0.0]),
        delay_quadratic=np.array([0.0]),
        sample_rate_hz=8.0,
        valid_starts=np.array([2], dtype=np.int64),
        valid_stops=np.array([6], dtype=np.int64),
        xp=cupy,
    )
    rotated, weight = front_rotate(
        source=source,
        source_validity=cupy.zeros(8, dtype=cupy.uint8),
        validity_stride=1,
        authoritative_validity=True,
        source_start_sample=0,
        block_parameters=parameters,
        fft_length=8,
        sample_rate_hz=8.0,
        fringe_reference_hz=0.0,
        lo_offset_hz=0.0,
        wall_lo_cycles=0.0,
        complex_input=False,
        xp=cupy,
    )

    np.testing.assert_array_equal(
        cupy.asnumpy(rotated.real), [[0, 0, 3, 4, 5, 6, 0, 0]]
    )
    np.testing.assert_array_equal(cupy.asnumpy(rotated.imag), 0)
    np.testing.assert_allclose(cupy.asnumpy(weight), [0.5])


@pytest.mark.gpu
def test_packed_real_band_front_matches_individual_channel_launches():
    cupy = _cupy()
    payload_bytes = 8
    channel_count = 2
    frame_count = 3
    samples_per_frame = payload_bytes * 4 // channel_count
    payloads = cupy.asarray(
        np.array(
            [0xE4, 0x1B, 0x9C, 0x72, 0x35, 0xCA, 0x87, 0x4D] * frame_count,
            dtype=np.uint8,
        )
    )
    validity = cupy.asarray([1, 0, 1], dtype=cupy.uint8)
    parameter_buffers = []
    expected_samples = []
    expected_weights = []
    fringe = [2.0, 3.5]
    lo = [0.5, -0.25]
    wall = [0.125, -0.375]
    starts = [-2, -2]
    for channel in range(channel_count):
        parameters = prepare_block_parameters(
            output_indices=np.array(
                [1, samples_per_frame + 1], dtype=np.int64
            ),
            coarse_samples=np.array([0, 1], dtype=np.int64),
            block_starts=np.array([0.25, 0.5], dtype=np.float64),
            delay_zero=np.array([0.125, -0.25], dtype=np.float64),
            delay_step=np.array([0.01, -0.02], dtype=np.float64),
            delay_quadratic=np.array([0.002, 0.001], dtype=np.float64),
            sample_rate_hz=16.0,
            centre_delay=np.zeros(2),
            xp=cupy,
        )
        parameter_buffers.append(parameters)
        samples, weights = front_rotate(
            source=payloads,
            source_validity=validity,
            validity_stride=samples_per_frame,
            source_start_sample=starts[channel],
            block_parameters=parameters,
            fft_length=8,
            sample_rate_hz=16.0,
            fringe_reference_hz=fringe[channel],
            lo_offset_hz=lo[channel],
            wall_lo_cycles=wall[channel],
            complex_input=False,
            source_sample_count=frame_count * samples_per_frame,
            packed_samples_per_frame=samples_per_frame,
            packed_payload_bytes=payload_bytes,
            packed_channel_count=channel_count,
            packed_channel_index=channel,
            xp=cupy,
        )
        expected_samples.append(cupy.asnumpy(samples))
        expected_weights.append(cupy.asnumpy(weights))

    actual_samples, actual_weights = front_rotate_packed_real_bands(
        source=payloads,
        source_validity=validity,
        validity_stride=samples_per_frame,
        source_sample_count=frame_count * samples_per_frame,
        block_parameters=cupy.concatenate(parameter_buffers),
        source_start_samples=starts,
        fringe_reference_hz=fringe,
        lo_offset_hz=lo,
        wall_lo_cycles=wall,
        packed_channel_indices=[0, 1],
        fft_length=8,
        sample_rate_hz=16.0,
        packed_samples_per_frame=samples_per_frame,
        packed_payload_bytes=payload_bytes,
        packed_channel_count=channel_count,
        xp=cupy,
    )
    np.testing.assert_allclose(
        cupy.asnumpy(actual_samples),
        np.asarray(expected_samples),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(actual_weights), np.asarray(expected_weights)
    )


@pytest.mark.gpu
def test_packed_multichannel_station_batch_matches_per_band_path():
    cupy = _cupy()

    epoch = MJDTime(58849, Decimal(0))
    station = Station(0, "AA", ClockPolynomial(epoch, (0.0,)))
    frequencies = (
        Frequency(0, 0.0, 128.0, Sideband.USB, 8),
        Frequency(1, 128.0, 128.0, Sideband.USB, 8),
    )
    bands = (RecordedBand(0, 0, "X"), RecordedBand(1, 1, "Y"))
    datastream = Datastream(
        0,
        0,
        SamplingKind.REAL,
        2,
        40,
        ("unused.vdif",),
        bands,
        sample_rate_hz=256.0,
    )
    model = DelayCoordinateModel(
        0, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,)
    )
    scan = Scan("scan", Decimal(0), Decimal("0.25"), "0", "0", (model,))
    job = Job(
        start_time=epoch,
        execution_duration_s=Decimal("0.25"),
        integration_duration_s=Decimal("0.25"),
        subintegration_duration_s=Decimal("0.25"),
        fringe_approximation_order=1,
        stations=(station,),
        frequencies=frequencies,
        datastreams=(datastream,),
        baselines=(),
        scans=(scan,),
    )
    payload = cupy.asarray(
        np.array(
            [0xE4, 0x1B, 0x9C, 0x72, 0x35, 0xCA, 0x87, 0x4D] * 4,
            dtype=np.uint8,
        )
    )
    validity = cupy.asarray([1, 1, 0, 1], dtype=cupy.uint8)
    payload_plan = object()
    group = []
    for index, (band, frequency) in enumerate(
        zip(bands, frequencies, strict=True)
    ):
        data = runtime_types._BandSamples(
            datastream=datastream,
            band=band,
            frequency=frequency,
            samples=np.empty(0, dtype=np.float32),
            validity=np.empty(0, dtype=np.bool_),
            start_sample=0,
            sample_rate_hz=256.0,
            device_validity=validity,
            validity_stride=16,
            device_payload=payload,
            source_sample_count=64,
            packed_samples_per_frame=16,
            packed_payload_bytes=8,
            packed_channel_count=2,
            packed_channel_index=index,
            payload_plan=payload_plan,
        )
        group.append(((0, index), data))
    starts = np.asarray([0.0, 0.0625, 0.125], dtype=np.float64)

    expected = {
        key: gpu_execution._station_blocks_for_scan(
            job, data, scan, starts, cupy
        )
        for key, data in group
    }
    actual = gpu_execution._station_blocks_packed_real_group_for_scan(
        job, group, scan, starts, cupy
    )

    for key in expected:
        np.testing.assert_allclose(
            cupy.asnumpy(actual[key].spectrum),
            cupy.asnumpy(expected[key].spectrum),
            rtol=3e-6,
            atol=3e-6,
        )
        np.testing.assert_array_equal(
            cupy.asnumpy(actual[key].weight),
            cupy.asnumpy(expected[key].weight),
        )
        np.testing.assert_array_equal(
            actual[key].baseband_hz, expected[key].baseband_hz
        )


@pytest.mark.parametrize(
    "frequencies",
    [
        np.ones(3, np.float32),
        np.ones((1, 3), np.float64),
        np.ones(2, np.float64),
        np.ones(6, np.float64)[::2],
    ],
    ids=["dtype", "rank", "length", "strided"],
)
def test_raw_residual_rejects_invalid_frequency_buffer(
    frequencies, monkeypatch
):
    def unexpected_kernel(*args):
        pytest.fail("invalid frequency buffer reached the raw kernel")

    monkeypatch.setattr(gpu_kernels, "_residual_kernel", unexpected_kernel)
    parameters = np.zeros(BLOCK_PARAMETERS_DTYPE.itemsize, dtype=np.uint8)

    with pytest.raises(TypeError, match="frequencies.*float64"):
        apply_residual_phase(
            np.ones((1, 3), np.complex64), parameters, frequencies, np
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_raw_residual_kernel_matches_vectorized_reference(dtype):
    cupy = _cupy()
    spectrum_host = np.array(
        [[1 + 2j, -3 + 4j, 5 - 6j], [2 - 1j, 4 + 3j, -5 + 2j]],
        dtype,
    )
    residual_host = np.array([0.125, -0.25], np.float64)
    frequencies_host = np.array([-3.0, -2.0, -1.0], np.float64)
    spectrum = cupy.asarray(spectrum_host)
    block_parameters = prepare_block_parameters(
        output_indices=np.zeros(2, dtype=np.int64),
        coarse_samples=np.zeros(2, dtype=np.int64),
        block_starts=np.zeros(2),
        delay_zero=np.zeros(2),
        delay_step=np.zeros(2),
        delay_quadratic=np.zeros(2),
        sample_rate_hz=1.0,
        centre_delay=-residual_host,
        xp=cupy,
    )
    apply_residual_phase(
        spectrum,
        block_parameters,
        cupy.asarray(frequencies_host),
        cupy,
    )
    cycles = residual_host[:, None] * frequencies_host[None, :]
    cycles -= np.floor(cycles + 0.5)
    expected = spectrum_host * np.exp(2j * np.pi * cycles)
    tolerance = 2e-6 if dtype is np.complex64 else 1e-13
    np.testing.assert_allclose(
        cupy.asnumpy(spectrum), expected, rtol=tolerance, atol=tolerance
    )


def test_block_parameter_layout_and_host_residual_are_exact():
    parameters = prepare_block_parameters(
        output_indices=np.array([10, 20], dtype=np.int64),
        coarse_samples=np.array([2, -3], dtype=np.int64),
        block_starts=np.array([0.25, 0.5]),
        delay_zero=np.array([1.0, 2.0]),
        delay_step=np.array([3.0, 4.0]),
        delay_quadratic=np.array([5.0, 6.0]),
        sample_rate_hz=8.0,
        centre_delay=np.array([0.125, -0.25]),
        source_offsets=np.array([0, 32], dtype=np.int64),
        source_cutovers=np.array([0, 16], dtype=np.int64),
        valid_starts=np.array([0, 7], dtype=np.int64),
        valid_stops=np.array([64, 61], dtype=np.int64),
        xp=np,
    )
    unpacked = parameters.view(BLOCK_PARAMETERS_DTYPE)
    assert BLOCK_PARAMETERS_DTYPE.itemsize == 88
    assert BLOCK_PARAMETERS_DTYPE.alignment == 8
    np.testing.assert_array_equal(unpacked["output_index"], [10, 20])
    np.testing.assert_array_equal(unpacked["coarse_sample"], [2, -3])
    np.testing.assert_array_equal(unpacked["residual"], [0.125, -0.125])
    np.testing.assert_array_equal(unpacked["source_offset"], [0, 32])
    np.testing.assert_array_equal(unpacked["source_cutover"], [0, 16])
    np.testing.assert_array_equal(unpacked["valid_start"], [0, 7])
    np.testing.assert_array_equal(unpacked["valid_stop"], [64, 61])


@pytest.mark.gpu
@pytest.mark.parametrize(
    (
        "fft_length",
        "mapping",
        "channel_count",
        "bin_offset",
        "raw_channels",
        "conjugated",
    ),
    [
        (5, "identity", 3, 0, [0, 1, 2], [False, False, False]),
        (5, "identity", 5, 0, [0, 1, 2, 3, 4], [False] * 5),
        (5, "real_lsb", 3, -3, [2, 3, 4], [False, False, False]),
        (
            5,
            "complex_lsb",
            5,
            -5,
            [0, 4, 3, 2, 1],
            [False, True, True, True, True],
        ),
        (5, "complex_dsb", 5, -2, [3, 4, 0, 1, 2], [False] * 5),
        (6, "complex_dsb", 6, -3, [3, 4, 5, 0, 1, 2], [False] * 6),
    ],
)
def test_fused_channel_mapping_and_residual_phase(
    fft_length, mapping, channel_count, bin_offset, raw_channels, conjugated
):
    cupy = _cupy()
    raw = (
        np.arange(2 * fft_length, dtype=np.float32).reshape(2, fft_length)
        + 1j
        * np.arange(31, 31 + 2 * fft_length, dtype=np.float32).reshape(
            2, fft_length
        )
    ).astype(np.complex64)
    residual = np.array([0.125, -0.0625], dtype=np.float64)
    parameters = prepare_block_parameters(
        output_indices=np.zeros(2, dtype=np.int64),
        coarse_samples=np.zeros(2, dtype=np.int64),
        block_starts=np.zeros(2),
        delay_zero=np.zeros(2),
        delay_step=np.zeros(2),
        delay_quadratic=np.zeros(2),
        sample_rate_hz=1.0,
        centre_delay=-residual,
        xp=cupy,
    )
    transformed = cupy.asarray(raw)
    actual = channelize_and_apply_residual(
        transformed,
        parameters,
        mapping=mapping,
        channel_count=channel_count,
        baseband_bin_offset=bin_offset,
        channel_spacing_hz=2.0,
        xp=cupy,
    )

    selected = raw[:, raw_channels].copy()
    selected[:, np.asarray(conjugated)] = np.conj(
        selected[:, np.asarray(conjugated)]
    )
    frequencies = (bin_offset + np.arange(channel_count)) * 2.0
    cycles = residual[:, None] * frequencies[None, :]
    cycles -= np.floor(cycles + 0.5)
    expected = selected * np.exp(2j * np.pi * cycles)
    assert actual.flags.c_contiguous
    assert actual.dtype == cupy.complex64
    np.testing.assert_allclose(
        cupy.asnumpy(actual), expected, rtol=2e-6, atol=2e-6
    )
    if mapping == "identity" and channel_count == fft_length:
        assert actual.data.ptr == transformed.data.ptr
    else:
        assert actual.data.ptr != transformed.data.ptr


@pytest.mark.gpu
def test_fused_residual_phase_recurrence_matches_direct_phasors():
    cupy = _cupy()
    fft_length = 16384
    generator = np.random.default_rng(20260805)
    raw = (
        generator.normal(size=(2, fft_length))
        + 1j * generator.normal(size=(2, fft_length))
    ).astype(np.complex64)
    residual = np.array([1.23456789e-7, -3.45678901e-7], dtype=np.float64)
    parameters = prepare_block_parameters(
        output_indices=np.zeros(2, dtype=np.int64),
        coarse_samples=np.zeros(2, dtype=np.int64),
        block_starts=np.zeros(2),
        delay_zero=np.zeros(2),
        delay_step=np.zeros(2),
        delay_quadratic=np.zeros(2),
        sample_rate_hz=1.0,
        centre_delay=-residual,
        xp=cupy,
    )
    actual = channelize_and_apply_residual(
        cupy.asarray(raw),
        parameters,
        mapping="identity",
        channel_count=fft_length,
        baseband_bin_offset=-fft_length // 2,
        channel_spacing_hz=125_000.0,
        xp=cupy,
    )
    frequencies = (
        -fft_length // 2 + np.arange(fft_length, dtype=np.float64)
    ) * 125_000.0
    cycles = residual[:, None] * frequencies[None, :]
    cycles -= np.floor(cycles + 0.5)
    expected = raw * np.exp(2j * np.pi * cycles)
    np.testing.assert_allclose(
        cupy.asnumpy(actual), expected, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("normalization", [np.nan, np.inf])
@pytest.mark.parametrize("batched", [False, True])
def test_visibility_rejects_nonfinite_normalization(
    normalization, batched, monkeypatch
):
    from cuphoton.xbeam.core import gpu_products

    def unexpected_kernel(*args):
        pytest.fail("non-finite normalization reached the raw kernel")

    for name in ("_visibility_kernel", "_batched_visibility_kernel"):
        monkeypatch.setattr(gpu_products, name, unexpected_kernel)
    spectrum = np.ones((1, 3), np.complex64)
    arguments = dict(
        normalization=normalization,
        channels_to_average=1,
        weight_a=np.ones(1, np.float64),
        xp=np,
    )
    helper = visibility_product
    if batched:
        helper = batched_visibility_product
        arguments.update(
            group_starts=[0],
            group_lengths=[1],
            output_offsets=[0],
            weight_offsets=[0],
            output=np.empty(3, np.complex64),
            weight_sum=np.empty(1, np.float64),
        )
    with pytest.raises(ValueError, match="normalization.*finite"):
        helper(spectrum, spectrum, **arguments)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("separate_weights", [False, True])
@pytest.mark.parametrize("preallocated", [False, True])
def test_raw_visibility_kernel_fuses_product_reduction_and_averaging(
    dtype, separate_weights, preallocated
):
    cupy = _cupy()
    generator = np.random.default_rng(1234)
    first = (
        generator.normal(size=(5, 8)) + 1j * generator.normal(size=(5, 8))
    ).astype(dtype)
    second = (
        generator.normal(size=(5, 8)) + 1j * generator.normal(size=(5, 8))
    ).astype(dtype)
    normalization = 7.25
    weights = cupy.linspace(0.25, 1.0, first.shape[0], dtype=cupy.float64)
    second_weights = cupy.linspace(
        1.0, 0.5, first.shape[0], dtype=cupy.float64
    )
    output = cupy.empty(4, dtype=dtype) if preallocated else None
    weight_output = (
        cupy.empty(1, dtype=cupy.float64) if preallocated else None
    )
    actual, weight_sum = visibility_product(
        cupy.asarray(first),
        cupy.asarray(second),
        normalization=normalization,
        channels_to_average=2,
        weight_a=weights,
        weight_b=second_weights if separate_weights else None,
        xp=cupy,
        output=output,
        weight_sum=weight_output,
    )
    if preallocated:
        assert actual is output
        assert weight_sum is weight_output
    assert actual.dtype == cupy.dtype(dtype)
    expected = np.sum(first * np.conj(second), axis=0) / normalization
    expected = expected.reshape(4, 2).mean(axis=1)
    tolerance = 2e-6 if dtype is np.complex64 else 1e-13
    np.testing.assert_allclose(
        cupy.asnumpy(actual), expected, rtol=tolerance, atol=tolerance
    )
    expected_weights = cupy.asnumpy(weights)
    if separate_weights:
        expected_weights *= cupy.asnumpy(second_weights)
    np.testing.assert_allclose(
        cupy.asnumpy(weight_sum), np.sum(expected_weights), rtol=0, atol=1e-12
    )


@pytest.mark.gpu
@pytest.mark.parametrize("channels_to_average", [1, 2, 4, 32])
@pytest.mark.parametrize("separate_weights", [False, True])
def test_batched_visibility_kernel_scatter_matches_group_reductions(
    channels_to_average, separate_weights
):
    cupy = _cupy()
    generator = np.random.default_rng(5678)
    input_channels = 64 if channels_to_average == 32 else 8
    first = (
        generator.normal(size=(6, input_channels))
        + 1j * generator.normal(size=(6, input_channels))
    ).astype(np.complex64)
    second = (
        generator.normal(size=(6, input_channels))
        + 1j * generator.normal(size=(6, input_channels))
    ).astype(np.complex64)
    weights_a = np.linspace(0.25, 1.0, 6, dtype=np.float64)
    weights_b = np.linspace(1.0, 0.5, 6, dtype=np.float64)
    starts = np.asarray([0, 2, 5], dtype=np.int32)
    lengths = np.asarray([2, 3, 1], dtype=np.int32)
    output_channels = first.shape[1] // channels_to_average
    output_offsets = np.asarray(
        [1, 1 + output_channels + 2, 1 + 2 * output_channels + 3],
        dtype=np.int64,
    )
    weight_offsets = np.asarray([2, 5, 7], dtype=np.int64)
    output = cupy.full(
        int(output_offsets[-1]) + output_channels + 1,
        cupy.nan + 1j * cupy.nan,
        dtype=cupy.complex64,
    )
    weight_output = cupy.full(9, cupy.nan, dtype=cupy.float64)
    normalization = 7.25

    batched_visibility_product(
        cupy.asarray(first),
        cupy.asarray(second),
        normalization=normalization,
        channels_to_average=channels_to_average,
        weight_a=cupy.asarray(weights_a),
        weight_b=cupy.asarray(weights_b) if separate_weights else None,
        group_starts=cupy.asarray(starts),
        group_lengths=cupy.asarray(lengths),
        output_offsets=cupy.asarray(output_offsets),
        weight_offsets=cupy.asarray(weight_offsets),
        output=output,
        weight_sum=weight_output,
        xp=cupy,
    )

    actual = cupy.asnumpy(output)
    actual_weights = cupy.asnumpy(weight_output)
    for start, length, output_offset, weight_offset in zip(
        starts, lengths, output_offsets, weight_offsets, strict=True
    ):
        expected = (
            np.sum(
                first[start : start + length]
                * np.conj(second[start : start + length]),
                axis=0,
            )
            / normalization
        )
        expected = expected.reshape(
            output_channels, channels_to_average
        ).mean(axis=1)
        np.testing.assert_allclose(
            actual[output_offset : output_offset + output_channels],
            expected,
            rtol=2e-6,
            atol=2e-6,
        )
        expected_weight = weights_a[start : start + length]
        if separate_weights:
            expected_weight = (
                expected_weight * weights_b[start : start + length]
            )
        np.testing.assert_allclose(
            actual_weights[weight_offset],
            np.sum(expected_weight),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.gpu
def test_prepared_visibility_indices_do_not_read_back_device_data(
    monkeypatch,
):
    cupy = _cupy()
    spectra = cupy.ones((2, 8), dtype=cupy.complex64)
    weights = cupy.ones(2, dtype=cupy.float64)
    starts = prepare_visibility_index(
        np.asarray([0], dtype=np.int32),
        device_dtype=cupy.int32,
        xp=cupy,
    )
    lengths = prepare_visibility_index(
        np.asarray([2], dtype=np.int32),
        device_dtype=cupy.int32,
        xp=cupy,
    )
    output_offsets = prepare_visibility_index(
        np.asarray([0], dtype=np.int64),
        device_dtype=cupy.int64,
        xp=cupy,
    )
    weight_offsets = prepare_visibility_index(
        np.asarray([0], dtype=np.int64),
        device_dtype=cupy.int64,
        xp=cupy,
    )
    output = cupy.empty(8, dtype=cupy.complex64)
    weight_output = cupy.empty(1, dtype=cupy.float64)

    def forbidden_readback(*_args, **_kwargs):
        raise AssertionError("prepared indices must not synchronize to host")

    monkeypatch.setattr(cupy, "asnumpy", forbidden_readback)
    batched_visibility_product(
        spectra,
        spectra,
        normalization=1.0,
        channels_to_average=1,
        weight_a=weights,
        group_starts=starts,
        group_lengths=lengths,
        output_offsets=output_offsets,
        weight_offsets=weight_offsets,
        output=output,
        weight_sum=weight_output,
        xp=cupy,
    )
    cupy.cuda.get_current_stream().synchronize()


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("starts", "lengths", "output_offset", "weight_offset", "message"),
    [
        ([1], [2], 0, 0, "group range"),
        ([0], [0], 0, 0, "group ranges"),
        ([0], [2], 1, 0, "output offset"),
        ([0], [2], 0, 1, "weight offset"),
    ],
)
def test_batched_visibility_rejects_out_of_range_indices(
    starts, lengths, output_offset, weight_offset, message
) -> None:
    cupy = _cupy()
    spectra = cupy.ones((2, 4), dtype=cupy.complex64)
    weights = cupy.ones(2, dtype=cupy.float64)

    with pytest.raises(ValueError, match=message):
        batched_visibility_product(
            spectra,
            spectra,
            normalization=1.0,
            channels_to_average=1,
            weight_a=weights,
            group_starts=cupy.asarray(starts, dtype=cupy.int32),
            group_lengths=cupy.asarray(lengths, dtype=cupy.int32),
            output_offsets=cupy.asarray([output_offset], dtype=cupy.int64),
            weight_offsets=cupy.asarray([weight_offset], dtype=cupy.int64),
            output=cupy.empty(4, dtype=cupy.complex64),
            weight_sum=cupy.empty(1, dtype=cupy.float64),
            xp=cupy,
        )


@pytest.mark.gpu
def test_fused_visibility_rejects_out_of_range_offsets() -> None:
    cupy = _cupy()
    spectra = [
        cupy.ones((2, 4), dtype=cupy.complex64),
        cupy.ones((2, 4), dtype=cupy.complex64),
    ]
    weights = cupy.ones(2, dtype=cupy.float64)

    with pytest.raises(ValueError, match="output offset"):
        batched_visibility_products(
            spectra,
            [(0, 1), (1, 0)],
            normalizations=[1.0, 1.0],
            channels_to_average=1,
            weight_pairs=[(weights, weights), (weights, weights)],
            multiply_weights=[True, True],
            group_starts=cupy.asarray([0], dtype=cupy.int32),
            group_lengths=cupy.asarray([2], dtype=cupy.int32),
            output_offsets=np.asarray([[0], [5]], dtype=np.int64),
            weight_offsets=np.asarray([[0], [1]], dtype=np.int64),
            output=cupy.empty(8, dtype=cupy.complex64),
            weight_sum=cupy.empty(2, dtype=cupy.float64),
            xp=cupy,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("channels_to_average", [1, 4, 32])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_product_kernel_reuses_bands_and_scatters(
    channels_to_average, strided
):
    cupy = _cupy()
    generator = np.random.default_rng(9123)
    input_channels = 64 if channels_to_average == 32 else 8
    bands = [
        (
            generator.normal(size=(7, input_channels))
            + 1j * generator.normal(size=(7, input_channels))
        ).astype(np.complex64)
        for _ in range(3)
    ]
    weights = [
        np.linspace(0.2 + index * 0.1, 1.0, 7, dtype=np.float64)
        for index in range(3)
    ]
    product_bands = [(0, 1), (0, 2), (2, 1)]
    normalizations = [7.5, 8.25, 9.0]
    starts = np.asarray([0, 3, 5], np.int32)
    lengths = np.asarray([3, 2, 2], np.int32)
    output_channels = input_channels // channels_to_average
    output_offsets = np.asarray(
        [
            [1, 1 + output_channels, 1 + 2 * output_channels],
            [30, 30 + output_channels, 30 + 2 * output_channels],
            [60, 60 + output_channels, 60 + 2 * output_channels],
        ],
        np.int64,
    )
    weight_offsets = np.asarray([[1, 2, 3], [5, 6, 7], [9, 10, 11]], np.int64)
    output = cupy.full(90, cupy.nan + 1j * cupy.nan, dtype=cupy.complex64)
    weight_output = cupy.full(14, cupy.nan, dtype=cupy.float64)
    if strided:
        device_bands = []
        for value in bands:
            padded = cupy.empty(
                (value.shape[0], value.shape[1] + 5), dtype=cupy.complex64
            )
            view = padded[:, 2 : 2 + value.shape[1]]
            view[...] = cupy.asarray(value)
            assert not view.flags.c_contiguous
            device_bands.append(view)
    else:
        device_bands = [cupy.asarray(value) for value in bands]
    device_weights = [cupy.asarray(value) for value in weights]

    batched_visibility_products(
        device_bands,
        product_bands,
        normalizations=normalizations,
        channels_to_average=channels_to_average,
        weight_pairs=[
            (device_weights[0], device_weights[1]),
            (device_weights[0], device_weights[2]),
            (device_weights[2], device_weights[1]),
        ],
        multiply_weights=[True, False, True],
        group_starts=cupy.asarray(starts),
        group_lengths=cupy.asarray(lengths),
        output_offsets=output_offsets,
        weight_offsets=weight_offsets,
        output=output,
        weight_sum=weight_output,
        xp=cupy,
    )

    actual = cupy.asnumpy(output)
    actual_weights = cupy.asnumpy(weight_output)
    for product, (left, right) in enumerate(product_bands):
        for group, (start, length) in enumerate(
            zip(starts, lengths, strict=True)
        ):
            expected = (
                np.sum(
                    bands[left][start : start + length]
                    * np.conj(bands[right][start : start + length]),
                    axis=0,
                )
                / normalizations[product]
            )
            expected = expected.reshape(
                output_channels, channels_to_average
            ).mean(1)
            offset = output_offsets[product, group]
            np.testing.assert_allclose(
                actual[offset : offset + output_channels],
                expected,
                rtol=2e-6,
                atol=2e-6,
            )
            expected_weight = weights[left][start : start + length]
            if product != 1:
                expected_weight = (
                    expected_weight * weights[right][start : start + length]
                )
            np.testing.assert_allclose(
                actual_weights[weight_offsets[product, group]],
                np.sum(expected_weight),
                rtol=0,
                atol=1e-12,
            )


@pytest.mark.gpu
def test_batched_visibility_avg32_accepts_different_row_strides():
    cupy = _cupy()
    generator = np.random.default_rng(4419)
    left = (
        generator.normal(size=(5, 64)) + 1j * generator.normal(size=(5, 64))
    ).astype(np.complex64)
    right = (
        generator.normal(size=(5, 64)) + 1j * generator.normal(size=(5, 64))
    ).astype(np.complex64)
    left_storage = cupy.empty((5, 69), dtype=cupy.complex64)
    right_storage = cupy.empty((5, 73), dtype=cupy.complex64)
    left_view = left_storage[:, 1:65]
    right_view = right_storage[:, 4:68]
    left_view[...] = cupy.asarray(left)
    right_view[...] = cupy.asarray(right)
    output = cupy.empty(2, dtype=cupy.complex64)
    weight_output = cupy.empty(1, dtype=cupy.float64)

    batched_visibility_product(
        left_view,
        right_view,
        normalization=3.5,
        channels_to_average=32,
        weight_a=cupy.ones(5, dtype=cupy.float64),
        group_starts=np.asarray([0], dtype=np.int32),
        group_lengths=np.asarray([5], dtype=np.int32),
        output_offsets=np.asarray([0], dtype=np.int64),
        weight_offsets=np.asarray([0], dtype=np.int64),
        output=output,
        weight_sum=weight_output,
        xp=cupy,
    )

    expected = np.sum(left * np.conj(right), axis=0) / 3.5
    expected = expected.reshape(2, 32).mean(axis=1)
    np.testing.assert_allclose(
        cupy.asnumpy(output), expected, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(cupy.asnumpy(weight_output), [5.0])


@pytest.mark.gpu
def test_four_station_products_fuse_across_baselines(monkeypatch):
    _cupy()

    job, source = _four_station_job_and_source()
    original_fused = gpu_kernels.batched_visibility_product_packs
    original_single = gpu_kernels.batched_visibility_product
    calls = {"fused": 0, "single": 0}

    def tracked_fused(*args, **kwargs):
        calls["fused"] += 1
        return original_fused(*args, **kwargs)

    def tracked_single(*args, **kwargs):
        calls["single"] += 1
        return original_single(*args, **kwargs)

    monkeypatch.setattr(
        gpu_execution, "batched_visibility_product_packs", tracked_fused
    )
    monkeypatch.setattr(
        gpu_execution, "batched_visibility_product", tracked_single
    )

    records = list(correlate(job, backend="cupy", recording_source=source))

    assert len(records) == 6
    assert calls["fused"] >= 1
    assert calls["single"] == 0


@pytest.mark.gpu
def test_truncated_autocorrelations_match_numpy():
    _cupy()
    base, source = _four_station_job_and_source()
    job = replace(
        base,
        frequencies=(replace(base.frequencies[0], channels_to_average=4),),
        write_autocorrelations=True,
    )

    expected = list(
        correlate(
            job,
            backend="numpy",
            recording_source=source,
            autocorrelation_mode=AutocorrelationMode.TRUNCATION,
        )
    )
    actual = list(
        correlate(
            job,
            backend="cupy",
            recording_source=source,
            autocorrelation_mode=AutocorrelationMode.TRUNCATION,
        )
    )

    assert len(actual) == len(expected) == 10
    for actual_record, expected_record in zip(actual, expected, strict=True):
        assert replace(actual_record, spectrum=None) == replace(
            expected_record, spectrum=None
        )
        np.testing.assert_allclose(
            actual_record.spectrum,
            expected_record.spectrum,
            rtol=3e-5,
            atol=3e-5,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "averaging,channels", [(1, 35), (4, 36), (32, 64), (3, 36)]
)
def test_auto_product_groups_match_singles_with_strided_rows(
    monkeypatch, averaging, channels
):
    cp = _cupy()

    rng = np.random.default_rng(521)
    host = (
        rng.normal(size=(5, 7, channels + 8))
        + 1j * rng.normal(size=(5, 7, channels + 8))
    ).astype(np.complex64)
    device = cp.asarray(host)
    weights = cp.asarray(rng.choice([0.0, 0.25, 0.75, 1.0], size=(5, 7)))
    blocks = [
        runtime_types._StationBlocks(
            device[i, :, 4 : 4 + channels], weights[i], np.arange(channels)
        )
        for i in range(5)
    ]
    pairs = [(0, 0), (1, 1), (0, 1), (2, 3), (4, 4)]
    output_channels = channels // averaging
    arena = SimpleNamespace(
        spectra=cp.full(10 * output_channels, cp.nan, dtype=cp.complex64),
        weights=cp.full(10, cp.nan, dtype=cp.float64),
    )
    starts = np.array([0, 3], dtype=np.int32)
    lengths = np.array([3, 4], dtype=np.int32)
    items = [
        gpu_execution._VisibilityProduct(
            blocks_a=blocks[a],
            blocks_b=blocks[b],
            reported_weight_a=blocks[a].weight,
            reported_weight_b=blocks[b].weight,
            boundary=(0, 0),
            normalization=2.5 + i,
            averaging=averaging,
            multiply_weights=a != b,
            output_offsets=[(2 * i + g) * output_channels for g in range(2)],
            weight_offsets=[2 * i, 2 * i + 1],
        )
        for i, (a, b) in enumerate(pairs)
    ]
    counts = {"fused": 0, "single": 0}
    fused = gpu_kernels.batched_visibility_product_packs

    def track_fused(*args, **kwargs):
        counts["fused"] += 1
        return fused(*args, **kwargs)

    monkeypatch.setattr(
        gpu_execution, "batched_visibility_product_packs", track_fused
    )

    def single(
        a, b, key_a, key_b, offsets, weight_offsets, norm, avg, multiply
    ):
        counts["single"] += 1
        batched_visibility_product(
            a.spectrum,
            b.spectrum,
            normalization=norm,
            channels_to_average=avg,
            weight_a=a.weight,
            weight_b=b.weight if multiply else None,
            group_starts=starts,
            group_lengths=lengths,
            output_offsets=offsets,
            weight_offsets=weight_offsets,
            output=arena.spectra,
            weight_sum=arena.weights,
            xp=cp,
        )

    gpu_execution._launch_visibility_groups(
        {(): items}, 2, {(0, 0): (starts, lengths)}, arena, cp, 4, single
    )
    assert counts == (
        {"fused": 0, "single": 5}
        if averaging == 3
        else {"fused": 1, "single": 1}
    )
    actual = cp.asnumpy(arena.spectra).copy()
    actual_weights = cp.asnumpy(arena.weights).copy()
    # Compare the identical products through the individual launch path.
    for item in items:
        single(
            item.blocks_a,
            item.blocks_b,
            (0, 0),
            (0, 0),
            item.output_offsets,
            item.weight_offsets,
            item.normalization,
            averaging,
            item.multiply_weights,
        )
    np.testing.assert_allclose(
        actual, cp.asnumpy(arena.spectra), rtol=3e-6, atol=3e-6
    )
    np.testing.assert_array_equal(actual_weights, cp.asnumpy(arena.weights))
    for i, (a, b) in enumerate(pairs):
        for g, (start, length) in enumerate(
            zip(starts, lengths, strict=True)
        ):
            left = host[a, start : start + length, 4 : 4 + channels]
            right = host[b, start : start + length, 4 : 4 + channels]
            expected = (left * np.conj(right)).sum(axis=0) / (2.5 + i)
            expected = expected.reshape(-1, averaging).mean(axis=1)
            offset = (2 * i + g) * output_channels
            np.testing.assert_allclose(
                actual[offset : offset + output_channels],
                expected,
                rtol=3e-6,
                atol=3e-6,
            )
    np.testing.assert_array_equal(cp.asnumpy(device), host)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", list(AutocorrelationMode))
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("fail_once", [False, True])
@pytest.mark.parametrize("dual_polarization", [False, True])
def test_auto_batching_reuses_host_layout_across_tiles(
    monkeypatch, mode, tiled, fail_once, dual_polarization
):
    cp = _cupy()

    base, source = _four_station_job_and_source()
    job = replace(
        base,
        write_autocorrelations=True,
        integration_duration_s=Decimal(4),
        subintegration_duration_s=Decimal(1),
        frequencies=(replace(base.frequencies[0], channels_to_average=4),),
    )
    if dual_polarization:
        job = replace(
            job,
            datastreams=tuple(
                replace(
                    stream,
                    recorded_bands=(
                        RecordedBand(0, 0, "R"),
                        RecordedBand(1, 0, "L"),
                    ),
                )
                for stream in job.datastreams
            ),
            baselines=tuple(
                replace(
                    baseline,
                    products=tuple(
                        BaselineProduct(a, b, 0)
                        for a in (0, 1)
                        for b in (0, 1)
                    ),
                )
                for baseline in job.baselines
            ),
        )

        class DualSource:
            def batches(self, datastream):
                for band in datastream.recorded_bands:
                    rng = np.random.default_rng(
                        20 * datastream.identifier + band.identifier
                    )
                    samples = rng.normal(size=2048).astype(np.float32)
                    validity = rng.random(2048) > 0.2
                    validity[:256] = False
                    yield VoltageBatch(
                        samples,
                        job.start_time,
                        256.0,
                        band.identifier,
                        band.polarization,
                        validity,
                    )

        source = DualSource()
    expected = list(
        correlate(
            job,
            backend="numpy",
            recording_source=source,
            autocorrelation_mode=mode,
        )
    )
    layouts = []
    build = gpu_execution._build_visibility_layout
    fused = gpu_kernels.batched_visibility_product_packs
    auto_packs = []
    failed = False

    def track_build(*args):
        result = build(*args)
        layouts.append(result)
        return result

    def track_fused(packs, **kwargs):
        nonlocal failed
        auto_packs.extend(p for p in packs if not all(p["multiply_weights"]))
        result = fused(packs, **kwargs)
        if tiled and fail_once and not failed:
            failed = True
            # Fail after a real GPU write: retry must discard this tile.
            raise cp.cuda.memory.OutOfMemoryError(1, 1)
        return result

    monkeypatch.setattr(engine, "_build_visibility_layout", track_build)
    monkeypatch.setattr(
        gpu_execution, "_build_visibility_layout", track_build
    )
    monkeypatch.setattr(
        gpu_execution, "batched_visibility_product_packs", track_fused
    )
    actual = list(
        correlate(
            job,
            backend="cupy",
            recording_source=source,
            autocorrelation_mode=mode,
            subintegration_tiling="auto" if tiled else "off",
            max_tile_subintegrations=3 if tiled else None,
            batch_blocks=1,
        )
    )
    assert len(layouts) == 1
    assert auto_packs
    assert failed == (tiled and fail_once)
    # Only immutable host identities and scalar policies are retained.
    for spec in layouts[0]:
        assert isinstance(spec.band_a, tuple)
        assert isinstance(spec.normalization, float)
        assert isinstance(spec.averaging, int)
    assert len(actual) == len(expected)
    for record, reference in zip(actual, expected, strict=True):
        assert replace(record, spectrum=None) == replace(
            reference, spectrum=None
        )
        np.testing.assert_allclose(
            record.spectrum, reference.spectrum, rtol=3e-5, atol=3e-5
        )


@pytest.mark.gpu
def test_segmented_station_models_match_numpy_across_boundary(monkeypatch):
    _cupy()

    job, source = _four_station_job_and_source()
    first_models = job.scans[0].station_models
    second_models = tuple(
        replace(
            model,
            epoch_offset_s=Decimal(4),
            delay_coefficients_s=(model.station_id / 256.0,),
            u_coefficients_m=(float(model.station_id + 10),),
        )
        for model in first_models
    )
    scan = replace(job.scans[0], station_models=first_models + second_models)
    segmented = replace(job, scans=(scan,))
    model_calls = 0
    original_model = compute_plan._model

    def tracked_model(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        return original_model(*args, **kwargs)

    monkeypatch.setattr(reference_execution, "_model", tracked_model)
    monkeypatch.setattr(output_records, "_model", tracked_model)

    expected = list(
        correlate(segmented, backend="numpy", recording_source=source)
    )
    actual = list(
        correlate(segmented, backend="cupy", recording_source=source)
    )

    assert len(actual) == len(expected)
    for actual_record, expected_record in zip(actual, expected, strict=True):
        assert replace(actual_record, spectrum=None) == replace(
            expected_record, spectrum=None
        )
        np.testing.assert_allclose(
            actual_record.spectrum,
            expected_record.spectrum,
            rtol=2e-5,
            atol=2e-5,
        )
    assert model_calls == 2 * len(job.stations)


@pytest.mark.gpu
def test_later_vdif_upload_overlaps_first_station_processing(
    tmp_path, monkeypatch
):
    _cupy()

    job, source = _packed_two_station_source(tmp_path)
    original_upload = ingest_scheduler._upload_vdif_plan_group
    original_station = gpu_execution._station_blocks
    second_upload_started = threading.Event()
    allow_second_upload = threading.Event()
    upload_count = 0

    def tracked_upload(*args, **kwargs):
        nonlocal upload_count
        upload_count += 1
        if upload_count == 2:
            second_upload_started.set()
            if not allow_second_upload.wait(timeout=5):
                raise AssertionError(
                    "station processing did not overlap upload"
                )
        return original_upload(*args, **kwargs)

    first_station_seen = False

    def tracked_station(job_arg, data, *args, **kwargs):
        nonlocal first_station_seen
        if not first_station_seen:
            first_station_seen = True
            assert second_upload_started.wait(timeout=5)
            allow_second_upload.set()
        return original_station(job_arg, data, *args, **kwargs)

    monkeypatch.setattr(engine, "_upload_vdif_plan_group", tracked_upload)
    monkeypatch.setattr(
        ingest_scheduler, "_upload_vdif_plan_group", tracked_upload
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks", tracked_station)

    records = list(correlate(job, backend="cupy", recording_source=source))

    assert len(records) == 1
    assert upload_count == 2
    assert first_station_seen


@pytest.mark.gpu
def test_ordered_posix_plans_use_frontier_uploads(tmp_path, monkeypatch):
    _cupy()

    base_job, source = _packed_two_station_source(tmp_path)
    job = replace(
        base_job,
        integration_duration_s=Decimal(4),
        subintegration_duration_s=Decimal(4),
    )
    expected = list(correlate(job, backend="numpy", recording_source=source))
    original_upload = ingest_scheduler._upload_vdif_plan_group
    upload_count = 0

    def tracked_upload(*args, **kwargs):
        nonlocal upload_count
        upload_count += 1
        return original_upload(*args, **kwargs)

    monkeypatch.setattr(engine, "_upload_vdif_plan_group", tracked_upload)
    monkeypatch.setattr(
        ingest_scheduler, "_upload_vdif_plan_group", tracked_upload
    )

    actual = list(
        correlate(
            job,
            backend="cupy",
            recording_source=source,
            vdif_upload_workers=1,
        )
    )

    assert upload_count == 4
    assert len(actual) == len(expected) == 2
    for actual_record, expected_record in zip(actual, expected, strict=True):
        assert replace(actual_record, spectrum=None) == replace(
            expected_record, spectrum=None
        )
        np.testing.assert_allclose(
            actual_record.spectrum,
            expected_record.spectrum,
            rtol=3e-5,
            atol=3e-5,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("authoritative_event", [False, True])
def test_correlate_dispatches_packed_multichannel_station_batches(
    tmp_path, monkeypatch, authoritative_event
):
    _cupy()

    job, source = _packed_multichannel_two_station_source(tmp_path)
    options = {}
    if authoritative_event:
        original_source = source

        class Source:
            def batches(self, stream):
                for batch in original_source.batches(stream):
                    if stream.identifier == 1:
                        batch.validity[:16] = False
                    yield batch

        source = Source()
        plan = GlobalFFTEligibilityPlan(
            (),
            (),
            (),
            (),
            "buffered-window-replay-v1",
            timeline_fft_count=4,
            source_validity_events=((0, 0, 256),),
            source_authoritative_validity_events=(0,),
        )
        options = {
            "profile": "interop-real-usb-buffered-window-replay-v1",
            "recovery_plans": {
                ("scan", stream.identifier, band.identifier): plan
                for stream in job.datastreams
                for band in stream.recorded_bands
            },
        }
    expected = list(
        correlate(job, backend="numpy", recording_source=source, **options)
    )
    original = gpu_execution._station_blocks_packed_real_group_for_scan
    calls = []

    def tracked(job_arg, group, *args, **kwargs):
        calls.append(len(group))
        return original(job_arg, group, *args, **kwargs)

    monkeypatch.setattr(
        gpu_execution,
        "_station_blocks_packed_real_group_for_scan",
        tracked,
    )
    actual = list(
        correlate(job, backend="cupy", recording_source=source, **options)
    )

    assert calls == ([] if authoritative_event else [2, 2])
    assert len(actual) == len(expected) == 2
    for actual_record, expected_record in zip(actual, expected, strict=True):
        assert actual_record.weight == expected_record.weight == 1
        assert replace(actual_record, spectrum=None) == replace(
            expected_record, spectrum=None
        )
        np.testing.assert_allclose(
            actual_record.spectrum,
            expected_record.spectrum,
            rtol=3e-5,
            atol=3e-5,
        )


@pytest.mark.parametrize("starts", ([0.0, 1.0], [1.0, 2.0]))
def test_eligibility_leaves_unaffected_station_buffers_untouched(
    monkeypatch, starts
):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    blocks = runtime_types._StationBlocks(
        np.ones((2, 32), dtype=np.complex64),
        np.ones(2),
        np.arange(32),
        np.full(2, 0.5),
    )
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=4,
        ineligible_intervals=((3, 4),),
    )
    actual = gpu_execution._apply_fft_eligibility(
        blocks,
        plan=plan,
        scan=SimpleNamespace(start_offset_s=Decimal(0)),
        data=SimpleNamespace(sample_rate_hz=64.0),
        block_starts=np.asarray(starts),
        xp=np,
    )
    assert actual is blocks


def test_eligibility_masks_spectrum_and_both_station_weights(monkeypatch):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    blocks = runtime_types._StationBlocks(
        np.ones((2, 32), dtype=np.complex64),
        np.ones(2),
        np.arange(32),
        np.full(2, 0.5),
    )
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=4,
        ineligible_intervals=((3, 4),),
    )
    actual = gpu_execution._apply_fft_eligibility(
        blocks,
        plan=plan,
        scan=SimpleNamespace(start_offset_s=Decimal(0)),
        data=SimpleNamespace(sample_rate_hz=64.0),
        block_starts=np.asarray([2.0, 3.0]),
        xp=np,
    )
    np.testing.assert_array_equal(actual.spectrum[0], blocks.spectrum[0])
    np.testing.assert_array_equal(actual.spectrum[1], 0)
    np.testing.assert_array_equal(actual.weight, [1.0, 0.0])
    np.testing.assert_array_equal(actual.reported_weight, [0.5, 0.0])
    np.testing.assert_array_equal(blocks.spectrum, 1)
    np.testing.assert_array_equal(blocks.weight, 1)
    np.testing.assert_array_equal(blocks.reported_weight, 0.5)


def test_packed_recovery_group_only_falls_back_in_replay_window(monkeypatch):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    data = SimpleNamespace(sample_rate_hz=64.0)
    group = [((0, 0), data), ((0, 1), data)]
    scan = SimpleNamespace(identifier="scan", start_offset_s=Decimal(0))
    replay = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=8,
        source_replay_events=((3, 32, 7),),
    )
    plans = {("scan", 0, 0): replay}

    assert not gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([0.0, 1.0, 2.0])
    )
    assert gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([2.0, 3.0, 4.0])
    )


def test_packed_recovery_group_accepts_identity_plans(monkeypatch):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    data = SimpleNamespace(sample_rate_hz=64.0)
    group = [((0, 0), data), ((0, 1), data)]
    scan = SimpleNamespace(identifier="scan", start_offset_s=Decimal(0))
    identity = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=8,
    )
    plans = {
        ("scan", 0, 0): identity,
        ("scan", 0, 1): identity,
    }

    assert not gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([0.0, 1.0, 2.0])
    )


@pytest.mark.parametrize("per_fft", [False, True])
def test_packed_recovery_group_falls_back_for_authoritative_validity(
    monkeypatch,
    per_fft,
):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    data = SimpleNamespace(sample_rate_hz=64.0)
    group = [((0, 0), data), ((0, 1), data)]
    scan = SimpleNamespace(identifier="scan", start_offset_s=Decimal(0))
    authoritative = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=8,
        source_validity_is_authoritative=not per_fft,
        source_validity_events=((1, 0, 64),) if per_fft else (),
        source_authoritative_validity_events=(1,) if per_fft else (),
    )
    plans = {("scan", 0, 0): authoritative}

    assert gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([0.0, 1.0, 2.0])
    )
    assert gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([3.0, 4.0])
    ) is (not per_fft)


def test_packed_recovery_group_falls_back_for_partial_sample_window(
    monkeypatch,
):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    data = SimpleNamespace(sample_rate_hz=64.0)
    group = [((0, 0), data), ((0, 1), data)]
    scan = SimpleNamespace(identifier="scan", start_offset_s=Decimal(0))
    partial = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=8,
        source_validity_events=((3, 7, 64),),
    )
    plans = {("scan", 0, 0): partial}

    assert not gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([0.0, 1.0, 2.0])
    )
    assert gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([2.0, 3.0, 4.0])
    )


def test_station_blocks_for_scan_sums_piecewise_source_segments(monkeypatch):
    fft_samples = 32
    frequency = SimpleNamespace(channel_count=fft_samples // 2)
    datastream = SimpleNamespace(
        sampling=SamplingKind.REAL,
        station_id=0,
    )
    data = SimpleNamespace(
        frequency=frequency,
        datastream=datastream,
        sample_rate_hz=float(fft_samples),
    )
    scan = SimpleNamespace(start_offset_s=0.0)
    starts = np.asarray((0.0, 1.0))
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=2,
        source_segment_events=((1, ((0, 8, 0), (16, 32, -64))),),
    )
    monkeypatch.setattr(
        gpu_execution,
        "_ordered_model_runs",
        lambda *_args: ((SimpleNamespace(), slice(0, 2)),),
    )
    calls = []

    def station_blocks(
        _job,
        _data,
        _model,
        selected_starts,
        _xp,
        source_offsets,
        _source_cutovers,
        valid_starts,
        valid_stops,
        _credits,
        _reported_credits,
        _authoritative_validity,
        source_timing_offsets=None,
    ):
        assert source_timing_offsets is None
        calls.append(
            (
                selected_starts.copy(),
                source_offsets.copy(),
                valid_starts.copy(),
                valid_stops.copy(),
            )
        )
        value = float(len(calls))
        return runtime_types._StationBlocks(
            np.full((selected_starts.size, frequency.channel_count), value),
            (valid_stops - valid_starts) / fft_samples,
            np.arange(frequency.channel_count),
        )

    monkeypatch.setattr(gpu_execution, "_station_blocks", station_blocks)
    result = gpu_execution._station_blocks_for_scan(
        SimpleNamespace(), data, scan, starts, np, plan
    )

    np.testing.assert_array_equal(result.spectrum[0], 1.0)
    np.testing.assert_array_equal(result.spectrum[1], 5.0)
    np.testing.assert_allclose(result.weight, (1.0, 0.75))
    assert [call[1].tolist() for call in calls] == [[0], [0], [-64]]


@pytest.mark.gpu
def test_packed_multichannel_station_batch_recovers_from_oom(
    tmp_path, monkeypatch
):
    cupy = _cupy()

    job, source = _packed_multichannel_two_station_source(tmp_path)
    expected = list(correlate(job, backend="numpy", recording_source=source))
    original_station = gpu_execution._station_blocks
    fallback_calls = 0

    def fail_batch(*args, **kwargs):
        raise cupy.cuda.memory.OutOfMemoryError(1, 1)

    def tracked_station(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return original_station(*args, **kwargs)

    monkeypatch.setattr(
        gpu_execution, "_station_blocks_packed_real_group", fail_batch
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks", tracked_station)

    actual = list(correlate(job, backend="cupy", recording_source=source))

    assert fallback_calls == 4
    assert len(actual) == len(expected) == 2
    for actual_record, expected_record in zip(actual, expected, strict=True):
        assert replace(actual_record, spectrum=None) == replace(
            expected_record, spectrum=None
        )
        np.testing.assert_allclose(
            actual_record.spectrum,
            expected_record.spectrum,
            rtol=3e-5,
            atol=3e-5,
        )


@pytest.mark.gpu
def test_late_vdif_upload_failure_is_propagated(tmp_path, monkeypatch):
    _cupy()

    job, source = _packed_two_station_source(tmp_path)
    original_upload = ingest_scheduler._upload_vdif_plan_group

    def fail_after_upload(*args, **kwargs):
        original_upload(*args, **kwargs)
        raise OSError("late upload failure")

    monkeypatch.setattr(engine, "_upload_vdif_plan_group", fail_after_upload)
    monkeypatch.setattr(
        ingest_scheduler, "_upload_vdif_plan_group", fail_after_upload
    )

    with pytest.raises(OSError, match="late upload failure"):
        list(correlate(job, backend="cupy", recording_source=source))


@pytest.mark.gpu
@pytest.mark.parametrize("quadratic", [0.0, 1e-23, -1e-23])
@pytest.mark.parametrize("rows", [1, 8])
def test_long_front_rotation_preserves_phase(quadratic, rows):
    """Long FFTs must not accumulate float32 phase drift."""
    cupy = _cupy()
    n, fs, edge = 262144, 4096000000.0, 210000000000.0
    delay = 0.0123456789012345
    step = 7.125e-7 / fs
    parameters = prepare_block_parameters(
        output_indices=np.zeros(rows, np.int64),
        coarse_samples=np.zeros(rows, np.int64),
        block_starts=np.zeros(rows),
        delay_zero=np.full(rows, delay),
        delay_step=np.full(rows, step),
        delay_quadratic=np.full(rows, quadratic),
        sample_rate_hz=fs,
        centre_delay=np.full(rows, delay),
        xp=cupy,
    )
    rotated, weights = front_rotate(
        source=cupy.ones(n, dtype=cupy.float32),
        source_validity=cupy.ones(n, dtype=cupy.uint8),
        source_start_sample=0,
        block_parameters=parameters,
        fft_length=n,
        sample_rate_hz=fs,
        fringe_reference_hz=edge,
        lo_offset_hz=0.0,
        wall_lo_cycles=0.0,
        complex_input=False,
        xp=cupy,
    )
    # Independent high-precision constant phase and direct per-sample phasors.
    from decimal import localcontext

    with localcontext() as ctx:
        ctx.prec = 60
        constant = float(
            (-Decimal.from_float(edge) * Decimal.from_float(delay)) % 1
        )
    indices = np.arange(n, dtype=np.float64)
    cycles = constant - edge * step * indices - edge * quadratic * indices**2
    expected = np.exp(2j * np.pi * cycles)
    actual = cupy.asnumpy(rotated[0])
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-5
    assert np.max(abs(actual - expected)) < 2e-5
    np.testing.assert_array_equal(cupy.asnumpy(weights), np.ones(rows))


@pytest.mark.parametrize(
    ("active", "enclosing", "expected"),
    [
        (1, 4350, 1),
        (2, 4350, 2),
        (3, 4350, 4),
        (1025, 4350, 2048),
        (2174, 4350, 4096),
        (2175, 4350, 4350),
        (2617, 4350, 4350),
        (4350, 4350, 4350),
        (949, 1900, 1024),
        (950, 1900, 1900),
        (1357, 1900, 1900),
    ],
)
def test_compat_fft_batch_capacity(active, enclosing, expected):
    assert gpu_execution._compat_fft_batch_rows(active, enclosing) == expected


@pytest.mark.parametrize(("active", "enclosing"), [(0, 10), (11, 10)])
def test_compat_fft_batch_capacity_rejects_invalid_counts(active, enclosing):
    with pytest.raises(ValueError, match="enclosing batch"):
        gpu_execution._compat_fft_batch_rows(active, enclosing)


@pytest.mark.gpu
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("authoritative", [False, True])
@pytest.mark.parametrize("fft_length", [32, 262144])
def test_padded_front_fft_preserves_active_rows(
    packed, authoritative, fft_length
):
    cupy = _cupy()
    from cupyx.scipy.fft import fft

    n = 3
    parameters = prepare_block_parameters(
        output_indices=np.arange(n, dtype=np.int64) * fft_length,
        coarse_samples=np.zeros(n, dtype=np.int64),
        block_starts=np.arange(n, dtype=np.float64) / 10,
        delay_zero=np.array([0.01, -0.02, 0.03]),
        delay_step=np.array([1e-9, -2e-9, 3e-9]),
        delay_quadratic=np.zeros(n),
        sample_rate_hz=4096000000.0,
        centre_delay=np.zeros(n),
        source_offsets=np.array([0, 1, -1], dtype=np.int64),
        source_cutovers=np.array([0, 8, 16], dtype=np.int64),
        valid_starts=np.array([0, 3, 0], dtype=np.int64),
        valid_stops=np.array([fft_length, fft_length - 2, fft_length]),
        xp=cupy,
    )
    kwargs = dict(
        source_validity=cupy.asarray([1, 0, 1, 1], dtype=cupy.uint8),
        validity_stride=fft_length,
        authoritative_validity=authoritative,
        source_start_sample=0,
        block_parameters=parameters,
        fft_length=fft_length,
        sample_rate_hz=4096000000.0,
        fringe_reference_hz=212052000000.0,
        lo_offset_hz=0.5,
        wall_lo_cycles=0.0,
        complex_input=False,
        xp=cupy,
    )
    if packed:
        kwargs.update(
            source=cupy.full(fft_length, 0xE4, dtype=cupy.uint8),
            source_sample_count=4 * fft_length,
            packed_samples_per_frame=fft_length,
            packed_payload_bytes=fft_length // 4,
            packed_channel_count=1,
            packed_channel_index=0,
        )
    else:
        kwargs["source"] = cupy.asarray(
            np.random.default_rng(73)
            .normal(size=4 * fft_length)
            .astype(np.float32)
        )
    exact, exact_weight = front_rotate(**kwargs)
    padded, padded_weight = front_rotate(**kwargs, _fft_batch_rows=5)
    assert padded.shape == (5, fft_length)
    np.testing.assert_array_equal(cupy.asnumpy(padded[3:]), 0)
    np.testing.assert_array_equal(
        cupy.asnumpy(padded[:3]), cupy.asnumpy(exact)
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(padded_weight), cupy.asnumpy(exact_weight)
    )
    np.testing.assert_allclose(
        cupy.asnumpy(fft(padded, axis=-1, overwrite_x=True)[:3]),
        cupy.asnumpy(fft(exact, axis=-1, overwrite_x=True)),
        rtol=2e-6,
        atol=2e-4,
    )


@pytest.mark.gpu
def test_padded_front_allocation_falls_back_to_active_rows(monkeypatch):
    cupy = _cupy()
    parameters = prepare_block_parameters(
        output_indices=np.array([0, 32], dtype=np.int64),
        coarse_samples=np.zeros(2, dtype=np.int64),
        block_starts=np.zeros(2),
        delay_zero=np.zeros(2),
        delay_step=np.zeros(2),
        delay_quadratic=np.zeros(2),
        sample_rate_hz=64.0,
        centre_delay=np.zeros(2),
        xp=cupy,
    )
    original_empty = cupy.empty
    attempts = []

    def empty(shape, *args, **kwargs):
        attempts.append(shape)
        if shape == (4, 32):
            raise cupy.cuda.memory.OutOfMemoryError(1024, 1024)
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(cupy, "empty", empty)
    rotated, weights = front_rotate(
        source=cupy.ones(64, dtype=cupy.float32),
        source_validity=cupy.ones(64, dtype=cupy.uint8),
        source_start_sample=0,
        block_parameters=parameters,
        fft_length=32,
        sample_rate_hz=64.0,
        fringe_reference_hz=0.0,
        lo_offset_hz=0.0,
        wall_lo_cycles=0.0,
        complex_input=False,
        xp=cupy,
        _fft_batch_rows=4,
    )
    assert attempts[:2] == [(4, 32), (2, 32)]
    np.testing.assert_array_equal(cupy.asnumpy(rotated), 1)
    np.testing.assert_array_equal(cupy.asnumpy(weights), 1)


@pytest.mark.gpu
@pytest.mark.parametrize("scenario", ["filtered", "segments", "all-rejected"])
def test_padded_station_recovery_preserves_spectra_and_reported_weights(
    monkeypatch, scenario
):
    cupy = _cupy()
    epoch = MJDTime(58849, Decimal(0))
    frequency = Frequency(0, 0.0, 128.0, Sideband.USB, 8)
    band = RecordedBand(0, 0, "X")
    datastream = Datastream(
        0,
        0,
        SamplingKind.REAL,
        2,
        40,
        ("unused.vdif",),
        (band,),
        sample_rate_hz=256.0,
    )
    model = DelayCoordinateModel(
        0, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,)
    )
    scan = Scan("scan", Decimal(0), Decimal("0.5"), "0", "0", (model,))
    job = Job(
        start_time=epoch,
        execution_duration_s=Decimal("0.5"),
        integration_duration_s=Decimal("0.5"),
        subintegration_duration_s=Decimal("0.5"),
        fringe_approximation_order=1,
        stations=(Station(0, "AA", ClockPolynomial(epoch, (0.0,))),),
        frequencies=(frequency,),
        datastreams=(datastream,),
        baselines=(),
        scans=(scan,),
    )
    samples = np.random.default_rng(82).normal(size=128).astype(np.float32)
    validity = np.ones(128, dtype=np.uint8)
    validity[64:80] = 0
    data = runtime_types._BandSamples(
        datastream=datastream,
        band=band,
        frequency=frequency,
        samples=samples,
        validity=validity.astype(bool),
        start_sample=0,
        sample_rate_hz=256.0,
        device_samples=cupy.asarray(samples),
        device_validity=cupy.asarray(validity),
        validity_stride=1,
    )
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "independent-test",
        timeline_fft_count=5,
        ineligible_intervals=(
            ((0, 5),) if scenario == "all-rejected" else ((1, 2), (3, 4))
        ),
        source_replay_events=((2, 2, 8), (3, 999999, 0)),
        source_validity_events=((2, 4, 16), (4, 0, 16)),
        source_authoritative_validity_events=(4,),
        source_weight_credit_intervals=((32, 36),),
        source_reported_weight_credit_intervals=(),
        source_segment_events=(
            ((4, ((0, 8, 0), (8, 16, 2))),) if scenario == "segments" else ()
        ),
    )
    starts = np.arange(5, dtype=np.float64) * 16 / 256
    chooser = gpu_execution._compat_fft_batch_rows
    dense_chooser = gpu_execution._use_dense_compat_fft_rows
    monkeypatch.setattr(
        gpu_execution, "_use_dense_compat_fft_rows", lambda *_: False
    )
    monkeypatch.setattr(
        gpu_execution,
        "_compat_fft_batch_rows",
        lambda active, _parent: active,
    )
    exact = gpu_execution._station_blocks_for_scan(
        job, data, scan, starts, cupy, plan
    )
    monkeypatch.setattr(gpu_execution, "_compat_fft_batch_rows", chooser)
    monkeypatch.setattr(
        gpu_execution, "_use_dense_compat_fft_rows", dense_chooser
    )
    padded = gpu_execution._station_blocks_for_scan(
        job, data, scan, starts, cupy, plan
    )
    np.testing.assert_allclose(
        cupy.asnumpy(padded.spectrum),
        cupy.asnumpy(exact.spectrum),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(padded.weight), cupy.asnumpy(exact.weight)
    )
    np.testing.assert_array_equal(padded.baseband_hz, exact.baseband_hz)
    if exact.reported_weight is not None:
        np.testing.assert_array_equal(
            cupy.asnumpy(padded.reported_weight),
            cupy.asnumpy(exact.reported_weight),
        )
    else:
        assert padded.reported_weight is None
    rejected = [0, 1, 2, 3, 4] if scenario == "all-rejected" else [1, 3]
    np.testing.assert_array_equal(cupy.asnumpy(padded.spectrum[rejected]), 0)
    np.testing.assert_array_equal(cupy.asnumpy(padded.weight[rejected]), 0)


@pytest.mark.gpu
@pytest.mark.parametrize("reported_alias", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_sparse_fft_clear_preserves_other_rows_and_views(
    reported_alias, strided
):
    cupy = _cupy()
    rng = np.random.default_rng(902)
    host = (rng.normal(size=(7, 66)) + 1j * rng.normal(size=(7, 66))).astype(
        np.complex64
    )
    backing = cupy.asarray(host)
    spectrum = backing[:, :33] if strided else backing
    weights = cupy.asarray(np.linspace(0.2, 1.0, 7))
    reported = weights if reported_alias else weights.copy() * 0.5
    expected = cupy.asnumpy(spectrum).copy()
    expected[[1, 5]] = 0
    assert gpu_kernels.clear_rejected_fft_rows(
        spectrum, weights, reported, np.array([1, 5]), xp=cupy
    )
    np.testing.assert_array_equal(cupy.asnumpy(spectrum), expected)
    expected_weights = np.linspace(0.2, 1.0, 7)
    expected_weights[[1, 5]] = 0
    np.testing.assert_array_equal(cupy.asnumpy(weights), expected_weights)
    np.testing.assert_array_equal(
        cupy.asnumpy(reported),
        expected_weights if reported_alias else expected_weights * 0.5,
    )
    if strided:
        np.testing.assert_array_equal(
            cupy.asnumpy(backing[:, 33:]), host[:, 33:]
        )


def test_sparse_fft_clear_declines_unsupported_layout_before_mutation():
    spectrum = np.ones((4, 8), np.complex128)
    weight = np.ones(4)
    assert not gpu_kernels.clear_rejected_fft_rows(
        spectrum, weight, None, np.array([1]), xp=np
    )
    np.testing.assert_array_equal(spectrum, 1)
    np.testing.assert_array_equal(weight, 1)


@pytest.mark.gpu
@pytest.mark.parametrize("owned", [False, True])
def test_station_eligibility_sparse_clear_respects_buffer_ownership(
    monkeypatch, owned
):
    cupy = _cupy()
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    blocks = runtime_types._StationBlocks(
        cupy.ones((4, 32), dtype=cupy.complex64),
        cupy.ones(4, dtype=cupy.float64),
        np.arange(32),
        None,
    )
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=4,
        ineligible_intervals=((1, 3),),
    )
    actual = gpu_execution._apply_fft_eligibility(
        blocks,
        plan=plan,
        scan=SimpleNamespace(start_offset_s=Decimal(0)),
        data=SimpleNamespace(sample_rate_hz=64.0),
        block_starts=np.arange(4, dtype=float),
        xp=cupy,
        _owns_buffers=owned,
    )
    np.testing.assert_array_equal(
        cupy.asnumpy(actual.spectrum[:, 0]), [1, 0, 0, 1]
    )
    np.testing.assert_array_equal(cupy.asnumpy(actual.weight), [1, 0, 0, 1])
    assert (actual is blocks) == owned
    if not owned:
        np.testing.assert_array_equal(cupy.asnumpy(blocks.spectrum), 1)
        np.testing.assert_array_equal(cupy.asnumpy(blocks.weight), 1)


@pytest.mark.gpu
@pytest.mark.parametrize("reject_outside", [False, True])
@pytest.mark.parametrize("segment_batch", [False, True])
@pytest.mark.parametrize("whole_replay", [False, True])
def test_packed_multichannel_station_batch_replay_matches_per_band_path(
    reject_outside,
    segment_batch,
    whole_replay,
):
    cupy = _cupy()

    epoch = MJDTime(58849, Decimal(0))
    station = Station(0, "AA", ClockPolynomial(epoch, (0.0,)))
    frequencies = (
        Frequency(0, 0.0, 128.0, Sideband.USB, 8),
        Frequency(1, 128.0, 128.0, Sideband.USB, 8),
    )
    bands = (RecordedBand(0, 0, "X"), RecordedBand(1, 1, "Y"))
    datastream = Datastream(
        0,
        0,
        SamplingKind.REAL,
        2,
        40,
        ("unused.vdif",),
        bands,
        sample_rate_hz=256.0,
    )
    model = DelayCoordinateModel(
        0, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,)
    )
    scan = Scan("scan", Decimal(0), Decimal("0.25"), "0", "0", (model,))
    job = Job(
        start_time=epoch,
        execution_duration_s=Decimal("0.25"),
        integration_duration_s=Decimal("0.25"),
        subintegration_duration_s=Decimal("0.25"),
        fringe_approximation_order=1,
        stations=(station,),
        frequencies=frequencies,
        datastreams=(datastream,),
        baselines=(),
        scans=(scan,),
    )
    payload = cupy.asarray(
        np.array(
            [0xE4, 0x1B, 0x9C, 0x72, 0x35, 0xCA, 0x87, 0x4D] * 4,
            dtype=np.uint8,
        )
    )
    validity = cupy.asarray([1, 1, 0, 1], dtype=cupy.uint8)
    payload_plan = object()
    group = []
    for index, (band, frequency) in enumerate(
        zip(bands, frequencies, strict=True)
    ):
        data = runtime_types._BandSamples(
            datastream=datastream,
            band=band,
            frequency=frequency,
            samples=np.empty(0, dtype=np.float32),
            validity=np.empty(0, dtype=np.bool_),
            start_sample=0,
            sample_rate_hz=256.0,
            device_validity=validity,
            validity_stride=16,
            device_payload=payload,
            source_sample_count=64,
            packed_samples_per_frame=16,
            packed_payload_bytes=8,
            packed_channel_count=2,
            packed_channel_index=index,
            payload_plan=payload_plan,
        )
        group.append(((0, index), data))
    starts = np.asarray([0.0, 0.0625, 0.125], dtype=np.float64)
    plans = None
    if reject_outside:
        cropped_payload = payload[:16]
        cropped_validity = validity[:2]
        for _key, data in group:
            data.device_payload = cropped_payload
            data.device_validity = cropped_validity
            data.source_sample_count = 32
            data.planned_sample_coverage = True
        plans = {
            ("scan", 0, band): GlobalFFTEligibilityPlan(
                (),
                (),
                (),
                (),
                "test",
                timeline_fft_count=4,
                ineligible_intervals=((1 if band == 0 else 2, 3),),
            )
            for band in range(2)
        }

    if whole_replay:
        plans = {
            ("scan", 0, band): replace(
                GlobalFFTEligibilityPlan(
                    (), (), (), (), "test", timeline_fft_count=4
                )
                if plans is None
                else plans[("scan", 0, band)],
                source_replay_events=((0, 16, 0), (1, -8, 0)),
            )
            for band in range(2)
        }

    if segment_batch:
        plans = {
            ("scan", 0, band): replace(
                GlobalFFTEligibilityPlan(
                    (), (), (), (), "test", timeline_fft_count=4
                )
                if plans is None
                else plans[("scan", 0, band)],
                source_segment_events=tuple(
                    (i, ((0, 8, 0), (8, 16, 8 if i == 0 else -8)))
                    for i in (0, 1)
                ),
                source_validity_events=((0, 0, 16),),
                source_authoritative_validity_events=((0,) if band else ()),
            )
            for band in range(2)
        }
    expected = {
        key: gpu_execution._station_blocks_for_scan(
            job,
            data,
            scan,
            starts,
            cupy,
            None if plans is None else plans[("scan", *key)],
        )
        for key, data in group
    }
    if segment_batch:
        for key, data in group:
            pieces = [
                gpu_execution._station_blocks_for_scan(
                    job,
                    data,
                    scan,
                    starts[i : i + 1],
                    cupy,
                    plans[("scan", *key)],
                )
                for i in range(starts.size)
            ]
            np.testing.assert_allclose(
                cupy.asnumpy(expected[key].spectrum),
                cupy.asnumpy(
                    cupy.concatenate([piece.spectrum for piece in pieces])
                ),
                rtol=3e-6,
                atol=3e-6,
            )
            np.testing.assert_array_equal(
                cupy.asnumpy(expected[key].weight),
                cupy.asnumpy(
                    cupy.concatenate([piece.weight for piece in pieces])
                ),
            )
    actual = (
        expected
        if segment_batch
        else gpu_execution._station_blocks_packed_real_group_for_scan(
            job, group, scan, starts, cupy, plans
        )
    )

    for key in expected:
        np.testing.assert_allclose(
            cupy.asnumpy(actual[key].spectrum),
            cupy.asnumpy(expected[key].spectrum),
            rtol=3e-6,
            atol=3e-6,
        )
        np.testing.assert_array_equal(
            cupy.asnumpy(actual[key].weight),
            cupy.asnumpy(expected[key].weight),
        )
        np.testing.assert_array_equal(
            actual[key].baseband_hz, expected[key].baseband_hz
        )


def test_packed_recovery_group_keeps_whole_fft_replay_batched(monkeypatch):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    data = SimpleNamespace(sample_rate_hz=64.0)
    scan = SimpleNamespace(identifier="scan", start_offset_s=Decimal(0))
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=8,
        source_replay_events=((3, 32, 0),),
    )
    assert not gpu_execution._packed_group_requires_source_replay(
        [((0, 0), data)],
        {("scan", 0, 0): plan},
        scan,
        np.asarray([2.0, 3.0, 4.0]),
    )


def test_packed_recovery_group_keeps_rejected_rows_batched(monkeypatch):
    monkeypatch.setattr(gpu_execution, "_fft_length", lambda _data: 64)
    data = SimpleNamespace(sample_rate_hz=64.0)
    group = [((0, 0), data), ((0, 1), data)]
    scan = SimpleNamespace(identifier="scan", start_offset_s=Decimal(0))
    plan = GlobalFFTEligibilityPlan(
        (),
        (),
        (),
        (),
        "test",
        timeline_fft_count=8,
        ineligible_intervals=((3, 5),),
    )
    plans = {("scan", 0, 1): plan}

    assert not gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([0.0, 1.0, 2.0])
    )
    assert not gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([2.0, 3.0, 4.0])
    )
    assert not gpu_execution._packed_group_requires_source_replay(
        group, plans, scan, np.asarray([3.0, 4.0])
    )


@pytest.mark.parametrize("all_empty", [False, True])
@pytest.mark.parametrize(
    "gpu", [False, pytest.param(True, marks=pytest.mark.gpu)]
)
def test_empty_fft_validity_does_not_require_source_coverage(all_empty, gpu):
    xp = _cupy() if gpu else np
    job, _ = _four_station_job_and_source()
    datastream = job.datastreams[0]
    samples = np.random.default_rng(73).normal(size=256).astype(np.float32)
    data = runtime_types._BandSamples(
        datastream=datastream,
        band=datastream.recorded_bands[0],
        frequency=job.frequencies[0],
        samples=samples,
        validity=np.ones(samples.size, dtype=bool),
        start_sample=0,
        sample_rate_hz=256.0,
        device_samples=xp.asarray(samples) if gpu else None,
        device_validity=xp.ones(samples.size, dtype=xp.uint8)
        if gpu
        else None,
    )
    model = job.scans[0].station_models[0]
    control = gpu_execution._station_blocks(
        job, data, model, np.array([0.0]), xp
    )
    result = gpu_execution._station_blocks(
        job,
        data,
        model,
        np.array([0.0, 4.0]),
        xp,
        valid_starts=np.array([0, 0]),
        valid_stops=np.array([0 if all_empty else 256, 0]),
    )
    asnumpy = xp.asnumpy if gpu else np.asarray
    np.testing.assert_array_equal(asnumpy(result.spectrum[1]), 0)
    np.testing.assert_array_equal(
        asnumpy(result.weight), [0 if all_empty else 1, 0]
    )
    if not all_empty:
        np.testing.assert_allclose(
            asnumpy(result.spectrum[0]), asnumpy(control.spectrum[0])
        )
    else:
        np.testing.assert_array_equal(asnumpy(result.spectrum[0]), 0)
    # A single live sample still requires planned source coverage.
    with pytest.raises(RuntimeError, match="planned sample buffer"):
        gpu_execution._station_blocks(
            job,
            data,
            model,
            np.array([4.0]),
            xp,
            valid_starts=np.array([0]),
            valid_stops=np.array([1]),
        )
