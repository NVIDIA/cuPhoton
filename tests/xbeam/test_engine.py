# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import weakref
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cuphoton.xbeam.adapters.radio.engine as engine_module
from cuphoton.xbeam import ingest_scheduler, runtime_types, upload_resources
from cuphoton.xbeam.adapters.radio import (
    compute_plan,
    gpu_execution,
    output_records,
)
from cuphoton.xbeam.adapters.radio.compute_plan import _timing_groups
from cuphoton.xbeam.adapters.radio.engine import (
    CorrelationRun,
    correlate,
    prepare_correlation,
)
from cuphoton.xbeam.adapters.radio.gpu_execution import (
    _automatic_xmac_fusion_width,
)
from cuphoton.xbeam.adapters.radio.models import (
    AutocorrelationMode,
    Baseline,
    BaselineProduct,
    BufferedReaderGeometry,
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
    ZoomBand,
)
from cuphoton.xbeam.adapters.radio.output_records import (
    _gpu_output_group_limit,
)
from cuphoton.xbeam.adapters.vdif.ingest import SampleInterval
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifPayloadFilePlan,
    VdifPayloadPlan,
    VdifSourceRange,
    VdifStreamingPlan,
)
from cuphoton.xbeam.adapters.vdif.vdif import (
    DuplicatePolicy,
    VdifHeader,
    absolute_frame_slot,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.ingest_scheduler import (
    _deduplicate_payload_plan_groups,
    _mapped_frontier_upload_tasks,
    _payload_group_chunk_bytes,
    _payload_upload_staging_plan,
    _PayloadProgress,
    _reusable_vdif_upload_staging_bytes,
    _reusable_vdif_upload_staging_covers,
)
from cuphoton.xbeam.missing_timestamp_recovery import (
    FFTEligibilityCheckpoint,
    GlobalFFTEligibilityPlan,
)
from cuphoton.xbeam.profiles import ANALYTIC_V1, INTEROP_REAL_USB_V1
from cuphoton.xbeam.runtime import backend
from cuphoton.xbeam.sample_loading import _load_band_samples
from cuphoton.xbeam.windowing import CorrelationWindow


@pytest.fixture
def buffered_file_job(tmp_path):
    from tests.xbeam.test_buffered_window_job_requests import model_job
    from tests.xbeam.test_recording import _frame as recording_frame

    path = tmp_path / "source.vdif"
    path.write_bytes(
        b"".join(
            recording_frame(
                payload=bytes([physical % 251]) * 8,
                frame_number=physical % 4,
                seconds_from_epoch=physical // 4,
                channels=1,
            )
            for physical in range(120)
        )
    )
    indexes = tmp_path / "indexes"
    build_vdif_binary_index(path, index_dir=indexes, backend="posix")
    job, ds, band, _frequency, scan, _geometry = model_job(
        (1.4 / 128,), rate=128.0, fft=32
    )
    ds = replace(
        ds,
        frame_size_bytes=40,
        source_files=(str(path),),
        recorded_bands=(replace(band, thread_id=5),),
    )
    scan = replace(scan, duration_s=Decimal(8))
    job = replace(
        job,
        datastreams=(ds,),
        scans=(scan,),
        execution_duration_s=Decimal(8),
        baselines=(Baseline(0, 0, 0, (BaselineProduct(0, 0, 0),)),),
        write_autocorrelations=True,
        buffered_reader=BufferedReaderGeometry(24, 8, 20),
    )
    return job, dict(
        backend="numpy",
        profile="interop-real-usb-buffered-window-replay-v1",
        index_dir=str(indexes),
        max_read_bytes=1 << 24,
    )


@pytest.mark.parametrize("plan_start", [0, 1])
def test_recovery_window_must_contain_execution(
    buffered_file_job, plan_start
):
    job, kwargs = buffered_file_job
    with pytest.raises(ConfigurationError, match="contain every selected"):
        prepare_correlation(
            job,
            window=CorrelationWindow("ordinary", 0, 2),
            recovery_plan_window=CorrelationWindow("ordinary", plan_start, 1),
            **kwargs,
        )


def test_file_recovery_scope_survives_explicit_plan_reuse(buffered_file_job):
    job, kwargs = buffered_file_job
    window = CorrelationWindow("ordinary", 0, 1)
    baseline = prepare_correlation(job, window=window, **kwargs)
    parent = prepare_correlation(
        job,
        window=window,
        recovery_plan_window=CorrelationWindow("ordinary", 0, 2),
        **kwargs,
    )
    assert parent.recovery_plans[("ordinary", 0, 0)].known_fft_scope == (0, 8)
    assert parent.recovery_plans[("ordinary", 0, 0)].source_replay_intervals
    reused = prepare_correlation(
        job, window=window, recovery_plans=parent.recovery_plans, **kwargs
    )
    expected = list(baseline.records())
    assert len(expected) == 2
    for run in (parent, reused):
        actual = list(run.records())
        assert len(actual) == len(expected)
        for wanted, observed in zip(expected, actual, strict=True):
            assert observed.weight == wanted.weight == 1.0
            assert observed.seconds_of_day == wanted.seconds_of_day
            np.testing.assert_array_equal(observed.spectrum, wanted.spectrum)
    later = prepare_correlation(
        job, window=CorrelationWindow("ordinary", 1, 1), **kwargs
    )
    assert later.recovery_plans[("ordinary", 0, 0)].known_fft_scope == (4, 8)
    with pytest.raises(ConfigurationError, match="does not cover execution"):
        prepare_correlation(
            job, window=window, recovery_plans=later.recovery_plans, **kwargs
        )


@pytest.mark.parametrize(
    ("free", "cached", "limit", "used", "expected"),
    [
        (800, 100, 0, 0, 800),
        (800, 100, 600, 550, 50),
        (50, 20, 0, 0, 0),
        (400, 300, 900, 200, 600),
    ],
)
def test_gpu_working_budget_respects_reserve_and_existing_pool_limit(
    free,
    cached,
    limit,
    used,
    expected,
):
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                memGetInfo=lambda: (free, 1000),
            )
        )
    )
    pool = SimpleNamespace(
        free_bytes=lambda: cached,
        get_limit=lambda: limit,
        used_bytes=lambda: used,
    )
    assert engine_module._available_gpu_working_bytes(xp, pool) == expected


@pytest.mark.parametrize("previous", [0, 700, 950])
@pytest.mark.parametrize("previous_cache", [-1, 0, 50, 200])
def test_gpu_pool_headroom_restores_limit_after_failure(
    previous, previous_cache
):
    limits, released, cache_limits = [], [], []
    cache = SimpleNamespace(
        get_memsize=lambda: previous_cache,
        set_memsize=lambda size: cache_limits.append(size),
    )
    pool = SimpleNamespace(
        get_limit=lambda: previous,
        total_bytes=lambda: 980,
        set_limit=lambda *, size: limits.append(size),
        free_all_blocks=lambda: released.append(True),
    )
    xp = SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        fft=SimpleNamespace(
            config=SimpleNamespace(get_plan_cache=lambda: cache)
        ),
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                memGetInfo=lambda: (20, 1000),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="correlation failure"):
        with engine_module._gpu_memory_headroom(xp):
            raise RuntimeError("correlation failure")
    assert limits == [min(previous or 900, 900), previous]
    assert released == [True]
    assert cache_limits == [
        min(previous_cache, 100) if previous_cache >= 0 else 100,
        previous_cache,
    ]


@pytest.mark.parametrize(
    ("capability", "expected"),
    [(8, 4), (9, 8), (10, 8), (12, 8)],
)
def test_automatic_xmac_fusion_width(capability, expected) -> None:
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(
                id=0, compute_capability=f"{capability}0"
            ),
            runtime=SimpleNamespace(
                getDeviceProperties=lambda identifier: {"major": capability}
            ),
        )
    )

    assert _automatic_xmac_fusion_width(xp) == expected


@pytest.mark.parametrize("mode", list(AutocorrelationMode))
@pytest.mark.parametrize("averaging", [1, 4])
def test_parallel_auto_reduction_preserves_strided_view(mode, averaging):
    storage = np.arange(3 * 40, dtype=np.float32).reshape(3, 40)
    spectrum = storage[:, 4:36].astype(np.complex64)
    # Retain a padded row stride as in a zoom-band view.
    padded = np.zeros((3, 40), dtype=np.complex64)
    padded[:, 4:36] = spectrum
    spectrum = padded[:, 4:36]
    weights = np.array([0.0, 0.25, 1.0])
    blocks = runtime_types._StationBlocks(
        spectrum, weights, np.arange(32, dtype=np.float64)
    )
    before = padded.copy()
    reduced, factor = gpu_execution._parallel_auto_reduction(
        blocks, averaging, mode, np
    )
    truncate = mode is AutocorrelationMode.TRUNCATION and averaging != 1
    count = 32 // averaging if truncate else 32
    assert factor == (1 if truncate else averaging)
    assert np.shares_memory(reduced.spectrum, padded)
    assert reduced.spectrum.strides == spectrum.strides
    assert reduced.weight is weights
    np.testing.assert_array_equal(reduced.spectrum, spectrum[:, :count])
    np.testing.assert_array_equal(reduced.baseband_hz, np.arange(count))
    np.testing.assert_array_equal(padded, before)


def test_spectrum_view_identity_includes_geometry():
    def spectrum(shape, strides):
        return SimpleNamespace(
            data=SimpleNamespace(ptr=1234), shape=shape, strides=strides
        )

    key = gpu_execution._spectrum_view_key
    assert key(spectrum((4, 8), (128, 8))) != key(spectrum((4, 8), (64, 8)))
    assert key(spectrum((4, 8), (128, 8))) != key(spectrum((4, 4), (128, 8)))


def test_correlation_run_waits_for_gpu_warmup_before_execution(
    monkeypatch,
) -> None:
    events = []

    def execute(*args, **kwargs):
        events.append("execute")
        return iter(("record",))

    monkeypatch.setattr(engine_module, "_execute_correlation", execute)
    warmup = SimpleNamespace(result=lambda: events.append("warmup"))
    run = CorrelationRun(
        job=SimpleNamespace(),
        backend="cupy",
        profile="analytic-v1",
        batch_blocks=None,
        vdif_io="posix",
        source=SimpleNamespace(),
        timing_groups=(),
        xp=SimpleNamespace(),
        gpu_warmup=warmup,
    )

    assert list(run.records()) == ["record"]
    assert events == ["warmup", "execute"]


def test_segmented_station_blocks_retry_model_runs_on_gpu_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOutOfMemoryError(MemoryError):
        pass

    free_calls = 0
    failed_buffers = []

    def free_all_blocks() -> None:
        nonlocal free_calls
        free_calls += 1
        assert all(buffer() is None for buffer in failed_buffers)

    xp = SimpleNamespace(
        __name__="cupy",
        cuda=SimpleNamespace(
            memory=SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError)
        ),
        empty=np.empty,
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=free_all_blocks
        ),
    )
    calls = []

    def station_blocks(_job, _data, model, starts, _xp):
        calls.append((model, len(starts)))
        if len(starts) > 2:
            temporary = np.empty(4)
            failed_buffers.append(weakref.ref(temporary))
            raise FakeOutOfMemoryError
        offset = 100 if model == "first" else 200
        values = np.asarray(starts) + offset
        return runtime_types._StationBlocks(
            np.repeat(values[:, None], 2, axis=1).astype(np.complex64),
            values.astype(np.float64),
            np.asarray((0.0, 1.0)),
        )

    monkeypatch.setattr(
        gpu_execution,
        "_ordered_model_runs",
        lambda *_args: (
            ("first", slice(0, 3)),
            ("second", slice(3, 6)),
        ),
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks", station_blocks)
    data = SimpleNamespace(
        datastream=SimpleNamespace(
            station_id=0, sampling=SamplingKind.COMPLEX
        ),
        frequency=SimpleNamespace(channel_count=2),
        sample_rate_hz=1.0,
    )

    blocks = gpu_execution._station_blocks_for_scan(
        SimpleNamespace(), data, SimpleNamespace(), np.arange(6.0), xp
    )

    expected = np.arange(6.0) + np.asarray((100, 100, 100, 200, 200, 200))
    np.testing.assert_array_equal(
        blocks.spectrum, np.repeat(expected[:, None], 2, axis=1)
    )
    np.testing.assert_array_equal(blocks.weight, expected)
    assert calls == [
        ("first", 3),
        ("first", 1),
        ("first", 1),
        ("first", 1),
        ("second", 3),
        ("second", 1),
        ("second", 1),
        ("second", 1),
    ]
    assert free_calls == 2


def test_segmented_station_blocks_keep_weight_credit_across_model_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credits = ((10, 20),)
    received = []

    def station_blocks(
        _job,
        _data,
        _model,
        starts,
        _xp,
        _offsets,
        _cutovers,
        _valid_starts,
        _valid_stops,
        weight_credit_intervals,
        _reported_weight_credit_intervals,
        _authoritative_validity,
        source_timing_offsets=None,
    ):
        assert source_timing_offsets is None
        received.append(weight_credit_intervals)
        return runtime_types._StationBlocks(
            np.zeros((len(starts), 2), dtype=np.complex128),
            np.ones(len(starts), dtype=np.float64),
            np.asarray((0.0, 1.0)),
        )

    monkeypatch.setattr(
        gpu_execution,
        "_ordered_model_runs",
        lambda *_args: (("first", slice(0, 2)), ("second", slice(2, 4))),
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks", station_blocks)
    data = SimpleNamespace(
        datastream=SimpleNamespace(
            station_id=0, sampling=SamplingKind.COMPLEX
        ),
        frequency=SimpleNamespace(channel_count=2),
        sample_rate_hz=1.0,
    )
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(4,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 4, 4, 0),),
        calibration_identifier="test-model-boundary-credit",
        timeline_fft_count=4,
        source_weight_credit_intervals=credits,
    )

    gpu_execution._station_blocks_for_scan(
        SimpleNamespace(),
        data,
        SimpleNamespace(start_offset_s=Decimal(0)),
        np.arange(4.0) * 2.0,
        np,
        plan,
    )

    assert received == [credits, credits]


def test_station_blocks_skip_ineligible_replay_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def station_blocks(
        _job,
        _data,
        _model,
        selected_starts,
        _xp,
        *_args,
        source_timing_offsets=None,
    ):
        assert source_timing_offsets is None
        calls.append(selected_starts.copy())
        values = selected_starts + 1
        return runtime_types._StationBlocks(
            np.repeat(values[:, None], 2, axis=1).astype(np.complex128),
            np.ones(selected_starts.size, dtype=np.float64),
            np.asarray((0.0, 0.5)),
            np.full(selected_starts.size, 0.5, dtype=np.float64),
        )

    monkeypatch.setattr(
        gpu_execution,
        "_ordered_model_runs",
        lambda *_args: ((SimpleNamespace(), slice(0, 4)),),
    )
    monkeypatch.setattr(gpu_execution, "_station_blocks", station_blocks)
    data = SimpleNamespace(
        datastream=SimpleNamespace(
            station_id=0, sampling=SamplingKind.COMPLEX
        ),
        band=SimpleNamespace(),
        frequency=SimpleNamespace(
            channel_count=2, sideband=gpu_execution.Sideband.USB
        ),
        sample_rate_hz=1.0,
    )
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(4,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 4, 2, 2),),
        calibration_identifier="test-ineligible-replay",
        timeline_fft_count=4,
        ineligible_intervals=((1, 3),),
        source_replay_intervals=((0, 1, -8), (1, 3, -8000), (3, 4, -8)),
    )

    result = gpu_execution._station_blocks_for_scan(
        SimpleNamespace(),
        data,
        SimpleNamespace(start_offset_s=Decimal(0)),
        np.arange(4.0) * 2.0,
        np,
        plan,
    )

    assert [item.tolist() for item in calls] == [[0.0, 6.0]]
    np.testing.assert_array_equal(result.spectrum[[1, 2]], 0)
    np.testing.assert_array_equal(result.weight, (1.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(
        result.reported_weight, (0.5, 0.0, 0.0, 0.5)
    )


def test_reported_station_weight_override_keeps_correlation_weight() -> None:
    data = SimpleNamespace(
        datastream=SimpleNamespace(
            station_id=0, sampling=SamplingKind.COMPLEX
        ),
        frequency=SimpleNamespace(channel_count=2),
        sample_rate_hz=1.0,
    )
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(2,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 2, 2, 0),),
        calibration_identifier="test-weight-override",
        timeline_fft_count=2,
        source_weight_override_events=((1, 2),),
    )

    weights = np.asarray((0.25, 0.0))
    blocks = gpu_execution._StationBlocks(
        np.zeros((2, 2), dtype=np.complex128),
        weights,
        np.arange(2),
    )
    reported = gpu_execution._reported_station_weights(
        blocks,
        plan=plan,
        scan=SimpleNamespace(start_offset_s=Decimal(0)),
        data=data,
        block_starts=np.arange(2.0) * 2.0,
        xp=np,
    )

    np.testing.assert_array_equal(weights, (0.25, 0.0))
    np.testing.assert_array_equal(reported, (0.25, 1.0))


def test_payload_progress_reports_errors_for_empty_ranges() -> None:
    progress = _PayloadProgress(1)
    error = OSError("late upload failure")
    progress.finish(error)
    xp = SimpleNamespace(
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace())
    )

    with pytest.raises(OSError, match="late upload failure"):
        progress.wait_samples(0, 0, 1, np.zeros(1, dtype=bool), xp)


def test_payload_progress_releases_consumed_events() -> None:
    waited = []
    stream = SimpleNamespace(wait_event=waited.append)
    xp = SimpleNamespace(
        cuda=SimpleNamespace(get_current_stream=lambda: stream)
    )
    progress = _PayloadProgress(4)

    for frame in range(4):
        event = object()
        progress.publish(np.asarray([frame], dtype=np.int32), event)
        progress.wait_samples(
            frame,
            frame + 1,
            1,
            np.ones(4, dtype=bool),
            xp,
        )

    assert len(waited) == 4
    assert progress._events == {}


def test_payload_progress_reports_publication_without_host_wait() -> None:
    validity = np.ones(2, dtype=bool)
    progress = _PayloadProgress(2, static_validity=validity)

    assert not progress.samples_published(0, 2, 1, validity)
    progress.publish(np.asarray([0], dtype=np.int32), object())
    assert progress.samples_published(0, 1, 1, validity)
    assert not progress.samples_published(0, 2, 1, validity)
    progress.publish(np.asarray([1], dtype=np.int32), object())
    assert progress.samples_published(0, 2, 1, validity)


def test_payload_progress_publication_propagates_terminal_gaps() -> None:
    validity = np.ones(1, dtype=bool)
    progress = _PayloadProgress(1, static_validity=validity)
    progress.finish(None)

    with pytest.raises(RuntimeError, match="without required valid frames"):
        progress.samples_published(0, 1, 1, validity)


def test_payload_progress_retains_event_until_all_frames_are_consumed() -> (
    None
):
    waited = []
    stream = SimpleNamespace(wait_event=waited.append)
    xp = SimpleNamespace(
        cuda=SimpleNamespace(get_current_stream=lambda: stream)
    )
    progress = _PayloadProgress(2)
    event = object()
    progress.publish(np.asarray([0, 1], dtype=np.int32), event)

    progress.wait_samples(0, 1, 1, np.ones(2, dtype=bool), xp)
    assert progress._events
    progress.wait_samples(1, 2, 1, np.ones(2, dtype=bool), xp)

    assert waited == [event, event]
    assert progress._events == {}
    assert progress._remaining == {}


def test_payload_progress_excludes_static_invalid_destinations() -> None:
    waited = []
    event = object()
    stream = SimpleNamespace(wait_event=waited.append)
    xp = SimpleNamespace(
        cuda=SimpleNamespace(get_current_stream=lambda: stream)
    )
    validity = np.asarray([True, False])
    progress = _PayloadProgress(2, static_validity=validity)

    progress.publish(np.asarray([0, 1], dtype=np.int32), event)

    assert progress._remaining == {1: 1}
    progress.wait_samples(0, 2, 1, validity, xp)
    assert waited == [event]
    assert progress._events == {}
    assert progress._remaining == {}


def test_payload_progress_does_not_retain_gap_tickets() -> None:
    waited = []
    event = object()
    stream = SimpleNamespace(wait_event=waited.append)
    xp = SimpleNamespace(
        cuda=SimpleNamespace(get_current_stream=lambda: stream)
    )
    progress = _PayloadProgress(4, dynamic_validity=True)
    validity = np.asarray([True, False, True, False])

    progress.publish_through(4, event, validity)

    assert progress._remaining == {1: 2}
    np.testing.assert_array_equal(progress._tickets, [1, -1, 1, -1])
    progress.wait_samples(0, 4, 1, validity, xp)
    assert waited == [event]
    assert progress._events == {}
    assert progress._remaining == {}


def _frame(station: str) -> bytes:
    payload = bytes([0xAA]) * 8  # 32 samples, all code +1
    word0 = 0
    word1 = 40 << 24  # 2020-01-01
    word2 = (32 + len(payload)) // 8
    station_id = (ord(station[0]) << 8) | ord(station[1])
    word3 = station_id | ((2 - 1) << 26)
    return (
        np.asarray(
            (word0, word1, word2, word3, 0, 0, 0, 0), dtype="<u4"
        ).tobytes()
        + payload
    )


def test_identical_payload_plans_share_one_upload(tmp_path: Path) -> None:
    path = tmp_path / "shared.vdif"
    slots = np.asarray((0, 1, -1), dtype=np.int32)

    def plan(mapping: np.ndarray) -> VdifPayloadPlan:
        return VdifPayloadPlan(
            40,
            8,
            2,
            (VdifPayloadFilePlan(path, 3, mapping),),
        )

    first = plan(slots)
    duplicate = plan(slots.copy())
    different = plan(np.asarray((1, 0, -1), dtype=np.int32))

    unique, aliases = _deduplicate_payload_plan_groups(
        [[first], [duplicate], [different]]
    )

    assert unique == [[first], [different]]
    assert aliases[id(first)] is first
    assert aliases[id(duplicate)] is first
    assert aliases[id(different)] is different


def _job(tmp_path: Path) -> Job:
    paths = []
    for name in ("AA", "BB"):
        path = tmp_path / f"{name}.vdif"
        path.write_bytes(_frame(name))
        build_vdif_binary_index(path, backend="posix", force=True)
        paths.append(path)
    epoch = MJDTime(58849, Decimal(0))
    stations = tuple(
        Station(index, name, ClockPolynomial(epoch, (0.0,)))
        for index, name in enumerate(("AA", "BB"))
    )
    frequency = Frequency(0, 0.0, 16.0, Sideband.USB, 4)
    streams = tuple(
        Datastream(
            index,
            index,
            SamplingKind.REAL,
            2,
            40,
            (str(paths[index]),),
            (RecordedBand(0, 0, "R"),),
            sample_rate_hz=32.0,
        )
        for index in range(2)
    )

    def zero_model(station_id):
        return DelayCoordinateModel(
            station_id,
            Decimal(0),
            (0.0,),
            (float(station_id),),
            (0.0,),
            (0.0,),
        )

    return Job(
        start_time=epoch,
        execution_duration_s=Decimal(1),
        integration_duration_s=Decimal("0.5"),
        subintegration_duration_s=Decimal("0.25"),
        fringe_approximation_order=1,
        stations=stations,
        frequencies=(frequency,),
        datastreams=streams,
        baselines=(Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),)),),
        scans=(
            Scan(
                "scan",
                Decimal(0),
                Decimal(1),
                "0",
                "0",
                (zero_model(0), zero_model(1)),
            ),
        ),
        write_autocorrelations=True,
    )


def test_replay_ingest_bounds_use_only_the_selected_window(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(4,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 4, 4, 0),),
        calibration_identifier="test",
        timeline_fft_count=4,
        source_replay_intervals=((0, 2, -8), (2, 4, -8000)),
    )

    selected = engine_module._extend_replay_intervals(
        job,
        ((0, job.scans[0], range(0, 1)),),
        {(0, 0): SampleInterval(100, 200)},
        {("scan", 0, 0): plan},
    )

    assert selected[(0, 0)] == SampleInterval(92, 192)


def test_physical_presentation_uses_recovery_planner_bounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.vdif"
    path.write_bytes(bytes(40 * 100))
    stat = path.stat()
    job = SimpleNamespace(
        datastreams=(
            SimpleNamespace(
                identifier=0,
                source_files=(str(path),),
                frame_size_bytes=40,
            ),
        )
    )
    ingest = engine_module.VdifIngestPlan(
        source_ranges={
            0: (
                VdifSourceRange(
                    path,
                    0,
                    1,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_dev,
                    stat.st_ino,
                ),
            )
        },
        thread_windows={
            0: {0: SimpleNamespace(start_slot=110, stop_slot=120)}
        },
        payload_intervals=None,
        selected_source_bytes=40,
        estimated_materialized_bytes=None,
        estimated_decoded_bytes=None,
        manifest={},
    )
    recovery = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(),
        checkpoints=(),
        calibration_identifier="test",
        timeline_fft_count=1,
        source_physical_frame_intervals=((0, 10, 90, 100),),
    )

    extended = engine_module._extend_physical_presentation_ranges(
        job,
        ingest,
        {("scan", 0, 0): recovery},
        max_read_bytes=1 << 20,
    )

    assert tuple(
        (item.first_frame, item.stop_frame)
        for item in extended.source_ranges[0]
    ) == ((20, 30),)
    assert extended.selected_source_bytes == 10 * 40
    assert extended.manifest["physical_presentation_source_bytes"] == 10 * 40


@pytest.mark.parametrize(
    "event_index,expected", [(0, (100, 200)), (1, (36, 200))]
)
def test_request_local_segments_use_only_the_selected_ingest_window(
    event_index: int,
    expected: tuple[int, int],
    tmp_path: Path,
) -> None:
    job = _job(tmp_path)
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(4,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 4, 4, 0),),
        calibration_identifier="test",
        timeline_fft_count=4,
        source_segment_events=((event_index, ((0, 8, -64),)),),
    )

    selected = engine_module._extend_replay_intervals(
        job,
        ((0, job.scans[0], range(1, 2)),),
        {(0, 0): SampleInterval(100, 200)},
        {("scan", 0, 0): plan},
    )

    assert selected[(0, 0)] == SampleInterval(*expected)


def test_source_recovery_masks_exact_packed_frames() -> None:
    payload_plan = SimpleNamespace(streaming=None)
    host_validity = np.ones(6, dtype=np.bool_)
    device_validity = np.ones(6, dtype=np.uint8)
    data = SimpleNamespace(
        payload_plan=payload_plan,
        payload_frame_validity=host_validity,
        packed_samples_per_frame=8,
        source_sample_count=48,
        start_sample=-8,
        validity=np.empty(0, dtype=np.bool_),
        device_validity=device_validity,
    )
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(1,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 1, 1, 0),),
        calibration_identifier="test",
        timeline_fft_count=1,
        source_invalid_intervals=((0, 8), (16, 24)),
    )

    engine_module._apply_source_invalid_intervals(
        {(2, 3): data}, {("scan", 2, 3): plan}
    )

    np.testing.assert_array_equal(host_validity, [1, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(device_validity, [1, 0, 1, 0, 1, 1])


def test_source_recovery_masks_exact_decoded_samples() -> None:
    validity = np.ones(12, dtype=np.bool_)
    data = SimpleNamespace(
        payload_plan=None,
        payload_frame_validity=None,
        packed_samples_per_frame=0,
        source_sample_count=None,
        start_sample=10,
        validity=validity,
        device_validity=None,
    )
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(1,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 1, 1, 0),),
        calibration_identifier="test",
        timeline_fft_count=1,
        source_invalid_intervals=((8, 13), (18, 25)),
    )

    engine_module._apply_source_invalid_intervals(
        {(2, 3): data}, {("scan", 2, 3): plan}
    )

    np.testing.assert_array_equal(
        validity,
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    )


def test_source_recovery_masks_cpu_decoded_packed_frames() -> None:
    frame_validity = np.ones(3, dtype=np.bool_)
    sample_validity = np.ones(24, dtype=np.bool_)
    data = SimpleNamespace(
        payload_plan=None,
        payload_frame_validity=frame_validity,
        packed_samples_per_frame=8,
        source_sample_count=24,
        start_sample=0,
        validity=sample_validity,
        device_validity=None,
    )
    plan = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(1,),
        checkpoints=(FFTEligibilityCheckpoint(0, 0, 1, 1, 0),),
        calibration_identifier="test",
        timeline_fft_count=1,
        source_invalid_intervals=((8, 16),),
    )

    engine_module._apply_source_invalid_intervals(
        {(2, 3): data}, {("scan", 2, 3): plan}
    )

    np.testing.assert_array_equal(frame_validity, [1, 0, 1])
    np.testing.assert_array_equal(
        sample_validity,
        [*([1] * 8), *([0] * 8), *([1] * 8)],
    )


def test_cufile_ingest_requires_cupy_backend(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError, match="cuFile ingest requires the CuPy backend"
    ):
        prepare_correlation(_job(tmp_path), vdif_io="cufile")


def test_reusable_upload_staging_matches_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest_scheduler,
        "cached_vdif_upload_staging_bytes",
        lambda device: 123,
    )
    assert _reusable_vdif_upload_staging_bytes("posix", 7) == 123
    assert _reusable_vdif_upload_staging_bytes("cufile", 7) == 0


def test_reusable_upload_staging_coverage_matches_backend(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int, tuple[int, ...], int]] = []

    def covers(
        backend: str,
        device: int,
        required: list[int],
        workers: int,
    ) -> bool:
        calls.append((backend, device, tuple(required), workers))
        return backend == "cufile"

    monkeypatch.setattr(
        ingest_scheduler,
        "cached_vdif_upload_staging_covers",
        lambda device, required, workers: covers(
            "posix", device, required, workers
        ),
    )
    assert not _reusable_vdif_upload_staging_covers("posix", 7, [20, 10], 2)
    assert not _reusable_vdif_upload_staging_covers("cufile", 7, [20, 10], 2)
    assert calls == [("posix", 7, (20, 10), 2)]


def test_constant_streams_cross_and_autocorrelations_match(tmp_path):
    records = list(correlate(_job(tmp_path), backend="numpy", batch_blocks=1))
    assert len(records) == 6
    for integration in range(2):
        cross, auto_a, auto_b = records[integration * 3 : integration * 3 + 3]
        np.testing.assert_allclose(cross.spectrum, auto_a.spectrum)
        np.testing.assert_allclose(cross.spectrum, auto_b.spectrum)
        np.testing.assert_allclose(cross.spectrum, [8, 0, 0, 0])
        assert cross.weight == auto_a.weight == auto_b.weight == 1.0
        assert (cross.u_m, cross.v_m, cross.w_m) == (-1.0, 0.0, 0.0)
        assert (auto_a.u_m, auto_a.v_m, auto_a.w_m) == (0.0, 0.0, 0.0)


def test_truncation_mode_changes_only_parallel_autocorrelations(tmp_path):
    base = _job(tmp_path)
    job = replace(
        base,
        frequencies=(replace(base.frequencies[0], channels_to_average=2),),
    )
    source = _DecodedSource(job)

    averaged = list(correlate(job, backend="numpy", recording_source=source))
    truncated = list(
        correlate(
            job,
            backend="numpy",
            recording_source=source,
            autocorrelation_mode=AutocorrelationMode.TRUNCATION,
        )
    )

    assert len(averaged) == len(truncated) == 6
    for integration in range(2):
        averaged_cross, averaged_a, averaged_b = averaged[
            integration * 3 : integration * 3 + 3
        ]
        truncated_cross, truncated_a, truncated_b = truncated[
            integration * 3 : integration * 3 + 3
        ]
        np.testing.assert_allclose(
            truncated_cross.spectrum, averaged_cross.spectrum
        )
        np.testing.assert_allclose(
            averaged_a.spectrum, averaged_cross.spectrum
        )
        np.testing.assert_allclose(
            averaged_b.spectrum, averaged_cross.spectrum
        )
        np.testing.assert_allclose(averaged_cross.spectrum, [4, 0])
        np.testing.assert_allclose(truncated_a.spectrum, [8, 0])
        np.testing.assert_allclose(truncated_b.spectrum, [8, 0])
        assert (
            truncated_cross.weight
            == truncated_a.weight
            == truncated_b.weight
            == 1.0
        )


def test_autocorrelation_mode_rejects_unknown_value(tmp_path):
    job = _job(tmp_path)

    with pytest.raises(ConfigurationError, match="autocorrelation_mode"):
        prepare_correlation(
            job,
            recording_source=_DecodedSource(job),
            autocorrelation_mode="unknown",
        )


def test_lo_phase_is_relative_to_job_start_not_wall_clock(tmp_path):
    base = _job(tmp_path)
    first_band = replace(
        base.datastreams[0].recorded_bands[0], lo_offset_hz=0.11
    )
    streams = (
        replace(base.datastreams[0], recorded_bands=(first_band,)),
        base.datastreams[1],
    )
    early = replace(base, datastreams=streams)
    late = replace(
        early,
        start_time=MJDTime(early.start_time.mjd, Decimal(47_880)),
    )

    early_cross = list(
        correlate(
            early,
            backend="numpy",
            recording_source=_DecodedSource(early),
        )
    )[::3]
    late_cross = list(
        correlate(
            late,
            backend="numpy",
            recording_source=_DecodedSource(late),
        )
    )[::3]

    for early_record, late_record in zip(
        early_cross, late_cross, strict=True
    ):
        np.testing.assert_allclose(
            early_record.spectrum,
            late_record.spectrum,
            rtol=1e-6,
            atol=1e-6,
        )


def test_gpu_output_group_limit_reserves_bounded_memory() -> None:
    assert _gpu_output_group_limit(0, 100, 1_600) == 0
    assert _gpu_output_group_limit(10, 0, 1_600) == 10
    assert _gpu_output_group_limit(10, 100, 1_600) == 4
    assert _gpu_output_group_limit(10, 100, 100) == 1


def test_payload_chunk_is_one_exact_integration_of_staging_frames(
    tmp_path: Path,
) -> None:
    plan_a = VdifPayloadPlan(
        8224,
        8192,
        10_000,
        (
            VdifPayloadFilePlan(
                tmp_path / "a.vdif",
                10_000,
                np.arange(10_000, dtype=np.int32),
            ),
        ),
    )
    plan_b = VdifPayloadPlan(
        8224,
        8192,
        10_000,
        (
            VdifPayloadFilePlan(
                tmp_path / "a.vdif",
                10_000,
                np.arange(10_000, dtype=np.int32),
            ),
        ),
    )
    data = SimpleNamespace(
        payload_plan=plan_a,
        sample_rate_hz=16_384_000_000,
        packed_samples_per_frame=32_768,
    )
    job = SimpleNamespace(integration_duration_s=Decimal("0.008192"))

    result = _payload_group_chunk_bytes(job, [plan_a, plan_b], {(0, 0): data})

    assert result == 4096 * (8224 + 2 * np.dtype(np.int32).itemsize)


@pytest.mark.parametrize(
    ("available_gib", "expected_integrations"),
    [(7, 1), (8, 2), (16, 4), (48, 8), (96, 8)],
)
def test_payload_chunk_scales_with_device_memory(
    tmp_path: Path, available_gib: int, expected_integrations: int
) -> None:
    plan = VdifPayloadPlan(
        8224,
        8192,
        10_000,
        (
            VdifPayloadFilePlan(
                tmp_path / "a.vdif",
                10_000,
                np.arange(10_000, dtype=np.int32),
            ),
        ),
    )
    data = SimpleNamespace(
        payload_plan=plan,
        sample_rate_hz=16_384_000_000,
        packed_samples_per_frame=32_768,
    )
    job = SimpleNamespace(integration_duration_s=Decimal("0.001024"))
    integration_bytes = 512 * (8224 + np.dtype(np.int32).itemsize)

    result = _payload_group_chunk_bytes(
        job,
        [plan],
        {(0, 0): data},
        available_device_bytes=available_gib * (1 << 30),
        upload_worker_count=2,
    )

    assert result == expected_integrations * integration_bytes


def test_mapped_frontier_upload_tasks_advance_destination_slices() -> None:
    mapping = np.arange(6, dtype=np.int32)
    plan = VdifPayloadPlan(
        frame_size_bytes=40,
        payload_bytes=8,
        slot_count=6,
        files=(
            VdifPayloadFilePlan(
                path=Path("stream.vdif"),
                frame_count=6,
                destination_slots=mapping,
                first_frame=10,
            ),
        ),
    )
    output = np.zeros(plan.byte_count, dtype=np.uint8)
    validity = np.ones(plan.slot_count, dtype=np.uint8)
    progress = _PayloadProgress(plan.slot_count, static_validity=validity)

    scheduled = _mapped_frontier_upload_tasks(
        [[plan]],
        {id(plan): output},
        {id(plan): validity},
        {id(plan): progress},
        [64],
        3,
        trust_slices=True,
    )

    assert scheduled is not None
    tasks, futures = scheduled
    assert len(tasks) == 3
    assert all(
        task.validated_mapped_token
        is upload_resources._TRUSTED_MAPPED_VDIF_PLANS
        for task in tasks
    )
    assert [task.plans[0].slot_count for task in tasks] == [2, 2, 2]
    assert [task.plans[0].files[0].first_frame for task in tasks] == [
        10,
        12,
        14,
    ]
    for task in tasks:
        np.testing.assert_array_equal(
            task.plans[0].files[0].destination_slots,
            np.asarray([0, 1], dtype=np.int32),
        )
        assert np.shares_memory(task.outputs[0], output)
        task.progresses[0].publish(
            np.asarray([0, 1], dtype=np.int32), object()
        )
        task.progresses[0].finish(None)

    completed_outputs = futures[id(plan)].result()
    assert completed_outputs == [output]
    assert progress.samples_published(0, 6, 1, validity)


def test_trusted_frontier_uses_physical_bounds_with_mapping_gaps() -> None:
    mapping = np.asarray([-1, 0, 1, -1, 2, 3, -1, 4, 5], dtype=np.int32)
    plan = VdifPayloadPlan(
        frame_size_bytes=40,
        payload_bytes=8,
        slot_count=6,
        files=(
            VdifPayloadFilePlan(
                path=Path("stream.vdif"),
                frame_count=mapping.size,
                destination_slots=mapping,
                first_frame=10,
            ),
        ),
    )
    output = np.zeros(plan.byte_count, dtype=np.uint8)
    validity = np.ones(plan.slot_count, dtype=np.uint8)
    progress = _PayloadProgress(plan.slot_count, static_validity=validity)

    scheduled = _mapped_frontier_upload_tasks(
        [[plan]],
        {id(plan): output},
        {id(plan): validity},
        {id(plan): progress},
        [64],
        3,
        trust_slices=True,
    )

    assert scheduled is not None
    tasks, _ = scheduled
    assert [task.plans[0].files[0].first_frame for task in tasks] == [
        11,
        14,
        17,
    ]
    for task in tasks:
        np.testing.assert_array_equal(
            task.plans[0].files[0].destination_slots,
            np.asarray([0, 1], dtype=np.int32),
        )


def test_trusted_frontier_slices_sparse_repeated_source_frames() -> None:
    destinations = np.arange(6, dtype=np.int32)
    sources = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)
    plan = VdifPayloadPlan(
        frame_size_bytes=40,
        payload_bytes=8,
        slot_count=6,
        files=(
            VdifPayloadFilePlan(
                path=Path("stream.vdif"),
                frame_count=6,
                destination_slots=destinations,
                first_frame=10,
                source_frame_indices=sources,
            ),
        ),
    )
    output = np.zeros(plan.byte_count, dtype=np.uint8)
    validity = np.ones(plan.slot_count, dtype=np.uint8)
    progress = _PayloadProgress(plan.slot_count, static_validity=validity)

    scheduled = _mapped_frontier_upload_tasks(
        [[plan]],
        {id(plan): output},
        {id(plan): validity},
        {id(plan): progress},
        [64],
        3,
        trust_slices=True,
    )

    assert scheduled is not None
    tasks, _ = scheduled
    assert [task.plans[0].files[0].first_frame for task in tasks] == [
        11,
        12,
        13,
    ]
    assert [task.plans[0].files[0].frame_count for task in tasks] == [
        1,
        1,
        1,
    ]
    for task, expected_sources in zip(
        tasks,
        (
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(
            task.plans[0].files[0].destination_slots,
            np.asarray([0, 1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            task.plans[0].files[0].source_frame_indices,
            expected_sources,
        )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (np.asarray([0, 1, 6], dtype=np.int32), "out of range"),
        (np.asarray([0, 1, 1], dtype=np.int32), "duplicate"),
    ],
)
def test_trusted_frontier_rejects_invalid_parent_mapping(
    mapping: np.ndarray,
    message: str,
) -> None:
    plan = VdifPayloadPlan(
        frame_size_bytes=40,
        payload_bytes=8,
        slot_count=6,
        files=(
            VdifPayloadFilePlan(
                path=Path("stream.vdif"),
                frame_count=mapping.size,
                destination_slots=mapping,
            ),
        ),
    )
    output = np.zeros(plan.byte_count, dtype=np.uint8)
    validity = np.ones(plan.slot_count, dtype=np.uint8)
    progress = _PayloadProgress(plan.slot_count, static_validity=validity)

    with pytest.raises(ValueError, match=message):
        _mapped_frontier_upload_tasks(
            [[plan]],
            {id(plan): output},
            {id(plan): validity},
            {id(plan): progress},
            [64],
            3,
            trust_slices=True,
        )


def test_cufile_frontier_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(engine_module._CUFILE_FRONTIER_ENV, raising=False)
    assert engine_module._cufile_frontier_enabled("cufile")

    monkeypatch.setenv(engine_module._CUFILE_FRONTIER_ENV, "yes")
    assert engine_module._cufile_frontier_enabled("cufile")
    with pytest.raises(ValueError, match="requires the cuFile backend"):
        engine_module._cufile_frontier_enabled("posix")

    monkeypatch.setenv(engine_module._CUFILE_FRONTIER_ENV, "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        engine_module._cufile_frontier_enabled("cufile")


def test_replay_readiness_defers_to_execution_source_dependencies(
    monkeypatch,
):
    calls = []
    progress = SimpleNamespace(
        samples_published=lambda *args: calls.append(args) or False
    )
    data = SimpleNamespace(
        payload_progress=progress,
        payload_frame_validity=np.ones(10, dtype=bool),
        packed_samples_per_frame=1,
    )
    plan = SimpleNamespace(
        source_offsets=(),
        source_replay_events=(),
        source_replay_intervals=(),
        has_source_segments=False,
        ineligible_intervals=(),
        eligible_ffts=(True,),
    )
    scan = SimpleNamespace(identifier="scan")
    monkeypatch.setattr(
        gpu_execution, "_station_payload_sample_bounds", lambda *args: (0, 10)
    )
    arguments = (None, [((1, 0), data)], scan, np.array([0.0]))
    plans = {("scan", 1, 0): plan}
    assert not gpu_execution._station_ready_to_attempt(*arguments, plans)
    assert len(calls) == 1
    plan.source_replay_events = ((0, 20, 0),)
    assert gpu_execution._station_ready_to_attempt(*arguments, plans)
    assert len(calls) == 1


def test_frontier_upload_queue_is_dynamic_and_frontier_ordered() -> None:
    tasks = [
        ingest_scheduler._FrontierUploadTask(
            frontier,
            [],
            [],
            [],
            [],
            64,
        )
        for frontier in (0, 0, 1, 1)
    ]
    queue = ingest_scheduler._FrontierUploadQueue(tasks)

    assert queue.take() is tasks[0]
    assert queue.take() is tasks[1]
    queue.finish(None)
    assert queue.take() is tasks[2]
    queue.finish(None)
    queue.finish(None)
    assert queue.take() is tasks[3]
    queue.finish(None)
    assert queue.take() is None


def test_frontier_upload_queue_cancels_pending_work_after_failure() -> None:
    tasks = [
        ingest_scheduler._FrontierUploadTask(
            frontier,
            [],
            [],
            [],
            [],
            64,
        )
        for frontier in (0, 1, 2)
    ]
    queue = ingest_scheduler._FrontierUploadQueue(tasks)

    assert queue.take() is tasks[0]
    assert queue.finish(RuntimeError("read failed")) == tasks[1:]
    assert queue.take() is None


def test_frontier_upload_queue_can_abort_with_active_work() -> None:
    tasks = [
        ingest_scheduler._FrontierUploadTask(
            frontier,
            [],
            [],
            [],
            [],
            64,
        )
        for frontier in (0, 1, 2)
    ]
    queue = ingest_scheduler._FrontierUploadQueue(tasks)

    assert queue.take() is tasks[0]
    assert queue.abort(RuntimeError("consumer stopped")) == tasks[1:]
    queue.finish(None)
    assert queue.take() is None


def test_payload_progress_can_share_one_notification_condition() -> None:
    condition = ingest_scheduler.Condition()
    first = _PayloadProgress(1, condition=condition)
    second = _PayloadProgress(1, condition=condition)

    assert first._condition is condition
    assert second._condition is condition


def test_payload_progress_notification_advances_shared_generation() -> None:
    notifier = ingest_scheduler._PayloadProgressNotifier()
    progress = _PayloadProgress(1, notifier=notifier)
    with notifier.condition:
        observed = notifier.generation

    progress.publish(np.asarray([0], dtype=np.int32), object())

    with notifier.condition:
        assert notifier.condition.wait_for(
            lambda: notifier.generation != observed,
            timeout=0.0,
        )


def test_mapped_frontier_upload_rejects_streaming_plan() -> None:
    plan = VdifPayloadPlan(
        frame_size_bytes=40,
        payload_bytes=8,
        slot_count=2,
        files=(),
        streaming=SimpleNamespace(),
    )

    assert (
        _mapped_frontier_upload_tasks(
            [[plan]],
            {id(plan): np.zeros(16, dtype=np.uint8)},
            {id(plan): np.ones(2, dtype=np.uint8)},
            {id(plan): _PayloadProgress(2)},
            [64],
            2,
        )
        is None
    )


def test_mapped_frontier_upload_rejects_unordered_mapping() -> None:
    plan = VdifPayloadPlan(
        frame_size_bytes=40,
        payload_bytes=8,
        slot_count=4,
        files=(
            VdifPayloadFilePlan(
                path=Path("unordered.vdif"),
                frame_count=4,
                destination_slots=np.asarray([0, 2, 1, 3], dtype=np.int32),
            ),
        ),
    )

    assert (
        _mapped_frontier_upload_tasks(
            [[plan]],
            {id(plan): np.zeros(plan.byte_count, dtype=np.uint8)},
            {id(plan): np.ones(plan.slot_count, dtype=np.uint8)},
            {
                id(plan): _PayloadProgress(
                    plan.slot_count,
                    static_validity=np.ones(plan.slot_count, dtype=bool),
                )
            },
            [64],
            2,
        )
        is None
    )


def test_payload_chunk_honours_concurrent_staging_budget(
    tmp_path: Path,
) -> None:
    plan = VdifPayloadPlan(
        8224,
        8192,
        10_000,
        (
            VdifPayloadFilePlan(
                tmp_path / "a.vdif",
                10_000,
                np.arange(10_000, dtype=np.int32),
            ),
        ),
    )
    data = SimpleNamespace(
        payload_plan=plan,
        sample_rate_hz=16_384_000_000,
        packed_samples_per_frame=32_768,
    )
    job = SimpleNamespace(integration_duration_s=Decimal("0.008192"))
    integration_bytes = 4096 * (8224 + np.dtype(np.int32).itemsize)

    result = _payload_group_chunk_bytes(
        job,
        [plan],
        {(0, 0): data},
        available_device_bytes=48 * (1 << 30),
        upload_worker_count=32,
    )

    assert result == integration_bytes


def test_payload_chunk_can_split_an_oversized_integration(
    tmp_path: Path,
) -> None:
    plan = VdifPayloadPlan(
        8224,
        8192,
        10_000,
        (
            VdifPayloadFilePlan(
                tmp_path / "a.vdif",
                10_000,
                np.arange(10_000, dtype=np.int32),
            ),
        ),
    )
    data = SimpleNamespace(
        payload_plan=plan,
        sample_rate_hz=16_384_000_000,
        packed_samples_per_frame=32_768,
    )
    job = SimpleNamespace(integration_duration_s=Decimal("0.02"))
    frame_bytes = 8224 + np.dtype(np.int32).itemsize

    result = _payload_group_chunk_bytes(
        job,
        [plan],
        {(0, 0): data},
        available_device_bytes=80 * (1 << 30),
        upload_worker_count=2,
    )

    assert result <= 64 * (1 << 20)
    assert result % frame_bytes == 0


def test_payload_chunk_uses_larger_direct_storage_requests(
    tmp_path: Path,
) -> None:
    plan = VdifPayloadPlan(
        8224,
        8192,
        10_000,
        (
            VdifPayloadFilePlan(
                tmp_path / "a.vdif",
                10_000,
                np.arange(10_000, dtype=np.int32),
            ),
        ),
    )
    data = SimpleNamespace(
        payload_plan=plan,
        sample_rate_hz=16_384_000_000,
        packed_samples_per_frame=32_768,
    )
    job = SimpleNamespace(integration_duration_s=Decimal("0.008192"))
    integration_bytes = 4096 * (8224 + np.dtype(np.int32).itemsize)

    result = _payload_group_chunk_bytes(
        job,
        [plan],
        {(0, 0): data},
        available_device_bytes=80 * (1 << 30),
        upload_worker_count=2,
        io_backend="cufile",
    )

    assert result == integration_bytes


def test_upload_staging_plan_sums_concurrent_workers(tmp_path: Path) -> None:
    plans = [
        VdifPayloadPlan(
            40,
            8,
            1,
            (
                VdifPayloadFilePlan(
                    tmp_path / f"{index}.vdif",
                    1,
                    np.asarray([0], dtype=np.int32),
                ),
            ),
        )
        for index in range(5)
    ]
    band_data = {
        (index, 0): SimpleNamespace(
            payload_plan=plan,
            sample_rate_hz=32,
            packed_samples_per_frame=32,
        )
        for index, plan in enumerate(plans)
    }
    available = 1 << 30

    workers, chunks, staging_bytes, per_worker = _payload_upload_staging_plan(
        SimpleNamespace(integration_duration_s=Decimal(1)),
        [[plan] for plan in plans],
        band_data,
        available,
    )

    assert workers == 4
    assert chunks == [44] * 5
    assert staging_bytes == 4 * 2 * 4 * 1024 * 1024
    assert per_worker == available // workers


def test_cufile_upload_staging_auto_scales_and_honours_override(
    tmp_path: Path,
) -> None:
    plans = [
        VdifPayloadPlan(
            40,
            8,
            1,
            (
                VdifPayloadFilePlan(
                    tmp_path / f"cufile-{index}.vdif",
                    1,
                    np.asarray([0], dtype=np.int32),
                ),
            ),
        )
        for index in range(11)
    ]
    band_data = {
        (index, 0): SimpleNamespace(
            payload_plan=plan,
            sample_rate_hz=32,
            packed_samples_per_frame=32,
        )
        for index, plan in enumerate(plans)
    }
    arguments = (
        SimpleNamespace(integration_duration_s=Decimal(1)),
        [[plan] for plan in plans],
        band_data,
        16 << 30,
    )

    automatic = _payload_upload_staging_plan(*arguments, io_backend="cufile")
    explicit = _payload_upload_staging_plan(
        *arguments, io_backend="cufile", requested_workers=6
    )

    assert automatic[0] == 4
    assert explicit[0] == 6


@pytest.mark.parametrize("value", ["0", "17", "invalid"])
def test_vdif_upload_worker_option_is_bounded(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ConfigurationError, match="vdif_upload_workers"):
        prepare_correlation(
            _job(tmp_path),
            backend="numpy",
            vdif_upload_workers=value,
        )


def test_subintegration_tiles_are_contiguous_and_preserve_remainder() -> None:
    source = range(10, 135)

    tiles = compute_plan._partition_subintegrations(source, 64)

    assert tiles == [range(10, 74), range(74, 135)]
    assert [item for tile in tiles for item in tile] == list(source)


def test_subintegration_tiling_options_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="must be 'auto' or 'off'"):
        prepare_correlation(_job(tmp_path), subintegration_tiling="always")
    with pytest.raises(ConfigurationError, match="positive integer"):
        prepare_correlation(_job(tmp_path), max_tile_subintegrations=0)
    with pytest.raises(ConfigurationError, match="tiling is off"):
        prepare_correlation(
            _job(tmp_path),
            subintegration_tiling="off",
            max_tile_subintegrations=1,
        )
    with pytest.raises(ConfigurationError, match="requires the CuPy backend"):
        prepare_correlation(_job(tmp_path), max_tile_subintegrations=1)


def test_subintegration_tile_oom_halves_only_the_failed_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOutOfMemoryError(MemoryError):
        pass

    freed = 0
    failed_buffers = []

    def free_all_blocks() -> None:
        nonlocal freed
        freed += 1
        assert all(buffer() is None for buffer in failed_buffers)

    xp = SimpleNamespace(
        empty=np.empty,
        uint8=np.uint8,
        complex64=np.complex64,
        float64=np.float64,
        cuda=SimpleNamespace(
            memory=SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError)
        ),
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=free_all_blocks
        ),
    )

    layout = ()

    def run_tile(*args, **_kwargs):
        assert _kwargs["visibility_layout"] is layout
        selected = args[2][0][2]
        arena = args[8]
        if len(selected) > 2:
            temporary = np.empty(4)
            failed_buffers.append(weakref.ref(temporary))
            raise FakeOutOfMemoryError
        arena.spectra.fill(len(selected))
        arena.weights.fill(len(selected))
        return ["record"]

    monkeypatch.setattr(gpu_execution, "_run_chunked_gpu", run_tile)
    pending, arena, sizes, retries = (
        gpu_execution._run_subintegration_tiled_gpu(
            SimpleNamespace(),
            xp,
            (4, SimpleNamespace(), range(5)),
            {},
            {},
            {},
            {},
            {},
            1,
            2,
            None,
            1,
            AutocorrelationMode.AVERAGE,
            None,
            5,
            visibility_layout=layout,
        )
    )

    assert pending == ["record"]
    assert sizes == [2, 1, 2]
    assert retries == 2
    assert freed == 2
    np.testing.assert_array_equal(arena.spectra, [5, 5])
    np.testing.assert_array_equal(arena.weights, [5])


@pytest.mark.gpu
def test_gpu_output_arena_slices_preserve_order_and_values(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.xbeam.test_gpu_backend import _cupy

    cupy = _cupy()
    import cuphoton.xbeam.adapters.radio.engine as engine_module

    job = _job(tmp_path)
    expected = list(engine_module.correlate(job, backend="cupy"))
    original_to_numpy = backend.to_numpy
    transfers: list[int] = []

    def tracked_to_numpy(value):
        if isinstance(value, cupy.ndarray) and value.dtype == cupy.uint8:
            transfers.append(int(value.size))
        return original_to_numpy(value)

    monkeypatch.setattr(output_records, "to_numpy", tracked_to_numpy)
    monkeypatch.setattr(
        engine_module, "_gpu_output_group_limit", lambda *_args: 1
    )
    actual = list(engine_module.correlate(job, backend="cupy"))

    assert len(transfers) == 2
    assert len(actual) == len(expected)
    for candidate, reference in zip(actual, expected, strict=True):
        assert (
            candidate.antenna_a,
            candidate.antenna_b,
            candidate.seconds_of_day,
            candidate.output_frequency_id,
            candidate.polarization_a,
            candidate.polarization_b,
        ) == (
            reference.antenna_a,
            reference.antenna_b,
            reference.seconds_of_day,
            reference.output_frequency_id,
            reference.polarization_a,
            reference.polarization_b,
        )
        np.testing.assert_allclose(
            candidate.spectrum, reference.spectrum, rtol=2e-6, atol=2e-6
        )
        assert candidate.weight == reference.weight


@pytest.mark.gpu
def test_gpu_subintegration_tiling_preserves_logical_integrations(
    tmp_path: Path,
) -> None:
    from tests.xbeam.test_gpu_backend import _cupy

    _cupy()

    job = _job(tmp_path)
    expected = list(
        correlate(job, backend="cupy", subintegration_tiling="off")
    )
    prepared = prepare_correlation(
        job,
        backend="cupy",
        max_tile_subintegrations=1,
    )
    actual = list(prepared.records())

    assert len(actual) == len(expected)
    for candidate, reference in zip(actual, expected, strict=True):
        assert (
            candidate.antenna_a,
            candidate.antenna_b,
            candidate.seconds_of_day,
            candidate.output_frequency_id,
            candidate.polarization_a,
            candidate.polarization_b,
        ) == (
            reference.antenna_a,
            reference.antenna_b,
            reference.seconds_of_day,
            reference.output_frequency_id,
            reference.polarization_a,
            reference.polarization_b,
        )
        np.testing.assert_allclose(
            candidate.spectrum, reference.spectrum, rtol=2e-6, atol=2e-6
        )
        assert candidate.weight == pytest.approx(reference.weight)
    assert prepared.ingest_manifest is not None
    tiling = prepared.ingest_manifest["subintegration_tiling"]
    assert tiling["effective"] is True
    assert tiling["automatic_limit"] == 1
    assert [item["subintegrations"] for item in tiling["tiles"]] == [
        [1, 1],
        [1, 1],
    ]


def test_batching_does_not_change_results(tmp_path):
    job = _job(tmp_path)
    unbatched = list(correlate(job, backend="numpy"))
    batched = list(correlate(job, backend="numpy", batch_blocks=1))
    for first, second in zip(unbatched, batched, strict=True):
        assert first.seconds_of_day == second.seconds_of_day
        np.testing.assert_allclose(first.spectrum, second.spectrum)


def test_zoom_is_channel_aligned_parent_slice(tmp_path):
    base = _job(tmp_path)
    zoom_frequency = Frequency(1, 4.0, 8.0, Sideband.USB, 2)
    streams = tuple(
        replace(
            stream,
            zoom_bands=(ZoomBand(1, 0, 1, 1, 2, "R"),),
        )
        for stream in base.datastreams
    )
    baseline = Baseline(0, 0, 1, (BaselineProduct(1, 1, 1),))
    job = replace(
        base,
        frequencies=base.frequencies + (zoom_frequency,),
        datastreams=streams,
        baselines=(baseline,),
        write_autocorrelations=False,
    )
    records = list(correlate(job, backend="numpy"))
    assert len(records) == 2
    assert all(record.output_frequency_id == 1 for record in records)
    assert all(record.spectrum.shape == (2,) for record in records)
    np.testing.assert_allclose(records[0].spectrum, [0, 0])


def test_autocorrelations_follow_requested_output_bands_only(tmp_path):
    base = _job(tmp_path)
    zoom_frequency = Frequency(1, 4.0, 8.0, Sideband.USB, 2)
    streams = tuple(
        replace(stream, zoom_bands=(ZoomBand(1, 0, 1, 1, 2, "R"),))
        for stream in base.datastreams
    )
    job = replace(
        base,
        frequencies=base.frequencies + (zoom_frequency,),
        datastreams=streams,
        baselines=(Baseline(0, 0, 1, (BaselineProduct(1, 1, 1),)),),
        write_autocorrelations=True,
    )

    records = list(correlate(job, backend="numpy"))

    assert len(records) == 6
    assert all(record.output_frequency_id == 1 for record in records)
    assert all(record.spectrum.shape == (2,) for record in records)


def test_integration_shorter_than_subintegration_omits_empty_windows(
    tmp_path,
):
    base = _job(tmp_path)
    job = replace(
        base,
        execution_duration_s=Decimal("0.5"),
        integration_duration_s=Decimal("0.125"),
        subintegration_duration_s=Decimal("0.25"),
    )
    records = list(correlate(job, backend="numpy"))
    cross_records = records[::3]
    assert len(cross_records) == 2
    assert [float(record.seconds_of_day) for record in cross_records] == [
        0.1875,
        0.4375,
    ]
    assert [record.weight for record in cross_records] == [2.0, 2.0]


def test_subintegration_is_included_when_midpoint_precedes_job_end(tmp_path):
    job = replace(
        _job(tmp_path),
        execution_duration_s=Decimal("0.4"),
        integration_duration_s=Decimal("0.5"),
        subintegration_duration_s=Decimal("0.25"),
    )

    groups = _timing_groups(job, ANALYTIC_V1)

    assert [(index, list(subs)) for index, _scan, subs in groups] == [
        (0, [0, 1])
    ]


def test_subintegrations_are_clipped_by_scan_midpoint(tmp_path):
    base = _job(tmp_path)
    job = replace(
        base,
        scans=(replace(base.scans[0], duration_s=Decimal("0.4")),),
    )

    groups = _timing_groups(job, ANALYTIC_V1)

    assert [(index, list(subs)) for index, _scan, subs in groups] == [
        (0, [0, 1])
    ]


def test_interop_fractional_start_uses_integer_second_execution_end(tmp_path):
    base = _job(tmp_path)
    job = replace(
        base,
        start_time=replace(base.start_time, seconds_of_day=Decimal("0.5")),
    )

    analytic = _timing_groups(job, ANALYTIC_V1)
    interop = _timing_groups(job, INTEROP_REAL_USB_V1)

    assert [(index, list(subs)) for index, _scan, subs in analytic] == [
        (0, [0, 1]),
        (1, [2, 3]),
    ]
    assert [(index, list(subs)) for index, _scan, subs in interop] == [
        (0, [0, 1])
    ]


def test_integration_cadence_restarts_at_each_scan(tmp_path):
    base = _job(tmp_path)
    first = replace(
        base.scans[0],
        identifier="first",
        duration_s=Decimal("0.4"),
        source_id="0",
    )
    second = replace(
        base.scans[0],
        identifier="second",
        start_offset_s=Decimal("0.6"),
        duration_s=Decimal("0.4"),
        source_id="1",
    )
    job = replace(base, scans=(first, second))

    groups = _timing_groups(job, ANALYTIC_V1)
    assert [
        (scan.identifier, index, list(subs)) for index, scan, subs in groups
    ] == [
        ("first", 0, [0, 1]),
        ("second", 0, [0, 1]),
    ]

    cross_records = list(correlate(job, backend="numpy"))[::3]
    assert [record.seconds_of_day for record in cross_records] == [
        Decimal("0.25"),
        Decimal("0.85"),
    ]


def test_long_timing_schedule_uses_compact_ranges(tmp_path):
    base = _job(tmp_path)
    job = replace(
        base,
        execution_duration_s=Decimal("100"),
        integration_duration_s=Decimal("0.008192"),
        subintegration_duration_s=Decimal("0.000004"),
        scans=(replace(base.scans[0], duration_s=Decimal("100")),),
    )

    groups = _timing_groups(job, ANALYTIC_V1)

    assert len(groups) == 12_208
    assert all(
        isinstance(subintegrations, range) for _, _, subintegrations in groups
    )
    assert (
        sum(len(subintegrations) for _, _, subintegrations in groups)
        == 25_000_000
    )


class _DecodedSource:
    def __init__(self, job: Job) -> None:
        self.job = job

    def batches(self, datastream: Datastream):
        yield VoltageBatch(
            samples=np.ones(32, dtype=np.float32),
            start_time=self.job.start_time,
            sample_rate_hz=32.0,
            band_id=0,
            polarization="R",
            validity=np.ones(32, dtype=np.bool_),
        )


def test_injected_recording_source_bypasses_filesystem(tmp_path):
    job = _job(tmp_path)
    streams = tuple(
        replace(stream, source_files=(str(tmp_path / "unavailable.vdif"),))
        for stream in job.datastreams
    )
    job = replace(job, datastreams=streams)

    records = list(
        correlate(job, backend="numpy", recording_source=_DecodedSource(job))
    )

    assert len(records) == 6
    np.testing.assert_allclose(records[0].spectrum, [8, 0, 0, 0])


def test_contiguous_decoded_batches_are_assembled(tmp_path):
    job = _job(tmp_path)

    class SplitSource:
        def batches(self, datastream: Datastream):
            for offset in (Decimal(0), Decimal("0.5")):
                yield VoltageBatch(
                    samples=np.ones(16, dtype=np.float32),
                    start_time=MJDTime(
                        job.start_time.mjd,
                        job.start_time.seconds_of_day + offset,
                    ),
                    sample_rate_hz=32.0,
                    band_id=0,
                    polarization="R",
                    validity=np.ones(16, dtype=np.bool_),
                )

    records = list(
        correlate(job, backend="numpy", recording_source=SplitSource())
    )

    np.testing.assert_allclose(records[0].spectrum, [8, 0, 0, 0])


@pytest.mark.parametrize(
    ("sampling", "source_dtype", "expected_dtype"),
    [
        (SamplingKind.REAL, np.float64, np.float32),
        (SamplingKind.COMPLEX, np.complex128, np.complex64),
    ],
)
def test_decoded_batches_are_normalized_for_gpu_front_end(
    tmp_path, sampling, source_dtype, expected_dtype
):
    job = _job(tmp_path)
    stream = replace(job.datastreams[0], sampling=sampling)
    samples = np.ones(32, dtype=source_dtype)
    batch = VoltageBatch(
        samples=samples,
        start_time=job.start_time,
        sample_rate_hz=32.0,
        band_id=0,
        polarization="R",
        validity=np.ones(32, dtype=np.bool_),
    )

    loaded = _load_band_samples(
        job,
        stream,
        stream.recorded_bands[0],
        job.frequencies[0],
        [batch],
    )

    assert loaded.samples.dtype == expected_dtype


@pytest.mark.parametrize("defer_file_payload", [False, True])
def test_streaming_validity_starts_zero_without_host_to_device_copy(
    tmp_path: Path, monkeypatch, defer_file_payload: bool
) -> None:
    job = _job(tmp_path)
    datastream = job.datastreams[0]
    raw = _frame("AA")
    template = VdifHeader.from_bytes(raw[:32])
    host_validity = np.ones(1, dtype=np.bool_)
    plan = VdifPayloadPlan(
        40,
        8,
        1,
        (
            VdifPayloadFilePlan(
                tmp_path / "stream.vdif",
                1,
                None,
            ),
        ),
        VdifStreamingPlan(
            template.thread_id,
            1,
            absolute_frame_slot(template, 1),
            template,
            DuplicatePolicy.FIRST_VALID,
            host_validity,
        ),
    )
    batch = VoltageBatch(
        samples=plan,
        start_time=job.start_time,
        sample_rate_hz=32.0,
        band_id=0,
        polarization="R",
        validity=host_validity,
        packed_layout=PackedVoltageLayout(8, 32, 1, 0, False),
    )
    asarray_calls = []

    def asarray(*args, **kwargs):
        asarray_calls.append((args, kwargs))
        return np.asarray(*args, **kwargs)

    fake_cupy = SimpleNamespace(
        __name__="cupy",
        uint8=np.uint8,
        asarray=asarray,
        zeros=np.zeros,
    )

    uploads = []

    def load_payload(plans, *, xp, validity_outputs, reservation_token):
        assert plans == [plan]
        np.testing.assert_array_equal(validity_outputs[0], [0])
        validity_outputs[0][:] = 1
        host_validity[:] = True
        uploads.append(validity_outputs[0])
        return [np.zeros(8, dtype=np.uint8)]

    monkeypatch.setattr(
        "cuphoton.xbeam.sample_loading.load_vdif_payload_plans", load_payload
    )
    device_cache = {}
    arguments = dict(
        xp=fake_cupy,
        device_cache=device_cache,
        defer_file_payload=defer_file_payload,
    )
    loaded = _load_band_samples(
        job,
        datastream,
        datastream.recorded_bands[0],
        job.frequencies[0],
        [batch],
        **arguments,
    )

    assert asarray_calls == []
    np.testing.assert_array_equal(host_validity, [not defer_file_payload])
    np.testing.assert_array_equal(
        loaded.device_validity, [int(not defer_file_payload)]
    )
    if not defer_file_payload:
        cached = _load_band_samples(
            job,
            datastream,
            datastream.recorded_bands[0],
            job.frequencies[0],
            [batch],
            **arguments,
        )
        assert cached.device_payload is loaded.device_payload
        assert cached.device_validity is loaded.device_validity
        assert len(uploads) == 1


def test_overlapping_recording_batches_are_rejected(tmp_path):
    job = _job(tmp_path)

    class OverlappingSource:
        def batches(self, datastream: Datastream):
            for offset in (Decimal(0), Decimal("0.25")):
                yield VoltageBatch(
                    samples=np.ones(16, dtype=np.float32),
                    start_time=MJDTime(job.start_time.mjd, offset),
                    sample_rate_hz=32.0,
                    band_id=0,
                    polarization="R",
                    validity=np.ones(16, dtype=np.bool_),
                )

    with pytest.raises(ConfigurationError, match="overlap"):
        list(
            correlate(
                job, backend="numpy", recording_source=OverlappingSource()
            )
        )


def test_sparse_recording_batches_are_rejected_before_allocation(tmp_path):
    job = _job(tmp_path)

    class SparseSource:
        def batches(self, datastream: Datastream):
            for offset in (Decimal(0), Decimal(100)):
                yield VoltageBatch(
                    samples=np.ones(1, dtype=np.float32),
                    start_time=MJDTime(job.start_time.mjd, offset),
                    sample_rate_hz=32.0,
                    band_id=0,
                    polarization="R",
                    validity=np.ones(1, dtype=np.bool_),
                )

    with pytest.raises(ConfigurationError, match="batch timestamp span"):
        list(correlate(job, backend="numpy", recording_source=SparseSource()))


def test_default_source_checks_declared_vdif_frame_size(tmp_path):
    job = _job(tmp_path)
    streams = tuple(
        replace(stream, frame_size_bytes=48) for stream in job.datastreams
    )

    with pytest.raises(ConfigurationError, match="frames; expected 48"):
        list(correlate(replace(job, datastreams=streams), backend="numpy"))


def test_default_source_checks_vdif_sampling_kind(tmp_path):
    job = _job(tmp_path)
    job = replace(
        job,
        frequencies=(replace(job.frequencies[0], bandwidth_hz=32.0),),
    )
    streams = tuple(
        replace(stream, sampling=SamplingKind.COMPLEX)
        for stream in job.datastreams
    )

    with pytest.raises(ConfigurationError, match="sampling"):
        list(correlate(replace(job, datastreams=streams), backend="numpy"))


def test_packed_injected_source_matches_default_source(tmp_path):
    job = _job(tmp_path)
    payload = np.frombuffer(_frame("AA")[32:], dtype=np.uint8)

    class PackedSource:
        def batches(self, datastream: Datastream):
            yield VoltageBatch(
                samples=payload,
                start_time=job.start_time,
                sample_rate_hz=32.0,
                band_id=0,
                polarization="R",
                validity=np.ones(1, dtype=np.bool_),
                packed_layout=PackedVoltageLayout(8, 32, 1, 0, False),
            )

    expected = list(correlate(job, backend="numpy"))
    actual = list(
        correlate(job, backend="numpy", recording_source=PackedSource())
    )

    for first, second in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(first.spectrum, second.spectrum)


@pytest.mark.gpu
def test_output_arena_oom_releases_failed_buffers_before_pool_retry(
    tmp_path, monkeypatch
):
    from tests.xbeam.test_gpu_backend import _cupy

    cupy = _cupy()
    job = _job(tmp_path)
    expected = list(correlate(job, backend="numpy"))
    allocate = engine_module._new_gpu_output_arena
    pool = cupy.get_default_memory_pool()
    failed_buffers, allocations = [], []

    class Pool:
        def __getattr__(self, name):
            return getattr(pool, name)

        def free_all_blocks(self, *args, **kwargs):
            assert all(buffer() is None for buffer in failed_buffers)
            return pool.free_all_blocks(*args, **kwargs)

    def allocation(xp, records, spectra):
        allocations.append(records)
        if len(allocations) == 1:
            temporary = np.empty(4)
            failed_buffers.append(weakref.ref(temporary))
            raise cupy.cuda.memory.OutOfMemoryError(1, 1)
        return allocate(xp, records, spectra)

    monkeypatch.setattr(cupy, "get_default_memory_pool", lambda: Pool())
    monkeypatch.setattr(engine_module, "_new_gpu_output_arena", allocation)
    monkeypatch.setattr(
        engine_module, "_gpu_output_group_limit", lambda *_: 2
    )
    actual = list(correlate(job, backend="cupy", subintegration_tiling="off"))
    assert allocations[1] == allocations[0] // 2
    assert len(actual) == len(expected)
    for candidate, reference in zip(actual, expected, strict=True):
        assert replace(candidate, spectrum=None) == replace(
            reference, spectrum=None
        )
        np.testing.assert_allclose(
            candidate.spectrum, reference.spectrum, rtol=3e-5, atol=3e-5
        )
