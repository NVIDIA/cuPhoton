# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cuphoton.xbeam.adapters.radio.engine as engine_module
import cuphoton.xbeam.adapters.vdif.ingest as ingest_module
from cuphoton.xbeam.adapters.radio import compute_plan
from cuphoton.xbeam.adapters.radio.compute_plan import (
    _required_sample_intervals,
    _timing_groups,
)
from cuphoton.xbeam.adapters.radio.engine import (
    prepare_correlation,
    prepare_vdif_ingest_plan,
)
from cuphoton.xbeam.adapters.radio.models import (
    Baseline,
    BaselineProduct,
    ClockPolynomial,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    MJDTime,
    RecordedBand,
    SamplingKind,
    Scan,
    Sideband,
    Station,
)
from cuphoton.xbeam.adapters.vdif.ingest import (
    SampleInterval,
    VdifIngestPlan,
    VdifPlanningCache,
    plan_vdif_ingest,
)
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifFileRecordingSource,
    VdifPayloadPlan,
)
from cuphoton.xbeam.adapters.vdif.vdif import (
    VdifError,
    VdifHeader,
    absolute_frame_slot,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.missing_timestamp_recovery import GlobalFFTEligibilityPlan
from cuphoton.xbeam.profiles import ANALYTIC_V1
from cuphoton.xbeam.windowing import CorrelationWindow


def _frame(
    station: str,
    seconds: int,
    *,
    thread_id: int,
    payload_value: int,
    frame_number: int = 0,
    reference_epoch: int = 40,
    extended_data_version: int = 0,
) -> bytes:
    payload = bytes((payload_value,)) * 8
    word0 = seconds
    word1 = frame_number | (reference_epoch << 24)
    word2 = (32 + len(payload)) // 8
    station_id = (ord(station[0]) << 8) | ord(station[1])
    word3 = station_id | (thread_id << 16) | ((2 - 1) << 26)
    return (
        np.asarray(
            (
                word0,
                word1,
                word2,
                word3,
                extended_data_version << 24,
                0,
                0,
                0,
            ),
            dtype="<u4",
        ).tobytes()
        + payload
    )


def _recorded_filler(*, thread_id: int = 3) -> bytes:
    return np.asarray(
        (
            0x8000_0000,
            0,
            5,
            thread_id << 16,
            0xABAD_DEED,
            0,
            0,
            0,
        ),
        dtype="<u4",
    ).tobytes() + bytes.fromhex("4433221144332211")


def _job(
    tmp_path: Path,
    *,
    source_seconds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> Job:
    paths = []
    for station in ("AA", "BB"):
        path = tmp_path / f"{station}.vdif"
        frames = []
        for second in source_seconds:
            frames.append(
                _frame(
                    station,
                    second,
                    thread_id=0,
                    payload_value=second + 1,
                )
            )
            frames.append(
                _frame(
                    station,
                    second,
                    thread_id=1,
                    payload_value=(0xE0 + second) & 0xFF,
                )
            )
        path.write_bytes(b"".join(frames))
        paths.append(path)

    epoch = MJDTime(58849, Decimal(0))
    stations = tuple(
        Station(index, name, ClockPolynomial(epoch, (0.0,)))
        for index, name in enumerate(("AA", "BB"))
    )
    frequency = Frequency(0, 0.0, 16.0, Sideband.USB, 4)
    streams = tuple(
        Datastream(
            identifier=index,
            station_id=index,
            sampling=SamplingKind.REAL,
            quantization_bits=2,
            frame_size_bytes=40,
            source_files=(str(paths[index]),),
            recorded_bands=(RecordedBand(0, 0, "R", thread_id=0),),
            sample_rate_hz=32.0,
        )
        for index in range(2)
    )

    def zero_model(station_id: int) -> DelayCoordinateModel:
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
        execution_duration_s=Decimal(5),
        integration_duration_s=Decimal(1),
        subintegration_duration_s=Decimal("0.5"),
        fringe_approximation_order=1,
        stations=stations,
        frequencies=(frequency,),
        datastreams=streams,
        baselines=(Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),)),),
        scans=(
            Scan(
                "scan",
                Decimal(0),
                Decimal(5),
                "0",
                "0",
                (zero_model(0), zero_model(1)),
            ),
        ),
        write_autocorrelations=True,
    )


def _assert_same_record(actual, expected) -> None:
    assert replace(actual, spectrum=None) == replace(expected, spectrum=None)
    np.testing.assert_array_equal(actual.spectrum, expected.spectrum)


def test_assume_ordered_window_matches_sidecar_window(tmp_path):
    job = _job(tmp_path)
    window = CorrelationWindow("scan", 2, 1)
    index_dir = tmp_path / "indexes"
    for datastream in job.datastreams:
        build_vdif_binary_index(
            datastream.source_files[0],
            frame_size_bytes=datastream.frame_size_bytes,
            index_dir=index_dir,
            backend="posix",
        )

    ordered = prepare_correlation(
        job, backend="numpy", window=window, assume_ordered=True
    )
    sidecar = prepare_correlation(
        job, backend="numpy", window=window, index_dir=str(index_dir)
    )

    assert ordered.ingest_manifest["mode"] == "assume-ordered"
    assert sidecar.ingest_manifest["mode"] == "binary-sidecar"
    ordered_records = list(ordered.records())
    sidecar_records = list(sidecar.records())
    assert len(ordered_records) == len(sidecar_records) == 3
    for candidate, reference in zip(
        ordered_records, sidecar_records, strict=True
    ):
        assert replace(candidate, spectrum=None) == replace(
            reference, spectrum=None
        )
        np.testing.assert_allclose(candidate.spectrum, reference.spectrum)


def test_binary_sidecar_replaces_ordered_probe_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    expected = prepare_correlation(
        job,
        backend="numpy",
        window=CorrelationWindow("scan", 2, 1),
        assume_ordered=True,
    )
    index_dir = tmp_path / "binary-indexes"
    for datastream in job.datastreams:
        for source in datastream.source_files:
            build_vdif_binary_index(
                source,
                frame_size_bytes=datastream.frame_size_bytes,
                index_dir=index_dir,
                backend="posix",
            )

    ingest_plan = prepare_vdif_ingest_plan(
        job,
        profile="analytic-v1",
        window=CorrelationWindow("scan", 2, 1),
        max_read_bytes=1 << 20,
        assume_ordered=True,
        index_dir=str(index_dir),
        vdif_io="cufile",
    )
    assert not ingest_plan.defer_ordered_headers
    assert ingest_plan.manifest["single_pass_effective"] is True
    assert ingest_plan.manifest["effective_io_mode"] == (
        "binary-sidecar-mapped"
    )

    prepared = prepare_correlation(
        job,
        backend="numpy",
        window=CorrelationWindow("scan", 2, 1),
        assume_ordered=True,
        index_dir=str(index_dir),
    )

    assert prepared.ingest_manifest["mode"] == "binary-sidecar"
    assert prepared.ingest_manifest["estimated_range_validation_bytes"] == 0
    assert {item["mode"] for item in prepared.ingest_manifest["sources"]} == {
        "binary-sidecar"
    }
    expected_records = list(expected.records())
    actual_records = list(prepared.records())
    assert len(actual_records) == len(expected_records)
    for actual, reference in zip(
        actual_records, expected_records, strict=True
    ):
        _assert_same_record(actual, reference)


def test_public_ingest_planner_includes_file_recovery_offsets(
    tmp_path, monkeypatch
):
    job = _job(tmp_path)
    window = CorrelationWindow("scan", 2, 1)
    ordinary = object()
    recovered = object()
    recovery_plans = {
        ("scan", 0, 0): SimpleNamespace(has_complete_source_replay=True)
    }
    calls = []

    def plan(*args, recovery_plans=None, **kwargs):
        calls.append(recovery_plans)
        return ordinary if recovery_plans is None else recovered

    monkeypatch.setattr(engine_module, "_plan_correlation_vdif_ingest", plan)
    monkeypatch.setattr(
        engine_module,
        "_build_file_recovery_plans",
        lambda *args, **kwargs: recovery_plans,
    )

    result = prepare_vdif_ingest_plan(
        job,
        profile="interop-real-usb-buffered-window-replay-v1",
        window=window,
        max_read_bytes=1 << 20,
        assume_ordered=False,
        index_dir=None,
        vdif_io="posix",
    )

    assert result is recovered
    assert calls == [None, recovery_plans]


def test_direct_replay_keeps_widened_materialized_window(monkeypatch):
    ordinary = {(0, 0): SampleInterval(100, 200)}
    widened = {(0, 0): SampleInterval(25, 200)}
    source_plan = object()
    recovery_plans = {
        ("scan", 0, 0): SimpleNamespace(
            source_shifted_fft_count=1,
            has_source_frame_presentation=False,
        )
    }
    calls = []

    monkeypatch.setattr(
        engine_module,
        "_required_sample_intervals",
        lambda *_args, **_kwargs: ordinary,
    )
    monkeypatch.setattr(
        engine_module,
        "_extend_replay_intervals",
        lambda *_args, **_kwargs: widened,
    )

    def plan(*_args, intervals, **_kwargs):
        calls.append(intervals)
        return source_plan

    monkeypatch.setattr(engine_module, "plan_vdif_ingest", plan)

    result = engine_module._plan_correlation_vdif_ingest(
        SimpleNamespace(),
        (),
        window=SimpleNamespace(),
        max_read_bytes=1 << 20,
        assume_ordered=False,
        index_dir=None,
        vdif_io="cufile",
        planning_cache=None,
        recovery_plans=recovery_plans,
    )

    assert result is source_plan
    assert calls == [widened]


def test_piecewise_replay_keeps_widened_materialized_window(monkeypatch):
    ordinary = {(0, 0): SampleInterval(100, 200)}
    widened = {(0, 0): SampleInterval(25, 200)}
    source_plan = object()
    recovery_plans = {
        ("scan", 0, 0): SimpleNamespace(
            source_shifted_fft_count=0,
            has_source_segments=True,
            has_source_frame_presentation=False,
        )
    }
    calls = []

    monkeypatch.setattr(
        engine_module,
        "_required_sample_intervals",
        lambda *_args, **_kwargs: ordinary,
    )
    monkeypatch.setattr(
        engine_module,
        "_extend_replay_intervals",
        lambda *_args, **_kwargs: widened,
    )

    def plan(*_args, intervals, **_kwargs):
        calls.append(intervals)
        return source_plan

    monkeypatch.setattr(engine_module, "plan_vdif_ingest", plan)

    result = engine_module._plan_correlation_vdif_ingest(
        SimpleNamespace(),
        (),
        window=SimpleNamespace(),
        max_read_bytes=1 << 20,
        assume_ordered=False,
        index_dir=None,
        vdif_io="cufile",
        planning_cache=None,
        recovery_plans=recovery_plans,
    )

    assert result is source_plan
    assert calls == [widened]


@pytest.mark.gpu
def test_gpu_binary_sidecar_uses_static_mapped_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.xbeam.test_gpu_backend import _cupy

    _cupy()
    job = _job(tmp_path)
    index_dir = tmp_path / "binary-indexes"
    for datastream in job.datastreams:
        for source in datastream.source_files:
            build_vdif_binary_index(
                source,
                frame_size_bytes=datastream.frame_size_bytes,
                index_dir=index_dir,
                backend="posix",
            )
    window = CorrelationWindow("scan", 2, 1)
    expected = list(
        prepare_correlation(
            job,
            backend="numpy",
            window=window,
            assume_ordered=True,
            index_dir=str(index_dir),
        ).records()
    )
    prepared = prepare_correlation(
        job,
        backend="cupy",
        window=window,
        assume_ordered=True,
        index_dir=str(index_dir),
        vdif_io="posix",
    )

    assert prepared.ingest_manifest["effective_io_mode"] == (
        "binary-sidecar-mapped"
    )
    actual = list(prepared.records())
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        _assert_same_record(observed, reference)


def test_vdif_ingest_plan_preserves_legacy_positional_construction() -> None:
    plan = VdifIngestPlan(None, None, None, 0, None, None, {})

    assert plan.manifest == {}
    assert plan.defer_ordered_headers is False


def test_assume_ordered_cpu_planning_preserves_header_validation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    observed = []
    locate = ingest_module.locate_ordered_vdif_range

    def record_validation_mode(*args, **kwargs):
        observed.append(kwargs.get("validate_all_headers"))
        return locate(*args, **kwargs)

    monkeypatch.setattr(
        ingest_module, "locate_ordered_vdif_range", record_validation_mode
    )

    prepared = prepare_correlation(
        job,
        backend="numpy",
        window=CorrelationWindow("scan", 2, 1),
        assume_ordered=True,
    )

    assert observed and set(observed) == {True}
    assert prepared.ingest_manifest["single_pass_requested"] is False
    assert prepared.ingest_manifest["single_pass_effective"] is False
    assert prepared.ingest_manifest["effective_io_mode"] == (
        "ordered-validated-three-pass"
    )
    assert prepared.ingest_manifest["estimated_range_validation_bytes"] == (
        prepared.estimated_read_bytes
    )
    assert prepared.ingest_manifest["estimated_header_pass_bytes"] == (
        prepared.estimated_read_bytes
    )
    assert prepared.ingest_manifest["estimated_total_io_bytes"] == (
        prepared.estimated_read_bytes * 3
    )
    assert len(list(prepared.records())) == 3


def test_planning_cache_reuses_unchanged_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    intervals = {
        (datastream.identifier, 0): SampleInterval(64, 96)
        for datastream in job.datastreams
    }
    cache = VdifPlanningCache()
    calls = 0
    inspect = ingest_module.inspect_vdif

    def count_inspections(*args, **kwargs):
        nonlocal calls
        calls += 1
        return inspect(*args, **kwargs)

    monkeypatch.setattr(ingest_module, "inspect_vdif", count_inspections)

    plans = [
        plan_vdif_ingest(
            job,
            intervals=intervals,
            max_read_bytes=1 << 20,
            assume_ordered=True,
            index_dir=None,
            defer_ordered_headers=True,
            planning_cache=cache,
        )
        for _ in range(2)
    ]

    assert calls == len(job.datastreams)
    assert all(
        not source["metadata_cache_hit"]
        for source in plans[0].manifest["sources"]
    )
    assert all(
        source["metadata_cache_hit"]
        for source in plans[1].manifest["sources"]
    )


def test_bulk_manifest_explicitly_records_no_thread_aliases(tmp_path):
    plan = plan_vdif_ingest(
        _job(tmp_path),
        intervals=None,
        max_read_bytes=1 << 20,
        assume_ordered=False,
        index_dir=None,
    )
    assert plan.manifest["thread_aliases"] == []


def test_bulk_planning_rejects_deferred_ordered_headers(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires bounded sample intervals",
    ):
        plan_vdif_ingest(
            _job(tmp_path),
            intervals=None,
            max_read_bytes=1 << 20,
            assume_ordered=True,
            index_dir=None,
            defer_ordered_headers=True,
        )


def test_ordered_planning_rejects_unsampled_timestamp_disorder(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seconds = list(range(100))
    seconds[52], seconds[53] = seconds[53], seconds[52]
    monkeypatch.setattr(ingest_module, "_DIRECT_METADATA_PROBES", 17)
    job = _job(tmp_path, source_seconds=tuple(seconds))
    intervals = {
        (datastream.identifier, 0): SampleInterval(52 * 32, 53 * 32)
        for datastream in job.datastreams
    }

    with pytest.raises(ConfigurationError, match="ordering decreases"):
        plan_vdif_ingest(
            job,
            intervals=intervals,
            max_read_bytes=1 << 20,
            assume_ordered=True,
            index_dir=None,
        )


def test_ordered_planning_rejects_unsampled_extended_format_change(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_module, "_DIRECT_METADATA_PROBES", 17)
    job = _job(tmp_path, source_seconds=tuple(range(100)))
    for datastream in job.datastreams:
        path = Path(datastream.source_files[0])
        raw = bytearray(path.read_bytes())
        frame_index = 52 * 2
        word4_offset = frame_index * 40 + 16
        raw[word4_offset : word4_offset + 4] = (1 << 24).to_bytes(4, "little")
        path.write_bytes(raw)
    intervals = {
        (datastream.identifier, 0): SampleInterval(52 * 32, 53 * 32)
        for datastream in job.datastreams
    }

    with pytest.raises(ConfigurationError, match="format changes"):
        plan_vdif_ingest(
            job,
            intervals=intervals,
            max_read_bytes=1 << 20,
            assume_ordered=True,
            index_dir=None,
        )


def test_ordered_plan_spans_reference_epoch_boundary(tmp_path: Path) -> None:
    job = _job(tmp_path)
    epoch_frames = (
        (40, 15_724_798),
        (40, 15_724_799),
        (41, 0),
        (41, 1),
    )
    for datastream, station in zip(
        job.datastreams, ("AA", "BB"), strict=True
    ):
        path = Path(datastream.source_files[0])
        path.write_bytes(
            b"".join(
                _frame(
                    station,
                    seconds,
                    thread_id=thread_id,
                    payload_value=index + 1,
                    reference_epoch=reference_epoch,
                )
                for index, (reference_epoch, seconds) in enumerate(
                    epoch_frames
                )
                for thread_id in (0, 1)
            )
        )
    job = replace(
        job,
        start_time=MJDTime(59030, Decimal(86399)),
        execution_duration_s=Decimal(2),
    )
    intervals = {
        (datastream.identifier, 0): SampleInterval(0, 32)
        for datastream in job.datastreams
    }

    plan = plan_vdif_ingest(
        job,
        intervals=intervals,
        max_read_bytes=1 << 20,
        assume_ordered=True,
        index_dir=None,
        defer_ordered_headers=True,
    )

    assert plan.defer_ordered_headers is True
    for windows in plan.thread_windows.values():
        window = windows[0]
        assert window.template.reference_epoch == 40
        assert window.stop_slot > window.start_slot


def test_distinct_file_datastreams_build_batches_concurrently(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_correlation(
        _job(tmp_path),
        backend="numpy",
        window=CorrelationWindow("scan", 2, 1),
        assume_ordered=True,
    )
    original = prepared.source.batches
    rendezvous = threading.Barrier(2)

    def synchronized_batches(datastream):
        rendezvous.wait(timeout=2.0)
        yield from original(datastream)

    monkeypatch.setattr(prepared.source, "batches", synchronized_batches)

    assert len(list(prepared.records())) == 3


def test_assume_ordered_resolves_required_threads_together(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    datastreams = tuple(
        replace(
            datastream,
            recorded_bands=(
                datastream.recorded_bands[0],
                RecordedBand(1, 0, "R", thread_id=1),
            ),
        )
        for datastream in job.datastreams
    )
    job = replace(job, datastreams=datastreams)
    intervals = {
        (datastream.identifier, band.identifier): SampleInterval(64, 96)
        for datastream in datastreams
        for band in datastream.recorded_bands
    }
    observed = []
    locate = ingest_module.locate_ordered_vdif_range

    def record_threads(*args, thread_ids, **kwargs):
        observed.append(thread_ids)
        return locate(*args, thread_ids=thread_ids, **kwargs)

    monkeypatch.setattr(
        ingest_module, "locate_ordered_vdif_range", record_threads
    )

    plan_vdif_ingest(
        job,
        intervals=intervals,
        max_read_bytes=1 << 20,
        assume_ordered=True,
        index_dir=None,
    )

    assert observed == [(0, 1), (0, 1)]


def test_assume_ordered_keeps_distinct_thread_windows_separate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path)
    datastreams = tuple(
        replace(
            datastream,
            recorded_bands=(
                datastream.recorded_bands[0],
                RecordedBand(1, 0, "R", thread_id=1),
            ),
        )
        for datastream in job.datastreams
    )
    job = replace(job, datastreams=datastreams)
    intervals = {
        (datastream.identifier, band.identifier): SampleInterval(
            64 + band.identifier * 32,
            96 + band.identifier * 32,
        )
        for datastream in datastreams
        for band in datastream.recorded_bands
    }
    observed = []
    locate = ingest_module.locate_ordered_vdif_range

    def record_threads(*args, thread_ids, **kwargs):
        observed.append(thread_ids)
        return locate(*args, thread_ids=thread_ids, **kwargs)

    monkeypatch.setattr(
        ingest_module, "locate_ordered_vdif_range", record_threads
    )

    plan_vdif_ingest(
        job,
        intervals=intervals,
        max_read_bytes=1 << 20,
        assume_ordered=True,
        index_dir=None,
    )

    assert observed == [(0,), (1,), (0,), (1,)]


def test_index_metadata_allows_empty_selected_thread_coverage(tmp_path):
    job = _job(tmp_path, source_seconds=(0,))
    index_dir = tmp_path / "indexes"
    for datastream in job.datastreams:
        build_vdif_binary_index(
            datastream.source_files[0],
            index_dir=index_dir,
            backend="posix",
        )
    prepared = prepare_correlation(
        job,
        backend="numpy",
        window=CorrelationWindow("scan", 2, 1),
        index_dir=str(index_dir),
    )

    batches = list(prepared.source.batches(job.datastreams[0]))

    assert prepared.ingest_manifest["mode"] == "binary-sidecar"
    assert len(batches) == 1
    assert not np.any(batches[0].validity)
    assert isinstance(batches[0].samples, VdifPayloadPlan)
    np.testing.assert_array_equal(
        batches[0].samples.materialize(), np.zeros(24, dtype=np.uint8)
    )


@pytest.mark.parametrize("assume_ordered", [True, False])
def test_single_logical_thread_aliases_single_physical_thread(
    tmp_path, assume_ordered
):
    job = _job(tmp_path)
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
        if not assume_ordered:
            build_vdif_binary_index(
                path,
                index_dir=tmp_path / "indexes",
                backend="posix",
            )
    intervals = {
        (datastream.identifier, 0): SampleInterval(64, 96)
        for datastream in job.datastreams
    }

    plan = plan_vdif_ingest(
        job,
        intervals=intervals,
        max_read_bytes=1 << 20,
        assume_ordered=assume_ordered,
        index_dir=tmp_path / "indexes",
    )
    source = VdifFileRecordingSource(
        job,
        ANALYTIC_V1,
        source_ranges=plan.source_ranges,
        thread_windows=plan.thread_windows,
        max_materialized_bytes=1 << 20,
        binary_sidecars=not assume_ordered,
        binary_index_dir=tmp_path / "indexes",
    )

    assert plan.manifest["thread_aliases"] == [
        {
            "datastream_id": 0,
            "logical_thread_id": 0,
            "physical_thread_id": 7,
        },
        {
            "datastream_id": 1,
            "logical_thread_id": 0,
            "physical_thread_id": 7,
        },
    ]
    assert plan.thread_windows[0][0].template.thread_id == 7
    batches = list(source.batches(job.datastreams[0]))
    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0].validity, [True, True, True])
    np.testing.assert_array_equal(
        batches[0].samples.materialize(),
        np.concatenate(
            [np.full(8, second + 1, dtype=np.uint8) for second in (1, 2, 3)]
        ),
    )


def test_recorded_filler_does_not_defeat_single_thread_alias(tmp_path):
    job = _job(tmp_path)
    index_dir = tmp_path / "indexes"
    for datastream in job.datastreams:
        station = job.stations[datastream.station_id].name
        path = Path(datastream.source_files[0])
        path.write_bytes(
            _frame(
                station,
                0,
                thread_id=7,
                payload_value=1,
            )
            + _recorded_filler()
            + b"".join(
                _frame(
                    station,
                    second,
                    thread_id=7,
                    payload_value=second + 1,
                )
                for second in range(1, 5)
            )
        )
        build_vdif_binary_index(path, index_dir=index_dir, backend="posix")

    plan = plan_vdif_ingest(
        job,
        intervals={
            (datastream.identifier, 0): SampleInterval(64, 96)
            for datastream in job.datastreams
        },
        max_read_bytes=1 << 20,
        assume_ordered=False,
        index_dir=index_dir,
    )

    assert [
        item["physical_thread_id"] for item in plan.manifest["thread_aliases"]
    ] == [
        7,
        7,
    ]


def _aliased_job(tmp_path, *, invalid_timestamp=False):
    job = _job(tmp_path)
    for datastream in job.datastreams:
        Path(datastream.source_files[0]).write_bytes(
            b"".join(
                _frame(
                    job.stations[datastream.station_id].name,
                    second,
                    thread_id=7,
                    payload_value=second + 1,
                    frame_number=int(invalid_timestamp and second == 2),
                )
                for second in range(5)
            )
        )
    return job


def test_alias_planning_error_identifies_logical_and_physical_threads(
    tmp_path,
):
    job = _aliased_job(tmp_path)
    job = replace(
        job,
        datastreams=tuple(
            replace(stream, sample_rate_hz=31.0) for stream in job.datastreams
        ),
    )
    with pytest.raises(ConfigurationError, match="integral VDIF") as caught:
        plan_vdif_ingest(
            job,
            intervals={
                (stream.identifier, 0): SampleInterval(64, 96)
                for stream in job.datastreams
            },
            max_read_bytes=1 << 20,
            assume_ordered=True,
            index_dir=None,
        )
    assert "logical thread 0 maps to physical thread 7" in str(caught.value)
    assert type(caught.value.__cause__) is type(caught.value)


@pytest.mark.parametrize("mode", ["ordered", "sidecar", "streaming"])
def test_alias_recording_error_identifies_logical_and_physical_threads(
    tmp_path, mode
):
    job = _aliased_job(tmp_path)
    if mode == "sidecar":
        for stream in job.datastreams:
            build_vdif_binary_index(stream.source_files[0], backend="posix")
    plan = plan_vdif_ingest(
        job,
        intervals={
            (stream.identifier, 0): SampleInterval(64, 96)
            for stream in job.datastreams
        },
        max_read_bytes=1 << 20,
        assume_ordered=mode != "sidecar",
        index_dir=None,
        defer_ordered_headers=mode == "streaming",
    )
    changed = replace(job.datastreams[0], sample_rate_hz=64.0)
    source = VdifFileRecordingSource(
        job,
        ANALYTIC_V1,
        source_ranges=plan.source_ranges,
        thread_windows=plan.thread_windows,
        binary_sidecars=mode == "sidecar",
        defer_ordered_headers=mode == "streaming",
    )
    with pytest.raises(
        ConfigurationError, match="frame rate changed"
    ) as caught:
        list(source.batches(changed))
    assert "logical thread 0 maps to physical thread 7" in str(caught.value)
    assert type(caught.value.__cause__) is type(caught.value)


@pytest.mark.parametrize("sidecar", [False, True])
def test_alias_timestamp_error_keeps_vdif_exception_type(tmp_path, sidecar):
    job = _aliased_job(tmp_path, invalid_timestamp=True)
    if sidecar:
        for stream in job.datastreams:
            build_vdif_binary_index(stream.source_files[0], backend="posix")
    plan = plan_vdif_ingest(
        job,
        intervals={
            (stream.identifier, 0): SampleInterval(64, 96)
            for stream in job.datastreams
        },
        max_read_bytes=1 << 20,
        assume_ordered=not sidecar,
        index_dir=None,
    )
    source = VdifFileRecordingSource(
        job,
        ANALYTIC_V1,
        source_ranges=plan.source_ranges,
        thread_windows=plan.thread_windows,
        binary_sidecars=sidecar,
    )
    with pytest.raises(
        VdifError, match="frame number 1 is outside"
    ) as caught:
        list(source.batches(job.datastreams[0]))
    assert "logical thread 0 maps to physical thread 7" in str(caught.value)
    assert type(caught.value.__cause__) is type(caught.value)


@pytest.mark.parametrize("assume_ordered", [True, False])
def test_single_logical_thread_does_not_alias_ambiguous_physical_threads(
    tmp_path,
    assume_ordered,
):
    job = _job(tmp_path)
    for datastream in job.datastreams:
        path = Path(datastream.source_files[0])
        path.write_bytes(
            b"".join(
                _frame(
                    job.stations[datastream.station_id].name,
                    second,
                    thread_id=thread_id,
                    payload_value=second + 1,
                )
                for second in range(5)
                for thread_id in (6, 7)
            )
        )
        if not assume_ordered:
            build_vdif_binary_index(
                path,
                index_dir=tmp_path / "indexes",
                backend="posix",
            )
    intervals = {
        (datastream.identifier, 0): SampleInterval(64, 96)
        for datastream in job.datastreams
    }

    with pytest.raises(ConfigurationError, match="thread 0 is unavailable"):
        plan_vdif_ingest(
            job,
            intervals=intervals,
            max_read_bytes=1 << 20,
            assume_ordered=assume_ordered,
            index_dir=tmp_path / "indexes",
        )


@pytest.mark.parametrize("interior_thread", [0, 7])
def test_single_observed_thread_requires_complete_inventory_to_alias(
    tmp_path,
    interior_thread,
):
    job = _job(tmp_path)
    for datastream in job.datastreams:
        path = Path(datastream.source_files[0])
        station = job.stations[datastream.station_id].name
        path.write_bytes(
            b"".join(
                _frame(
                    station,
                    second,
                    thread_id=interior_thread if second == 3001 else 7,
                    payload_value=0xAA if second == 3001 else 0x77,
                )
                for second in range(6001)
            )
        )
    intervals = {
        (datastream.identifier, 0): SampleInterval(3001 * 32, 3002 * 32)
        for datastream in job.datastreams
    }

    with pytest.raises(ConfigurationError, match="build an index"):
        plan_vdif_ingest(
            job,
            intervals=intervals,
            max_read_bytes=1 << 20,
            assume_ordered=True,
            index_dir=None,
        )

    if interior_thread == 7:
        return
    for datastream in job.datastreams:
        build_vdif_binary_index(datastream.source_files[0], backend="posix")
    plan = plan_vdif_ingest(
        job,
        intervals=intervals,
        max_read_bytes=1 << 20,
        assume_ordered=True,
        index_dir=None,
    )
    assert plan.manifest["thread_aliases"] == []
    assert all(
        source["thread_inventory_complete"]
        for source in plan.manifest["sources"]
    )
    source = VdifFileRecordingSource(
        job,
        ANALYTIC_V1,
        source_ranges=plan.source_ranges,
        thread_windows=plan.thread_windows,
        max_materialized_bytes=1 << 20,
        binary_sidecars=True,
    )
    (batch,) = source.batches(job.datastreams[0])
    np.testing.assert_array_equal(batch.validity, [False, True, False])
    np.testing.assert_array_equal(
        batch.samples.materialize()[8:16], np.full(8, 0xAA, dtype=np.uint8)
    )


def test_empty_bounded_thread_is_not_limited_by_observed_frame_count(
    tmp_path,
):
    job = _job(tmp_path, source_seconds=(10,))
    job = replace(
        job,
        execution_duration_s=Decimal(1),
        scans=(replace(job.scans[0], duration_s=Decimal(1)),),
    )
    index_dir = tmp_path / "indexes"
    for datastream in job.datastreams:
        build_vdif_binary_index(
            datastream.source_files[0],
            index_dir=index_dir,
            backend="posix",
        )
    prepared = prepare_correlation(
        job,
        backend="numpy",
        window=CorrelationWindow("scan", 0, 1),
        index_dir=str(index_dir),
    )

    batches = list(prepared.source.batches(job.datastreams[0]))

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0].validity, [False, False, False])


@pytest.mark.parametrize("assume_ordered", [True, False])
@pytest.mark.parametrize("corrupt", [True, False])
def test_existing_bad_sidecar_never_falls_back_to_ordered_probes(
    tmp_path, assume_ordered, corrupt
):
    job = _job(tmp_path)
    directory = tmp_path / "indexes"
    for datastream in job.datastreams:
        source = Path(datastream.source_files[0])
        index = build_vdif_binary_index(
            source, index_dir=directory, backend="posix"
        ).index_path
        if corrupt:
            index.write_bytes(b"")
        else:
            with source.open("r+b") as stream:
                stream.seek(32)
                stream.write(b"\xff")

    with pytest.raises(ConfigurationError, match="sidecar"):
        plan_vdif_ingest(
            job,
            intervals={
                (stream.identifier, 0): SampleInterval(64, 96)
                for stream in job.datastreams
            },
            max_read_bytes=1 << 20,
            assume_ordered=assume_ordered,
            index_dir=directory,
        )


@pytest.mark.parametrize("assume_ordered", [True, False])
def test_candidate_extents_are_filtered_to_planned_thread_window(
    tmp_path, assume_ordered
):
    job = _job(tmp_path)
    if not assume_ordered:
        for datastream in job.datastreams:
            build_vdif_binary_index(
                datastream.source_files[0],
                index_dir=tmp_path / "indexes",
                backend="posix",
            )
    intervals = {
        (datastream.identifier, 0): SampleInterval(64, 96)
        for datastream in job.datastreams
    }
    plan = plan_vdif_ingest(
        job,
        intervals=intervals,
        max_read_bytes=1 << 20,
        assume_ordered=assume_ordered,
        index_dir=tmp_path / "indexes",
    )
    ranges = plan.source_ranges[job.datastreams[0].identifier]
    source = VdifFileRecordingSource(
        job,
        ANALYTIC_V1,
        source_ranges=plan.source_ranges,
        thread_windows=plan.thread_windows,
        max_materialized_bytes=1 << 20,
    )

    batches = list(source.batches(job.datastreams[0]))

    assert ranges
    assert sum(item.frame_count for item in ranges) > 3
    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0].validity, [True, True, True])
    assert isinstance(batches[0].samples, VdifPayloadPlan)
    expected_payload = np.concatenate(
        [np.full(8, second + 1, dtype=np.uint8) for second in (1, 2, 3)]
    )
    np.testing.assert_array_equal(
        batches[0].samples.materialize(), expected_payload
    )


@pytest.mark.parametrize(
    ("clock_offset_s", "expected"),
    [
        (1.0 / 64.0, SampleInterval(64, 96)),
        (3.0 / 64.0, SampleInterval(62, 94)),
        (-3.0 / 64.0, SampleInterval(66, 98)),
    ],
    ids=("half-to-even", "positive-delay", "negative-delay"),
)
def test_required_sample_halo_uses_execution_rounding(
    tmp_path, clock_offset_s, expected
):
    job = _job(tmp_path)
    datastreams = tuple(
        replace(
            datastream,
            recorded_bands=(
                replace(
                    datastream.recorded_bands[0],
                    clock_offset_s=clock_offset_s,
                ),
            ),
        )
        for datastream in job.datastreams
    )
    job = replace(job, datastreams=datastreams)
    selected = [
        group for group in _timing_groups(job, ANALYTIC_V1) if group[0] == 2
    ]

    intervals = _required_sample_intervals(job, selected)

    assert intervals == {(0, 0): expected, (1, 0): expected}


def test_polynomial_sample_bounds_contain_discrete_execution_range(tmp_path):
    job = _job(tmp_path)
    datastreams = tuple(
        replace(
            datastream,
            recorded_bands=(
                replace(
                    datastream.recorded_bands[0],
                    clock_offset_s=0.001,
                ),
            ),
        )
        for datastream in job.datastreams
    )
    models = tuple(
        replace(
            model,
            delay_coefficients_s=(0.002, -0.003, 0.001),
        )
        for model in job.scans[0].station_models
    )
    job = replace(
        job,
        datastreams=datastreams,
        scans=(replace(job.scans[0], station_models=models),),
    )
    selected = [
        group for group in _timing_groups(job, ANALYTIC_V1) if group[0] == 2
    ]

    bounded = _required_sample_intervals(job, selected)
    scan = selected[0][1]
    subintegrations = [
        value for _index, _scan, values in selected for value in values
    ]
    for datastream in job.datastreams:
        band = datastream.recorded_bands[0]
        frequency = job.frequencies[band.frequency_id]
        sample_rate = compute_plan._sample_rate(datastream, frequency)
        fft_length = frequency.channel_count * 2
        starts = compute_plan._block_starts(
            job,
            sample_rate,
            fft_length,
            scan,
            subintegrations,
        )
        _, _, coarse, output = compute_plan._source_block_indices_for_scan(
            job,
            datastream,
            band,
            frequency,
            scan,
            starts,
        )
        exact = SampleInterval(
            int(np.min(output - coarse)),
            int(np.max(output - coarse + fft_length)),
        )
        actual = bounded[(datastream.identifier, band.identifier)]
        assert actual.first_sample <= exact.first_sample
        assert actual.stop_sample >= exact.stop_sample


def test_segmented_sample_bounds_contain_delay_jump(tmp_path):
    job = _job(tmp_path)
    first_models = tuple(
        replace(model, delay_coefficients_s=(-0.25,))
        for model in job.scans[0].station_models
    )
    second_models = tuple(
        replace(
            model,
            epoch_offset_s=Decimal("2.5"),
            delay_coefficients_s=(0.5,),
        )
        for model in first_models
    )
    scan = replace(job.scans[0], station_models=first_models + second_models)
    job = replace(job, scans=(scan,))
    selected = [
        group for group in _timing_groups(job, ANALYTIC_V1) if group[0] == 2
    ]

    bounded = _required_sample_intervals(job, selected)
    subintegrations = [
        value for _index, _scan, values in selected for value in values
    ]
    for datastream in job.datastreams:
        band = datastream.recorded_bands[0]
        frequency = job.frequencies[band.frequency_id]
        sample_rate = compute_plan._sample_rate(datastream, frequency)
        starts = compute_plan._block_starts(
            job,
            sample_rate,
            frequency.channel_count * 2,
            scan,
            subintegrations,
        )
        fft_length, _, coarse, output = (
            compute_plan._source_block_indices_for_scan(
                job, datastream, band, frequency, scan, starts
            )
        )
        np.testing.assert_array_equal(np.unique(coarse), (-8, 16))
        exact = SampleInterval(
            int(np.min(output - coarse)),
            int(np.max(output - coarse + fft_length)),
        )
        actual = bounded[(datastream.identifier, band.identifier)]
        assert actual.first_sample <= exact.first_sample
        assert actual.stop_sample >= exact.stop_sample


def test_prepared_window_rejects_same_size_atomic_source_replacement(
    tmp_path,
):
    job = _job(tmp_path)
    prepared = prepare_correlation(
        job,
        backend="numpy",
        window=CorrelationWindow("scan", 2, 1),
        assume_ordered=True,
    )
    source = Path(job.datastreams[0].source_files[0])
    original_stat = source.stat()
    replacement = tmp_path / "replacement.vdif"
    replacement_bytes = bytearray(source.read_bytes())
    replacement_bytes[-1] ^= 0xFF
    replacement.write_bytes(replacement_bytes)
    os.utime(
        replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    replacement.replace(source)

    with pytest.raises(OSError, match="changed after ingest planning"):
        list(prepared.records())


def test_validated_headers_reject_atomic_source_replacement(tmp_path):
    job = _job(tmp_path)
    prepared = prepare_correlation(
        job,
        backend="numpy",
        window=CorrelationWindow("scan", 2, 1),
        assume_ordered=True,
    )
    source = Path(job.datastreams[0].source_files[0])
    original_stat = source.stat()
    replacement = tmp_path / "headers-replacement.vdif"
    replacement.write_bytes(source.read_bytes())
    os.utime(
        replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    replacement.replace(source)

    with pytest.raises(OSError, match="changed after ingest planning"):
        list(prepared.source.batches(job.datastreams[0]))


def test_prepared_ordered_run_rejects_source_growth_after_guard(tmp_path):
    job = _job(tmp_path)
    prepared = prepare_correlation(
        job,
        backend="numpy",
        max_read_bytes=3000,
        assume_ordered=True,
    )
    source = Path(job.datastreams[0].source_files[0])
    source.write_bytes(source.read_bytes() + source.read_bytes()[:40])

    with pytest.raises(OSError, match="changed after ingest planning"):
        list(prepared.records())


def test_unwindowed_ordered_plan_limits_payload_to_scheduled_groups(
    tmp_path,
):
    job = _job(tmp_path)
    job = replace(
        job,
        execution_duration_s=Decimal(2),
        scans=(replace(job.scans[0], duration_s=Decimal(2)),),
    )

    prepared = prepare_correlation(
        job,
        backend="numpy",
        max_read_bytes=2000,
        assume_ordered=True,
    )
    batches = list(prepared.source.batches(job.datastreams[0]))

    assert prepared.ingest_manifest["mode"] == "assume-ordered"
    assert prepared.ingest_manifest["bounded"] is True
    assert len(batches) == 1
    assert isinstance(batches[0].samples, VdifPayloadPlan)
    assert batches[0].samples.slot_count == 4
    np.testing.assert_array_equal(
        batches[0].validity, [False, True, True, True]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(batches[0].samples.files[0].destination_slots >= 0),
        [0, 2, 4],
    )


def test_window_rejects_legacy_custom_recording_source(tmp_path):
    job = _job(tmp_path)
    custom_source = VdifFileRecordingSource(job, ANALYTIC_V1)

    with pytest.raises(ConfigurationError, match="recording_source"):
        prepare_correlation(
            job,
            backend="numpy",
            window=CorrelationWindow("scan", 0, 1),
            recording_source=custom_source,
        )


def test_unordered_source_requires_binary_sidecar(tmp_path):
    job = _job(tmp_path)
    intervals = {
        (datastream.identifier, 0): SampleInterval(64, 96)
        for datastream in job.datastreams
    }

    with pytest.raises(ConfigurationError, match="sidecar does not exist"):
        plan_vdif_ingest(
            job,
            intervals=intervals,
            max_read_bytes=100,
            assume_ordered=False,
            index_dir=tmp_path / "indexes",
        )


@pytest.mark.parametrize("assume_ordered", [False, True])
def test_unwindowed_timestamp_recovery_matches_full_window(
    tmp_path, monkeypatch, assume_ordered
):
    job = _job(tmp_path, source_seconds=tuple(range(9)))
    for datastream in job.datastreams:
        build_vdif_binary_index(
            datastream.source_files[0], frame_size_bytes=40, backend="posix"
        )
    header = VdifHeader.from_bytes(
        _frame("AA", 0, thread_id=0, payload_value=0)
    )
    first_slot = absolute_frame_slot(header, 1)
    # Physical seconds 2..7 are presented at job seconds 0..5. The read
    # extent moves by two frames; the execution timeline must stay fixed.
    recovery = GlobalFFTEligibilityPlan(
        missing_timestamps=(),
        eligible_ffts=(),
        checkpoint_stops=(),
        checkpoints=(),
        calibration_identifier="test",
        timeline_fft_count=20,
        source_frame_intervals=(
            (first_slot + 2, first_slot + 7, first_slot),
        ),
        source_ingest_offset_intervals=((0, 20, 64, 64),),
    )
    plans = {
        ("scan", stream.identifier, 0): recovery for stream in job.datastreams
    }
    monkeypatch.setattr(
        engine_module,
        "get_profile",
        lambda _name: replace(ANALYTIC_V1, missing_timestamp_recovery="test"),
    )
    kwargs = dict(recovery_plans=plans, assume_ordered=assume_ordered)
    expected = list(
        prepare_correlation(
            job, window=CorrelationWindow("scan", 0, 5), **kwargs
        ).records()
    )
    actual = list(prepare_correlation(job, **kwargs).records())

    assert len(actual) == len(expected) == 15
    for record, reference in zip(actual, expected, strict=True):
        _assert_same_record(record, reference)
