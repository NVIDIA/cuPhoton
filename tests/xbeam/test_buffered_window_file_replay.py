# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import (
    Baseline,
    BaselineProduct,
    BufferedReaderGeometry,
)
from cuphoton.xbeam.adapters.vdif.ingest import (
    SampleInterval,
    plan_vdif_ingest,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.buffered_reader.adapter import (
    build_raw_file_replay_plans,
    build_returned_window_replay_plans,
)
from cuphoton.xbeam.buffered_reader.replay import (
    RetainedRawWindowContract,
)
from tests.xbeam.test_buffered_window_job_requests import model_job
from tests.xbeam.test_recording import _frame
from tests.xbeam.test_vdif_sidecar import _recorded_filler

RATE, FFT = 4096000000, 262144


@pytest.fixture
def recording(tmp_path, request):
    parameter = getattr(request, "param", 1000)
    frames, channels = (
        parameter if isinstance(parameter, tuple) else (parameter, 1)
    )
    path = tmp_path / "own-input.vdif"
    with path.open("wb") as stream:
        for physical in range(frames):
            stream.write(
                _frame(
                    payload=bytes([physical % 251]) * 8192,
                    frame_number=physical,
                    channels=channels,
                    invalid=physical == 431,
                )
            )
    indexes = tmp_path / "indexes"
    build_vdif_binary_index(path, index_dir=indexes, backend="posix")
    job, ds, band, frequency, scan, _geometry = model_job(
        rate=RATE // channels, fft=FFT // channels
    )
    band = replace(band, thread_id=5)
    ds = replace(ds, source_files=(str(path),), recorded_bands=(band,))
    duration = Decimal("0.0049" if frames == 1000 else "0.0098")
    scan = replace(scan, duration_s=duration)
    job = replace(
        job,
        datastreams=(ds,),
        scans=(scan,),
        execution_duration_s=duration,
        subintegration_duration_s=Decimal("0.0032"),
        buffered_reader=BufferedReaderGeometry(24, 8, 20),
    )
    ingest = plan_vdif_ingest(
        job,
        intervals={
            (0, 0): SampleInterval(
                50 * FFT // channels, 100 * FFT // channels
            )
        },
        max_read_bytes=1 << 23,
        assume_ordered=False,
        index_dir=indexes,
    )
    return job, scan, ingest, indexes


def build(args, *, indices=(1,), **kwargs):
    job, scan, ingest, indexes = args
    if "window_clock_contract_factory" not in kwargs:
        return build_returned_window_replay_plans(
            job,
            [(0, scan, indices)],
            ingest,
            index_dir=str(indexes),
            alignment_bytes=8192,
            word_bytes=4,
            control_word_bits=30,
            timestamp_quantum_nanoseconds=1,
            replay_identifier="independent-file-adapter-test",
            **kwargs,
        )
    return build_raw_file_replay_plans(
        job,
        [(0, scan, indices)],
        ingest,
        index_dir=str(indexes),
        alignment_bytes=8192,
        word_bytes=4,
        control_word_bits=30,
        **kwargs,
    )


def test_actual_file_models_and_ingest_keep_complete_final_subintegration(
    recording,
):
    plans = build(recording)
    plan = plans[("ordinary", 0, 0)]
    # Physical midpoint admits subintegration1 although its FFT tail reaches
    # 0.0064s, beyond the scan's0.0049s endpoint (nominal FFT ceiling77).
    assert plan.fft_count == 100
    assert plan.timeline_fft_count == 100
    assert not plan.ineligible_intervals
    assert not plan.source_validity_is_authoritative
    assert not plan.source_weight_override_events
    assert plan.source_physical_frame_intervals[0][1:3] == (400, 800)
    for fft in (50, 53, 99):
        assert plan.source_replay(fft) == (0, 0)
    tiled = build(recording, maximum_requests_per_tile=1)
    assert plans == tiled
    with pytest.raises(TypeError):
        plans["unexpected"] = plan


@pytest.mark.parametrize("recording", (750,), indirect=True)
def test_actual_short_file_clips_payload_without_unknown_return(recording):
    plans = build(recording)
    plan = plans[("ordinary", 0, 0)]
    # Eight physical frames fill each FFT. Frame 750 lies three quarters
    # through FFT 93; subsequent FFTs have no physical payload. This fixture
    # builds and reads a real sidecar with no patched planner or timeline.
    assert plan.fft_count == 100
    assert plan.source_validity_events == ((93, 0, 3 * FFT // 4),)
    # This EOF mask intersects ingest validity; it must not re-enable the
    # explicitly invalid physical frame elsewhere in the recording.
    assert not plan.source_validity_is_authoritative
    assert plan.ineligible_intervals == ((94, 100),)
    assert plan.source_physical_frame_intervals[0][1:3] == (400, 750)
    assert plans == build(recording, maximum_requests_per_tile=1)


@pytest.mark.parametrize("indices", ((1,), (1, 2)))
def test_trailing_recorded_filler_is_independent_of_window_and_tile(
    recording, indices
):
    job, scan, _ingest, indexes = recording
    scan = replace(scan, duration_s=Decimal("0.0098"))
    job = replace(job, execution_duration_s=scan.duration_s, scans=(scan,))
    source = Path(job.datastreams[0].source_files[0])
    with source.open("r+b") as stream:
        stream.seek(750 * 8224)
        stream.write(_recorded_filler(frame_size=8224) * 250)
    build_vdif_binary_index(
        source, index_dir=indexes, backend="posix", force=True
    )
    ingest = plan_vdif_ingest(
        job,
        intervals={(0, 0): SampleInterval(50 * FFT, 150 * FFT)},
        max_read_bytes=1 << 23,
        assume_ordered=False,
        index_dir=indexes,
    )
    arguments = (job, scan, ingest, indexes)
    plans = build(arguments, indices=indices)
    plan = plans[("ordinary", 0, 0)]
    # Eight physical frames fill each FFT. The trailing filler begins
    # three quarters through FFT 93, regardless of the requested scope.
    assert plan.source_validity_events == ((93, 0, 3 * FFT // 4),)
    assert plan.ineligible_intervals == ((94, 50 * (indices[-1] + 1)),)
    assert not plan.source_validity_is_authoritative
    assert plans == build(
        arguments, indices=indices, maximum_requests_per_tile=1
    )


@pytest.mark.parametrize("recording", (750,), indirect=True)
@pytest.mark.parametrize(
    "backend", ("numpy", pytest.param("cupy", marks=pytest.mark.gpu))
)
def test_actual_short_file_execution_preserves_fractional_exposure(
    recording, backend
):
    from cuphoton.xbeam.adapters.radio.engine import correlate
    from cuphoton.xbeam.windowing import CorrelationWindow

    if backend == "cupy":
        from tests.xbeam.test_gpu_backend import _cupy

        _cupy()
    job, scan, _ingest, indexes = recording
    second_station = replace(job.stations[0], identifier=1, name="other")
    second_path = Path(job.datastreams[0].source_files[0]).with_name(
        "other.vdif"
    )
    second_path.write_bytes(
        Path(job.datastreams[0].source_files[0]).read_bytes()
    )
    build_vdif_binary_index(second_path, index_dir=indexes, backend="posix")
    second_stream = replace(
        job.datastreams[0],
        identifier=1,
        station_id=1,
        source_files=(str(second_path),),
    )
    scan = replace(
        scan,
        duration_s=Decimal(1),
        station_models=(
            *scan.station_models,
            replace(scan.station_models[0], station_id=1),
        ),
    )
    job = replace(
        job,
        execution_duration_s=Decimal(1),
        integration_duration_s=Decimal("0.0032"),
        stations=(*job.stations, second_station),
        datastreams=(*job.datastreams, second_stream),
        baselines=(Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),)),),
        scans=(scan,),
        write_autocorrelations=True,
    )
    records = list(
        correlate(
            job,
            backend=backend,
            profile="interop-real-usb-buffered-window-replay-v1",
            window=CorrelationWindow("ordinary", 1, 1),
            index_dir=str(indexes),
            max_read_bytes=1 << 28,
        )
    )
    assert len(records) == 3
    for record in records:
        # Forty-two complete FFTs, one with 7/8 valid samples (the invalid
        # frame), and one with 3/4 (EOF). Six remaining FFTs have no data.
        expected = (
            (42 + (7 / 8) ** 2 + (3 / 4) ** 2) / 50
            if record.antenna_a != record.antenna_b
            else (42 + 7 / 8 + 3 / 4) / 50
        )
        assert record.weight == pytest.approx(expected)
        assert np.isfinite(record.spectrum).all()


@pytest.mark.parametrize(
    "failure", ["two-files", "thread", "frequency", "scope"]
)
def test_strict_input_context_failures_do_not_fallback(recording, failure):
    job, scan, ingest, indexes = recording
    ds = job.datastreams[0]
    if failure == "two-files":
        ds = replace(ds, source_files=ds.source_files * 2)
        message = "exactly one source file"
    elif failure == "thread":
        ds = replace(
            ds, recorded_bands=(replace(ds.recorded_bands[0], thread_id=6),)
        )
        message = "physical ingest thread"
    elif failure == "frequency":
        other = replace(job.frequencies[0], identifier=1, channel_count=65536)
        ds = replace(
            ds,
            recorded_bands=ds.recorded_bands
            + (replace(ds.recorded_bands[0], identifier=1, frequency_id=1),),
        )
        job = replace(job, frequencies=job.frequencies + (other,))
        message = "incompatible raw geometry"
    else:
        message = "beyond scan execution"
    job = replace(job, datastreams=(ds,))
    with pytest.raises(ValueError, match=message):
        build(
            (job, scan, ingest, indexes),
            indices=(2,) if failure == "scope" else (1,),
        )


def test_protocol_arguments_are_required(recording):
    job, scan, ingest, indexes = recording
    with pytest.raises(TypeError, match="alignment_bytes"):
        build_raw_file_replay_plans(
            job, [(0, scan, (1,))], ingest, index_dir=str(indexes)
        )


@pytest.mark.parametrize("recording", (2000,), indirect=True)
def test_file_adapter_bounds_each_real_request_tile(recording, monkeypatch):
    import cuphoton.xbeam.buffered_reader.adapter as replay

    whole = build(recording, indices=(1, 2))
    actual_evaluate = replay.evaluate_raw_job_requests
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(
            (
                kwargs["first_fft"],
                kwargs["stop_fft"],
                kwargs["maximum_requests"],
            )
        )
        return actual_evaluate(*args, **kwargs)

    monkeypatch.setattr(replay, "evaluate_raw_job_requests", evaluate)
    tiled = build(recording, indices=(1, 2), maximum_requests_per_tile=1)
    assert calls == [(50, 100, 1), (100, 150, 1)]
    assert tiled == whole


def test_shared_scans_reuse_one_actual_validated_context(
    recording, monkeypatch
):
    import cuphoton.xbeam.buffered_reader.adapter as replay

    job, scan, ingest, indexes = recording
    other = replace(scan, identifier="second")
    job = replace(job, scans=(scan, other))
    calls = {"validate": 0, "context": 0}
    actual_validate, actual_context = (
        replay.validate_vdif_binary_index,
        replay.raw_sidecar_context,
    )

    def validate(*args, **kwargs):
        calls["validate"] += 1
        return actual_validate(*args, **kwargs)

    def context(*args, **kwargs):
        calls["context"] += 1
        return actual_context(*args, **kwargs)

    # Count real calls; these wrappers keep actual validation and compilation.
    monkeypatch.setattr(replay, "validate_vdif_binary_index", validate)
    monkeypatch.setattr(replay, "raw_sidecar_context", context)
    plans = replay.build_returned_window_replay_plans(
        job,
        [(0, scan, range(1, 2)), (0, other, range(1, 2))],
        ingest,
        index_dir=str(indexes),
        alignment_bytes=8192,
        word_bytes=4,
        control_word_bits=30,
        timestamp_quantum_nanoseconds=1,
        replay_identifier="independent-cache-test",
    )
    assert calls == {"validate": 1, "context": 1}
    assert (
        plans[("ordinary", 0, 0)].source_physical_frame_intervals
        == plans[("second", 0, 0)].source_physical_frame_intervals
    )


@pytest.mark.parametrize("failure", ["none", "scope", "callable"])
def test_file_contract_factory_rejects_invalid_binding(recording, failure):
    options = {}

    def factory(scope, inputs, state):
        if failure == "none":
            return None
        size = inputs.stop_fft - inputs.first_fft
        return RetainedRawWindowContract(
            inputs.first_fft + 1,
            np.zeros(size, dtype=np.int64),
            np.zeros(size, dtype=np.int64),
            np.zeros(size, dtype=np.float64),
            np.ones(size, dtype=np.int64),
            np.ones(size, dtype=np.int64),
            np.zeros(size, dtype=np.int64),
            np.ones(size, dtype=bool),
        )

    options["window_clock_contract_factory"] = factory
    error, message = TypeError, "RetainedRawWindowContract"
    if failure == "scope":
        error, message = ValueError, "match tile scope"
    elif failure == "callable":
        options["window_clock_contract_factory"] = object()
        error, message = TypeError, "callable contract factory"
    with pytest.raises(error, match=message):
        build(recording, **options)


@pytest.mark.parametrize("recording", (2000,), indirect=True)
@pytest.mark.parametrize(
    "reader", ((24, 8, 20), (16, 8, 20), (32, 8, 20), (12, 4, 10))
)
def test_returned_policy_reads_actual_file_without_output_calibration(
    recording,
    reader,
):
    from cuphoton.xbeam.buffered_reader.adapter import (
        build_returned_window_replay_plans,
    )

    job, scan, ingest, indexes = recording
    job = replace(job, buffered_reader=BufferedReaderGeometry(*reader))
    arguments = dict(
        index_dir=str(indexes),
        alignment_bytes=8192,
        word_bytes=4,
        control_word_bits=30,
        timestamp_quantum_nanoseconds=1,
        replay_identifier="independent-returned-window-fixture",
    )
    whole = build_returned_window_replay_plans(
        job, [(0, scan, (1,))], ingest, **arguments
    )
    tiled = build_returned_window_replay_plans(
        job,
        [(0, scan, (1,))],
        ingest,
        maximum_requests_per_tile=1,
        **arguments,
    )
    assert whole == tiled
    plan = whole[("ordinary", 0, 0)]
    assert plan.calibration_identifier == arguments["replay_identifier"]
    assert not plan.ineligible_intervals
    assert not plan.source_validity_is_authoritative
    assert not plan.source_weight_override_events
    assert plan.source_physical_frame_intervals[0][1:3] == (400, 800)
    assert plan.source_timing_events == ()
    with pytest.raises(TypeError):
        whole["unexpected"] = plan


@pytest.mark.parametrize("recording", [(1000, 32)], indirect=True)
def test_multichannel_file_replay_retains_per_channel_sample_coordinates(
    recording,
):
    plans = build(recording)
    plan = plans[("ordinary", 0, 0)]
    assert recording[0].frequencies[0].channel_count * 2 == FFT // 32
    assert plan.source_physical_frame_intervals[0][1:3] == (400, 800)
    for fft in (50, 53, 99):
        assert plan.source_replay(fft) == (0, 0)
    assert plan.fft_count == 100
    assert plan.timeline_fft_count == 100
    assert plan.calibration_identifier == "independent-file-adapter-test"


def test_file_replay_uses_validated_single_thread_alias(recording):
    job, scan, _ingest, indexes = recording
    ds = job.datastreams[0]
    band = replace(ds.recorded_bands[0], thread_id=0)
    ds = replace(ds, recorded_bands=(band,))
    job = replace(job, datastreams=(ds,))
    ingest = plan_vdif_ingest(
        job,
        intervals={(0, 0): SampleInterval(50 * FFT, 100 * FFT)},
        max_read_bytes=1 << 23,
        assume_ordered=False,
        index_dir=indexes,
    )
    assert ingest.thread_windows[0][0].template.thread_id == 5
    plans = build((job, scan, ingest, indexes))
    assert plans[("ordinary", 0, 0)].source_physical_frame_intervals[0][
        1:3
    ] == (400, 800)
