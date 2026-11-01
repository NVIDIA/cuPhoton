# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio import engine as engine_module
from cuphoton.xbeam.adapters.radio import gpu_execution as execution
from cuphoton.xbeam.adapters.radio.engine import correlate
from cuphoton.xbeam.adapters.radio.models import (
    ClockPolynomial,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    MJDTime,
    PackedVoltageLayout,
    RecordedBand,
    SamplingKind,
    Sideband,
    Station,
    VoltageBatch,
)
from cuphoton.xbeam.runtime_types import _BandSamples
from tests.xbeam.test_fft_validity_policy import (
    FRAME_SAMPLES,
    SAMPLE_COUNT,
    SAMPLE_RATE,
    _assert_sample_masks,
    _job,
    _write_vdif,
)


def fixture():
    epoch = MJDTime(58230, Decimal(47880))
    frequency = Frequency(0, 0.0, 16.0, Sideband.USB, 4)
    band = RecordedBand(0, 0, "R")
    stream = Datastream(0, 0, SamplingKind.REAL, 2, 40, (), (band,))
    station = Station(0, "S", ClockPolynomial(epoch, (0.0,)))
    model = DelayCoordinateModel(
        0, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,)
    )
    job = Job(
        epoch,
        Decimal(1),
        Decimal(1),
        Decimal(1),
        1,
        (station,),
        (frequency,),
        (stream,),
        (),
        (),
    )
    data = _BandSamples(
        stream,
        band,
        frequency,
        np.ones(8, np.float32),
        np.ones(8, bool),
        0,
        32.0,
    )
    return job, model, data


@pytest.mark.parametrize("start", [-1 / 32, 1 / 32])
@pytest.mark.parametrize("storage", ["host", "device", "packed"])
@pytest.mark.parametrize("source_origin", [0, 37])
def test_coverage_fails_before_upload_wait_or_device_work(
    start, storage, source_origin
):
    job, model, data = fixture()
    data.start_sample = source_origin
    if storage == "device":
        data.device_samples = SimpleNamespace(size=8)
        data.samples = np.empty(0, np.float32)
    elif storage == "packed":
        data.device_payload = object()
        data.source_sample_count = 8
        data.samples = np.empty(0, np.float32)
    data.payload_progress = Mock()
    data.payload_frame_validity = np.ones(1, bool)
    with pytest.raises(RuntimeError, match="planned sample buffer"):
        execution._station_blocks(
            job,
            data,
            model,
            np.array([start + source_origin / data.sample_rate_hz]),
            SimpleNamespace(),
        )
    data.payload_progress.wait_samples.assert_not_called()


def test_missing_samples_within_buffer_still_reduce_weight():
    job, model, data = fixture()
    data.validity[:3] = False
    result = execution._station_blocks(job, data, model, np.array([0.0]), np)
    np.testing.assert_array_equal(result.weight, [5 / 8])


def test_reader_tail_credit_does_not_restore_a_fully_empty_fft_weight():
    job, model, data = fixture()
    data.validity[:] = False

    result = execution._station_blocks(
        job,
        data,
        model,
        np.array([0.0]),
        np,
        weight_credit_intervals=((0, 8),),
    )

    np.testing.assert_array_equal(result.weight, [0.0])


def test_reported_weight_uses_only_trustworthy_reader_tail_credit():
    job, model, data = fixture()
    data.validity[4:] = False

    result = execution._station_blocks(
        job,
        data,
        model,
        np.array([0.0]),
        np,
        weight_credit_intervals=((4, 8),),
        reported_weight_credit_intervals=(),
    )

    np.testing.assert_array_equal(result.weight, [1.0])
    np.testing.assert_array_equal(result.reported_weight, [0.5])


def test_replay_coverage_omits_unused_unshifted_fft():
    job, model, data = fixture()
    data.start_sample = -8

    result = execution._station_blocks(
        job,
        data,
        model,
        np.array([0.0]),
        np,
        source_offsets=np.array([-8], dtype=np.int64),
        source_cutovers=np.array([0], dtype=np.int64),
    )

    np.testing.assert_array_equal(result.weight, [1.0])


@pytest.mark.parametrize("preceding_band", [False, True])
def test_packed_group_checks_coverage_before_parameter_upload(
    monkeypatch, preceding_band
):
    job, model, data = fixture()
    data.device_payload = object()
    data.source_sample_count = 7
    monkeypatch.setattr(
        execution, "_packed_real_group_compatible", lambda a, b: True
    )
    upload = Mock(side_effect=AssertionError("unexpected device work"))
    monkeypatch.setattr(execution, "_packed_real_block_parameters", upload)
    group = [((0, 0), data)]
    if preceding_band:
        group.insert(0, ((0, 1), replace(data, source_sample_count=8)))
    with pytest.raises(RuntimeError, match="planned sample buffer"):
        execution._station_blocks_packed_real_group(
            job, group, model, np.array([0.0]), SimpleNamespace()
        )
    upload.assert_not_called()


def test_planned_file_source_still_requires_complete_allocation(
    tmp_path, monkeypatch
):
    job = _job(tmp_path)
    _write_vdif(job, (0, 0))
    original = engine_module._load_band_samples

    def undersized(*args, **kwargs):
        data = original(*args, **kwargs)
        return replace(
            data,
            samples=data.samples[:SAMPLE_RATE],
            validity=data.validity[:SAMPLE_RATE],
        )

    monkeypatch.setattr(engine_module, "_load_band_samples", undersized)
    with pytest.raises(RuntimeError, match="planned sample buffer"):
        list(correlate(job, backend="numpy", assume_ordered=True))


@pytest.mark.parametrize("representation", ["decoded", "packed"])
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize(
    ("first", "stop", "delay_samples"),
    [(32, 512, 0), (0, 480, 0), (0, 512, 8), (0, 512, -8)],
)
def test_recording_source_boundary_gaps_reduce_exposure(
    tmp_path, representation, backend, tiled, first, stop, delay_samples
):
    if backend == "cupy":
        from tests.xbeam.test_gpu_backend import _cupy

        _cupy()
    job = _job(tmp_path)
    scan = job.scans[0]
    models = (
        scan.station_models[0],
        replace(
            scan.station_models[1],
            delay_coefficients_s=(delay_samples / SAMPLE_RATE,),
        ),
    )
    job = replace(job, scans=(replace(scan, station_models=models),))

    class Source:
        def batches(self, datastream):
            lower, upper = (
                (first, stop)
                if datastream.station_id == 1
                else (0, SAMPLE_COUNT)
            )
            count = upper - lower
            if representation == "packed":
                samples = np.full(count // 4, 0xAA, dtype=np.uint8)
                validity = np.ones(count // FRAME_SAMPLES, dtype=np.bool_)
                layout = PackedVoltageLayout(8, FRAME_SAMPLES, 1, 0, False)
            else:
                samples = np.ones(count, dtype=np.float32)
                validity = np.ones(count, dtype=np.bool_)
                layout = None
            yield VoltageBatch(
                samples,
                MJDTime(
                    job.start_time.mjd,
                    job.start_time.seconds_of_day
                    + Decimal(lower) / SAMPLE_RATE,
                ),
                float(SAMPLE_RATE),
                0,
                "R",
                validity,
                layout,
            )

    records = list(
        correlate(
            job,
            backend=backend,
            recording_source=Source(),
            subintegration_tiling="auto" if tiled else "off",
            max_tile_subintegrations=1
            if tiled and backend == "cupy"
            else None,
        )
    )
    indices = np.arange(SAMPLE_COUNT) - delay_samples
    _assert_sample_masks(
        records,
        {
            "AA": np.ones(SAMPLE_COUNT),
            "BB": ((indices >= first) & (indices < stop)).astype(float),
        },
    )
