# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent sample-count and DFT oracles for explicit invalid intervals."""

import gc
import weakref
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.engine import correlate
from cuphoton.xbeam.adapters.radio.models import (
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
    ZoomBand,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.errors import UnsupportedProfileError
from cuphoton.xbeam.missing_timestamp_recovery import (
    ExactFFTEligibilityRule,
    GlobalFFTEligibilityPlan,
    MissingTimestampCalibration,
    build_global_fft_eligibility_plan,
    with_exact_source_replay,
)
from cuphoton.xbeam.profiles import PROFILES, get_profile

SAMPLE_RATE = 256
FFT_LENGTH = 64
SAMPLE_COUNT = 512
FRAME_SAMPLES = 32


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_station_input_released_without_cyclic_collection(
    tmp_path, monkeypatch, replay, fail
):
    from cuphoton.xbeam.adapters.radio import gpu_execution
    from cuphoton.xbeam.runtime_types import _BandSamples

    job = _job(tmp_path)
    plan = (
        GlobalFFTEligibilityPlan(
            (),
            (),
            (),
            (),
            "buffered-window-replay-v1",
            timeline_fft_count=8,
            ineligible_intervals=((1, 2),),
        )
        if replay
        else None
    )
    if fail:

        def fail_station(*args, **kwargs):
            raise RuntimeError("station failure")

        monkeypatch.setattr(gpu_execution, "_station_blocks", fail_station)

    def execute():
        samples = np.ones(SAMPLE_COUNT, np.float32)
        reference = weakref.ref(samples)
        stream = job.datastreams[0]
        data = _BandSamples(
            stream,
            stream.recorded_bands[0],
            job.frequencies[0],
            samples,
            np.ones(SAMPLE_COUNT, bool),
            0,
            float(SAMPLE_RATE),
        )
        try:
            blocks = gpu_execution._station_blocks_for_scan(
                job, data, job.scans[0], np.arange(4) * 0.25, np, plan
            )
        except RuntimeError as exc:
            assert fail and str(exc) == "station failure"
        else:
            assert not fail
            np.testing.assert_array_equal(
                blocks.weight, [1, 0 if replay else 1, 1, 1]
            )
        return reference

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert execute()() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("narrow", [False, True])
def test_authoritative_validity_survives_a_neighboring_segment(
    tmp_path, backend, narrow
):
    if backend == "cupy":
        from tests.xbeam.test_gpu_backend import _cupy

        _cupy()
    job = _job(tmp_path)
    identity = GlobalFFTEligibilityPlan(
        (), (), (), (), "buffered-window-replay-v1", timeline_fft_count=8
    )
    plan = replace(
        identity,
        source_validity_events=((0, 0, FFT_LENGTH),),
        source_authoritative_validity_events=(0,),
        source_segment_events=((1, ((0, 32, 0), (32, 64, 1))),),
    )

    class Source(_Source):
        def batches(self, datastream):
            for batch in super().batches(datastream):
                samples = batch.samples.copy()
                if datastream.identifier == 1:
                    samples[:FFT_LENGTH] = 2
                yield replace(batch, samples=samples)

    records = list(
        correlate(
            job,
            backend=backend,
            profile="interop-real-usb-buffered-window-replay-v1",
            recording_source=Source(job, (0, FFT_LENGTH), packed=False),
            recovery_plans={("scan", 0, 0): identity, ("scan", 1, 0): plan},
            batch_blocks=1 if narrow else None,
            subintegration_tiling="auto" if narrow else "off",
            max_tile_subintegrations=(
                1 if narrow and backend == "cupy" else None
            ),
        )
    )
    assert len(records) == 6
    for record in records:
        expected_dc = 64
        if record.seconds_of_day < 1:
            if (record.antenna_a, record.antenna_b) == ("AA", "BB"):
                expected_dc = 80
            elif record.antenna_a == record.antenna_b == "BB":
                expected_dc = 112
        assert record.weight == 1
        np.testing.assert_allclose(record.spectrum[0], expected_dc)
        np.testing.assert_allclose(record.spectrum[1:], 0, atol=1e-6)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("accounting", ["separate-credit", "override"])
def test_reported_exposure_preserves_normalization_for_all_products(
    tmp_path, backend, accounting
):
    if backend == "cupy":
        from tests.xbeam.test_gpu_backend import _cupy

        _cupy()
    base = _job(tmp_path)
    bands = (RecordedBand(0, 0, "R"), RecordedBand(1, 0, "L"))
    streams = tuple(
        replace(
            stream,
            recorded_bands=bands,
            zoom_bands=(
                ZoomBand(2, 0, 1, 2, 8, "R"),
                ZoomBand(3, 1, 1, 2, 8, "L"),
            ),
        )
        for stream in base.datastreams
    )
    job = replace(
        base,
        datastreams=streams,
        frequencies=base.frequencies
        + (Frequency(1, 8.0, 32.0, Sideband.USB, 8),),
        baselines=(
            Baseline(
                0,
                0,
                1,
                tuple(
                    BaselineProduct(
                        a + 2 * frequency, b + 2 * frequency, frequency
                    )
                    for frequency in (0, 1)
                    for a in (0, 1)
                    for b in (0, 1)
                ),
            ),
        ),
    )
    rng = np.random.default_rng(734)
    values = {
        (station, polarization): rng.normal(size=SAMPLE_COUNT).astype(
            np.float32
        )
        for station in ("AA", "BB")
        for polarization in ("R", "L")
    }

    class Source:
        def batches(self, stream):
            station = job.stations[stream.station_id].name
            for band in stream.recorded_bands:
                validity = np.ones(SAMPLE_COUNT, bool)
                if accounting == "separate-credit" and station == "BB":
                    validity[:FRAME_SAMPLES] = False
                yield VoltageBatch(
                    values[(station, band.polarization)],
                    job.start_time,
                    float(SAMPLE_RATE),
                    band.identifier,
                    band.polarization,
                    validity,
                )

    identity = GlobalFFTEligibilityPlan(
        (), (), (), (), "buffered-window-replay-v1", timeline_fft_count=8
    )
    adjusted = (
        replace(
            identity,
            source_weight_credit_intervals=((0, FRAME_SAMPLES),),
            source_reported_weight_credit_intervals=(),
        )
        if accounting == "separate-credit"
        else replace(
            identity, source_weight_override_events=((0, FRAME_SAMPLES),)
        )
    )
    records = list(
        correlate(
            job,
            backend=backend,
            profile="interop-real-usb-buffered-window-replay-v1",
            recording_source=Source(),
            recovery_plans={
                ("scan", stream.identifier, band.identifier): (
                    identity if stream.identifier == 0 else adjusted
                )
                for stream in streams
                for band in bands
            },
        )
    )
    spectra = {}
    reported = {"AA": np.ones(8), "BB": np.ones(8)}
    reported["BB"][0] = 0.5
    for key, samples in values.items():
        expected_samples = samples.copy()
        if accounting == "separate-credit" and key[0] == "BB":
            expected_samples[:FRAME_SAMPLES] = 0
        spectra[key] = np.fft.fft(expected_samples.reshape(8, FFT_LENGTH))
    assert len(records) == 48
    for record in records:
        start = int(record.seconds_of_day) * 4
        selected = slice(start, start + 4)
        a = (record.antenna_a, record.polarization_a)
        b = (record.antenna_b, record.polarization_b)
        product = spectra[a][selected] * spectra[b][selected].conj()
        expected = product.sum(axis=0) / (4 * FFT_LENGTH)
        channels = (
            slice(0, 32) if record.output_frequency_id == 0 else slice(2, 10)
        )
        np.testing.assert_allclose(
            record.spectrum, expected[channels], rtol=2e-5, atol=2e-5
        )
        weights = reported[a[0]][selected]
        if a != b:
            weights = weights * reported[b[0]][selected]
        assert record.weight == pytest.approx(weights.sum() / 4)


def test_unknown_compatibility_profile_fails_explicitly():
    name = "unknown-profile"
    assert name not in PROFILES
    with pytest.raises(
        UnsupportedProfileError, match="not frozen or supported"
    ):
        get_profile(name)


def test_missing_timestamp_recovery_profile_is_explicit_and_versioned():
    name = "interop-real-usb-buffered-window-replay-v1"
    profile = get_profile(name)
    assert PROFILES[name] is profile
    assert profile.missing_timestamp_recovery == "buffered-window-replay-v1"
    assert (
        get_profile("interop-real-usb-v1").missing_timestamp_recovery is None
    )


def test_missing_timestamp_recovery_profile_requires_exact_plan(tmp_path):
    job = _job(tmp_path)
    with pytest.raises(
        ValueError, match="exact FFT fingerprint artifact is required"
    ):
        correlate(
            job,
            backend="numpy",
            profile="interop-real-usb-buffered-window-replay-v1",
            recording_source=_Source(job, (0, 0), packed=False),
        )


def test_mask_only_recovery_cannot_run_file_backed_data(tmp_path):
    job = _job(tmp_path)
    with pytest.raises(ValueError, match="complete exact source replay plan"):
        correlate(
            job,
            backend="numpy",
            profile="interop-real-usb-buffered-window-replay-v1",
            recovery_plans={},
        )


def test_exact_recovery_plan_excludes_same_fft_from_weight_and_spectrum(
    tmp_path,
):
    job = _job(tmp_path)
    calibration = MissingTimestampCalibration(
        "buffered-window-replay-v1",
        8,
        (4, 8),
        exact_artifacts=(ExactFFTEligibilityRule((1,), (1,)),),
    )
    plan = build_global_fft_eligibility_plan(
        fft_count=8,
        checkpoint_stops=(4, 8),
        missing_timestamps=(1,),
        calibration=calibration,
    )
    plans = {
        ("scan", datastream.identifier, 0): plan
        for datastream in job.datastreams
    }
    records = list(
        correlate(
            job,
            backend="numpy",
            profile="interop-real-usb-buffered-window-replay-v1",
            recording_source=_Source(job, (0, 0), packed=False),
            recovery_plans=plans,
        )
    )
    assert len(records) == 6
    for record in records:
        integration = int(record.seconds_of_day)
        assert record.weight == (0.75 if integration == 0 else 1.0)
        assert np.isfinite(record.spectrum).all()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
def test_exact_source_replay_selects_partial_fft_payload_and_drops_tail(
    tmp_path, packed, backend
):
    if backend == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("no CUDA device")
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA device unavailable: {exc}")
    job = _job(tmp_path)
    rng = np.random.default_rng(17)
    codes = {
        station: rng.integers(0, 4, SAMPLE_COUNT, dtype=np.uint8)
        for station in ("AA", "BB")
    }
    levels = np.asarray([-3.3359, -1.0, 1.0, 3.3359], dtype=np.float32)
    values = {station: levels[row] for station, row in codes.items()}

    class Source:
        def batches(self, datastream):
            name = job.stations[datastream.station_id].name
            if packed:
                row = codes[name].reshape(-1, 4)
                samples = np.ascontiguousarray(
                    row[:, 0]
                    | (row[:, 1] << 2)
                    | (row[:, 2] << 4)
                    | (row[:, 3] << 6)
                )
                validity = np.ones(SAMPLE_COUNT // FRAME_SAMPLES, bool)
                layout = PackedVoltageLayout(8, FRAME_SAMPLES, 1, 0, False)
            else:
                samples = values[name]
                validity = np.ones(SAMPLE_COUNT, bool)
                layout = None
            yield VoltageBatch(
                samples,
                job.start_time,
                float(SAMPLE_RATE),
                0,
                "R",
                validity,
                layout,
            )

    calibration = MissingTimestampCalibration(
        "buffered-window-replay-v1",
        8,
        (4, 8),
        exact_artifacts=(ExactFFTEligibilityRule((5,), (4,)),),
    )
    mask = build_global_fft_eligibility_plan(
        fft_count=8,
        checkpoint_stops=(4, 8),
        missing_timestamps=(5,),
        calibration=calibration,
    )
    identity = build_global_fft_eligibility_plan(
        fft_count=8,
        checkpoint_stops=(4, 8),
        missing_timestamps=(),
        calibration=calibration,
    )
    offsets = (0, 0, FRAME_SAMPLES, FRAME_SAMPLES, 0, 0, 0, 0)
    fractional_cutover = FRAME_SAMPLES // 2 + 3
    cutovers = (0, 0, fractional_cutover, 0, 0, 0, 0, 0)
    plans = {
        ("scan", 0, 0): with_exact_source_replay(
            identity,
            source_offsets=(0,) * 8,
            source_cutovers=(0,) * 8,
        ),
        ("scan", 1, 0): with_exact_source_replay(
            mask,
            source_offsets=offsets,
            source_cutovers=cutovers,
        ),
    }
    records = list(
        correlate(
            job,
            backend=backend,
            profile="interop-real-usb-buffered-window-replay-v1",
            recording_source=Source(),
            recovery_plans=plans,
        )
    )
    expected_b = values["BB"].copy()
    expected_b[2 * FFT_LENGTH + fractional_cutover : 3 * FFT_LENGTH] = values[
        "BB"
    ][
        2 * FFT_LENGTH + fractional_cutover + FRAME_SAMPLES : 3 * FFT_LENGTH
        + FRAME_SAMPLES
    ]
    expected_b[3 * FFT_LENGTH : 4 * FFT_LENGTH] = values["BB"][
        3 * FFT_LENGTH + FRAME_SAMPLES : 4 * FFT_LENGTH + FRAME_SAMPLES
    ]
    for record in records:
        if (record.antenna_a, record.antenna_b) != ("AA", "BB"):
            continue
        integration = int(record.seconds_of_day)
        starts = range(integration * 4, (integration + 1) * 4)
        usable = [index for index in starts if index != 4]
        accumulated = sum(
            np.fft.fft(
                values["AA"][index * FFT_LENGTH : (index + 1) * FFT_LENGTH]
            )
            * np.fft.fft(
                expected_b[index * FFT_LENGTH : (index + 1) * FFT_LENGTH]
            ).conj()
            for index in usable
        )
        expected = accumulated[: FFT_LENGTH // 2] / (len(usable) * FFT_LENGTH)
        assert record.weight == pytest.approx(len(usable) / 4)
        np.testing.assert_allclose(
            record.spectrum, expected, rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
def test_file_backed_source_replay_uses_frames_after_missing_timestamp(
    tmp_path, backend
):
    if backend == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("no CUDA device")
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA device unavailable: {exc}")
    job = _job(tmp_path)
    rng = np.random.default_rng(23)
    codes = {
        name: rng.integers(0, 4, SAMPLE_COUNT, dtype=np.uint8)
        for name in ("AA", "BB")
    }
    levels = np.asarray([-3.3359, -1.0, 1.0, 3.3359], dtype=np.float32)
    values = {name: levels[row] for name, row in codes.items()}
    for station, stream in zip(job.stations, job.datastreams, strict=True):
        with open(stream.source_files[0], "wb") as output:
            for frame in range(SAMPLE_COUNT // FRAME_SAMPLES):
                if station.name == "BB" and frame == 5:
                    continue
                second, number = divmod(frame, SAMPLE_RATE // FRAME_SAMPLES)
                station_code = (ord(station.name[0]) << 8) | ord(
                    station.name[1]
                )
                words = np.asarray(
                    [
                        second,
                        number | (40 << 24),
                        5,
                        station_code | (1 << 26),
                        0,
                        0,
                        0,
                        0,
                    ],
                    dtype="<u4",
                )
                row = codes[station.name][
                    frame * FRAME_SAMPLES : (frame + 1) * FRAME_SAMPLES
                ].reshape(-1, 4)
                payload = np.ascontiguousarray(
                    row[:, 0]
                    | (row[:, 1] << 2)
                    | (row[:, 2] << 4)
                    | (row[:, 3] << 6)
                )
                output.write(words.tobytes())
                output.write(payload.tobytes())
        build_vdif_binary_index(stream.source_files[0], backend="posix")

    calibration = MissingTimestampCalibration(
        "buffered-window-replay-v1",
        8,
        (4, 8),
        exact_artifacts=(ExactFFTEligibilityRule((5,), (4,)),),
    )
    mask = build_global_fft_eligibility_plan(
        fft_count=8,
        checkpoint_stops=(4, 8),
        missing_timestamps=(5,),
        calibration=calibration,
    )
    identity = build_global_fft_eligibility_plan(
        fft_count=8,
        checkpoint_stops=(4, 8),
        missing_timestamps=(),
        calibration=calibration,
    )
    plans = {
        ("scan", 0, 0): with_exact_source_replay(
            identity,
            source_offsets=(0,) * 8,
            source_cutovers=(0,) * 8,
        ),
        ("scan", 1, 0): with_exact_source_replay(
            mask,
            source_offsets=(0, 0, FRAME_SAMPLES, FRAME_SAMPLES, 0, 0, 0, 0),
            source_cutovers=(0, 0, FRAME_SAMPLES, 0, 0, 0, 0, 0),
        ),
    }
    records = list(
        correlate(
            job,
            backend=backend,
            max_read_bytes=1 << 20,
            profile="interop-real-usb-buffered-window-replay-v1",
            recovery_plans=plans,
        )
    )
    cross = [
        row
        for row in records
        if (row.antenna_a, row.antenna_b) == ("AA", "BB")
    ]
    assert len(cross) == 2
    expected_b = values["BB"].copy()
    expected_b[5 * FRAME_SAMPLES : 4 * FFT_LENGTH] = values["BB"][
        6 * FRAME_SAMPLES : 4 * FFT_LENGTH + FRAME_SAMPLES
    ]
    for integration, record in enumerate(cross):
        usable = [
            index
            for index in range(integration * 4, (integration + 1) * 4)
            if index != 4
        ]
        spectrum = sum(
            np.fft.fft(
                values["AA"][index * FFT_LENGTH : (index + 1) * FFT_LENGTH]
            )
            * np.fft.fft(
                expected_b[index * FFT_LENGTH : (index + 1) * FFT_LENGTH]
            ).conj()
            for index in usable
        )
        assert record.weight == pytest.approx(len(usable) / 4)
        np.testing.assert_allclose(
            record.spectrum,
            spectrum[: FFT_LENGTH // 2] / (len(usable) * FFT_LENGTH),
            rtol=2e-5,
            atol=2e-5,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("representation", ["decoded", "packed"])
@pytest.mark.parametrize("tiled", [False, True])
def test_gpu_exact_recovery_plan_matches_cpu_weight(
    tmp_path, representation, tiled
):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    job = _job(tmp_path)
    calibration = MissingTimestampCalibration(
        "buffered-window-replay-v1",
        8,
        (4, 8),
        exact_artifacts=(ExactFFTEligibilityRule((1,), (1,)),),
    )
    plan = build_global_fft_eligibility_plan(
        fft_count=8,
        checkpoint_stops=(4, 8),
        missing_timestamps=(1,),
        calibration=calibration,
    )
    records = list(
        correlate(
            job,
            backend="cupy",
            profile="interop-real-usb-buffered-window-replay-v1",
            recording_source=_Source(
                job, (0, 0), packed=representation == "packed"
            ),
            recovery_plans={
                ("scan", datastream.identifier, 0): plan
                for datastream in job.datastreams
            },
            subintegration_tiling="auto" if tiled else "off",
            max_tile_subintegrations=1 if tiled else None,
        )
    )
    assert len(records) == 6
    for record in records:
        integration = int(record.seconds_of_day)
        assert record.weight == pytest.approx(
            0.75 if integration == 0 else 1.0
        )
        assert np.isfinite(record.spectrum).all()


def _job(tmp_path):
    epoch = MJDTime(58849, Decimal(0))
    stations = tuple(
        Station(i, name, ClockPolynomial(epoch, (0.0,)))
        for i, name in enumerate(("AA", "BB"))
    )
    streams = tuple(
        Datastream(
            i,
            i,
            SamplingKind.REAL,
            2,
            40,
            (str(tmp_path / f"{station.name}.vdif"),),
            (RecordedBand(0, 0, "R"),),
            sample_rate_hz=float(SAMPLE_RATE),
        )
        for i, station in enumerate(stations)
    )
    models = tuple(
        DelayCoordinateModel(i, Decimal(0), (0.0,), (0.0,), (0.0,), (0.0,))
        for i in range(2)
    )
    return Job(
        epoch,
        Decimal(2),
        Decimal(1),
        Decimal("0.25"),
        1,
        stations,
        (Frequency(0, 0.0, SAMPLE_RATE / 2, Sideband.USB, FFT_LENGTH // 2),),
        streams,
        (Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),)),),
        (Scan("scan", Decimal(0), Decimal(2), "0", "0", models),),
        write_autocorrelations=True,
    )


class _Source:
    def __init__(self, job, interval, packed, *, interval_a=None):
        self.job = job
        self.intervals = {0: interval_a, 1: interval}
        self.packed = packed

    def batches(self, datastream):
        validity = np.ones(SAMPLE_COUNT, dtype=np.bool_)
        interval = self.intervals[datastream.station_id]
        if interval is not None:
            validity[slice(*interval)] = False
        if self.packed:
            # Offset-binary code 2 decodes to +1, earliest sample in low bits.
            samples = np.full(SAMPLE_COUNT // 4, 0xAA, dtype=np.uint8)
            frame_validity = validity.reshape(-1, FRAME_SAMPLES)
            assert np.all(frame_validity == frame_validity[:, :1])
            validity = frame_validity[:, 0].copy()
            layout = PackedVoltageLayout(8, FRAME_SAMPLES, 1, 0, False)
        else:
            samples = np.ones(SAMPLE_COUNT, dtype=np.float32)
            layout = None
        yield VoltageBatch(
            samples,
            self.job.start_time,
            float(SAMPLE_RATE),
            0,
            "R",
            validity,
            layout,
        )


def _write_vdif(job, interval, *, interval_a=None):
    """Encode independently specified invalid intervals in ordered frames."""
    intervals = {0: interval_a, 1: interval}
    for station, stream in zip(job.stations, job.datastreams, strict=True):
        current_interval = intervals[station.identifier]
        start, stop = current_interval or (0, 0)
        assert start % FRAME_SAMPLES == stop % FRAME_SAMPLES == 0
        with open(stream.source_files[0], "wb") as output:
            for frame in range(SAMPLE_COUNT // FRAME_SAMPLES):
                second, number = divmod(frame, SAMPLE_RATE // FRAME_SAMPLES)
                invalid = start <= frame * FRAME_SAMPLES < stop
                station_code = (ord(station.name[0]) << 8) | ord(
                    station.name[1]
                )
                words = np.asarray(
                    [
                        second | (int(invalid) << 31),
                        number | (40 << 24),
                        5,
                        station_code | (1 << 26),
                        0,
                        0,
                        0,
                        0,
                    ],
                    dtype="<u4",
                )
                output.write(words.tobytes())
                output.write(bytes([0xAA]) * 8)


def _assert_exact_invalid_time(records, interval):
    # The independent oracle zeros precisely the specified sample interval.
    # Only BB is affected, so no dual-partial station-weight rule is assumed.
    values = {name: np.ones(SAMPLE_COUNT) for name in ("AA", "BB")}
    values["BB"][slice(*interval)] = 0
    _assert_sample_masks(records, values)


def _assert_sample_masks(
    records, values, *, sample_rate=SAMPLE_RATE, fft_length=FFT_LENGTH
):
    """Independent DFT and per-station fractional exposure contract."""
    assert len(records) == 6
    seen = set()
    for record in records:
        integration = int(record.seconds_of_day)
        key = (integration, record.antenna_a, record.antenna_b)
        assert key not in seen
        seen.add(key)
        region = slice(
            integration * sample_rate, (integration + 1) * sample_rate
        )
        a = values[record.antenna_a][region]
        b = values[record.antenna_b][region]
        a = a.reshape(-1, fft_length)
        b = b.reshape(-1, fft_length)
        fractions = np.mean(a != 0, axis=-1)
        if record.antenna_a != record.antenna_b:
            fractions *= np.mean(b != 0, axis=-1)
        exposure = np.sum(fractions) * fft_length
        assert record.weight == pytest.approx(exposure / sample_rate)
        fa = np.fft.fft(a, axis=-1)
        fb = np.fft.fft(b, axis=-1)
        spectrum = np.sum(fa * fb.conj(), axis=0) / exposure
        np.testing.assert_allclose(
            record.spectrum, spectrum[: fft_length // 2], rtol=2e-5, atol=2e-5
        )
    assert seen == {
        (i, a, b)
        for i in range(2)
        for a, b in [("AA", "BB"), ("AA", "AA"), ("BB", "BB")]
    }


@pytest.mark.parametrize("profile", ["analytic-v1", "interop-real-usb-v1"])
@pytest.mark.parametrize("interval", [(1, 2), (63, 65), (255, 257), (0, 64)])
def test_decoded_invalid_interval_preserves_other_fft_samples(
    tmp_path, profile, interval
):
    job = _job(tmp_path)
    records = list(
        correlate(
            job,
            backend="numpy",
            profile=profile,
            recording_source=_Source(job, interval, packed=False),
        )
    )
    _assert_exact_invalid_time(records, interval)


@pytest.mark.parametrize("representation", ["packed", "file"])
@pytest.mark.parametrize("interval", [(32, 64), (32, 96), (224, 288)])
def test_invalid_frames_lose_exact_duration(
    tmp_path, representation, interval
):
    job = _job(tmp_path)
    if representation == "file":
        _write_vdif(job, interval)
        source = None
    else:
        source = _Source(job, interval, packed=True)
    records = list(
        correlate(
            job,
            backend="numpy",
            profile="interop-real-usb-v1",
            recording_source=source,
            assume_ordered=True,
        )
    )
    _assert_exact_invalid_time(records, interval)


@pytest.mark.gpu
@pytest.mark.parametrize("representation", ["decoded", "packed", "file"])
@pytest.mark.parametrize("tiled", [False, True])
def test_gpu_invalid_interval_matches_independent_oracle(
    tmp_path, representation, tiled
):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    job = _job(tmp_path)
    interval = (255, 257) if representation == "decoded" else (224, 288)
    if representation == "file":
        _write_vdif(job, interval)
        source = None
    else:
        source = _Source(job, interval, packed=representation == "packed")
    records = list(
        correlate(
            job,
            backend="cupy",
            profile="interop-real-usb-v1",
            recording_source=source,
            assume_ordered=True,
            subintegration_tiling="auto" if tiled else "off",
            max_tile_subintegrations=1 if tiled else None,
        )
    )
    _assert_exact_invalid_time(records, interval)


@pytest.mark.parametrize("profile", ["analytic-v1", "interop-real-usb-v1"])
@pytest.mark.parametrize("interval_b", [(0, 32), (32, 64)])
@pytest.mark.parametrize("representation", ["decoded", "packed", "file"])
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("tiled", [False, True])
def test_dual_partial_stations_use_product_of_fft_fractions(
    tmp_path, profile, interval_b, representation, backend, tiled
):
    if backend == "cupy":
        from tests.xbeam.test_gpu_backend import _cupy

        _cupy()
    job = _job(tmp_path)
    interval_a = (0, 32)
    if representation == "file":
        _write_vdif(job, interval_b, interval_a=interval_a)
        source = None
    else:
        source = _Source(
            job,
            interval_b,
            packed=representation == "packed",
            interval_a=interval_a,
        )
    records = list(
        correlate(
            job,
            backend=backend,
            profile=profile,
            recording_source=source,
            assume_ordered=True,
            subintegration_tiling="auto" if tiled else "off",
            max_tile_subintegrations=1
            if tiled and backend == "cupy"
            else None,
        )
    )
    values = {name: np.ones(SAMPLE_COUNT) for name in ("AA", "BB")}
    values["AA"][slice(*interval_a)] = 0
    values["BB"][slice(*interval_b)] = 0
    _assert_sample_masks(records, values)
    cross = next(
        record
        for record in records
        if int(record.seconds_of_day) == 0
        and record.antenna_a != record.antenna_b
    )
    # Both cases have one block with 1/2 * 1/2 exposure and three full blocks.
    # Simultaneous-sample overlap would instead produce 0.875 or 0.75.
    assert cross.weight == pytest.approx(0.8125)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
def test_non_power_of_two_fft_fractional_exposure(tmp_path, backend):
    if backend == "cupy":
        from tests.xbeam.test_gpu_backend import _cupy

        _cupy()
    sample_rate, fft_length = 240, 60
    job = _job(tmp_path)
    job = replace(
        job,
        frequencies=(
            replace(job.frequencies[0], bandwidth_hz=120, channel_count=30),
        ),
        datastreams=tuple(
            replace(stream, sample_rate_hz=float(sample_rate))
            for stream in job.datastreams
        ),
    )
    values = {name: np.ones(2 * sample_rate) for name in ("AA", "BB")}
    values["AA"][3:11] = 0
    values["BB"][:7] = 0

    class Source:
        def batches(self, datastream):
            mask = values[job.stations[datastream.station_id].name] != 0
            yield VoltageBatch(
                np.ones(mask.size, dtype=np.float32),
                job.start_time,
                float(sample_rate),
                0,
                "R",
                mask,
            )

    records = list(correlate(job, backend=backend, recording_source=Source()))
    _assert_sample_masks(
        records, values, sample_rate=sample_rate, fft_length=fft_length
    )
