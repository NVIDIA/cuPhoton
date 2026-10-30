# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from cuphoton.xbeam import (
    Baseline,
    BaselineProduct,
    ClockPolynomial,
    ConfigurationError,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    JobLoadError,
    MJDTime,
    RecordedBand,
    SamplingKind,
    Scan,
    Sideband,
    Station,
    UnsupportedFeatureError,
    UnsupportedFeatures,
    ZoomBand,
    dump_job,
    job_from_dict,
    job_to_dict,
    load_job,
    load_job_schema,
    validate_job,
)


def valid_job() -> Job:
    start = MJDTime(60000, Decimal("86399.125000000"))
    clock = ClockPolynomial(MJDTime(60000, Decimal("0")), (0.0, 0.0))
    stations = (Station(0, "A", clock), Station(1, "B", clock))
    frequency = Frequency(0, 200_000_000.0, 4_000_000.0, Sideband.USB, 16)
    datastreams = (
        Datastream(
            0,
            0,
            SamplingKind.REAL,
            2,
            8032,
            ("a.vdif",),
            (RecordedBand(0, 0, "R"),),
        ),
        Datastream(
            1,
            1,
            SamplingKind.REAL,
            2,
            8032,
            ("b.vdif",),
            (RecordedBand(0, 0, "R"),),
        ),
    )
    baseline = Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),))
    models = tuple(
        DelayCoordinateModel(
            station_id, Decimal("0"), (0.0, 0.0), (0.0,), (0.0,), (0.0,)
        )
        for station_id in (0, 1)
    )
    scan = Scan(
        "scan-0",
        Decimal("-1.25"),
        Decimal("11.25"),
        "source",
        "default",
        models,
    )
    return Job(
        start,
        Decimal("10.000000000"),
        Decimal("1.048576"),
        Decimal("0.524288"),
        1,
        stations,
        (frequency,),
        datastreams,
        (baseline,),
        (scan,),
        True,
    )


def test_valid_job_and_lossless_json_round_trip(tmp_path) -> None:
    job = valid_job()
    validate_job(job)
    path = tmp_path / "job.json"
    dump_job(job, path)

    encoded = path.read_text(encoding="utf-8")
    assert '"seconds_of_day": "86399.125000000"' in encoded
    assert load_job(path) == job
    assert job_from_dict(job_to_dict(job)) == job


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("frequencies", 0, "channel_count"), 16.5),
        (("frequencies", 0, "channel_count"), True),
        (("schema_version",), 1.9),
        (("fringe_approximation_order",), 2.9),
        (("start_time", "mjd"), 60000.5),
    ],
)
def test_load_job_rejects_lossy_integer_metadata(tmp_path, path, value):
    document = job_to_dict(valid_job())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    filename = tmp_path / "job.json"
    filename.write_text(json.dumps(document))

    with pytest.raises(JobLoadError, match="expected an integer"):
        load_job(filename)


@pytest.mark.parametrize("value", [16, 16.0, "16"])
def test_load_job_preserves_exact_integer_normalization(tmp_path, value):
    expected = valid_job()
    document = job_to_dict(expected)
    document["frequencies"][0]["channel_count"] = value
    filename = tmp_path / "job.json"
    filename.write_text(json.dumps(document))

    assert load_job(filename) == expected


def test_bundled_schema_describes_normalized_v1_jobs() -> None:
    schema = load_job_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema_version"] == {"const": 1}
    assert "datastreams" in schema["required"]


@pytest.mark.parametrize(
    ("replacement", "field"),
    [
        ({"execution_duration_s": Decimal("0")}, "execution_duration_s"),
        ({"fringe_approximation_order": 3}, "fringe_approximation_order"),
        ({"schema_version": 2}, "schema_version"),
    ],
)
def test_rejects_invalid_job_scalars(replacement, field) -> None:
    with pytest.raises(ConfigurationError, match=field):
        validate_job(replace(valid_job(), **replacement))


def test_rejects_reference_to_unknown_station() -> None:
    job = valid_job()
    broken_stream = replace(job.datastreams[0], station_id=99)
    with pytest.raises(ConfigurationError) as raised:
        validate_job(
            replace(job, datastreams=(broken_stream, job.datastreams[1]))
        )

    assert raised.value.entity == "datastream[0]"
    assert raised.value.field == "station_id"
    assert "99" in raised.value.reason


def test_rejects_unsupported_features_before_any_source_access() -> None:
    job = replace(
        valid_job(),
        unsupported_features=UnsupportedFeatures(pulsar_binning=True),
    )
    with pytest.raises(UnsupportedFeatureError) as raised:
        validate_job(job)

    assert raised.value.field == "pulsar_binning"


@pytest.mark.parametrize(
    "frequency",
    [
        Frequency(0, 1.0, 1.0, Sideband.USB, 16, channels_to_average=3),
        Frequency(0, 1.0, 1.0, Sideband.USB, 16, oversampling=2),
        Frequency(0, 1.0, 1.0, Sideband.USB, 16, decimation=2),
    ],
)
def test_rejects_invalid_or_unsupported_frequency_modes(frequency) -> None:
    with pytest.raises(ConfigurationError):
        validate_job(replace(valid_job(), frequencies=(frequency,)))


def test_negative_scan_start_is_allowed_when_scan_overlaps_execution() -> (
    None
):
    validate_job(valid_job())


def test_scan_requires_models_for_every_used_station() -> None:
    job = valid_job()
    scan = replace(
        job.scans[0], station_models=job.scans[0].station_models[:1]
    )
    with pytest.raises(
        ConfigurationError, match="missing models for stations"
    ):
        validate_job(replace(job, scans=(scan,)))


def test_segmented_models_require_matching_station_epoch_grids() -> None:
    job = valid_job()
    first_a, first_b = job.scans[0].station_models
    scan = replace(
        job.scans[0],
        station_models=(
            first_a,
            first_b,
            replace(first_a, epoch_offset_s=Decimal("5")),
        ),
    )

    with pytest.raises(
        ConfigurationError, match="identical model epoch grids"
    ):
        validate_job(replace(job, scans=(scan,)))


def test_segmented_models_reject_duplicate_station_epoch() -> None:
    job = valid_job()
    duplicate = job.scans[0].station_models[0]
    scan = replace(
        job.scans[0],
        station_models=job.scans[0].station_models + (duplicate,),
    )

    with pytest.raises(ConfigurationError, match="duplicate model epoch"):
        validate_job(replace(job, scans=(scan,)))


def test_segmented_models_reject_epochs_collapsed_at_runtime_precision() -> (
    None
):
    job = valid_job()
    collapsed = tuple(
        replace(model, epoch_offset_s=epoch)
        for epoch in (
            Decimal("1.00000000000000001"),
            Decimal("1.00000000000000002"),
        )
        for model in job.scans[0].station_models
    )
    scan = replace(
        job.scans[0],
        station_models=job.scans[0].station_models + collapsed,
    )

    with pytest.raises(ConfigurationError, match="runtime precision"):
        validate_job(replace(job, scans=(scan,)))


def test_segmented_models_reject_nonfinite_epoch_cleanly() -> None:
    job = valid_job()
    model = replace(
        job.scans[0].station_models[0], epoch_offset_s=Decimal("sNaN")
    )
    scan = replace(
        job.scans[0],
        station_models=(model,) + job.scans[0].station_models[1:],
    )

    with pytest.raises(ConfigurationError, match="must be finite"):
        validate_job(replace(job, scans=(scan,)))


def test_static_validation_does_not_touch_recording_paths(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.vdif"
    job = valid_job()
    stream = replace(job.datastreams[0], source_files=(str(missing),))
    validate_job(replace(job, datastreams=(stream, job.datastreams[1])))


def test_parser_rejects_unknown_fields() -> None:
    document = job_to_dict(valid_job())
    document["mystery"] = 1
    with pytest.raises(JobLoadError, match="unknown field 'mystery'"):
        job_from_dict(document)


def test_parser_does_not_treat_nonempty_strings_as_true() -> None:
    document = job_to_dict(valid_job())
    document["write_autocorrelations"] = "false"
    with pytest.raises(JobLoadError, match="expected a boolean"):
        job_from_dict(document)


def test_timestamp_seconds_must_be_within_day() -> None:
    job = replace(valid_job(), start_time=MJDTime(60000, Decimal("86400")))
    with pytest.raises(ConfigurationError, match="seconds_of_day"):
        validate_job(job)


def test_equivalent_usb_and_lsb_frequency_grids_are_accepted() -> None:
    job = valid_job()
    lsb = Frequency(1, 204_000_000.0, 4_000_000.0, Sideband.LSB, 16)
    second = replace(
        job.datastreams[1], recorded_bands=(RecordedBand(0, 1, "R"),)
    )

    validate_job(
        replace(
            job,
            frequencies=job.frequencies + (lsb,),
            datastreams=(job.datastreams[0], second),
        )
    )


def test_complex_dsb_lsb_is_rejected_until_ordering_is_implemented() -> None:
    job = valid_job()
    frequency = replace(job.frequencies[0], sideband=Sideband.LSB)
    datastreams = tuple(
        replace(stream, sampling=SamplingKind.COMPLEX_DSB)
        for stream in job.datastreams
    )

    with pytest.raises(UnsupportedFeatureError, match="lower-sideband"):
        validate_job(
            replace(job, frequencies=(frequency,), datastreams=datastreams)
        )


def test_complex_dsb_lsb_zoom_output_is_rejected() -> None:
    job = valid_job()
    lsb = Frequency(1, 204_000_000.0, 4_000_000.0, Sideband.LSB, 16)
    first = replace(
        job.datastreams[0],
        sampling=SamplingKind.COMPLEX_DSB,
        zoom_bands=(ZoomBand(1, 0, 1, 0, 16, "R"),),
    )

    with pytest.raises(UnsupportedFeatureError, match="lower-sideband"):
        validate_job(
            replace(
                job,
                frequencies=job.frequencies + (lsb,),
                datastreams=(first, job.datastreams[1]),
            )
        )


def test_complex_dsb_lsb_baseline_output_is_rejected() -> None:
    job = valid_job()
    lsb = Frequency(1, 204_000_000.0, 4_000_000.0, Sideband.LSB, 16)
    datastreams = tuple(
        replace(stream, sampling=SamplingKind.COMPLEX_DSB)
        for stream in job.datastreams
    )
    baseline = replace(job.baselines[0], products=(BaselineProduct(0, 0, 1),))

    with pytest.raises(UnsupportedFeatureError, match="lower-sideband"):
        validate_job(
            replace(
                job,
                frequencies=job.frequencies + (lsb,),
                datastreams=datastreams,
                baselines=(baseline,),
            )
        )


def test_rejects_product_frequency_grid_mismatch_before_io() -> None:
    job = valid_job()
    shifted = Frequency(1, 201_000_000.0, 4_000_000.0, Sideband.USB, 16)
    second = replace(
        job.datastreams[1], recorded_bands=(RecordedBand(0, 1, "R"),)
    )
    broken = replace(
        job,
        frequencies=job.frequencies + (shifted,),
        datastreams=(job.datastreams[0], second),
    )

    with pytest.raises(UnsupportedFeatureError, match="spectral placement"):
        validate_job(broken)


def test_rejects_output_frequency_grid_mismatch_before_io() -> None:
    job = valid_job()
    shifted = Frequency(1, 201_000_000.0, 4_000_000.0, Sideband.USB, 16)
    baseline = replace(job.baselines[0], products=(BaselineProduct(0, 0, 1),))

    with pytest.raises(UnsupportedFeatureError, match="output frequency"):
        validate_job(
            replace(
                job,
                frequencies=job.frequencies + (shifted,),
                baselines=(baseline,),
            )
        )


def test_rejects_mixed_sample_rates_within_one_vdif_thread() -> None:
    job = valid_job()
    narrow = Frequency(1, 204_000_000.0, 2_000_000.0, Sideband.USB, 16)
    first = replace(
        job.datastreams[0],
        recorded_bands=(
            RecordedBand(0, 0, "R", thread_id=3),
            RecordedBand(1, 1, "L", thread_id=3),
        ),
    )

    with pytest.raises(ConfigurationError, match="one sample rate"):
        validate_job(
            replace(
                job,
                frequencies=job.frequencies + (narrow,),
                datastreams=(first, job.datastreams[1]),
            )
        )


def test_rejects_directly_constructed_non_file_data_source() -> None:
    job = valid_job()
    first = replace(job.datastreams[0], data_source="network")

    with pytest.raises(UnsupportedFeatureError, match="data_source"):
        validate_job(replace(job, datastreams=(first, job.datastreams[1])))


@pytest.mark.parametrize("frame_size", [32, 36, 40])
def test_vdif_frames_require_an_aligned_payload(frame_size):
    job = valid_job()
    job = replace(
        job,
        datastreams=tuple(
            replace(stream, frame_size_bytes=frame_size)
            for stream in job.datastreams
        ),
    )
    if frame_size == 40:
        validate_job(job)
    else:
        with pytest.raises(ConfigurationError, match="frame_size_bytes"):
            validate_job(job)


@pytest.mark.parametrize("value", ["0E-9", "1E-9", "1E+3"])
def test_decimal_serializer_uses_schema_compatible_fixed_notation(value):
    import re

    job = replace(valid_job(), execution_duration_s=Decimal(value))
    encoded = job_to_dict(job)["execution_duration_s"]
    assert re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", encoded)
    assert Decimal(encoded) == Decimal(value)
    assert job_from_dict(job_to_dict(job)) == job


@pytest.mark.parametrize(
    "path,value",
    [
        (("stations", 0, "name"), 7),
        (("datastreams", 0, "recorded_bands", 0, "polarization"), None),
        (("datastreams", 0, "source_files", 0), None),
    ],
)
def test_normalized_schema_rejects_nonstring_metadata(path, value):
    document = job_to_dict(valid_job())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(JobLoadError, match="expected a string"):
        job_from_dict(document)


def test_normalized_schema_requires_explicit_version():
    document = job_to_dict(valid_job())
    del document["schema_version"]
    with pytest.raises(JobLoadError, match="missing field 'schema_version'"):
        job_from_dict(document)


@pytest.mark.parametrize(
    "sampling,factor",
    [
        (SamplingKind.REAL, 2),
        (SamplingKind.COMPLEX, 1),
    ],
)
@pytest.mark.parametrize("scale", [0.5, 1, 2, 4])
def test_explicit_rate_must_match_supported_fft_grid(sampling, factor, scale):
    job = valid_job()
    rate = job.frequencies[0].bandwidth_hz * factor * scale
    job = replace(
        job,
        datastreams=tuple(
            replace(stream, sampling=sampling, sample_rate_hz=rate)
            for stream in job.datastreams
        ),
    )
    if scale == 1:
        validate_job(job)
    else:
        with pytest.raises(UnsupportedFeatureError, match="FFT grid"):
            validate_job(job)
