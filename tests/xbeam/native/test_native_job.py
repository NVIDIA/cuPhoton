# SPDX-License-Identifier: Apache-2.0
"""Differential job-adapter tests using independently generated metadata."""

import json
import subprocess
from pathlib import Path

import pytest

from cuphoton.xbeam.adapters.radio.models import job_from_dict
from cuphoton.xbeam.commands import load_interop_job
from tests.xbeam.native.job_fixtures import write_job


@pytest.mark.parametrize("zoom", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_native_job_matches_python(
    tmp_path, zoom, explicit, native_executable
):
    target, band = (2, 1) if zoom else (0, 0)
    path = write_job(
        tmp_path,
        band=band,
        zoom=zoom,
        targets=(target,) if explicit else (None,),
    )
    executable = native_executable
    actual = json.loads(
        subprocess.check_output(
            [
                str(executable),
                "inspect-job",
                str(path),
            ]
        )
    )
    assert job_from_dict(actual) == load_interop_job(path)


@pytest.mark.parametrize("transition", [False, True])
def test_native_source_blocks_match_python(tmp_path, transition):
    import importlib.util
    import os

    import numpy as np

    from cuphoton.xbeam.adapters.radio.compute_plan import (
        _source_block_indices_for_scan,
    )

    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    path = write_job(tmp_path)
    model = tmp_path / "job.im"
    text = model.read_text().replace(
        "DELAY (us): 0 0", "DELAY (us): -7.125 0.007"
    )
    if transition:
        text = text.replace("NUM POLY: 1", "NUM POLY: 2")
        text += "SCAN 0 POLY 1 MJD: 58849\nSCAN 0 POLY 1 SEC: 2\n"
        for station in range(2):
            for field in ("DELAY (us)", "U (m)", "V (m)", "W (m)"):
                text += f"SRC 0 ANT {station} {field}: 3.5 -0.002\n"
    model.write_text(text)
    job = load_interop_job(path)
    compiled = native.NativeJob(path)
    starts = np.ascontiguousarray(np.linspace(0, 3.99, 256))
    for stream in job.datastreams:
        for band in stream.recorded_bands:
            fft, delay, coarse, output = _source_block_indices_for_scan(
                job,
                stream,
                band,
                job.frequencies[band.frequency_id],
                job.scans[0],
                starts,
            )
            rows = compiled.source_blocks(
                job.scans[0].identifier,
                stream.identifier,
                band.identifier,
                starts,
            )
            assert all(row[0] == fft for row in rows)
            assert np.array_equal([row[1] for row in rows], output)
            assert np.array_equal([row[2] for row in rows], coarse)
            assert np.array_equal([row[3] for row in rows], delay)


@pytest.mark.parametrize(
    "first,stop,count", [(0, 99, 99), (13, 58, 101), (980, 1001, 1001)]
)
@pytest.mark.parametrize("endpoint_mean", [False, True])
def test_native_request_clocks_match_python(
    tmp_path, first, stop, count, endpoint_mean
):
    import importlib.util
    import os
    from fractions import Fraction

    import numpy as np

    from cuphoton.xbeam.buffered_reader.geometry import (
        BufferedRequestGeometry,
    )
    from cuphoton.xbeam.buffered_reader.requests import (
        ReadClockConvention,
        evaluate_raw_job_requests,
    )

    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    path = write_job(tmp_path)
    model = tmp_path / "job.im"
    model.write_text(
        model.read_text().replace(
            "DELAY (us): 0 0", "DELAY (us): -7.125 0.007"
        )
    )
    job = load_interop_job(path)
    compiled = native.NativeJob(path)
    geometry = BufferedRequestGeometry(Fraction(4), 64, 50, 8192, 4)
    for stream in job.datastreams:
        band = stream.recorded_bands[0]
        reference = evaluate_raw_job_requests(
            job,
            stream,
            band,
            job.frequencies[band.frequency_id],
            job.scans[0],
            geometry,
            first_fft=first,
            stop_fft=stop,
            timeline_fft_count=count,
            job_start_slot=100000,
            timestamp_origin_frame=100000,
            read_clock_convention=ReadClockConvention.ENDPOINT_MEAN
            if endpoint_mean
            else None,
        )
        actual = compiled.requests(
            job.scans[0].identifier,
            stream.identifier,
            band.identifier,
            4,
            1,
            64,
            50,
            8192,
            4,
            first,
            stop,
            count,
            100000,
            100000,
            endpoint_mean,
        )
        for key in (
            "request_fft_starts",
            "request_fft_counts",
            "desired_sample_starts",
            "endpoint_drifts",
            "ordinary_source_starts",
        ):
            np.testing.assert_array_equal(
                actual[key], getattr(reference, key), err_msg=key
            )
        np.testing.assert_array_equal(
            actual["left_delays"],
            reference.leading_head_inputs.endpoint_delays_seconds,
        )
        if endpoint_mean:
            for key in (
                "output_sample_starts",
                "centre_delays_seconds",
                "read_average_delays_seconds",
            ):
                np.testing.assert_array_equal(
                    actual[key], getattr(reference, key), err_msg=key
                )


@pytest.mark.parametrize("integration", [0, 1, 999, 43200, 172800])
def test_native_record_time_and_coordinates(tmp_path, integration):
    import importlib.util
    import os
    from decimal import Decimal

    from cuphoton.xbeam.adapters.radio.compute_plan import (
        _coordinates,
        _model,
    )

    directory = os.environ.get("XBEAM_NATIVE_BINDING_DIR")
    if not directory:
        pytest.skip("set XBEAM_NATIVE_BINDING_DIR")
    files = list(Path(directory).rglob("_binding*.so"))
    spec = importlib.util.spec_from_file_location("_binding", files[0])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    path = write_job(tmp_path)
    job = load_interop_job(path)
    actual = native.NativeJob(path).record_metadata(
        job.scans[0].identifier, integration, 0, 1, 0, b"RR"
    )
    midpoint = (
        Decimal(integration) + Decimal("0.5")
    ) * job.integration_duration_s
    absolute = job.start_time.seconds_of_day + midpoint
    days = int(absolute // 86400)
    assert actual["mjd"] == job.start_time.mjd + days
    assert actual["seconds"] == float(absolute - Decimal(days) * 86400)
    assert actual["baseline"] == 258
    scan = job.scans[0]
    expected = _coordinates(
        _model(scan, 0, float(midpoint)),
        _model(scan, 1, float(midpoint)),
        float(midpoint),
    )
    assert actual["uvw"] == expected
