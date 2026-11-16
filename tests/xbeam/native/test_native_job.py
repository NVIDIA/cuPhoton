# SPDX-License-Identifier: Apache-2.0
"""Differential job-adapter tests using independently generated metadata."""

import json
import subprocess
from pathlib import Path

import pytest

from cuphoton.xbeam.adapters.radio.models import job_from_dict
from cuphoton.xbeam.commands import load_interop_job
from cuphoton.xbeam.errors import (
    ConfigurationError,
    JobLoadError,
    UnsupportedFeatureError,
)
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


@pytest.mark.parametrize("target", [None, 0])
def test_native_rejects_incompatible_sky_grids(
    tmp_path, native_executable, target
):
    path = write_job(tmp_path, second_edge=1001, targets=(target,))
    with pytest.raises(UnsupportedFeatureError, match="regridding"):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "regridding" in result.stderr


def test_native_accepts_equivalent_usb_lsb_grids(tmp_path, native_executable):
    path = write_job(tmp_path, second_edge=1016, second_sideband="L")
    actual = json.loads(
        subprocess.check_output(
            [str(native_executable), "inspect-job", str(path)]
        )
    )
    assert job_from_dict(actual) == load_interop_job(path)


def test_native_rejects_duplicate_baseline_products(
    tmp_path, native_executable
):
    path = write_job(tmp_path, targets=(0, 0))
    with pytest.raises(
        ConfigurationError, match="duplicate baseline product"
    ):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "duplicate baseline product" in result.stderr


@pytest.mark.parametrize("second", ["1", "1.00000000000000001", "2"])
def test_native_model_epochs_match_python(
    tmp_path, native_executable, second
):
    path = write_job(tmp_path)
    calc = tmp_path / "job.calc"
    calc.write_text(calc.read_text().replace("START (S): 0", "START (S): 2"))
    model = tmp_path / "job.im"
    text = model.read_text()
    polynomial = text[text.index("SCAN 0 POLY 0 MJD") :]
    model.write_text(
        text.replace("NUM POLY: 1", "NUM POLY: 2").replace(
            "POLY 0 SEC: 0", "POLY 0 SEC: 1"
        )
        + polynomial.replace("POLY 0", "POLY 1").replace(
            "POLY 1 SEC: 0", f"POLY 1 SEC: {second}"
        )
    )
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    if second == "2":
        assert result.returncode == 0, result.stderr
        assert job_from_dict(json.loads(result.stdout)) == load_interop_job(
            path
        )
    else:
        with pytest.raises(ConfigurationError, match="model epoch"):
            load_interop_job(path)
        assert result.returncode == 1
        assert "model epoch" in result.stderr


def test_native_rejects_negative_file_count(tmp_path, native_executable):
    path = write_job(tmp_path)
    path.write_text(
        path.read_text().replace(
            "D/STREAM 0 FILES: 1", "D/STREAM 0 FILES: -1"
        )
    )
    with pytest.raises(JobLoadError, match="FILES must be nonnegative"):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "file count must be nonnegative" in result.stderr


def test_native_rejects_duplicate_scan_identifiers(
    tmp_path, native_executable
):
    path = write_job(tmp_path)
    calc = tmp_path / "job.calc"
    text = calc.read_text()
    second_scan = "\n".join(
        line.replace("SCAN 0", "SCAN 1")
        for line in text.splitlines()
        if line.startswith("SCAN 0")
    )
    calc.write_text(
        text.replace("NUM SCANS: 1", "NUM SCANS: 2") + second_scan + "\n"
    )
    model = tmp_path / "job.im"
    text = model.read_text()
    second_model = text[text.index("SCAN 0 POINTING SRC") :].replace(
        "SCAN 0", "SCAN 1"
    )
    model.write_text(
        text.replace("NUM SCANS: 1", "NUM SCANS: 2") + second_model
    )
    with pytest.raises(ConfigurationError, match="duplicate scan identifier"):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "duplicate scan identifier" in result.stderr


def test_native_rejects_model_starting_after_scan(
    tmp_path, native_executable
):
    path = write_job(tmp_path)
    model = tmp_path / "job.im"
    model.write_text(
        model.read_text().replace("POLY 0 SEC: 0", "POLY 0 SEC: 1")
    )
    with pytest.raises(ConfigurationError, match="starts after"):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "first model starts after" in result.stderr


def test_missing_reader_geometry_is_inspectable_but_cannot_replay(
    tmp_path, native_executable
):
    path = write_job(tmp_path)
    metadata = json.loads(
        subprocess.check_output(
            [str(native_executable), "inspect-job", str(path)]
        )
    )
    assert metadata["buffered_reader"] is None
    result = subprocess.run(
        [
            str(native_executable),
            "inspect-replay",
            str(path),
            "0",
            "0",
            str(tmp_path / "absent.vdix"),
            "scan",
            "0",
            "1",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "native replay requires DATA BUFFER FACTOR" in result.stderr


@pytest.mark.parametrize("field", ["OVERSAMPLE FAC.", "DECIMATION FAC."])
def test_native_rejects_unimplemented_sampling_factors(
    tmp_path, native_executable, field
):
    path = write_job(tmp_path)
    path.write_text(
        path.read_text().replace(f"{field} 0: 1", f"{field} 0: 2")
    )
    with pytest.raises(UnsupportedFeatureError, match="factor of 1"):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "factors must be 1" in result.stderr


def test_native_rejects_zoom_with_different_channel_spacing(
    tmp_path, native_executable
):
    path = write_job(tmp_path, zoom=True, band=1)
    path.write_text(
        path.read_text().replace("NUM CHANNELS 2: 16", "NUM CHANNELS 2: 8")
    )
    with pytest.raises(UnsupportedFeatureError, match="parent slice"):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "parent slice" in result.stderr


@pytest.mark.parametrize(
    "original,replacement",
    [
        ("NUM CHANNELS 1: 32", "NUM CHANNELS 1: 64"),
        ("BW (MHZ) 1: 16", "BW (MHZ) 1: 8"),
    ],
)
def test_native_rejects_mixed_recorded_fft_grids(
    tmp_path, native_executable, original, replacement
):
    path = write_job(tmp_path)
    text = path.read_text().replace(original, replacement)
    text = text.replace("NUM RECORDED FREQS: 1", "NUM RECORDED FREQS: 2")
    text = text.replace(
        "NUM REC POLS 0: 1",
        "NUM REC POLS 0: 1\nREC FREQ INDEX 1: 1\nNUM REC POLS 1: 1\n"
        "CLK OFFSET 1 (us): 0\nFREQ OFFSET 1 (Hz): 0",
    )
    text = text.replace(
        "REC BAND 0 INDEX: 0",
        "REC BAND 0 INDEX: 0\nREC BAND 1 POL: L\nREC BAND 1 INDEX: 1",
    )
    path.write_text(text)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "recorded bands must share one FFT grid" in result.stderr


@pytest.mark.parametrize("owner", ["clock", "delay"])
def test_native_rejects_cubic_terms_for_quadratic_fringe_order(
    tmp_path, native_executable, owner
):
    path = write_job(tmp_path)
    text = path.read_text().replace(
        "FRINGE ROTN ORDER: 1", "FRINGE ROTN ORDER: 2"
    )
    if owner == "clock":
        text = text.replace("CLOCK POLY ORDER 0: 0", "CLOCK POLY ORDER 0: 3")
        text = text.replace(
            "CLOCK COEFF 0/0: 0",
            "CLOCK COEFF 0/0: 0\nCLOCK COEFF 0/1: 0\n"
            "CLOCK COEFF 0/2: 0\nCLOCK COEFF 0/3: 1",
        )
    else:
        model = tmp_path / "job.im"
        model.write_text(
            model.read_text().replace(
                "DELAY (us): 0 0", "DELAY (us): 0 0 0 1"
            )
        )
    path.write_text(text)
    with pytest.raises(UnsupportedFeatureError, match="cubic or higher"):
        load_interop_job(path)
    result = subprocess.run(
        [str(native_executable), "inspect-job", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "cubic or higher" in result.stderr


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
