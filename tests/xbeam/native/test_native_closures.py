# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Raw VDIF to serialized multi-station closures, with an independent DFT."""

import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.closure import compute_closures
from cuphoton.xbeam.adapters.radio.engine import correlate
from cuphoton.xbeam.adapters.radio.swin import (
    from_visibility,
    read_swin,
    write_swin,
)
from cuphoton.xbeam.commands import load_interop_job
from cuphoton.xbeam.windowing import CorrelationWindow
from tests.xbeam.native.closure_fixtures import (
    CHANNELS,
    INTEGRATION_SECONDS,
    PAIRS,
    PROFILE,
    QUADRANGLES,
    STATIONS,
    TRIANGLES,
    direct_visibilities,
    station_coordinates,
    write_closure_job,
)


@pytest.fixture
def closure_job(tmp_path):
    path, samples = write_closure_job(tmp_path)
    return path, direct_visibilities(samples)


def reference_records(path):
    records = correlate(
        load_interop_job(path),
        backend="numpy",
        profile=PROFILE,
        window=CorrelationWindow("closure", 0, 2),
        index_dir=str(path.parent / "indexes"),
    )
    output = path.parent / "reference.swin"
    write_swin(
        output,
        (
            from_visibility(
                record, dict(zip(STATIONS, range(4), strict=True))
            )
            for record in records
        ),
    )
    return read_swin(output, channel_count=CHANNELS)


def assert_visibilities(records, expected):
    seen = set()
    for record in records:
        integration = int(record.seconds) // INTEGRATION_SECONDS
        a, b = record.antenna_pair
        key = integration, a, b
        assert key not in seen
        seen.add(key)
        spectrum, weight = expected[key]
        assert record.mjd == 58849
        assert record.seconds == (integration + 0.5) * INTEGRATION_SECONDS
        assert record.configuration_id == record.source_id == 0
        assert record.frequency_id == record.pulsar_bin == 0
        assert record.polarization == "RR"
        assert record.weight == weight
        np.testing.assert_array_equal(
            (record.u_m, record.v_m, record.w_m),
            station_coordinates(a, record.seconds)
            - station_coordinates(b, record.seconds),
        )
        np.testing.assert_allclose(
            record.spectrum, spectrum, rtol=3e-6, atol=2e-6
        )
    assert seen == set(expected)
    assert len(records) == 2 * (len(PAIRS) + len(STATIONS))


def assert_closures(records, expected):
    closures = compute_closures(
        records, triangles=TRIANGLES, quadrangles=QUADRANGLES
    )
    assert len(closures) == 2 * (len(TRIANGLES) + len(QUADRANGLES))
    for closure in closures:
        integration = int(closure.key.seconds_of_day) // INTEGRATION_SECONDS

        def visibility(a, b, integration=integration):
            if (integration, a, b) in expected:
                return expected[integration, a, b][0]
            return expected[integration, b, a][0].conj()

        assert closure.valid.all()
        if closure.kind == "phase":
            a, b, c = closure.stations
            value = np.angle(
                visibility(a, b) * visibility(b, c) * visibility(c, a)
            )
            # Nonzero, non-pi phases detect global conjugation of spectra.
            assert np.max(np.abs(np.sin(value))) > 0.2
            residual = np.angle(np.exp(1j * (closure.values - value)))
            np.testing.assert_allclose(residual, 0, atol=2e-5)
        else:
            a, b, c, d = closure.stations
            value = np.log(
                np.abs(
                    visibility(a, b)
                    * visibility(c, d)
                    / (visibility(a, c) * visibility(b, d))
                )
            )
            assert np.max(np.abs(value)) > 0.2
            np.testing.assert_allclose(closure.values, value, atol=2e-5)


def test_reference_vdif_closures_match_direct_dft(closure_job):
    path, expected = closure_job
    records = reference_records(path)
    assert_visibilities(records, expected)
    assert_closures(records, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("prefetch", ["on", "off"])
def test_native_vdif_closures_match_reference(closure_job, prefetch):
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        pytest.skip("CUDA devices are explicitly disabled")
    executable = (
        Path(__file__).resolve().parents[3]
        / "build/xbeam-native/xbeam-native-correlate"
    )
    if not executable.is_file():
        pytest.skip("build the standalone xbeam-native-correlate with CUDA")
    path, expected = closure_job
    output = path.parent / "native"
    completed = subprocess.run(
        [
            str(executable),
            str(path),
            "--output",
            str(output),
            "--index-dir",
            str(path.parent / "indexes"),
            "--scan",
            "closure",
            "--first-integration",
            "0",
            "--integration-count",
            "2",
            "--fft-tile",
            "256",
            "--vdif-io",
            "posix",
            "--device",
            "0",
            "--prefetch",
            prefetch,
            "--profile",
            PROFILE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "native-run.json").read_text())
    assert report["backend"] == "native-cuda"
    assert report["records"] == 20
    assert report["integration_count"] == 2
    records = read_swin(output / "visibilities.swin", channel_count=CHANNELS)
    assert_visibilities(records, expected)
    assert_closures(records, expected)
    reference = reference_records(path)
    for actual, wanted in zip(records, reference, strict=True):
        for field in fields(actual):
            if field.name != "spectrum":
                assert getattr(actual, field.name) == getattr(
                    wanted, field.name
                ), field.name
        np.testing.assert_allclose(
            actual.spectrum, wanted.spectrum, rtol=3e-6, atol=2e-6
        )
