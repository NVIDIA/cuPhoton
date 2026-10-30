# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Analytic sky-tone checks for spectrally inverted complex recordings."""

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
    RecordedBand,
    SamplingKind,
    Scan,
    Sideband,
    Station,
    VoltageBatch,
)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("lo_offset", [0.0, 5.0])
def test_conjugated_lsb_and_usb_match_with_band_edge_and_fractional_delay(
    backend, lo_offset
):
    if backend == "cupy":
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    epoch = MJDTime(58849, Decimal(0))
    rate, edge, channels = 1024.0, 4.0e5, 64
    delay = (0.0, -3.37 / rate)
    usb = Frequency(0, edge, rate, Sideband.USB, channels)
    job = Job(
        epoch,
        Decimal(2),
        Decimal(2),
        Decimal("0.5"),
        1,
        tuple(
            Station(i, f"S{i}", ClockPolynomial(epoch, (0.0,)))
            for i in range(2)
        ),
        (usb,),
        tuple(
            Datastream(
                i,
                i,
                SamplingKind.COMPLEX,
                2,
                40,
                ("unused.vdif",),
                (
                    RecordedBand(
                        0, 0, "R", lo_offset_hz=lo_offset if i else 0.0
                    ),
                ),
                sample_rate_hz=rate,
            )
            for i in range(2)
        ),
        (Baseline(0, 0, 1, (BaselineProduct(0, 0, 0),)),),
        (
            Scan(
                "s",
                Decimal(0),
                Decimal(2),
                "0",
                "0",
                tuple(
                    DelayCoordinateModel(
                        i, Decimal(0), (delay[i],), (0.0,), (0.0,), (0.0,)
                    )
                    for i in range(2)
                ),
            ),
        ),
    )
    # Channel-centred tones have an exactly real, positive cross-spectrum
    # after removing the independently prescribed station delays.
    bins = np.array([3, 9, 27])
    frequencies = bins * rate / channels
    amplitudes = np.array([1 + 0.5j, 0.3 - 0.2j, -0.7 + 0.1j])
    time = np.arange(2 * 1024 + 64) / rate
    signals = []
    for station, tau in enumerate(delay):
        signal = (
            np.exp(2j * np.pi * (time + tau)[:, None] * frequencies)
            @ amplitudes
        )
        signal *= np.exp(
            2j * np.pi * (edge * tau - (lo_offset if station else 0) * time)
        )
        signals.append(signal.astype(np.complex64))

    class Source:
        def __init__(self, inverted=False):
            self.inverted = inverted

        def batches(self, stream):
            samples = signals[stream.identifier]
            if self.inverted:
                samples = np.conj(samples)
            yield VoltageBatch(
                samples, epoch, rate, 0, "R", np.ones(samples.size, bool)
            )

    (upper,) = correlate(job, backend=backend, recording_source=Source())
    lower_job = replace(
        job,
        frequencies=(
            replace(usb, band_edge_hz=edge + rate, sideband=Sideband.LSB),
        ),
    )
    (lower,) = correlate(
        lower_job, backend=backend, recording_source=Source(True)
    )
    np.testing.assert_allclose(
        lower.spectrum[1:], upper.spectrum[1:], rtol=2e-5, atol=2e-4
    )
    assert lower.weight == upper.weight == 1.0
    if lo_offset == 0:
        np.testing.assert_allclose(
            np.angle(upper.spectrum[bins]), 0, atol=2e-5
        )
        assert np.all(upper.spectrum[bins].real > 0)
