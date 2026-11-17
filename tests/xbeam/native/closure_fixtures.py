# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small four-station recordings, independent of any observation archive."""

from itertools import combinations

import numpy as np

from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)

SAMPLE_RATE = 4096
# Align each reader request to the profile's 8192-byte packed-data grid.
SUBINTEGRATION_SECONDS = 8
# Two 256-FFT tiles exercise prefetch and accumulation between tiles.
INTEGRATION_SECONDS = 2 * SUBINTEGRATION_SECONDS
INTEGRATION_SAMPLES = SAMPLE_RATE * INTEGRATION_SECONDS
FFT_SAMPLES = 128
CHANNELS = FFT_SAMPLES // 2
STATIONS = ("AA", "BB", "CC", "DD")
# Reversed edges exercise the visibility orientation through SWIN encoding.
PAIRS = ((0, 1), (2, 0), (0, 3), (1, 2), (3, 1), (2, 3))
PROFILE = "interop-real-usb-buffered-window-replay-v1"


def station_coordinates(station, time):
    return np.asarray(
        (100 + 13 * station, -20 + 7 * station**2, 3 - 5 * station)
    ) + time * np.asarray((0.25 * station, -0.5 * station, 0.125 * station))


def write_closure_job(directory):
    """Write two integration periods plus padding and return decoded samples.

    Two independent noise sources reach each station with different integer
    lags and mixing coefficients. Quantization retains nontrivial closure
    phase and amplitude; a single coherent signal would not test either.
    One invalid frame at DD also exercises fractional exposure.
    """
    rng = np.random.default_rng(8531)
    count = 4 * INTEGRATION_SAMPLES
    sources = rng.normal(size=(2, count))
    samples = []
    for station, name in enumerate(STATIONS):
        voltage = (
            np.roll(sources[0], station)
            + (0.4 + 0.2 * station)
            * np.roll(sources[1], (-2, 3, 1, 5)[station])
            + 0.15 * rng.normal(size=count)
        )
        voltage /= np.std(voltage)
        codes = np.digitize(voltage, (-0.98, 0, 0.98)).astype(np.uint8)
        decoded = np.asarray((-3.3359, -1, 1, 3.3359), dtype=np.float32)[
            codes
        ]
        path = directory / f"{name}.vdif"
        with path.open("wb") as stream:
            for frame, frame_codes in enumerate(
                codes.reshape(-1, FFT_SAMPLES)
            ):
                seconds, number = divmod(frame, SAMPLE_RATE // FFT_SAMPLES)
                invalid = station == 3 and frame == 300
                words = np.asarray(
                    (
                        seconds | (int(invalid) << 31),
                        number | (40 << 24),
                        8,  # 32-byte header and 32-byte payload.
                        (ord(name[0]) << 8) | ord(name[1]) | (1 << 26),
                        0,
                        0,
                        0,
                        0,
                    ),
                    dtype="<u4",
                )
                packed = np.bitwise_or.reduce(
                    frame_codes.reshape(-1, 4)
                    << np.asarray((0, 2, 4, 6), dtype=np.uint8),
                    axis=1,
                )
                stream.write(words.tobytes())
                stream.write(packed.tobytes())
                if invalid:
                    decoded[
                        frame * FFT_SAMPLES : (frame + 1) * FFT_SAMPLES
                    ] = 0
        samples.append(decoded)
        build_vdif_binary_index(
            path, index_dir=directory / "indexes", backend="posix"
        )

    lines = [
        "CALC FILENAME: closure.calc",
        f"EXECUTE TIME (SEC): {2 * INTEGRATION_SECONDS}",
        "START MJD: 58849",
        "START SECONDS: 0",
        "OUTPUT FORMAT: SWIN",
        "OUTPUT FILENAME: unused",
        "NUM CONFIGURATIONS: 1",
        f"INT TIME (SEC): {INTEGRATION_SECONDS}",
        f"SUBINT NANOSECONDS: {SUBINTEGRATION_SECONDS * 10**9}",
        "FRINGE ROTN ORDER: 1",
        "WRITE AUTOCORRS: TRUE",
        "PULSAR BINNING: FALSE",
        "PHASED ARRAY: FALSE",
        "DATA BUFFER FACTOR: 24",
        "NUM DATA SEGMENTS: 8",
        "NUM BUFFERED FFTS: 256",
        "FREQ ENTRIES: 1",
        "FREQ (MHZ) 0: 0",
        "BW (MHZ) 0: 0.002048",
        "SIDEBAND 0: U",
        f"NUM CHANNELS 0: {CHANNELS}",
        "CHANS TO AVG 0: 1",
        "OVERSAMPLE FAC. 0: 1",
        "DECIMATION FAC. 0: 1",
        "TELESCOPE ENTRIES: 4",
    ]
    for station, name in enumerate(STATIONS):
        lines.extend(
            [
                f"TELESCOPE NAME {station}: {name}",
                f"CLOCK REF MJD {station}: 58849",
                f"CLOCK POLY ORDER {station}: 0",
                f"CLOCK COEFF {station}/0: 0",
            ]
        )
    lines.append("DATASTREAM ENTRIES: 4")
    for station in range(4):
        lines.extend(
            [
                f"TELESCOPE INDEX: {station}",
                "DATA FORMAT: VDIF",
                "QUANTISATION BITS: 2",
                "DATA FRAME SIZE: 64",
                "DATA SAMPLING: REAL",
                "NUM RECORDED FREQS: 1",
                "REC FREQ INDEX 0: 0",
                "CLK OFFSET 0 (us): 0",
                "FREQ OFFSET 0 (Hz): 0",
                "NUM REC POLS 0: 1",
                "REC BAND 0 POL: R",
                "REC BAND 0 INDEX: 0",
                "NUM ZOOM FREQS: 0",
            ]
        )
    lines.append(f"BASELINE ENTRIES: {len(PAIRS)}")
    for baseline, (a, b) in enumerate(PAIRS):
        lines.extend(
            [
                f"D/STREAM A INDEX {baseline}: {a}",
                f"D/STREAM B INDEX {baseline}: {b}",
                f"NUM FREQS {baseline}: 1",
                f"TARGET FREQ {baseline}/0: 0",
                f"POL PRODUCTS {baseline}/0: 1",
                "D/STREAM A BAND 0: 0",
                "D/STREAM B BAND 0: 0",
            ]
        )
    for station, name in enumerate(STATIONS):
        lines.extend(
            [f"D/STREAM {station} FILES: 1", f"FILE {station}/0: {name}.vdif"]
        )
    path = directory / "closure.input"
    path.write_text("\n".join(lines) + "\n")
    (directory / "closure.calc").write_text(
        "IM FILENAME: closure.im\nNUM SCANS: 1\nSCAN 0 IDENTIFIER: closure\n"
        f"SCAN 0 START (S): 0\nSCAN 0 DUR (S): {2 * INTEGRATION_SECONDS}\n"
        "SCAN 0 POINTING SRC: 0\nSCAN 0 NUM PHS CTRS: 1\n"
    )
    model = [
        "NUM SCANS: 1",
        "SCAN 0 POINTING SRC: 0",
        "SCAN 0 NUM POLY: 1",
        "SCAN 0 POLY 0 MJD: 58849",
        "SCAN 0 POLY 0 SEC: 0",
    ]
    for station in range(4):
        model.append(f"SRC 0 ANT {station} DELAY (us): 0 0")
        origin = station_coordinates(station, 0)
        rate = station_coordinates(station, 1) - origin
        for field, value, slope in zip("UVW", origin, rate, strict=True):
            model.append(f"SRC 0 ANT {station} {field} (m): {value} {slope}")
    (directory / "closure.im").write_text("\n".join(model) + "\n")
    return path, np.asarray(samples)


def direct_visibilities(samples):
    """Direct DFT and averaging, without xBeam DSP or FFT routines."""
    dft = np.exp(
        -2j
        * np.pi
        * np.arange(CHANNELS)[:, None]
        * np.arange(FFT_SAMPLES)[None, :]
        / FFT_SAMPLES
    )
    result = {}
    for integration in range(2):
        frames = samples[
            :,
            integration * INTEGRATION_SAMPLES : (integration + 1)
            * INTEGRATION_SAMPLES,
        ].reshape(4, -1, FFT_SAMPLES)
        spectra = frames @ dft.T
        fractions = np.mean(frames != 0, axis=-1)
        for a, b in (*PAIRS, *((i, i) for i in range(4))):
            exposure = fractions[a] if a == b else fractions[a] * fractions[b]
            result[integration, a, b] = (
                np.sum(spectra[a] * spectra[b].conj(), axis=0)
                / (np.sum(exposure) * FFT_SAMPLES),
                np.mean(exposure),
            )
    return result


TRIANGLES = tuple(combinations(range(4), 3))
QUADRANGLES = ((0, 1, 2, 3), (0, 1, 3, 2))
