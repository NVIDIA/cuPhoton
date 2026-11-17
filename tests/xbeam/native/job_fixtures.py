# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate small text jobs from synthetic stations, bands, and clocks."""


def write_job(
    directory,
    *,
    targets=(None,),
    band=0,
    zoom=False,
    stream_a=0,
    stream_b=1,
    second_edge=1000,
    second_sideband="U",
):
    """Create metadata only; loading must never open the named recordings."""
    lines = [
        "CALC FILENAME: job.calc",
        "EXECUTE TIME (SEC): 4",
        "START MJD: 58849",
        "START SECONDS: 0",
        "OUTPUT FORMAT: SWIN",
        "OUTPUT FILENAME: output",
        "NUM CONFIGURATIONS: 1",
        "INT TIME (SEC): 1",
        "SUBINT NANOSECONDS: 20000000",
        "FRINGE ROTN ORDER: 1",
        "WRITE AUTOCORRS: TRUE",
        "PULSAR BINNING: FALSE",
        "PHASED ARRAY: FALSE",
        f"FREQ ENTRIES: {3 if zoom else 2}",
    ]
    frequencies = [
        (1000, 16, "U", 32),
        (second_edge, 16, second_sideband, 32),
    ]
    if zoom:
        frequencies.append((1004, 8, "U", 16))
    for index, (edge, bandwidth, sideband, channels) in enumerate(
        frequencies
    ):
        lines.extend(
            [
                f"FREQ (MHZ) {index}: {edge}",
                f"BW (MHZ) {index}: {bandwidth}",
                f"SIDEBAND {index}: {sideband}",
                f"NUM CHANNELS {index}: {channels}",
                f"CHANS TO AVG {index}: 1",
                f"OVERSAMPLE FAC. {index}: 1",
                f"DECIMATION FAC. {index}: 1",
            ]
        )
    lines.append("TELESCOPE ENTRIES: 2")
    for station in range(2):
        lines.extend(
            [
                f"TELESCOPE NAME {station}: station{station}",
                f"CLOCK REF MJD {station}: 58849",
                f"CLOCK POLY ORDER {station}: 0",
                f"CLOCK COEFF {station}/0: 0",
            ]
        )
    lines.append("DATASTREAM ENTRIES: 2")
    for station in range(2):
        lines.extend(
            [
                f"TELESCOPE INDEX: {station}",
                "DATA FORMAT: VDIF",
                "QUANTISATION BITS: 2",
                "DATA FRAME SIZE: 8032",
                "DATA SAMPLING: REAL",
                "NUM RECORDED FREQS: 1",
                f"REC FREQ INDEX 0: {station}",
                "CLK OFFSET 0 (us): 0",
                "FREQ OFFSET 0 (Hz): 0",
                "NUM REC POLS 0: 1",
                "REC BAND 0 POL: R",
                "REC BAND 0 INDEX: 0",
                f"NUM ZOOM FREQS: {int(zoom)}",
            ]
        )
        if zoom:
            lines.extend(
                [
                    "ZOOM FREQ INDEX 0: 2",
                    "NUM ZOOM POLS 0: 1",
                    "ZOOM BAND 0 POL: R",
                    "ZOOM BAND 0 INDEX: 0",
                ]
            )
    lines.extend(
        [
            "BASELINE ENTRIES: 1",
            f"D/STREAM A INDEX 0: {stream_a}",
            f"D/STREAM B INDEX 0: {stream_b}",
            f"NUM FREQS 0: {len(targets)}",
        ]
    )
    for group, target in enumerate(targets):
        if target is not None:
            lines.append(f"TARGET FREQ 0/{group}: {target}")
        lines.extend(
            [
                f"POL PRODUCTS 0/{group}: 1",
                f"D/STREAM A BAND 0: {band}",
                f"D/STREAM B BAND 0: {band}",
            ]
        )
    for stream in range(2):
        lines.extend(
            [
                f"D/STREAM {stream} FILES: 1",
                f"FILE {stream}/0: absent-{stream}.vdif",
            ]
        )
    path = directory / "job.input"
    path.write_text("\n".join(lines) + "\n")
    (directory / "job.calc").write_text(
        "IM FILENAME: job.im\nNUM SCANS: 1\nSCAN 0 IDENTIFIER: scan\n"
        "SCAN 0 START (S): 0\nSCAN 0 DUR (S): 4\n"
        "SCAN 0 POINTING SRC: 0\nSCAN 0 NUM PHS CTRS: 1\n"
    )
    model = [
        "NUM SCANS: 1",
        "SCAN 0 POINTING SRC: 0",
        "SCAN 0 NUM POLY: 1",
        "SCAN 0 POLY 0 MJD: 58849",
        "SCAN 0 POLY 0 SEC: 0",
    ]
    for station in range(2):
        for field in ("DELAY (us)", "U (m)", "V (m)", "W (m)"):
            model.append(f"SRC 0 ANT {station} {field}: 0 0")
    (directory / "job.im").write_text("\n".join(model) + "\n")
    return path
