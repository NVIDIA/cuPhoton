# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic text fixtures from the adapter's declared metadata fields."""

from decimal import Decimal

import pytest

from cuphoton.xbeam.adapters.radio.difx_job import load_interop_job
from cuphoton.xbeam.adapters.radio.models import MJDTime
from cuphoton.xbeam.errors import (
    JobLoadError,
    UnsupportedFeatureError,
    XBeamError,
)


def _write_job(
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


@pytest.mark.parametrize("zoom", [False, True])
def test_legacy_inference_matches_explicit_recorded_or_zoom_target(
    tmp_path, zoom
):
    target, band = (2, 1) if zoom else (0, 0)
    legacy = load_interop_job(_write_job(tmp_path, band=band, zoom=zoom))
    explicit = load_interop_job(
        _write_job(tmp_path, targets=(target,), band=band, zoom=zoom)
    )
    assert legacy == explicit
    assert legacy.baselines[0].products[0].output_frequency_id == target


def test_legacy_inference_accepts_equal_usb_lsb_grids(tmp_path):
    legacy = load_interop_job(
        _write_job(tmp_path, second_edge=1016, second_sideband="L")
    )
    explicit = load_interop_job(
        _write_job(
            tmp_path, targets=(0,), second_edge=1016, second_sideband="L"
        )
    )
    assert legacy == explicit


def test_explicit_and_inferred_frequency_groups_can_coexist(tmp_path):
    job = load_interop_job(_write_job(tmp_path, targets=(None, 1)))
    assert [p.output_frequency_id for p in job.baselines[0].products] == [
        0,
        1,
    ]


@pytest.mark.parametrize("target", [None, 0])
@pytest.mark.parametrize("side", ["stream_a", "stream_b"])
@pytest.mark.parametrize("identifier", [-1, 2])
def test_bad_datastream_reference_has_context(
    tmp_path, target, side, identifier
):
    path = _write_job(tmp_path, targets=(target,), **{side: identifier})
    with pytest.raises(
        JobLoadError, match=r"baseline\[0\].*datastream table"
    ):
        load_interop_job(path)


@pytest.mark.parametrize("target", [None, 0])
def test_incompatible_frequency_grids_remain_rejected(tmp_path, target):
    path = _write_job(tmp_path, targets=(target,), second_edge=1001)
    with pytest.raises(UnsupportedFeatureError, match="regridding"):
        load_interop_job(path)


def test_missing_recorded_or_zoom_band_is_rejected(tmp_path):
    with pytest.raises(XBeamError, match="band 9"):
        load_interop_job(_write_job(tmp_path, band=9))


def test_duplicate_explicit_target_is_rejected(tmp_path):
    path = _write_job(tmp_path, targets=(0,))
    path.write_text(
        path.read_text().replace(
            "TARGET FREQ 0/0: 0", "TARGET FREQ 0/0: 0\nTARGET FREQ 0/0: 1"
        )
    )
    with pytest.raises(JobLoadError, match="TARGET FREQ.*repeated"):
        load_interop_job(path)


@pytest.mark.parametrize("missing", [0, 1])
def test_explicitly_empty_feed_preserves_surviving_ids(tmp_path, missing):
    path = _write_job(tmp_path)
    path.write_text(
        path.read_text()
        .replace(
            f"D/STREAM {missing} FILES: 1", f"D/STREAM {missing} FILES: 0"
        )
        .replace(f"FILE {missing}/0: absent-{missing}.vdif\n", "")
    )
    with pytest.warns(RuntimeWarning, match=f"datastreams {missing};"):
        job = load_interop_job(path)
    assert [stream.identifier for stream in job.datastreams] == [1 - missing]
    assert [station.identifier for station in job.stations] == [0, 1]
    assert not job.baselines
    assert job.write_autocorrelations
    # A named file need not be opened during metadata normalization, but it
    # remains a real source requirement; it is never pruned because of a stat.
    assert (
        job.datastreams[0]
        .source_files[0]
        .endswith(f"absent-{1 - missing}.vdif")
    )


def test_negative_file_count_is_not_an_unavailable_feed(tmp_path):
    path = _write_job(tmp_path)
    path.write_text(
        path.read_text().replace(
            "D/STREAM 0 FILES: 1", "D/STREAM 0 FILES: -1"
        )
    )
    with pytest.raises(JobLoadError, match="FILES must be nonnegative"):
        load_interop_job(path)


def test_all_empty_feeds_cannot_become_a_successful_empty_job(tmp_path):
    path = _write_job(tmp_path)
    text = path.read_text()
    for index in (0, 1):
        text = text.replace(
            f"D/STREAM {index} FILES: 1", f"D/STREAM {index} FILES: 0"
        ).replace(f"FILE {index}/0: absent-{index}.vdif\n", "")
    path.write_text(text)
    with pytest.warns(RuntimeWarning, match="datastreams 0, 1;"):
        with pytest.raises(XBeamError, match="at least one datastream"):
            load_interop_job(path)


@pytest.mark.parametrize("sideband,edge", [("u", 1000), ("l", 1016)])
def test_sideband_tokens_are_case_insensitive(tmp_path, sideband, edge):
    load_interop_job(
        _write_job(tmp_path, second_edge=edge, second_sideband=sideband)
    )


def test_unknown_sideband_is_rejected(tmp_path):
    path = _write_job(tmp_path, second_edge=1016, second_sideband="invalid")
    with pytest.raises(JobLoadError, match="SIDEBAND 1 must be U or L"):
        load_interop_job(path)


@pytest.mark.parametrize("zoom", [False, True])
@pytest.mark.parametrize("identifier", [-1, 3])
def test_frequency_references_are_checked_before_lookup(
    tmp_path, zoom, identifier
):
    path = _write_job(tmp_path, zoom=zoom, band=int(zoom))
    field, original = (
        ("ZOOM FREQ INDEX 0", 2) if zoom else ("REC FREQ INDEX 0", 0)
    )
    path.write_text(
        path.read_text().replace(
            f"{field}: {original}", f"{field}: {identifier}"
        )
    )
    with pytest.raises(JobLoadError, match=f"{field}.*frequency table"):
        load_interop_job(path)


def test_empty_recorded_frequency_table_is_rejected(tmp_path):
    path = _write_job(tmp_path)
    path.write_text(
        path.read_text().replace(
            "NUM RECORDED FREQS: 1", "NUM RECORDED FREQS: 0"
        )
    )
    with pytest.raises(
        JobLoadError, match="NUM RECORDED FREQS must be positive"
    ):
        load_interop_job(path)


@pytest.mark.parametrize(
    "data_format,threads",
    [
        ("VDIF", (0, 1)),
        ("INTERLACEDVDIF/7:11/2/2", (7, 11)),
    ],
)
def test_nonzero_difx_units_epochs_and_thread_mapping(
    tmp_path, data_format, threads
):
    path = _write_job(tmp_path)
    text = path.read_text().replace(
        "START SECONDS: 0", "START SECONDS: 123.25"
    )
    text = text.replace(
        "SUBINT NANOSECONDS: 20000000", "SUBINT NANOSECONDS: 25000000"
    )
    text = text.replace("DATA FRAME SIZE: 8032", "DATA FRAME SIZE: 4032")
    text = text.replace("DATA FORMAT: VDIF", f"DATA FORMAT: {data_format}")
    text = text.replace("CLK OFFSET 0 (us): 0", "CLK OFFSET 0 (us): -2.25")
    text = text.replace("FREQ OFFSET 0 (Hz): 0", "FREQ OFFSET 0 (Hz): 17.5")
    text = text.replace("NUM REC POLS 0: 1", "NUM REC POLS 0: 2")
    text = text.replace(
        "REC BAND 0 INDEX: 0",
        "REC BAND 0 INDEX: 0\nREC BAND 1 POL: L\nREC BAND 1 INDEX: 0",
    )
    for station in range(2):
        text = text.replace(
            f"CLOCK REF MJD {station}: 58849",
            f"CLOCK REF MJD {station}: 58848.75",
        )
        text = text.replace(
            f"CLOCK COEFF {station}/0: 0", f"CLOCK COEFF {station}/0: 1.5"
        )
    path.write_text(text)
    model = tmp_path / "job.im"
    model.write_text(
        model.read_text()
        .replace("POLY 0 SEC: 0", "POLY 0 SEC: 123")
        .replace("DELAY (us): 0 0", "DELAY (us): 3.5 -0.25")
    )

    job = load_interop_job(path)
    assert job.start_time == MJDTime(58849, Decimal("123.25"))
    assert job.subintegration_duration_s == Decimal("0.025")
    assert job.scans[0].start_offset_s == Decimal("-0.25")
    for station in job.stations:
        assert station.clock.epoch == MJDTime(58848, Decimal("64800"))
        assert station.clock.coefficients_s == pytest.approx((1.5e-6,))
    for stream in job.datastreams:
        assert stream.frame_size_bytes == 4032
        assert (
            tuple(band.thread_id for band in stream.recorded_bands) == threads
        )
        for band in stream.recorded_bands:
            assert band.clock_offset_s == pytest.approx(-2.25e-6)
            assert band.lo_offset_hz == 17.5
    for model in job.scans[0].station_models:
        assert model.epoch_offset_s == Decimal("-0.25")
        assert model.delay_coefficients_s == pytest.approx((3.5e-6, -0.25e-6))
