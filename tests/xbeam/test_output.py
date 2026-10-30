# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from decimal import Decimal

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.output import (
    OutputFormatError,
    OutputRecord,
    read_xbeam_v1,
    write_xbeam_v1,
)


def make_record(scale=1.0):
    return OutputRecord(
        antenna_a="T1",
        antenna_b="T2",
        mjd=58948,
        seconds_of_day=Decimal("25200.500000000"),
        source_id=0,
        configuration_id=0,
        output_frequency_id=3,
        polarization_a="R",
        polarization_b="L",
        weight=0.75,
        u_m=1.0,
        v_m=2.0,
        w_m=3.0,
        spectrum=np.array([1 + 2j, 3 - 4j], dtype=np.complex64) * scale,
    )


def test_xbeam_v1_round_trip_and_binary_layout(tmp_path):
    records = [make_record(), make_record(2)]
    assert write_xbeam_v1(tmp_path, records) == 2
    raw = (tmp_path / "spectra.c64").read_bytes()
    expected = (
        np.concatenate([r.spectrum for r in records]).astype("<c8").tobytes()
    )
    assert raw == expected
    lines = (tmp_path / "metadata.jsonl").read_text().splitlines()
    assert [json.loads(line)["spectrum_offset"] for line in lines] == [0, 16]
    assert json.loads(lines[0])["seconds_of_day"] == "25200.500000000"
    actual = read_xbeam_v1(tmp_path)
    assert actual[0].seconds_of_day == Decimal("25200.500000000")
    np.testing.assert_array_equal(actual[1].spectrum, records[1].spectrum)


def test_writer_accepts_structural_mapping_and_seconds_alias(tmp_path):
    record = (
        dict(make_record().__dict__)
        if hasattr(make_record(), "__dict__")
        else {
            name: getattr(make_record(), name)
            for name in make_record().__slots__
        }
    )
    record["seconds"] = record.pop("seconds_of_day")
    write_xbeam_v1(tmp_path, [record])
    assert read_xbeam_v1(tmp_path)[0].seconds_of_day == Decimal(
        "25200.500000000"
    )


def test_reader_rejects_truncated_or_trailing_spectra(tmp_path):
    write_xbeam_v1(tmp_path, [make_record()])
    path = tmp_path / "spectra.c64"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(OutputFormatError, match="truncated"):
        read_xbeam_v1(tmp_path)


def test_writer_refuses_to_overwrite(tmp_path):
    write_xbeam_v1(tmp_path, [make_record()])
    with pytest.raises(FileExistsError):
        write_xbeam_v1(tmp_path, [make_record()])


def test_reader_reports_malformed_seconds_as_output_error(tmp_path):
    write_xbeam_v1(tmp_path, [make_record()])
    path = tmp_path / "metadata.jsonl"
    record = json.loads(path.read_text())
    record["seconds_of_day"] = "invalid"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(
        OutputFormatError, match="metadata.jsonl:1: invalid record"
    ):
        read_xbeam_v1(tmp_path)
