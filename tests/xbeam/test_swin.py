# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal
from io import BytesIO

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import VisibilityRecord
from cuphoton.xbeam.adapters.radio.swin import (
    HEADER,
    SwinFormatError,
    SwinRecord,
    SwinWriter,
    decode_baseline,
    encode_baseline,
    from_visibility,
    iter_swin,
    read_record,
    write_record,
)


def test_baseline_code_round_trip():
    assert encode_baseline(0, 1) == 258
    assert decode_baseline(514) == (1, 1)
    with pytest.raises(SwinFormatError):
        decode_baseline(0)


def test_truncation_and_bad_sync_are_rejected():
    with pytest.raises(SwinFormatError, match="header"):
        read_record(BytesIO(b"123"), 1)
    bad = BytesIO(b"\0" * HEADER.size + b"\0" * 8)
    with pytest.raises(SwinFormatError, match="sync"):
        read_record(bad, 1)


def test_record_validates_polarization_and_spectrum_shape():
    record = SwinRecord(257, 1, 2.0, 0, 0, 0, "R", 0, 1, 0, 0, 0, np.zeros(1))
    with pytest.raises(ValueError, match="polarization"):
        write_record(BytesIO(), record)
    invalid_baseline = SwinRecord(
        0, 1, 2.0, 0, 0, 0, "RR", 0, 1, 0, 0, 0, np.zeros(1)
    )
    with pytest.raises(SwinFormatError, match="baseline"):
        write_record(BytesIO(), invalid_baseline)


@pytest.mark.parametrize("named_ids", [False, True])
def test_visibility_sink_adapter(tmp_path, named_ids):
    visibility = VisibilityRecord(
        "T1",
        "T2",
        58948,
        Decimal("25200.5"),
        "0",
        "0",
        3,
        "R",
        "L",
        0.5,
        1.0,
        2.0,
        3.0,
        np.array([1 + 2j], dtype=np.complex64),
    )
    path = tmp_path / "visibility.swin"
    indices = {}
    if named_ids:
        visibility = replace(
            visibility, configuration_id="default", source_id="target"
        )
        indices = {
            "configuration_indices": {"default": 4},
            "source_indices": {"target": 7},
        }
        with pytest.raises(ValueError, match="no SWIN source_id index"):
            from_visibility(
                visibility,
                {"T1": 0, "T2": 1},
                configuration_indices={"default": 4},
                source_indices={},
            )
    with SwinWriter(path, {"T1": 0, "T2": 1}, **indices) as writer:
        writer.write(visibility)
    decoded = next(iter_swin(path, 1))
    assert decoded.baseline == 258
    assert decoded.polarization == "RL"
    assert decoded.frequency_id == 3
    assert decoded.configuration_id == (4 if named_ids else 0)
    assert decoded.source_id == (7 if named_ids else 0)
    np.testing.assert_array_equal(decoded.spectrum, visibility.spectrum)
