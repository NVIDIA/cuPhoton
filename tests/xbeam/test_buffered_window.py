# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from decimal import Decimal
from io import StringIO

import numpy as np
import pytest

from cuphoton.xbeam import (
    BufferedReaderGeometry,
    ConfigurationError,
    JobLoadError,
    job_from_dict,
    job_to_dict,
    validate_job,
)
from cuphoton.xbeam.adapters.radio.difx_job import _buffered_reader_geometry
from cuphoton.xbeam.buffered_reader.geometry import (
    BufferedWindowGeometry,
    frame_edge_underflows,
)
from cuphoton.xbeam.job_text import parse_job_text
from tests.xbeam.test_models_validation import valid_job


def test_reader_settings_survive_normalization():
    reader = BufferedReaderGeometry(24, 8, 20)
    job = replace(valid_job(), buffered_reader=reader)
    validate_job(job)
    assert job_from_dict(job_to_dict(job)) == job
    assert "buffered_reader" not in job_to_dict(valid_job())
    assert job_from_dict(job_to_dict(valid_job())).buffered_reader is None


def test_text_adapter_preserves_reader_settings():
    doc = parse_job_text(
        StringIO(
            "DATA BUFFER FACTOR: 24\nNUM DATA SEGMENTS: 8\n"
            "NUM BUFFERED FFTS: 20\n"
        )
    )
    assert _buffered_reader_geometry(doc) == BufferedReaderGeometry(24, 8, 20)
    assert _buffered_reader_geometry(parse_job_text(StringIO(""))) is None


@pytest.mark.parametrize("raw", ["0", "-1", "2.5", "abc"])
def test_text_adapter_rejects_invalid_reader_geometry(raw):
    doc = parse_job_text(
        StringIO(
            f"DATA BUFFER FACTOR: {raw}\nNUM DATA SEGMENTS: 8\n"
            "NUM BUFFERED FFTS: 20\n"
        )
    )
    with pytest.raises(JobLoadError, match="DATA BUFFER FACTOR"):
        _buffered_reader_geometry(doc)


def test_partial_reader_geometry_is_not_defaulted():
    doc = parse_job_text(StringIO("DATA BUFFER FACTOR: 24\n"))
    with pytest.raises(JobLoadError, match="NUM DATA SEGMENTS"):
        _buffered_reader_geometry(doc)


@pytest.mark.parametrize("value", [True, 0, -1, 2.5, "24"])
def test_normalized_reader_geometry_requires_positive_integers(value):
    job = replace(
        valid_job(), buffered_reader=BufferedReaderGeometry(value, 8, 20)
    )
    with pytest.raises(
        ConfigurationError, match="buffered_reader.buffer_factor"
    ):
        validate_job(job)
    with pytest.raises(JobLoadError, match="buffered_reader.buffer_factor"):
        job_from_dict(job_to_dict(job))


@pytest.mark.parametrize(
    "factor,segments,expected", [(24, 8, 1206), (16, 8, 804)]
)
def test_allocation_geometry_is_derived(factor, segments, expected):
    geometry = BufferedWindowGeometry.from_reader(
        BufferedReaderGeometry(factor, segments, 20),
        samples_per_frame=32768,
        fft_samples=262144,
        sample_rate=4096000000,
        subintegration_s=Decimal("0.0032"),
    )
    assert geometry.frames_per_fft == 8
    assert geometry.ffts_per_request == 50
    assert geometry.allocation_frames == expected
    assert geometry.ring_segments == segments


def test_nonintegral_buffer_geometry_is_rejected():
    with pytest.raises(ConfigurationError, match="divisible"):
        BufferedWindowGeometry.from_reader(
            BufferedReaderGeometry(25, 8, 20),
            samples_per_frame=32768,
            fft_samples=262144,
            sample_rate=4096000000,
            subintegration_s=Decimal("0.0032"),
        )


@pytest.mark.parametrize(
    "origin", [0, 43200 * 4096000000, 86400 * 4096000000]
)
def test_frame_edge_rule_is_independent_of_absolute_origin(origin):
    delay_samples = np.asarray([0.0, 2.0, 2.5, 3.0, 4.125, 4.25])
    selected = frame_edge_underflows(
        np.arange(6),
        delay_samples / 4096000000,
        np.full(6, origin, dtype=np.int64),
        sample_rate=4096000000,
        samples_per_frame=32768,
        minimum_samples=-4.125,
        maximum_samples=-2.5,
    )
    assert selected == (2, 3, 4)
