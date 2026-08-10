# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cuphoton.xdr.reader import GpuCompImageReader, _validate_comp_geometry


def test_comp_reader_keeps_input_alive_until_non_null_stream_finishes(
    monkeypatch,
):
    class Stream:
        def __init__(self):
            self.synchronize_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def synchronize(self):
            self.synchronize_calls += 1

    class Handle:
        def wait(self):
            return "device-input", "relative-offsets"

    class Loader:
        def load_tiles_async(self, offsets, lengths):
            return Handle()

    plan = {
        "sel_abs_offsets": "absolute-offsets",
        "sel_lengths": "lengths",
    }
    stream = Stream()
    keepalives = []
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(Stream=SimpleNamespace(null=None))
        ),
    )
    monkeypatch.setattr(
        GpuCompImageReader, "prepare_plan", lambda self, section=None: plan
    )

    def decode(d_concat, rel_offsets, plan, **kwargs):
        kwargs["keepalive"].append(d_concat)
        keepalives.append(kwargs["keepalive"])
        return "decoded"

    monkeypatch.setattr(
        GpuCompImageReader,
        "decode_from_device_heap",
        staticmethod(decode),
    )
    reader = object.__new__(GpuCompImageReader)

    result = reader.read(loader=Loader(), stream=stream)

    assert result == "decoded"
    assert keepalives == [["device-input"]]
    assert stream.synchronize_calls == 1


@pytest.mark.parametrize(
    ("data_shape", "tile_shape"),
    [
        ((100, 100), (0, 64)),
        ((100, 100), (64, 0)),
        ((100, 100), (-1, 64)),
        ((0, 100), (64, 64)),
        ((100, -5), (64, 64)),
    ],
)
def test_comp_geometry_rejects_nonpositive_dims(data_shape, tile_shape):
    with pytest.raises(ValueError, match="must be positive"):
        _validate_comp_geometry(data_shape, tile_shape)


def test_comp_geometry_accepts_positive_dims_case():
    _validate_comp_geometry((100, 100), (64, 64))
