# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest


def test_public_batch_functions_import_without_astropy_loaded():
    code = """
import sys
import cuphoton.xdr as module
assert callable(module.batch_to_device)
assert callable(module.batch_to_device_stream)
raise SystemExit("astropy" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_batch_api_keeps_existing_keyword_surface():
    from cuphoton import xdr

    expected = (
        "paths",
        "hdu_indices",
        "out",
        "stream",
        "section",
        "parallel",
        "prefetch_depth",
        "decode_batch_files",
        "batch_queue_depth",
        "native_read_threads",
        "native_plan_threads",
        "native_batcher",
    )
    actual = tuple(inspect.signature(xdr.batch_to_device).parameters)

    assert actual == expected


def test_parallel_batch_delegates_to_stream_frontend(monkeypatch):
    from cuphoton.xdr import convenience

    calls = []

    def fake_stream(paths, **kwargs):
        calls.append((paths, kwargs))
        return ("ok",)

    monkeypatch.setattr(convenience, "batch_to_device_stream", fake_stream)

    result = convenience.batch_to_device(
        ["a.fits", "b.fits"],
        hdu_indices=(1, 2),
        out="out",
        stream="stream",
        section=(slice(0, 1), slice(0, 1)),
        parallel=True,
        prefetch_depth=3,
        decode_batch_files=4,
        batch_queue_depth=5,
        native_read_threads=6,
        native_plan_threads=7,
        native_batcher=True,
    )

    assert result == ("ok",)
    assert calls == [
        (
            ["a.fits", "b.fits"],
            {
                "hdu_indices": (1, 2),
                "out": "out",
                "prefetch_depth": 3,
                "decode_batch_files": 4,
                "batch_queue_depth": 5,
                "native_read_threads": 6,
                "native_plan_threads": 7,
                "native_batcher": True,
                "section": (slice(0, 1), slice(0, 1)),
                "stream": "stream",
            },
        )
    ]


@pytest.mark.parametrize(
    "message",
    [
        "RICE_1 is not a supported GPU compression format",
        "HCOMPRESS_1 is not supported by the GPU path",
    ],
)
def test_unsupported_gpu_compression_errors_become_not_implemented(
    message,
):
    from cuphoton.xdr.planning import normalize_planning_error

    with pytest.raises(
        NotImplementedError, match="not supported on the GPU path"
    ):
        normalize_planning_error(RuntimeError(message))
