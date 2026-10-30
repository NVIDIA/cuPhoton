# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line behavior for xBeam vdif sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from cuphoton.core.cli import run_component
from tests.xbeam.test_vdif_sidecar import (
    _frame,
)


def test_cli_builds_multiple_sidecars_concurrently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sources = []
    for name in ("first.vdif", "second.vdif"):
        source = tmp_path / name
        source.write_bytes(_frame(0) + _frame(1))
        sources.append(source)
    index_dir = tmp_path / "sidecars"

    status = run_component(
        "xbeam",
        [
            "build-vdif-sidecar",
            *(str(source) for source in sources),
            "--index-dir",
            str(index_dir),
            "--backend",
            "posix",
            "--devices",
            "0,1",
            "--json",
        ],
    )

    assert status == 0
    report = __import__("json").loads(capsys.readouterr().out)
    assert report["workers"] == 2
    assert report["devices"] == [0, 1]
    assert report["source_bytes"] == sum(
        item.stat().st_size for item in sources
    )
    assert len(report["sources"]) == 2
    assert all(
        Path(item["index_path"]).is_file() for item in report["sources"]
    )


@pytest.mark.gpu
def test_gpu_cufile_builds_multiple_recordings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    if device_count < 1:
        pytest.skip("no CUDA device")
    sources = []
    for name in ("gpu-first.vdif", "gpu-second.vdif"):
        source = tmp_path / name
        source.write_bytes(b"".join(_frame(value) for value in range(8)))
        sources.append(source)

    status = run_component(
        "xbeam",
        [
            "build-vdif-sidecar",
            *(str(source) for source in sources),
            "--index-dir",
            str(tmp_path / "indexes"),
            "--backend",
            "cufile",
            "--devices",
            "0",
            "--json",
        ],
    )

    assert status == 0
    report = __import__("json").loads(capsys.readouterr().out)
    assert report["workers"] == 1
    assert len(report["sources"]) == 2
