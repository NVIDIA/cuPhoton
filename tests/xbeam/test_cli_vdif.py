# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from cuphoton.core.cli import (
    build_component_cli,
    get_component,
    run_component,
)
from cuphoton.xbeam import __version__


def _frame(*, frame_number: int, thread_id: int) -> bytes:
    payload = bytes((0xAA,)) * 8
    word0 = 0
    word1 = frame_number | (40 << 24)
    word2 = (32 + len(payload)) // 8
    word3 = (ord("A") << 8) | ord("A") | (thread_id << 16) | (1 << 26)
    header = np.asarray((word0, word1, word2, word3, 0, 0, 0, 0), dtype="<u4")
    return header.tobytes() + payload


def test_cli_uses_shared_component_command_discovery() -> None:
    cli = build_component_cli(get_component("xbeam"))

    assert cli.command_names == (
        "correlate",
        "correlate-series",
        "merge-shards",
        "inspect-vdif",
        "build-vdif-sidecar",
    )
    assert get_component("xbeam").module_name == "cuphoton.xbeam.commands"


def test_all_command_parsers_build_without_invariant_collisions() -> None:
    output = StringIO()
    error = StringIO()
    cli = build_component_cli(get_component("xbeam"), out=output, err=error)

    for command_name in cli.command_names:
        assert cli.run(["help", command_name]) == 0

    assert error.getvalue() == ""


def test_cli_help_uses_shared_component_format(capsys) -> None:
    assert run_component("xbeam", ["help"]) == 0
    captured = capsys.readouterr()

    assert "Available subcommands:" in captured.out
    assert "correlate" in captured.out
    assert "inspect-vdif" in captured.out
    assert captured.err == ""


def test_command_help_uses_shared_program_name(capsys) -> None:
    assert run_component("xbeam", ["help", "inspect-vdif"]) == 0
    captured = capsys.readouterr()

    assert "Usage: cuphoton xbeam inspect-vdif" in captured.out
    assert "--probe-count" in captured.out
    assert "--extract-headers" in captured.out
    assert captured.err == ""


def test_cli_reports_cuphoton_distribution_version() -> None:
    output = StringIO()

    assert run_component("xbeam", ["--version"], out=output) == 0
    assert output.getvalue() == f"cuphoton xbeam {__version__}\n"


def test_inspect_vdif_json_does_not_create_an_index_by_default(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sample.vdif"
    source.write_bytes(
        _frame(frame_number=0, thread_id=0)
        + _frame(frame_number=0, thread_id=1)
    )

    status = run_component(
        "xbeam", ["inspect-vdif", str(source), "--probe-count", "2", "--json"]
    )

    assert status == 0
    report = json.loads(capsys.readouterr().out)
    inspection = report["sources"][0]["inspection"]
    assert inspection["frame_count"] == 2
    assert inspection["thread_ids"] == [0, 1]
    assert "index" not in report["sources"][0]


def test_inspect_vdif_rejects_removed_build_index_option(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.vdif"
    source.write_bytes(_frame(frame_number=0, thread_id=0))
    status = run_component(
        "xbeam",
        [
            "inspect-vdif",
            str(source),
            "--build-index",
        ],
    )

    assert status == 1


@pytest.mark.parametrize(
    "arguments",
    (
        ("correlate", "missing.json", "--vdif-io", "cufile-async"),
        ("correlate", "missing.json", "--vdif-io", "cufile-sync"),
        ("correlate", "missing.json", "--xmac-fusion-width", "2"),
        ("correlate", "missing.json", "--preplanned-ingest", "plan.json"),
        ("correlate", "missing.json", "--local-worker-ready", "ready"),
        ("correlate", "missing.json", "--local-worker-release", "go"),
        ("build-vdif-sidecar", "missing.vdif", "--ring-slots", "2"),
        ("build-vdif-sidecar", "missing.vdif", "--chunk-mib", "256"),
        ("build-vdif-sidecar", "missing.vdif", "--workers", "2"),
        ("build-vdif-sidecar", "missing.vdif", "--device", "0"),
    ),
)
def test_cli_rejects_removed_pipeline_options(arguments) -> None:
    assert run_component("xbeam", list(arguments)) != 0


def test_inspect_vdif_can_extract_a_bounded_raw_header_file(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sample.vdif"
    frames = [
        _frame(frame_number=index, thread_id=index % 2) for index in range(4)
    ]
    source.write_bytes(b"".join(frames))
    output = tmp_path / "headers.bin"

    status = run_component(
        "xbeam",
        [
            "inspect-vdif",
            str(source),
            "--extract-headers",
            str(output),
            "--header-first-frame",
            "1",
            "--header-frame-count",
            "2",
            "--json",
        ],
    )

    assert status == 0
    report = json.loads(capsys.readouterr().out)
    extraction = report["sources"][0]["header_extraction"]
    assert extraction["first_frame"] == 1
    assert extraction["frame_count"] == 2
    assert extraction["output_bytes"] == 64
    assert output.read_bytes() == frames[1][:32] + frames[2][:32]


def test_inspect_vdif_requires_output_for_header_bounds(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sample.vdif"
    source.write_bytes(_frame(frame_number=0, thread_id=0))

    status = run_component(
        "xbeam",
        [
            "inspect-vdif",
            str(source),
            "--header-first-frame",
            "1",
        ],
    )

    assert status == 1
    assert "require --extract-headers" in capsys.readouterr().err


def test_inspect_vdif_rejects_nonpositive_probe_count(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sample.vdif"
    source.write_bytes(_frame(frame_number=0, thread_id=0))

    status = run_component(
        "xbeam", ["inspect-vdif", str(source), "--probe-count", "0", "--json"]
    )

    assert status == 1
    assert capsys.readouterr().err.startswith("ERROR:")
