# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line behavior for xBeam ingest planning."""

from cuphoton.core.cli import run_component
from cuphoton.xbeam.adapters.radio.models import (
    dump_job,
)
from tests.xbeam.test_ingest_planning import (
    _job,
)


def test_max_read_guard_rejects_before_cli_output_creation(tmp_path, capsys):
    job_path = tmp_path / "job.json"
    output = tmp_path / "output"
    dump_job(_job(tmp_path), job_path)

    status = run_component(
        "xbeam",
        [
            "correlate",
            str(job_path),
            "--output",
            str(output),
            "--scan",
            "scan",
            "--first-integration",
            "2",
            "--integration-count",
            "1",
            "--assume-ordered",
            "--max-read-bytes",
            "100",
        ],
    )

    assert status == 1
    assert "max_read_bytes" in capsys.readouterr().err
    assert not output.exists()


def test_numpy_decoded_guard_rejects_before_cli_output_creation(
    tmp_path, capsys
):
    job_path = tmp_path / "job.json"
    output = tmp_path / "output"
    dump_job(_job(tmp_path), job_path)

    status = run_component(
        "xbeam",
        [
            "correlate",
            str(job_path),
            "--output",
            str(output),
            "--scan",
            "scan",
            "--first-integration",
            "2",
            "--integration-count",
            "1",
            "--assume-ordered",
            "--max-read-bytes",
            "500",
        ],
    )

    assert status == 1
    assert "decoded NumPy samples" in capsys.readouterr().err
    assert not output.exists()
