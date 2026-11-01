# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line behavior for xBeam engine."""

import json
from io import StringIO

import numpy as np
import pytest

import cuphoton.xbeam.commands as commands_module
from cuphoton.core.cli import run_component
from cuphoton.xbeam.adapters.radio.models import (
    dump_job,
)
from cuphoton.xbeam.adapters.radio.output import read_xbeam_v1
from cuphoton.xbeam.commands import _split_window
from cuphoton.xbeam.windowing import CorrelationWindow
from tests.xbeam.test_engine import (
    _job,
)


def test_cli_can_bound_ingest_windows_in_one_process(
    tmp_path, capsys, monkeypatch
):
    job_path = tmp_path / "job.json"
    output = tmp_path / "output"
    dump_job(_job(tmp_path), job_path)

    error = StringIO()
    prepare_calls = []
    real_prepare_correlation = commands_module.prepare_correlation

    def prepare_with_call_capture(*args, **kwargs):
        prepared = real_prepare_correlation(*args, **kwargs)
        prepare_calls.append(
            (kwargs.get("recovery_plans", "absent"), prepared.recovery_plans)
        )
        return prepared

    monkeypatch.setattr(
        commands_module, "prepare_correlation", prepare_with_call_capture
    )
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
            "0",
            "--integration-count",
            "2",
            "--ingest-window-integrations",
            "1",
            "--autocorrelation-mode",
            "truncation",
            "--assume-ordered",
        ],
        err=error,
    )

    assert status == 0, error.getvalue() + capsys.readouterr().err
    assert "correlating ingest window 1/2" in error.getvalue()
    assert "completed ingest window 2/2" in error.getvalue()
    assert len(read_xbeam_v1(output)) == 6
    manifest = json.loads((output / "ingest-manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["autocorrelation_mode"] == "truncation"
    assert manifest["ingest_window_integrations"] == 1
    assert [
        item["window"]["first_integration"]
        for item in manifest["ingest_windows"]
    ] == [0, 1]
    assert [
        item["window"]["integration_count"]
        for item in manifest["ingest_windows"]
    ] == [1, 1]
    assert manifest["ingest"] is None
    assert prepare_calls[0][0] == "absent"
    assert prepare_calls[1][0] is prepare_calls[0][1]


def test_split_window_preserves_remainder() -> None:
    windows = _split_window(CorrelationWindow("scan", 4, 5), 2)

    assert windows == (
        CorrelationWindow("scan", 4, 2),
        CorrelationWindow("scan", 6, 2),
        CorrelationWindow("scan", 8, 1),
    )


def test_cli_writes_native_output(tmp_path):
    job = _job(tmp_path)
    job_path = tmp_path / "job.json"
    dump_job(job, job_path)
    output = tmp_path / "output"
    assert (
        run_component(
            "xbeam",
            [
                "correlate",
                str(job_path),
                "--output",
                str(output),
                "--backend",
                "numpy",
                "--batch-blocks",
                "1",
            ],
        )
        == 0
    )
    records = read_xbeam_v1(output)
    assert len(records) == 6
    np.testing.assert_allclose(records[0].spectrum, [8, 0, 0, 0])
    manifest = json.loads((output / "ingest-manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["estimated_read_bytes"] == 80
    assert manifest["ingest"]["mode"] == "binary-sidecar"
    assert manifest["ingest"]["bounded"] is True


@pytest.mark.parametrize(
    "exception", [RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_interrupted_correlation_records_failure(
    tmp_path, monkeypatch, exception
):
    job_path = tmp_path / "job.json"
    dump_job(_job(tmp_path), job_path)
    output = tmp_path / "output"
    failure = exception("stop fixture")

    def interrupted(records, sink):
        # Interrupt after a real record has reached the actual output writer.
        sink.write(next(iter(records)))
        raise failure

    monkeypatch.setattr(commands_module, "_write_records", interrupted)
    command = commands_module.CorrelateCommand()
    command.job = str(job_path)
    command.output_path = output
    with pytest.raises(exception) as caught:
        command._run_correlation()
    assert caught.value is failure
    manifest = json.loads((output / "ingest-manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["failure"] == {
        "type": exception.__name__,
        "message": "stop fixture",
    }
    assert "output_integrity" not in manifest
