# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed-wheel probe process handling without a GPU."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "wheel_test_installed",
    Path(__file__).resolve().parents[1] / "scripts/wheels/test_installed.py",
)
installed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installed)


def test_native_probes_share_remaining_budget(monkeypatch):
    times = iter((1000, 1200))
    monkeypatch.setattr(installed.time, "monotonic", lambda: next(times))
    timeouts = []

    def run(command, **kwargs):
        timeouts.append(
            (kwargs["timeout"], int(command[command.index("--timeout") + 1]))
        )
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"ok": True, "checks": []})
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(installed.subprocess, "run", run)
    for probe in ("imaging", "xfit-float32-difference"):
        installed.run_native_probe(probe, {"checks": []}, deadline=1300)

    assert timeouts == [(180, 175), (90, 85)]


@pytest.mark.parametrize("remaining", [0, 15])
def test_native_probe_does_not_start_without_budget(monkeypatch, remaining):
    monkeypatch.setattr(installed.time, "monotonic", lambda: 1000)

    def unexpected_run(*args, **kwargs):
        pytest.fail("An exhausted probe must not launch a subprocess")

    monkeypatch.setattr(installed.subprocess, "run", unexpected_run)
    with pytest.raises(RuntimeError, match="imaging"):
        installed.run_native_probe(
            "imaging", {"checks": []}, deadline=1000 + remaining
        )


@pytest.mark.parametrize("as_bytes", [False, True])
def test_native_probe_timeout_retains_output(monkeypatch, as_bytes):
    monkeypatch.setattr(installed.time, "monotonic", lambda: 1000)
    stdout = "Completed extension load\n"
    stderr = "Thread traceback from child\n"
    expired = subprocess.TimeoutExpired(
        ["python"],
        90,
        output=stdout.encode() if as_bytes else stdout,
        stderr=stderr.encode() if as_bytes else stderr,
    )

    def timeout(*args, **kwargs):
        raise expired

    monkeypatch.setattr(installed.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="imaging") as caught:
        installed.run_native_probe("imaging", {"checks": []}, deadline=1100)

    assert stdout in str(caught.value)
    assert stderr in str(caught.value)
    assert caught.value.__cause__ is expired
