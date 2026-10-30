# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line behavior for xBeam sharding."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import cuphoton.xbeam.commands as commands_module
from cuphoton.core.cli import collect_invariants, run_component
from cuphoton.xbeam.commands import (
    _local_worker_argv,
    _parse_devices,
    _resolve_local_devices,
    _run_local_shards,
    _terminate_local_workers,
)
from cuphoton.xbeam.sharding import (
    load_ordered_shard_outputs,
    plan_time_shards,
)
from cuphoton.xbeam.windowing import CorrelationWindow


class _FakeLocalProcess:
    def __init__(
        self,
        *,
        pid: int,
        returncode: int | None,
        stubborn: bool = False,
        interrupt_once: bool = False,
        ignore_kill: bool = False,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self.stubborn = stubborn
        self.interrupt_once = interrupt_once
        self.ignore_kill = ignore_kill
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        if not self.ignore_kill:
            self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.interrupt_once:
            self.interrupt_once = False
            raise KeyboardInterrupt
        if self.stubborn and self.returncode is None:
            raise commands_module.subprocess.TimeoutExpired("worker", timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def _local_command() -> SimpleNamespace:
    return SimpleNamespace(
        assume_ordered=True,
        index_dir=None,
        max_read_bytes=1 << 20,
        vdif_io="posix",
        batch_blocks=None,
        ingest_window_integrations=None,
        error=StringIO(),
    )


def test_local_device_parser_and_worker_contract(tmp_path: Path) -> None:
    assert _parse_devices("0, GPU-acde") == ("0", "GPU-acde")
    with pytest.raises(ValueError, match="duplicates"):
        _parse_devices("0,0")
    assert _resolve_local_devices(("0", "1"), "4,5") == ("4", "5")
    assert _resolve_local_devices(("GPU-b",), "GPU-a,GPU-b") == ("GPU-b",)
    with pytest.raises(ValueError, match="outside inherited"):
        _resolve_local_devices(("2",), "4,5")

    command = SimpleNamespace(
        max_read_bytes=1234,
        vdif_io="posix",
        batch_blocks=32,
        ingest_window_integrations=5,
        assume_ordered=True,
        index_dir=tmp_path / "indexes",
    )
    arguments = _local_worker_argv(
        command,
        job_path=tmp_path / "job.json",
        output=tmp_path / "shard",
        profile="analytic-v1",
        parent_window=CorrelationWindow("scan", 10, 20),
        shard_count=4,
        shard_index=2,
    )

    assert arguments[1:5] == ["-m", "cuphoton", "xbeam", "correlate"]
    assert "--shard-count=4" in arguments
    assert "--shard-index=2" in arguments
    assert "--integration-count=20" in arguments
    assert "--assume-ordered" in arguments


def test_source_snapshot_numbers_sources_within_each_datastream() -> None:
    def source(datastream_id: int, path: str) -> dict[str, object]:
        return {
            "datastream_id": datastream_id,
            "path": path,
            "size_bytes": 40,
            "mtime_ns": 123,
            "device": 1,
            "inode": 2,
        }

    snapshot = commands_module._source_snapshot(
        {
            "sources": [
                source(2, "/resolved/first.vdif"),
                source(1, "/resolved/only.vdif"),
                source(2, "/resolved/second.vdif"),
            ]
        }
    )

    assert [
        (item["datastream_id"], item["source_index"]) for item in snapshot
    ] == [(1, 0), (2, 0), (2, 1)]


@pytest.mark.parametrize(
    ("output_format", "expected"),
    (
        (commands_module.OutputFormat.NATIVE, "records"),
        (commands_module.OutputFormat.SWIN, "bytes"),
    ),
)
def test_assembled_metric_unit_is_explicit(output_format, expected) -> None:
    job = SimpleNamespace(output=SimpleNamespace(format=output_format))

    assert commands_module._assembled_metric_unit(job) == expected


def test_local_coordinator_launches_and_assembles_successful_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processes = [
        _FakeLocalProcess(pid=100, returncode=0),
        _FakeLocalProcess(pid=101, returncode=0),
    ]
    launches = []
    merges = []

    def launch(arguments, *, env):
        launches.append((arguments, env))
        return processes[len(launches) - 1]

    def merge(job, shard_outputs, output):
        merges.append((job, tuple(shard_outputs), output))
        return tuple(shard_outputs), 12

    monkeypatch.setattr(commands_module.subprocess, "Popen", launch)
    monkeypatch.setattr(commands_module, "_merge_shard_set", merge)
    monkeypatch.setattr(
        commands_module, "_job_fingerprint", lambda _job: "job"
    )
    output = tmp_path / "output"
    job = SimpleNamespace(
        datastreams=(),
        output=SimpleNamespace(format=commands_module.OutputFormat.NATIVE),
    )

    _run_local_shards(
        command=_local_command(),
        job_path=tmp_path / "job.json",
        job=job,
        profile="analytic-v1",
        parent_window=CorrelationWindow("scan", 0, 2),
        devices=("4", "5"),
        output=output,
    )

    assert [item[1]["CUDA_VISIBLE_DEVICES"] for item in launches] == [
        "4",
        "5",
    ]
    assert [
        next(
            argument.removeprefix("--shard-index=")
            for argument in item[0]
            if argument.startswith("--shard-index=")
        )
        for item in launches
    ] == ["0", "1"]
    assert len(merges) == 1
    manifest = json.loads(
        (output / "ingest-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["assembled_metric"] == 12
    assert manifest["assembled_metric_unit"] == "records"
    assert [worker["status"] for worker in manifest["workers"]] == [
        "complete",
        "complete",
    ]


def test_local_coordinator_shares_gds_read_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processes = [
        _FakeLocalProcess(pid=100, returncode=0),
        _FakeLocalProcess(pid=101, returncode=0),
    ]
    launches = []

    def launch(arguments, *, env, pass_fds):
        launches.append((arguments, env, pass_fds))
        return processes[len(launches) - 1]

    monkeypatch.setenv("CUPHOTON_XBEAM_GLOBAL_GDS_READS", "3")
    monkeypatch.setattr(commands_module.subprocess, "Popen", launch)
    monkeypatch.setattr(
        commands_module,
        "_merge_shard_set",
        lambda _job, shard_outputs, _output: (tuple(shard_outputs), 12),
    )
    monkeypatch.setattr(
        commands_module, "_job_fingerprint", lambda _job: "job"
    )
    command = _local_command()
    command.vdif_io = "cufile"
    output = tmp_path / "output"
    job = SimpleNamespace(
        datastreams=(),
        output=SimpleNamespace(format=commands_module.OutputFormat.NATIVE),
    )

    _run_local_shards(
        command=command,
        job_path=tmp_path / "job.json",
        job=job,
        profile="analytic-v1",
        parent_window=CorrelationWindow("scan", 0, 2),
        devices=("4", "5"),
        output=output,
    )

    assert len(launches) == 2
    assert launches[0][2] == launches[1][2]
    assert len(launches[0][2]) == 2
    for _arguments, environment, pass_fds in launches:
        assert environment["_CUPHOTON_XBEAM_CUFILE_READ_TOKEN_FD"] == str(
            pass_fds[0]
        )
        assert environment["_CUPHOTON_XBEAM_CUFILE_RETURN_TOKEN_FD"] == str(
            pass_fds[1]
        )
    manifest = json.loads(
        (output / "ingest-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["global_gds_reads"] == 3


def test_local_coordinator_output_can_be_an_outer_time_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.vdif"
    source.write_bytes(b"source")
    job = SimpleNamespace(
        datastreams=(SimpleNamespace(identifier=7, source_files=(source,)),),
        output=SimpleNamespace(format=commands_module.OutputFormat.NATIVE),
    )
    parent = CorrelationWindow("scan", 0, 4)
    outer_shards = plan_time_shards(parent, 2)
    next_pid = 100

    def launch(_arguments, *, env):
        nonlocal next_pid
        process = _FakeLocalProcess(pid=next_pid, returncode=0)
        next_pid += 1
        return process

    monkeypatch.setattr(commands_module.subprocess, "Popen", launch)
    monkeypatch.setattr(
        commands_module,
        "_merge_shard_set",
        lambda _job, shard_outputs, _output: (tuple(shard_outputs), 12),
    )
    monkeypatch.setattr(
        commands_module, "_job_fingerprint", lambda _job: "job"
    )
    outputs = []
    for outer in outer_shards:
        output = tmp_path / f"outer-{outer.index}"
        _run_local_shards(
            command=_local_command(),
            job_path=tmp_path / "job.json",
            job=job,
            profile="analytic-v1",
            parent_window=outer.window,
            distributed_shard=outer,
            devices=(str(outer.index),),
            output=output,
        )
        outputs.append(output)

    ordered, actual = load_ordered_shard_outputs(tuple(reversed(outputs)))

    assert ordered == tuple(outputs)
    assert actual == outer_shards


def test_local_worker_cleanup_does_not_hang_after_kill_timeout() -> None:
    process = _FakeLocalProcess(
        pid=100,
        returncode=None,
        stubborn=True,
        ignore_kill=True,
    )

    outcomes = _terminate_local_workers({0: process}, timeout_s=0)

    assert outcomes == {0: (None, "unreaped")}
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert all(timeout is not None for timeout in process.wait_timeouts)


def test_local_worker_cleanup_kills_every_worker_before_reaping() -> None:
    events = []

    class TrackedProcess(_FakeLocalProcess):
        def kill(self) -> None:
            events.append(("kill", self.pid))
            super().kill()

        def wait(self, timeout: float | None = None) -> int:
            if self.kill_calls:
                events.append(("reap", self.pid))
            return super().wait(timeout)

    processes = {
        index: TrackedProcess(pid=100 + index, returncode=None, stubborn=True)
        for index in range(2)
    }

    outcomes = _terminate_local_workers(processes, timeout_s=0)

    assert events[:2] == [("kill", 100), ("kill", 101)]
    assert outcomes == {0: (-9, "killed"), 1: (-9, "killed")}


@pytest.mark.parametrize("explicit_options", [False, True])
def test_every_worker_option_round_trips_through_cli(
    tmp_path, monkeypatch, explicit_options
):
    job_path = tmp_path / "job.json"
    job_path.write_text("{}")
    output = tmp_path / "parent"
    worker_output = output / "shard-2"
    parsed = []
    monkeypatch.setattr(
        commands_module.CorrelateCommand,
        "run",
        lambda self: parsed.append(self),
    )
    arguments = [
        "correlate",
        str(job_path),
        "--output",
        str(output),
        "--backend",
        "cupy",
        "--devices",
        "0,1,2,3",
        "--scan",
        "fixture-scan",
        "--first-integration",
        "10",
        "--integration-count",
        "20",
    ]
    if explicit_options:
        arguments.extend(
            [
                "--profile",
                "interop-real-usb-v1",
                "--batch-blocks",
                "17",
                "--ingest-window-integrations",
                "3",
                "--max-read-bytes",
                "12345",
                "--assume-ordered",
                "--index-dir",
                str(tmp_path / "indexes"),
                "--vdif-io",
                "cufile",
                "--autocorrelation-mode",
                "truncation",
                "--subintegration-tiling",
                "off",
                "--max-tile-subintegrations",
                "7",
                "--vdif-upload-workers",
                "2",
            ]
        )
    assert run_component("xbeam", arguments) == 0
    parent = parsed[0]
    resolved_profile = parent.profile or "analytic-v1"
    worker_argv = _local_worker_argv(
        parent,
        job_path=job_path,
        output=worker_output,
        profile=resolved_profile,
        parent_window=CorrelationWindow(
            parent.scan, parent.first_integration, parent.integration_count
        ),
        shard_count=4,
        shard_index=2,
    )
    assert worker_argv[1:5] == ["-m", "cuphoton", "xbeam", "correlate"]
    assert run_component("xbeam", worker_argv[4:]) == 0
    worker = parsed[1]
    overrides = {
        "output_path": worker_output,
        "devices": None,
        "shard_count": 4,
        "shard_index": 2,
        "profile": resolved_profile,
    }
    for name, _ in collect_invariants(commands_module.CorrelateCommand):
        assert getattr(worker, name) == overrides.get(
            name, getattr(parent, name)
        ), name
