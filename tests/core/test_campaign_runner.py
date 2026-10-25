# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "examples/benchmarks/run_campaign.py"
_SPEC = importlib.util.spec_from_file_location("campaign_runner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

_SHA = "a" * 40
_ADAPTER = """
import json, os, pathlib, sys
stage, folder, mode = sys.argv[1:]
p = pathlib.Path(folder)
if stage == 'validate':
    if mode == 'validation-preexisting': sys.exit(0)
    assert (p / 'output.json').read_text() == '[1, 4, 9]'
    value = {'status': 'ok', 'quality_checks': [
        {'name': 'squares', 'passed': True}]}
    if mode == 'validation-source': value['source_sha'] = 'b' * 40
    if mode == 'validation-count': value['completed_work'] = 2
    if mode == 'validation-bool': value['completed_work'] = True
    if mode == 'round-valid':
        value.update(source_sha='a' * 40, completed_work=3)
    if mode == 'validation-mutate':
        original = json.loads((p / 'result.json').read_text())
        original['validator_note'] = 'changed'
        (p / 'result.json').write_text(json.dumps(original))
    (p / 'validation.json').write_text(json.dumps(value))
else:
    print('workload log', flush=True)
    if mode == 'fail':
        sys.exit(7)
    (p / 'output.json').write_text('[1, 4, 9]')
    value = {
        'status': 'ok', 'source_sha': 'a' * 40, 'actual_backend': 'cpu',
        'completed_work': 3,
        'quality_checks': [{'name': 'finite', 'passed': True}],
        'input_identities': {'seed': 1},
        'output_identities': {'path': 'output.json'},
        'warm_rounds': [{'elapsed_seconds': 1.0}, {'elapsed_seconds': 3.0}],
        'pid': os.getpid(),
    }
    if mode == 'backend': value['actual_backend'] = 'unexpected'
    if mode == 'source': value['source_sha'] = 'b' * 40
    if mode == 'count': value['completed_work'] = 2
    if mode == 'bool': value['completed_work'] = True
    if mode == 'quality': value['quality_checks'] = []
    if mode == 'failed-check': value['quality_checks'][0]['passed'] = False
    if mode == 'nan':
        value['warm_rounds'][0]['elapsed_seconds'] = float('nan')
    if mode == 'negative': value['warm_rounds'][0]['elapsed_seconds'] = -1
    if mode == 'zero': value['warm_rounds'][0]['elapsed_seconds'] = 0
    if mode == 'single-shot': value['warm_rounds'] = []
    if mode == 'validation-preexisting':
        (p / 'validation.json').write_text(json.dumps({
            'status': 'ok', 'quality_checks': [
                {'name': 'not independent', 'passed': True}]}))
    if mode in ('round-valid', 'round-total', 'round-partial', 'round-bool'):
        value['warm_rounds'][0]['completed_work'] = 1
        if mode != 'round-partial':
            value['warm_rounds'][1]['completed_work'] = (
                True if mode == 'round-bool'
                else 1 if mode == 'round-total' else 2)
    (p / 'result.json').write_text(json.dumps(value))
"""


def _manifest(tmp_path, *, modes=("ok",), blocks=1, kind="pilot"):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(_ADAPTER)
    manifest = {
        "schema_version": 1,
        "experiment_kind": kind,
        "blocks": blocks,
        "timeout_seconds": 10,
        **{
            key: {"name": "test"}
            for key in (
                "comparison",
                "workload",
                "output_contract",
                "cache",
                "disclosure",
            )
        },
        "treatments": [
            {
                "name": f"treatment-{i}",
                "cwd": str(tmp_path),
                "argv": [
                    sys.executable,
                    str(adapter),
                    "run",
                    "{launch_dir}",
                    mode,
                ],
                "validation_argv": [
                    sys.executable,
                    str(adapter),
                    "validate",
                    "{launch_dir}",
                    mode,
                ],
                "source_sha": _SHA,
                "backend": "cpu",
                "expected_completed_work": 3,
            }
            for i, mode in enumerate(modes)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_dry_run_rotation_fresh_launches_and_nested_rounds(tmp_path):
    path, _ = _manifest(
        tmp_path, modes=("ok", "ok"), blocks=6, kind="headline"
    )
    output = tmp_path / "results"
    plan = runner.prepare(path, output)
    assert not output.exists()
    assert plan == runner.prepare(path, output)
    assert [item["treatment"] for item in plan["schedule"][:4]] == [
        "treatment-0",
        "treatment-1",
        "treatment-1",
        "treatment-0",
    ]
    report = runner.execute(plan)
    assert report["status"] == "ok" and report["headline_eligible"]
    assert len(report["launches"]) == 12
    assert len({row["result"]["pid"] for row in report["launches"]}) == 12
    assert len({row["directory"] for row in report["launches"]}) == 12
    summary = report["treatments"]["treatment-0"]
    assert summary["accepted_launches"] == 6
    assert summary["launch_warm_medians_seconds"] == [2.0] * 6
    assert summary["realized_position_counts"] == {"0": 3, "1": 3}
    assert summary["warm_round_counts"] == [2] * 6
    assert summary["warm_round_count_consistent"]
    for row in report["launches"]:
        assert row["workload"]["elapsed_seconds"] > 0
        assert row["validation"]["elapsed_seconds"] > 0
        assert row["launch_wall_seconds"] >= (
            row["workload"]["elapsed_seconds"]
            + row["validation"]["elapsed_seconds"]
        )
        launch = Path(row["directory"])
        assert row["result_sha256"] == row["result_sha256_after_validation"]
        assert (
            row["result_sha256"]
            == hashlib.sha256(
                (launch / "result.json").read_bytes()
            ).hexdigest()
        )
        assert (
            row["validation_sha256"]
            == hashlib.sha256(
                (launch / "validation.json").read_bytes()
            ).hexdigest()
        )
    with pytest.raises(ValueError, match="already exists"):
        runner.prepare(path, output)
    with pytest.raises(FileExistsError):
        runner.execute(plan)


@pytest.mark.parametrize(
    "mode",
    [
        "fail",
        "backend",
        "source",
        "count",
        "bool",
        "quality",
        "failed-check",
        "nan",
        "negative",
        "zero",
        "validation-source",
        "validation-count",
        "validation-bool",
        "validation-mutate",
        "validation-preexisting",
        "round-total",
        "round-partial",
        "round-bool",
    ],
)
def test_bad_launch_retained_and_other_treatment_still_runs(tmp_path, mode):
    path, _ = _manifest(tmp_path, modes=(mode, "ok"))
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert report["status"] == "failed" and not report["headline_eligible"]
    assert len(report["launches"]) == 2
    bad, good = report["launches"]
    assert bad["status"] == "failed" and bad["errors"]
    assert good["status"] == "ok"
    if mode == "validation-mutate":
        assert "validation modified result.json" in bad["errors"][0]
        assert bad["result_sha256"] != bad["result_sha256_after_validation"]
    if mode == "validation-preexisting":
        assert "before independent validation" in bad["errors"][0]
        assert "validation" not in bad
    if mode in ("validation-source", "validation-count"):
        assert "validation.json:" in bad["errors"][0]
        assert "does not match treatment" in bad["errors"][0]
    assert (
        "workload log"
        in (Path(bad["directory"]) / "workload.stdout.log").read_text()
    )
    summary = report["treatments"]["treatment-0"]
    assert summary["planned_launches"] == summary["failed_launches"] == 1
    assert summary["launch_wall_median_seconds"] is None


def test_missing_independent_validation_rejects_result(tmp_path):
    path, manifest = _manifest(tmp_path)
    manifest["treatments"][0]["validation_argv"] = [
        sys.executable,
        "-c",
        "pass",
    ]
    path.write_text(json.dumps(manifest))
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert report["status"] == "failed"
    assert "validation.json" in report["launches"][0]["errors"][0]


def test_helpers_finishing_during_grace_do_not_reject_launch(
    tmp_path, monkeypatch
):
    path, _ = _manifest(tmp_path)
    # Model a helper that is still shutting down when the leader is reaped.
    probes = iter([True, True, False])
    monkeypatch.setattr(
        runner, "_group_exists", lambda _pid: next(probes, False)
    )

    def unexpected_termination(_process):
        pytest.fail("helper finished within grace; no termination needed")

    monkeypatch.setattr(runner, "_terminate", unexpected_termination)
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert report["status"] == "ok"
    stage = report["launches"][0]["workload"]
    assert stage["group_linger_seconds"] > 0
    assert not stage["leaked_children"]


def test_single_shot_headline_does_not_require_warm_rounds(tmp_path):
    path, _ = _manifest(
        tmp_path, modes=("single-shot",), blocks=6, kind="headline"
    )
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert report["status"] == "ok" and report["headline_eligible"]
    assert report["treatments"]["treatment-0"]["warm_round_counts"] == [0] * 6


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_timeout_terminates_adapter_child_and_retains_failure(tmp_path):
    path, manifest = _manifest(tmp_path)
    child = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    )
    program = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "pathlib.Path(sys.argv[1], 'child.pid').write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    manifest["timeout_seconds"] = 0.4
    manifest["treatments"][0]["argv"] = [
        sys.executable,
        "-c",
        program,
        "{launch_dir}",
    ]
    path.write_text(json.dumps(manifest))
    started = time.monotonic()
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert time.monotonic() - started < 5
    row = report["launches"][0]
    assert row["workload"]["timed_out"] and row["status"] == "failed"
    pid = int((Path(row["directory"]) / "child.pid").read_text())
    proc = Path(f"/proc/{pid}/stat")
    for _ in range(100):
        if not proc.exists() or proc.read_text().split(") ")[1].startswith(
            "Z"
        ):
            break
        time.sleep(0.01)
    else:
        pytest.fail("adapter child remained running after timeout")


def test_headline_rejects_small_pilot_and_boolean_work_count(tmp_path):
    path, manifest = _manifest(tmp_path, kind="headline")
    with pytest.raises(ValueError, match="six blocks"):
        runner.prepare(path, tmp_path / "results")
    manifest["experiment_kind"] = "pilot"
    manifest["treatments"][0]["expected_completed_work"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="positive integer"):
        runner.prepare(path, tmp_path / "results")


def test_unlabelled_manifest_stays_pilot(tmp_path):
    path, manifest = _manifest(tmp_path)
    del manifest["experiment_kind"]
    path.write_text(json.dumps(manifest))
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert report["status"] == "ok"
    assert report["experiment_kind"] == "pilot"
    assert not report["headline_eligible"]


@pytest.mark.parametrize("expected_rounds", [2, 3])
def test_declared_round_count_and_completed_work(tmp_path, expected_rounds):
    path, manifest = _manifest(tmp_path, modes=("round-valid",))
    manifest["workload"]["measured_rounds"] = expected_rounds
    path.write_text(json.dumps(manifest))
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert report["status"] == ("ok" if expected_rounds == 2 else "failed")
    if expected_rounds == 3:
        assert "measured_rounds" in report["launches"][0]["errors"][0]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_child_outliving_successful_parent_is_cleaned_and_rejected(tmp_path):
    path, manifest = _manifest(tmp_path, modes=("ok", "ok"))
    child = (
        "import pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1],'child.ready').touch();time.sleep(30)"
    )
    program = (
        "import pathlib,subprocess,sys,time\n"
        f"p=subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]])\n"
        "root=pathlib.Path(sys.argv[1])\n"
        "(root/'child.pid').write_text(str(p.pid))\n"
        "while not (root/'child.ready').exists():time.sleep(0.01)\n"
    )
    manifest["treatments"][0]["argv"] = [
        sys.executable,
        "-c",
        program,
        "{launch_dir}",
    ]
    path.write_text(json.dumps(manifest))
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    row = report["launches"][0]
    assert row["workload"]["returncode"] == 0
    assert row["workload"]["leaked_children"]
    assert row["status"] == "failed"
    assert "validation" not in row
    pid = int((Path(row["directory"]) / "child.pid").read_text())
    proc = Path(f"/proc/{pid}/stat")
    assert not proc.exists() or proc.read_text().split(") ")[1].startswith(
        "Z"
    )
    if row["workload"]["terminate_incomplete"]:
        assert report["halt_reason"]
        assert len(report["launches"]) == 1
        assert report["treatments"]["treatment-1"]["skipped_launches"] == 1


def test_unreapable_process_has_bounded_cleanup(tmp_path, monkeypatch):
    class StuckProcess:
        pid = 123

        def poll(self):
            return None

    signals = []
    monkeypatch.setattr(
        runner.os, "killpg", lambda pid, sig: signals.append(sig)
    )
    monkeypatch.setattr(runner, "_TERMINATE_GRACE_SECONDS", 0.02)
    started = time.monotonic()
    receipt = runner._terminate(StuckProcess())
    assert time.monotonic() - started < 0.5
    assert signal.SIGTERM in signals and signal.SIGKILL in signals
    assert receipt["returncode"] is None and receipt["terminate_incomplete"]


def test_incomplete_cleanup_halts_remaining_launches(tmp_path, monkeypatch):
    path, _ = _manifest(tmp_path, modes=("ok", "ok"), blocks=2)
    calls = []

    def incomplete(*args):
        calls.append(args)
        return {
            "returncode": None,
            "terminate_incomplete": True,
            "timed_out": True,
            "leaked_children": False,
        }

    monkeypatch.setattr(runner, "_command", incomplete)
    report = runner.execute(runner.prepare(path, tmp_path / "results"))
    assert len(calls) == len(report["launches"]) == 1
    assert report["status"] == "failed" and report["halt_reason"]
    assert not (tmp_path / "results/block-001").exists()
    assert report["treatments"]["treatment-1"]["skipped_launches"] == 2
    assert (
        report["treatments"]["treatment-1"]["launch_wall_median_seconds"]
        is None
    )
    assert report["treatments"]["treatment-1"][
        "realized_position_counts"
    ] == {"0": 0, "1": 0}


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
@pytest.mark.parametrize(
    ("signum", "during_timeout"),
    [
        (signal.SIGINT, False),
        (signal.SIGTERM, False),
        (signal.SIGHUP, False),
        (signal.SIGTERM, True),
    ],
)
def test_interrupt_preserves_receipt_and_stops_before_next_launch(
    tmp_path, signum, during_timeout
):
    path, manifest = _manifest(tmp_path, blocks=2)
    child = (
        "import pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1],'child.ready').touch();time.sleep(30)"
    )
    program = (
        "import os,pathlib,signal,subprocess,sys,time\n"
        "root=pathlib.Path(sys.argv[1])\n"
        "(root/'adapter.pid').write_text(str(os.getpid()))\n"
        "signal.signal(signal.SIGTERM,"
        "lambda *_: (root/'cleanup.started').touch())\n"
        f"p=subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]])\n"
        "while not (root/'child.ready').exists():time.sleep(0.01)\n"
        "(root/'child.pid').write_text(str(p.pid))\n"
        "while True:time.sleep(30)\n"
    )
    if during_timeout:
        manifest["timeout_seconds"] = 0.4
    manifest["treatments"][0]["argv"] = [
        sys.executable,
        "-c",
        program,
        "{launch_dir}",
    ]
    path.write_text(json.dumps(manifest))
    output = tmp_path / "results"
    launch = output / "block-000/treatment-0"
    process = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--manifest",
            str(path),
            "--output-dir",
            str(output),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(200):
            if (launch / "child.pid").exists():
                break
            time.sleep(0.01)
        else:
            pytest.fail("adapter did not start")
        if during_timeout:
            for _ in range(200):
                if (launch / "cleanup.started").exists():
                    break
                time.sleep(0.01)
            else:
                pytest.fail("timeout cleanup did not start")
        cleanup_before = (
            (launch / "cleanup.started").stat().st_mtime_ns
            if during_timeout
            else 0
        )
        process.send_signal(signum)
        for _ in range(200):
            if (launch / "cleanup.started").exists() and (
                launch / "cleanup.started"
            ).stat().st_mtime_ns > cleanup_before:
                break
            time.sleep(0.01)
        else:
            pytest.fail("interrupted cleanup did not start")
        for repeated in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            process.send_signal(repeated)
            time.sleep(0.01)
        assert process.wait(timeout=6) == 128 + signum
        pid = int((launch / "child.pid").read_text())
        proc = Path(f"/proc/{pid}/stat")
        assert not proc.exists() or proc.read_text().split(") ")[
            1
        ].startswith("Z")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if (launch / "adapter.pid").exists():
            try:
                os.killpg(
                    int((launch / "adapter.pid").read_text()), signal.SIGKILL
                )
            except ProcessLookupError:
                pass
    row = json.loads((launch / "launch-result.json").read_text())
    assert row["status"] == "interrupted"
    assert row["workload"]["interrupted"]
    assert row["workload"]["timed_out"] == during_timeout
    assert not (output / "block-001").exists()
