# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from cuphoton.xmonitor import (
    Dashboard,
    Finding,
    InferenceConfig,
    Investigator,
    Measurement,
    Monitor,
    MonitorSession,
    Sample,
)
from cuphoton.xmonitor.dashboard import make_server

pytest.importorskip("httpx")


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.02)
    raise AssertionError("session did not reach expected state")


def sample(revision=1):
    return Sample(
        "signal",
        4,
        {"lag": Measurement(7.0 + revision, "sample")},
        revision=revision,
        findings=(Finding("changed", "Inspect lag."),),
    )


@pytest.fixture
def controlled(monkeypatch):
    observed = {"calls": [], "delay": 0.02}

    async def request(self, evidence, question, instructions):
        observed["calls"].append(evidence)
        await asyncio.sleep(observed["delay"])
        item = evidence["samples"][0]
        measurement = item["measurements"]["lag"]
        return {
            "narrative": "Inspect the changed lag.",
            "citations": [
                {
                    "sample_id": item["id"],
                    "revision": item["revision"],
                    "measurement": "lag",
                    "value": measurement["value"],
                    "unit": measurement["unit"],
                }
            ],
        }

    monkeypatch.setattr(Investigator, "_request", request)
    return observed


def reports(monitor):
    return monitor.snapshot()["reports"]


@pytest.mark.parametrize("prefix", ["/", "/review/run/"])
def test_http_question_access_progress_and_saved_export(
    tmp_path, monkeypatch, controlled, prefix
):
    token = "fixture-investigation-access-token"
    monkeypatch.setenv("XMONITOR_TEST_ACCESS", token)
    monitor = Monitor(tmp_path / "run", run_id="interactive")
    monitor.publish(sample())
    session = MonitorSession(
        monitor,
        config=InferenceConfig("http://127.0.0.1:1/v1", "fixture"),
        access_token_env="XMONITOR_TEST_ACCESS",
    )
    dashboard = Dashboard(
        monitor, render_sample=lambda item: f"<p>revision {item.revision}</p>"
    )
    server = make_server(dashboard, port=0, session=session, prefix=prefix)
    session.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(access, origin=base):
        request = urllib.request.Request(
            base + prefix + "api/ask",
            data=json.dumps(
                {
                    "question": "Explain lag",
                    "sample_ids": ["signal"],
                    "scope": "selected",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "X-XMonitor-Token": access,
                "Origin": origin,
            },
        )
        return urllib.request.urlopen(request)

    try:
        for access, origin, status in [
            ("wrong", base, 401),
            (token, "http://unrelated.invalid", 403),
        ]:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(access, origin)
            assert error.value.code == status
        assert not controlled["calls"]
        with post(token) as response:
            assert response.status == 202
            identifier = json.load(response)["report_id"]
        wait_for(
            lambda: any(
                r["id"] == identifier and r["status"] == "completed"
                for r in reports(monitor)
            )
        )
        with urllib.request.urlopen(
            base + prefix + "api/dashboard?sample_id=signal"
        ) as response:
            display = json.load(response)
        assert display["ui"]["authentication_required"]
        assert display["reports"][0]["events"]
        assert "revision 1" in display["domain_view"]["html"]
        assert token not in json.dumps(display)
        exported = dashboard.export(tmp_path / "offline.html").read_text()
        assert token not in exported and "Inspect the changed lag" in exported
        assert '"live":false' in exported
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        session.close()


def test_automatic_revision_replay_and_recovery_selection(
    tmp_path, controlled
):
    monitor = Monitor(tmp_path / "run", run_id="watch")
    pending = [sample()]

    def poll():
        if pending:
            monitor.publish(pending.pop(0))

    config = InferenceConfig("http://127.0.0.1:1/v1", "fixture")
    session = MonitorSession(
        monitor,
        config=config,
        poll=poll,
        poll_interval=0.05,
        auto_investigate=True,
    )
    session.start()
    try:
        wait_for(
            lambda: any(r["status"] == "completed" for r in reports(monitor))
        )
        pending.append(sample())
        wait_for(lambda: not pending)
        assert len(controlled["calls"]) == 1
        pending.append(sample(2))
        wait_for(
            lambda: (
                len(
                    [
                        r
                        for r in reports(monitor)
                        if r["status"] == "completed"
                    ]
                )
                == 1
                and len(controlled["calls"]) == 2
                and reports(monitor)[-1]["status"] == "completed"
            )
        )
        assert reports(monitor)[0]["status"] == "stale"
    finally:
        session.close()
    assert monitor.snapshot()["collector"] == "stopped"
    # A comparison in another automatic report is evidence, not its trigger.
    monitor.publish(replace(sample(), id="context-only"))
    monitor.save_report(
        {
            "id": "context",
            "status": "completed",
            "trigger": "automatic",
            "selected_evidence": {"signal": 2},
            "evidence": {"signal": 2, "context-only": 1},
        }
    )
    recovered = MonitorSession(
        Monitor.load(monitor.directory), config=config, auto_investigate=True
    )
    recovered.start()
    try:
        wait_for(
            lambda: any(
                r["status"] == "completed"
                and r.get("selected_evidence") == {"context-only": 1}
                for r in reports(recovered.monitor)
            )
        )
        assert len(controlled["calls"]) == 3
    finally:
        recovered.close()


def test_cancellation_and_visible_collection_failure(tmp_path, controlled):
    monitor = Monitor(tmp_path / "run", run_id="cancel")
    monitor.publish(sample())
    controlled["delay"] = 30

    def broken_poll():
        raise OSError("private diagnostic must not be saved")

    session = MonitorSession(
        monitor,
        config=InferenceConfig("http://127.0.0.1:1/v1", "fixture"),
        poll=broken_poll,
        poll_interval=1,
    )
    session.start()
    try:
        first = session.submit("Inspect", ["signal"])
        wait_for(lambda: controlled["calls"])
        second = session.submit("Compare", ["signal"])
        assert session.cancel(second)
        assert session.cancel(first)
        wait_for(
            lambda: all(r["status"] == "cancelled" for r in reports(monitor))
        )
        assert monitor.snapshot()["collector"] == "failed"
        assert monitor.snapshot()["errors"][0]["source_id"] == "live-source"
        assert "private diagnostic" not in json.dumps(monitor.snapshot())
    finally:
        session.close()


def test_report_limit_stops_automatic_investigation_but_keeps_collecting(
    tmp_path, controlled
):
    monitor = Monitor(tmp_path / "run", run_id="report-limit")
    monitor.publish(sample())
    state = monitor.snapshot()
    state.pop("storage_error")
    state["reports"] = [
        {"id": str(index), "status": "completed", "evidence": {}}
        for index in range(1000)
    ]
    monitor._commit(state)
    polls = []

    def poll():
        polls.append(1)
        if len(polls) == 2:
            monitor.publish(replace(sample(), id="later-input"))

    session = MonitorSession(
        monitor,
        config=InferenceConfig("http://127.0.0.1:1/v1", "fixture"),
        poll=poll,
        poll_interval=0.05,
        auto_investigate=True,
    )
    session.start()
    try:
        wait_for(lambda: "later-input" in monitor.snapshot()["samples"])
        assert not session.auto_investigate
        assert not controlled["calls"]
        assert monitor.snapshot()["collector"] == "watching"
        errors = monitor.snapshot()["errors"]
        assert len(errors) == 1
        assert errors[0]["source_id"] == "auto-investigation-stopped"
    finally:
        session.close()


def test_idle_polling_does_not_copy_or_rewrite_full_state(
    tmp_path, monkeypatch
):
    monitor = Monitor(tmp_path / "run", run_id="idle")
    polls = []
    writes = []
    snapshots = []
    original_commit = monitor._commit
    original_snapshot = monitor.snapshot

    def commit(state):
        writes.append(state["collector"])
        return original_commit(state)

    def snapshot():
        snapshots.append(1)
        return original_snapshot()

    monkeypatch.setattr(monitor, "_commit", commit)
    monkeypatch.setattr(monitor, "snapshot", snapshot)
    session = MonitorSession(
        monitor, poll=lambda: polls.append(1), poll_interval=0.05
    )
    session.start()
    try:
        wait_for(lambda: len(polls) >= 2)
        counts = len(writes), len(snapshots)
        first = len(polls)
        wait_for(lambda: len(polls) >= first + 2)
        assert (len(writes), len(snapshots)) == counts
    finally:
        session.close()


@pytest.mark.parametrize(
    ("method", "path", "headers", "body", "expected"),
    [
        ("GET", "/", {"Host": "attacker.example"}, None, 421),
        ("GET", "/api/state", {"Host": "attacker.example"}, None, 421),
        ("GET", "/health", {"Host": "user@localhost"}, None, 421),
        (
            "POST",
            "/api/ask",
            {
                "Host": "attacker.example",
                "Origin": "http://attacker.example",
                "Sec-Fetch-Site": "same-origin",
            },
            b"{}",
            421,
        ),
        ("POST", "/api/ask", {"Sec-Fetch-Site": "cross-site"}, b"{}", 403),
        ("POST", "/api/ask", {"Sec-Fetch-Site": "same-site"}, b"{}", 403),
        ("POST", "/api/ask", {"Content-Type": "text/plain"}, b"{}", 415),
        ("POST", "/api/ask", {}, b" " * 16385, 400),
        ("POST", "/api/ask", {}, b'{"unknown":true}', 400),
    ],
)
def test_http_request_guards(
    tmp_path, controlled, method, path, headers, body, expected
):
    import http.client

    monitor = Monitor(tmp_path / "run", run_id="guards")
    monitor.publish(sample())
    session = MonitorSession(
        monitor, config=InferenceConfig("http://localhost:1/v1", "fixture")
    )
    server = make_server(Dashboard(monitor), port=0, session=session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    try:
        connection.request(
            method,
            path,
            body,
            {"Content-Type": "application/json", **headers},
        )
        response = connection.getresponse()
        assert response.status == expected
        response.read()
        assert not controlled["calls"]
        assert not reports(monitor)
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
        session.close()


def test_explicit_host_allowlist_and_bound_loopback_port(tmp_path):
    import http.client

    monitor = Monitor(tmp_path / "run", run_id="hosts")
    server = make_server(
        Dashboard(monitor), port=0, allowed_hosts=["monitor.example:8443"]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for host in (
            f"localhost:{server.server_port}",
            f"[::1]:{server.server_port}",
            "monitor.example:8443",
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port
            )
            try:
                connection.request("GET", "/health", headers={"Host": host})
                response = connection.getresponse()
                assert response.status == 200
                response.read()
            finally:
                connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_remote_investigation_and_token_requirements(tmp_path, monkeypatch):
    monitor = Monitor(tmp_path / "run", run_id="access")
    config = InferenceConfig("http://localhost:1/v1", "fixture")
    session = MonitorSession(monitor, config=config)
    with pytest.raises(ValueError, match="non-loopback investigation"):
        make_server(
            Dashboard(monitor), host="0.0.0.0", port=0, session=session
        )
    monkeypatch.setenv("XMONITOR_SHORT_TOKEN", "x" * 23)
    with pytest.raises(ValueError, match="at least 24"):
        MonitorSession(
            monitor, config=config, access_token_env="XMONITOR_SHORT_TOKEN"
        )


@pytest.mark.parametrize("prerequisite", ["available", "httpx", "nooa"])
def test_ask_prints_detached_report_without_writing_run(
    tmp_path, monkeypatch, capsys, controlled, prerequisite
):
    from cuphoton.core.cli import main
    from cuphoton.xmonitor import inference
    from cuphoton.xmonitor.nooa_backend import NooaError

    monitor = Monitor(tmp_path / "run", run_id="owned")
    monitor.publish(sample())
    monitor._collector("watching")
    before = (monitor.directory / "state.json").read_bytes()
    config = tmp_path / "inference.toml"
    config.write_text(
        '[inference]\nbase_url="http://localhost:1/v1"\nmodel="fixture"\n'
        + ('backend="nooa"\n' if prerequisite == "nooa" else "")
    )
    if prerequisite == "httpx":
        monkeypatch.setattr(
            inference.importlib.util, "find_spec", lambda name: None
        )
    elif prerequisite == "nooa":

        def unavailable():
            raise NooaError("NOOA requires Linux bubblewrap")

        monkeypatch.setattr(inference, "_command", unavailable)
    result = main(
        [
            "xmonitor",
            "ask",
            "--state",
            str(monitor.directory),
            "--config",
            str(config),
            "--sample",
            "signal",
            "--question",
            "Explain lag",
        ]
    )
    output = capsys.readouterr()
    assert (monitor.directory / "state.json").read_bytes() == before
    assert monitor.snapshot()["reports"] == []
    assert list(
        Monitor.load(monitor.directory, recover=False).snapshot()["samples"]
    ) == ["signal"]
    if prerequisite == "available":
        assert result == 0
        assert json.loads(output.out)["status"] == "completed"
    else:
        assert result != 0
        assert not controlled["calls"]


def test_ask_preserves_samples_published_during_inference(
    tmp_path, monkeypatch, capsys
):
    from cuphoton.core.cli import main

    writer = Monitor(tmp_path / "run", run_id="live")
    writer.publish(sample())
    config = tmp_path / "inference.toml"
    config.write_text(
        '[inference]\nbase_url="http://localhost:1/v1"\nmodel="fixture"\n'
    )

    async def request(self, evidence, question, instructions):
        writer.publish(replace(sample(), id="arrived-during-ask"))
        return {
            "narrative": "Recorded lag.",
            "citations": [
                {
                    "sample_id": "signal",
                    "revision": 1,
                    "measurement": "lag",
                    "value": 8.0,
                    "unit": "sample",
                }
            ],
        }

    monkeypatch.setattr(Investigator, "_request", request)
    assert (
        main(
            [
                "xmonitor",
                "ask",
                "--state",
                str(writer.directory),
                "--config",
                str(config),
                "--sample",
                "signal",
                "--question",
                "Explain lag",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    saved = Monitor.load(writer.directory, recover=False).snapshot()
    assert set(saved["samples"]) == {"signal", "arrived-during-ask"}
    assert saved["reports"] == []
