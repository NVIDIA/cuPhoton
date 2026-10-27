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

from cuphoton.xmon import (
    Dashboard,
    Finding,
    InferenceConfig,
    Investigator,
    Measurement,
    Monitor,
    MonitorSession,
    Sample,
)
from cuphoton.xmon.dashboard import make_server

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


def test_http_question_access_progress_and_saved_export(
    tmp_path, monkeypatch, controlled
):
    token = "fixture-investigation-access-token"
    monkeypatch.setenv("XMON_TEST_ACCESS", token)
    monitor = Monitor(tmp_path / "run", run_id="interactive")
    monitor.publish(sample())
    session = MonitorSession(
        monitor,
        config=InferenceConfig("http://127.0.0.1:1/v1", "fixture"),
        access_token_env="XMON_TEST_ACCESS",
    )
    dashboard = Dashboard(
        monitor, render_sample=lambda item: f"<p>revision {item.revision}</p>"
    )
    server = make_server(dashboard, port=0, session=session)
    session.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(access, origin=base):
        request = urllib.request.Request(
            base + "/api/ask",
            data=json.dumps(
                {
                    "question": "Explain lag",
                    "sample_ids": ["signal"],
                    "scope": "selected",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "X-XMon-Token": access,
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
            base + "/api/dashboard?sample_id=signal"
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
