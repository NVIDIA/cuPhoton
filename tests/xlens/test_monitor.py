# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import threading
import urllib.request
from dataclasses import replace

import pytest

from cuphoton.xlens import Dashboard, Finding, Measurement, Monitor, Sample
from cuphoton.xlens.dashboard import make_server
from cuphoton.xlens.quickstart import quickstart


def sample(value=3.0, revision=1):
    return Sample(
        "window-a",
        20,
        {"level": Measurement(value, "arb", coverage=0.8)},
        revision=revision,
        observed_start=0.4,
        observed_end=0.9,
        findings=(Finding("level", "Inspect measured level."),),
    )


def test_revision_duplicate_recovery_and_detached_evidence(tmp_path):
    monitor = Monitor(tmp_path / "state", run_id="run")
    original = sample()
    assert monitor.publish(original)
    assert not monitor.publish(original)
    original.measurements["level"] = Measurement(99)
    assert monitor.samples[0].measurements["level"].value == 3
    snapshot = monitor.snapshot()
    snapshot["samples"]["window-a"]["sample"]["measurements"].clear()
    assert len(monitor.samples[0].measurements) == 1
    with pytest.raises(FileExistsError):
        Monitor(tmp_path / "state", run_id="different-run")
    with pytest.raises(ValueError, match="revision"):
        monitor.publish(sample(4))
    monitor.save_report(
        {"id": "report", "status": "completed", "evidence": {"window-a": 1}}
    )
    monitor.publish(sample(4, 2))
    recovered = Monitor.load(tmp_path / "state")
    assert len(recovered.snapshot()["history"]) == 2
    assert recovered.snapshot()["reports"][0]["status"] == "stale"
    assert recovered.samples[0].observed_start == 0.4
    recovered.save_report(
        {"id": "pending", "status": "running", "evidence": {"window-a": 2}}
    )
    viewer = Monitor.load(tmp_path / "state", recover=False)
    assert viewer.snapshot()["reports"][-1]["status"] == "running"
    assert (
        Monitor.load(tmp_path / "state").snapshot()["reports"][-1]["status"]
        == "interrupted"
    )


def test_invalid_values_and_explicit_time_contract(tmp_path):
    monitor = Monitor(tmp_path / "state", run_id="finite")
    monitor.publish(sample(float("nan")))
    measurement = monitor.samples[0].measurements["level"]
    assert measurement.value is None and measurement.status == "invalid"
    assert Measurement(None).status == "missing"
    assert Measurement(None, status="unavailable").status == "unavailable"
    json.dumps(monitor.snapshot(), allow_nan=False)
    with pytest.raises(ValueError):
        Measurement(3, coverage=float("nan"))
    with pytest.raises(ValueError):
        replace(sample(), observed_end=0.1)
    with pytest.raises(ValueError):
        replace(sample(), metadata={"array": [1] * 20000})


def test_ingestion_errors_and_failed_source_are_visible(tmp_path):
    monitor = Monitor(tmp_path / "state", run_id="errors")

    def analyze(value):
        if value == 1:
            raise ValueError("private path or credential must not be saved")
        return replace(sample(), id=f"sample-{value}")

    monitor.replay([0, 1, 2], analyze)
    assert len(monitor.samples) == 2
    assert monitor.snapshot()["collector"] == "complete"
    assert monitor.snapshot()["errors"][0]["error_type"] == "ValueError"
    assert "credential" not in json.dumps(monitor.snapshot())

    def broken_source():
        yield 3
        raise OSError("source stopped")

    with pytest.raises(OSError):
        monitor.replay(broken_source(), analyze)
    assert len(monitor.samples) == 3
    assert monitor.snapshot()["collector"] == "failed"
    assert monitor.snapshot()["errors"][-1]["source_id"] == "source"


def test_storage_failure_preserves_previous_state(tmp_path, monkeypatch):
    monitor = Monitor(tmp_path / "state", run_id="storage")
    monitor.publish(sample())

    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr("cuphoton.xlens.monitor.os.fsync", fail)
    with pytest.raises(OSError):
        monitor.publish(sample(4, 2))
    assert monitor.snapshot()["storage_error"]
    assert monitor.samples[0].revision == 1
    assert Monitor.load(tmp_path / "state").samples[0].revision == 1


def test_dashboard_installed_resources_and_refresh(tmp_path):
    monitor = quickstart(tmp_path / "quickstart")
    page = (tmp_path / "quickstart/report.html").read_text()
    assert "Invented measurement monitor" in page
    assert "Apache License 2.0" in page
    viewer = Monitor.load(monitor.directory, recover=False)
    server = make_server(Dashboard(viewer), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monitor.publish(sample())
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/api/state"
        ) as response:
            body = json.load(response)
        assert "window-a" in body["samples"]
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/"
        ) as response:
            assert b"window-a" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_dashboard_preserves_placeholder_text_in_evidence_and_views(tmp_path):
    monitor = Monitor(tmp_path / "state", run_id="placeholders")
    message = "__SCRIPT__ __DATA__ </script> &"
    monitor.publish(
        replace(sample(), findings=(Finding("literal", message),))
    )
    page = Dashboard(
        monitor,
        title="__STYLE__ <title>",
        render_sample=lambda _sample: "<p>__DATA__ __SCRIPT__</p>",
    ).render()
    state = re.search(
        r'<script id="state" type="application/json">(.*?)</script>',
        page,
        flags=re.DOTALL,
    )
    assert state is not None
    display = json.loads(state.group(1))
    assert (
        display["records"][0]["sample"]["findings"][0]["message"] == message
    )
    assert "<title>__STYLE__ &lt;title&gt;</title>" in page
    assert "<p>__DATA__ __SCRIPT__</p>" in page


def test_unchanged_collector_and_report_do_not_rewrite_state(
    tmp_path, monkeypatch
):
    monitor = Monitor(tmp_path / "state", run_id="no-op")
    monitor.publish(sample())
    monitor._collector("watching")
    report = {
        "id": "report",
        "status": "completed",
        "evidence": {"window-a": 1},
    }
    monitor.save_report(report)

    def unexpected_commit(_state):
        raise AssertionError("unchanged state must not be rewritten")

    monkeypatch.setattr(monitor, "_commit", unexpected_commit)
    monitor._collector("watching")
    returned = monitor.save_report(report)
    returned["evidence"].clear()
    assert monitor.snapshot()["reports"][0]["evidence"] == {"window-a": 1}


def test_dashboard_csp_hash_matches_embedded_script(tmp_path):
    import base64
    import hashlib
    import re

    monitor = Monitor(tmp_path / "run", run_id="csp")
    page = Dashboard(monitor).render()
    script = re.search(r"<script>(.*?)</script>", page, re.DOTALL).group(1)
    digest = base64.b64encode(
        hashlib.sha256(script.encode()).digest()
    ).decode()
    assert f"script-src 'sha256-{digest}'" in page
    assert "script-src 'unsafe-inline'" not in page
