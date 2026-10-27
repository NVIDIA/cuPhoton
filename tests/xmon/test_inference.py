# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cuphoton.xmon import (
    Dashboard,
    InferenceConfig,
    Investigator,
    Measurement,
    Monitor,
    Sample,
)

pytest.importorskip("httpx")


@pytest.fixture
def endpoint():
    seen = []
    behavior = {
        "status": 200,
        "delay": 0,
        "wrong_value": False,
        "large": False,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            seen.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": payload,
                }
            )
            evidence = json.loads(payload["messages"][-1]["content"])[
                "evidence"
            ]
            sample = evidence["samples"][0]
            name, measurement = next(iter(sample["measurements"].items()))
            answer = {
                "narrative": "Inspect the measured value.",
                "citations": [
                    {
                        "sample_id": sample["id"],
                        "revision": sample["revision"],
                        "measurement": name,
                        "value": 999
                        if behavior["wrong_value"]
                        else measurement["value"],
                        "unit": measurement["unit"],
                    }
                ],
            }
            body = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(answer)},
                        }
                    ]
                }
            ).encode()
            if behavior["large"]:
                body += b" " * 5000
            time.sleep(behavior["delay"])
            self.send_response(behavior["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen, behavior
    server.shutdown()
    server.server_close()
    thread.join()


def make_monitor(tmp_path):
    monitor = Monitor(tmp_path / "state", run_id="fixture")
    monitor.publish(
        Sample(
            "observation",
            3,
            {"flux": Measurement(2.5, "arb")},
            metadata={"local_only": "not model evidence"},
        )
    )
    return monitor


@pytest.mark.parametrize("auth", ["none", "bearer"])
def test_portable_client_auth_model_and_report_separation(
    tmp_path, monkeypatch, endpoint, auth
):
    url, seen, _ = endpoint
    monkeypatch.setenv("CUPHOTON_XMON_API_KEY", "fixture-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-credential")
    monitor = make_monitor(tmp_path)
    before = monitor.snapshot()["samples"]
    config = InferenceConfig(
        url,
        "institution/exact-model-id",
        auth=auth,
        api_key_env="CUPHOTON_XMON_API_KEY" if auth == "bearer" else None,
    )
    report = asyncio.run(
        Investigator(config).investigate(monitor, ["observation"])
    )
    assert report["status"] == "completed"
    assert seen[0]["path"] == "/v1/chat/completions"
    assert seen[0]["body"]["model"] == "institution/exact-model-id"
    assert seen[0]["authorization"] == (
        "Bearer fixture-credential" if auth == "bearer" else None
    )
    assert "local_only" not in json.dumps(seen[0]["body"])
    assert "tools" not in seen[0]["body"]
    assert monitor.snapshot()["samples"] == before
    page = Dashboard(monitor).render()
    assert "fixture-credential" not in page
    assert "unrelated-credential" not in page
    assert url not in page
    assert "institution/exact-model-id" not in page


def test_reject_incorrect_evidence_and_unsupported_capability(
    tmp_path, endpoint
):
    url, seen, behavior = endpoint
    monitor = make_monitor(tmp_path)
    config = InferenceConfig(url, "custom", response_format="json_object")
    behavior["wrong_value"] = True
    report = asyncio.run(
        Investigator(config).investigate(monitor, ["observation"])
    )
    assert report["status"] == "failed"
    assert "evidence validation" in report["diagnostic"]
    behavior["status"] = 400
    report = asyncio.run(
        Investigator(config).investigate(monitor, ["observation"])
    )
    assert report["status"] == "failed"
    assert "contract" in report["diagnostic"]
    assert len(seen) == 2  # No model, endpoint or output-mode fallback.
    assert seen[-1]["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        (
            {"value": 999},
            "citation does not match the recorded measurement",
        ),
        ({"sample_id": ["observation"]}, "invalid answer structure"),
    ],
)
def test_nooa_invalid_answer_reports_validation_not_transport(
    tmp_path, monkeypatch, changed, reason
):
    monitor = make_monitor(tmp_path)
    before = monitor.snapshot()["samples"]
    answer = {
        "narrative": "private response text must not enter a failed report",
        "citations": [
            {
                "sample_id": "observation",
                "revision": 1,
                "measurement": "flux",
                "value": 2.5,
                "unit": "arb",
                **changed,
            }
        ],
    }

    async def completed_nooa(*args, **kwargs):
        return answer

    monkeypatch.setattr("cuphoton.xmon.nooa_backend.run_nooa", completed_nooa)
    investigator = Investigator(
        InferenceConfig(
            "http://inference.example/v1", "custom", backend="nooa"
        )
    )
    report = asyncio.run(investigator.investigate(monitor, ["observation"]))
    assert report["status"] == "failed"
    assert report["diagnostic"] == (
        "NOOA response failed evidence validation: " + reason
    )
    assert answer["narrative"] not in json.dumps(report)
    assert monitor.snapshot()["reports"][-1] == report
    assert monitor.snapshot()["samples"] == before

    answer["citations"][0].update(sample_id="observation", value=2.5)
    report = asyncio.run(investigator.investigate(monitor, ["observation"]))
    assert report["status"] == "completed"
    assert report["citations"][0]["value"] == 2.5


def test_bounds_missing_credentials_and_timeouts(
    tmp_path, monkeypatch, endpoint
):
    url, seen, behavior = endpoint
    monitor = make_monitor(tmp_path)
    monkeypatch.delenv("CUPHOTON_XMON_API_KEY", raising=False)
    missing = InferenceConfig(
        url, "custom", auth="bearer", api_key_env="CUPHOTON_XMON_API_KEY"
    )
    assert (
        asyncio.run(
            Investigator(missing).investigate(monitor, ["observation"])
        )["status"]
        == "failed"
    )
    assert not seen
    behavior["large"] = True
    bounded = InferenceConfig(url, "custom", max_response_bytes=1024)
    report = asyncio.run(
        Investigator(bounded).investigate(monitor, ["observation"])
    )
    assert "byte limit" in report["diagnostic"]
    behavior.update(large=False, delay=0.3)
    timed = replace(bounded, timeout_seconds=0.1)
    report = asyncio.run(
        Investigator(timed).investigate(monitor, ["observation"])
    )
    assert report["status"] == "timed_out"
    with pytest.raises(ValueError):
        asyncio.run(
            Investigator(bounded).investigate(monitor, ["observation"] * 65)
        )


def test_cancellation_admission_and_revised_inflight_evidence(
    tmp_path, monkeypatch
):
    monitor = make_monitor(tmp_path)
    config = InferenceConfig("http://inference.example/v1", "custom")
    investigator = Investigator(config)

    async def check():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def controlled(evidence, question, instructions):
            entered.set()
            await release.wait()
            return {"narrative": "Measured value.", "citations": []}

        monkeypatch.setattr(investigator, "_request", controlled)
        task = asyncio.create_task(
            investigator.investigate(monitor, ["observation"])
        )
        await entered.wait()
        busy = await investigator.investigate(monitor, ["observation"])
        assert busy["status"] == "busy"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert monitor.snapshot()["reports"][-1]["status"] == "cancelled"
        entered.clear()
        task = asyncio.create_task(
            investigator.investigate(monitor, ["observation"])
        )
        await entered.wait()
        monitor.publish(
            replace(
                monitor.samples[0],
                revision=2,
                measurements={"flux": Measurement(3, "arb")},
            )
        )
        release.set()
        assert (await task)["status"] == "stale"

    asyncio.run(check())
