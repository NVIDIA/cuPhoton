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

from cuphoton.xlens import (
    Dashboard,
    InferenceConfig,
    Investigator,
    Measurement,
    Monitor,
    Sample,
)


@pytest.fixture
def endpoint():
    pytest.importorskip("httpx")
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
            messages = payload.get("messages", payload.get("input"))
            evidence = json.loads(messages[-1]["content"])["evidence"]
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
            result = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(answer)},
                    }
                ]
            }
            if self.path.endswith("/responses"):
                result = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(answer),
                                }
                            ],
                        }
                    ],
                }
            body = json.dumps(result).encode()
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
@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_portable_client_auth_model_and_report_separation(
    tmp_path, monkeypatch, endpoint, auth, api
):
    url, seen, _ = endpoint
    monkeypatch.setenv("CUPHOTON_XLENS_API_KEY", "fixture-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-credential")
    monitor = make_monitor(tmp_path)
    before = monitor.snapshot()["samples"]
    config = InferenceConfig(
        url,
        "institution/exact-model-id",
        auth=auth,
        api_key_env="CUPHOTON_XLENS_API_KEY" if auth == "bearer" else None,
        api=api,
    )
    report = asyncio.run(
        Investigator(config).investigate(monitor, ["observation"])
    )
    assert report["status"] == "completed"
    assert seen[0]["path"] == (
        "/v1/responses" if api == "responses" else "/v1/chat/completions"
    )
    if api == "responses":
        assert seen[0]["body"]["store"] is False
        assert seen[0]["body"]["max_output_tokens"] == config.max_tokens
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


def test_bounds_missing_credentials_and_timeouts(
    tmp_path, monkeypatch, endpoint
):
    url, seen, behavior = endpoint
    monitor = make_monitor(tmp_path)
    monkeypatch.delenv("CUPHOTON_XLENS_API_KEY", raising=False)
    missing = InferenceConfig(
        url, "custom", auth="bearer", api_key_env="CUPHOTON_XLENS_API_KEY"
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


@pytest.mark.parametrize(
    "url",
    [
        "http://inference.example/v1",
        "http://192.0.2.1/v1",
        "http://[2001:db8::1]/v1",
    ],
)
def test_remote_bearer_requires_https(url):
    with pytest.raises(ValueError, match="requires HTTPS"):
        InferenceConfig(
            url, "fixture", auth="bearer", api_key_env="FIXTURE_KEY"
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://inference.example/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_bearer_allows_https_or_loopback(url):
    assert (
        InferenceConfig(
            url, "fixture", auth="bearer", api_key_env="FIXTURE_KEY"
        ).base_url
        == url
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "valid",
        "revision",
        "sample_id",
        "unit",
        "boolean",
        "narrative",
        "citations",
        "finish_reason",
        "tool_calls",
    ],
)
def test_response_contract_validates_answers(tmp_path, monkeypatch, invalid):
    monitor = make_monitor(tmp_path)
    # A unit-valued measurement makes bool/numeric equality otherwise pass.
    monitor.publish(
        Sample(
            "observation", 3, {"flux": Measurement(1.0, "arb")}, revision=2
        )
    )
    answer = {
        "narrative": "Measured flux.",
        "citations": [
            {
                "sample_id": "observation",
                "revision": 2,
                "measurement": "flux",
                "value": 1.0,
                "unit": "arb",
            }
        ],
    }
    if invalid == "revision":
        answer["citations"][0]["revision"] = 1
    elif invalid == "sample_id":
        answer["citations"][0]["sample_id"] = ["observation"]
    elif invalid == "unit":
        answer["citations"][0]["unit"] = "other"
    elif invalid == "boolean":
        answer["citations"][0]["value"] = True
    elif invalid == "narrative":
        answer["narrative"] = "x" * 8001
    elif invalid == "citations":
        answer["citations"] *= 65
    message = {"content": json.dumps(answer)}
    if invalid == "tool_calls":
        message["tool_calls"] = [{"function": {"name": "unexpected"}}]

    async def chat(self, payload):
        return {
            "choices": [
                {
                    "finish_reason": "length"
                    if invalid == "finish_reason"
                    else "stop",
                    "message": message,
                }
            ]
        }

    monkeypatch.setattr(Investigator, "_chat", chat)
    report = asyncio.run(
        Investigator(
            InferenceConfig("http://localhost:1/v1", "fixture")
        ).investigate(monitor, ["observation"])
    )
    if invalid == "valid":
        assert report["status"] == "completed"
        assert report["narrative"] == answer["narrative"]
        assert report["citations"] == answer["citations"]
    else:
        assert report["status"] == "failed"
        expected_error = {
            "finish_reason": (
                "endpoint did not complete a bounded text response"
            ),
            "tool_calls": "model requested unsupported tool execution",
        }.get(invalid, "model response failed evidence validation")
        assert report["diagnostic"] == expected_error
        assert "narrative" not in report
        assert "citations" not in report


@pytest.mark.parametrize(
    ("backend", "runtime_error", "expected"),
    [
        (
            "basic",
            None,
            "Install cuphoton[xlens-inference] for investigation.",
        ),
        (
            "nooa",
            "NOOA requires Linux and Python 3.12 or 3.13",
            "NOOA requires Linux and Python 3.12 or 3.13",
        ),
        (
            "nooa",
            "install cuphoton[xlens-nooa] for NOOA",
            "install cuphoton[xlens-nooa] for NOOA",
        ),
        (
            "nooa",
            "NOOA sandbox unavailable: permission denied",
            "NOOA sandbox unavailable: permission denied",
        ),
        (
            "nooa",
            None,
            "Install cuphoton[xlens-nooa] for investigation.",
        ),
    ],
)
def test_missing_runtime_is_reported_before_creating_a_report(
    tmp_path, monkeypatch, backend, runtime_error, expected
):
    from cuphoton.xlens import MonitorSession, inference
    from cuphoton.xlens.inference import InferenceError
    from cuphoton.xlens.nooa_backend import NooaError

    calls = []

    def command():
        calls.append("runtime")
        if runtime_error:
            raise NooaError(runtime_error)
        return []

    monkeypatch.setattr(inference, "_command", command)
    monkeypatch.setattr(
        inference.importlib.util, "find_spec", lambda name: None
    )
    config = InferenceConfig(
        "http://localhost:1/v1", "fixture", backend=backend
    )
    with pytest.raises((InferenceError, NooaError)) as error:
        Investigator(config).check_available()
    assert str(error.value) == expected
    monitor = make_monitor(tmp_path)
    session = MonitorSession(monitor, config=config)
    assert session.reason == expected
    assert not session.can_ask
    assert monitor.snapshot()["reports"] == []
    assert calls == (["runtime", "runtime"] if backend == "nooa" else [])


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize(
    "outcome",
    ["complete", "changed_before", "changed_during", "bad_citation"],
)
def test_nooa_diagnostic_bridge_and_answer_validation_without_worker(
    tmp_path, monkeypatch, api, outcome
):
    from cuphoton.xlens import nooa_backend
    from cuphoton.xlens._responses import ResponsesAdapter

    monitor = make_monitor(tmp_path)
    context = Sample("context", 4, {"flux": Measurement(3.0, "arb")})
    monitor.publish(context)
    calls = []
    adapters = []
    progress = []
    answer = {
        "narrative": "Compare the recorded flux.",
        "citations": [
            {
                "sample_id": "context",
                "revision": 1,
                "measurement": "flux",
                "value": 999 if outcome == "bad_citation" else 3.0,
                "unit": "arb",
            }
        ],
    }

    def revise():
        monitor.publish(replace(context, revision=2))

    def domain_query(sample_id, query):
        calls.append((sample_id, query))
        if outcome == "changed_during":
            revise()
        return {"peak": 3.0}

    async def chat(self, payload, *, adapter=None):
        assert payload == {
            "messages": [{"role": "user", "content": "Inspect"}]
        }
        adapters.append(adapter)
        return {"choices": [{"message": {"content": json.dumps(answer)}}]}

    async def worker(evidence, question, instructions, **kwargs):
        assert evidence["selected_ids"] == ["observation"]
        assert {s["id"] for s in evidence["samples"]} == {
            "observation",
            "context",
        }
        assert question == "Compare flux"
        assert instructions == "Cite measurements."
        assert kwargs["domain_views"] == {"profile": "Measured profile"}
        if outcome == "changed_before":
            revise()
        assert kwargs["domain_query"]("context", "profile") == {"peak": 3.0}
        kwargs["progress"]({"kind": "tool", "message": "Read profile."})
        payload = {"messages": [{"role": "user", "content": "Inspect"}]}
        await kwargs["request"](payload)
        response = await kwargs["request"](payload)
        return json.loads(response["choices"][0]["message"]["content"])

    monkeypatch.setattr(Investigator, "_chat", chat)
    monkeypatch.setattr(nooa_backend, "run_nooa", worker)
    config = InferenceConfig(
        "http://localhost:1/v1", "fixture", backend="nooa", api=api
    )
    report = asyncio.run(
        Investigator(config).investigate(
            monitor,
            ["observation"],
            question="Compare flux",
            instructions="Cite measurements.",
            domain_query=domain_query,
            domain_views={"profile": "Measured profile"},
            progress=progress.append,
        )
    )
    assert report["selected_evidence"] == {"observation": 1}
    assert report["evidence"] == {"observation": 1, "context": 1}
    assert calls == (
        [] if outcome == "changed_before" else [("context", "profile")]
    )
    if outcome.startswith("changed_"):
        assert report["status"] == "stale"
        assert report["diagnostic"] == (
            "diagnostic evidence changed; start another investigation"
        )
        assert not adapters and not progress
    else:
        assert len(adapters) == 2
        assert adapters[0] is adapters[1]
        if api == "responses":
            assert isinstance(adapters[0], ResponsesAdapter)
        else:
            assert adapters == [None, None]
        assert progress[0]["message"] == "Read profile."
        assert report["events"] == progress
        if outcome == "complete":
            assert report["status"] == "completed"
            assert report["citations"] == answer["citations"]
        else:
            assert report["status"] == "failed"
            assert report["diagnostic"] == (
                "NOOA response failed evidence validation: "
                "citation does not match the recorded measurement"
            )
    if outcome != "complete":
        assert "citations" not in report and "narrative" not in report
