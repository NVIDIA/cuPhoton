# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cuphoton.xlens import (
    InferenceConfig,
    Investigator,
    Measurement,
    Monitor,
    Sample,
)

try:
    INSTALLED = importlib.metadata.version("nooa") == "0.0.12"
except importlib.metadata.PackageNotFoundError:
    INSTALLED = False

requires_nooa = pytest.mark.skipif(
    os.environ.get("CUPHOTON_REQUIRE_NOOA_TESTS") != "1"
    and (
        not INSTALLED
        or sys.platform != "linux"
        or sys.version_info[:2] not in {(3, 12), (3, 13)}
        or not shutil.which("bwrap")
    ),
    reason="requires the optional NOOA profile and Linux bubblewrap",
)

SAMPLES = {"transport-alpha": 2.5, "transport-beta": 7.5}


def answer(sample_id):
    return {
        "narrative": f"Inspect the recorded flux for {sample_id}.",
        "citations": [
            {
                "sample_id": sample_id,
                "revision": 1,
                "measurement": "flux",
                "value": SAMPLES[sample_id],
                "unit": "count",
            }
        ],
    }


@pytest.fixture
def tool_endpoint():
    requests = {sample_id: [] for sample_id in SAMPLES}
    provider_outputs = {}
    failures = []
    first_turns = threading.Barrier(2)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                serialized = json.dumps(payload)
                matching = [key for key in SAMPLES if key in serialized]
                assert len(matching) == 1, "investigation evidence crossed"
                sample_id = matching[0]
                with lock:
                    requests[sample_id].append(
                        {
                            "path": self.path,
                            "authorization": self.headers.get(
                                "Authorization"
                            ),
                            "body": payload,
                        }
                    )
                    turn = len(requests[sample_id])
                assert turn in {1, 2}, "unexpected model retry"
                if turn == 1:
                    # Neither worker can finish before both investigations
                    # have reached the actual HTTP transport concurrently.
                    first_turns.wait(timeout=20)
                    name = "execute_python"
                    arguments = {
                        "code": (
                            "import os\n"
                            "assert 'XLENS_TRANSPORT_TOKEN' "
                            "not in os.environ\n"
                            f"print(self.get_sample({sample_id!r}))"
                        )
                    }
                    call_id = "read-evidence"
                else:
                    name = "return_result"
                    arguments = {"result": answer(sample_id)}
                    call_id = "finish-investigation"
                if self.path == "/v1/responses":
                    output = [
                        {
                            "id": f"fc-{sample_id}-{turn}",
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": json.dumps(arguments),
                            "status": "completed",
                        }
                    ]
                    if turn == 1:
                        output.insert(
                            0,
                            {
                                "id": f"rs-{sample_id}",
                                "type": "reasoning",
                                "summary": [],
                                "encrypted_content": f"opaque-{sample_id}",
                            },
                        )
                        provider_outputs[sample_id] = output
                    result = {"status": "completed", "output": output}
                else:
                    assert self.path == "/v1/chat/completions"
                    result = {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(
                                                    arguments
                                                ),
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                status = 200
            except Exception as exc:
                failures.append(repr(exc))
                result, status = {}, 500
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f"http://127.0.0.1:{server.server_port}/v1",
            requests,
            provider_outputs,
            failures,
        )
    finally:
        first_turns.abort()
        server.shutdown()
        server.server_close()
        thread.join()


@requires_nooa
@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_concurrent_nooa_investigations_through_http(
    tmp_path, monkeypatch, tool_endpoint, api
):
    url, requests, provider_outputs, failures = tool_endpoint
    monkeypatch.setenv("XLENS_TRANSPORT_TOKEN", "synthetic-transport-secret")
    config = InferenceConfig(
        url,
        "fixture/exact-model",
        backend="nooa",
        api=api,
        auth="bearer",
        api_key_env="XLENS_TRANSPORT_TOKEN",
        max_tokens=1024,
        token_parameter="max_completion_tokens",
        max_concurrent=2,
        timeout_seconds=45,
    )
    investigator = Investigator(config)
    monitors = []
    for sample_id, value in SAMPLES.items():
        monitor = Monitor(tmp_path / sample_id, run_id=sample_id)
        monitor.publish(
            Sample(sample_id, 1, {"flux": Measurement(value, "count")})
        )
        monitors.append(monitor)

    async def investigate_both():
        return await asyncio.gather(
            *(
                investigator.investigate(
                    monitor,
                    [sample_id],
                    question=f"Inspect {sample_id} using its recorded flux.",
                )
                for monitor, sample_id in zip(monitors, SAMPLES, strict=True)
            )
        )

    reports = asyncio.run(investigate_both())
    assert not failures
    for sample_id, report in zip(SAMPLES, reports, strict=True):
        assert report["status"] == "completed", report
        assert report["citations"] == answer(sample_id)["citations"]
        assert len(requests[sample_id]) == 2
        for request in requests[sample_id]:
            payload = request["body"]
            assert request["authorization"] == (
                "Bearer synthetic-transport-secret"
            )
            assert "synthetic-transport-secret" not in json.dumps(payload)
            assert payload["model"] == config.model
            assert payload["stream"] is False
            if api == "responses":
                assert request["path"] == "/v1/responses"
                assert payload["store"] is False
                assert payload["max_output_tokens"] == config.max_tokens
                assert {tool["name"] for tool in payload["tools"]} == {
                    "execute_python",
                    "return_result",
                }
            else:
                assert request["path"] == "/v1/chat/completions"
                assert payload["max_completion_tokens"] == config.max_tokens
                assert {
                    tool["function"]["name"] for tool in payload["tools"]
                } == {"execute_python", "return_result"}

        second = requests[sample_id][1]["body"]
        if api == "responses":
            replay = [
                item
                for item in second["input"]
                if item.get("type") == "reasoning"
                or item.get("call_id") == "read-evidence"
            ]
            assert replay[:-1] == provider_outputs[sample_id]
            assert replay[-1]["type"] == "function_call_output"
            assert replay[-1]["call_id"] == "read-evidence"
            assert replay[-1]["output"] == "status: accepted"
            messages = second["input"]
            assert f"opaque-{sample_id}" not in json.dumps(report)
        else:
            messages = second["messages"]
            assistant = next(
                item
                for item in messages
                if item.get("tool_calls", [{}])[0].get("id")
                == "read-evidence"
            )
            assert assistant["tool_calls"][0]["function"]["name"] == (
                "execute_python"
            )
            tool = next(
                item
                for item in messages
                if item.get("tool_call_id") == "read-evidence"
            )
            assert tool["role"] == "tool"
            assert tool["content"] == "status: accepted"
        # NOOA acknowledges execution in the tool message, then supplies
        # the actual Python output in a separate user message.
        evidence_output = next(
            item["content"]
            for item in messages
            if item.get("role") == "user"
            and "PythonOutput" in item["content"]
            and "read-evidence" in item["content"]
        )
        assert sample_id in evidence_output
        assert str(SAMPLES[sample_id]) in evidence_output
        assert "AssertionError" not in evidence_output
        assert "synthetic-transport-secret" not in json.dumps(report)
