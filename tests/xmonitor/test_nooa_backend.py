# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

import pytest

from cuphoton.xmonitor import nooa_backend as backend

try:
    INSTALLED = importlib.metadata.version("nooa") == "0.0.10"
except importlib.metadata.PackageNotFoundError:
    INSTALLED = False

requires_nooa = pytest.mark.skipif(
    not INSTALLED or sys.platform != "linux" or not shutil.which("bwrap"),
    reason="requires the optional NOOA profile and Linux bubblewrap",
)


def evidence():
    return {
        "run_id": "fixture",
        "selected_ids": ["second"],
        "samples": [
            {
                "id": identifier,
                "sequence": index,
                "revision": 1,
                "findings": [{"code": "review", "message": "Inspect flux"}],
                "measurements": {
                    "flux": {
                        "value": value,
                        "unit": "count",
                        "status": "valid",
                    }
                },
            }
            for index, (identifier, value) in enumerate(
                [("first", 2.0), ("second", 5.0)]
            )
        ],
    }


def response(name, arguments):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    }


def answer():
    return {
        "narrative": "The second sample records 5 count.",
        "citations": [
            {
                "sample_id": "second",
                "revision": 1,
                "measurement": "flux",
                "value": 5.0,
                "unit": "count",
            }
        ],
    }


async def run(request, progress=lambda event: None, **kwargs):
    options = {
        "max_iterations": 5,
        "max_tool_calls": 20,
        "timeout_seconds": 30,
        "max_context_bytes": 65536,
        "max_response_bytes": 65536,
    }
    options.update(kwargs)
    return await backend.run_nooa(
        evidence(),
        "Explain the changed flux",
        "",
        request=request,
        progress=progress,
        **options,
    )


@requires_nooa
def test_real_iterative_queries_and_application_view(monkeypatch):
    monkeypatch.setenv("XMONITOR_PARENT_SECRET", "fixture-parent-only")
    calls, events, domain_calls = [], [], []
    original = evidence()
    unchanged = copy.deepcopy(original)

    async def request(payload):
        calls.append(payload)
        assert set(payload) == {"messages", "tools"}
        assert "fixture-parent-only" not in json.dumps(payload)
        if len(calls) == 1:
            return response(
                "execute_python", {"code": "print(self.get_sample('second'))"}
            )
        if len(calls) == 2:
            assert "5.0" in json.dumps(payload)
            return response(
                "execute_python",
                {
                    "code": (
                        "print(self.metric_series('flux'))\n"
                        "print(self.compare_samples(['first', 'second']))\n"
                        "print(self.recent_findings())\n"
                        "print(self.domain_view('second', 'residual'))"
                    )
                },
            )
        assert "residual_peak" in json.dumps(payload)
        return response("return_result", {"result": answer()})

    def domain_query(sample_id, name):
        domain_calls.append((sample_id, name))
        return {"residual_peak": 0.25}

    result = asyncio.run(
        backend.run_nooa(
            original,
            "Explain flux",
            "",
            request=request,
            progress=events.append,
            max_iterations=5,
            max_tool_calls=20,
            timeout_seconds=30,
            max_context_bytes=65536,
            max_response_bytes=65536,
            domain_query=domain_query,
            domain_views={"residual": "Bounded residual summary"},
        )
    )
    assert result == answer() and len(calls) == 3
    assert original == unchanged
    assert domain_calls == [("second", "residual")]
    reads = [event for event in events if event["kind"] == "evidence"]
    assert reads[0]["evidence"] == {"second": 1}
    assert reads[1]["evidence"] == {"first": 1, "second": 1}
    assert reads[-1]["evidence"] == {"second": 1}


@requires_nooa
def test_sandbox_hides_host_files_environment_and_network(
    tmp_path, monkeypatch
):
    sentinel = tmp_path / "private.txt"
    sentinel.write_text("parent-only")
    monkeypatch.setenv("XMONITOR_PARENT_SECRET", "fixture-parent-only")
    host_network = os.readlink("/proc/self/ns/net")
    script = (
        "import json,os,pathlib,sys; "
        "print(json.dumps({'environment':dict(os.environ),"
        f"'host_visible':pathlib.Path({str(sentinel)!r}).exists(),"
        "'unrelated_package':pathlib.Path('/packages/pytest').exists(),"
        "'network':os.readlink('/proc/self/ns/net'),"
        "'paths':sys.path}))"
    )
    command = backend._command()
    command[-1:] = ["-c", script]
    completed = subprocess.run(
        command,
        env={},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    observed = json.loads(completed.stdout)
    assert not observed["host_visible"]
    assert not observed["unrelated_package"]
    assert "XMONITOR_PARENT_SECRET" not in observed["environment"]
    assert observed["network"] != host_network
    assert all(path.startswith("/runtime/") for path in observed["paths"])


@requires_nooa
def test_iteration_and_domain_read_budgets():
    turns = []

    async def request(payload):
        turns.append(payload)
        return response("execute_python", {"code": "print(self.status())"})

    with pytest.raises(backend.NooaError):
        asyncio.run(run(request, max_iterations=2))
    assert len(turns) == 2

    domain_calls = []

    async def domain_request(payload):
        return response(
            "execute_python",
            {"code": "print(self.domain_view('second', 'residual'))"},
        )

    with pytest.raises(backend.NooaError, match="tool limit"):
        asyncio.run(
            run(
                domain_request,
                max_tool_calls=1,
                domain_views={"residual": "Summary"},
                domain_query=lambda *args: domain_calls.append(args) or {},
            )
        )
    assert domain_calls == []


@pytest.mark.parametrize("cancel", [False, True])
@requires_nooa
def test_deadline_and_cancel_reap_worker(monkeypatch, cancel):
    processes = []
    launch = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await launch(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)

    async def exercise():
        entered = asyncio.Event()

        async def request(payload):
            entered.set()
            if cancel:
                await asyncio.Event().wait()
            return response("execute_python", {"code": "while True: pass"})

        task = asyncio.create_task(run(request, timeout_seconds=8))
        if cancel:
            await asyncio.wait_for(entered.wait(), 7)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(backend.NooaError, match="deadline"):
                await task

    asyncio.run(exercise())
    assert len(processes) == 1 and processes[0].returncode is not None


@requires_nooa
def test_missing_isolation_fails_before_model_call(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    async def request(payload):
        raise AssertionError("model request must not run")

    with pytest.raises(backend.NooaError, match="bubblewrap"):
        asyncio.run(run(request))


@pytest.mark.skipif(
    os.name != "posix", reason="parent broker uses POSIX process groups"
)
@pytest.mark.parametrize(
    "probe",
    [
        "get_sample",
        "unknown_query",
        "query_bound",
        "domain_allowlist",
        "model_allowlist",
        "iteration_limit",
    ],
)
def test_parent_broker_with_stdio_worker(monkeypatch, probe):
    # Exercise the real parent protocol without NOOA, bubblewrap, or a model.
    frames = {
        "get_sample": {
            "kind": "query",
            "name": "get_sample",
            "args": {"sample_id": "second"},
        },
        "unknown_query": {
            "kind": "query",
            "name": "read_file",
            "args": {"path": "/etc/passwd"},
        },
        "query_bound": {
            "kind": "query",
            "name": "recent_findings",
            "args": {"limit": 21},
        },
        "domain_allowlist": {
            "kind": "query",
            "name": "domain_view",
            "args": {"sample_id": "second", "query": "unlisted"},
        },
        "model_allowlist": {
            "kind": "model",
            "payload": {"messages": [], "tools": []},
        },
        "iteration_limit": {
            "kind": "model",
            "payload": {"messages": [], "tools": []},
        },
    }
    script = "\n".join(
        [
            "import json,sys",
            "json.loads(sys.stdin.readline())",
            f"frame = {frames[probe]!r}",
            f"for _ in range({3 if probe == 'iteration_limit' else 1}):",
            "    print(json.dumps(frame), flush=True)",
            "    reply = json.loads(sys.stdin.readline())",
            "print(json.dumps({'kind':'result','answer':reply}), flush=True)",
        ]
    )
    monkeypatch.setattr(
        backend, "_command", lambda: [sys.executable, "-I", "-c", script]
    )
    domain_calls, requests = [], []

    async def request(payload):
        requests.append(payload)
        return response(
            "forbidden_tool"
            if probe == "model_allowlist"
            else "execute_python",
            {"code": "pass"},
        )

    options = dict(
        max_iterations=2,
        domain_views={"residual": "Allowed view"},
        domain_query=lambda *args: domain_calls.append(args),
    )
    if probe in {"model_allowlist", "iteration_limit"}:
        with pytest.raises(
            backend.NooaError,
            match="invalid NOOA model response"
            if probe == "model_allowlist"
            else "iteration limit",
        ):
            asyncio.run(run(request, **options))
        assert len(requests) == (1 if probe == "model_allowlist" else 2)
    else:
        result = asyncio.run(run(request, **options))
        if probe == "get_sample":
            assert result["value"]["id"] == "second"
        else:
            assert result == {"error": "unavailable evidence query"}
        assert not requests
    assert not domain_calls
