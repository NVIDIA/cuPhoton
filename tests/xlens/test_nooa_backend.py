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
import signal
import socket
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest

from cuphoton.xlens import nooa_backend as backend

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
        or os.getuid() == 0
    ),
    reason="requires the optional NOOA profile and non-root Linux bubblewrap",
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
    monkeypatch.setenv("XLENS_PARENT_SECRET", "fixture-parent-only")
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
def test_sandbox_hides_host_files_environment_and_network(monkeypatch):
    monkeypatch.setenv("XLENS_PARENT_SECRET", "fixture-parent-only")
    script = (
        "import errno,json,os,pathlib,socket,sys\n"
        "assert list(pathlib.Path('/proc').iterdir()) == []\n"
        "for prefix in ('self', 'thread-self', '1', str(os.getpid())):\n"
        "    for entry in ('mountinfo', 'mounts', 'mountstats', 'cmdline',\n"
        "                  f'task/{os.getpid()}/mountinfo'):\n"
        "        try:\n"
        "            (pathlib.Path('/proc') / prefix / entry).read_bytes()\n"
        "        except FileNotFoundError:\n"
        "            pass\n"
        "        else:\n"
        "            raise AssertionError('process metadata exposed')\n"
        "try:\n"
        "    connection = socket.create_connection(ENDPOINT, timeout=1)\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    connection.close()\n"
        "    raise AssertionError('host loopback reachable')\n"
        "for directory in ('/', '/dev', '/dev/shm'):\n"
        "    assert os.statvfs(directory).f_flag & os.ST_RDONLY\n"
        "    try:\n"
        "        pathlib.Path(directory, 'write-probe').write_text('x')\n"
        "    except OSError as error:\n"
        "        assert error.errno == errno.EROFS\n"
        "    else:\n"
        "        raise AssertionError('unexpected writable filesystem')\n"
        "scratch = pathlib.Path('/work/write-probe')\n"
        "scratch.write_text('x')\n"
        "scratch.unlink()\n"
        "space = os.statvfs('/work')\n"
        "assert space.f_blocks * space.f_frsize <= 16 * 1024**2\n"
        "print(json.dumps({'environment':dict(os.environ),"
        "'roots':sorted(os.listdir('/')),"
        "'host_visible':pathlib.Path(SENTINEL).exists(),"
        "'unrelated_package':pathlib.Path('/packages/pytest').exists(),"
        "'paths':sys.path}))"
    )
    # Keep this outside /tmp, which is independently hidden by /work.
    with (
        tempfile.NamedTemporaryFile(dir=Path(__file__).parent) as sentinel,
        socket.socket() as listener,
    ):
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        endpoint = listener.getsockname()
        # Prove that this exact endpoint is reachable in the parent namespace.
        with socket.create_connection(endpoint, timeout=1):
            pass
        with backend._worker_command() as (command, pass_fds):
            command[-1:] = [
                "-c",
                script.replace("SENTINEL", repr(sentinel.name)).replace(
                    "ENDPOINT", repr(endpoint)
                ),
            ]
            completed = subprocess.run(
                command,
                pass_fds=pass_fds,
                env=dict(os.environ),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    expected_roots = {
        "dev",
        "packages",
        "proc",
        "runtime",
        "tmp",
        "work",
        "worker.py",
    }
    expected_roots.update(
        directory.lstrip("/")
        for directory in ("/lib", "/lib64", "/usr")
        if Path(directory).exists()
    )
    assert set(observed["roots"]) == expected_roots
    assert not observed["host_visible"]
    assert not observed["unrelated_package"]
    assert "XLENS_PARENT_SECRET" not in observed["environment"]
    assert all(path.startswith("/runtime/") for path in observed["paths"])


@requires_nooa
def test_generated_code_cannot_spawn_or_exceed_address_space():
    code = """import ctypes, errno, mmap, os, resource, threading
assert os.getuid() != 0
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(21, 0, 0, 0, 0) == 2  # PR_GET_SECCOMP: filter mode
libc.personality.argtypes = [ctypes.c_ulong]
ctypes.set_errno(0)
assert libc.personality(0xffffffff) == -1
assert ctypes.get_errno() == errno.EPERM
# statmount/listmount have these UAPI numbers on both supported architectures.
libc.syscall.argtypes = [ctypes.c_long]
for number in (457, 458):
    ctypes.set_errno(0)
    assert libc.syscall(number, 0, 0, 0, 0) == -1
    assert ctypes.get_errno() == errno.EPERM
assert resource.getrlimit(resource.RLIMIT_NPROC) == (0, 0)
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
except (ValueError, PermissionError):
    pass
else:
    raise AssertionError('worker raised its process limit')
try:
    pid = os.fork()
except OSError as error:
    assert error.errno == errno.EAGAIN
else:
    raise AssertionError('worker created a child')
thread = threading.Thread(target=lambda: None)
try:
    thread.start()
except RuntimeError as error:
    assert "can't start new thread" in str(error)
else:
    thread.join()
    raise AssertionError('worker created a thread')
limit = 2 * 1024**3
assert resource.getrlimit(resource.RLIMIT_AS) == (limit, limit)
try:
    allocation = mmap.mmap(-1, limit + 1)
except OSError as error:
    assert error.errno == errno.ENOMEM
else:
    allocation.close()
    raise AssertionError('worker exceeded its address-space limit')
print('kernel resource limits enforced')
"""
    turns = []

    async def request(payload):
        turns.append(payload)
        if len(turns) == 1:
            return response("execute_python", {"code": code})
        outputs = [
            message["content"]
            for message in payload["messages"]
            if message.get("role") == "user"
            and "PythonOutput(" in message.get("content", "")
        ]
        assert "kernel resource limits enforced" in outputs[-1]
        assert "ResultStatus.COMPLETE" in outputs[-1]
        assert "AssertionError" not in outputs[-1]
        return response("return_result", {"result": answer()})

    assert asyncio.run(run(request)) == answer()
    assert len(turns) == 2


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
                await bounded(task, processes)
        else:
            with pytest.raises(backend.NooaError, match="deadline"):
                await bounded(task, processes)

    asyncio.run(exercise())
    assert len(processes) == 1
    assert processes[0].returncode == -signal.SIGKILL


@requires_nooa
def test_missing_isolation_fails_before_model_call(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    async def request(payload):
        raise AssertionError("model request must not run")

    with pytest.raises(backend.NooaError, match="bubblewrap"):
        asyncio.run(run(request))


async def bounded(awaitable, processes, timeout=12):
    """Fail promptly even if production cleanup forgets to kill the worker."""
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(process.wait() for process in processes))
        pytest.fail("NOOA worker did not stop within the test deadline")
    return task.result()


@pytest.fixture
def stdio_worker(monkeypatch):
    """Run the actual broker with an independent, blocking protocol peer."""
    if os.name != "posix":
        pytest.skip("parent broker uses POSIX process groups")
    processes = []
    launch = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await launch(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)

    def exercise(frames, model_response=None, **options):
        script = "\n".join(
            [
                "import json,os,sys",
                "settings = json.loads(sys.stdin.readline())",
                "replies = []",
                f"for frame in {frames!r}:",
                "    print(json.dumps(frame), flush=True)",
                "    replies.append(json.loads(sys.stdin.readline()))",
                "print(json.dumps({'kind':'result', 'answer':{",
                "    'replies':replies, 'environment':dict(os.environ)",
                "}}), flush=True)",
                # Deliberately stay alive: success and failure must kill us.
                "sys.stdin.read()",
            ]
        )
        monkeypatch.setattr(
            backend,
            "_worker_command",
            lambda: nullcontext(([sys.executable, "-I", "-c", script], ())),
        )
        requests, events = [], []

        async def request(payload):
            requests.append(payload)
            return model_response or response(
                "execute_python", {"code": "pass"}
            )

        async def invoke():
            return await bounded(
                run(request, events.append, **options), processes, 3
            )

        try:
            result = asyncio.run(invoke())
        finally:
            assert processes[-1].returncode == -signal.SIGKILL
        return result, requests, events

    return exercise


def model_frame(**extra):
    return {
        "kind": "model",
        "payload": {"messages": [], "tools": [], **extra},
    }


def query_frame(name, **args):
    return {"kind": "query", "name": name, "args": args}


def test_parent_broker_reads_detached_evidence(stdio_worker, monkeypatch):
    monkeypatch.setenv("XLENS_PARENT_SECRET", "parent-only")
    result, requests, events = stdio_worker(
        [
            query_frame("get_sample", sample_id="second"),
            query_frame("compare_samples", sample_ids=["first", "second"]),
            query_frame("metric_series", measurement="flux"),
            query_frame("recent_findings", limit=1),
        ]
    )
    assert result["replies"][0]["value"]["id"] == "second"
    assert [sample["id"] for sample in result["replies"][1]["value"]] == [
        "first",
        "second",
    ]
    assert [sample["value"] for sample in result["replies"][2]["value"]] == [
        2.0,
        5.0,
    ]
    assert result["replies"][3]["value"][0]["sample_id"] == "second"
    assert "XLENS_PARENT_SECRET" not in result["environment"]
    assert not requests
    assert events[0] == {"kind": "started", "message": "NOOA worker started"}
    assert [event["evidence"] for event in events[1:]] == [
        {"second": 1},
        {"first": 1, "second": 1},
        {"first": 1, "second": 1},
        {"second": 1},
    ]


@pytest.mark.parametrize(
    "frame",
    [
        query_frame("read_file", path="/etc/passwd"),
        query_frame("status", path="/etc/passwd"),
        query_frame("recent_findings", limit=21),
        query_frame("recent_findings", limit=True),
        query_frame("compare_samples", sample_ids=["second"] * 9),
        query_frame("domain_view", sample_id="second", query="unlisted"),
        query_frame("domain_view", sample_id="unknown", query="residual"),
    ],
)
def test_parent_broker_rejects_unavailable_queries(stdio_worker, frame):
    domain_calls = []
    result, requests, events = stdio_worker(
        [frame],
        domain_views={"residual": "Allowed view"},
        domain_query=lambda *args: domain_calls.append(args),
    )
    assert result["replies"] == [{"error": "unavailable evidence query"}]
    assert events[-1]["evidence"] == {}
    assert not domain_calls and not requests


@pytest.mark.parametrize(
    ("frames", "options", "diagnostic"),
    [
        ([model_frame(model="untrusted")], {}, "invalid NOOA model request"),
        (
            [model_frame(messages=["x" * 2048])],
            {"max_context_bytes": 1024},
            "byte limit",
        ),
        (
            [query_frame("compare_samples", sample_ids=["first", "second"])],
            {"max_context_bytes": 250},
            "byte limit",
        ),
        (
            [{"kind": "result", "answer": {"text": "x" * 2048}}],
            {"max_response_bytes": 1024},
            "byte limit",
        ),
        (
            [{"kind": "result", "answer": "not an object"}],
            {},
            "invalid NOOA answer",
        ),
        (
            [{"kind": "result", "answer": {"text": "x" * 8192}}],
            {"max_context_bytes": 1024, "max_response_bytes": 1024},
            "invalid NOOA worker message",
        ),
        ([model_frame()] * 3, {"max_iterations": 2}, "iteration limit"),
        ([query_frame("status")] * 2, {"max_tool_calls": 1}, "tool limit"),
        (
            [{"kind": "error", "message": "private generated output"}],
            {},
            "NOOA could not complete the investigation",
        ),
        ([{"kind": "unrecognized"}], {}, "unsupported NOOA worker message"),
    ],
)
def test_parent_broker_enforces_protocol_and_limits(
    stdio_worker, frames, options, diagnostic
):
    with pytest.raises(backend.NooaError, match=diagnostic):
        stdio_worker(frames, **options)


@pytest.mark.parametrize("total", [False, True])
def test_parent_broker_bounds_domain_results(stdio_worker, total):
    calls = []

    def query(*args):
        calls.append(args)
        return {"summary": "x" * (16000 if total else 16384)}

    with pytest.raises(
        backend.NooaError,
        match="domain evidence limit" if total else "byte limit",
    ):
        stdio_worker(
            [query_frame("domain_view", sample_id="second", query="residual")]
            * (5 if total else 1),
            domain_views={"residual": "Summary"},
            domain_query=query,
        )
    assert len(calls) == (5 if total else 1)


@pytest.mark.parametrize("calls", [None, [], "omitted"])
def test_parent_broker_accepts_text_only_turn(stdio_worker, calls):
    message = {"role": "assistant", "content": "Considering the evidence."}
    if calls != "omitted":
        message["tool_calls"] = calls
    result, requests, _ = stdio_worker(
        [model_frame()], model_response={"choices": [{"message": message}]}
    )
    assert len(requests) == 1
    forwarded = result["replies"][0]["response"]["choices"][0]["message"]
    assert forwarded == {**message, "tool_calls": []}


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("arguments", ['{"code": "print(1)"', "[]"])
def test_parent_broker_rejects_invalid_tool_arguments(
    stdio_worker, api, arguments
):
    model_response = response("execute_python", {})
    model_response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = arguments
    if api == "responses":
        from cuphoton.xlens._responses import ResponsesAdapter

        model_response = ResponsesAdapter().response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call",
                        "name": "execute_python",
                        "arguments": arguments,
                    }
                ],
            }
        )
    with pytest.raises(
        backend.NooaError, match="^model returned invalid tool arguments$"
    ):
        stdio_worker([model_frame()], model_response=model_response)


@pytest.mark.parametrize(
    "failure", ["name", "second_name", "id", "tool_limit", "response_bytes"]
)
def test_parent_broker_checks_model_output(stdio_worker, failure):
    model_response = response("execute_python", {"code": "pass"})
    calls = model_response["choices"][0]["message"]["tool_calls"]
    options = {}
    diagnostic = "invalid NOOA model response"
    if failure == "name":
        calls[0]["function"]["name"] = "forbidden_tool"
    elif failure == "second_name":
        calls.append(
            {
                "id": "second_call",
                "type": "function",
                "function": {"name": "forbidden_tool", "arguments": "{}"},
            }
        )
    elif failure == "id":
        calls[0]["id"] = 42
    elif failure == "tool_limit":
        calls.append({**calls[0], "id": "second_call"})
        options["max_tool_calls"] = 1
        diagnostic = "tool limit"
    else:
        model_response["metadata"] = "x" * 2048
        options["max_response_bytes"] = 1024
        diagnostic = "byte limit"
    with pytest.raises(backend.NooaError, match=diagnostic):
        stdio_worker(
            [model_frame()], model_response=model_response, **options
        )


@pytest.mark.parametrize("failure", ["eof", "initial_write", "reply_write"])
def test_worker_start_failure_is_consistent(monkeypatch, failure):
    class Pipe:
        writes = 0

        def write(self, data):
            self.writes += 1

        async def drain(self):
            if failure == "initial_write" or self.writes == 2:
                raise BrokenPipeError("private path or worker output")

    class Reader:
        async def readline(self):
            if failure == "reply_write":
                return json.dumps(query_frame("status")).encode() + b"\n"
            return b""

    class Process:
        stdin = Pipe()
        stdout = Reader()
        pid = 123

        async def wait(self):
            pass

    async def launch(*args, **kwargs):
        return Process()

    monkeypatch.setattr(
        backend, "_worker_command", lambda: nullcontext(([], ()))
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(os, "killpg", lambda *args: None)
    events = []
    with pytest.raises(backend.NooaError, match="^NOOA worker stopped;"):
        asyncio.run(run(lambda payload: None, events.append))
    assert bool(events) == (failure == "reply_write")


def test_namespace_probe_reports_bwrap_cause_without_host_paths(monkeypatch):
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 5 and kwargs["env"] == {}
        return subprocess.CompletedProcess(
            command,
            1,
            stderr=(
                b"unrelated output\nbwrap: /private/host/path: "
                b"setting up uid map: Permission denied\n"
                b"private follow-up line\n"
            ),
        )

    monkeypatch.setattr(subprocess, "run", probe)
    monkeypatch.setattr(backend, "seccomp_filter", lambda: nullcontext(123))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "version_info", (3, 12))
    monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.12")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")
    with pytest.raises(backend.NooaError) as caught:
        backend._command()
    assert str(caught.value) == (
        "NOOA isolation unavailable: bwrap: <path> "
        "setting up uid map: Permission denied"
    )
    assert calls[0][-1] == "/bin/true"
    assert {"--unshare-all", "--unshare-user", "--disable-userns"} <= set(
        calls[0]
    )
