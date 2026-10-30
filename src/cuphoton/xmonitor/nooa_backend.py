# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional NOOA execution with a credential-free Linux worker.

The application process owns the HTTP transport and authoritative evidence.
The worker receives only JSON and read-only Python runtime mounts. Bubblewrap
isolates its filesystem, processes and network; unavailable isolation fails
closed. Python restrictions alone are not a security boundary.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


class NooaError(RuntimeError):
    """A bounded worker diagnostic without remote content or host paths."""


def _json(value: Any, limit: int) -> bytes:
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > limit:
        raise NooaError("NOOA evidence or message exceeds its byte limit")
    return data


def _package_mounts() -> list[str]:
    """Mount declared NOOA dependencies, excluding installed applications.

    Read installed wheel metadata without importing packages or running .pth
    files. Shared third-party namespace roots are trusted runtime input too.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending: list[tuple[str, set[str]]] = [("nooa", set())]
    visited: dict[str, set[str]] = {}
    roots: dict[str, Path] = {}
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        if name in visited and extras <= visited[name]:
            continue
        extras |= visited.get(name, set())
        visited[name] = extras
        distribution = importlib.metadata.distribution(name)
        if distribution.files is None:
            raise NooaError(
                "NOOA requires installed dependency file metadata"
            )
        for entry in distribution.files:
            part = entry.parts[0]
            if part in {"..", "/"} or part.endswith(".pth"):
                continue
            root = Path(str(distribution.locate_file(part))).resolve()
            if root.exists():
                roots[part] = root
        for dependency in distribution.requires or []:
            requirement = Requirement(dependency)
            if requirement.marker is None or any(
                requirement.marker.evaluate({"extra": extra})
                for extra in {"", *extras}
            ):
                pending.append((requirement.name, set(requirement.extras)))
    mounts = ["--dir", "/packages"]
    for name, source in sorted(roots.items()):
        destination = "/packages/" + name
        if name.endswith(".dist-info"):
            # Version/dependency introspection needs only METADATA. Exclude
            # local install URLs and unrelated entry-point/plugin discovery.
            mounts.extend(
                (
                    "--dir",
                    destination,
                    "--ro-bind",
                    str(source / "METADATA"),
                    destination + "/METADATA",
                )
            )
        else:
            mounts.extend(("--ro-bind", str(source), destination))
    return mounts


def _command() -> list[str]:
    if sys.platform != "linux" or sys.version_info[:2] not in {
        (3, 12),
        (3, 13),
    }:
        raise NooaError("NOOA requires Linux and Python 3.12 or 3.13")
    try:
        version = importlib.metadata.version("nooa")
    except importlib.metadata.PackageNotFoundError as exc:
        raise NooaError("install cuphoton[xmonitor-nooa] for NOOA") from exc
    if version != "0.0.10":
        raise NooaError("the NOOA backend requires nooa==0.0.10")
    executable = shutil.which("bwrap")
    if not executable:
        raise NooaError("NOOA requires bubblewrap with user namespaces")
    prefix = Path(sys.base_prefix)
    command = [
        executable,
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--setenv",
        "HOME",
        "/work",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "LD_LIBRARY_PATH",
        "/runtime/lib",
        "--ro-bind",
        str(Path(sys.executable).resolve()),
        "/runtime/bin/python",
        "--ro-bind",
        str(prefix / "lib"),
        "/runtime/lib",
        "--ro-bind",
        str(Path(__file__).with_name("_nooa_worker.py")),
        "/worker.py",
    ]
    for directory in ("/usr/lib", "/lib", "/lib64"):
        if Path(directory).exists():
            command.extend(("--ro-bind", directory, directory))
    # The interpreter's library mount must not expose base-environment apps.
    for site_directory in (prefix / "lib").glob("python*/site-packages"):
        target = "/runtime/lib/" + str(
            site_directory.relative_to(prefix / "lib")
        )
        command.extend(("--tmpfs", target, "--remount-ro", target))
    try:
        command.extend(_package_mounts())
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise NooaError(
            "install the complete cuphoton[xmonitor-nooa] profile"
        ) from exc
    command.extend(
        (
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--size",
            "16777216",
            "--tmpfs",
            "/work",
            "--symlink",
            "/work",
            "/tmp",
            "--chdir",
            "/work",
            "/runtime/bin/python",
            "-I",
            "-S",
            "/worker.py",
        )
    )
    return command


def _query(evidence: dict[str, Any], name: str, args: Any) -> Any:
    """Read detached scalar evidence; never accept a path or expression."""
    samples = evidence["samples"]
    by_id = {sample["id"]: sample for sample in samples}
    if not isinstance(args, dict):
        raise ValueError("query arguments must be an object")
    if name == "status" and not args:
        return {
            "run_id": evidence["run_id"],
            "selected_ids": evidence.get("selected_ids", []),
            "samples": [
                {
                    key: sample[key]
                    for key in ("id", "sequence", "revision", "findings")
                }
                for sample in samples
            ],
        }
    if name == "get_sample" and set(args) == {"sample_id"}:
        return by_id[args["sample_id"]]
    if name == "compare_samples" and set(args) == {"sample_ids"}:
        ids = args["sample_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 8:
            raise ValueError("compare between one and eight samples")
        return [by_id[identifier] for identifier in ids]
    if name == "recent_findings" and set(args) == {"limit"}:
        limit = args["limit"]
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("findings limit must be between one and twenty")
        return [
            {
                "sample_id": sample["id"],
                "revision": sample["revision"],
                "finding": finding,
            }
            for sample in samples
            for finding in sample["findings"]
        ][-limit:]
    if name == "metric_series" and set(args) == {"measurement"}:
        measurement = args["measurement"]
        if not isinstance(measurement, str):
            raise ValueError("measurement requires a name")
        return [
            {
                "sample_id": sample["id"],
                "sequence": sample["sequence"],
                "revision": sample["revision"],
                "observed_start": sample.get("observed_start"),
                "observed_end": sample.get("observed_end"),
                "time_unit": sample.get("time_unit", "s"),
                "measurement": measurement,
                **sample["measurements"][measurement],
            }
            for sample in samples
            if measurement in sample["measurements"]
        ]
    raise ValueError("unsupported evidence query")


async def run_nooa(
    evidence: dict[str, Any],
    question: str,
    instructions: str,
    *,
    request: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    progress: Callable[[dict[str, Any]], None],
    max_iterations: int,
    max_tool_calls: int,
    timeout_seconds: float,
    max_context_bytes: int,
    max_response_bytes: int,
    domain_query: Callable[[str, str], dict[str, Any]] | None = None,
    domain_views: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run real iterative CodeAct; validate final citations in the caller.

    ``request`` receives messages and tools only. Its parent-owned transport
    must supply the configured endpoint, model, authentication and token cap.
    Tool calls and evidence reads share a finite budget. Callbacks are trusted
    application code; domain callbacks must return promptly and perform only
    bounded reads. Cancellation kills the isolated session and descendants.
    """
    command = _command()
    # A serialization round trip prevents mutation of live caller records.
    evidence = json.loads(_json(evidence, 16 * 1024 * 1024))
    views = domain_views or {}
    initial = {
        "question": question,
        "instructions": instructions,
        "selected_ids": evidence.get("selected_ids", []),
        "domain_views": views,
        "max_iterations": max_iterations,
        "max_tool_calls": max_tool_calls,
        "max_context_bytes": max_context_bytes,
        "timeout_seconds": timeout_seconds,
    }
    initial_bytes = _json(initial, max_context_bytes)
    frame_limit = max(max_context_bytes, max_response_bytes) + 4096
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={},
        start_new_session=True,
        limit=frame_limit,
    )
    assert process.stdin is not None and process.stdout is not None
    iterations = 0
    tool_calls = 0
    domain_bytes = 0
    try:
        async with asyncio.timeout(timeout_seconds):
            process.stdin.write(initial_bytes + b"\n")
            await process.stdin.drain()
            progress({"kind": "started", "message": "NOOA worker started"})
            while True:
                try:
                    line = await process.stdout.readline()
                    if not line:
                        raise NooaError(
                            "NOOA worker stopped; verify bubblewrap user "
                            "namespace support and the optional installation"
                        )
                    if len(line) > frame_limit:
                        raise ValueError("oversized frame")
                    frame = json.loads(line)
                    if not isinstance(frame, dict):
                        raise ValueError("invalid frame")
                except (ValueError, UnicodeError) as exc:
                    raise NooaError("invalid NOOA worker message") from exc
                kind = frame.get("kind")
                reply: dict[str, Any]
                if kind == "result":
                    answer = frame.get("answer")
                    _json(answer, max_response_bytes)
                    if not isinstance(answer, dict):
                        raise NooaError("invalid NOOA answer")
                    return answer
                if kind == "error":
                    raise NooaError(
                        "NOOA could not complete the investigation"
                    )
                if kind == "model":
                    iterations += 1
                    if iterations > max_iterations:
                        raise NooaError("NOOA iteration limit reached")
                    payload = frame.get("payload")
                    if not isinstance(payload, dict) or set(payload) != {
                        "messages",
                        "tools",
                    }:
                        raise NooaError("invalid NOOA model request")
                    _json(payload, max_context_bytes)
                    progress(
                        {
                            "kind": "model",
                            "message": f"NOOA reasoning turn {iterations}",
                        }
                    )
                    response = await request(payload)
                    _json(response, max_response_bytes)
                    try:
                        choice = response["choices"][0]
                        message = choice["message"]
                        calls = message.get("tool_calls", [])
                        if not isinstance(calls, list):
                            raise ValueError("invalid tool calls")
                        tool_calls += len(calls)
                        if tool_calls > max_tool_calls:
                            raise NooaError("NOOA tool limit reached")
                        clean_calls = []
                        for call in calls:
                            name = call["function"]["name"]
                            if name not in {
                                "execute_python",
                                "return_result",
                            }:
                                raise ValueError("unsupported tool")
                            if not isinstance(
                                call["id"], str
                            ) or not isinstance(
                                call["function"]["arguments"], str
                            ):
                                raise ValueError("invalid tool fields")
                            clean_calls.append(
                                {
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": call["function"][
                                            "arguments"
                                        ],
                                    },
                                }
                            )
                            progress(
                                {
                                    "kind": "tool",
                                    "tool": name,
                                    "message": f"NOOA selected {name}",
                                }
                            )
                    except (
                        KeyError,
                        IndexError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise NooaError(
                            "invalid NOOA model response"
                        ) from exc
                    # Provider metadata can repeat the configured model ID.
                    # Only the actual assistant message crosses the broker.
                    reply = {
                        "response": {
                            "choices": [
                                {
                                    "finish_reason": choice.get(
                                        "finish_reason", "stop"
                                    ),
                                    "message": {
                                        "role": "assistant",
                                        "content": message.get("content"),
                                        "tool_calls": clean_calls,
                                    },
                                }
                            ]
                        }
                    }
                elif kind == "query":
                    tool_calls += 1
                    if tool_calls > max_tool_calls:
                        raise NooaError("NOOA tool limit reached")
                    name, args = frame.get("name"), frame.get("args")
                    if not isinstance(name, str):
                        raise NooaError("invalid NOOA query name")
                    references: dict[str, int] = {}
                    try:
                        if name == "domain_view":
                            if not isinstance(args, dict) or set(args) != {
                                "sample_id",
                                "query",
                            }:
                                raise ValueError("invalid domain query")
                            if (
                                domain_query is None
                                or args["query"] not in views
                                or args["sample_id"]
                                not in {s["id"] for s in evidence["samples"]}
                            ):
                                raise ValueError("unavailable domain view")
                            result = domain_query(
                                args["sample_id"], args["query"]
                            )
                            domain_bytes += len(_json(result, 16384))
                            if domain_bytes > 65536:
                                raise NooaError(
                                    "NOOA domain evidence limit reached"
                                )
                            identifiers = [args["sample_id"]]
                        else:
                            result = _query(evidence, name, args)
                            if name == "status":
                                identifiers = [
                                    s["id"] for s in result["samples"]
                                ]
                            elif name == "get_sample":
                                identifiers = [result["id"]]
                            elif name == "compare_samples":
                                identifiers = [s["id"] for s in result]
                            else:
                                identifiers = [s["sample_id"] for s in result]
                        _json(result, max_context_bytes)
                        references = {
                            s["id"]: s["revision"]
                            for s in evidence["samples"]
                            if s["id"] in identifiers
                        }
                        reply = {"value": result}
                    except (ValueError, TypeError, KeyError):
                        reply = {"error": "unavailable evidence query"}
                    progress(
                        {
                            "kind": "evidence",
                            "tool": str(name)[:80],
                            "message": "Evidence query completed"
                            if "value" in reply
                            else "Evidence query unavailable",
                            "evidence": references,
                        }
                    )
                else:
                    raise NooaError("unsupported NOOA worker message")
                if tool_calls > max_tool_calls:
                    raise NooaError("NOOA tool limit reached")
                process.stdin.write(_json(reply, frame_limit) + b"\n")
                await process.stdin.drain()
    except TimeoutError as exc:
        raise NooaError("NOOA investigation deadline exceeded") from exc
    finally:
        # SIGKILL is deliberate: no generated cleanup code runs after cancel.
        # The private PID namespace also contains reparented descendants.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
