# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Translate the bounded xLens chat contract to stateless Responses calls."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def _assistant(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("invalid assistant text")
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ValueError("invalid assistant tools")
    clean = []
    for call in calls:
        function = call["function"]
        # The shared broker validates argument JSON for both APIs.
        if (
            call.get("type") != "function"
            or not isinstance(call["id"], str)
            or not call["id"]
            or not isinstance(function["name"], str)
            or not function["name"]
            or not isinstance(function["arguments"], str)
        ):
            raise ValueError("invalid assistant tool call")
        clean.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": function["arguments"],
                },
            }
        )
    if len({call["id"] for call in clean}) != len(clean):
        raise ValueError("duplicate assistant tool call")
    return {"content": content or None, "tool_calls": clean}


class ResponsesAdapter:
    """Own the opaque provider history for exactly one investigation.

    The caller bounds each incoming response and the final outgoing payload.
    Only turns still present in the worker's transcript are retained. Keeping
    the original output items preserves encrypted reasoning and message phase
    fields without exposing either to the generated-code worker.
    """

    def __init__(self) -> None:
        self._turns: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self._started = False

    def request(self, chat_payload: dict[str, Any]) -> dict[str, Any]:
        """Translate messages and replay the original provider outputs."""
        try:
            return self._request(chat_payload)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError("invalid Responses request history") from exc

    def _request(self, chat: dict[str, Any]) -> dict[str, Any]:
        messages = chat["messages"]
        if not isinstance(messages, list):
            raise ValueError("invalid messages")
        items: list[dict[str, Any]] = []
        retained = []
        cursor = 0
        pending: set[str] = set()
        seen: set[str] = set()
        for message in messages:
            role = message["role"]
            content = message.get("content")
            if role == "assistant":
                normalized = _assistant(message)
                match = next(
                    (
                        index
                        for index in range(cursor, len(self._turns))
                        if self._turns[index][0] == normalized
                    ),
                    None,
                )
                if match is not None:
                    turn = self._turns[match]
                    items.extend(deepcopy(turn[1]))
                    retained.append(turn)
                    cursor = match + 1
                elif normalized["tool_calls"]:
                    # NOOA 0.0.12 executes input introspection before its
                    # first model call. These synthetic calls have no
                    # provider history to replay. Translate only that
                    # initial prefill; all later calls require cached
                    # provider items, including opaque reasoning.
                    if (
                        self._started
                        or self._turns
                        or normalized["content"]
                        or not all(
                            re.fullmatch(r"prefill_[0-9a-f]{8}", call["id"])
                            and call["function"]["name"] == "execute_python"
                            for call in normalized["tool_calls"]
                        )
                    ):
                        raise ValueError("unavailable Responses tool history")
                    prefill = [
                        {
                            "type": "function_call",
                            "call_id": call["id"],
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        }
                        for call in normalized["tool_calls"]
                    ]
                    items.extend(deepcopy(prefill))
                    retained.append((deepcopy(normalized), prefill))
                else:
                    items.append({"role": role, "content": content or ""})
                for call in normalized["tool_calls"]:
                    if call["id"] in seen:
                        raise ValueError("duplicate tool call")
                    seen.add(call["id"])
                    pending.add(call["id"])
            elif role == "tool":
                call_id = message["tool_call_id"]
                if call_id not in pending or not isinstance(content, str):
                    raise ValueError("unmatched tool output")
                pending.remove(call_id)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": content,
                    }
                )
            elif role in {"system", "developer", "user"} and isinstance(
                content, str
            ):
                items.append({"role": role, "content": content})
            else:
                raise ValueError("unsupported message")
        if pending:
            raise ValueError("missing tool output")
        payload: dict[str, Any] = {
            "model": chat["model"],
            "input": items,
            "max_output_tokens": chat.get(
                "max_completion_tokens", chat.get("max_tokens")
            ),
            "stream": False,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if "tools" in chat:
            if not isinstance(chat["tools"], list):
                raise ValueError("invalid tools")
            tools = []
            for tool in chat["tools"]:
                function = tool["function"]
                if (
                    tool["type"] != "function"
                    or not isinstance(function["parameters"], dict)
                    or not isinstance(function["name"], str)
                    or not function["name"]
                    or not isinstance(function.get("description", ""), str)
                ):
                    raise ValueError("unsupported tool")
                tools.append(
                    {
                        "type": "function",
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "parameters": deepcopy(function["parameters"]),
                        "strict": False,
                    }
                )
            payload["tools"] = tools
        if "response_format" in chat:
            if chat["response_format"] != {"type": "json_object"}:
                raise ValueError("unsupported output format")
            payload["text"] = {"format": {"type": "json_object"}}
        self._turns = retained
        self._started = True
        return payload

    def response(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a completed provider response for the existing broker."""
        try:
            return self._response(raw)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError(
                "invalid or incomplete Responses result"
            ) from exc

    def _response(self, raw: dict[str, Any]) -> dict[str, Any]:
        if raw.get("status") != "completed":
            raise ValueError("response did not complete")
        output = raw["output"]
        if not isinstance(output, list):
            raise ValueError("invalid output")
        text: list[str] = []
        calls = []
        for item in output:
            kind = item["type"]
            if kind == "reasoning":
                continue
            if kind == "function_call":
                if item.get("status", "completed") != "completed":
                    raise ValueError("tool call did not complete")
                calls.append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"],
                        },
                    }
                )
            elif kind == "message":
                if (
                    item.get("role") != "assistant"
                    or item.get("status", "completed") != "completed"
                    or not isinstance(item["content"], list)
                ):
                    raise ValueError("invalid assistant message")
                for part in item["content"]:
                    if part["type"] != "output_text" or not isinstance(
                        part["text"], str
                    ):
                        raise ValueError("unsupported output or refusal")
                    text.append(part["text"])
            else:
                raise ValueError("unsupported provider output")
        normalized = _assistant(
            {"content": "".join(text) or None, "tool_calls": calls}
        )
        if not normalized["content"] and not normalized["tool_calls"]:
            raise ValueError("empty result")
        self._turns.append((deepcopy(normalized), deepcopy(output)))
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": {"role": "assistant", **normalized},
                }
            ]
        }
