# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from cuphoton.xlens._responses import ResponsesAdapter


def payload(messages, **kwargs):
    return {
        "model": "configured-model",
        "messages": messages,
        "max_tokens": 4096,
        **kwargs,
    }


def tool_result(call_id="call_1"):
    return {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "opaque-reasoning",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": call_id,
                "name": "execute_python",
                "arguments": json.dumps(
                    {"code": 'self.get_sample("sample-1")'}
                ),
                "status": "completed",
            },
        ],
    }


def text_result(text):
    return {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_text",
                "summary": [],
                "encrypted_content": "opaque-final-reasoning",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
    }


def test_multiturn_tools_preserve_opaque_outputs():
    adapter = ResponsesAdapter()
    history = [
        {"role": "system", "content": "Investigate measurements."},
        {"role": "user", "content": "Explain sample-1."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Inspect evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]
    request = adapter.request(payload(history, tools=tools))
    assert request["model"] == "configured-model"
    assert request["max_output_tokens"] == 4096
    assert request["store"] is False
    assert request["stream"] is False
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["tools"][0] == {
        "type": "function",
        **tools[0]["function"],
        "strict": False,
    }
    first = tool_result()
    chat = adapter.response(first)
    assert chat["choices"][0]["finish_reason"] == "tool_calls"
    history.extend(
        [
            chat["choices"][0]["message"],
            {"role": "tool", "tool_call_id": "call_1", "content": "2.5"},
        ]
    )
    second_request = adapter.request(payload(history, tools=tools))
    assert second_request["input"][2:4] == first["output"]
    assert second_request["input"][4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "2.5",
    }
    second = tool_result("call_2")
    history.extend(
        [
            adapter.response(second)["choices"][0]["message"],
            {"role": "tool", "tool_call_id": "call_2", "content": "3.0"},
        ]
    )
    third_request = adapter.request(payload(history, tools=tools))
    assert third_request["input"][2:4] == first["output"]
    assert third_request["input"][5:7] == second["output"]
    final = adapter.response(text_result('{"narrative":"Inspect flux."}'))
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["choices"][0]["message"]["content"] == (
        '{"narrative":"Inspect flux."}'
    )


def test_text_retry_preserves_reasoning_and_phase_and_translates_json_mode():
    adapter = ResponsesAdapter()
    history = [{"role": "user", "content": "Return JSON."}]
    request = adapter.request(
        payload(
            history,
            max_completion_tokens=8192,
            response_format={"type": "json_object"},
        )
    )
    assert request["max_output_tokens"] == 8192
    assert request["text"] == {"format": {"type": "json_object"}}
    raw = text_result("The answer.")
    history.append(adapter.response(raw)["choices"][0]["message"])
    history.append({"role": "user", "content": "Please return JSON."})
    expected = deepcopy(raw["output"])
    raw["output"][0]["encrypted_content"] = "mutated by caller"
    assert adapter.request(payload(history))["input"][1:3] == expected


def test_investigations_cannot_reuse_each_others_tool_history():
    first = ResponsesAdapter()
    history = [first.response(tool_result())["choices"][0]["message"]]
    history.append(
        {"role": "tool", "tool_call_id": "call_1", "content": "2.5"}
    )
    with pytest.raises(ValueError, match="invalid Responses request history"):
        ResponsesAdapter().request(payload(history))
    first.request(payload(history))
    # Dropping a turn from the transcript also drops its opaque cached data.
    first.request(payload([{"role": "user", "content": "New question."}]))
    with pytest.raises(ValueError, match="invalid Responses request history"):
        first.request(payload(history))


def test_nooa_initial_prefill_is_preserved_without_fabricating_later_turns():
    adapter = ResponsesAdapter()
    initial = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "prefill_0123abcd",
                    "type": "function",
                    "function": {
                        "name": "execute_python",
                        "arguments": '{"code":"print(selected_ids)"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "prefill_0123abcd",
            "content": "status: accepted",
        },
        {"role": "user", "content": "PythonOutput: ['sample-1']"},
    ]
    expected = [
        {
            "type": "function_call",
            "call_id": "prefill_0123abcd",
            "name": "execute_python",
            "arguments": '{"code":"print(selected_ids)"}',
        },
        {
            "type": "function_call_output",
            "call_id": "prefill_0123abcd",
            "output": "status: accepted",
        },
        initial[-1],
    ]
    assert adapter.request(payload(initial))["input"] == expected
    provider = tool_result()
    continued = initial + [
        adapter.response(provider)["choices"][0]["message"],
        {"role": "tool", "tool_call_id": "call_1", "content": "2.5"},
    ]
    request = adapter.request(payload(continued))
    assert request["input"][:3] == expected
    assert request["input"][3:5] == provider["output"]
    changed = deepcopy(continued)
    changed[0]["tool_calls"][0]["function"]["arguments"] = '{"code":"other"}'
    with pytest.raises(ValueError, match="invalid Responses request history"):
        adapter.request(payload(changed))


@pytest.mark.parametrize(
    "raw",
    [
        {"status": "incomplete", "output": tool_result()["output"]},
        {"status": "failed", "error": {"message": "private detail"}},
        {"status": "completed", "output": []},
        {"status": "completed", "output": [None]},
        {"status": "completed", "output": [{"type": "web_search_call"}]},
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "private"}],
                }
            ],
        },
    ],
)
def test_incomplete_unsupported_and_refused_outputs_have_fixed_errors(raw):
    with pytest.raises(ValueError) as error:
        ResponsesAdapter().response(raw)
    assert str(error.value) == "invalid or incomplete Responses result"


@pytest.mark.parametrize(
    "change",
    [
        {"call_id": ""},
        {"name": []},
        {"arguments": "not JSON"},
        {"arguments": "[]"},
        {"status": "in_progress"},
    ],
)
def test_malformed_tool_calls_are_rejected_before_reaching_worker(change):
    raw = tool_result()
    raw["output"][-1].update(change)
    with pytest.raises(ValueError, match="invalid or incomplete"):
        ResponsesAdapter().response(raw)


def test_duplicate_calls_and_mismatched_outputs_are_rejected():
    raw = tool_result()
    raw["output"].append(deepcopy(raw["output"][-1]))
    with pytest.raises(ValueError, match="invalid or incomplete"):
        ResponsesAdapter().response(raw)
    adapter = ResponsesAdapter()
    message = adapter.response(tool_result())["choices"][0]["message"]
    with pytest.raises(ValueError, match="invalid Responses request history"):
        adapter.request(payload([message]))
    with pytest.raises(ValueError, match="invalid Responses request history"):
        adapter.request(
            payload(
                [
                    message,
                    {"role": "tool", "tool_call_id": "wrong", "content": "x"},
                ]
            )
        )
