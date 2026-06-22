"""
Unit tests for the Ollama client translation logic.

Pure translation — no API calls. Verifies our format <-> Ollama format.
"""
from __future__ import annotations

import pytest

ollama = pytest.importorskip("ollama")

from src.agent.llm_client import ToolCall, ToolResult, ToolSpec
from src.agent.ollama_client import OllamaClient


def test_tools_translation():
    spec = ToolSpec(
        name="get_pace",
        description="Get pace data",
        parameters={
            "type": "object",
            "properties": {"driver": {"type": "string"}},
            "required": ["driver"],
        },
    )
    result = OllamaClient._tools_to_ollama([spec])
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_pace"
    assert result[0]["function"]["description"] == "Get pace data"


def test_tools_translation_empty():
    assert OllamaClient._tools_to_ollama([]) == []


def test_messages_include_system_prompt():
    messages = [{"role": "user", "content": "Hi"}]
    result = OllamaClient._build_messages("You are a helper.", messages)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a helper."
    assert result[1]["role"] == "user"


def test_assistant_text_translation():
    messages = [{"role": "assistant", "content": "Box this lap"}]
    result = OllamaClient._build_messages("S", messages)
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Box this lap"


def test_assistant_tool_call_translation():
    tc = ToolCall(name="get_pace", arguments={"driver": "NOR"}, call_id="x")
    messages = [{"role": "assistant", "tool_calls": [tc]}]
    result = OllamaClient._build_messages("S", messages)
    assert result[1]["role"] == "assistant"
    assert "tool_calls" in result[1]
    assert result[1]["tool_calls"][0]["function"]["name"] == "get_pace"
    assert result[1]["tool_calls"][0]["function"]["arguments"] == {"driver": "NOR"}


def test_tool_result_translation():
    tr = ToolResult(call_id="x", name="get_pace", content='{"pace": 103.5}')
    messages = [{"role": "tool", "tool_result": tr}]
    result = OllamaClient._build_messages("S", messages)
    assert result[1]["role"] == "tool"
    assert result[1]["content"] == '{"pace": 103.5}'
    assert result[1]["tool_name"] == "get_pace"


def test_parse_response_text_only():
    """Parse a response that's just text, no tool calls."""
    # Ollama SDK returns objects; we accept both objects and dicts
    response = type("R", (), {
        "message": type("M", (), {
            "content": "Box this lap.",
            "tool_calls": None,
        })()
    })()
    parsed = OllamaClient._parse_response(response)
    assert parsed.text == "Box this lap."
    assert parsed.tool_calls == []


def test_parse_response_tool_call():
    """Parse a response containing a tool call."""
    fn = type("F", (), {"name": "stay_silent", "arguments": {"reason": "ok"}})()
    tc = type("TC", (), {"function": fn})()
    response = type("R", (), {
        "message": type("M", (), {"content": "", "tool_calls": [tc]})()
    })()
    parsed = OllamaClient._parse_response(response)
    assert parsed.text is None
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "stay_silent"
    assert parsed.tool_calls[0].arguments == {"reason": "ok"}


def test_parse_response_dict_format():
    """Some SDK versions return dicts instead of objects — handle both."""
    response = {
        "message": {
            "content": "OK",
            "tool_calls": None,
        }
    }
    # Wrap in object-like to match the .message attr access
    response_obj = type("R", (), {"message": response["message"]})()
    parsed = OllamaClient._parse_response(response_obj)
    assert parsed.text == "OK"
