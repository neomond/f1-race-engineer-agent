"""
Unit tests for Gemini client translation logic.

These test the pure translation functions (our format <-> Gemini format)
without making any API calls.
"""
from __future__ import annotations

import pytest

# Skip these tests if google-genai isn't installed yet
genai = pytest.importorskip("google.genai")

from src.agent.gemini_client import GeminiClient
from src.agent.llm_client import ToolCall, ToolResult, ToolSpec


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
    result = GeminiClient._tools_to_gemini([spec])
    assert len(result) == 1
    assert len(result[0].function_declarations) == 1
    assert result[0].function_declarations[0].name == "get_pace"


def test_tools_translation_empty():
    result = GeminiClient._tools_to_gemini([])
    assert result == []


def test_user_message_translation():
    messages = [{"role": "user", "content": "Hello"}]
    result = GeminiClient._messages_to_gemini(messages)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].parts[0].text == "Hello"


def test_assistant_text_translation():
    messages = [{"role": "assistant", "content": "Box this lap"}]
    result = GeminiClient._messages_to_gemini(messages)
    assert len(result) == 1
    assert result[0].role == "model"
    assert result[0].parts[0].text == "Box this lap"


def test_assistant_tool_call_translation():
    tc = ToolCall(
        name="get_pace",
        arguments={"driver": "NOR"},
        call_id="abc-123",
    )
    messages = [{"role": "assistant", "tool_calls": [tc]}]
    result = GeminiClient._messages_to_gemini(messages)
    assert len(result) == 1
    assert result[0].role == "model"
    assert result[0].parts[0].function_call.name == "get_pace"
    assert dict(result[0].parts[0].function_call.args) == {"driver": "NOR"}


def test_tool_result_translation():
    tr = ToolResult(call_id="abc-123", name="get_pace", content='{"pace": 103.5}')
    messages = [{"role": "tool", "tool_result": tr}]
    result = GeminiClient._messages_to_gemini(messages)
    assert len(result) == 1
    # Gemini uses 'user' role for function responses
    assert result[0].role == "user"
    assert result[0].parts[0].function_response.name == "get_pace"


def test_full_conversation_translation():
    tc = ToolCall(name="get_pace", arguments={"driver": "NOR"}, call_id="x")
    tr = ToolResult(call_id="x", name="get_pace", content='{"pace": 103.5}')
    messages = [
        {"role": "user", "content": "How is Norris doing?"},
        {"role": "assistant", "tool_calls": [tc]},
        {"role": "tool", "tool_result": tr},
        {"role": "assistant", "content": "Pace is strong, 103.5s."},
    ]
    result = GeminiClient._messages_to_gemini(messages)
    assert len(result) == 4
    assert [c.role for c in result] == ["user", "model", "user", "model"]
