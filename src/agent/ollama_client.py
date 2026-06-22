"""
Ollama implementation of the LLMClient interface.

Connects to a local Ollama server (default: http://localhost:11434) and
translates our provider-agnostic types to Ollama's chat API format.

This file proves the LLMClient abstraction does its job — same agent loop,
same tools, same memory, completely different model running locally with
no API keys, no rate limits, no network dependency.

Recommended models for this project:
  - qwen3:8b      — best balance of quality and speed on a typical Mac
  - qwen3:4b      — faster if 8b feels slow; still does tool calling well
  - llama3.2:3b   — very fast, weaker reasoning

All of these support function calling via Ollama's tool API.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from ollama import Client as OllamaSDK

from src.agent.llm_client import (
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolSpec,
)

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """LLM client backed by a local Ollama server."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.3,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.client = OllamaSDK(host=self.host)

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        ollama_messages = self._build_messages(system_prompt, messages)
        ollama_tools = self._tools_to_ollama(tools)

        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            tools=ollama_tools if ollama_tools else None,
            options={"temperature": self.temperature},
            # Qwen 3 supports an explicit "think" mode; we leave the default
            # but turn off thinking to keep output focused.
            think=False,
        )

        return self._parse_response(response)

    # ---- Translation helpers ---------------------------------------------

    @staticmethod
    def _tools_to_ollama(tools: list[ToolSpec]) -> list[dict]:
        """Convert our ToolSpecs to Ollama's tool JSON schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _build_messages(
        system_prompt: str, messages: list[dict[str, Any]]
    ) -> list[dict]:
        """
        Translate our message format to Ollama's chat format.

        Our format:
          {"role": "user", "content": "text"}
          {"role": "assistant", "content": "text"} or
          {"role": "assistant", "tool_calls": [ToolCall, ...]}
          {"role": "tool", "tool_result": ToolResult}

        Ollama format:
          {"role": "system", "content": "..."}
          {"role": "user", "content": "..."}
          {"role": "assistant", "content": "..."} or with "tool_calls"
          {"role": "tool", "content": "...", "tool_name": "..."}
        """
        out: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"]
            if role == "user":
                out.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                if "tool_calls" in msg:
                    out.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in msg["tool_calls"]
                        ],
                    })
                else:
                    out.append({"role": "assistant", "content": msg["content"]})
            elif role == "tool":
                tr = msg["tool_result"]
                out.append({
                    "role": "tool",
                    "content": tr.content,
                    "tool_name": tr.name,
                })
        return out

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Extract text and/or tool calls from Ollama's response."""
        message = response.message if hasattr(response, "message") else response.get("message", {})

        # Tool calls
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls is None and isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls")

        tool_calls: list[ToolCall] = []
        if raw_tool_calls:
            for tc in raw_tool_calls:
                # tc can be an object or a dict depending on SDK version
                if hasattr(tc, "function"):
                    name = tc.function.name
                    args = dict(tc.function.arguments) if tc.function.arguments else {}
                else:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {}) or {}
                tool_calls.append(ToolCall(
                    name=name,
                    arguments=args,
                    call_id=str(uuid.uuid4()),
                ))

        # Text content
        text = getattr(message, "content", None)
        if text is None and isinstance(message, dict):
            text = message.get("content")
        text = (text or "").strip() or None

        return LLMResponse(text=text, tool_calls=tool_calls, raw=response)
