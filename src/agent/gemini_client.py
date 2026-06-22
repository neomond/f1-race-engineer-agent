"""
Gemini implementation of the LLMClient interface.

Uses the google-genai SDK (the new one, not the legacy google-generativeai).
Translates our provider-agnostic types to Gemini's specific format and back.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from google import genai
from google.genai import types as gtypes
from google.genai import errors as genai_errors

from src.agent.llm_client import (
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolSpec,
)

logger = logging.getLogger(__name__)


# Retry config for rate-limited / transient failures
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 8.0


class GeminiClient(LLMClient):
    """LLM client backed by Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it to .env or pass api_key explicitly."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        contents = self._messages_to_gemini(messages)
        gemini_tools = self._tools_to_gemini(tools)

        config = gtypes.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            tools=gemini_tools if gemini_tools else None,
        )

        # Retry loop for rate limits / transient failures
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return self._parse_response(response)
            except genai_errors.ClientError as e:
                # 429 = rate limit; some 400s are misreported rate limits
                msg = str(e).lower()
                is_rate_limit = (
                    "429" in msg
                    or "rate" in msg
                    or "quota" in msg
                    or "resource_exhausted" in msg
                )
                if is_rate_limit and attempt < MAX_RETRIES:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}), "
                        f"backing off {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise

    # ---- Translation helpers ---------------------------------------------

    @staticmethod
    def _tools_to_gemini(tools: list[ToolSpec]) -> list[gtypes.Tool]:
        if not tools:
            return []
        function_decls = [
            gtypes.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in tools
        ]
        return [gtypes.Tool(function_declarations=function_decls)]

    @staticmethod
    def _messages_to_gemini(messages: list[dict[str, Any]]) -> list[gtypes.Content]:
        """
        Translate our message format to Gemini Contents.

        Our format:
          {"role": "user", "content": "text"}
          {"role": "assistant", "content": "text"} or
          {"role": "assistant", "tool_calls": [ToolCall, ...]}
          {"role": "tool", "tool_result": ToolResult}
        """
        contents: list[gtypes.Content] = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                contents.append(gtypes.Content(
                    role="user",
                    parts=[gtypes.Part(text=msg["content"])],
                ))
            elif role == "assistant":
                if "tool_calls" in msg:
                    parts = [
                        gtypes.Part(function_call=gtypes.FunctionCall(
                            name=tc.name,
                            args=tc.arguments,
                        ))
                        for tc in msg["tool_calls"]
                    ]
                    contents.append(gtypes.Content(role="model", parts=parts))
                else:
                    contents.append(gtypes.Content(
                        role="model",
                        parts=[gtypes.Part(text=msg["content"])],
                    ))
            elif role == "tool":
                tr = msg["tool_result"]
                contents.append(gtypes.Content(
                    role="user",  # Gemini uses 'user' for tool responses
                    parts=[gtypes.Part(function_response=gtypes.FunctionResponse(
                        name=tr.name,
                        response={"result": tr.content},
                    ))],
                ))
        return contents

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Extract text and/or tool calls from Gemini's response.

        We deduplicate text parts to avoid Gemini occasionally emitting the
        same content multiple times (intermediate thinking + final answer).
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if not response.candidates:
            return LLMResponse(text=None, tool_calls=[], raw=response)

        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCall(
                    name=fc.name,
                    arguments=args,
                    call_id=str(uuid.uuid4()),
                ))
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text.strip())

        # Deduplicate identical text parts while preserving order.
        # Gemini sometimes emits the same answer twice when reasoning + final.
        seen = set()
        unique_parts = []
        for p in text_parts:
            if p and p not in seen:
                seen.add(p)
                unique_parts.append(p)

        return LLMResponse(
            text="\n".join(unique_parts) if unique_parts else None,
            tool_calls=tool_calls,
            raw=response,
        )
