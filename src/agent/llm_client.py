"""
LLM client abstraction.

Defines a provider-agnostic interface for tool-using language models.
Concrete implementations (Gemini, Claude, local models) plug in behind
this interface so the agent loop never needs to change.

Design: keep the surface area small. We need:
  - send a list of messages + tool definitions
  - get back either text, or a list of tool calls to execute
  - feed tool results back in and continue
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    name: str
    arguments: dict[str, Any]
    call_id: str  # provider-specific ID for matching results back to calls


@dataclass
class ToolResult:
    """Result of executing a tool, to be sent back to the model."""
    call_id: str
    name: str
    content: str  # JSON-serialised result


@dataclass
class LLMResponse:
    """A single turn from the model."""
    text: Optional[str]
    tool_calls: list[ToolCall]
    raw: Any = None  # provider-specific original response, for debugging

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolSpec:
    """Provider-agnostic tool definition."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class LLMClient(ABC):
    """Abstract LLM client. Subclass per provider."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        """
        Single generation step.

        messages: list of {"role": "user"|"assistant"|"tool", "content": ...}
                  where tool messages are ToolResult objects serialised.
        Returns a LLMResponse with either text (final answer) or tool_calls
        (model wants to invoke tools).
        """
        ...
