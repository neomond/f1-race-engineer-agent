"""
Unit tests for the RaceEngineerAgent loop.

We mock the LLM client with scripted responses to verify each control flow
path: direct text, tool-then-text, immediate stay_silent, tool-then-silent,
iteration cap, and tool dispatch correctness.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.agent.llm_client import LLMClient, LLMResponse, ToolCall
from src.agent.race_engineer import RaceEngineerAgent
from src.data.lap_context import LapContext, GapInfo


def make_ctx(lap=10):
    return LapContext(
        lap_number=lap,
        total_laps=51,
        laps_remaining=51 - lap,
        driver_code="NOR",
        driver_name="Lando Norris",
        position=5,
        lap_time_seconds=103.5,
        last_3_lap_times=[103.5, 103.6, 103.5],
        tyre_compound="MEDIUM",
        tyre_age_laps=9,
        stint_number=1,
        in_pit=False,
        ahead=GapInfo("PIA", "Oscar Piastri", -1.5),
        behind=GapInfo("LEC", "Charles Leclerc", 2.0),
        track_temp_c=42.0,
        air_temp_c=28.0,
        rainfall=False,
        track_status="green",
    )


class ScriptedLLM(LLMClient):
    """LLM that returns a pre-scripted list of responses, one per call."""
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, system_prompt, messages, tools):
        self.calls.append({
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": tools,
        })
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


def _text(s: str) -> LLMResponse:
    return LLMResponse(text=s, tool_calls=[], raw=None)


def _tool(name: str, args: dict, call_id="x") -> LLMResponse:
    return LLMResponse(
        text=None,
        tool_calls=[ToolCall(name=name, arguments=args, call_id=call_id)],
        raw=None,
    )


def make_agent(scripted: list[LLMResponse]):
    """Build an agent with a scripted LLM and a mocked RaceData."""
    race = MagicMock()
    race.meta.pit_loss_seconds = 22.0
    race.pit_loss_estimate.source = "measured"
    llm = ScriptedLLM(scripted)
    agent = RaceEngineerAgent(llm=llm, race=race)
    return agent, llm, race


# ---- Tests ----------------------------------------------------------------


def test_direct_text_radio_call():
    """Model returns text immediately → that text is the radio call."""
    agent, llm, _ = make_agent([_text("Box this lap, undercut window.")])

    call = agent.step(make_ctx(lap=15))

    assert call.message == "Box this lap, undercut window."
    assert call.silent_reason is None
    assert call.tools_used == []
    assert call.lap_number == 15


def test_stay_silent_tool_path():
    """Model calls stay_silent → call.message is None, reason recorded."""
    agent, llm, _ = make_agent([
        _tool("stay_silent", {"reason": "pace stable, no decisions pending"}),
    ])

    call = agent.step(make_ctx(lap=10))

    assert call.message is None
    assert call.silent_reason == "pace stable, no decisions pending"
    assert "stay_silent" in call.tools_used


def test_tool_then_text():
    """Model calls compare_pace, gets result, then produces a radio call."""
    agent, llm, race = make_agent([
        _tool("compare_pace", {
            "driver_a": "NOR", "driver_b": "PIA", "mode": "last_n",
        }),
        _text("Matching Piastri's pace, gap stable. Continue."),
    ])

    # Stub the tool registry dispatch
    from src.tools import registry
    registry.TOOL_HANDLERS["compare_pace"] = lambda r, l, a: json.dumps({
        "avg_lap_delta_seconds": 0.05, "trend_direction": "stable",
    })

    call = agent.step(make_ctx(lap=20))

    assert call.message == "Matching Piastri's pace, gap stable. Continue."
    assert call.tools_used == ["compare_pace"]
    assert len(llm.calls) == 2  # one for tool decision, one for final text


def test_iteration_cap_handled():
    """If model keeps calling tools beyond the cap, return silent fallback."""
    # 5 tool calls in a row, but cap is 4 → after 4 iterations we bail
    excess = [
        _tool("compare_pace", {"driver_a": "NOR", "driver_b": "PIA", "mode": "last_n"})
        for _ in range(10)
    ]
    agent, llm, _ = make_agent(excess)

    from src.tools import registry
    registry.TOOL_HANDLERS["compare_pace"] = lambda r, l, a: '{"ok": true}'

    call = agent.step(make_ctx(lap=15))

    assert call.message is None
    assert "cap" in call.silent_reason


def test_memory_records_lap_and_call():
    """After step(), memory should contain the lap and the call."""
    agent, _, _ = make_agent([_text("Push now.")])
    ctx = make_ctx(lap=25)
    agent.step(ctx)

    assert len(agent.memory.recent_contexts) == 1
    assert agent.memory.recent_contexts[0].lap_number == 25
    assert len(agent.memory.recent_calls) == 1
    assert agent.memory.recent_calls[0].message == "Push now."


def test_memory_includes_previous_lap_in_prompt():
    """Second step() should include first lap's context in the prompt."""
    agent, llm, _ = make_agent([
        _text("First call."),
        _text("Second call."),
    ])

    agent.step(make_ctx(lap=10))
    agent.step(make_ctx(lap=11))

    # The second LLM call should reference lap 10 in its prompt
    second_call_messages = llm.calls[1]["messages"]
    second_user_msg = next(
        m["content"] for m in second_call_messages if m["role"] == "user"
    )
    assert "L10" in second_user_msg or "Lap 10" in second_user_msg
    assert "First call" in second_user_msg  # previous radio call shown


def test_empty_text_treated_as_silent():
    """If model returns empty string, treat as silent with diagnostic."""
    agent, _, _ = make_agent([_text("")])
    call = agent.step(make_ctx(lap=10))
    assert call.message is None
    assert call.silent_reason == "empty response"
