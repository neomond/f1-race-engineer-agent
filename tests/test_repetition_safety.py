"""
Unit tests for the agent's repetition safety net.

The model SHOULD vary its output. When it doesn't (smaller local models
sometimes loop), our agent code catches it and converts to silent.
We test that:
  - Exact repeats are caught
  - Near-repeats with trivial punctuation diffs are caught
  - Genuinely different calls about the same rival pass through
  - Calls referencing different rivals pass through
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.agent.race_engineer import RaceEngineerAgent
from src.memory.agent_memory import AgentMemory, RadioCall


def make_agent_with_recent_calls(messages: list[str]) -> RaceEngineerAgent:
    """Build an agent whose memory has the given recent radio calls."""
    agent = RaceEngineerAgent(llm=MagicMock(), race=MagicMock(),
                              memory=AgentMemory())
    for i, msg in enumerate(messages, start=1):
        agent.memory.record_call(RadioCall(
            lap_number=i,
            message=msg,
            silent_reason=None,
            tools_used=[],
        ))
    return agent


# ---- Exact + near-exact repeats ------------------------------------------


def test_exact_repeat_caught():
    agent = make_agent_with_recent_calls([
        "Box this lap, box this lap. Undercut window on HAM.",
    ])
    assert agent._is_repetitive(
        "Box this lap, box this lap. Undercut window on HAM."
    )


def test_punctuation_only_diff_caught():
    """Same content, different punctuation, should be flagged as repeat."""
    agent = make_agent_with_recent_calls([
        "Box this lap, box this lap. Undercut window on HAM.",
    ])
    assert agent._is_repetitive(
        "Box this lap box this lap undercut window on HAM"
    )


def test_repeat_across_window():
    """Repeat of a call from 2 laps ago is still a repeat."""
    agent = make_agent_with_recent_calls([
        "Box this lap, undercut window on HAM.",
        "VER pace dropping, gap widening.",
        "ALO two tenths quicker in S2. Manage rears.",
    ])
    # The first call (3 laps ago) is within our window=3
    assert agent._is_repetitive("Box this lap, undercut window on HAM.")


def test_too_old_repeat_not_caught():
    """A repeat from outside the window is NOT flagged."""
    # Window of 2 — only check most recent 2
    agent = make_agent_with_recent_calls([
        "Box this lap, undercut window on HAM.",       # 3 laps ago
        "VER pace dropping, gap widening.",            # 2 laps ago
        "ALO two tenths quicker in S2. Manage rears.", # 1 lap ago
    ])
    # Repeating the oldest call, but window is 2 → not flagged
    assert not agent._is_repetitive(
        "Box this lap, undercut window on HAM.", window=2
    )


# ---- Legitimate variation passes -----------------------------------------


def test_different_rival_passes():
    """Same advice pattern but different target = legitimate."""
    agent = make_agent_with_recent_calls([
        "Box this lap, undercut window on HAM.",
    ])
    # Different rival named — meaningfully different call
    assert not agent._is_repetitive(
        "Box this lap, undercut window on LEC."
    )


def test_different_advice_same_rival_passes():
    """Same rival but different advice = legitimate."""
    agent = make_agent_with_recent_calls([
        "HAM closing, gap 1.5s. Defend hard.",
    ])
    # Same rival, but advice shifted from defend → strategic
    assert not agent._is_repetitive(
        "HAM matched pace last 3 laps. Switch to engine mode 2."
    )


def test_completely_different_call_passes():
    agent = make_agent_with_recent_calls([
        "Box this lap, undercut window on HAM.",
    ])
    assert not agent._is_repetitive(
        "Yellow flag sector 2. Lift and coast through turn 8."
    )


# ---- Edge cases ----------------------------------------------------------


def test_no_recent_calls_passes_everything():
    """First call of the race is never a repeat."""
    agent = make_agent_with_recent_calls([])
    assert not agent._is_repetitive("Any call goes here.")


def test_empty_candidate_not_flagged():
    agent = make_agent_with_recent_calls(["Some call"])
    assert not agent._is_repetitive("")


def test_only_stopwords_in_candidate_not_flagged():
    """If the call is just stopwords, it's noise — don't crash, don't flag."""
    agent = make_agent_with_recent_calls(["Some real call"])
    assert not agent._is_repetitive("the a is on")


def test_silent_calls_in_memory_ignored():
    """Silent entries in memory shouldn't influence repetition detection."""
    agent = RaceEngineerAgent(llm=MagicMock(), race=MagicMock(),
                              memory=AgentMemory())
    agent.memory.record_call(RadioCall(
        lap_number=1, message=None,
        silent_reason="stable", tools_used=[],
    ))
    # No real messages in memory — candidate should pass
    assert not agent._is_repetitive("Box this lap, undercut window on HAM.")


def test_partial_overlap_below_threshold():
    """A call sharing some tokens (rival name) but otherwise different passes."""
    agent = make_agent_with_recent_calls([
        "HAM closing fast, brake earlier into turn 1.",
    ])
    # Shares "HAM" but the substance is completely different
    assert not agent._is_repetitive(
        "Pit strategy update: overcut HAM by staying out 3 more laps."
    )
