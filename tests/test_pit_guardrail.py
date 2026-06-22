"""
Unit tests for the pit-recommendation guardrail.

A driver pits ~1-2 times per race. If the agent recommends a stop and
the driver does NOT actually pit (no pit_stop event in memory), further
box recommendations within the cooldown window are suppressed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.agent.race_engineer import RaceEngineerAgent
from src.memory.agent_memory import AgentMemory, RadioCall, RaceEvent


def make_agent_with_history(
    radio_calls: list[tuple[int, str | None]],
    events: list[tuple[int, str]] | None = None,
) -> RaceEngineerAgent:
    """
    radio_calls: list of (lap, message). message=None means silent.
    events: list of (lap, event_type).
    """
    agent = RaceEngineerAgent(llm=MagicMock(), race=MagicMock(),
                              memory=AgentMemory())
    for lap, msg in radio_calls:
        agent.memory.record_call(RadioCall(
            lap_number=lap,
            message=msg,
            silent_reason=None if msg else "test silent",
            tools_used=[],
        ))
    for lap, etype in events or []:
        agent.memory.events.append(RaceEvent(
            lap_number=lap, event_type=etype, description="test event",
        ))
    return agent


# ---- mentions_pit detector ------------------------------------------------


def test_mentions_pit_detects_box_this_lap():
    assert RaceEngineerAgent._mentions_pit("Box this lap, box this lap.")


def test_mentions_pit_detects_box_box():
    assert RaceEngineerAgent._mentions_pit("Box, box, box. Undercut now.")


def test_mentions_pit_detects_undercut_window_opens():
    assert RaceEngineerAgent._mentions_pit("Undercut window opens on LEC.")
    assert RaceEngineerAgent._mentions_pit("Undercut window opening — HAM.")


def test_mentions_pit_ignores_undercut_window_closing():
    """'Closing' is a warning, not a recommendation."""
    assert not RaceEngineerAgent._mentions_pit(
        "Undercut window closing on Leclerc."
    )


def test_mentions_pit_ignores_overcut_advice():
    assert not RaceEngineerAgent._mentions_pit(
        "Overcut favoured, hold position."
    )


def test_mentions_pit_ignores_unrelated_box_mentions():
    """The word 'box' alone shouldn't trigger — only with 'this lap'."""
    assert not RaceEngineerAgent._mentions_pit("Watch the gearbox temperature.")


# ---- Premature pit-call detection ----------------------------------------


def test_recent_box_call_no_pit_event_blocks_repeat():
    """Classic 2023-style failure: box recommended at L27, no pit yet, L29 repeats."""
    agent = make_agent_with_history(
        radio_calls=[(27, "Gap to HUL closing. Box this lap, undercut window opens.")]
    )
    assert agent._is_premature_pit_call(
        "Gap to HUL now 1.00. Undercut window opens, box this lap.",
        current_lap=29,
    )


def test_box_call_after_pit_event_passes():
    """If the driver actually pitted, a new box recommendation may be valid."""
    agent = make_agent_with_history(
        radio_calls=[(20, "Box this lap. Undercut window on HAM.")],
        events=[(21, "pit_stop")],   # driver actually pitted on lap 21
    )
    # New box call at lap 28 — driver pitted in between, so this is OK
    assert not agent._is_premature_pit_call(
        "Box this lap. Undercut window on VER.", current_lap=28,
    )


def test_box_call_outside_cooldown_passes():
    """If the cooldown elapsed and no pit happened, that's a different problem
    but we shouldn't gate it forever — let the model retry."""
    agent = make_agent_with_history(
        radio_calls=[(10, "Box this lap. Undercut window on HAM.")]
    )
    # Lap 17 = 7 laps later, beyond default cooldown of 5
    assert not agent._is_premature_pit_call(
        "Box this lap. Undercut window on LEC.", current_lap=17,
    )


def test_non_pit_call_after_box_passes():
    """A non-pit follow-up call is fine even right after a box."""
    agent = make_agent_with_history(
        radio_calls=[(20, "Box this lap. Undercut window on HAM.")]
    )
    assert not agent._is_premature_pit_call(
        "VER closing, 0.3s. Defend hard.", current_lap=21,
    )


def test_no_prior_box_call_passes():
    """First box recommendation of the race should always pass."""
    agent = make_agent_with_history(
        radio_calls=[(15, "VER closing, watch the gap.")]
    )
    assert not agent._is_premature_pit_call(
        "Box this lap. Undercut window on HAM.", current_lap=20,
    )


def test_three_consecutive_box_calls_blocked():
    """Real 2025 failure mode: box for LEC, then LAW, then HAM in 3 laps."""
    agent = make_agent_with_history(
        radio_calls=[
            (19, "Box this lap. Undercut window on LEC opening."),
        ]
    )
    # Lap 20: box for LAW
    assert agent._is_premature_pit_call(
        "Box this lap. Undercut window on LAW opening.", current_lap=20,
    )

    # Simulate that L20 was suppressed (silent), then L21 tries again
    agent.memory.record_call(RadioCall(
        lap_number=20, message=None,
        silent_reason="premature box suppressed", tools_used=[],
    ))
    # Lap 21: box for HAM — still within cooldown of original L19 call
    assert agent._is_premature_pit_call(
        "Box this lap. Undercut window on HAM opening.", current_lap=21,
    )


def test_silent_calls_in_memory_dont_count_as_box():
    """A silent entry can't trigger the guardrail even if its reason mentions box."""
    agent = make_agent_with_history(
        radio_calls=[(20, None)],  # silent
    )
    assert not agent._is_premature_pit_call(
        "Box this lap. Undercut window on HAM.", current_lap=21,
    )
