"""
Unit tests for _extract_radio_call.

The model output can be messy — reasoning leaking out, repeated content,
contradictory text. We test that the extractor reliably pulls out just
the actual radio call.
"""
from __future__ import annotations

from src.agent.race_engineer import RaceEngineerAgent


extract = RaceEngineerAgent._extract_radio_call


def test_plain_quoted_call():
    text = '"Box this lap, undercut window on Leclerc."'
    assert extract(text) == "Box this lap, undercut window on Leclerc."


def test_reasoning_before_quoted_call():
    """Real Gemini output: reasoning paragraph + quoted call at end."""
    text = (
        "NOR is losing pace to ALB and PER is gaining. PER is only 0.01s "
        "behind. NOR should pit.\n\n"
        '"Box this lap, box this lap. Perez is closing."'
    )
    assert extract(text) == "Box this lap, box this lap. Perez is closing."


def test_takes_last_quoted_when_multiple():
    """If the call appears twice, prefer the last one (it's usually final)."""
    text = (
        '"Initial thought: Box this lap."\n'
        '"Box this lap, box this lap. Perez is closing."'
    )
    assert extract(text) == "Box this lap, box this lap. Perez is closing."


def test_no_quotes_fallback_to_first_useful_line():
    text = (
        "Here is the call:\n"
        "Based on the analysis,\n"
        "Box this lap, undercut window."
    )
    assert extract(text) == "Box this lap, undercut window."


def test_empty_returns_empty():
    assert extract("") == ""
    assert extract("   ") == ""


def test_caps_very_long_output():
    long_text = "Push harder and ".strip() * 50  # ~800 chars unquoted
    result = extract(long_text)
    assert len(result) <= 203  # 200 + "..."


def test_ignores_short_quoted_artifacts():
    """Tiny quoted strings (like 'NOR') shouldn't be picked over real call."""
    text = (
        "Comparing \"NOR\" and \"PIA\".\n"
        '"PIA two tenths quicker. Manage tyres."'
    )
    assert extract(text) == "PIA two tenths quicker. Manage tyres."


def test_handles_contradiction_safely():
    """
    Real failure case: model emits a call AND says it will stay silent.
    We extract the call — silent should have been a tool invocation.
    """
    text = (
        "NOR is losing pace. Overcut is favoured.\n\n"
        '"Hold position, overcut favoured."\n\n'
        "I will stay silent."
    )
    assert extract(text) == "Hold position, overcut favoured."
