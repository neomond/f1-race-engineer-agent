"""
Unit tests for AgentMemory.
"""
from __future__ import annotations

from src.data.lap_context import LapContext, GapInfo
from src.memory.agent_memory import AgentMemory, RadioCall


def make_ctx(lap=10, stint=1, compound="MEDIUM", track_status="green"):
    return LapContext(
        lap_number=lap,
        total_laps=51,
        laps_remaining=51 - lap,
        driver_code="NOR",
        driver_name="Lando Norris",
        position=5,
        lap_time_seconds=103.5,
        last_3_lap_times=[103.5, 103.6, 103.5],
        tyre_compound=compound,
        tyre_age_laps=9,
        stint_number=stint,
        in_pit=False,
        ahead=GapInfo("PIA", "Oscar Piastri", -1.5),
        behind=GapInfo("LEC", "Charles Leclerc", 2.0),
        track_temp_c=42.0,
        air_temp_c=28.0,
        rainfall=False,
        track_status=track_status,
    )


def test_short_term_lap_window_caps():
    mem = AgentMemory(short_term_laps=3)
    for i in range(5):
        mem.record_lap(make_ctx(lap=i + 1))
    # Only last 3 retained
    assert len(mem.recent_contexts) == 3
    assert [c.lap_number for c in mem.recent_contexts] == [3, 4, 5]


def test_short_term_call_window_caps():
    mem = AgentMemory(short_term_calls=2)
    for i in range(4):
        mem.record_call(RadioCall(
            lap_number=i + 1, message=f"call {i}",
            silent_reason=None, tools_used=[],
        ))
    assert len(mem.recent_calls) == 2
    assert [c.lap_number for c in mem.recent_calls] == [3, 4]


def test_detects_pit_stop_on_stint_change():
    mem = AgentMemory()
    mem.record_lap(make_ctx(lap=15, stint=1, compound="MEDIUM"))
    mem.record_lap(make_ctx(lap=16, stint=2, compound="HARD"))

    pit_events = [e for e in mem.events if e.event_type == "pit_stop"]
    assert len(pit_events) == 1
    assert pit_events[0].lap_number == 16
    assert "HARD" in pit_events[0].description


def test_no_pit_event_on_first_lap():
    """No previous stint to compare → no event."""
    mem = AgentMemory()
    mem.record_lap(make_ctx(lap=1, stint=1))
    assert not any(e.event_type == "pit_stop" for e in mem.events)


def test_detects_safety_car_start_and_end():
    mem = AgentMemory()
    mem.record_lap(make_ctx(lap=10, track_status="green"))
    mem.record_lap(make_ctx(lap=11, track_status="safety_car"))
    mem.record_lap(make_ctx(lap=12, track_status="safety_car"))
    mem.record_lap(make_ctx(lap=13, track_status="green"))

    starts = [e for e in mem.events if e.event_type == "safety_car_start"]
    ends = [e for e in mem.events if e.event_type == "safety_car_end"]
    assert len(starts) == 1 and starts[0].lap_number == 11
    assert len(ends) == 1 and ends[0].lap_number == 13


def test_prompt_render_includes_all_sections():
    mem = AgentMemory()
    mem.record_lap(make_ctx(lap=15, stint=1))
    mem.record_lap(make_ctx(lap=16, stint=2, compound="HARD"))  # triggers pit event
    mem.record_call(RadioCall(
        lap_number=15, message="Test call",
        silent_reason=None, tools_used=["compare_pace"],
    ))

    rendered = mem.render_for_prompt()
    assert "RECENT EVENTS:" in rendered
    assert "pit_stop" in rendered
    assert "RECENT RADIO CALLS" in rendered
    assert "Test call" in rendered
    assert "LAP TREND" in rendered


def test_prompt_render_empty_when_no_data():
    mem = AgentMemory()
    rendered = mem.render_for_prompt()
    assert rendered == "(no prior context)"


def test_silent_call_rendered_distinctly():
    mem = AgentMemory()
    mem.record_call(RadioCall(
        lap_number=10, message=None,
        silent_reason="pace stable", tools_used=[],
    ))
    rendered = mem.render_for_prompt()
    assert "silent" in rendered
    assert "pace stable" in rendered


def test_to_log_dict_serializable():
    import json
    mem = AgentMemory()
    mem.record_lap(make_ctx(lap=15, stint=1))
    mem.record_lap(make_ctx(lap=16, stint=2))
    mem.record_call(RadioCall(10, "test", None, []))
    log = mem.to_log_dict()
    s = json.dumps(log)
    assert "events" in json.loads(s)
