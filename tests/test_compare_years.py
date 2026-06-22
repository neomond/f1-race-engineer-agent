"""
Unit tests for cross-year comparison report builder.

Synthesises log payloads matching the schema produced by run_cross_year.py
and verifies each section of the report renders meaningfully.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.compare_years import (
    build_report,
    load_logs,
    pit_loss_drift_section,
    sample_calls_section,
    strategic_moments_section,
    summary_table,
    tool_usage_section,
)


def make_log(
    year: int,
    pit_loss: float = 20.5,
    pit_loss_source: str = "measured",
    calls: list | None = None,
):
    """Build a minimal log payload that matches the schema."""
    calls = calls or []
    return {
        "_path": Path(f"/tmp/{year}_azerbaijan_NOR_L10_35.json"),
        "meta": {
            "year": year,
            "race": "Azerbaijan",
            "circuit": "Azerbaijan Grand Prix",
            "driver": "NOR",
            "lap_range": [10, 35],
            "total_laps": 51,
            "pit_loss_seconds": pit_loss,
            "pit_loss_source": pit_loss_source,
            "model": "Gemini (gemini-2.5-flash)",
            "timestamp": "2026-05-24T17:00:00+00:00",
        },
        "calls": calls,
        "events": [],
        "summary": {
            "total_laps_processed": 26,
            "radio_calls": sum(1 for c in calls if c.get("message")),
            "silent_laps": sum(1 for c in calls if not c.get("message")),
        },
    }


def make_call(lap, message=None, silent_reason=None, tools=None):
    return {
        "lap_number": lap,
        "message": message,
        "silent_reason": silent_reason,
        "tools_used": tools or [],
        "timestamp": "2026-05-24T17:00:00+00:00",
    }


# ---- Section-level tests --------------------------------------------------


def test_summary_table_has_one_row_per_year():
    logs = [make_log(2023), make_log(2024), make_log(2025)]
    table = summary_table(logs)
    assert "| 2023 |" in table
    assert "| 2024 |" in table
    assert "| 2025 |" in table


def test_summary_table_call_rate_percent():
    """Call rate is (radio calls / total laps) * 100, rounded to int."""
    calls = [
        make_call(10, message="defend"),
        make_call(11, silent_reason="stable"),
        make_call(12, message="box this lap"),
        make_call(13, silent_reason="stable"),
    ]
    log = make_log(2025, calls=calls)
    log["summary"]["total_laps_processed"] = 4
    log["summary"]["radio_calls"] = 2
    log["summary"]["silent_laps"] = 2
    table = summary_table([log])
    assert "50%" in table


def test_tool_usage_counts_correctly():
    calls = [
        make_call(10, message="x", tools=["compare_pace"]),
        make_call(11, message="y", tools=["compare_pace", "simulate_pit_window"]),
        make_call(12, silent_reason="ok", tools=["stay_silent"]),
    ]
    log = make_log(2025, calls=calls)
    section = tool_usage_section([log])
    # All three tool names should appear as columns
    assert "compare_pace" in section
    assert "simulate_pit_window" in section
    assert "stay_silent" in section
    # Counts row: compare_pace=2, simulate_pit_window=1, stay_silent=1, total=4
    assert "| 2025 |" in section
    assert "| 4 |" in section  # total
    # Sorted alphabetically: compare_pace, simulate_pit_window, stay_silent
    # So row should be: | 2025 | 2 | 1 | 1 | 4 |
    assert "| 2 | 1 | 1 | 4 |" in section


def test_tool_usage_handles_unknown_tools():
    """New tools that didn't exist when the report was written should still appear."""
    calls = [
        make_call(10, message="x", tools=["get_tyre_degradation"]),
        make_call(11, message="y", tools=["some_future_tool"]),
    ]
    log = make_log(2025, calls=calls)
    section = tool_usage_section([log])
    assert "get_tyre_degradation" in section
    assert "some_future_tool" in section


def test_tool_usage_handles_no_tools_called():
    """When no tools were called at all, surface that explicitly."""
    calls = [
        make_call(10, message="defend"),
        make_call(11, silent_reason="stable"),
    ]
    log = make_log(2025, calls=calls)
    section = tool_usage_section([log])
    assert "No tool calls recorded" in section
    assert "known behaviour of smaller local LLMs" in section


def test_strategic_moments_highlights_pit_window_calls():
    calls = [
        make_call(10, message="defend", tools=["compare_pace"]),
        make_call(20, message="box this lap, undercut on Leclerc",
                  tools=["compare_pace", "simulate_pit_window"]),
        make_call(30, silent_reason="stable"),
    ]
    log = make_log(2025, calls=calls)
    section = strategic_moments_section([log])
    assert "L20" in section
    assert "undercut" in section.lower()
    # Lap 10 had no simulate_pit_window — should NOT appear
    assert "L10" not in section


def test_strategic_moments_handles_no_pit_window_calls():
    """If no year had simulate_pit_window, surface that clearly."""
    log = make_log(2025, calls=[make_call(10, message="defend",
                                         tools=["compare_pace"])])
    section = strategic_moments_section([log])
    assert "No `simulate_pit_window`" in section


def test_pit_loss_drift_with_one_year():
    """Single-year case should explain there's no comparison to make."""
    section = pit_loss_drift_section([make_log(2025, pit_loss=20.5)])
    assert "Only one race" in section


def test_pit_loss_drift_shows_spread():
    logs = [
        make_log(2023, pit_loss=19.8),
        make_log(2024, pit_loss=20.6),
        make_log(2025, pit_loss=20.1),
    ]
    section = pit_loss_drift_section(logs)
    # Spread = 20.6 - 19.8 = 0.8
    assert "Spread: 0.80s" in section
    assert "19.8s" in section
    assert "20.6s" in section


def test_sample_calls_picks_calls_with_messages():
    """Should sample radio calls (with messages), skip silent ones."""
    calls = [make_call(i, message=f"call {i}", tools=["compare_pace"])
             for i in range(10, 20)]
    calls.append(make_call(21, silent_reason="stable"))  # should be excluded
    log = make_log(2025, calls=calls)
    section = sample_calls_section([log], n_per_year=3)
    assert "### 2025" in section
    assert "call 10" in section or "call 11" in section  # at least one sampled


def test_sample_calls_handles_empty():
    log = make_log(2025, calls=[make_call(10, silent_reason="all silent")])
    section = sample_calls_section([log])
    assert "No radio calls" in section


# ---- End-to-end report build ---------------------------------------------


def test_full_report_has_all_sections():
    calls_2023 = [make_call(15, message="defend", tools=["compare_pace"])]
    calls_2024 = [
        make_call(20, message="box now", tools=["simulate_pit_window"]),
    ]
    logs = [make_log(2023, calls=calls_2023), make_log(2024, calls=calls_2024)]
    report = build_report(logs)

    assert "Summary across editions" in report
    assert "Tool usage by year" in report
    assert "Measured pit loss drift" in report
    assert "Strategic moments" in report
    assert "Representative radio calls" in report


def test_empty_logs_produces_safe_report():
    report = build_report([])
    assert "No logs found" in report


def test_load_logs_sorts_by_year(tmp_path):
    """load_logs should return logs sorted ascending by year."""
    for year in (2025, 2023, 2024):
        log = make_log(year)
        log.pop("_path")
        (tmp_path / f"{year}_az.json").write_text(json.dumps(log))
    loaded = load_logs(tmp_path)
    years = [l["meta"]["year"] for l in loaded]
    assert years == [2023, 2024, 2025]


def test_load_logs_skips_malformed(tmp_path):
    """Garbage files shouldn't crash the loader."""
    (tmp_path / "good.json").write_text(json.dumps(make_log(2025) | {"_path": None}))
    (tmp_path / "broken.json").write_text("{ not valid json")
    loaded = load_logs(tmp_path)
    assert len(loaded) == 1
