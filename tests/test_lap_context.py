"""
Unit tests for the data layer.

Uses mocked race data to verify the lap context builder produces
correctly structured snapshots without hitting the FastF1 servers.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.lap_context import build_lap_context, LapContext
from src.data.race_data import RaceMeta


def make_mock_race():
    """Build a minimal mock RaceData for testing."""
    race = MagicMock()
    race.meta = RaceMeta(
        year=2024,
        name="Azerbaijan",
        circuit="Baku City Circuit",
        total_laps=51,
        pit_loss_seconds=19.5,
    )

    # Norris lap 10: P5, on mediums, age 9, lap time 1:43.5
    nor_lap = pd.Series({
        "LapNumber": 10,
        "Position": 5.0,
        "LapTime": timedelta(seconds=103.5),
        "Compound": "MEDIUM",
        "TyreLife": 9.0,
        "Stint": 1.0,
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "TrackStatus": "1",
        "Time": timedelta(minutes=18, seconds=20),
        "Driver": "NOR",
    })
    race.lap_at.return_value = nor_lap

    # Recent history: 3 prior laps
    history = pd.DataFrame({
        "LapTime": [
            timedelta(seconds=104.2),
            timedelta(seconds=103.8),
            timedelta(seconds=103.5),
        ],
    })
    race.laps_up_to.return_value = history

    # Field at lap 10 — Norris P5, Piastri P4 (ahead), Leclerc P6 (behind)
    field = pd.DataFrame({
        "Driver": ["VER", "HAM", "RUS", "PIA", "NOR", "LEC"],
        "Position": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Time": [
            timedelta(minutes=18, seconds=15),
            timedelta(minutes=18, seconds=16),
            timedelta(minutes=18, seconds=17),
            timedelta(minutes=18, seconds=18.5),
            timedelta(minutes=18, seconds=20),
            timedelta(minutes=18, seconds=22),
        ],
    })
    race.all_drivers_at.return_value = field

    # Weather
    race.weather_at.return_value = pd.Series({
        "TrackTemp": 42.0,
        "AirTemp": 28.0,
        "Rainfall": False,
    })

    race.driver_code_to_name.side_effect = lambda c: {
        "NOR": "Lando Norris",
        "PIA": "Oscar Piastri",
        "LEC": "Charles Leclerc",
    }.get(c, c)

    return race


def test_build_lap_context_basic():
    race = make_mock_race()
    ctx = build_lap_context(race, "NOR", 10)

    assert ctx is not None
    assert ctx.lap_number == 10
    assert ctx.driver_code == "NOR"
    assert ctx.driver_name == "Lando Norris"
    assert ctx.position == 5
    assert ctx.tyre_compound == "MEDIUM"
    assert ctx.tyre_age_laps == 9
    assert ctx.lap_time_seconds == pytest.approx(103.5)
    assert ctx.laps_remaining == 41


def test_build_lap_context_gaps():
    race = make_mock_race()
    ctx = build_lap_context(race, "NOR", 10)

    # Norris is P5, Piastri P4 ahead by 1.5s
    assert ctx.ahead is not None
    assert ctx.ahead.driver_code == "PIA"
    assert ctx.ahead.gap_seconds == pytest.approx(-1.5)

    # Leclerc P6 behind by 2.0s
    assert ctx.behind is not None
    assert ctx.behind.driver_code == "LEC"
    assert ctx.behind.gap_seconds == pytest.approx(2.0)


def test_build_lap_context_recent_pace():
    race = make_mock_race()
    ctx = build_lap_context(race, "NOR", 10)
    assert len(ctx.last_3_lap_times) == 3
    assert ctx.last_3_lap_times == pytest.approx([104.2, 103.8, 103.5])


def test_build_lap_context_weather():
    race = make_mock_race()
    ctx = build_lap_context(race, "NOR", 10)
    assert ctx.track_temp_c == 42.0
    assert ctx.air_temp_c == 28.0
    assert ctx.rainfall is False


def test_build_lap_context_track_status():
    race = make_mock_race()
    ctx = build_lap_context(race, "NOR", 10)
    assert ctx.track_status == "green"


def test_lap_context_serializable():
    """Context must be JSON-serializable for the agent prompt."""
    race = make_mock_race()
    ctx = build_lap_context(race, "NOR", 10)
    d = ctx.to_dict()
    assert isinstance(d, dict)
    assert d["lap_number"] == 10
