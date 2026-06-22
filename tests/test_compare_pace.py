"""
Unit tests for compare_pace.

We mock RaceData with synthetic lap data covering the interesting cases:
- Clear pace gap
- Trend (one driver gaining)
- Sector-level differences
- Pit lap filtering
- Stint mode windowing
- Insufficient data
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.tools.compare_pace import compare_pace


def _td(seconds: float) -> pd.Timedelta:
    return pd.Timedelta(seconds=seconds)


def make_laps(
    driver: str,
    lap_times: list[float],
    sector_splits: tuple[float, float, float] = (0.30, 0.35, 0.35),
    stint: int = 1,
    pit_in_at: int | None = None,
) -> pd.DataFrame:
    """Build a synthetic laps DataFrame for one driver."""
    rows = []
    for i, lt in enumerate(lap_times, start=1):
        s1 = lt * sector_splits[0]
        s2 = lt * sector_splits[1]
        s3 = lt * sector_splits[2]
        rows.append({
            "LapNumber": i,
            "Driver": driver,
            "LapTime": _td(lt),
            "Sector1Time": _td(s1),
            "Sector2Time": _td(s2),
            "Sector3Time": _td(s3),
            "Stint": stint,
            "PitInTime": _td(lt) if i == pit_in_at else pd.NaT,
            "PitOutTime": pd.NaT,
        })
    return pd.DataFrame(rows)


def make_mock_race(laps_a: pd.DataFrame, laps_b: pd.DataFrame):
    """Mock RaceData that returns the right driver's laps."""
    race = MagicMock()

    def laps_up_to(driver, current_lap):
        df = laps_a if driver == "NOR" else laps_b
        return df[df["LapNumber"] <= current_lap].reset_index(drop=True)

    race.laps_up_to.side_effect = laps_up_to
    return race


# ---- Tests ----------------------------------------------------------------


def test_clear_pace_gap_last_n():
    """Driver B is consistently 0.5s faster — should show positive delta."""
    nor = make_laps("NOR", [104.0, 104.0, 104.0, 104.0, 104.0])
    pia = make_laps("PIA", [103.5, 103.5, 103.5, 103.5, 103.5])
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=5, mode="last_n", last_n=5)

    assert result.laps_compared == 5
    assert result.avg_lap_delta_seconds == pytest.approx(0.5, abs=0.01)
    assert result.median_lap_delta_seconds == pytest.approx(0.5, abs=0.01)
    assert result.trend_direction == "stable"  # constant gap


def test_target_driver_faster():
    """If driver_a is faster, delta should be negative."""
    nor = make_laps("NOR", [103.0] * 5)
    pia = make_laps("PIA", [104.0] * 5)
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=5, mode="last_n", last_n=5)
    assert result.avg_lap_delta_seconds == pytest.approx(-1.0, abs=0.01)


def test_closing_trend():
    """Driver A starts behind, gains 0.2s per lap → trend should be 'closing'."""
    nor = make_laps("NOR", [105.0, 104.8, 104.6, 104.4, 104.2])
    pia = make_laps("PIA", [104.0, 104.0, 104.0, 104.0, 104.0])
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=5, mode="last_n", last_n=5)

    assert result.trend_direction == "closing"
    assert result.trend_seconds_per_lap < 0


def test_losing_trend():
    """Driver A starts level, loses 0.15s per lap → 'losing'."""
    nor = make_laps("NOR", [104.0, 104.15, 104.3, 104.45, 104.6])
    pia = make_laps("PIA", [104.0] * 5)
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=5, mode="last_n", last_n=5)
    assert result.trend_direction == "losing"
    assert result.trend_seconds_per_lap > 0


def test_sector_deltas():
    """Driver A is faster in S1, equal in S2, slower in S3."""
    # Total lap = 100s, but split unevenly
    nor = make_laps("NOR", [100.0] * 5, sector_splits=(0.28, 0.35, 0.37))
    pia = make_laps("PIA", [100.0] * 5, sector_splits=(0.30, 0.35, 0.35))
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=5, mode="last_n", last_n=5)

    # NOR S1 = 28s, PIA S1 = 30s → NOR faster by 2s
    assert result.sector_delta.sector_1_seconds == pytest.approx(-2.0, abs=0.01)
    # S2 equal
    assert result.sector_delta.sector_2_seconds == pytest.approx(0.0, abs=0.01)
    # NOR S3 = 37s, PIA S3 = 35s → NOR slower by 2s
    assert result.sector_delta.sector_3_seconds == pytest.approx(2.0, abs=0.01)


def test_pit_lap_filtered():
    """Pit-in lap should be excluded from the comparison."""
    # NOR lap 5 is a pit-in (would be ~120s), should not pollute average
    nor = make_laps("NOR", [104.0, 104.0, 104.0, 104.0, 120.0], pit_in_at=5)
    pia = make_laps("PIA", [104.0] * 5)
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=5, mode="last_n", last_n=5)

    # Should compare 4 clean laps, not 5
    assert result.laps_compared == 4
    # And average should be ~0, not skewed by the 120s pit lap
    assert result.avg_lap_delta_seconds == pytest.approx(0.0, abs=0.01)


def test_stint_mode_windowing():
    """In stint mode, only the current stint's laps are compared."""
    # NOR pitted after lap 3 — stint 1 (laps 1-3), stint 2 (laps 4-6)
    nor_stint1 = make_laps("NOR", [105.0, 105.0, 105.0], stint=1)
    nor_stint2 = make_laps("NOR", [103.0, 103.0, 103.0], stint=2)
    nor_stint2["LapNumber"] = [4, 5, 6]
    nor = pd.concat([nor_stint1, nor_stint2], ignore_index=True)

    pia = make_laps("PIA", [104.0] * 6)
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=6, mode="stint")

    # Should only use stint 2 (laps 4-6) where NOR averages 103, PIA 104
    assert result.laps_compared == 3
    assert result.avg_lap_delta_seconds == pytest.approx(-1.0, abs=0.01)


def test_no_overlap_returns_empty():
    """If drivers share no clean comparable laps, return empty result."""
    nor = make_laps("NOR", [104.0, 104.0], pit_in_at=1)
    nor.loc[1, "PitInTime"] = _td(104.0)  # both NOR laps are pit laps
    pia = make_laps("PIA", [104.0, 104.0])
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=2, mode="last_n", last_n=5)
    assert result.laps_compared == 0
    assert result.avg_lap_delta_seconds is None
    assert result.trend_direction == "insufficient_data"


def test_serializable():
    """Result must be JSON-serialisable for the agent."""
    import json
    nor = make_laps("NOR", [104.0] * 5)
    pia = make_laps("PIA", [103.5] * 5)
    race = make_mock_race(nor, pia)

    result = compare_pace(race, "NOR", "PIA", current_lap=5, mode="last_n", last_n=5)
    s = json.dumps(result.to_dict())
    parsed = json.loads(s)
    assert parsed["driver_a"] == "NOR"
    assert parsed["sector_delta"]["sector_1_seconds"] is not None
