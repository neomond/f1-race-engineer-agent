"""
Unit tests for pit_loss estimator.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.tools.pit_loss import estimate_pit_loss


def _td(seconds: float) -> pd.Timedelta:
    return pd.Timedelta(seconds=seconds)


def make_stint(
    driver: str,
    start_lap: int,
    n_laps: int,
    pace: float,
    stint: int,
    pit_in_last: bool = False,
    pit_out_first: bool = False,
) -> list[dict]:
    """Build a list of lap dicts representing a stint."""
    rows = []
    for i in range(n_laps):
        lap_num = start_lap + i
        is_first = i == 0
        is_last = i == n_laps - 1

        # Mark pit lap times longer than baseline
        lap_time = pace
        if pit_in_last and is_last:
            lap_time = pace + 8  # in-lap is slow
        if pit_out_first and is_first:
            lap_time = pace + 14  # out-lap is much slower

        rows.append({
            "Driver": driver,
            "LapNumber": lap_num,
            "Stint": stint,
            "LapTime": _td(lap_time),
            "PitInTime": _td(lap_time) if (pit_in_last and is_last) else pd.NaT,
            "PitOutTime": _td(lap_time) if (pit_out_first and is_first) else pd.NaT,
        })
    return rows


def test_measured_with_clean_stops():
    """Multiple drivers with clean stops should give a measured estimate."""
    rows = []
    for driver in ["NOR", "PIA", "LEC", "VER"]:
        # Stint 1: laps 1-15, pit in on 15
        rows.extend(make_stint(driver, 1, 15, pace=100.0, stint=1, pit_in_last=True))
        # Stint 2: laps 16-30, pit out on 16
        rows.extend(make_stint(driver, 16, 15, pace=100.0, stint=2, pit_out_first=True))

    laps = pd.DataFrame(rows)
    result = estimate_pit_loss(laps, fallback_seconds=22.0, min_samples=3)

    assert result.source == "measured"
    assert result.sample_size == 4
    # in-lap = 108, out-lap = 114, baseline = 100
    # loss = (108 + 114) - 200 = 22
    assert result.seconds == pytest.approx(22.0, abs=0.5)


def test_fallback_when_too_few_samples():
    """If only one stop happens, we should fall back to the static value."""
    rows = make_stint("NOR", 1, 15, pace=100.0, stint=1, pit_in_last=True)
    rows.extend(make_stint("NOR", 16, 15, pace=100.0, stint=2, pit_out_first=True))
    laps = pd.DataFrame(rows)

    result = estimate_pit_loss(laps, fallback_seconds=22.0, min_samples=3)
    assert result.source == "fallback"
    assert result.seconds == 22.0


def test_outlier_pit_loss_filtered():
    """A 60s 'pit stop' (e.g. data anomaly) should be excluded."""
    rows = []
    # Three normal stops
    for driver in ["NOR", "PIA", "LEC"]:
        rows.extend(make_stint(driver, 1, 15, pace=100.0, stint=1, pit_in_last=True))
        rows.extend(make_stint(driver, 16, 15, pace=100.0, stint=2, pit_out_first=True))

    # One garbage stop (anomalous in-lap of 200s)
    bad = make_stint("HAM", 1, 15, pace=100.0, stint=1, pit_in_last=False)
    bad[-1]["LapTime"] = _td(200.0)
    bad[-1]["PitInTime"] = _td(200.0)
    bad.extend(make_stint("HAM", 16, 15, pace=100.0, stint=2, pit_out_first=True))
    rows.extend(bad)

    laps = pd.DataFrame(rows)
    result = estimate_pit_loss(laps, fallback_seconds=22.0, min_samples=3)

    # Bad stop excluded — three good ones remain
    assert result.source == "measured"
    assert result.sample_size == 3
    assert result.seconds == pytest.approx(22.0, abs=1.0)
