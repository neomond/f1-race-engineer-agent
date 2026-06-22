"""
Unit tests for simulate_pit_window.

Math reminder for the corrected model:
- Both scenarios assume both drivers pit eventually, so pit_loss cancels.
- UNDERCUT: target gains fresh_tyre_advantage_s. final_gap = current - fresh_adv
- OVERCUT:  rival gains fresh_tyre_advantage_s. final_gap = current + fresh_adv
- Emergence reflects the in-between state (one car pitted, other hasn't).
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.tools.pit_loss import PitLossEstimate
from src.tools.simulate_pit_window import simulate_pit_window


def make_mock_race(
    target_time_s: float,
    rival_time_s: float,
    pit_loss_s: float = 22.0,
    source: str = "measured",
):
    race = MagicMock()
    race.meta.pit_loss_seconds = pit_loss_s
    race.pit_loss_estimate = PitLossEstimate(
        seconds=pit_loss_s, sample_size=10, source=source, raw_values=[],
    )

    field = pd.DataFrame({
        "Driver": ["NOR", "PIA"],
        "Position": [5.0, 4.0],
        "Time": [
            timedelta(seconds=target_time_s),
            timedelta(seconds=rival_time_s),
        ],
    })
    race.all_drivers_at.return_value = field
    return race


def test_undercut_gains_fresh_tyre_advantage():
    """
    Target 4s behind rival. fresh_adv = 1.2.
    Undercut: 4 - 1.2 = 2.8s behind (target gained 1.2s)
    Overcut:  4 + 1.2 = 5.2s behind (target lost 1.2s)
    """
    race = make_mock_race(target_time_s=3604, rival_time_s=3600, pit_loss_s=22.0)
    result = simulate_pit_window(race, "NOR", "PIA", current_lap=20)

    assert result is not None
    assert result.current_gap_seconds == pytest.approx(4.0)

    assert result.undercut.projected_gap_seconds == pytest.approx(2.8, abs=0.01)
    assert result.undercut.margin_vs_status_quo_seconds == pytest.approx(1.2, abs=0.01)

    assert result.overcut.projected_gap_seconds == pytest.approx(5.2, abs=0.01)
    assert result.overcut.margin_vs_status_quo_seconds == pytest.approx(-1.2, abs=0.01)


def test_undercut_favoured_when_target_behind():
    race = make_mock_race(target_time_s=3601, rival_time_s=3600)
    result = simulate_pit_window(race, "NOR", "PIA", current_lap=20)
    assert "Undercut favoured" in result.recommendation


def test_overcut_favoured_when_target_ahead():
    race = make_mock_race(target_time_s=3595, rival_time_s=3600)
    result = simulate_pit_window(race, "NOR", "PIA", current_lap=20)
    assert "Overcut favoured" in result.recommendation


def test_emergence_gap_reflects_pit_loss():
    """
    Undercut emergence: current_gap + pit_loss (target stopped, rival hasn't).
    Overcut emergence: current_gap - pit_loss.
    """
    race = make_mock_race(target_time_s=3604, rival_time_s=3600, pit_loss_s=20.0)
    result = simulate_pit_window(race, "NOR", "PIA", current_lap=20)
    assert result.undercut.emergence_gap_seconds == pytest.approx(24.0)
    assert result.overcut.emergence_gap_seconds == pytest.approx(-16.0)


def test_pit_loss_does_not_appear_in_final_gap():
    """
    Two simulations with different pit_loss but same fresh_adv should
    produce identical final gaps — pit_loss cancels in correct math.
    """
    race_short_pit = make_mock_race(3604, 3600, pit_loss_s=18.0)
    race_long_pit  = make_mock_race(3604, 3600, pit_loss_s=28.0)

    r1 = simulate_pit_window(race_short_pit, "NOR", "PIA", 20)
    r2 = simulate_pit_window(race_long_pit, "NOR", "PIA", 20)

    assert r1.undercut.projected_gap_seconds == r2.undercut.projected_gap_seconds
    assert r1.overcut.projected_gap_seconds == r2.overcut.projected_gap_seconds
    # Emergence differs (it depends on pit_loss)
    assert r1.undercut.emergence_gap_seconds != r2.undercut.emergence_gap_seconds


def test_pit_loss_source_propagated():
    race = make_mock_race(3601, 3600, source="fallback")
    result = simulate_pit_window(race, "NOR", "PIA", 20)
    assert result.pit_loss_source == "fallback"


def test_returns_none_when_data_missing():
    race = MagicMock()
    race.meta.pit_loss_seconds = 22.0
    race.pit_loss_estimate = PitLossEstimate(22.0, 0, "fallback", [])
    race.all_drivers_at.return_value = pd.DataFrame({
        "Driver": ["NOR"],
        "Time": [timedelta(seconds=3600)],
    })
    result = simulate_pit_window(race, "NOR", "PIA", 20)
    assert result is None


def test_serializable():
    import json
    race = make_mock_race(3601, 3600)
    result = simulate_pit_window(race, "NOR", "PIA", 20)
    s = json.dumps(result.to_dict())
    parsed = json.loads(s)
    assert parsed["target_driver"] == "NOR"
    assert "undercut" in parsed
    assert "overcut" in parsed
    assert "recommendation" in parsed
    assert "emergence_gap_seconds" in parsed["undercut"]


def test_fresh_tyre_advantage_parameterised():
    """Larger fresh-tyre advantage should make undercut gap smaller."""
    race = make_mock_race(3604, 3600)
    weak = simulate_pit_window(race, "NOR", "PIA", 20, fresh_tyre_advantage_s=0.5)
    strong = simulate_pit_window(race, "NOR", "PIA", 20, fresh_tyre_advantage_s=2.0)
    assert (
        strong.undercut.projected_gap_seconds
        < weak.undercut.projected_gap_seconds
    )


def test_classify_gap_alongside():
    """1s behind, 1.2 fresh adv → undercut: -0.2 (alongside)."""
    race = make_mock_race(3601, 3600)
    result = simulate_pit_window(race, "NOR", "PIA", 20)
    assert result.undercut.verdict == "alongside"
