"""
Unit tests for tyre degradation fitter.

We synthesise stints with known degradation rates and verify the fitter
recovers them correctly, picks the right model, handles edge cases.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.tools.tyre_degradation import (
    MAX_PLAUSIBLE_DEG,
    MIN_LAPS_FOR_LINEAR,
    compare_degradation,
    fit_degradation,
    project_forward,
)


def make_stint(
    base_lap_time: float,
    deg_rate: float,
    n_laps: int,
    *,
    quadratic_term: float = 0.0,
    noise_std: float = 0.0,
    compound: str = "MEDIUM",
    stint_number: int = 1,
    start_lap: int = 10,
    pit_in_last: bool = False,
    pit_out_first: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesise a stint with a known degradation curve:
      lap_time(age) = base_lap_time + deg_rate * age + quadratic_term * age^2 + noise
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_laps):
        age = i + 1
        clean_time = (
            base_lap_time + deg_rate * age + quadratic_term * age * age
        )
        if noise_std > 0:
            clean_time += rng.normal(0, noise_std)

        is_first = i == 0
        is_last = i == n_laps - 1

        rows.append({
            "Driver": "TST",
            "LapNumber": start_lap + i,
            "TyreLife": float(age),
            "Stint": float(stint_number),
            "Compound": compound,
            "LapTime": timedelta(seconds=clean_time),
            "PitInTime": timedelta(seconds=clean_time) if (pit_in_last and is_last) else pd.NaT,
            "PitOutTime": timedelta(seconds=clean_time) if (pit_out_first and is_first) else pd.NaT,
        })
    return pd.DataFrame(rows)


# ---- Linear fit recovery -------------------------------------------------


def test_recovers_linear_degradation_rate():
    """Synthetic stint with deg = 0.08 s/lap → fitter should find it."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=10)
    fit = fit_degradation(stint, "NOR")

    assert fit.model == "linear"
    assert fit.slope_seconds_per_lap == pytest.approx(0.08, abs=0.005)
    # Intercept is now fit to the windowed laps, not lap 1, so check slope
    # and r_squared rather than intercept directly
    assert fit.r_squared == pytest.approx(1.0, abs=0.001)  # noise-free
    # n_laps_in_fit is capped at the window size (default 8)
    assert fit.n_laps_in_fit == 8
    assert fit.stint_total_laps == 10
    assert fit.fit_window == 8


def test_handles_negative_slope_improving_pace():
    """Tyres warming up early in a stint → slightly negative slope."""
    stint = make_stint(base_lap_time=102.0, deg_rate=-0.10, n_laps=6)
    fit = fit_degradation(stint, "NOR")
    assert fit.slope_seconds_per_lap < 0
    assert fit.slope_seconds_per_lap == pytest.approx(-0.10, abs=0.01)


def test_with_noise_still_recovers_approximate_rate():
    """Realistic noise should still give a sensible slope."""
    stint = make_stint(base_lap_time=103.0, deg_rate=0.06, n_laps=15,
                       noise_std=0.15)
    fit = fit_degradation(stint, "NOR")
    assert fit.slope_seconds_per_lap == pytest.approx(0.06, abs=0.05)
    assert fit.r_squared > 0.5  # signal still dominates noise


# ---- Quadratic fit selection ---------------------------------------------


def test_quadratic_picked_when_truly_quadratic():
    """Strongly quadratic curve → fitter should select quadratic.

    With the recent_window default of 8, the curvature signal across only 8
    laps competes with the simpler linear fit's adjusted-R² advantage. We
    pass a wider window here to give the quadratic shape room to dominate.
    """
    stint = make_stint(base_lap_time=100.0, deg_rate=0.02,
                       quadratic_term=0.010, n_laps=14)
    fit = fit_degradation(stint, "NOR", recent_window=14)
    assert fit.model == "quadratic"
    # Quadratic term should be nonzero and positive (accelerating deg)
    assert fit.quadratic_term is not None
    assert fit.quadratic_term > 0


def test_linear_picked_when_curve_is_linear():
    """Pure linear data with enough laps → still linear, not over-fit."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=12,
                       quadratic_term=0.0, noise_std=0.05)
    fit = fit_degradation(stint, "NOR")
    assert fit.model == "linear"


def test_quadratic_not_picked_on_short_stint():
    """With only 5 laps, refuse quadratic even if it'd fit better."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.05,
                       quadratic_term=0.005, n_laps=5)
    fit = fit_degradation(stint, "NOR")
    assert fit.model == "linear"  # not enough laps for quadratic


# ---- Recent window behaviour --------------------------------------------


def test_window_uses_only_recent_laps():
    """A 20-lap stint should be fit on only the most recent `window` laps."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=20)
    fit = fit_degradation(stint, "NOR", recent_window=8)
    assert fit.n_laps_in_fit == 8
    assert fit.stint_total_laps == 20
    assert fit.fit_window == 8


def test_window_isolates_recent_deg_from_early_warmup():
    """
    Simulates the real F1 picture: early laps slower (warmup), late laps
    slower again (deg). A full-stint fit would call this 'flat'; the
    windowed fit should recover the late-stint degradation.
    """
    import pandas as pd
    from datetime import timedelta

    rows = []
    # Laps 1-5: tyres warming up — slow then fast
    for i, t in enumerate([102.0, 101.5, 101.0, 100.6, 100.3], start=1):
        rows.append({
            "Driver": "TST", "LapNumber": i, "TyreLife": float(i), "Stint": 1.0,
            "Compound": "MEDIUM", "LapTime": timedelta(seconds=t),
            "PitInTime": pd.NaT, "PitOutTime": pd.NaT,
        })
    # Laps 6-13: degradation phase — +0.10 s/lap
    for i, t in enumerate([100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0], start=6):
        rows.append({
            "Driver": "TST", "LapNumber": i, "TyreLife": float(i), "Stint": 1.0,
            "Compound": "MEDIUM", "LapTime": timedelta(seconds=t),
            "PitInTime": pd.NaT, "PitOutTime": pd.NaT,
        })
    stint = pd.DataFrame(rows)

    # Full-stint fit (window=13) would give a confused slope due to warmup
    full = fit_degradation(stint, "TST", recent_window=13)
    # Windowed fit (window=8) sees only the deg phase — should recover ~0.10
    windowed = fit_degradation(stint, "TST", recent_window=8)

    assert windowed.slope_seconds_per_lap == pytest.approx(0.10, abs=0.02)
    # The windowed fit should be SHARPER (higher R²) than the full-stint fit
    assert windowed.r_squared > full.r_squared


def test_short_stint_smaller_than_window_uses_all():
    """If the stint is shorter than the window, use all available laps."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=5)
    fit = fit_degradation(stint, "NOR", recent_window=8)
    assert fit.n_laps_in_fit == 5
    assert fit.stint_total_laps == 5
    assert fit.fit_window == 8  # the requested window is preserved


def test_recent_window_passed_through_compare_degradation():
    """compare_degradation should propagate the window to both fits."""
    target = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=20)
    rival  = make_stint(base_lap_time=100.0, deg_rate=0.10, n_laps=20)
    cmp = compare_degradation(target, rival, "NOR", "PIA", recent_window=6)
    assert cmp.target_fit.fit_window == 6
    assert cmp.rival_fit.fit_window == 6
    assert cmp.target_fit.n_laps_in_fit == 6
    assert cmp.rival_fit.n_laps_in_fit == 6





def test_insufficient_data_returns_safe_result():
    """Below MIN_LAPS_FOR_LINEAR → no fit, but result is well-formed."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08,
                       n_laps=MIN_LAPS_FOR_LINEAR - 1)
    fit = fit_degradation(stint, "NOR")
    assert fit.model == "insufficient_data"
    assert fit.slope_seconds_per_lap is None
    assert fit.error is not None
    assert "clean laps" in fit.error


def test_empty_stint_returns_safe_result():
    fit = fit_degradation(pd.DataFrame(), "NOR")
    assert fit.model == "insufficient_data"


# ---- Pit lap filtering ---------------------------------------------------


def test_pit_in_lap_excluded_from_fit():
    """The in-lap should be excluded from the cleaned set before windowing."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=10,
                       pit_in_last=True)
    fit = fit_degradation(stint, "NOR")
    # Stint has 9 clean laps after excluding the in-lap; window caps at 8
    assert fit.stint_total_laps == 9
    assert fit.n_laps_in_fit == 8


def test_pit_out_lap_excluded_from_fit():
    """The out-lap should be excluded from the cleaned set before windowing."""
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=10,
                       pit_out_first=True)
    fit = fit_degradation(stint, "NOR")
    assert fit.stint_total_laps == 9
    assert fit.n_laps_in_fit == 8


# ---- Slope plausibility cap ----------------------------------------------


def test_implausible_slope_capped():
    """If the data somehow produces a >MAX_PLAUSIBLE_DEG slope, cap it."""
    # Construct a stint with absurd degradation (5s/lap)
    stint = make_stint(base_lap_time=100.0, deg_rate=5.0, n_laps=6)
    fit = fit_degradation(stint, "NOR")
    assert abs(fit.slope_seconds_per_lap) <= MAX_PLAUSIBLE_DEG


# ---- Projection ----------------------------------------------------------


def test_projection_uses_linear_model():
    stint = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=10)
    fit = fit_degradation(stint, "NOR")
    # current age is 10, project 5 laps ahead → age 15
    # projected = 100 + 0.08*15 = 101.2
    proj = project_forward(fit, laps_ahead=5)
    assert proj.projected_lap_time_seconds == pytest.approx(101.2, abs=0.01)
    # Loss = 0.08 * 5 = 0.4
    assert proj.projected_total_loss_seconds == pytest.approx(0.4, abs=0.01)


def test_projection_with_insufficient_data_safe():
    fit = fit_degradation(pd.DataFrame(), "NOR")
    proj = project_forward(fit, laps_ahead=5)
    assert proj.projected_lap_time_seconds is None
    assert proj.projected_total_loss_seconds is None


# ---- Two-driver comparison -----------------------------------------------


def test_compare_matched_when_similar_degradation():
    target = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=8)
    rival  = make_stint(base_lap_time=100.5, deg_rate=0.082, n_laps=8)
    cmp = compare_degradation(target, rival, "NOR", "PIA", project_laps_ahead=5)
    assert cmp.verdict == "matched"
    assert abs(cmp.delta_seconds_per_lap) < 0.03


def test_compare_target_degrading_faster():
    target = make_stint(base_lap_time=100.0, deg_rate=0.15, n_laps=8)
    rival  = make_stint(base_lap_time=100.0, deg_rate=0.05, n_laps=8)
    cmp = compare_degradation(target, rival, "NOR", "PIA")
    assert cmp.verdict == "target_degrading_faster"
    assert cmp.delta_seconds_per_lap > 0


def test_compare_rival_degrading_faster():
    target = make_stint(base_lap_time=100.0, deg_rate=0.04, n_laps=8)
    rival  = make_stint(base_lap_time=100.0, deg_rate=0.18, n_laps=8)
    cmp = compare_degradation(target, rival, "NOR", "PIA")
    assert cmp.verdict == "rival_degrading_faster"
    assert cmp.delta_seconds_per_lap < 0


def test_compare_no_rival_data():
    target = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=8)
    cmp = compare_degradation(target, None, "NOR", None)
    assert cmp.verdict == "no_comparison"
    assert cmp.rival_fit is None
    assert cmp.target_projection is not None  # target still has projection


def test_compare_rival_insufficient_data():
    """If rival's stint is too short, verdict should be no_comparison."""
    target = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=10)
    rival  = make_stint(base_lap_time=100.0, deg_rate=0.08,
                        n_laps=MIN_LAPS_FOR_LINEAR - 1)
    cmp = compare_degradation(target, rival, "NOR", "PIA")
    assert cmp.verdict == "no_comparison"
    assert cmp.delta_seconds_per_lap is None
    assert cmp.target_fit.model == "linear"  # target was still fit


# ---- Serialisation -------------------------------------------------------


def test_comparison_serializable():
    import json
    target = make_stint(base_lap_time=100.0, deg_rate=0.08, n_laps=8)
    rival  = make_stint(base_lap_time=100.0, deg_rate=0.10, n_laps=8)
    cmp = compare_degradation(target, rival, "NOR", "PIA")
    s = json.dumps(cmp.to_dict())
    parsed = json.loads(s)
    assert parsed["target_fit"]["driver_code"] == "NOR"
    assert parsed["rival_fit"]["driver_code"] == "PIA"
    assert "verdict" in parsed
