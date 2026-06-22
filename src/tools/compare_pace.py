"""
compare_pace tool.

Compares the pace of two drivers over a window — either the last N laps,
or the entire current stint up to the given lap. Returns lap-level delta,
sector-by-sector deltas, and a simple trend indicator.

Design:
- Pure function. No FastF1 imports here; takes a RaceData object and
  returns a structured dict that's JSON-serialisable for the agent.
- Defensive: skips outlier laps (pit in/out, safety car) when computing
  averages, since those would distort the pace picture.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.data.race_data import RaceData


WindowMode = Literal["last_n", "stint"]


@dataclass
class SectorDelta:
    """Sector-level pace delta. Negative = target driver is faster."""
    sector_1_seconds: Optional[float]
    sector_2_seconds: Optional[float]
    sector_3_seconds: Optional[float]


@dataclass
class PaceComparison:
    """Result of comparing two drivers over a window."""
    driver_a: str
    driver_b: str
    mode: str
    window_description: str
    laps_compared: int

    # Lap-level. Negative = driver_a is faster than driver_b.
    avg_lap_delta_seconds: Optional[float]
    median_lap_delta_seconds: Optional[float]

    # Trend over the window. Negative = driver_a closing the gap.
    trend_seconds_per_lap: Optional[float]
    trend_direction: str  # "closing", "stable", "losing"

    # Sector breakdown
    sector_delta: SectorDelta

    # Raw averages for transparency
    driver_a_avg_lap_seconds: Optional[float]
    driver_b_avg_lap_seconds: Optional[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# Threshold for classifying trend as "stable" rather than closing/losing
_STABLE_TREND_THRESHOLD_S = 0.05


def compare_pace(
    race: RaceData,
    driver_a: str,
    driver_b: str,
    current_lap: int,
    mode: WindowMode = "last_n",
    last_n: int = 5,
) -> PaceComparison:
    """
    Compare pace between two drivers.

    Args:
        race: Loaded RaceData.
        driver_a: Target driver code (e.g. 'NOR').
        driver_b: Comparison driver code (e.g. 'PIA').
        current_lap: Current lap number — comparison runs on laps <= this.
        mode: 'last_n' for sliding window, 'stint' for current stint.
        last_n: Window size when mode='last_n' (default 5).

    Returns:
        PaceComparison with lap delta, sector deltas, trend.
    """
    laps_a = _get_window(race, driver_a, current_lap, mode, last_n)
    laps_b = _get_window(race, driver_b, current_lap, mode, last_n)

    # Align on lap number — only compare laps both drivers ran cleanly
    aligned = _align_clean_laps(laps_a, laps_b)
    n = len(aligned)

    if n == 0:
        return _empty_comparison(driver_a, driver_b, mode, last_n)

    avg_a = aligned["LapTime_a_s"].mean()
    avg_b = aligned["LapTime_b_s"].mean()
    avg_delta = avg_a - avg_b
    median_delta = (aligned["LapTime_a_s"] - aligned["LapTime_b_s"]).median()

    trend, direction = _compute_trend(aligned)
    sectors = _compute_sector_deltas(aligned)

    window_desc = (
        f"last {n} laps (lap {int(aligned['LapNumber'].min())}"
        f"–{int(aligned['LapNumber'].max())})"
        if mode == "last_n"
        else f"current stint, {n} laps (up to lap {current_lap})"
    )

    return PaceComparison(
        driver_a=driver_a,
        driver_b=driver_b,
        mode=mode,
        window_description=window_desc,
        laps_compared=n,
        avg_lap_delta_seconds=round(avg_delta, 3),
        median_lap_delta_seconds=round(median_delta, 3),
        trend_seconds_per_lap=round(trend, 3) if trend is not None else None,
        trend_direction=direction,
        sector_delta=sectors,
        driver_a_avg_lap_seconds=round(avg_a, 3),
        driver_b_avg_lap_seconds=round(avg_b, 3),
    )


# ---- Helpers --------------------------------------------------------------


def _get_window(
    race: RaceData,
    driver: str,
    current_lap: int,
    mode: WindowMode,
    last_n: int,
) -> pd.DataFrame:
    """Return the relevant slice of laps for this driver."""
    df = race.laps_up_to(driver, current_lap)
    if mode == "stint":
        # FastF1 increments Stint on each pit stop; latest stint = current
        if "Stint" in df.columns and df["Stint"].notna().any():
            current_stint = df["Stint"].iloc[-1]
            df = df[df["Stint"] == current_stint]
    elif mode == "last_n":
        df = df.tail(last_n)
    return df.reset_index(drop=True)


def _align_clean_laps(
    laps_a: pd.DataFrame, laps_b: pd.DataFrame
) -> pd.DataFrame:
    """
    Inner-join on lap number, drop laps where either driver was in pits or
    had no lap time recorded. Returns a tidy DataFrame with both drivers'
    times on the same row.
    """
    def _prep(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        keep = ["LapNumber", "LapTime", "Sector1Time", "Sector2Time",
                "Sector3Time", "PitInTime", "PitOutTime"]
        keep = [c for c in keep if c in df.columns]
        sub = df[keep].copy()

        # Filter pit laps
        if "PitInTime" in sub.columns:
            sub = sub[sub["PitInTime"].isna()]
        if "PitOutTime" in sub.columns:
            sub = sub[sub["PitOutTime"].isna()]

        # Filter laps with no recorded lap time
        sub = sub[sub["LapTime"].notna()]

        # Convert timedeltas to seconds
        sub[f"LapTime_{suffix}_s"] = sub["LapTime"].dt.total_seconds()
        for s in ("Sector1Time", "Sector2Time", "Sector3Time"):
            if s in sub.columns:
                sub[f"{s}_{suffix}_s"] = sub[s].dt.total_seconds()

        cols = ["LapNumber", f"LapTime_{suffix}_s"] + [
            f"{s}_{suffix}_s"
            for s in ("Sector1Time", "Sector2Time", "Sector3Time")
            if s in sub.columns
        ]
        return sub[cols]

    a = _prep(laps_a, "a")
    b = _prep(laps_b, "b")
    merged = a.merge(b, on="LapNumber", how="inner")
    return merged.sort_values("LapNumber").reset_index(drop=True)


def _compute_trend(aligned: pd.DataFrame) -> tuple[Optional[float], str]:
    """
    Linear-regression slope of (delta) vs lap number.
    Negative slope = driver_a is closing the gap.
    """
    if len(aligned) < 3:
        return None, "insufficient_data"

    delta = aligned["LapTime_a_s"] - aligned["LapTime_b_s"]
    laps = aligned["LapNumber"].astype(float)

    # numpy polyfit deg=1 returns [slope, intercept]
    slope, _ = np.polyfit(laps, delta, 1)

    if abs(slope) < _STABLE_TREND_THRESHOLD_S:
        direction = "stable"
    elif slope < 0:
        direction = "closing"
    else:
        direction = "losing"

    return float(slope), direction


def _compute_sector_deltas(aligned: pd.DataFrame) -> SectorDelta:
    """Per-sector mean delta. Negative = driver_a faster in that sector."""
    deltas: dict[str, Optional[float]] = {}
    for i, key in enumerate(("sector_1_seconds", "sector_2_seconds",
                             "sector_3_seconds"), start=1):
        col_a = f"Sector{i}Time_a_s"
        col_b = f"Sector{i}Time_b_s"
        if col_a in aligned.columns and col_b in aligned.columns:
            valid = aligned.dropna(subset=[col_a, col_b])
            if len(valid):
                deltas[key] = round(
                    (valid[col_a] - valid[col_b]).mean(), 3
                )
                continue
        deltas[key] = None
    return SectorDelta(**deltas)


def _empty_comparison(
    driver_a: str, driver_b: str, mode: str, last_n: int
) -> PaceComparison:
    return PaceComparison(
        driver_a=driver_a,
        driver_b=driver_b,
        mode=mode,
        window_description="no comparable laps",
        laps_compared=0,
        avg_lap_delta_seconds=None,
        median_lap_delta_seconds=None,
        trend_seconds_per_lap=None,
        trend_direction="insufficient_data",
        sector_delta=SectorDelta(None, None, None),
        driver_a_avg_lap_seconds=None,
        driver_b_avg_lap_seconds=None,
    )
