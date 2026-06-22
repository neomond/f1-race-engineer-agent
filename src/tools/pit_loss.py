"""
Pit loss estimation.

Computes the time cost of a pit stop for this specific race by analysing
actual pit stops that occurred. More accurate than a static circuit
average because it accounts for this weekend's pit lane speed limit
enforcement, traffic, weather, etc.

Method:
  pit_loss = (in_lap_time + out_lap_time) - 2 * baseline_lap_time

  where baseline_lap_time is a clean reference lap for that driver
  (median of their non-pit laps in the same stint).

We average across all valid pit stops in the race. Outliers (botched stops,
SC pit stops) get filtered with simple percentile bounds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PitLossEstimate:
    seconds: float
    sample_size: int
    source: str  # 'measured' or 'fallback'
    raw_values: list[float]


def estimate_pit_loss(
    laps: pd.DataFrame,
    fallback_seconds: float,
    min_samples: int = 3,
) -> PitLossEstimate:
    """
    Estimate pit loss in seconds from actual stops in the race.

    Args:
        laps: full session laps DataFrame from FastF1.
        fallback_seconds: circuit-average value to use if too few samples.
        min_samples: minimum number of clean stops needed to trust measurement.

    Returns:
        PitLossEstimate with measured value if possible, else fallback.
    """
    raw_values: list[float] = []

    for driver in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
        if not len(driver_laps):
            continue

        in_laps = driver_laps[driver_laps["PitInTime"].notna()]

        for _, in_lap in in_laps.iterrows():
            in_lap_num = in_lap["LapNumber"]
            stint = in_lap.get("Stint")

            # Find matching out-lap (next lap for same driver)
            out_lap_row = driver_laps[
                driver_laps["LapNumber"] == in_lap_num + 1
            ]
            if not len(out_lap_row):
                continue
            out_lap = out_lap_row.iloc[0]

            # Both laps need recorded times
            if pd.isna(in_lap["LapTime"]) or pd.isna(out_lap["LapTime"]):
                continue

            # Baseline: median of clean (non-pit) laps in the IN-lap's stint
            baseline = _baseline_for_stint(driver_laps, stint)
            if baseline is None:
                continue

            in_s = in_lap["LapTime"].total_seconds()
            out_s = out_lap["LapTime"].total_seconds()
            loss = (in_s + out_s) - (2 * baseline)

            # Sanity bound — real pit losses are 15–35s on most circuits.
            # Outside this is almost always a SC stop or data anomaly.
            if 12.0 <= loss <= 45.0:
                raw_values.append(loss)

    if len(raw_values) < min_samples:
        logger.info(
            f"Only {len(raw_values)} clean pit samples, using fallback "
            f"{fallback_seconds}s"
        )
        return PitLossEstimate(
            seconds=fallback_seconds,
            sample_size=len(raw_values),
            source="fallback",
            raw_values=raw_values,
        )

    # Trim outliers (top/bottom 10%) before averaging
    arr = np.array(raw_values)
    lo, hi = np.percentile(arr, [10, 90])
    trimmed = arr[(arr >= lo) & (arr <= hi)]
    measured = float(np.mean(trimmed))

    logger.info(
        f"Measured pit loss: {measured:.2f}s "
        f"({len(raw_values)} stops, trimmed mean of middle 80%)"
    )
    return PitLossEstimate(
        seconds=round(measured, 2),
        sample_size=len(raw_values),
        source="measured",
        raw_values=raw_values,
    )


def _baseline_for_stint(
    driver_laps: pd.DataFrame, stint: Optional[float]
) -> Optional[float]:
    """Median lap time on clean laps in this stint."""
    if stint is None or pd.isna(stint):
        return None
    stint_laps = driver_laps[
        (driver_laps["Stint"] == stint)
        & driver_laps["PitInTime"].isna()
        & driver_laps["PitOutTime"].isna()
        & driver_laps["LapTime"].notna()
    ]
    if len(stint_laps) < 3:
        return None
    times = stint_laps["LapTime"].dt.total_seconds()
    return float(times.median())
