"""
Race data loader using FastF1.

Loads a session, caches it locally, and exposes clean accessors for
lap-by-lap state. All downstream tools and the agent consume data through
this module rather than touching FastF1 directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

# Cache to disk so we don't re-download on every run
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


@dataclass
class RaceMeta:
    """Static metadata about a race."""
    year: int
    name: str
    circuit: str
    total_laps: int
    pit_loss_seconds: float  # approximate pit stop time loss for this circuit


# Approximate pit loss values (seconds). Real teams compute these per race
# from FP data; these are reasonable defaults for v1.
PIT_LOSS_BY_CIRCUIT = {
    "Baku City Circuit": 19.5,
    "Hungaroring": 20.5,
    "Monza": 22.0,
    "Singapore Street Circuit": 28.0,
    "default": 22.0,
}


class RaceData:
    """Loaded race session with convenience accessors."""

    def __init__(self, year: int, race_name: str):
        self.year = year
        self.race_name = race_name
        logger.info(f"Loading {year} {race_name} GP...")

        self.session = fastf1.get_session(year, race_name, "R")
        self.session.load(telemetry=False, weather=True, messages=True)

        self.laps = self.session.laps
        self.weather = self.session.weather_data
        self.results = self.session.results

        circuit = self.session.event["EventName"]
        fallback_pit_loss = PIT_LOSS_BY_CIRCUIT.get(
            self.session.event.get("Location", ""),
            PIT_LOSS_BY_CIRCUIT["default"],
        )

        # Lazy import to avoid circular dependency
        from src.tools.pit_loss import estimate_pit_loss
        self.pit_loss_estimate = estimate_pit_loss(self.laps, fallback_pit_loss)

        self.meta = RaceMeta(
            year=year,
            name=race_name,
            circuit=circuit,
            total_laps=int(self.laps["LapNumber"].max()),
            pit_loss_seconds=self.pit_loss_estimate.seconds,
        )
        logger.info(
            f"Loaded {self.meta.total_laps} laps, "
            f"{len(self.results)} drivers, circuit: {self.meta.circuit}, "
            f"pit loss: {self.meta.pit_loss_seconds}s "
            f"({self.pit_loss_estimate.source})"
        )

    # ---- Accessors --------------------------------------------------------

    def driver_laps(self, driver_code: str) -> pd.DataFrame:
        """All laps for a driver (e.g. 'NOR', 'PIA', 'LEC')."""
        return self.laps.pick_drivers(driver_code).reset_index(drop=True)

    def lap_at(self, driver_code: str, lap_number: int) -> Optional[pd.Series]:
        """Single lap row for driver at given lap number."""
        df = self.driver_laps(driver_code)
        row = df[df["LapNumber"] == lap_number]
        return row.iloc[0] if len(row) else None

    def laps_up_to(self, driver_code: str, lap_number: int) -> pd.DataFrame:
        """All laps for a driver up to and including the given lap."""
        df = self.driver_laps(driver_code)
        return df[df["LapNumber"] <= lap_number].reset_index(drop=True)

    def current_stint_laps(
        self, driver_code: str, lap_number: int
    ) -> pd.DataFrame:
        """
        All clean-ish laps in the driver's CURRENT stint up to lap_number.

        The stint is determined by FastF1's Stint column, which increments
        on each pit stop. We return all laps where Stint matches the value
        at the most recent lap. Empty DataFrame if Stint data is missing.
        """
        df = self.laps_up_to(driver_code, lap_number)
        if not len(df) or "Stint" not in df.columns:
            return df
        if df["Stint"].isna().all():
            return df.iloc[0:0]  # empty
        current_stint = df["Stint"].iloc[-1]
        return df[df["Stint"] == current_stint].reset_index(drop=True)


    def all_drivers_at(self, lap_number: int) -> pd.DataFrame:
        """All drivers' lap data at a specific lap number."""
        return self.laps[self.laps["LapNumber"] == lap_number].reset_index(drop=True)

    def weather_at(self, lap_number: int) -> Optional[pd.Series]:
        """Closest weather sample to the given lap."""
        if self.weather is None or len(self.weather) == 0:
            return None
        # weather samples are time-indexed; map via lap start time
        ref_lap = self.laps[self.laps["LapNumber"] == lap_number]
        if not len(ref_lap):
            return None
        ref_time = ref_lap.iloc[0]["LapStartTime"]
        if pd.isna(ref_time):
            return None
        idx = (self.weather["Time"] - ref_time).abs().idxmin()
        return self.weather.loc[idx]

    def driver_code_to_name(self, code: str) -> str:
        """e.g. 'NOR' -> 'Lando Norris'."""
        row = self.results[self.results["Abbreviation"] == code]
        if len(row):
            return f"{row.iloc[0]['FirstName']} {row.iloc[0]['LastName']}"
        return code
