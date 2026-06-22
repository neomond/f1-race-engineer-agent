"""
Lap context builder.

Each lap, we construct a structured snapshot of the race state from the
target driver's perspective. This is what the agent sees on every tick.

Design principle: keep this purely descriptive. No analysis, no opinions,
no recommendations. The agent does the reasoning; this just reports facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

from src.data.race_data import RaceData


@dataclass
class GapInfo:
    driver_code: str
    driver_name: str
    gap_seconds: Optional[float]  # negative = ahead, positive = behind, None if N/A


@dataclass
class LapContext:
    """Snapshot of race state at a given lap, from one driver's POV."""
    lap_number: int
    total_laps: int
    laps_remaining: int

    # Target driver state
    driver_code: str
    driver_name: str
    position: Optional[int]
    lap_time_seconds: Optional[float]
    last_3_lap_times: list[float] = field(default_factory=list)
    tyre_compound: Optional[str] = None
    tyre_age_laps: Optional[int] = None
    stint_number: Optional[int] = None
    in_pit: bool = False

    # Race context
    ahead: Optional[GapInfo] = None
    behind: Optional[GapInfo] = None

    # Track conditions
    track_temp_c: Optional[float] = None
    air_temp_c: Optional[float] = None
    rainfall: bool = False

    # Race control
    track_status: Optional[str] = None  # e.g. 'green', 'yellow', 'safety_car'

    def to_dict(self) -> dict:
        return asdict(self)


_TRACK_STATUS_MAP = {
    "1": "green",
    "2": "yellow",
    "4": "safety_car",
    "5": "red",
    "6": "vsc",
    "7": "vsc_ending",
}


def build_lap_context(
    race: RaceData, driver_code: str, lap_number: int
) -> Optional[LapContext]:
    """Build a structured snapshot for the agent."""
    lap = race.lap_at(driver_code, lap_number)
    if lap is None:
        return None

    driver_name = race.driver_code_to_name(driver_code)

    # Recent pace (last 3 lap times where available)
    history = race.laps_up_to(driver_code, lap_number).tail(3)
    recent_times = [
        t.total_seconds()
        for t in history["LapTime"].tolist()
        if pd.notna(t)
    ]

    # Position + gaps
    position = int(lap["Position"]) if pd.notna(lap.get("Position")) else None
    ahead, behind = _compute_gaps(race, driver_code, lap_number, position)

    # Tyre info
    compound = lap.get("Compound") if pd.notna(lap.get("Compound")) else None
    tyre_age = int(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else None
    stint = int(lap["Stint"]) if pd.notna(lap.get("Stint")) else None

    # Pit status — FastF1 marks PitInTime / PitOutTime on relevant laps
    in_pit = pd.notna(lap.get("PitInTime")) or pd.notna(lap.get("PitOutTime"))

    # Weather
    weather = race.weather_at(lap_number)
    track_temp = float(weather["TrackTemp"]) if weather is not None else None
    air_temp = float(weather["AirTemp"]) if weather is not None else None
    rainfall = bool(weather["Rainfall"]) if weather is not None else False

    # Track status
    track_status_raw = lap.get("TrackStatus")
    track_status = (
        _TRACK_STATUS_MAP.get(str(track_status_raw)[0], "green")
        if pd.notna(track_status_raw)
        else None
    )

    lap_time = (
        lap["LapTime"].total_seconds() if pd.notna(lap.get("LapTime")) else None
    )

    return LapContext(
        lap_number=lap_number,
        total_laps=race.meta.total_laps,
        laps_remaining=race.meta.total_laps - lap_number,
        driver_code=driver_code,
        driver_name=driver_name,
        position=position,
        lap_time_seconds=lap_time,
        last_3_lap_times=recent_times,
        tyre_compound=compound,
        tyre_age_laps=tyre_age,
        stint_number=stint,
        in_pit=in_pit,
        ahead=ahead,
        behind=behind,
        track_temp_c=track_temp,
        air_temp_c=air_temp,
        rainfall=rainfall,
        track_status=track_status,
    )


def _compute_gaps(
    race: RaceData,
    driver_code: str,
    lap_number: int,
    position: Optional[int],
) -> tuple[Optional[GapInfo], Optional[GapInfo]]:
    """Compute gap to car ahead and behind based on lap-end timing."""
    if position is None:
        return None, None

    field_at_lap = race.all_drivers_at(lap_number)
    if "Position" not in field_at_lap.columns:
        return None, None

    field_at_lap = field_at_lap.dropna(subset=["Position", "Time"]).sort_values("Position")
    if not len(field_at_lap):
        return None, None

    target_row = field_at_lap[field_at_lap["Driver"] == driver_code]
    if not len(target_row):
        return None, None

    target_time = target_row.iloc[0]["Time"]
    ahead_info = behind_info = None

    ahead_row = field_at_lap[field_at_lap["Position"] == position - 1]
    if len(ahead_row):
        gap = (target_time - ahead_row.iloc[0]["Time"]).total_seconds()
        code = ahead_row.iloc[0]["Driver"]
        ahead_info = GapInfo(
            driver_code=code,
            driver_name=race.driver_code_to_name(code),
            gap_seconds=-abs(gap),
        )

    behind_row = field_at_lap[field_at_lap["Position"] == position + 1]
    if len(behind_row):
        gap = (behind_row.iloc[0]["Time"] - target_time).total_seconds()
        code = behind_row.iloc[0]["Driver"]
        behind_info = GapInfo(
            driver_code=code,
            driver_name=race.driver_code_to_name(code),
            gap_seconds=abs(gap),
        )

    return ahead_info, behind_info
