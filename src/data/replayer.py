"""
Race replayer.

Iterates through a race lap-by-lap, yielding LapContext snapshots as if
the race were unfolding live. The agent consumes one context per tick.

Usage:
    race = RaceData(2024, "Azerbaijan")
    replayer = RaceReplayer(race, driver_code="NOR")
    for context in replayer:
        # feed context to agent
        ...
"""
from __future__ import annotations

from typing import Iterator, Optional

from src.data.race_data import RaceData
from src.data.lap_context import LapContext, build_lap_context


class RaceReplayer:
    """Lap-by-lap iterator over a race."""

    def __init__(
        self,
        race: RaceData,
        driver_code: str,
        start_lap: int = 1,
        end_lap: Optional[int] = None,
    ):
        self.race = race
        self.driver_code = driver_code
        self.start_lap = start_lap
        self.end_lap = end_lap or race.meta.total_laps

    def __iter__(self) -> Iterator[LapContext]:
        for lap_num in range(self.start_lap, self.end_lap + 1):
            ctx = build_lap_context(self.race, self.driver_code, lap_num)
            if ctx is not None:
                yield ctx
