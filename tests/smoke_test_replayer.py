"""Quick smoke test: load Azerbaijan 2024 and replay first 5 laps for Norris."""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.data.race_data import RaceData
from src.data.replayer import RaceReplayer


def main():
    race = RaceData(2024, "Azerbaijan")
    print(f"\n=== {race.meta.year} {race.meta.name} GP ===")
    print(f"Circuit: {race.meta.circuit}")
    print(f"Total laps: {race.meta.total_laps}")
    print(f"Pit loss: {race.meta.pit_loss_seconds}s\n")

    replayer = RaceReplayer(race, driver_code="NOR", start_lap=1, end_lap=5)

    for ctx in replayer:
        ahead = (
            f"{ctx.ahead.driver_code} {ctx.ahead.gap_seconds:+.2f}s"
            if ctx.ahead else "—"
        )
        behind = (
            f"{ctx.behind.driver_code} {ctx.behind.gap_seconds:+.2f}s"
            if ctx.behind else "—"
        )
        lap_time = (
            f"{ctx.lap_time_seconds:.3f}s" if ctx.lap_time_seconds else "—"
        )
        print(
            f"L{ctx.lap_number:2d}  P{ctx.position}  "
            f"{ctx.tyre_compound or '?':<6} age={ctx.tyre_age_laps}  "
            f"lap={lap_time}  ahead={ahead}  behind={behind}"
        )


if __name__ == "__main__":
    main()
