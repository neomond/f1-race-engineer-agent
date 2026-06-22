"""
Demo: run compare_pace on real Baku 2024 data.

This is the end-to-end check that the tool works with real FastF1 data,
not just synthetic test inputs. Run after smoke_test_replayer.py succeeds.
"""
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

from src.data.race_data import RaceData
from src.tools.compare_pace import compare_pace
from src.tools.registry import dispatch_tool


def main():
    print("Loading 2024 Azerbaijan GP...")
    race = RaceData(2024, "Azerbaijan")
    print(f"Loaded. Total laps: {race.meta.total_laps}\n")

    # Scenario 1: Norris vs Piastri at lap 20 (early in race)
    print("=" * 70)
    print("Scenario 1: NOR vs PIA at lap 20 — last 5 laps")
    print("=" * 70)
    result = compare_pace(race, "NOR", "PIA", current_lap=20, mode="last_n", last_n=5)
    _print_result(result)

    # Scenario 2: Norris vs Leclerc at lap 35 — full stint
    print("\n" + "=" * 70)
    print("Scenario 2: NOR vs LEC at lap 35 — entire current stint")
    print("=" * 70)
    result = compare_pace(race, "NOR", "LEC", current_lap=35, mode="stint")
    _print_result(result)

    # Scenario 3: agent-style invocation through dispatcher
    print("\n" + "=" * 70)
    print("Scenario 3: agent-style call through dispatch_tool (returns JSON)")
    print("=" * 70)
    json_result = dispatch_tool(
        "compare_pace",
        race,
        current_lap=40,
        args={"driver_a": "NOR", "driver_b": "PIA", "mode": "last_n", "last_n": 5},
    )
    parsed = json.loads(json_result)
    print(json.dumps(parsed, indent=2))


def _print_result(r):
    print(f"  Window: {r.window_description}")
    print(f"  Laps compared: {r.laps_compared}")
    print(f"  {r.driver_a} avg: {r.driver_a_avg_lap_seconds}s")
    print(f"  {r.driver_b} avg: {r.driver_b_avg_lap_seconds}s")
    print(f"  Avg delta:    {r.avg_lap_delta_seconds:+.3f}s "
          f"({'A faster' if r.avg_lap_delta_seconds < 0 else 'B faster'})")
    print(f"  Median delta: {r.median_lap_delta_seconds:+.3f}s")
    print(f"  Trend:        {r.trend_direction} "
          f"({r.trend_seconds_per_lap:+.3f}s/lap)")
    sd = r.sector_delta
    print(f"  Sector deltas: S1={sd.sector_1_seconds:+.3f}  "
          f"S2={sd.sector_2_seconds:+.3f}  S3={sd.sector_3_seconds:+.3f}")


if __name__ == "__main__":
    main()
