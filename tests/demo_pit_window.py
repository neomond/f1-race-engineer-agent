"""
Demo: run simulate_pit_window on real Baku 2024 data.
"""
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

from src.data.race_data import RaceData
from src.tools.simulate_pit_window import simulate_pit_window
from src.tools.registry import dispatch_tool


def main():
    print("Loading 2024 Azerbaijan GP...")
    race = RaceData(2024, "Azerbaijan")
    print(f"\nMeasured pit loss: {race.meta.pit_loss_seconds}s "
          f"({race.pit_loss_estimate.source}, "
          f"{race.pit_loss_estimate.sample_size} samples)\n")

    # Scenario 1: Norris vs car ahead at lap 18 (just before Norris pit window)
    print("=" * 70)
    print("Scenario 1: NOR vs PIA at lap 18")
    print("=" * 70)
    result = simulate_pit_window(race, "NOR", "PIA", current_lap=18)
    _print(result)

    # Scenario 2: Mid-race, Leclerc P2 leading Piastri P1 — undercut classic
    print("\n" + "=" * 70)
    print("Scenario 2: LEC vs PIA at lap 17 (LEC leading early, PIA chasing)")
    print("=" * 70)
    result = simulate_pit_window(race, "PIA", "LEC", current_lap=17)
    _print(result)

    # Scenario 3: Late race - Norris running down Russell
    print("\n" + "=" * 70)
    print("Scenario 3: NOR vs RUS at lap 40")
    print("=" * 70)
    result = simulate_pit_window(race, "NOR", "RUS", current_lap=40)
    _print(result)

    # Scenario 4: through dispatcher (agent-style)
    print("\n" + "=" * 70)
    print("Scenario 4: agent-style call via dispatch_tool")
    print("=" * 70)
    json_result = dispatch_tool(
        "simulate_pit_window", race, current_lap=15,
        args={"target_driver": "NOR", "rival_driver": "PIA"},
    )
    print(json.dumps(json.loads(json_result), indent=2))


def _print(r):
    if r is None:
        print("  (No result — data unavailable)")
        return
    print(f"  Current gap:  {r.current_gap_seconds:+.2f}s "
          f"({'target ahead' if r.current_gap_seconds < 0 else 'target behind'})")
    print(f"  Pit loss:     {r.pit_loss_seconds}s ({r.pit_loss_source})")
    print(f"  Fresh tyre:   {r.fresh_tyre_advantage_s}s/lap advantage")
    print()
    print(f"  UNDERCUT — pit now, rival pits next lap:")
    print(f"    Projected final gap: {r.undercut.projected_gap_seconds:+.2f}s "
          f"({r.undercut.verdict})")
    print(f"    Margin vs status quo: "
          f"{r.undercut.margin_vs_status_quo_seconds:+.2f}s")
    print(f"    Out-lap emergence: {r.undercut.emergence_gap_seconds:+.2f}s "
          f"(check for traffic)")
    print()
    print(f"  OVERCUT — rival pits now, target pits next lap:")
    print(f"    Projected final gap: {r.overcut.projected_gap_seconds:+.2f}s "
          f"({r.overcut.verdict})")
    print(f"    Margin vs status quo: "
          f"{r.overcut.margin_vs_status_quo_seconds:+.2f}s")
    print(f"    Out-lap emergence: {r.overcut.emergence_gap_seconds:+.2f}s")
    print()
    print(f"  💡 {r.recommendation}")


if __name__ == "__main__":
    main()
