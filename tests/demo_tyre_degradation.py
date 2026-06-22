"""
Demo: tyre degradation tool on real Baku data.

Shows the fitter recovering degradation rates from actual F1 stints across
multiple drivers, and the head-to-head comparison the agent will see.
"""
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

from src.data.race_data import RaceData
from src.tools.tyre_degradation import compare_degradation
from src.tools.registry import dispatch_tool


def main():
    print("Loading 2024 Azerbaijan GP...")
    race = RaceData(2024, "Azerbaijan")
    print()

    # Scenario 1: target only — Norris at lap 25
    print("=" * 70)
    print("Scenario 1: NOR degradation alone, lap 25")
    print("=" * 70)
    target_stint = race.current_stint_laps("NOR", 25)
    result = compare_degradation(
        target_stint=target_stint,
        rival_stint=None,
        target_code="NOR",
        rival_code=None,
        project_laps_ahead=5,
    )
    _print(result)

    # Scenario 2: NOR vs PIA — same constructor, comparable tyres
    print("\n" + "=" * 70)
    print("Scenario 2: NOR vs PIA degradation, lap 25 (teammates)")
    print("=" * 70)
    target = race.current_stint_laps("NOR", 25)
    rival = race.current_stint_laps("PIA", 25)
    result = compare_degradation(
        target_stint=target, rival_stint=rival,
        target_code="NOR", rival_code="PIA",
        project_laps_ahead=5,
    )
    _print(result)

    # Scenario 3: NOR vs LEC late in race — different tyre situations
    print("\n" + "=" * 70)
    print("Scenario 3: NOR vs LEC degradation, lap 40")
    print("=" * 70)
    target = race.current_stint_laps("NOR", 40)
    rival = race.current_stint_laps("LEC", 40)
    result = compare_degradation(
        target_stint=target, rival_stint=rival,
        target_code="NOR", rival_code="LEC",
        project_laps_ahead=5,
    )
    _print(result)

    # Scenario 4: through the dispatcher
    print("\n" + "=" * 70)
    print("Scenario 4: agent-style call via dispatch_tool")
    print("=" * 70)
    json_result = dispatch_tool(
        "get_tyre_degradation", race, current_lap=30,
        args={"target_driver": "NOR", "rival_driver": "PIA",
              "project_laps_ahead": 5},
    )
    print(json.dumps(json.loads(json_result), indent=2))


def _print(r):
    tf = r.target_fit
    print(f"\n  TARGET: {tf.driver_code}")
    if tf.model == "insufficient_data":
        print(f"    ⚠️  {tf.error}")
    else:
        print(f"    Model: {tf.model}  ({tf.n_laps_in_fit} clean laps in fit)")
        print(f"    Compound: {tf.tyre_compound}  Age: {tf.current_tyre_age}")
        print(f"    Slope: {tf.slope_seconds_per_lap:+.3f}s/lap")
        print(f"    Fit quality: adjR² = {tf.adjusted_r_squared:.3f} "
              f"(lin {tf.linear_adjusted_r_squared:.3f}, "
              f"quad {tf.quadratic_adjusted_r_squared if tf.quadratic_adjusted_r_squared is not None else 'n/a'})")

    if r.rival_fit:
        rf = r.rival_fit
        print(f"\n  RIVAL: {rf.driver_code}")
        if rf.model == "insufficient_data":
            print(f"    ⚠️  {rf.error}")
        else:
            print(f"    Model: {rf.model}  ({rf.n_laps_in_fit} clean laps)")
            print(f"    Compound: {rf.tyre_compound}  Age: {rf.current_tyre_age}")
            print(f"    Slope: {rf.slope_seconds_per_lap:+.3f}s/lap")

    if r.delta_seconds_per_lap is not None:
        print(f"\n  Delta: {r.delta_seconds_per_lap:+.3f}s/lap "
              f"({'target' if r.delta_seconds_per_lap > 0 else 'rival'} degrading faster)")
        print(f"  Verdict: {r.verdict}")

    if r.target_projection and r.target_projection.projected_lap_time_seconds:
        tp = r.target_projection
        print(f"\n  Projection {tp.laps_ahead} laps ahead:")
        print(f"    Target lap time: {tp.projected_lap_time_seconds}s "
              f"(+{tp.projected_total_loss_seconds}s vs now)")
        if r.rival_projection and r.rival_projection.projected_lap_time_seconds:
            rp = r.rival_projection
            print(f"    Rival lap time:  {rp.projected_lap_time_seconds}s "
                  f"(+{rp.projected_total_loss_seconds}s vs now)")


if __name__ == "__main__":
    main()
