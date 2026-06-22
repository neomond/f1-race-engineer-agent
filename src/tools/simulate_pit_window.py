"""
simulate_pit_window tool.

Given a target driver and a rival, simulate two strategic scenarios:

  UNDERCUT: target pits THIS lap, rival pits NEXT lap.
  OVERCUT:  rival pits THIS lap, target pits NEXT lap.

Both scenarios assume both drivers eventually pit — pit_loss is paid by
everyone and CANCELS OUT in the gap calculation. What actually moves the
needle is the one-lap window where one driver is on fresh rubber and the
other is on old. Whoever has the fresh tyres in that window gains
`fresh_tyre_advantage_s` on the other.

We also report the "emergence gap" — the absolute gap to the rival the
moment the target rejoins from a stop, before tyre swing kicks in. This
tells you whether the target rejoins in clean air or in traffic, which is
a real strategic risk separate from the eventual gap.

The model is deliberately simple — a 1-lap horizon. Real teams run multi-
lap Monte Carlo simulations with degradation curves; for an agent making
real-time calls, the 1-lap horizon captures the right signal without false
precision. The agent can chain calls across laps to reason longer-term.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from src.data.race_data import RaceData


DEFAULT_FRESH_TYRE_ADVANTAGE_S = 1.2


@dataclass
class ScenarioResult:
    """Outcome of one strategic scenario."""
    projected_gap_seconds: float           # gap once both have pitted. + = behind
    verdict: str                           # ahead_of_rival | behind_rival | alongside
    margin_vs_status_quo_seconds: float    # + = target gained vs current_gap
    emergence_gap_seconds: float           # absolute gap during the one-lap window

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PitWindowResult:
    target_driver: str
    rival_driver: str
    current_lap: int

    current_gap_seconds: float
    pit_loss_seconds: float
    pit_loss_source: str
    fresh_tyre_advantage_s: float

    undercut: ScenarioResult
    overcut: ScenarioResult

    recommendation: str

    def to_dict(self) -> dict:
        return {
            "target_driver": self.target_driver,
            "rival_driver": self.rival_driver,
            "current_lap": self.current_lap,
            "current_gap_seconds": self.current_gap_seconds,
            "pit_loss_seconds": self.pit_loss_seconds,
            "pit_loss_source": self.pit_loss_source,
            "fresh_tyre_advantage_s": self.fresh_tyre_advantage_s,
            "undercut": self.undercut.to_dict(),
            "overcut": self.overcut.to_dict(),
            "recommendation": self.recommendation,
        }


def simulate_pit_window(
    race: RaceData,
    target_driver: str,
    rival_driver: str,
    current_lap: int,
    fresh_tyre_advantage_s: float = DEFAULT_FRESH_TYRE_ADVANTAGE_S,
) -> Optional[PitWindowResult]:
    """Simulate undercut and overcut for target vs rival."""
    current_gap = _compute_current_gap(race, target_driver, rival_driver, current_lap)
    if current_gap is None:
        return None

    pit_loss = race.meta.pit_loss_seconds
    pit_loss_source = race.pit_loss_estimate.source

    # UNDERCUT: target pits now, rival pits next lap.
    # During the 1-lap window: target is fresh, rival is old → target gains.
    # Once both have stopped, the pit_loss cost is paid by both and cancels.
    undercut_final_gap = current_gap - fresh_tyre_advantage_s
    # Emergence: the moment target rejoins, before rival has pitted.
    # Target has just paid pit_loss; absolute gap to rival is current + pit_loss.
    undercut_emergence = current_gap + pit_loss

    undercut = ScenarioResult(
        projected_gap_seconds=round(undercut_final_gap, 2),
        verdict=_classify_gap(undercut_final_gap),
        margin_vs_status_quo_seconds=round(current_gap - undercut_final_gap, 2),
        emergence_gap_seconds=round(undercut_emergence, 2),
    )

    # OVERCUT: rival pits now, target pits next lap.
    # During the 1-lap window: rival is fresh, target is old → rival gains.
    overcut_final_gap = current_gap + fresh_tyre_advantage_s
    # Emergence (from rival's perspective): right after rival's stop,
    # target leads/trails by pit_loss less than before.
    overcut_emergence = current_gap - pit_loss

    overcut = ScenarioResult(
        projected_gap_seconds=round(overcut_final_gap, 2),
        verdict=_classify_gap(overcut_final_gap),
        margin_vs_status_quo_seconds=round(current_gap - overcut_final_gap, 2),
        emergence_gap_seconds=round(overcut_emergence, 2),
    )

    recommendation = _build_recommendation(
        target_driver, rival_driver, current_gap, undercut, overcut,
        fresh_tyre_advantage_s,
    )

    return PitWindowResult(
        target_driver=target_driver,
        rival_driver=rival_driver,
        current_lap=current_lap,
        current_gap_seconds=round(current_gap, 2),
        pit_loss_seconds=pit_loss,
        pit_loss_source=pit_loss_source,
        fresh_tyre_advantage_s=fresh_tyre_advantage_s,
        undercut=undercut,
        overcut=overcut,
        recommendation=recommendation,
    )


# ---- Helpers --------------------------------------------------------------


def _compute_current_gap(
    race: RaceData, target: str, rival: str, lap: int
) -> Optional[float]:
    """Gap from target to rival at end of lap. + = target behind."""
    field = race.all_drivers_at(lap)
    if "Time" not in field.columns:
        return None

    target_row = field[field["Driver"] == target]
    rival_row = field[field["Driver"] == rival]
    if not len(target_row) or not len(rival_row):
        return None

    t_target = target_row.iloc[0]["Time"]
    t_rival = rival_row.iloc[0]["Time"]
    if pd.isna(t_target) or pd.isna(t_rival):
        return None

    return (t_target - t_rival).total_seconds()


def _classify_gap(gap: float) -> str:
    if gap < -1.0:
        return "ahead_of_rival"
    if gap > 1.0:
        return "behind_rival"
    return "alongside"


def _build_recommendation(
    target: str,
    rival: str,
    current_gap: float,
    undercut: ScenarioResult,
    overcut: ScenarioResult,
    fresh_advantage: float,
) -> str:
    """One-sentence summary the agent can quote."""
    relation = (
        f"{target} ahead of {rival} by {abs(current_gap):.1f}s"
        if current_gap < 0
        else f"{target} behind {rival} by {abs(current_gap):.1f}s"
    )

    swing = round(2 * fresh_advantage, 2)  # spread between the two scenarios

    if current_gap > 0:
        # Target behind — undercut shrinks the gap.
        action = (
            f"Undercut favoured: pitting first projects "
            f"{undercut.projected_gap_seconds:+.1f}s vs rival "
            f"(overcut: {overcut.projected_gap_seconds:+.1f}s, "
            f"swing {swing:.1f}s). "
            f"Out-lap emergence ~{undercut.emergence_gap_seconds:+.1f}s "
            f"behind rival — verify no traffic before committing."
        )
    elif current_gap < 0:
        # Target ahead — overcut keeps the rival on old rubber longer.
        action = (
            f"Overcut favoured: staying out projects "
            f"{overcut.projected_gap_seconds:+.1f}s "
            f"(undercut: {undercut.projected_gap_seconds:+.1f}s, "
            f"swing {swing:.1f}s)."
        )
    else:
        action = (
            f"Cars level. Whoever pits first inherits the fresh-tyre "
            f"advantage of ~{fresh_advantage:.1f}s/lap."
        )

    return f"{relation}. {action}"
