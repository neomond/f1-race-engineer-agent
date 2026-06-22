"""
Agent memory.

Two layers:

SHORT-TERM (rolling window):
  - Last N lap contexts (so the agent sees recent trends, not just current)
  - Last N radio calls (so it doesn't repeat itself, and follows up coherently)

LONG-TERM (event log):
  - Pinned race events: pit stops, SC/VSC periods, stint changes
  - Stint summaries (compound, length, avg pace) for completed stints

Design: structured JSON, not vector search. For a ~50-lap race with a
single driver of interest, we don't need RAG — we need clean state.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from src.data.lap_context import LapContext


# Window sizes tuned for ~50-lap races
SHORT_TERM_LAPS = 5
SHORT_TERM_CALLS = 5


@dataclass
class RadioCall:
    """A call the agent made (or chose not to make)."""
    lap_number: int
    message: Optional[str]            # None when stayed silent
    silent_reason: Optional[str]      # filled when message is None
    tools_used: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RaceEvent:
    """A pinned long-term event."""
    lap_number: int
    event_type: str   # 'pit_stop', 'sc_start', 'sc_end', 'stint_change', etc.
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


class AgentMemory:
    """Short-term + long-term memory for the race engineer agent."""

    def __init__(
        self,
        short_term_laps: int = SHORT_TERM_LAPS,
        short_term_calls: int = SHORT_TERM_CALLS,
    ):
        self.recent_contexts: deque[LapContext] = deque(maxlen=short_term_laps)
        self.recent_calls: deque[RadioCall] = deque(maxlen=short_term_calls)
        self.events: list[RaceEvent] = []
        self._last_stint: Optional[int] = None
        self._last_track_status: Optional[str] = None

    # ---- Short-term updates ----------------------------------------------

    def record_lap(self, context: LapContext) -> None:
        """Add a lap context; auto-detect pinned events vs previous state."""
        self._detect_events(context)
        self.recent_contexts.append(context)

    def record_call(self, call: RadioCall) -> None:
        self.recent_calls.append(call)

    # ---- Event detection -------------------------------------------------

    def _detect_events(self, ctx: LapContext) -> None:
        """Detect stint changes and SC/VSC transitions."""
        # Stint change → new pit stop
        if (
            self._last_stint is not None
            and ctx.stint_number is not None
            and ctx.stint_number != self._last_stint
        ):
            self.events.append(RaceEvent(
                lap_number=ctx.lap_number,
                event_type="pit_stop",
                description=(
                    f"Stint change: now on {ctx.tyre_compound or '?'} "
                    f"(stint {ctx.stint_number})"
                ),
            ))
        self._last_stint = ctx.stint_number

        # Track status transitions (green ↔ SC/VSC/yellow)
        prev = self._last_track_status
        curr = ctx.track_status
        if prev is not None and curr is not None and prev != curr:
            if curr in ("safety_car", "vsc"):
                self.events.append(RaceEvent(
                    lap_number=ctx.lap_number,
                    event_type=f"{curr}_start",
                    description=f"Track status: {curr}",
                ))
            elif prev in ("safety_car", "vsc") and curr == "green":
                self.events.append(RaceEvent(
                    lap_number=ctx.lap_number,
                    event_type=f"{prev}_end",
                    description=f"Track status: green (was {prev})",
                ))
        self._last_track_status = curr

    # ---- Prompt rendering ------------------------------------------------

    def render_for_prompt(self) -> str:
        """
        Build a compact string representation for the agent prompt.
        Newest items last (LLMs attend better to recent context).
        """
        parts: list[str] = []

        if self.events:
            recent_events = self.events[-5:]
            parts.append("RECENT EVENTS:")
            for e in recent_events:
                parts.append(f"  L{e.lap_number}: [{e.event_type}] {e.description}")

        if self.recent_calls:
            parts.append("\nRECENT RADIO CALLS (avoid repeating):")
            for c in self.recent_calls:
                if c.message:
                    parts.append(f"  L{c.lap_number}: \"{c.message}\"")
                else:
                    parts.append(f"  L{c.lap_number}: [silent — {c.silent_reason}]")

        if self.recent_contexts:
            parts.append("\nLAP TREND (last few laps):")
            for c in self.recent_contexts:
                lap_time = (
                    f"{c.lap_time_seconds:.2f}s" if c.lap_time_seconds else "—"
                )
                ahead = (
                    f"{c.ahead.driver_code} {c.ahead.gap_seconds:+.2f}"
                    if c.ahead else "—"
                )
                behind = (
                    f"{c.behind.driver_code} {c.behind.gap_seconds:+.2f}"
                    if c.behind else "—"
                )
                parts.append(
                    f"  L{c.lap_number} P{c.position} "
                    f"{c.tyre_compound}({c.tyre_age_laps}) "
                    f"{lap_time}  ahead={ahead}  behind={behind}"
                )

        return "\n".join(parts) if parts else "(no prior context)"

    # ---- Export ----------------------------------------------------------

    def to_log_dict(self) -> dict:
        """Full memory snapshot for end-of-race logging."""
        return {
            "events": [e.to_dict() for e in self.events],
            "all_calls": [c.to_dict() for c in self.recent_calls],
        }
