"""
stay_silent tool.

A meta-tool that lets the agent explicitly decline to make a radio call on
the current lap. The model logs a brief internal reason (for evaluation
and debugging), but no message is delivered to the driver.

Design rationale:
  - Without this, models tend to chatter — generating plausible-sounding
    but useless calls every lap.
  - Making silence an explicit, equal-footing tool choice removes the
    pressure to produce text, while still requiring a reasoned decision.
  - The "reason" field is invaluable in evaluation: did the agent stay
    quiet because nothing was happening, or did it miss something?
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent.llm_client import ToolSpec
from src.data.race_data import RaceData


@dataclass
class StaySilentResult:
    reason: str
    lap_number: int

    def to_dict(self) -> dict:
        return {"reason": self.reason, "lap_number": self.lap_number}


STAY_SILENT_SPEC = ToolSpec(
    name="stay_silent",
    description=(
        "Decline to make a radio call this lap. Use when the race state is "
        "stable, no strategic decision is pending, and there is nothing the "
        "driver needs to know. Briefly note your reasoning for the log. "
        "Calling this is preferred over generating filler chatter."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Brief internal note explaining why no call is warranted, "
                    "e.g. 'pace stable, gaps unchanged, no pit window open'."
                ),
            },
        },
        "required": ["reason"],
    },
)


def handle_stay_silent(
    race: RaceData, current_lap: int, args: dict[str, Any]
) -> str:
    result = StaySilentResult(
        reason=args.get("reason", "no reason given"),
        lap_number=current_lap,
    )
    return json.dumps(result.to_dict())
