"""
Tool registry.

Each tool the agent can call is defined here as:
  1. A ToolSpec (schema seen by the LLM)
  2. A handler function that executes it given a RaceData context

The agent loop dispatches tool calls by name through TOOL_HANDLERS.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src.agent.llm_client import ToolSpec
from src.data.race_data import RaceData
from src.tools.compare_pace import compare_pace
from src.tools.simulate_pit_window import simulate_pit_window
from src.tools.stay_silent import STAY_SILENT_SPEC, handle_stay_silent
from src.tools.tyre_degradation import compare_degradation


# ---- Tool: compare_pace ---------------------------------------------------

COMPARE_PACE_SPEC = ToolSpec(
    name="compare_pace",
    description=(
        "Compare lap pace between two drivers over a window. Returns average "
        "and median lap-time delta, sector-by-sector deltas, and a trend "
        "(closing/stable/losing). Negative deltas mean driver_a is faster. "
        "Use this to assess threats from behind, undercut potential ahead, "
        "or pace consistency vs a rival."
    ),
    parameters={
        "type": "object",
        "properties": {
            "driver_a": {
                "type": "string",
                "description": "Target driver code, e.g. 'NOR', 'PIA', 'LEC'.",
            },
            "driver_b": {
                "type": "string",
                "description": "Comparison driver code.",
            },
            "mode": {
                "type": "string",
                "enum": ["last_n", "stint"],
                "description": (
                    "'last_n' for a sliding window of recent laps "
                    "(use for current pace). 'stint' for entire current "
                    "stint (use for stint-long trends)."
                ),
            },
            "last_n": {
                "type": "integer",
                "description": (
                    "Window size when mode='last_n'. Default 5. "
                    "Ignored when mode='stint'."
                ),
            },
        },
        "required": ["driver_a", "driver_b", "mode"],
    },
)


def _handle_compare_pace(race: RaceData, current_lap: int, args: dict[str, Any]) -> str:
    result = compare_pace(
        race=race,
        driver_a=args["driver_a"],
        driver_b=args["driver_b"],
        current_lap=current_lap,
        mode=args.get("mode", "last_n"),
        last_n=args.get("last_n", 5),
    )
    return json.dumps(result.to_dict())


# ---- Tool: simulate_pit_window -------------------------------------------

SIMULATE_PIT_WINDOW_SPEC = ToolSpec(
    name="simulate_pit_window",
    description=(
        "Simulate undercut and overcut scenarios between the target driver "
        "and a rival. Returns the projected gap one lap after each scenario "
        "plus a recommendation. Use when considering whether to pit early "
        "(undercut), let a rival pit first (overcut), or hold position. "
        "Pit loss is measured from this race's actual stops where possible."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_driver": {
                "type": "string",
                "description": "The driver we are advising, e.g. 'NOR'.",
            },
            "rival_driver": {
                "type": "string",
                "description": (
                    "The competitor to evaluate against (typically the car "
                    "directly ahead or behind), e.g. 'PIA', 'LEC'."
                ),
            },
            "fresh_tyre_advantage_s": {
                "type": "number",
                "description": (
                    "Optional: seconds-per-lap advantage of fresh tyres vs "
                    "current rubber. Default 1.2. Use 0.8 for hard "
                    "compounds in cool conditions; 1.5+ for very degraded "
                    "softs."
                ),
            },
        },
        "required": ["target_driver", "rival_driver"],
    },
)


def _handle_simulate_pit_window(
    race: RaceData, current_lap: int, args: dict[str, Any]
) -> str:
    result = simulate_pit_window(
        race=race,
        target_driver=args["target_driver"],
        rival_driver=args["rival_driver"],
        current_lap=current_lap,
        fresh_tyre_advantage_s=args.get("fresh_tyre_advantage_s", 1.2),
    )
    if result is None:
        return json.dumps({
            "error": "Could not compute current gap (data unavailable)."
        })
    return json.dumps(result.to_dict())


# ---- Tool: get_tyre_degradation ------------------------------------------

GET_TYRE_DEGRADATION_SPEC = ToolSpec(
    name="get_tyre_degradation",
    description=(
        "Fit a degradation curve to the target driver's current stint and "
        "(optionally) a rival's. Returns degradation rate in seconds/lap, "
        "projected lap time N laps ahead, and a verdict comparing the two "
        "drivers' degradation. Linear and quadratic models are both fit and "
        "the better one (by adjusted R²) is reported. "
        "Use when deciding whether tyres are still alive, when pit pressure "
        "is building, or when explaining why pace is dropping."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_driver": {
                "type": "string",
                "description": "The driver we are advising, e.g. 'NOR'.",
            },
            "rival_driver": {
                "type": "string",
                "description": (
                    "Optional: a competitor to compare degradation against. "
                    "Useful for 'are my tyres going off faster than theirs?' "
                    "questions. Pass empty string or omit for target-only."
                ),
            },
            "project_laps_ahead": {
                "type": "integer",
                "description": (
                    "How many laps ahead to forecast. Default 5. Use 3 for "
                    "near-term decisions, 10 for stint-length planning."
                ),
            },
        },
        "required": ["target_driver"],
    },
)


def _handle_get_tyre_degradation(
    race: RaceData, current_lap: int, args: dict[str, Any]
) -> str:
    target = args["target_driver"]
    rival = args.get("rival_driver") or None
    laps_ahead = args.get("project_laps_ahead", 5)
    recent_window = args.get("recent_window", 8)

    target_stint = race.current_stint_laps(target, current_lap)
    rival_stint = (
        race.current_stint_laps(rival, current_lap) if rival else None
    )

    result = compare_degradation(
        target_stint=target_stint,
        rival_stint=rival_stint,
        target_code=target,
        rival_code=rival,
        project_laps_ahead=laps_ahead,
        recent_window=recent_window,
    )
    return json.dumps(result.to_dict())


# ---- Registry -------------------------------------------------------------

ALL_TOOL_SPECS: list[ToolSpec] = [
    COMPARE_PACE_SPEC,
    SIMULATE_PIT_WINDOW_SPEC,
    GET_TYRE_DEGRADATION_SPEC,
    STAY_SILENT_SPEC,
]

TOOL_HANDLERS: dict[str, Callable[[RaceData, int, dict[str, Any]], str]] = {
    "compare_pace": _handle_compare_pace,
    "simulate_pit_window": _handle_simulate_pit_window,
    "get_tyre_degradation": _handle_get_tyre_degradation,
    "stay_silent": handle_stay_silent,
}


def dispatch_tool(
    name: str,
    race: RaceData,
    current_lap: int,
    args: dict[str, Any],
) -> str:
    """Execute a tool by name. Returns JSON string for the agent."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return handler(race, current_lap, args)
    except Exception as e:
        return json.dumps({"error": f"Tool {name} failed: {type(e).__name__}: {e}"})
