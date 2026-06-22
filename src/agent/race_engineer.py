"""
Race engineer agent loop.

One tick per lap:
  1. Receive a LapContext
  2. Build a prompt: system + memory snapshot + current lap state
  3. Run an LLM iteration with tools, allowing multiple tool calls
  4. Resolve final output: a RadioCall (either spoken or silent)
  5. Update memory

Designed to be model-agnostic via LLMClient. The loop itself never imports
Gemini-specific anything.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Optional

from src.agent.llm_client import LLMClient, ToolResult
from src.agent.prompts import SYSTEM_PROMPT
from src.data.lap_context import LapContext
from src.data.race_data import RaceData
from src.memory.agent_memory import AgentMemory, RadioCall
from src.tools.registry import ALL_TOOL_SPECS, dispatch_tool

logger = logging.getLogger(__name__)


# Cap on tool-calling iterations per lap. Real models converge in 1-3.
MAX_TOOL_ITERATIONS = 4


class RaceEngineerAgent:
    """The agent: state + step function."""

    def __init__(
        self,
        llm: LLMClient,
        race: RaceData,
        memory: Optional[AgentMemory] = None,
    ):
        self.llm = llm
        self.race = race
        self.memory = memory or AgentMemory()

    def step(self, ctx: LapContext) -> RadioCall:
        """
        Process one lap. Returns the RadioCall produced (spoken or silent).
        Side effect: updates memory.
        """
        self.memory.record_lap(ctx)

        user_prompt = self._build_user_prompt(ctx)
        messages = [{"role": "user", "content": user_prompt}]
        tools_used: list[str] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
                tools=ALL_TOOL_SPECS,
            )

            # Case 1: model wants to call tools — execute and continue loop
            if response.wants_tools:
                # Append assistant turn with the tool calls
                messages.append({
                    "role": "assistant",
                    "tool_calls": response.tool_calls,
                })

                # Detect: did the model call stay_silent? That ends the lap.
                silent_call = next(
                    (tc for tc in response.tool_calls if tc.name == "stay_silent"),
                    None,
                )
                if silent_call:
                    tools_used.extend(tc.name for tc in response.tool_calls)
                    call = RadioCall(
                        lap_number=ctx.lap_number,
                        message=None,
                        silent_reason=silent_call.arguments.get("reason", ""),
                        tools_used=tools_used,
                    )
                    self.memory.record_call(call)
                    return call

                # Otherwise: execute each tool, feed results back
                for tc in response.tool_calls:
                    tools_used.append(tc.name)
                    result_json = dispatch_tool(
                        tc.name, self.race, ctx.lap_number, tc.arguments
                    )
                    messages.append({
                        "role": "tool",
                        "tool_result": ToolResult(
                            call_id=tc.call_id,
                            name=tc.name,
                            content=result_json,
                        ),
                    })
                continue

            # Case 2: model produced text — that's the radio call
            raw_text = (response.text or "").strip()
            message = self._extract_radio_call(raw_text)

            # Safety net 1: token-similarity repetition check.
            # Prevents "Box, box, box" 16-times-in-a-row from smaller models.
            if message and self._is_repetitive(message):
                logger.info(
                    f"Lap {ctx.lap_number}: model repeated recent call, "
                    f"converting to silent"
                )
                call = RadioCall(
                    lap_number=ctx.lap_number,
                    message=None,
                    silent_reason="repeat of recent call (auto-converted)",
                    tools_used=tools_used,
                )
                self.memory.record_call(call)
                return call

            # Safety net 2: pit-recommendation guardrail.
            # A driver pits ~1-2 times per race. If the agent already
            # recommended a stop within the last N laps and no pit_stop
            # event was recorded, suppress further box calls.
            if message and self._is_premature_pit_call(message, ctx.lap_number):
                logger.info(
                    f"Lap {ctx.lap_number}: premature pit recommendation "
                    f"suppressed (recent box call with no actual stop)"
                )
                call = RadioCall(
                    lap_number=ctx.lap_number,
                    message=None,
                    silent_reason=(
                        "premature box recommendation suppressed — "
                        "no pit_stop event since last 'box' call"
                    ),
                    tools_used=tools_used,
                )
                self.memory.record_call(call)
                return call

            call = RadioCall(
                lap_number=ctx.lap_number,
                message=message if message else None,
                silent_reason=None if message else "empty response",
                tools_used=tools_used,
            )
            self.memory.record_call(call)
            return call

        # Iteration cap hit — treat as silent with diagnostic
        logger.warning(
            f"Lap {ctx.lap_number}: hit tool iteration cap, recording silent"
        )
        call = RadioCall(
            lap_number=ctx.lap_number,
            message=None,
            silent_reason="iteration cap hit",
            tools_used=tools_used,
        )
        self.memory.record_call(call)
        return call

    # ---- Output post-processing ------------------------------------------

    def _is_premature_pit_call(
        self, candidate: str, current_lap: int, cooldown_laps: int = 5
    ) -> bool:
        """
        Detect a pit recommendation made too soon after a previous one.

        A driver pits ~1-2 times per race; recommending two stops within
        `cooldown_laps` without a recorded pit_stop event between them is
        nonsensical. We catch and silence the second.

        Returns True if:
          1. The candidate message is a pit recommendation
          2. A previous pit recommendation exists within `cooldown_laps`
          3. No pit_stop event has been recorded since that earlier call
        """
        if not self._mentions_pit(candidate):
            return False

        # Look for a recent prior "box" recommendation
        prior_box_lap = None
        for prev in self.memory.recent_calls:
            if prev.message and self._mentions_pit(prev.message):
                prior_box_lap = prev.lap_number

        if prior_box_lap is None:
            return False

        # Within cooldown?
        if current_lap - prior_box_lap > cooldown_laps:
            return False

        # Did an actual pit_stop event happen since that earlier call?
        for event in self.memory.events:
            if event.event_type == "pit_stop" and event.lap_number > prior_box_lap:
                return False  # Driver pitted — a new box recommendation is fine

        return True  # Recent box call, no actual stop → premature

    @staticmethod
    def _mentions_pit(message: str) -> bool:
        """Detect pit-recommendation phrases."""
        import re
        text = message.lower()
        # Match 'box this lap', 'box, box, box', 'undercut window opens',
        # 'pit this lap', etc. Avoid false positives on 'box you out',
        # 'undercut window closing', 'overcut favoured'.
        patterns = [
            r"\bbox\s+this\s+lap\b",
            r"\bbox,\s*box\b",
            r"\bpit\s+this\s+lap\b",
            r"\bundercut\s+window\s+(opens?|opening)\b",
        ]
        return any(re.search(p, text) for p in patterns)


    def _is_repetitive(self, candidate: str, window: int = 3,
                       similarity_threshold: float = 0.75) -> bool:
        """
        Return True if the candidate is too similar to one of the last
        `window` non-silent radio calls.

        We compare normalised tokens (lowercase, punctuation stripped) and
        compute Jaccard similarity. Above the threshold = treat as repeat.
        """
        recent_messages = [
            c.message for c in self.memory.recent_calls
            if c.message
        ][-window:]
        if not recent_messages:
            return False

        cand_tokens = self._tokenise(candidate)
        if not cand_tokens:
            return False

        for prev in recent_messages:
            prev_tokens = self._tokenise(prev)
            if not prev_tokens:
                continue
            overlap = len(cand_tokens & prev_tokens)
            union = len(cand_tokens | prev_tokens)
            similarity = overlap / union if union else 0.0
            if similarity >= similarity_threshold:
                return True
        return False

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        """Lowercase + strip punctuation + return content tokens."""
        import re
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        stop = {"the", "a", "an", "is", "are", "on", "in", "to", "of", "and",
                "this", "lap", "now"}
        return {tok for tok in cleaned.split() if tok and tok not in stop}

    @staticmethod
    def _extract_radio_call(text: str) -> str:
        """
        Pull the actual radio call out of potentially-chatty model output.

        Models sometimes emit reasoning followed by the call in quotes,
        or repeat the call multiple times. We:
          1. Find the LAST quoted string (the "final answer" in chatty output)
          2. Strip explanatory preambles ("Here is the call:", etc.)
          3. Cap length at 200 chars to prevent essay-as-radio-call

        If nothing parses cleanly, return the first non-empty line trimmed.
        """
        import re

        if not text:
            return ""

        # Look for double-quoted strings; prefer the last one (final answer)
        quoted = re.findall(r'"([^"\n]{3,200})"', text)
        if quoted:
            return quoted[-1].strip()

        # Fallback: drop common preamble lines, take first sensible line
        lines = [
            ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().lower().startswith((
                "here is", "based on", "the call", "i will", "i would",
                "analysis:", "reasoning:", "thought:", "therefore",
                "the previous", "according to",
            ))
        ]
        if not lines:
            return ""

        # Take the first remaining line, cap at 200 chars
        first = lines[0]
        return first[:200] + "..." if len(first) > 200 else first

    # ---- Prompt construction ---------------------------------------------

    def _build_user_prompt(self, ctx: LapContext) -> str:
        memory_block = self.memory.render_for_prompt()
        current_state = self._render_current_state(ctx)
        return (
            f"{memory_block}\n\n"
            f"CURRENT LAP:\n{current_state}\n\n"
            f"Decide: radio call or stay silent. Use tools if needed first."
        )

    @staticmethod
    def _render_current_state(ctx: LapContext) -> str:
        lines = [
            f"Lap {ctx.lap_number}/{ctx.total_laps} "
            f"({ctx.laps_remaining} remaining)",
            f"Driver: {ctx.driver_code} ({ctx.driver_name})",
            f"Position: P{ctx.position}" if ctx.position else "Position: unknown",
            f"Tyre: {ctx.tyre_compound} (age {ctx.tyre_age_laps} laps, "
            f"stint {ctx.stint_number})",
        ]
        if ctx.lap_time_seconds:
            lines.append(f"Lap time: {ctx.lap_time_seconds:.3f}s")
        if ctx.last_3_lap_times:
            recent = ", ".join(
                f"{t:.2f}" for t in ctx.last_3_lap_times if t is not None
            )
            if recent:
                lines.append(f"Last 3 laps: [{recent}]")
        if ctx.ahead and ctx.ahead.gap_seconds is not None:
            lines.append(
                f"Ahead: {ctx.ahead.driver_code} {ctx.ahead.gap_seconds:+.2f}s"
            )
        if ctx.behind and ctx.behind.gap_seconds is not None:
            lines.append(
                f"Behind: {ctx.behind.driver_code} {ctx.behind.gap_seconds:+.2f}s"
            )
        if ctx.track_temp_c is not None and ctx.air_temp_c is not None:
            lines.append(
                f"Track: {ctx.track_temp_c:.1f}°C, Air: {ctx.air_temp_c:.1f}°C, "
                f"Rain: {ctx.rainfall}"
            )
        if ctx.track_status:
            lines.append(f"Track status: {ctx.track_status}")
        if ctx.in_pit:
            lines.append("⚠️  IN PIT this lap")
        return "\n".join(lines)
