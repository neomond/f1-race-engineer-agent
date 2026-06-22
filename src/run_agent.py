"""
Run the race engineer agent through a race replay.

Streams radio calls to terminal as they happen, and saves a structured
JSON log to evals/runs/ for later analysis.

Usage:
    python -m src.run_agent
    python -m src.run_agent --year 2024 --race Azerbaijan --driver NOR --laps 1-20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Make sure src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.gemini_client import GeminiClient
from src.agent.ollama_client import OllamaClient
from src.agent.race_engineer import RaceEngineerAgent
from src.data.race_data import RaceData
from src.data.replayer import RaceReplayer
from src.memory.agent_memory import RadioCall


def parse_lap_range(s: str, total: int) -> tuple[int, int]:
    """Parse '1-20' or '15' or 'all' → (start, end)."""
    s = s.strip().lower()
    if s in ("all", ""):
        return 1, total
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


def main():
    parser = argparse.ArgumentParser(description="Race engineer agent runner")
    parser.add_argument("--year", type=int, default=int(os.getenv("RACE_YEAR", 2024)))
    parser.add_argument("--race", default=os.getenv("RACE_NAME", "Azerbaijan"))
    parser.add_argument("--driver", default=os.getenv("TARGET_DRIVER", "NOR"))
    parser.add_argument("--laps", default="all", help="Range, e.g. '1-20' or 'all'")
    parser.add_argument(
        "--provider", choices=["ollama", "gemini"], default="ollama",
        help="LLM provider: ollama (local, default) or gemini (cloud)."
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model name. Defaults: qwen3:8b (ollama), "
             "gemini-2.5-flash (gemini)."
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help=(
            "Seconds between laps. Default: 0 for ollama, 15 for gemini "
            "(free tier = 5 req/min for gemini-2.5-flash). "
            "Use --model gemini-2.5-flash-lite with shorter delays "
            "for faster development cycles."
        )
    )
    parser.add_argument(
        "--log-dir", default="evals/runs",
        help="Where to save the structured run log"
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # ---- Load race + agent -----------------------------------------------
    print(f"Loading {args.year} {args.race} GP...")
    race = RaceData(args.year, args.race)
    start_lap, end_lap = parse_lap_range(args.laps, race.meta.total_laps)

    print(f"\n=== {args.year} {args.race} GP — Engineer for {args.driver} ===")
    print(f"Circuit: {race.meta.circuit}")
    print(f"Laps: {start_lap}–{end_lap} of {race.meta.total_laps}")
    print(f"Pit loss: {race.meta.pit_loss_seconds}s ({race.pit_loss_estimate.source})")
    print("=" * 70)
    print()

    # ---- Build LLM client based on provider -----------------------------
    if args.provider == "ollama":
        llm = OllamaClient(model=args.model or os.getenv("OLLAMA_MODEL", "qwen3:8b"))
        default_delay = 0.0
        provider_label = f"Ollama ({llm.model}, local)"
    else:
        llm = GeminiClient(
            model=args.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
        # Gemini 2.5 Flash free tier: 5 requests/min. The agent makes 1-3
        # API calls per lap (1 for silent, 2+ for tool-using laps), so
        # 15s minimum between laps keeps us under the limit.
        default_delay = 15.0
        provider_label = f"Gemini ({llm.model})"

    effective_delay = args.delay if args.delay is not None else default_delay
    print(f"Provider: {provider_label}")
    print(f"Inter-lap delay: {effective_delay}s")
    print("=" * 70)
    print()

    agent = RaceEngineerAgent(llm=llm, race=race)

    # ---- Run the race ----------------------------------------------------
    replayer = RaceReplayer(race, args.driver, start_lap, end_lap)
    radio_log: list[RadioCall] = []

    for ctx in replayer:
        try:
            call = agent.step(ctx)
        except Exception as e:
            print(f"L{ctx.lap_number}  ❌ ERROR: {type(e).__name__}: {e}")
            continue

        _print_call(call)
        radio_log.append(call)

        if effective_delay > 0:
            time.sleep(effective_delay)

    # ---- Persist log -----------------------------------------------------
    log_path = _save_log(args, race, radio_log, agent, start_lap, end_lap, provider_label)
    print()
    print("=" * 70)
    print(f"Race complete. {sum(1 for c in radio_log if c.message)} radio calls, "
          f"{sum(1 for c in radio_log if not c.message)} silent laps.")
    print(f"Log saved: {log_path}")


def _print_call(call: RadioCall):
    if call.message:
        tools = f" ({','.join(call.tools_used)})" if call.tools_used else ""
        print(f"L{call.lap_number:2d}  📻 {call.message}{tools}")
    else:
        print(f"L{call.lap_number:2d}  · silent: {call.silent_reason}")


def _save_log(args, race, radio_log, agent, start_lap, end_lap, provider_label) -> Path:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = (
        f"{args.year}_{args.race.lower().replace(' ', '')}_"
        f"{args.driver}_L{start_lap}-{end_lap}_{timestamp}.json"
    )
    path = log_dir / fname

    payload = {
        "meta": {
            "year": args.year,
            "race": args.race,
            "circuit": race.meta.circuit,
            "driver": args.driver,
            "lap_range": [start_lap, end_lap],
            "total_laps": race.meta.total_laps,
            "pit_loss_seconds": race.meta.pit_loss_seconds,
            "pit_loss_source": race.pit_loss_estimate.source,
            "model": provider_label,
            "timestamp": timestamp,
        },
        "calls": [c.to_dict() for c in radio_log],
        "events": [e.to_dict() for e in agent.memory.events],
        "summary": {
            "total_laps_processed": len(radio_log),
            "radio_calls": sum(1 for c in radio_log if c.message),
            "silent_laps": sum(1 for c in radio_log if not c.message),
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


if __name__ == "__main__":
    main()
