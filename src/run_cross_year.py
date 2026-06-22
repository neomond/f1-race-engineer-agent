"""
Run the race engineer agent across multiple Baku editions for cross-year
validation.

Designed to be deployment-friendly: each year's run is independent, logs
land in evals/runs/ with consistent naming, and the script is rerunnable
(skips years already logged unless --force is passed).

Usage:
    python -m src.run_cross_year                       # all defaults
    python -m src.run_cross_year --years 2023 2024 2025
    python -m src.run_cross_year --laps 20-30          # narrower window
    python -m src.run_cross_year --force               # re-run even if logged
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.race_data import RaceData
from src.data.replayer import RaceReplayer
from src.memory.agent_memory import RadioCall
from src.agent.race_engineer import RaceEngineerAgent


# Edit-friendly defaults — what races to run if user passes nothing
DEFAULT_YEARS = [2023, 2024, 2025]
DEFAULT_RACE = "Azerbaijan"
DEFAULT_DRIVER = "NOR"
DEFAULT_LAPS = "10-35"

# Cross-year runs go in a dedicated subfolder so they're easy to batch-analyse
LOG_SUBDIR = "cross_year"


def parse_lap_range(s: str, total: int) -> tuple[int, int]:
    s = s.strip().lower()
    if s in ("all", ""):
        return 1, total
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


def build_llm(provider: str, model: str | None):
    """Defer imports so we don't require both SDKs for either provider."""
    if provider == "ollama":
        from src.agent.ollama_client import OllamaClient
        llm = OllamaClient(model=model or os.getenv("OLLAMA_MODEL", "qwen3:8b"))
        return llm, f"Ollama ({llm.model}, local)", 0.0
    else:
        from src.agent.gemini_client import GeminiClient
        llm = GeminiClient(
            model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
        return llm, f"Gemini ({llm.model})", 15.0


def log_path_for(year: int, race: str, driver: str, lap_range: str,
                 log_dir: Path) -> Path:
    """Stable filename so we can detect previously-run years."""
    fname = (
        f"{year}_{race.lower().replace(' ', '')}_"
        f"{driver}_L{lap_range.replace('-', '_')}.json"
    )
    return log_dir / fname


def run_one_year(
    year: int,
    race_name: str,
    driver: str,
    lap_spec: str,
    llm,
    provider_label: str,
    delay: float,
    log_dir: Path,
    force: bool,
) -> dict | None:
    """Run a single race, save log, return summary dict."""
    laps_canonical = lap_spec  # used in filename for stability
    out_path = log_path_for(year, race_name, driver, laps_canonical, log_dir)

    if out_path.exists() and not force:
        print(f"  ↪ {year}: skipping (log exists at {out_path.name}). "
              f"Pass --force to re-run.")
        return json.loads(out_path.read_text())

    print(f"\n  → Loading {year} {race_name} GP...")
    try:
        race = RaceData(year, race_name)
    except Exception as e:
        print(f"  ❌ {year}: failed to load race data: {type(e).__name__}: {e}")
        return None

    start_lap, end_lap = parse_lap_range(lap_spec, race.meta.total_laps)
    print(f"     Circuit: {race.meta.circuit}")
    print(f"     Laps {start_lap}–{end_lap}/{race.meta.total_laps}, "
          f"pit loss: {race.meta.pit_loss_seconds}s "
          f"({race.pit_loss_estimate.source})")

    agent = RaceEngineerAgent(llm=llm, race=race)
    replayer = RaceReplayer(race, driver, start_lap, end_lap)
    radio_log: list[RadioCall] = []

    for ctx in replayer:
        try:
            call = agent.step(ctx)
        except Exception as e:
            print(f"     L{ctx.lap_number}  ❌ {type(e).__name__}: {e}")
            continue

        if call.message:
            print(f"     L{call.lap_number:2d}  📻 {call.message}")
        else:
            print(f"     L{call.lap_number:2d}  · silent")
        radio_log.append(call)

        if delay > 0:
            time.sleep(delay)

    # Persist with stable filename for cross-year detection
    payload = {
        "meta": {
            "year": year,
            "race": race_name,
            "circuit": race.meta.circuit,
            "driver": driver,
            "lap_range": [start_lap, end_lap],
            "total_laps": race.meta.total_laps,
            "pit_loss_seconds": race.meta.pit_loss_seconds,
            "pit_loss_source": race.pit_loss_estimate.source,
            "model": provider_label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "calls": [c.to_dict() for c in radio_log],
        "events": [e.to_dict() for e in agent.memory.events],
        "summary": {
            "total_laps_processed": len(radio_log),
            "radio_calls": sum(1 for c in radio_log if c.message),
            "silent_laps": sum(1 for c in radio_log if not c.message),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"     ✓ saved {out_path.name}")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Cross-year race engineer batch runner"
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--race", default=DEFAULT_RACE)
    parser.add_argument("--driver", default=DEFAULT_DRIVER)
    parser.add_argument("--laps", default=DEFAULT_LAPS,
                        help="Same lap range applied to every year.")
    parser.add_argument("--provider", choices=["ollama", "gemini"],
                        default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--log-dir", default=f"evals/{LOG_SUBDIR}")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a log already exists.")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    llm, provider_label, default_delay = build_llm(args.provider, args.model)
    delay = args.delay if args.delay is not None else default_delay

    print(f"=== Cross-year run: {args.race} GP for {args.driver} ===")
    print(f"Years: {args.years}")
    print(f"Laps:  {args.laps}")
    print(f"Provider: {provider_label}")
    print(f"Inter-lap delay: {delay}s")
    print(f"Log dir: {log_dir}")
    print("=" * 70)

    runs = []
    for year in args.years:
        result = run_one_year(
            year=year,
            race_name=args.race,
            driver=args.driver,
            lap_spec=args.laps,
            llm=llm,
            provider_label=provider_label,
            delay=delay,
            log_dir=log_dir,
            force=args.force,
        )
        if result is not None:
            runs.append(result)

    # ---- End-of-batch summary -------------------------------------------
    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    for r in runs:
        s = r["summary"]
        m = r["meta"]
        print(f"  {m['year']}: {s['radio_calls']:>2} calls, "
              f"{s['silent_laps']:>2} silent over "
              f"{s['total_laps_processed']} laps  "
              f"(pit loss: {m['pit_loss_seconds']}s {m['pit_loss_source']})")
    print()
    print(f"Logs in: {log_dir}")
    print(f"Next step: python src/compare_years.py --log-dir {log_dir}")


if __name__ == "__main__":
    main()
