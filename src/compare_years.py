"""
Cross-year comparison report.

Reads all JSON race logs in a folder and produces a markdown report
comparing the agent's behaviour across years. Designed to be runnable
unattended — drop in next year's log and re-run, get a fresh report.

Usage:
    python -m src.compare_years
    python -m src.compare_years --log-dir evals/cross_year
    python -m src.compare_years --output evals/reports/baku_cross_year.md

Sections produced:
  1. Race-by-race summary table (circuit, lap range, pit loss, call rate)
  2. Tool usage frequency per year
  3. Strategic moment detection (laps where simulate_pit_window was called)
  4. Sample radio calls per year for qualitative comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_logs(log_dir: Path) -> list[dict]:
    """Load all JSON logs in a directory, sorted by year."""
    logs = []
    for path in sorted(log_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["_path"] = path
            logs.append(data)
        except (json.JSONDecodeError, KeyError):
            print(f"  ! skipping malformed log: {path.name}", file=sys.stderr)
    logs.sort(key=lambda d: d.get("meta", {}).get("year", 0))
    return logs


# ---- Report sections ------------------------------------------------------


def summary_table(logs: list[dict]) -> str:
    lines = [
        "## Summary across editions",
        "",
        "| Year | Circuit | Laps | Pit loss | Source | Calls | Silent | Call rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for log in logs:
        m = log["meta"]
        s = log["summary"]
        total = max(s["total_laps_processed"], 1)
        rate = s["radio_calls"] / total * 100
        lap_range = f"{m['lap_range'][0]}–{m['lap_range'][1]}"
        lines.append(
            f"| {m['year']} | {m['circuit']} | {lap_range} | "
            f"{m['pit_loss_seconds']}s | {m['pit_loss_source']} | "
            f"{s['radio_calls']} | {s['silent_laps']} | {rate:.0f}% |"
        )
    return "\n".join(lines)


def tool_usage_section(logs: list[dict]) -> str:
    """
    Dynamic tool-usage table — counts whatever tools appear in the logs.
    Works regardless of how many tools the agent has, including future ones.
    """
    # Discover every tool name across all logs
    all_tools: set[str] = set()
    per_year: dict[int, Counter[str]] = {}
    for log in logs:
        counter: Counter[str] = Counter()
        for call in log["calls"]:
            counter.update(call.get("tools_used", []))
        per_year[log["meta"]["year"]] = counter
        all_tools.update(counter.keys())

    lines = ["## Tool usage by year", ""]

    if not all_tools:
        lines.append(
            "_No tool calls recorded in any logged year. The agent produced "
            "radio calls directly from lap context without invoking tools. "
            "This is a known behaviour of smaller local LLMs under structured "
            "prompts — they may skip tool use when they think they can answer "
            "from the context alone. Cloud models (e.g. Gemini 2.5 Flash) "
            "reliably invoke tools on the same prompt._"
        )
        return "\n".join(lines)

    # Sort tools alphabetically for stable output
    tools_sorted = sorted(all_tools)
    header = "| Year | " + " | ".join(tools_sorted) + " | total |"
    separator = "| --- | " + " | ".join("---" for _ in tools_sorted) + " | --- |"
    lines.append(header)
    lines.append(separator)

    for log in logs:
        year = log["meta"]["year"]
        counter = per_year[year]
        row_cells = [str(counter.get(t, 0)) for t in tools_sorted]
        total = sum(counter.values())
        lines.append(f"| {year} | " + " | ".join(row_cells) + f" | {total} |")

    return "\n".join(lines)


def strategic_moments_section(logs: list[dict]) -> str:
    """Laps where the agent invoked simulate_pit_window — strategic windows."""
    lines = ["## Strategic moments (laps with simulate_pit_window calls)", ""]
    any_found = False
    for log in logs:
        year = log["meta"]["year"]
        moments = [
            (call["lap_number"], call.get("message") or
             f"(silent: {call.get('silent_reason')})")
            for call in log["calls"]
            if "simulate_pit_window" in call.get("tools_used", [])
        ]
        if moments:
            any_found = True
            lines.append(f"### {year}")
            for lap, msg in moments:
                lines.append(f"- **L{lap}**: {msg}")
            lines.append("")
    if not any_found:
        lines.append("_No `simulate_pit_window` calls in any year. "
                     "Strategic decisions may have been outside the lap window, "
                     "or the agent didn't identify an undercut/overcut opportunity._")
    return "\n".join(lines)


def sample_calls_section(logs: list[dict], n_per_year: int = 6) -> str:
    """A handful of representative calls per year for qualitative read."""
    lines = ["## Representative radio calls", ""]
    for log in logs:
        year = log["meta"]["year"]
        calls = [c for c in log["calls"] if c.get("message")]
        if not calls:
            lines.append(f"### {year}\n\n_No radio calls in this window._\n")
            continue
        # Evenly sample across the window
        step = max(len(calls) // n_per_year, 1)
        sampled = calls[::step][:n_per_year]
        lines.append(f"### {year}")
        for c in sampled:
            tools = (
                f" *(tools: {', '.join(c['tools_used'])})*"
                if c.get("tools_used") else ""
            )
            lines.append(f"- **L{c['lap_number']}**: \"{c['message']}\"{tools}")
        lines.append("")
    return "\n".join(lines)


def pit_loss_drift_section(logs: list[dict]) -> str:
    """Compare measured pit loss across years — sanity check on the measurement."""
    lines = ["## Measured pit loss drift across editions", ""]
    if len(logs) < 2:
        lines.append("_Only one race logged — no cross-year comparison._")
        return "\n".join(lines)
    values = [(l["meta"]["year"], l["meta"]["pit_loss_seconds"]) for l in logs]
    pit_min = min(v for _, v in values)
    pit_max = max(v for _, v in values)
    spread = pit_max - pit_min
    lines.append(
        f"Pit loss values measured directly from each race's pit stops "
        f"(see `src/tools/pit_loss.py`):"
    )
    lines.append("")
    for year, val in values:
        lines.append(f"- **{year}**: {val}s")
    lines.append("")
    lines.append(
        f"**Spread: {spread:.2f}s.** A small spread (~1-2s) suggests the "
        f"measurement is robust to year-on-year variation in pit lane traffic "
        f"and stop quality; a larger spread suggests external factors "
        f"(SC stops included in the sample, regulation changes, etc.)."
    )
    return "\n".join(lines)


def build_report(logs: list[dict]) -> str:
    if not logs:
        return "# Cross-year report\n\n_No logs found._\n"

    title = "# Cross-year race engineer agent — Baku editions"
    intro = (
        "Cross-year validation of the race engineer agent, run on multiple "
        "Azerbaijan GPs with the same code, tools, and prompt. The agent "
        "has no foreknowledge of subsequent laps — it processes each lap "
        "as it arrives, identical to how it would run on live timing data."
    )
    parts = [
        title,
        "",
        intro,
        "",
        summary_table(logs),
        "",
        tool_usage_section(logs),
        "",
        pit_loss_drift_section(logs),
        "",
        strategic_moments_section(logs),
        "",
        sample_calls_section(logs),
        "",
        "---",
        "",
        f"_Report generated from {len(logs)} run log(s) in "
        f"`{logs[0]['_path'].parent}`._",
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Cross-year comparison report")
    parser.add_argument("--log-dir", default="evals/cross_year")
    parser.add_argument("--output", default="evals/reports/cross_year_report.md")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    logs = load_logs(log_dir)
    if not logs:
        print(f"No logs in {log_dir}", file=sys.stderr)
        sys.exit(1)

    report = build_report(logs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Report written: {out_path}")
    print(f"Years covered: {[l['meta']['year'] for l in logs]}")


if __name__ == "__main__":
    main()
