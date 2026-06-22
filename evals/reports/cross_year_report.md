# Cross-year race engineer agent — Baku editions

Cross-year validation of the race engineer agent, run on multiple Azerbaijan GPs with the same code, tools, and prompt. The agent has no foreknowledge of subsequent laps — it processes each lap as it arrives, identical to how it would run on live timing data.

## Summary across editions

| Year | Circuit | Laps | Pit loss | Source | Calls | Silent | Call rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | Azerbaijan Grand Prix | 10–35 | 21.33s | measured | 6 | 20 | 23% |
| 2024 | Azerbaijan Grand Prix | 10–35 | 20.56s | measured | 6 | 20 | 23% |
| 2025 | Azerbaijan Grand Prix | 10–35 | 19.53s | measured | 5 | 21 | 19% |

## Tool usage by year

| Year | simulate_pit_window | total |
| --- | --- | --- |
| 2023 | 0 | 0 |
| 2024 | 1 | 1 |
| 2025 | 0 | 0 |

## Measured pit loss drift across editions

Pit loss values measured directly from each race's pit stops (see `src/tools/pit_loss.py`):

- **2023**: 21.33s
- **2024**: 20.56s
- **2025**: 19.53s

**Spread: 1.80s.** A small spread (~1-2s) suggests the measurement is robust to year-on-year variation in pit lane traffic and stop quality; a larger spread suggests external factors (SC stops included in the sample, regulation changes, etc.).

## Strategic moments (laps with simulate_pit_window calls)

### 2024
- **L29**: Overcut favoured. Staying out projects +0.6s gap to VER.


## Representative radio calls

### 2023
- **L10**: "Box this lap, box this lap. Undercut window on MAG."
- **L16**: "Box this lap, box this lap. Undercut window on HUL."
- **L22**: "Tyre degradation increasing. Box next lap if gap to HUL closes below 1.0."
- **L27**: "Box this lap. Undercut window on HUL, gap closing."
- **L33**: "Gap to HUL stable at 0.59. Tyres holding, no immediate pit needed."
- **L34**: "Gap to HUL widening to 1.0. Tyres holding, no immediate pit needed."

### 2024
- **L10**: "Box this lap, box this lap. Undercut window on Leclerc."
- **L16**: "Box this lap, box this lap. Undercut window on PER."
- **L22**: "Box this lap. Undercut window on ALB, gap closing."
- **L28**: "Box this lap, box this lap. Undercut window on Verstappen."
- **L29**: "Overcut favoured. Staying out projects +0.6s gap to VER." *(tools: simulate_pit_window)*
- **L35**: "Box this lap, box this lap. Undercut window on RUS."

### 2025
- **L10**: "Box this lap, box this lap. Undercut window on Leclerc."
- **L16**: "Box this lap, box this lap. Undercut window on Leclerc."
- **L22**: "Box this lap, box this lap. Undercut window on TSU."
- **L28**: "Box this lap. Undercut window on TSU, gap closing."
- **L34**: "Box this lap. Undercut window on TSU, gap closing."


---

_Report generated from 3 run log(s) in `evals/cross_year`._