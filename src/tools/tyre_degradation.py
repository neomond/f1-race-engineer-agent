"""
Tyre degradation fitter.

Fits both a linear (lap_time = a + b*age) and a quadratic
(lap_time = a + b*age + c*age^2) model to a single stint's clean laps,
selects the better fit via adjusted R^2, and projects forward.

WINDOWING — IMPORTANT
We fit only the RECENT laps of a stint (default 8), not the full stint.
Why: lap times in F1 are driven by two competing effects:
  - Tyres degrade (typically +0.02 to +0.10 s/lap on modern compounds)
  - Fuel burns off (typically -0.05 to -0.08 s/lap as car gets lighter)
Across a long stint, fuel burn often dominates, so a full-stint regression
of lap_time vs tyre_age produces a NEGATIVE slope and misleads downstream
decisions. By restricting the fit to a short recent window, fuel burn is
roughly constant across the window and the recovered slope reflects mostly
tyre degradation. This is a pragmatic mitigation; a production-grade fitter
would include an explicit fuel-correction term.

Design principles:
- Pure functions. Takes a stint DataFrame, returns a typed result.
- Defensive on small samples: refuses to fit below MIN_LAPS_FOR_FIT.
- Confidence is exposed via n_laps_in_fit and r_squared so the agent
  knows how much to trust the projection.
- Linear vs quadratic comparison uses adjusted R^2, which penalises
  extra parameters — prevents quadratic overfitting on short stints.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# Need enough laps to fit meaningfully. Linear needs 2 to fit but is unreliable
# below ~4; quadratic needs 3 to fit but is unreliable below ~6.
MIN_LAPS_FOR_LINEAR = 4
MIN_LAPS_FOR_QUADRATIC = 6

# Default window: how many recent laps to use for the fit. Short enough that
# fuel burn is roughly constant across the window (so it doesn't dominate the
# slope), long enough to give degradation signal room to show.
DEFAULT_RECENT_WINDOW = 8

# Cap on plausible degradation rates (seconds per lap).
# Anything outside ±2.0 s/lap is almost certainly fit noise on a tiny sample.
MAX_PLAUSIBLE_DEG = 2.0

ModelChoice = Literal["linear", "quadratic", "insufficient_data"]


@dataclass
class DegradationFit:
    """A fitted degradation curve for one driver's stint."""
    driver_code: str
    n_laps_in_fit: int
    model: ModelChoice
    fit_window: int           # how many recent laps the fit used
    stint_total_laps: int     # how many clean laps exist in the full stint

    # Linear coefficients (always populated when fit succeeds).
    # lap_time ≈ intercept + slope_per_lap * tyre_age
    intercept_seconds: Optional[float]
    slope_seconds_per_lap: Optional[float]   # degradation rate

    # Quadratic coefficients (populated if we fit a quadratic).
    # lap_time ≈ intercept + linear_term*age + quadratic_term*age^2
    linear_term: Optional[float]
    quadratic_term: Optional[float]

    # Fit quality
    r_squared: Optional[float]              # for chosen model
    adjusted_r_squared: Optional[float]     # for chosen model
    linear_adjusted_r_squared: Optional[float]
    quadratic_adjusted_r_squared: Optional[float]

    # Stint context
    tyre_compound: Optional[str]
    current_tyre_age: Optional[int]
    stint_start_lap: Optional[int]

    # Reason if we couldn't fit
    error: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DegradationProjection:
    """Forecasted lap times based on a fit."""
    laps_ahead: int
    projected_lap_time_seconds: Optional[float]
    projected_total_loss_seconds: Optional[float]  # vs current pace

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DegradationComparison:
    """Comparison of degradation between two drivers."""
    target_fit: DegradationFit
    rival_fit: Optional[DegradationFit]
    delta_seconds_per_lap: Optional[float]   # target_slope - rival_slope
    verdict: str                             # "target_degrading_faster" | "matched" | "rival_degrading_faster" | "no_comparison"
    target_projection: Optional[DegradationProjection]
    rival_projection: Optional[DegradationProjection]

    def to_dict(self) -> dict:
        return {
            "target_fit": self.target_fit.to_dict(),
            "rival_fit": self.rival_fit.to_dict() if self.rival_fit else None,
            "delta_seconds_per_lap": self.delta_seconds_per_lap,
            "verdict": self.verdict,
            "target_projection": (
                self.target_projection.to_dict() if self.target_projection else None
            ),
            "rival_projection": (
                self.rival_projection.to_dict() if self.rival_projection else None
            ),
        }


# Threshold for "matched" verdict (seconds/lap delta below this = equivalent)
_MATCHED_THRESHOLD_S = 0.03


# ---- Core fitting ---------------------------------------------------------


def fit_degradation(
    stint_laps: pd.DataFrame,
    driver_code: str,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> DegradationFit:
    """
    Fit linear and quadratic degradation models on the last `recent_window`
    clean laps of the stint, return the better one.

    The recent-window restriction is a fuel-burn mitigation: over 6-10
    laps fuel burn is approximately constant, so the recovered slope
    reflects mostly tyre degradation rather than the combined effect.

    stint_laps must contain columns:
      LapTime (Timedelta), TyreLife (numeric), PitInTime, PitOutTime, Compound,
      LapNumber, Stint (consistent across rows).
    """
    full_clean = _clean_stint(stint_laps)
    stint_total = len(full_clean)

    # Slice to the recent window
    clean = full_clean.tail(recent_window).reset_index(drop=True)
    n = len(clean)

    compound = (
        stint_laps["Compound"].iloc[0] if "Compound" in stint_laps.columns
        and len(stint_laps) and pd.notna(stint_laps["Compound"].iloc[0])
        else None
    )
    current_age = (
        int(stint_laps["TyreLife"].iloc[-1])
        if "TyreLife" in stint_laps.columns and len(stint_laps)
        and pd.notna(stint_laps["TyreLife"].iloc[-1])
        else None
    )
    stint_start = (
        int(stint_laps["LapNumber"].min())
        if "LapNumber" in stint_laps.columns and len(stint_laps) else None
    )

    if n < MIN_LAPS_FOR_LINEAR:
        return DegradationFit(
            driver_code=driver_code,
            n_laps_in_fit=n,
            model="insufficient_data",
            fit_window=recent_window,
            stint_total_laps=stint_total,
            intercept_seconds=None,
            slope_seconds_per_lap=None,
            linear_term=None,
            quadratic_term=None,
            r_squared=None,
            adjusted_r_squared=None,
            linear_adjusted_r_squared=None,
            quadratic_adjusted_r_squared=None,
            tyre_compound=compound,
            current_tyre_age=current_age,
            stint_start_lap=stint_start,
            error=f"Only {n} clean laps in recent window (need {MIN_LAPS_FOR_LINEAR}+)",
        )

    ages = clean["TyreLife"].astype(float).to_numpy()
    times = clean["LapTime_s"].to_numpy()

    # ---- Linear fit (OLS via sklearn) -----------------------------------
    lin_slope, lin_intercept, lin_pred = _fit_ols(ages, times, degree=1)
    lin_r2 = _r_squared(times, lin_pred)
    lin_adj_r2 = _adjusted_r_squared(lin_r2, n, k=1)

    # ---- Quadratic fit (if we have enough data) -------------------------
    quad_coeffs = None
    quad_r2 = None
    quad_adj_r2 = None
    if n >= MIN_LAPS_FOR_QUADRATIC:
        # a*x^2 + b*x + c
        b, a, c, quad_pred = _fit_ols(ages, times, degree=2)
        quad_coeffs = (a, b, c)
        quad_r2 = _r_squared(times, quad_pred)
        quad_adj_r2 = _adjusted_r_squared(quad_r2, n, k=2)

    # ---- Pick the better model -----------------------------------------
    use_quadratic = (
        quad_adj_r2 is not None and quad_adj_r2 > lin_adj_r2 + 0.02
        # Small margin requirement so we don't pick quadratic for tiny gains
    )

    if use_quadratic:
        a, b, c = quad_coeffs
        return DegradationFit(
            driver_code=driver_code,
            n_laps_in_fit=n,
            model="quadratic",
            fit_window=recent_window,
            stint_total_laps=stint_total,
            intercept_seconds=round(float(c), 3),
            slope_seconds_per_lap=round(float(b), 4),  # linear term as "current rate"
            linear_term=round(float(b), 4),
            quadratic_term=round(float(a), 5),
            r_squared=round(float(quad_r2), 4),
            adjusted_r_squared=round(float(quad_adj_r2), 4),
            linear_adjusted_r_squared=round(float(lin_adj_r2), 4),
            quadratic_adjusted_r_squared=round(float(quad_adj_r2), 4),
            tyre_compound=compound,
            current_tyre_age=current_age,
            stint_start_lap=stint_start,
            error=None,
        )

    # Sanity-bound the slope for the linear model
    slope_capped = _cap_implausible(lin_slope)
    return DegradationFit(
        driver_code=driver_code,
        n_laps_in_fit=n,
        model="linear",
        fit_window=recent_window,
        stint_total_laps=stint_total,
        intercept_seconds=round(float(lin_intercept), 3),
        slope_seconds_per_lap=round(float(slope_capped), 4),
        linear_term=None,
        quadratic_term=None,
        r_squared=round(float(lin_r2), 4),
        adjusted_r_squared=round(float(lin_adj_r2), 4),
        linear_adjusted_r_squared=round(float(lin_adj_r2), 4),
        quadratic_adjusted_r_squared=(
            round(float(quad_adj_r2), 4) if quad_adj_r2 is not None else None
        ),
        tyre_compound=compound,
        current_tyre_age=current_age,
        stint_start_lap=stint_start,
        error=None,
    )


def project_forward(fit: DegradationFit, laps_ahead: int) -> DegradationProjection:
    """Project lap time `laps_ahead` laps into the future from current age."""
    if fit.model == "insufficient_data" or fit.current_tyre_age is None:
        return DegradationProjection(
            laps_ahead=laps_ahead,
            projected_lap_time_seconds=None,
            projected_total_loss_seconds=None,
        )

    future_age = fit.current_tyre_age + laps_ahead

    if fit.model == "linear":
        current = fit.intercept_seconds + fit.slope_seconds_per_lap * fit.current_tyre_age
        projected = fit.intercept_seconds + fit.slope_seconds_per_lap * future_age
    else:  # quadratic
        current = (
            fit.intercept_seconds
            + fit.linear_term * fit.current_tyre_age
            + fit.quadratic_term * fit.current_tyre_age**2
        )
        projected = (
            fit.intercept_seconds
            + fit.linear_term * future_age
            + fit.quadratic_term * future_age**2
        )

    return DegradationProjection(
        laps_ahead=laps_ahead,
        projected_lap_time_seconds=round(float(projected), 3),
        projected_total_loss_seconds=round(float(projected - current), 3),
    )


def compare_degradation(
    target_stint: pd.DataFrame,
    rival_stint: Optional[pd.DataFrame],
    target_code: str,
    rival_code: Optional[str],
    project_laps_ahead: int = 5,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> DegradationComparison:
    """Fit degradation for target and optionally rival, return comparison."""
    target_fit = fit_degradation(target_stint, target_code, recent_window)
    target_proj = (
        project_forward(target_fit, project_laps_ahead)
        if target_fit.model != "insufficient_data" else None
    )

    if rival_stint is None or rival_code is None:
        return DegradationComparison(
            target_fit=target_fit,
            rival_fit=None,
            delta_seconds_per_lap=None,
            verdict="no_comparison",
            target_projection=target_proj,
            rival_projection=None,
        )

    rival_fit = fit_degradation(rival_stint, rival_code, recent_window)
    rival_proj = (
        project_forward(rival_fit, project_laps_ahead)
        if rival_fit.model != "insufficient_data" else None
    )

    if (
        target_fit.slope_seconds_per_lap is None
        or rival_fit.slope_seconds_per_lap is None
    ):
        verdict = "no_comparison"
        delta = None
    else:
        delta = target_fit.slope_seconds_per_lap - rival_fit.slope_seconds_per_lap
        if abs(delta) < _MATCHED_THRESHOLD_S:
            verdict = "matched"
        elif delta > 0:
            verdict = "target_degrading_faster"
        else:
            verdict = "rival_degrading_faster"

    return DegradationComparison(
        target_fit=target_fit,
        rival_fit=rival_fit,
        delta_seconds_per_lap=round(delta, 4) if delta is not None else None,
        verdict=verdict,
        target_projection=target_proj,
        rival_projection=rival_proj,
    )


# ---- Helpers --------------------------------------------------------------


def _fit_ols(x: np.ndarray, y: np.ndarray, degree: int):
    """
    Ordinary least squares via sklearn.

    degree=1 → (slope, intercept, predictions)
    degree=2 → (linear_term, quadratic_term, intercept, predictions)
               matching lap_time ≈ c + b*age + a*age^2
    """
    X = x.reshape(-1, 1)
    if degree == 1:
        model = LinearRegression().fit(X, y)
        return float(model.coef_[0]), float(model.intercept_), model.predict(X)

    pipe = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )
    pipe.fit(X, y)
    lin: LinearRegression = pipe.named_steps["linearregression"]
    linear_term, quadratic_term = (float(c) for c in lin.coef_)
    intercept = float(lin.intercept_)
    return linear_term, quadratic_term, intercept, pipe.predict(X)



def _clean_stint(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pit laps and rows missing essential data, return tidy DataFrame."""
    if not len(df):
        return pd.DataFrame(columns=["LapTime_s", "TyreLife"])

    cols = df.copy()

    # Filter pit-in/out laps
    if "PitInTime" in cols.columns:
        cols = cols[cols["PitInTime"].isna()]
    if "PitOutTime" in cols.columns:
        cols = cols[cols["PitOutTime"].isna()]

    # Filter rows with no lap time or no tyre age
    cols = cols[cols["LapTime"].notna()]
    if "TyreLife" in cols.columns:
        cols = cols[cols["TyreLife"].notna()]
    else:
        return pd.DataFrame(columns=["LapTime_s", "TyreLife"])

    if not len(cols):
        return pd.DataFrame(columns=["LapTime_s", "TyreLife"])

    # Convert lap time to seconds
    out = pd.DataFrame({
        "TyreLife": cols["TyreLife"].astype(float).values,
        "LapTime_s": cols["LapTime"].dt.total_seconds().values,
    })
    return out.reset_index(drop=True)


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _adjusted_r_squared(r2: float, n: int, k: int) -> float:
    """k = number of predictors (linear = 1, quadratic = 2)."""
    if n - k - 1 <= 0:
        return r2  # can't compute, return raw
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)


def _cap_implausible(slope: float) -> float:
    """Clamp wildly implausible slopes to MAX_PLAUSIBLE_DEG."""
    if slope > MAX_PLAUSIBLE_DEG:
        return MAX_PLAUSIBLE_DEG
    if slope < -MAX_PLAUSIBLE_DEG:
        return -MAX_PLAUSIBLE_DEG
    return slope
