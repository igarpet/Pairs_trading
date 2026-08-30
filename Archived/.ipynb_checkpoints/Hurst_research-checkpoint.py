"""
Hurst_research.py

Research-only utilities for studying whether Hurst dynamics help explain
future spread convergence.

IMPORTANT:
- May create forward-looking labels such as convergence within 1HL, 2HL, 3HL.
- Use ONLY on the training/development sample.
- Do not feed research labels into the out-of-sample trading engine.
"""
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 40) -> float:
    x = pd.Series(series).dropna().astype(float)
    if len(x) < max_lag + 10:
        return np.nan
    lags = np.arange(min_lag, max_lag)
    tau = []
    for lag in lags:
        diffs = x.diff(lag).dropna().to_numpy()
        if len(diffs) < 5:
            tau.append(np.nan)
            continue
        scale = np.std(diffs, ddof=1)
        tau.append(np.nan if (not np.isfinite(scale) or scale <= 0) else np.sqrt(scale))
    tau = np.asarray(tau, dtype=float)
    valid = np.isfinite(tau) & (tau > 0)
    if valid.sum() < 5:
        return np.nan
    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(2.0 * slope)


def rolling_hurst(spread: pd.Series, window: int = 120, min_lag: int = 2, max_lag: int = 40) -> pd.Series:
    spread = pd.Series(spread).astype(float)
    if window <= max_lag + 10:
        raise ValueError("window should be materially larger than max_lag.")
    values = np.full(len(spread), np.nan)
    for i in range(window - 1, len(spread)):
        sample = spread.iloc[i - window + 1:i + 1]
        values[i] = hurst_exponent(sample, min_lag=min_lag, max_lag=max_lag)
    return pd.Series(values, index=spread.index, name="hurst")


def smooth_hurst(hurst: pd.Series, ema_span: int = 10) -> pd.Series:
    if ema_span <= 1:
        raise ValueError("ema_span must be greater than 1.")
    return pd.Series(hurst).ewm(span=ema_span, adjust=False, min_periods=ema_span).mean().rename("hurst_smooth")


def local_hurst_dynamics(smoothed_hurst: pd.Series, fit_window: int = 10) -> pd.DataFrame:
    h = pd.Series(smoothed_hurst).astype(float)
    if fit_window < 5:
        raise ValueError("fit_window should be at least 5.")
    slopes = np.full(len(h), np.nan)
    accelerations = np.full(len(h), np.nan)
    x = np.arange(fit_window, dtype=float)
    x_last = x[-1]
    for i in range(fit_window - 1, len(h)):
        y = h.iloc[i - fit_window + 1:i + 1].to_numpy(dtype=float)
        if not np.all(np.isfinite(y)):
            continue
        c2, c1, _ = np.polyfit(x, y, deg=2)
        slopes[i] = c1 + 2.0 * c2 * x_last
        accelerations[i] = 2.0 * c2
    return pd.DataFrame({"hurst_slope": slopes, "hurst_acceleration": accelerations}, index=h.index)


def rolling_zscore(spread: pd.Series, window: int = 120) -> pd.Series:
    spread = pd.Series(spread).astype(float)
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std(ddof=1)
    z = (spread - mean) / std
    return z.where(std > 0).rename("zscore")


def build_research_features(spread: pd.Series, hurst_window: int = 120, hurst_min_lag: int = 2,
                            hurst_max_lag: int = 40, ema_span: int = 10,
                            dynamics_window: int = 10, z_window: int = 120) -> pd.DataFrame:
    spread = pd.Series(spread).astype(float).rename("spread")
    h = rolling_hurst(spread, window=hurst_window, min_lag=hurst_min_lag, max_lag=hurst_max_lag)
    hs = smooth_hurst(h, ema_span=ema_span)
    dyn = local_hurst_dynamics(hs, fit_window=dynamics_window)
    z = rolling_zscore(spread, window=z_window)
    return pd.concat([spread, z, h, hs, dyn], axis=1)


def _crosses_zero(future_z: pd.Series, entry_z: float) -> Tuple[bool, Optional[int]]:
    if not np.isfinite(entry_z) or entry_z == 0:
        return False, None
    for step, z in enumerate(pd.Series(future_z).dropna().to_numpy(dtype=float), start=1):
        if entry_z > 0 and z <= 0:
            return True, step
        if entry_z < 0 and z >= 0:
            return True, step
    return False, None


def maximum_adverse_excursion(future_z: pd.Series, entry_z: float) -> float:
    values = pd.Series(future_z).dropna().to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(entry_z):
        return np.nan
    if entry_z > 0:
        return float(max(0.0, np.max(values) - entry_z))
    if entry_z < 0:
        return float(max(0.0, entry_z - np.min(values)))
    return 0.0


def add_future_convergence_labels(features: pd.DataFrame, half_life: float,
                                  horizons: Iterable[float] = (1.0, 2.0, 3.0)) -> pd.DataFrame:
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive.")
    out = features.copy()
    z = out["zscore"]
    horizon_days = {float(h): max(1, int(np.ceil(float(h) * half_life))) for h in horizons}
    for h in horizon_days:
        label = str(h).rstrip("0").rstrip(".").replace(".", "_")
        out[f"converged_{label}hl"] = False
    max_days = max(horizon_days.values())
    out["days_to_convergence_3hl"] = np.nan
    out["max_adverse_excursion_3hl"] = np.nan
    for i in range(len(out)):
        entry_z = z.iloc[i]
        if not np.isfinite(entry_z) or entry_z == 0:
            continue
        for h, days in horizon_days.items():
            label = str(h).rstrip("0").rstrip(".").replace(".", "_")
            future = z.iloc[i + 1:i + 1 + days]
            converged, _ = _crosses_zero(future, entry_z)
            out.iat[i, out.columns.get_loc(f"converged_{label}hl")] = converged
        future_max = z.iloc[i + 1:i + 1 + max_days]
        converged, days_to_conv = _crosses_zero(future_max, entry_z)
        if converged:
            out.iat[i, out.columns.get_loc("days_to_convergence_3hl")] = days_to_conv
        out.iat[i, out.columns.get_loc("max_adverse_excursion_3hl")] = maximum_adverse_excursion(future_max, entry_z)
    return out


def make_research_event_table(features_with_labels: pd.DataFrame, pair: str, half_life: float,
                              min_abs_z: float = 0.5) -> pd.DataFrame:
    data = features_with_labels.loc[features_with_labels["zscore"].abs() >= min_abs_z].copy()
    data["pair"] = pair
    data["half_life"] = half_life
    data["abs_z"] = data["zscore"].abs()
    data["hurst_falling"] = data["hurst_slope"] < 0
    data["hurst_acceleration_negative"] = data["hurst_acceleration"] < 0
    data["hurst_state"] = np.select(
        [
            (data["hurst_slope"] < 0) & (data["hurst_acceleration"] < 0),
            (data["hurst_slope"] < 0) & (data["hurst_acceleration"] >= 0),
            (data["hurst_slope"] >= 0) & (data["hurst_acceleration"] < 0),
            (data["hurst_slope"] >= 0) & (data["hurst_acceleration"] >= 0),
        ],
        ["falling_accelerating", "falling_decelerating", "rising_decelerating", "rising_accelerating"],
        default="insufficient_data",
    )
    return data


def summarize_research_states(event_data: pd.DataFrame) -> pd.DataFrame:
    return (event_data.dropna(subset=["hurst_slope", "hurst_acceleration"])
            .groupby("hurst_state")
            .agg(observations=("pair", "size"), mean_abs_z=("abs_z", "mean"),
                 convergence_rate_1hl=("converged_1hl", "mean"),
                 convergence_rate_2hl=("converged_2hl", "mean"),
                 convergence_rate_3hl=("converged_3hl", "mean"),
                 median_days_to_convergence=("days_to_convergence_3hl", "median"),
                 mean_max_adverse_excursion=("max_adverse_excursion_3hl", "mean"))
            .reset_index())