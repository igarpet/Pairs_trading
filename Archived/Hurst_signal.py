"""
Hurst_signal.py

LIVE / OUT-OF-SAMPLE Hurst feature module.

Contains only backward-looking quantities allowed in the final strategy.

Working Hurst confirmation:
    1. Smoothed Hurst < 0.5
    2. Local Hurst slope crosses from >= 0 to < 0

Acceleration is deliberately excluded from live trading logic.
"""
from __future__ import annotations
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


def local_hurst_slope(smoothed_hurst: pd.Series, fit_window: int = 10) -> pd.Series:
    h = pd.Series(smoothed_hurst).astype(float)
    if fit_window < 5:
        raise ValueError("fit_window should be at least 5.")
    slopes = np.full(len(h), np.nan)
    x = np.arange(fit_window, dtype=float)
    x_last = x[-1]
    for i in range(fit_window - 1, len(h)):
        y = h.iloc[i - fit_window + 1:i + 1].to_numpy(dtype=float)
        if not np.all(np.isfinite(y)):
            continue
        c2, c1, _ = np.polyfit(x, y, deg=2)
        slopes[i] = c1 + 2.0 * c2 * x_last
    return pd.Series(slopes, index=h.index, name="hurst_slope")


def hurst_slope_crosses_negative(hurst_slope: pd.Series) -> pd.Series:
    slope = pd.Series(hurst_slope).astype(float)
    previous = slope.shift(1)
    signal = (previous >= 0) & (slope < 0)
    return signal.fillna(False).rename("hurst_slope_negative_cross")


def hurst_entry_confirmation(hurst_smooth: pd.Series, hurst_slope: pd.Series,
                             max_hurst: float = 0.5) -> pd.Series:
    cross = hurst_slope_crosses_negative(hurst_slope)
    confirmation = (pd.Series(hurst_smooth) < max_hurst) & cross
    return confirmation.fillna(False).rename("hurst_entry_confirmation")


def build_live_hurst_features(spread: pd.Series, hurst_window: int = 120,
                              hurst_min_lag: int = 2, hurst_max_lag: int = 40,
                              ema_span: int = 10, slope_window: int = 10,
                              max_hurst: float = 0.5) -> pd.DataFrame:
    spread = pd.Series(spread).astype(float).rename("spread")
    h = rolling_hurst(spread, window=hurst_window, min_lag=hurst_min_lag, max_lag=hurst_max_lag)
    hs = smooth_hurst(h, ema_span=ema_span)
    slope = local_hurst_slope(hs, fit_window=slope_window)
    cross = hurst_slope_crosses_negative(slope)
    confirmation = hurst_entry_confirmation(hs, slope, max_hurst=max_hurst)
    return pd.concat([spread, h, hs, slope, cross, confirmation], axis=1)