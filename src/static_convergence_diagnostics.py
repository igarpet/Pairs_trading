"""Static OOS convergence-calibration diagnostics for Module 11.

This module evaluates whether the fOU model-implied probability recorded at
trade entry was empirically calibrated in the original static backtest.
It does not modify the strategy or generate new trading signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.tz_localize(None).dt.normalize()


def trading_days_between(index: pd.DatetimeIndex, start, end) -> int:
    """Number of observed OOS trading steps after start through end, inclusive of end."""
    idx = pd.DatetimeIndex(index).tz_localize(None).normalize().sort_values().unique()
    start = pd.Timestamp(start).tz_localize(None).normalize()
    end = pd.Timestamp(end).tz_localize(None).normalize()
    if end < start:
        return 0
    return int(((idx > start) & (idx <= end)).sum())


def build_trade_calibration_table(trades: pd.DataFrame, test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create one-row-per-trade convergence calibration diagnostics.

    Primary event definition:
      realized_within_selected_horizon = 1 only when the recorded convergence
      exit occurs within the model-selected trading horizon.

    Trades force-closed at end of test are treated as censored unless the full
    selected horizon was observable before the test ended. Expiry trades are
    fully observed failures because the option lived beyond the selected fOU
    horizon by construction.
    """
    required = {
        "pair", "entry_date", "exit_date", "exit_reason",
        "convergence_horizon_trading_days", "probability_at_selected_horizon",
        "probability_at_max_horizon", "pnl", "trade_return",
    }
    missing = sorted(required - set(trades.columns))
    if missing:
        raise KeyError(f"Trades are missing required columns: {missing}")

    t = trades.copy().reset_index(drop=True)
    t["entry_date"] = _normalize_dates(t["entry_date"])
    t["exit_date"] = _normalize_dates(t["exit_date"])
    idx = pd.DatetimeIndex(test_index).tz_localize(None).normalize().sort_values().unique()
    final_date = pd.Timestamp(idx.max())

    t["observed_trading_days_to_exit"] = [
        trading_days_between(idx, a, b) for a, b in zip(t["entry_date"], t["exit_date"])
    ]
    t["available_trading_days_after_entry"] = [
        trading_days_between(idx, a, final_date) for a in t["entry_date"]
    ]

    t["realized_convergence"] = (t["exit_reason"].astype(str) == "convergence").astype(int)
    t["realized_within_selected_horizon"] = (
        (t["realized_convergence"] == 1)
        & (t["observed_trading_days_to_exit"] <= t["convergence_horizon_trading_days"].astype(int))
    ).astype(int)

    t["selected_horizon_assessable"] = (
        (t["exit_reason"].astype(str) != "end_of_test")
        | (t["available_trading_days_after_entry"] >= t["convergence_horizon_trading_days"].astype(int))
    )
    t["censored_end_of_test"] = ~t["selected_horizon_assessable"]

    p = t["probability_at_selected_horizon"].astype(float).clip(0.0, 1.0)
    y = t["realized_within_selected_horizon"].astype(float)
    t["calibration_error"] = y - p
    t["brier_component"] = (y - p) ** 2
    return t


def calibration_summary(calibration: pd.DataFrame) -> pd.Series:
    """Headline calibration statistics on assessable trades."""
    d = calibration.loc[calibration["selected_horizon_assessable"]].copy()
    if d.empty:
        raise ValueError("No assessable trades for calibration.")

    p = d["probability_at_selected_horizon"].astype(float)
    y = d["realized_within_selected_horizon"].astype(float)
    conv = d["realized_convergence"].astype(float)
    expiry = (d["exit_reason"].astype(str) == "expiry").astype(float)

    return pd.Series({
        "n_total_trades": int(len(calibration)),
        "n_assessable_selected_horizon": int(len(d)),
        "n_censored_end_of_test": int((~calibration["selected_horizon_assessable"]).sum()),
        "mean_model_probability_selected_horizon": float(p.mean()),
        "observed_convergence_rate": float(conv.mean()),
        "observed_convergence_within_selected_horizon": float(y.mean()),
        "calibration_gap_observed_minus_model": float(y.mean() - p.mean()),
        "mean_model_probability_max_horizon": float(d["probability_at_max_horizon"].astype(float).mean()),
        "expiry_rate": float(expiry.mean()),
        "brier_score_selected_horizon": float(np.mean((y - p) ** 2)),
        "mean_trade_return": float(d["trade_return"].astype(float).mean()),
        "median_trade_return": float(d["trade_return"].astype(float).median()),
        "total_realized_pnl": float(d["pnl"].astype(float).sum()),
    })


def exit_reason_summary(calibration: pd.DataFrame) -> pd.DataFrame:
    """Economic outcomes conditional on exit reason."""
    rows = []
    for reason, g in calibration.groupby("exit_reason", dropna=False):
        rows.append({
            "exit_reason": str(reason),
            "n_trades": int(len(g)),
            "total_pnl": float(g["pnl"].astype(float).sum()),
            "mean_trade_return": float(g["trade_return"].astype(float).mean()),
            "median_trade_return": float(g["trade_return"].astype(float).median()),
            "win_rate": float((g["pnl"].astype(float) > 0).mean()),
            "mean_selected_horizon": float(g["convergence_horizon_trading_days"].astype(float).mean()),
            "mean_observed_days_to_exit": float(g["observed_trading_days_to_exit"].astype(float).mean()),
        })
    return pd.DataFrame(rows).sort_values("n_trades", ascending=False).reset_index(drop=True)


def horizon_bucket_summary(calibration: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Calibration by selected-horizon quartile (or fewer bins if necessary)."""
    d = calibration.loc[calibration["selected_horizon_assessable"]].copy()
    if d.empty:
        return pd.DataFrame()
    q = min(int(n_bins), int(d["convergence_horizon_trading_days"].nunique()))
    if q < 1:
        return pd.DataFrame()
    d["horizon_bucket"] = pd.qcut(
        d["convergence_horizon_trading_days"], q=q, duplicates="drop"
    )
    out = d.groupby("horizon_bucket", observed=True).agg(
        n_trades=("pair", "size"),
        mean_horizon=("convergence_horizon_trading_days", "mean"),
        mean_model_probability=("probability_at_selected_horizon", "mean"),
        observed_within_horizon=("realized_within_selected_horizon", "mean"),
        observed_eventual_convergence=("realized_convergence", "mean"),
        mean_trade_return=("trade_return", "mean"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    out["calibration_gap"] = out["observed_within_horizon"] - out["mean_model_probability"]
    return out
