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


def build_trade_calibration_table(trades, test_index, prices=None):
    """Require actual spread paths; exit reasons cannot identify first passages."""
    from src.forecast_calibration import label_forecasts
    if prices is None:
        raise ValueError('Calibration now requires prices=full_prices; exit reasons alone are insufficient.')
    result = label_forecasts(trades, prices)
    if result.empty:
        return result
    result['realized_convergence'] = result['event_status'].eq('success').astype(int)
    result['censored_end_of_test'] = result['event_status'].eq('censored')
    result['observed_trading_days_to_exit'] = [
        trading_days_between(prices.index,a,b) for a,b in zip(result.entry_date,result.exit_date)]
    return result


def calibration_summary(calibration: pd.DataFrame) -> pd.Series:
    """Headline calibration statistics on assessable trades."""
    d = calibration.loc[calibration["complete_horizon_observed"]].copy()
    if d.empty:
        raise ValueError("No assessable trades for calibration.")

    p = d["probability_at_selected_horizon"].astype(float)
    y = d["realized_within_selected_horizon"].astype(float)
    conv = d["realized_convergence"].astype(float)
    expiry = (d["exit_reason"].astype(str) == "expiry").astype(float)

    return pd.Series({
        "n_total_trades": int(len(calibration)),
        "n_assessable_selected_horizon": int(len(d)),
        "n_censored_end_of_test": int((~calibration["complete_horizon_observed"]).sum()),
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
    d = calibration.loc[calibration["complete_horizon_observed"]].copy()
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
