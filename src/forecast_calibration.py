"""First-passage event labels from prices, independently of option exits."""

import numpy as np
import pandas as pd


def label_forecasts(forecasts, prices):
    rows = []
    for r in forecasts.to_dict("records"):
        start = pd.Timestamp(r.get("signal_date", r.get("entry_date")))
        h = int(r["convergence_horizon_trading_days"])
        if h < 1 or start not in prices.index:
            raise ValueError("Forecast origin must be on the price session index.")
        p = float(r["probability_at_selected_horizon"])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Forecast probability must lie in [0,1].")
        i = prices.index.get_loc(start)
        future = prices.iloc[i + 1 : i + h + 1]
        values = (
            np.log(future[r["dependent"]])
            - r["alpha"]
            - r["beta"] * np.log(future[r["independent"]])
        )
        valid = np.isfinite(values.to_numpy())
        # Do not bridge a missing session: its first crossing is unknowable.
        prefix = int(np.argmax(~valid)) if (~valid).any() else len(values)
        observed = values.iloc[:prefix]
        hit = observed.le(r["mu"]) if r["direction"] > 0 else observed.ge(r["mu"])
        success = bool(hit.any())
        complete = prefix == h
        status = "success" if success else ("failure" if complete else "censored")
        y = float(success) if (success or complete) else np.nan
        rows.append(
            {
                **r,
                "event_status": status,
                "selected_horizon_assessable": bool(success or complete),
                "complete_horizon_observed": complete,
                "observed_path_steps": prefix,
                "first_crossing_date": (
                    hit.index[np.flatnonzero(hit)[0]] if success else pd.NaT
                ),
                "forecast_horizon_date": (
                    prices.index[i + h] if i + h < len(prices) else pd.NaT
                ),
                "realized_within_selected_horizon": y,
                "brier_component": (y - p) ** 2,
                "calibration_error": y - p,
            }
        )
    return pd.DataFrame(rows)


def summarize_forecasts(labels):
    if labels.empty:
        return dict(n_forecasts=0, n_complete_horizon=0, n_censored=0, brier_score=None)
    # Complete horizons for headline: including early hits but dropping late censored
    # failures would select on outcomes. The complete-window subset avoids that.
    full = labels.loc[labels.complete_horizon_observed]
    return dict(
        n_forecasts=len(labels),
        n_complete_horizon=len(full),
        n_censored=int(labels.event_status.eq("censored").sum()),
        mean_prediction=(
            float(full.probability_at_selected_horizon.mean()) if len(full) else None
        ),
        observed_rate=(
            float(full.realized_within_selected_horizon.mean()) if len(full) else None
        ),
        brier_score=float(full.brier_component.mean()) if len(full) else None,
    )
