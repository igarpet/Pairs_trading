"""First-passage event labels from prices, independently of option exits."""

import numpy as np
import pandas as pd


def label_forecasts(forecasts, prices):
    rows = []
    for row in forecasts.to_dict("records"):
        start = pd.Timestamp(row.get("signal_date", row.get("entry_date")))
        horizon = int(row["convergence_horizon_trading_days"])
        probability = float(row["probability_at_selected_horizon"])
        i = prices.index.get_loc(start)
        future = prices.iloc[i + 1 : i + horizon + 1]
        values = (
            np.log(future[row["dependent"]])
            - row["alpha"]
            - row["beta"] * np.log(future[row["independent"]])
        )

        valid = np.isfinite(values.to_numpy())
        prefix = int(np.argmax(~valid)) if (~valid).any() else len(values)
        observed = values.iloc[:prefix]
        hit = observed.le(row["mu"]) if row["direction"] > 0 else observed.ge(row["mu"])
        success = bool(hit.any())
        complete = prefix == horizon
        status = "success" if success else ("failure" if complete else "censored")
        realized = float(success) if (success or complete) else np.nan

        rows.append(
            {
                **row,
                "event_status": status,
                "selected_horizon_assessable": bool(success or complete),
                "complete_horizon_observed": complete,
                "observed_path_steps": prefix,
                "first_crossing_date": hit.index[np.flatnonzero(hit)[0]] if success else pd.NaT,
                "forecast_horizon_date": prices.index[i + horizon] if i + horizon < len(prices) else pd.NaT,
                "realized_within_selected_horizon": realized,
                "brier_component": (realized - probability) ** 2,
                "calibration_error": realized - probability,
            }
        )
    return pd.DataFrame(rows)


def summarize_forecasts(labels):
    if labels.empty:
        return dict(n_forecasts=0, n_complete_horizon=0, n_censored=0, brier_score=None)

    full = labels.loc[labels.complete_horizon_observed]
    return dict(
        n_forecasts=len(labels),
        n_complete_horizon=len(full),
        n_censored=int(labels.event_status.eq("censored").sum()),
        mean_prediction=float(full.probability_at_selected_horizon.mean()) if len(full) else None,
        observed_rate=float(full.realized_within_selected_horizon.mean()) if len(full) else None,
        brier_score=float(full.brier_component.mean()) if len(full) else None,
    )
