import numpy as np
import pandas as pd
import statsmodels.api as sm


def aligned_returns(equity, benchmark, rates):
    index = equity.index
    benchmark = benchmark.reindex(index)
    rf = rates.reindex(index, method="ffill").shift(1)
    data = pd.DataFrame(
        {
            "strategy_return": equity.equity.pct_change(fill_method=None),
            "market_return": benchmark.pct_change(fill_method=None),
            "rf_daily": (1 + rf) ** (1 / 252) - 1,
        }
    ).iloc[1:]
    data["strategy_excess"] = data.strategy_return - data.rf_daily
    data["market_excess"] = data.market_return - data.rf_daily
    return data


def market_regression(aligned, lags=5):
    x = sm.add_constant(aligned.market_excess, has_constant="add")
    fit = sm.OLS(aligned.strategy_excess, x).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    pvalue = float(fit.pvalues["const"])
    tstat = float(fit.tvalues["const"])
    return {
        "alpha_daily": float(fit.params["const"]),
        "alpha_annualized_simple": float(252 * fit.params["const"]),
        "alpha_HAC_se": float(fit.bse["const"]),
        "alpha_t_stat": tstat,
        "alpha_p_value_two_sided": pvalue,
        "alpha_p_value_one_sided_positive": pvalue / 2 if tstat > 0 else 1 - pvalue / 2,
        "market_beta": float(fit.params["market_excess"]),
        "r_squared": float(fit.rsquared),
        "n_obs": len(aligned),
        "hac_lags": lags,
    }


def bootstrap_mean(returns, blocks=(10, 20, 40), replications=10000, seed=42):
    x = np.asarray(returns, dtype=float)
    rows = []
    for block in blocks:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(block)]))
        means = np.empty(replications)
        for i in range(0, replications, 200):
            n = min(200, replications - i)
            starts = rng.integers(
                0,
                len(x) - block + 1,
                size=(n, int(np.ceil(len(x) / block))),
            )
            sample = x[(starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, : len(x)]]
            means[i : i + n] = sample.mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        rows.append(
            {
                "block_length": block,
                "n_replications": replications,
                "observed_mean": float(x.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def placebo_samples(pool, n_pairs, n_portfolios, seed):
    ordered = pool.sort_values("pair").reset_index(drop=True)
    for j in range(n_portfolios):
        rng = np.random.default_rng(np.random.SeedSequence([seed, j]))
        sample = rng.choice(len(ordered), n_pairs, replace=False)
        yield j, ordered.iloc[sample].sort_values("pair").reset_index(drop=True)


def backtest_summary(trades, equity_curve, initial_capital):
    equity = equity_curve.equity.astype(float)
    running_max = equity.cummax().clip(lower=initial_capital)
    drawdown = equity / running_max - 1

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / initial_capital - 1),
        "max_drawdown": float(drawdown.min()),
        "n_trades": int(len(trades)),
        "win_rate": float((trades.pnl > 0).mean()) if len(trades) else np.nan,
        "average_trade_return": float(trades.trade_return.mean()) if len(trades) else np.nan,
        "median_trade_return": float(trades.trade_return.median()) if len(trades) else np.nan,
        "max_concurrent_positions": int(equity_curve.n_open_positions.max()),
    }


def portfolio_metrics(result, initial_capital):
    metrics = backtest_summary(result["trades"], result["equity_curve"], initial_capital)
    returns = result["equity_curve"].equity.pct_change(fill_method=None).dropna()
    metrics["daily_sharpe"] = float(np.sqrt(252) * returns.mean() / returns.std())
    return metrics


def compare_placebos(actual, placebos):
    rows = []
    for metric in ["total_return", "daily_sharpe", "average_trade_return"]:
        values = pd.to_numeric(placebos[metric], errors="coerce").dropna()
        value = actual[metric]
        rows.append(
            {
                "metric": metric,
                "actual": value,
                "n_draws": len(values),
                "p_value": (1 + int((values >= value).sum())) / (len(values) + 1),
                "percentile": 100 * float((values < value).mean()),
            }
        )
    return pd.DataFrame(rows)


def label_forecasts(forecasts, prices):
    rows = []
    for row in forecasts.to_dict("records"):
        start = pd.Timestamp(row["signal_date"])
        horizon = int(row["convergence_horizon_trading_days"])
        probability = float(row["probability_at_selected_horizon"])
        i = prices.index.get_loc(start)
        future = prices.iloc[i + 1 : i + horizon + 1]
        spread = (
            np.log(future[row["dependent"]])
            - row["alpha"]
            - row["beta"] * np.log(future[row["independent"]])
        )

        valid = np.isfinite(spread.to_numpy())
        observed_steps = int(np.argmax(~valid)) if (~valid).any() else len(spread)
        observed = spread.iloc[:observed_steps]
        crossed = observed.le(row["mu"]) if row["direction"] > 0 else observed.ge(row["mu"])
        success = bool(crossed.any())
        complete = observed_steps == horizon
        status = "success" if success else "failure" if complete else "censored"
        realized = float(success) if success or complete else np.nan

        rows.append(
            {
                **row,
                "event_status": status,
                "complete_horizon_observed": complete,
                "observed_path_steps": observed_steps,
                "first_crossing_date": crossed.index[np.flatnonzero(crossed)[0]] if success else pd.NaT,
                "forecast_horizon_date": prices.index[i + horizon] if i + horizon < len(prices) else pd.NaT,
                "realized_within_selected_horizon": realized,
                "brier_component": (realized - probability) ** 2,
                "calibration_error": realized - probability,
            }
        )
    return pd.DataFrame(rows)


def summarize_forecasts(labels):
    complete = labels.loc[labels.complete_horizon_observed] if len(labels) else labels
    return {
        "n_forecasts": len(labels),
        "n_complete_horizon": len(complete),
        "n_censored": int(labels.event_status.eq("censored").sum()) if len(labels) else 0,
        "mean_prediction": float(complete.probability_at_selected_horizon.mean()) if len(complete) else np.nan,
        "observed_rate": float(complete.realized_within_selected_horizon.mean()) if len(complete) else np.nan,
        "brier_score": float(complete.brier_component.mean()) if len(complete) else np.nan,
    }
