"""Frozen benchmark regression, raw-mean bootstrap and matched placebo statistics."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def aligned_returns(equity, benchmark, rates):
    idx = equity.index
    benchmark = benchmark.reindex(idx)
    rf = rates.reindex(idx, method="ffill").shift(1)
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
    if len(aligned) <= max(3, lags) or aligned.market_excess.std() < 1e-12:
        return {"status": "insufficient_market_variation", "n_obs": len(aligned)}

    x = sm.add_constant(aligned.market_excess, has_constant="add")
    fit = sm.OLS(aligned.strategy_excess, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    pvalue = float(fit.pvalues["const"])
    tstat = float(fit.tvalues["const"])

    if not np.isfinite([pvalue, tstat]).all():
        return {"status": "degenerate_regression", "n_obs": len(aligned)}

    return dict(
        status="ok",
        alpha_daily=float(fit.params["const"]),
        alpha_annualized_simple=float(252 * fit.params["const"]),
        alpha_HAC_se=float(fit.bse["const"]),
        alpha_t_stat=tstat,
        alpha_p_value_two_sided=pvalue,
        alpha_p_value_one_sided_positive=pvalue / 2 if tstat > 0 else 1 - pvalue / 2,
        market_beta=float(fit.params["market_excess"]),
        r_squared=float(fit.rsquared),
        n_obs=len(aligned),
        hac_lags=lags,
        rf_convention="annual_effective_lagged_252",
        start=str(aligned.index[0]),
        end=str(aligned.index[-1]),
    )


def bootstrap_mean(returns, blocks=(10, 20, 40), replications=10000, seed=42):
    x = np.asarray(returns, dtype=float)
    rows = []
    for block in blocks:
        if block > len(x):
            rows.append(dict(block_length=block, status="insufficient_observations"))
            continue

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

        lo, hi = np.quantile(means, [0.025, 0.975])
        rows.append(
            dict(
                block_length=block,
                status="ok",
                n_replications=replications,
                observed_mean=float(x.mean()),
                ci_low=float(lo),
                ci_high=float(hi),
                estimand="unconditional_mean_raw_daily_return",
            )
        )
    return pd.DataFrame(rows)


def placebo_samples(pool, n_pairs, n_portfolios, seed):
    """Uniform subsets without replacement within a draw; repeated draws are allowed."""
    ordered = pool.sort_values("pair").reset_index(drop=True)
    for j in range(n_portfolios):
        rng = np.random.default_rng(np.random.SeedSequence([seed, j]))
        sample = rng.choice(len(ordered), n_pairs, replace=False)
        yield j, ordered.iloc[sample].sort_values("pair").reset_index(drop=True)


def portfolio_metrics(result, initial_capital):
    metrics = backtest_summary(result["trades"], result["equity_curve"], initial_capital).to_dict()
    returns = result["equity_curve"].equity.pct_change(fill_method=None).dropna()
    metrics["daily_sharpe"] = (
        float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else None
    )
    return metrics


def compare_placebos(actual, placebos):
    rows = []
    for key in ["total_return", "daily_sharpe", "average_trade_return"]:
        actual_value = actual.get(key)
        values = pd.to_numeric(placebos[key], errors="coerce") if key in placebos else pd.Series(dtype=float)
        finite = values[np.isfinite(values)]

        if (
            actual_value is None
            or not np.isfinite(actual_value)
            or len(finite) != len(placebos)
            or len(finite) == 0
        ):
            rows.append(dict(metric=key, status="undefined_or_incomplete", n_draws=len(values)))
            continue

        rows.append(
            dict(
                metric=key,
                status="ok",
                actual=actual_value,
                n_draws=len(finite),
                p_value=(1 + int((finite >= actual_value).sum())) / (len(finite) + 1),
                percentile=100 * float((finite < actual_value).mean()),
            )
        )
    return pd.DataFrame(rows)


def backtest_summary(trades, equity_curve, initial_capital):
    """Compact Module 07 sanity-check summary; full analysis belongs in Module 08."""
    equity = equity_curve["equity"].astype(float)
    running_max = equity.cummax().clip(lower=initial_capital)
    drawdown = equity / running_max - 1.0

    if trades.empty:
        wins = np.nan
        avg_trade_return = np.nan
        median_trade_return = np.nan
    else:
        wins = float((trades["pnl"] > 0).mean())
        avg_trade_return = float(trades["trade_return"].mean())
        median_trade_return = float(trades["trade_return"].median())

    return pd.Series(
        {
            "initial_capital": float(initial_capital),
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1] / initial_capital - 1.0),
            "max_drawdown": float(drawdown.min()),
            "n_trades": int(len(trades)),
            "win_rate": wins,
            "average_trade_return": avg_trade_return,
            "median_trade_return": median_trade_return,
            "max_concurrent_positions": int(equity_curve["n_open_positions"].max()),
        },
        name="module_07_summary",
    )
