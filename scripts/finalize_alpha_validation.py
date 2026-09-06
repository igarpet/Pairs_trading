from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

from src.backtest import run_walk_forward_backtest, backtest_summary

SEED = 42
INITIAL_CAPITAL = 100_000.0
N_PATHS = 5000
OUT = Path("data/processed/alpha_validation")
OUT.mkdir(parents=True, exist_ok=True)


def load_rf():
    rf_data = pd.read_parquet("data/processed/risk_free_rates.parquet")
    s = rf_data.iloc[:, 0] if isinstance(rf_data, pd.DataFrame) else pd.Series(rf_data)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().astype(float)


def load_equity():
    equity = pd.read_parquet("data/processed/equity_curve.parquet").copy()
    if "date" in equity.columns:
        equity["date"] = pd.to_datetime(equity["date"])
        equity = equity.set_index("date")
    equity.index = pd.to_datetime(equity.index).tz_localize(None)
    return equity.sort_index()


def run_actual_top40():
    train_prices = pd.read_parquet("data/processed/train_prices.parquet")
    test_prices = pd.read_parquet("data/processed/test_prices.parquet")
    top40 = pd.read_parquet("data/processed/eligible_pairs.parquet")
    cointegrated_pairs = pd.read_parquet("data/processed/cointegrated_pairs.parquet")
    risk_free_rates = load_rf()
    res = run_walk_forward_backtest(
        train_prices=train_prices,
        test_prices=test_prices,
        eligible_pairs=top40,
        cointegrated_pairs=cointegrated_pairs,
        risk_free_rates=risk_free_rates,
        initial_capital=INITIAL_CAPITAL,
        entry_z=1.5,
        target_probability=0.70,
        memory_window=60,
        max_horizon_days=126,
        n_paths=N_PATHS,
        ewma_lambda=0.94,
        seed=SEED,
    )
    eq = res["equity_curve"].copy()
    tr = res["trades"].copy()
    s = backtest_summary(tr, eq, INITIAL_CAPITAL)
    daily = eq["equity"].pct_change().dropna()
    daily_std = daily.std(ddof=1)
    return {
        "total_return": float(s["total_return"]),
        "max_drawdown": float(s["max_drawdown"]),
        "final_equity": float(s["final_equity"]),
        "n_trades": int(len(tr)),
        "mean_trade_return": float(tr["trade_return"].mean()) if len(tr) else np.nan,
        "total_trade_pnl": float(tr["pnl"].sum()) if len(tr) else 0.0,
        "daily_sharpe": float(np.sqrt(252) * daily.mean() / daily_std) if daily_std > 0 else np.nan,
    }


def market_alpha_and_bootstrap(equity, risk_free_rates):
    start = equity.index.min()
    end = equity.index.max() + pd.Timedelta(days=1)
    spy = yf.download("SPY", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if spy.empty:
        raise RuntimeError("Could not download SPY for alpha validation.")
    spy_close = spy["Close"].iloc[:, 0] if isinstance(spy.columns, pd.MultiIndex) else spy["Close"]
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)

    strategy_ret = equity["equity"].pct_change().rename("strategy_return")
    market_ret = spy_close.pct_change().rename("market_return")
    rf_daily = (risk_free_rates.reindex(strategy_ret.index, method="ffill") / 252.0).rename("rf_daily")
    reg = pd.concat([strategy_ret, market_ret, rf_daily], axis=1).dropna()
    reg["strategy_excess"] = reg["strategy_return"] - reg["rf_daily"]
    reg["market_excess"] = reg["market_return"] - reg["rf_daily"]
    fit = sm.OLS(reg["strategy_excess"], sm.add_constant(reg["market_excess"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    alpha_daily = float(fit.params["const"])
    alpha_t = float(fit.tvalues["const"])
    alpha_p_two = float(fit.pvalues["const"])
    alpha_p_one = alpha_p_two / 2 if alpha_t > 0 else 1 - alpha_p_two / 2
    factor_alpha = pd.DataFrame([{
        "alpha_daily": alpha_daily,
        "alpha_annualized_simple": alpha_daily * 252,
        "alpha_HAC_se": float(fit.bse["const"]),
        "alpha_t_stat": alpha_t,
        "alpha_p_value_two_sided": alpha_p_two,
        "alpha_p_value_one_sided_positive": alpha_p_one,
        "market_beta": float(fit.params["market_excess"]),
        "r_squared": float(fit.rsquared),
        "n_obs": int(fit.nobs),
    }])
    factor_alpha.to_csv(OUT / "market_adjusted_alpha.csv", index=False)

    returns = strategy_ret.dropna().to_numpy(float)
    block_length = 20
    n_bootstrap = 10_000
    batch = 1_000
    rng = np.random.default_rng(SEED)
    n = len(returns)
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    means = np.empty(n_bootstrap, dtype=float)
    for lo in range(0, n_bootstrap, batch):
        hi = min(lo + batch, n_bootstrap)
        starts = rng.integers(0, n - block_length + 1, size=(hi - lo, n_blocks))
        idx = starts[..., None] + offsets
        samples = returns[idx].reshape(hi - lo, -1)[:, :n]
        means[lo:hi] = samples.mean(axis=1)
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    observed_mean = float(returns.mean())
    bootstrap = pd.DataFrame([{
        "observed_mean_daily_return": observed_mean,
        "observed_annualized_simple_mean": observed_mean * 252,
        "bootstrap_ci_2_5_daily": float(ci_low),
        "bootstrap_ci_97_5_daily": float(ci_high),
        "bootstrap_ci_2_5_annualized_simple": float(ci_low * 252),
        "bootstrap_ci_97_5_annualized_simple": float(ci_high * 252),
        "block_length_trading_days": block_length,
        "n_bootstrap": n_bootstrap,
    }])
    bootstrap.to_csv(OUT / "moving_block_bootstrap.csv", index=False)
    return factor_alpha, bootstrap


def load_placebos():
    files = sorted(Path("placebo_shards").rglob("placebos_shard_*.parquet"))
    if len(files) != 10:
        raise RuntimeError(f"Expected 10 placebo shard files, found {len(files)}")
    placebos = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    placebos = placebos.sort_values("placebo_id").reset_index(drop=True)
    if len(placebos) != 100:
        raise RuntimeError(f"Expected 100 placebo portfolios, found {len(placebos)}")
    if placebos["placebo_id"].nunique() != 100:
        raise RuntimeError("Duplicate placebo_id values detected.")
    if placebos["sampled_pairs"].nunique() != 100:
        raise RuntimeError("Exact duplicate placebo portfolios detected.")
    placebos.to_parquet(OUT / "pair_selection_placebos.parquet", index=False)
    return placebos


def randomization_test(actual, placebos):
    rows = []
    for metric in ["total_return", "daily_sharpe", "total_trade_pnl", "mean_trade_return"]:
        a = float(actual[metric])
        null = placebos[metric].dropna().to_numpy(float)
        p = (1 + np.sum(null >= a)) / (len(null) + 1)
        rows.append({
            "metric": metric,
            "actual": a,
            "placebo_mean": float(null.mean()),
            "placebo_median": float(np.median(null)),
            "one_sided_randomization_p_value": float(p),
            "actual_percentile_vs_placebos": float(100 * np.mean(null < a)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "pair_selection_randomization_test.csv", index=False)
    return out


def main():
    equity = load_equity()
    risk_free_rates = load_rf()
    factor_alpha, bootstrap = market_alpha_and_bootstrap(equity, risk_free_rates)
    placebos = load_placebos()
    actual = run_actual_top40()
    actual_df = pd.DataFrame([actual])
    actual_df.to_csv(OUT / "actual_top40_5000_paths.csv", index=False)
    test = randomization_test(actual, placebos)

    alpha_pass = bool(factor_alpha.loc[0, "alpha_p_value_one_sided_positive"] < 0.05)
    bootstrap_pass = bool(bootstrap.loc[0, "bootstrap_ci_2_5_daily"] > 0)
    ret_row = test.loc[test["metric"].eq("total_return")].iloc[0]
    placebo_pass = bool(ret_row["one_sided_randomization_p_value"] < 0.05)
    scorecard = pd.DataFrame([
        {"test": "HAC positive alpha", "reject_null_at_5pct": alpha_pass, "evidence": f"alpha={factor_alpha.loc[0, 'alpha_daily']:.6f}/day, one-sided p={factor_alpha.loc[0, 'alpha_p_value_one_sided_positive']:.4f}"},
        {"test": "20-day moving-block bootstrap", "reject_null_at_5pct": bootstrap_pass, "evidence": f"95% CI daily mean=[{bootstrap.loc[0, 'bootstrap_ci_2_5_daily']:.6f}, {bootstrap.loc[0, 'bootstrap_ci_97_5_daily']:.6f}]"},
        {"test": "Structural Top40 pair-selection placebo", "reject_null_at_5pct": placebo_pass, "evidence": f"actual return={ret_row['actual']:.3%}, randomization p={ret_row['one_sided_randomization_p_value']:.4f}"},
    ])
    scorecard.to_csv(OUT / "alpha_validation_scorecard.csv", index=False)

    canonical_final = float(equity["equity"].iloc[-1])
    canonical_return = canonical_final / INITIAL_CAPITAL - 1
    summary = Path("workflow_final_summary.md")
    summary.write_text(
        "# Final distributed thesis run\n\n"
        f"- Canonical 5,000-path final equity: ${canonical_final:,.2f}\n"
        f"- Canonical 5,000-path total return: {canonical_return:.3%}\n"
        f"- Actual Top40 rerun at 5,000 paths: {actual['total_return']:.3%}\n"
        f"- Placebo portfolios: {len(placebos)} (10 shards x 10 portfolios)\n"
        f"- Exact duplicate placebo portfolios: {100 - placebos['sampled_pairs'].nunique()}\n"
        f"- Placebo mean total return: {placebos['total_return'].mean():.3%}\n"
        f"- Randomization p-value (total return): {float(ret_row['one_sided_randomization_p_value']):.4f}\n"
        f"- HAC one-sided alpha p-value: {factor_alpha.loc[0, 'alpha_p_value_one_sided_positive']:.4f}\n"
        f"- Bootstrap 95% daily-mean CI: [{bootstrap.loc[0, 'bootstrap_ci_2_5_daily']:.6f}, {bootstrap.loc[0, 'bootstrap_ci_97_5_daily']:.6f}]\n",
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
