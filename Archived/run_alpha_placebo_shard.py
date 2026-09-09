from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest, backtest_summary
from src.pair_eligibility import filter_antipersistent_pairs

SEED = 42
INITIAL_CAPITAL = 100_000.0
N_PLACEBOS = 100
N_PATHS = 5000
N_SHARDS = 10


def load_inputs():
    train_prices = pd.read_parquet("data/processed/train_prices.parquet")
    test_prices = pd.read_parquet("data/processed/test_prices.parquet")
    top40 = pd.read_parquet("data/processed/eligible_pairs.parquet")
    cointegrated_pairs = pd.read_parquet("data/processed/cointegrated_pairs.parquet")
    fou_parameters = pd.read_parquet("data/processed/fractional_ou_parameters.parquet")
    rf_data = pd.read_parquet("data/processed/risk_free_rates.parquet")
    risk_free_rates = rf_data.iloc[:, 0] if isinstance(rf_data, pd.DataFrame) else pd.Series(rf_data)
    risk_free_rates.index = pd.to_datetime(risk_free_rates.index)
    risk_free_rates = risk_free_rates.sort_index().astype(float)
    return train_prices, test_prices, top40, cointegrated_pairs, fou_parameters, risk_free_rates


def generate_unique_samples(pool: pd.DataFrame, n_pairs: int) -> list[np.ndarray]:
    """Generate 100 deterministic, exact-portfolio-unique placebo samples.

    Pair overlap between different placebo portfolios is intentional under the
    random-selection null. Only exact duplicate 40-pair portfolios are rejected.
    """
    rng = np.random.default_rng(SEED)
    samples: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()

    while len(samples) < N_PLACEBOS:
        idx = rng.choice(len(pool), size=n_pairs, replace=False)
        pairs = tuple(sorted(pool.iloc[idx]["pair"].astype(str).tolist()))
        if pairs in seen:
            continue
        seen.add(pairs)
        samples.append(idx)
    return samples


def run_portfolio(pair_frame, train_prices, test_prices, cointegrated_pairs, risk_free_rates):
    res = run_walk_forward_backtest(
        train_prices=train_prices,
        test_prices=test_prices,
        eligible_pairs=pair_frame,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True, choices=range(N_SHARDS))
    parser.add_argument("--out", type=Path, default=Path("placebo_shards"))
    args = parser.parse_args()

    train_prices, test_prices, top40, cointegrated_pairs, fou_parameters, risk_free_rates = load_inputs()
    pool = filter_antipersistent_pairs(fou_parameters).drop_duplicates("pair").reset_index(drop=True)
    n_pairs = len(top40)
    samples = generate_unique_samples(pool, n_pairs)

    start = args.shard * (N_PLACEBOS // N_SHARDS)
    stop = start + (N_PLACEBOS // N_SHARDS)
    rows = []

    print(f"Shard {args.shard}: placebo ids {start}..{stop - 1}")
    print(f"H<0.5 pool: {len(pool)}, pairs/portfolio: {n_pairs}, paths/signal: {N_PATHS}")

    for placebo_id in range(start, stop):
        sampled = pool.iloc[samples[placebo_id]].copy()
        stats = run_portfolio(sampled, train_prices, test_prices, cointegrated_pairs, risk_free_rates)
        stats["placebo_id"] = placebo_id
        stats["sampled_pairs"] = "|".join(sorted(sampled["pair"].astype(str)))
        rows.append(stats)
        print(f"Completed placebo {placebo_id} ({placebo_id - start + 1}/{stop - start})")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"placebos_shard_{args.shard:02d}.parquet"
    pd.DataFrame(rows).sort_values("placebo_id").to_parquet(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
