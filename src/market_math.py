"""Black-Scholes, EWMA, log spreads and pair parameters.

See METHODOLOGY.md for timing, risk budgets and forecast event definitions.
"""

from __future__ import annotations

from math import exp, log, sqrt
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm


TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
CONTRACT_MULTIPLIER = 100


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
) -> float:
    """European Black-Scholes value without dividends."""
    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry_years)
    r = float(risk_free_rate)
    sigma = float(volatility)
    typ = str(option_type).lower()

    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be positive.")
    if typ not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    if T <= 0:
        return float(max(S - K, 0.0) if typ == "call" else max(K - S, 0.0))

    sigma = max(sigma, 1e-12)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if typ == "call":
        return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
    return float(K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def black_scholes_delta(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
) -> float:
    """European Black-Scholes delta without dividends."""
    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry_years)
    r = float(risk_free_rate)
    sigma = max(float(volatility), 1e-12)
    typ = str(option_type).lower()

    if typ not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    if T <= 0:
        if typ == "call":
            return float(1.0 if S > K else 0.0)
        return float(-1.0 if S < K else 0.0)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    call_delta = float(norm.cdf(d1))
    return call_delta if typ == "call" else call_delta - 1.0


def option_types_from_spread_direction(direction: int) -> tuple[str, str]:
    """
    direction = +1 means spread is above equilibrium:
        dependent PUT, independent CALL.
    direction = -1 means spread is below equilibrium:
        dependent CALL, independent PUT.
    """
    if direction == 1:
        return "put", "call"
    if direction == -1:
        return "call", "put"
    raise ValueError("direction must be +1 or -1.")


def _risk_free_at(
    risk_free_rates,
    date,
) -> float:
    """
    Accept either a scalar decimal rate or a dated pandas Series of decimal rates.
    Uses the latest rate observable on or before `date`.
    """
    if np.isscalar(risk_free_rates):
        return float(risk_free_rates)

    s = pd.Series(risk_free_rates).dropna().astype(float).sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    available = s.loc[: pd.Timestamp(date).normalize()]
    if available.empty:
        raise ValueError(f"No risk-free rate available on or before {date}.")
    return float(available.iloc[-1])


def compute_log_spread(
    prices: pd.DataFrame,
    dependent: str,
    independent: str,
    alpha: float,
    beta: float,
) -> pd.Series:
    dep = prices[dependent].astype(float)
    ind = prices[independent].astype(float)
    if (dep <= 0).any() or (ind <= 0).any():
        raise ValueError("Prices must be strictly positive for log spread.")
    spread = np.log(dep) - float(alpha) - float(beta) * np.log(ind)
    return spread.rename(f"{dependent}-{independent}")


def precompute_oos_ewma_volatility(
    train_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    lambda_: float = 0.94,
) -> pd.DataFrame:
    """
    Precompute no-look-ahead OOS EWMA annualized volatilities.

    The variance entering the first OOS day is initialized from training returns.
    On OOS date t, sigma_t^2 uses information through return r_{t-1}, consistent
    with sigma_t^2 = lambda*sigma_{t-1}^2 + (1-lambda)*r_{t-1}^2.
    """
    if not 0 < float(lambda_) < 1:
        raise ValueError("lambda_ must be in (0,1).")

    train = train_prices.sort_index().copy()
    test = test_prices.sort_index().copy()

    common = [c for c in test.columns if c in train.columns]
    out = pd.DataFrame(index=test.index, columns=common, dtype=float)

    for ticker in common:
        tr = np.log(train[ticker].astype(float)).diff().dropna()
        if len(tr) < 2:
            raise ValueError(f"Insufficient training returns for {ticker}.")

        var = max(float(tr.var(ddof=1)), 1e-12)

        # Bring EWMA state through the complete training return history.
        vals = tr.to_numpy(dtype=float)
        for r_prev in vals:
            var = float(lambda_) * var + (1.0 - float(lambda_)) * float(r_prev) ** 2

        combined = pd.concat([train[ticker].tail(1), test[ticker]])
        test_returns = np.log(combined.astype(float)).diff().iloc[1:]

        # At date t, store variance known before incorporating r_t.
        for date, r_t in test_returns.items():
            out.at[date, ticker] = sqrt(TRADING_DAYS_PER_YEAR * var)
            var = float(lambda_) * var + (1.0 - float(lambda_)) * float(r_t) ** 2

    return out


def _stable_seed(base_seed: int, pair: str, date) -> int:
    token = f"{pair}|{pd.Timestamp(date).date()}".encode("utf-8")
    extra = zlib.crc32(token)
    return int((int(base_seed) + extra) % (2**32 - 1))


def _has_converged(current_spread: float, mu: float, entry_direction: int) -> bool:
    if entry_direction > 0:
        return bool(current_spread <= mu)
    return bool(current_spread >= mu)


def _prepare_pair_parameters(
    eligible_pairs: pd.DataFrame,
    cointegrated_pairs: pd.DataFrame,
) -> pd.DataFrame:
    e = eligible_pairs.copy()
    c = cointegrated_pairs.copy()

    if "pair" not in e.columns:
        e["pair"] = e["dependent"].astype(str) + "-" + e["independent"].astype(str)
    if "pair" not in c.columns:
        c["pair"] = c["dependent"].astype(str) + "-" + c["independent"].astype(str)

    # Keep fOU columns from eligible set and add alpha/beta from Module 02.
    needed_c = c[["pair", "alpha", "beta"]].drop_duplicates("pair")
    p = e.merge(needed_c, on="pair", how="left", validate="one_to_one")

    required = {
        "pair",
        "dependent",
        "independent",
        "alpha",
        "beta",
        "mu",
        "kappa",
        "sigma",
        "hurst",
        "variance",
    }
    missing = sorted(required - set(p.columns))
    if missing:
        raise KeyError(f"Missing required pair parameter columns: {missing}")

    if p[["alpha", "beta"]].isna().any().any():
        bad = p.loc[p[["alpha", "beta"]].isna().any(axis=1), "pair"].tolist()
        raise ValueError(f"Missing alpha/beta for pairs: {bad}")

    return p.reset_index(drop=True)
