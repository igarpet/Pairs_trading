"""Black-Scholes, EWMA, log spreads and pair parameters."""

from __future__ import annotations

from math import exp, log, sqrt
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm


TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
CONTRACT_MULTIPLIER = 100


def black_scholes_price(spot, strike, time_to_expiry_years, risk_free_rate, volatility, option_type):
    """European Black-Scholes value without dividends."""
    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry_years)
    r = float(risk_free_rate)
    sigma = max(float(volatility), 1e-12)
    typ = str(option_type).lower()

    if T <= 0:
        return float(max(S - K, 0.0) if typ == "call" else max(K - S, 0.0))

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if typ == "call":
        return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
    return float(K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def black_scholes_delta(spot, strike, time_to_expiry_years, risk_free_rate, volatility, option_type):
    """European Black-Scholes delta without dividends."""
    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry_years)
    r = float(risk_free_rate)
    sigma = max(float(volatility), 1e-12)
    typ = str(option_type).lower()

    if T <= 0:
        if typ == "call":
            return float(1.0 if S > K else 0.0)
        return float(-1.0 if S < K else 0.0)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    call_delta = float(norm.cdf(d1))
    return call_delta if typ == "call" else call_delta - 1.0


def option_types_from_spread_direction(direction):
    """Map spread direction to the two long option legs."""
    return ("put", "call") if direction > 0 else ("call", "put")


def _risk_free_at(risk_free_rates, date):
    """Latest decimal risk-free rate observable on or before date."""
    if np.isscalar(risk_free_rates):
        return float(risk_free_rates)
    rates = pd.Series(risk_free_rates).dropna().astype(float).sort_index()
    rates.index = pd.to_datetime(rates.index).normalize()
    return float(rates.loc[: pd.Timestamp(date).normalize()].iloc[-1])


def compute_log_spread(prices, dependent, independent, alpha, beta):
    dep = prices[dependent].astype(float)
    ind = prices[independent].astype(float)
    return (np.log(dep) - float(alpha) - float(beta) * np.log(ind)).rename(
        f"{dependent}-{independent}"
    )


def precompute_oos_ewma_volatility(train_prices, test_prices, lambda_=0.94):
    """No-look-ahead OOS EWMA annualized volatilities."""
    train = train_prices.sort_index().copy()
    test = test_prices.sort_index().copy()
    common = [c for c in test.columns if c in train.columns]
    out = pd.DataFrame(index=test.index, columns=common, dtype=float)

    for ticker in common:
        train_returns = np.log(train[ticker].astype(float)).diff().dropna()
        var = max(float(train_returns.var(ddof=1)), 1e-12)

        for r_prev in train_returns.to_numpy(dtype=float):
            var = lambda_ * var + (1 - lambda_) * r_prev**2

        combined = pd.concat([train[ticker].tail(1), test[ticker]])
        test_returns = np.log(combined.astype(float)).diff().iloc[1:]

        for date, r_t in test_returns.items():
            out.at[date, ticker] = sqrt(TRADING_DAYS_PER_YEAR * var)
            var = lambda_ * var + (1 - lambda_) * float(r_t) ** 2

    return out


def _stable_seed(base_seed, pair, date):
    token = f"{pair}|{pd.Timestamp(date).date()}".encode("utf-8")
    return int((int(base_seed) + zlib.crc32(token)) % (2**32 - 1))


def _has_converged(current_spread, mu, entry_direction):
    if entry_direction > 0:
        return bool(current_spread <= mu)
    return bool(current_spread >= mu)


def _prepare_pair_parameters(eligible_pairs, cointegrated_pairs):
    eligible = eligible_pairs.copy()
    cointegrated = cointegrated_pairs.copy()

    if "pair" not in eligible.columns:
        eligible["pair"] = eligible["dependent"].astype(str) + "-" + eligible["independent"].astype(str)
    if "pair" not in cointegrated.columns:
        cointegrated["pair"] = cointegrated["dependent"].astype(str) + "-" + cointegrated["independent"].astype(str)

    hedge = cointegrated[["pair", "alpha", "beta"]].drop_duplicates("pair")
    return eligible.merge(hedge, on="pair", how="left").reset_index(drop=True)
