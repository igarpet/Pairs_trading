from math import exp, log, sqrt
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm


TRADING_DAYS_PER_YEAR = 252
CONTRACT_MULTIPLIER = 100


def black_scholes(spot, strike, time_to_expiry_years, risk_free_rate, volatility, option_type):
    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry_years)
    r = float(risk_free_rate)
    sigma = max(float(volatility), 1e-12)
    is_call = str(option_type).lower() == "call"

    if T <= 0:
        price = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        delta = (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
        return float(price), float(delta)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_delta = float(norm.cdf(d1))

    if is_call:
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = call_delta
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = call_delta - 1.0
    return float(price), float(delta)


def option_types_from_spread_direction(direction):
    return ("put", "call") if direction > 0 else ("call", "put")


def risk_free_at(rates, date):
    if np.isscalar(rates):
        return float(rates)
    series = pd.Series(rates).dropna().astype(float).sort_index()
    series.index = pd.to_datetime(series.index).normalize()
    return float(series.loc[: pd.Timestamp(date).normalize()].iloc[-1])


def compute_log_spread(prices, dependent, independent, alpha, beta):
    return (
        np.log(prices[dependent].astype(float))
        - float(alpha)
        - float(beta) * np.log(prices[independent].astype(float))
    ).rename(f"{dependent}-{independent}")


def ewma_volatility(train_prices, test_prices, lambda_=0.94):
    train = train_prices.sort_index()
    test = test_prices.sort_index()
    out = pd.DataFrame(index=test.index, columns=test.columns, dtype=float)

    for ticker in test.columns:
        train_returns = np.log(train[ticker].astype(float)).diff().dropna()
        var = max(float(train_returns.var(ddof=1)), 1e-12)

        for previous_return in train_returns.to_numpy():
            var = lambda_ * var + (1 - lambda_) * previous_return**2

        combined = pd.concat([train[ticker].tail(1), test[ticker]])
        test_returns = np.log(combined.astype(float)).diff().iloc[1:]
        for date, current_return in test_returns.items():
            out.at[date, ticker] = sqrt(TRADING_DAYS_PER_YEAR * var)
            var = lambda_ * var + (1 - lambda_) * float(current_return) ** 2
    return out


def stable_seed(base_seed, pair, date):
    token = f"{pair}|{pd.Timestamp(date).date()}".encode()
    return int((int(base_seed) + zlib.crc32(token)) % (2**32 - 1))


def prepare_pair_parameters(eligible_pairs, cointegrated_pairs):
    hedge = cointegrated_pairs[["pair", "alpha", "beta"]].drop_duplicates("pair")
    return eligible_pairs.merge(hedge, on="pair", how="left").reset_index(drop=True)
