"""
backtest.py

Module 07 — Out-of-sample walk-forward backtest for the fOU pairs strategy
implemented with synthetic long equity options.

Core design
-----------
Formation parameters are frozen from the 70% training sample:
    alpha, beta, H, mu, sigma, kappa, stationary variance, eligible pair set.

During the 30% out-of-sample period, only state variables are updated:
    - current log-price spread and Z-score,
    - recent inferred fGn history and conditional convergence horizon,
    - EWMA volatility for each equity leg,
    - risk-free rate,
    - synthetic option mark-to-model value.

Entry
-----
A pair can be entered only when:
    |Z_t| >= entry_z
and Module 04 finds a convergence horizon T^70 within max_horizon_days.

At most one position per pair is open at a time.

Synthetic option construction
-----------------------------
Spread:
    X_t = log(P_dep,t) - alpha - beta * log(P_ind,t)

If Z_t > 0:
    spread above equilibrium -> short spread:
        dependent PUT + independent CALL

If Z_t < 0:
    spread below equilibrium -> long spread:
        dependent CALL + independent PUT

Synthetic options are ATM at entry:
    K = S_entry

Required calendar maturity:
    ceil(T^70 * 365 / 252)

The synthetic expiry is the first observed trading date on or after the
required calendar target date when such a date exists in the test sample.
If the target is beyond the test sample, the calendar target date itself is
retained and the position is marked to model until the test ends.

Contract sizing
---------------
For a log-price spread, delta-dollar exposures are matched approximately:

    N_ind * |Delta_ind| * S_ind
        ~= beta * N_dep * |Delta_dep| * S_dep

Define:
    r = N_ind / N_dep
      = beta * |Delta_dep| * S_dep
        / (|Delta_ind| * S_ind)

The smallest transparent integer hedge unit is used:
    - if r >= 1: N_dep = 1, N_ind = round(r)
    - if r <  1: N_ind = 1, N_dep = round(1/r)

with each leg constrained to at least one contract.

Capital
-------
Long options are fully premium-funded. No leverage or borrowing is used.
Cash is reduced by entry premium and restored by exit value. If cash is
insufficient for the minimum hedge unit, the signal is skipped.

Exit
----
1. Statistical convergence: spread reaches/crosses frozen equilibrium mu.
2. Option expiry: intrinsic value.
3. End of test: remaining positions are marked to model and force-closed.

No transaction costs or bid/ask spread are included because the options are
synthetic/model-priced rather than historical quoted contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, log, sqrt
from typing import Optional
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm

from convergence_signal import calculate_convergence_signal


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


def trading_to_calendar_days(trading_days: int) -> int:
    if int(trading_days) < 1:
        raise ValueError("trading_days must be >= 1.")
    return int(np.ceil(int(trading_days) * CALENDAR_DAYS_PER_YEAR / TRADING_DAYS_PER_YEAR))


def next_observed_date_on_or_after(
    target_date,
    observed_dates: pd.DatetimeIndex,
) -> pd.Timestamp:
    """
    Use the first observed trading date on/after target when available.
    Otherwise retain target_date (useful for positions extending past test end).
    """
    target = pd.Timestamp(target_date).normalize()
    idx = pd.DatetimeIndex(observed_dates).normalize().sort_values()
    future = idx[idx >= target]
    return pd.Timestamp(future[0]) if len(future) else target


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
    available = s.loc[:pd.Timestamp(date).normalize()]
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


def size_log_spread_option_legs(
    beta: float,
    dependent_spot: float,
    independent_spot: float,
    dependent_delta: float,
    independent_delta: float,
) -> dict:
    """
    Integer contract sizing for log-price spread using delta-dollar exposure.
    """
    beta = float(beta)
    if beta <= 0:
        raise ValueError(
            "Current option-direction mapping assumes beta > 0. "
            "Negative-beta pairs require a different leg-direction mapping."
        )

    dep_exposure_1 = abs(float(dependent_delta)) * float(dependent_spot)
    ind_exposure_1 = abs(float(independent_delta)) * float(independent_spot)

    if dep_exposure_1 <= 0 or ind_exposure_1 <= 0:
        raise ValueError("Option delta-dollar exposure must be positive.")

    target_ratio = beta * dep_exposure_1 / ind_exposure_1  # N_ind / N_dep

    if target_ratio >= 1.0:
        n_dep = 1
        n_ind = max(1, int(np.rint(target_ratio)))
    else:
        n_ind = 1
        n_dep = max(1, int(np.rint(1.0 / target_ratio)))

    lhs = n_ind * ind_exposure_1
    rhs = beta * n_dep * dep_exposure_1
    hedge_error = abs(lhs - rhs) / max(abs(rhs), 1e-12)

    return {
        "dependent_contracts": int(n_dep),
        "independent_contracts": int(n_ind),
        "target_contract_ratio_ind_over_dep": float(target_ratio),
        "realized_contract_ratio_ind_over_dep": float(n_ind / n_dep),
        "relative_hedge_error": float(hedge_error),
    }


def _stable_seed(base_seed: int, pair: str, date) -> int:
    token = f"{pair}|{pd.Timestamp(date).date()}".encode("utf-8")
    extra = zlib.crc32(token)
    return int((int(base_seed) + extra) % (2**32 - 1))


def _position_market_value(
    position: dict,
    date,
    dep_spot: float,
    indep_spot: float,
    dep_vol: float,
    indep_vol: float,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    date = pd.Timestamp(date).normalize()
    expiry = pd.Timestamp(position["expiry_date"]).normalize()
    remaining_days = max(int((expiry - date).days), 0)
    T = remaining_days / CALENDAR_DAYS_PER_YEAR

    dep_px = black_scholes_price(
        dep_spot,
        position["dependent_strike"],
        T,
        risk_free_rate,
        dep_vol,
        position["dependent_option_type"],
    )
    ind_px = black_scholes_price(
        indep_spot,
        position["independent_strike"],
        T,
        risk_free_rate,
        indep_vol,
        position["independent_option_type"],
    )

    total = CONTRACT_MULTIPLIER * (
        position["dependent_contracts"] * dep_px
        + position["independent_contracts"] * ind_px
    )
    return float(total), float(dep_px), float(ind_px)


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
        "pair", "dependent", "independent", "alpha", "beta",
        "mu", "kappa", "sigma", "hurst", "variance",
    }
    missing = sorted(required - set(p.columns))
    if missing:
        raise KeyError(f"Missing required pair parameter columns: {missing}")

    if p[["alpha", "beta"]].isna().any().any():
        bad = p.loc[p[["alpha", "beta"]].isna().any(axis=1), "pair"].tolist()
        raise ValueError(f"Missing alpha/beta for pairs: {bad}")

    return p.reset_index(drop=True)


def run_walk_forward_backtest(
    train_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    cointegrated_pairs: pd.DataFrame,
    risk_free_rates,
    initial_capital: float = 100_000.0,
    entry_z: float = 1.5,
    target_probability: float = 0.70,
    memory_window: int = 60,
    max_horizon_days: int = 126,
    n_paths: int = 5000,
    ewma_lambda: float = 0.94,
    seed: int = 42,
) -> dict:
    """
    Run the complete daily OOS backtest.

    Returns
    -------
    dict with:
        trades         : one row per closed trade
        equity_curve   : daily portfolio state
        skipped_signals: signals not opened because of capital/data constraints
        pair_parameters: frozen pair table actually used
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

    train = train_prices.copy().sort_index()
    test = test_prices.copy().sort_index()
    train.index = pd.to_datetime(train.index).normalize()
    test.index = pd.to_datetime(test.index).normalize()

    if not isinstance(train.index, pd.DatetimeIndex) or not isinstance(test.index, pd.DatetimeIndex):
        raise TypeError("Price data must use DatetimeIndex.")
    if train.index.max() >= test.index.min():
        # Equality can happen only if datasets were intentionally overlapped.
        raise ValueError("Training and test periods must not overlap.")

    params = _prepare_pair_parameters(eligible_pairs, cointegrated_pairs)

    tickers = sorted(set(params["dependent"]).union(params["independent"]))
    missing_train = sorted(set(tickers) - set(train.columns))
    missing_test = sorted(set(tickers) - set(test.columns))
    if missing_train or missing_test:
        raise KeyError(
            f"Missing price columns. train={missing_train}, test={missing_test}"
        )

    full_prices = pd.concat([train[tickers], test[tickers]], axis=0).sort_index()
    vol = precompute_oos_ewma_volatility(
        train_prices=train[tickers],
        test_prices=test[tickers],
        lambda_=ewma_lambda,
    )

    # Precompute each pair's full spread series once.
    spread_series = {}
    for _, row in params.iterrows():
        spread_series[row["pair"]] = compute_log_spread(
            full_prices,
            row["dependent"],
            row["independent"],
            row["alpha"],
            row["beta"],
        )

    cash = float(initial_capital)
    open_positions: dict[str, dict] = {}
    trades = []
    equity_rows = []
    skipped = []

    test_dates = pd.DatetimeIndex(test.index).sort_values()

    for date in test_dates:
        date = pd.Timestamp(date).normalize()

        # ---------------------------------------------------------------
        # A. Manage positions that were already open before today's entry scan
        # ---------------------------------------------------------------
        for pair in list(open_positions.keys()):
            pos = open_positions[pair]
            row = params.loc[params["pair"] == pair].iloc[0]
            dep, ind = row["dependent"], row["independent"]

            dep_spot = float(test.at[date, dep])
            ind_spot = float(test.at[date, ind])
            dep_vol = float(vol.at[date, dep])
            ind_vol = float(vol.at[date, ind])
            rf = _risk_free_at(risk_free_rates, date)
            current_spread = float(spread_series[pair].loc[date])

            converged = _has_converged(
                current_spread=current_spread,
                mu=float(row["mu"]),
                entry_direction=int(pos["direction"]),
            )
            expired = date >= pd.Timestamp(pos["expiry_date"]).normalize()

            if converged or expired:
                exit_value, dep_exit_px, ind_exit_px = _position_market_value(
                    pos, date, dep_spot, ind_spot, dep_vol, ind_vol, rf
                )
                cash += exit_value

                pnl = exit_value - pos["entry_premium"]
                trade_return = pnl / pos["entry_premium"]
                exit_reason = "convergence" if converged else "expiry"

                trades.append({
                    **pos,
                    "exit_date": date,
                    "exit_reason": exit_reason,
                    "exit_spread": current_spread,
                    "exit_dependent_spot": dep_spot,
                    "exit_independent_spot": ind_spot,
                    "exit_dependent_volatility": dep_vol,
                    "exit_independent_volatility": ind_vol,
                    "exit_risk_free_rate": rf,
                    "exit_dependent_option_price": dep_exit_px,
                    "exit_independent_option_price": ind_exit_px,
                    "exit_value": exit_value,
                    "pnl": pnl,
                    "trade_return": trade_return,
                    "holding_trading_days": int(
                        test_dates.get_loc(date) - test_dates.get_loc(pos["entry_date"])
                    ),
                    "holding_calendar_days": int((date - pos["entry_date"]).days),
                })
                del open_positions[pair]

        # ---------------------------------------------------------------
        # B. Scan flat pairs for new entries
        # ---------------------------------------------------------------
        for _, row in params.iterrows():
            pair = row["pair"]
            if pair in open_positions:
                continue

            dep, ind = row["dependent"], row["independent"]
            if float(row["beta"]) <= 0:
                skipped.append({
                    "date": date, "pair": pair, "reason": "nonpositive_beta"
                })
                continue

            history = spread_series[pair].loc[:date].dropna()
            if len(history) < memory_window + 1:
                continue

            current_spread = float(history.iloc[-1])
            current_z = float(
                (current_spread - float(row["mu"])) / sqrt(float(row["variance"]))
            )

            # Cheap dislocation screen before the expensive conditional fOU MC.
            if abs(current_z) < float(entry_z):
                continue

            signal_seed = _stable_seed(seed, pair, date)
            try:
                signal, _ = calculate_convergence_signal(
                    spread_history=history,
                    mu=float(row["mu"]),
                    kappa=float(row["kappa"]),
                    sigma=float(row["sigma"]),
                    hurst=float(row["hurst"]),
                    stationary_variance=float(row["variance"]),
                    target_probability=float(target_probability),
                    memory_window=int(memory_window),
                    max_horizon_days=int(max_horizon_days),
                    n_paths=int(n_paths),
                    dt=1.0,
                    seed=signal_seed,
                )
            except Exception as exc:
                skipped.append({
                    "date": date, "pair": pair,
                    "reason": f"signal_error:{type(exc).__name__}"
                })
                continue

            horizon = signal.selected_dte_trading_days
            if horizon is None:
                continue

            direction = 1 if current_z > 0 else -1
            dep_type, ind_type = option_types_from_spread_direction(direction)

            dep_spot = float(test.at[date, dep])
            ind_spot = float(test.at[date, ind])
            dep_vol = float(vol.at[date, dep])
            ind_vol = float(vol.at[date, ind])
            rf = _risk_free_at(risk_free_rates, date)

            required_calendar_days = trading_to_calendar_days(int(horizon))
            target_expiry = date + pd.Timedelta(days=required_calendar_days)
            expiry = next_observed_date_on_or_after(target_expiry, test_dates)
            calendar_dte = int((expiry - date).days)
            T = calendar_dte / CALENDAR_DAYS_PER_YEAR

            # Synthetic ATM options.
            dep_strike = dep_spot
            ind_strike = ind_spot

            dep_option_px = black_scholes_price(
                dep_spot, dep_strike, T, rf, dep_vol, dep_type
            )
            ind_option_px = black_scholes_price(
                ind_spot, ind_strike, T, rf, ind_vol, ind_type
            )
            dep_delta = black_scholes_delta(
                dep_spot, dep_strike, T, rf, dep_vol, dep_type
            )
            ind_delta = black_scholes_delta(
                ind_spot, ind_strike, T, rf, ind_vol, ind_type
            )

            try:
                sizing = size_log_spread_option_legs(
                    beta=float(row["beta"]),
                    dependent_spot=dep_spot,
                    independent_spot=ind_spot,
                    dependent_delta=dep_delta,
                    independent_delta=ind_delta,
                )
            except Exception as exc:
                skipped.append({
                    "date": date, "pair": pair,
                    "reason": f"sizing_error:{type(exc).__name__}"
                })
                continue

            n_dep = sizing["dependent_contracts"]
            n_ind = sizing["independent_contracts"]

            premium = CONTRACT_MULTIPLIER * (
                n_dep * dep_option_px + n_ind * ind_option_px
            )

            if premium > cash:
                skipped.append({
                    "date": date,
                    "pair": pair,
                    "reason": "insufficient_cash",
                    "required_premium": float(premium),
                    "available_cash": float(cash),
                })
                continue

            cash -= premium

            open_positions[pair] = {
                "pair": pair,
                "dependent": dep,
                "independent": ind,
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "entry_date": date,
                "entry_spread": current_spread,
                "entry_z": current_z,
                "mu": float(row["mu"]),
                "direction": int(direction),
                "target_probability": float(target_probability),
                "convergence_horizon_trading_days": int(horizon),
                "probability_at_selected_horizon":
                    float(signal.probability_at_selected_dte),
                "probability_at_max_horizon":
                    float(signal.probability_at_max_horizon),
                "required_calendar_days": int(required_calendar_days),
                "expiry_date": expiry,
                "option_calendar_dte": int(calendar_dte),
                "dependent_option_type": dep_type,
                "independent_option_type": ind_type,
                "dependent_strike": dep_strike,
                "independent_strike": ind_strike,
                "entry_dependent_spot": dep_spot,
                "entry_independent_spot": ind_spot,
                "entry_dependent_volatility": dep_vol,
                "entry_independent_volatility": ind_vol,
                "entry_risk_free_rate": rf,
                "entry_dependent_delta": dep_delta,
                "entry_independent_delta": ind_delta,
                "dependent_contracts": int(n_dep),
                "independent_contracts": int(n_ind),
                "target_contract_ratio_ind_over_dep":
                    sizing["target_contract_ratio_ind_over_dep"],
                "realized_contract_ratio_ind_over_dep":
                    sizing["realized_contract_ratio_ind_over_dep"],
                "relative_hedge_error": sizing["relative_hedge_error"],
                "entry_dependent_option_price": dep_option_px,
                "entry_independent_option_price": ind_option_px,
                "entry_premium": float(premium),
            }

        # ---------------------------------------------------------------
        # C. End-of-day portfolio mark
        # ---------------------------------------------------------------
        open_value = 0.0
        for pair, pos in open_positions.items():
            row = params.loc[params["pair"] == pair].iloc[0]
            dep, ind = row["dependent"], row["independent"]
            dep_spot = float(test.at[date, dep])
            ind_spot = float(test.at[date, ind])
            dep_vol = float(vol.at[date, dep])
            ind_vol = float(vol.at[date, ind])
            rf = _risk_free_at(risk_free_rates, date)

            value, _, _ = _position_market_value(
                pos, date, dep_spot, ind_spot, dep_vol, ind_vol, rf
            )
            open_value += value

        equity_rows.append({
            "date": date,
            "cash": float(cash),
            "open_position_value": float(open_value),
            "equity": float(cash + open_value),
            "n_open_positions": int(len(open_positions)),
        })

    # -------------------------------------------------------------------
    # D. Force-close residual positions on last OOS date at model value
    # -------------------------------------------------------------------
    if len(test_dates):
        final_date = pd.Timestamp(test_dates[-1]).normalize()

        for pair in list(open_positions.keys()):
            pos = open_positions[pair]
            row = params.loc[params["pair"] == pair].iloc[0]
            dep, ind = row["dependent"], row["independent"]

            dep_spot = float(test.at[final_date, dep])
            ind_spot = float(test.at[final_date, ind])
            dep_vol = float(vol.at[final_date, dep])
            ind_vol = float(vol.at[final_date, ind])
            rf = _risk_free_at(risk_free_rates, final_date)
            current_spread = float(spread_series[pair].loc[final_date])

            exit_value, dep_exit_px, ind_exit_px = _position_market_value(
                pos, final_date, dep_spot, ind_spot, dep_vol, ind_vol, rf
            )
            cash += exit_value
            pnl = exit_value - pos["entry_premium"]

            trades.append({
                **pos,
                "exit_date": final_date,
                "exit_reason": "end_of_test",
                "exit_spread": current_spread,
                "exit_dependent_spot": dep_spot,
                "exit_independent_spot": ind_spot,
                "exit_dependent_volatility": dep_vol,
                "exit_independent_volatility": ind_vol,
                "exit_risk_free_rate": rf,
                "exit_dependent_option_price": dep_exit_px,
                "exit_independent_option_price": ind_exit_px,
                "exit_value": exit_value,
                "pnl": exit_value - pos["entry_premium"],
                "trade_return": pnl / pos["entry_premium"],
                "holding_trading_days": int(
                    test_dates.get_loc(final_date) - test_dates.get_loc(pos["entry_date"])
                ),
                "holding_calendar_days": int((final_date - pos["entry_date"]).days),
            })
            del open_positions[pair]

        # Replace final equity row after forced liquidation.
        if equity_rows:
            equity_rows[-1] = {
                "date": final_date,
                "cash": float(cash),
                "open_position_value": 0.0,
                "equity": float(cash),
                "n_open_positions": 0,
            }

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)

    return {
        "trades": trades_df,
        "equity_curve": equity_df,
        "skipped_signals": skipped_df,
        "pair_parameters": params,
        "oos_ewma_volatility": vol,
    }


def backtest_summary(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
) -> pd.Series:
    """Compact Module 07 sanity-check summary; full performance analysis belongs in Module 08."""
    if equity_curve.empty:
        raise ValueError("equity_curve is empty.")

    eq = equity_curve["equity"].astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0

    if trades.empty:
        wins = np.nan
        avg_trade_return = np.nan
        median_trade_return = np.nan
    else:
        wins = float((trades["pnl"] > 0).mean())
        avg_trade_return = float(trades["trade_return"].mean())
        median_trade_return = float(trades["trade_return"].median())

    return pd.Series({
        "initial_capital": float(initial_capital),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / initial_capital - 1.0),
        "max_drawdown": float(drawdown.min()),
        "n_trades": int(len(trades)),
        "win_rate": wins,
        "average_trade_return": avg_trade_return,
        "median_trade_return": median_trade_return,
        "max_concurrent_positions": int(equity_curve["n_open_positions"].max()),
    }, name="module_07_summary")
