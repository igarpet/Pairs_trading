"""
option_pricing.py

Module 06 — Option pricing and expiry selection.

Purpose
-------
Translate Module 04 statistical convergence horizons and Module 05 EWMA
volatility estimates into option contracts and theoretical Black-Scholes prices.

Design
------
- The Module 04 convergence horizon is NOT itself an option expiry.
- It is converted from trading days to calendar days.
- The actual option expiry is selected as the first listed expiry on or after
  the required convergence horizon.
- The option leg direction follows the pair-trading direction.
- Each leg is priced separately with Black-Scholes using the equity-specific
  EWMA annualized volatility from Module 05.
- No alternate pricing models or benchmark branches are retained.

Strategy mapping
----------------
Spread:
    X_t = log(P_dep) - alpha - beta * log(P_indep)

If Z_t > 0:
    spread is above equilibrium -> short spread:
        dependent: PUT
        independent: CALL

If Z_t < 0:
    spread is below equilibrium -> long spread:
        dependent: CALL
        independent: PUT

This module uses long options only, so maximum loss is premium paid.

Strike rule
-----------
ATM strike by default:
    strike = closest listed strike to current spot

This avoids adding a separate moneyness optimization branch.

Pricing
-------
European Black-Scholes:

Call:
    C = S N(d1) - K exp(-rT) N(d2)

Put:
    P = K exp(-rT) N(-d2) - S N(-d1)

where:
    d1 = [ln(S/K) + (r + 0.5 sigma^2)T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

Dividends are omitted here. If the later data source contains a reliable
dividend yield field, q can be added directly without changing architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, log, sqrt
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class OptionLeg:
    ticker: str
    option_type: str
    spot: float
    strike: float
    volatility: float
    expiry_date: pd.Timestamp
    calendar_dte: int
    year_fraction: float
    risk_free_rate: float
    theoretical_price: float

    def to_dict(self) -> dict:
        return asdict(self)


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
) -> float:
    """European Black-Scholes option price without dividends."""
    S = float(spot)
    K = float(strike)
    T = float(time_to_expiry_years)
    r = float(risk_free_rate)
    sigma = float(volatility)

    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be strictly positive.")
    if T <= 0:
        raise ValueError("time_to_expiry_years must be strictly positive.")
    if sigma <= 0:
        raise ValueError("volatility must be strictly positive.")

    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    d1 = (
        log(S / K)
        + (r + 0.5 * sigma ** 2) * T
    ) / (sigma * sqrt(T))

    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        return float(
            S * norm.cdf(d1)
            - K * exp(-r * T) * norm.cdf(d2)
        )

    return float(
        K * exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )


def trading_to_calendar_days(
    trading_days: int,
    trading_days_per_year: int = 252,
    calendar_days_per_year: int = 365,
) -> int:
    """Convert trading-day horizon to an approximate calendar-day horizon."""
    if trading_days < 1:
        raise ValueError("trading_days must be >= 1.")

    return int(
        np.ceil(
            trading_days
            * calendar_days_per_year
            / trading_days_per_year
        )
    )


def select_expiry_on_or_after(
    evaluation_date,
    required_calendar_days: int,
    available_expiries: Iterable,
) -> pd.Timestamp:
    """
    Select the first listed expiry on or after the required convergence horizon.
    """
    eval_date = pd.Timestamp(evaluation_date).normalize()

    expiries = pd.DatetimeIndex(
        pd.to_datetime(list(available_expiries))
    ).normalize().sort_values()

    if len(expiries) == 0:
        raise ValueError("No available expiries supplied.")

    target_date = (
        eval_date
        + pd.Timedelta(days=int(required_calendar_days))
    )

    eligible = expiries[expiries >= target_date]

    if len(eligible) == 0:
        raise ValueError(
            "No listed expiry reaches the required convergence horizon."
        )

    return pd.Timestamp(eligible[0])


def nearest_atm_strike(
    spot: float,
    available_strikes: Iterable[float],
) -> float:
    """Select listed strike closest to current spot."""
    strikes = np.asarray(
        list(available_strikes),
        dtype=float,
    )

    if len(strikes) == 0:
        raise ValueError("No available strikes supplied.")

    if np.any(strikes <= 0):
        raise ValueError("All strikes must be strictly positive.")

    return float(
        strikes[
            np.argmin(np.abs(strikes - float(spot)))
        ]
    )


def option_types_from_spread_direction(
    direction: int,
) -> tuple[str, str]:
    """
    Map Module 04 spread direction into option directions.

    direction = +1:
        spread above equilibrium -> short spread
        dependent PUT, independent CALL

    direction = -1:
        spread below equilibrium -> long spread
        dependent CALL, independent PUT
    """
    if direction == 1:
        return "put", "call"

    if direction == -1:
        return "call", "put"

    raise ValueError("direction must be +1 or -1.")


def price_option_leg(
    ticker: str,
    option_type: str,
    spot: float,
    volatility: float,
    evaluation_date,
    expiry_date,
    strike: float,
    risk_free_rate: float,
) -> OptionLeg:
    eval_date = pd.Timestamp(evaluation_date).normalize()
    expiry = pd.Timestamp(expiry_date).normalize()

    calendar_dte = int(
        (expiry - eval_date).days
    )

    if calendar_dte <= 0:
        raise ValueError("expiry_date must be after evaluation_date.")

    T = calendar_dte / DAYS_PER_YEAR

    price = black_scholes_price(
        spot=spot,
        strike=strike,
        time_to_expiry_years=T,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type,
    )

    return OptionLeg(
        ticker=ticker,
        option_type=option_type,
        spot=float(spot),
        strike=float(strike),
        volatility=float(volatility),
        expiry_date=expiry,
        calendar_dte=calendar_dte,
        year_fraction=float(T),
        risk_free_rate=float(risk_free_rate),
        theoretical_price=float(price),
    )


def build_pair_option_trade(
    dependent: str,
    independent: str,
    direction: int,
    dependent_spot: float,
    independent_spot: float,
    dependent_volatility: float,
    independent_volatility: float,
    convergence_horizon_trading_days: int,
    evaluation_date,
    available_expiries: Iterable,
    dependent_strikes: Iterable[float],
    independent_strikes: Iterable[float],
    risk_free_rate: float,
) -> dict:
    """
    Build and price the two long-option legs implementing the pair signal.
    """
    required_calendar_days = trading_to_calendar_days(
        convergence_horizon_trading_days
    )

    expiry = select_expiry_on_or_after(
        evaluation_date=evaluation_date,
        required_calendar_days=required_calendar_days,
        available_expiries=available_expiries,
    )

    dep_type, indep_type = option_types_from_spread_direction(
        direction
    )

    dep_strike = nearest_atm_strike(
        spot=dependent_spot,
        available_strikes=dependent_strikes,
    )

    indep_strike = nearest_atm_strike(
        spot=independent_spot,
        available_strikes=independent_strikes,
    )

    dep_leg = price_option_leg(
        ticker=dependent,
        option_type=dep_type,
        spot=dependent_spot,
        volatility=dependent_volatility,
        evaluation_date=evaluation_date,
        expiry_date=expiry,
        strike=dep_strike,
        risk_free_rate=risk_free_rate,
    )

    indep_leg = price_option_leg(
        ticker=independent,
        option_type=indep_type,
        spot=independent_spot,
        volatility=independent_volatility,
        evaluation_date=evaluation_date,
        expiry_date=expiry,
        strike=indep_strike,
        risk_free_rate=risk_free_rate,
    )

    total_premium = (
        dep_leg.theoretical_price
        + indep_leg.theoretical_price
    )

    return {
        "dependent": dependent,
        "independent": independent,
        "direction": int(direction),
        "required_convergence_calendar_days":
            int(required_calendar_days),
        "selected_expiry": expiry,
        "selected_option_calendar_dte":
            dep_leg.calendar_dte,
        "dependent_option_type":
            dep_leg.option_type,
        "independent_option_type":
            indep_leg.option_type,
        "dependent_spot":
            dep_leg.spot,
        "independent_spot":
            indep_leg.spot,
        "dependent_strike":
            dep_leg.strike,
        "independent_strike":
            indep_leg.strike,
        "dependent_volatility":
            dep_leg.volatility,
        "independent_volatility":
            indep_leg.volatility,
        "dependent_option_price":
            dep_leg.theoretical_price,
        "independent_option_price":
            indep_leg.theoretical_price,
        "total_premium_per_share":
            float(total_premium),
        "total_premium_100x":
            float(100.0 * total_premium),
    }
