"""
volatility_model.py

Module 05 — EWMA volatility estimation for option pricing.

Final model
-----------
For each equity leg, daily conditional variance is estimated recursively as:

    sigma_t^2 = lambda * sigma_{t-1}^2
                + (1-lambda) * r_{t-1}^2

with lambda = 0.94.

The annualized volatility passed to Module 06 is:

    sigma_ann = sqrt(252 * sigma_t^2)

Under the standard EWMA forecasting assumption, the conditional variance
forecast is flat across future horizons:

    E_t[sigma_{t+h}^2] = sigma_t^2

Therefore Module 05 does not alter the convergence horizon from Module 04.
It only supplies one current annualized volatility estimate for each equity leg.

Important
---------
- Fit recursively using only prices available up to the evaluation date.
- Volatility is estimated separately for dependent and independent equities.
- No GARCH benchmark or alternative volatility model is retained.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_LAMBDA = 0.94


@dataclass(frozen=True)
class EWMAResult:
    lambda_: float
    daily_variance: float
    daily_volatility: float
    annualized_volatility: float
    n_obs: int

    def to_dict(self) -> dict:
        return asdict(self)


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from a strictly positive price series."""
    p = pd.Series(prices).dropna().astype(float)

    if len(p) < 3:
        raise ValueError("At least 3 price observations are required.")

    if (p <= 0).any():
        raise ValueError("Prices must be strictly positive.")

    return np.log(p).diff().dropna().rename("log_return")


def ewma_variance_series(
    returns: pd.Series,
    lambda_: float = DEFAULT_LAMBDA,
    initial_variance: float | None = None,
) -> pd.Series:
    """
    Compute recursive EWMA conditional variance.

    sigma_t^2 =
        lambda * sigma_{t-1}^2
        + (1-lambda) * r_{t-1}^2
    """
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must lie strictly between 0 and 1.")

    r = pd.Series(returns).dropna().astype(float)

    if len(r) < 2:
        raise ValueError("At least 2 returns are required.")

    if initial_variance is None:
        initial_variance = float(np.var(r, ddof=1))

    initial_variance = max(float(initial_variance), 1e-12)

    variance = np.empty(len(r), dtype=float)
    variance[0] = initial_variance

    values = r.to_numpy(dtype=float)

    for t in range(1, len(r)):
        variance[t] = (
            lambda_ * variance[t - 1]
            + (1.0 - lambda_) * values[t - 1] ** 2
        )

    return pd.Series(
        variance,
        index=r.index,
        name="ewma_variance",
    )


def fit_ewma(
    returns: pd.Series,
    lambda_: float = DEFAULT_LAMBDA,
    min_obs: int = 60,
) -> EWMAResult:
    """
    Estimate current EWMA volatility from historical daily returns.
    """
    r = pd.Series(returns).dropna().astype(float)

    if len(r) < min_obs:
        raise ValueError(
            f"Need at least {min_obs} daily returns for EWMA volatility."
        )

    var_series = ewma_variance_series(
        returns=r,
        lambda_=lambda_,
    )

    daily_variance = float(var_series.iloc[-1])
    daily_volatility = float(np.sqrt(daily_variance))
    annualized_volatility = float(
        np.sqrt(TRADING_DAYS_PER_YEAR * daily_variance)
    )

    return EWMAResult(
        lambda_=float(lambda_),
        daily_variance=daily_variance,
        daily_volatility=daily_volatility,
        annualized_volatility=annualized_volatility,
        n_obs=len(r),
    )


def fit_and_forecast_volatility(
    prices: pd.Series,
    horizon_days: int,
    lambda_: float = DEFAULT_LAMBDA,
    min_obs: int = 60,
) -> dict:
    """
    Estimate EWMA volatility for one equity.

    The horizon is retained for interface consistency with Module 04/06.
    Under EWMA, the expected variance forecast is flat across the horizon.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")

    returns = log_returns(prices)

    result = fit_ewma(
        returns=returns,
        lambda_=lambda_,
        min_obs=min_obs,
    )

    return {
        **result.to_dict(),
        "horizon_days": int(horizon_days),
    }


def forecast_pair_leg_volatilities(
    prices: pd.DataFrame,
    dependent: str,
    independent: str,
    horizon_days: int,
    evaluation_date=None,
    lambda_: float = DEFAULT_LAMBDA,
    min_obs: int = 60,
) -> dict:
    """
    Estimate current EWMA volatility separately for both equity legs.

    Parameters
    ----------
    prices:
        Wide adjusted-close DataFrame indexed by date.

    dependent, independent:
        Pair tickers.

    horizon_days:
        Module 04 convergence horizon. Retained for downstream consistency.

    evaluation_date:
        Optional cutoff date. When supplied, only information available through
        that date is used. This prevents look-ahead bias in the backtest.
    """
    if dependent not in prices.columns:
        raise KeyError(f"{dependent} not found in price columns.")

    if independent not in prices.columns:
        raise KeyError(f"{independent} not found in price columns.")

    px = prices.sort_index()

    if evaluation_date is not None:
        px = px.loc[:evaluation_date]

    dep_result = fit_and_forecast_volatility(
        prices=px[dependent],
        horizon_days=horizon_days,
        lambda_=lambda_,
        min_obs=min_obs,
    )

    indep_result = fit_and_forecast_volatility(
        prices=px[independent],
        horizon_days=horizon_days,
        lambda_=lambda_,
        min_obs=min_obs,
    )

    return {
        "dependent": dependent,
        "independent": independent,
        "horizon_days": int(horizon_days),
        "lambda": float(lambda_),
        "dependent_annualized_volatility":
            dep_result["annualized_volatility"],
        "independent_annualized_volatility":
            indep_result["annualized_volatility"],
        "dependent_daily_variance":
            dep_result["daily_variance"],
        "independent_daily_variance":
            indep_result["daily_variance"],
        "dependent_n_obs":
            dep_result["n_obs"],
        "independent_n_obs":
            indep_result["n_obs"],
    }
