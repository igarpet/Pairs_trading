from typing import Dict, Tuple

import numpy as np
import pandas as pd

from statsmodels.api import OLS, add_constant


# ============================================================
# Half-Life Estimation
# ============================================================

def estimate_half_life(spread: pd.Series) -> Tuple[float, float]:
    """
    Estimate the Ornstein-Uhlenbeck mean reversion speed (theta)
    and the corresponding half-life.

    Parameters
    ----------
    spread : pd.Series

    Returns
    -------
    theta : float
    half_life : float
    """

    spread = spread.dropna()

    lagged = spread.shift(1)
    delta = spread.diff()

    df = pd.concat([lagged, delta], axis=1).dropna()

    X = add_constant(df.iloc[:, 0])
    y = df.iloc[:, 1]

    model = OLS(y, X).fit()

    lambda_hat = model.params.iloc[1]

    # Stationarity requires lambda < 0
    if lambda_hat >= 0:
        return np.nan, np.nan

    # Exact OU relationship
    theta = -np.log(1 + lambda_hat)

    if theta <= 0:
        return np.nan, np.nan

    half_life = np.log(2) / theta

    return theta, half_life


# ============================================================
# Hurst Exponent
# ============================================================

def hurst_exponent(series: pd.Series,
                   max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent.

    H < 0.5  -> Mean reverting
    H = 0.5  -> Random walk
    H > 0.5  -> Trending
    """

    series = series.dropna()

    if len(series) < max_lag:
        return np.nan

    lags = range(2, max_lag)

    tau = []

    for lag in lags:

        diff = series.diff(lag).dropna()

        tau.append(np.sqrt(np.std(diff)))

    poly = np.polyfit(np.log(lags),
                      np.log(tau),
                      1)

    return poly[0] * 2.0


# ============================================================
# Main Analysis Function
# ============================================================

def analyze_spreads(
    spreads: Dict[Tuple[str, str], pd.Series],
    ou_window: int = 504
) -> pd.DataFrame:
    """
    Analyse all stationary spreads.

    Parameters
    ----------
    spreads : dict
        Dictionary of spreads.

    ou_window : int
        Number of most recent observations used
        for OU estimation.

    Returns
    -------
    pd.DataFrame
    """

    results = []

    for pair, spread in spreads.items():

        spread = spread.dropna()

        if len(spread) < ou_window:
            continue

        # Only use the most recent observations
        spread = spread.iloc[-ou_window:]

        theta, half_life = estimate_half_life(spread)

        if np.isnan(half_life):
            continue

        hurst = hurst_exponent(spread)

        mean = spread.mean()
        std = spread.std()

        current_z = (spread.iloc[-1] - mean) / std

        results.append({

            "pair": f"{pair[0]}-{pair[1]}",

            "dependent": pair[0],

            "independent": pair[1],

            "mean": mean,

            "std": std,

            "min spread": spread.min(),

            "max spread": spread.max(),

            "theta": theta,

            "half_life": half_life,

            "expected_convergence": 3 * half_life,

            "hurst": hurst,

            "current z": current_z

        })

    results = (
        pd.DataFrame(results)
        .sort_values("half_life")
        .reset_index(drop=True)
    )

    return results