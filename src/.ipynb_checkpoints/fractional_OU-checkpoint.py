"""
fractional_OU.py

Module 03 — Fractional Ornstein-Uhlenbeck estimation.

Model
-----
    dX_t = kappa * (mu - X_t) dt + sigma dB_t^H

where B^H is fractional Brownian motion.

Estimation
----------
Two-stage procedure:
1) Estimate H from a second-order change-of-frequency statistic.
2) Conditional on H, estimate sigma from second-order variation,
   mu from the sample mean, and kappa from the stationary variance identity

    Var(X) = sigma^2 * Gamma(2H + 1) / (2 * kappa^(2H)).

Use formation/training spread observations only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
from scipy.special import gamma


@dataclass(frozen=True)
class FOUParameters:
    mu: float
    kappa: float
    sigma: float
    hurst: float
    variance: float
    drift_half_life: float
    n_obs: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _clean_array(series: pd.Series) -> np.ndarray:
    x = np.asarray(pd.Series(series), dtype=float)
    return x[np.isfinite(x)]


def estimate_hurst_cof(
    series: pd.Series,
    min_h: float = 0.01,
    max_h: float = 0.99,
) -> float:
    """
    Estimate H using second-order change-of-frequency variation.

    D1_t = X_t - 2X_{t-1} + X_{t-2}
    D2_t = X_t - 2X_{t-2} + X_{t-4}

    For fractional scaling:
        sum(D2^2) / sum(D1^2) ~ 2^(2H)
    hence
        H_hat = 0.5 * log2(COF).
    """
    x = _clean_array(series)

    if len(x) < 50:
        raise ValueError("At least 50 observations are required to estimate H.")

    d1 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    d2 = x[4:] - 2.0 * x[2:-2] + x[:-4]
    d1_common = d1[2:]

    denom = np.sum(d1_common ** 2)
    numer = np.sum(d2 ** 2)

    if denom <= 0 or numer <= 0:
        raise ValueError("Second-order variation is degenerate.")

    cof = numer / denom
    h = 0.5 * np.log2(cof)

    if not np.isfinite(h):
        raise ValueError("Hurst estimate is not finite.")

    return float(np.clip(h, min_h, max_h))


def estimate_sigma_second_variation(
    series: pd.Series,
    hurst: float,
    dt: float = 1.0,
) -> float:
    """
    Estimate sigma from second-order variation.

    Var(Delta^2 B^H) = (4 - 2^(2H)) * dt^(2H).
    """
    if not (0 < hurst < 1):
        raise ValueError("hurst must lie in (0, 1).")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    x = _clean_array(series)
    if len(x) < 20:
        raise ValueError("Not enough observations to estimate sigma.")

    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    denom = (4.0 - 2.0 ** (2.0 * hurst)) * (dt ** (2.0 * hurst))

    if denom <= 0:
        raise ValueError("Invalid denominator in sigma estimator.")

    sigma2 = np.mean(d2 ** 2) / denom

    if sigma2 <= 0 or not np.isfinite(sigma2):
        raise ValueError("Estimated sigma^2 is invalid.")

    return float(np.sqrt(sigma2))


def estimate_kappa_stationary_variance(
    series: pd.Series,
    hurst: float,
    sigma: float,
) -> tuple[float, float]:
    """
    Estimate kappa from
        Var(X) = sigma^2 Gamma(2H+1) / [2 kappa^(2H)].

    Returns
    -------
    kappa, sample_variance
    """
    if not (0 < hurst < 1):
        raise ValueError("hurst must lie in (0, 1).")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    x = _clean_array(series)
    variance = float(np.var(x, ddof=1))

    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Sample variance is invalid.")

    numerator = sigma ** 2 * gamma(2.0 * hurst + 1.0)
    ratio = numerator / (2.0 * variance)

    if ratio <= 0 or not np.isfinite(ratio):
        raise ValueError("Invalid moment ratio for kappa.")

    kappa = ratio ** (1.0 / (2.0 * hurst))

    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("Estimated kappa is invalid.")

    return float(kappa), variance


def estimate_fractional_ou(
    spread: pd.Series,
    dt: float = 1.0,
    min_h: float = 0.01,
    max_h: float = 0.99,
) -> FOUParameters:
    """Estimate all fOU parameters with the two-stage procedure."""
    x = pd.Series(spread).dropna().astype(float)

    if len(x) < 100:
        raise ValueError("At least 100 observations are recommended for fOU estimation.")

    mu = float(x.mean())
    hurst = estimate_hurst_cof(x, min_h=min_h, max_h=max_h)
    sigma = estimate_sigma_second_variation(x, hurst=hurst, dt=dt)
    kappa, variance = estimate_kappa_stationary_variance(x, hurst=hurst, sigma=sigma)
    drift_half_life = float(np.log(2.0) / kappa)

    return FOUParameters(
        mu=mu,
        kappa=kappa,
        sigma=sigma,
        hurst=hurst,
        variance=variance,
        drift_half_life=drift_half_life,
        n_obs=len(x),
    )


def standardized_spread(
    spread: pd.Series,
    mu: float,
    variance: float,
) -> pd.Series:
    if variance <= 0:
        raise ValueError("variance must be positive.")

    return (
        (pd.Series(spread).astype(float) - mu) / np.sqrt(variance)
    ).rename("fou_z")


def fit_pairs_fractional_ou(
    spreads: dict,
    eligible_pairs: pd.DataFrame,
    dependent_col: str = "dependent",
    independent_col: str = "independent",
    pair_col: str = "pair",
    dt: float = 1.0,
) -> pd.DataFrame:
    """Estimate fOU parameters for all eligible pairs."""
    rows = []

    for _, row in eligible_pairs.iterrows():
        dep = row[dependent_col]
        indep = row[independent_col]
        pair = row[pair_col]
        key = (dep, indep)

        if key not in spreads:
            continue

        params = estimate_fractional_ou(spreads[key], dt=dt)

        rows.append({
            "pair": pair,
            "dependent": dep,
            "independent": indep,
            **params.to_dict(),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["hurst", "drift_half_life"])
        .reset_index(drop=True)
    )
