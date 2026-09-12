from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.special import gamma
from src.convergence_signal import structural_convergence_horizon


@dataclass(frozen=True)
class FOUParameters:
    mu: float
    kappa: float
    sigma: float
    hurst: float
    variance: float
    drift_half_life: float
    n_obs: int

    def to_dict(self):
        return asdict(self)


def _clean_array(series):
    x = np.asarray(pd.Series(series), dtype=float)
    return x[np.isfinite(x)]


def estimate_hurst_cof(series, min_h=0.01, max_h=0.99):
    x = _clean_array(series)
    d1 = x[2:] - 2 * x[1:-1] + x[:-2]
    d2 = x[4:] - 2 * x[2:-2] + x[:-4]
    h = 0.5 * np.log2(np.sum(d2**2) / np.sum(d1[2:] ** 2))
    return float(np.clip(h, min_h, max_h))


def estimate_sigma_second_variation(series, hurst, dt=1.0):
    x = _clean_array(series)
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    denom = (4 - 2 ** (2 * hurst)) * dt ** (2 * hurst)
    return float(np.sqrt(np.mean(d2**2) / denom))


def estimate_kappa_stationary_variance(series, hurst, sigma):
    x = _clean_array(series)
    variance = float(np.var(x, ddof=1))
    ratio = sigma**2 * gamma(2 * hurst + 1) / (2 * variance)
    return float(ratio ** (1 / (2 * hurst))), variance


def estimate_fractional_ou(spread, dt=1.0, min_h=0.01, max_h=0.99):
    x = pd.Series(spread).dropna().astype(float)
    mu = float(x.mean())
    hurst = estimate_hurst_cof(x, min_h=min_h, max_h=max_h)
    sigma = estimate_sigma_second_variation(x, hurst=hurst, dt=dt)
    kappa, variance = estimate_kappa_stationary_variance(x, hurst=hurst, sigma=sigma)
    return FOUParameters(mu, kappa, sigma, hurst, variance, float(np.log(2) / kappa), len(x))


def fit_cointegrated_pairs_fractional_ou(
    spreads,
    cointegrated_pairs,
    dependent_col="dependent",
    independent_col="independent",
    pair_col="pair",
    dt=1.0,
    return_audit=False,
):
    rows, audit = [], []
    for _, row in cointegrated_pairs.iterrows():
        dep, indep = row[dependent_col], row[independent_col]
        pair = row[pair_col] if pair_col in row.index else f"{dep}-{indep}"
        key = (dep, indep)

        if key not in spreads:
            audit.append({"pair": pair, "status": "missing_spread", "error": "Spread key absent"})
            continue

        try:
            params = estimate_fractional_ou(spreads[key], dt=dt)
        except Exception as exc:
            audit.append({"pair": pair, "status": "fit_error", "error": str(exc)})
            continue

        audit.append(
            {
                "pair": pair,
                "status": "fitted",
                "error": "",
                "hurst_at_clip_boundary": bool(params.hurst <= 0.01 or params.hurst >= 0.99),
                "daily_euler_stable": bool(0 < params.kappa < 2),
            }
        )
        rows.append({"pair": pair, "dependent": dep, "independent": indep, **params.to_dict()})

    result = pd.DataFrame(rows).reset_index(drop=True)
    return (result, pd.DataFrame(audit)) if return_audit else result


def filter_antipersistent_pairs(fou_parameters):
    cols = ["hurst", "kappa", "sigma", "variance", "drift_half_life"]
    mask = (
        (fou_parameters["hurst"] > 0)
        & (fou_parameters["hurst"] < 0.5)
        & (fou_parameters["kappa"] > 0)
        & (fou_parameters["kappa"] < 2)
        & (fou_parameters["sigma"] > 0)
        & (fou_parameters["variance"] > 0)
        & np.isfinite(fou_parameters[cols]).all(axis=1)
    )
    return fou_parameters.loc[mask].copy().reset_index(drop=True)


def compute_structural_t70(
    eligible_pairs,
    starting_z=1.5,
    target_probability=0.70,
    max_horizon_days=252,
    n_paths=5000,
    dt=1.0,
    seed=42,
):
    rows = []
    for _, row in eligible_pairs.iterrows():
        try:
            result = structural_convergence_horizon(
                mu=row["mu"],
                kappa=row["kappa"],
                sigma=row["sigma"],
                hurst=row["hurst"],
                stationary_variance=row["variance"],
                starting_z=starting_z,
                target_probability=target_probability,
                max_horizon_days=max_horizon_days,
                n_paths=n_paths,
                dt=dt,
                seed=seed,
            )
            rows.append({**row.to_dict(), **result})
        except Exception as exc:
            rows.append(
                {
                    **row.to_dict(),
                    "structural_t70": np.nan,
                    "structural_probability_max": np.nan,
                    "eligibility_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def select_top_pairs_by_structural_t70(structural_results, top_n=40):
    valid = structural_results[structural_results["structural_t70"].notna()].copy()
    valid = valid.sort_values(
        ["structural_t70", "structural_probability_max", "pair"],
        ascending=[True, False, True],
        kind="stable",
    )
    return valid.head(top_n).reset_index(drop=True)
