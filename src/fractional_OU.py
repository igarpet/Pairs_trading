import numpy as np
import pandas as pd
from scipy.special import gamma

from src.convergence_signal import structural_convergence_horizon


def estimate_fractional_ou(spread, dt=1.0, min_h=0.01, max_h=0.99):
    x = pd.Series(spread).dropna().astype(float).to_numpy()
    mu = float(x.mean())

    d1 = x[2:] - 2 * x[1:-1] + x[:-2]
    d2 = x[4:] - 2 * x[2:-2] + x[:-4]
    hurst = float(np.clip(0.5 * np.log2(np.sum(d2**2) / np.sum(d1[2:] ** 2)), min_h, max_h))

    sigma = float(
        np.sqrt(
            np.mean(d1**2)
            / ((4 - 2 ** (2 * hurst)) * dt ** (2 * hurst))
        )
    )
    variance = float(np.var(x, ddof=1))
    kappa = float(
        (sigma**2 * gamma(2 * hurst + 1) / (2 * variance)) ** (1 / (2 * hurst))
    )

    return {
        "mu": mu,
        "kappa": kappa,
        "sigma": sigma,
        "hurst": hurst,
        "variance": variance,
        "drift_half_life": float(np.log(2) / kappa),
        "n_obs": len(x),
    }


def fit_fractional_ou(spreads, cointegrated_pairs, dt=1.0):
    rows = []
    for row in cointegrated_pairs.itertuples():
        params = estimate_fractional_ou(spreads[(row.dependent, row.independent)], dt=dt)
        rows.append(
            {
                "pair": row.pair,
                "dependent": row.dependent,
                "independent": row.independent,
                **params,
            }
        )
    return pd.DataFrame(rows)


def filter_antipersistent_pairs(parameters):
    finite = np.isfinite(parameters[["hurst", "kappa", "sigma", "variance"]]).all(axis=1)
    mask = (
        finite
        & (parameters.hurst > 0)
        & (parameters.hurst < 0.5)
        & (parameters.kappa > 0)
        & (parameters.kappa < 2)
        & (parameters.sigma > 0)
        & (parameters.variance > 0)
    )
    return parameters.loc[mask].reset_index(drop=True)


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
    for row in eligible_pairs.itertuples(index=False):
        result = structural_convergence_horizon(
            row.mu,
            row.kappa,
            row.sigma,
            row.hurst,
            row.variance,
            starting_z,
            target_probability,
            max_horizon_days,
            n_paths,
            dt,
            seed,
        )
        rows.append({**row._asdict(), **result})
    return pd.DataFrame(rows)


def select_top_pairs_by_structural_t70(results, top_n=40):
    return (
        results.loc[results.structural_t70.notna()]
        .sort_values(
            ["structural_t70", "structural_probability_max", "pair"],
            ascending=[True, False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )
