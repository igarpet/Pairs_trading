from __future__ import annotations
from dataclasses import dataclass, asdict
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
    def to_dict(self):
        return asdict(self)

def _clean_array(series):
    x = np.asarray(pd.Series(series), dtype=float)
    return x[np.isfinite(x)]

def estimate_hurst_cof(series, min_h=0.01, max_h=0.99):
    x = _clean_array(series)
    if len(x) < 50:
        raise ValueError('At least 50 observations are required to estimate H.')
    d1 = x[2:] - 2*x[1:-1] + x[:-2]
    d2 = x[4:] - 2*x[2:-2] + x[:-4]
    d1_common = d1[2:]
    denom = np.sum(d1_common**2)
    numer = np.sum(d2**2)
    if denom <= 0 or numer <= 0:
        raise ValueError('Second-order variation is degenerate.')
    h = 0.5*np.log2(numer/denom)
    if not np.isfinite(h):
        raise ValueError('Hurst estimate is not finite.')
    return float(np.clip(h, min_h, max_h))

def estimate_sigma_second_variation(series, hurst, dt=1.0):
    if not (0 < hurst < 1):
        raise ValueError('hurst must lie in (0,1).')
    x = _clean_array(series)
    d2 = x[2:] - 2*x[1:-1] + x[:-2]
    denom = (4 - 2**(2*hurst))*(dt**(2*hurst))
    sigma2 = np.mean(d2**2)/denom
    if sigma2 <= 0 or not np.isfinite(sigma2):
        raise ValueError('Estimated sigma^2 is invalid.')
    return float(np.sqrt(sigma2))

def estimate_kappa_stationary_variance(series, hurst, sigma):
    x = _clean_array(series)
    variance = float(np.var(x, ddof=1))
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError('Sample variance is invalid.')
    ratio = (sigma**2 * gamma(2*hurst + 1))/(2*variance)
    kappa = ratio**(1/(2*hurst))
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError('Estimated kappa is invalid.')
    return float(kappa), variance

def estimate_fractional_ou(spread, dt=1.0, min_h=0.01, max_h=0.99):
    x = pd.Series(spread).dropna().astype(float)
    if len(x) < 100:
        raise ValueError('At least 100 observations are recommended for fOU estimation.')
    mu = float(x.mean())
    hurst = estimate_hurst_cof(x, min_h=min_h, max_h=max_h)
    sigma = estimate_sigma_second_variation(x, hurst=hurst, dt=dt)
    kappa, variance = estimate_kappa_stationary_variance(x, hurst=hurst, sigma=sigma)
    return FOUParameters(mu, kappa, sigma, hurst, variance, float(np.log(2)/kappa), len(x))

def standardized_spread(spread, mu, variance):
    if variance <= 0:
        raise ValueError('variance must be positive.')
    return ((pd.Series(spread).astype(float)-mu)/np.sqrt(variance)).rename('fou_z')

def fit_cointegrated_pairs_fractional_ou(spreads, cointegrated_pairs, dependent_col='dependent', independent_col='independent', pair_col='pair', dt=1.0):
    rows = []
    for _, row in cointegrated_pairs.iterrows():
        dep, indep = row[dependent_col], row[independent_col]
        pair = row[pair_col] if pair_col in row.index else f'{dep}-{indep}'
        key = (dep, indep)
        if key not in spreads:
            continue
        try:
            params = estimate_fractional_ou(spreads[key], dt=dt)
        except Exception as exc:
            print(f'Skipped {pair}: {exc}')
            continue
        rows.append({'pair': pair, 'dependent': dep, 'independent': indep, **params.to_dict()})
    return pd.DataFrame(rows).reset_index(drop=True)
