from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

@dataclass(frozen=True)
class ConvergenceSignal:
    current_spread: float
    current_z: float
    direction: int
    target_probability: float
    selected_dte_trading_days: Optional[int]
    probability_at_selected_dte: float
    probability_at_max_horizon: float
    statistical_signal: bool
    def to_dict(self):
        return asdict(self)

def fgn_autocovariance(lag, hurst, dt=1.0):
    k = np.asarray(lag, dtype=float)
    return 0.5*(dt**(2*hurst))*(np.abs(k+1)**(2*hurst)-2*np.abs(k)**(2*hurst)+np.abs(k-1)**(2*hurst))

def fgn_covariance_matrix(n, hurst, dt=1.0):
    idx = np.arange(n)
    return fgn_autocovariance(np.abs(idx[:,None]-idx[None,:]), hurst, dt)

def simulate_unconditional_fou_paths(current_spread, mu, kappa, sigma, hurst, horizon_days, n_paths=5000, dt=1.0, seed=42):
    cov = fgn_covariance_matrix(horizon_days, hurst, dt)
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(np.zeros(horizon_days), cov, size=n_paths, method='cholesky')
    paths = np.empty((n_paths, horizon_days+1))
    paths[:,0] = current_spread
    for t in range(horizon_days):
        x = paths[:,t]
        paths[:,t+1] = x + kappa*(mu-x)*dt + sigma*noise[:,t]
    return paths

def infer_fgn_innovations(spread_history, mu, kappa, sigma, dt=1.0):
    x = np.asarray(pd.Series(spread_history).dropna(), dtype=float)
    dx = x[1:]-x[:-1]
    drift = kappa*(mu-x[:-1])*dt
    return (dx-drift)/sigma

def conditional_future_fgn(past_innovations, future_steps, hurst, dt=1.0, ridge=1e-10):
    past = np.asarray(past_innovations, dtype=float)
    m, n = len(past), int(future_steps)
    full = fgn_covariance_matrix(m+n, hurst, dt)
    cpp, cpf = full[:m,:m].copy(), full[:m,m:]
    cfp, cff = full[m:,:m], full[m:,m:].copy()
    cpp.flat[::m+1] += ridge
    factor = cho_factor(cpp, lower=True, check_finite=False)
    mean = cfp @ cho_solve(factor, past, check_finite=False)
    cov = cff - cfp @ cho_solve(factor, cpf, check_finite=False)
    cov = 0.5*(cov+cov.T)
    eig_min = np.min(np.linalg.eigvalsh(cov))
    if eig_min < 0:
        cov += (abs(eig_min)+ridge)*np.eye(n)
    return mean, cov

def simulate_conditional_fou_paths(current_spread, past_innovations, mu, kappa, sigma, hurst, horizon_days, n_paths=5000, dt=1.0, seed=42):
    mean, cov = conditional_future_fgn(past_innovations, horizon_days, hurst, dt)
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(mean, cov, size=n_paths, method='cholesky')
    paths = np.empty((n_paths, horizon_days+1))
    paths[:,0] = current_spread
    for t in range(horizon_days):
        x = paths[:,t]
        paths[:,t+1] = x + kappa*(mu-x)*dt + sigma*noise[:,t]
    return paths

def first_passage_days(paths, mu):
    paths = np.asarray(paths, dtype=float)
    x0, future = paths[:,0], paths[:,1:]
    side = np.sign(x0-mu)
    crossed = np.where(side[:,None] > 0, future <= mu, future >= mu)
    any_cross = crossed.any(axis=1)
    first = np.argmax(crossed, axis=1)+1
    out = np.full(paths.shape[0], np.nan)
    out[any_cross] = first[any_cross]
    return out

def convergence_probability_curve(first_passage, horizon_days):
    fp = np.asarray(first_passage, dtype=float)
    vals = [np.mean(np.isfinite(fp) & (fp <= d)) for d in range(1, horizon_days+1)]
    return pd.Series(vals, index=pd.RangeIndex(1, horizon_days+1, name='dte'), name='convergence_probability')

def choose_dte_from_probability(probability_curve, target_probability=0.70):
    reached = probability_curve[probability_curve >= target_probability]
    if reached.empty:
        return None, float(probability_curve.iloc[-1])
    return int(reached.index[0]), float(reached.iloc[0])

def structural_convergence_horizon(mu, kappa, sigma, hurst, stationary_variance, starting_z=1.5, target_probability=0.70, max_horizon_days=252, n_paths=5000, dt=1.0, seed=42):
    if not (0 < hurst < 0.5):
        raise ValueError('Structural eligibility requires 0 < H < 0.5.')
    x0 = mu + starting_z*np.sqrt(stationary_variance)
    paths = simulate_unconditional_fou_paths(x0, mu, kappa, sigma, hurst, max_horizon_days, n_paths, dt, seed)
    fp = first_passage_days(paths, mu)
    curve = convergence_probability_curve(fp, max_horizon_days)
    t70, p = choose_dte_from_probability(curve, target_probability)
    return {
        'structural_starting_z': float(starting_z),
        'structural_target_probability': float(target_probability),
        'structural_t70': t70,
        'structural_probability_at_t70': float(p),
        'structural_probability_max': float(curve.iloc[-1]),
    }

def calculate_convergence_signal(spread_history, mu, kappa, sigma, hurst, stationary_variance, target_probability=0.70, entry_z = 1.5, memory_window=60, max_horizon_days=126, n_paths=5000, dt=1.0, seed=42):
    history = pd.Series(spread_history).dropna().astype(float)
    if len(history) < memory_window+1:
        raise ValueError(f'Need at least {memory_window+1} observations.')
    current = float(history.iloc[-1])
    current_z = float((current-mu)/np.sqrt(stationary_variance))
    direction = int(np.sign(current-mu))
    if direction == 0:
        sig = ConvergenceSignal(current, current_z, 0, target_probability, None, 0.0, 1.0, False)
        return sig, pd.Series(dtype=float)
    innovations = infer_fgn_innovations(history.iloc[-(memory_window+1):], mu, kappa, sigma, dt)
    paths = simulate_conditional_fou_paths(current, innovations, mu, kappa, sigma, hurst, max_horizon_days, n_paths, dt, seed)
    fp = first_passage_days(paths, mu)
    curve = convergence_probability_curve(fp, max_horizon_days)
    dte, psel = choose_dte_from_probability(curve, target_probability)
    statistical_signal = (abs(current_z) >= entry_z and dte is not None)
    sig = ConvergenceSignal(current, current_z, direction, target_probability, dte, psel, float(curve.iloc[-1]), statistical_signal)
    return sig, curve

def trading_days_to_calendar_days(trading_days, trading_days_per_year=252, calendar_days_per_year=365):
    return int(np.ceil(trading_days*calendar_days_per_year/trading_days_per_year))
