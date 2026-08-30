"""
convergence_signal.py

Module 04 — fOU convergence-probability signal and probability-implied DTE.

For each date:
1. infer recent fractional-Gaussian-noise innovations from the fitted fOU;
2. condition future fGn on that recent history;
3. simulate future fOU spread paths;
4. estimate P(first passage to equilibrium by T);
5. choose the smallest T for which the target probability is met.

The default target probability is 70%.

Because fOU is non-Markovian, this implementation conditions on a finite
recent innovation history rather than only on the current spread.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

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

    def to_dict(self) -> Dict:
        return asdict(self)


def fgn_autocovariance(
    lag: np.ndarray | int,
    hurst: float,
    dt: float = 1.0,
) -> np.ndarray:
    k = np.asarray(lag, dtype=float)

    return (
        0.5
        * (dt ** (2.0 * hurst))
        * (
            np.abs(k + 1.0) ** (2.0 * hurst)
            - 2.0 * np.abs(k) ** (2.0 * hurst)
            + np.abs(k - 1.0) ** (2.0 * hurst)
        )
    )


def fgn_covariance_matrix(
    n: int,
    hurst: float,
    dt: float = 1.0,
) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1.")

    idx = np.arange(n)
    lags = np.abs(idx[:, None] - idx[None, :])
    return fgn_autocovariance(lags, hurst=hurst, dt=dt)


def infer_fgn_innovations(
    spread_history: pd.Series,
    mu: float,
    kappa: float,
    sigma: float,
    dt: float = 1.0,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    x = np.asarray(pd.Series(spread_history).dropna(), dtype=float)

    if len(x) < 2:
        raise ValueError("At least two spread observations are required.")

    dx = x[1:] - x[:-1]
    drift = kappa * (mu - x[:-1]) * dt
    return (dx - drift) / sigma


def conditional_future_fgn(
    past_innovations: np.ndarray,
    future_steps: int,
    hurst: float,
    dt: float = 1.0,
    ridge: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    past = np.asarray(past_innovations, dtype=float)
    m = len(past)
    n = int(future_steps)

    if m < 1:
        raise ValueError("At least one past innovation is required.")
    if n < 1:
        raise ValueError("future_steps must be >= 1.")

    full_cov = fgn_covariance_matrix(m + n, hurst=hurst, dt=dt)

    cpp = full_cov[:m, :m].copy()
    cpf = full_cov[:m, m:]
    cfp = full_cov[m:, :m]
    cff = full_cov[m:, m:].copy()

    cpp.flat[:: m + 1] += ridge

    factor = cho_factor(cpp, lower=True, check_finite=False)
    alpha = cho_solve(factor, past, check_finite=False)
    conditional_mean = cfp @ alpha

    beta = cho_solve(factor, cpf, check_finite=False)
    conditional_cov = cff - cfp @ beta
    conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)

    eig_min = np.min(np.linalg.eigvalsh(conditional_cov))
    if eig_min < 0:
        conditional_cov += (abs(eig_min) + ridge) * np.eye(n)

    return conditional_mean, conditional_cov


def simulate_conditional_fou_paths(
    current_spread: float,
    past_innovations: np.ndarray,
    mu: float,
    kappa: float,
    sigma: float,
    hurst: float,
    horizon_days: int,
    n_paths: int = 5000,
    dt: float = 1.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    if n_paths < 100:
        raise ValueError("Use at least 100 Monte Carlo paths.")

    mean_fgn, cov_fgn = conditional_future_fgn(
        past_innovations=past_innovations,
        future_steps=horizon_days,
        hurst=hurst,
        dt=dt,
    )

    rng = np.random.default_rng(seed)
    future_noise = rng.multivariate_normal(
        mean=mean_fgn,
        cov=cov_fgn,
        size=n_paths,
        method="cholesky",
    )

    paths = np.empty((n_paths, horizon_days + 1), dtype=float)
    paths[:, 0] = current_spread

    for t in range(horizon_days):
        x = paths[:, t]
        paths[:, t + 1] = (
            x
            + kappa * (mu - x) * dt
            + sigma * future_noise[:, t]
        )

    return paths


def first_passage_days(paths: np.ndarray, mu: float) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must be a 2D array with at least 2 columns.")

    x0 = paths[:, 0]
    future = paths[:, 1:]
    initial_side = np.sign(x0 - mu)

    crossed = np.where(
        initial_side[:, None] > 0,
        future <= mu,
        future >= mu,
    )

    any_cross = crossed.any(axis=1)
    first = np.argmax(crossed, axis=1) + 1

    result = np.full(paths.shape[0], np.nan)
    result[any_cross] = first[any_cross]
    return result


def convergence_probability_curve(
    first_passage: np.ndarray,
    horizon_days: int,
) -> pd.Series:
    fp = np.asarray(first_passage, dtype=float)

    probabilities = [
        np.mean(np.isfinite(fp) & (fp <= day))
        for day in range(1, horizon_days + 1)
    ]

    return pd.Series(
        probabilities,
        index=pd.RangeIndex(1, horizon_days + 1, name="dte"),
        name="convergence_probability",
    )


def choose_dte_from_probability(
    probability_curve: pd.Series,
    target_probability: float = 0.70,
) -> tuple[Optional[int], float]:
    if not (0 < target_probability < 1):
        raise ValueError("target_probability must lie in (0, 1).")

    reached = probability_curve[probability_curve >= target_probability]

    if reached.empty:
        return None, float(probability_curve.iloc[-1])

    selected = int(reached.index[0])
    return selected, float(reached.iloc[0])


def calculate_convergence_signal(
    spread_history: pd.Series,
    mu: float,
    kappa: float,
    sigma: float,
    hurst: float,
    stationary_variance: float,
    target_probability: float = 0.70,
    memory_window: int = 60,
    max_horizon_days: Optional[int] = None,
    horizon_half_life_multiple: float = 3.0,
    n_paths: int = 5000,
    dt: float = 1.0,
    seed: Optional[int] = 42,
) -> tuple[ConvergenceSignal, pd.Series]:
    """
    Generate the statistical signal and probability-implied DTE.

    If max_horizon_days is None, the search is capped at
        ceil(horizon_half_life_multiple * ln(2)/kappa).

    The cap only limits the search. DTE itself is determined by the
    convergence-probability target.
    """
    history = pd.Series(spread_history).dropna().astype(float)

    if len(history) < memory_window + 1:
        raise ValueError(
            f"Need at least {memory_window + 1} observations to condition "
            "the fractional-noise history."
        )

    current = float(history.iloc[-1])

    if stationary_variance <= 0:
        raise ValueError("stationary_variance must be positive.")

    current_z = float((current - mu) / np.sqrt(stationary_variance))
    direction = int(np.sign(current - mu))

    if direction == 0:
        signal = ConvergenceSignal(
            current_spread=current,
            current_z=current_z,
            direction=0,
            target_probability=target_probability,
            selected_dte_trading_days=None,
            probability_at_selected_dte=0.0,
            probability_at_max_horizon=1.0,
            statistical_signal=False,
        )
        return signal, pd.Series(dtype=float)

    drift_half_life = np.log(2.0) / kappa

    if max_horizon_days is None:
        max_horizon_days = max(
            5,
            int(np.ceil(horizon_half_life_multiple * drift_half_life)),
        )

    innovations = infer_fgn_innovations(
        history.iloc[-(memory_window + 1):],
        mu=mu,
        kappa=kappa,
        sigma=sigma,
        dt=dt,
    )

    paths = simulate_conditional_fou_paths(
        current_spread=current,
        past_innovations=innovations,
        mu=mu,
        kappa=kappa,
        sigma=sigma,
        hurst=hurst,
        horizon_days=max_horizon_days,
        n_paths=n_paths,
        dt=dt,
        seed=seed,
    )

    fp = first_passage_days(paths, mu=mu)
    curve = convergence_probability_curve(fp, horizon_days=max_horizon_days)
    selected_dte, prob_selected = choose_dte_from_probability(
        curve,
        target_probability=target_probability,
    )

    signal = ConvergenceSignal(
        current_spread=current,
        current_z=current_z,
        direction=direction,
        target_probability=target_probability,
        selected_dte_trading_days=selected_dte,
        probability_at_selected_dte=prob_selected,
        probability_at_max_horizon=float(curve.iloc[-1]),
        statistical_signal=selected_dte is not None,
    )

    return signal, curve


def trading_days_to_calendar_days(
    trading_days: int,
    trading_days_per_year: int = 252,
    calendar_days_per_year: int = 365,
) -> int:
    if trading_days < 1:
        raise ValueError("trading_days must be >= 1.")

    return int(np.ceil(
        trading_days * calendar_days_per_year / trading_days_per_year
    ))
