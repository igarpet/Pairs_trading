import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def fgn_covariance(n, hurst, dt=1.0):
    idx = np.arange(n)
    lag = np.abs(idx[:, None] - idx[None, :]).astype(float)
    return 0.5 * dt ** (2 * hurst) * (
        np.abs(lag + 1) ** (2 * hurst)
        - 2 * lag ** (2 * hurst)
        + np.abs(lag - 1) ** (2 * hurst)
    )


def simulate_fou_paths(
    current_spread,
    mu,
    kappa,
    sigma,
    hurst,
    horizon_days,
    n_paths=5000,
    dt=1.0,
    seed=42,
    past_innovations=None,
):
    if past_innovations is None:
        mean = np.zeros(horizon_days)
        cov = fgn_covariance(horizon_days, hurst, dt)
    else:
        past = np.asarray(past_innovations, dtype=float)
        m = len(past)
        full = fgn_covariance(m + horizon_days, hurst, dt)
        cpp = full[:m, :m].copy()
        cpf = full[:m, m:]
        cfp = full[m:, :m]
        cff = full[m:, m:].copy()

        ridge = 1e-10
        cpp.flat[:: m + 1] += ridge
        factor = cho_factor(cpp, lower=True, check_finite=False)
        mean = cfp @ cho_solve(factor, past, check_finite=False)
        cov = cff - cfp @ cho_solve(factor, cpf, check_finite=False)
        cov = 0.5 * (cov + cov.T)
        eig_min = np.linalg.eigvalsh(cov).min()
        if eig_min < 0:
            cov += (abs(eig_min) + ridge) * np.eye(horizon_days)

    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(mean, cov, size=n_paths, method="cholesky")
    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = current_spread

    for t in range(horizon_days):
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (mu - paths[:, t]) * dt
            + sigma * noise[:, t]
        )
    return paths


def convergence_probability(paths, mu, target_probability):
    x0 = paths[:, 0]
    future = paths[:, 1:]
    crossed = np.where(
        (x0 - mu)[:, None] > 0,
        future <= mu,
        future >= mu,
    )
    any_cross = crossed.any(axis=1)
    first = np.argmax(crossed, axis=1) + 1
    first = np.where(any_cross, first, np.inf)

    curve = pd.Series(
        [np.mean(first <= day) for day in range(1, future.shape[1] + 1)],
        index=pd.RangeIndex(1, future.shape[1] + 1, name="dte"),
        name="convergence_probability",
    )
    reached = curve[curve >= target_probability]
    horizon = int(reached.index[0]) if len(reached) else None
    probability = float(reached.iloc[0]) if len(reached) else float(curve.iloc[-1])
    return curve, horizon, probability


def structural_convergence_horizon(
    mu,
    kappa,
    sigma,
    hurst,
    stationary_variance,
    starting_z=1.5,
    target_probability=0.70,
    max_horizon_days=252,
    n_paths=5000,
    dt=1.0,
    seed=42,
):
    current = mu + starting_z * np.sqrt(stationary_variance)
    paths = simulate_fou_paths(
        current, mu, kappa, sigma, hurst, max_horizon_days, n_paths, dt, seed
    )
    curve, horizon, probability = convergence_probability(paths, mu, target_probability)
    return {
        "structural_starting_z": float(starting_z),
        "structural_target_probability": float(target_probability),
        "structural_t70": horizon,
        "structural_probability_at_t70": probability,
        "structural_probability_max": float(curve.iloc[-1]),
    }


def calculate_convergence_signal(
    spread_history,
    mu,
    kappa,
    sigma,
    hurst,
    stationary_variance,
    target_probability=0.70,
    entry_z=1.5,
    memory_window=60,
    max_horizon_days=126,
    n_paths=5000,
    dt=1.0,
    seed=42,
):
    history = pd.Series(spread_history).dropna().astype(float)
    current = float(history.iloc[-1])
    current_z = float((current - mu) / np.sqrt(stationary_variance))
    direction = int(np.sign(current - mu))

    if direction == 0:
        return {
            "current_spread": current,
            "current_z": current_z,
            "direction": 0,
            "selected_dte_trading_days": None,
            "probability_at_selected_dte": 0.0,
            "probability_at_max_horizon": 1.0,
            "statistical_signal": False,
        }, pd.Series(dtype=float)

    x = history.iloc[-(memory_window + 1) :].to_numpy()
    dx = x[1:] - x[:-1]
    drift = kappa * (mu - x[:-1]) * dt
    innovations = (dx - drift) / sigma

    paths = simulate_fou_paths(
        current,
        mu,
        kappa,
        sigma,
        hurst,
        max_horizon_days,
        n_paths,
        dt,
        seed,
        past_innovations=innovations,
    )
    curve, horizon, probability = convergence_probability(paths, mu, target_probability)

    return {
        "current_spread": current,
        "current_z": current_z,
        "direction": direction,
        "selected_dte_trading_days": horizon,
        "probability_at_selected_dte": probability,
        "probability_at_max_horizon": float(curve.iloc[-1]),
        "statistical_signal": abs(current_z) >= entry_z and horizon is not None,
    }, curve
