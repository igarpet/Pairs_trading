"""Formation-only Engle–Granger screening with a fixed orientation.

The alphabetical first ticker is the dependent series. Candidate pairs are
screened using raw Engle–Granger p-values, without multiple-testing adjustment.
This is an individual screening rule, not a family-wise significance claim.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


def estimate_hedge_ratio(y, x):
    fit = OLS(y, add_constant(x, has_constant="add")).fit()
    return float(fit.params.iloc[0]), float(fit.params.iloc[1]), fit.resid


def screen_cointegration(prices, candidate_pairs, significance=0.01, integration_alpha=0.05):
    if not 0 < significance < 1 or not 0 < integration_alpha < 1:
        raise ValueError("Significance levels must be in (0, 1).")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Formation prices must be finite and positive.")
    logs = np.log(prices)
    candidates = sorted(set(tuple(sorted(p)) for p in candidate_pairs))
    unit_roots = {}
    for ticker in sorted({t for p in candidates for t in p}):
        x = logs[ticker]
        try:
            level = adfuller(x, regression="c", autolag="AIC")
            diff = adfuller(x.diff().dropna(), regression="c", autolag="AIC")
            unit_roots[ticker] = dict(
                level_p=float(level[1]),
                diff_p=float(diff[1]),
                level_lag=int(level[2]),
                diff_lag=int(diff[2]),
                i1=bool(level[1] >= integration_alpha and diff[1] < integration_alpha),
            )
        except ValueError as exc:
            unit_roots[ticker] = dict(level_p=np.nan, diff_p=np.nan, i1=False, error=str(exc))
    rows, spreads = [], {}
    for dep, ind in candidates:
        row = dict(
            pair=f"{dep}-{ind}",
            dependent=dep,
            independent=ind,
            pvalue=1.0,
            alpha=np.nan,
            beta=np.nan,
            adf=np.nan,
            trend="c",
            autolag="aic",
            maxlag="statsmodels_default",
            orientation="alphabetical",
            n_obs=len(logs),
            formation_start=logs.index.min(),
            formation_end=logs.index.max(),
            integration_screen=bool(unit_roots[dep]["i1"] and unit_roots[ind]["i1"]),
            test_error="",
        )
        for label, ticker in [("dependent", dep), ("independent", ind)]:
            row.update({f"{label}_{k}": v for k, v in unit_roots[ticker].items()})
        try:
            alpha, beta, residual = estimate_hedge_ratio(logs[dep], logs[ind])
            stat, p, critical = coint(logs[dep], logs[ind], trend="c", autolag="aic")
            if not np.isfinite(stat) or not np.isfinite(p):
                raise ValueError("Degenerate or nearly collinear pair.")
            # coint uses no constant in the residual ADF regression.
            residual_adf = adfuller(residual, regression="n", autolag="AIC")
            row.update(
                alpha=alpha,
                beta=beta,
                adf=float(stat),
                pvalue=float(p),
                residual_lag=int(residual_adf[2]),
                critical_1=float(critical[0]),
                critical_5=float(critical[1]),
                critical_10=float(critical[2]),
            )
            spreads[(dep, ind)] = residual
        except ValueError as exc:
            row["test_error"] = str(exc)
        rows.append(row)
    columns = [
        "pair",
        "dependent",
        "independent",
        "alpha",
        "beta",
        "adf",
        "pvalue",
        "adjusted_pvalue",
        "selected",
        "integration_screen",
        "test_error",
    ]
    audit = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    if rows:
        family_size = len(prices.columns) * (len(prices.columns) - 1) // 2
        if len(candidates) > family_size:
            raise ValueError("Invalid candidate pair family.")
        # Retain this legacy audit column as an explicit unadjusted alias.
        audit["adjusted_pvalue"] = audit["pvalue"]
        audit["selected"] = (
            (audit.pvalue <= significance)
            & audit.integration_screen
            & (audit.beta > 0)
            & audit.test_error.eq("")
        )
    audit["multiplicity_method"] = "none"
    audit["n_candidate_tests"] = len(candidates)
    audit["n_family_tests"] = len(prices.columns) * (len(prices.columns) - 1) // 2
    selected = audit.loc[audit.selected.astype(bool)].reset_index(drop=True)
    keys = set(zip(selected.dependent, selected.independent))
    return selected, {k: v for k, v in spreads.items() if k in keys}, audit


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    "Daily log returns: log(P_t / P_(t-1))."

    if prices.isnull().values.any():
        raise ValueError("Prices contain NaN values.")

    returns = np.log(prices / prices.shift(1))

    return returns.dropna()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    "Pearson correlations between return series."

    return returns.corr(method="pearson")


def generate_candidate_pairs(corr_matrix: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, str]]:
    "Take each asset's top correlations; deduplicate and orient alphabetically."

    pairs = set()

    for stock in corr_matrix.columns:

        correlations = (
            corr_matrix[stock].drop(labels=stock).sort_values(ascending=False).head(top_n)
        )

        for candidate in correlations.index:

            pair = tuple(sorted((stock, candidate)))

            pairs.add(pair)

    return sorted(list(pairs))
