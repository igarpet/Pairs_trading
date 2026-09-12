"""Formation-only Engle-Granger screening with a fixed orientation."""

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
    logs = np.log(prices)
    candidates = sorted(set(tuple(sorted(p)) for p in candidate_pairs))
    unit_roots = {}

    for ticker in sorted({t for pair in candidates for t in pair}):
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
            stat, pvalue, critical = coint(logs[dep], logs[ind], trend="c", autolag="aic")
            if not np.isfinite(stat) or not np.isfinite(pvalue):
                raise ValueError("Degenerate or nearly collinear pair.")
            residual_adf = adfuller(residual, regression="n", autolag="AIC")
            row.update(
                alpha=alpha,
                beta=beta,
                adf=float(stat),
                pvalue=float(pvalue),
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
    return selected, {key: value for key, value in spreads.items() if key in keys}, audit


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: log(P_t / P_(t-1))."""
    return np.log(prices / prices.shift(1)).dropna()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr(method="pearson")


def generate_candidate_pairs(corr_matrix: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, str]]:
    """Take each asset's top correlations; deduplicate and orient alphabetically."""
    pairs = set()
    for stock in corr_matrix.columns:
        correlations = corr_matrix[stock].drop(labels=stock).sort_values(ascending=False).head(top_n)
        for candidate in correlations.index:
            pairs.add(tuple(sorted((stock, candidate))))
    return sorted(pairs)
