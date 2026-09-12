from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


def compute_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def correlation_matrix(returns):
    return returns.corr()


def generate_candidate_pairs(corr_matrix, top_n=10) -> List[Tuple[str, str]]:
    pairs = set()
    for stock in corr_matrix.columns:
        neighbors = corr_matrix[stock].drop(stock).nlargest(top_n).index
        pairs.update(tuple(sorted((stock, other))) for other in neighbors)
    return sorted(pairs)


def estimate_hedge_ratio(y, x):
    fit = OLS(y, add_constant(x, has_constant="add")).fit()
    return float(fit.params.iloc[0]), float(fit.params.iloc[1]), fit.resid


def screen_cointegration(prices, candidate_pairs, significance=0.01, integration_alpha=0.05):
    logs = np.log(prices)
    candidates = sorted(set(tuple(sorted(pair)) for pair in candidate_pairs))
    tickers = sorted({ticker for pair in candidates for ticker in pair})

    roots = {}
    for ticker in tickers:
        level = adfuller(logs[ticker], regression="c", autolag="AIC")
        diff = adfuller(logs[ticker].diff().dropna(), regression="c", autolag="AIC")
        roots[ticker] = {
            "level_p": float(level[1]),
            "diff_p": float(diff[1]),
            "i1": bool(level[1] >= integration_alpha and diff[1] < integration_alpha),
        }

    rows = []
    spreads = {}
    for dependent, independent in candidates:
        alpha, beta, residual = estimate_hedge_ratio(logs[dependent], logs[independent])
        stat, pvalue, critical = coint(
            logs[dependent], logs[independent], trend="c", autolag="aic"
        )
        residual_adf = adfuller(residual, regression="n", autolag="AIC")
        integration_screen = roots[dependent]["i1"] and roots[independent]["i1"]
        selected = bool(pvalue <= significance and integration_screen and beta > 0)

        rows.append(
            {
                "pair": f"{dependent}-{independent}",
                "dependent": dependent,
                "independent": independent,
                "alpha": alpha,
                "beta": beta,
                "adf": float(stat),
                "pvalue": float(pvalue),
                "critical_1": float(critical[0]),
                "critical_5": float(critical[1]),
                "critical_10": float(critical[2]),
                "residual_lag": int(residual_adf[2]),
                "dependent_level_p": roots[dependent]["level_p"],
                "dependent_diff_p": roots[dependent]["diff_p"],
                "independent_level_p": roots[independent]["level_p"],
                "independent_diff_p": roots[independent]["diff_p"],
                "integration_screen": bool(integration_screen),
                "selected": selected,
            }
        )
        if selected:
            spreads[(dependent, independent)] = residual

    results = pd.DataFrame(rows)
    selected = results.loc[results.selected].reset_index(drop=True)
    return selected, spreads, results
