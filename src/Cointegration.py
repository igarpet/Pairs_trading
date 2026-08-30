from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


# =============================================================================
# OLS REGRESSION
# =============================================================================

def estimate_hedge_ratio(
    y: pd.Series,
    x: pd.Series
) -> Tuple[float, float, pd.Series]:
    """
    Estimate the hedge ratio using OLS.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.

    x : pd.Series
        Independent variable.

    Returns
    -------
    alpha : float
    beta : float
    residuals : pd.Series
    """

    X = add_constant(x)

    model = OLS(y, X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    residuals = model.resid

    return alpha, beta, residuals


# =============================================================================
# ADF TEST
# =============================================================================

def adf_test(
    residuals: pd.Series
) -> Tuple[float, float]:
    """
    Perform an Augmented Dickey-Fuller test.

    Parameters
    ----------
    residuals : pd.Series

    Returns
    -------
    adf_statistic : float
    p_value : float
    """

    result = adfuller(residuals)

    return result[0], result[1]


# =============================================================================
# COINTEGRATION
# =============================================================================

def find_cointegrated_pairs(
    prices: pd.DataFrame,
    candidate_pairs: List[Tuple[str, str]],
    significance: float = 0.01
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.Series]]:
    """
    Test candidate pairs for Engle-Granger cointegration.

    Parameters
    ----------
    prices : pd.DataFrame

    candidate_pairs : list

    significance : float

    Returns
    -------
    results : pd.DataFrame

    spreads : dict
    """

    log_prices = np.log(prices)

    results = []

    spreads = {}

    for stock1, stock2 in tqdm(candidate_pairs):

        s1 = log_prices[stock1]

        s2 = log_prices[stock2]

        # ------------------------------------------------------------
        # Regression: stock1 ~ stock2
        # ------------------------------------------------------------

        alpha1, beta1, resid1 = estimate_hedge_ratio(s1, s2)

        adf1, p1 = adf_test(resid1)

        # ------------------------------------------------------------
        # Regression: stock2 ~ stock1
        # ------------------------------------------------------------

        alpha2, beta2, resid2 = estimate_hedge_ratio(s2, s1)

        adf2, p2 = adf_test(resid2)

        # ------------------------------------------------------------
        # Select best regression
        # ------------------------------------------------------------

        if p1 < p2:

            dependent = stock1
            independent = stock2

            alpha = alpha1
            beta = beta1

            adf = adf1
            pvalue = p1

            spread = resid1

        else:

            dependent = stock2
            independent = stock1

            alpha = alpha2
            beta = beta2

            adf = adf2
            pvalue = p2

            spread = resid2

        # ------------------------------------------------------------
        # Keep only significant pairs
        # ------------------------------------------------------------

        if pvalue < significance:

            pair = (dependent, independent)

            spreads[pair] = spread

            results.append({

                "dependent": dependent,
                "independent": independent,
                "alpha": alpha,
                "beta": beta,
                "adf": adf,
                "pvalue": pvalue

            })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "pvalue",
        ascending=True
    ).reset_index(drop=True)

    return results, spreads