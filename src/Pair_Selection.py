
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# DATA
# =============================================================================

def load_prices(filepath: str | Path) -> pd.DataFrame:
    """
    Load adjusted closing prices from a parquet file.

    Parameters
    ----------
    filepath : str or Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        Price dataframe indexed by date.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist.")

    prices = pd.read_parquet(filepath)

    if prices.empty:
        raise ValueError("Price dataframe is empty.")

    return prices


# =============================================================================
# RETURNS
# =============================================================================

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily logarithmic returns.

    Parameters
    ----------
    prices : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    if prices.isnull().values.any():
        raise ValueError("Prices contain NaN values.")

    returns = np.log(prices / prices.shift(1))

    return returns.dropna()


# =============================================================================
# CORRELATION
# =============================================================================

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix.

    Parameters
    ----------
    returns : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    return returns.corr(method="pearson")


# =============================================================================
# CANDIDATE PAIRS
# =============================================================================

def generate_candidate_pairs(
    corr_matrix: pd.DataFrame,
    top_n: int = 10
) -> List[Tuple[str, str]]:
    """
    Generate candidate pairs using the Top-N correlation approach.

    Parameters
    ----------
    corr_matrix : pd.DataFrame

    top_n : int
        Number of most correlated stocks retained for each asset.

    Returns
    -------
    list[tuple]
        Unique candidate pairs.
    """

    pairs = set()

    for stock in corr_matrix.columns:

        correlations = (
            corr_matrix[stock]
            .drop(labels=stock)
            .sort_values(ascending=False)
            .head(top_n)
        )

        for candidate in correlations.index:

            pair = tuple(sorted((stock, candidate)))

            pairs.add(pair)

    return sorted(list(pairs))


# =============================================================================
# SUMMARY
# =============================================================================

def candidate_summary(candidate_pairs: List[Tuple[str, str]]) -> None:
    """
    Print a summary of candidate pairs.

    Parameters
    ----------
    candidate_pairs : list
    """

    print("=" * 60)
    print("PAIR SELECTION SUMMARY")
    print("=" * 60)

    print(f"Candidate pairs : {len(candidate_pairs):,}")

    print("=" * 60)