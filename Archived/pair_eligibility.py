from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd
import yfinance as yf


# ============================================================
# 1. KNOWN SAME-ISSUER / SHARE-CLASS PAIRS
# ============================================================

# These are pairs where both tickers represent different
# securities of essentially the same issuer.
#
# This list can be expanded if additional examples appear in
# the candidate universe.

SAME_ISSUER_PAIRS = {
    frozenset(("GOOG", "GOOGL")),
    frozenset(("NWS", "NWSA")),
    frozenset(("FOX", "FOXA")),
}


def is_same_issuer_pair(
    stock1: str,
    stock2: str,
) -> bool:
    """
    Check whether two securities are known to belong to the
    same issuer / share-class structure.
    """

    return frozenset((stock1, stock2)) in SAME_ISSUER_PAIRS


# ============================================================
# 2. OPTION AVAILABILITY
# ============================================================

def get_option_expirations(
    ticker: str,
) -> Tuple[bool, list]:
    """
    Retrieve available option expiration dates for a ticker.

    Returns
    -------
    has_options : bool
        True if at least one option expiration is available.

    expirations : list
        Available expiration dates.
    """

    try:

        asset = yf.Ticker(ticker)

        expirations = asset.options

        if expirations is None:
            return False, []

        expirations = list(expirations)

        return len(expirations) > 0, expirations

    except Exception:

        return False, []


def check_option_availability(
    ticker: str,
) -> dict:
    """
    Check whether a ticker has listed options.

    Returns
    -------
    dict containing:

        has_options
        number_expirations
        first_expiration
        last_expiration
    """

    has_options, expirations = get_option_expirations(ticker)

    if not expirations:

        return {
            "has_options": False,
            "number_expirations": 0,
            "first_expiration": None,
            "last_expiration": None,
        }

    return {
        "has_options": True,
        "number_expirations": len(expirations),
        "first_expiration": expirations[0],
        "last_expiration": expirations[-1],
    }


# ============================================================
# 3. SCREEN ONE PAIR
# ============================================================

def screen_pair(
    stock1: str,
    stock2: str,
) -> dict:
    """
    Perform the mechanical eligibility checks for one pair.
    """

    # --------------------------------------------------------
    # Same issuer
    # --------------------------------------------------------

    same_issuer = is_same_issuer_pair(
        stock1,
        stock2,
    )

    # --------------------------------------------------------
    # Options
    # --------------------------------------------------------

    option_1 = check_option_availability(stock1)
    option_2 = check_option_availability(stock2)

    both_have_options = (
        option_1["has_options"]
        and option_2["has_options"]
    )

    # --------------------------------------------------------
    # Preliminary eligibility
    # --------------------------------------------------------

    if same_issuer:

        eligible = False
        exclusion_reason = "Same issuer / share class"

    elif not both_have_options:

        eligible = False
        exclusion_reason = "Options unavailable for one or both underlyings"

    else:

        eligible = True
        exclusion_reason = None

    return {

        "stock1": stock1,
        "stock2": stock2,

        "same_issuer": same_issuer,

        "stock1_has_options":
            option_1["has_options"],

        "stock2_has_options":
            option_2["has_options"],

        "stock1_num_expirations":
            option_1["number_expirations"],

        "stock2_num_expirations":
            option_2["number_expirations"],

        "stock1_first_expiration":
            option_1["first_expiration"],

        "stock2_first_expiration":
            option_2["first_expiration"],

        "stock1_last_expiration":
            option_1["last_expiration"],

        "stock2_last_expiration":
            option_2["last_expiration"],

        "eligible":
            eligible,

        "exclusion_reason":
            exclusion_reason,

    }


# ============================================================
# 4. SCREEN CANDIDATE PAIRS
# ============================================================

def screen_pairs(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Screen all candidate pairs.

    Parameters
    ----------
    pairs : pd.DataFrame

        DataFrame containing at least:

            dependent
            independent

        and preferably:

            pair
            half_life
            pvalue
            hurst

    Returns
    -------
    pd.DataFrame
        Original pair information plus eligibility information.
    """

    results = []

    for _, row in pairs.iterrows():

        stock1 = row["dependent"]
        stock2 = row["independent"]

        screening = screen_pair(
            stock1,
            stock2,
        )

        result = row.to_dict()

        result.update(screening)

        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# 5. FINAL ELIGIBLE PAIRS
# ============================================================

def get_eligible_pairs(
    screened_pairs: pd.DataFrame,
    n_pairs: int = 20,
) -> pd.DataFrame:
    """
    Select the n_pairs with the shortest half-life among
    eligible pairs.

    No composite score is used.

    Parameters
    ----------
    screened_pairs : pd.DataFrame

    n_pairs : int
        Number of final pairs.

    Returns
    -------
    pd.DataFrame
    """

    eligible = screened_pairs[
        screened_pairs["eligible"]
    ].copy()

    eligible = eligible.sort_values(
        "half_life",
        ascending=True,
    )

    eligible = eligible.head(
        n_pairs
    ).reset_index(drop=True)

    return eligible


# ============================================================
# 6. SAVE SCREENING RESULTS
# ============================================================

def save_screening_results(
    screened_pairs: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    screening_path: str,
    selected_path: str,
) -> None:
    """
    Save screening results and final selected pairs.
    """

    screened_pairs.to_parquet(
        screening_path,
        index=False,
    )

    eligible_pairs.to_parquet(
        selected_path,
        index=False,
    )