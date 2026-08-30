"""
strategy.py

Trading signal engine for the pairs-trading thesis.

The strategy uses:
    1. OU spread dynamics
    2. Current spread Z-score
    3. Leung-Li liquidation boundary
    4. Volatility-based stop-loss
    5. Maximum option DTE = 3 x OU half-life

Important:
    This module does NOT price options.
    It generates the underlying trading signal that will later
    be used by both the stock and option implementations.
"""

from typing import Optional, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Z-SCORE
# ---------------------------------------------------------------------

def calculate_z_score(
    spread: float,
    mean: float,
    sigma: float
) -> float:
    """
    Calculate the standardized spread.

    Z_t = (S_t - mu) / sigma
    """

    if sigma <= 0:
        raise ValueError("Spread standard deviation must be positive.")

    return (spread - mean) / sigma


# ---------------------------------------------------------------------
# ENTRY SIGNAL
# ---------------------------------------------------------------------

def determine_entry_signal(
    z_score: float,
    entry_z: float = 2.0
) -> int:
    """
    Determine whether an entry signal exists.

    Returns
    -------
    1  : positive divergence
    -1 : negative divergence
    0  : no signal

    Interpretation
    --------------
    +1:
        Dependent stock is relatively overvalued.
        Buy a PUT on the dependent stock.

    -1:
        Dependent stock is relatively undervalued.
        Buy a CALL on the dependent stock.
    """

    if z_score >= entry_z:
        return 1

    if z_score <= -entry_z:
        return -1

    return 0


# ---------------------------------------------------------------------
# STOP LOSS
# ---------------------------------------------------------------------

def calculate_stop_loss(
    mean: float,
    sigma: float,
    stop_multiplier: float = 2.0
) -> float:
    """
    Calculate the adverse spread stop-loss boundary.

    L = mu - stop_multiplier * sigma

    The negative side of the spread is used because the Leung-Li
    liquidation calculation is formulated relative to the lower
    boundary.

    Parameters
    ----------
    mean : float
        OU long-run mean.

    sigma : float
        Spread standard deviation.

    stop_multiplier : float
        Number of standard deviations used for the stop-loss.

    Returns
    -------
    float
        Stop-loss boundary.
    """

    return mean - stop_multiplier * sigma


def calculate_upper_stop_loss(
    mean: float,
    sigma: float,
    stop_multiplier: float = 2.0
) -> float:
    """
    Calculate the symmetric upper stop-loss boundary.
    """

    return mean + stop_multiplier * sigma


# ---------------------------------------------------------------------
# LEUNG-LI LIQUIDATION BOUNDARY
# ---------------------------------------------------------------------

def calculate_liquidation_boundary(
    theta: float,
    mean: float,
    sigma: float,
    stop_loss: float,
    discount_rate: float
) -> float:
    """
    Calculate the Leung-Li liquidation boundary.

    This function assumes that the validated Leung-Li solver used
    earlier in the project is represented by the expression below.

    The boundary is expressed in spread units.

    Parameters
    ----------
    theta : float
        OU mean-reversion speed.

    mean : float
        Long-run OU mean.

    sigma : float
        OU spread volatility.

    stop_loss : float
        Lower stopping boundary L.

    discount_rate : float
        Continuous discount rate.

    Returns
    -------
    float
        Optimal liquidation boundary b*.
    """

    if theta <= 0:
        raise ValueError("theta must be positive.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if discount_rate <= 0:
        raise ValueError("discount_rate must be positive.")

    if stop_loss >= mean:
        raise ValueError(
            "stop_loss must be below the OU mean."
        )

    # -----------------------------------------------------------------
    # Standardized lower boundary
    # -----------------------------------------------------------------

    a = (stop_loss - mean) / sigma

    # -----------------------------------------------------------------
    # Numerical solution of the Leung-Li boundary.
    #
    # We solve for the standardized liquidation boundary x = (b-mu)/sigma.
    #
    # The exact solver was validated separately in the previous stage.
    # -----------------------------------------------------------------

    from scipy.optimize import brentq
    from scipy.special import roots_hermitenorm
    from scipy.integrate import quad

    # Dimensionless discount parameter
    rho = discount_rate / theta

    # -----------------------------------------------------------------
    # Expected discounted first-passage formulation
    # -----------------------------------------------------------------

    def value_function(x: float) -> float:
        """
        Dimensionless value function component.
        """

        integral, _ = quad(
            lambda y: np.exp(
                -rho * (y - x)
            ) * np.exp(
                -(y ** 2) / 2
            ),
            a,
            x,
            limit=200
        )

        return integral

    def objective(x: float) -> float:
        """
        First-order optimality condition.

        The liquidation boundary is found numerically.
        """

        h = 1e-5

        derivative = (
            value_function(x + h)
            - value_function(x - h)
        ) / (2 * h)

        return derivative

    # Search interval above the mean.
    lower = max(1e-6, (0.01))
    upper = 10.0

    grid = np.linspace(lower, upper, 500)

    previous_x = grid[0]
    previous_value = objective(previous_x)

    root = None

    for x in grid[1:]:

        current_value = objective(x)

        if previous_value * current_value < 0:

            root = brentq(
                objective,
                previous_x,
                x
            )

            break

        previous_x = x
        previous_value = current_value

    if root is None:
        raise RuntimeError(
            "Could not find a liquidation boundary. "
            "Check OU parameters and discount rate."
        )

    return mean + root * sigma


# ---------------------------------------------------------------------
# EXIT CONDITIONS
# ---------------------------------------------------------------------

def check_exit(
    spread: float,
    position: int,
    mean: float,
    sigma: float,
    liquidation_boundary: float,
    stop_loss: float
) -> bool:
    """
    Determine whether an open position should be closed.

    Parameters
    ----------
    spread : float
        Current spread.

    position : int
        +1 = positive divergence / long put
        -1 = negative divergence / long call

    mean : float
        OU mean.

    sigma : float
        Spread standard deviation.

    liquidation_boundary : float
        Positive liquidation boundary.

    stop_loss : float
        Lower stop-loss boundary.

    Returns
    -------
    bool
        True if the position should be closed.
    """

    if position == 1:
        # Positive divergence:
        # exit when spread has reverted to the liquidation region
        # or when adverse movement reaches the stop-loss.

        if spread <= mean:
            return True

        if spread <= stop_loss:
            return True

    elif position == -1:
        # Negative divergence:
        # symmetric liquidation boundary.

        symmetric_boundary = (
            2 * mean - liquidation_boundary
        )

        symmetric_stop = (
            2 * mean - stop_loss
        )

        if spread >= mean:
            return True

        if spread >= symmetric_stop:
            return True

    return False


# ---------------------------------------------------------------------
# MAXIMUM DTE
# ---------------------------------------------------------------------

def calculate_max_dte(
    half_life: float,
    multiplier: float = 3.0
) -> int:
    """
    Maximum option maturity.

    DTE_max = multiplier * half_life

    Thesis assumption:
        DTE_max = 3 x half-life
    """

    if half_life <= 0:
        raise ValueError("Half-life must be positive.")

    return int(np.ceil(multiplier * half_life))


# ---------------------------------------------------------------------
# COMPLETE SIGNAL
# ---------------------------------------------------------------------

def generate_signal(
    spread: float,
    mean: float,
    sigma: float,
    theta: float,
    half_life: float,
    discount_rate: float,
    entry_z: float = 1.0,
    stop_multiplier: float = 2.0
) -> Dict:
    """
    Generate a complete trading signal for one pair.

    Returns a dictionary containing all quantities needed by the
    backtester.
    """

    z_score = calculate_z_score(
        spread,
        mean,
        sigma
    )

    entry_signal = determine_entry_signal(
        z_score,
        entry_z
    )

    stop_loss = calculate_stop_loss(
        mean,
        sigma,
        stop_multiplier
    )

    liquidation_boundary = calculate_liquidation_boundary(
        theta=theta,
        mean=mean,
        sigma=sigma,
        stop_loss=stop_loss,
        discount_rate=discount_rate
    )

    max_dte = calculate_max_dte(
        half_life
    )

    return {
        "spread": spread,
        "mean": mean,
        "sigma": sigma,
        "z_score": z_score,
        "entry_signal": entry_signal,
        "stop_loss": stop_loss,
        "liquidation_boundary": liquidation_boundary,
        "half_life": half_life,
        "max_dte": max_dte
    }


# ---------------------------------------------------------------------
# SIGNAL LABEL
# ---------------------------------------------------------------------

def signal_to_label(signal: int) -> str:
    """
    Convert numerical signal into readable label.
    """

    if signal == 1:
        return "LONG_PUT"

    if signal == -1:
        return "LONG_CALL"

    return "NO_SIGNAL"