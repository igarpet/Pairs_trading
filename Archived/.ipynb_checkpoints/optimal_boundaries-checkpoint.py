from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


# ============================================================
# RESULT CONTAINER
# ============================================================

@dataclass
class OptimalBoundaries:

    stop_loss: float

    entry_boundary: float

    exit_boundary: float

    success: bool = True

    message: str = ""


# ============================================================
# BASIC HELPERS
# ============================================================

def stop_loss_from_volatility(
    mu: float,
    sigma: float,
    k: float = 2.0
) -> float:
    """
    Thesis-specific volatility stop.

    L = mu - k*sigma
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if k <= 0:
        raise ValueError("k must be positive.")

    return mu - k * sigma


def annual_discount_factor(
    annual_rate: float,
    trading_days: float,
    days_per_year: int = 252
) -> float:
    """
    Discount factor for an annual continuously compounded
    risk-free rate.

    DF = exp(-r*T/252)

    IMPORTANT:
    The rate is NOT divided by sqrt(252).
    """

    if days_per_year <= 0:
        raise ValueError(
            "days_per_year must be positive."
        )

    return np.exp(
        -annual_rate
        * trading_days
        / days_per_year
    )


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def _validate_ou_parameters(
    kappa: float,
    sigma: float,
    r: float
) -> None:

    if kappa <= 0:
        raise ValueError(
            "kappa must be positive."
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be positive."
        )

    if r <= 0:
        raise ValueError(
            "r must be positive."
        )


# ============================================================
# LEUNG & LI FUNDAMENTAL SOLUTIONS
# EQUATIONS (3.3) AND (3.4)
# ============================================================

def F_function(
    x: float,
    kappa: float,
    mu: float,
    sigma: float,
    r: float
) -> float:
    """
    Leung & Li equation (3.3).

    F(x) =
        integral_0^inf

        u^(r/kappa - 1)

        * exp(
            sqrt(2*kappa)/sigma
            * (x-mu)*u
            - u^2/2
        )

        du
    """

    _validate_ou_parameters(
        kappa,
        sigma,
        r
    )

    coefficient = (
        np.sqrt(2.0 * kappa)
        / sigma
    )

    power = (
        r / kappa
        - 1.0
    )

    def integrand(u):

        if u == 0:
            return 0.0

        exponent = (
            power * np.log(u)
            +
            coefficient
            * (x - mu)
            * u
            -
            0.5 * u * u
        )

        if exponent < -745:
            return 0.0

        if exponent > 700:
            return np.inf

        return np.exp(exponent)

    value, _ = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-9,
        limit=250
    )

    return float(value)


def G_function(
    x: float,
    kappa: float,
    mu: float,
    sigma: float,
    r: float
) -> float:
    """
    Leung & Li equation (3.4).

    G(x) =
        integral_0^inf

        u^(r/kappa - 1)

        * exp(
            sqrt(2*kappa)/sigma
            * (mu-x)*u
            - u^2/2
        )

        du
    """

    _validate_ou_parameters(
        kappa,
        sigma,
        r
    )

    coefficient = (
        np.sqrt(2.0 * kappa)
        / sigma
    )

    power = (
        r / kappa
        - 1.0
    )

    def integrand(u):

        if u == 0:
            return 0.0

        exponent = (
            power * np.log(u)
            +
            coefficient
            * (mu - x)
            * u
            -
            0.5 * u * u
        )

        if exponent < -745:
            return 0.0

        if exponent > 700:
            return np.inf

        return np.exp(exponent)

    value, _ = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-9,
        limit=250
    )

    return float(value)


# ============================================================
# DERIVATIVES
# ============================================================

def _numerical_derivative(
    function,
    x: float,
    relative_step: float = 1e-5
) -> float:

    h = (
        relative_step
        * max(1.0, abs(x))
    )

    return (
        function(x + h)
        -
        function(x - h)
    ) / (2.0 * h)


def F_prime(
    x,
    kappa,
    mu,
    sigma,
    r
):

    return _numerical_derivative(

        lambda z:
        F_function(
            z,
            kappa,
            mu,
            sigma,
            r
        ),

        x
    )


def G_prime(
    x,
    kappa,
    mu,
    sigma,
    r
):

    return _numerical_derivative(

        lambda z:
        G_function(
            z,
            kappa,
            mu,
            sigma,
            r
        ),

        x
    )


# ============================================================
# CRITICAL STOP-LOSS LEVEL
# ============================================================

def critical_stop_loss(
    kappa: float,
    mu: float,
    r: float,
    c: float
) -> float:
    """
    Leung & Li critical level.

    L* = (kappa*mu + r*c)
         / (kappa + r)
    """

    if kappa <= 0:
        raise ValueError(
            "kappa must be positive."
        )

    if r <= 0:
        raise ValueError(
            "r must be positive."
        )

    if c <= 0:
        raise ValueError(
            "c must be positive."
        )

    return (
        kappa * mu
        +
        r * c
    ) / (
        kappa + r
    )


# ============================================================
# THEOREM 5.1 — C AND D
# ============================================================

def liquidation_constants(
    b: float,
    L: float,
    c: float,
    kappa: float,
    mu: float,
    sigma: float,
    r: float
) -> Tuple[float, float]:
    """
    Leung & Li equation (5.4).

    V_L(x) = C F(x) + D G(x)
    """

    Fb = F_function(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    Gb = G_function(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    FL = F_function(
        L,
        kappa,
        mu,
        sigma,
        r
    )

    GL = G_function(
        L,
        kappa,
        mu,
        sigma,
        r
    )

    denominator = (
        Fb * GL
        -
        FL * Gb
    )

    if abs(denominator) < 1e-14:

        raise RuntimeError(
            "Near-singular denominator "
            "in equation (5.4)."
        )

    C = (

        (b - c) * GL
        -
        (L - c) * Gb

    ) / denominator

    D = (

        (L - c) * Fb
        -
        (b - c) * FL

    ) / denominator

    return (
        float(C),
        float(D)
    )


# ============================================================
# EQUATION (5.5)
# ============================================================

def liquidation_boundary_equation(
    b: float,
    L: float,
    c: float,
    kappa: float,
    mu: float,
    sigma: float,
    r: float
) -> float:
    """
    Leung & Li equation (5.5).

    [(L-c)G(b) - (b-c)G(L)] F'(b)

    +

    [(b-c)F(L) - (L-c)F(b)] G'(b)

    =

    G(b)F(L) - G(L)F(b)

    Returns:

        LHS - RHS
    """

    Fb = F_function(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    Gb = G_function(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    FL = F_function(
        L,
        kappa,
        mu,
        sigma,
        r
    )

    GL = G_function(
        L,
        kappa,
        mu,
        sigma,
        r
    )

    Fpb = F_prime(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    Gpb = G_prime(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    lhs = (

        (
            (L - c) * Gb
            -
            (b - c) * GL
        )
        *
        Fpb

        +

        (
            (b - c) * FL
            -
            (L - c) * Fb
        )
        *
        Gpb
    )

    rhs = (

        Gb * FL
        -
        GL * Fb

    )

    return float(
        lhs - rhs
    )


# ============================================================
# SOLVE b_L*
# ============================================================

def solve_liquidation_boundary(
    kappa: float,
    mu: float,
    sigma: float,
    r: float,
    c: float,
    L: float,
    upper_sigma_multiple: float = 15.0,
    grid_points: int = 250
) -> float:
    """
    Solve for b_L* using equation (5.5).

    Corollary 5.2:

        if L < L*

        there exists a unique

        b_L* in (L*, infinity)

    where:

        L* =
        (kappa*mu + r*c)
        /(kappa+r)
    """

    _validate_ou_parameters(
        kappa,
        sigma,
        r
    )

    if c <= 0:
        raise ValueError(
            "c must be positive."
        )

    L_star = critical_stop_loss(
        kappa,
        mu,
        r,
        c
    )

    if L >= L_star:

        raise ValueError(

            f"No non-trivial "
            f"liquidation boundary exists. "

            f"L={L:.8f} >= "
            f"L*={L_star:.8f}."

        )

    lower = (
        L_star
        +
        1e-7
    )

    upper = max(

        mu
        +
        upper_sigma_multiple
        * sigma,

        lower
        +
        2.0 * sigma,

        lower
        +
        1.0

    )

    grid = np.linspace(
        lower,
        upper,
        grid_points
    )

    values = np.array([

        liquidation_boundary_equation(

            b,

            L,

            c,

            kappa,

            mu,

            sigma,

            r

        )

        for b in grid

    ])

    for i in range(
        len(grid) - 1
    ):

        f1 = values[i]
        f2 = values[i + 1]

        if not (
            np.isfinite(f1)
            and
            np.isfinite(f2)
        ):
            continue

        if f1 == 0:

            return float(
                grid[i]
            )

        if f1 * f2 < 0:

            root = brentq(

                lambda b:

                liquidation_boundary_equation(

                    b,

                    L,

                    c,

                    kappa,

                    mu,

                    sigma,

                    r

                ),

                grid[i],
                grid[i + 1],

                xtol=1e-9,
                rtol=1e-10,
                maxiter=200

            )

            return float(root)

    raise RuntimeError(

        "Could not find the unique "
        "root of Leung & Li equation "
        "(5.5). Increase the search "
        "range and inspect the residual."

    )


# ============================================================
# VALUE FUNCTION — EQUATION (5.3)
# ============================================================

def liquidation_value(
    x: float,
    b: float,
    L: float,
    c: float,
    kappa: float,
    mu: float,
    sigma: float,
    r: float
) -> float:
    """
    Leung & Li equation (5.3).
    """

    if (
        x <= L
        or
        x >= b
    ):

        return x - c

    C, D = liquidation_constants(

        b,
        L,
        c,
        kappa,
        mu,
        sigma,
        r

    )

    return (

        C * F_function(
            x,
            kappa,
            mu,
            sigma,
            r
        )

        +

        D * G_function(
            x,
            kappa,
            mu,
            sigma,
            r
        )

    )


# ============================================================
# VALIDATION
# ============================================================

def validate_liquidation_boundary(
    b: float,
    L: float,
    c: float,
    kappa: float,
    mu: float,
    sigma: float,
    r: float
) -> dict:
    """
    Validate the numerical solution.

    Checks:

    1. equation (5.5)
    2. value matching at b
    3. smooth pasting at b
    4. value matching at L
    """

    C, D = liquidation_constants(

        b,
        L,
        c,
        kappa,
        mu,
        sigma,
        r

    )

    Fb = F_function(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    Gb = G_function(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    FL = F_function(
        L,
        kappa,
        mu,
        sigma,
        r
    )

    GL = G_function(
        L,
        kappa,
        mu,
        sigma,
        r
    )

    Fpb = F_prime(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    Gpb = G_prime(
        b,
        kappa,
        mu,
        sigma,
        r
    )

    value_b = (
        C * Fb
        +
        D * Gb
    )

    derivative_b = (
        C * Fpb
        +
        D * Gpb
    )

    value_L = (
        C * FL
        +
        D * GL
    )

    return {

        "equation_5_5_residual":
        liquidation_boundary_equation(
            b,
            L,
            c,
            kappa,
            mu,
            sigma,
            r
        ),

        "value_matching_b_residual":
        value_b - (b - c),

        "smooth_pasting_b_residual":
        derivative_b - 1.0,

        "value_matching_L_residual":
        value_L - (L - c)

    }


# ============================================================
# HIGH-LEVEL API
# ============================================================

def calculate_optimal_boundaries(
    kappa: float,
    mu: float,
    sigma: float,
    risk_free_rate: float,
    transaction_cost: float,
    stop_multiplier: float = 2.0
) -> OptimalBoundaries:
    """
    Calculate the volatility stop and the validated
    Leung & Li liquidation boundary.

    Entry remains NaN until Theorem 5.5 is implemented.
    """

    L = stop_loss_from_volatility(

        mu,
        sigma,
        stop_multiplier

    )

    b_star = solve_liquidation_boundary(

        kappa=kappa,

        mu=mu,

        sigma=sigma,

        r=risk_free_rate,

        c=transaction_cost,

        L=L

    )

    return OptimalBoundaries(

        stop_loss=L,

        entry_boundary=np.nan,

        exit_boundary=b_star,

        success=True,

        message=(
            "Exit boundary calculated directly "
            "from Leung & Li Theorem 5.1 "
            "and equation (5.5). "
            "Entry boundary pending Theorem 5.5."
        )

    )