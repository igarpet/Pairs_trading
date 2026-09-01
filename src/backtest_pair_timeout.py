"""
Static Module 07 extension with a pair-level loss-streak timeout.

This preserves the original frozen pair universe and all original entry/exit,
volatility, fOU convergence, option-pricing and sizing rules. The only added
rule is:

    after 3 consecutive losing closed trades for the same pair,
    block new entries in that pair for 6 calendar months.

A winning closed trade resets the consecutive-loss counter to zero. When a
cooldown is triggered, the counter is reset to zero. Existing open positions
are never force-closed by the timeout rule.

This is a post-hoc risk-control extension and should not be presented as the
original untouched out-of-sample specification.
"""

from __future__ import annotations

from math import sqrt
import numpy as np
import pandas as pd

from src.convergence_signal import calculate_convergence_signal
from src.backtest import (
    CALENDAR_DAYS_PER_YEAR,
    CONTRACT_MULTIPLIER,
    _has_converged,
    _position_market_value,
    _prepare_pair_parameters,
    _risk_free_at,
    _stable_seed,
    black_scholes_delta,
    black_scholes_price,
    compute_log_spread,
    next_observed_date_on_or_after,
    option_types_from_spread_direction,
    precompute_oos_ewma_volatility,
    size_log_spread_option_legs,
    trading_to_calendar_days,
)


def run_walk_forward_backtest_with_pair_timeout(
    train_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    cointegrated_pairs: pd.DataFrame,
    risk_free_rates,
    initial_capital: float = 100_000.0,
    entry_z: float = 1.5,
    target_probability: float = 0.70,
    memory_window: int = 60,
    max_horizon_days: int = 126,
    n_paths: int = 5000,
    ewma_lambda: float = 0.94,
    seed: int = 42,
    loss_streak_trigger: int = 3,
    timeout_months: int = 6,
) -> dict:
    """Run the original static OOS strategy with pair-level cooldowns."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if int(loss_streak_trigger) < 1:
        raise ValueError("loss_streak_trigger must be >= 1.")
    if int(timeout_months) < 1:
        raise ValueError("timeout_months must be >= 1.")

    train = train_prices.copy().sort_index()
    test = test_prices.copy().sort_index()
    train.index = pd.to_datetime(train.index).normalize()
    test.index = pd.to_datetime(test.index).normalize()

    if train.index.max() >= test.index.min():
        raise ValueError("Training and test periods must not overlap.")

    params = _prepare_pair_parameters(eligible_pairs, cointegrated_pairs)
    tickers = sorted(set(params["dependent"]).union(params["independent"]))
    missing_train = sorted(set(tickers) - set(train.columns))
    missing_test = sorted(set(tickers) - set(test.columns))
    if missing_train or missing_test:
        raise KeyError(f"Missing price columns. train={missing_train}, test={missing_test}")

    full_prices = pd.concat([train[tickers], test[tickers]], axis=0).sort_index()
    vol = precompute_oos_ewma_volatility(
        train_prices=train[tickers], test_prices=test[tickers], lambda_=ewma_lambda
    )

    spread_series = {}
    for _, row in params.iterrows():
        spread_series[row["pair"]] = compute_log_spread(
            full_prices,
            row["dependent"],
            row["independent"],
            row["alpha"],
            row["beta"],
        )

    cash = float(initial_capital)
    open_positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    skipped: list[dict] = []
    timeout_events: list[dict] = []

    # Pair-level state. Values are updated only when a trade closes.
    consecutive_losses = {str(p): 0 for p in params["pair"]}
    timeout_until = {str(p): pd.NaT for p in params["pair"]}

    test_dates = pd.DatetimeIndex(test.index).sort_values()

    for date in test_dates:
        date = pd.Timestamp(date).normalize()

        # A. Manage existing positions first.
        for pair in list(open_positions.keys()):
            pos = open_positions[pair]
            row = params.loc[params["pair"] == pair].iloc[0]
            dep, ind = row["dependent"], row["independent"]

            dep_spot = float(test.at[date, dep])
            ind_spot = float(test.at[date, ind])
            dep_vol = float(vol.at[date, dep])
            ind_vol = float(vol.at[date, ind])
            rf = _risk_free_at(risk_free_rates, date)
            current_spread = float(spread_series[pair].loc[date])

            converged = _has_converged(
                current_spread=current_spread,
                mu=float(row["mu"]),
                entry_direction=int(pos["direction"]),
            )
            expired = date >= pd.Timestamp(pos["expiry_date"]).normalize()

            if converged or expired:
                exit_value, dep_exit_px, ind_exit_px = _position_market_value(
                    pos, date, dep_spot, ind_spot, dep_vol, ind_vol, rf
                )
                cash += exit_value
                pnl = float(exit_value - pos["entry_premium"])
                trade_return = float(pnl / pos["entry_premium"])
                exit_reason = "convergence" if converged else "expiry"

                if pnl < 0:
                    consecutive_losses[pair] += 1
                else:
                    consecutive_losses[pair] = 0

                triggered_timeout = False
                triggered_until = pd.NaT
                if consecutive_losses[pair] >= int(loss_streak_trigger):
                    triggered_timeout = True
                    triggered_until = date + pd.DateOffset(months=int(timeout_months))
                    timeout_until[pair] = triggered_until
                    timeout_events.append({
                        "pair": pair,
                        "trigger_date": date,
                        "timeout_until": triggered_until,
                        "loss_streak_trigger": int(loss_streak_trigger),
                        "timeout_months": int(timeout_months),
                    })
                    # New cycle after serving the cooldown.
                    consecutive_losses[pair] = 0

                trades.append({
                    **pos,
                    "exit_date": date,
                    "exit_reason": exit_reason,
                    "exit_spread": current_spread,
                    "exit_dependent_spot": dep_spot,
                    "exit_independent_spot": ind_spot,
                    "exit_dependent_volatility": dep_vol,
                    "exit_independent_volatility": ind_vol,
                    "exit_risk_free_rate": rf,
                    "exit_dependent_option_price": dep_exit_px,
                    "exit_independent_option_price": ind_exit_px,
                    "exit_value": float(exit_value),
                    "pnl": pnl,
                    "trade_return": trade_return,
                    "holding_trading_days": int(
                        test_dates.get_loc(date) - test_dates.get_loc(pos["entry_date"])
                    ),
                    "holding_calendar_days": int((date - pos["entry_date"]).days),
                    "consecutive_losses_after_exit": int(consecutive_losses[pair]),
                    "timeout_triggered": bool(triggered_timeout),
                    "timeout_until_after_exit": triggered_until,
                })
                del open_positions[pair]

        # B. Scan flat pairs for new entries.
        for _, row in params.iterrows():
            pair = str(row["pair"])
            if pair in open_positions:
                continue

            blocked_until = timeout_until[pair]
            if pd.notna(blocked_until) and date < pd.Timestamp(blocked_until).normalize():
                skipped.append({
                    "date": date,
                    "pair": pair,
                    "reason": "pair_timeout",
                    "timeout_until": pd.Timestamp(blocked_until).normalize(),
                })
                continue

            dep, ind = row["dependent"], row["independent"]
            if float(row["beta"]) <= 0:
                skipped.append({"date": date, "pair": pair, "reason": "nonpositive_beta"})
                continue

            history = spread_series[pair].loc[:date].dropna()
            if len(history) < memory_window + 1:
                continue

            current_spread = float(history.iloc[-1])
            current_z = float(
                (current_spread - float(row["mu"])) / sqrt(float(row["variance"]))
            )
            if abs(current_z) < float(entry_z):
                continue

            signal_seed = _stable_seed(seed, pair, date)
            try:
                signal, _ = calculate_convergence_signal(
                    spread_history=history,
                    mu=float(row["mu"]),
                    kappa=float(row["kappa"]),
                    sigma=float(row["sigma"]),
                    hurst=float(row["hurst"]),
                    stationary_variance=float(row["variance"]),
                    target_probability=float(target_probability),
                    memory_window=int(memory_window),
                    max_horizon_days=int(max_horizon_days),
                    n_paths=int(n_paths),
                    dt=1.0,
                    seed=signal_seed,
                )
            except Exception as exc:
                skipped.append({
                    "date": date,
                    "pair": pair,
                    "reason": f"signal_error:{type(exc).__name__}",
                })
                continue

            horizon = signal.selected_dte_trading_days
            if horizon is None:
                continue

            direction = 1 if current_z > 0 else -1
            dep_type, ind_type = option_types_from_spread_direction(direction)
            dep_spot = float(test.at[date, dep])
            ind_spot = float(test.at[date, ind])
            dep_vol = float(vol.at[date, dep])
            ind_vol = float(vol.at[date, ind])
            rf = _risk_free_at(risk_free_rates, date)

            required_calendar_days = trading_to_calendar_days(int(horizon))
            target_expiry = date + pd.Timedelta(days=required_calendar_days)
            expiry = next_observed_date_on_or_after(target_expiry, test_dates)
            calendar_dte = int((expiry - date).days)
            T = calendar_dte / CALENDAR_DAYS_PER_YEAR

            dep_strike = dep_spot
            ind_strike = ind_spot
            dep_option_px = black_scholes_price(dep_spot, dep_strike, T, rf, dep_vol, dep_type)
            ind_option_px = black_scholes_price(ind_spot, ind_strike, T, rf, ind_vol, ind_type)
            dep_delta = black_scholes_delta(dep_spot, dep_strike, T, rf, dep_vol, dep_type)
            ind_delta = black_scholes_delta(ind_spot, ind_strike, T, rf, ind_vol, ind_type)

            try:
                sizing = size_log_spread_option_legs(
                    beta=float(row["beta"]),
                    dependent_spot=dep_spot,
                    independent_spot=ind_spot,
                    dependent_delta=dep_delta,
                    independent_delta=ind_delta,
                )
            except Exception as exc:
                skipped.append({
                    "date": date,
                    "pair": pair,
                    "reason": f"sizing_error:{type(exc).__name__}",
                })
                continue

            n_dep = sizing["dependent_contracts"]
            n_ind = sizing["independent_contracts"]
            premium = CONTRACT_MULTIPLIER * (
                n_dep * dep_option_px + n_ind * ind_option_px
            )

            if premium > cash:
                skipped.append({
                    "date": date,
                    "pair": pair,
                    "reason": "insufficient_cash",
                    "required_premium": float(premium),
                    "available_cash": float(cash),
                })
                continue

            cash -= premium
            open_positions[pair] = {
                "pair": pair,
                "dependent": dep,
                "independent": ind,
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "entry_date": date,
                "entry_spread": current_spread,
                "entry_z": current_z,
                "mu": float(row["mu"]),
                "direction": int(direction),
                "target_probability": float(target_probability),
                "convergence_horizon_trading_days": int(horizon),
                "probability_at_selected_horizon": float(signal.probability_at_selected_dte),
                "probability_at_max_horizon": float(signal.probability_at_max_horizon),
                "required_calendar_days": int(required_calendar_days),
                "expiry_date": expiry,
                "option_calendar_dte": int(calendar_dte),
                "dependent_option_type": dep_type,
                "independent_option_type": ind_type,
                "dependent_strike": dep_strike,
                "independent_strike": ind_strike,
                "entry_dependent_spot": dep_spot,
                "entry_independent_spot": ind_spot,
                "entry_dependent_volatility": dep_vol,
                "entry_independent_volatility": ind_vol,
                "entry_risk_free_rate": rf,
                "entry_dependent_delta": dep_delta,
                "entry_independent_delta": ind_delta,
                "dependent_contracts": int(n_dep),
                "independent_contracts": int(n_ind),
                "target_contract_ratio_ind_over_dep": sizing["target_contract_ratio_ind_over_dep"],
                "realized_contract_ratio_ind_over_dep": sizing["realized_contract_ratio_ind_over_dep"],
                "relative_hedge_error": sizing["relative_hedge_error"],
                "entry_dependent_option_price": dep_option_px,
                "entry_independent_option_price": ind_option_px,
                "entry_premium": float(premium),
            }

        # C. End-of-day portfolio mark.
        open_value = 0.0
        for pair, pos in open_positions.items():
            dep, ind = pos["dependent"], pos["independent"]
            dep_spot = float(test.at[date, dep])
            ind_spot = float(test.at[date, ind])
            dep_vol = float(vol.at[date, dep])
            ind_vol = float(vol.at[date, ind])
            rf = _risk_free_at(risk_free_rates, date)
            value, _, _ = _position_market_value(
                pos, date, dep_spot, ind_spot, dep_vol, ind_vol, rf
            )
            open_value += value

        equity_rows.append({
            "date": date,
            "cash": float(cash),
            "open_position_value": float(open_value),
            "equity": float(cash + open_value),
            "n_open_positions": int(len(open_positions)),
            "n_pairs_in_timeout": int(sum(
                pd.notna(v) and date < pd.Timestamp(v).normalize()
                for v in timeout_until.values()
            )),
        })

    # D. Force-close residual positions at end of test. No timeout is applied
    # afterward because there are no future entries.
    if len(test_dates):
        final_date = pd.Timestamp(test_dates[-1]).normalize()
        for pair in list(open_positions.keys()):
            pos = open_positions[pair]
            dep, ind = pos["dependent"], pos["independent"]
            dep_spot = float(test.at[final_date, dep])
            ind_spot = float(test.at[final_date, ind])
            dep_vol = float(vol.at[final_date, dep])
            ind_vol = float(vol.at[final_date, ind])
            rf = _risk_free_at(risk_free_rates, final_date)
            current_spread = float(spread_series[pair].loc[final_date])
            exit_value, dep_exit_px, ind_exit_px = _position_market_value(
                pos, final_date, dep_spot, ind_spot, dep_vol, ind_vol, rf
            )
            cash += exit_value
            pnl = float(exit_value - pos["entry_premium"])

            trades.append({
                **pos,
                "exit_date": final_date,
                "exit_reason": "end_of_test",
                "exit_spread": current_spread,
                "exit_dependent_spot": dep_spot,
                "exit_independent_spot": ind_spot,
                "exit_dependent_volatility": dep_vol,
                "exit_independent_volatility": ind_vol,
                "exit_risk_free_rate": rf,
                "exit_dependent_option_price": dep_exit_px,
                "exit_independent_option_price": ind_exit_px,
                "exit_value": float(exit_value),
                "pnl": pnl,
                "trade_return": float(pnl / pos["entry_premium"]),
                "holding_trading_days": int(
                    test_dates.get_loc(final_date) - test_dates.get_loc(pos["entry_date"])
                ),
                "holding_calendar_days": int((final_date - pos["entry_date"]).days),
                "consecutive_losses_after_exit": np.nan,
                "timeout_triggered": False,
                "timeout_until_after_exit": pd.NaT,
            })
            del open_positions[pair]

        if equity_rows:
            equity_rows[-1] = {
                "date": final_date,
                "cash": float(cash),
                "open_position_value": 0.0,
                "equity": float(cash),
                "n_open_positions": 0,
                "n_pairs_in_timeout": int(sum(
                    pd.notna(v) and final_date < pd.Timestamp(v).normalize()
                    for v in timeout_until.values()
                )),
            }

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)
    timeout_df = pd.DataFrame(timeout_events)
    state_df = pd.DataFrame({
        "pair": list(consecutive_losses.keys()),
        "consecutive_losses": list(consecutive_losses.values()),
        "timeout_until": [timeout_until[p] for p in consecutive_losses],
    })

    return {
        "trades": trades_df,
        "equity_curve": equity_df,
        "skipped_signals": skipped_df,
        "pair_parameters": params,
        "oos_ewma_volatility": vol,
        "timeout_events": timeout_df,
        "pair_timeout_state": state_df,
    }
