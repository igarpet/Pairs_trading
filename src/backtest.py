from math import sqrt

import numpy as np
import pandas as pd

from src.convergence_signal import calculate_convergence_signal
from src.market_math import (
    CONTRACT_MULTIPLIER,
    black_scholes,
    compute_log_spread,
    ewma_volatility,
    option_types_from_spread_direction,
    prepare_pair_parameters,
    risk_free_at,
    stable_seed,
)


def budgeted_size(beta, spots, deltas, option_prices, budget, max_error, slippage_bps, commission):
    costs = CONTRACT_MULTIPLIER * np.asarray(option_prices) * (1 + slippage_bps / 10000) + commission
    exposure = np.abs(deltas) * np.asarray(spots)
    ratio = beta * exposure[0] / exposure[1]
    best = None

    for dependent_contracts in range(
        1, max(0, int(np.floor((budget - costs[1]) / costs[0]))) + 1
    ):
        low = max(1, int(np.ceil((1 - max_error) * ratio * dependent_contracts - 1e-12)))
        high = min(
            int(np.floor((1 + max_error) * ratio * dependent_contracts + 1e-12)),
            int(np.floor((budget - costs[0] * dependent_contracts) / costs[1] + 1e-12)),
        )
        if high < low:
            continue

        error = abs(high / (dependent_contracts * ratio) - 1)
        debit = dependent_contracts * costs[0] + high * costs[1]
        rank = (debit, -error, -(dependent_contracts + high))
        if best is None or rank > best[0]:
            best = (rank, dependent_contracts, high, error)

    if best is None:
        return None

    _, dependent_contracts, independent_contracts, error = best
    return {
        "dependent_contracts": dependent_contracts,
        "independent_contracts": independent_contracts,
        "relative_hedge_error": float(error),
        "target_contract_ratio_ind_over_dep": float(ratio),
        "realized_contract_ratio_ind_over_dep": independent_contracts / dependent_contracts,
    }


def entry_option_terms(instruction, date, prices, volatility, risk_free_rates):
    date = pd.Timestamp(date)
    T = (pd.Timestamp(instruction["expiry_date"]) - date).days / 365
    legs = ("dependent", "independent")
    spots = [float(prices.at[date, instruction[leg]]) for leg in legs]
    vols = [float(volatility.at[date, instruction[leg]]) for leg in legs]
    rf = risk_free_at(risk_free_rates, date)
    types = option_types_from_spread_direction(instruction["direction"])
    terms = [black_scholes(s, s, T, rf, v, typ) for s, v, typ in zip(spots, vols, types)]
    option_prices = [term[0] for term in terms]
    deltas = [term[1] for term in terms]
    return spots, types, option_prices, deltas


def run_backtest(
    train_prices,
    test_prices,
    eligible_pairs,
    cointegrated_pairs,
    risk_free_rates,
    config,
    signal_cache=None,
):
    train = train_prices.copy()
    test = test_prices.copy()
    train.index = pd.to_datetime(train.index).tz_localize(None).normalize()
    test.index = pd.to_datetime(test.index).tz_localize(None).normalize()

    params = prepare_pair_parameters(eligible_pairs, cointegrated_pairs).sort_values("pair")
    full = pd.concat([train, test])
    dates = full.index
    volatility = ewma_volatility(train, test, config.ewma_lambda)
    spreads = {
        row.pair: compute_log_spread(full, row.dependent, row.independent, row.alpha, row.beta)
        for row in params.itertuples()
    }

    cash = float(config.initial_capital)
    opened = {}
    pending = {}
    trades = []
    forecasts = []
    equity = []
    slip = config.slippage_bps / 10000

    def quote(position, date):
        T = max((position["expiry_date"] - date).days, 0) / 365
        rf = risk_free_at(risk_free_rates, date)
        option_prices = [
            black_scholes(
                test.at[date, position[leg]],
                position[leg + "_strike"],
                T,
                rf,
                volatility.at[date, position[leg]],
                position[leg + "_option_type"],
            )[0]
            for leg in ("dependent", "independent")
        ]
        return float(CONTRACT_MULTIPLIER * sum(
            position[leg + "_contracts"] * price
            for leg, price in zip(("dependent", "independent"), option_prices)
        ))

    for date in test.index:
        i = dates.get_loc(date)

        for pair in list(opened):
            position = opened[pair]
            expired = date >= position["expiry_date"]
            last = date == test.index[-1]
            previous_spread = spreads[pair].iloc[i - 1]
            converged = (
                previous_spread <= position["mu"]
                if position["direction"] > 0
                else previous_spread >= position["mu"]
            )

            if expired or last or (date > position["entry_date"] and converged):
                gross = quote(position, date)
                fees = config.commission_per_contract * (
                    position["dependent_contracts"] + position["independent_contracts"]
                )
                exit_cost = 0.0 if expired else gross * slip + fees
                value = gross - exit_cost
                cash += value
                trades.append(
                    {
                        **position,
                        "exit_date": date,
                        "exit_reason": "expiry" if expired else "end_of_test" if last else "convergence",
                        "exit_spread": float(spreads[pair].loc[date]),
                        "exit_value": value,
                        "exit_cost": exit_cost,
                        "pnl": value - position["entry_premium"],
                        "trade_return": value / position["entry_premium"] - 1,
                    }
                )
                del opened[pair]

        nav = cash + sum(quote(position, date) for position in opened.values())
        for pair in sorted(pending):
            instruction = pending[pair]
            if pair in opened:
                continue
            if config.max_open_pairs is not None and len(opened) >= config.max_open_pairs:
                continue
            if date >= instruction["expiry_date"] or date == test.index[-1]:
                continue

            position = instruction.copy()
            spots, types, option_prices, deltas = entry_option_terms(
                position, date, test, volatility, risk_free_rates
            )
            budget = max(0.0, min(cash, nav * config.premium_budget_fraction))
            sizing = budgeted_size(
                position["beta"],
                spots,
                deltas,
                option_prices,
                budget,
                config.max_hedge_error,
                config.slippage_bps,
                config.commission_per_contract,
            )
            if sizing is None:
                continue

            position.update(sizing)
            gross = CONTRACT_MULTIPLIER * sum(
                position[leg + "_contracts"] * price
                for leg, price in zip(("dependent", "independent"), option_prices)
            )
            entry_cost = gross * slip + config.commission_per_contract * (
                position["dependent_contracts"] + position["independent_contracts"]
            )
            debit = gross + entry_cost
            position.update(
                entry_date=date,
                entry_spread=float(spreads[pair].loc[date]),
                entry_premium=debit,
                entry_cost=entry_cost,
                entry_budget=budget,
            )

            for j, leg in enumerate(("dependent", "independent")):
                position.update(
                    {
                        leg + "_strike": spots[j],
                        leg + "_option_type": types[j],
                    }
                )

            cash -= debit
            opened[pair] = position

        pending = {}

        if date != test.index[-1]:
            for row in params.itertuples():
                if row.pair in opened:
                    continue

                history = spreads[row.pair].loc[:date]
                z = (history.iloc[-1] - row.mu) / sqrt(row.variance)
                if abs(z) < config.entry_z or len(history) < config.memory_window + 1:
                    continue

                key = (row.pair, pd.Timestamp(date))
                signal = signal_cache.get(key) if signal_cache is not None else None
                if signal is None:
                    signal, _ = calculate_convergence_signal(
                        history,
                        row.mu,
                        row.kappa,
                        row.sigma,
                        row.hurst,
                        row.variance,
                        config.target_probability,
                        config.entry_z,
                        config.memory_window,
                        config.max_horizon_days,
                        config.n_paths,
                        seed=stable_seed(config.seed, row.pair, date),
                    )
                    if signal_cache is not None:
                        signal_cache[key] = signal

                horizon = signal["selected_dte_trading_days"]
                if horizon is None:
                    continue

                record = {
                    "pair": row.pair,
                    "dependent": row.dependent,
                    "independent": row.independent,
                    "alpha": row.alpha,
                    "beta": row.beta,
                    "mu": row.mu,
                    "signal_date": date,
                    "signal_spread": float(history.iloc[-1]),
                    "entry_z": float(z),
                    "direction": int(signal["direction"]),
                    "convergence_horizon_trading_days": int(horizon),
                    "target_probability": config.target_probability,
                    "probability_at_selected_horizon": float(signal["probability_at_selected_dte"]),
                    "probability_at_max_horizon": float(signal["probability_at_max_horizon"]),
                    "signal_seed": stable_seed(config.seed, row.pair, date),
                    "forecast_id": f"{row.pair}|{date.date()}",
                }
                forecasts.append(record.copy())

                if horizon <= 1 or i + horizon >= len(dates):
                    continue

                record["expiry_date"] = dates[i + horizon]
                record["forecast_horizon_date"] = dates[i + horizon]
                pending[row.pair] = record

        mark = sum(quote(position, date) for position in opened.values())
        equity.append(
            {
                "date": date,
                "cash": cash,
                "open_position_value": mark,
                "equity": cash + mark,
                "n_open_positions": len(opened),
            }
        )

    trade_columns = [
        "forecast_id",
        "pair",
        "entry_date",
        "exit_date",
        "exit_reason",
        "pnl",
        "trade_return",
        "entry_premium",
    ]
    return {
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(columns=trade_columns),
        "equity_curve": pd.DataFrame(equity).set_index("date"),
        "pair_parameters": params.reset_index(drop=True),
        "oos_ewma_volatility": volatility,
        "forecasts": pd.DataFrame(forecasts),
    }
