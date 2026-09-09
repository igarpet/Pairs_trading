"""Next-close execution of lagged signals, with exact forecast horizons."""
from math import sqrt
import numpy as np
import pandas as pd
from src.backtest import (black_scholes_price, black_scholes_delta, _prepare_pair_parameters,
    compute_log_spread, precompute_oos_ewma_volatility, _risk_free_at,
    _has_converged, _stable_seed, option_types_from_spread_direction, CONTRACT_MULTIPLIER)
from src.convergence_signal import calculate_convergence_signal


def budgeted_size(beta, spots, deltas, prices, budget, max_error, slippage_bps, commission):
    """Exhaustive integer search; maximize invested premium subject to error cap.

    For each dependent count, the largest feasible independent count maximizes
    cost for that count. Ties prefer lower hedge error, then fewer contracts.
    """
    costs = CONTRACT_MULTIPLIER * np.asarray(prices) * (1 + slippage_bps/10000) + commission
    exposure = np.abs(deltas) * np.asarray(spots)
    if not np.isfinite(costs).all() or (costs <= 0).any() or beta <= 0 or (exposure <= 0).any():
        raise ValueError('Invalid sizing inputs.')
    ratio = beta * exposure[0] / exposure[1]
    best = None
    for nd in range(1, max(0, int(np.floor((budget-costs[1])/costs[0])))+1):
        low = max(1, int(np.ceil((1-max_error)*ratio*nd - 1e-12)))
        high = min(int(np.floor((1+max_error)*ratio*nd + 1e-12)),
                   int(np.floor((budget-costs[0]*nd)/costs[1] + 1e-12)))
        if high < low:
            continue
        error = abs(high/(nd*ratio)-1)
        debit = nd*costs[0] + high*costs[1]
        rank = (debit, -error, -(nd+high))
        if best is None or rank > best[0]:
            best = (rank, nd, high, error)
    if best is None:
        return None
    _, nd, ni, error = best
    return dict(dependent_contracts=nd, independent_contracts=ni,
                relative_hedge_error=float(error), target_contract_ratio_ind_over_dep=float(ratio),
                realized_contract_ratio_ind_over_dep=ni/nd)


def entry_option_terms(instruction, date, prices, volatility, risk_free_rates):
    """Shared synthetic ATM terms for Module 06 and the actual execution engine."""
    date=pd.Timestamp(date)
    if date <= pd.Timestamp(instruction['signal_date']):
        raise ValueError('Execution must follow the signal date.')
    T=(pd.Timestamp(instruction['expiry_date'])-date).days/365
    if T <= 0:raise ValueError('No remaining option maturity at execution.')
    spots=[float(prices.at[date,instruction[l]]) for l in ('dependent','independent')]
    vols=[float(volatility.at[date,instruction[l]]) for l in ('dependent','independent')]
    rf=_risk_free_at(risk_free_rates,date)
    types=option_types_from_spread_direction(instruction['direction'])
    px=[black_scholes_price(s,s,T,rf,v,k) for s,v,k in zip(spots,vols,types)]
    deltas=[black_scholes_delta(s,s,T,rf,v,k) for s,v,k in zip(spots,vols,types)]
    return spots,vols,rf,types,px,deltas


def run_backtest(train_prices, test_prices, eligible_pairs, cointegrated_pairs, risk_free_rates,
                 initial_capital=100000., entry_z=1.5, target_probability=.70,
                 memory_window=60, max_horizon_days=126, n_paths=5000, ewma_lambda=.94,
                 seed=42, max_open_pairs=10, premium_budget_fraction=.05,
                 max_hedge_error=.10, slippage_bps=10., commission_per_contract=.65, signal_cache=None):
    from src.research_config import ResearchConfig
    ResearchConfig(initial_capital=initial_capital,entry_z=entry_z,target_probability=target_probability,
                   memory_window=memory_window,max_horizon_days=max_horizon_days,n_paths=n_paths,
                   ewma_lambda=ewma_lambda,seed=seed,max_open_pairs=max_open_pairs,
                   premium_budget_fraction=premium_budget_fraction,max_hedge_error=max_hedge_error,
                   slippage_bps=slippage_bps,commission_per_contract=commission_per_contract).validate()
    train, test = train_prices.copy(), test_prices.copy()
    for frame in (train,test):
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError('Prices require nonempty, sorted, unique session indices.')
        if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise ValueError('Missing/nonpositive prices: supply an explicit market-data policy; no OOS filling.')
    if train.index[-1] >= test.index[0]:
        raise ValueError('Training and test periods must not overlap.')
    params = _prepare_pair_parameters(eligible_pairs, cointegrated_pairs).sort_values('pair').reset_index(drop=True)
    if (params.beta <= 0).any():
        raise ValueError('This strategy supports positive-beta pairs only.')
    if not np.isfinite(params[['alpha','beta','mu','kappa','sigma','hurst','variance']]).all().all():
        raise ValueError('Nonfinite pair parameters.')
    if ((params.kappa <= 0) | (params.kappa >= 2) | (params.sigma <= 0) | (params.variance <= 0) | (params.hurst <= 0) | (params.hurst >= .5)).any():
        raise ValueError('Requires positive variance/sigma, 0<H<.5 and stable daily Euler 0<kappa<2.')
    full = pd.concat([train,test])
    dates = full.index
    vol = precompute_oos_ewma_volatility(train,test,ewma_lambda)
    spreads = {r.pair: compute_log_spread(full,r.dependent,r.independent,r.alpha,r.beta) for r in params.itertuples()}
    cash, opened, pending, trades, skipped, forecasts = float(initial_capital), {}, {}, [], [], []
    equity = []  # First OOS close is the all-cash initial valuation; first fill is the next close.
    slip = slippage_bps / 10000

    def quote(pos, date):
        T = max((pos['expiry_date']-date).days,0)/365
        rf = _risk_free_at(risk_free_rates,date)
        px = [black_scholes_price(test.at[date,pos[leg]],pos[leg+'_strike'],T,rf,
                                 vol.at[date,pos[leg]],pos[leg+'_option_type'])
              for leg in ('dependent','independent')]
        gross = CONTRACT_MULTIPLIER*sum(pos[leg+'_contracts']*p for leg,p in zip(('dependent','independent'),px))
        return float(gross),px

    for date in test.index:
        i = dates.get_loc(date)
        # Exit instructions are based on yesterday's close, except known expiry/end.
        for pair in list(opened):
            pos = opened[pair]
            expired = date >= pos['expiry_date']
            last = date == test.index[-1]
            lagged_cross = _has_converged(spreads[pair].iloc[i-1],pos['mu'],pos['direction'])
            if expired or last or (date > pos['entry_date'] and lagged_cross):
                gross,px = quote(pos,date)
                fees = commission_per_contract*(pos['dependent_contracts']+pos['independent_contracts'])
                exit_cost = 0. if expired else gross*slip + fees
                value = gross-exit_cost
                cash += value
                trades.append({**pos,'exit_date':date,'exit_reason':'expiry' if expired else ('end_of_test' if last else 'convergence'),
                               'exit_spread':float(spreads[pair].loc[date]),'exit_value':value,
                               'exit_model_value':gross,'exit_cost':exit_cost,
                               'exit_dependent_option_price':px[0],'exit_independent_option_price':px[1],
                               'pnl':value-pos['entry_premium'],'trade_return':value/pos['entry_premium']-1,
                               'holding_trading_days':int(i-dates.get_loc(pos['entry_date'])),
                               'holding_calendar_days':int((date-pos['entry_date']).days)})
                del opened[pair]
        # Fixed budget based on equity before today's entries, including current marks.
        nav = cash + sum(quote(p,date)[0] for p in opened.values())
        for pair in sorted(pending):
            instruction = pending[pair]
            reason = None
            if pair in opened: reason = 'already_open'
            elif len(opened) >= max_open_pairs: reason = 'max_open_pairs'
            elif date >= instruction['expiry_date']: reason = 'no_remaining_maturity'
            elif date == test.index[-1]: reason = 'last_session'
            if reason:
                skipped.append(dict(date=date,pair=pair,reason=reason,signal_date=instruction['signal_date']))
                continue
            pos = instruction.copy()
            # Fixed option types and horizon come from t; spots and ATM strikes from execution t+1.
            spots,vols,rf,types,px,deltas = entry_option_terms(pos,date,test,vol,risk_free_rates)
            budget = max(0.,min(cash,nav*premium_budget_fraction))
            sizing = budgeted_size(pos['beta'],spots,deltas,px,budget,max_hedge_error,slippage_bps,commission_per_contract)
            if sizing is None:
                skipped.append(dict(date=date,pair=pair,signal_date=pos['signal_date'],reason='no_feasible_integer_hedge',budget=budget))
                continue
            pos.update(sizing)
            gross = CONTRACT_MULTIPLIER*sum(pos[l+'_contracts']*p for l,p in zip(('dependent','independent'),px))
            entry_cost = gross*slip+commission_per_contract*(pos['dependent_contracts']+pos['independent_contracts'])
            debit = gross+entry_cost
            pos.update(entry_date=date,entry_spread=float(spreads[pair].loc[date]),entry_premium=debit,
                       entry_model_value=gross,entry_cost=entry_cost,entry_budget=budget,
                       entry_risk_free_rate=rf,option_calendar_dte=int((pos['expiry_date']-date).days),
                       execution_convention='next_close',option_model='European_BS_synthetic_q0',contract_multiplier=100)
            for j,leg in enumerate(('dependent','independent')):
                pos.update({leg+'_strike':spots[j],leg+'_option_type':types[j],
                            'entry_'+leg+'_spot':spots[j],'entry_'+leg+'_volatility':vols[j],
                            'entry_'+leg+'_option_price':px[j],'entry_'+leg+'_delta':deltas[j]})
            cash -= debit
            opened[pair] = pos
        pending = {}
        # Forecast at today's close; never trade it at today's price.
        if date != test.index[-1]:
            for row in params.itertuples():
                if row.pair in opened: continue
                history = spreads[row.pair].loc[:date]
                z = (history.iloc[-1]-row.mu)/sqrt(row.variance)
                if abs(z) < entry_z or len(history) < memory_window+1: continue
                signal_seed = _stable_seed(seed,row.pair,date)
                key = (row.pair,str(date),row.mu,row.kappa,row.sigma,row.hurst,row.variance,
                       target_probability,entry_z,memory_window,max_horizon_days,n_paths,signal_seed,
                       history.iloc[-(memory_window+1):].to_numpy().tobytes())
                sig = signal_cache.get(key) if signal_cache is not None else None
                if sig is None:
                    sig,_ = calculate_convergence_signal(history,row.mu,row.kappa,row.sigma,row.hurst,row.variance,
                        target_probability=target_probability,entry_z=entry_z,memory_window=memory_window,
                        max_horizon_days=max_horizon_days,n_paths=n_paths,seed=signal_seed)
                    if signal_cache is not None:
                        signal_cache[key] = sig
                h = sig.selected_dte_trading_days
                if h is None: continue
                rec = dict(pair=row.pair,dependent=row.dependent,independent=row.independent,
                           alpha=row.alpha,beta=row.beta,mu=row.mu,signal_date=date,
                           signal_spread=float(history.iloc[-1]),entry_z=float(z),direction=int(sig.direction),
                           convergence_horizon_trading_days=int(h),target_probability=target_probability,
                           probability_at_selected_horizon=float(sig.probability_at_selected_dte),
                           probability_at_max_horizon=float(sig.probability_at_max_horizon),signal_seed=signal_seed)
                rec['forecast_id'] = f'{row.pair}|{date.date()}'
                forecasts.append(rec.copy())
                if i+h >= len(dates):
                    skipped.append(dict(date=date,pair=row.pair,reason='horizon_beyond_test'))
                    continue
                rec['expiry_date'] = dates[i+h]
                rec['forecast_horizon_date'] = dates[i+h]
                if h <= 1:
                    skipped.append(dict(date=date,pair=row.pair,reason='no_remaining_maturity'))
                    continue
                pending[row.pair] = rec
        mark = sum(quote(p,date)[0] for p in opened.values())
        if cash < -1e-7 or len(opened)>max_open_pairs:
            raise AssertionError('Cash or concurrency invariant failed.')
        equity.append(dict(date=date,cash=cash,open_position_value=mark,equity=cash+mark,n_open_positions=len(opened)))
    trade_columns = ['pair','entry_date','exit_date','exit_reason','pnl','trade_return','entry_premium']
    return dict(trades=pd.DataFrame(trades) if trades else pd.DataFrame(columns=trade_columns),
                equity_curve=pd.DataFrame(equity).set_index('date'),skipped_signals=pd.DataFrame(skipped),
                pair_parameters=params,oos_ewma_volatility=vol,forecasts=pd.DataFrame(forecasts))
