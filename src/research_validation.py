"""Frozen benchmark regression, raw-mean bootstrap and matched placebo statistics."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def aligned_returns(equity, benchmark, rates):
    # Benchmark must include the initial pre-OOS valuation date.
    idx = equity.index
    b = benchmark.reindex(idx)
    if b.isna().any() or (b <= 0).any() or not np.isfinite(b).all():
        raise ValueError('Frozen benchmark must cover every equity date including the initial date.')
    # Annual effective decimal rate known at start of return interval.
    rf = rates.reindex(idx,method='ffill').shift(1)
    d = pd.DataFrame({'strategy_return':equity.equity.pct_change(fill_method=None),
                      'market_return':b.pct_change(fill_method=None),
                      'rf_daily':(1+rf)**(1/252)-1}).iloc[1:]
    if not np.isfinite(d.to_numpy()).all():
        raise ValueError('Return alignment contains missing/nonfinite values.')
    d['strategy_excess']=d.strategy_return-d.rf_daily
    d['market_excess']=d.market_return-d.rf_daily
    return d


def market_regression(aligned, lags=5):
    if len(aligned) <= max(3,lags) or aligned.market_excess.std() < 1e-12:
        return {'status':'insufficient_market_variation','n_obs':len(aligned)}
    x=sm.add_constant(aligned.market_excess,has_constant='add')
    fit=sm.OLS(aligned.strategy_excess,x).fit(cov_type='HAC',cov_kwds={'maxlags':lags})
    p=float(fit.pvalues['const']); t=float(fit.tvalues['const'])
    # Degenerate all-zero/exact-fit response has no meaningful asymptotic test.
    if not np.isfinite([p,t]).all():
        return {'status':'degenerate_regression','n_obs':len(aligned)}
    out=dict(status='ok',alpha_daily=float(fit.params['const']),
             alpha_annualized_simple=float(252*fit.params['const']),
             alpha_HAC_se=float(fit.bse['const']),alpha_t_stat=t,
             alpha_p_value_two_sided=p,alpha_p_value_one_sided_positive=p/2 if t>0 else 1-p/2,
             market_beta=float(fit.params['market_excess']),r_squared=float(fit.rsquared),
             n_obs=len(aligned),hac_lags=lags,rf_convention='annual_effective_lagged_252',
             start=str(aligned.index[0]),end=str(aligned.index[-1]))
    return out


def bootstrap_mean(returns, blocks=(10,20,40), replications=10000, seed=42):
    x=np.asarray(returns,dtype=float); rows=[]
    if not np.isfinite(x).all(): raise ValueError('Nonfinite returns.')
    for b in blocks:
        if b > len(x):
            rows.append(dict(block_length=b,status='insufficient_observations')); continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(b)]))
        means=np.empty(replications)
        for i in range(0,replications,200):
            n=min(200,replications-i)
            starts=rng.integers(0,len(x)-b+1,size=(n,int(np.ceil(len(x)/b))))
            sample=x[(starts[:,:,None]+np.arange(b)).reshape(n,-1)[:,:len(x)]]
            means[i:i+n]=sample.mean(axis=1)
        lo,hi=np.quantile(means,[.025,.975])
        rows.append(dict(block_length=b,status='ok',n_replications=replications,
                         observed_mean=float(x.mean()),ci_low=float(lo),ci_high=float(hi),
                         estimand='unconditional_mean_raw_daily_return'))
    return pd.DataFrame(rows)


def placebo_samples(pool, n_pairs, n_portfolios, seed):
    """IID uniform subsets. Repeated subsets across draws are allowed and recorded.

    Avoids the old infinite loop when fewer than 100 unique subsets exist.
    """
    ordered=pool.sort_values('pair').reset_index(drop=True)
    if n_pairs <= 0 or n_pairs > len(ordered):
        raise ValueError('Invalid placebo portfolio size.')
    for j in range(n_portfolios):
        rng=np.random.default_rng(np.random.SeedSequence([seed,j]))
        yield j,ordered.iloc[rng.choice(len(ordered),n_pairs,replace=False)].sort_values('pair').reset_index(drop=True)


def portfolio_metrics(result, initial_capital):
    from src.backtest import backtest_summary
    m=backtest_summary(result['trades'],result['equity_curve'],initial_capital).to_dict()
    ret=result['equity_curve'].equity.pct_change(fill_method=None).dropna()
    m['daily_sharpe']=float(np.sqrt(252)*ret.mean()/ret.std()) if ret.std()>0 else None
    return m


def compare_placebos(actual, placebos):
    rows=[]
    for key in ['total_return','daily_sharpe','average_trade_return']:
        a=actual.get(key)
        values=pd.to_numeric(placebos[key],errors='coerce') if key in placebos else pd.Series(dtype=float)
        finite=values[np.isfinite(values)]
        if a is None or not np.isfinite(a) or len(finite)!=len(placebos) or len(finite)==0:
            rows.append(dict(metric=key,status='undefined_or_incomplete',n_draws=len(values))); continue
        rows.append(dict(metric=key,status='ok',actual=a,n_draws=len(finite),
                         p_value=(1+int((finite>=a).sum()))/(len(finite)+1),
                         percentile=100*float((finite<a).mean())))
    return pd.DataFrame(rows)
