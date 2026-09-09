"""Shared research functions and optional fixed-folder command-line execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# Supports both python -m scripts.run_research and python scripts/run_research.py.
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.research_config import ResearchConfig
from src.research_data import normalize_frame, prepare_prices
from src.Cointegration import screen_cointegration
from src.Pair_Selection import compute_returns, correlation_matrix, generate_candidate_pairs
from src.fractional_OU import fit_cointegrated_pairs_fractional_ou
from src.pair_eligibility import filter_antipersistent_pairs, compute_structural_t70, select_top_pairs_by_structural_t70
from src.backtest import run_walk_forward_backtest
from src.forecast_calibration import label_forecasts, summarize_forecasts
from src.research_validation import aligned_returns,market_regression,bootstrap_mean,placebo_samples,portfolio_metrics,compare_placebos


def write_json(path, value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
        if isinstance(x,np.generic):return x.item()
        return x
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,default=str,allow_nan=False)+'\n')
    tmp.replace(path)


def save_frame(run,name,frame):
    frame.to_parquet(run/(name+'.parquet'))


def read_series(path):
    x=normalize_frame(pd.read_parquet(path))
    if x.shape[1]!=1:raise ValueError(f'{path} must contain exactly one data column.')
    s=x.iloc[:,0].astype(float)
    if not np.isfinite(s).all():raise ValueError(f'{path} contains nonfinite values.')
    return s


def backtest_kwargs(c):
    names=['initial_capital','entry_z','target_probability','memory_window','max_horizon_days',
           'n_paths','ewma_lambda','seed','max_open_pairs','premium_budget_fraction','max_hedge_error',
           'slippage_bps','commission_per_contract']
    return {k:getattr(c,k) for k in names}


def formation(run,c,allow_legacy=True):
    p=pd.read_parquet(run/'inputs/prices.parquet')
    m=pd.read_csv(run/'inputs/membership.csv') if (run/'inputs/membership.csv').exists() else None
    train,test,availability=prepare_prices(p,c,m,allow_legacy)
    for name,frame in [('train_prices',train),('test_prices',test),('availability',availability)]:save_frame(run,name,frame)
    # Validate benchmark/rates before expensive selection and simulations.
    rf=read_series(run/'inputs/rates.parquet'); benchmark=read_series(run/'inputs/benchmark.parquet')
    if (rf<=-1).any():raise ValueError('Risk-free rates must be annual effective decimals above -1.')
    aligned_returns(pd.DataFrame({'equity':c.initial_capital},index=test.index),benchmark,rf)
    candidates=generate_candidate_pairs(correlation_matrix(compute_returns(train)),c.correlation_neighbors)
    print(f'Formation: {len(train)} sessions, {len(train.columns)} assets, {len(candidates)} candidates',flush=True)
    selected,spreads,audit=screen_cointegration(train,candidates,c.cointegration_alpha,c.integration_alpha)
    save_frame(run,'cointegration_audit',audit);save_frame(run,'cointegrated_pairs',selected)
    print(f'{len(selected)} pairs pass raw-p-value cointegration and integration screens',flush=True)
    if selected.empty:raise ValueError('No selected pairs. Audit saved; do not relax thresholds based on OOS results.')
    fou,fit_audit=fit_cointegrated_pairs_fractional_ou(spreads,selected,return_audit=True)
    save_frame(run,'fou_fit_audit',fit_audit)
    if fou.empty:raise ValueError('No valid fOU fits.')
    save_frame(run,'fractional_ou_parameters',fou)
    pool=filter_antipersistent_pairs(fou)
    if pool.empty:raise ValueError('No stable anti-persistent fOU fits.')
    structural=compute_structural_t70(pool,starting_z=c.entry_z,target_probability=c.target_probability,
        max_horizon_days=c.structural_horizon,n_paths=c.n_paths,seed=c.seed)
    save_frame(run,'structural_results',structural)
    eligible=structural.loc[structural.structural_t70.notna()].copy()
    save_frame(run,'eligible_pool',eligible)
    top=select_top_pairs_by_structural_t70(eligible,c.top_n)
    save_frame(run,'eligible_pairs',top)
    if top.empty:raise ValueError('No finite structural horizons.')
    # Generated formation tables use exactly the fitted and selected populations.
    fou.hurst.describe().to_csv(run/'hurst_summary.csv')
    top.structural_t70.describe().to_csv(run/'selected_horizon_summary.csv')


def load_backtest(run,c,pairs=None,signal_cache=None):
    rd=lambda name:pd.read_parquet(run/(name+'.parquet'))
    return run_walk_forward_backtest(rd('train_prices'),rd('test_prices'),
        rd('eligible_pairs') if pairs is None else pairs,rd('cointegrated_pairs'),
        read_series(run/'inputs/rates.parquet'),signal_cache=signal_cache,**backtest_kwargs(c))


def backtest(run,c):
    result=load_backtest(run,c)
    for name,frame in result.items():save_frame(run,name,frame)
    write_json(run/'backtest_summary.json',portfolio_metrics(result,c.initial_capital))
    labels=label_forecasts(result['forecasts'],pd.read_parquet(run/'test_prices.parquet'))
    save_frame(run,'forecast_calibration',labels)
    write_json(run/'calibration_summary.json',summarize_forecasts(labels))
    trade_forecasts=result['forecasts']
    if not result['trades'].empty:
        trade_forecasts=trade_forecasts.loc[trade_forecasts.forecast_id.isin(result['trades'].forecast_id)]
    else: trade_forecasts=trade_forecasts.iloc[:0]
    write_json(run/'traded_forecast_calibration_summary.json',summarize_forecasts(label_forecasts(trade_forecasts,pd.read_parquet(run/'test_prices.parquet'))))
    # Save useful plots locally; no fixed numbers in captions.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));result['equity_curve'].equity.plot(ax=ax)
    ax.set(title='Revised synthetic option portfolio',ylabel='Model equity');fig.tight_layout();fig.savefig(run/'equity_curve.png',dpi=200);plt.close(fig)


def validate_market(run,c):
    equity=pd.read_parquet(run/'equity_curve.parquet')
    aligned=aligned_returns(equity,read_series(run/'inputs/benchmark.parquet'),read_series(run/'inputs/rates.parquet'))
    save_frame(run,'market_alignment',aligned)
    write_json(run/'market_alpha.json',market_regression(aligned,c.hac_lags))

def validate_bootstrap(run,c):
    aligned=pd.read_parquet(run/'market_alignment.parquet')
    bootstrap_mean(aligned.strategy_return,c.bootstrap_blocks,c.bootstrap_replications,c.seed).to_csv(run/'bootstrap_mean.csv',index=False)

def validate_placebos(run,c):
    pool=pd.read_parquet(run/'eligible_pool.parquet');top=pd.read_parquet(run/'eligible_pairs.parquet')
    actual=json.loads((run/'backtest_summary.json').read_text())
    rows=[]
    signal_cache={}  # Reuse forecasts only within this invocation.
    # Recompute on every invocation; stale results never enter a new comparison.
    for old in run.glob('placebo_[0-9][0-9][0-9][0-9].json'):
        old.unlink()
    for j,sample in placebo_samples(pool,len(top),c.n_placebos,c.seed):
        pairs=sample.pair.tolist()
        row={**portfolio_metrics(load_backtest(run,c,sample,signal_cache),c.initial_capital),
             'placebo_id':j,'sampled_pairs':pairs,'seed':c.seed}
        write_json(run/f'placebo_{j:04d}.json',row)
        rows.append(row)
        print(f'Placebo {j+1}/{c.n_placebos} completed',flush=True)
    frame=pd.DataFrame(rows)
    save_frame(run,'placebos',frame)
    compare_placebos(actual,frame).to_csv(run/'placebo_comparison.csv',index=False)
    write_json(run/'placebo_design.json',{'null':'uniform subsets of finite-structural-horizon pool',
        'pool_size':len(pool),'portfolio_size':len(top),'n_draws':c.n_placebos,
        'duplicate_draws_allowed':True,'degenerate_membership_null':len(pool)==len(top),
        'execution_order':'alphabetical pair ID for baseline and every placebo',
        'scope':'conditional descriptive reference; does not adjust all research specification searches'})


def validation(run,c):
    validate_market(run,c)
    validate_bootstrap(run,c)
    validate_placebos(run,c)


def main(argv=None):
    from src import project_io as io
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','formation','backtest','validation'],default='all')
    p.add_argument('--prices',type=Path,default=ROOT/'data/processed/prices.parquet')
    p.add_argument('--rates',type=Path,default=ROOT/'data/processed/risk_free_rates.parquet')
    p.add_argument('--benchmark',type=Path,default=ROOT/'data/processed/systematic_risk/sp500_prices.parquet')
    p.add_argument('--membership',type=Path)
    p.add_argument('--config',type=Path,help='JSON settings overrides for formation')
    args=p.parse_args(argv)
    if args.stage in ['all','formation']:
        c=ResearchConfig(**(json.loads(args.config.read_text()) if args.config else {})).validate()
        io.initialize(c,args.prices,args.rates,args.benchmark,args.membership)
    else:
        if args.config:p.error('Change settings in formation, then rerun the later stages.')
        c=io.load_config()
    for stage in (['formation','backtest','validation'] if args.stage=='all' else [args.stage]):
        globals()[stage](io.OUTPUT_DIR,c)
        print(f'Completed {stage}: {io.OUTPUT_DIR}',flush=True)

if __name__=='__main__':main()
