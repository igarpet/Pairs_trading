"""Canonical local CLI. No network calls and no writes to historical outputs."""
from __future__ import annotations
import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def source_hashes():
    paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+[ROOT/'requirements.txt',ROOT/'METHODOLOGY.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths if p.exists()}


def environment():
    return {'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','pyarrow']}}


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


def formation(run,c,manifest):
    p=pd.read_parquet(run/'inputs/prices.parquet')
    m=pd.read_csv(run/'inputs/membership.csv') if (run/'inputs/membership.csv').exists() else None
    train,test,availability=prepare_prices(p,c,m,manifest['allow_legacy_universe'])
    for name,frame in [('train_prices',train),('test_prices',test),('availability',availability)]:save_frame(run,name,frame)
    # Validate benchmark/rates before expensive selection and simulations.
    rf=read_series(run/'inputs/rates.parquet'); benchmark=read_series(run/'inputs/benchmark.parquet')
    if (rf<=-1).any():raise ValueError('Risk-free rates must be annual effective decimals above -1.')
    aligned_returns(pd.DataFrame({'equity':c.initial_capital},index=test.index),benchmark,rf)
    candidates=generate_candidate_pairs(correlation_matrix(compute_returns(train)),c.correlation_neighbors)
    print(f'Formation: {len(train)} sessions, {len(train.columns)} assets, {len(candidates)} candidates',flush=True)
    selected,spreads,audit=screen_cointegration(train,candidates,c.cointegration_alpha,c.integration_alpha)
    save_frame(run,'cointegration_audit',audit);save_frame(run,'cointegrated_pairs',selected)
    print(f'{len(selected)} pairs pass Holm-adjusted cointegration and integration screens',flush=True)
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


def validation(run,c):
    equity=pd.read_parquet(run/'equity_curve.parquet')
    aligned=aligned_returns(equity,read_series(run/'inputs/benchmark.parquet'),read_series(run/'inputs/rates.parquet'))
    save_frame(run,'market_alignment',aligned)
    write_json(run/'market_alpha.json',market_regression(aligned,c.hac_lags))
    bootstrap_mean(aligned.strategy_return,c.bootstrap_blocks,c.bootstrap_replications,c.seed).to_csv(run/'bootstrap_mean.csv',index=False)
    pool=pd.read_parquet(run/'eligible_pool.parquet');top=pd.read_parquet(run/'eligible_pairs.parquet')
    actual=json.loads((run/'backtest_summary.json').read_text())
    rows=[]
    signal_cache={}  # Same frozen run; reusable forecasts do not depend on holdings or portfolio membership.
    for j,sample in placebo_samples(pool,len(top),c.n_placebos,c.seed):
        # Each completed portfolio is checkpointed for an interrupted long run.
        out=run/f'placebo_{j:04d}.json'
        pairs=sample.pair.tolist()
        if out.exists():
            row=json.loads(out.read_text())
            if row['sampled_pairs']!=pairs:raise ValueError('Placebo checkpoint sample mismatch.')
        else:
            row={**portfolio_metrics(load_backtest(run,c,sample,signal_cache),c.initial_capital),
                 'placebo_id':j,'sampled_pairs':pairs,'seed':c.seed}
            write_json(out,row)
            checkpoint_manifest=json.loads((run/'manifest.json').read_text())
            checkpoint_manifest.setdefault('checkpoint_hashes',{})[out.name]=sha(out)
            write_json(run/'manifest.json',checkpoint_manifest)
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


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['all','formation','backtest','validation'],default='all')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prices',type=Path,default=ROOT/'data/processed/prices.parquet')
    p.add_argument('--rates',type=Path,default=ROOT/'data/processed/risk_free_rates.parquet')
    p.add_argument('--benchmark',type=Path,default=ROOT/'data/processed/systematic_risk/sp500_prices.parquet')
    p.add_argument('--benchmark-name',default='S&P 500 price index ^GSPC')
    p.add_argument('--membership',type=Path)
    p.add_argument('--allow-legacy-universe',action='store_true')
    p.add_argument('--config',type=Path,help='JSON overrides to ResearchConfig; saved once per run')
    args=p.parse_args(argv);run=args.run_dir.resolve()
    legacy=(ROOT/'data').resolve()
    if run==ROOT or run==legacy or legacy in run.parents:
        p.error('Use a NEW runs/<id> directory; never write into historical data.')
    manifest_path=run/'manifest.json'
    if args.resume:
        manifest=json.loads(manifest_path.read_text())
        if args.config: p.error('Resume uses frozen config; start a new run to change it.')
        if manifest['source_hashes']!=source_hashes() or manifest['environment']!=environment():
            p.error('Code/environment changed since run creation; use a new run directory.')
        for name,checksum in manifest['input_hashes'].items():
            if sha(run/name)!=checksum:p.error(f'Input changed: {name}')
        for name,checksum in {**manifest.get('output_hashes',{}),**manifest.get('checkpoint_hashes',{})}.items():
            if sha(run/name)!=checksum:p.error(f'Completed output changed: {name}')
        c=ResearchConfig(**manifest['config']).validate()
    else:
        if args.stage not in ['all','formation']:p.error('Start with formation or all; later stages need --resume.')
        overrides=json.loads(args.config.read_text()) if args.config else {}
        c=ResearchConfig(**overrides).validate()
        if not args.membership and not args.allow_legacy_universe:p.error('Supply --membership or acknowledge --allow-legacy-universe.')
        if run.exists():p.error('Run directory already exists; use --resume or a new directory.')
        for path in [args.prices,args.rates,args.benchmark]+([args.membership] if args.membership else []):
            if not path.is_file():p.error(f'Missing input: {path}')
        (run/'inputs').mkdir(parents=True)
        files={'prices.parquet':args.prices,'rates.parquet':args.rates,'benchmark.parquet':args.benchmark}
        if args.membership:files['membership.csv']=args.membership
        for name,path in files.items():shutil.copyfile(path,run/'inputs'/name)
        try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        except (OSError,subprocess.CalledProcessError):commit='unknown'
        manifest={'methodology_version':2,'created_utc':datetime.now(timezone.utc).isoformat(),
                  'code_commit':commit,'source_hashes':source_hashes(),'environment':environment(),
                  'config':c.to_dict(),'allow_legacy_universe':args.allow_legacy_universe,
                  'universe_status':'legacy_preselected_survivorship_and_availability_bias' if not args.membership else 'user_supplied_dated_membership',
                  'benchmark_name':args.benchmark_name,'option_data':'synthetic_adjusted_spot_European_q0',
                  'input_sources':{name:str(path.resolve()) for name,path in files.items()},
                  'input_hashes':{'inputs/'+name:sha(run/'inputs'/name) for name in files},'completed_stages':[]}
        write_json(manifest_path,manifest)
    stages=['formation','backtest','validation'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage in manifest['completed_stages']:continue
        previous={'backtest':'formation','validation':'backtest'}.get(stage)
        if previous and previous not in manifest['completed_stages']:p.error(f'Complete {previous} first.')
        manifest['active_stage']=stage;manifest['status']='running';write_json(manifest_path,manifest)
        try:
            formation(run,c,manifest) if stage == 'formation' else globals()[stage](run,c)
        except Exception as exc:
            manifest['checkpoint_hashes']=json.loads(manifest_path.read_text()).get('checkpoint_hashes',{})
            manifest['status']='failed';manifest['error']=str(exc);write_json(manifest_path,manifest)
            raise
        if stage=='validation':
            manifest['checkpoint_hashes']=json.loads(manifest_path.read_text()).get('checkpoint_hashes',{})
        manifest['completed_stages'].append(stage);manifest['status']='completed';manifest.pop('error',None)
        manifest['output_hashes']={str(f.relative_to(run)):sha(f) for f in sorted(run.glob('*')) if f.is_file() and f.name!='manifest.json'}
        import pyarrow.parquet as pq
        manifest['output_table_rows']={f.name:pq.read_metadata(f).num_rows for f in run.glob('*.parquet')}
        write_json(manifest_path,manifest)
        print(f'Completed {stage}: {run}',flush=True)

if __name__=='__main__':main()
