from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from src.research_config import ResearchConfig
from src.research_data import prepare_prices
from src.forecast_calibration import label_forecasts,summarize_forecasts
from src.execution import run_backtest,budgeted_size
from src.research_validation import aligned_returns,bootstrap_mean,placebo_samples,compare_placebos
from src.backtest import black_scholes_price,backtest_summary
from src.Cointegration import screen_cointegration


def prices_fixture():
    dates=pd.bdate_range('2020-01-01',periods=130)
    rng=np.random.default_rng(7)
    b=100*np.exp(np.cumsum(rng.normal(0,.007,len(dates))))
    a=b*np.exp(.10+rng.normal(0,.01,len(dates)))
    p=pd.DataFrame({'A':a,'B':b},index=dates)
    params=pd.DataFrame([dict(pair='A-B',dependent='A',independent='B',mu=0.,kappa=.1,sigma=.02,hurst=.3,variance=.001)])
    c=pd.DataFrame([dict(pair='A-B',alpha=0.,beta=1.)])
    return p.iloc[:110],p.iloc[110:],params,c


def fake_signal(*args,**kw):
    return SimpleNamespace(selected_dte_trading_days=4,direction=1,
                           probability_at_selected_dte=.75,probability_at_max_horizon=.9),None


def test_next_close_budget_costs_and_exact_maturity(monkeypatch):
    monkeypatch.setattr('src.execution.calculate_convergence_signal',fake_signal)
    train,test,pairs,c=prices_fixture()
    r=run_backtest(train,test,pairs,c,.03)
    tr=r['trades'];assert len(tr)>0
    assert (tr.entry_date>tr.signal_date).all()
    idx=train.index.append(test.index)
    for row in tr.itertuples():
        assert idx.get_loc(row.entry_date)==idx.get_loc(row.signal_date)+1
        assert idx.get_loc(row.expiry_date)==idx.get_loc(row.signal_date)+4
        assert row.entry_dependent_spot==test.at[row.entry_date,'A']
        assert row.entry_premium<=row.entry_budget+1e-8
        assert row.relative_hedge_error<=.1+1e-10
        assert row.entry_cost>0
    eq=r['equity_curve']
    assert (eq.cash>=0).all()
    assert eq.equity.iloc[0]==100000
    assert eq.n_open_positions.max()<=10
    assert eq.open_position_value.iloc[-1]==0
    assert eq.equity.iloc[-1]==pytest.approx(100000+tr.pnl.sum())
    assert eq.equity.equals(eq.cash+eq.open_position_value)


def test_decisions_do_not_use_execution_day_close(monkeypatch):
    monkeypatch.setattr('src.execution.calculate_convergence_signal',fake_signal)
    train,test,pairs,c=prices_fixture()
    one=run_backtest(train,test,pairs,c,.03)
    changed=test.copy();changed.iloc[1,0]*=1.04
    two=run_backtest(train,changed,pairs,c,.03)
    a,b=one['trades'].iloc[0],two['trades'].iloc[0]
    assert a.signal_date==b.signal_date
    assert a.signal_spread==b.signal_spread
    assert a.signal_seed==b.signal_seed
    assert a.entry_dependent_spot!=b.entry_dependent_spot


def test_convergence_exit_is_delayed(monkeypatch):
    monkeypatch.setattr('src.execution.calculate_convergence_signal',fake_signal)
    train,test,pairs,c=prices_fixture()
    test=test.copy();test.iloc[2,0]=test.iloc[2,1]*.9
    r=run_backtest(train,test,pairs,c,.03)
    first=r['trades'].iloc[0]
    assert first.exit_reason=='convergence'
    assert first.exit_date==test.index[3]


def test_limit_and_order_independent_of_dataframe(monkeypatch):
    monkeypatch.setattr('src.execution.calculate_convergence_signal',fake_signal)
    train,test,pairs,c=prices_fixture()
    for f in [train,test]:
        f['C']=f.A*1.01;f['D']=f.B
    other=pairs.copy();other['pair']='C-D';other['dependent']='C';other['independent']='D'
    allpairs=pd.concat([pairs,other],ignore_index=True)
    c=pd.concat([c,pd.DataFrame([dict(pair='C-D',alpha=0.,beta=1.)])],ignore_index=True)
    a=run_backtest(train,test,allpairs,c,.03,max_open_pairs=1)
    b=run_backtest(train,test,allpairs.iloc[::-1],c,.03,max_open_pairs=1)
    pd.testing.assert_frame_equal(a['trades'],b['trades'])
    assert a['equity_curve'].n_open_positions.max()==1
    assert 'max_open_pairs' in a['skipped_signals'].reason.values


def test_sizing_matches_exhaustive_integer_reference():
    spots=np.array([120.,90.]);deltas=np.array([-.4,.6]);prices=np.array([3.,2.])
    s=budgeted_size(1.2,spots,deltas,prices,5000,.1,10,.65)
    ratio=1.2*48/54; costs=100*prices*1.001+.65
    feasible=[]
    for a in range(1,25):
        for b in range(1,25):
            err=abs(b/(a*ratio)-1);debit=a*costs[0]+b*costs[1]
            if err<=.1 and debit<=5000:feasible.append((debit,-err,-(a+b),a,b))
    best=max(feasible)
    assert (s['dependent_contracts'],s['independent_contracts'])==best[-2:]
    assert budgeted_size(1,spots,deltas,prices,1,.1,10,.65) is None


def test_no_trades_and_immediate_horizon(monkeypatch):
    def signal(*a,**k):
        x,_=fake_signal();x.selected_dte_trading_days=1;return x,None
    monkeypatch.setattr('src.execution.calculate_convergence_signal',signal)
    train,test,pairs,c=prices_fixture();r=run_backtest(train,test,pairs,c,.03)
    assert r['trades'].empty
    assert r['equity_curve'].equity.eq(100000).all()
    assert backtest_summary(r['trades'],r['equity_curve'],100000)['total_return']==0


def event_fixture():
    idx=pd.bdate_range('2024-01-01',periods=6)
    prices=pd.DataFrame({'A':np.exp([.3,.2,.1,-.1,.1,.1])*100,'B':100},index=idx)
    forecast=dict(pair='A-B',dependent='A',independent='B',alpha=0.,beta=1.,mu=0.,direction=1,
                  signal_date=idx[0],convergence_horizon_trading_days=4,probability_at_selected_horizon=.7,
                  exit_date=idx[1],exit_reason='expiry')
    return prices,forecast


def test_early_expiry_does_not_censor_observable_path():
    prices,f=event_fixture();out=label_forecasts(pd.DataFrame([f]),prices)
    assert out.event_status.iloc[0]=='success'
    assert out.first_crossing_date.iloc[0]==prices.index[3]
    assert out.complete_horizon_observed.iloc[0]


def test_incomplete_success_and_failure_have_distinct_status():
    prices,f=event_fixture();f['convergence_horizon_trading_days']=10
    hit=label_forecasts(pd.DataFrame([f]),prices)
    assert hit.event_status.iloc[0]=='success'
    assert not hit.complete_horizon_observed.iloc[0]
    assert summarize_forecasts(hit)['n_complete_horizon']==0
    prices.A=110
    no=label_forecasts(pd.DataFrame([f]),prices)
    assert no.event_status.iloc[0]=='censored'
    assert np.isnan(no.realized_within_selected_horizon.iloc[0])


def test_gap_cannot_be_bridged():
    prices,f=event_fixture();prices.iloc[1,0]=np.nan
    out=label_forecasts(pd.DataFrame([f]),prices)
    assert out.event_status.iloc[0]=='censored'


def test_horizon_inclusive_boundary():
    prices,f=event_fixture();f['convergence_horizon_trading_days']=2
    assert label_forecasts(pd.DataFrame([f]),prices).event_status.iloc[0]=='failure'
    f['convergence_horizon_trading_days']=3
    assert label_forecasts(pd.DataFrame([f]),prices).event_status.iloc[0]=='success'


def test_formation_ignores_future_availability():
    train,test,_,_=prices_fixture();p=pd.concat([train,test])
    c=replace(ResearchConfig(),train_fraction=110/130)
    t1,_,a1=prepare_prices(p,c,allow_legacy=True)
    changed=p.copy();changed.iloc[-1,0]=np.nan
    # Stops on unavailable OOS valuation, rather than changing the formation set.
    with pytest.raises(ValueError,match='OOS missing'):prepare_prices(changed,c,allow_legacy=True)
    changed=p.copy();changed.iloc[-1,0]*=2
    t2,_,a2=prepare_prices(changed,c,allow_legacy=True)
    pd.testing.assert_frame_equal(t1,t2);pd.testing.assert_frame_equal(a1,a2)
    with pytest.raises(ValueError,match='membership'):prepare_prices(p,c)


def test_membership_known_at_cutoff():
    train,test,_,_=prices_fixture();p=pd.concat([train,test]);p['C']=p.A
    m=pd.DataFrame(dict(ticker=['A','B','C'],member_from=['2010-01-01']*3,
                        member_to=[None]*3,known_at=['2010-01-01','2010-01-01','2030-01-01']))
    a,_,_=prepare_prices(p,replace(ResearchConfig(),train_fraction=110/130),m)
    assert list(a.columns)==['A','B']


def test_cointegration_fixed_orientation_and_unadjusted_metadata():
    from statsmodels.tsa.stattools import coint
    rng=np.random.default_rng(132)
    x=4+np.cumsum(rng.normal(0,.01,500));y=x+rng.normal(0,.004,500)
    p=pd.DataFrame({'A':np.exp(y),'B':np.exp(x)},index=pd.bdate_range('2020-01-01',periods=500))
    _,_,a=screen_cointegration(p,[('B','A')])
    stat,pval,_=coint(y,x,trend='c',autolag='aic')
    assert a.iloc[0].dependent=='A'
    assert a.iloc[0].pvalue==pytest.approx(pval)
    assert a.iloc[0].adf==pytest.approx(stat)
    assert a.iloc[0].adjusted_pvalue==pytest.approx(pval)
    assert a.iloc[0].multiplicity_method=='none'
    _,_,b=screen_cointegration(p,[('A','B'),('B','A')])
    pd.testing.assert_frame_equal(a,b)


def test_option_expiry_and_parity():
    for kind in ['call','put']:
        p=black_scholes_price(110,100,0,.03,.2,kind)
        assert p==(10 if kind=='call' else 0)
    c=black_scholes_price(110,100,1,.03,.2,'call')
    p=black_scholes_price(110,100,1,.03,.2,'put')
    assert c-p==pytest.approx(110-100*np.exp(-.03))


def test_alignment_uses_previous_rate_and_initial_equity():
    idx=pd.bdate_range('2024-01-01',periods=4)
    e=pd.DataFrame({'equity':[100,101,102,103]},index=idx)
    b=pd.Series([100,102,99,103],index=idx);rates=pd.Series([.01,.02,.03,.04],index=idx)
    a=aligned_returns(e,b,rates)
    assert len(a)==3
    assert a.rf_daily.iloc[0]==pytest.approx(1.01**(1/252)-1)
    with pytest.raises(ValueError,match='cover'):aligned_returns(e,b.iloc[1:],rates)


def test_placebo_small_pool_terminates_and_repeatability():
    p=pd.DataFrame({'pair':['B-C','A-B']})
    samples=list(placebo_samples(p,2,100,42))
    assert len(samples)==100
    assert all(list(s.pair)==['A-B','B-C'] for _,s in samples)
    out=compare_placebos({'total_return':.1},pd.DataFrame({'total_return':[.1,.2,0]}))
    assert out.iloc[0].p_value==.75


def test_bootstrap_reproducible_and_constant_return():
    a=bootstrap_mean(np.ones(60)*.01,blocks=(10,20,100),replications=100,seed=2)
    b=bootstrap_mean(np.ones(60)*.01,blocks=(10,20,100),replications=100,seed=2)
    pd.testing.assert_frame_equal(a,b)
    assert a.iloc[0].ci_low==pytest.approx(.01)
    assert a.iloc[-1].status=='insufficient_observations'


def test_legacy_calibration_fails_without_paths():
    from src.static_convergence_diagnostics import build_trade_calibration_table
    with pytest.raises(ValueError,match='requires prices'):build_trade_calibration_table(pd.DataFrame(),pd.DatetimeIndex([]))


def test_cli_manifest_rejects_changed_input(tmp_path,monkeypatch):
    from scripts import run_research as cli
    train,test,_,_=prices_fixture();p=pd.concat([train,test]);prices=tmp_path/'prices.parquet';p.to_parquet(prices)
    rates=tmp_path/'rates.parquet';pd.DataFrame({'rate':.03},index=p.index).to_parquet(rates)
    bench=tmp_path/'benchmark.parquet';p[['B']].to_parquet(bench)
    config=tmp_path/'c.json';config.write_text(json.dumps({'train_fraction':110/130}))
    monkeypatch.setattr(cli,'formation',lambda run,c,m: (run/'done.txt').write_text('done'))
    out=tmp_path/'run'
    args=['--run-dir',str(out),'--stage','formation','--prices',str(prices),'--rates',str(rates),'--benchmark',str(bench),'--config',str(config),'--allow-legacy-universe']
    cli.main(args)
    m=json.loads((out/'manifest.json').read_text())
    assert m['completed_stages']==['formation']
    cli.main(['--run-dir',str(out),'--resume','--stage','formation'])
    with (out/'inputs/prices.parquet').open('ab') as f:f.write(b'changed')
    with pytest.raises(SystemExit):cli.main(['--run-dir',str(out),'--resume'])


def test_raw_cointegration_selection_is_not_multiplied_by_family(monkeypatch):
    rng=np.random.default_rng(132);x=4+np.cumsum(rng.normal(0,.01,300));y=x+rng.normal(0,.02,300)
    p=pd.DataFrame({'A':np.exp(y),'B':np.exp(x),'C':np.exp(x+.1)},index=pd.bdate_range('2020-01-01',periods=300))
    # A raw p=.005 passes at 1%, whereas Holm over three pairs would fail.
    monkeypatch.setattr('src.Cointegration.coint',lambda *a,**k:(-4.,.005,[-4.,-3.,-2.]))
    monkeypatch.setattr('src.Cointegration.adfuller',lambda v,**k:(-3.,.5 if len(v)==300 else .001,0,0,{},0))
    selected,_,a=screen_cointegration(p,[('A','B')])
    assert a.n_family_tests.iloc[0]==3
    assert a.adjusted_pvalue.iloc[0]==pytest.approx(.005)
    assert a.multiplicity_method.iloc[0]=='none'
    assert selected.pair.tolist()==['A-B']
    rejected,_,_=screen_cointegration(p,[('A','B')],significance=.001)
    assert rejected.empty


def test_cached_forecasts_are_identical_to_uncached(monkeypatch):
    monkeypatch.setattr('src.execution.calculate_convergence_signal',fake_signal)
    train,test,pairs,c=prices_fixture();cache={}
    a=run_backtest(train,test,pairs,c,.03,signal_cache=cache)
    assert cache
    def fail(*a,**k):raise AssertionError('Identical forecast must be cached')
    monkeypatch.setattr('src.execution.calculate_convergence_signal',fail)
    b=run_backtest(train,test,pairs,c,.03,signal_cache=cache)
    pd.testing.assert_frame_equal(a['trades'],b['trades'])


def test_unmocked_full_cli_pipeline(tmp_path):
    from scripts.run_research import main
    from src.convergence_signal import simulate_unconditional_fou_paths
    idx=pd.bdate_range('2020-01-01',periods=400)
    x=4.5+np.cumsum(np.random.default_rng(51).normal(0,.015,400))
    noise=simulate_unconditional_fou_paths(0,0,.2,.015,.3,399,n_paths=1,seed=32)[0]
    p=pd.DataFrame({'A':np.exp(x+noise),'B':np.exp(x)},index=idx)
    prices=tmp_path/'prices.parquet';p.to_parquet(prices)
    rates=tmp_path/'rates.parquet';pd.DataFrame({'rate':.03},index=idx).to_parquet(rates)
    bench=tmp_path/'benchmark.parquet';p[['B']].to_parquet(bench)
    cfg=tmp_path/'cfg.json';cfg.write_text(json.dumps(dict(memory_window=10,structural_horizon=30,
        max_horizon_days=20,n_paths=100,n_placebos=2,bootstrap_replications=100,entry_z=.5)))
    out=tmp_path/'complete'
    main(['--run-dir',str(out),'--prices',str(prices),'--rates',str(rates),'--benchmark',str(bench),
          '--config',str(cfg),'--allow-legacy-universe'])
    m=json.loads((out/'manifest.json').read_text())
    assert m['completed_stages']==['formation','backtest','validation']
    tr=pd.read_parquet(out/'trades.parquet');eq=pd.read_parquet(out/'equity_curve.parquet')
    assert len(tr)>0
    assert eq.equity.iloc[-1]==pytest.approx(100000+tr.pnl.sum())
    assert m['output_table_rows']['trades.parquet']==len(tr)
    assert len(m['checkpoint_hashes'])==2
    main(['--run-dir',str(out),'--resume'])
