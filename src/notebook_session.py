"""Shared saved-run state for independent numbered Jupyter modules.

Only Module 01 chooses a run/configuration. Later notebooks read the saved
selection automatically, so restarting a kernel does not lose dependencies.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import subprocess
import pandas as pd
from scripts.run_research import ROOT,sha,write_json,environment,source_hashes
from src.research_config import ResearchConfig

ACTIVE=ROOT/'runs'/'active_notebook_run.json'
ORDER=['01_stock','01_rf','02_pairs','03_fou','03_eligibility','04_signal','05_volatility',
       '06_options','07_backtest','08_drawdown','09_market','10_equilibrium','11_calibration','12_alpha']


def notebook_hashes():
    # Saved plots and execution counts may change without changing the method.
    import hashlib
    return {p.name:hashlib.sha256(json.dumps([
        (c['cell_type'],c['source']) for c in json.loads(p.read_text(encoding='utf-8'))['cells']
    ],sort_keys=True).encode()).hexdigest() for p in ROOT.glob('*.ipynb') if p.name[:2].isdigit() and p.name[:2]!='00'}


class NotebookSession:
    def __init__(self,run):
        self.run=Path(run).resolve()
        self.manifest=self.run/'manifest.json'
        if not self.manifest.exists():raise FileNotFoundError('Run 01_Stock_Data.ipynb first.')
        m=self._read()
        if m.get('interface')!='numbered_notebooks':
            raise ValueError('This is not a notebook run. Choose a new RUN_NAME in Module 01.')
        if m['source_hashes']!=source_hashes() or m['environment']!=environment():
            raise ValueError('Code or environment changed. Restart the kernel, then use a new RUN_NAME in Module 01.')
        if m['notebook_sources']!=notebook_hashes():
            raise ValueError('Notebook source changed since this run began. Choose a new RUN_NAME in Module 01.')
        for name,h in {**m['input_hashes'],**m.get('output_hashes',{}),**m.get('checkpoint_hashes',{})}.items():
            if not (self.run/name).exists() or sha(self.run/name)!=h:
                raise ValueError(f'Run file changed or missing: {name}. Use a new RUN_NAME.')
        self.config=ResearchConfig(**m['config']).validate()

    def _read(self):return json.loads(self.manifest.read_text(encoding='utf-8'))

    @classmethod
    def create(cls,run_name,config,prices,rates,benchmark,membership=None,
               allow_legacy=True,benchmark_name='S&P 500 price index ^GSPC'):
        config.validate()
        if Path(run_name).name!=run_name or run_name in ('','.','..') or any(x in run_name for x in '/\\'):
            raise ValueError('RUN_NAME must be a simple folder name, for example jupyter_v2_01.')
        run=ROOT/'runs'/run_name
        if run.exists():
            session=cls(run)
            if json.loads(json.dumps(config.to_dict()))!=session._read()['config']:
                raise ValueError('Settings changed. Use a new RUN_NAME.')
            old=session._read()
            requested={'prices.parquet':prices,'rates.parquet':rates,'benchmark.parquet':benchmark}
            if membership is not None:requested['membership.csv']=membership
            if set(requested)!=set(old['input_sources']) or any(str(Path(v).resolve())!=old['input_sources'][k] or sha(v)!=old['input_hashes']['inputs/'+k] for k,v in requested.items()):
                raise ValueError('Input files changed. Use a new RUN_NAME.')
            if old['allow_legacy_universe']!=allow_legacy or old['benchmark_name']!=benchmark_name:
                raise ValueError('Data conventions changed. Use a new RUN_NAME.')
        else:
            if membership is None and not allow_legacy:
                raise ValueError('Supply historical membership or acknowledge the existing universe.')
            files={'prices.parquet':prices,'rates.parquet':rates,'benchmark.parquet':benchmark}
            if membership is not None:files['membership.csv']=membership
            for path in files.values():
                if not Path(path).is_file():raise FileNotFoundError(path)
            (run/'inputs').mkdir(parents=True)
            for name,path in files.items():shutil.copyfile(path,run/'inputs'/name)
            try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            except (OSError,subprocess.CalledProcessError):commit='unknown'
            write_json(run/'manifest.json',dict(methodology_version=2,interface='numbered_notebooks',
                created_utc=datetime.now(timezone.utc).isoformat(),code_commit=commit,
                source_hashes=source_hashes(),notebook_sources=notebook_hashes(),environment=environment(),
                config=config.to_dict(),allow_legacy_universe=allow_legacy,
                universe_status='legacy_preselected_survivorship_and_availability_bias' if membership is None else 'user_supplied_dated_membership',
                benchmark_name=benchmark_name,option_data='synthetic_adjusted_spot_European_q0',
                input_sources={k:str(Path(v).resolve()) for k,v in files.items()},
                input_hashes={'inputs/'+k:sha(run/'inputs'/k) for k in files},
                completed_modules=[],completed_stages=[],output_hashes={}))
            session=cls(run)
        ACTIVE.parent.mkdir(parents=True,exist_ok=True)
        write_json(ACTIVE,{'run_dir':str(run.resolve())})
        print(f'Active notebook run: {run.name}')
        return session

    @classmethod
    def active(cls):
        if not ACTIVE.exists():raise FileNotFoundError('Run 01_Stock_Data.ipynb first to select a run.')
        return cls(json.loads(ACTIVE.read_text())['run_dir'])

    def begin(self,module,requires=()):
        m=self._read()
        missing=[x for x in requires if x not in m['completed_modules']]
        if missing:raise ValueError('Run these modules first: '+', '.join(missing))
        if module not in ORDER:raise ValueError(module)
        # If repeating an upstream module, require downstream notebooks again.
        m['completed_modules']=[x for x in m['completed_modules'] if ORDER.index(x)<ORDER.index(module)]
        m['active_module']=module;m['module_status']='running'
        write_json(self.manifest,m)
        print(f'Module {module} | {self.run.name}')

    def frame(self,name):return pd.read_parquet(self.run/(name+'.parquet'))
    def save(self,name,frame):
        path=self.run/(name+'.parquet');frame.to_parquet(path)
        self._record(path)
    def json(self,name,value):
        path=self.run/(name+'.json');write_json(path,value);self._record(path)
    def _record(self,path):
        m=self._read();m.setdefault('output_hashes',{})[str(path.relative_to(self.run))]=sha(path)
        write_json(self.manifest,m)
    def finish(self,module):
        m=self._read()
        if m.get('active_module')!=module:raise ValueError('Module session changed; run this notebook from its first cell.')
        m['completed_modules']=list(dict.fromkeys(m['completed_modules']+[module]))
        m['module_status']='completed'
        # Include files written by the shared CLI statistical routines.
        m['output_hashes']={p.name:sha(p) for p in self.run.iterdir() if p.is_file() and p.name!='manifest.json'}
        import pyarrow.parquet as pq
        m['output_table_rows']={p.name:pq.read_metadata(p).num_rows for p in self.run.glob('*.parquet')}
        write_json(self.manifest,m)
        print(f'Completed {module}. Outputs saved in {self.run}')
