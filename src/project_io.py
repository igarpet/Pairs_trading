"""Plain shared files for the numbered notebook workflow."""
from pathlib import Path
import json
import shutil
import pandas as pd
from src.research_config import ResearchConfig

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'data' / 'processed' / 'current'


def initialize(config, prices, rates, benchmark, membership=None, allow_legacy=True):
    """Overwrite current input copies and settings; no run identity or history checks."""
    config.validate()
    files = {'prices.parquet': prices, 'rates.parquet': rates, 'benchmark.parquet': benchmark}
    if membership is not None:
        files['membership.csv'] = membership
    for path in files.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing input file: {path}')
    inputs = OUTPUT_DIR / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        if Path(path).resolve() != (inputs / name).resolve():
            shutil.copyfile(path, inputs / name)
    if membership is None:
        (inputs / 'membership.csv').unlink(missing_ok=True)
    (OUTPUT_DIR / 'settings.json').write_text(json.dumps({
        'config': config.to_dict(), 'allow_legacy_universe': allow_legacy,
        'input_sources': {k: str(Path(v).resolve()) for k, v in files.items()}
    }, indent=2) + '\n')
    print(f'Files saved to {OUTPUT_DIR}')


def load_config():
    path = OUTPUT_DIR / 'settings.json'
    if not path.exists():
        raise FileNotFoundError('Run 01_Stock_Data.ipynb first to save the settings and price split.')
    return ResearchConfig(**json.loads(path.read_text())['config']).validate()


def load_frame(name):
    path = OUTPUT_DIR / (name + '.parquet')
    if not path.exists():
        raise FileNotFoundError(f'{path.name} is missing. Run the notebook that produces it first; see 00_START_HERE.ipynb.')
    return pd.read_parquet(path)


def save_frame(name, frame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_DIR / (name + '.parquet'))


def save_json(name, value):
    from scripts.run_research import write_json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / (name + '.json'), value)
