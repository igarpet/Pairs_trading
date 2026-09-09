# Pairs Trading

The current local pipeline implements methodology version 2. Read [METHODOLOGY.md](METHODOLOGY.md)
for the exact selection, timing, sizing and statistical definitions. The previous 216-trade
thesis outputs remain in `data/processed`; they are historical and are NOT revised results.

## Install and test on Windows

Use Python 3.11 or 3.12 in a new environment, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

If using Anaconda, create a fresh Python 3.12 environment instead, activate it and use `python`
in place of `.\.venv\Scripts\python.exe`. Avoid installing these pinned dependencies over
an existing project environment. There are no live data downloads in the new runner.

## Run locally

The included prices already have historical universe-selection bias. To run this usable
but limited input, acknowledge it explicitly; that acknowledgement is saved in the manifest:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_research --run-dir runs/v2_local_01 --allow-legacy-universe
```

This executes formation, the main backtest and 100 placebos. It can take substantial time.
It rebuilds selection from prices; it does not reuse old cointegrated/eligible pair tables.
Do not lower the significance threshold automatically if the corrected screen selects few
or no pairs. Inspect `cointegration_audit.parquet` and discuss the implications first.

For staged execution, stop after formation to inspect the new eligible population:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_research --run-dir runs/v2_local_02 --allow-legacy-universe --stage formation
.\.venv\Scripts\python.exe -m scripts.run_research --run-dir runs/v2_local_02 --resume --stage backtest
.\.venv\Scripts\python.exe -m scripts.run_research --run-dir runs/v2_local_02 --resume --stage validation
```

An interrupted run can be resumed with `--resume` (same source code, environment and frozen
inputs). Validation resumes completed placebo checkpoints. A stage that failed before a
checkpoint is recomputed. Use a NEW directory when changing source code or configuration.

For different declared research settings, save a JSON file and pass `--config filename.json`
when creating a new run. Field names are in `src/research_config.py`. Example quick mechanics
check: `{"n_paths": 100, "n_placebos": 2, "bootstrap_replications": 200}`. Such small simulations
are for execution checking only; they do not replace the default statistical run.

For an improved data input, supply `--prices full_historical_panel.parquet --membership dated_members.csv`
and omit `--allow-legacy-universe`. Membership schema is in METHODOLOGY.md. Default benchmark
is the frozen S&P 500 price index in the repository. To use another series provide
`--benchmark file.parquet --benchmark-name "accurate series description"`. It must be a single
positive price column covering the entire OOS valuation index. `--rates` similarly takes a
single annual-effective decimal rate column. The code does not fabricate missing constituents,
option quotes, dividends or delisting prices.

## Outputs to review together

* `manifest.json`: methodology, configuration, limitations and file hashes.
* `cointegration_audit.parquet`, `fractional_ou_parameters.parquet`, `structural_results.parquet`:
  screening decisions and rebuilt model fits; `eligible_pool.parquet` and `eligible_pairs.parquet`
  distinguish the full eligible universe from the selected portfolio.
* `trades.parquet`, `equity_curve.parquet`, `skipped_signals.parquet`, `forecasts.parquet` and
  `backtest_summary.json`: timing, costs, sizing and portfolio outcomes; `equity_curve.png`.
* `forecast_calibration.parquet`, `calibration_summary.json` and
  `traded_forecast_calibration_summary.json`: path-based forecast diagnostics.
* `market_alignment.parquet`, `market_alpha.json`, `bootstrap_mean.csv`, `placebos.parquet`,
  `placebo_comparison.csv` and `placebo_design.json`: aligned validation with explicit definitions.

Send these new results for review before changing the thesis. The `runs/` directory is ignored
by git so local outputs do not accidentally replace historical evidence. Root historical
notebooks are now under `Archived/pre_v2_notebooks/`; use `00_run_research.ipynb` if you prefer
a notebook. Automated CI runs only the tests and never commits result files.
