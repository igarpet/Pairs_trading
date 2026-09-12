# Pairs Trading

Open Jupyter in this repository and use a Python 3.12 kernel. Install the pinned
libraries once with `python -m pip install -r requirements.txt` in your environment.

## Notebook order

Run each notebook from top to bottom:

1. 01_Stock_Data
2. 01_RF_Data
3. 02_Pair_Selection
4. 03_fractional_OU
5. 03_pair_eligibility
6. 04_convergence_signal
7. 05_volatility_model
8. 06_option_pricing
9. 07_backtest
10. 08_drawdown_diagnostics
11. 09_systematic_risk_diagnostics
12. 10_equilibrium_shift_diagnostics
13. 11_static_convergence_calibration
14. 12_alpha_validation

The first cell of each notebook contains imports only. Shared mathematical
functions are in `src/`; stage-specific calculations are visible in the notebooks.
No startup notebook, test folder, workflow or automatic executor is required.

## Files and settings

Module 01_Stock_Data downloads the S&P 500 constituents from the supplied GitHub
CSV and adjusted prices from Yahoo Finance. Edit START_DATE and END_DATE there
(defaults: 2016-01-01 through 2025-12-31). It performs the chronological 70/30 split.
Module 01_RF_Data downloads Yahoo's ^IRX Treasury yields and ^GSPC index
for the required dates. Both modules require internet access; neither loads the
old market-data files in `data/inputs/`.

Insert or edit filenames directly in each notebook's loading and saving cells.
Generated files default to the notebook
working folder: for example, `train_prices.parquet`, `trades.parquet` and
`equity_curve.parquet`. If you choose other filenames or locations, change the
matching reads in subsequent notebooks. Create any destination folders yourself.
There is no path discovery, directory creation, input copying or saved-settings layer.

Every notebook reads the shared defaults from `src/research_config.py`. Change
research settings there, then restart the kernels and rerun the notebooks in order.
Defaults are $100,000 starting equity, 5% of pre-entry equity per trade subject to
cash, no numeric open-position cap, and at most one position per pair. Simulation
uses 5,000 paths; Module 12 uses 100 placebos and 10,000 bootstrap replications.

Rerunning overwrites the named outputs. Module 12 recalculates every placebo and
builds its comparison from those calculations, not from old output files. Notebooks
are distributed without stored outputs; execution generates their displays.

Summary dictionaries use `.pkl` files through `pd.to_pickle` and `pd.read_pickle`.
Tables remain Parquet. Only load pickle files you generated or otherwise trust.
After pulling this version, rerun Modules 07–12 to generate the pickle outputs.

## Shared mathematics

The nine source files have direct roles:

- `research_config.py`: shared settings.
- `research_data.py`: input preparation and simple file helpers.
- `cointegration.py`: candidate pairs and statistical screening.
- `fractional_OU.py`: parameter estimation and pair eligibility.
- `convergence_signal.py`: fractional simulations and convergence probabilities.
- `market_math.py`: option pricing, volatility and spread calculations.
- `backtest.py`: position sizing and daily portfolio accounting.
- `forecast_calibration.py`: forecast outcomes and calibration.
- `research_validation.py`: portfolio statistics, alpha, bootstrap and placebos.

The backtest receives the shared settings object directly. There is no run registry,
run identifier or saved configuration to load. Later modules need the ordinary
data files produced by earlier modules and can execute in a fresh kernel.

See `METHODOLOGY.md` and `data/inputs/README.md` for model, data and inference
limitations. Options are model-valued, not historical listed-option quotes.
The downloaded constituents are a current snapshot, so historical survivorship
bias remains. Fresh downloads can also reflect vendor revisions.

## AI assistance

ChatGPT/Codex assisted with code generation, debugging, restructuring and results
review. This assistance must be disclosed under university requirements. The author
must personally understand and verify the submitted code, results and references;
the simplified structure does not change the scope of that assistance.
