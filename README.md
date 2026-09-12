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

Insert or edit filenames directly in each notebook's loading and saving cells.
The supplied inputs are in `data/inputs/`. Generated files default to the notebook
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
builds its comparison from those calculations, not from old JSON files. Existing
notebook displays retain the evaluated results until replaced by execution.

See `METHODOLOGY.md` and `data/inputs/README.md` for model, data and inference
limitations. Options are model-valued, not historical listed-option quotes.

## AI assistance

ChatGPT/Codex assisted with code generation, debugging, restructuring and results
review. This assistance must be disclosed under university requirements. The author
must personally understand and verify the submitted code, results and references;
the simplified structure does not change the scope of that assistance.
