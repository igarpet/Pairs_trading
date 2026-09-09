# Pairs Trading

The current local pipeline implements methodology version 2. Read [METHODOLOGY.md](METHODOLOGY.md)
for the exact selection, timing, sizing and statistical definitions. The previous 216-trade
thesis outputs remain in `data/processed`; they are historical and are NOT revised results.

## Run in Jupyter one module at a time

Open `00_START_HERE.ipynb` for one-time dependency installation and the module guide.
Then open each notebook below and choose **Run All**, in this order:

1. `01_Stock_Data.ipynb` — choose the run name and settings, load and split prices.
2. `01_RF_Data.ipynb` — load frozen risk-free rates and check benchmark coverage.
3. `02_Pair_Selection.ipynb` — correlation candidates and corrected cointegration screening.
4. `03_fractional_OU.ipynb` — fit all selected spreads and inspect Hurst estimates.
5. `03_pair_eligibility.ipynb` — structural forecasts, eligible pool and Top 40.
6. `04_convergence_signal.ipynb` — inspect conditional forecasts on a preview date.
7. `05_volatility_model.ipynb` — lagged EWMA volatility.
8. `06_option_pricing.ipynb` — preview next-close synthetic option terms and sizing.
9. `07_backtest.ipynb` — execute the full daily backtest and save trades/equity.
10. `08_drawdown_diagnostics.ipynb` — drawdown episodes and trade-level losses.
11. `09_systematic_risk_diagnostics.ipynb` — aligned market regression and rolling exposure.
12. `10_equilibrium_shift_diagnostics.ipynb` — frozen spread displacement diagnostics.
13. `11_static_convergence_calibration.ipynb` — path-based first-passage calibration.
14. `12_alpha_validation.ipynb` — consistent alpha, bootstrap and matched placebos.

Each notebook contains readable calculation cells, tables and plots. Intermediate files
are saved automatically, so each module can use its own kernel. Run one module at a time.
Settings and the run name are chosen only in Module 01; the active run is remembered under
`runs/active_notebook_run.json`. Default output folder: `runs/jupyter_v2_01`.

Use the existing input defaults for your first run. The historical universe limitation is
explicitly acknowledged in Module 01 and saved in the manifest. To supply improved data,
change its price/membership paths. Their schema is described in METHODOLOGY.md.

Modules 04 and 06 are inspection snapshots, not substitutes for the daily simulation.
Module 07 applies the same signal/pricing functions throughout the test sample. Modules
08–11 load results without repeating the backtest. Module 12 alone simulates placebo
portfolios; rerun that notebook after an interruption to reuse completed checkpoints.

To change model settings, use a NEW RUN_NAME in Module 01. Restart the kernel after code or
package updates. Saved input/source hashes prevent accidental mixing of experiments.
Repeating an upstream notebook invalidates later completion flags, so continue through
later notebooks again. Old artifacts remain available for inspection but cannot satisfy
a missing prerequisite. No code touches the old thesis results in data/processed.

If no pairs survive the corrected statistical screen, inspect the saved audit rather than
loosening thresholds to force an attractive result. Empty one-date previews are different:
they are valid, and you can continue to the next module.

The original pre-v2 notebooks remain in Archived/pre_v2_notebooks for historical reference.
The notebooks at the repository root are the updated executable versions. The optional
command-line runner remains available via `python -m scripts.run_research --help`; notebook
runs use their own saved module state and should be continued through the notebooks.

## Validation and reviewing results

Install `requirements-notebooks.txt` and run `python -m pytest -q` to test both the shared
methodology code and notebook execution. CI executes a small synthetic example through
all numbered notebooks with separate Jupyter kernels. This checks mechanics, not historical
performance. The default full 5,000-path experiment and 100 placebos are for local execution.

Send the entire `runs/jupyter_v2_01` output folder for review before changing the thesis.
It includes the manifest, formation audit, fitted parameters, trades, equity, diagnostics,
calibration and statistical comparisons. Notebook output displays are generated from those
same files; no earlier thesis numbers are hardcoded.
