# Pairs Trading

## Run in Jupyter

Use a Python 3.12 kernel. Open `00_START_HERE.ipynb` for dependency installation,
then choose **Run All** in each notebook below, one at a time.

1. `01_Stock_Data.ipynb` — choose settings, load and split prices.
2. `01_RF_Data.ipynb` — load frozen risk-free rates and check benchmark coverage.
3. `02_Pair_Selection.ipynb` — correlation candidates and raw Engle–Granger screening at 1%.
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


## Simple shared files

All intermediate tables and results go to **data/processed/current/**. Each module
loads the files it needs and overwrites its own outputs. Settings are chosen in
01_Stock_Data and saved as ordinary settings.json for subsequent modules.
There are no run names, manifests, source/environment hashes or completion flags.
You can close Jupyter, restart a kernel, edit a notebook, and continue using the saved files.

After changing inputs, settings or upstream calculations, rerun the affected module and
its downstream modules to refresh their files. Existing files are not automatically
checked for freshness. Run one notebook at a time. Copy the current folder elsewhere
if you want to keep a particular set of results before overwriting them.

Module 01 copies the selected local inputs into current/inputs and overwrites those
copies each time. The original data/processed source files and old thesis results remain
available. Missing required files produce a normal message to run the producing module.

Modules 04 and 06 are one-date previews; empty previews are valid. Module 07 computes
signals and option prices throughout the test period. Modules 08–11 read the saved
backtest. Module 12 recomputes every placebo draw on each execution, using the current
files and settings. It never resumes old checkpoints.

The included stock universe remains historically filtered and options remain model-valued.
Selection uses raw Engle–Granger p-values at 1%, with no Holm adjustment. See
METHODOLOGY.md for the research definitions. No surviving pairs is a possible research
outcome, not a file-management error.

## Tests and results

Install requirements-notebooks.txt and run `python -m pytest -q`. CI executes a synthetic
example through all numbered notebooks with separate kernels. The full historical
experiment remains for local execution. Send data/processed/current/ for review before
updating the thesis. Saved notebook displays may show earlier results until rerun.

The optional `python -m scripts.run_research` uses the same fixed folder. Its `--stage`
option supports formation, backtest or validation without run identifiers or resume flags.
Archived notebooks and workflows are historical references, not current entry points.
