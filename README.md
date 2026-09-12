# Pairs Trading

Cointegrated equity spreads, fractional OU forecasts and synthetic European options.

## Run in Jupyter

Use Python 3.12. Open `00_START_HERE.ipynb` and install `requirements.txt` in the
selected kernel. Use **Run All** in each notebook in this order:

| Notebook | Purpose / principal output |
|---|---|
| 01_Stock_Data | Settings and formation/test prices |
| 01_RF_Data | Rates and benchmark coverage |
| 02_Pair_Selection | Correlation candidates and cointegration audit |
| 03_fractional_OU | Hurst and fractional OU parameters |
| 03_pair_eligibility | Structural horizons and selected pairs |
| 04_convergence_signal | One-date forecast preview |
| 05_volatility_model | Lagged EWMA volatility |
| 06_option_pricing | One-date option and sizing preview |
| 07_backtest | Trades, forecasts, equity and summary |
| 08_drawdown_diagnostics | Drawdowns and pair contributions |
| 09_systematic_risk_diagnostics | Market alpha and rolling exposure |
| 10_equilibrium_shift_diagnostics | Spread displacement diagnostics |
| 11_static_convergence_calibration | Forecast calibration |
| 12_alpha_validation | 100 placebos and block-bootstrap intervals |

Each notebook can run in a fresh kernel. They exchange ordinary files in
`data/processed/current/`; Module 01 saves settings there. After changing settings
or upstream data, rerun dependent notebooks in order. Modules 04 and 06 are previews;
an empty preview is valid. Module 12 recomputes its draws. Copy outputs elsewhere
before overwriting them if needed. There are no named-run or manifest gates.

## Files

* Root notebooks: stage-by-stage analysis, plots and stage-specific functions.
* `src/`: shared functions grouped by purpose. The strategy and placebo portfolios
  use the same simulation and execution functions.
* `data/inputs/`: supplied prices, rates and benchmark.
* `data/processed/current/`: generated tables, JSON summaries and placebo records;
  created on execution and excluded from Git.
* `tests/`: accounting, timing, statistics and separate-kernel notebook checks.
* `execute_notebooks.py`: optional executor, also used by GitHub Actions.

For example, `python execute_notebooks.py 07_backtest.ipynb` executes Module 07
after its prerequisites exist and saves outputs even on failure. Start the
**Research notebooks** workflow manually in GitHub Actions to execute tests,
01–06, 07, 08–11 and finally 12 in dependent jobs with artifacts at each stage.
The test workflow runs automatically on pushes and pull requests.

## Settings and verification

Defaults are in `src/research_config.py`; select overrides in Module 01.
Starting equity is **$100,000**. Each entry's all-in debit is capped at **5% of
pre-entry equity and available cash**. There is **no numeric open-pair cap**,
with at most one position per pair, no borrowing and no continuous rebalancing.
Signals use 5,000 paths; validation uses 100 placebos and 10,000 bootstrap
replications per block length. See `METHODOLOGY.md` for seeds and assumptions.
`requirements.txt` pins versions; Module 00 displays installed versions.

Run `python -m pytest -q` for verification. Notebook integration tests use small
synthetic inputs without altering research defaults. Retained notebook displays
contain evaluated historical results; rerunning replaces their displays. Full raw
outputs can be regenerated or downloaded from research-workflow artifacts.
The supplied universe has survivorship/availability limitations and options are
model-valued. A positive return alone does not establish significant alpha.

## AI assistance and author review

ChatGPT/Codex assisted with code generation, debugging, restructuring, tests and
results review. This assistance must be disclosed under the university requirements;
repository organisation does not establish independent authorship. The student
must personally review and understand the submitted code, verify results and
references, and accurately describe the scope of assistance. Automated checks do
not substitute for that review.
