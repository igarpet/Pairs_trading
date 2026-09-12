# Pairs Trading

Open Jupyter in this repository with a Python 3.12 kernel and install the pinned
libraries with `python -m pip install -r requirements.txt`.

## Notebook order

Run the notebooks from top to bottom:

1. `01_Stock_Data.ipynb` — constituents, stock prices, benchmark, risk-free rate and 70/30 split
2. `02_Pair_Selection.ipynb`
3. `03_fractional_OU.ipynb`
4. `03_pair_eligibility.ipynb`
5. `04_convergence_signal.ipynb`
6. `05_volatility_model.ipynb`
7. `06_option_pricing.ipynb`
8. `07_backtest.ipynb`
9. `08_drawdown_diagnostics.ipynb`
10. `09_systematic_risk_diagnostics.ipynb`
11. `10_equilibrium_shift_diagnostics.ipynb`
12. `11_static_convergence_calibration.ipynb`
13. `12_alpha_validation.ipynb`

Module 01 is the single market-data entry point. It downloads the current S&P 500
constituent list, adjusted stock prices, `^GSPC` benchmark prices and `^IRX` Treasury
yields. It then creates the 70/30 chronological split directly in the notebook. Assets
must have at most 5% missing observations in formation, a valid first formation price and
complete out-of-sample prices. Formation gaps are forward-filled from earlier observations.
The full-sample availability requirement is a documented selection limitation.

Module 01 saves `train_prices.parquet`, `test_prices.parquet`, `benchmark_prices.parquet`,
`risk_free_rates.parquet`, `availability.parquet` and `constituents.csv` for later modules.
The constituent list is a current snapshot, so historical survivorship bias remains.

Shared settings are in `src/research_config.py`. Generated files are written directly to
the notebook working folder. There is no run registry, saved-run layer, path discovery or
separate validation framework.

## Source modules

The simplified `src/` folder contains the mathematics and reusable execution logic:

- `research_config.py`: shared parameters.
- `cointegration.py`: return correlations, ADF/Engle-Granger screening and hedge ratios.
- `fractional_OU.py`: fOU estimation and pair eligibility.
- `convergence_signal.py`: fractional Gaussian simulation and first-passage probabilities.
- `market_math.py`: Black-Scholes, EWMA volatility, spreads and shared market calculations.
- `backtest.py`: option sizing and daily portfolio simulation.
- `research_validation.py`: performance, alpha, bootstrap, placebo and forecast calibration statistics.

The notebooks contain no explicit validation method calls or deliberate error raises. Model
selection rules and trading branches remain because they are part of the research methodology.

See `METHODOLOGY.md` for model, execution and inference assumptions. Options are synthetic
model values rather than historical listed-option quotes.

## AI assistance

ChatGPT/Codex assisted with code generation, debugging, restructuring and results review.
This assistance should be disclosed according to university requirements. The author remains
responsible for understanding and verifying the submitted code, results and references.
