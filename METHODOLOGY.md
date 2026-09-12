# Synthetic pairs experiment

The numbered notebooks are the analysis entry points. Shared functions in `src/` hold the
mathematics and execution rules used by the strategy and matched placebos.

## Data and inference scope

Module 01 is the single market-data module. It downloads the current S&P 500 constituents
CSV, Yahoo Finance adjusted stock prices, the `^GSPC` price index and the `^IRX` 13-week
Treasury yield. The current constituent snapshot introduces survivorship bias because it is
not historical membership. Fresh downloads may also reflect vendor revisions.

The first 70% of market sessions are formation and the remaining 30% are out of sample.
Asset missingness and initial availability are assessed only in formation. Formation gaps
are forward-filled from earlier observations; the out-of-sample panel is left unchanged.
Prices are adjusted synthetic underlying levels, not historical listed-option spots.

## Pair selection and fractional model

- Each asset's ten highest return correlations define the unordered candidate family.
- Each pair is oriented once alphabetically, with the first ticker dependent.
- ADF with a constant and AIC lags must fail to reject a level unit root at 5% and reject a
  unit root in first differences at 5%. This is a screening convention, not proof of I(1).
- Engle-Granger with a constant and AIC supplies the cointegration p-value. Candidates are
  retained at raw p <= 1%, subject to the I(1) screen and a positive hedge ratio. No Holm or
  other multiple-testing adjustment is applied and no orientation search is performed.
- fOU parameters use the second-variation H and sigma estimators and the continuous-time
  stationary-variance kappa estimate. Eligibility requires positive sigma and variance,
  0 < H < 0.5 and 0 < kappa < 2 for the daily Euler recursion.
- Structural forecasts use 5,000 paths, starting z=1.5, target probability 0.70 and a maximum
  horizon of 252 sessions. Up to 40 pairs are selected by shortest finite horizon, then
  highest maximum convergence probability, then pair ID.
- Daily forecasts use the latest 60 inferred fGn innovations, a maximum horizon of 126
  sessions and 5,000 paths. Seeds depend on the configured seed, pair and forecast date.

## Signal and execution timing

At close t the model uses information through t. A qualifying instruction can execute at
close t+1. The signal direction and forecast are frozen at t; entry spots and ATM strikes
come from t+1. EWMA volatility at t+1 uses returns only through t.

Forecast horizon H counts observed sessions from the signal date. Synthetic expiry is the
session t+H, while remaining maturity at next-close fill is expressed in calendar days / 365.
H=1 has no positive remaining maturity after the one-session execution lag and is not traded.
Forecasts whose horizon extends beyond the evaluation window remain forecast records but are
not traded.

Convergence for an open position is observed at one close and executed at the next close.
Known maturity settles intrinsically at expiry. Open positions are liquidated on the final
out-of-sample close. One position per pair is allowed; the default configuration has no
numeric portfolio-wide position cap.

## Option model and capital

Positive spread deviations use a dependent put plus independent call; negative deviations
use a dependent call plus independent put. Both legs are long European ATM options. Values
use zero-dividend Black-Scholes on adjusted synthetic spots with historical EWMA volatility.
This is a synthetic q=0 experiment, not a claim about historical American equity-option prices.

The contract multiplier is 100. Each entry is limited by the smaller of available cash and
5% of marked equity. An integer search maximizes invested debit subject to at least one
contract in each leg and at most 10% relative delta-dollar hedge error. Default costs are
10 bps adverse proportional price adjustment plus $0.65 per contract per side. Expiry
settlement has no sale commission or spread. Cash earns zero and long premiums are fully
cash-funded.

## Forecast calibration

Every eligible forecast on a flat pair is recorded before funding and execution decisions.
First passage is measured directly from observed spread paths after signal t through t+H.
Equality counts as crossing. A crossing before sample end is success; no crossing with the
full horizon observed is failure; an incomplete no-crossing path is censored.

Headline calibration rates and Brier scores use forecasts whose full selected horizon is
observed. Traded-only and all-forecast summaries are reported separately. Overlapping
forecasts are not treated as independent binomial trials.

## Benchmark and uncertainty

Module 01 saves the `^GSPC` price index and `^IRX` yield proxy. `^IRX` is divided by 100 and
treated as an annual decimal effective-rate proxy. Daily rates are `(1+r)^(1/252)-1` using
the prior available rate for each return interval.

Market alpha is estimated by OLS on excess returns with HAC covariance and five lags by
default. Daily alpha is multiplied by 252 for the labelled annualized display. Moving-block
bootstrap intervals estimate the raw daily mean return using 10, 20 and 40-session blocks
with 10,000 replications.

Matched placebos sample portfolios uniformly without replacement within each draw from the
same finite structural-horizon pool as the baseline. Repeated draws across portfolios are
allowed. Baseline and placebo portfolios use the same execution order, costs, sizing,
calendar, pair parameters and pair/date signal seeds. Upper-tail comparison uses
`(1 + count(placebo >= actual)) / (N + 1)`.

## Reproducibility

Each notebook uses explicit filenames and can run in a fresh kernel once prior notebook
outputs exist. Shared defaults are in `src/research_config.py`. There is no saved-run registry,
path discovery, separate validation layer or automatic executor. Modules 04 and 06 are
single-date previews; Module 07 runs the complete out-of-sample portfolio and Module 12 reruns
the same engine for matched placebos.

The final evaluation period was used during research development, so the finished allocation
rule should not be described as an untouched confirmatory test. Independent future or
otherwise untouched data would be required for confirmatory evaluation.
