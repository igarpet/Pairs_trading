# Synthetic pairs experiment

The numbered notebooks are the analysis entry points. Shared functions in `src/`
define identical numerical and execution rules for the strategy and placebos.

## Data and inference scope

Module 01 downloads the current S&P 500 constituents CSV from
https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv
and Yahoo Finance adjusted closes, with explicit start and exclusive end dates.
The current constituent snapshot introduces survivorship bias; it is not historical
membership. The notebook explicitly permits this limited universe with
`allow_legacy=True`. Missing full histories stop execution for inspection, and
formation-only availability screening follows. Vendor revisions may affect fresh downloads.
For improved universe formation, supply a full historical price panel and dated membership
CSV with `ticker,member_from,member_to,known_at`. `member_to` is exclusive (blank means open).
Membership is frozen at the formation cutoff. A later exit from the index does not trigger
removal of a frozen pair. The user must establish the historical source's completeness;
a CSV alone does not prove absence of survivorship bias or adjusted-data revisions.

The first 70% of sessions are formation. Missingness and initial availability are assessed
ONLY there. Formation gaps are forward-filled from earlier observations; leading gaps
exclude the asset. The OOS panel must be finite and positive; missing quotes/delistings
stop the run instead of silently removing an asset based on future information. A complete
corporate-action/delisting policy and corresponding data would be a separate data extension.
Prices represent adjusted synthetic underlying levels, not historical listed-option spots.

## Pair selection and fractional model

* Ten highest return correlations per asset define an unordered candidate family.
* Each pair is oriented once, alphabetically, with the first ticker dependent.
* ADF with constant and AIC lags must fail to reject a level unit root at 5% and reject
  a unit root in first differences at 5%. This is a screening convention, not proof of I(1).
* Engle–Granger `statsmodels.tsa.stattools.coint`, constant and AIC, supplies the residual-test
  p-value. Retain candidates with a raw p-value at or below 1%, subject to the integration
  screen and a positive hedge ratio. No multiple-testing adjustment is applied and no
  minimum-p orientation search is performed. Failed tests receive p=1.
* This is individual candidate screening, not family-wise significance or a correction
  for data snooping. The audit records `multiplicity_method='none'`; its legacy
  `adjusted_pvalue` column equals the raw `pvalue` for compatibility. Candidate and full
  universe pair counts are descriptive only and do not change selection thresholds.
* fOU parameters retain the existing second-variation H and sigma estimates and continuous-time
  stationary-variance kappa estimate. Require positive sigma/variance, 0<H<0.5 and 0<kappa<2
  for stability of the daily Euler recursion. The continuous/discrete discrepancy remains a
  model approximation; it is not cured by a small Monte Carlo standard error.
* Structural forecasts use 5,000 paths, z=1.5, probability target 0.70, maximum 252 sessions.
  Select up to 40 finite horizons, ascending horizon then descending maximum probability,
  with pair ID breaking ties. Fewer than 40 is reported without relaxing thresholds.
* Live forecasts use the existing 60-innovation conditional fGn construction, daily Euler
  recursion, maximum 126 sessions and 5,000 paths. Seeds depend on configured seed, pair and
  forecast date. Structural simulation uses a shared fixed seed (common random numbers).

## Signal and execution timing

At close t, compute the signal from data through t. If it qualifies, submit an instruction
for close t+1. The instruction's direction and forecast are frozen at t; execution uses
spots and ATM strikes at t+1. Lagged EWMA at t+1 includes returns only through t. Rates are
as-of the valuation date. This is a declared next-close synthetic fill convention, not a
claim to obtain historical closing option quotes. Signals are not recomputed or cancelled
based on the t+1 closing spread. In particular, a crossing during the delay may occur.

Forecast horizon H counts observed sessions FROM SIGNAL DATE. Synthetic expiry is the
session at t+H, and the remaining option maturity at fill is H-1 sessions expressed as
actual calendar days / 365 for Black–Scholes. H=1 cannot leave positive time after fill and
is skipped. Horizons beyond the OOS calendar remain in forecast diagnostics but are not
traded; this is an explicitly finite evaluation-window policy. Only calendar membership
and the declared sample end are used for maturity, never future prices.

Convergence is checked at close t for an existing position and executed at close t+1.
Known maturity settles intrinsically at expiry; final forced liquidation occurs at the
last observed close. A forecast event and an executed exit are distinct records. One
position per pair is enforced; there is no numeric total-position cap. Simultaneous instructions use
alphabetical pair order for baseline and placebos. No same-close signal-and-fill path exists.

## Option model and capital

Two long European ATM options implement each positive-beta spread direction: positive
spread deviation uses dependent put plus independent call; negative uses the reverse.
Contract multiplier is 100. Prices are zero-dividend Black–Scholes values on adjusted
synthetic spots with historical EWMA volatility. This is a synthetic q=0 experiment;
actual dividend-paying American equity options require quote, dividend and corporate-action
data plus an appropriate exercise model. Zero-dividend pricing is explicit, not a claim
that real constituents pay no dividends.

Default round-trip cost convention: 10 bps adverse proportional price adjustment on each
purchase/sale, plus $0.65 per contract per side. Expiry settlement has no sale commission
or spread. These are configured research assumptions, not measured market costs. Marked
NAV uses model mid values; realized PnL includes entry and exit costs. Cash earns zero,
long premiums are fully cash-funded, no borrowing, and equity starts at the first OOS close
with initial capital before any fill.

Initial equity is $100,000. Each entry's all-in debit is bounded by the smaller of
available cash and 5% of equity marked before that session's entries. There is no
numeric cap on simultaneously open pairs. One position per pair is allowed.
Cash is not reserved for a fixed number of positions.
An integer search maximizes invested debit subject to both legs
having at least one contract and at most 10% relative delta-dollar hedge error; ties prefer
lower error and fewer contracts. If no feasible pair exists, skip and log it. Hedge matching
uses the original log-spread derivative ratio. Budgets cap premium loss, not all portfolio
risk; exposure may drift because there is no delta rebalancing. Aggregate cash and count
limits remain binding. Final exit fees can exceed a negligible residual option value and
are included in net proceeds.

## Calibration

Every eligible forecast on a flat pair is logged before execution/funding decisions.
First passage is measured directly on the spread path after signal t through t+H, irrespective
of option exit or whether the forecast was traded. Equality counts as crossing. A crossing
observed before sample end establishes success even with an incomplete horizon; no crossing
with the full horizon observed is failure; an incomplete no-crossing path is censored.
A missing session stops observation rather than allowing the event path to jump across it.
Early option expiry alone NEVER causes censoring if prices through t+H are available.

Headline rates and Brier scores use the subset whose full H-session window is observed,
avoiding a denominator that keeps early successes but drops incomplete failures. Traded-only
and all-forecast summaries are separate. The predicted probability at the maximum horizon
is never compared to the selected-H event. No ordinary independent-binomial significance
claim is made for overlapping forecasts.

## Benchmark and uncertainty

Module 01_RF_Data downloads the ^GSPC price index from Yahoo Finance (not SPY or a
total-return index), and Yahoo's ^IRX 13-week Treasury bill yields.
The latter are discount-basis yields in percent; division by 100 supplies an annual
decimal yield treated as an effective-rate proxy, not an exact realized Treasury return.
Downstream notebooks read these generated files. Daily rates are calculated as
(1+r)^(1/252)-1 using the prior-session value for each return interval.
All market analysis uses one saved aligned excess-return table and HAC with five lags.
Daily alpha is simply multiplied by 252 for its labelled annualized display; its significance
is unchanged. Degenerate and insufficient-data regressions report status instead of a
spurious alpha p-value. Bootstrap intervals estimate raw daily mean return, not alpha;
10, 20 and 40-session overlapping-block lengths use 10,000 replications. Stationarity and
block-length dependence remain limitations.

Placebos draw portfolios uniformly without replacement WITHIN each draw from the same
finite structural-horizon pool as the baseline. Across draws, repetitions are allowed and
recorded (IID Monte Carlo draws). The baseline may contain fewer than 40 pairs; placebo
size always matches. If the entire pool is selected, the membership null is degenerate
and explicitly flagged. All portfolios use identical entry order, costs, sizing, dates,
model parameters and signal seeds. Upper-tail comparison uses (1 + count(value >= actual))
/ (N + 1); strict-less percentile and ties are explicit. Undefined metrics are labelled,
not silently dropped. This is a conditional descriptive reference, not formal proof of
alpha or correction for all strategy choices considered during research.

## Reproducibility

Each notebook starts with imports and uses explicit filenames in its reading and
saving cells. There is no directory discovery, input copying or saved-settings
layer. Shared defaults are in src/research_config.py. After editing settings,
restart kernels and rerun the notebooks in order. Generated files default to the
working folder; manually edit matching upstream writes and downstream reads if
using different locations. The user creates any required destination folders.

Modules 04/06 are one-date previews. Module 07 calculates the full daily portfolio.
Module 12 uses the same numerical and execution functions and recomputes all draws;
its comparison is built from those draws rather than pre-existing placebo files.
Library versions are pinned in requirements.txt. No automated tests or GitHub
workflows are included in the submission repository.

References for the implemented APIs:
* https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html

## Research selection limitation

Allocation settings were examined using results from this evaluation period. It must
not be described as an untouched confirmatory test of the final allocation rule.
The conditional placebo comparison and bootstrap do not adjust for that specification
search. Independent future or otherwise untouched data would be needed for confirmatory
evaluation. See README.md for the scope of AI assistance.
