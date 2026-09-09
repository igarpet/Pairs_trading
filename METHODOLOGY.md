# Revised synthetic pairs experiment

This is methodology version 2. The primary interface is the restored numbered Jupyter notebooks.
The optional CLI `python -m scripts.run_research` uses the same shared research functions.
Root data files and `Archived/` describe older experiments and are never overwritten.
The revised runner recomputes formation, execution and validation from frozen price inputs.
It does not claim that historical results survive these changes.

## Data and inference scope

The default price input already contains a historically filtered 466-stock universe.
Its original survivorship and full-sample availability biases CANNOT be undone by filtering
it again. `--allow-legacy-universe` is required to run that input, and the limitation is
written to the manifest. This is the immediately runnable, explicitly limited experiment.
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
  p-value. Holm adjustment at 1% applies across ALL unordered asset pairs in the formation
  universe. Correlation-screened-out pairs and failed tests receive p=1. Only positive hedge ratios are supported and retained. No minimum-p orientation search.
* Padding the full asset-pair family with p=1 for pairs not evaluated prevents the
  correlation screen from shrinking the multiplicity penalty. Holm controls family-wise
  error under valid underlying p-values; the I(1) assumptions and repeated research
  specification choices remain qualifications, not a universal data-snooping correction.
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
position per pair and at most ten total are enforced. Simultaneous instructions use
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

Each entry's all-in debit is bounded by the lesser of available cash and 5% of pre-entry
marked portfolio equity. An integer search maximizes invested debit subject to both legs
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

The default benchmark is the saved ^GSPC price-index series (not SPY or a total-return index).
It covers the first OOS valuation date. Files and benchmark identity are frozen and hashed;
no network call occurs during finalization. The annual risk-free input is explicitly treated
as effective decimal, converted to (1+r)^(1/252)-1 using the prior-session value for each
return interval. Historical rate-source quote conventions remain a data assumption.
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

## Reproducibility and acceptance

Run directories are exclusive, preserve input copies, source hashes, Python/package versions,
configuration, completed stages and output hashes. Resume rejects changed input, source,
environment or completed outputs. Placebo results are checkpointed individually. No old
row counts or old returns are enforced on new results. Original notebook copies and old workflows are archived with their historic assumptions.
The root numbered notebooks are updated to this methodology. Each saves its own intermediate
outputs and completion state. Modules 04/06 are one-date explanatory snapshots; Module 07
recalculates daily signals and prices with the same functions. Diagnostic modules do not
rerun the main portfolio. Notebook run manifests also hash notebook sources, excluding
outputs and execution counts; use a new run name when changing methodology or inputs.
Tests cover horizon events, lagged decisions, costs, budgets, position limits, frozen inputs,
fixed orientation, deterministic ordering, statistics and no-trade/degenerate cases.

A full formation/backtest/100-placebo run is intentionally left for local execution. After
reviewing it, update the thesis from THAT run's artifacts. Additional experiments should
change configuration into new directories (for example path counts/seeds, costs, memory
windows, risk budgets). Do not tune against OOS returns and present the same sample as
untouched evidence. No matched rolling-recalibration robustness claim exists until a new,
explicit rolling experiment is implemented and evaluated.

References for the implemented APIs:
* https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html
* https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
