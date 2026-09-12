# Input data

The files preserve the supplied dataset; notebooks do not download replacements.

| File | Content |
|---|---|
| prices.parquet | Adjusted equity prices, USD, daily sessions, 466-stock universe |
| risk_free_rates.parquet | Annual rate input, decimals, treated as effective |
| sp500_prices.parquet | S&P 500 price-index benchmark (^GSPC) |

The research sample covers January 2016 through December 2025. Its first 70% forms
the estimation sample. The equity/rate source is Yahoo Finance; the rate proxy is
^IRX. The constituent list and full-sample availability filter are not a point-in-time
historical universe. Prices contain vendor adjustments and are not listed-option
quotes. See METHODOLOGY.md for cleaning and valuation assumptions.

The original download date and precise constituent-list snapshot are not established
by these files. Verify them against the author's acquisition records before submission;
no date or complete historical-membership claim is inferred here.
