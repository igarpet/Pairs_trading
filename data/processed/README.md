# Historical inputs and outputs

These files predate methodology v2. The current runner reads explicitly named frozen
price/rate/benchmark inputs only, then writes ALL revised outputs under runs/<id>.
It never reads the old trade, eligible-pair or fitted-parameter files as revised results.

Known older outputs coexist here: selected_pairs has 20 rows; eligible_pairs has 40;
fou_parameters has 446 fits; fractional_ou_parameters has 481. The nested
static_convergence_calibration directory describes a different 254-trade generation;
its 19-byte horizon_bucket_calibration.parquet is damaged. The root 216-trade calibration
and older timeout experiments have different provenance. Do not combine them.

The revised run directories include a manifest and generated formation summaries so
these naming and population inconsistencies cannot silently propagate into new tables.
