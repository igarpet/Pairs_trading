"""Input-series checks, JSON serialization and shared backtest arguments."""

import json
import numpy as np
import pandas as pd
from src.research_data import normalize_frame


def write_json(filename, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(clean(value), file, indent=2, default=str, allow_nan=False)


def read_series(path):
    x = normalize_frame(pd.read_parquet(path))
    if x.shape[1] != 1:
        raise ValueError(f"{path} must contain exactly one data column.")
    s = x.iloc[:, 0].astype(float)
    if not np.isfinite(s).all():
        raise ValueError(f"{path} contains nonfinite values.")
    return s


def backtest_kwargs(c):
    names = [
        "initial_capital",
        "entry_z",
        "target_probability",
        "memory_window",
        "max_horizon_days",
        "n_paths",
        "ewma_lambda",
        "seed",
        "max_open_pairs",
        "premium_budget_fraction",
        "max_hedge_error",
        "slippage_bps",
        "commission_per_contract",
    ]
    return {k: getattr(c, k) for k in names}
