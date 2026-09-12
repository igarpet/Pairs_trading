"""Research settings shared by notebooks and placebo portfolios."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ResearchConfig:
    train_fraction: float = 0.70
    max_missing_fraction: float = 0.05
    correlation_neighbors: int = 10
    cointegration_alpha: float = 0.01
    integration_alpha: float = 0.05
    top_n: int = 40
    initial_capital: float = 100000.0
    entry_z: float = 1.5
    target_probability: float = 0.70
    memory_window: int = 60
    structural_horizon: int = 252
    max_horizon_days: int = 126
    n_paths: int = 5000
    seed: int = 42
    ewma_lambda: float = 0.94
    max_open_pairs: int | None = None
    premium_budget_fraction: float = 0.05
    max_hedge_error: float = 0.10
    slippage_bps: float = 10.0
    commission_per_contract: float = 0.65
    n_placebos: int = 100
    bootstrap_replications: int = 10000
    bootstrap_blocks: tuple = (10, 20, 40)
    hac_lags: int = 5

    def to_dict(self):
        return asdict(self)
