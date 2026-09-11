"""Single configuration for the v2 synthetic research experiment."""
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

    def validate(self):
        if not 0 < self.train_fraction < 1 or not 0 <= self.max_missing_fraction < 1:
            raise ValueError('Invalid training fraction or missingness threshold.')
        if not 0 < self.cointegration_alpha < 1 or not 0 < self.integration_alpha < 1:
            raise ValueError('Invalid statistical significance level.')
        for key in ('top_n','correlation_neighbors','memory_window','structural_horizon',
                    'max_horizon_days','n_paths','bootstrap_replications'):
            value = getattr(self,key)
            if not isinstance(value,int) or value < 1:
                raise ValueError(f'{key} must be a positive integer.')
        if self.max_open_pairs is not None and (not isinstance(self.max_open_pairs,int) or self.max_open_pairs < 1):
            raise ValueError('max_open_pairs must be positive or None.')
        if self.n_placebos < 0 or self.hac_lags < 0 or self.seed < 0:
            raise ValueError('Counts, seed and lags must be nonnegative.')
        if self.initial_capital <= 0 or not 0 < self.premium_budget_fraction <= 1:
            raise ValueError('Invalid capital or premium budget.')
        if not 0 <= self.max_hedge_error < 1 or not 0 <= self.slippage_bps < 10000 or self.commission_per_contract < 0:
            raise ValueError('Invalid hedge tolerance or transaction costs.')
        if not 0 < self.ewma_lambda < 1 or not 0 < self.target_probability < 1 or self.entry_z <= 0:
            raise ValueError('Invalid model thresholds.')
        if any(not isinstance(b,int) or b < 1 for b in self.bootstrap_blocks):
            raise ValueError('Bootstrap block lengths must be positive integers.')
        return self

    def to_dict(self):
        return asdict(self)
