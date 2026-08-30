import numpy as np
import pandas as pd
from src.convergence_signal import structural_convergence_horizon

def filter_antipersistent_pairs(fou_parameters):
    cols = ['hurst','kappa','sigma','variance','drift_half_life']
    mask = ((fou_parameters['hurst'] > 0) & (fou_parameters['hurst'] < 0.5) &
            (fou_parameters['kappa'] > 0) & (fou_parameters['sigma'] > 0) &
            (fou_parameters['variance'] > 0) & np.isfinite(fou_parameters[cols]).all(axis=1))
    return fou_parameters.loc[mask].copy().reset_index(drop=True)

def compute_structural_t70(eligible_pairs, starting_z=1.5, target_probability=0.70, max_horizon_days=252, n_paths=5000, dt=1.0, seed=42):
    rows = []
    for _, row in eligible_pairs.iterrows():
        try:
            result = structural_convergence_horizon(
                mu=row['mu'], kappa=row['kappa'], sigma=row['sigma'], hurst=row['hurst'],
                stationary_variance=row['variance'], starting_z=starting_z,
                target_probability=target_probability, max_horizon_days=max_horizon_days,
                n_paths=n_paths, dt=dt, seed=seed)
            rows.append({**row.to_dict(), **result})
        except Exception as exc:
            rows.append({**row.to_dict(), 'structural_t70': np.nan,
                         'structural_probability_max': np.nan,
                         'eligibility_error': str(exc)})
    return pd.DataFrame(rows)

def select_top_pairs_by_structural_t70(structural_results, top_n=40):
    valid = structural_results[structural_results['structural_t70'].notna()].copy()
    valid = valid.sort_values(['structural_t70','structural_probability_max'], ascending=[True,False])
    return valid.head(top_n).reset_index(drop=True)
