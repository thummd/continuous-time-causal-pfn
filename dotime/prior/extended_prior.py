"""Extended CausalTimePrior wrapper for Do-Over-Time-PFN.

Wraps CausalTimePrior.generate_pair() to produce model-ready dicts with:
- Padding to N_max=41 with variable masks
- Intervention time windows (start, end) instead of single time
- Intervention type encoding (0=hard, 1=soft, 2=time_varying)
- Query target/time sampling with downstream probability
"""

import torch
import numpy as np
from typing import Dict, Optional

from causal_time_prior.prior import CausalTimePrior
from causal_time_prior.interventions import InterventionType


# Map intervention types to integers
INTERVENTION_TYPE_MAP = {
    InterventionType.HARD: 0,
    InterventionType.SOFT: 1,
    InterventionType.TIME_VARYING: 2,
}


def pad_to_max_nodes(X: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Pad time series to have max_nodes variables."""
    T, N = X.shape
    if N < max_nodes:
        padding = torch.zeros(T, max_nodes - N, dtype=X.dtype, device=X.device)
        return torch.cat([X, padding], dim=1)
    return X[:, :max_nodes]


class ExtendedCausalTimePrior:
    """CTP wrapper that produces model-ready dicts for Do-Over-Time-PFN."""

    def __init__(
        self,
        n_max: int = 41,
        n_min: int = 3,
        n_max_prior: int = 10,
        t_range: tuple = (50, 200),
        burn_in: int = 50,
        downstream_prob: float = 0.7,
        seed: int = 42,
        chain_prob: float = 0.15,
        regime_switching_prob: float = 0.15,
    ):
        self.n_max = n_max
        self.t_range = t_range
        self.downstream_prob = downstream_prob

        config = {
            'N_max': n_max_prior,
            'burn_in': burn_in,
        }
        self.prior = CausalTimePrior(
            config=config,
            seed=seed,
            chain_prob=chain_prob,
            regime_switching_prob=regime_switching_prob,
        )
        self.rng = np.random.RandomState(seed)

    def sample_T(self) -> int:
        """Sample a time series length uniformly from t_range."""
        return self.rng.randint(self.t_range[0], self.t_range[1] + 1)

    def generate_sample(self, T: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Generate a single model-ready sample.

        Returns dict with:
            X_obs: (T, N_max) padded observational series
            X_int: (T, N_max) padded interventional series
            variable_mask: (N_max,) binary mask for real variables
            intervention_target: scalar int
            intervention_type: scalar int (0=hard, 1=soft, 2=time_varying)
            intervention_value: scalar float
            intervention_time_start: scalar float in [0, 1]
            intervention_time_end: scalar float in [0, 1]
            query_target: scalar int
            query_time: scalar float in [0, 1]
            Y_true: scalar float (ground truth)
            num_vars: scalar int (actual number of variables)
        """
        if T is None:
            T = self.sample_T()

        # Generate with divergence retry (up to 5 attempts)
        for _ in range(5):
            X_obs, X_int, intervention, scm = self.prior.generate_pair(T=T)
            if (not torch.isnan(X_obs).any() and not torch.isnan(X_int).any()
                    and X_obs.abs().max() < 500 and X_int.abs().max() < 500):
                break

        N = X_obs.shape[1]

        # Pad to N_max
        X_obs_padded = pad_to_max_nodes(X_obs, self.n_max)
        X_int_padded = pad_to_max_nodes(X_int, self.n_max)

        # Variable mask
        variable_mask = torch.zeros(self.n_max)
        variable_mask[:N] = 1.0

        # Intervention info
        intervention_target = intervention.targets[0] if intervention.targets else 0
        time_start = min(intervention.times)
        time_end = max(intervention.times)

        # Intervention value (scalar representation)
        if callable(intervention.values):
            # For time-varying: evaluate at midpoint
            mid_time = (time_start + time_end) // 2
            intervention_value = float(intervention.values(mid_time))
        else:
            intervention_value = float(intervention.values)

        intervention_type = INTERVENTION_TYPE_MAP[intervention.intervention_type]

        # Query sampling (downstream with probability downstream_prob)
        is_downstream = self.rng.rand() < self.downstream_prob
        if is_downstream and N > 1:
            other_vars = [v for v in range(N) if v != intervention_target]
            query_target = int(self.rng.choice(other_vars))
            query_time_idx = min(
                int(time_start + self.rng.randint(1, 6)),
                T - 1,
            )
        else:
            query_target = intervention_target
            query_time_idx = min(
                int(np.mean(intervention.times)),
                T - 1,
            )

        # Ground truth: raw interventional value and causal effect
        y_int = float(X_int_padded[query_time_idx, query_target].item())
        y_obs = float(X_obs_padded[query_time_idx, query_target].item())
        Y_true = y_int
        Y_causal_effect = y_int - y_obs

        return {
            'X_obs': X_obs_padded,                                    # (T, N_max)
            'X_int': X_int_padded,                                    # (T, N_max)
            'variable_mask': variable_mask,                           # (N_max,)
            'intervention_target': torch.tensor(intervention_target, dtype=torch.long),
            'intervention_type': torch.tensor(intervention_type, dtype=torch.long),
            'intervention_value': torch.tensor(intervention_value, dtype=torch.float32),
            'intervention_time_start': torch.tensor(time_start / T, dtype=torch.float32),
            'intervention_time_end': torch.tensor(time_end / T, dtype=torch.float32),
            'query_target': torch.tensor(query_target, dtype=torch.long),
            'query_time': torch.tensor(query_time_idx / T, dtype=torch.float32),
            'Y_true': torch.tensor(Y_true, dtype=torch.float32),
            'Y_causal_effect': torch.tensor(Y_causal_effect, dtype=torch.float32),
            'num_vars': torch.tensor(N, dtype=torch.long),
        }

    def generate_batch(
        self, batch_size: int, T: Optional[int] = None, **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate a batch of model-ready samples.

        All samples in a batch share the same T (sampled once if not provided).
        Returns dict with batched tensors.
        """
        if T is None:
            T = self.sample_T()

        samples = [self.generate_sample(T=T) for _ in range(batch_size)]

        return {
            key: torch.stack([s[key] for s in samples])
            for key in samples[0].keys()
        }
