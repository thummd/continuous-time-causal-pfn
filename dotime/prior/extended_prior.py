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
from causal_time_prior.interventions import InterventionType, InterventionSpec
from dotime.prior.tscm_sampler import TSCMSampler, TSCMStructure


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


class TSCMPrior:
    """Drop-in replacement for CausalTimePrior that generates from a single TSCM structure.

    Has the same ``generate_pair(T)`` interface so ``ExtendedCausalTimePrior``
    can swap it in transparently.
    """

    def __init__(self, structure: TSCMStructure, burn_in: int = 50, seed: int = 42,
                 use_lagged_edges: bool = True, intervention_scale: float = 2.0,
                 sigma_w: float = 0.5):
        self.sampler = TSCMSampler(structure, max_lag=1, use_lagged_edges=use_lagged_edges,
                                   sigma_w=sigma_w, sigma_b=sigma_w * 0.5)
        self.hidden_vars = self.sampler.get_hidden_vars()
        self.burn_in = burn_in
        self.intervention_scale = intervention_scale
        self.gen = torch.Generator().manual_seed(seed)
        self.config = {'burn_in': burn_in}

    def generate_pair(self, T: int):
        """Return (X_obs, X_int, intervention, scm) like CausalTimePrior."""
        scm = self.sampler.sample(generator=self.gen)
        N = len(scm._topo)

        X_obs = scm.sample_observational(T=T, burn_in=self.burn_in, generator=self.gen)

        # Pick intervention target: first non-hidden variable
        valid = [i for i in range(N) if i not in self.hidden_vars]
        int_target = valid[0] if valid else 0

        # Intervention at a random time in [10, T-10], matching CTP's range
        t_lo = min(10, T - 1)
        t_hi = max(t_lo + 1, T - 10)
        int_time = int(torch.randint(t_lo, t_hi, (1,), generator=self.gen).item())
        int_value = float(torch.randn(1, generator=self.gen).item() * self.intervention_scale)

        intervention = InterventionSpec(
            targets=[int_target],
            times=[int_time],
            intervention_type=InterventionType.HARD,
            values=int_value,
        )

        X_int = scm.sample_interventional(
            T=T, intervention=intervention, burn_in=self.burn_in, generator=self.gen,
        )
        return X_obs, X_int, intervention, scm


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
        intervention_source: str = "prior",
        tscm_structure: Optional[str] = None,
        use_lagged_edges: bool = True,
        intervention_scale: float = 2.0,
        causal_mask_mode: str = "full",
    ):
        self.n_max = n_max
        self.t_range = t_range
        self.downstream_prob = downstream_prob
        self.intervention_source = intervention_source
        self.causal_mask_mode = causal_mask_mode

        if tscm_structure is not None:
            structure_enum = TSCMStructure(tscm_structure)
            self.prior = TSCMPrior(
                structure_enum, burn_in=burn_in, seed=seed,
                use_lagged_edges=use_lagged_edges,
                intervention_scale=intervention_scale,
            )
        else:
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

    def generate_sample(
        self, T: Optional[int] = None, n_queries: int = 1,
        query_mode: str = "single",
    ) -> Dict[str, torch.Tensor]:
        """Generate a single model-ready sample with one or more query points.

        Parameters
        ----------
        T : int, optional
            Time series length (sampled from t_range if None).
        n_queries : int
            Number of (query_target, query_time) pairs per trajectory.
            When > 1, query_target/query_time/Y_true/Y_causal_effect
            are tensors of shape (n_queries,) instead of scalars.

        Returns dict with:
            X_obs: (T, N_max) padded observational series
            X_int: (T, N_max) padded interventional series
            variable_mask: (N_max,) binary mask for real variables
            intervention_target: scalar int
            intervention_type: scalar int (0=hard, 1=soft, 2=time_varying)
            intervention_value: scalar float
            intervention_time_start: scalar float in [0, 1]
            intervention_time_end: scalar float in [0, 1]
            query_target: scalar int or (n_queries,) ints
            query_time: scalar float or (n_queries,) floats
            Y_true: scalar float or (n_queries,) floats
            Y_causal_effect: scalar float or (n_queries,) floats
            num_vars: scalar int
        """
        if T is None:
            T = self.sample_T()

        # Generate with divergence retry (up to 20 attempts for long trajectories)
        for _ in range(20):
            X_obs, X_int, intervention, scm = self.prior.generate_pair(T=T)
            if (not torch.isnan(X_obs).any() and not torch.isnan(X_int).any()
                    and X_obs.abs().max() < 10 and X_int.abs().max() < 10):
                break

        N = X_obs.shape[1]

        # Causal masking: zero out X_obs at and after intervention onset.
        int_onset = min(intervention.times)
        intervention_target = intervention.targets[0] if intervention.targets else 0
        X_obs_masked = X_obs.clone()
        X_obs_masked[int_onset:] = 0.0

        if self.causal_mask_mode == "interpolation":
            # Restore the treatment variable at int_onset with its OBSERVATIONAL
            # value. The causal model additionally receives the intervention spec
            # (A_int = v) via the mixer; the obs-only model only sees A_obs here.
            X_obs_masked[int_onset, intervention_target] = X_obs[int_onset, intervention_target]

        # Pad to N_max
        X_obs_padded = pad_to_max_nodes(X_obs_masked, self.n_max)
        X_int_padded = pad_to_max_nodes(X_int, self.n_max)

        # Variable mask
        variable_mask = torch.zeros(self.n_max)
        variable_mask[:N] = 1.0

        # Intervention info (intervention_target already set above for masking)
        time_start = min(intervention.times)
        time_end = max(intervention.times)

        # Intervention value (scalar representation)
        if callable(intervention.values):
            mid_time = (time_start + time_end) // 2
            intervention_value = float(intervention.values(mid_time))
        else:
            intervention_value = float(intervention.values)

        # Re-simulate with an observed-scale intervention value if requested.
        # Sample a value from the pre-intervention history of the intervention
        # target, create a new InterventionSpec, and re-run the SCM so that
        # (intervention_value, X_int) stays consistent.
        #
        # Modes:
        #   "prior"             — keep the CTP-sampled value (no re-simulation).
        #   "positivity_aware"  — clip prior value to [obs_mean - 3σ, obs_mean + 3σ]
        #                         and re-simulate. Preserves prior shape, enforces
        #                         positivity (intervention within observed support).
        #   "observed_discrete" — pick a random past value (measure-zero for
        #                         continuous variables, kept for backward compat).
        #   "observed_normal"   — sample from N(mean(pre_int), std(pre_int)).
        #   "observed_uniform"  — sample from U[min(pre_int), max(pre_int)].
        #
        # "observed" is accepted as a legacy alias for "observed_discrete".
        mode = self.intervention_source
        if mode == "observed":
            mode = "observed_discrete"

        if mode == "positivity_aware":
            pre_int = X_obs[:int_onset, intervention_target]
            if pre_int.numel() > 1 and float(pre_int.std().item()) > 1e-4:
                mu = float(pre_int.mean().item())
                sigma = float(pre_int.std().item())
                clipped = float(np.clip(intervention_value, mu - 3 * sigma, mu + 3 * sigma))
                if clipped != intervention_value:
                    new_intervention = InterventionSpec(
                        targets=intervention.targets,
                        times=intervention.times,
                        intervention_type=InterventionType.HARD,
                        values=clipped,
                    )
                    X_int_new = scm.sample_interventional(
                        T=T, intervention=new_intervention,
                        burn_in=self.prior.config.get('burn_in', 50),
                    )
                    if (not torch.isnan(X_int_new).any() and X_int_new.abs().max() < 10):
                        X_int = X_int_new
                        X_int_padded = pad_to_max_nodes(X_int, self.n_max)
                        intervention = new_intervention
                        intervention_value = clipped

        elif mode in ("observed_discrete", "observed_normal", "observed_uniform"):
            pre_int = X_obs[:int_onset, intervention_target]
            if pre_int.numel() > 0 and float(pre_int.std().item()) > 1e-4:
                pre_np = pre_int.detach().cpu().numpy()
                if mode == "observed_discrete":
                    obs_value = float(pre_np[self.rng.randint(len(pre_np))])
                elif mode == "observed_normal":
                    mu = float(pre_np.mean())
                    sigma = float(pre_np.std(ddof=1)) if len(pre_np) > 1 else 0.0
                    obs_value = float(self.rng.randn() * max(sigma, 1e-4) + mu)
                else:  # observed_uniform
                    lo = float(pre_np.min())
                    hi = float(pre_np.max())
                    if hi > lo:
                        obs_value = float(self.rng.uniform(lo, hi))
                    else:
                        obs_value = lo

                new_intervention = InterventionSpec(
                    targets=intervention.targets,
                    times=intervention.times,
                    intervention_type=InterventionType.HARD,
                    values=obs_value,
                )
                X_int_new = scm.sample_interventional(
                    T=T, intervention=new_intervention,
                    burn_in=self.prior.config.get('burn_in', 50),
                )
                if (not torch.isnan(X_int_new).any() and X_int_new.abs().max() < 10):
                    X_int = X_int_new
                    X_int_padded = pad_to_max_nodes(X_int, self.n_max)
                    intervention = new_intervention
                    intervention_value = obs_value

        intervention_type = INTERVENTION_TYPE_MAP[intervention.intervention_type]

        # Positivity score: how OOD is intervention_value relative to observed support?
        pre_int = X_obs[:int_onset, intervention_target]
        if pre_int.numel() > 1 and float(pre_int.std().item()) > 1e-4:
            obs_mu = float(pre_int.mean().item())
            obs_sigma = float(pre_int.std().item())
            positivity_score = max(0.0, abs(intervention_value - obs_mu) / obs_sigma - 3.0)
        else:
            positivity_score = 0.0

        # Query sampling — aligned with identifiability theory:
        # P(Y_t | do(A_t), H_{t-1},...,H_{t-K})
        other_vars = [v for v in range(N) if v != intervention_target]
        int_time = min(int(np.mean(intervention.times)), T - 1)

        query_targets = []
        query_time_idxs = []

        if query_mode == "all_pairs" and other_vars:
            # Query ALL non-intervention variables at intervention time.
            # This forces the model to learn the full causal structure.
            for qt in other_vars:
                query_targets.append(qt)
                query_time_idxs.append(int_time)
        else:
            # Single mode: random queries at intervention time
            for _ in range(n_queries):
                if other_vars:
                    qt = int(self.rng.choice(other_vars))
                else:
                    qt = intervention_target
                query_targets.append(qt)
                query_time_idxs.append(int_time)

        # Ground truth: raw interventional value and causal effect
        y_trues = [float(X_int_padded[qti, qt].item())
                   for qt, qti in zip(query_targets, query_time_idxs)]
        y_obs_vals = [float(X_obs_padded[qti, qt].item())
                      for qt, qti in zip(query_targets, query_time_idxs)]
        y_effects = [yi - yo for yi, yo in zip(y_trues, y_obs_vals)]

        # Flatten to scalars if single query (backwards compatible)
        actual_n_queries = len(query_targets)
        if actual_n_queries == 1:
            query_target_t = torch.tensor(query_targets[0], dtype=torch.long)
            query_time_t = torch.tensor(query_time_idxs[0] / T, dtype=torch.float32)
            y_true_t = torch.tensor(y_trues[0], dtype=torch.float32)
            y_effect_t = torch.tensor(y_effects[0], dtype=torch.float32)
        else:
            query_target_t = torch.tensor(query_targets, dtype=torch.long)
            query_time_t = torch.tensor([qti / T for qti in query_time_idxs], dtype=torch.float32)
            y_true_t = torch.tensor(y_trues, dtype=torch.float32)
            y_effect_t = torch.tensor(y_effects, dtype=torch.float32)

        return {
            'X_obs': X_obs_padded,                                    # (T, N_max)
            'X_int': X_int_padded,                                    # (T, N_max)
            'variable_mask': variable_mask,                           # (N_max,)
            'int_onset_idx': torch.tensor(int_onset, dtype=torch.long),
            'intervention_target': torch.tensor(intervention_target, dtype=torch.long),
            'intervention_type': torch.tensor(intervention_type, dtype=torch.long),
            'intervention_value': torch.tensor(intervention_value, dtype=torch.float32),
            'intervention_time_start': torch.tensor(time_start / T, dtype=torch.float32),
            'intervention_time_end': torch.tensor(time_end / T, dtype=torch.float32),
            'positivity_score': torch.tensor(positivity_score, dtype=torch.float32),
            'query_target': query_target_t,
            'query_time': query_time_t,
            'Y_true': y_true_t,
            'Y_causal_effect': y_effect_t,
            'num_vars': torch.tensor(N, dtype=torch.long),
        }

    def generate_batch(
        self, batch_size: int, T: Optional[int] = None,
        n_queries: int = 1, num_workers: int = 0,
        query_mode: str = "single", **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate a batch of model-ready samples.

        All samples in a batch share the same T (sampled once if not provided).
        Multi-query batches include a '_traj_idx' field that maps each query
        to its source trajectory (for encoder caching).

        Parameters
        ----------
        query_mode : "single" (random queries) or "all_pairs" (all outcome vars)

        Returns dict with:
            X_obs, variable_mask: (B, ...) unique trajectories
            intervention_*, query_*, Y_*: (B_total,) per-query (B_total = sum of queries)
            _traj_idx: (B_total,) index into trajectory dimension
        """
        if T is None:
            T = self.sample_T()

        if num_workers > 0:
            samples = self._generate_parallel(batch_size, T, n_queries, num_workers, query_mode)
        else:
            samples = [self.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)
                       for _ in range(batch_size)]

        return self._collate_batch(samples)

    def _generate_parallel(self, batch_size, T, n_queries, num_workers, query_mode):
        """Generate samples in parallel using multiprocessing (fork)."""
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        args = [(self, T, n_queries, query_mode)] * batch_size
        with ctx.Pool(processes=min(num_workers, batch_size)) as pool:
            samples = pool.map(_generate_sample_worker, args)
        return samples

    @staticmethod
    def _collate_batch(samples):
        """Collate samples into a batch with _traj_idx for encoder caching.

        Trajectory-level fields (X_obs, variable_mask) are stacked to (B, ...).
        Query-level fields are concatenated to (B_total,) with _traj_idx mapping
        each query back to its trajectory.
        """
        query_keys = {'query_target', 'query_time', 'Y_true', 'Y_causal_effect'}
        # Check if any sample has multi-query (tensor with dim > 0 for query fields)
        is_multi = any(s['query_target'].dim() > 0 for s in samples)

        if not is_multi:
            # All scalar queries — simple stack, no _traj_idx needed
            return {
                key: torch.stack([s[key] for s in samples])
                for key in samples[0].keys()
            }

        # Multi-query: build _traj_idx and separate trajectory vs query fields
        batch = {}
        traj_indices = []
        intervention_keys = {'intervention_target', 'intervention_type',
                             'intervention_value', 'intervention_time_start',
                             'intervention_time_end'}

        # Stack trajectory-level fields (unique per trajectory)
        traj_keys = {k for k in samples[0].keys() if k not in query_keys and k not in intervention_keys}
        for key in traj_keys:
            batch[key] = torch.stack([s[key] for s in samples])

        # Build _traj_idx and concatenate query + intervention fields
        for i, s in enumerate(samples):
            nq = s['query_target'].numel()
            traj_indices.append(torch.full((nq,), i, dtype=torch.long))
        batch['_traj_idx'] = torch.cat(traj_indices)

        for key in query_keys:
            parts = []
            for s in samples:
                v = s[key]
                parts.append(v.unsqueeze(0) if v.dim() == 0 else v)
            batch[key] = torch.cat(parts)

        # Intervention fields: repeat per query count
        for key in intervention_keys:
            parts = []
            for s in samples:
                nq = s['query_target'].numel()
                v = s[key]
                parts.append(v.unsqueeze(0).expand(nq) if v.dim() == 0 else v)
            batch[key] = torch.cat(parts)

        return batch


def _generate_sample_worker(args):
    """Top-level function for multiprocessing Pool.map (must be picklable)."""
    prior, T, n_queries, query_mode = args
    return prior.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)
