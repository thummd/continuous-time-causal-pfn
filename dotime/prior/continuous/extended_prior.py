"""Model-ready batch generator for continuous-time temporal causal PFNs.

This is the continuous-time counterpart to
:class:`dotime.prior.extended_prior.ExtendedCausalTimePrior`.  It wraps a
:class:`ContinuousTSCMSampler`, draws an observation schedule from one
of the variable-Delta-t families in :mod:`time_schedule`, and produces a
model-ready dict for every call.

Contract with the rest of the pipeline
--------------------------------------
The returned dict matches the discrete-time contract used by
:mod:`dotime.data.temporal_dataloader` and :mod:`dotime.model`, plus two
new fields:

- ``times`` : ``(T,)`` float tensor of absolute observation times.
- ``dts`` : ``(T - 1,)`` float tensor with ``dts[i] = times[i+1] -
  times[i]``.  Useful for encoder variants that consume log-Delta-t
  features separately.

Two existing fields change their semantics in the continuous setting:

- ``int_onset_idx`` : integer index into ``times`` at which the
  intervention starts.  Matches the discrete-time field exactly.
- ``intervention_time_start`` / ``intervention_time_end`` : stay
  normalised to ``[0, 1]`` over the observation window
  ``[times[0], times[-1]]`` so the existing mixer head is unchanged.

Additionally, we expose ``t_int_start``, ``t_int_end``, and ``t_query``
in absolute time units for encoders that want them.

Counterfactual vs interventional pairs
--------------------------------------
``pair_mode`` selects between
:meth:`ContinuousSCM.sample_counterfactual_pair` (shared noise; Pearl
rung 3) and :meth:`ContinuousSCM.sample_interventional_pair`
(independent noise; rung 2 — matches the discrete-time DoT-PFN default).
Counterfactual pairs are the natural training signal for the workshop
paper; interventional pairs are kept for regression testing against
DoT-PFN behaviour.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from dotime.prior.tscm_sampler import TSCMStructure

from .continuous_scm import (
    ContinuousIntervention,
    ContinuousSCM,
    InterventionKind,
)
from .time_schedule import (
    exponential_schedule,
    jittered_schedule,
    regular_schedule,
)
from .tscm_sampler import ContinuousTSCMSampler


def _pad_to_max_nodes(X: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Right-pad ``X`` of shape ``(T, N)`` to ``(T, max_nodes)`` with zeros."""
    T, N = X.shape
    if N >= max_nodes:
        return X[:, :max_nodes]
    padding = torch.zeros(T, max_nodes - N, dtype=X.dtype, device=X.device)
    return torch.cat([X, padding], dim=1)


def _build_schedule(
    schedule: str,
    T: int,
    dt: float,
    jitter: float,
    exp_rate: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the requested schedule family."""
    if schedule == "regular":
        return regular_schedule(T=T, dt=dt)
    if schedule == "jittered":
        return jittered_schedule(T=T, dt=dt, jitter=jitter, generator=generator)
    if schedule == "exponential":
        return exponential_schedule(T=T, rate=exp_rate, generator=generator)
    raise ValueError(f"unknown schedule: {schedule!r}")


class ContinuousExtendedPrior:
    """Continuous-time model-ready sample generator.

    Parameters
    ----------
    tscm_structure : str
        One of the :class:`TSCMStructure` values (``back_door``,
        ``front_door``, ...).  Random-topology sampling (the CTP path in
        the discrete prior) is intentionally not supported here; the
        workshop paper is scoped to the named identifiability
        structures.
    n_max : int
        Pad width along the variable axis.  Defaults to 41 to match
        DoT-PFN's CausalChamber-motivated default.
    t_range : tuple of int
        Uniform prior on ``T`` (number of observations).
    schedule : {"regular", "jittered", "exponential"}
        Observation schedule family.  ``regular`` reproduces the
        discrete-time behaviour at ``dt=1.0``.
    dt : float
        Mean inter-observation gap for ``regular`` / ``jittered``
        schedules.
    jitter : float
        Used only by ``jittered``; see :func:`jittered_schedule`.
    exp_rate : float
        Used only by ``exponential``; inter-arrival rate.
    pair_mode : {"counterfactual", "interventional"}
        Selects paired-sample semantics (see module docstring).
    intervention_value_scale : float
        Standard deviation of the Gaussian prior on hard-intervention
        values.
    intervention_window_frac : tuple of float
        Lower/upper bounds (as fractions of total trajectory duration)
        for the intervention window length.  Default ``(0.1, 0.3)``
        matches the discrete-time roughly-30% default.
    theta_range, sigma_range, weight_scale : forwarded to :class:`ContinuousTSCMSampler`.
    seed : int
        Seeds the initial ``torch.Generator`` and ``numpy`` RNG.
    """

    # -- Which intervention kinds are currently implemented end to end. The
    # discrete pipeline supports hard, soft, and time-varying interventions;
    # for the workshop scope we start with hard interventions and will add the
    # others once the training path is validated.
    _SUPPORTED_KINDS = (InterventionKind.HARD,)

    def __init__(
        self,
        tscm_structure: str = "back_door",
        n_max: int = 41,
        t_range: tuple = (50, 200),
        schedule: str = "regular",
        dt: float = 1.0,
        jitter: float = 0.3,
        exp_rate: float = 1.0,
        pair_mode: str = "counterfactual",
        intervention_value_scale: float = 2.0,
        intervention_window_frac: tuple = (0.1, 0.3),
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        weight_scale: float = 0.5,
        seed: int = 42,
    ) -> None:
        if pair_mode not in ("counterfactual", "interventional"):
            raise ValueError(f"invalid pair_mode: {pair_mode!r}")

        self.n_max = n_max
        self.t_range = tuple(t_range)
        self.schedule = schedule
        self.dt = float(dt)
        self.jitter = float(jitter)
        self.exp_rate = float(exp_rate)
        self.pair_mode = pair_mode
        self.intervention_value_scale = float(intervention_value_scale)
        self.intervention_window_frac = tuple(intervention_window_frac)

        self.sampler = ContinuousTSCMSampler(
            structure=TSCMStructure(tscm_structure),
            theta_range=theta_range,
            sigma_range=sigma_range,
            weight_scale=weight_scale,
        )
        self.hidden_vars = self.sampler.get_hidden_vars()

        # Canonical permutation: A at index 0, Y at index N-1.  Matches
        # ``TSCMPrior`` so downstream evaluation code can reuse its
        # index conventions.
        a_idx_topo = self.sampler.get_intervention_target()
        y_idx_topo = self.sampler.get_outcome_var()
        N = self.sampler.n_vars
        middle = [i for i in range(N) if i != a_idx_topo and i != y_idx_topo]
        self.canonical_perm = [a_idx_topo] + middle + [y_idx_topo]

        self._seed = seed
        self._torch_gen = torch.Generator().manual_seed(seed)
        self._np_rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------ helpers

    @property
    def n_vars(self) -> int:
        return self.sampler.n_vars

    def sample_T(self) -> int:
        """Sample a trajectory length uniformly from ``t_range``."""
        return int(self._np_rng.randint(self.t_range[0], self.t_range[1] + 1))

    def _permute(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the canonical topological-order permutation."""
        return X[:, self.canonical_perm]

    # ------------------------------------------------------------------ sample

    def generate_sample(
        self,
        T: Optional[int] = None,
        n_queries: int = 1,
        query_mode: str = "single",
    ) -> Dict[str, torch.Tensor]:
        """Draw one trajectory with a single or multi-query batch entry.

        Parameters
        ----------
        T : int, optional
            Number of observations; sampled from ``t_range`` if omitted.
        n_queries : int
            Number of (variable, time) query points attached to this
            trajectory.  ``query_target`` / ``query_time`` / ``Y_true`` /
            ``Y_causal_effect`` become 1-D tensors of this length when
            > 1.
        query_mode : {"single", "all_pairs"}
            "single" picks ``n_queries`` random (variable, time) points.
            "all_pairs" queries every variable at every sampled time
            (``n_queries`` becomes the number of distinct times).

        Returns
        -------
        dict
            See the module docstring for field semantics.
        """
        if T is None:
            T = self.sample_T()

        # 1. Sample observation schedule
        times, dts = _build_schedule(
            schedule=self.schedule,
            T=T,
            dt=self.dt,
            jitter=self.jitter,
            exp_rate=self.exp_rate,
            generator=self._torch_gen,
        )
        span = float((times[-1] - times[0]).item())

        # 2. Sample SCM
        scm = self.sampler.sample(generator=self._torch_gen)

        # 3. Sample intervention: variable = A (canonical index 0 after perm,
        #    but we use topological index internally until after the permutation).
        a_idx_topo = self.sampler.get_intervention_target()
        win_frac = float(self._np_rng.uniform(*self.intervention_window_frac))
        # intervention window: place centred roughly in the second half
        # of the trajectory so the encoder has a meaningful pre-window.
        win_len = max(self.dt * 2, win_frac * span)
        earliest_start = times[0].item() + 0.3 * span
        latest_start = times[-1].item() - win_len
        if latest_start <= earliest_start:
            earliest_start = times[0].item() + 0.1 * span
            latest_start = times[-1].item() - self.dt
        t_int_start = float(self._np_rng.uniform(earliest_start, latest_start))
        t_int_end = t_int_start + win_len

        intervention_value = float(
            self._np_rng.randn() * self.intervention_value_scale
        )
        intervention = ContinuousIntervention(
            target=a_idx_topo,
            t_start=t_int_start,
            t_end=t_int_end,
            kind=InterventionKind.HARD,
            value=intervention_value,
        )

        # 4. Simulate paired trajectories
        if self.pair_mode == "counterfactual":
            _, X_obs, X_int = scm.sample_counterfactual_pair(
                times, dts, intervention, generator=self._torch_gen,
            )
        else:  # "interventional"
            _, X_obs, X_int = scm.sample_interventional_pair(
                times, dts, intervention, generator=self._torch_gen,
            )

        # 5. Apply canonical permutation (A -> 0, Y -> N-1, ...)
        X_obs = self._permute(X_obs)
        X_int = self._permute(X_int)
        # Re-map intervention_target / hidden_vars / outcome_var to canonical indices
        topo_to_canon = [0] * self.n_vars
        for canon_idx, topo_idx in enumerate(self.canonical_perm):
            topo_to_canon[topo_idx] = canon_idx
        intervention_target_canon = topo_to_canon[a_idx_topo]
        outcome_canon = topo_to_canon[self.sampler.get_outcome_var()]

        # 6. Find int_onset_idx (first observation time >= t_int_start)
        onset_mask = times >= t_int_start
        if onset_mask.any():
            int_onset_idx = int(onset_mask.float().argmax().item())
        else:
            int_onset_idx = T - 1

        # 7. Causal masking: zero out post-intervention observations
        X_obs_masked = X_obs.clone()
        X_obs_masked[int_onset_idx:] = 0.0

        # 8. Pad to n_max
        X_obs_padded = _pad_to_max_nodes(X_obs_masked, self.n_max)
        X_int_padded = _pad_to_max_nodes(X_int, self.n_max)
        variable_mask = torch.zeros(self.n_max)
        variable_mask[: self.n_vars] = 1.0

        # 9. Sample queries (variable, time) pairs.  Query time defaults
        #    to the intervention window midpoint offset by a small jitter
        #    sampled from the post-intervention region.
        query_target_idx, query_time_idx = self._sample_queries(
            T=T,
            n_queries=n_queries,
            query_mode=query_mode,
            int_onset_idx=int_onset_idx,
            intervention_target_canon=intervention_target_canon,
        )
        query_time_abs = times[query_time_idx]
        y_true = X_int[query_time_idx, query_target_idx]
        y_obs = X_obs[query_time_idx, query_target_idx]
        y_causal_effect = y_true - y_obs

        # 10. Times normalised to [0, 1] for compatibility with the existing mixer.
        times_norm = (times - times[0]) / max(span, 1e-6)
        t_int_start_norm = (t_int_start - times[0].item()) / max(span, 1e-6)
        t_int_end_norm = (t_int_end - times[0].item()) / max(span, 1e-6)
        query_time_norm = times_norm[query_time_idx]

        sample: Dict[str, torch.Tensor] = {
            # Trajectories
            "X_obs": X_obs_padded,
            "X_int": X_int_padded,
            "variable_mask": variable_mask,
            "num_vars": torch.tensor(self.n_vars),
            # Schedule (new for continuous-time)
            "times": times,
            "dts": dts,
            # Intervention (existing contract)
            "int_onset_idx": torch.tensor(int_onset_idx, dtype=torch.long),
            "intervention_target": torch.tensor(intervention_target_canon, dtype=torch.long),
            "intervention_type": torch.tensor(0, dtype=torch.long),  # HARD
            "intervention_value": torch.tensor(intervention_value, dtype=torch.float32),
            "intervention_time_start": torch.tensor(t_int_start_norm, dtype=torch.float32),
            "intervention_time_end": torch.tensor(t_int_end_norm, dtype=torch.float32),
            # Absolute-time intervention fields (new for continuous-time)
            "t_int_start": torch.tensor(t_int_start, dtype=torch.float32),
            "t_int_end": torch.tensor(t_int_end, dtype=torch.float32),
            # Query
            "query_target": query_target_idx.long() if n_queries > 1 else torch.tensor(int(query_target_idx.item()), dtype=torch.long),
            "query_time": query_time_norm if n_queries > 1 else torch.tensor(float(query_time_norm.item()), dtype=torch.float32),
            "t_query": query_time_abs if n_queries > 1 else torch.tensor(float(query_time_abs.item()), dtype=torch.float32),
            "Y_true": y_true if n_queries > 1 else torch.tensor(float(y_true.item()), dtype=torch.float32),
            "Y_obs": y_obs if n_queries > 1 else torch.tensor(float(y_obs.item()), dtype=torch.float32),
            "Y_causal_effect": y_causal_effect if n_queries > 1 else torch.tensor(float(y_causal_effect.item()), dtype=torch.float32),
        }
        return sample

    # ------------------------------------------------------------------ queries

    def _sample_queries(
        self,
        T: int,
        n_queries: int,
        query_mode: str,
        int_onset_idx: int,
        intervention_target_canon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(query_target_idx, query_time_idx)`` both shape ``(n_queries,)``.

        Query times are sampled from the post-intervention region so the
        downstream effect is observable.  Query targets are drawn
        uniformly from the observed variables (excluding hidden ones).
        The intervention target itself is allowed as a query target so
        the model also learns the direct effect of ``do(A := c)`` on A.
        """
        observable_vars = [
            v for v in range(self.n_vars)
            if v not in [self._canonical_hidden(h) for h in self.hidden_vars]
        ]

        t_lo = int_onset_idx
        t_hi = max(int_onset_idx + 1, T - 1)
        if t_lo >= t_hi:
            t_lo = max(0, T - 2)
            t_hi = T - 1

        if query_mode == "all_pairs":
            times_idx = torch.tensor(
                self._np_rng.randint(t_lo, t_hi + 1, size=n_queries),
                dtype=torch.long,
            )
            targets_idx = torch.tensor(
                self._np_rng.choice(observable_vars, size=n_queries, replace=True),
                dtype=torch.long,
            )
        else:
            times_idx = torch.tensor(
                self._np_rng.randint(t_lo, t_hi + 1, size=n_queries),
                dtype=torch.long,
            )
            targets_idx = torch.tensor(
                self._np_rng.choice(observable_vars, size=n_queries, replace=True),
                dtype=torch.long,
            )
        return targets_idx, times_idx

    def _canonical_hidden(self, topo_idx: int) -> int:
        """Map a topological-order hidden-variable index to canonical order."""
        topo_to_canon = [0] * self.n_vars
        for canon_idx, t_idx in enumerate(self.canonical_perm):
            topo_to_canon[t_idx] = canon_idx
        return topo_to_canon[topo_idx]

    # ------------------------------------------------------------------ batch

    def generate_batch(
        self,
        batch_size: int,
        n_queries: int = 1,
        query_mode: str = "single",
        T: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stack ``batch_size`` samples into batched tensors.

        All samples in a batch share the same ``T`` (sampled once at the
        start of the call) so the trajectory tensors can be stacked
        cleanly.  Query tensors stack along dim 0 regardless.
        """
        if T is None:
            T = self.sample_T()
        samples = [
            self.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)
            for _ in range(batch_size)
        ]
        batch: Dict[str, torch.Tensor] = {}
        for key in samples[0]:
            batch[key] = torch.stack([s[key] for s in samples], dim=0)
        return batch
