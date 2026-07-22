"""Vectorised Euler--Maruyama simulation for :class:`ContinuousSCM`.

The reference implementation in ``continuous_scm.py`` advances the state one
variable at a time inside a Python loop (``ContinuousSCM._step``).  That is
correct but slow: at ``num_substeps = 8`` a single fine-grid trajectory spends
almost all of its time in per-variable tensor bookkeeping, and training the
neural-drift ablation cells took ~10 h each because of it.

The ``-finegrid`` tree added a vectorised path, but only for *homogeneous*
graphs (all-OU or all-neural).  The paper's neural prior is a per-variable
Bernoulli(``p_neural``) *mixture*, so every neural cell silently fell back to
the slow loop.  This module vectorises the **mixed** case: OU and neural
variables are integrated together, across all ``n`` variables at once, by
computing both drift forms and selecting per variable.

Correctness is not assumed -- ``tests/test_vectorized_equiv.py`` asserts this
path reproduces ``ContinuousSCM._step`` to floating-point tolerance on OU,
neural, and mixed graphs, with hard / soft / time-varying interventions and
under the shared-noise counterfactual pairing.  (The sibling discrete-time
repo learned the hard way that a vectorisation bug which re-samples mechanisms
or noise between the observational and interventional runs silently destroys
the counterfactual target; this path re-uses the caller's ``noise`` verbatim
and never touches the mechanisms.)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .ou_mechanism import OUMechanism
from .neural_drift_mechanism import NeuralDriftMechanism
from .continuous_scm import ContinuousIntervention, InterventionKind


class _VectorizedPlan:
    """Padded per-variable tensors precomputed once per SCM.

    Encodes exactly the arithmetic of ``OUMechanism.drift`` /
    ``NeuralDriftMechanism.drift`` so the batched step is numerically the same
    map, only evaluated across all variables simultaneously.
    """

    def __init__(self, mechanisms, device, dtype):
        n = len(mechanisms)
        self.n = n
        self.device = device
        self.dtype = dtype

        self.theta = torch.tensor([float(m.theta) for m in mechanisms],
                                  device=device, dtype=dtype)
        self.sigma = torch.tensor([float(m.sigma) for m in mechanisms],
                                  device=device, dtype=dtype)
        self.is_neural = torch.tensor(
            [isinstance(m, NeuralDriftMechanism) for m in mechanisms],
            device=device, dtype=torch.bool,
        )

        # --- OU parent linear map: W[v, u] = w_{v,u} for OU parents of v ---
        # Applied as (W @ x); rows for neural variables are left at zero and
        # their contribution is supplied by the MLP branch instead.
        self.W = torch.zeros(n, n, device=device, dtype=dtype)
        for v, m in enumerate(mechanisms):
            if isinstance(m, OUMechanism) and len(m.parents) > 0:
                w = m.parent_weights.to(device=device, dtype=dtype)
                for k, u in enumerate(m.parents):
                    self.W[v, int(u)] = w[k]

        # --- Neural MLP batch: pad every neural var to a common input dim ---
        neural = [(v, m) for v, m in enumerate(mechanisms)
                  if isinstance(m, NeuralDriftMechanism)]
        self.has_neural = len(neural) > 0
        if self.has_neural:
            hidden = neural[0][1].W1.shape[0]
            # input dim per neural var is 1 (self) + #parents; pad to max.
            in_dims = [1 + len(m.parents) for _, m in neural]
            d_max = max(in_dims)
            nn = len(neural)

            self.neural_rows = torch.tensor([v for v, _ in neural],
                                            device=device, dtype=torch.long)
            self.W1 = torch.zeros(nn, hidden, d_max, device=device, dtype=dtype)
            self.b1 = torch.zeros(nn, hidden, device=device, dtype=dtype)
            self.W2 = torch.zeros(nn, 1, hidden, device=device, dtype=dtype)
            self.b2 = torch.zeros(nn, 1, device=device, dtype=dtype)
            self.out_scale = torch.tensor([float(m.out_scale) for _, m in neural],
                                          device=device, dtype=dtype)
            # gather_idx[j, d] = state index feeding input slot d of neural var j
            # (slot 0 = self, slots 1.. = parents); padded slots gather index 0
            # but are masked to 0 so they contribute nothing.
            self.gather_idx = torch.zeros(nn, d_max, device=device, dtype=torch.long)
            self.gather_mask = torch.zeros(nn, d_max, device=device, dtype=dtype)
            for j, (v, m) in enumerate(neural):
                self.W1[j, :, :in_dims[j]] = m.W1.to(device=device, dtype=dtype)
                self.b1[j] = m.b1.to(device=device, dtype=dtype)
                self.W2[j, 0, :] = m.W2.to(device=device, dtype=dtype).reshape(-1)
                self.b2[j, 0] = m.b2.to(device=device, dtype=dtype).reshape(-1)[0]
                self.gather_idx[j, 0] = v
                self.gather_mask[j, 0] = 1.0
                for k, u in enumerate(m.parents):
                    self.gather_idx[j, k + 1] = int(u)
                    self.gather_mask[j, k + 1] = 1.0

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorised drift for the whole state vector ``x`` of shape (n,)."""
        d = -self.theta * x                       # shared dissipative term
        # OU parent sum for every variable (neural rows of W are zero).
        d = d + self.W @ x
        if self.has_neural:
            inp = x[self.gather_idx] * self.gather_mask          # (nn, d_max)
            h = torch.tanh(torch.bmm(self.W1, inp.unsqueeze(-1)).squeeze(-1) + self.b1)
            o = torch.tanh(torch.bmm(self.W2, h.unsqueeze(-1)).squeeze(-1) + self.b2).squeeze(-1)
            nn_drift = self.out_scale * o                        # (nn,)
            # Neural variables: replace the (zero) OU parent sum with the MLP.
            d = d.index_copy(0, self.neural_rows,
                             -self.theta[self.neural_rows] * x[self.neural_rows] + nn_drift)
        return d


def simulate_vectorized(
    scm,
    times: torch.Tensor,
    dts: torch.Tensor,
    intervention: Optional[ContinuousIntervention],
    x0: Optional[torch.Tensor],
    noise: torch.Tensor,
    num_substeps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in vectorised replacement for ``ContinuousSCM.simulate``'s loop.

    Assumes ``noise`` is already drawn (shape ``((T-1)*num_substeps, n)``) and
    every mechanism is OU or neural.  Preserves the reference implementation's
    noise indexing and intervention timing (evaluated at the forward time of
    each sub-step, snapped to the observation time on the last sub-step).
    """
    plan: _VectorizedPlan = scm._vec_plan
    n = plan.n
    T = times.numel()
    x = (torch.zeros(n, device=plan.device, dtype=plan.dtype)
         if x0 is None else x0.to(device=plan.device, dtype=plan.dtype).clone())

    traj = torch.empty(T, n, device=plan.device, dtype=plan.dtype)
    traj[0] = x
    tgt = intervention.target if intervention is not None else -1
    for i in range(T - 1):
        fine_dt = dts[i] / num_substeps
        sqrt_dt = torch.sqrt(fine_dt)
        t_i = float(times[i].item())
        t_next = float(times[i + 1].item())
        for k in range(num_substeps):
            t_fine = t_next if k == num_substeps - 1 else t_i + (k + 1) * float(fine_dt.item())
            drift = plan.drift(x)
            if intervention is not None and intervention.is_active(t_fine) \
                    and intervention.kind is InterventionKind.SOFT:
                drift = drift.clone()
                drift[tgt] = drift[tgt] + float(intervention.eval_value(t_fine))
            x = x + drift * fine_dt + plan.sigma * sqrt_dt * noise[i * num_substeps + k]
            if intervention is not None and intervention.is_active(t_fine) and \
                    intervention.kind in (InterventionKind.HARD, InterventionKind.TIME_VARYING):
                x = x.clone()
                x[tgt] = float(intervention.eval_value(t_fine))
        traj[i + 1] = x
    return times, traj


def can_vectorize(mechanisms) -> bool:
    """True when every mechanism is a plain OU or neural drift.

    Anything else (e.g. regime-switching wrappers) falls back to the reference
    loop, so the vectorised path never changes what those samples generate.
    """
    return all(isinstance(m, (OUMechanism, NeuralDriftMechanism)) for m in mechanisms)
