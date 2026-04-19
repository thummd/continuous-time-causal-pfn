"""Batch-vectorized SCM simulation for fixed graph structures.

For sanity experiments where all B samples share the same graph topology
(e.g., all back_door), we can vectorize across the batch dimension,
eliminating per-sample Python loops and moving computation to GPU.

The time loop is still sequential (inherent to autoregressive dynamics),
but the inner variable loop and batch dimension are fully vectorized.

Usage
-----
    sim = BatchedTSCMSimulator(TSCMStructure.BACK_DOOR, use_lagged_edges=True)
    X_obs = sim.simulate(B=64, T=200, burn_in=2050, device='cuda')
    X_int = sim.simulate(B=64, T=200, burn_in=2050, device='cuda',
                         int_target=1, int_time=195, int_value=2.0)
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple

from dotime.prior.tscm_sampler import TSCMSampler, TSCMStructure


# Activation functions that work on batched tensors
BATCHED_ACTIVATIONS = [
    torch.nn.Identity(),
    torch.tanh,
    lambda x: torch.tanh(torch.relu(x)),
    torch.relu,
]


class BatchedTSCMSimulator:
    """Vectorized SCM simulation for a fixed graph structure.

    All B samples share the same graph topology but have independently
    sampled mechanism weights, biases, and noise. The simulation runs
    vectorized over (B, N) at each time step.
    """

    def __init__(
        self,
        structure: TSCMStructure,
        max_lag: int = 1,
        use_lagged_edges: bool = True,
        sigma_w: float = 0.5,
        noise_std: float = 0.3,
    ):
        self.structure = structure
        self.max_lag = max_lag
        self.sigma_w = sigma_w
        self.noise_std = noise_std

        # Build the graph once to get topology and adjacency
        sampler = TSCMSampler(structure, max_lag=max_lag, use_lagged_edges=use_lagged_edges)
        self.dag = sampler._build_dag()
        self.topo = self.dag.topo_order
        self.N = len(self.topo)
        self.hidden_vars = sampler.get_hidden_vars()

        # Topological order indices for sequential processing
        self.topo_indices = list(range(self.N))

        # Build adjacency tensors from the graph
        # adj_instant[i, j] = 1.0 if j is an instantaneous parent of i (in topo order)
        self.adj_instant = self._build_instant_adj()
        # adj_lag[k][i, j] = 1.0 if j at lag k+1 is a parent of i
        self.adj_lags = self._build_lag_adjs()

    def _build_instant_adj(self) -> torch.Tensor:
        """Build (N, N) adjacency matrix for instantaneous edges."""
        adj = torch.zeros(self.N, self.N)
        for j, parent in enumerate(self.topo):
            for i, child in enumerate(self.topo):
                if self.dag.G_0.has_edge(parent, child):
                    adj[i, j] = 1.0  # j is parent of i
        return adj

    def _build_lag_adjs(self) -> List[torch.Tensor]:
        """Build list of (N, N) adjacency matrices for lagged edges."""
        adjs = []
        for k in range(self.max_lag):
            G_k = self.dag.G_lags[k]
            adj = torch.tensor(G_k, dtype=torch.float32)  # (N, N) from numpy
            # G_k[j, i] = 1.0 means j(t-k-1) -> i(t), which is adj[i, j] in our convention
            adjs.append(adj.T)  # transpose: adj[i,j] = j is lagged parent of i
        return adjs

    def simulate(
        self,
        B: int,
        T: int,
        burn_in: int = 50,
        device: str = "cpu",
        int_target: Optional[int] = None,
        int_time: Optional[int] = None,
        int_value=None,
        divergence_threshold: float = 10.0,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate B trajectories in parallel.

        Parameters
        ----------
        B : int
            Batch size.
        T : int
            Recorded trajectory length (after burn-in).
        burn_in : int
            Steps to discard before recording.
        device : str
            Device for computation ('cpu' or 'cuda:X').
        int_target : int, optional
            Variable index to intervene on (in topo order).
        int_time : int, optional
            Time step of intervention (relative to recorded T, not burn-in).
        int_value : float, optional
            Intervention value (scalar, applied to all B samples).
            If None, sampled per-sample from N(0, sigma_w*2).
        divergence_threshold : float
            Max |value| before a sample is considered diverged.
        seed : int, optional
            Random seed.

        Returns
        -------
        buffer : (B, T, N) recorded trajectories
        valid_mask : (B,) bool tensor, True for non-diverged samples
        """
        total_T = burn_in + T
        dev = torch.device(device)

        if seed is not None:
            # Generator must match device for torch.randn(device=dev, generator=gen)
            gen = torch.Generator(device=dev).manual_seed(seed)
        else:
            gen = None

        # Move adjacency to device
        adj_inst = self.adj_instant.to(dev)
        adj_lags = [a.to(dev) for a in self.adj_lags]

        # Sample mechanism weights: (B, N, N) masked by adjacency
        W_inst = torch.randn(B, self.N, self.N, device=dev, generator=gen) * self.sigma_w
        W_inst = W_inst * adj_inst.unsqueeze(0)  # zero out non-edges

        W_lag = []
        for k in range(self.max_lag):
            Wk = torch.randn(B, self.N, self.N, device=dev, generator=gen) * self.sigma_w
            Wk = Wk * adj_lags[k].unsqueeze(0)
            W_lag.append(Wk)

        bias = torch.randn(B, self.N, device=dev, generator=gen) * self.sigma_w * 0.5

        # Sample per-variable activation index: (B, N) -> index into BATCHED_ACTIVATIONS
        n_acts = len(BATCHED_ACTIVATIONS)
        act_idx = torch.randint(0, n_acts, (B, self.N), device=dev, generator=gen)

        # Pre-sample noise: (B, total_T, N)
        noise = torch.randn(B, total_T, self.N, device=dev, generator=gen) * self.noise_std

        # Intervention setup — int_value can be None, scalar, or a (B,) tensor
        do_intervention = (int_target is not None and int_time is not None)
        if do_intervention and int_value is None:
            int_value_t = torch.randn(B, device=dev, generator=gen) * self.sigma_w * 2.0
        elif do_intervention and torch.is_tensor(int_value):
            int_value_t = int_value.to(dev).float()
        elif do_intervention:
            int_value_t = torch.full((B,), float(int_value), device=dev)
        else:
            int_value_t = None

        # Forward simulation buffer: (B, total_T, N)
        buffer = torch.zeros(B, total_T, self.N, device=dev)
        valid = torch.ones(B, dtype=torch.bool, device=dev)

        for t in range(total_T):
            # Process variables in topological order
            for i in self.topo_indices:
                # Check hard intervention
                if do_intervention and i == int_target and (t - burn_in) == int_time:
                    buffer[:, t, i] = int_value_t
                    continue

                # Instantaneous contribution: sum_j W[b,i,j] * buffer[b,t,j] for parents j
                instant = (W_inst[:, i, :] * buffer[:, t, :]).sum(dim=-1)  # (B,)

                # Lagged contributions
                lagged = torch.zeros(B, device=dev)
                for k in range(self.max_lag):
                    if t >= k + 1:
                        lagged = lagged + (W_lag[k][:, i, :] * buffer[:, t - k - 1, :]).sum(dim=-1)

                # Combined + bias + noise
                combined = instant + lagged + bias[:, i] + noise[:, t, i]

                # Apply per-sample activation. Compute all 4 activations in
                # parallel (no kernel-launch overhead penalty for tiny ops on
                # GPU) and select via per-sample index. Avoids mask indexing
                # which forces sync/dynamic shape on GPU.
                # Activations: 0=Identity, 1=tanh, 2=tanh(relu), 3=relu
                act_id = combined
                act_tanh = torch.tanh(combined)
                act_relu = torch.relu(combined)
                act_tanhrelu = torch.tanh(act_relu)
                # Stack: (B, 4) then gather by act_idx[:, i]
                stacked = torch.stack([act_id, act_tanh, act_tanhrelu, act_relu], dim=-1)
                result = stacked.gather(-1, act_idx[:, i:i+1]).squeeze(-1)

                buffer[:, t, i] = result

            # Periodic divergence check
            if t > 0 and t % 50 == 0:
                diverged = buffer[:, t, :].abs().max(dim=-1).values > divergence_threshold
                valid = valid & ~diverged
                # Zero out diverged samples to prevent NaN propagation
                buffer[diverged, t:, :] = 0.0

        recorded = buffer[:, burn_in:, :]  # (B, T, N)
        return recorded, valid

    def generate_pairs(
        self,
        B: int,
        T: int,
        burn_in: int = 50,
        device: str = "cpu",
        intervention_scale: float = 4.0,
        seed: int = 42,
        positivity_clip: bool = False,
    ) -> dict:
        """Generate B observational + interventional trajectory pairs.

        Parameters
        ----------
        positivity_clip : bool
            If True, per-sample clip each int_value to [obs_mu - 3σ, obs_mu + 3σ]
            where (obs_mu, obs_sigma) are computed from the pre-intervention
            observational window of that sample's treatment variable.

        Returns a dict with tensors ready for batched processing:
            X_obs: (B, T, N)
            X_int: (B, T, N)
            int_target: (B,) int — intervention target index
            int_value: (B,) float — intervention values (per-sample)
            int_time: (B,) int — intervention time (single common value for batch)
            valid: (B,) bool — non-diverged samples
        """
        int_target_idx = self.topo.index('A')

        gen = torch.Generator().manual_seed(seed)
        t_lo = min(10, T - 1)
        t_hi = max(t_lo + 1, T - 10)
        # Use a single common int_time for the whole batch (so we can vectorize
        # the interventional simulate() call). Per-sample values are still used.
        common_int_time = int(torch.randint(t_lo, t_hi, (1,), generator=gen).item())

        # Per-sample intervention values: N(0, intervention_scale)
        int_values = torch.randn(B, generator=gen) * intervention_scale

        # 1) Observational simulation
        X_obs, valid_obs = self.simulate(
            B, T, burn_in=burn_in, device=device, seed=seed,
        )

        # 2) Per-sample positivity clip — keeps int_value within observed 3σ of
        #    the treatment variable's pre-intervention support. Without this,
        #    many int_values land in the tails and the model sees distributions
        #    it can't interpolate.
        if positivity_clip:
            pre_int = X_obs[:, :common_int_time, int_target_idx].detach().cpu()  # (B, t<)
            if pre_int.shape[1] > 1:
                mu = pre_int.mean(dim=1)                       # (B,)
                sigma = pre_int.std(dim=1).clamp(min=1e-4)     # (B,)
                lo = mu - 3.0 * sigma
                hi = mu + 3.0 * sigma
                int_values = torch.clamp(int_values, min=lo, max=hi)

        # 3) Interventional simulation with per-sample int_values
        X_int, valid_int = self.simulate(
            B, T, burn_in=burn_in, device=device,
            int_target=int_target_idx,
            int_time=common_int_time,
            int_value=int_values,
            seed=seed + 1,
        )

        valid = valid_obs & valid_int

        return {
            'X_obs': X_obs,
            'X_int': X_int,
            'int_target': torch.full((B,), int_target_idx, dtype=torch.long),
            'int_value': int_values,
            'int_time': torch.full((B,), common_int_time, dtype=torch.long),
            'valid': valid,
            'N': self.N,
            'topo': self.topo,
            'hidden_vars': self.hidden_vars,
        }
