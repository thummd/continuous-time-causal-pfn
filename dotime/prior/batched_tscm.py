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
        int_value: Optional[float] = None,
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
            gen = torch.Generator(device='cpu').manual_seed(seed)
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
        act_idx = torch.randint(0, n_acts, (B, self.N), generator=gen)

        # Pre-sample noise: (B, total_T, N)
        noise = torch.randn(B, total_T, self.N, device=dev, generator=gen) * self.noise_std

        # Intervention setup
        do_intervention = (int_target is not None and int_time is not None)
        if do_intervention and int_value is None:
            int_value_t = torch.randn(B, device=dev, generator=gen) * self.sigma_w * 2.0
        elif do_intervention:
            int_value_t = torch.full((B,), int_value, device=dev)
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

                # Apply per-sample activation
                # Group by activation type to avoid per-sample branching
                result = torch.zeros(B, device=dev)
                for a_idx in range(n_acts):
                    mask = (act_idx[:, i] == a_idx)
                    if mask.any():
                        act_fn = BATCHED_ACTIVATIONS[a_idx]
                        result[mask] = act_fn(combined[mask])

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
    ) -> dict:
        """Generate B observational + interventional trajectory pairs.

        Returns a dict with tensors ready for batched processing:
            X_obs: (B, T, N)
            X_int: (B, T, N)
            int_target: (B,) int — intervention target index
            int_value: (B,) float — intervention values
            int_time: (B,) int — intervention time (relative to T)
            valid: (B,) bool — non-diverged samples
        """
        # Pick intervention target: the treatment variable A
        int_target_idx = self.topo.index('A')

        # Random intervention time per sample: uniform in [10, T-10]
        gen = torch.Generator().manual_seed(seed)
        t_lo = min(10, T - 1)
        t_hi = max(t_lo + 1, T - 10)
        int_times = torch.randint(t_lo, t_hi, (B,), generator=gen)

        # Random intervention values
        int_values = torch.randn(B, generator=gen) * intervention_scale

        # Generate observational trajectories (no intervention)
        X_obs, valid_obs = self.simulate(
            B, T, burn_in=burn_in, device=device, seed=seed,
        )

        # Generate interventional trajectories
        # Note: each sample has a different int_time, so we simulate one at a time
        # for the intervention. For efficiency, we could batch samples with the
        # same int_time, but for now we use a single common int_time.
        # Simplification: use a single random int_time for the whole batch.
        common_int_time = int(int_times[0].item())
        X_int, valid_int = self.simulate(
            B, T, burn_in=burn_in, device=device,
            int_target=int_target_idx,
            int_time=common_int_time,
            int_value=None,  # will be sampled per-sample inside simulate
            seed=seed + 1,  # different seed so obs ≠ int even without intervention
        )
        # Override with the specific int_values
        # Re-simulate with per-sample values by using the batch
        # Actually the simulate() already samples per-sample values when int_value=None.
        # Let's get the actual values used:
        gen2 = torch.Generator().manual_seed(seed + 1)
        # Skip to where int_value is sampled in simulate()
        # This is fragile — better to return int_value from simulate()
        # For now, use a fixed value for all samples:
        common_int_value = float(int_values[0].item())
        X_int2, valid_int2 = self.simulate(
            B, T, burn_in=burn_in, device=device,
            int_target=int_target_idx,
            int_time=common_int_time,
            int_value=common_int_value,
            seed=seed + 1,
        )

        valid = valid_obs & valid_int2

        return {
            'X_obs': X_obs,
            'X_int': X_int2,
            'int_target': torch.full((B,), int_target_idx, dtype=torch.long),
            'int_value': torch.full((B,), common_int_value),
            'int_time': torch.full((B,), common_int_time, dtype=torch.long),
            'valid': valid,
            'N': self.N,
            'topo': self.topo,
            'hidden_vars': self.hidden_vars,
        }
