"""Targeted Temporal SCM Sampler for identifiability case studies.

Generates specific causal structures (confounder, mediator, etc.)
with random mechanisms and noise, enabling systematic identifiability
analysis of the trained model.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Optional, List
from enum import Enum

from causal_time_prior.temporal_graph import TemporalDAG
from causal_time_prior.temporal_scm import TemporalSCM
from causal_time_prior.temporal_mechanism import TemporalMechanism
from dopfnprior.utils.sampling import TorchDistributionSampler
from dopfnprior.configs.default_config import Tanh, TanhReLU


class TSCMStructure(Enum):
    """Named temporal causal structures for identifiability studies."""
    OBSERVED_CONFOUNDER = "observed_confounder"       # Z -> X, Z -> Y
    MEDIATOR = "mediator"                              # X -> M -> Y
    CONFOUNDER_MEDIATOR = "confounder_mediator"        # Z -> X -> M -> Y, Z -> Y
    UNOBSERVED_CONFOUNDER = "unobserved_confounder"    # U -> X, U -> Y (U hidden)
    BACK_DOOR = "back_door"                            # Z -> X, Z -> Y, X -> Y
    FRONT_DOOR = "front_door"                          # X -> M -> Y, U -> X, U -> Y (U hidden)


# Default activations for random mechanism sampling
DEFAULT_ACTIVATIONS = [
    nn.Identity(),
    Tanh(),
    TanhReLU(),
    nn.ReLU(),
]


class TSCMSampler:
    """Samples temporal SCMs with specific causal structures.

    Unlike CausalTimePrior which randomly samples graph structure,
    this builds a fixed named structure with random mechanisms/noise.
    """

    def __init__(
        self,
        structure: TSCMStructure,
        max_lag: int = 1,
        activations: Optional[List[nn.Module]] = None,
        sigma_w: float = 1.0,
        sigma_b: float = 0.5,
        noise_std: float = 0.3,
        device: str = "cpu",
    ):
        self.structure = structure
        self.max_lag = max_lag
        self.activations = activations or DEFAULT_ACTIVATIONS
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.noise_std = noise_std
        self.device = torch.device(device)

    def sample(self, generator: Optional[torch.Generator] = None) -> TemporalSCM:
        """Sample a temporal SCM with the specified structure.

        Returns
        -------
        TemporalSCM with the fixed structure but random mechanisms/noise.
        """
        dag = self._build_dag()
        node_names = dag.topo_order
        mechanisms = self._sample_mechanisms(node_names, dag, generator)
        noise = self._sample_noise(node_names)
        return TemporalSCM(dag, mechanisms, noise, device=self.device)

    def get_hidden_vars(self) -> List[int]:
        """Return indices of hidden (unobserved) variables, if any."""
        if self.structure == TSCMStructure.UNOBSERVED_CONFOUNDER:
            return [0]  # U is first variable
        elif self.structure == TSCMStructure.FRONT_DOOR:
            return [0]  # U is first variable
        return []

    def _build_dag(self) -> TemporalDAG:
        """Construct a TemporalDAG for the specified structure."""
        builder = getattr(self, f'_build_{self.structure.value}')
        return builder()

    # --- Structure builders ---

    def _build_observed_confounder(self) -> TemporalDAG:
        """Z -> X, Z -> Y. All observed. Lag on Z -> Y."""
        G_0 = nx.DiGraph()
        nodes = ["Z", "X", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("Z", "X")  # instantaneous
        # Z -> Y via lag

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:  # lag-1
                G_k[0, 2] = 1.0  # Z(t-1) -> Y(t)
                # Autoregressive
                for i in range(N):
                    G_k[i, i] = 1.0
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_mediator(self) -> TemporalDAG:
        """X -> M -> Y with lag on X -> M."""
        G_0 = nx.DiGraph()
        nodes = ["X", "M", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("M", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                G_k[0, 1] = 1.0  # X(t-1) -> M(t)
                for i in range(N):
                    G_k[i, i] = 1.0
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_confounder_mediator(self) -> TemporalDAG:
        """Z -> X -> M -> Y, Z -> Y."""
        G_0 = nx.DiGraph()
        nodes = ["Z", "X", "M", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("Z", "X")  # instantaneous
        G_0.add_edge("X", "M")  # instantaneous
        G_0.add_edge("M", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                G_k[0, 3] = 1.0  # Z(t-1) -> Y(t)
                for i in range(N):
                    G_k[i, i] = 1.0
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_unobserved_confounder(self) -> TemporalDAG:
        """U -> X, U -> Y where U is hidden (index 0)."""
        G_0 = nx.DiGraph()
        nodes = ["U", "X", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("U", "X")  # instantaneous
        G_0.add_edge("U", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                for i in range(N):
                    G_k[i, i] = 1.0
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_back_door(self) -> TemporalDAG:
        """Z -> X, Z -> Y, X -> Y. Z blocks the back-door path."""
        G_0 = nx.DiGraph()
        nodes = ["Z", "X", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("Z", "X")  # instantaneous
        G_0.add_edge("Z", "Y")  # instantaneous
        G_0.add_edge("X", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                for i in range(N):
                    G_k[i, i] = 1.0
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_front_door(self) -> TemporalDAG:
        """X -> M -> Y, U -> X, U -> Y. U is hidden. M satisfies front-door."""
        G_0 = nx.DiGraph()
        nodes = ["U", "X", "M", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("U", "X")  # instantaneous
        G_0.add_edge("U", "Y")  # instantaneous
        G_0.add_edge("X", "M")  # instantaneous
        G_0.add_edge("M", "Y")  # instantaneous

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                for i in range(N):
                    G_k[i, i] = 1.0
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    # --- Mechanism/noise sampling ---

    def _sample_mechanisms(
        self,
        node_names: List[str],
        dag: TemporalDAG,
        generator: Optional[torch.Generator],
    ) -> dict:
        """Sample random mechanisms for each node."""
        mechanisms = {}
        for v in node_names:
            # Pick random activation
            act_idx = int(torch.randint(0, len(self.activations), (1,),
                                        generator=generator).item())
            activation = self.activations[act_idx]

            mechanisms[v] = TemporalMechanism(
                node_names=node_names,
                activation=activation,
                num_lags=dag.K,
                device=self.device,
                generator=generator,
                sigma_w=self.sigma_w,
                sigma_b=self.sigma_b,
            )
        return mechanisms

    def _sample_noise(self, node_names: List[str]) -> dict:
        """Create noise distributions for each node."""
        noise = {}
        for v in node_names:
            noise[v] = TorchDistributionSampler(
                torch.distributions.Normal(0.0, self.noise_std)
            )
        return noise
