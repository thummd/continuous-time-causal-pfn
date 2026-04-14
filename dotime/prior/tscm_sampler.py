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
    """Named temporal causal structures for identifiability studies.

    Identification strategies (cf. identifiability.tex):
    - Backdoor: OBSERVED_CONFOUNDER, BACK_DOOR, CONFOUNDER_MEDIATOR
      Adjust via observed covariates that block all back-door paths.
    - Frontdoor: FRONT_DOOR, MEDIATOR
      Adjust via mediator when confounder is hidden.
    - Trivially identified: RCT_NO_CONFOUNDING
      No confounding => p(Y|do(A)) = p(Y|A).
    - IV: INSTRUMENTAL_VARIABLE
      Z -> A -> Y with hidden confounding U -> A, U -> Y.
    - Non-identifiable: UNOBSERVED_CONFOUNDER
      Hidden confounder, no mediator or instrument. Tests model robustness.
    """
    OBSERVED_CONFOUNDER = "observed_confounder"       # Z -> X, Z -> Y (backdoor)
    MEDIATOR = "mediator"                              # X -> M -> Y (frontdoor, trivial)
    CONFOUNDER_MEDIATOR = "confounder_mediator"        # Z -> X -> M -> Y, Z -> Y (backdoor)
    UNOBSERVED_CONFOUNDER = "unobserved_confounder"    # U -> X, U -> Y (non-identifiable)
    BACK_DOOR = "back_door"                            # Z -> X, Z -> Y, X -> Y (backdoor)
    FRONT_DOOR = "front_door"                          # X -> M -> Y, U -> X, U -> Y (frontdoor)
    INSTRUMENTAL_VARIABLE = "instrumental_variable"    # Z -> A -> Y, U -> A, U -> Y (IV)
    RCT_NO_CONFOUNDING = "rct_no_confounding"          # A -> Y (trivially identified)


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
        use_lagged_edges: bool = True,
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
        self.use_lagged_edges = use_lagged_edges

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
        if self.structure in (
            TSCMStructure.UNOBSERVED_CONFOUNDER,
            TSCMStructure.FRONT_DOOR,
            TSCMStructure.INSTRUMENTAL_VARIABLE,
        ):
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
        """X -> A (inst.), A -> Y (inst.), X(t-1) -> Y(t) (lagged if enabled).

        X = confounder, A = treatment, Y = outcome.
        X causes both A and Y. Backdoor adjustment requires conditioning on X.
        The lagged X(t-1)->Y(t) edge (when use_lagged_edges=True) makes the
        confounding observable from history.
        """
        G_0 = nx.DiGraph()
        nodes = ["X", "A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("X", "A")  # confounder -> treatment
        G_0.add_edge("A", "Y")  # treatment -> outcome

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                if self.use_lagged_edges:
                    G_k[0, 2] = 1.0  # X(t-1) -> Y(t): lagged confounding
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive self-loops
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_front_door(self) -> TemporalDAG:
        """U -> X, U -> Y (inst.), X -> M (inst.), M(t-1) -> Y(t) (lagged if enabled).

        U = hidden confounder, X = treatment, M = mediator, Y = outcome.
        M satisfies the front-door criterion. The lagged M(t-1)->Y(t) edge
        (when use_lagged_edges=True) makes the mediation observable from history.
        """
        G_0 = nx.DiGraph()
        nodes = ["U", "X", "M", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("U", "X")  # hidden confounder -> treatment
        G_0.add_edge("U", "Y")  # hidden confounder -> outcome
        G_0.add_edge("X", "M")  # treatment -> mediator

        N = len(nodes)
        G_lags = []
        for k in range(self.max_lag):
            G_k = np.zeros((N, N), dtype=np.float32)
            if k == 0:
                if self.use_lagged_edges:
                    # Topo order is [U, X, Y, M] so M=3, Y=2
                    G_k[3, 2] = 1.0  # M(t-1) -> Y(t): lagged mediation
                for i in range(N):
                    G_k[i, i] = 1.0  # autoregressive self-loops
            G_lags.append(G_k)

        topo = list(nx.topological_sort(G_0))
        return TemporalDAG(G_0, G_lags, self.max_lag, topo)

    def _build_instrumental_variable(self) -> TemporalDAG:
        """Z -> A -> Y, U -> A, U -> Y. U hidden. Z is an instrument for A->Y."""
        G_0 = nx.DiGraph()
        nodes = ["U", "Z", "A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("U", "A")  # hidden confounder -> treatment
        G_0.add_edge("U", "Y")  # hidden confounder -> outcome
        G_0.add_edge("Z", "A")  # instrument -> treatment
        G_0.add_edge("A", "Y")  # treatment -> outcome

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

    def _build_rct_no_confounding(self) -> TemporalDAG:
        """A -> Y. No confounding. Trivially identified: p(Y|do(A)) = p(Y|A)."""
        G_0 = nx.DiGraph()
        nodes = ["A", "Y"]
        G_0.add_nodes_from(nodes)
        G_0.add_edge("A", "Y")  # direct causal effect

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
