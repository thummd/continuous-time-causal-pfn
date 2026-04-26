"""CausalChamber dataset loading and formatting for Do-Over-Time-PFN.

Loads CausalChamber experiments, extracts intervention episodes,
and converts them to model-ready input format.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from causalchamber.datasets import Dataset


# Light tunnel variable groups
LT_ACTUATORS = ["pol_1", "pol_2", "l_11", "l_12", "l_21", "l_22", "l_31", "l_32"]
LT_SENSORS = [
    "red", "green", "blue", "osr_c", "v_c", "current",
    "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3",
    "diode_ir_1", "diode_vis_1", "diode_ir_2", "diode_vis_2",
    "diode_ir_3", "diode_vis_3",
]
LT_META = ["timestamp", "config", "counter", "flag", "intervention"]

# Recommended 5-variable subgraph: polarizer -> color sensors
LT_SUBGRAPH_5VAR = ["pol_1", "red", "green", "blue", "osr_c"]


class CausalChamberLoader:
    """Load and format CausalChamber data for model evaluation."""

    def __init__(
        self,
        dataset_name: str = "lt_walks_v1",
        root: str = "/tmp/causalchamber",
        subgraph_vars: Optional[List[str]] = None,
        n_max: int = 41,
        download: bool = True,
    ):
        self.dataset_name = dataset_name
        self.n_max = n_max
        self.subgraph_vars = subgraph_vars or LT_SUBGRAPH_5VAR
        self.ds = Dataset(dataset_name, root=root, download=download)
        self.var_to_idx = {v: i for i, v in enumerate(self.subgraph_vars)}

    def list_experiments(self) -> List[str]:
        return self.ds.available_experiments()

    def load_experiment(self, name: str) -> pd.DataFrame:
        """Load an experiment as a pandas DataFrame."""
        exp = self.ds.get_experiment(name)
        return exp.as_pandas_dataframe()

    def extract_episodes(
        self,
        df: pd.DataFrame,
        obs_window: int = 50,
        post_window: int = 20,
    ) -> List[Dict]:
        """Extract intervention episodes from the data.

        An episode consists of:
        - obs_window time steps of observational data before intervention change
        - The intervention specification (which actuator, what value)
        - post_window time steps of post-intervention data

        Parameters
        ----------
        df : DataFrame with time series and 'intervention' column
        obs_window : number of pre-intervention time steps
        post_window : number of post-intervention time steps

        Returns
        -------
        List of episode dicts
        """
        # Select only subgraph variables
        var_cols = [v for v in self.subgraph_vars if v in df.columns]
        data = df[var_cols].values  # (T_total, N_sub)
        intervention_col = df["intervention"].values if "intervention" in df.columns else None
        timestamps = df["timestamp"].values if "timestamp" in df.columns else None

        # Find intervention changepoints (where actuator values change significantly)
        actuator_cols = [i for i, v in enumerate(var_cols) if v in LT_ACTUATORS]
        if not actuator_cols:
            return []

        actuator_data = data[:, actuator_cols]
        diffs = np.abs(np.diff(actuator_data, axis=0))
        changepoints = np.where(diffs.max(axis=1) > 0.5)[0] + 1

        # Filter changepoints to ensure enough window before and after
        valid_cps = [cp for cp in changepoints
                     if cp >= obs_window and cp + post_window <= len(data)]

        episodes = []
        for cp in valid_cps:
            # Pre-intervention: observational
            X_obs = data[cp - obs_window: cp]  # (obs_window, N_sub)

            # Which actuator changed most at this point
            cp_diff = diffs[cp - 1]
            int_actuator_local = actuator_cols[cp_diff.argmax()]
            int_var_name = var_cols[int_actuator_local]

            # Post-intervention value
            int_value = float(data[cp, int_actuator_local])

            # Post-intervention data
            X_post = data[cp: cp + post_window]  # (post_window, N_sub)

            episode = {
                'X_obs': X_obs,
                'X_post': X_post,
                'intervention_var': int_var_name,
                'intervention_var_idx': self.var_to_idx.get(int_var_name, 0),
                'intervention_value': int_value,
                'changepoint': cp,
                'var_names': var_cols,
            }
            if timestamps is not None:
                episode['timestamps'] = np.asarray(
                    timestamps[cp - obs_window: cp + post_window], dtype=np.float64,
                )
            episodes.append(episode)

        return episodes

    def episode_to_model_input(
        self,
        episode: Dict,
        query_var: str,
        query_time_offset: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """Convert an episode to model-ready input format.

        Parameters
        ----------
        episode : dict from extract_episodes()
        query_var : variable name to query
        query_time_offset : time steps after intervention to query

        Returns
        -------
        Model input dict compatible with DoOverTimePFN.forward()
        """
        X_obs = episode['X_obs']  # (T, N_sub)
        X_post = episode['X_post']
        var_names = episode['var_names']
        T, N_sub = X_obs.shape

        # Pad to n_max
        X_obs_padded = np.zeros((T, self.n_max), dtype=np.float32)
        X_obs_padded[:, :N_sub] = X_obs

        # Variable mask
        variable_mask = np.zeros(self.n_max, dtype=np.float32)
        variable_mask[:N_sub] = 1.0

        # Intervention spec
        int_target = episode['intervention_var_idx']
        int_value = episode['intervention_value']

        # Query
        query_idx = self.var_to_idx.get(query_var, 0)
        query_time_idx = min(query_time_offset, len(X_post) - 1)
        Y_true = float(X_post[query_time_idx, query_idx])

        # Per-variable normalization (compute from X_obs)
        means = X_obs.mean(axis=0)  # (N_sub,)
        stds = X_obs.std(axis=0) + 1e-8

        X_obs_norm = np.zeros_like(X_obs_padded)
        X_obs_norm[:, :N_sub] = (X_obs - means) / stds

        Y_true_norm = (Y_true - means[query_idx]) / stds[query_idx]

        # Last observational value for this query variable (naive counterfactual)
        Y_last_obs = float(X_obs[-1, query_idx])
        Y_last_obs_norm = (Y_last_obs - means[query_idx]) / stds[query_idx]

        return {
            'X_obs': torch.tensor(X_obs_padded).unsqueeze(0),           # (1, T, N_max)
            'X_obs_norm': torch.tensor(X_obs_norm, dtype=torch.float32).unsqueeze(0),
            'variable_mask': torch.tensor(variable_mask).unsqueeze(0),   # (1, N_max)
            'intervention_target': torch.tensor([int_target], dtype=torch.long),
            'intervention_type': torch.tensor([0], dtype=torch.long),    # hard intervention
            'intervention_value': torch.tensor([int_value / (stds[int_target] + 1e-8)], dtype=torch.float32),
            'intervention_time_start': torch.tensor([1.0], dtype=torch.float32),  # at end of obs
            'intervention_time_end': torch.tensor([1.0], dtype=torch.float32),
            'query_target': torch.tensor([query_idx], dtype=torch.long),
            'query_time': torch.tensor([1.0 + query_time_offset / T], dtype=torch.float32),
            'Y_true': torch.tensor([Y_true], dtype=torch.float32),
            'Y_true_norm': torch.tensor([Y_true_norm], dtype=torch.float32),
            'Y_last_obs_norm': torch.tensor([Y_last_obs_norm], dtype=torch.float32),
            '_norm_means': torch.zeros(1, self.n_max),
            '_norm_stds': torch.ones(1, self.n_max),
        }
