"""CausalChamber dataset loading and formatting for Do-Over-Time-PFN.

Loads CausalChamber experiments, extracts intervention episodes,
and converts them to model-ready input format.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional, Tuple

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


# Wind-tunnel variable groups (wt_intake_impulse_v1).  The chamber's
# `intervention` column is a 1-row pulse marker that flags toggle
# events on the intake-fan setpoint `load_in`.  `load_in` takes two
# discrete values (low: 0.01, high: 1.0) and roughly half the
# intervention pulses are "decoys" that re-assert the current value
# (no SCM-level effect); the other half are real load_in transitions
# (do(load_in := c) for c in {0.01, 1.0}).
WT_TREATMENT = "load_in"
WT_SUBGRAPH_5VAR = ["load_in", "current_in", "rpm_in", "pressure_intake", "pressure_downwind"]


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
        intervention_source: Literal["actuator_step", "intervention_column"] = "actuator_step",
        treatment_var: Optional[str] = None,
        require_value_change: bool = True,
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
        intervention_source : which signal defines an episode boundary.
            ``"actuator_step"`` (default, used by ``lt_walks_v1``) runs
            a change-point detector over the actuator columns of the
            chosen subgraph.  ``"intervention_column"`` (used by
            ``wt_intake_impulse_v1``) reads the chamber's explicit
            ``intervention`` flag column directly.
        treatment_var : variable that toggles at each intervention
            pulse (only used when ``intervention_source ==
            "intervention_column"``).  Must be present in
            ``self.subgraph_vars``.  The intervention value reported
            per episode is the post-pulse value of this variable.
        require_value_change : when ``intervention_source ==
            "intervention_column"``, drop pulses where
            ``treatment_var`` does not actually change across the
            pulse (the wind-tunnel rig sends ~one decoy pulse for
            every two real toggles -- they reassert the current
            setpoint and are valid SCM events but produce no
            observable response).  Default ``True`` keeps only real
            transitions.

        Returns
        -------
        List of episode dicts
        """
        if intervention_source == "intervention_column":
            return self._extract_episodes_from_intervention_column(
                df,
                obs_window=obs_window,
                post_window=post_window,
                treatment_var=treatment_var,
                require_value_change=require_value_change,
            )

        # Select only subgraph variables
        var_cols = [v for v in self.subgraph_vars if v in df.columns]
        data = df[var_cols].values  # (T_total, N_sub)

        # Real per-row physical timestamps (seconds).  CausalChamber
        # ``lt_walks_v1`` exposes a ``timestamp`` float column; rows are
        # nominally ~6.75 Hz with occasional ~0.25 s gaps (jitter from
        # the chamber control loop).  Treating these as a uniform 10 Hz
        # grid -- as previous versions of the CT adapter did -- is
        # exactly the schedule conflation the workshop paper critiques,
        # so we propagate the real values to downstream code.
        if "timestamp" in df.columns:
            timestamps_full = df["timestamp"].astype(float).values
        else:
            timestamps_full = None

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
            if timestamps_full is not None:
                # Slice and re-anchor to seconds-since-episode-start so
                # downstream code does not have to know that
                # CausalChamber's raw timestamps live in some absolute
                # epoch (~ 446 763 s in the v1 release).
                ts_obs = timestamps_full[cp - obs_window: cp]
                ts_post = timestamps_full[cp: cp + post_window]
                t0 = float(ts_obs[0])
                episode['timestamps_obs'] = (ts_obs - t0).astype(np.float32)
                episode['timestamps_post'] = (ts_post - t0).astype(np.float32)
                # Useful for downstream window construction: physical
                # gap from the last pre-intervention sample to the
                # first post-intervention sample.
                episode['intervention_dt'] = float(ts_post[0] - ts_obs[-1])

            episodes.append(episode)

        return episodes

    def _extract_episodes_from_intervention_column(
        self,
        df: pd.DataFrame,
        obs_window: int,
        post_window: int,
        treatment_var: Optional[str],
        require_value_change: bool,
    ) -> List[Dict]:
        """Episode boundaries straight from the chamber's `intervention` flag.

        The wind-tunnel ``wt_intake_impulse_v1`` rig encodes every
        treatment toggle as a single-row ``intervention=1`` pulse.
        We treat each rising edge (``0 -> 1``) as the intervention
        onset and read the new ``treatment_var`` value off the post-
        pulse row.
        """
        if "intervention" not in df.columns:
            raise ValueError(
                "intervention_source='intervention_column' requires the "
                "DataFrame to have an 'intervention' column"
            )
        if treatment_var is None:
            raise ValueError(
                "intervention_source='intervention_column' requires "
                "`treatment_var` (the variable that toggles at each pulse)"
            )

        var_cols = [v for v in self.subgraph_vars if v in df.columns]
        if treatment_var not in var_cols:
            raise ValueError(
                f"treatment_var={treatment_var!r} not in subgraph "
                f"({var_cols}); add it to ``subgraph_vars``"
            )
        treat_idx_local = var_cols.index(treatment_var)
        data = df[var_cols].values  # (T_total, N_sub)

        if "timestamp" in df.columns:
            timestamps_full = df["timestamp"].astype(float).values
        else:
            timestamps_full = None

        iv = df["intervention"].values
        # Rising edges of the intervention flag: each marks one toggle event.
        rising = np.where(np.diff(iv) > 0)[0] + 1

        episodes: List[Dict] = []
        for cp in rising:
            if cp < obs_window or cp + post_window > len(data):
                continue
            # Pre-pulse value (one row before the flag rises).
            pre_val = float(data[cp - 1, treat_idx_local])
            post_val = float(data[cp, treat_idx_local])
            if require_value_change and post_val == pre_val:
                # Decoy pulse: the chamber re-asserted the current
                # setpoint without changing it.  Skip because it
                # produces no observable response and would inflate
                # naive-baseline RMSE without testing causal-effect
                # tracking.
                continue

            X_obs = data[cp - obs_window: cp]
            X_post = data[cp: cp + post_window]

            episode = {
                "X_obs": X_obs,
                "X_post": X_post,
                "intervention_var": treatment_var,
                "intervention_var_idx": self.var_to_idx.get(treatment_var, treat_idx_local),
                "intervention_value": post_val,
                "intervention_value_pre": pre_val,
                "changepoint": int(cp),
                "var_names": var_cols,
            }
            if timestamps_full is not None:
                ts_obs = timestamps_full[cp - obs_window: cp]
                ts_post = timestamps_full[cp: cp + post_window]
                t0 = float(ts_obs[0])
                episode["timestamps_obs"] = (ts_obs - t0).astype(np.float32)
                episode["timestamps_post"] = (ts_post - t0).astype(np.float32)
                episode["intervention_dt"] = float(ts_post[0] - ts_obs[-1])

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
