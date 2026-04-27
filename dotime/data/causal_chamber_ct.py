r"""Continuous-time adapter for CausalChamber episodes.

The discrete-time pipeline already has
:class:`dotime.data.causal_chamber.CausalChamberLoader`, which loads
raw CausalChamber experiments, extracts intervention episodes, and
emits batch dicts for :class:`DoOverTimePFN`.  This module adds the
continuous-time fields (absolute ``times``, ``dts``, ``t_int_start``,
``t_query``, plus the ``int_onset_idx`` expected by
:class:`ContinuousTemporalEncoder`) without re-implementing the
loading logic.

Real timestamps
---------------
CausalChamber ``lt_walks_v1`` exposes a per-row ``timestamp`` column
(seconds, ``float64``), with median inter-row gap ~0.147 s but with
genuine sampling jitter (some gaps reach ~0.25 s).  Earlier versions
of this adapter discarded those values and synthesised a uniform 10 Hz
grid (``np.arange * dt_seconds``), which is exactly the schedule
conflation we critique in the workshop paper.  Phase 12 propagates the
real per-row timestamps from
:meth:`CausalChamberLoader.extract_episodes` through this adapter, so
the encoder sees the same irregular schedule the chamber produced.

Intervention window
-------------------
CausalChamber interventions are actuator set-point changes that occur
(nearly) instantaneously between two samples.  We represent them as
hard interventions on a window
``[t_intervention, t_post[intervention_width_steps - 1]]`` -- the first
``intervention_width_steps`` post-intervention samples -- so the window
length tracks the actual chamber timing rather than a pretended
uniform rate.  ``intervention_width_steps=2`` by default; widen if the
actuator's physical transient spans more samples.

Backward compatibility
----------------------
When an episode dict does not include the per-row timestamp arrays
(e.g.\ tests using synthetic episodes), the adapter falls back to a
uniform grid generated from ``dt_seconds`` and emits a one-time
warning.  Real CausalChamber episodes from the loader always include
the real timestamps; the fallback is for tests only.

Import safety
-------------
Depends on the ``causalchamber`` package (optional).  Import happens at
call time so unit tests can patch or skip when the package is missing.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def _lazy_loader(**kwargs):
    """Import CausalChamberLoader lazily -- keeps import failures local."""
    from dotime.data.causal_chamber import CausalChamberLoader  # noqa: WPS433

    return CausalChamberLoader(**kwargs)


def build_causal_chamber_batch(
    episode: Dict,
    query_var: str,
    *,
    n_max: int,
    dt_seconds: float = 0.1,
    intervention_width_steps: int = 2,
    query_time_offsets: Optional[Sequence[int]] = None,
    normalize_eps: float = 1e-2,
) -> Dict[str, torch.Tensor]:
    """Convert one CausalChamber episode into a continuous-time batch dict.

    Parameters
    ----------
    episode : dict
        Output of
        :meth:`CausalChamberLoader.extract_episodes`.  Must have
        ``X_obs`` (numpy array, shape ``(T_obs, N)``), ``X_post``
        (shape ``(T_post, N)``), ``intervention_var_idx``,
        ``intervention_value``, and ``var_names``.
    query_var : str
        Name of the variable to query.  Must be in
        ``episode['var_names']``.
    n_max : int
        Variable-axis padding (should match the model's ``n_max``).
    dt_seconds : float
        Used **only** as a fallback when the episode dict lacks the
        per-row ``timestamps_obs`` / ``timestamps_post`` arrays (e.g.
        synthetic test episodes).  Real episodes from
        :class:`CausalChamberLoader` always carry the actual chamber
        timestamps and ignore this argument.
    intervention_width_steps : int
        Number of observation steps spanned by the intervention
        window.  Two steps is a short pulse around the setpoint
        change; widen if the actuator's physical transient is long.
    query_time_offsets : sequence of int, optional
        Indices into ``X_post`` to query.  Defaults to every
        post-intervention time step with offset >= 1 (the actuator-set
        step itself, offset 0, is the intervention time and therefore
        skipped).
    normalize_eps : float
        Small constant added to per-variable std to avoid div-by-zero
        when computing z-scores for nearly-constant sensors.

    Returns
    -------
    dict
        Batch dict compatible with :class:`ContinuousDoOverTimePFN`,
        with one entry per queried time offset.  Includes
        ``_subject_id`` / ``_query_time_hours`` style bookkeeping
        fields for downstream reporting.
    """
    X_obs = np.asarray(episode["X_obs"], dtype=np.float32)           # (T_obs, N)
    X_post = np.asarray(episode["X_post"], dtype=np.float32)         # (T_post, N)
    var_names: List[str] = list(episode["var_names"])
    N = len(var_names)
    T_obs = X_obs.shape[0]
    T_post = X_post.shape[0]
    T_total = T_obs + T_post

    if query_var not in var_names:
        raise ValueError(f"{query_var!r} not in episode variables {var_names}")
    if n_max < N:
        raise ValueError(f"n_max must be >= {N} (episode variable count), got {n_max}")

    # Trajectory: [X_obs; X_post] stacked along the time axis.  X_obs is
    # pre-intervention observational; X_post is the interventional
    # trajectory used as the source of ground-truth query values.
    X_full = np.concatenate([X_obs, X_post], axis=0)                 # (T_total, N)

    # Absolute times.  Prefer the per-row timestamps that the loader
    # propagates (see Phase 12).  Fall back to a uniform grid only for
    # synthetic episodes that do not carry timestamps.
    if "timestamps_obs" in episode and "timestamps_post" in episode:
        ts_obs = np.asarray(episode["timestamps_obs"], dtype=np.float32)
        ts_post = np.asarray(episode["timestamps_post"], dtype=np.float32)
        if ts_obs.shape != (T_obs,) or ts_post.shape != (T_post,):
            raise ValueError(
                "episode timestamp arrays have shape "
                f"({ts_obs.shape}, {ts_post.shape}); expected "
                f"(({T_obs},), ({T_post},)) to match X_obs / X_post"
            )
        # Re-anchor so times start at zero, matching the rest of the
        # continuous-time pipeline (the prior generates trajectories
        # starting at t = 0 and the encoder's Fourier embedding is
        # centred on the intervention onset, so absolute offsets do
        # not matter).
        t0 = float(ts_obs[0])
        times_np = np.concatenate([ts_obs - t0, ts_post - t0]).astype(np.float32)
    else:
        warnings.warn(
            "CausalChamber episode lacks per-row timestamps; falling "
            "back to a synthetic uniform grid at dt_seconds. This "
            "ignores the real chamber sampling jitter and only matches "
            "tests with hand-built episode dicts.",
            stacklevel=2,
        )
        times_np = (np.arange(T_total, dtype=np.float32) * dt_seconds)
    dts_np = np.diff(times_np)
    if dts_np.size > 0 and (dts_np <= 0).any():
        raise ValueError(
            "non-positive inter-observation gap in CausalChamber "
            "timestamps -- check the source CSV"
        )

    int_onset_idx = T_obs  # first step of X_post = intervention onset
    t_int_start = float(times_np[int_onset_idx])
    # Window length tracks real time: span the first
    # ``intervention_width_steps`` post-intervention samples so the
    # encoder sees a non-empty interval whose physical length matches
    # the chamber's sampling cadence rather than a pretended 10 Hz.
    width_offset = max(1, min(int(intervention_width_steps), T_post)) - 1
    t_int_end_idx = int_onset_idx + width_offset
    t_int_end = float(times_np[t_int_end_idx])
    if t_int_end <= t_int_start:
        # Edge case: the chosen window collapsed to a single point
        # (e.g. T_post == 1).  Pad by the median observed gap to keep
        # the encoder happy.
        median_gap = float(np.median(dts_np)) if dts_np.size else 1.0
        t_int_end = t_int_start + median_gap

    span = float(times_np[-1] - times_np[0]) or 1.0
    t_int_start_norm = (t_int_start - float(times_np[0])) / span
    t_int_end_norm = (t_int_end - float(times_np[0])) / span

    # Causal masking: zero post-intervention observational context.
    X_obs_masked = np.zeros_like(X_full)
    X_obs_masked[:int_onset_idx, :] = X_full[:int_onset_idx, :]

    # Per-variable normalisation stats from the pre-intervention window.
    pre = X_full[:int_onset_idx, :]
    means = pre.mean(axis=0)                                         # (N,)
    stds = pre.std(axis=0, ddof=0) + normalize_eps                    # (N,)

    # Pad variable axis to n_max.
    X_obs_padded = np.zeros((T_total, n_max), dtype=np.float32)
    X_obs_padded[:, :N] = X_obs_masked
    X_full_padded = np.zeros((T_total, n_max), dtype=np.float32)
    X_full_padded[:, :N] = X_full

    # Normalise and pad.
    X_obs_norm = np.zeros_like(X_obs_padded)
    X_obs_norm[:int_onset_idx, :N] = (pre - means) / stds

    variable_mask = np.zeros(n_max, dtype=np.float32)
    variable_mask[:N] = 1.0

    means_padded = np.zeros(n_max, dtype=np.float32)
    stds_padded = np.ones(n_max, dtype=np.float32)
    means_padded[:N] = means
    stds_padded[:N] = stds

    # Intervention + query specs.
    int_target_idx = int(episode["intervention_var_idx"])
    int_value_raw = float(episode["intervention_value"])
    int_value_norm = (int_value_raw - means[int_target_idx]) / stds[int_target_idx]

    query_idx = var_names.index(query_var)
    if query_time_offsets is None:
        query_time_offsets = list(range(1, T_post))
    q_offsets = np.asarray(list(query_time_offsets), dtype=np.int64)
    if q_offsets.size == 0:
        raise ValueError("no query time offsets supplied and X_post has fewer than 2 steps")
    q_abs_idx = int_onset_idx + q_offsets                            # absolute time indices
    q_times_abs = times_np[q_abs_idx]
    q_times_norm = (q_times_abs - float(times_np[0])) / span

    Y_true = X_post[q_offsets, query_idx].astype(np.float32)
    Y_true_norm = np.clip(
        (Y_true - means[query_idx]) / stds[query_idx], -10.0, 10.0,
    )

    B = q_offsets.shape[0]

    def _expand(x: np.ndarray) -> torch.Tensor:
        """Broadcast a (T, ...) or (N,) numpy array to a (B, ...) torch tensor."""
        t = torch.from_numpy(x)
        return t.unsqueeze(0).expand((B, *t.shape)).contiguous()

    batch: Dict[str, torch.Tensor] = {
        "X_obs": _expand(X_obs_padded),
        "X_obs_norm": _expand(X_obs_norm),
        "X_int": _expand(X_full_padded),
        "variable_mask": _expand(variable_mask),
        "num_vars": torch.full((B,), N, dtype=torch.long),
        "_norm_means": _expand(means_padded),
        "_norm_stds": _expand(stds_padded),
        "times": _expand(times_np),
        "dts": _expand(dts_np),
        "int_onset_idx": torch.full((B,), int_onset_idx, dtype=torch.long),
        "intervention_target": torch.full((B,), int_target_idx, dtype=torch.long),
        "intervention_type": torch.zeros(B, dtype=torch.long),  # HARD
        "intervention_value": torch.full((B,), float(int_value_norm), dtype=torch.float32),
        "intervention_time_start": torch.full((B,), t_int_start_norm, dtype=torch.float32),
        "intervention_time_end": torch.full((B,), t_int_end_norm, dtype=torch.float32),
        "t_int_start": torch.full((B,), t_int_start, dtype=torch.float32),
        "t_int_end": torch.full((B,), t_int_end, dtype=torch.float32),
        "query_target": torch.full((B,), query_idx, dtype=torch.long),
        "query_time": torch.from_numpy(q_times_norm.astype(np.float32)),
        "t_query": torch.from_numpy(q_times_abs.astype(np.float32)),
        "Y_true": torch.from_numpy(Y_true),
        "Y_true_norm": torch.from_numpy(Y_true_norm.astype(np.float32)),
        "_query_time_offset": torch.from_numpy(q_offsets),
    }
    return batch


def iter_causal_chamber_batches(
    episodes: Iterable[Dict],
    query_var: str,
    **kwargs,
) -> Iterable[Dict[str, torch.Tensor]]:
    """Yield one continuous-time batch per episode."""
    for ep in episodes:
        yield build_causal_chamber_batch(ep, query_var=query_var, **kwargs)


def load_chamber_episodes(
    dataset_name: str = "lt_walks_v1",
    root: str = "/tmp/causalchamber",
    subgraph_vars: Optional[Sequence[str]] = None,
    download: bool = True,
    obs_window: int = 50,
    post_window: int = 20,
    max_experiments: Optional[int] = None,
    max_episodes: Optional[int] = None,
) -> List[Dict]:
    """Convenience wrapper: load + extract episodes in one call.

    Parameters mirror :class:`CausalChamberLoader`.  ``max_experiments``
    and ``max_episodes`` cap the amount of data loaded, which is useful
    for smoke tests.
    """
    loader = _lazy_loader(
        dataset_name=dataset_name,
        root=root,
        subgraph_vars=list(subgraph_vars) if subgraph_vars else None,
        download=download,
    )
    experiments = loader.list_experiments()
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    all_episodes: List[Dict] = []
    for name in experiments:
        df = loader.load_experiment(name)
        eps = loader.extract_episodes(df, obs_window=obs_window, post_window=post_window)
        for ep in eps:
            ep.setdefault("experiment", name)
        all_episodes.extend(eps)
        if max_episodes is not None and len(all_episodes) >= max_episodes:
            all_episodes = all_episodes[:max_episodes]
            break
    return all_episodes
