"""Adapter from Warfarin subject records to model-ready batches.

Maps the three-variable Warfarin PK/PD dataset onto the
:class:`ContinuousDoOverTimePFN` input contract.  Compared to the
two-variable :mod:`theophylline_adapter`, this adapter exercises the
model's mediator reasoning: the model intervenes on ``Dose`` and is
queried on ``concentration`` (the PK mediator) and ``PCA`` (the PD
outcome) at a variety of irregular observation times.

Variable layout (canonical order, shared with the TSCM ``front_door``
structure so a model trained on the synthetic front-door prior can be
evaluated zero-shot on Warfarin):

- **Canonical 0 = Dose** (treatment A).
- **Canonical 1 = concentration cp** (mediator M).
- **Canonical 2 = PCA** (outcome Y, pharmacodynamic response).

Per subject we build a single trajectory on the union time grid of
all cp and PCA observations for that subject.  Each measured
``(variable, time)`` pair becomes one entry in the returned batch, so
the caller gets ``n_cp + n_pca`` queries per subject.

Normalisation follows the same ``peek_target`` strategy as
:mod:`theophylline_adapter`: per-variable mean / std are taken from
the subject's own observation window so the model's training-scale
z-score inputs stay in range.  This is the same mild zero-shot scale
calibration used for Theophylline.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from .warfarin import WarfarinSubject


_DOSE_IDX = 0
_CP_IDX = 1
_PCA_IDX = 2
_N_VARS = 3


def _union_sorted_times(
    a: torch.Tensor, b: torch.Tensor, include_zero: bool = True,
) -> torch.Tensor:
    """Return the strictly-increasing union of two 1-D time tensors."""
    pieces = [a, b]
    if include_zero:
        pieces.append(torch.tensor([0.0], dtype=torch.float32))
    return torch.unique(torch.cat(pieces))  # returns sorted + unique


def build_warfarin_batch(
    subject: WarfarinSubject,
    n_max: int = 16,
    absorption_window_hours: float = 1.0,
    dose_scale: float = 0.01,
    conc_std_floor: float = 0.5,
    pca_std_floor: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Build a zero-shot CT batch dict for one Warfarin subject.

    Parameters
    ----------
    subject : WarfarinSubject
        Subject to convert.
    n_max : int
        Variable-axis padding, must match the model's ``n_max``
        (and must be ``>= 3``).
    absorption_window_hours : float
        Length of the hard-intervention window on ``Dose``.  Warfarin's
        absorption half-life is roughly 1 hour.
    dose_scale : float
        Multiplier applied to the raw dose (mg) before it enters the
        batch as ``intervention_value``.  The default ``0.01`` puts a
        100 mg dose at normalised value 1.0, keeping the intervention
        inside the synthetic prior's typical range.
    conc_std_floor, pca_std_floor : float
        Lower bounds on the per-variable normalisation std, to keep
        zero-shot predictions bounded when a subject's observation
        window is unusually flat.

    Returns
    -------
    dict
        Batch dict with ``B = subject.n_cp + subject.n_pca`` entries.
        Fields follow the :class:`ContinuousDoOverTimePFN` contract and
        include ``_query_variable_name``, ``_subject_id``, and
        ``_query_time_hours`` for downstream bookkeeping.
    """
    if n_max < _N_VARS:
        raise ValueError(f"n_max must be >= {_N_VARS}, got {n_max}")
    if subject.n_cp == 0 and subject.n_pca == 0:
        raise ValueError(f"subject {subject.subject_id} has no observations")

    # 1. Union time grid for this subject (always includes 0.0).
    times = _union_sorted_times(subject.cp_times, subject.pca_times)
    T = times.numel()

    # 2. Map observations into the union grid.  X_int holds the true
    #    trajectories; X_obs is zero everywhere (no pre-dose context).
    def _map_to_grid(obs_times: torch.Tensor, obs_values: torch.Tensor) -> tuple:
        """Return (grid_indices, values) for each observation."""
        if obs_times.numel() == 0:
            return (torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.float32))
        # Exact match via torch.searchsorted + equality check.
        idx = torch.searchsorted(times, obs_times)
        return idx.long(), obs_values

    cp_idx, cp_vals = _map_to_grid(subject.cp_times, subject.cp_values)
    pca_idx, pca_vals = _map_to_grid(subject.pca_times, subject.pca_values)

    X_full = torch.zeros(T, _N_VARS, dtype=torch.float32)
    if cp_idx.numel() > 0:
        X_full[cp_idx, _CP_IDX] = cp_vals
    if pca_idx.numel() > 0:
        X_full[pca_idx, _PCA_IDX] = pca_vals

    X_obs = torch.zeros_like(X_full)  # no pre-dose observational window

    # 3. Per-variable normalisation stats.
    dose_raw = float(subject.dose_mg) * dose_scale
    if subject.n_cp > 1 and float(subject.cp_values.std(unbiased=True).item()) > 0:
        cp_mean = float(subject.cp_values.mean().item())
        cp_std = max(float(subject.cp_values.std(unbiased=True).item()), conc_std_floor)
    else:
        cp_mean, cp_std = 0.0, conc_std_floor
    if subject.n_pca > 1 and float(subject.pca_values.std(unbiased=True).item()) > 0:
        pca_mean = float(subject.pca_values.mean().item())
        pca_std = max(float(subject.pca_values.std(unbiased=True).item()), pca_std_floor)
    else:
        pca_mean, pca_std = 0.0, pca_std_floor

    means_padded = torch.zeros(n_max, dtype=torch.float32)
    stds_padded = torch.ones(n_max, dtype=torch.float32)
    means_padded[_DOSE_IDX] = 0.0
    stds_padded[_DOSE_IDX] = 1.0
    means_padded[_CP_IDX] = cp_mean
    stds_padded[_CP_IDX] = cp_std
    means_padded[_PCA_IDX] = pca_mean
    stds_padded[_PCA_IDX] = pca_std

    # 4. Pad the variable axis to n_max.
    def _pad(X: torch.Tensor) -> torch.Tensor:
        if n_max == _N_VARS:
            return X
        pad = torch.zeros(X.shape[0], n_max - _N_VARS, dtype=X.dtype)
        return torch.cat([X, pad], dim=1)

    X_obs_padded = _pad(X_obs)
    X_int_padded = _pad(X_full)
    X_obs_norm_padded = torch.zeros_like(X_obs_padded)

    variable_mask = torch.zeros(n_max, dtype=torch.float32)
    variable_mask[:_N_VARS] = 1.0

    # 5. Intervention spec.
    intervention_value_norm = dose_raw  # dose_scale already applied; std=1
    t_int_start = 0.0
    t_int_end = max(
        float(absorption_window_hours),
        float(times[1].item() - times[0].item()) if T >= 2 else 1.0,
    )
    span = float(times[-1].item()) or 1.0
    t_int_start_norm = 0.0
    t_int_end_norm = (t_int_end - float(times[0].item())) / span

    # 6. Per-observation queries (one batch entry per real observation).
    query_grid_idx: List[int] = []
    query_target_canon: List[int] = []
    query_var_name: List[str] = []
    for g, v in zip(cp_idx.tolist(), cp_vals.tolist()):
        query_grid_idx.append(int(g))
        query_target_canon.append(_CP_IDX)
        query_var_name.append("cp")
    for g, v in zip(pca_idx.tolist(), pca_vals.tolist()):
        query_grid_idx.append(int(g))
        query_target_canon.append(_PCA_IDX)
        query_var_name.append("pca")

    B = len(query_grid_idx)
    q_times_abs = times[torch.tensor(query_grid_idx, dtype=torch.long)]
    q_times_norm = (q_times_abs - float(times[0].item())) / max(span, 1e-6)
    q_target_t = torch.tensor(query_target_canon, dtype=torch.long)

    # Y_true denormalised (model predicts normalised, evaluator denormalises back)
    Y_true = torch.tensor(
        [X_full[g, c].item() for g, c in zip(query_grid_idx, query_target_canon)],
        dtype=torch.float32,
    )
    q_stds = stds_padded[q_target_t]
    q_means = means_padded[q_target_t]
    Y_true_norm = torch.clamp((Y_true - q_means) / q_stds, -10.0, 10.0)

    batch: Dict[str, torch.Tensor] = {
        "X_obs": X_obs_padded.unsqueeze(0).expand(B, T, n_max).contiguous(),
        "X_obs_norm": X_obs_norm_padded.unsqueeze(0).expand(B, T, n_max).contiguous(),
        "X_int": X_int_padded.unsqueeze(0).expand(B, T, n_max).contiguous(),
        "variable_mask": variable_mask.unsqueeze(0).expand(B, n_max).contiguous(),
        "num_vars": torch.full((B,), _N_VARS, dtype=torch.long),
        "_norm_means": means_padded.unsqueeze(0).expand(B, n_max).contiguous(),
        "_norm_stds": stds_padded.unsqueeze(0).expand(B, n_max).contiguous(),
        "times": times.unsqueeze(0).expand(B, T).contiguous(),
        "dts": times.diff().unsqueeze(0).expand(B, T - 1).contiguous(),
        "int_onset_idx": torch.zeros(B, dtype=torch.long),
        "intervention_target": torch.full((B,), _DOSE_IDX, dtype=torch.long),
        "intervention_type": torch.zeros(B, dtype=torch.long),  # HARD
        "intervention_value": torch.full((B,), intervention_value_norm, dtype=torch.float32),
        "intervention_time_start": torch.full((B,), t_int_start_norm, dtype=torch.float32),
        "intervention_time_end": torch.full((B,), t_int_end_norm, dtype=torch.float32),
        "t_int_start": torch.full((B,), t_int_start, dtype=torch.float32),
        "t_int_end": torch.full((B,), t_int_end, dtype=torch.float32),
        "query_target": q_target_t,
        "query_time": q_times_norm.float(),
        "t_query": q_times_abs.float(),
        "Y_true": Y_true,
        "Y_true_norm": Y_true_norm.float(),
        "_subject_id": torch.full((B,), subject.subject_id, dtype=torch.long),
        "_query_time_hours": q_times_abs.float(),
        "_query_variable_name": query_var_name,  # Python list, not a tensor
    }
    return batch


def iter_warfarin_batches(
    subjects: Iterable[WarfarinSubject],
    **kwargs,
) -> Iterable[Dict[str, torch.Tensor]]:
    """Yield one batch dict per subject."""
    for s in subjects:
        yield build_warfarin_batch(s, **kwargs)
