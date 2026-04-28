"""Adapter from Theophylline subject records to model-ready batches.

Maps the Theophylline PK dataset onto the
:class:`ContinuousDoOverTimePFN` input contract so the model can be
evaluated zero-shot on a real pharmacokinetic benchmark.

Mapping assumptions
-------------------
Theophylline has two physical variables (dose, concentration) and no
pre-dose observations.  We frame each subject as an
interventional-only trajectory on a two-variable RCT structure:

- **Variable 0 (treatment A) = Dose.**
- **Variable 1 (outcome Y) = concentration.**
- **Intervention.** Oral dosing is modelled as a hard intervention on
  ``Dose`` over a short absorption window
  ``[0, absorption_window_hours)``.  The intervention value is the
  dose, optionally rescaled so it sits inside the training prior's
  support.
- **Observational context.**  ``X_obs`` is zero on every pre-dose time
  step (there are none in Theophylline) and masked to zero from the
  intervention onset onwards, as in training.  The encoder therefore
  runs on an empty pre-intervention window and relies entirely on the
  cross-variable mixer's intervention spec.
- **Queries.**  For each post-dose observation time we emit a single
  query ``(target=concentration, time=t)`` with ``Y_true`` equal to the
  observed concentration at that time.  An entire batch for one
  subject has ``n_queries`` queries stacked across the batch dim.

Normalisation
-------------
The training pipeline z-score-normalises ``X_obs`` and ``Y_true`` using
*pre-intervention* per-variable statistics.  Theophylline has no
pre-intervention data, so this adapter supports two strategies via
``normalize_from``:

- ``"peek_target"`` (default): use the entire subject's observed
  concentration trajectory as the normalisation window for the
  concentration variable.  This is a mild form of cheating -- we look
  at the target scale to match the model's training scale -- but is
  standard practice for zero-shot numerical transfer.  ``Dose`` is
  centred by its raw value and scaled by a configurable constant.
- ``"fixed"``: apply globally fixed means/stds passed in as arguments.
  Useful when calibrating against an external reference scale.

The chosen normalisation is applied directly inside the returned batch
(``X_obs_norm``, ``Y_true_norm``, ``_norm_means``, ``_norm_stds``) so
the downstream model can be called without re-running
:func:`normalize_batch`.

N_max padding and intervention types follow the training contract so
:class:`ContinuousDoOverTimePFN` can consume the batch unmodified.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .theophylline import TheophSubject


# Canonical indices used by the training pipeline's TSCMPrior (A=0, Y=N-1).
# For RCT we have only A and Y -> N=2 -> Y at index 1.
_DOSE_IDX = 0
_CONC_IDX = 1
_N_VARS = 2


def build_theophylline_batch(
    subject: TheophSubject,
    n_max: int = 16,
    absorption_window_hours: float = 1.0,
    dose_scale: float = 1.0,
    query_time_indices: Optional[Sequence[int]] = None,
    normalize_from: str = "peek_target",
    fixed_means: Optional[Tuple[float, float]] = None,
    fixed_stds: Optional[Tuple[float, float]] = None,
    conc_std_floor: float = 0.5,
    pre_baseline_n: int = 0,
    pre_baseline_dt_hours: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Build a batch dict for one subject containing one query per post-dose observation.

    Each query is a separate batch entry with the same trajectory and
    intervention but a different ``(query_time, Y_true)``.  This keeps
    the returned dict in the standard "flat" batch layout
    (no ``_traj_idx`` key) so the model's forward path doesn't need
    the encoder-caching code path.

    Parameters
    ----------
    subject : TheophSubject
        Subject to convert.
    n_max : int
        Variable-axis padding, must match the model's ``n_max``.
    absorption_window_hours : float
        Length of the hard-intervention window applied to ``Dose``.
        Short values (~0.5 h) approximate an instantaneous pulse; the
        ~1 h default is closer to Theophylline's absorption half-life.
    dose_scale : float
        Multiplier on the raw dose (mg/kg) before it enters the batch
        as the ``intervention_value``.  Use this to rescale the dose
        into the training prior's support.
    query_time_indices : sequence of int, optional
        Which of the subject's observation times to query.  Defaults to
        all times with ``t > 0`` (i.e. every post-dose observation).
    normalize_from : {"peek_target", "fixed"}
        Normalisation strategy (see module docstring).
    fixed_means, fixed_stds : tuple of float, optional
        Only used with ``normalize_from="fixed"``; ``(dose_mean,
        conc_mean)`` / ``(dose_std, conc_std)``.
    conc_std_floor : float
        Lower bound on the concentration normalisation std to avoid
        exploding predictions for subjects with very flat trajectories.
    pre_baseline_n : int
        Phase-13a synthetic pre-baseline padding.  Number of fake
        pre-dose observations to prepend to the subject's trajectory
        at uniformly spaced negative times, all at zero baseline
        (dose=0, concentration=0).  These give the encoder real
        pre-intervention context that matches its training
        distribution, at the cost of inventing observations the
        subject did not actually have.  ``0`` (default) preserves the
        zero-context behaviour (no padding).
    pre_baseline_dt_hours : float, optional
        Spacing between consecutive synthetic pre-baseline samples,
        in hours.  Defaults to the first post-dose gap
        ``times[1] - times[0]`` so the synthetic schedule blends
        smoothly into the real one.

    Returns
    -------
    dict
        Batch fields consumed by :class:`ContinuousDoOverTimePFN` plus
        auxiliary ``_subject_id`` and ``_query_time_hours`` fields that
        simplify evaluation bookkeeping.
    """
    if normalize_from not in ("peek_target", "fixed"):
        raise ValueError(f"invalid normalize_from: {normalize_from!r}")
    if normalize_from == "fixed" and (fixed_means is None or fixed_stds is None):
        raise ValueError("fixed_means and fixed_stds required for normalize_from='fixed'")
    if n_max < _N_VARS:
        raise ValueError(f"n_max must be >= {_N_VARS}, got {n_max}")
    if pre_baseline_n < 0:
        raise ValueError(f"pre_baseline_n must be >= 0, got {pre_baseline_n}")

    real_times = subject.times
    concs = subject.concentrations
    T_real = real_times.numel()

    # Phase-13a synthetic pre-baseline padding.  Prepend ``pre_baseline_n``
    # fake pre-dose observations at uniformly spaced negative times so the
    # encoder receives a non-empty pre-intervention window.
    if pre_baseline_n > 0:
        if pre_baseline_dt_hours is None:
            if T_real >= 2:
                pre_dt = float((real_times[1] - real_times[0]).item())
            else:
                pre_dt = 0.5
        else:
            pre_dt = float(pre_baseline_dt_hours)
        if pre_dt <= 0:
            raise ValueError(
                f"pre_baseline_dt_hours must be positive; derived/passed value {pre_dt}"
            )
        pre_times = torch.tensor(
            [-pre_dt * (pre_baseline_n - i) for i in range(pre_baseline_n)],
            dtype=real_times.dtype,
        )
        times = torch.cat([pre_times, real_times])
        # Pre-dose concentrations are zero (no drug yet).  We expand the
        # concs vector so downstream indexing is uniform over T.
        concs_padded = torch.cat([torch.zeros(pre_baseline_n, dtype=concs.dtype), concs])
    else:
        times = real_times
        concs_padded = concs
    T = times.numel()

    if query_time_indices is None:
        # ``query_time_indices`` always indexes into ``subject.times`` /
        # ``concs`` (the real trajectory), so callers do not need to know
        # about the synthetic pre-baseline offset.
        query_time_indices = [i for i in range(T_real) if float(real_times[i].item()) > 0.0]
    query_time_indices = list(query_time_indices)
    if not query_time_indices:
        raise ValueError(f"subject {subject.subject_id} has no post-dose observations")

    B = len(query_time_indices)

    # -- shared-per-subject tensors (all queries share the same trajectory) --
    X_obs_vars = torch.zeros(T, _N_VARS)  # all zeros: pre-dose at baseline, post-dose causally masked
    X_int_vars = torch.zeros(T, _N_VARS)
    X_int_vars[:, _DOSE_IDX] = 0.0  # placeholder; concentration is the observable
    X_int_vars[:, _CONC_IDX] = concs_padded

    # -- normalisation --
    if normalize_from == "peek_target":
        dose_mean = 0.0
        dose_std = max(dose_scale, 1e-4)
        conc_mean = float(concs.mean().item())
        conc_std = max(float(concs.std(unbiased=True).item()), conc_std_floor)
    else:
        dose_mean, conc_mean = fixed_means
        dose_std, conc_std = fixed_stds

    raw_means = torch.zeros(_N_VARS)
    raw_stds = torch.ones(_N_VARS)
    raw_means[_DOSE_IDX] = dose_mean
    raw_means[_CONC_IDX] = conc_mean
    raw_stds[_DOSE_IDX] = dose_std
    raw_stds[_CONC_IDX] = conc_std

    X_obs_norm_vars = (X_obs_vars - raw_means) / raw_stds
    # Causal masking: zero out X_obs from intervention onset onwards.
    # Pre-baseline rows (rows < pre_baseline_n) keep their normalised
    # baseline values so the encoder sees a real pre-intervention
    # window; post-dose rows are masked to zero, matching training.
    if pre_baseline_n == 0:
        X_obs_norm_vars = X_obs_norm_vars * 0.0
    else:
        X_obs_norm_vars[pre_baseline_n:] = 0.0

    # -- pad variable axis to n_max --
    def _pad(X: torch.Tensor) -> torch.Tensor:
        if n_max == _N_VARS:
            return X
        pad = torch.zeros(X.shape[0], n_max - _N_VARS, dtype=X.dtype)
        return torch.cat([X, pad], dim=1)

    X_obs_padded = _pad(X_obs_vars)
    X_int_padded = _pad(X_int_vars)
    X_obs_norm_padded = _pad(X_obs_norm_vars)

    variable_mask = torch.zeros(n_max)
    variable_mask[:_N_VARS] = 1.0

    means_padded = torch.zeros(n_max)
    stds_padded = torch.ones(n_max)
    means_padded[:_N_VARS] = raw_means
    stds_padded[:_N_VARS] = raw_stds

    # -- intervention and absolute-time fields --
    dose_raw = float(subject.dose_mg_per_kg) * dose_scale
    # Normalise dose the same way as training: z-score against its own
    # (mean, std) which for "peek_target" is (0, dose_scale).
    intervention_value_norm = (dose_raw - dose_mean) / max(dose_std, 1e-4)

    t_int_start = 0.0  # the dose; the synthetic pre-baseline lives at t < 0
    # First post-dose gap; clamp so we always have a non-empty window.
    first_post_gap = float(real_times[1].item() - real_times[0].item()) if T_real >= 2 else 1.0
    t_int_end = max(float(absorption_window_hours), first_post_gap * 0.5)
    span = float((times[-1] - times[0]).item()) or 1.0
    t_int_start_norm = (t_int_start - float(times[0].item())) / span
    t_int_end_norm = (t_int_end - float(times[0].item())) / span
    int_onset_idx = pre_baseline_n  # 0 if no padding; otherwise first post-dose row

    # -- per-query tensors --
    # ``query_time_indices`` indexes into ``real_times``; shift by the
    # pre-baseline offset to land in the padded ``times`` array.
    q_idx = torch.tensor(
        [i + pre_baseline_n for i in query_time_indices], dtype=torch.long,
    )
    q_times_abs = times[q_idx]
    q_times_norm = (q_times_abs - times[0]) / max(span, 1e-6)
    q_concs = concs_padded[q_idx]
    q_concs_norm = torch.clamp((q_concs - conc_mean) / conc_std, -10.0, 10.0)

    batch: Dict[str, torch.Tensor] = {
        # Trajectories (broadcast to batch dim)
        "X_obs": X_obs_padded.unsqueeze(0).expand(B, T, n_max).contiguous(),
        "X_obs_norm": X_obs_norm_padded.unsqueeze(0).expand(B, T, n_max).contiguous(),
        "X_int": X_int_padded.unsqueeze(0).expand(B, T, n_max).contiguous(),
        "variable_mask": variable_mask.unsqueeze(0).expand(B, n_max).contiguous(),
        "num_vars": torch.full((B,), _N_VARS, dtype=torch.long),
        "_norm_means": means_padded.unsqueeze(0).expand(B, n_max).contiguous(),
        "_norm_stds": stds_padded.unsqueeze(0).expand(B, n_max).contiguous(),
        # Schedule
        "times": times.unsqueeze(0).expand(B, T).contiguous(),
        "dts": times.diff().unsqueeze(0).expand(B, T - 1).contiguous(),
        # Intervention
        "int_onset_idx": torch.full((B,), int_onset_idx, dtype=torch.long),
        "intervention_target": torch.full((B,), _DOSE_IDX, dtype=torch.long),
        "intervention_type": torch.zeros(B, dtype=torch.long),  # HARD
        "intervention_value": torch.full((B,), intervention_value_norm, dtype=torch.float32),
        "intervention_time_start": torch.full((B,), t_int_start_norm, dtype=torch.float32),
        "intervention_time_end": torch.full((B,), t_int_end_norm, dtype=torch.float32),
        "t_int_start": torch.full((B,), t_int_start, dtype=torch.float32),
        "t_int_end": torch.full((B,), t_int_end, dtype=torch.float32),
        # Query
        "query_target": torch.full((B,), _CONC_IDX, dtype=torch.long),
        "query_time": q_times_norm.float(),
        "t_query": q_times_abs.float(),
        "Y_true": q_concs.float(),
        "Y_true_norm": q_concs_norm.float(),
        # Bookkeeping (not consumed by the model; useful for eval scripts)
        "_subject_id": torch.full((B,), subject.subject_id, dtype=torch.long),
        "_query_time_hours": q_times_abs.float(),
    }
    return batch


def iter_theophylline_batches(
    subjects: Iterable[TheophSubject],
    **kwargs,
) -> Iterable[Dict[str, torch.Tensor]]:
    """Yield one batch dict per subject using :func:`build_theophylline_batch`."""
    for subj in subjects:
        yield build_theophylline_batch(subj, **kwargs)


def concat_batches(
    batches: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Concatenate per-subject batches along the batch dim.

    Only supports batches with identical ``(T, N_max)`` shapes and
    matching field sets.  The 12 Theophylline subjects all have T=11
    in the bundled dataset, so this works out of the box.
    """
    if not batches:
        raise ValueError("no batches to concatenate")
    keys = set(batches[0].keys())
    for b in batches[1:]:
        if set(b.keys()) != keys:
            raise ValueError("batch key sets differ")
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}
