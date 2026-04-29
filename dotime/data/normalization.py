"""Per-variable normalization for heterogeneous-scale time series."""

import torch
from typing import Dict, Tuple


# Outer bound on |Y_true_norm| (in units of pre-window std).  Raised
# from the original ±10 to ±50 after the Phase 13b shakedown showed
# that even with stable Euler--Maruyama, the prior occasionally
# produces samples > 10 sigma from their pre-window mean -- usually
# because a hard intervention lands ~5 sigma away and the chain of
# parental weights with std ~0.5 amplifies it.  At ±10 the clip was
# pinning ~30-50% of training batches at the ceiling and corrupting
# the regression target distribution.  ±50 is loose enough that
# legitimate samples never hit it; the trainer still skips truly
# pathological steps via its own ``y_max > 100`` guard.
Y_NORM_CLIP = 50.0


def per_variable_normalize(
    X_obs: torch.Tensor,
    variable_mask: torch.Tensor,
    eps: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each variable independently using its observational statistics.

    Parameters
    ----------
    X_obs : (B, T, N_max) observational time series
    variable_mask : (B, N_max) binary mask for real variables
    eps : numerical stability

    Returns
    -------
    X_norm : (B, T, N_max) normalized series
    means : (B, N_max) per-variable means
    stds : (B, N_max) per-variable stds
    """
    # Compute per-variable mean and std over pre-intervention timesteps only.
    # X_obs is zero-masked after int_onset, so we exclude zeros to avoid
    # diluting statistics with the post-intervention mask.
    # X_obs: (B, T, N_max)
    nonzero_mask = (X_obs != 0).float()  # (B, T, N_max)
    n_valid = nonzero_mask.sum(dim=1).clamp(min=1)  # (B, N_max)
    means = (X_obs * nonzero_mask).sum(dim=1) / n_valid  # (B, N_max)
    sq_diff = ((X_obs - means.unsqueeze(1)) * nonzero_mask) ** 2
    # Bessel's correction: divide by (n-1) to match torch.std default
    stds = (sq_diff.sum(dim=1) / (n_valid - 1).clamp(min=1)).sqrt() + eps  # (B, N_max)

    # Zero out stats for padded variables
    mask = variable_mask                 # (B, N_max)
    means = means * mask
    stds = stds * mask + (1 - mask)     # padded vars get std=1 to avoid div-by-zero

    # Normalize
    X_norm = (X_obs - means.unsqueeze(1)) / stds.unsqueeze(1)

    # Zero out padded variables
    X_norm = X_norm * mask.unsqueeze(1)

    return X_norm, means, stds


def normalize_target(
    Y_true: torch.Tensor,
    query_target: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    """Normalize target values using the same per-variable statistics.

    Parameters
    ----------
    Y_true : (B,) target values
    query_target : (B,) variable indices
    means : (B, N_max)
    stds : (B, N_max)

    Returns
    -------
    Y_norm : (B,) normalized targets
    """
    batch_idx = torch.arange(Y_true.shape[0], device=Y_true.device)
    target_mean = means[batch_idx, query_target]
    target_std = stds[batch_idx, query_target]
    y_norm = (Y_true - target_mean) / target_std
    return torch.clamp(y_norm, -Y_NORM_CLIP, Y_NORM_CLIP)


def normalize_batch(
    batch: Dict[str, torch.Tensor],
    target_key: str = "Y_true",
) -> Dict[str, torch.Tensor]:
    """Normalize a full batch in-place, adding normalized fields.

    Handles both flat batches (B trajectories, B queries) and multi-query
    batches where _traj_idx maps B_total queries to B unique trajectories.

    Adds: X_obs_norm, Y_true_norm, _norm_means, _norm_stds
    """
    X_norm, means, stds = per_variable_normalize(
        batch['X_obs'], batch['variable_mask']
    )
    batch['X_obs_norm'] = X_norm
    batch['_norm_means'] = means
    batch['_norm_stds'] = stds

    # For multi-query batches, expand means/stds to query dimension via _traj_idx
    if '_traj_idx' in batch:
        traj_idx = batch['_traj_idx']
        q_means = means[traj_idx]   # (B_total, N_max)
        q_stds = stds[traj_idx]     # (B_total, N_max)
    else:
        q_means = means
        q_stds = stds

    query_target = batch['query_target']
    q_idx = torch.arange(query_target.shape[0], device=query_target.device)

    if target_key == "Y_causal_effect" and "Y_causal_effect" in batch:
        target_std = q_stds[q_idx, query_target]
        Y_norm = torch.clamp(
            batch['Y_causal_effect'] / target_std, -Y_NORM_CLIP, Y_NORM_CLIP,
        )
    else:
        target_mean = q_means[q_idx, query_target]
        target_std = q_stds[q_idx, query_target]
        Y_norm = torch.clamp(
            (batch[target_key] - target_mean) / target_std,
            -Y_NORM_CLIP, Y_NORM_CLIP,
        )

    batch['Y_true_norm'] = Y_norm
    return batch
