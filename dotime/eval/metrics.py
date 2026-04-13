"""Evaluation metrics for Do-Over-Time-PFN."""

import torch
from typing import Dict, List, Optional, Union
from sklearn.metrics import r2_score


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Root mean squared error."""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

def compute_nmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Normalized mean squared error."""

    mse = torch.mean((predictions - targets) ** 2)
    target_range = torch.max(targets) - torch.min(targets)
    if target_range == 0:
        return mse.item()
    nmse = mse / (target_range ** 2)
    
    return nmse.item()

def compute_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """R² score."""
    return r2_score(targets.cpu().numpy(), predictions.cpu().numpy())


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error."""
    return torch.mean(torch.abs(predictions - targets)).item()


@torch.no_grad()
def compute_quantile_calibration(
    all_logits: torch.Tensor,
    borders: torch.Tensor,
    all_targets: torch.Tensor,
    tau_levels: Union[torch.Tensor, List[float]],
) -> Dict[str, float]:
    """Compute quantile calibration: fraction of targets below each predicted quantile.

    Perfect calibration means the fraction equals tau for each quantile level.

    Parameters
    ----------
    all_logits : (N, K) logits over K buckets
    borders : (K+1,) bucket boundaries
    all_targets : (N,) true values
    tau_levels : quantile levels to evaluate

    Returns
    -------
    dict mapping f"coverage_tau{tau}" to observed coverage fraction,
    plus "calibration_error" (mean absolute difference from ideal).
    """
    from dotime.model.pinball_loss import extract_quantiles

    if not isinstance(tau_levels, torch.Tensor):
        tau_levels = torch.tensor(tau_levels, device=all_logits.device, dtype=all_logits.dtype)

    quantiles = extract_quantiles(all_logits, borders, tau_levels)  # (N, Q)
    targets = all_targets.unsqueeze(-1)  # (N, 1)
    below = (targets <= quantiles).float()  # (N, Q)
    coverage = below.mean(dim=0)  # (Q,)

    results = {}
    cal_errors = []
    for i, tau in enumerate(tau_levels.tolist()):
        cov = coverage[i].item()
        results[f"coverage_tau{tau:.2f}"] = cov
        cal_errors.append(abs(cov - tau))
    results["calibration_error"] = sum(cal_errors) / len(cal_errors)
    return results


@torch.no_grad()
def compute_pinball_metric(
    all_logits: torch.Tensor,
    borders: torch.Tensor,
    all_targets: torch.Tensor,
    tau_levels: Union[torch.Tensor, List[float]],
) -> float:
    """Compute pinball loss as an evaluation metric.

    Parameters
    ----------
    all_logits : (N, K) logits
    borders : (K+1,) bucket boundaries
    all_targets : (N,) true values
    tau_levels : quantile levels

    Returns
    -------
    mean pinball loss (scalar)
    """
    from dotime.model.pinball_loss import extract_quantiles, pinball_loss

    if not isinstance(tau_levels, torch.Tensor):
        tau_levels = torch.tensor(tau_levels, device=all_logits.device, dtype=all_logits.dtype)

    quantile_preds = extract_quantiles(all_logits, borders, tau_levels)
    return pinball_loss(quantile_preds, all_targets, tau_levels).item()


@torch.no_grad()
def compute_crps_from_quantiles(
    quantile_preds: torch.Tensor,
    targets: torch.Tensor,
    tau_levels: Union[torch.Tensor, List[float]],
) -> float:
    """Approximate CRPS from quantile predictions via numerical integration.

    CRPS = ∫₀¹ 2·L_τ(y, q_τ) dτ ≈ Σᵢ 2·L_τᵢ(y, q_τᵢ)·Δτᵢ

    Parameters
    ----------
    quantile_preds : (N, Q) predicted quantiles
    targets : (N,) true values
    tau_levels : (Q,) quantile levels

    Returns
    -------
    mean CRPS (scalar)
    """
    if not isinstance(tau_levels, torch.Tensor):
        tau_levels = torch.tensor(tau_levels, dtype=quantile_preds.dtype)

    # Compute pinball loss per quantile
    error = targets.unsqueeze(-1) - quantile_preds  # (N, Q)
    tau = tau_levels.unsqueeze(0)  # (1, Q)
    pinball = torch.where(error >= 0, tau * error, (tau - 1.0) * error)  # (N, Q)

    # Numerical integration weights (trapezoid rule over [0, 1])
    tau_sorted = tau_levels.sort().values
    # Widths: half-intervals at boundaries, full intervals in between
    widths = torch.zeros_like(tau_sorted)
    widths[0] = (tau_sorted[0] + tau_sorted[1]) / 2
    widths[-1] = 1.0 - (tau_sorted[-2] + tau_sorted[-1]) / 2
    for i in range(1, len(tau_sorted) - 1):
        widths[i] = (tau_sorted[i + 1] - tau_sorted[i - 1]) / 2

    # CRPS = 2 * weighted sum of pinball losses
    crps_per_sample = 2.0 * (pinball * widths.unsqueeze(0)).sum(dim=-1)  # (N,)
    return crps_per_sample.mean().item()


@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu", tau_levels=None) -> Dict[str, float]:
    """Evaluate model on a dataloader, returning metrics dict.

    Returns
    -------
    dict with keys: loss, rmse, mae
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        logits = model(batch)
        loss = model.bar_head.loss(logits, batch['Y_true_norm'])
        total_loss += loss.item()

        preds = model.bar_head.predict_mean(logits)
        all_preds.append(preds.cpu())
        all_targets.append(batch['Y_true_norm'].cpu())
        if tau_levels is not None:
            all_logits.append(logits.cpu())
        n_batches += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    results = {
        'loss': total_loss / max(n_batches, 1),
        'rmse': compute_rmse(all_preds, all_targets),
        'mae': compute_mae(all_preds, all_targets),
        'nmse': compute_nmse(all_preds, all_targets),
        'r2': compute_r2(all_preds, all_targets),
    }

    if tau_levels is not None and all_logits:
        logits_cat = torch.cat(all_logits)
        borders = model.bar_head.borders.cpu()
        results['pinball_loss'] = compute_pinball_metric(
            logits_cat, borders, all_targets, tau_levels)
        cal = compute_quantile_calibration(
            logits_cat, borders, all_targets, tau_levels)
        results.update(cal)

    return results
