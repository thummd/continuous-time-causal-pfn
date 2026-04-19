#!/usr/bin/env python
"""Evaluate s5 checkpoints on BD and FD structures with detailed metrics.

Runs each of the 4 s5 checkpoints against its trained structure using the
same data generation as training (no lag, interpolation mask) and reports
RMSE, MAE, nMSE, R2, direction accuracy, and causal effect metrics.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotime.model.do_over_time_pfn import DoOverTimePFN
from dotime.data.temporal_dataloader import TemporalInterventionDataLoader
from dotime.data.normalization import normalize_batch


def evaluate_checkpoint(ckpt_path, structure, n_batches=50, batch_size=16,
                        observational_only=False, device='cuda:0'):
    """Evaluate one checkpoint on the given TSCM structure."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DoOverTimePFN(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    loader = TemporalInterventionDataLoader(
        num_steps=n_batches, batch_size=batch_size,
        n_max=12, n_max_prior=10,
        t_range=(50, 200), burn_in=50,
        seed=12345, device=device,
        target_key='Y_true',
        n_queries=100, query_mode='all_pairs',
        tscm_structure=structure,
        use_lagged_edges=False,
        intervention_scale=4.0,
        causal_mask_mode='interpolation',
    )

    all_preds, all_targets, all_effects, all_pred_effects = [], [], [], []
    all_positivity = []

    with torch.no_grad():
        for batch in loader:
            if observational_only:
                batch['intervention_target'] = torch.zeros_like(batch['intervention_target'])
                batch['intervention_type'] = torch.zeros_like(batch['intervention_type'])
                batch['intervention_value'] = torch.zeros_like(batch['intervention_value'])
                batch['intervention_time_start'] = torch.zeros_like(batch['intervention_time_start'])
                batch['intervention_time_end'] = torch.zeros_like(batch['intervention_time_end'])

            # Handle multi-query caching
            if '_traj_idx' in batch:
                h_vars = model.encode(batch)
                h_vars_exp = h_vars[batch['_traj_idx']]
                qb = {k: v for k, v in batch.items() if k not in ('X_obs_norm', '_traj_idx')}
                qb['variable_mask'] = batch['variable_mask'][batch['_traj_idx']]
                output = model.query(h_vars_exp, qb)
            else:
                output = model(batch)

            pred_norm = model.head.predict_mean(output)  # normalized
            # Denormalize predictions
            q_idx = torch.arange(pred_norm.shape[0], device=device)
            traj_idx = batch.get('_traj_idx', q_idx)
            q_target = batch['query_target']
            means = batch['_norm_means'][traj_idx, q_target]
            stds = batch['_norm_stds'][traj_idx, q_target]
            pred = pred_norm * stds + means

            all_preds.append(pred.cpu())
            all_targets.append(batch['Y_true'].cpu())
            all_effects.append(batch['Y_causal_effect'].cpu())
            # Pred effect = pred - obs_value at query time
            # X_int[t, var] - X_obs[t, var] = Y_causal_effect; so pred_effect = pred - (Y_true - Y_causal_effect)
            obs_val = batch['Y_true'].cpu() - batch['Y_causal_effect'].cpu()
            all_pred_effects.append(pred.cpu() - obs_val)
            if 'positivity_score' in batch:
                ps = batch['positivity_score'].cpu()
                if ps.dim() == 0:
                    ps = ps.unsqueeze(0)
                if '_traj_idx' in batch:
                    ps = ps[batch['_traj_idx'].cpu()]
                all_positivity.append(ps)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    effects = torch.cat(all_effects)
    pred_effects = torch.cat(all_pred_effects)
    positivity = torch.cat(all_positivity) if all_positivity else None

    # Metrics
    err = preds - targets
    rmse = torch.sqrt((err ** 2).mean()).item()
    mae = err.abs().mean().item()
    var = targets.var(unbiased=False).item()
    nmse = (err ** 2).mean().item() / max(var, 1e-8)
    ss_res = (err ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    # Direction accuracy (non-trivial targets)
    mask = targets.abs() > 0.1
    if mask.any():
        dir_acc = (torch.sign(preds[mask]) == torch.sign(targets[mask])).float().mean().item()
    else:
        dir_acc = float('nan')

    effect_rmse = torch.sqrt(((pred_effects - effects) ** 2).mean()).item()
    effect_mae = (pred_effects - effects).abs().mean().item()
    mean_effect = effects.abs().mean().item()

    result = {
        'rmse': rmse, 'mae': mae, 'nmse': nmse, 'r2': r2,
        'direction_accuracy': dir_acc,
        'effect_rmse': effect_rmse, 'effect_mae': effect_mae,
        'mean_effect_magnitude': mean_effect,
        'n_queries': preds.numel(),
    }
    if positivity is not None:
        pos = positivity.numpy()
        result['mean_positivity_score'] = float(pos.mean())
        result['frac_ood'] = float((pos > 0).mean())
    return result


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 80)

    experiments = [
        ('s5_bd_nolag_interp_causal', 'back_door', False),
        ('s5_bd_nolag_interp_obs',    'back_door', True),
        ('s5_fd_nolag_interp_causal', 'front_door', False),
        ('s5_fd_nolag_interp_obs',    'front_door', True),
    ]

    results = {}
    for name, structure, obs_only in experiments:
        ckpt_path = f'checkpoints/{name}/do_over_time_pfn_best.pt'
        if not os.path.exists(ckpt_path):
            print(f"MISSING: {ckpt_path}")
            continue
        print(f"\n--- {name} (structure={structure}, obs_only={obs_only}) ---")
        result = evaluate_checkpoint(ckpt_path, structure, observational_only=obs_only,
                                     device=device)
        results[name] = result
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save JSON
    os.makedirs('results/s5_analysis', exist_ok=True)
    with open('results/s5_analysis/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/s5_analysis/metrics.json")

    # Summary comparison
    print("\n" + "=" * 100)
    print(f"{'Name':<35} {'RMSE':>8} {'MAE':>8} {'nMSE':>8} {'R2':>8} {'Dir':>7} {'Eff.RMSE':>9} {'|Eff|':>7}")
    print("-" * 100)
    for name, r in results.items():
        print(f"{name:<35} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['nmse']:>8.4f} "
              f"{r['r2']:>8.4f} {r['direction_accuracy']:>6.1%} "
              f"{r['effect_rmse']:>9.4f} {r['mean_effect_magnitude']:>7.4f}")
    print("=" * 100)


if __name__ == '__main__':
    main()
