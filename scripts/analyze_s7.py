#!/usr/bin/env python
"""Evaluate s7 checkpoints on BD and FD with positivity-clipped interventions.

Matches the s7 training protocol: intervention_source=positivity_aware,
query_offset_range=(0,5). Reports RMSE, MAE, nMSE, R^2, direction accuracy,
causal-effect metrics, and %OOD.

Recommended collaborator usage (T=5000 eval on steady-state trajectories):

    python scripts/analyze_s7.py \\
        --t-range 5000 5000 --n-batches 20 --sim-device cpu \\
        --results-dir results/s7_analysis_T5000

The encoder truncates to the last context_window=200 pre-intervention steps
regardless of T. At T=5000 those 200 steps reflect a warmed-up steady-state
SCM, which empirically gives ~17% lower RMSE and much better R^2 than the
T=200 setting used during training. Runtime: ~16s/batch on CPU sim.
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


def evaluate_checkpoint(ckpt_path, structure, n_batches=50, batch_size=16,
                        observational_only=False, device='cuda:0',
                        intervention_source='positivity_aware',
                        query_offset_range=(0, 5),
                        t_range=(50, 200),
                        dynamics_burn_in=500,
                        sim_device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DoOverTimePFN(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    loader = TemporalInterventionDataLoader(
        num_steps=n_batches, batch_size=batch_size,
        n_max=12, n_max_prior=10,
        t_range=t_range, burn_in=50,
        seed=12345, device=device,
        target_key='Y_true',
        n_queries=100, query_mode='all_pairs',
        tscm_structure=structure,
        use_lagged_edges=False,
        intervention_scale=4.0,
        causal_mask_mode='interpolation',
        dynamics_burn_in=dynamics_burn_in,
        intervention_source=intervention_source,
        query_offset_range=query_offset_range,
        sim_device=sim_device,
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

            if '_traj_idx' in batch:
                h_vars = model.encode(batch)
                h_vars_exp = h_vars[batch['_traj_idx']]
                qb = {k: v for k, v in batch.items() if k not in ('X_obs_norm', '_traj_idx')}
                qb['variable_mask'] = batch['variable_mask'][batch['_traj_idx']]
                output = model.query(h_vars_exp, qb)
            else:
                output = model(batch)

            pred_norm = model.head.predict_mean(output)
            q_idx = torch.arange(pred_norm.shape[0], device=device)
            traj_idx = batch.get('_traj_idx', q_idx)
            q_target = batch['query_target']
            means = batch['_norm_means'][traj_idx, q_target]
            stds = batch['_norm_stds'][traj_idx, q_target]
            pred = pred_norm * stds + means

            all_preds.append(pred.cpu())
            all_targets.append(batch['Y_true'].cpu())
            all_effects.append(batch['Y_causal_effect'].cpu())
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

    err = preds - targets
    rmse = torch.sqrt((err ** 2).mean()).item()
    mae = err.abs().mean().item()
    var = targets.var(unbiased=False).item()
    nmse = (err ** 2).mean().item() / max(var, 1e-8)
    ss_res = (err ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)

    mask = targets.abs() > 0.1
    dir_acc = ((torch.sign(preds[mask]) == torch.sign(targets[mask])).float().mean().item()
               if mask.any() else float('nan'))

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
    import argparse
    parser = argparse.ArgumentParser(description="Analyze s7 checkpoints.")
    parser.add_argument('--t-range', type=int, nargs=2, default=[50, 200],
                        metavar=('LO', 'HI'),
                        help="Eval trajectory length range, uniform in [LO, HI]. "
                             "Use e.g. 1000 5000 to evaluate on long trajectories.")
    parser.add_argument('--dynamics-burn-in', type=int, default=500,
                        help="Extra burn-in steps for SCM dynamics (default 500).")
    parser.add_argument('--n-batches', type=int, default=50,
                        help="Number of eval batches (default 50).")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--sim-device', type=str, default='cpu',
                        help="BatchedTSCMSimulator device. 'cpu' is faster for B<=64; "
                             "try 'cuda:0' only with large batches or very long T.")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default='results/s7_analysis',
                        help="Where to write metrics.json")
    args = parser.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    t_range = tuple(args.t_range)
    print(f"Device: {device} | t_range: {t_range} | n_batches: {args.n_batches}")
    print("=" * 80)

    experiments = [
        ('s7_bd_nolag_interp_causal', 'back_door',  False),
        ('s7_bd_nolag_interp_obs',    'back_door',  True),
        ('s7_fd_nolag_interp_causal', 'front_door', False),
        ('s7_fd_nolag_interp_obs',    'front_door', True),
    ]

    results = {}
    for name, structure, obs_only in experiments:
        ckpt_path = f'checkpoints/{name}/do_over_time_pfn_best.pt'
        if not os.path.exists(ckpt_path):
            print(f"MISSING: {ckpt_path}")
            continue
        print(f"\n--- {name} (structure={structure}, obs_only={obs_only}) ---")
        result = evaluate_checkpoint(
            ckpt_path, structure, observational_only=obs_only,
            device=device,
            t_range=t_range,
            dynamics_burn_in=args.dynamics_burn_in,
            n_batches=args.n_batches,
            batch_size=args.batch_size,
            sim_device=args.sim_device,
        )
        results[name] = result
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n" + "=" * 110)
    print(f"{'Name':<35} {'RMSE':>8} {'MAE':>8} {'nMSE':>8} {'R2':>8} "
          f"{'Dir':>7} {'Eff.RMSE':>9} {'|Eff|':>7} {'%OOD':>6}")
    print("-" * 110)
    for name, r in results.items():
        print(f"{name:<35} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['nmse']:>8.4f} "
              f"{r['r2']:>8.4f} {r['direction_accuracy']:>6.1%} "
              f"{r['effect_rmse']:>9.4f} {r['mean_effect_magnitude']:>7.4f} "
              f"{r.get('frac_ood', float('nan')):>5.1%}")
    print("=" * 110)


if __name__ == '__main__':
    main()
