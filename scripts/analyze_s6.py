#!/usr/bin/env python
"""Evaluate s6 checkpoints on BD and FD structures with detailed metrics.

For each of the 4 s6 checkpoints, generates evaluation batches with the
same TSCM structure as training (no-lag, interpolation mask) and reports
RMSE, MAE, nMSE, R2, direction accuracy, and causal effect metrics.

For obs-only models (trained on Y_obs via Fix 2d), we zero the intervention
context at eval time (matching training) but still compare predictions
against Y_int — this is the real test: can the model predict the
counterfactual without knowing the intervention?
"""

import os
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Reuse the s5 evaluator — data pipeline and metrics are identical
from analyze_s5 import evaluate_checkpoint


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 80)

    experiments = [
        ('s6_bd_nolag_interp_causal', 'back_door',  False),
        ('s6_bd_nolag_interp_obs',    'back_door',  True),
        ('s6_fd_nolag_interp_causal', 'front_door', False),
        ('s6_fd_nolag_interp_obs',    'front_door', True),
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

    os.makedirs('results/s6_analysis', exist_ok=True)
    with open('results/s6_analysis/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/s6_analysis/metrics.json")

    # Summary comparison
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
