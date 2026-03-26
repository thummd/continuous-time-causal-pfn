#!/usr/bin/env python
"""TSCM identifiability case studies.

For each causal structure, generate data and evaluate whether the model
correctly handles the identification strategy.

Modes:
  --verify-only    Just verify TSCM data generation for all 6 structures
  --checkpoint     Evaluate a trained model on identifiability scenarios
"""

import argparse
import torch
import numpy as np

from dotime.prior.tscm_sampler import TSCMSampler, TSCMStructure
from dotime.prior.extended_prior import pad_to_max_nodes


def verify_data_generation(n_samples: int = 10, T: int = 50):
    """Verify that all 6 TSCM structures generate valid data.

    Checks: shapes, finite values, hidden variable masking, and
    that interventional data differs from observational.
    """
    from causal_time_prior.interventions import InterventionSpec, InterventionType

    print("=" * 60)
    print("TSCM Data Generation Verification")
    print("=" * 60)

    all_ok = True

    for structure in TSCMStructure:
        print(f"\n--- {structure.value} ---")
        sampler = TSCMSampler(structure, max_lag=1)
        hidden_vars = sampler.get_hidden_vars()
        print(f"  Hidden variables: {hidden_vars if hidden_vars else 'none'}")

        gen = torch.Generator().manual_seed(42)
        n_valid = 0
        n_divergent = 0
        obs_shapes = []
        int_shapes = []
        obs_int_diffs = []

        for i in range(n_samples):
            scm = sampler.sample(generator=gen)
            N = len(scm._topo)

            # Generate observational data
            X_obs = scm.sample_observational(T=T, burn_in=30, generator=gen)

            if X_obs.abs().max() > 900:
                n_divergent += 1
                continue

            obs_shapes.append(tuple(X_obs.shape))

            # Pick intervention target (first non-hidden variable)
            valid_targets = [j for j in range(N) if j not in hidden_vars]
            if not valid_targets:
                continue
            int_target = valid_targets[0]

            # Create intervention: single-step do(A_t) by default
            int_times = [max(0, T - 5)]
            intervention = InterventionSpec(
                targets=[int_target],
                times=int_times,
                intervention_type=InterventionType.HARD,
                values=1.0,
            )

            X_int = scm.sample_interventional(
                T=T, intervention=intervention, burn_in=30, generator=gen,
            )
            int_shapes.append(tuple(X_int.shape))

            # Check interventional differs from observational
            diff = (X_int - X_obs).abs().mean().item()
            obs_int_diffs.append(diff)

            # Check finiteness
            if not torch.isfinite(X_obs).all():
                print(f"  WARNING: Non-finite values in X_obs (sample {i})")
                all_ok = False
            if not torch.isfinite(X_int).all():
                print(f"  WARNING: Non-finite values in X_int (sample {i})")
                all_ok = False

            n_valid += 1

        print(f"  Valid samples: {n_valid}/{n_samples} (divergent: {n_divergent})")
        if obs_shapes:
            print(f"  X_obs shape: {obs_shapes[0]} (T={T}, N={obs_shapes[0][1]})")
        if obs_int_diffs:
            mean_diff = np.mean(obs_int_diffs)
            print(f"  Mean |X_int - X_obs|: {mean_diff:.4f}")
            if mean_diff < 1e-6:
                print(f"  WARNING: Interventional data identical to observational!")
                all_ok = False
        print(f"  Node names: {scm._topo}")

    print("\n" + "=" * 60)
    if all_ok:
        print("All structures OK")
    else:
        print("Some issues found — see warnings above")
    print("=" * 60)
    return all_ok


def evaluate_structure(
    model,
    structure: TSCMStructure,
    n_samples: int = 50,
    T: int = 50,
    n_max: int = 41,
    device: str = "cpu",
    multi_step: bool = False,
):
    """Evaluate model on a specific causal structure.

    Returns dict with metrics about causal correctness.
    """
    from causal_time_prior.interventions import InterventionSpec, InterventionType

    sampler = TSCMSampler(structure, max_lag=1)
    hidden_vars = sampler.get_hidden_vars()

    gen = torch.Generator().manual_seed(42)
    preds_list = []
    targets_list = []
    logits_list = []

    for _ in range(n_samples):
        scm = sampler.sample(generator=gen)
        N = len(scm._topo)

        # Generate observational data
        X_obs = scm.sample_observational(T=T, burn_in=30, generator=gen)
        if X_obs.abs().max() > 900:
            continue  # Skip divergent SCMs

        # Pick intervention target (first non-hidden variable that has children)
        valid_targets = [i for i in range(N) if i not in hidden_vars]
        if not valid_targets:
            continue

        int_target = valid_targets[0]

        # Intervention timing: single-step by default, multi-step for g-computation
        if multi_step:
            int_times = list(range(max(0, T - 10), T))
        else:
            int_times = [max(0, T - 5)]
        intervention = InterventionSpec(
            targets=[int_target],
            times=int_times,
            intervention_type=InterventionType.HARD,
            values=1.0,
        )

        # Generate interventional data for ground truth
        X_int = scm.sample_interventional(
            T=T, intervention=intervention, burn_in=30, generator=gen,
        )
        if X_int.abs().max() > 900:
            continue

        # Pad and normalize
        X_obs_padded = pad_to_max_nodes(X_obs, n_max)
        variable_mask = torch.zeros(n_max)
        for i in range(N):
            if i not in hidden_vars:
                variable_mask[i] = 1.0

        means = X_obs_padded.mean(dim=0)
        stds = X_obs_padded.std(dim=0) + 1e-8
        stds[variable_mask == 0] = 1.0
        X_norm = (X_obs_padded - means.unsqueeze(0)) / stds.unsqueeze(0)
        X_norm = X_norm * variable_mask.unsqueeze(0)

        # Intervention time normalized
        time_start = min(int_times) / T
        time_end = max(int_times) / T

        # Query each non-hidden, non-intervention variable
        for q_idx in range(N):
            if q_idx in hidden_vars or q_idx == int_target:
                continue

            # Ground truth: interventional value at query time
            query_time_idx = min(int(min(int_times) + 3), T - 1)
            y_true = X_int[query_time_idx, q_idx].item()
            y_true_norm = (y_true - means[q_idx].item()) / stds[q_idx].item()

            batch = {
                'X_obs_norm': X_norm.unsqueeze(0).to(device),
                'variable_mask': variable_mask.unsqueeze(0).to(device),
                'intervention_target': torch.tensor([int_target], dtype=torch.long, device=device),
                'intervention_type': torch.tensor([0], dtype=torch.long, device=device),
                'intervention_value': torch.tensor([1.0], dtype=torch.float32, device=device),
                'intervention_time_start': torch.tensor([time_start], dtype=torch.float32, device=device),
                'intervention_time_end': torch.tensor([time_end], dtype=torch.float32, device=device),
                'query_target': torch.tensor([q_idx], dtype=torch.long, device=device),
                'query_time': torch.tensor([query_time_idx / T], dtype=torch.float32, device=device),
            }

            with torch.no_grad():
                logits = model(batch)
                pred = model.bar_head.predict_mean(logits)

            preds_list.append(pred.cpu().item())
            targets_list.append(y_true_norm)
            logits_list.append(logits.cpu())

    if not preds_list:
        return {'total': 0}

    preds = torch.tensor(preds_list)
    targets = torch.tensor(targets_list)
    all_logits = torch.cat(logits_list)

    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    mae = torch.mean(torch.abs(preds - targets)).item()

    # Direction correctness: does the prediction move in the same direction
    # as the ground truth relative to the observational mean (0 in normalized space)?
    correct_dir = ((preds * targets) > 0).float().mean().item()

    results = {
        'total': len(preds_list),
        'rmse': rmse,
        'mae': mae,
        'direction_accuracy': correct_dir,
    }

    # Quantile calibration if bar distribution is available
    if model.bar_head.bar_dist is not None:
        from dotime.eval.metrics import compute_quantile_calibration, compute_pinball_metric
        tau_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        borders = model.bar_head.borders.cpu()
        cal = compute_quantile_calibration(all_logits, borders, targets, tau_levels)
        results['calibration_error'] = cal['calibration_error']
        results['pinball_loss'] = compute_pinball_metric(
            all_logits, borders, targets, tau_levels)
        results['coverage'] = {k: v for k, v in cal.items() if k.startswith('coverage_')}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify data generation, no model evaluation")
    parser.add_argument("--multi-step", action="store_true",
                        help="Use multi-step interventions (for g-computation)")
    args = parser.parse_args()

    if args.verify_only:
        verify_data_generation(n_samples=args.n_samples)
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --verify-only is set")

    # Load model
    from dotime.model.do_over_time_pfn import DoOverTimePFN
    from pfns.model.bar_distribution import FullSupportBarDistribution

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt['config']
    model = DoOverTimePFN(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    borders = ckpt['borders']
    bar_dist = FullSupportBarDistribution(borders)
    model.bar_head.set_bar_distribution(bar_dist, borders)
    model = model.to(args.device)
    model.eval()

    print("=" * 60)
    print("TSCM Identifiability Case Studies")
    print("=" * 60)

    all_results = {}
    for structure in TSCMStructure:
        print(f"\n--- {structure.value} ---")
        results = evaluate_structure(
            model, structure, n_samples=args.n_samples, device=args.device,
            multi_step=args.multi_step,
        )
        all_results[structure.value] = results

        if results['total'] == 0:
            print("  No valid queries generated")
            continue

        print(f"  Queries: {results['total']}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  Direction accuracy: {results['direction_accuracy']:.2%}")
        if 'calibration_error' in results:
            print(f"  Calibration error: {results['calibration_error']:.4f}")
            print(f"  Pinball loss: {results['pinball_loss']:.4f}")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Structure':<25} {'N':>5} {'RMSE':>8} {'MAE':>8} {'Dir.Acc':>8} {'Cal.Err':>8}")
    print("-" * 72)
    for name, r in all_results.items():
        if r['total'] == 0:
            print(f"{name:<25} {'—':>5}")
            continue
        cal = r.get('calibration_error', float('nan'))
        print(f"{name:<25} {r['total']:>5} {r['rmse']:>8.4f} {r['mae']:>8.4f} "
              f"{r['direction_accuracy']:>7.1%} {cal:>8.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
