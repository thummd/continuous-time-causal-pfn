#!/usr/bin/env python
"""TSCM identifiability case studies.

For each causal structure, generate data and evaluate whether the model
correctly handles the identification strategy.

Modes:
  --verify-only    Just verify TSCM data generation for all structures
  --checkpoint     Evaluate a trained model on identifiability scenarios
"""

import argparse
import torch
import numpy as np

from dotime.prior.tscm_sampler import TSCMSampler, TSCMStructure
from dotime.prior.extended_prior import pad_to_max_nodes


def verify_data_generation(n_samples: int = 10, T: int = 50):
    """Verify that all TSCM structures generate valid data.

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

            # Create intervention: single-step do(A_t) with diverse values
            int_times = [max(0, T - 5)]
            int_value = float(torch.randn(1, generator=gen).item() * 2.0)
            intervention = InterventionSpec(
                targets=[int_target],
                times=int_times,
                intervention_type=InterventionType.HARD,
                values=int_value,
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


def compute_confounding_strength(X_obs, hidden_vars, int_target, N):
    """Estimate confounding strength from observational data.

    Measures correlation between the intervention target and other observed
    variables, which indicates potential confounding paths.

    Returns dict with:
        mean_abs_corr: mean absolute correlation of int_target with other vars
        max_abs_corr: max absolute correlation (strongest confounding path)
    """
    observed = [i for i in range(N) if i not in hidden_vars]
    if len(observed) < 2 or int_target not in observed:
        return {'mean_abs_corr': float('nan'), 'max_abs_corr': float('nan')}

    # Correlation matrix of observed variables
    X = X_obs[:, observed].numpy()
    if X.std(axis=0).min() < 1e-8:
        return {'mean_abs_corr': float('nan'), 'max_abs_corr': float('nan')}

    corr = np.corrcoef(X, rowvar=False)
    # Index of int_target within observed list
    target_obs_idx = observed.index(int_target)

    # Correlations of target with other observed variables
    other_corrs = [abs(corr[target_obs_idx, j])
                   for j in range(len(observed)) if j != target_obs_idx]

    if not other_corrs:
        return {'mean_abs_corr': 0.0, 'max_abs_corr': 0.0}

    return {
        'mean_abs_corr': float(np.mean(other_corrs)),
        'max_abs_corr': float(np.max(other_corrs)),
    }


def _evaluate_single_query(
    model, X_obs, X_int, X_obs_padded, X_norm, means, stds,
    variable_mask, int_target, int_times, q_idx, query_time_idx, T, device,
    int_value: float = 1.0,
):
    """Evaluate a single query (one variable at one time offset)."""
    time_start = min(int_times) / T
    time_end = max(int_times) / T

    y_int = X_int[query_time_idx, q_idx].item()
    y_obs = X_obs[query_time_idx, q_idx].item() if query_time_idx < X_obs.shape[0] else 0.0
    y_true_norm = (y_int - means[q_idx].item()) / stds[q_idx].item()
    causal_effect_norm = (y_int - y_obs) / stds[q_idx].item()

    # Normalize intervention value using the intervened variable's stats
    int_value_norm = (int_value - means[int_target].item()) / stds[int_target].item()

    batch = {
        'X_obs_norm': X_norm.unsqueeze(0).to(device),
        'variable_mask': variable_mask.unsqueeze(0).to(device),
        'intervention_target': torch.tensor([int_target], dtype=torch.long, device=device),
        'intervention_type': torch.tensor([0], dtype=torch.long, device=device),
        'intervention_value': torch.tensor([int_value_norm], dtype=torch.float32, device=device),
        'intervention_time_start': torch.tensor([time_start], dtype=torch.float32, device=device),
        'intervention_time_end': torch.tensor([time_end], dtype=torch.float32, device=device),
        'query_target': torch.tensor([q_idx], dtype=torch.long, device=device),
        'query_time': torch.tensor([query_time_idx / T], dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        output = model(batch)
        pred = model.head.predict_mean(output)

    pred_val = pred.cpu().item()
    obs_norm = (y_obs - means[q_idx].item()) / stds[q_idx].item()
    pred_effect = pred_val - obs_norm

    return {
        'pred': pred_val,
        'target': y_true_norm,
        'causal_effect': causal_effect_norm,
        'pred_effect': pred_effect,
        'output': output.cpu(),
    }


def _aggregate_results(records, all_confounding):
    """Compute aggregate metrics from a list of query records."""
    if not records:
        return {'total': 0}

    preds = torch.tensor([r['pred'] for r in records])
    targets = torch.tensor([r['target'] for r in records])
    effects = torch.tensor([r['causal_effect'] for r in records])
    pred_effects = torch.tensor([r['pred_effect'] for r in records])

    valid_conf = [c for c in all_confounding if not np.isnan(c['mean_abs_corr'])]
    mean_confounding = np.mean([c['mean_abs_corr'] for c in valid_conf]) if valid_conf else float('nan')

    return {
        'total': len(records),
        'rmse': torch.sqrt(torch.mean((preds - targets) ** 2)).item(),
        'mae': torch.mean(torch.abs(preds - targets)).item(),
        'direction_accuracy': ((preds * targets) > 0).float().mean().item(),
        'effect_rmse': torch.sqrt(torch.mean((pred_effects - effects) ** 2)).item(),
        'effect_mae': torch.mean(torch.abs(pred_effects - effects)).item(),
        'mean_effect_magnitude': effects.abs().mean().item(),
        'mean_confounding': mean_confounding,
    }


def evaluate_structure(
    model,
    structure: TSCMStructure,
    n_samples: int = 50,
    T: int = 50,
    n_max: int = 41,
    device: str = "cpu",
    multi_step: bool = False,
    query_offsets: list = None,
):
    """Evaluate model on a specific causal structure.

    Parameters
    ----------
    query_offsets : list of int, optional
        Time offsets after intervention to query. Default [3].
        E.g. [1, 2, 3, 5, 10] queries at 1, 2, 3, 5, 10 steps after do(A_t).

    Returns dict with aggregate metrics, per-TSCM metrics, and per-offset metrics.
    """
    from causal_time_prior.interventions import InterventionSpec, InterventionType

    if query_offsets is None:
        query_offsets = [3]

    sampler = TSCMSampler(structure, max_lag=1)
    hidden_vars = sampler.get_hidden_vars()

    gen = torch.Generator().manual_seed(42)

    per_tscm = []
    all_confounding = []
    # Per-offset record buckets
    per_offset_records = {offset: [] for offset in query_offsets}
    all_records = []

    for sample_idx in range(n_samples):
        scm = sampler.sample(generator=gen)
        N = len(scm._topo)

        X_obs = scm.sample_observational(T=T, burn_in=30, generator=gen)
        if X_obs.abs().max() > 900:
            continue

        valid_targets = [i for i in range(N) if i not in hidden_vars]
        if not valid_targets:
            continue
        int_target = valid_targets[0]

        conf = compute_confounding_strength(X_obs, hidden_vars, int_target, N)
        all_confounding.append(conf)

        if multi_step:
            int_times = list(range(max(0, T - 10), T))
        else:
            int_times = [max(0, T - 5)]

        # Sample intervention value from training distribution N(0, 2)
        # to match CTP's hard intervention sampling and avoid positivity issues
        int_value = float(torch.randn(1, generator=gen).item() * 2.0)

        intervention = InterventionSpec(
            targets=[int_target],
            times=int_times,
            intervention_type=InterventionType.HARD,
            values=int_value,
        )

        X_int = scm.sample_interventional(
            T=T, intervention=intervention, burn_in=30, generator=gen,
        )
        if X_int.abs().max() > 900:
            continue

        X_obs_padded = pad_to_max_nodes(X_obs, n_max)
        variable_mask = torch.zeros(n_max)
        for i in range(N):
            if i not in hidden_vars:
                variable_mask[i] = 1.0

        means = X_obs_padded.mean(dim=0)
        stds = X_obs_padded.std(dim=0) + 1e-2
        stds[variable_mask == 0] = 1.0
        X_norm = (X_obs_padded - means.unsqueeze(0)) / stds.unsqueeze(0)
        X_norm = X_norm * variable_mask.unsqueeze(0)

        tscm_records = []

        for q_idx in range(N):
            if q_idx in hidden_vars or q_idx == int_target:
                continue

            for offset in query_offsets:
                query_time_idx = min(int(min(int_times) + offset), T - 1)

                rec = _evaluate_single_query(
                    model, X_obs, X_int, X_obs_padded, X_norm, means, stds,
                    variable_mask, int_target, int_times, q_idx, query_time_idx,
                    T, device, int_value=int_value,
                )
                rec['offset'] = offset
                rec['q_idx'] = q_idx

                per_offset_records[offset].append(rec)
                all_records.append(rec)
                tscm_records.append(rec)

        if tscm_records:
            p = torch.tensor([r['pred'] for r in tscm_records])
            t = torch.tensor([r['target'] for r in tscm_records])
            e = torch.tensor([r['causal_effect'] for r in tscm_records])
            pe = torch.tensor([r['pred_effect'] for r in tscm_records])
            per_tscm.append({
                'sample_idx': sample_idx,
                'n_queries': len(tscm_records),
                'rmse': torch.sqrt(torch.mean((p - t) ** 2)).item(),
                'mae': torch.mean(torch.abs(p - t)).item(),
                'effect_rmse': torch.sqrt(torch.mean((pe - e) ** 2)).item(),
                'effect_mae': torch.mean(torch.abs(pe - e)).item(),
                'mean_causal_effect': e.mean().item(),
                'direction_accuracy': ((p * t) > 0).float().mean().item(),
                'confounding': conf,
            })

    if not all_records:
        return {'total': 0}

    # Aggregate across all offsets
    results = _aggregate_results(all_records, all_confounding)
    results['n_tscms'] = len(per_tscm)
    results['per_tscm'] = per_tscm

    # Per-offset breakdown
    results['per_offset'] = {}
    for offset in query_offsets:
        results['per_offset'][offset] = _aggregate_results(
            per_offset_records[offset], all_confounding,
        )

    # Quantile calibration if bar distribution is available
    if hasattr(model, 'bar_head') and model.bar_head is not None and model.bar_head.bar_dist is not None:
        from dotime.eval.metrics import compute_quantile_calibration, compute_pinball_metric
        tau_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        borders = model.bar_head.borders.cpu()
        all_logits = torch.cat([r['output'] for r in all_records])
        targets = torch.tensor([r['target'] for r in all_records])
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
    parser.add_argument("--query-offsets", type=int, nargs="+", default=[3],
                        help="Time offsets after intervention to query (default: 3). "
                             "Use e.g. '1 2 3 5 10' to study effect decay.")
    args = parser.parse_args()

    if args.verify_only:
        verify_data_generation(n_samples=args.n_samples)
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --verify-only is set")

    # Load model
    from dotime.model.do_over_time_pfn import DoOverTimePFN

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt['config']
    model = DoOverTimePFN(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Restore bar distribution if applicable
    head_type = ckpt.get('head_type', 'bar')
    if head_type == 'bar' and ckpt.get('borders') is not None:
        from pfns.model.bar_distribution import FullSupportBarDistribution
        borders = ckpt['borders']
        bar_dist = FullSupportBarDistribution(borders)
        model.bar_head.set_bar_distribution(bar_dist, borders)

    model = model.to(args.device)
    model.eval()

    print("=" * 80)
    print("TSCM Identifiability Case Studies")
    print("=" * 80)

    all_results = {}
    for structure in TSCMStructure:
        print(f"\n--- {structure.value} ---")
        results = evaluate_structure(
            model, structure, n_samples=args.n_samples, device=args.device,
            multi_step=args.multi_step, query_offsets=args.query_offsets,
        )
        all_results[structure.value] = results

        if results['total'] == 0:
            print("  No valid queries generated")
            continue

        print(f"  Queries: {results['total']} across {results['n_tscms']} TSCMs")
        print(f"  RMSE: {results['rmse']:.4f} | MAE: {results['mae']:.4f}")
        print(f"  Direction accuracy: {results['direction_accuracy']:.2%}")
        print(f"  Causal effect RMSE: {results['effect_rmse']:.4f} | MAE: {results['effect_mae']:.4f}")
        print(f"  Mean |causal effect|: {results['mean_effect_magnitude']:.4f}")
        print(f"  Mean confounding (corr): {results['mean_confounding']:.4f}")

        # Per-TSCM summary
        if results['per_tscm']:
            per_rmse = [t['rmse'] for t in results['per_tscm']]
            per_effect = [t['mean_causal_effect'] for t in results['per_tscm']]
            print(f"  Per-TSCM RMSE: min={min(per_rmse):.4f} median={np.median(per_rmse):.4f} max={max(per_rmse):.4f}")
            print(f"  Per-TSCM effect: min={min(per_effect):.4f} median={np.median(per_effect):.4f} max={max(per_effect):.4f}")

        if 'calibration_error' in results:
            print(f"  Calibration error: {results['calibration_error']:.4f}")

        # Per-offset effect decay
        if 'per_offset' in results and len(results['per_offset']) > 1:
            print(f"  Effect decay by query offset:")
            for offset in sorted(results['per_offset'].keys()):
                r_off = results['per_offset'][offset]
                if r_off['total'] == 0:
                    continue
                print(f"    t+{offset}: RMSE={r_off['rmse']:.4f} "
                      f"Eff.RMSE={r_off['effect_rmse']:.4f} "
                      f"|Effect|={r_off['mean_effect_magnitude']:.4f} "
                      f"Dir={r_off['direction_accuracy']:.1%}")

    # Summary table
    print("\n" + "=" * 105)
    print(f"{'Structure':<25} {'N':>5} {'RMSE':>7} {'MAE':>7} {'Dir':>6} "
          f"{'Eff.RMSE':>8} {'|Effect|':>8} {'Confound':>8}")
    print("-" * 105)
    for name, r in all_results.items():
        if r['total'] == 0:
            print(f"{name:<25} {'--':>5}")
            continue
        print(f"{name:<25} {r['total']:>5} {r['rmse']:>7.4f} {r['mae']:>7.4f} "
              f"{r['direction_accuracy']:>5.1%} "
              f"{r['effect_rmse']:>8.4f} {r['mean_effect_magnitude']:>8.4f} "
              f"{r['mean_confounding']:>8.4f}")
    print("=" * 105)

    # Per-offset summary table (if multiple offsets)
    if len(args.query_offsets) > 1:
        print("\n" + "=" * 90)
        print("Effect decay by query offset (averaged across structures)")
        print("-" * 90)
        print(f"{'Offset':>8} {'N':>6} {'RMSE':>8} {'Eff.RMSE':>9} {'|Effect|':>9} {'Dir.Acc':>8}")
        print("-" * 90)
        for offset in sorted(args.query_offsets):
            # Pool across structures
            off_totals = []
            for name, r in all_results.items():
                if r['total'] > 0 and offset in r.get('per_offset', {}):
                    off_totals.append(r['per_offset'][offset])
            if off_totals:
                n = sum(o['total'] for o in off_totals)
                avg_rmse = np.mean([o['rmse'] for o in off_totals])
                avg_eff = np.mean([o['effect_rmse'] for o in off_totals])
                avg_mag = np.mean([o['mean_effect_magnitude'] for o in off_totals])
                avg_dir = np.mean([o['direction_accuracy'] for o in off_totals])
                print(f"  t+{offset:<5} {n:>6} {avg_rmse:>8.4f} {avg_eff:>9.4f} "
                      f"{avg_mag:>9.4f} {avg_dir:>7.1%}")
        print("=" * 90)


if __name__ == "__main__":
    main()
