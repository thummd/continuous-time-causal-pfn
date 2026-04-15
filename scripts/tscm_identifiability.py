#!/usr/bin/env python
"""TSCM identifiability case studies.

For each causal structure, generate data and evaluate whether the model
correctly handles the identification strategy.

Modes:
  --verify-only    Just verify TSCM data generation for all structures
  --checkpoint     Evaluate a trained model on identifiability scenarios
"""

import argparse
import json
import os
import torch
import numpy as np

from dotime.prior.tscm_sampler import TSCMSampler, TSCMStructure
from dotime.prior.extended_prior import pad_to_max_nodes
from dotime.eval.metrics import compute_rmse, compute_mae, compute_r2, compute_nmse


# Near-zero targets are ambiguous for sign-based direction accuracy.
# Targets with |t| < DIR_ACC_EPS are excluded from the metric and reported separately.
DIR_ACC_EPS = 0.1


def _direction_accuracy(preds: torch.Tensor, targets: torch.Tensor, eps: float = DIR_ACC_EPS):
    """Sign-consistent direction accuracy, excluding near-zero targets.

    Returns a dict with:
        accuracy: fraction of non-excluded predictions with matching sign
        n_valid:  number of samples used (|target| >= eps)
        n_excluded: number of samples skipped (|target| < eps)
    """
    if preds.numel() == 0:
        return {'accuracy': float('nan'), 'n_valid': 0, 'n_excluded': 0}
    mask = targets.abs() >= eps
    n_valid = int(mask.sum().item())
    n_excluded = int(preds.numel() - n_valid)
    if n_valid == 0:
        return {'accuracy': float('nan'), 'n_valid': 0, 'n_excluded': n_excluded}
    p = preds[mask]
    t = targets[mask]
    acc = ((p.sign() == t.sign()).float().mean().item())
    return {'accuracy': acc, 'n_valid': n_valid, 'n_excluded': n_excluded}


def _bootstrap_ci(values, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap mean, std, and (1-alpha) CI from a list of per-sample values.

    Returns (mean, std, ci_low, ci_high). Returns NaNs when values is empty.
    """
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
                     dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    if arr.size == 1:
        v = float(arr[0])
        return v, 0.0, v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    boot_means = arr[idx].mean(axis=1)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return mean, std, lo, hi


def verify_data_generation(n_samples: int = 10, T: int = 50, max_lag: int = 1,
                           t_range: tuple = None):
    """Verify that all TSCM structures generate valid data.

    Checks: shapes, finite values, hidden variable masking, and
    that interventional data differs from observational.
    """
    from causal_time_prior.interventions import InterventionSpec, InterventionType

    print("=" * 60)
    print("TSCM Data Generation Verification")
    print("=" * 60)
    t_rng = np.random.RandomState(42)

    all_ok = True

    for structure in TSCMStructure:
        print(f"\n--- {structure.value} ---")
        sampler = TSCMSampler(structure, max_lag=max_lag)
        hidden_vars = sampler.get_hidden_vars()
        print(f"  Hidden variables: {hidden_vars if hidden_vars else 'none'}")

        gen = torch.Generator().manual_seed(42)
        n_valid = 0
        n_divergent = 0
        obs_shapes = []
        int_shapes = []
        obs_int_diffs = []

        for i in range(n_samples):
            T_sample = t_rng.randint(t_range[0], t_range[1] + 1) if t_range else T

            scm = sampler.sample(generator=gen)
            N = len(scm._topo)

            # Generate observational data
            X_obs = scm.sample_observational(T=T_sample, burn_in=30, generator=gen)

            if X_obs.abs().max() > 10:
                n_divergent += 1
                continue

            obs_shapes.append(tuple(X_obs.shape))

            # Pick intervention target (first non-hidden variable)
            valid_targets = [j for j in range(N) if j not in hidden_vars]
            if not valid_targets:
                continue
            int_target = valid_targets[0]

            # Create intervention: single-step do(A_t) with diverse values
            int_times = [max(0, T_sample - 5)]
            int_value = float(torch.randn(1, generator=gen).item() * 2.0)
            intervention = InterventionSpec(
                targets=[int_target],
                times=int_times,
                intervention_type=InterventionType.HARD,
                values=int_value,
            )

            X_int = scm.sample_interventional(
                T=T_sample, intervention=intervention, burn_in=30, generator=gen,
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
            if t_range:
                print(f"  X_obs shapes: T in [{t_range[0]}, {t_range[1]}], N={obs_shapes[0][1]}")
            else:
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
    y_true_norm = max(-10.0, min(10.0, (y_int - means[q_idx].item()) / stds[q_idx].item()))
    causal_effect_norm = max(-10.0, min(10.0, (y_int - y_obs) / stds[q_idx].item()))

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
    mean_confounding = (np.mean([c['mean_abs_corr'] for c in valid_conf])
                        if valid_conf else float('nan'))

    dir_acc = _direction_accuracy(preds, targets)

    return {
        'total': len(records),
        'rmse': compute_rmse(preds, targets),
        'mae': compute_mae(preds, targets),
        'nmse': compute_nmse(preds, targets),
        'r2': compute_r2(preds, targets),
        'direction_accuracy': dir_acc['accuracy'],
        'direction_n_valid': dir_acc['n_valid'],
        'direction_n_excluded': dir_acc['n_excluded'],
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
    t_range: tuple = None,
    n_max: int = 41,
    device: str = "cpu",
    multi_step: bool = False,
    query_offsets: list = None,
    max_lag: int = 1,
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

    sampler = TSCMSampler(structure, max_lag=max_lag)
    hidden_vars = sampler.get_hidden_vars()

    gen = torch.Generator().manual_seed(42)
    t_rng = np.random.RandomState(42)

    per_tscm = []
    all_confounding = []
    # Per-offset record buckets
    per_offset_records = {offset: [] for offset in query_offsets}
    all_records = []

    for sample_idx in range(n_samples):
        T_sample = t_rng.randint(t_range[0], t_range[1] + 1) if t_range else T

        scm = sampler.sample(generator=gen)
        N = len(scm._topo)

        X_obs = scm.sample_observational(T=T_sample, burn_in=30, generator=gen)
        if X_obs.abs().max() > 10:
            continue

        valid_targets = [i for i in range(N) if i not in hidden_vars]
        if not valid_targets:
            continue
        int_target = valid_targets[0]

        conf = compute_confounding_strength(X_obs, hidden_vars, int_target, N)
        all_confounding.append(conf)

        if multi_step:
            int_times = list(range(max(0, T_sample - 10), T_sample))
        else:
            int_times = [max(0, T_sample - 5)]

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
            T=T_sample, intervention=intervention, burn_in=30, generator=gen,
        )
        if X_int.abs().max() > 10:
            continue
        # Skip all-zero samples (diverged SCMs that returned zeros)
        if X_obs.abs().max() < 1e-6 or X_int.abs().max() < 1e-6:
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
                query_time_idx = min(int(min(int_times) + offset), T_sample - 1)

                rec = _evaluate_single_query(
                    model, X_obs, X_int, X_obs_padded, X_norm, means, stds,
                    variable_mask, int_target, int_times, q_idx, query_time_idx,
                    T_sample, device, int_value=int_value,
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
            tscm_dir = _direction_accuracy(p, t)
            per_tscm.append({
                'sample_idx': sample_idx,
                'n_queries': len(tscm_records),
                'rmse': compute_rmse(p, t),
                'mae': compute_mae(p, t),
                'nmse': compute_nmse(p, t),
                'r2': compute_r2(p, t),
                'effect_rmse': torch.sqrt(torch.mean((pe - e) ** 2)).item(),
                'effect_mae': torch.mean(torch.abs(pe - e)).item(),
                'mean_causal_effect': e.mean().item(),
                'direction_accuracy': tscm_dir['accuracy'],
                'direction_n_valid': tscm_dir['n_valid'],
                'direction_n_excluded': tscm_dir['n_excluded'],
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
    parser.add_argument("--max-lag", type=int, default=1,
                        help="Max lag K for TSCM structures (1=Markov, 2-3 for higher-order)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="If set, write full results dict to this JSON path. "
                             "Default: results/<ckpt_parent>/tscm_eval.json when --checkpoint is given.")
    parser.add_argument("--bootstrap-n", type=int, default=1000,
                        help="Number of bootstrap resamples for CI estimation (default: 1000)")
    parser.add_argument("--T", type=int, default=None,
                        help="Fixed trajectory length (default: 50 if --t-range not given)")
    parser.add_argument("--t-range", type=int, nargs=2, default=None,
                        metavar=("T_MIN", "T_MAX"),
                        help="Sample T uniformly from [T_MIN, T_MAX] per sample, "
                             "matching training. Mutually exclusive with --T.")
    args = parser.parse_args()

    if args.T is not None and args.t_range is not None:
        parser.error("--T and --t-range are mutually exclusive")
    if args.t_range is not None and args.t_range[0] > args.t_range[1]:
        parser.error("--t-range: T_MIN must be <= T_MAX")
    if args.T is None and args.t_range is None:
        args.T = 50

    if args.verify_only:
        verify_data_generation(n_samples=args.n_samples, max_lag=args.max_lag,
                               t_range=tuple(args.t_range) if args.t_range else None)
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
    if args.t_range:
        print(f"T: sampled uniformly from [{args.t_range[0]}, {args.t_range[1]}]")
    else:
        print(f"T: {args.T} (fixed)")
    print("=" * 80)

    t_range_tuple = tuple(args.t_range) if args.t_range else None
    all_results = {}
    for structure in TSCMStructure:
        print(f"\n--- {structure.value} ---")
        results = evaluate_structure(
            model, structure, n_samples=args.n_samples,
            T=args.T if args.T is not None else 50,
            t_range=t_range_tuple,
            device=args.device,
            multi_step=args.multi_step, query_offsets=args.query_offsets,
            max_lag=args.max_lag,
        )
        all_results[structure.value] = results

        if results['total'] == 0:
            print("  No valid queries generated")
            continue

        # Bootstrap CIs from per-TSCM values
        per_tscm = results.get('per_tscm', [])
        boot = {}
        if per_tscm:
            for key in ('rmse', 'mae', 'direction_accuracy', 'effect_rmse', 'effect_mae', 'nmse', 'r2'):
                vals = [t[key] for t in per_tscm if t.get(key) is not None]
                mean, std, lo, hi = _bootstrap_ci(vals, n=args.bootstrap_n)
                # Also compute median (robust to outliers, especially for nMSE)
                clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                median = float(np.median(clean)) if clean else float('nan')
                boot[key] = {'mean': mean, 'std': std, 'ci_low': lo, 'ci_high': hi, 'median': median}
        results['bootstrap'] = boot

        print(f"  Queries: {results['total']} across {results['n_tscms']} TSCMs")
        def _fmt(key, pct=False):
            b = boot.get(key, {})
            m = b.get('mean', float('nan'))
            s = b.get('std', float('nan'))
            l = b.get('ci_low', float('nan'))
            h = b.get('ci_high', float('nan'))
            if pct:
                return f"{m:.2} ± {s:.2} ({l:.2}, {h:.2})"
            return f"{m:.4} ± {s:.4} ({l:.4}, {h:.4})"
        nmse_med = boot.get('nmse', {}).get('median', float('nan'))
        print(f"  RMSE: {_fmt('rmse')} | MAE: {_fmt('mae')} | nMSE: {_fmt('nmse')} (median={nmse_med:.4f}) | R2 {_fmt('r2')}" )
        n_valid = results.get('direction_n_valid', 0)
        n_excl = results.get('direction_n_excluded', 0)
        print(f"  Direction accuracy: {_fmt('direction_accuracy', pct=True)} "
              f"(n_valid={n_valid}, excluded={n_excl}, |t|<{DIR_ACC_EPS})")
        print(f"  Causal effect RMSE: {_fmt('effect_rmse')} | MAE: {_fmt('effect_mae')}")
        print(f"  Mean |causal effect|: {results['mean_effect_magnitude']:.4f}")
        print(f"  Mean confounding (corr): {results['mean_confounding']:.4f}")

        # Per-TSCM summary
        if per_tscm:
            per_rmse = [t['rmse'] for t in per_tscm]
            per_effect = [t['mean_causal_effect'] for t in per_tscm]
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
    header_w = 155
    print("\n" + "=" * header_w)
    print(f"{'Structure':<25} {'N':>4} {'Nt':>4} "
          f"{'RMSE':>8} {'±std':>7} "
          f"{'nMSE':>8} {'med':>8} "
          f"{'Dir':>7} {'±std':>7} "
          f"{'Nv':>4} {'Nx':>4} "
          f"{'Eff.RMSE':>9} {'|Effect|':>9} {'Confound':>8}")
    print("-" * header_w)
    for name, r in all_results.items():
        if r['total'] == 0:
            print(f"{name:<25} {'--':>4}")
            continue
        b = r.get('bootstrap', {})
        rmse_m = b.get('rmse', {}).get('mean', r['rmse'])
        rmse_s = b.get('rmse', {}).get('std', 0.0)
        nmse_mean = b.get('nmse', {}).get('mean', r.get('nmse', float('nan')))
        nmse_med = b.get('nmse', {}).get('median', float('nan'))
        dir_m = b.get('direction_accuracy', {}).get('mean', r['direction_accuracy'])
        dir_s = b.get('direction_accuracy', {}).get('std', 0.0)
        eff_m = b.get('effect_rmse', {}).get('mean', r['effect_rmse'])
        n_valid = r.get('direction_n_valid', 0)
        n_excl = r.get('direction_n_excluded', 0)
        print(f"{name:<25} {r['total']:>4} {r.get('n_tscms', 0):>4} "
              f"{rmse_m:>8.4f} {rmse_s:>7.4f} "
              f"{nmse_mean:>8.3f} {nmse_med:>8.4f} "
              f"{dir_m:>6.1%} {dir_s:>6.1%} "
              f"{n_valid:>4} {n_excl:>4} "
              f"{eff_m:>9.4f} {r['mean_effect_magnitude']:>9.4f} "
              f"{r['mean_confounding']:>8.4f}")
    print("=" * header_w)
    print(f"(N=total queries, Nt=TSCMs, nMSE=MSE/Var, med=median nMSE, Nv/Nx=direction valid/excluded |t|<{DIR_ACC_EPS})")

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

    # JSON export
    json_path = args.json_out
    if json_path is None:
        ckpt_dir = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(repo_root, 'results', ckpt_dir, 'tscm_eval.json')

    def _json_clean(obj):
        """Recursively convert non-JSON-serializable values to primitives."""
        if isinstance(obj, dict):
            return {str(k): _json_clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    export = {
        'checkpoint': args.checkpoint,
        'n_samples': args.n_samples,
        'T': args.T,
        't_range': args.t_range,
        'max_lag': args.max_lag,
        'multi_step': args.multi_step,
        'query_offsets': args.query_offsets,
        'dir_acc_eps': DIR_ACC_EPS,
        'bootstrap_n': args.bootstrap_n,
        'results': _json_clean(all_results),
    }
    # Strip the heavy 'output' (model logits) from per_tscm records if present.
    for structure_name, r in export['results'].items():
        if isinstance(r, dict):
            r.pop('coverage', None)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\nResults written to {json_path}")


if __name__ == "__main__":
    main()
