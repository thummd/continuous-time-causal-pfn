#!/usr/bin/env python
"""Prior-based identifiability evaluation.

Evaluates baselines on samples drawn from ExtendedCausalTimePrior, mirroring
training conditions. Intervention target, query target, and intervention time
are chosen by the prior rather than hard-coded to a fixed TSCM structure.

Usage
-----
python prior_eval.py eval_configs/back_door_backdoor_tabpfn.yaml
python prior_eval.py eval_configs/verify.yaml
"""

import argparse
import json
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import yaml

import numpy as np
import torch
from tqdm import tqdm

from dotime.prior.extended_prior import ExtendedCausalTimePrior
from dotime.data.normalization import normalize_batch
from dotime.eval.metrics import compute_rmse, compute_mae, compute_r2, compute_nmse
from baselines import (
    AR1Baseline,
    Chronos2Observational,
    BackDoorTabPFNInterventional,
    BackDoorTabPFNObservational,
    BackDoorTabPFNCausalEffect,
    BackDoorDoTPFNCausalEffect,
    BackDoorObsPFNCausalEffect,
    FrontDoorTabPFNInterventional,
    FrontDoorTabPFNObservational,
    FrontDoorTabPFNCausalEffect,
    ZeroBaseline,
)

# Targets with |value| < DIR_ACC_EPS are excluded from direction accuracy
DIR_ACC_EPS = 0.1

BASELINES = {
    "AR1Baseline": lambda _: AR1Baseline(),
    "ZeroBaseline": lambda _: ZeroBaseline(),
    "Chronos2Observational": Chronos2Observational,
    "BackDoorTabPFNInterventional": BackDoorTabPFNInterventional,
    "BackDoorTabPFNObservational": BackDoorTabPFNObservational,
    "BackDoorTabPFNCausalEffect": BackDoorTabPFNCausalEffect,
    "BackDoorDoTPFNCausalEffect": BackDoorDoTPFNCausalEffect,
    "BackDoorObsPFNCausalEffect": BackDoorObsPFNCausalEffect,
    "FrontDoorTabPFNInterventional": FrontDoorTabPFNInterventional,
    "FrontDoorTabPFNObservational": FrontDoorTabPFNObservational,
    "FrontDoorTabPFNCausalEffect": FrontDoorTabPFNCausalEffect,
}


def _direction_accuracy(preds: torch.Tensor, targets: torch.Tensor, eps: float = DIR_ACC_EPS):
    """Sign accuracy excluding near-zero targets (|target| < eps)."""
    if preds.numel() == 0:
        return {'accuracy': float('nan'), 'n_valid': 0, 'n_excluded': 0}
    mask = targets.abs() >= eps
    n_valid = int(mask.sum().item())
    n_excluded = preds.numel() - n_valid
    if n_valid == 0:
        return {'accuracy': float('nan'), 'n_valid': 0, 'n_excluded': n_excluded}
    acc = (preds[mask].sign() == targets[mask].sign()).float().mean().item()
    return {'accuracy': acc, 'n_valid': n_valid, 'n_excluded': n_excluded}


def _bootstrap_ci(values, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap (mean, std, lo, hi) from a list of per-sample scalars."""
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    if arr.size == 1:
        v = float(arr[0])
        return v, 0.0, v, v
    rng = np.random.default_rng(seed)
    boot_means = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
    return (
        float(arr.mean()),
        float(arr.std(ddof=1)),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def compute_confounding_strength(X_obs, X_int, hidden_vars, int_target, outcome_var, N):
    """Estimate confounding between A and Y from observational data."""
    observed = [i for i in range(N) if i not in hidden_vars]
    nan_result = {'corr_A_Y': float('nan'), 'mean_confounder_corr': float('nan')}

    if int_target not in observed or outcome_var not in observed:
        return nan_result

    X = X_obs[:, observed].float().numpy()
    if X.std(axis=0).min() < 1e-8:
        return nan_result

    corr = np.corrcoef(X, rowvar=False)
    obs_A = observed.index(int_target)
    obs_Y = observed.index(outcome_var)

    A_obs = X_obs[:, int_target]
    A_int = X_int[-1, int_target]

    covariate_indices = [j for j, i in enumerate(observed) if i != int_target and i != outcome_var]
    mean_confounder_corr = float(np.mean([
        (abs(corr[j, obs_A]) + abs(corr[j, obs_Y])) / 2.0
        for j in covariate_indices
    ])) if covariate_indices else 0.0

    return {
        'corr_A_Y': float(abs(corr[obs_A, obs_Y])),
        'mean_confounder_corr': mean_confounder_corr,
        'A_stds_away': float((A_int - A_obs.mean()) / (A_obs.std() + 1e-9)),
        'A_positivity': bool(A_int > A_obs.min() and A_int < A_obs.max()),
    }


def _aggregate(records, all_confounding):
    """Aggregate metrics across a list of query records."""
    if not records:
        return {'total': 0}

    preds = torch.tensor([r['pred'] for r in records])
    targets = torch.tensor([r['target'] for r in records])
    dir_acc = _direction_accuracy(preds, targets)

    valid_conf = [c for c in all_confounding if not np.isnan(c['corr_A_Y'])]
    mean_confounding = np.mean([c['corr_A_Y'] for c in valid_conf]) if valid_conf else float('nan')

    return {
        'total': len(records),
        'rmse': compute_rmse(preds, targets),
        'mae': compute_mae(preds, targets),
        'nmse': compute_nmse(preds, targets),
        'r2': compute_r2(preds, targets),
        'direction_accuracy': dir_acc['accuracy'],
        'direction_n_valid': dir_acc['n_valid'],
        'direction_n_excluded': dir_acc['n_excluded'],
        'mean_confounding': float(mean_confounding),
    }


def verify_data_generation(n_samples: int = 10, T: int = 200,
                           tscm_structure: str = None, n_max_prior: int = 10,
                           burn_in: int = 50):
    """Sanity-check the prior: print valid/divergent counts and mean |X_int - X_obs|."""
    label = tscm_structure or "full CausalTimePrior (random)"
    print("=" * 60)
    print(f"Prior Data Generation Verification — {label}")
    print("=" * 60)

    prior = ExtendedCausalTimePrior(
        n_max=41, n_max_prior=n_max_prior, t_range=(T, T),
        burn_in=burn_in, tscm_structure=tscm_structure,
        intervention_source="observed_uniform", seed=42,
    )

    n_valid, n_divergent, diffs = 0, 0, []
    for i in range(n_samples):
        sample = prior.generate_sample(T=T)
        X_obs, X_int = sample['X_obs'], sample['X_int']
        if X_obs.abs().max() > 900 or X_int.abs().max() > 900:
            n_divergent += 1
            continue
        if not torch.isfinite(X_obs).all():
            print(f"  WARNING: non-finite X_obs (sample {i})")
            continue
        if not torch.isfinite(X_int).all():
            print(f"  WARNING: non-finite X_int (sample {i})")
            continue
        diffs.append((X_int - X_obs).abs().mean().item())
        n_valid += 1

    print(f"  Valid: {n_valid}/{n_samples}  Divergent: {n_divergent}")
    if diffs:
        mean_diff = np.mean(diffs)
        print(f"  Mean |X_int - X_obs|: {mean_diff:.4f}")
        if mean_diff < 1e-6:
            print("  WARNING: interventional data identical to observational!")
    print(f"  X_obs shape: {sample['X_obs'].shape}")
    print("=" * 60)


def evaluate_prior(
    model,
    n_samples: int = 50,
    T: int = 1000,
    n_max: int = 41,
    device: str = "cpu",
    query_offsets: list = None,
    tscm_structure: str = None,
    n_max_prior: int = 10,
    burn_in: int = 50,
):
    """Evaluate a baseline on samples from ExtendedCausalTimePrior.

    Parameters
    ----------
    model : SinglePointTimeSeriesBaseline
    n_samples : number of valid prior samples to collect
    T : time-series length
    n_max : max variables passed to ExtendedCausalTimePrior (model capacity)
    device : torch device string
    query_offsets : list of time offsets after intervention to query (0 = same step)
    tscm_structure : restrict prior to this named structure (e.g. 'back_door'),
                     or None for the full random CTP prior
    n_max_prior : max variables in the CTP prior graph
    burn_in : SCM simulation burn-in steps
    """
    if query_offsets is None:
        query_offsets = [0]

    prior = ExtendedCausalTimePrior(
        n_max=n_max, n_max_prior=n_max_prior, t_range=(T, T),
        burn_in=burn_in, tscm_structure=tscm_structure,
        intervention_source="observed_uniform", seed=42,
    )

    per_sample, all_confounding = [], []
    per_offset_records = {o: [] for o in query_offsets}
    all_records = []

    sample_idx = 0
    attempts = 0

    with tqdm(total=n_samples, desc="Evaluating") as pbar:
        while sample_idx < n_samples and attempts < n_samples * 5:
            attempts += 1

            # Per-attempt deterministic seed so divergent samples are skipped cleanly
            # and all baselines see the same accepted dataset (same attempt order).
            seed = 42 + attempts
            torch.manual_seed(seed)
            np.random.seed(seed)
            if hasattr(prior.prior, 'gen'):
                prior.prior.gen.manual_seed(seed)
            prior.rng = np.random.RandomState(seed)

            sample = prior.generate_sample(T=T)
            X_obs, X_int = sample['X_obs'], sample['X_int']

            if X_obs.abs().max() > 900 or X_int.abs().max() > 900:
                continue
            if X_obs.abs().max() < 1e-6 or X_int.abs().max() < 1e-6:
                continue

            N = int(sample['num_vars'].item())
            int_target = int(sample['intervention_target'].item())
            q_idx = int(sample['query_target'].item())
            int_time = int(round(float(sample['intervention_time_start'].item()) * T))
            variable_mask = sample['variable_mask']
            hidden_vars = [i for i in range(N) if variable_mask[i] < 0.5]

            batch_1 = {k: v.unsqueeze(0) for k, v in sample.items() if isinstance(v, torch.Tensor)}
            batch_1 = normalize_batch(batch_1)
            X_obs_norm = batch_1['X_obs_norm'][0]  # (T, n_max)
            means = batch_1['_norm_means'][0]       # (n_max,)
            stds = batch_1['_norm_stds'][0]         # (n_max,)

            conf = compute_confounding_strength(
                X_obs[:, :N], X_int[:, :N], hidden_vars, int_target, q_idx, N
            )
            all_confounding.append(conf)

            sample_records = []
            for offset in query_offsets:
                query_time_idx = min(int_time + offset, T - 1)

                if offset == 0:
                    expected_t = float(sample['intervention_time_start'].item())
                    actual_t = query_time_idx / T
                    assert abs(actual_t - expected_t) < 2.0 / T, (
                        f"query_time {actual_t:.8f} != intervention_time_start {expected_t:.8f} "
                        f"(diff={abs(actual_t - expected_t):.2e}, T={T})"
                    )

                y_int = sample['X_int'][query_time_idx, q_idx].item()
                y_true_norm = float(max(-10.0, min(10.0,
                    (y_int - means[q_idx].item()) / stds[q_idx].item())))

                batch = {
                    'X_obs_norm': X_obs_norm.unsqueeze(0).to(device),
                    'variable_mask': variable_mask.unsqueeze(0).to(device),
                    'intervention_target': torch.tensor([int_target], dtype=torch.long, device=device),
                    'intervention_type': sample['intervention_type'].unsqueeze(0).to(device),
                    'intervention_value': sample['intervention_value'].unsqueeze(0).to(device),
                    'intervention_time_start': sample['intervention_time_start'].unsqueeze(0).to(device),
                    'intervention_time_end': sample['intervention_time_end'].unsqueeze(0).to(device),
                    'query_target': torch.tensor([q_idx], dtype=torch.long, device=device),
                    'query_time': torch.tensor([query_time_idx / T], dtype=torch.float32, device=device),
                }

                pred_val, output = model.forward(batch)
                rec = {'pred': pred_val, 'target': y_true_norm, 'output': output, 'offset': offset}
                per_offset_records[offset].append(rec)
                all_records.append(rec)
                sample_records.append(rec)

            if sample_records:
                p = torch.tensor([r['pred'] for r in sample_records])
                t = torch.tensor([r['target'] for r in sample_records])
                dir_acc = _direction_accuracy(p, t)
                per_sample.append({
                    'sample_idx': sample_idx,
                    'n_queries': len(sample_records),
                    'rmse': compute_rmse(p, t),
                    'mae': compute_mae(p, t),
                    'nmse': compute_nmse(p, t),
                    'r2': compute_r2(p, t),
                    'direction_accuracy': dir_acc['accuracy'],
                    'direction_n_valid': dir_acc['n_valid'],
                    'direction_n_excluded': dir_acc['n_excluded'],
                    'confounding': conf,
                })
                sample_idx += 1
                pbar.update(1)

    if not all_records:
        return {'total': 0}

    results = _aggregate(all_records, all_confounding)
    results['n_samples'] = len(per_sample)
    results['per_sample'] = per_sample
    results['per_offset'] = {
        offset: _aggregate(per_offset_records[offset], all_confounding)
        for offset in query_offsets
    }

    # Quantile calibration (only if model has a bar distribution head)
    if hasattr(model, 'bar_head') and model.bar_head is not None and model.bar_head.bar_dist is not None:
        from dotime.eval.metrics import compute_quantile_calibration, compute_pinball_metric
        tau_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        borders = model.bar_head.borders.cpu()
        all_logits = torch.cat([r['output'] for r in all_records])
        targets_t = torch.tensor([r['target'] for r in all_records])
        cal = compute_quantile_calibration(all_logits, borders, targets_t, tau_levels)
        results['calibration_error'] = cal['calibration_error']
        results['pinball_loss'] = compute_pinball_metric(all_logits, borders, targets_t, tau_levels)
        results['coverage'] = {k: v for k, v in cal.items() if k.startswith('coverage_')}

    return results


def main():
    parser = argparse.ArgumentParser(description="Prior-based identifiability evaluation")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    cfg_path = parser.parse_args().config

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # `baseline` may be a single string or a list of strings
    baseline_names = cfg["baseline"]
    if isinstance(baseline_names, str):
        baseline_names = [baseline_names]

    # Optional with defaults
    case_study    = cfg.get("case_study", None)
    device        = cfg.get("device", "cpu")
    n_samples     = cfg.get("n_samples", 50)
    verify_only   = cfg.get("verify_only", False)
    query_offsets = cfg.get("query_offsets", [0])
    bootstrap_n   = cfg.get("bootstrap_n", 1000)
    T             = cfg.get("T", 1000)
    n_max_prior   = cfg.get("n_max_prior", 10)
    burn_in       = cfg.get("burn_in", 50)

    if any(o > 0 for o in query_offsets):
        warnings.warn(
            f"query_offsets > 0 {[o for o in query_offsets if o > 0]}: "
            "predicting p(y_{{t+k}} | do(a_t)) rather than p(y_t | do(a_t)); "
            "correct identification requires a multi-step rollout.",
            UserWarning, stacklevel=2,
        )

    if verify_only:
        verify_data_generation(
            n_samples=n_samples, T=T,
            tscm_structure=case_study,
            n_max_prior=n_max_prior, burn_in=burn_in,
        )
        return

    label = case_study or "ctp_prior"

    # Output directory created once for the whole run
    now = datetime.now()
    cfg_stem = os.path.splitext(os.path.basename(cfg_path))[0]
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results',
        now.strftime('%Y-%m-%d'),
        f'{cfg_stem}_{now.strftime("%H-%M-%S")}',
    )
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(out_dir, os.path.basename(cfg_path)))

    def _json_clean(obj):
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

    all_results = {}   # baseline_name -> results dict
    all_boots   = {}   # baseline_name -> boot dict

    for baseline_name in baseline_names:
        print("\n" + "=" * 80)
        print(f"  Baseline:  {baseline_name}")
        print(f"  Structure: {label}  |  N samples: {n_samples}  |  T={T}  |  offsets={query_offsets}")
        print("=" * 80)

        try:
            model = BASELINES[baseline_name](device)
        except Exception as e:
            print(f"  [SKIP] Could not load {baseline_name}: {e}")
            continue

        try:
            results = evaluate_prior(
                model,
                n_samples=n_samples,
                T=T,
                device=device,
                query_offsets=query_offsets,
                tscm_structure=case_study,
                n_max_prior=n_max_prior,
                burn_in=burn_in,
            )
        except Exception as e:
            print(f"  [SKIP] evaluate_prior failed for {baseline_name}: {e}")
            continue

        if results['total'] == 0:
            print("  No valid queries generated.")
            continue

        # Bootstrap CIs
        per_sample = results.get('per_sample', [])
        boot = {}
        if per_sample:
            for key in ('rmse', 'mae', 'nmse', 'r2', 'direction_accuracy'):
                vals = [s[key] for s in per_sample if s.get(key) is not None]
                mean, std, lo, hi = _bootstrap_ci(vals, n=bootstrap_n)
                boot[key] = {'mean': mean, 'std': std, 'ci_low': lo, 'ci_high': hi}
        results['bootstrap'] = boot
        all_results[baseline_name] = results
        all_boots[baseline_name]   = boot

        def _fmt(key, pct=False):
            b = boot.get(key, {})
            m, s, lo, hi = (b.get('mean', float('nan')), b.get('std', float('nan')),
                            b.get('ci_low', float('nan')), b.get('ci_high', float('nan')))
            fmt = ".2" if pct else ".4"
            return f"{m:{fmt}} ± {s:{fmt}} ({lo:{fmt}}, {hi:{fmt}})"

        nv = results.get('direction_n_valid', 0)
        nx = results.get('direction_n_excluded', 0)
        print(f"\n  Queries: {results['total']} across {results['n_samples']} samples")
        print(f"  RMSE: {_fmt('rmse')} | MAE: {_fmt('mae')} | NMSE: {_fmt('nmse')} | R2: {_fmt('r2')}")
        print(f"  Direction accuracy: {_fmt('direction_accuracy', pct=True)} "
              f"(n_valid={nv}, excluded={nx}, |t|<{DIR_ACC_EPS})")
        print(f"  Mean confounding (corr A-Y): {results['mean_confounding']:.4f}")

        if per_sample:
            per_rmse = [s['rmse'] for s in per_sample]
            print(f"  Per-sample RMSE: min={min(per_rmse):.4f}  "
                  f"median={np.median(per_rmse):.4f}  max={max(per_rmse):.4f}")

    if not all_results:
        print("\nNo baselines produced results.")
        return

    # Combined summary table
    W = 110
    print("\n\n" + "=" * W)
    print("SUMMARY")
    print(f"{'Baseline':<38} {'N':>4} {'Ns':>4} "
          f"{'RMSE':>8} {'±std':>7} {'nMSE':>6} "
          f"{'Dir':>7} {'±std':>7} {'Nv':>4} {'Nx':>4} {'Confound':>8}")
    print("-" * W)
    for bname, r in all_results.items():
        b = all_boots[bname]
        rmse_m = b.get('rmse', {}).get('mean', r['rmse'])
        rmse_s = b.get('rmse', {}).get('std', 0.0)
        dir_m  = b.get('direction_accuracy', {}).get('mean', r['direction_accuracy'])
        dir_s  = b.get('direction_accuracy', {}).get('std', 0.0)
        nv = r.get('direction_n_valid', 0)
        nx = r.get('direction_n_excluded', 0)
        print(f"{bname:<38} {r['total']:>4} {r['n_samples']:>4} "
              f"{rmse_m:>8.4f} {rmse_s:>7.4f} {r.get('nmse', float('nan')):>6.3f} "
              f"{dir_m:>6.1%} {dir_s:>6.1%} "
              f"{nv:>4} {nx:>4} {r['mean_confounding']:>8.4f}")
    print("=" * W)
    print(f"(N=total queries, Ns=samples, Nv=direction-valid, Nx=excluded |target|<{DIR_ACC_EPS})")

    # Save combined results
    export = {
        'tscm_structure': case_study,
        'n_samples': n_samples,
        'T': T,
        'query_offsets': query_offsets,
        'dir_acc_eps': DIR_ACC_EPS,
        'bootstrap_n': bootstrap_n,
        'results': _json_clean({
            bname: {**r, 'coverage': None}
            for bname, r in all_results.items()
        }),
    }
    for r in export['results'].values():
        if isinstance(r, dict):
            r.pop('coverage', None)

    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\nResults written to {out_dir}/")

    # Consolidate: generate plots + markdown summary
    try:
        from consolidate_results import (
            load_results, build_per_sample_df, build_summary_df,
            plot_bar_metrics, plot_critical_difference, build_markdown_tables,
        )
        plots_dir = Path(out_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        _data       = load_results(Path(json_path))
        _per_sample = build_per_sample_df(_data)
        _summary    = build_summary_df(_data)
        print("\nGenerating plots and summary...")
        plot_bar_metrics(_per_sample, plots_dir)
        plot_critical_difference(_per_sample, plots_dir, metric="nmse")
        plot_critical_difference(_per_sample, plots_dir, metric="direction_accuracy")
        _md = (f"# Results: {case_study or 'ctp_prior'}  "
               f"(n={n_samples}, T={T})\n\n"
               + build_markdown_tables(_summary))
        (plots_dir / "summary.md").write_text(_md + "\n")
        print(f"Plots and summary written to {plots_dir}/")
    except Exception as e:
        print(f"[WARNING] Consolidation failed: {e}")


if __name__ == "__main__":
    main()
