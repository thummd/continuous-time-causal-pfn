#!/usr/bin/env python
"""Generate data summary and stationarity report for the training prior.

Draws a sample of trajectories from ``ExtendedCausalTimePrior`` and reports
per-variable distributional statistics (mean, std, min, max, lag-1
autocorrelation), optional ADF / KPSS stationarity tests (if statsmodels is
installed), and simple per-variable histograms + a few example traces.

Usage
-----
    python scripts/data_summary.py --n-samples 500 --out-dir figures/data_summary
"""

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotime.prior.extended_prior import ExtendedCausalTimePrior


def _try_import_statsmodels():
    try:
        from statsmodels.tsa.stattools import adfuller, kpss  # noqa: F401
        return True
    except Exception:
        return False


def _autocorr_lag1(x: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation, NaN-safe."""
    if x.size < 2:
        return float('nan')
    x = x - x.mean()
    denom = (x * x).sum()
    if denom < 1e-12:
        return float('nan')
    return float((x[:-1] * x[1:]).sum() / denom)


def _stationarity_tests(x: np.ndarray):
    """Run ADF / KPSS if statsmodels is available. Returns dict of p-values or NaN."""
    res = {'adf_p': float('nan'), 'kpss_p': float('nan')}
    if x.size < 20 or np.allclose(x, x[0]):
        return res
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res['adf_p'] = float(adfuller(x, autolag='AIC')[1])
            except Exception:
                pass
            try:
                res['kpss_p'] = float(kpss(x, regression='c', nlags='auto')[1])
            except Exception:
                pass
    except ImportError:
        pass
    return res


def collect_stats(
    n_samples: int, T_fixed: int, seed: int, n_max_prior: int,
    intervention_source: str = "prior",
):
    """Generate n_samples trajectories and return stats + raw arrays for plotting."""
    prior = ExtendedCausalTimePrior(
        n_max=max(n_max_prior + 2, 12),
        n_max_prior=n_max_prior,
        t_range=(T_fixed, T_fixed),
        seed=seed,
        intervention_source=intervention_source,
    )

    per_sample = []
    for i in range(n_samples):
        b = prior.generate_sample(T=T_fixed)
        X_obs = b['X_obs'].numpy()          # (T, N_max)
        X_int = b['X_int'].numpy()
        mask = b['variable_mask'].numpy().astype(bool)
        per_sample.append({
            'X_obs': X_obs,
            'X_int': X_int,
            'mask': mask,
            'int_target': int(b['intervention_target'].item()),
            'int_value': float(b['intervention_value'].item()),
            't_start': float(b['intervention_time_start'].item()),
            't_end': float(b['intervention_time_end'].item()),
        })

    # Flatten per-variable time series across all samples (only real vars).
    flat_obs = []
    flat_int = []
    auto1 = []
    stationarity = []
    for s in per_sample:
        for j in np.nonzero(s['mask'])[0]:
            obs = s['X_obs'][:, j]
            intr = s['X_int'][:, j]
            flat_obs.append(obs)
            flat_int.append(intr)
            auto1.append(_autocorr_lag1(obs))
            stationarity.append(_stationarity_tests(obs))

    if not flat_obs:
        return {}, per_sample

    stacked_obs = np.concatenate(flat_obs)
    stacked_int = np.concatenate(flat_int)
    auto1_arr = np.array([a for a in auto1 if not np.isnan(a)])
    adf_ps = np.array([s['adf_p'] for s in stationarity if not np.isnan(s['adf_p'])])
    kpss_ps = np.array([s['kpss_p'] for s in stationarity if not np.isnan(s['kpss_p'])])

    summary = {
        'n_samples': n_samples,
        'T': T_fixed,
        'n_variables_total': len(flat_obs),
        'X_obs': {
            'mean': float(stacked_obs.mean()),
            'std': float(stacked_obs.std()),
            'min': float(stacked_obs.min()),
            'max': float(stacked_obs.max()),
            'abs_median': float(np.median(np.abs(stacked_obs))),
            'pct5': float(np.quantile(stacked_obs, 0.05)),
            'pct95': float(np.quantile(stacked_obs, 0.95)),
        },
        'X_int': {
            'mean': float(stacked_int.mean()),
            'std': float(stacked_int.std()),
            'min': float(stacked_int.min()),
            'max': float(stacked_int.max()),
        },
        'lag1_autocorr': {
            'mean': float(auto1_arr.mean()) if auto1_arr.size else float('nan'),
            'median': float(np.median(auto1_arr)) if auto1_arr.size else float('nan'),
            'pct10': float(np.quantile(auto1_arr, 0.10)) if auto1_arr.size else float('nan'),
            'pct90': float(np.quantile(auto1_arr, 0.90)) if auto1_arr.size else float('nan'),
        },
        'stationarity': {
            'n_adf': int(adf_ps.size),
            'adf_reject_frac': float((adf_ps < 0.05).mean()) if adf_ps.size else float('nan'),
            'adf_median_p': float(np.median(adf_ps)) if adf_ps.size else float('nan'),
            'n_kpss': int(kpss_ps.size),
            'kpss_reject_frac': float((kpss_ps < 0.05).mean()) if kpss_ps.size else float('nan'),
            'kpss_median_p': float(np.median(kpss_ps)) if kpss_ps.size else float('nan'),
        },
        'intervention_source': intervention_source,
    }
    return summary, per_sample


def make_plots(per_sample, summary, out_dir: str, n_example_trajs: int = 6):
    os.makedirs(out_dir, exist_ok=True)

    # Histogram of all observational values
    flat_obs = np.concatenate([s['X_obs'][:, s['mask']].ravel() for s in per_sample if s['mask'].any()])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(flat_obs, bins=80, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title(f"Observational value histogram (n={flat_obs.size})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "histogram_X_obs.png"), dpi=140)
    plt.close(fig)

    # Histogram of intervention values
    int_values = np.array([s['int_value'] for s in per_sample])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(int_values, bins=60, color="#d62728", alpha=0.85)
    ax.set_xlabel("intervention value")
    ax.set_ylabel("count")
    ax.set_title(f"Intervention values (src={summary.get('intervention_source','?')})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "histogram_interventions.png"), dpi=140)
    plt.close(fig)

    # Example trajectories
    n_plot = min(n_example_trajs, len(per_sample))
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.4 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for ax, s in zip(axes, per_sample[:n_plot]):
        real_vars = np.nonzero(s['mask'])[0]
        for j in real_vars:
            ax.plot(s['X_obs'][:, j], alpha=0.75, linewidth=1.1)
        # shade intervention window
        T_here = s['X_obs'].shape[0]
        t_lo = int(s['t_start'] * T_here)
        t_hi = int(s['t_end'] * T_here)
        ax.axvspan(t_lo, max(t_lo + 1, t_hi + 1), alpha=0.15, color="gray",
                   label=f"do(var={s['int_target']}={s['int_value']:+.2f})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time step")
    fig.suptitle("Example observational trajectories", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "example_trajectories.png"), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--T", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-max-prior", type=int, default=10)
    parser.add_argument("--intervention-source", type=str, default="prior",
                        choices=["prior", "observed_discrete", "observed_normal", "observed_uniform"])
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    has_sm = _try_import_statsmodels()
    if not has_sm:
        print("statsmodels not installed — ADF/KPSS will be skipped. "
              "`pip install statsmodels` to enable.")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.out_dir is None:
        args.out_dir = os.path.join(repo_root, "figures", "data_summary")
    if args.json_out is None:
        args.json_out = os.path.join(repo_root, "results", "data_summary.json")

    summary, per_sample = collect_stats(
        n_samples=args.n_samples, T_fixed=args.T, seed=args.seed,
        n_max_prior=args.n_max_prior, intervention_source=args.intervention_source,
    )
    if not summary:
        print("No valid samples generated.")
        return

    make_plots(per_sample, summary, args.out_dir)

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Data summary ({args.n_samples} samples, T={args.T})")
    print("=" * 60)
    print(f"Total variable-trajectories: {summary['n_variables_total']}")
    print(f"X_obs:  mean={summary['X_obs']['mean']:+.3f}  "
          f"std={summary['X_obs']['std']:.3f}  "
          f"range=[{summary['X_obs']['min']:+.2f}, {summary['X_obs']['max']:+.2f}]  "
          f"p5/p95={summary['X_obs']['pct5']:+.2f}/{summary['X_obs']['pct95']:+.2f}")
    print(f"X_int:  mean={summary['X_int']['mean']:+.3f}  "
          f"std={summary['X_int']['std']:.3f}  "
          f"range=[{summary['X_int']['min']:+.2f}, {summary['X_int']['max']:+.2f}]")
    print(f"Lag-1 autocorr: mean={summary['lag1_autocorr']['mean']:.3f}  "
          f"median={summary['lag1_autocorr']['median']:.3f}  "
          f"p10/p90={summary['lag1_autocorr']['pct10']:.3f}/{summary['lag1_autocorr']['pct90']:.3f}")
    s = summary['stationarity']
    if s['n_adf'] > 0:
        print(f"ADF: {s['n_adf']} tested, {s['adf_reject_frac']:.0%} reject H0 "
              f"(median p={s['adf_median_p']:.3f})  "
              f"KPSS: {s['n_kpss']} tested, {s['kpss_reject_frac']:.0%} reject H0 "
              f"(median p={s['kpss_median_p']:.3f})")
    else:
        print("Stationarity tests: skipped (statsmodels not available).")

    print(f"\nWrote plots to {args.out_dir}")
    print(f"Wrote summary JSON to {args.json_out}")


if __name__ == "__main__":
    main()
