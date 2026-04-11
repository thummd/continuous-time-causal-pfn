#!/usr/bin/env python
"""Per-structure diagnostic plots for a Do-Over-Time-PFN checkpoint.

For each TSCM structure:
  - Sample a few SCMs.
  - Run observational and interventional trajectories.
  - Overlay the model's prediction at the intervention time on top of the
    ground-truth observational / interventional traces.

This script is read-only w.r.t. the checkpoint and can be run on any existing
model to diagnose per-structure failure modes. It complements the aggregate
metrics reported by ``scripts/tscm_identifiability.py``.

Usage
-----
    python scripts/plot_tscm_diagnostics.py \
        --checkpoint checkpoints/v5_combined/do_over_time_pfn_best.pt \
        --device cuda:0 --n-samples-per-structure 3
"""

import argparse
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dotime.prior.tscm_sampler import TSCMSampler, TSCMStructure
from dotime.prior.extended_prior import pad_to_max_nodes


def _load_model(checkpoint: str, device: str):
    from dotime.model.do_over_time_pfn import DoOverTimePFN

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    model = DoOverTimePFN(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    head_type = ckpt.get('head_type', 'bar')
    if head_type == 'bar' and ckpt.get('borders') is not None:
        from pfns.model.bar_distribution import FullSupportBarDistribution
        borders = ckpt['borders']
        bar_dist = FullSupportBarDistribution(borders)
        model.bar_head.set_bar_distribution(bar_dist, borders)

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict(
    model, X_norm, variable_mask, int_target, int_time, q_idx, int_value_norm, T, device,
):
    batch = {
        'X_obs_norm': X_norm.unsqueeze(0).to(device),
        'variable_mask': variable_mask.unsqueeze(0).to(device),
        'intervention_target': torch.tensor([int_target], dtype=torch.long, device=device),
        'intervention_type': torch.tensor([0], dtype=torch.long, device=device),
        'intervention_value': torch.tensor([int_value_norm], dtype=torch.float32, device=device),
        'intervention_time_start': torch.tensor([int_time / T], dtype=torch.float32, device=device),
        'intervention_time_end': torch.tensor([int_time / T], dtype=torch.float32, device=device),
        'query_target': torch.tensor([q_idx], dtype=torch.long, device=device),
        'query_time': torch.tensor([int_time / T], dtype=torch.float32, device=device),
    }
    output = model(batch)
    pred = model.head.predict_mean(output)
    return float(pred.cpu().item())


def plot_structure(
    model,
    structure: TSCMStructure,
    out_dir: str,
    n_samples: int = 3,
    T: int = 80,
    n_max: int = 12,
    device: str = "cpu",
    max_lag: int = 1,
    seed: int = 0,
):
    """Plot traces (obs / int / prediction) for a single structure."""
    from causal_time_prior.interventions import InterventionSpec, InterventionType

    sampler = TSCMSampler(structure, max_lag=max_lag)
    hidden_vars = sampler.get_hidden_vars()
    gen = torch.Generator().manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    figures_saved = []

    plotted = 0
    attempts = 0
    while plotted < n_samples and attempts < n_samples * 10:
        attempts += 1
        scm = sampler.sample(generator=gen)
        N = len(scm._topo)

        X_obs = scm.sample_observational(T=T, burn_in=30, generator=gen)
        if X_obs.abs().max() > 900:
            continue

        valid_targets = [i for i in range(N) if i not in hidden_vars]
        if not valid_targets:
            continue
        int_target = valid_targets[0]

        int_time = max(0, T - 10)
        int_value = float(torch.randn(1, generator=gen).item() * 2.0)
        intervention = InterventionSpec(
            targets=[int_target], times=[int_time],
            intervention_type=InterventionType.HARD, values=int_value,
        )
        X_int = scm.sample_interventional(T=T, intervention=intervention, burn_in=30, generator=gen)
        if X_int.abs().max() > 900:
            continue
        # Skip degenerate (all-zero) runs.
        if X_obs.abs().max() < 1e-6 or X_int.abs().max() < 1e-6:
            continue

        # Causal-mask X_obs for the model input
        X_obs_masked = X_obs.clone()
        X_obs_masked[int_time:] = 0.0
        X_padded = pad_to_max_nodes(X_obs_masked, n_max)
        variable_mask = torch.zeros(n_max)
        for j in range(N):
            if j not in hidden_vars:
                variable_mask[j] = 1.0

        means = X_padded.mean(dim=0)
        stds = X_padded.std(dim=0) + 1e-2
        stds[variable_mask == 0] = 1.0
        X_norm = (X_padded - means.unsqueeze(0)) / stds.unsqueeze(0)
        X_norm = X_norm * variable_mask.unsqueeze(0)

        int_value_norm = (int_value - means[int_target].item()) / stds[int_target].item()

        # Plot each non-hidden, non-intervention variable in its own subplot.
        query_vars = [q for q in range(N) if q not in hidden_vars and q != int_target]
        if not query_vars:
            continue
        fig, axes = plt.subplots(
            len(query_vars), 1, figsize=(10, 2.8 * len(query_vars)), sharex=True, squeeze=False,
        )
        axes = axes[:, 0]
        t_axis = np.arange(T)

        for ax, q_idx in zip(axes, query_vars):
            obs_trace = X_obs[:, q_idx].numpy()
            int_trace = X_int[:, q_idx].numpy()
            ax.plot(t_axis, obs_trace, label=f"obs ({scm._topo[q_idx]})", color="#1f77b4", alpha=0.85)
            ax.plot(t_axis, int_trace, label=f"int ({scm._topo[q_idx]})", color="#d62728", alpha=0.85)
            ax.axvline(int_time, color="gray", linestyle="--", linewidth=1,
                       label=f"do({scm._topo[int_target]}={int_value:+.2f}) @ t={int_time}")

            # Model prediction at the intervention time (in normalized space,
            # rescaled back to raw for overlay).
            try:
                pred_norm = _predict(
                    model, X_norm, variable_mask, int_target, int_time, q_idx,
                    int_value_norm, T, device,
                )
                pred_raw = pred_norm * stds[q_idx].item() + means[q_idx].item()
                target_raw = X_int[int_time, q_idx].item()
                ax.scatter([int_time], [pred_raw], marker="x", s=80, color="black",
                           zorder=5,
                           label=f"pred={pred_raw:+.2f}")
                ax.scatter([int_time], [target_raw], marker="o", s=50, color="#2ca02c",
                           zorder=5,
                           label=f"target={target_raw:+.2f}")
            except Exception as exc:
                ax.text(0.02, 0.95, f"predict failed: {exc}",
                        transform=ax.transAxes, color="red", fontsize=7)

            ax.set_ylabel(f"var {q_idx}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=7, ncol=2)

        axes[-1].set_xlabel("time step")
        fig.suptitle(f"{structure.value} — sample {plotted+1}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = os.path.join(out_dir, f"{structure.value}_{plotted+1}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        figures_saved.append(out_path)
        plotted += 1

    return figures_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples-per-structure", type=int, default=3)
    parser.add_argument("--T", type=int, default=80)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. Defaults to figures/<ckpt_parent>_diagnostics.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.out_dir is None:
        ckpt_dir = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.out_dir = os.path.join(repo_root, "figures", f"{ckpt_dir}_diagnostics")

    model = _load_model(args.checkpoint, args.device)
    print(f"Loaded model from {args.checkpoint}")
    print(f"Writing diagnostic plots to {args.out_dir}")

    for structure in TSCMStructure:
        paths = plot_structure(
            model, structure, out_dir=args.out_dir,
            n_samples=args.n_samples_per_structure, T=args.T, n_max=args.n_max,
            device=args.device, max_lag=args.max_lag, seed=args.seed,
        )
        print(f"  {structure.value}: {len(paths)} plot(s)")


if __name__ == "__main__":
    main()
