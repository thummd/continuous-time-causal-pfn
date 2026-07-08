#!/usr/bin/env python
"""Pinball-weight comparison experiment.

Trains two models side-by-side:
  A) pinball_weight = 0.0  (bar distribution loss only)
  B) pinball_weight > 0    (bar distribution + auxiliary pinball loss)

Then evaluates both on quantile calibration, pinball metric, RMSE, and
direction accuracy across all TSCM identifiability structures.

Usage:
  python scripts/pinball_comparison.py --device cuda --total-steps 5000
  python scripts/pinball_comparison.py --device cpu --total-steps 1000  # quick test
"""

import argparse
import os
import sys
import time
import json
import torch
import yaml
import numpy as np


from dotime.training.trainer import train
from dotime.model.do_over_time_pfn import DoOverTimePFN
from dotime.model.bar_head import calibrate_bar_distribution
from dotime.data.temporal_dataloader import TemporalInterventionDataLoader
from dotime.eval.metrics import evaluate_model
from pfns.model.bar_distribution import FullSupportBarDistribution
from scripts.tscm_identifiability import evaluate_structure
from dotime.prior.tscm_sampler import TSCMStructure


def load_checkpoint(ckpt_path, device="cpu"):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = DoOverTimePFN(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    borders = ckpt['borders']
    bar_dist = FullSupportBarDistribution(borders)
    model.bar_head.set_bar_distribution(bar_dist, borders.to(device))
    model = model.to(device)
    model.eval()
    return model, ckpt


def run_comparison(args):
    """Run the A/B pinball-weight comparison."""
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    prior_cfg = config["prior"]
    train_cfg = config["training"]

    # Override for faster experimentation
    embed_size = args.embed_size or model_cfg["embed_size"]
    n_encoder_layers = args.n_encoder_layers or model_cfg["n_encoder_layers"]
    n_buckets = args.n_buckets or model_cfg["n_buckets"]
    total_steps = args.total_steps or train_cfg["total_steps"]
    batch_size = args.batch_size or train_cfg["batch_size"]

    common_kwargs = dict(
        n_max=model_cfg["n_max"],
        embed_size=embed_size,
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=n_encoder_layers,
        n_cross_attn_heads=model_cfg["n_cross_attn_heads"],
        n_buckets=n_buckets,
        encoder_backend=args.backend or "transformer",
        encoder_config=model_cfg.get("encoder"),
        n_max_prior=prior_cfg["n_max"],
        t_range=tuple(prior_cfg["t_range"]),
        burn_in=prior_cfg["burn_in"],
        downstream_prob=prior_cfg["downstream_prob"],
        batch_size=batch_size,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=min(train_cfg["warmup_steps"], total_steps // 5),
        total_steps=total_steps,
        grad_clip=train_cfg["grad_clip"],
        eval_every=min(train_cfg["eval_every"], total_steps // 4),
        bucket_calibration_samples=train_cfg["bucket_calibration_samples"],
        seed=train_cfg["seed"],
        device=args.device,
    )

    tau_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

    # ---- Experiment A: pinball_weight = 0 (baseline) ----
    print("\n" + "#" * 70)
    print("# EXPERIMENT A: Bar distribution only (pinball_weight=0)")
    print("#" * 70)
    save_dir_a = os.path.join(args.save_dir, "baseline_pw0")
    t0 = time.time()
    train(save_dir=save_dir_a, pinball_weight=0.0, pinball_quantiles=None, **common_kwargs)
    time_a = time.time() - t0

    # ---- Experiment B: pinball_weight > 0 ----
    print("\n" + "#" * 70)
    print(f"# EXPERIMENT B: Bar + pinball loss (pinball_weight={args.pinball_weight})")
    print("#" * 70)
    save_dir_b = os.path.join(args.save_dir, f"pinball_pw{args.pinball_weight}")
    t0 = time.time()
    train(
        save_dir=save_dir_b,
        pinball_weight=args.pinball_weight,
        pinball_quantiles=tau_levels,
        **common_kwargs,
    )
    time_b = time.time() - t0

    # ---- Evaluation ----
    print("\n" + "#" * 70)
    print("# EVALUATION")
    print("#" * 70)

    ckpt_a_path = os.path.join(save_dir_a, "do_over_time_pfn_best.pt")
    ckpt_b_path = os.path.join(save_dir_b, "do_over_time_pfn_best.pt")

    if not os.path.exists(ckpt_a_path) or not os.path.exists(ckpt_b_path):
        print("ERROR: One or both checkpoints not found. Training may have failed.")
        return

    model_a, ckpt_a = load_checkpoint(ckpt_a_path, args.device)
    model_b, ckpt_b = load_checkpoint(ckpt_b_path, args.device)

    # --- Eval on held-out prior data ---
    print("\n--- Held-out prior data evaluation ---")
    eval_loader = TemporalInterventionDataLoader(
        num_steps=50, batch_size=batch_size,
        n_max=model_cfg["n_max"], n_max_prior=prior_cfg["n_max"],
        t_range=tuple(prior_cfg["t_range"]), burn_in=prior_cfg["burn_in"],
        downstream_prob=prior_cfg["downstream_prob"],
        seed=train_cfg["seed"] + 9999,
        device=args.device,
    )

    tau_tensor = tau_levels
    metrics_a = evaluate_model(model_a, eval_loader, args.device, tau_levels=tau_tensor)
    metrics_b = evaluate_model(model_b, eval_loader, args.device, tau_levels=tau_tensor)

    print(f"\n{'Metric':<25} {'Baseline (pw=0)':>15} {'Pinball (pw={pw})':>15}".format(
        pw=args.pinball_weight))
    print("-" * 60)
    for key in ['loss', 'rmse', 'mae', 'pinball_loss', 'calibration_error']:
        if key in metrics_a and key in metrics_b:
            va, vb = metrics_a[key], metrics_b[key]
            better = "<" if vb < va else ">" if vb > va else "="
            print(f"  {key:<23} {va:>15.4f} {vb:>14.4f} {better}")

    # Coverage table
    cov_keys = [k for k in metrics_a if k.startswith("coverage_")]
    if cov_keys:
        print(f"\n{'Quantile':<25} {'Ideal':>8} {'Baseline':>10} {'Pinball':>10}")
        print("-" * 58)
        for k in sorted(cov_keys):
            tau = float(k.split("tau")[1])
            ca = metrics_a[k]
            cb = metrics_b[k]
            err_a = abs(ca - tau)
            err_b = abs(cb - tau)
            marker = " *" if err_b < err_a else ""
            print(f"  {k:<23} {tau:>8.2f} {ca:>10.3f} {cb:>10.3f}{marker}")

    # --- TSCM identifiability case studies ---
    print("\n--- TSCM Identifiability Case Studies ---")

    results_all = {}
    for structure in TSCMStructure:
        name = structure.value
        print(f"\n  {name}:")
        res_a = evaluate_structure(model_a, structure, n_samples=args.n_tscm_samples,
                                   device=args.device)
        res_b = evaluate_structure(model_b, structure, n_samples=args.n_tscm_samples,
                                   device=args.device)
        results_all[name] = {'baseline': res_a, 'pinball': res_b}

        if res_a['total'] == 0 and res_b['total'] == 0:
            print("    No valid queries")
            continue

        for label, res in [("Baseline", res_a), ("Pinball ", res_b)]:
            if res['total'] == 0:
                print(f"    {label}: no valid queries")
                continue
            cal = res.get('calibration_error', float('nan'))
            pb = res.get('pinball_loss', float('nan'))
            print(f"    {label}: RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  "
                  f"Dir.Acc={res['direction_accuracy']:.1%}  Cal.Err={cal:.4f}  "
                  f"Pinball={pb:.4f}")

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("SUMMARY: TSCM Identifiability")
    print("=" * 90)
    print(f"{'Structure':<25} {'Model':<10} {'N':>5} {'RMSE':>8} {'MAE':>8} "
          f"{'Dir.Acc':>8} {'Cal.Err':>8} {'Pinball':>8}")
    print("-" * 90)
    for name, res_pair in results_all.items():
        for label, key in [("pw=0", "baseline"), (f"pw={args.pinball_weight}", "pinball")]:
            r = res_pair[key]
            if r['total'] == 0:
                print(f"{name:<25} {label:<10} {'—':>5}")
                continue
            cal = r.get('calibration_error', float('nan'))
            pb = r.get('pinball_loss', float('nan'))
            print(f"{name:<25} {label:<10} {r['total']:>5} {r['rmse']:>8.4f} {r['mae']:>8.4f} "
                  f"{r['direction_accuracy']:>7.1%} {cal:>8.4f} {pb:>8.4f}")
        print()

    print(f"\nTraining time: Baseline={time_a:.0f}s, Pinball={time_b:.0f}s")
    print(f"Checkpoints saved in: {args.save_dir}/")

    # Save results as JSON
    json_path = os.path.join(args.save_dir, "comparison_results.json")
    json_results = {
        'config': {
            'total_steps': total_steps,
            'embed_size': embed_size,
            'n_encoder_layers': n_encoder_layers,
            'n_buckets': n_buckets,
            'batch_size': batch_size,
            'pinball_weight': args.pinball_weight,
        },
        'held_out_eval': {
            'baseline': {k: v for k, v in metrics_a.items()},
            'pinball': {k: v for k, v in metrics_b.items()},
        },
        'tscm': {
            name: {
                'baseline': {k: v for k, v in rp['baseline'].items() if k != 'coverage'},
                'pinball': {k: v for k, v in rp['pinball'].items() if k != 'coverage'},
            }
            for name, rp in results_all.items()
        },
        'training_time': {'baseline': time_a, 'pinball': time_b},
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Pinball-weight comparison experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=5000,
                        help="Training steps per experiment")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embed-size", type=int, default=None)
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--n-buckets", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--pinball-weight", type=float, default=0.1,
                        help="Pinball weight for experiment B")
    parser.add_argument("--n-tscm-samples", type=int, default=50,
                        help="TSCM evaluation samples per structure")
    parser.add_argument("--save-dir", type=str, default="checkpoints/pinball_comparison")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")
    print(f"Pinball weight for experiment B: {args.pinball_weight}")
    run_comparison(args)


if __name__ == "__main__":
    main()
