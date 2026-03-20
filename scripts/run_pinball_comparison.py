#!/usr/bin/env python
"""Pinball loss comparison experiment.

Trains two models — bar-only (baseline) and bar+pinball (augmented) —
then compares quantile calibration and robustness metrics.
"""

import argparse
import sys
import torch
import yaml

from dotime.training.trainer import train
from dotime.eval.metrics import evaluate_model
from dotime.data.temporal_dataloader import TemporalInterventionDataLoader
from pfns.model.bar_distribution import FullSupportBarDistribution


def load_model_for_eval(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    from dotime.model.do_over_time_pfn import DoOverTimePFN

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = DoOverTimePFN(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    borders = ckpt['borders']
    bar_dist = FullSupportBarDistribution(borders)
    model.bar_head.set_bar_distribution(bar_dist, borders)
    model = model.to(device)
    model.eval()
    return model, ckpt


def run_experiment(args, config, pinball_weight, save_name):
    """Train a model with given pinball weight and return checkpoint path."""
    import os

    model_cfg = config["model"]
    prior_cfg = config["prior"]
    train_cfg = config["training"]

    save_dir = os.path.join(args.save_dir, save_name)

    model = train(
        n_max=model_cfg["n_max"],
        embed_size=args.embed_size or model_cfg["embed_size"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=args.n_encoder_layers or model_cfg["n_encoder_layers"],
        n_cross_attn_heads=model_cfg["n_cross_attn_heads"],
        n_buckets=model_cfg["n_buckets"],
        encoder_backend="transformer",
        encoder_config=model_cfg.get("encoder"),
        n_max_prior=prior_cfg["n_max"],
        t_range=tuple(prior_cfg["t_range"]),
        burn_in=prior_cfg["burn_in"],
        downstream_prob=prior_cfg["downstream_prob"],
        batch_size=args.batch_size or train_cfg["batch_size"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=min(train_cfg["warmup_steps"], args.total_steps // 5),
        total_steps=args.total_steps,
        grad_clip=train_cfg["grad_clip"],
        eval_every=max(args.total_steps // 10, 100),
        bucket_calibration_samples=train_cfg["bucket_calibration_samples"],
        seed=train_cfg["seed"],
        device=args.device,
        save_dir=save_dir,
        pinball_weight=pinball_weight,
        pinball_quantiles=train_cfg.get("pinball_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]),
    )

    ckpt_path = os.path.join(save_dir, "do_over_time_pfn_best.pt")
    return ckpt_path


def evaluate_with_quantiles(model, config, device, tau_levels):
    """Evaluate a model and return metrics including quantile calibration."""
    prior_cfg = config["prior"]
    train_cfg = config["training"]

    eval_loader = TemporalInterventionDataLoader(
        num_steps=50,
        batch_size=train_cfg["batch_size"],
        n_max=config["model"]["n_max"],
        n_max_prior=prior_cfg["n_max"],
        t_range=tuple(prior_cfg["t_range"]),
        burn_in=prior_cfg["burn_in"],
        downstream_prob=prior_cfg["downstream_prob"],
        seed=12345,  # fixed seed for fair comparison
        device=device,
    )

    return evaluate_model(model, eval_loader, device=device, tau_levels=tau_levels)


def print_comparison(baseline_metrics, pinball_metrics, tau_levels):
    """Print a comparison table of metrics."""
    print("\n" + "=" * 72)
    print("COMPARISON: Bar-only (baseline) vs Bar+Pinball (augmented)")
    print("=" * 72)

    header = f"{'Metric':<30} {'Baseline':>18} {'Pinball':>18}"
    print(header)
    print("-" * 72)

    for key in ['loss', 'rmse', 'mae', 'pinball_loss', 'calibration_error']:
        if key in baseline_metrics and key in pinball_metrics:
            print(f"{key:<30} {baseline_metrics[key]:>18.4f} {pinball_metrics[key]:>18.4f}")

    print("\nQuantile Coverage (ideal: coverage = tau):")
    print(f"{'Tau':<10} {'Baseline':>18} {'Pinball':>18} {'Ideal':>10}")
    print("-" * 60)
    for tau in tau_levels:
        key = f"coverage_tau{tau:.2f}"
        if key in baseline_metrics and key in pinball_metrics:
            print(f"{tau:<10.2f} {baseline_metrics[key]:>18.4f} {pinball_metrics[key]:>18.4f} {tau:>10.2f}")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Pinball loss comparison experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embed-size", type=int, default=None)
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--pinball-weight", type=float, default=0.1,
                        help="Pinball weight for augmented model")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/pinball_comparison")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tau_levels = config["training"].get("pinball_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])

    # --- Experiment 1: Baseline (bar-only) ---
    print("\n" + "#" * 72)
    print("# EXPERIMENT 1: Baseline (pinball_weight=0)")
    print("#" * 72)
    ckpt_baseline = run_experiment(args, config, pinball_weight=0.0, save_name="baseline")

    # --- Experiment 2: Pinball-augmented ---
    print("\n" + "#" * 72)
    print(f"# EXPERIMENT 2: Pinball-augmented (pinball_weight={args.pinball_weight})")
    print("#" * 72)
    ckpt_pinball = run_experiment(args, config, pinball_weight=args.pinball_weight,
                                  save_name="pinball")

    # --- Evaluation ---
    print("\n" + "#" * 72)
    print("# EVALUATION")
    print("#" * 72)

    model_baseline, _ = load_model_for_eval(ckpt_baseline, args.device)
    model_pinball, _ = load_model_for_eval(ckpt_pinball, args.device)

    print("\nEvaluating baseline model...")
    baseline_metrics = evaluate_with_quantiles(model_baseline, config, args.device, tau_levels)

    print("Evaluating pinball model...")
    pinball_metrics = evaluate_with_quantiles(model_pinball, config, args.device, tau_levels)

    print_comparison(baseline_metrics, pinball_metrics, tau_levels)


if __name__ == "__main__":
    main()
