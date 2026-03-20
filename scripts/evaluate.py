#!/usr/bin/env python
"""Evaluation entry point for Do-Over-Time-PFN."""

import argparse
import torch

from dotime.model.do_over_time_pfn import DoOverTimePFN
from dotime.model.bar_head import BarDistributionHead
from dotime.eval.evaluate_chamber import evaluate_on_chamber
from pfns.model.bar_distribution import FullSupportBarDistribution


def main():
    parser = argparse.ArgumentParser(description="Evaluate Do-Over-Time-PFN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="lt_walks_v1")
    parser.add_argument("--experiment", type=str, default="smooth_polarizers")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-episodes", type=int, default=100)
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt['config']

    # Recreate model
    model = DoOverTimePFN(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Restore bar distribution
    borders = ckpt['borders']
    bar_dist = FullSupportBarDistribution(borders)
    model.bar_head.set_bar_distribution(bar_dist, borders)
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded (step {ckpt['step']}, eval_loss={ckpt['eval_loss']:.4f})")

    # Evaluate on CausalChamber
    results = evaluate_on_chamber(
        model,
        dataset_name=args.dataset,
        experiment_name=args.experiment,
        max_episodes=args.max_episodes,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    if "overall" in results and results["overall"]:
        print(f"\nOverall: RMSE={results['overall']['rmse']:.4f}, "
              f"MAE={results['overall']['mae']:.4f}, "
              f"N={results['overall']['n_episodes']}")

    if "per_var" in results:
        for var, metrics in results["per_var"].items():
            print(f"  {var}: RMSE={metrics['rmse']:.4f}, "
                  f"MAE={metrics['mae']:.4f}, N={metrics['n_episodes']}")


if __name__ == "__main__":
    main()
