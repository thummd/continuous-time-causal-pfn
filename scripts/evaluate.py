#!/usr/bin/env python
"""Evaluation entry point for Do-Over-Time-PFN.

Evaluates a trained model (bar or quantile head) on CausalChamber.
Supports multiple query time offsets to study effect decay.

Usage
-----
python scripts/evaluate.py --checkpoint checkpoints/quantile/do_over_time_pfn_best.pt
python scripts/evaluate.py --checkpoint ckpt.pt --query-offsets 1 2 3 5 10
"""

import argparse
import torch

from dotime.model.do_over_time_pfn import DoOverTimePFN
from dotime.eval.evaluate_chamber import evaluate_on_chamber


def load_model(checkpoint_path, device="cpu"):
    """Load a Do-Over-Time-PFN from checkpoint (bar or quantile head)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = DoOverTimePFN(**config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Restore bar distribution if applicable
    head_type = ckpt.get('head_type', config.get('head_type', 'bar'))
    if head_type == 'bar' and ckpt.get('borders') is not None:
        from pfns.model.bar_distribution import FullSupportBarDistribution
        borders = ckpt['borders']
        bar_dist = FullSupportBarDistribution(borders)
        model.bar_head.set_bar_distribution(bar_dist, borders)

    model = model.to(device)
    model.eval()

    print(f"Model loaded: head={head_type}, step={ckpt.get('step', '?')}, "
          f"eval_loss={ckpt.get('eval_loss', float('nan')):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Do-Over-Time-PFN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="lt_walks_v1")
    parser.add_argument("--experiment", type=str, default="actuators_white")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--obs-window", type=int, default=50)
    parser.add_argument("--query-offsets", type=int, nargs="+", default=[5])
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)

    for offset in args.query_offsets:
        print(f"\n--- query_time_offset = {offset} ---")
        results = evaluate_on_chamber(
            model,
            dataset_name=args.dataset,
            experiment_name=args.experiment,
            obs_window=args.obs_window,
            query_time_offset=offset,
            max_episodes=args.max_episodes,
            device=args.device,
        )

        print("\n" + "=" * 60)
        print(f"Results  (offset={offset})")
        print("=" * 60)

        if results.get("overall"):
            r = results["overall"]
            print(f"\nOverall: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}, N={r['n_episodes']}")

        for var, m in results.get("per_var", {}).items():
            print(f"  {var}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, N={m['n_episodes']}")


if __name__ == "__main__":
    main()
