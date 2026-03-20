#!/usr/bin/env python
"""Training entry point for Do-Over-Time-PFN."""

import argparse
import yaml
import torch

from dotime.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train Do-Over-Time-PFN")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embed-size", type=int, default=None)
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--n-buckets", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["transformer", "gdp"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--pinball-weight", type=float, default=None,
                        help="Auxiliary pinball loss weight (0 = disabled)")
    parser.add_argument("--pinball-quantiles", type=float, nargs="+", default=None,
                        help="Quantile levels for pinball loss")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    prior_cfg = config["prior"]
    train_cfg = config["training"]

    train(
        n_max=model_cfg["n_max"],
        embed_size=args.embed_size or model_cfg["embed_size"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=args.n_encoder_layers or model_cfg["n_encoder_layers"],
        n_cross_attn_heads=model_cfg["n_cross_attn_heads"],
        n_buckets=args.n_buckets or model_cfg["n_buckets"],
        encoder_backend=args.backend or "transformer",
        encoder_config=model_cfg.get("encoder"),
        n_max_prior=prior_cfg["n_max"],
        t_range=tuple(prior_cfg["t_range"]),
        burn_in=prior_cfg["burn_in"],
        downstream_prob=prior_cfg["downstream_prob"],
        batch_size=args.batch_size or train_cfg["batch_size"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=args.total_steps or train_cfg["total_steps"],
        grad_clip=train_cfg["grad_clip"],
        eval_every=train_cfg["eval_every"],
        bucket_calibration_samples=train_cfg["bucket_calibration_samples"],
        seed=train_cfg["seed"],
        device=device,
        save_dir=args.save_dir,
        pinball_weight=(args.pinball_weight if args.pinball_weight is not None
                        else train_cfg.get("pinball_weight", 0.0)),
        pinball_quantiles=(args.pinball_quantiles if args.pinball_quantiles is not None
                           else train_cfg.get("pinball_quantiles")),
    )


if __name__ == "__main__":
    main()
