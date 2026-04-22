#!/usr/bin/env python
"""Training entry point for Continuous-time Causal PFN.

Parallel to ``scripts/train.py``.  Loads a YAML config, applies CLI
overrides, and calls :func:`dotime.training.continuous_trainer.train_continuous`.
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml

from dotime.training.continuous_trainer import train_continuous


def _float_or_none(x):
    return None if x in (None, "None", "none") else float(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Continuous-time Causal PFN")
    parser.add_argument("--config", type=str, default="configs/continuous_default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/ct")

    # Training-loop CLI overrides
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefetch", type=int, default=None)

    # Model CLI overrides
    parser.add_argument("--embed-size", type=int, default=None)
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument(
        "--encoder-backend", type=str, default=None, choices=["transformer", "gdp"]
    )
    parser.add_argument("--n-mixer-layers", type=int, default=None)
    parser.add_argument(
        "--head-type", type=str, default=None, choices=["quantile", "bar"],
        help="Output head: quantile (pinball loss) or bar (bucket distribution)",
    )
    parser.add_argument(
        "--n-buckets", type=int, default=None,
        help="Number of bar-distribution buckets (bar head only)",
    )

    # Prior CLI overrides (most useful for ablations)
    parser.add_argument(
        "--prior-mode", type=str, default=None, choices=["tscm", "random"],
        help="tscm: fixed named structure; random: random-graph per trajectory",
    )
    parser.add_argument("--n-min-prior", type=int, default=None)
    parser.add_argument("--n-max-prior", type=int, default=None)
    parser.add_argument("--edge-prob", type=float, default=None)
    parser.add_argument(
        "--hidden-prob", type=float, default=None,
        help="Probability that each non-(A, Y) node is hidden "
             "(random-graph prior only). Hidden nodes still drive the "
             "dynamics but are masked out of X_obs / variable_mask.",
    )
    parser.add_argument(
        "--regime-prob", type=float, default=None,
        help="Fraction of trajectories drawn from a regime-switching "
             "continuous-time SCM (random-graph prior only). 0.0 = "
             "always stationary; 0.15 matches the discrete-time CTP mix.",
    )
    parser.add_argument(
        "--regime-count-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
        help="Uniform prior on the number of regimes R in [LO, HI].",
    )
    parser.add_argument(
        "--mechanism-kind", type=str, default=None,
        choices=["linear", "neural", "mixed"],
        help="Per-variable drift family. 'linear' = OU (default). "
             "'neural' = small MLP. 'mixed' = Bernoulli(p_neural) per "
             "variable, OU otherwise. Random-graph prior only.",
    )
    parser.add_argument(
        "--p-neural", type=float, default=None,
        help="Fraction of variables drawn with a neural drift when "
             "mechanism_kind='mixed'.",
    )
    parser.add_argument(
        "--neural-hidden-dim", type=int, default=None,
        help="Hidden width of the MLP drift (default 8).",
    )
    parser.add_argument(
        "--num-substeps", type=int, default=None,
        help="Phase-11 fine-grid integration. Each observation gap is "
             "split into this many Euler-Maruyama sub-steps with "
             "independent noise per step. 1 (default) = tier-(B) naive "
             "observation-grid integration. Large values approximate "
             "tier-(C) schedule-invariant continuous integration.",
    )
    parser.add_argument("--tscm-structure", type=str, default=None)
    parser.add_argument(
        "--schedule", type=str, default=None, choices=["regular", "jittered", "exponential"]
    )
    parser.add_argument(
        "--pair-mode", type=str, default=None, choices=["counterfactual", "interventional"]
    )
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument(
        "--intervention-source",
        type=str,
        default=None,
        choices=["prior", "positivity_aware"],
    )
    parser.add_argument(
        "--intervention-kind-probs", type=float, nargs=3, default=None,
        metavar=("HARD", "SOFT", "TIME_VARYING"),
    )
    parser.add_argument("--intervention-value-scale", type=float, default=None)
    parser.add_argument("--t-range", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--n-queries", type=int, default=None)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = config["model"]
    p = config["prior"]
    t = config["training"]

    train_continuous(
        # model
        n_max=m["n_max"],
        embed_size=args.embed_size or m["embed_size"],
        n_heads=m["n_heads"],
        n_encoder_layers=args.n_encoder_layers or m["n_encoder_layers"],
        n_cross_attn_heads=m["n_cross_attn_heads"],
        encoder_backend=args.encoder_backend or "transformer",
        context_window=args.context_window or m.get("context_window", 128),
        n_mixer_layers=args.n_mixer_layers or m.get("n_mixer_layers", 1),
        num_time_frequencies=m.get("num_time_frequencies", 64),
        time_min_freq=m.get("time_min_freq", 0.01),
        time_max_freq=m.get("time_max_freq", 10.0),
        tau_levels=m.get("tau_levels"),
        head_type=args.head_type or m.get("head_type", "quantile"),
        n_buckets=args.n_buckets or m.get("n_buckets", 1000),
        # prior
        prior_mode=args.prior_mode or p.get("mode", "tscm"),
        n_min_prior=args.n_min_prior or p.get("n_min_prior", 3),
        n_max_prior=args.n_max_prior or p.get("n_max_prior", 10),
        edge_prob=args.edge_prob if args.edge_prob is not None else p.get("edge_prob", 0.3),
        hidden_prob=args.hidden_prob if args.hidden_prob is not None else p.get("hidden_prob", 0.0),
        regime_prob=args.regime_prob if args.regime_prob is not None else p.get("regime_prob", 0.0),
        regime_count_range=(
            tuple(args.regime_count_range)
            if args.regime_count_range is not None
            else tuple(p.get("regime_count_range", [2, 3]))
        ),
        mechanism_kind=args.mechanism_kind or p.get("mechanism_kind", "linear"),
        p_neural=args.p_neural if args.p_neural is not None else p.get("p_neural", 0.0),
        neural_hidden_dim=args.neural_hidden_dim or p.get("neural_hidden_dim", 8),
        neural_out_scale_range=tuple(p.get("neural_out_scale_range", [0.5, 2.0])),
        num_substeps=args.num_substeps or p.get("num_substeps", 1),
        tscm_structure=args.tscm_structure or p["tscm_structure"],
        schedule=args.schedule or p["schedule"],
        dt=args.dt if args.dt is not None else p["dt"],
        jitter=p.get("jitter", 0.3),
        exp_rate=p.get("exp_rate", 1.0),
        pair_mode=args.pair_mode or p["pair_mode"],
        t_range=tuple(args.t_range) if args.t_range else tuple(p["t_range"]),
        intervention_kind_probs=tuple(args.intervention_kind_probs) if args.intervention_kind_probs else tuple(p["intervention_kind_probs"]),
        intervention_source=args.intervention_source or p["intervention_source"],
        intervention_value_scale=args.intervention_value_scale if args.intervention_value_scale is not None else p["intervention_value_scale"],
        intervention_window_frac=tuple(p["intervention_window_frac"]),
        soft_shift_scale=p.get("soft_shift_scale", 1.0),
        time_varying_profile=p.get("time_varying_profile", "random"),
        theta_range=tuple(p["theta_range"]),
        sigma_range=tuple(p["sigma_range"]),
        weight_scale=p.get("weight_scale", 0.5),
        # training
        batch_size=args.batch_size or t["batch_size"],
        lr=args.lr if args.lr is not None else t["lr"],
        weight_decay=t["weight_decay"],
        warmup_steps=args.warmup_steps or t["warmup_steps"],
        total_steps=args.total_steps or t["total_steps"],
        grad_clip=t["grad_clip"],
        eval_every=args.eval_every or t["eval_every"],
        eval_num_steps=t.get("eval_num_steps", 20),
        seed=args.seed or t["seed"],
        device=device,
        save_dir=args.save_dir,
        prefetch=args.prefetch if args.prefetch is not None else t.get("prefetch", 2),
        target_key=t.get("target_key", "Y_true"),
        n_queries=args.n_queries or t.get("n_queries", 1),
        query_mode=t.get("query_mode", "single"),
        early_stop_patience=t.get("early_stop_patience", 0),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=(
            args.wandb_run_name
            or (os.path.basename(args.save_dir) if args.save_dir else None)
        ),
    )


if __name__ == "__main__":
    main()
