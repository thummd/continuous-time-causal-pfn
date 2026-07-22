#!/usr/bin/env python
"""Cross-prior validation eval for continuous-time PFN checkpoints.

Loads a checkpoint saved by ``scripts/ct_train.py``, rebuilds the
model, then runs a fresh ``ContinuousTemporalInterventionDataLoader``
with prior knobs supplied via CLI flags (rather than from the
checkpoint's training config).  The goal is to evaluate every model
against a *common* prior so the ablation table becomes apples-to-
apples instead of self-referential.

Two named eval-prior presets mirror the workshop paper's
``tab:continuity_val`` columns:

  --eval-mode ou_mixed     : linear OU mechanism, mixed schedule
                             (continuous-time analogue of AR(1) on a
                             stochastic schedule).
  --eval-mode dt1_regular  : linear OU mechanism, regular schedule
                             with dt=1 -- the discrete-time tier-(A)
                             limit every continuous-time prior reduces
                             to at that step size.

Either preset is overridable via the lower-level flags
--mechanism-kind / --schedule / --dt / --jitter / --substeps for
ad-hoc eval priors.  Outputs a JSON with the aggregate ``eval_loss``
and ``eval_rmse`` over ``--n-eval-batches`` batches.

Note: checkpoints that use ``positional_only=True`` (the workshop
ablation's positional-encoder cells) require the model class
extension that ships in the ``-finegrid`` repo; the upstream model
class does not yet expose that flag.  The eval script will raise on
``load_state_dict`` mismatch in that case until the extension is
ported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.continuous import ContinuousDoOverTimePFN


@torch.no_grad()
def _evaluate(model, dataloader, device) -> dict:
    """Mirror of dotime.training.continuous_trainer._evaluate."""
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0
    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        output = model(batch)
        loss = model.head.loss(output, batch["Y_true_norm"])
        mean_pred = model.head.predict_mean(output)
        rmse = torch.sqrt(((mean_pred - batch["Y_true_norm"]) ** 2).mean())
        total_loss += float(loss.item())
        total_rmse += float(rmse.item())
        n_batches += 1
    return {
        "eval_loss": total_loss / max(1, n_batches),
        "eval_rmse": total_rmse / max(1, n_batches),
        "n_batches": n_batches,
    }


def _load_model(checkpoint_path: Path, device: str) -> ContinuousDoOverTimePFN:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    head_type = cfg.get("head_type", "quantile")
    model_kwargs = dict(
        n_max=cfg["n_max"],
        embed_size=cfg["embed_size"],
        n_encoder_layers=cfg["n_encoder_layers"],
        n_cross_attn_heads=cfg["n_cross_attn_heads"],
        encoder_backend=cfg.get("encoder_backend", "transformer"),
        context_window=cfg.get("context_window", 128),
        n_mixer_layers=cfg.get("n_mixer_layers", 1),
        num_time_frequencies=cfg.get("num_time_frequencies", 64),
        time_min_freq=cfg.get("time_min_freq", 0.01),
        time_max_freq=cfg.get("time_max_freq", 10.0),
        head_type=head_type,
        tau_levels=cfg.get("tau_levels"),
        n_buckets=cfg.get("n_buckets", 1000),
    )
    if "positional_only" in cfg:
        model_kwargs["positional_only"] = cfg["positional_only"]
    model = ContinuousDoOverTimePFN(**model_kwargs).to(device)
    if head_type == "bar" and ckpt.get("borders") is not None:
        from dotime.model.bar_head import FullSupportBarDistribution
        borders = ckpt["borders"].to(device)
        bar_dist = FullSupportBarDistribution(borders)
        model.bar_head.set_bar_distribution(bar_dist, borders)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


PRESETS = {
    "ou_mixed": dict(
        mechanism_kind="linear", p_neural=0.0,
        schedule="mixed", dt=1.0, jitter=0.3, exp_rate=1.0,
    ),
    "dt1_regular": dict(
        mechanism_kind="linear", p_neural=0.0,
        schedule="regular", dt=1.0, jitter=0.0, exp_rate=1.0,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-prior CT-PFN validation eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--eval-mode", choices=tuple(PRESETS) + ("custom",), default="ou_mixed",
        help="Named eval prior preset. 'custom' = use the lower-level flags.",
    )
    parser.add_argument("--mechanism-kind", choices=["linear", "neural", "mixed"], default=None)
    parser.add_argument("--p-neural", type=float, default=None)
    parser.add_argument("--schedule", choices=["regular", "jittered", "exponential", "mixed"], default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--jitter", type=float, default=None)
    parser.add_argument("--exp-rate", type=float, default=None)
    parser.add_argument("--substeps", type=int, default=None,
                        help="EM integration substeps per observation gap.")
    parser.add_argument(
        "--intervention-value-scale", type=float, default=1.0,
        help="Intervention magnitude of the eval prior. MUST match the "
             "training config for a train==eval comparison; the reported "
             "table pins both to 1.0. (The pre-release eval hardcoded 2.0, "
             "silently mismatching 1.0-trained checkpoints.)",
    )
    parser.add_argument("--n-eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260508,
                        help="Eval seed; offset from training seeds.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_cfg = _load_model(args.checkpoint, device=device)

    if args.eval_mode == "custom":
        prior_cfg = {}
    else:
        prior_cfg = dict(PRESETS[args.eval_mode])
    for key in ("mechanism_kind", "p_neural", "schedule", "dt", "jitter", "exp_rate"):
        v = getattr(args, key)
        if v is not None:
            prior_cfg[key] = v
    substeps = args.substeps if args.substeps is not None else 1

    n_max = ckpt_cfg["n_max"]
    tscm_structure = ckpt_cfg.get("tscm_structure", "back_door")
    pair_mode = ckpt_cfg.get("pair_mode", "counterfactual")

    # ``substeps`` vs ``num_substeps``: the upstream dataloader uses
    # ``num_substeps``; the finegrid branch uses ``substeps``.  Try
    # both so the script runs from either repo.
    loader_kwargs = dict(
        num_steps=args.n_eval_batches,
        batch_size=args.batch_size,
        prior_mode="random",
        tscm_structure=tscm_structure,
        n_min_prior=3,
        n_max_prior=10,
        edge_prob=0.3,
        hidden_prob=0.0,
        regime_prob=0.0,
        n_max=n_max,
        normalize=True,
        target_key="Y_true",
        n_queries=1,
        query_mode="single",
        theta_range=(0.1, 0.5),
        sigma_range=(0.2, 0.6),
        weight_scale=0.3,
        intervention_value_scale=args.intervention_value_scale,
        intervention_window_frac=(0.1, 0.3),
        mechanism_kind=prior_cfg.get("mechanism_kind", "linear"),
        p_neural=prior_cfg.get("p_neural", 0.0),
        schedule=prior_cfg.get("schedule", "mixed"),
        dt=prior_cfg.get("dt", 1.0),
        jitter=prior_cfg.get("jitter", 0.3),
        exp_rate=prior_cfg.get("exp_rate", 1.0),
        pair_mode=pair_mode,
        t_range=(60, 120),
        seed=args.seed,
        device=device,
        prefetch=0,
    )
    try:
        loader = ContinuousTemporalInterventionDataLoader(
            substeps=substeps, **loader_kwargs,
        )
    except TypeError:
        loader = ContinuousTemporalInterventionDataLoader(
            num_substeps=substeps, **loader_kwargs,
        )

    metrics = _evaluate(model, loader, device=device)
    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_mode": args.eval_mode,
        "eval_prior_config": {
            "mechanism_kind": prior_cfg.get("mechanism_kind", "linear"),
            "schedule": prior_cfg.get("schedule", "mixed"),
            "dt": prior_cfg.get("dt", 1.0),
            "jitter": prior_cfg.get("jitter", 0.3),
            "num_substeps": substeps,
        },
        "n_eval_batches": args.n_eval_batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "metrics": metrics,
        "checkpoint_config": {
            k: ckpt_cfg.get(k)
            for k in ("n_max", "embed_size", "positional_only",
                      "tscm_structure", "schedule", "pair_mode")
            if k in ckpt_cfg
        },
    }
    print(f"=== {args.eval_mode} eval of {args.checkpoint.name} ===")
    print(f"  eval_loss : {metrics['eval_loss']:.4f}")
    print(f"  eval_rmse : {metrics['eval_rmse']:.4f}")
    print(f"  n_batches : {metrics['n_batches']}")
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  -> {args.save_json}")


if __name__ == "__main__":
    main()
