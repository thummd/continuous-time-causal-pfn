"""Evaluate all grid_v4 checkpoints on a fixed discrete-time OU prior.

Loads each (encoder x integrator x mechanism) checkpoint and computes mean
head loss + RMSE on a held-out batch stream drawn from the *same* prior:
random-DAG, OU mechanism, regular schedule, dt=1, substeps=1.  This is the
"tier-A home turf" cross-distribution probe missing from tab:continuity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.continuous import ContinuousDoOverTimePFN


GRID_ROOT = REPO / "checkpoints" / "ct" / "grid_v4"
CELLS = [
    "pos_naive_OU",     "pos_fine_OU",
    "time_naive_OU",    "time_fine_OU",
    "pos_naive_neural", "pos_fine_neural",
    "time_naive_neural","time_fine_neural",
    "pos_naive_mixed",  "pos_fine_mixed",
    "time_naive_mixed", "time_fine_mixed",
]


def _load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ContinuousDoOverTimePFN(
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
        positional_only=cfg.get("positional_only", False),
        head_type=cfg.get("head_type", "quantile"),
        tau_levels=cfg.get("tau_levels"),
        n_buckets=cfg.get("n_buckets", 1000),
    ).to(device)
    if cfg.get("head_type", "quantile") == "bar" and ckpt.get("borders") is not None:
        from dotime.model.bar_head import FullSupportBarDistribution
        borders = ckpt["borders"].to(device)
        bar_dist = FullSupportBarDistribution(borders)
        model.bar_head.set_bar_distribution(bar_dist, borders)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    losses, rmses = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        loss = model.head.loss(out, batch["Y_true_norm"])
        if not torch.isfinite(loss):
            continue
        pred = model.head.predict_mean(out)
        rmse = (pred - batch["Y_true_norm"]).pow(2).mean().sqrt()
        losses.append(float(loss.item()))
        rmses.append(float(rmse.item()))
    import numpy as np
    arr = np.asarray(losses)
    rng = np.random.default_rng(0)
    boots = rng.choice(arr, size=(2000, len(arr)), replace=True).mean(axis=1)
    return {
        "loss": float(arr.mean()),
        "loss_ci_lo": float(np.quantile(boots, 0.025)),
        "loss_ci_hi": float(np.quantile(boots, 0.975)),
        "rmse_norm": float(np.mean(rmses)),
        "n_batches": len(losses),
        "per_batch_loss": losses,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--n-eval-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-seed", type=int, default=99999)
    p.add_argument("--mechanism-kind", default="linear", choices=["linear", "neural", "mixed"])
    p.add_argument("--schedule", default="regular", choices=["regular", "jittered", "exponential", "mixed"])
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--save-json", type=Path, default=None)
    p.add_argument("--cells", nargs="+", default=None,
                   help="Subset of cells to evaluate. If --save-json exists, "
                        "matching rows are replaced and others kept.")
    args = p.parse_args()
    if args.save_json is None:
        tag = f"{args.mechanism_kind}_{args.schedule}_dt{args.dt}_s{args.substeps}"
        args.save_json = REPO / "results" / "grid_v4" / f"eval_{tag}.json"

    cells_to_run = args.cells if args.cells else CELLS
    unknown = [c for c in cells_to_run if c not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells: {unknown}; valid: {CELLS}")

    existing_rows: dict[str, dict] = {}
    if args.cells and args.save_json.exists():
        prev = json.loads(args.save_json.read_text())
        existing_rows = {r["cell"]: r for r in prev.get("rows", [])}
        print(f"[merge] loaded {len(existing_rows)} existing rows from {args.save_json}")

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    new_rows: dict[str, dict] = {}
    for cell in cells_to_run:
        ckpt_path = GRID_ROOT / cell / "seed_0" / "continuous_do_over_time_pfn_best.pt"
        if not ckpt_path.exists():
            print(f"[skip] {cell} (missing checkpoint)")
            continue
        model, cfg = _load_model(ckpt_path, device)

        loader = ContinuousTemporalInterventionDataLoader(
            num_steps=args.n_eval_batches,
            batch_size=args.batch_size,
            seed=args.eval_seed,
            device=device,
            prefetch=0,
            prior_mode="random",
            n_min_prior=3,
            n_max_prior=cfg.get("n_max", 10),
            edge_prob=0.3,
            hidden_prob=0.0,
            mechanism_kind=args.mechanism_kind,
            schedule=args.schedule,
            dt=args.dt,
            substeps=args.substeps,
            pair_mode="counterfactual",
            t_range=(60, 120),
            n_max=cfg["n_max"],
            theta_range=(0.1, 0.5),
            sigma_range=(0.2, 0.6),
            weight_scale=0.3,
            intervention_value_scale=1.0,
            normalize=True,
            target_key="Y_true",
            n_queries=1,
            query_mode="single",
            vectorize=True,
        )
        m = _evaluate(model, loader, device)
        print(f"[{cell:24s}] loss={m['loss']:.4f}  "
              f"[{m['loss_ci_lo']:.4f}, {m['loss_ci_hi']:.4f}]  "
              f"rmse_norm={m['rmse_norm']:.4f}  n_batches={m['n_batches']}")
        new_rows[cell] = {"cell": cell, **{k: v for k, v in m.items() if k != "per_batch_loss"}}

    merged = {**existing_rows, **new_rows}
    rows = [merged[c] for c in CELLS if c in merged]

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps({
        "eval_distribution": {
            "mechanism_kind": args.mechanism_kind,
            "schedule": args.schedule,
            "dt": args.dt,
            "substeps": args.substeps,
            "n_eval_batches": args.n_eval_batches,
            "batch_size": args.batch_size,
            "eval_seed": args.eval_seed,
        },
        "rows": rows,
    }, indent=2))
    print(f"\nSaved -> {args.save_json}")


if __name__ == "__main__":
    main()
