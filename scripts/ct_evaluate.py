#!/usr/bin/env python
"""Zero-shot evaluation entry point for continuous-time PFN checkpoints.

Currently supports the Theophylline PK benchmark.  Loads a checkpoint
saved by ``scripts/ct_train.py``, rebuilds the model, and runs
:func:`evaluate_dataset` on the bundled dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from dotime.data.pk_pd.theophylline import load_theophylline
from dotime.eval.continuous_pk_eval import (
    evaluate_dataset,
    format_summary,
    metrics_to_dict,
)
from dotime.model.continuous import ContinuousDoOverTimePFN


def _load_model(checkpoint_path: Path, device: str) -> ContinuousDoOverTimePFN:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        head_type="quantile",
        tau_levels=cfg.get("tau_levels"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot CT-PFN evaluation")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a continuous_do_over_time_pfn_best.pt file",
    )
    parser.add_argument(
        "--benchmark", choices=["theophylline"], default="theophylline",
        help="Real-world benchmark to evaluate on",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-json", type=Path, default=None,
        help="If set, write the full metrics dict to this JSON file",
    )
    parser.add_argument(
        "--absorption-window-hours", type=float, default=1.0,
        help="Length of the PK hard-intervention window on the Dose variable",
    )
    parser.add_argument(
        "--dose-scale", type=float, default=1.0,
        help="Multiplier on the raw mg/kg dose before it enters the batch",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(args.checkpoint, device=device)
    subjects = load_theophylline()

    result = evaluate_dataset(
        model,
        subjects,
        device=device,
        n_max=model.temporal_encoder.n_max,
        absorption_window_hours=args.absorption_window_hours,
        dose_scale=args.dose_scale,
    )

    print(format_summary(result))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(metrics_to_dict(result), f, indent=2)
        print(f"\nWrote metrics to {args.save_json}")


if __name__ == "__main__":
    main()
