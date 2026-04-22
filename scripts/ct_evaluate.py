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
    head_type = cfg.get("head_type", "quantile")  # older checkpoints pre-date bar head
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
        head_type=head_type,
        tau_levels=cfg.get("tau_levels"),
        n_buckets=cfg.get("n_buckets", 1000),
    ).to(device)
    # For bar-head checkpoints we must rehydrate the bar distribution
    # (bucket borders sampled at training time) before load_state_dict,
    # otherwise the head's internal buffers won't line up.
    if head_type == "bar" and ckpt.get("borders") is not None:
        from dotime.model.bar_head import FullSupportBarDistribution
        borders = ckpt["borders"].to(device)
        bar_dist = FullSupportBarDistribution(borders)
        model.bar_head.set_bar_distribution(bar_dist, borders)
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
        "--benchmark",
        choices=["theophylline", "warfarin", "causal_chamber"],
        default="theophylline",
        help="Real-world benchmark to evaluate on",
    )
    # Warfarin-specific flags.
    parser.add_argument(
        "--warfarin-dose-scale", type=float, default=0.01,
        help="Multiplier on raw Warfarin dose (mg) before normalisation",
    )
    parser.add_argument(
        "--warfarin-absorption-hours", type=float, default=1.0,
        help="Intervention window length for Warfarin dosing",
    )
    # Chamber-specific flags (ignored for theophylline).
    parser.add_argument(
        "--chamber-dataset", type=str, default="lt_walks_v1",
        help="CausalChamber dataset name (e.g. lt_walks_v1)",
    )
    parser.add_argument(
        "--chamber-root", type=str, default="/tmp/causalchamber",
        help="Local directory for CausalChamber downloads",
    )
    parser.add_argument(
        "--chamber-query-var", type=str, default="red",
        help="Which sensor variable to query",
    )
    parser.add_argument(
        "--chamber-max-episodes", type=int, default=None,
        help="Cap on the number of episodes to evaluate",
    )
    parser.add_argument(
        "--chamber-dt-seconds", type=float, default=0.1,
        help="Inter-sample spacing in seconds (10 Hz default)",
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

    if args.benchmark == "theophylline":
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
        payload = metrics_to_dict(result)
    elif args.benchmark == "warfarin":
        from dotime.data.pk_pd.warfarin import load_warfarin
        from dotime.eval.continuous_warfarin_eval import (
            evaluate_warfarin_dataset,
            format_summary as warfarin_format_summary,
            metrics_to_dict as warfarin_metrics_to_dict,
        )

        subjects = load_warfarin()
        result = evaluate_warfarin_dataset(
            model,
            subjects,
            device=device,
            n_max=model.temporal_encoder.n_max,
            absorption_window_hours=args.warfarin_absorption_hours,
            dose_scale=args.warfarin_dose_scale,
        )
        print(warfarin_format_summary(result))
        payload = warfarin_metrics_to_dict(result)
    else:
        # causal_chamber
        from dotime.data.causal_chamber_ct import load_chamber_episodes
        from dotime.eval.continuous_chamber_eval import (
            evaluate_episodes,
            format_summary as chamber_format_summary,
            metrics_to_dict as chamber_metrics_to_dict,
        )

        episodes = load_chamber_episodes(
            dataset_name=args.chamber_dataset,
            root=args.chamber_root,
            max_episodes=args.chamber_max_episodes,
        )
        if not episodes:
            raise SystemExit(
                f"No intervention episodes found in dataset {args.chamber_dataset!r}. "
                "Check that the dataset contains actuator changepoints and that "
                "--chamber-max-episodes isn't set to 0."
            )
        result = evaluate_episodes(
            model,
            episodes,
            query_var=args.chamber_query_var,
            device=device,
            n_max=model.temporal_encoder.n_max,
            dt_seconds=args.chamber_dt_seconds,
        )
        print(chamber_format_summary(result))
        payload = chamber_metrics_to_dict(result)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote metrics to {args.save_json}")


if __name__ == "__main__":
    main()
