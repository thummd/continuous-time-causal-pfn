#!/usr/bin/env python
"""Per-TSCM histograms of RMSE and direction accuracy from a tscm_eval JSON.

Consumes a JSON produced by ``scripts/tscm_identifiability.py`` and
produces one histogram per structure so we can spot bimodal or
heavy-tailed failure modes that aggregate statistics hide.

Usage
-----
    python scripts/plot_per_tscm_histograms.py \
        --json results/v5_combined/tscm_eval_n200.json \
        --out-dir figures/v5_combined_hist
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _per_tscm_values(results: dict, key: str):
    out = {}
    for structure, r in results.items():
        if not isinstance(r, dict):
            continue
        per = r.get('per_tscm', [])
        vals = [t.get(key) for t in per if t.get(key) is not None]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            out[structure] = vals
    return out


def _plot_grid(values_by_structure, metric_label, out_path, bins=20, color="#1f77b4"):
    n = len(values_by_structure)
    if n == 0:
        return
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows), squeeze=False)
    for ax, (name, vals) in zip(axes.flat, values_by_structure.items()):
        arr = np.asarray(vals, dtype=np.float64)
        ax.hist(arr, bins=bins, color=color, alpha=0.85)
        ax.axvline(arr.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean={arr.mean():.2f}")
        ax.axvline(np.median(arr), color="red", linestyle="--", linewidth=1,
                   label=f"median={np.median(arr):.2f}")
        ax.set_title(f"{name} (n={arr.size})", fontsize=9)
        ax.set_xlabel(metric_label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    # Hide unused axes
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Per-TSCM {metric_label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    if args.out_dir is None:
        base = os.path.splitext(os.path.basename(args.json))[0]
        args.out_dir = os.path.join("figures", f"{base}_hist")

    with open(args.json) as f:
        data = json.load(f)
    results = data.get("results", {})
    if not results:
        print("No results in JSON.")
        return

    for metric, label, color in [
        ("rmse", "RMSE", "#1f77b4"),
        ("direction_accuracy", "Direction accuracy", "#2ca02c"),
        ("effect_rmse", "Effect RMSE", "#d62728"),
        ("mean_causal_effect", "Mean causal effect", "#9467bd"),
    ]:
        vals = _per_tscm_values(results, metric)
        out_path = os.path.join(args.out_dir, f"{metric}.png")
        _plot_grid(vals, label, out_path, bins=args.bins,
                   color=color if metric != "direction_accuracy" else "#2ca02c")
        if vals:
            print(f"Wrote {out_path} for {len(vals)} structures.")

    print(f"\nDone. Figures in {args.out_dir}")


if __name__ == "__main__":
    main()
