#!/usr/bin/env python
"""Render the projective-consistency convergence figure for the paper.

Reads the JSONs written by ``scripts/projective_consistency.py`` and plots
schedule dependence (coordinate-averaged W1 between X(T) marginals under
nested schedules) against the EM resolution ``s``, one panel per mechanism
family, with the same-law Monte-Carlo noise floor and a 1/s reference.

Reproduction:
    python scripts/plot_projective_consistency.py \
        --linear results/projective_consistency/linear.json \
        --neural results/projective_consistency/neural.json \
        --out figures/projective_consistency

Raises
------
FileNotFoundError
    If either input JSON is missing.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--linear", default="results/projective_consistency/linear.json")
    ap.add_argument("--neural", default="results/projective_consistency/neural.json")
    ap.add_argument("--out", default="figures/projective_consistency")
    a = ap.parse_args()

    data = {"OU": json.load(open(a.linear)),
            "Neural": json.load(open(a.neural))}
    subs = data["OU"]["config"]["substeps"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    pairs = [("1v2", "Schedule pair $\\{0,T\\}$ vs $\\{0,T/2,T\\}$", "o-"),
             ("1v8", "Schedule pair $\\{0,T\\}$ vs $\\{0,\\ldots,T/8,\\ldots\\}$", "s-"),
             ("4v8", "Schedule pair $\\{0,\\ldots,T/4,\\ldots\\}$ vs $\\{0,\\ldots,T/8,\\ldots\\}$", "^-")]
    for ax, (mech, d) in zip(axes, data.items()):
        for key, label, style in pairs:
            ys = [d["pairwise_w1"][f"{key}@s{s}"]["mean"] for s in subs]
            ax.loglog(subs, ys, style, ms=3.5, lw=1.2, label=label)
        floor = [d["noise_floor"][f"s{s}"]["mean"] for s in subs]
        ax.loglog(subs, floor, "k--", lw=1.0,
                  label="Same-law sampling floor")
        # 1/s reference anchored at the first 1v8 point.
        y0 = d["pairwise_w1"][f"1v8@s{subs[0]}"]["mean"]
        ax.loglog(subs, [y0 * subs[0] / s for s in subs], ":",
                  color="gray", lw=1.0, label="Slope $1/s$")
        ax.set_title(f"{mech} drift", fontsize=9)
        ax.set_xlabel("Substeps per gap $s$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(subs)
        ax.set_xticklabels([str(s) for s in subs])
    axes[0].set_ylabel("Wasserstein-1 distance", fontsize=8)
    axes[1].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
