#!/usr/bin/env python
"""Aggregate multi-seed real-data eval JSONs into mean+/-std for tab:real.

Reads the per-seed eval outputs written by ``eval_multiseed_realdata.sh``
(``results/phase14c_multiseed/``) and produces, for every row of the
paper's real-data table, the across-seed mean and population std of the
four reported quantities: pooled RMSE, naive pooled RMSE, lift over naive,
and mean Pearson r.

Usage:
    python scripts/aggregate_multiseed_realdata.py \
        --results-dir results/phase14c_multiseed \
        --seeds 0 1 2 3 4 \
        --out results/phase14c_multiseed/aggregate_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return None, None, 0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n  # population std (n), not n-1
    return mean, math.sqrt(var), n


def _load(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


# (row label, benchmark, mechanism, field-extractor) -------------------------
# Each extractor pulls (rmse, naive_rmse, lift, pearson) from an aggregate dict.
def _chamber_fields(agg):
    return (
        agg["pooled_rmse"],
        agg["naive_pooled_rmse"],
        agg["lift_over_naive"],
        agg["mean_pearson_r"],
    )


def _warfarin_fields(agg, suffix):
    return (
        agg[f"pooled_rmse_{suffix}"],
        agg[f"naive_pooled_rmse_{suffix}"],
        agg[f"lift_over_naive_{suffix}"],
        agg[f"mean_pearson_r_{suffix}"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--mechs", nargs="+", default=["linear", "mixed"])
    ap.add_argument("--qvars", nargs="+",
                    default=["rpm_in", "current_in", "pressure_downwind"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rd = args.results_dir
    rows = []  # each: dict with label, mech, and per-metric mean/std/n lists

    def add_row(label, mech, per_seed_tuples):
        rmse = _mean_std([t[0] for t in per_seed_tuples])
        naive = _mean_std([t[1] for t in per_seed_tuples])
        lift = _mean_std([t[2] for t in per_seed_tuples])
        pear = _mean_std([t[3] for t in per_seed_tuples])
        rows.append({
            "label": label, "mech": mech,
            "n_seeds": rmse[2],
            "rmse_mean": rmse[0], "rmse_std": rmse[1],
            "naive_rmse_mean": naive[0], "naive_rmse_std": naive[1],
            "lift_mean": lift[0], "lift_std": lift[1],
            "pearson_mean": pear[0], "pearson_std": pear[1],
        })

    for mech in args.mechs:
        # Warfarin concentration (cp) and PD (pca)
        wf = {s: _load(rd / f"warfarin_p13b_pnc000_{mech}_seed{s}.json")
              for s in args.seeds}
        cp = [_warfarin_fields(wf[s]["aggregate"], "cp")
              for s in args.seeds if wf[s] is not None]
        pca = [_warfarin_fields(wf[s]["aggregate"], "pca")
               for s in args.seeds if wf[s] is not None]
        if cp:
            add_row("Warfarin (concentration)", mech, cp)
        if pca:
            add_row("Warfarin (PD response)", mech, pca)

        # Chamber-WT query vars
        for qvar in args.qvars:
            ch = {s: _load(rd / f"chamber_p13b_pnc000_{mech}_seed{s}_{qvar}.json")
                  for s in args.seeds}
            tuples = [_chamber_fields(ch[s]["aggregate"])
                      for s in args.seeds if ch[s] is not None]
            if tuples:
                add_row(f"Chamber-WT ({qvar})", mech, tuples)

    # ---- pretty table ----
    hdr = f"{'Row':<30} {'mech':<7} {'seeds':>5}  {'RMSE (mean+/-std)':>22}  {'lift%':>16}  {'Pearson r':>16}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rmse = f"{r['rmse_mean']:.2f} +/- {r['rmse_std']:.2f}"
        lift = f"{100*r['lift_mean']:.1f} +/- {100*r['lift_std']:.1f}"
        pear = f"{r['pearson_mean']:+.2f} +/- {r['pearson_std']:.2f}"
        print(f"{r['label']:<30} {r['mech']:<7} {r['n_seeds']:>5}  {rmse:>22}  {lift:>16}  {pear:>16}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({"seeds": args.seeds, "rows": rows}, f, indent=2)
        print(f"\nWrote aggregate summary to {args.out}")


if __name__ == "__main__":
    main()
