#!/usr/bin/env python
"""Aggregate the faithful finegrid grid_v4 eval (8 cells x 3 seeds x 3 cfgs)
into multi-seed reproductions of tab:reg and tab:substeps.

Reads results/grid_v4_synth_finegrid/{enc}_{integ}_{prior}_seed{S}_{cfg}.json
(metrics.eval_loss). Deltas are paired by seed index. Lower eval_loss = better.
"""
import json, glob, statistics, os, argparse

def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None, 0)
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0, len(xs))

def fmt(m, s):
    return "  n/a  " if m is None else f"{m:.4f}±{s:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/grid_v4_synth_finegrid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # Default the summary next to the inputs, not to a hardcoded sibling
    # directory -- otherwise `--dir <other>` silently overwrites this file.
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    D, SEEDS = args.dir, args.seeds
    if args.out is None:
        args.out = os.path.join(D, "aggregate_summary.json")

    def load(enc, integ, prior, seed, cfg):
        f = os.path.join(D, f"{enc}_{integ}_{prior}_seed{seed}_{cfg}.json")
        try:
            return json.load(open(f))["metrics"]["eval_loss"]
        except Exception:
            return None

    def paired_delta(a_list, b_list):
        return [a - b for a, b in zip(a_list, b_list) if a is not None and b is not None]

    PRIORS = ["OU", "neural"]
    summary = {"tab_reg": [], "tab_substeps": [], "fine_vs_naive_reg": []}

    # ===== tab:reg  (regular eval, s=1): encoder gap pos - time, per integrator =====
    print("=" * 78)
    print("TABLE tab:reg  (regular eval, s_eval=1)  --  multi-seed mean±std")
    print(f"{'Prior':<7}{'Trained':<8}{'pos':>16}{'time':>16}{'Δ(pos−time)':>18}  per-seed Δ")
    print("-" * 90)
    for prior in PRIORS:
        for integ in ["naive", "fine"]:
            pos = [load("pos", integ, prior, s, "reg_s1") for s in SEEDS]
            tim = [load("time", integ, prior, s, "reg_s1") for s in SEEDS]
            d = paired_delta(pos, tim)
            pm, ps, _ = ms(pos); tm, ts, _ = ms(tim); dm, ds, _ = ms(d)
            ps_str = "[" + ", ".join(f"{x:+.4f}" for x in d) + "]"
            row = {"prior": prior, "trained": integ, "pos_mean": pm, "pos_std": ps,
                   "time_mean": tm, "time_std": ts, "delta_pos_time_mean": dm,
                   "delta_pos_time_std": ds, "per_seed": d}
            summary["tab_reg"].append(row)
            print(f"{prior:<7}{integ:<8}{fmt(pm,ps):>16}{fmt(tm,ts):>16}{fmt(dm,ds):>18}  {ps_str}")

    # ===== fine vs naive on regular (both encoders), Δ(naive - fine) =====
    print("\n" + "=" * 78)
    print("FINE vs NAIVE on regular eval (Δ=naive−fine; fine wins if Δ>0)")
    print(f"{'Prior':<7}{'Enc':<6}{'naive':>16}{'fine':>16}{'Δ(naive−fine)':>18}  fine-wins")
    print("-" * 86)
    for prior in PRIORS:
        for enc in ["pos", "time"]:
            nv = [load(enc, "naive", prior, s, "reg_s1") for s in SEEDS]
            fn = [load(enc, "fine", prior, s, "reg_s1") for s in SEEDS]
            d = paired_delta(nv, fn)
            nm, ns, _ = ms(nv); fm, fs, _ = ms(fn); dm, ds, _ = ms(d)
            wins = sum(1 for x in d if x > 0)
            summary["fine_vs_naive_reg"].append({"prior": prior, "enc": enc,
                "naive_mean": nm, "fine_mean": fm, "delta_mean": dm, "delta_std": ds,
                "fine_wins": wins, "n": len(d)})
            print(f"{prior:<7}{enc:<6}{fmt(nm,ns):>16}{fmt(fm,fs):>16}{fmt(dm,ds):>18}  {wins}/{len(d)}")

    # ===== tab:substeps  (mixed eval, time encoder): naive vs fine at s=1,8 =====
    print("\n" + "=" * 78)
    print("TABLE tab:substeps  (mixed eval, time encoder)  --  Δ=naive−fine, fine wins if Δ>0")
    print(f"{'Prior':<7}{'s_eval':<7}{'naive':>16}{'fine':>16}{'Δ(naive−fine)':>18}  fine-wins")
    print("-" * 86)
    refine = {}
    for prior in PRIORS:
        for cfg, lab in [("mixed_s1", "1"), ("mixed_s8", "8")]:
            nv = [load("time", "naive", prior, s, cfg) for s in SEEDS]
            fn = [load("time", "fine", prior, s, cfg) for s in SEEDS]
            d = paired_delta(nv, fn)
            nm, ns, _ = ms(nv); fm, fs, _ = ms(fn); dm, ds, _ = ms(d)
            wins = sum(1 for x in d if x > 0)
            refine.setdefault(prior, {})[lab] = dm
            summary["tab_substeps"].append({"prior": prior, "s_eval": lab,
                "naive_mean": nm, "fine_mean": fm, "delta_mean": dm, "delta_std": ds,
                "fine_wins": wins, "n": len(d)})
            print(f"{prior:<7}{lab:<7}{fmt(nm,ns):>16}{fmt(fm,fs):>16}{fmt(dm,ds):>18}  {wins}/{len(d)}")
    print("\n  Refinement (does fine's lead grow s_eval 1→8?):")
    for prior in PRIORS:
        d1, d8 = refine.get(prior, {}).get("1"), refine.get(prior, {}).get("8")
        if d1 is not None and d8 is not None:
            print(f"    {prior}: Δ {d1:+.4f} (s1) → {d8:+.4f} (s8)  [{'grows' if d8 > d1 else 'shrinks'}]")

    # ===== headline summaries =====
    reg_fine = summary["fine_vs_naive_reg"]
    sub = summary["tab_substeps"]
    print("\n" + "=" * 78)
    print("SUMMARY")
    fn_cells = reg_fine + sub
    mean_pos = sum(1 for r in fn_cells if r["delta_mean"] is not None and r["delta_mean"] > 0)
    robust = sum(1 for r in fn_cells if r["delta_mean"] is not None and r["delta_mean"] > r["delta_std"])
    perseed = sum(r["fine_wins"] for r in fn_cells)
    perseed_tot = sum(r["n"] for r in fn_cells)
    print(f"  fine-vs-naive: {mean_pos}/{len(fn_cells)} cells mean Δ>0; "
          f"{robust}/{len(fn_cells)} robust (mean>std); "
          f"{perseed}/{perseed_tot} per-seed comparisons fine wins")
    # encoder gap pattern (tab:reg): paper expects pos-time >0 in naive, ~0 in fine
    enc_naive = [r for r in summary["tab_reg"] if r["trained"] == "naive"]
    enc_fine = [r for r in summary["tab_reg"] if r["trained"] == "fine"]
    print("  encoder gap (pos−time):")
    for r in enc_naive + enc_fine:
        if r["delta_pos_time_mean"] is not None:
            print(f"    {r['prior']:<7} {r['trained']:<5} Δ={r['delta_pos_time_mean']:+.4f}±{r['delta_pos_time_std']:.4f}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
