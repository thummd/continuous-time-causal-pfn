#!/usr/bin/env python
"""Aggregate grid_v4 time_* common-prior eval into multi-seed fine-vs-naive.
Reports, per (prior, eval-config), naive vs fine eval_loss (mean+/-std over
3 seeds) and the paired Delta(naive-fine) (fine wins when Delta>0), plus
per-seed sign consistency."""
import json, glob, statistics, sys
from collections import defaultdict

OUT="/home/dennis/repos/continuous-time-causal-pfn/results/grid_v4_synth"
def load(cell,seed,cfg):
    f=f"{OUT}/{cell}_seed{seed}_{cfg}.json"
    try: return json.load(open(f))["metrics"]["eval_loss"]
    except Exception: return None

CFGS=["reg_s1","reg_s8"]
PRIORS=["OU","neural"]
def ms(xs):
    xs=[x for x in xs if x is not None]
    if not xs: return (None,None,0)
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs)>1 else 0.0, len(xs))

print(f"{'prior':<7}{'cfg':<10}{'naive (mean±std)':>20}{'fine (mean±std)':>20}{'Δ(naive−fine) mean±std':>26}  per-seed Δ  (fine wins if Δ>0)")
print("-"*128)
summary={}
for prior in PRIORS:
    for cfg in CFGS:
        nv=[load(f"time_naive_{prior}",s,cfg) for s in (0,1,2)]
        fn=[load(f"time_fine_{prior}",s,cfg) for s in (0,1,2)]
        deltas=[(n-f) for n,f in zip(nv,fn) if n is not None and f is not None]
        nm,ns,_=ms(nv); fm,fs,_=ms(fn); dm,ds,_=ms(deltas)
        wins=sum(1 for d in deltas if d>0)
        summary[(prior,cfg)]=(dm,ds,wins,len(deltas))
        if nm is None or fm is None or dm is None:
            print(f"{prior:<7}{cfg:<10}  (incomplete: naive={nm} fine={fm})"); continue
        ps="["+", ".join(f"{d:+.4f}" for d in deltas)+"]"
        print(f"{prior:<7}{cfg:<10}{nm:>13.4f}±{ns:.4f}{fm:>13.4f}±{fs:.4f}{dm:>16.4f}±{ds:.4f}   {ps}  {wins}/{len(deltas)} fine")

# Headline: fine-vs-naive replication count
tot=sum(v[3] for v in summary.values()); finewins=sum(v[2] for v in summary.values())
cells_meanpos=sum(1 for v in summary.values() if v[0] is not None and v[0]>0)
cells_robust=sum(1 for v in summary.values() if v[0] is not None and v[0]>v[1])  # mean>std
print("\n=== Fine-vs-naive summary (time_* cells only) ===")
print(f"  per-seed: {finewins}/{tot} individual (prior,cfg,seed) comparisons have fine < naive")
print(f"  per-cell mean: {cells_meanpos}/{len(summary)} (prior,cfg) cells have mean Δ>0 (fine wins on average)")
print(f"  robust (mean Δ > std): {cells_robust}/{len(summary)} cells")
print("\n  s_eval refinement effect (mixed): does fine's lead grow s1->s8?")
for prior in PRIORS:
    d1=summary[(prior,'reg_s1')][0]; d8=summary[(prior,'reg_s8')][0]
    if d1 is not None and d8 is not None:
        print(f"    {prior}: Δ reg_s1={d1:+.4f} -> reg_s8={d8:+.4f}  ({'grows' if d8>d1 else 'shrinks'})")
