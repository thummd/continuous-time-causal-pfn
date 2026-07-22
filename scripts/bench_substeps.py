#!/usr/bin/env python
"""Granularity-vs-cost micro-benchmark for the SDE-prior substep tier.

Measures how training-data-generation cost and model compute scale with
num_substeps (EM substeps per observation gap).  Key point being tested:
num_substeps enters ONLY the prior/data-gen (continuous_scm EM loop), not
the model -- so inference/deployment cost is invariant to the training grid.
"""
import time, json, statistics, sys, gc
import torch
from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.continuous import ContinuousDoOverTimePFN

DEVICE = sys.argv[1] if len(sys.argv) > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
SUBSTEPS = [1, 2, 4, 8, 16, 32]
N_TIME, N_WARM = 20, 3   # timed batches, warmup batches
BATCH = 32

def make_loader(ns):
    return ContinuousTemporalInterventionDataLoader(
        num_steps=N_TIME + N_WARM + 1, batch_size=BATCH, prior_mode="random",
        tscm_structure="back_door", n_min_prior=3, n_max_prior=8, edge_prob=0.3,
        hidden_prob=0.3, regime_prob=0.0, n_max=10, normalize=True,
        target_key="Y_true", n_queries=1, query_mode="single",
        theta_range=(0.1, 0.5), sigma_range=(0.2, 0.6), weight_scale=0.3,
        intervention_value_scale=2.0, intervention_window_frac=(0.1, 0.3),
        mechanism_kind="linear", p_neural=0.0, schedule="regular", dt=1.0,
        jitter=0.0, exp_rate=1.0, pair_mode="counterfactual", t_range=(60, 120),
        seed=12345, device=DEVICE, prefetch=0, num_substeps=ns,
    )

def time_datagen(ns):
    it = iter(make_loader(ns))
    for _ in range(N_WARM): next(it)
    if DEVICE.startswith("cuda"): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(N_TIME): b = next(it)
    if DEVICE.startswith("cuda"): torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N_TIME
    peak = torch.cuda.max_memory_allocated()/1e6 if DEVICE.startswith("cuda") else float("nan")
    del it; gc.collect()
    return dt, peak, b

# Build a model once; its fwd/bwd cost is independent of num_substeps.
def time_model(batch):
    cfg = dict(n_max=10, embed_size=128, n_encoder_layers=4, n_cross_attn_heads=4,
               context_window=128, n_mixer_layers=1, num_time_frequencies=64,
               head_type="quantile", n_buckets=1000)
    m = ContinuousDoOverTimePFN(**cfg).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
    for _ in range(3):  # warmup
        opt.zero_grad(); out=m(b); loss=m.head.loss(out, b["Y_true_norm"]); loss.backward(); opt.step()
    if DEVICE.startswith("cuda"): torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(10):
        opt.zero_grad(); out=m(b); loss=m.head.loss(out, b["Y_true_norm"]); loss.backward(); opt.step()
    if DEVICE.startswith("cuda"): torch.cuda.synchronize()
    return (time.perf_counter()-t0)/10

print(f"device={DEVICE}  batch={BATCH}  timed_batches={N_TIME}\n")
rows=[]; base=None; lastbatch=None
for ns in SUBSTEPS:
    dt, peak, b = time_datagen(ns); lastbatch=b
    if base is None: base=dt
    rows.append((ns, dt, dt/base, peak))
    print(f"  num_substeps={ns:3d}  datagen={dt*1000:8.1f} ms/batch  x{dt/base:5.2f} vs s1  peak_mem={peak:7.1f} MB")
mdt = time_model(lastbatch)
print(f"\n  model fwd+bwd+step: {mdt*1000:.1f} ms/batch  (INDEPENDENT of num_substeps)")
print("\n=== JSON ===")
print(json.dumps({"device":DEVICE,"batch":BATCH,
    "datagen_ms_per_batch":{str(ns):dt*1000 for ns,dt,_,_ in rows},
    "datagen_rel_to_s1":{str(ns):r for ns,_,r,_ in rows},
    "peak_mem_mb":{str(ns):p for ns,_,_,p in rows},
    "model_fwd_bwd_ms":mdt*1000}, indent=2))
