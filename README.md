# Do-Over-Time-PFN

In-context causal effect estimation for temporal data using Prior-Fitted Networks.

## Overview

Do-Over-Time-PFN is a neural network that estimates interventional distributions `P(Y | do(X), history)` from observational time series — without retraining per dataset. It is trained once on synthetic data from many temporal causal structures, then performs causal inference in-context at test time.

### Architecture

The model has three stages that mirror the causal identification pipeline:

1. **Temporal Encoder** — Encodes each variable's time series independently via Transformer or GatedDeltaProduct (from TempoPFN). Produces per-variable temporal representations `h_vars: (B, N, E)`.

2. **Cross-Variable Mixer** — Combines per-variable representations with intervention and query specifications via cross-attention. The model learns which variables are causally relevant (e.g., adjusting for confounders via back-door, tracing through mediators via front-door).

3. **Output Head** — Two options:
   - **QuantileHead** (recommended): Direct quantile predictions via pinball loss at 5 levels [0.1, 0.25, 0.5, 0.75, 0.9]. No calibration needed, simpler, more robust.
   - **BarDistributionHead**: Full predictive distribution over ~1000 buckets (classification-as-regression, from PFN). Requires bucket border calibration.

### Identification Strategy

Given observational history `H_{0:t-1}` and an intervention `do(a_t)`, the model implicitly performs the identification:

```
p(y_t | do(a_t), H_{t-1}) = integral p(y_t | a_t, x_t) p(x_t | x_{t-1}) dx_t
```

The temporal encoder learns `p(x_t | x_{t-1})` from each variable's history; the cross-variable mixer learns the adjustment/marginalization; and the output head produces the result.

## Project Structure

```
dotime/
  model/
    do_over_time_pfn.py      # Main model (3-stage, head_type="bar"|"quantile")
    encoder.py               # Temporal encoder (Transformer / GatedDeltaProduct)
    cross_variable_mixer.py  # Cross-attention causal reasoning
    quantile_head.py         # Pure quantile predictions via pinball loss
    bar_head.py              # Bar distribution output (legacy, for existing checkpoints)
    pinball_loss.py          # Differentiable quantile extraction and pinball loss
  data/
    temporal_dataloader.py   # On-the-fly data generation with prefetching
    causal_chamber.py        # CausalChamber dataset integration
    normalization.py         # Per-variable z-score normalization with clamping
  prior/
    extended_prior.py        # CausalTimePrior wrapper (random graphs for training)
    tscm_sampler.py          # 8 named causal structures for identifiability testing
  training/
    trainer.py               # Training loop (AdamW, cosine LR, quantile/bar loss)
  eval/
    metrics.py               # RMSE, MAE, quantile calibration, pinball metrics
    evaluate_chamber.py      # CausalChamber evaluation

scripts/
  train.py                   # Training entry point
  evaluate.py                # Model evaluation on CausalChamber
  tscm_identifiability.py    # TSCM identifiability case studies
  pinball_comparison.py      # A/B comparison: bar-only vs bar+pinball
  plot_tscm_structures.py    # Publication-quality DAG visualizations
  run_full_scale.sh          # Full-scale server training launcher

configs/
  default.yaml               # Full config (512-dim, 10 layers, 1000 buckets, 100K steps)
  server.yaml                # Server config (6 layers, batch_size=16 for shared A100)
  fast_comparison.yaml       # Reduced config for quick experiments

tests/                       # 33 tests covering model, quantile head, prior, dataloader
```

## TSCM Identifiability Case Studies

The TSCM sampler generates 8 specific causal structures to test whether the model correctly handles each identification strategy:

| Structure | Nodes | Identification | Strategy |
|-----------|-------|----------------|----------|
| Observed Confounder | Z, X, Y | Identifiable | Back-door adjustment through Z |
| Back-Door | Z, X, Y | Identifiable | Z satisfies back-door criterion |
| Mediator | X, M, Y | Identifiable | Trivial (no confounding) |
| Confounder + Mediator | Z, X, M, Y | Identifiable | Hybrid: back-door + front-door |
| Front-Door | **U**, X, M, Y | Identifiable | Front-door criterion through M |
| Instrumental Variable | **U**, Z, A, Y | Identifiable | Z is instrument for A->Y |
| RCT (No Confounding) | A, Y | Identifiable | Trivially: p(Y\|do(A)) = p(Y\|A) |
| Unobserved Confounder | **U**, X, Y | **Not identifiable** | Tests model robustness |

**Bold** = unobserved variable. See `figures/tscm_structures.png` for DAG visualizations.

### Evaluation metrics

- **Per-TSCM error**: RMSE, MAE, direction accuracy tracked per individual SCM sample
- **Causal effect RMSE**: Error on the treatment effect (X_int - X_obs), not just the raw prediction
- **Confounding strength**: Mean absolute correlation between intervention target and other variables
- **Effect decay**: Query at multiple offsets after intervention (`--query-offsets 1 2 3 5 10`) to study how the causal effect propagates through lagged dependencies

## Ground Truth and Evaluation Philosophy

Our two evaluation settings have fundamentally different ground truths:

**Synthetic TSCMs**: Ground truth is exact. We generate data from a known SCM, so the true interventional outcome `X_int[t, var]` and causal effect `X_int - X_obs` are available by construction. The oracle is the SCM simulator itself (RMSE = 0). The model's goal is to recover this from observational data alone.

**CausalChamber (real-world)**: Ground truth is the actual sensor reading after a physical intervention. There is no analytical oracle — the true causal mechanisms (optics, electronics) are complex. Crucially, we only observe the factual outcome, not the counterfactual "what would have happened without the intervention." A naive time-series model can score well by extrapolating trends without any causal reasoning.

To distinguish causal understanding from mere prediction, we report **confounding-aware metrics**:
- **Naive RMSE**: Predicting the last observational value (assumes nothing changes)
- **Lift over naive**: How much better the model is than this baseline. A model with genuine causal understanding should show positive lift, especially for large interventions.
- **Effect-error correlation**: Correlation between prediction error and intervention magnitude. High correlation suggests the model struggles when interventions are strong — a sign of confounding bias.

## Training Modes

### Quantile head (recommended)

```bash
python scripts/train.py --config configs/server.yaml --head-type quantile --device cuda:0
```

No bucket calibration needed. Outputs 5 quantile predictions directly via pinball loss.

### Causal effect target

```bash
python scripts/train.py --head-type quantile --target-key Y_causal_effect --device cuda:0
```

Trains the model to predict `Y_int - Y_obs` (the treatment effect) instead of the raw interventional value.

### Observational-only ablation

```bash
python scripts/train.py --head-type quantile --observational-only --device cuda:0
```

Zeros out all intervention context (target, type, value, timing) to create a predictive baseline. Demonstrates the value gained from causal/interventional information.

### Bar distribution head (legacy)

```bash
python scripts/train.py --config configs/default.yaml --head-type bar --device cuda:0
```

## Dependencies

- [pfns](https://pypi.org/project/pfns/) — Bar distribution implementation (only needed for bar head)
- [dopfnprior](https://github.com/) — Do-PFN prior utilities (install from `ctp/Do-PFN-prior`)
- [causal_time_prior](https://github.com/) — Temporal SCM generation (add `ctp/` to `PYTHONPATH`)
- [CausalChamber](https://pypi.org/project/causalchamber/) — Real-world evaluation dataset
- PyTorch >= 2.0

## Quick Start

### Install

```bash
pip install -e .
pip install pfns
pip install -e /path/to/ctp/Do-PFN-prior
export PYTHONPATH=/path/to/ctp:$PYTHONPATH
```

### Train

```bash
# Quick test (500 steps, small model)
python scripts/train.py --config configs/fast_comparison.yaml --head-type quantile --device cuda:0

# Full training (100K steps, quantile head)
python scripts/train.py --config configs/server.yaml --head-type quantile --device cuda:0
```

### Evaluate

```bash
# TSCM identifiability (verify data generation for all 8 structures)
python scripts/tscm_identifiability.py --verify-only

# TSCM identifiability (evaluate trained model)
python scripts/tscm_identifiability.py --checkpoint checkpoints/model_best.pt --device cuda:0

# Effect decay analysis (query at multiple offsets after intervention)
python scripts/tscm_identifiability.py --checkpoint checkpoints/model_best.pt \
    --query-offsets 1 2 3 5 10

# CausalChamber evaluation
python scripts/evaluate.py --checkpoint checkpoints/model_best.pt --device cuda:0
```

### Tests

```bash
pytest tests/ -v  # 33 tests
```
