# Do-Over-Time-PFN

In-context causal effect estimation for temporal data using Prior-Fitted Networks.

## Overview

Do-Over-Time-PFN is a neural network that estimates interventional distributions `P(Y | do(X), history)` from observational time series — without retraining per dataset. It is trained once on synthetic data from many temporal causal structures, then performs causal inference in-context at test time.

### Architecture

The model has three stages that mirror the causal identification pipeline:

1. **Temporal Encoder** — Encodes each variable's time series independently via Transformer or GatedDeltaProduct (from [TempoPFN](https://github.com/)). Produces per-variable temporal representations `h_vars: (B, N, E)`.

2. **Cross-Variable Mixer** — Combines per-variable representations with intervention and query specifications via cross-attention. The model learns which variables are causally relevant (e.g., adjusting for confounders via back-door, tracing through mediators via front-door).

3. **Bar Distribution Head** — Outputs a full predictive distribution over ~1000 buckets (classification-as-regression, from [PFN](https://github.com/automl/PFNs)). Supports an optional auxiliary pinball loss for quantile robustness.

### Identification Strategy

Given observational history `H_{0:t-1}` and an intervention `do(a_t)`, the model implicitly performs the identification:

```
p(y_t | do(a_t), H_{t-1}) = integral p(y_t | a_t, x_t) p(x_t | x_{t-1}) dx_t
```

The temporal encoder learns `p(x_t | x_{t-1})` from each variable's history; the cross-variable mixer learns the adjustment/marginalization; and the bar distribution head outputs the result as a discretized distribution.

## Project Structure

```
dotime/
  model/
    do_over_time_pfn.py   # Main model (3-stage architecture)
    encoder.py            # Temporal encoder (Transformer / GatedDeltaProduct)
    cross_variable_mixer.py  # Cross-attention causal reasoning
    bar_head.py           # Bar distribution output + pinball loss integration
    pinball_loss.py       # Differentiable quantile extraction and pinball loss
  data/
    temporal_dataloader.py  # On-the-fly data generation from causal structures
    causal_chamber.py       # CausalChamber dataset integration
    normalization.py        # Per-variable z-score normalization
  prior/
    extended_prior.py     # CausalTimePrior wrapper (random graphs for training)
    tscm_sampler.py       # 6 named causal structures for identifiability testing
  training/
    trainer.py            # Training loop (AdamW, cosine LR, bar + pinball loss)
  eval/
    metrics.py            # RMSE, MAE, quantile calibration, pinball metrics
    evaluate_chamber.py   # CausalChamber evaluation

scripts/
  train.py                # Training entry point
  evaluate.py             # Model evaluation on CausalChamber
  pinball_comparison.py   # A/B comparison: bar-only vs bar+pinball
  tscm_identifiability.py # TSCM identifiability case studies
  plot_tscm_structures.py # Publication-quality DAG visualizations
  run_full_scale.sh       # Full-scale server training launcher

configs/
  default.yaml            # Full config (512-dim, 10 layers, 1000 buckets, 100K steps)
  fast_comparison.yaml    # Reduced config for quick experiments

tests/
  test_model.py           # Forward pass, loss, gradients
  test_pinball.py         # Quantile extraction, pinball loss (13 tests)
  test_prior.py           # TSCM sampler, extended prior
  test_dataloader.py      # Data generation, normalization
```

## TSCM Identifiability Case Studies

The TSCM sampler generates 6 specific causal structures to test whether the model correctly handles each identification strategy:

| Structure | Nodes | Identification |
|-----------|-------|----------------|
| Observed Confounder | Z, X, Y | Back-door adjustment through Z |
| Mediator | X, M, Y | Causal effect traced through M |
| Confounder + Mediator | Z, X, M, Y | Hybrid: adjust for Z + trace through M |
| Unobserved Confounder | **U**, X, Y | Not point-identifiable (tests model limits) |
| Back-Door | Z, X, Y | Z satisfies back-door criterion |
| Front-Door | **U**, X, M, Y | Front-door criterion through M |

**Bold** = unobserved variable. See `figures/tscm_structures.png` for DAG visualizations.

Each structure uses random mechanisms (weights, activations) but a fixed graph topology, testing whether the model learned the identification strategy rather than overfitting to specific functional forms.

## Pinball Loss

The optional auxiliary pinball loss improves quantile calibration of the bar distribution output:

```
L_total = L_bar + pinball_weight * L_pinball
```

- `L_bar`: Cross-entropy over discretized buckets (primary objective)
- `L_pinball`: Asymmetric quantile loss at levels [0.1, 0.25, 0.5, 0.75, 0.9]
- `pinball_weight`: 0 = disabled (default), 0.1 = recommended starting point

Quantiles are extracted differentiably from the bar distribution logits via soft bucket selection over the CDF.

## Dependencies

- [pfns](https://pypi.org/project/pfns/) — Bar distribution implementation
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
python scripts/train.py --config configs/fast_comparison.yaml --device cuda:0

# Full training (100K steps)
python scripts/train.py --config configs/default.yaml --device cuda:0
```

### Evaluate

```bash
# TSCM identifiability (verify data generation)
python scripts/tscm_identifiability.py --verify-only

# TSCM identifiability (evaluate trained model)
python scripts/tscm_identifiability.py --checkpoint checkpoints/model_best.pt --device cuda:0

# CausalChamber evaluation
python scripts/evaluate.py --checkpoint checkpoints/model_best.pt --device cuda:0
```

### Pinball comparison

```bash
# Train baseline vs pinball-augmented and compare
python scripts/pinball_comparison.py --config configs/fast_comparison.yaml --device cuda:0
```

### Tests

```bash
pytest tests/ -v
```
