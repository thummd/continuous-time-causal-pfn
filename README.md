# Continuous-time Causal Prior Fitted Networks (ICML FMSD 2026)

> **This is a private fork of [do-over-time-pfn](https://github.com/thummd/do-over-time-pfn).**
> It hosts the development for the ICML 2026 *Foundation Models for Structured
> Data* workshop paper on **continuous-time causal PFNs**. The upstream
> DoT-PFN repo continues to be the home of the full NeurIPS 2026 submission.

## Fork relationship

| Aspect | DoT-PFN (upstream) | ct-cpfn (this fork) |
|---|---|---|
| Target venue | NeurIPS 2026 (full paper) | ICML FMSD 2026 workshop (4-page non-archival) |
| Scope | Discrete-time temporal causal PFN + identifiability benchmark | Continuous-time extension: SDE prior, Delta-t aware encoder, irregular-sampling real-world eval |
| GitHub | `thummd/do-over-time-pfn` | `thummd/continuous-time-causal-pfn` |
| Branch layout | `main`, `dennis` (latest s8) | `main`, `dennis` (mirror), `ct-dev` (workshop work) |

### Remotes

```bash
git remote -v
# origin    git@github.com:thummd/continuous-time-causal-pfn.git  (fetch + push)
# upstream  git@github.com:thummd/do-over-time-pfn.git             (fetch + push)
```

### Where new code goes

Continuous-time extensions live under `continuous/` subdirectories so that
they can be cleanly upstreamed into DoT-PFN later:

```
dotime/prior/continuous/   -> OU mechanism sampler, ContinuousSCM, Delta-t schedules
dotime/model/continuous/   -> Fourier time embeddings (Delta-t aware)
dotime/data/pk_pd/         -> Theophylline / Warfarin loaders (stub)
paper/icml_fmsd/           -> workshop paper draft (NeurIPS draft stays in paper/)
```

All pre-existing discrete-time code is unchanged.

### Current continuous-time module (phases 1 – 9)

End-to-end **training** pipeline: config → CLI entry → prior →
dataloader → model → loss → checkpoint.  The prior supports both
**named TSCM structures** (back_door, front_door, IV, ...) and a
**random-graph** continuous-time prior (variable N, random DAG, random
(A, Y) roles, configurable fraction of **hidden confounders** per
trajectory, and a **regime-switching** slice for non-stationary
dynamics).  Output head is **quantile** (pinball loss) or **bar**
(classification-as-regression over bucket borders calibrated from the
prior).

End-to-end **zero-shot evaluation** pipeline on three real-world
benchmarks:
- **Theophylline** (pharmacokinetics, 12 subjects × 11 irregular PK
  observations, 2 variables: dose → concentration).
- **Warfarin** (PK/PD, 32 subjects with irregular cp + PCA observations,
  3 variables: dose → concentration → PD outcome — a direct front-door
  structure match for the synthetic TSCM prior).
- **CausalChamber** (~10 Hz physical-system walks with known actuator
  interventions).

243/243 tests pass in `tests/`.

Quick start — train:

```bash
PYTHONPATH=. python scripts/ct_train.py \
    --config configs/continuous_default.yaml \
    --total-steps 5000 \
    --save-dir checkpoints/ct/back_door_cf_regular
```

Quick start — train on a random-graph prior (with or without hidden confounders):

```bash
# all-observed random-graph prior
PYTHONPATH=. python scripts/ct_train.py \
    --config configs/continuous_default.yaml \
    --prior-mode random \
    --n-min-prior 3 --n-max-prior 8 --edge-prob 0.3 \
    --total-steps 5000 \
    --save-dir checkpoints/ct/random_graph

# with ~30% of non-(A, Y) nodes hidden per trajectory
PYTHONPATH=. python scripts/ct_train.py \
    --config configs/continuous_default.yaml \
    --prior-mode random \
    --n-min-prior 4 --n-max-prior 8 --edge-prob 0.3 --hidden-prob 0.3 \
    --total-steps 5000 \
    --save-dir checkpoints/ct/random_graph_hidden
```

Quick start — evaluate zero-shot on Theophylline:

```bash
PYTHONPATH=. python scripts/ct_evaluate.py \
    --checkpoint checkpoints/ct/back_door_cf_regular/continuous_do_over_time_pfn_best.pt \
    --benchmark theophylline \
    --save-json results/theoph_zero_shot.json
```

Quick start — evaluate zero-shot on CausalChamber:

```bash
PYTHONPATH=. python scripts/ct_evaluate.py \
    --checkpoint checkpoints/ct/random_graph/continuous_do_over_time_pfn_best.pt \
    --benchmark causal_chamber \
    --chamber-dataset lt_walks_v1 \
    --chamber-query-var red \
    --chamber-max-episodes 50 \
    --save-json results/chamber_zero_shot.json
```

Quick start — evaluate zero-shot on Warfarin (dose → cp → PCA):

```bash
PYTHONPATH=. python scripts/ct_evaluate.py \
    --checkpoint checkpoints/ct/random_graph/continuous_do_over_time_pfn_best.pt \
    --benchmark warfarin \
    --save-json results/warfarin_zero_shot.json
```

Quick start — train with a bar-distribution head:

```bash
PYTHONPATH=. python scripts/ct_train.py \
    --config configs/continuous_default.yaml \
    --head-type bar --n-buckets 1000 \
    --total-steps 5000 \
    --save-dir checkpoints/ct/bar_head
```

Quick start — train with a regime-switching prior mix:

```bash
# Mirrors the discrete-time CTP's 15% regime-switching slice, plus
# hidden confounders for realism.
PYTHONPATH=. python scripts/ct_train.py \
    --config configs/continuous_default.yaml \
    --prior-mode random \
    --n-min-prior 3 --n-max-prior 8 --edge-prob 0.3 \
    --hidden-prob 0.2 --regime-prob 0.15 --regime-count-range 2 3 \
    --total-steps 5000 \
    --save-dir checkpoints/ct/regime_switching
```

See `configs/continuous_default.yaml` for the full list of knobs and
`scripts/ct_train.py --help` / `scripts/ct_evaluate.py --help` for CLI
overrides.

**Phase 1 — SDE primitives** (verified in `tests/test_continuous_prior.py`
and `tests/test_continuous_encoder.py`):

- **`dotime.prior.continuous.OUMechanism`** — single-variable linear-drift
  Ornstein-Uhlenbeck spec:
  `dX_v = (-theta_v * X_v + sum_u w_{v,u} * X_u) dt + sigma_v dW_v`.
  At `dt = 1` Euler-Maruyama recovers the AR(1) form of the discrete-time
  `batched_tscm.py` mechanism exactly.
- **`dotime.prior.continuous.ContinuousSCM`** — multivariate SCM that
  integrates a vector of `OUMechanism` on any observation schedule via
  Euler-Maruyama. Supports hard / soft / time-varying interventions plus
  `sample_counterfactual_pair` (shared noise, Pearl rung 3) and
  `sample_interventional_pair` (independent noise, DoT-PFN-compatible).
- **`dotime.prior.continuous.time_schedule`** — `regular_schedule`,
  `jittered_schedule`, `exponential_schedule`, `from_times`.
- **`dotime.model.continuous.FourierTimeEmbedding`** / **`DeltaTEmbedding`** —
  scale-free sinusoidal embeddings over time / log-Δt.

**Phase 2 — end-to-end pipeline** (verified in
`tests/test_continuous_integration.py`):

- **`dotime.prior.continuous.ContinuousTSCMSampler`** — reuses the 8 named
  `TSCMStructure` topologies from `dotime.prior.tscm_sampler` (back-door,
  front-door, IV, RCT, mediator, confounder+mediator, observed /
  unobserved confounder) with OU mechanisms. Both instantaneous and
  lagged parent edges from the discrete DAG collapse into the continuous
  SCM's single parent set, since Euler-Maruyama advances all variables
  simultaneously on pre-step parent values.
- **`dotime.prior.continuous.ContinuousExtendedPrior`** — model-ready
  batch generator (analogue of `ExtendedCausalTimePrior`). Returns the
  full discrete-time batch dict plus two new fields:
  - `times: (T,)` absolute observation times.
  - `dts: (T-1,)` inter-observation gaps.
  - `t_int_start`, `t_int_end`, `t_query` in absolute time units.
  Selects between counterfactual and interventional pair semantics via
  `pair_mode`. Applies the same canonical `A → 0, Y → N-1` permutation
  as the discrete `TSCMPrior` so downstream evaluation code works
  unchanged.
- **`dotime.data.continuous_dataloader.ContinuousTemporalInterventionDataLoader`** —
  infinite on-the-fly loader (analogue of
  `TemporalInterventionDataLoader`). Supports background prefetching
  and per-variable z-score normalisation via the existing
  `normalize_batch` helper (unchanged for continuous batches).
- **`dotime.model.continuous.ContinuousTemporalEncoder`** — subclasses
  `TemporalEncoder`, overriding the forward to replace the learnable
  `rel_pos_encoding` with `FourierTimeEmbedding(times - t_int_start)`.
  Truncation to `context_window` pre-intervention observations is
  identical to the discrete case (index-based, not time-based).
- **`dotime.model.continuous.ContinuousDoOverTimePFN`** — drop-in
  variant of `DoOverTimePFN` that swaps the encoder and overrides
  `encode()` to pass `times` / `t_int_start` / `int_onset_idx` from
  the batch dict. Everything else — the cross-variable mixer, the
  output heads, `loss` / `predict` / `forward` — is inherited
  unchanged.

**Phase 3 — training pipeline** (verified in
`tests/test_continuous_phase3.py`):

- **Soft + time-varying interventions** in `ContinuousExtendedPrior`.
  Sampled via `intervention_kind_probs=(p_hard, p_soft, p_tv)`; the
  SCM already supported all three kinds, the extended prior now
  dispatches accordingly.  Time-varying profiles
  (`_StepProfile`, `_RampProfile`, `_SineProfile`) are picklable
  dataclasses at module level so multiprocessing dataloaders still
  work.
- **Positivity-aware intervention clipping** via
  `intervention_source="positivity_aware"`.  After simulating the
  observational trajectory, the hard-intervention value is clipped to
  the 3-sigma band around the pre-intervention target variable.  Soft
  and time-varying interventions are no-ops.
- **`dotime.training.continuous_trainer.train_continuous`** — training
  loop for `ContinuousDoOverTimePFN`.  Cosine-with-warmup LR, AdamW,
  gradient clipping, background prefetch, early stopping, optional
  Wandb logging.  Quantile head only (see module docstring for why).
- **`configs/continuous_default.yaml`** — full YAML config surface
  covering model, prior, and training hyperparameters.
- **`scripts/ct_train.py`** — CLI entry that loads the config, applies
  overrides for the common ablation knobs
  (`--schedule`, `--pair-mode`, `--tscm-structure`,
  `--intervention-kind-probs`, `--intervention-source`, ...), and
  calls `train_continuous`.

**Phase 4 — zero-shot Theophylline PK evaluation** (verified in
`tests/test_continuous_pk_pd.py`):

- **`dotime.data.pk_pd.theophylline.load_theophylline`** — reads the
  bundled CSV at `data/pk_pd/theophylline.csv` and returns 12
  `TheophSubject` records with per-subject times (hours) and
  concentrations (mg/L). Offline — no network required.
- **`dotime.data.pk_pd.theophylline_adapter.build_theophylline_batch`** —
  maps a subject record to the `ContinuousDoOverTimePFN` batch dict.
  Models dosing as a hard intervention on variable 0 (Dose) with a
  short absorption window; queries concentration (variable 1) at each
  post-dose observation time. Supports two normalisation strategies
  (`peek_target` for zero-shot scale alignment, `fixed` for
  calibration against an external reference).
- **`dotime.eval.continuous_pk_eval.evaluate_dataset`** — runs a
  trained model over all 12 subjects, denormalises predictions back to
  mg/L, and reports per-subject RMSE / MAE / Pearson r plus aggregate
  metrics including **lift over a naive mean-concentration baseline**.
- **`scripts/ct_evaluate.py`** — CLI that loads a checkpoint and
  writes a human-readable summary and optional JSON results file.

**Phase 5 — random-graph prior + CausalChamber evaluation** (verified
in `tests/test_continuous_phase5.py`):

- **`dotime.prior.continuous.RandomContinuousSCMSampler`** — samples a
  fresh :class:`ContinuousSCM` per trajectory with random N in
  ``[n_min, n_max_prior]``, Erdos-Renyi edges over the topological
  order, and per-sample (A, Y) roles.  Analogous to CausalTimePrior's
  random-graph path in the discrete-time pipeline.
- **`dotime.prior.continuous.RandomContinuousExtendedPrior`** —
  drop-in replacement for :class:`ContinuousExtendedPrior` that plugs
  the random sampler into the existing generate_sample contract.
  :class:`ContinuousExtendedPrior` was refactored to expose a
  ``_sample_scm_context`` hook so both samplers share the
  schedule / intervention / query logic.
- **`--prior-mode random`** on the training CLI and
  ``prior.mode: random`` in `configs/continuous_default.yaml` switch
  the dataloader between named-TSCM and random-graph training.
- **`dotime.data.causal_chamber_ct.build_causal_chamber_batch`** —
  thin adapter over :class:`CausalChamberLoader` episodes.  Stitches
  ``X_obs + X_post`` into a single trajectory, lays ``times`` on a
  uniform ``dt_seconds`` grid (default 10 Hz), sets
  ``int_onset_idx`` at the changepoint, and emits a batch dict
  compatible with :class:`ContinuousDoOverTimePFN`.
- **`dotime.eval.continuous_chamber_eval.evaluate_episodes`** — runs
  the model on a list of chamber episodes, denormalises predictions
  back to raw sensor units, reports per-episode + aggregate RMSE /
  MAE / Pearson r, and computes **lift over a naive last-value
  baseline**.
- **`scripts/ct_evaluate.py --benchmark causal_chamber`** — CLI entry
  that loads CausalChamber data via ``load_chamber_episodes`` and runs
  zero-shot evaluation with the same output format as Theophylline.

**Phase 6 — hidden-variable confounders** (verified in
`tests/test_continuous_phase6.py`):

- **Bug fix**: :class:`ContinuousExtendedPrior` now correctly masks
  hidden variables in both ``X_obs`` and ``X_int`` and zeros the
  corresponding ``variable_mask`` entries.  Previously, structures
  like ``front_door`` and ``instrumental_variable`` exposed the
  hidden ``U``'s trajectory to the model, silently defeating the
  purpose of the identifiability benchmark.  All five TSCM structures
  with a hidden ``U`` now fully hide it from the encoder's view
  while still letting ``U`` drive the dynamics of observed variables.
- **`hidden_prob` on the random-graph prior**
  (:class:`RandomContinuousSCMSampler`, :class:`RandomContinuousExtendedPrior`):
  each non-(A, Y) node is independently marked hidden with this
  probability.  ``A`` (treatment) and ``Y`` (outcome) are never
  hidden.  Hidden nodes still participate in the OU simulation, so
  their causal influence on observed variables remains in the
  training data -- only the *observations* of the hidden node are
  stripped.
- **CLI / config plumbing**: new ``prior.hidden_prob`` key in
  ``configs/continuous_default.yaml`` and ``--hidden-prob`` flag on
  ``scripts/ct_train.py``.
- **Query sampler** already respected hidden variables from phase 5
  (``_sample_queries`` excludes them as query targets).  Phase 6
  added a regression test that catches any future regression there.

**Phase 7 — Warfarin PK/PD + bar-distribution head** (verified in
`tests/test_continuous_warfarin.py` and `tests/test_continuous_bar_head.py`):

- **`dotime.data.pk_pd.warfarin.load_warfarin`** — reads the bundled
  `data/pk_pd/warfarin.csv` (ported from `nlmixr2data`) and returns 32
  `WarfarinSubject` records.  Each subject has 1 oral dose at t=0, an
  irregular concentration trajectory (``cp``), and an irregular
  PCA/pharmacodynamic trajectory.  Duplicate same-timestamp replicates
  are averaged so the per-stream time axes stay strictly increasing.
- **`dotime.data.pk_pd.warfarin_adapter.build_warfarin_batch`** —
  3-variable adapter (canonical ``0 = Dose, 1 = cp, 2 = pca``) on the
  per-subject union time grid.  Each observed ``(variable, time)`` pair
  becomes one query entry, so the batch carries ``n_cp + n_pca`` queries
  per subject and exercises both the PK mediator and the PD outcome in
  one pass.
- **`dotime.eval.continuous_warfarin_eval.evaluate_warfarin_dataset`** —
  reports per-subject and aggregate RMSE / MAE / Pearson r **split by
  variable** (cp vs pca) plus lift over a per-subject mean-value naive
  baseline for each stream.
- **`scripts/ct_evaluate.py --benchmark warfarin`** — CLI entry point.

- **Bar-distribution head for continuous-time** now available in
  :class:`ContinuousDoOverTimePFN` via ``head_type="bar"``.  The
  continuous trainer calibrates bucket borders from the prior at
  training start and stores them in the checkpoint so
  ``ct_evaluate.py`` can rehydrate the head on load.  Calibration
  reuses the existing discrete-time ``calibrate_bar_distribution``
  helper unchanged.  New CLI flags ``--head-type {quantile,bar}`` and
  ``--n-buckets N`` on ``scripts/ct_train.py`` plus matching
  ``model.head_type`` / ``model.n_buckets`` keys in
  ``configs/continuous_default.yaml``.

**Phase 8 — regime-switching continuous-time prior** (verified in
`tests/test_continuous_phase8.py`):

- **`dotime.prior.continuous.ContinuousRegimeSwitchingSCM`** — holds
  ``R`` independent :class:`ContinuousSCM` regimes sharing the same
  variable count and a sticky Markov transition matrix over regime
  state.  ``simulate(times, dts, noise, regime_trajectory)`` returns
  the same ``(times, trajectory)`` shape as :class:`ContinuousSCM` so
  it is a drop-in replacement in the extended prior's dispatch.  The
  regime trajectory is pre-sampled once per pair and **shared with the
  noise realisation** between observational and counterfactual runs,
  keeping the "identical before intervention" property intact across
  regime switches.
- **`sample_sticky_transition_matrix(n_regimes, sticky_alpha=9.0,
  other_alpha=0.5)`** — Dirichlet-sampled transition matrix whose
  diagonal is typically ~0.9 (expected regime duration ~10
  observation steps), matching the discrete-time CTP convention.
- **`RandomContinuousSCMSampler`** gained ``regime_prob``,
  ``regime_count_range``, ``sticky_alpha``, ``other_alpha`` parameters.
  With probability ``regime_prob`` the returned SCM is a
  :class:`ContinuousRegimeSwitchingSCM`; the rest of the interface
  (``(scm, n_vars, A_topo, Y_topo, hidden_topo)``) is unchanged.
  ``ContinuousExtendedPrior.generate_sample`` automatically pre-samples
  a shared regime trajectory whenever the SCM exposes
  ``_draw_regime_trajectory``, so counterfactual pair mode works for
  both stationary and regime-switching samples uniformly.
- **Plumbing**: new ``--regime-prob`` and ``--regime-count-range`` CLI
  flags on ``scripts/ct_train.py``, new ``prior.regime_prob`` and
  ``prior.regime_count_range`` keys in
  ``configs/continuous_default.yaml``.

**Phase 9 — seeded Marsaglia-Tsang Gamma + Dirichlet** (verified in
`tests/test_continuous_phase9.py`):

- **`_sample_gamma_seeded(alphas, generator)`** in
  ``dotime.prior.continuous.regime_switching`` — element-wise Gamma
  sampler driven entirely by :meth:`torch.Tensor.normal_` and
  :meth:`torch.Tensor.uniform_` so it honours a
  :class:`torch.Generator` bit-for-bit.  Marsaglia-Tsang (2000)
  rejection for ``alpha >= 1``, combined with the shape-augmentation
  identity ``Gamma(a) = Gamma(a + 1) * U^{1/a}`` for ``a < 1``.
  Typical acceptance rate > 99% per iteration.
- **`_sample_dirichlet_seeded(alphas, generator)`** — Dirichlet row
  via Gamma normalisation; seeded by the same generator.
- **`sample_sticky_transition_matrix`** now uses the seeded path when
  a generator is supplied, replacing the Wilson-Hilferty
  approximation introduced in phase 8.  Wilson-Hilferty was visibly
  biased for the default ``other_alpha=0.5`` off-diagonal (negative
  "Gamma" samples clamped to zero after the cube step), which skewed
  Dirichlet rows toward the diagonal.  Marsaglia-Tsang recovers the
  correct moments:
  - ``Gamma(alpha=0.5)`` empirical mean 0.501 vs target 0.5, variance
    0.51 vs target 0.5 (20 000 samples).
  - Sticky transition matrix diagonal mean 0.9 (matches the
    theoretical ``9/(9+0.5+0.5)``).

### Not yet implemented (phase 10+)

- DREAM4 (or similar) in-silico gene-regulatory-network benchmark as
  a third synthetic-but-realistic eval dataset.
- GatedDeltaProduct continuous-time encoder backend (currently only
  the Transformer backend is wired through the `times` / `t_int_start`
  path end to end).

### Quick example: end-to-end training (Python API)

```python
import torch
from dotime.training.continuous_trainer import train_continuous

model = train_continuous(
    # Model
    n_max=16, embed_size=128, n_encoder_layers=2,
    context_window=64, num_time_frequencies=32,
    tau_levels=[0.1, 0.5, 0.9],
    # Prior
    tscm_structure="front_door",
    schedule="jittered",
    dt=1.0, jitter=0.3,
    pair_mode="counterfactual",
    intervention_kind_probs=(0.5, 0.3, 0.2),  # hard / soft / time-varying
    intervention_source="positivity_aware",
    t_range=(50, 100),
    # Training
    batch_size=16, total_steps=5000,
    eval_every=500, save_dir="checkpoints/ct/front_door",
)
```

### CLI equivalent

```bash
PYTHONPATH=. python scripts/ct_train.py \
    --config configs/continuous_default.yaml \
    --tscm-structure front_door \
    --schedule jittered \
    --pair-mode counterfactual \
    --intervention-kind-probs 0.5 0.3 0.2 \
    --intervention-source positivity_aware \
    --total-steps 5000 \
    --save-dir checkpoints/ct/front_door
```

### SDE primitives (phase 1 standalone)

```python
import torch
from dotime.prior.continuous import (
    ContinuousSCM, ContinuousIntervention, InterventionKind,
    regular_schedule, exponential_schedule,
)

scm = ContinuousSCM.sample_random(
    n_vars=5, edge_prob=0.3,
    theta_range=(0.5, 2.0), sigma_range=(0.2, 0.5),
    generator=torch.Generator().manual_seed(0),
)
times, dts = regular_schedule(T=100, dt=0.1)

intv = ContinuousIntervention(
    target=0, t_start=3.0, t_end=6.0, kind=InterventionKind.HARD, value=2.5,
)
times, X_obs, X_cf = scm.sample_counterfactual_pair(times, dts, intv)
# Before t_start: X_obs == X_cf (same noise, same mechanism).
# Inside window:  X_cf[:, 0] clamped to 2.5.
# After window:   X_cf diverges from X_obs only through causal propagation.
```

### LFS checkpoints

This fork was cloned **without LFS blobs**. The `.gitattributes` LFS
pointers are present, but the underlying checkpoint files are only stored
in the upstream DoT-PFN LFS server. If you need a DoT-PFN checkpoint for
baselines:

```bash
# One-off pull of specific file from upstream LFS
git lfs pull upstream --include "checkpoints/s8_*/do_over_time_pfn_best.pt"
```

New checkpoints produced during continuous-time experiments should be
committed to this fork's own LFS (omit `lfs.allowincompletepush=true` for
those commits).

### Syncing with DoT-PFN

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. In short:

```bash
git fetch upstream
git checkout main && git merge upstream/main && git push origin main
git checkout ct-dev && git merge main   # pull NeurIPS changes into workshop branch
```

---

## Upstream Do-Over-Time-PFN README

*(Everything below is inherited from DoT-PFN.)*

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
