"""Phase 13b tests: zero-context training augmentation.

The PK adapters (``build_theophylline_batch``, ``build_warfarin_batch``)
emit batches with ``int_onset_idx == 0`` because pharmacokinetic
datasets have no pre-dose observations.  Pre-Phase-13b training never
saw such samples, so the cross-variable mixer was extrapolating on
those benchmarks.  Phase 13b adds a ``p_no_context`` knob to
:class:`ContinuousExtendedPrior` that, with the configured probability,
forces a training sample's intervention to start at ``times[0]``, so
the encoder runs on an empty pre-intervention window -- matching the
PK eval regime.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dotime.prior.continuous import ContinuousExtendedPrior


def _mk_prior(p_no_context: float, seed: int = 0) -> ContinuousExtendedPrior:
    return ContinuousExtendedPrior(
        tscm_structure="back_door",
        n_max=8,
        t_range=(20, 20),
        schedule="regular",
        dt=1.0,
        intervention_kind_probs=(1.0, 0.0, 0.0),  # hard only for stable behaviour
        intervention_window_frac=(0.1, 0.3),
        p_no_context=p_no_context,
        seed=seed,
    )


# ---------------------------------------------------------------------- validation


def test_rejects_invalid_p_no_context():
    with pytest.raises(ValueError):
        _mk_prior(p_no_context=-0.1)
    with pytest.raises(ValueError):
        _mk_prior(p_no_context=1.5)


def test_default_p_no_context_is_zero():
    prior = _mk_prior(p_no_context=0.0)
    assert prior.p_no_context == 0.0


# ---------------------------------------------------------------------- behaviour


def test_p_no_context_zero_never_emits_zero_context_samples():
    """With p_no_context=0 we recover the pre-Phase-13b distribution.

    Empirically check by sampling many trajectories and confirming
    none of them start with ``int_onset_idx == 0``.
    """
    prior = _mk_prior(p_no_context=0.0, seed=42)
    onsets = []
    for _ in range(50):
        sample = prior.generate_sample(T=20, n_queries=1)
        onsets.append(int(sample["int_onset_idx"].item()))
    # The default earliest_start is 30% into the trajectory, so no
    # sample should land at index 0.
    assert min(onsets) > 0, f"unexpected zero-context sample at p_no_context=0: onsets={onsets[:5]}"


def test_p_no_context_one_always_emits_zero_context_samples():
    prior = _mk_prior(p_no_context=1.0, seed=43)
    for _ in range(20):
        sample = prior.generate_sample(T=20, n_queries=1)
        assert int(sample["int_onset_idx"].item()) == 0
        # And the t_int_start_norm field is exactly 0 in this regime.
        assert float(sample["intervention_time_start"].item()) == pytest.approx(0.0, abs=1e-6)


def test_p_no_context_half_emits_mixture():
    """At p_no_context=0.5 we should see roughly half zero-context and half rich-context."""
    prior = _mk_prior(p_no_context=0.5, seed=44)
    n_zero = 0
    N = 200
    for _ in range(N):
        sample = prior.generate_sample(T=20, n_queries=1)
        if int(sample["int_onset_idx"].item()) == 0:
            n_zero += 1
    # With N=200 the binomial 99% CI for p=0.5 is roughly +/- 0.09.
    frac = n_zero / N
    assert 0.40 <= frac <= 0.60, f"expected ~50% zero-context at p=0.5, got {frac:.3f}"


# ---------------------------------------------------------------------- semantic invariants


def test_zero_context_sample_has_empty_pre_window_in_X_obs():
    """When ``int_onset_idx == 0`` the pre-intervention window of
    ``X_obs`` is the entire first row, which the causal-masking step
    zeroes.  After masking, ``X_obs`` is all zeros."""
    prior = _mk_prior(p_no_context=1.0, seed=45)
    sample = prior.generate_sample(T=20, n_queries=1)
    X_obs = sample["X_obs"]
    n_vars = int(sample["num_vars"].item())
    assert torch.all(X_obs[:, :n_vars] == 0.0), "X_obs should be all zeros under zero-context"


def test_zero_context_sample_keeps_non_trivial_X_int():
    """The interventional trajectory must still respond to the
    intervention -- zero-context only zeros ``X_obs`` (the encoder's
    view), not the simulator output."""
    prior = _mk_prior(p_no_context=1.0, seed=46)
    sample = prior.generate_sample(T=20, n_queries=1)
    X_int = sample["X_int"]
    n_vars = int(sample["num_vars"].item())
    # X_int post-intervention should have non-trivial values for at
    # least the intervention target.
    assert float(X_int[:, :n_vars].abs().max().item()) > 0.0


def test_zero_context_query_time_is_in_post_intervention_region():
    """Queries must still target post-intervention timesteps; with
    zero-context the entire trajectory is post-intervention."""
    prior = _mk_prior(p_no_context=1.0, seed=47)
    for _ in range(10):
        sample = prior.generate_sample(T=20, n_queries=3)
        # int_onset_idx == 0 so any query_time_idx >= 0 is valid; the
        # only invariant is that we get exactly n_queries queries.
        assert sample["query_time"].shape == (3,)
        assert int(sample["int_onset_idx"].item()) == 0


def test_zero_context_intervention_window_starts_at_times_zero():
    prior = _mk_prior(p_no_context=1.0, seed=48)
    sample = prior.generate_sample(T=20, n_queries=1)
    # Absolute time field
    t_int_start = float(sample["t_int_start"].item())
    times_first = float(sample["times"][0].item())
    assert t_int_start == pytest.approx(times_first, abs=1e-6)


def test_zero_context_random_graph_path_works():
    """RandomContinuousExtendedPrior forwards p_no_context to its
    ContinuousExtendedPrior super-init via **kwargs; verify the
    forwarded knob actually fires at the random-graph path too."""
    from dotime.prior.continuous import RandomContinuousExtendedPrior

    prior = RandomContinuousExtendedPrior(
        n_min=3, n_max_prior=5, edge_prob=0.4,
        n_max=8, t_range=(20, 20), schedule="regular", dt=1.0,
        intervention_kind_probs=(1.0, 0.0, 0.0),
        p_no_context=1.0, seed=49,
    )
    for _ in range(5):
        sample = prior.generate_sample(T=20, n_queries=1)
        assert int(sample["int_onset_idx"].item()) == 0


def test_zero_context_dataloader_path():
    """Smoke check: ContinuousTemporalInterventionDataLoader forwards
    p_no_context through to the prior."""
    from dotime.data.continuous_dataloader import (
        ContinuousTemporalInterventionDataLoader,
    )
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=3,
        batch_size=4,
        prior_mode="tscm",
        tscm_structure="back_door",
        n_max=8, t_range=(20, 20), schedule="regular", dt=1.0,
        p_no_context=1.0,
        seed=50,
        device="cpu",
        prefetch=0,
    )
    batch = next(iter(loader))
    # Every sample in the batch is zero-context.
    onsets = batch["int_onset_idx"].tolist()
    assert all(o == 0 for o in onsets), f"expected all zero-context, got {onsets}"
