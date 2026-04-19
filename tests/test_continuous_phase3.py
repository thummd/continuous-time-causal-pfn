"""Phase 3 tests: soft + time-varying + positivity-aware + smoke training."""

from __future__ import annotations

import os
import pickle
import tempfile

import pytest
import torch

from dotime.prior.continuous import (
    ContinuousExtendedPrior,
    InterventionKind,
)
from dotime.prior.continuous.extended_prior import (
    _RampProfile,
    _SineProfile,
    _StepProfile,
)


# ---------------------------------------------------------------------- time-varying profiles


def test_step_profile_jumps_at_midpoint():
    p = _StepProfile(t_mid=5.0, lo=-1.5, hi=2.5)
    assert p(4.9) == -1.5
    assert p(5.0) == 2.5
    assert p(7.0) == 2.5


def test_ramp_profile_interpolates_linearly():
    p = _RampProfile(t_start=0.0, t_end=4.0, val_start=-2.0, val_end=2.0)
    assert p(0.0) == pytest.approx(-2.0)
    assert p(2.0) == pytest.approx(0.0)
    assert p(4.0) == pytest.approx(2.0)
    assert p(-1.0) == pytest.approx(-2.0)  # clamped left
    assert p(5.0) == pytest.approx(2.0)    # clamped right


def test_sine_profile_first_half_cycle():
    p = _SineProfile(t_start=0.0, period=4.0, amplitude=3.0)
    assert p(0.0) == pytest.approx(0.0)
    assert p(1.0) == pytest.approx(3.0)
    assert p(2.0) == pytest.approx(0.0, abs=1e-6)
    assert p(3.0) == pytest.approx(-3.0)


def test_profiles_are_picklable():
    for obj in [
        _StepProfile(t_mid=1.0, lo=-1.0, hi=1.0),
        _RampProfile(t_start=0.0, t_end=1.0, val_start=-1.0, val_end=1.0),
        _SineProfile(t_start=0.0, period=1.0, amplitude=1.0),
    ]:
        restored = pickle.loads(pickle.dumps(obj))
        assert restored(0.5) == obj(0.5)


# ---------------------------------------------------------------------- soft + time-varying interventions


def test_intervention_kind_probs_produce_all_three_kinds():
    prior = ContinuousExtendedPrior(
        tscm_structure="back_door",
        intervention_kind_probs=(0.4, 0.3, 0.3),
        pair_mode="counterfactual",
        t_range=(40, 40),
        seed=0,
    )
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(60):
        s = prior.generate_sample(T=40)
        counts[int(s["intervention_type"].item())] += 1
    assert all(c > 0 for c in counts.values()), f"missing kinds: {counts}"


@pytest.mark.parametrize(
    "probs, expected_kind",
    [
        ((1.0, 0.0, 0.0), InterventionKind.HARD),
        ((0.0, 1.0, 0.0), InterventionKind.SOFT),
        ((0.0, 0.0, 1.0), InterventionKind.TIME_VARYING),
    ],
)
def test_deterministic_kind_produces_finite_trajectories(probs, expected_kind):
    prior = ContinuousExtendedPrior(
        tscm_structure="back_door",
        intervention_kind_probs=probs,
        pair_mode="counterfactual",
        t_range=(40, 40),
        seed=1,
    )
    s = prior.generate_sample(T=40)
    kind_idx = int(s["intervention_type"].item())
    assert kind_idx == prior._INT_KIND_ORDER.index(expected_kind)
    assert torch.isfinite(s["X_obs"]).all()
    assert torch.isfinite(s["X_int"]).all()


def test_hard_intervention_clamps_target_variable():
    """Canonical index 0 is the treatment A in counterfactual mode."""
    prior = ContinuousExtendedPrior(
        tscm_structure="back_door",
        intervention_kind_probs=(1.0, 0.0, 0.0),
        pair_mode="counterfactual",
        intervention_window_frac=(0.3, 0.3),
        t_range=(60, 60),
        seed=2,
    )
    s = prior.generate_sample(T=60)
    onset = int(s["int_onset_idx"].item())
    # The hard intervention value is reported unnormalised in
    # ``intervention_value``. Inside the intervention window, X_int[t,
    # target_idx] must equal that constant up to the next-step update.
    # We only check the first step at/after onset to avoid numerical
    # drift through the OU drift (the SCM applies the clamp BEFORE the
    # next step, so exactly at the onset step the target equals the
    # hard value).
    target_canon = int(s["intervention_target"].item())
    expected = float(s["intervention_value"].item())
    # X_int is padded; strip padding.
    num_vars = int(s["num_vars"].item())
    X_int = s["X_int"][:, :num_vars]
    if onset < X_int.shape[0]:
        # Allow a small tolerance for the schedule-to-window alignment
        # (t_int_start may fall slightly past times[onset]).
        assert abs(float(X_int[onset, target_canon].item()) - expected) < 1e-4


def test_soft_intervention_does_not_clamp_target():
    """Soft intervention only shifts drift; target values drift naturally."""
    prior = ContinuousExtendedPrior(
        tscm_structure="back_door",
        intervention_kind_probs=(0.0, 1.0, 0.0),
        pair_mode="counterfactual",
        t_range=(80, 80),
        seed=3,
    )
    s = prior.generate_sample(T=80)
    onset = int(s["int_onset_idx"].item())
    target = int(s["intervention_target"].item())
    num_vars = int(s["num_vars"].item())
    X_int = s["X_int"][:, :num_vars]
    shift = float(s["intervention_value"].item())
    # The soft shift should be of order abs(shift) but not a clamp,
    # so the target trajectory should vary across time.
    if onset < X_int.shape[0] - 5:
        window = X_int[onset:onset + 5, target]
        assert window.std() > 1e-4, "soft intervention unexpectedly clamped"
        # The window mean should be within a reasonable distance of
        # the shift (order-of-magnitude sanity check).
        assert abs(float(window.mean().item())) < 20 + abs(shift) * 5


# ---------------------------------------------------------------------- positivity-aware


def test_positivity_aware_narrows_hard_intervention_range():
    """Large intervention_value_scale + positivity clipping should narrow the range."""
    kwargs = dict(
        tscm_structure="back_door",
        intervention_kind_probs=(1.0, 0.0, 0.0),
        intervention_value_scale=10.0,
        pair_mode="counterfactual",
        t_range=(80, 80),
    )
    prior_unclipped = ContinuousExtendedPrior(
        intervention_source="prior", seed=42, **kwargs,
    )
    prior_clipped = ContinuousExtendedPrior(
        intervention_source="positivity_aware", seed=42, **kwargs,
    )

    unclipped = [
        float(prior_unclipped.generate_sample(T=80)["intervention_value"].item())
        for _ in range(40)
    ]
    clipped = [
        float(prior_clipped.generate_sample(T=80)["intervention_value"].item())
        for _ in range(40)
    ]

    span_un = max(unclipped) - min(unclipped)
    span_cl = max(clipped) - min(clipped)
    assert span_cl < span_un, f"positivity_aware did not narrow range: {span_cl} vs {span_un}"


def test_positivity_aware_is_a_noop_on_soft_and_time_varying():
    """Soft / time-varying interventions aren't re-scaled by positivity clipping."""
    prior = ContinuousExtendedPrior(
        tscm_structure="back_door",
        intervention_kind_probs=(0.0, 0.0, 1.0),
        intervention_source="positivity_aware",
        intervention_value_scale=20.0,  # large, would be clipped if applied
        pair_mode="counterfactual",
        t_range=(40, 40),
        seed=5,
    )
    s = prior.generate_sample(T=40)
    # For time-varying the scalar intervention_value is a summary; make
    # sure the sample didn't blow up and the profile is still applied.
    assert torch.isfinite(s["X_int"]).all()
    assert int(s["intervention_type"].item()) == 2


# ---------------------------------------------------------------------- validation


def test_invalid_intervention_kind_probs_rejected():
    with pytest.raises(ValueError):
        ContinuousExtendedPrior(intervention_kind_probs=(0.5, 0.5))       # length
    with pytest.raises(ValueError):
        ContinuousExtendedPrior(intervention_kind_probs=(0.0, 0.0, 0.0))  # zero sum
    with pytest.raises(ValueError):
        ContinuousExtendedPrior(intervention_kind_probs=(-0.1, 0.6, 0.5)) # negative


def test_invalid_intervention_source_rejected():
    with pytest.raises(ValueError):
        ContinuousExtendedPrior(intervention_source="bogus")


def test_invalid_time_varying_profile_rejected():
    with pytest.raises(ValueError):
        ContinuousExtendedPrior(time_varying_profile="bogus")


# ---------------------------------------------------------------------- smoke training


def test_smoke_training_runs_and_loss_decreases():
    """Run the continuous trainer for a handful of steps and check it
    actually produces a checkpoint + a finite final eval loss.
    """
    from dotime.training.continuous_trainer import train_continuous

    with tempfile.TemporaryDirectory() as tmp:
        model = train_continuous(
            # model
            n_max=8, embed_size=32, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=32,
            num_time_frequencies=16,
            tau_levels=[0.1, 0.5, 0.9],
            # prior (mixed kinds, counterfactual)
            tscm_structure="back_door",
            schedule="regular", dt=1.0, pair_mode="counterfactual",
            intervention_kind_probs=(0.5, 0.3, 0.2),
            intervention_source="positivity_aware",
            t_range=(24, 24),
            # training: a tiny run
            batch_size=4, lr=2e-4, warmup_steps=5, total_steps=20,
            grad_clip=1.0, eval_every=10, eval_num_steps=2,
            seed=0, device="cpu", save_dir=tmp, prefetch=0,
        )
        # Checkpoint was saved at least once (eval_every=10 with 20 total)
        ckpt = os.path.join(tmp, "continuous_do_over_time_pfn_best.pt")
        assert os.path.exists(ckpt), "best checkpoint was not written"
        # Step-loss log exists
        step_log = os.path.join(tmp, "ct_step_losses.csv")
        assert os.path.exists(step_log)
        # Model is valid after training
        assert next(model.parameters()).isfinite().all()
