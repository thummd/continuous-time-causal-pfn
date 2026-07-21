"""End-to-end integration tests for the continuous-time pipeline.

Covers:

- :class:`ContinuousExtendedPrior` produces batches that the
  :class:`ContinuousDoOverTimePFN` can forward through without shape
  errors.
- Loss is a finite scalar and produces gradients on all parameters.
- Counterfactual pair-mode trajectories match exactly before the
  intervention window (shared noise), as a structural regression
  against the phase-1 SDE property.
- Irregular schedules (jittered, exponential) don't break the batch.
"""

from __future__ import annotations

import pytest
import torch

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.continuous import ContinuousDoOverTimePFN
from dotime.prior.continuous import ContinuousExtendedPrior


N_MAX = 8       # Smaller than the 41 default so tests stay fast.
EMBED_SIZE = 32
CONTEXT_WINDOW = 32


def _tiny_model() -> ContinuousDoOverTimePFN:
    """Small model configuration used across tests."""
    return ContinuousDoOverTimePFN(
        n_max=N_MAX,
        embed_size=EMBED_SIZE,
        n_heads=2,
        n_encoder_layers=1,
        n_cross_attn_heads=2,
        head_type="quantile",
        tau_levels=[0.1, 0.5, 0.9],
        context_window=CONTEXT_WINDOW,
        num_time_frequencies=8,
    )


# ---------------------------------------------------------------------- batch / model


def test_dataloader_batch_has_continuous_fields():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        tscm_structure="back_door",
        schedule="regular",
        dt=1.0,
        t_range=(20, 20),
        n_max=N_MAX,
        seed=0,
    )
    batch = next(iter(loader))
    assert batch["X_obs_norm"].shape == (2, 20, N_MAX)
    assert batch["times"].shape == (2, 20)
    assert batch["dts"].shape == (2, 19)
    assert batch["t_int_start"].shape == (2,)
    assert "Y_true_norm" in batch


def test_model_forward_on_continuous_batch():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=3,
        tscm_structure="front_door",
        schedule="regular",
        dt=1.0,
        t_range=(25, 25),
        n_max=N_MAX,
        seed=1,
    )
    batch = next(iter(loader))
    model = _tiny_model()
    out = model(batch)
    assert out.shape == (3, 3)  # (B, Q) with 3 quantiles
    assert torch.isfinite(out).all()


def test_loss_is_finite_and_backward_flows():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        tscm_structure="back_door",
        schedule="regular",
        dt=1.0,
        t_range=(30, 30),
        n_max=N_MAX,
        seed=2,
    )
    batch = next(iter(loader))
    model = _tiny_model()

    loss = model.loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    # At least one parameter in each stage should receive a gradient.
    enc_grad = sum(
        (p.grad is not None and p.grad.abs().sum() > 0)
        for p in model.temporal_encoder.parameters()
    )
    mix_grad = sum(
        (p.grad is not None and p.grad.abs().sum() > 0)
        for p in model.cross_variable_mixer.parameters()
    )
    head_grad = sum(
        (p.grad is not None and p.grad.abs().sum() > 0)
        for p in model.quantile_head.parameters()
    )
    assert enc_grad > 0, "no gradients reached the temporal encoder"
    assert mix_grad > 0, "no gradients reached the cross-variable mixer"
    assert head_grad > 0, "no gradients reached the quantile head"


# ---------------------------------------------------------------------- schedules


@pytest.mark.parametrize("schedule", ["regular", "jittered", "exponential"])
def test_all_schedule_types_train_a_step(schedule):
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        tscm_structure="back_door",
        schedule=schedule,
        dt=1.0,
        jitter=0.3,
        exp_rate=1.0,
        t_range=(24, 24),
        n_max=N_MAX,
        seed=3,
    )
    batch = next(iter(loader))
    model = _tiny_model()
    loss = model.loss(batch)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------- pair-mode


def test_counterfactual_mode_batch_matches_sde_property():
    """Counterfactual prior: X_obs and X_int agree before int_onset_idx.

    We check at least one real variable (non-padded) has identical
    values between X_obs and X_int for timesteps strictly before
    int_onset. This is the high-level consequence of shared noise that
    should propagate through the batch generator.
    """
    prior = ContinuousExtendedPrior(
        tscm_structure="back_door",
        schedule="regular",
        dt=1.0,
        pair_mode="counterfactual",
        n_max=N_MAX,
        t_range=(25, 25),
        seed=4,
    )
    sample = prior.generate_sample(T=25)
    # X_obs is masked to zeros after the intervention onset (causal
    # masking); X_int is not masked. So compare on strictly
    # pre-intervention timesteps.
    onset = int(sample["int_onset_idx"].item())
    num_vars = int(sample["num_vars"].item())
    if onset == 0 or num_vars == 0:
        pytest.skip("degenerate sample")
    real_var_slice = slice(0, num_vars)
    pre_obs = sample["X_obs"][:onset, real_var_slice]
    pre_int = sample["X_int"][:onset, real_var_slice]
    assert torch.allclose(pre_obs, pre_int, atol=1e-5)


def test_interventional_mode_does_not_share_noise():
    prior = ContinuousExtendedPrior(
        tscm_structure="back_door",
        schedule="regular",
        dt=1.0,
        pair_mode="interventional",
        n_max=N_MAX,
        t_range=(25, 25),
        seed=5,
    )
    sample = prior.generate_sample(T=25)
    onset = int(sample["int_onset_idx"].item())
    num_vars = int(sample["num_vars"].item())
    if onset < 2 or num_vars == 0:
        pytest.skip("degenerate sample")
    pre_obs = sample["X_obs"][:onset, :num_vars]
    pre_int = sample["X_int"][:onset, :num_vars]
    assert (pre_obs - pre_int).abs().max() > 1e-3


# ---------------------------------------------------------------------- TSCM structures


@pytest.mark.parametrize(
    "structure",
    ["back_door", "front_door", "instrumental_variable", "bivariate"],
)
def test_all_key_tscm_structures_run_through_model(structure):
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        tscm_structure=structure,
        schedule="regular",
        dt=1.0,
        t_range=(24, 24),
        n_max=N_MAX,
        seed=6,
    )
    batch = next(iter(loader))
    model = _tiny_model()
    out = model(batch)
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()
