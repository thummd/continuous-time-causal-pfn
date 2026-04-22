"""Phase 8 tests: regime-switching continuous-time prior."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.continuous import ContinuousDoOverTimePFN
from dotime.prior.continuous import (
    ContinuousIntervention,
    ContinuousRegimeSwitchingSCM,
    ContinuousSCM,
    InterventionKind,
    RandomContinuousExtendedPrior,
    RandomContinuousSCMSampler,
    regular_schedule,
    sample_sticky_transition_matrix,
)


N_MAX = 8
EMBED = 16
CONTEXT_WINDOW = 16


def _tiny_model(n_max: int = N_MAX) -> ContinuousDoOverTimePFN:
    return ContinuousDoOverTimePFN(
        n_max=n_max, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
        n_cross_attn_heads=2, head_type="quantile",
        tau_levels=[0.1, 0.5, 0.9],
        context_window=CONTEXT_WINDOW, num_time_frequencies=8,
    )


# ---------------------------------------------------------------------- sticky transition


def test_sticky_transition_rows_are_stochastic():
    g = torch.Generator().manual_seed(0)
    P = sample_sticky_transition_matrix(n_regimes=4, generator=g)
    row_sums = P.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-4)
    assert (P >= 0).all()


def test_sticky_transition_diagonal_is_high():
    """With default concentrations (9.0, 0.5), diagonal should be well above 0.5."""
    g = torch.Generator().manual_seed(3)
    P = sample_sticky_transition_matrix(n_regimes=3, generator=g)
    assert (P.diag() > 0.5).all(), f"diag={P.diag().tolist()}"


def test_sticky_transition_rejects_bad_inputs():
    with pytest.raises(ValueError):
        sample_sticky_transition_matrix(n_regimes=1)
    with pytest.raises(ValueError):
        sample_sticky_transition_matrix(n_regimes=3, sticky_alpha=0.0)
    with pytest.raises(ValueError):
        sample_sticky_transition_matrix(n_regimes=3, other_alpha=-0.5)


# ---------------------------------------------------------------------- SCM correctness


def test_regime_switching_scm_rejects_inconsistent_regimes():
    # Two regimes with different n_vars must be rejected.
    scm_a = ContinuousSCM.sample_random(
        n_vars=3, generator=torch.Generator().manual_seed(0),
    )
    scm_b = ContinuousSCM.sample_random(
        n_vars=4, generator=torch.Generator().manual_seed(1),
    )
    P = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(ValueError):
        ContinuousRegimeSwitchingSCM([scm_a, scm_b], P)


def test_regime_switching_scm_rejects_nonstochastic_transition_matrix():
    scm_a = ContinuousSCM.sample_random(
        n_vars=3, generator=torch.Generator().manual_seed(0),
    )
    scm_b = ContinuousSCM.sample_random(
        n_vars=3, generator=torch.Generator().manual_seed(1),
    )
    P_bad = torch.tensor([[0.5, 0.3], [0.1, 0.9]])  # row 0 sums to 0.8
    with pytest.raises(ValueError):
        ContinuousRegimeSwitchingSCM([scm_a, scm_b], P_bad)


def test_regime_switching_scm_rejects_negative_transitions():
    scm_a = ContinuousSCM.sample_random(
        n_vars=3, generator=torch.Generator().manual_seed(0),
    )
    scm_b = ContinuousSCM.sample_random(
        n_vars=3, generator=torch.Generator().manual_seed(1),
    )
    P_bad = torch.tensor([[1.1, -0.1], [0.1, 0.9]])
    with pytest.raises(ValueError):
        ContinuousRegimeSwitchingSCM([scm_a, scm_b], P_bad)


def test_regime_switching_scm_rejects_single_regime():
    scm = ContinuousSCM.sample_random(
        n_vars=3, generator=torch.Generator().manual_seed(0),
    )
    with pytest.raises(ValueError):
        ContinuousRegimeSwitchingSCM([scm], torch.tensor([[1.0]]))


def test_regime_switching_scm_simulate_shape():
    g = torch.Generator().manual_seed(0)
    scm = ContinuousRegimeSwitchingSCM.sample_random(
        n_vars=4, n_regimes=2, generator=g,
    )
    times, dts = regular_schedule(T=30, dt=0.5)
    _, traj = scm.simulate(times, dts, generator=torch.Generator().manual_seed(1))
    assert traj.shape == (30, 4)
    assert torch.isfinite(traj).all()


def test_regime_switching_scm_last_regime_trajectory_is_recorded():
    g = torch.Generator().manual_seed(0)
    scm = ContinuousRegimeSwitchingSCM.sample_random(
        n_vars=3, n_regimes=3, generator=g,
    )
    times, dts = regular_schedule(T=50, dt=0.5)
    _, _ = scm.simulate(times, dts, generator=torch.Generator().manual_seed(1))
    assert hasattr(scm, "last_regime_trajectory")
    assert scm.last_regime_trajectory.shape == (50,)
    assert set(scm.last_regime_trajectory.unique().tolist()).issubset({0, 1, 2})


def test_regime_switching_sticky_trajectory_visits_multiple_regimes():
    """Over a long trajectory the sticky chain should cover >=2 regimes."""
    g = torch.Generator().manual_seed(0)
    scm = ContinuousRegimeSwitchingSCM.sample_random(
        n_vars=3, n_regimes=3, generator=g,
    )
    # Use enough steps that a sticky chain will almost certainly switch.
    times, dts = regular_schedule(T=500, dt=0.5)
    _, _ = scm.simulate(times, dts, generator=torch.Generator().manual_seed(2))
    unique_regimes = scm.last_regime_trajectory.unique()
    assert unique_regimes.numel() >= 2


# ---------------------------------------------------------------------- pair semantics


def test_regime_switching_counterfactual_pair_matches_pre_intervention():
    """Shared noise + shared regime trajectory => identical pre-intervention obs/cf."""
    g = torch.Generator().manual_seed(5)
    scm = ContinuousRegimeSwitchingSCM.sample_random(
        n_vars=4, n_regimes=2, generator=g,
    )
    times, dts = regular_schedule(T=50, dt=0.5)
    intv = ContinuousIntervention(
        target=0, t_start=10.0, t_end=15.0,
        kind=InterventionKind.HARD, value=2.5,
    )
    _, x_obs, x_cf = scm.sample_counterfactual_pair(
        times, dts, intv, generator=torch.Generator().manual_seed(7),
    )
    pre_mask = times < intv.t_start
    max_err = (x_obs[pre_mask] - x_cf[pre_mask]).abs().max().item()
    assert max_err < 1e-5


def test_regime_switching_interventional_pair_differs_pre_intervention():
    """Independent noise + independent regime trajectory => obs / int diverge even pre-intervention."""
    g = torch.Generator().manual_seed(6)
    scm = ContinuousRegimeSwitchingSCM.sample_random(
        n_vars=4, n_regimes=2, generator=g,
    )
    times, dts = regular_schedule(T=50, dt=0.5)
    intv = ContinuousIntervention(
        target=0, t_start=10.0, t_end=15.0,
        kind=InterventionKind.HARD, value=2.5,
    )
    _, x_obs, x_int = scm.sample_interventional_pair(
        times, dts, intv, generator=torch.Generator().manual_seed(7),
    )
    pre_mask = times < intv.t_start
    assert (x_obs[pre_mask] - x_int[pre_mask]).abs().max() > 1e-3


def test_regime_switching_hard_intervention_clamps_target():
    g = torch.Generator().manual_seed(7)
    scm = ContinuousRegimeSwitchingSCM.sample_random(
        n_vars=3, n_regimes=2, generator=g,
    )
    times, dts = regular_schedule(T=40, dt=0.5)
    intv = ContinuousIntervention(
        target=0, t_start=5.0, t_end=10.0,
        kind=InterventionKind.HARD, value=3.14,
    )
    _, traj = scm.simulate(
        times, dts, intervention=intv,
        generator=torch.Generator().manual_seed(8),
    )
    in_window = (times >= intv.t_start) & (times < intv.t_end)
    # All regime mechanisms share the same hard-intervention clamp, so
    # target values inside the window must all be the intervention value.
    assert torch.allclose(traj[in_window, 0], torch.full((int(in_window.sum()),), 3.14))


# ---------------------------------------------------------------------- sampler


def test_random_sampler_honours_regime_prob_zero():
    s = RandomContinuousSCMSampler(
        n_min=3, n_max_prior=5, regime_prob=0.0, seed=0,
    )
    for _ in range(20):
        scm, *_ = s.sample()
        assert isinstance(scm, ContinuousSCM)
        assert not isinstance(scm, ContinuousRegimeSwitchingSCM)


def test_random_sampler_honours_regime_prob_one():
    s = RandomContinuousSCMSampler(
        n_min=3, n_max_prior=5, regime_prob=1.0, seed=1,
    )
    for _ in range(10):
        scm, n_vars, a, y, hidden = s.sample()
        assert isinstance(scm, ContinuousRegimeSwitchingSCM)
        # n_vars consistency across regimes is checked inside the class; the
        # sampler must respect the N range.
        assert 3 <= n_vars <= 5


def test_random_sampler_regime_count_range_respected():
    s = RandomContinuousSCMSampler(
        n_min=3, n_max_prior=5, regime_prob=1.0,
        regime_count_range=(2, 4), seed=2,
    )
    Rs = set()
    for _ in range(30):
        scm, *_ = s.sample()
        Rs.add(scm.n_regimes)
    # Over 30 samples with range [2, 4] we should see at least two distinct R values.
    assert len(Rs) >= 2
    assert Rs.issubset({2, 3, 4})


def test_random_sampler_rejects_invalid_regime_prob():
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(
            n_min=3, n_max_prior=5, regime_prob=1.5,
        )
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(
            n_min=3, n_max_prior=5, regime_count_range=(3, 2),
        )


# ---------------------------------------------------------------------- extended prior


def test_random_extended_prior_regime_samples_produce_finite_trajectories():
    p = RandomContinuousExtendedPrior(
        n_min=4, n_max_prior=6, regime_prob=1.0,
        t_range=(30, 30), seed=42,
    )
    s = p.generate_sample(T=30)
    assert torch.isfinite(s["X_obs"]).all()
    assert torch.isfinite(s["X_int"]).all()


def test_random_extended_prior_counterfactual_pre_intervention_matches_with_regimes():
    """With counterfactual pair_mode, regime-switching samples still match
    pre-intervention. This validates that shared_regime_traj propagates."""
    p = RandomContinuousExtendedPrior(
        n_min=4, n_max_prior=6, regime_prob=1.0,
        pair_mode="counterfactual",
        intervention_kind_probs=(1.0, 0.0, 0.0),
        t_range=(40, 40), seed=100,
    )
    s = p.generate_sample(T=40)
    num_vars = int(s["num_vars"].item())
    onset = int(s["int_onset_idx"].item())
    # Pre-intervention X_obs (before causal masking) and X_int should match.
    # X_obs is zero-masked post-intervention, so only checking pre-intervention window.
    if onset > 1:
        pre_obs = s["X_obs"][:onset, :num_vars]
        pre_int = s["X_int"][:onset, :num_vars]
        assert torch.allclose(pre_obs, pre_int, atol=1e-5)


# ---------------------------------------------------------------------- dataloader + training


def test_dataloader_supports_regime_prob():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=3, prior_mode="random",
        n_min_prior=4, n_max_prior=6,
        regime_prob=1.0, regime_count_range=(2, 3),
        t_range=(20, 20), n_max=N_MAX, seed=1,
    )
    batch = next(iter(loader))
    assert batch["X_obs_norm"].shape == (3, 20, N_MAX)
    assert torch.isfinite(batch["Y_true"]).all()


def test_model_forward_on_regime_switching_batch():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=2, prior_mode="random",
        n_min_prior=3, n_max_prior=5,
        regime_prob=1.0, regime_count_range=(2, 3),
        t_range=(20, 20), n_max=N_MAX, seed=2,
    )
    batch = next(iter(loader))
    model = _tiny_model()
    out = model(batch)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_smoke_training_with_regime_prob():
    from dotime.training.continuous_trainer import train_continuous

    with tempfile.TemporaryDirectory() as tmp:
        train_continuous(
            n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=CONTEXT_WINDOW,
            num_time_frequencies=8,
            prior_mode="random",
            n_min_prior=4, n_max_prior=6,
            regime_prob=1.0, regime_count_range=(2, 3),
            schedule="regular", dt=1.0, pair_mode="counterfactual",
            intervention_kind_probs=(1.0, 0.0, 0.0),
            t_range=(16, 16),
            batch_size=2, lr=2e-4, warmup_steps=2, total_steps=6,
            grad_clip=1.0, eval_every=3, eval_num_steps=1,
            seed=0, device="cpu", save_dir=tmp, prefetch=0,
        )
        ckpt = Path(tmp) / "continuous_do_over_time_pfn_best.pt"
        assert ckpt.exists()
