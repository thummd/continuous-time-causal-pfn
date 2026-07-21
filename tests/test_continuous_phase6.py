"""Phase 6 tests: hidden-variable masking in TSCM + random-graph paths."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.continuous import ContinuousDoOverTimePFN
from dotime.prior.continuous import (
    ContinuousExtendedPrior,
    RandomContinuousExtendedPrior,
    RandomContinuousSCMSampler,
)


N_MAX = 12
EMBED = 16
CONTEXT_WINDOW = 16


def _tiny_model(n_max: int = N_MAX) -> ContinuousDoOverTimePFN:
    return ContinuousDoOverTimePFN(
        n_max=n_max,
        embed_size=EMBED,
        n_heads=2,
        n_encoder_layers=1,
        n_cross_attn_heads=2,
        head_type="quantile",
        tau_levels=[0.1, 0.5, 0.9],
        context_window=CONTEXT_WINDOW,
        num_time_frequencies=8,
    )


# ---------------------------------------------------------------------- TSCM hidden-var masking


@pytest.mark.parametrize(
    "structure, expected_hidden_count",
    [
        ("back_door", 0),
        ("bivariate", 0),
        ("front_door", 1),              # U hidden
        ("instrumental_variable", 1),   # U hidden
        ("unobserved_confounder", 1),   # U hidden
    ],
)
def test_tscm_structures_mask_hidden_vars_correctly(
    structure, expected_hidden_count,
):
    """Known-topology structures with hidden U should expose
    variable_mask==0 at the hidden position in the output batch."""
    p = ContinuousExtendedPrior(
        tscm_structure=structure,
        intervention_kind_probs=(1.0, 0.0, 0.0),
        pair_mode="counterfactual",
        t_range=(30, 30),
        seed=0,
    )
    s = p.generate_sample(T=30)
    n_vars = int(s["num_vars"].item())
    vmask = s["variable_mask"][:n_vars]
    hidden_positions = (vmask == 0).nonzero(as_tuple=True)[0].tolist()
    assert len(hidden_positions) == expected_hidden_count


def test_tscm_front_door_hidden_position_has_zero_x_obs():
    """In front_door, the hidden U must have identically-zero X_obs and X_int."""
    p = ContinuousExtendedPrior(
        tscm_structure="front_door",
        intervention_kind_probs=(1.0, 0.0, 0.0),
        pair_mode="counterfactual",
        t_range=(40, 40),
        seed=1,
    )
    s = p.generate_sample(T=40)
    n_vars = int(s["num_vars"].item())
    vmask = s["variable_mask"][:n_vars]
    hidden_positions = (vmask == 0).nonzero(as_tuple=True)[0].tolist()
    assert len(hidden_positions) == 1
    h = hidden_positions[0]
    # X_obs is already causally masked post-intervention, so check the
    # hidden variable's *entire* slice is zero.
    assert torch.all(s["X_obs"][:, h] == 0.0)
    assert torch.all(s["X_int"][:, h] == 0.0)


def test_tscm_a_and_y_never_hidden():
    """Treatment A (canon 0) and outcome Y (canon N-1) must stay observable."""
    for struct in ("front_door", "instrumental_variable", "unobserved_confounder"):
        p = ContinuousExtendedPrior(
            tscm_structure=struct,
            intervention_kind_probs=(1.0, 0.0, 0.0),
            t_range=(24, 24),
            seed=7,
        )
        s = p.generate_sample(T=24)
        n_vars = int(s["num_vars"].item())
        vmask = s["variable_mask"]
        assert vmask[0].item() == 1.0, f"{struct}: A is hidden (canon idx 0)"
        assert vmask[n_vars - 1].item() == 1.0, f"{struct}: Y is hidden (canon idx N-1)"


# ---------------------------------------------------------------------- random-graph hidden


def test_random_sampler_hidden_prob_zero_leaves_all_observed():
    s = RandomContinuousSCMSampler(n_min=3, n_max_prior=6, hidden_prob=0.0, seed=0)
    for _ in range(20):
        _, _, _, _, hidden = s.sample()
        assert hidden == []


def test_random_sampler_hidden_prob_positive_produces_hidden():
    s = RandomContinuousSCMSampler(n_min=5, n_max_prior=5, hidden_prob=1.0, seed=1)
    for _ in range(10):
        _, n_vars, a, y, hidden = s.sample()
        # With hidden_prob=1.0 every non-(A, Y) node is hidden.
        expected_hidden = {i for i in range(n_vars) if i != a and i != y}
        assert set(hidden) == expected_hidden


def test_random_sampler_rejects_invalid_hidden_prob():
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(n_min=3, n_max_prior=5, hidden_prob=-0.1)
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(n_min=3, n_max_prior=5, hidden_prob=1.5)


def test_random_sampler_a_and_y_not_in_hidden():
    """Even at hidden_prob=1.0, A and Y must be excluded from the hidden set."""
    s = RandomContinuousSCMSampler(n_min=4, n_max_prior=6, hidden_prob=1.0, seed=42)
    for _ in range(30):
        _, _, a, y, hidden = s.sample()
        assert a not in hidden
        assert y not in hidden


def test_random_extended_prior_hidden_fraction_matches_expectation():
    """With hidden_prob=0.5 and n_vars=6, expect ~2 hidden nodes per sample
    (out of 4 non-(A,Y) eligible nodes).  Averaged over 100 samples the
    mean should land between 1.5 and 2.5 for seed=0."""
    p = RandomContinuousExtendedPrior(
        n_min=6, n_max_prior=6, hidden_prob=0.5,
        t_range=(20, 20), seed=0,
    )
    hidden_counts = []
    for _ in range(100):
        s = p.generate_sample(T=20)
        vmask = s["variable_mask"][:int(s["num_vars"].item())]
        hidden_counts.append(int((vmask == 0).sum().item()))
    mean = sum(hidden_counts) / len(hidden_counts)
    assert 1.5 < mean < 2.5, f"mean hidden count {mean} outside expected range"


def test_random_extended_prior_hidden_positions_are_zero_in_x_obs():
    p = RandomContinuousExtendedPrior(
        n_min=6, n_max_prior=6, hidden_prob=0.8,
        t_range=(30, 30), seed=5,
    )
    s = p.generate_sample(T=30)
    n_vars = int(s["num_vars"].item())
    vmask = s["variable_mask"][:n_vars]
    hidden_positions = (vmask == 0).nonzero(as_tuple=True)[0].tolist()
    if not hidden_positions:
        pytest.skip("sample happened to produce no hidden vars at this seed")
    for h in hidden_positions:
        assert torch.all(s["X_obs"][:, h] == 0.0), f"X_obs nonzero at hidden canon idx {h}"
        assert torch.all(s["X_int"][:, h] == 0.0), f"X_int nonzero at hidden canon idx {h}"


# ---------------------------------------------------------------------- queries


def test_queries_never_target_hidden_variable():
    """_sample_queries respects hidden_vars_topo; check it in the random path."""
    p = RandomContinuousExtendedPrior(
        n_min=5, n_max_prior=5, hidden_prob=0.8,
        t_range=(20, 20), seed=11,
    )
    for _ in range(50):
        s = p.generate_sample(T=20, n_queries=3)
        n_vars = int(s["num_vars"].item())
        vmask = s["variable_mask"][:n_vars]
        hidden = set((vmask == 0).nonzero(as_tuple=True)[0].tolist())
        q_targets = s["query_target"].tolist()
        for q in q_targets:
            assert q not in hidden, f"query targeted hidden var {q}"


# ---------------------------------------------------------------------- dataloader + training


def test_dataloader_forwards_hidden_prob_to_random_prior():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=4,
        prior_mode="random",
        n_min_prior=5, n_max_prior=5, hidden_prob=1.0,
        t_range=(20, 20), n_max=N_MAX, seed=3,
    )
    batch = next(iter(loader))
    # With hidden_prob=1.0 and N=5, 3 nodes (non-A, non-Y) must be hidden.
    vmask = batch["variable_mask"]
    hidden_counts = (vmask[:, :5] == 0).sum(dim=1).tolist()
    assert all(c == 3 for c in hidden_counts), f"unexpected hidden counts {hidden_counts}"


def test_model_forward_on_hidden_var_batch():
    """Sanity: the model can forward a batch with hidden-var masks."""
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=2,
        prior_mode="random",
        n_min_prior=4, n_max_prior=6, hidden_prob=0.5,
        t_range=(20, 20), n_max=N_MAX, seed=2,
    )
    batch = next(iter(loader))
    model = _tiny_model()
    out = model(batch)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_smoke_training_with_hidden_vars():
    """Train a tiny CT model for a few steps with hidden variables enabled."""
    from dotime.training.continuous_trainer import train_continuous

    with tempfile.TemporaryDirectory() as tmp:
        train_continuous(
            n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=CONTEXT_WINDOW,
            num_time_frequencies=8,
            tau_levels=[0.1, 0.5, 0.9],
            prior_mode="random",
            n_min_prior=4, n_max_prior=6, hidden_prob=0.3,
            schedule="regular", dt=1.0, pair_mode="counterfactual",
            intervention_kind_probs=(1.0, 0.0, 0.0),
            t_range=(16, 16),
            batch_size=2, lr=2e-4, warmup_steps=2, total_steps=6,
            grad_clip=1.0, eval_every=3, eval_num_steps=1,
            seed=0, device="cpu", save_dir=tmp, prefetch=0,
        )
        ckpt = Path(tmp) / "continuous_do_over_time_pfn_best.pt"
        assert ckpt.exists()
