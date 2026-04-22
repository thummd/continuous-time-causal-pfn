"""Phase 9 tests: seeded Marsaglia-Tsang Gamma + Dirichlet for the
sticky transition matrix sampler."""

from __future__ import annotations

import math

import pytest
import torch

from dotime.prior.continuous.regime_switching import (
    _sample_dirichlet_seeded,
    _sample_gamma_seeded,
    sample_sticky_transition_matrix,
)


# ---------------------------------------------------------------------- Gamma sampler moments


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0, 9.0, 20.0])
def test_gamma_sampler_mean_matches_target(alpha):
    """For Gamma(alpha, 1): E[X] == alpha."""
    g = torch.Generator().manual_seed(0)
    alphas = torch.full((20_000,), float(alpha))
    samples = _sample_gamma_seeded(alphas, g)
    mean = float(samples.mean().item())
    # Empirical mean should be within 3% of alpha with 20k samples.
    assert math.isclose(mean, alpha, rel_tol=0.03), f"mean {mean} vs target {alpha}"


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0, 9.0])
def test_gamma_sampler_variance_matches_target(alpha):
    """For Gamma(alpha, 1): Var[X] == alpha."""
    g = torch.Generator().manual_seed(1)
    alphas = torch.full((20_000,), float(alpha))
    samples = _sample_gamma_seeded(alphas, g)
    var = float(samples.var(unbiased=True).item())
    # Small-alpha Gamma has a heavy tail — allow a wider tolerance.
    tol = 0.10 if alpha < 1.0 else 0.05
    assert math.isclose(var, alpha, rel_tol=tol), f"var {var} vs target {alpha}"


def test_gamma_sampler_is_bit_exact_under_same_generator():
    alphas = torch.tensor([0.3, 0.9, 1.5, 9.0, 17.0])
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s1 = _sample_gamma_seeded(alphas, g1)
    s2 = _sample_gamma_seeded(alphas, g2)
    assert torch.equal(s1, s2)


def test_gamma_sampler_rejects_nonpositive_alpha():
    with pytest.raises(ValueError):
        _sample_gamma_seeded(
            torch.tensor([0.5, 0.0, 1.0]),
            torch.Generator().manual_seed(0),
        )
    with pytest.raises(ValueError):
        _sample_gamma_seeded(
            torch.tensor([-0.5, 1.0]),
            torch.Generator().manual_seed(0),
        )


def test_gamma_sampler_outputs_are_strictly_positive():
    alphas = torch.tensor([0.1, 0.5, 1.0, 5.0])
    g = torch.Generator().manual_seed(7)
    samples = _sample_gamma_seeded(alphas, g)
    assert (samples > 0).all()
    assert torch.isfinite(samples).all()


# ---------------------------------------------------------------------- Dirichlet sampler


def test_dirichlet_row_is_normalized():
    alphas = torch.tensor([0.5, 0.5, 9.0, 0.5])
    g = torch.Generator().manual_seed(0)
    row = _sample_dirichlet_seeded(alphas, g)
    assert row.shape == (4,)
    assert math.isclose(float(row.sum().item()), 1.0, rel_tol=1e-5)
    assert (row > 0).all()


def test_dirichlet_row_moments_match_target():
    """E[Dirichlet(a)_i] = a_i / sum(a).  Averaged over many draws, the
    empirical mean should match the theoretical fraction."""
    alphas = torch.tensor([9.0, 0.5, 0.5])
    total = float(alphas.sum().item())
    target = (alphas / total).tolist()

    rows = []
    for seed in range(2000):
        g = torch.Generator().manual_seed(seed)
        rows.append(_sample_dirichlet_seeded(alphas, g))
    emp = torch.stack(rows).mean(dim=0).tolist()
    # Rel tolerance of 5% is comfortable at this sample size.
    for e, t in zip(emp, target):
        assert math.isclose(e, t, rel_tol=0.05), f"emp={emp} target={target}"


def test_dirichlet_rejects_multidim_alphas():
    with pytest.raises(ValueError):
        _sample_dirichlet_seeded(
            torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            torch.Generator().manual_seed(0),
        )


# ---------------------------------------------------------------------- sticky transition


def test_sticky_transition_matrix_is_bit_exact_under_same_generator():
    """Regression: phase 8 used a Wilson-Hilferty approximation for the
    seeded path; phase 9's Marsaglia-Tsang path is still reproducible
    bit-for-bit under the same generator."""
    g1 = torch.Generator().manual_seed(100)
    g2 = torch.Generator().manual_seed(100)
    P1 = sample_sticky_transition_matrix(n_regimes=3, generator=g1)
    P2 = sample_sticky_transition_matrix(n_regimes=3, generator=g2)
    assert torch.equal(P1, P2)


def test_sticky_transition_matrix_diagonal_concentrates_around_expected():
    """Averaged over many seeds, diag[i] should concentrate around
    sticky_alpha / (sticky_alpha + (R - 1) * other_alpha) = 9/10 = 0.9."""
    n_regimes = 3
    diagonals = []
    for seed in range(500):
        g = torch.Generator().manual_seed(seed)
        P = sample_sticky_transition_matrix(n_regimes=n_regimes, generator=g)
        diagonals.append(P.diag())
    mean_diag = torch.stack(diagonals).mean().item()
    target = 9.0 / (9.0 + (n_regimes - 1) * 0.5)
    assert math.isclose(mean_diag, target, rel_tol=0.05), (
        f"mean diagonal {mean_diag} vs target {target}"
    )


def test_sticky_transition_matrix_off_diagonal_is_unbiased():
    """Same check for off-diagonal entries: averaged, they should match
    the theoretical fraction other_alpha / sum(alphas)."""
    n_regimes = 3
    rows_row0 = []
    for seed in range(500):
        g = torch.Generator().manual_seed(seed)
        P = sample_sticky_transition_matrix(n_regimes=n_regimes, generator=g)
        rows_row0.append(P[0])
    mean_row0 = torch.stack(rows_row0).mean(dim=0).tolist()
    # row 0 is Dirichlet(9.0, 0.5, 0.5) => expected [0.9, 0.05, 0.05]
    target = [0.9, 0.05, 0.05]
    for e, t in zip(mean_row0, target):
        # Off-diagonal has relatively high variance; widen tolerance.
        assert abs(e - t) < 0.015, f"row 0 empirical {mean_row0} vs target {target}"


def test_sticky_transition_matrix_large_scale_comparison_with_reference():
    """Seeded Marsaglia-Tsang should match the global-RNG
    torch.distributions.Dirichlet in distribution (not exactly, but on
    first / second moments).  This catches a future regression where
    the seeded path diverges from the reference."""
    # Seeded path
    seeded_diag = []
    for seed in range(1000):
        g = torch.Generator().manual_seed(seed)
        P = sample_sticky_transition_matrix(n_regimes=4, generator=g)
        seeded_diag.append(P.diag().mean().item())
    # Reference path (global RNG)
    torch.manual_seed(0)
    reference_diag = []
    for _ in range(1000):
        P = sample_sticky_transition_matrix(n_regimes=4, generator=None)
        reference_diag.append(P.diag().mean().item())

    import statistics
    seeded_mean = statistics.mean(seeded_diag)
    ref_mean = statistics.mean(reference_diag)
    # Both should match the theoretical 9 / (9 + 3 * 0.5) = 0.857...
    target = 9.0 / (9.0 + 3 * 0.5)
    assert abs(seeded_mean - target) < 0.01
    assert abs(ref_mean - target) < 0.01
    # And should agree with each other.
    assert abs(seeded_mean - ref_mean) < 0.01


# ---------------------------------------------------------------------- regression


def test_phase8_integration_still_works_with_phase9_sampler():
    """End-to-end check: a random-graph regime-switching trajectory
    still simulates cleanly with the new seeded Dirichlet path."""
    from dotime.prior.continuous import RandomContinuousExtendedPrior

    p = RandomContinuousExtendedPrior(
        n_min=4, n_max_prior=6, regime_prob=1.0,
        pair_mode="counterfactual",
        intervention_kind_probs=(1.0, 0.0, 0.0),
        t_range=(30, 30), seed=0,
    )
    s = p.generate_sample(T=30)
    assert torch.isfinite(s["X_obs"]).all()
    assert torch.isfinite(s["X_int"]).all()
    # Counterfactual pre-intervention match (carried through shared
    # noise + shared regime trajectory).
    onset = int(s["int_onset_idx"].item())
    num_vars = int(s["num_vars"].item())
    if onset > 1:
        pre_obs = s["X_obs"][:onset, :num_vars]
        pre_int = s["X_int"][:onset, :num_vars]
        assert torch.allclose(pre_obs, pre_int, atol=1e-5)
