"""Phase 10 tests: small-MLP (neural) drift mechanism.

Covers the construction of :class:`NeuralDriftMechanism`, the sampling
factory :func:`sample_neural_drift_mechanism`, integration with
:class:`ContinuousSCM` and :class:`RandomContinuousSCMSampler`, and
end-to-end trajectory stability.
"""

from __future__ import annotations

import math

import pytest
import torch

from dotime.prior.continuous import (
    ContinuousSCM,
    NeuralDriftMechanism,
    RandomContinuousSCMSampler,
    regular_schedule,
    sample_neural_drift_mechanism,
)
from dotime.prior.continuous.ou_mechanism import OUMechanism


# ---------------------------------------------------------------------- shape / validation


def test_neural_mechanism_rejects_bad_hyperparams():
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        sample_neural_drift_mechanism(parents=(), theta_range=(-1.0, 1.0), generator=g)
    with pytest.raises(ValueError):
        sample_neural_drift_mechanism(parents=(), sigma_range=(0.0, 0.1), generator=g)
    with pytest.raises(ValueError):
        sample_neural_drift_mechanism(parents=(), out_scale_range=(-0.1, 1.0), generator=g)
    with pytest.raises(ValueError):
        sample_neural_drift_mechanism(parents=(), hidden_dim=0, generator=g)
    with pytest.raises(ValueError):
        sample_neural_drift_mechanism(parents=(), weight_scale=0.0, generator=g)


def test_neural_mechanism_root_no_parents_has_correct_input_shape():
    g = torch.Generator().manual_seed(0)
    m = sample_neural_drift_mechanism(parents=(), hidden_dim=8, generator=g)
    # in_dim = 1 (just x_self)
    assert m.W1.shape == (8, 1)
    assert m.W2.shape == (1, 8)
    # Drift should be a 0-d tensor
    d = m.drift(torch.tensor(0.5), torch.empty(0))
    assert d.dim() == 0


def test_neural_mechanism_with_parents_builds_correct_input():
    g = torch.Generator().manual_seed(0)
    m = sample_neural_drift_mechanism(parents=(0, 1, 2), hidden_dim=4, generator=g)
    assert m.W1.shape == (4, 4)  # hidden x (1 + 3 parents)
    d = m.drift(torch.tensor(0.5), torch.tensor([0.1, -0.2, 0.3]))
    assert d.dim() == 0


def test_neural_mechanism_raises_on_shape_mismatch():
    g = torch.Generator().manual_seed(0)
    m = sample_neural_drift_mechanism(parents=(0, 1), hidden_dim=4, generator=g)
    # W1 hidden dim mismatch with b1
    with pytest.raises(ValueError):
        NeuralDriftMechanism(
            theta=1.0, sigma=0.5, out_scale=1.0,
            W1=torch.zeros(4, 3), b1=torch.zeros(5),   # b1 wrong length
            W2=torch.zeros(1, 4), b2=torch.zeros(1),
            parents=(0, 1),
        )


# ---------------------------------------------------------------------- physical properties


def test_neural_drift_is_bounded_by_theta_plus_out_scale():
    """|drift| <= theta * |x_self| + out_scale (tanh output in [-1, 1])."""
    g = torch.Generator().manual_seed(1)
    m = sample_neural_drift_mechanism(parents=(0, 1), generator=g)
    x_self = torch.tensor(5.0)
    x_parents = torch.tensor([3.0, -2.0])
    d = m.drift(x_self, x_parents)
    bound = m.theta * abs(float(x_self)) + m.out_scale
    assert abs(float(d)) <= bound + 1e-5, f"drift {d.item()} exceeds bound {bound}"


def test_neural_drift_reduces_to_pure_mean_reversion_at_zero_weights():
    """If the MLP weights are all zero, drift = -theta * x_self (parents ignored).

    This is the key "contains OU" property that lets Phase 10 gracefully
    degrade to the Phase 1 mechanism when the nonlinear capacity is unused.
    """
    hidden = 4
    m = NeuralDriftMechanism(
        theta=1.5, sigma=0.5, out_scale=1.0,
        W1=torch.zeros(hidden, 3),
        b1=torch.zeros(hidden),
        W2=torch.zeros(1, hidden),
        b2=torch.zeros(1),
        parents=(0, 1),
    )
    x_self = torch.tensor(0.4)
    x_parents = torch.tensor([99.0, -99.0])  # extreme values, should be ignored
    d = m.drift(x_self, x_parents)
    assert math.isclose(float(d), -1.5 * 0.4, abs_tol=1e-6)


def test_neural_drift_is_nonlinear_by_construction():
    """drift(2x) != 2 * drift(x) - OU is linear, neural drift is not."""
    g = torch.Generator().manual_seed(2)
    m = sample_neural_drift_mechanism(
        parents=(0,), generator=g, out_scale_range=(1.5, 2.5),
    )
    x1 = torch.tensor(0.5)
    x2 = torch.tensor(1.0)
    d1 = m.drift(x1, torch.tensor([0.3]))
    d2 = m.drift(x2, torch.tensor([0.6]))
    # If the mechanism were linear, d2 / d1 == 2 exactly.
    assert not math.isclose(float(d2), 2.0 * float(d1), rel_tol=1e-3)


def test_neural_step_is_bit_exact_under_same_generator():
    alphas = torch.tensor([0.3, -0.2])
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    m1 = sample_neural_drift_mechanism(parents=(0,), generator=g1)
    m2 = sample_neural_drift_mechanism(parents=(0,), generator=g2)
    x_self = torch.tensor(0.1)
    x_parents = torch.tensor([alphas[0].item()])
    dt = torch.tensor(0.5)
    noise = torch.tensor(0.7)
    s1 = m1.step(x_self, x_parents, dt, noise)
    s2 = m2.step(x_self, x_parents, dt, noise)
    assert torch.equal(s1, s2)


# ---------------------------------------------------------------------- SCM integration


def test_continuous_scm_accepts_neural_mechanisms():
    g = torch.Generator().manual_seed(3)
    scm = ContinuousSCM.sample_random(
        n_vars=4, mechanism_kind="neural", edge_prob=0.5, generator=g,
    )
    assert scm.n_vars == 4
    assert all(isinstance(m, NeuralDriftMechanism) for m in scm.mechanisms)


def test_continuous_scm_mixed_kind_produces_both_types():
    g = torch.Generator().manual_seed(4)
    scm = ContinuousSCM.sample_random(
        n_vars=20, mechanism_kind="mixed", p_neural=0.5, generator=g,
    )
    kinds = {type(m).__name__ for m in scm.mechanisms}
    # With p_neural = 0.5 over 20 vars, the probability of seeing only one
    # kind is 2 * 0.5^20 << 1e-6.
    assert kinds == {"OUMechanism", "NeuralDriftMechanism"}


def test_continuous_scm_default_still_linear():
    g = torch.Generator().manual_seed(5)
    scm = ContinuousSCM.sample_random(n_vars=4, generator=g)
    assert all(isinstance(m, OUMechanism) for m in scm.mechanisms)


def test_continuous_scm_invalid_mechanism_kind():
    with pytest.raises(ValueError):
        ContinuousSCM.sample_random(n_vars=3, mechanism_kind="deep_ensemble")
    with pytest.raises(ValueError):
        ContinuousSCM.sample_random(n_vars=3, p_neural=1.5)


# ---------------------------------------------------------------------- trajectory stability


@pytest.mark.parametrize("mechanism_kind", ["linear", "neural", "mixed"])
def test_trajectory_stays_finite(mechanism_kind):
    """Explicit mean-reversion keeps trajectories bounded for any mechanism_kind."""
    g = torch.Generator().manual_seed(6)
    scm = ContinuousSCM.sample_random(
        n_vars=6,
        edge_prob=0.4,
        mechanism_kind=mechanism_kind,
        p_neural=0.5,
        generator=g,
    )
    times, dts = regular_schedule(T=100, dt=0.5)
    _, traj = scm.simulate(times, dts, generator=torch.Generator().manual_seed(7))
    assert traj.shape == (100, 6)
    assert torch.isfinite(traj).all()
    # Bounded absolute value check: with theta in (0.5, 2.0), out_scale in
    # (0.5, 2.0), and sigma in (0.2, 0.8), the long-run std should stay
    # well below ~20 over 100 steps.  A very loose sanity cap of 1e3.
    assert float(traj.abs().max()) < 1e3


# ---------------------------------------------------------------------- random-graph sampler


def test_random_sampler_forwards_mechanism_kind():
    sampler = RandomContinuousSCMSampler(
        n_min=4, n_max_prior=4,
        mechanism_kind="neural", seed=8,
    )
    scm, n, a, y, hidden = sampler.sample()
    assert all(isinstance(m, NeuralDriftMechanism) for m in scm.mechanisms)


def test_random_sampler_mixed_mode_produces_both():
    sampler = RandomContinuousSCMSampler(
        n_min=10, n_max_prior=10,
        mechanism_kind="mixed", p_neural=0.5, seed=9,
    )
    scm, _, _, _, _ = sampler.sample()
    kinds = {type(m).__name__ for m in scm.mechanisms}
    assert kinds == {"OUMechanism", "NeuralDriftMechanism"}


def test_random_sampler_rejects_bad_mechanism_kind():
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(mechanism_kind="transformer")
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(mechanism_kind="mixed", p_neural=-0.1)
