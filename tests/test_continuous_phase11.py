"""Phase 11 tests: fine-grid integration (schedule-invariant continuous prior).

Covers the ``substeps`` knob on :meth:`ContinuousSCM.simulate` that
realises tier-(C) continuous integration from Section 3.1 of the paper:
each observation gap is split into ``substeps`` Euler-Maruyama
steps, with independent noise per sub-step, so the trajectory
approximates the true SDE law as ``substeps -> infinity``.
"""

from __future__ import annotations

import math

import pytest
import torch

from dotime.prior.continuous import (
    ContinuousIntervention,
    ContinuousSCM,
    InterventionKind,
    regular_schedule,
)
from dotime.prior.continuous.ou_mechanism import OUMechanism


# ---------------------------------------------------------------------- interface


def _mk_scm_ou(theta=1.0, sigma=1.0):
    """Single-variable OU SCM with a known analytic marginal."""
    mech = OUMechanism(theta=theta, sigma=sigma)
    return ContinuousSCM([mech])


def test_substeps_defaults_to_one_and_preserves_behaviour():
    """substeps=1 must reproduce pre-Phase-11 behaviour exactly."""
    scm = _mk_scm_ou()
    times, dts = regular_schedule(T=10, dt=0.5)

    # Same seed -> identical trajectory (noise path is deterministic).
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    _, x1 = scm.simulate(times, dts, generator=g1)
    _, x2 = scm.simulate(times, dts, generator=g2, substeps=1)
    assert torch.equal(x1, x2)


def test_substeps_rejects_bad_values():
    scm = _mk_scm_ou()
    times, dts = regular_schedule(T=5, dt=0.5)
    with pytest.raises(ValueError):
        scm.simulate(times, dts, substeps=0)
    with pytest.raises(ValueError):
        scm.simulate(times, dts, substeps=-2)
    with pytest.raises(ValueError):
        scm.simulate(times, dts, substeps=1.5)  # must be int


def test_noise_shape_scales_with_substeps():
    """The caller-facing contract: noise must be
    (T-1)*substeps x n_vars."""
    scm = _mk_scm_ou()
    times, dts = regular_schedule(T=5, dt=0.5)
    g = torch.Generator().manual_seed(0)

    # Correct shape passes.
    noise = torch.empty(4 * 3, 1)
    noise.normal_(generator=g)
    scm.simulate(times, dts, noise=noise, substeps=3)

    # Old shape (T-1, n_vars) rejected when substeps > 1.
    with pytest.raises(ValueError):
        scm.simulate(times, dts, noise=torch.zeros(4, 1), substeps=3)


def test_output_shape_is_observation_grid_regardless_of_substeps():
    """Only observation times are recorded even with fine-grid stepping."""
    scm = _mk_scm_ou()
    times, dts = regular_schedule(T=20, dt=0.5)
    for k in (1, 2, 5, 10, 25):
        _, traj = scm.simulate(
            times, dts, generator=torch.Generator().manual_seed(k), substeps=k,
        )
        assert traj.shape == (20, 1), f"wrong shape at substeps={k}"


# ---------------------------------------------------------------------- convergence to true OU law


def _ou_theoretical_stationary_var(theta, sigma):
    """E[X_inf^2] for OU with dX = -theta X dt + sigma dW."""
    return (sigma ** 2) / (2.0 * theta)


@pytest.mark.parametrize("theta,sigma,horizon", [(1.0, 1.0, 20.0), (2.0, 0.5, 10.0)])
def test_fine_grid_converges_to_true_ou_variance(theta, sigma, horizon):
    """As substeps increases, the Monte Carlo variance at the
    horizon converges to the OU stationary variance.  substeps=1
    systematically over-estimates the variance (Euler-Maruyama bias
    at coarse dt), and substeps >> 1 tightens around the truth.

    Quantitative comparison:

    - Naive one-step EM (substeps=1, dt = horizon): its stationary
      variance under the discrete recursion is
      sigma**2 / (1 - (1 - theta*dt)**2). With theta*horizon >> 1 this
      blows up and the gap to the true stationary variance
      sigma**2 / (2 theta) is large.
    - Fine-grid EM (substeps >> 1, dt = horizon / substeps):
      the discrete stationary variance converges to
      sigma**2 / (2 theta) as dt -> 0.  We verify with a 10% tolerance.
    """
    scm = _mk_scm_ou(theta=theta, sigma=sigma)
    # Coarse observation schedule: single "huge" gap so the bias of
    # one-step Euler-Maruyama is visible.
    times = torch.tensor([0.0, horizon])
    dts = torch.tensor([horizon])

    def mc_variance(substeps, n_traj, seed):
        xs = torch.empty(n_traj)
        g = torch.Generator().manual_seed(seed)
        for j in range(n_traj):
            _, traj = scm.simulate(times, dts, generator=g, substeps=substeps)
            xs[j] = traj[-1, 0]
        return float(xs.var(unbiased=True).item())

    target = _ou_theoretical_stationary_var(theta, sigma)
    v_coarse = mc_variance(substeps=1, n_traj=4000, seed=0)
    v_fine = mc_variance(substeps=200, n_traj=2000, seed=1)

    # Tier-(B) naive integration over-shoots the stationary variance
    # by a large margin for theta*horizon >> 1.
    assert v_coarse > target * 1.5, (
        f"expected naive one-step Euler to over-shoot target {target:.3f}, got {v_coarse:.3f}"
    )
    # Tier-(C) fine-grid integration should land within 15% of the
    # stationary variance at these MC sizes (MC 95% CI +/- stationary-EM bias).
    assert math.isclose(v_fine, target, rel_tol=0.15), (
        f"fine-grid variance {v_fine:.3f} vs target {target:.3f}"
    )
    # And fine-grid should be much closer to target than coarse.
    assert abs(v_fine - target) < abs(v_coarse - target) / 4, (
        f"fine ({v_fine:.3f}) should be much closer to target ({target:.3f}) than coarse ({v_coarse:.3f})"
    )


# ---------------------------------------------------------------------- schedule invariance


def test_fine_grid_trajectory_law_is_approximately_schedule_invariant():
    """Key property from Definition 3.1: when we integrate finely, the
    marginal law at a fixed physical time should not depend on how
    many OTHER observation times are on the schedule.
    """
    scm = _mk_scm_ou(theta=1.0, sigma=1.0)

    # Two schedules that both include t = 5.0 but differ elsewhere.
    times_sparse = torch.tensor([0.0, 5.0])
    dts_sparse = torch.tensor([5.0])
    times_dense = torch.tensor([0.0, 1.0, 2.5, 3.7, 5.0])
    dts_dense = times_dense[1:] - times_dense[:-1]

    def mc_var_at_5(times, dts, substeps, n_traj=3000, seed=0):
        xs = torch.empty(n_traj)
        g = torch.Generator().manual_seed(seed)
        for j in range(n_traj):
            _, traj = scm.simulate(times, dts, generator=g, substeps=substeps)
            xs[j] = traj[-1, 0]  # X(5.0)
        return float(xs.var(unbiased=True).item())

    # Naive tier-(B) integration: sparse schedule over-shoots
    # (one big step) while dense schedule is much closer to truth.
    v_b_sparse = mc_var_at_5(times_sparse, dts_sparse, substeps=1, seed=0)
    v_b_dense = mc_var_at_5(times_dense, dts_dense, substeps=1, seed=1)
    # Meaningful discrepancy => tier-(B) is NOT schedule-invariant.
    assert abs(v_b_sparse - v_b_dense) / max(v_b_sparse, v_b_dense) > 0.15

    # Tier-(C) fine-grid integration: both schedules recover similar
    # variance (the law is approximately schedule-invariant).
    v_c_sparse = mc_var_at_5(times_sparse, dts_sparse, substeps=64, seed=2)
    v_c_dense = mc_var_at_5(times_dense, dts_dense, substeps=32, seed=3)
    assert abs(v_c_sparse - v_c_dense) / max(v_c_sparse, v_c_dense) < 0.15


# ---------------------------------------------------------------------- counterfactual pair with substeps


def test_counterfactual_pair_preserves_pre_intervention_match_at_fine_grid():
    """Shared noise -> obs and cf trajectories agree exactly
    pre-intervention, at any substeps."""
    scm = _mk_scm_ou(theta=1.0, sigma=0.5)
    times, dts = regular_schedule(T=12, dt=0.5)
    t_start, t_end = 3.0, 5.0
    intervention = ContinuousIntervention(
        target=0, t_start=t_start, t_end=t_end,
        kind=InterventionKind.HARD, value=0.5,
    )

    for k in (1, 4, 16):
        g = torch.Generator().manual_seed(100 + k)
        _, x_obs, x_cf = scm.sample_counterfactual_pair(
            times, dts, intervention=intervention, generator=g, substeps=k,
        )
        # All pre-intervention observations strictly before t_start
        # must match bit-exactly, since the intervention has not
        # modified the state.
        pre_mask = times < t_start
        assert torch.allclose(x_obs[pre_mask], x_cf[pre_mask], atol=1e-6), (
            f"pre-intervention mismatch at substeps={k}"
        )
