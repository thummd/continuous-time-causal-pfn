"""Tests for the continuous-time prior (dotime.prior.continuous)."""

from __future__ import annotations

import pickle

import pytest
import torch

from dotime.prior.continuous import (
    ContinuousIntervention,
    ContinuousSCM,
    InterventionKind,
    OUMechanism,
    exponential_schedule,
    from_times,
    jittered_schedule,
    regular_schedule,
    sample_ou_mechanism,
)


# ---------------------------------------------------------------------- schedule


def test_regular_schedule_shapes_and_values():
    times, dts = regular_schedule(T=10, dt=0.5, t0=2.0)
    assert times.shape == (10,)
    assert dts.shape == (9,)
    assert torch.allclose(times, torch.linspace(2.0, 2.0 + 9 * 0.5, 10))
    assert torch.allclose(dts, torch.full((9,), 0.5))


def test_jittered_schedule_is_strictly_increasing():
    g = torch.Generator().manual_seed(0)
    times, dts = jittered_schedule(T=50, dt=1.0, jitter=0.5, generator=g)
    assert (times.diff() > 0).all()
    assert (dts > 0).all()


def test_exponential_schedule_is_strictly_increasing():
    g = torch.Generator().manual_seed(0)
    times, dts = exponential_schedule(T=200, rate=1.0, generator=g)
    assert (times.diff() > 0).all()
    # inter-arrival mean should be roughly 1 / rate
    assert 0.5 < dts.mean().item() < 2.0


def test_schedule_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        regular_schedule(T=1)
    with pytest.raises(ValueError):
        regular_schedule(T=5, dt=-1.0)
    with pytest.raises(ValueError):
        jittered_schedule(T=5, dt=1.0, jitter=1.0)
    with pytest.raises(ValueError):
        exponential_schedule(T=5, rate=0.0)


def test_from_times_wraps_arbitrary_schedule():
    t = torch.tensor([0.0, 0.3, 1.1, 2.8])
    times, dts = from_times(t)
    assert torch.allclose(times, t)
    assert torch.allclose(dts, torch.tensor([0.3, 0.8, 1.7]))

    with pytest.raises(ValueError):
        from_times(torch.tensor([0.0, 1.0, 0.5]))  # not monotone


# ---------------------------------------------------------------------- mechanism


def test_ou_mechanism_rejects_nonpositive_theta_sigma():
    with pytest.raises(ValueError):
        OUMechanism(theta=0.0, sigma=1.0)
    with pytest.raises(ValueError):
        OUMechanism(theta=1.0, sigma=0.0)


def test_ou_mechanism_rejects_mismatched_parents():
    with pytest.raises(ValueError):
        OUMechanism(
            theta=1.0,
            sigma=0.1,
            parent_weights=torch.tensor([0.5]),
            parents=(0, 1),  # 2 parents but only 1 weight
        )


def test_sample_ou_mechanism_respects_seed():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    m1 = sample_ou_mechanism(parents=[0, 1], generator=g1)
    m2 = sample_ou_mechanism(parents=[0, 1], generator=g2)
    assert m1.theta == m2.theta
    assert m1.sigma == m2.sigma
    assert torch.allclose(m1.parent_weights, m2.parent_weights)


# ---------------------------------------------------------------------- continuous-SCM


def test_ar1_equivalence_at_dt_one():
    """Univariate OU with dt=1 reproduces AR(1) exactly."""
    theta, sigma, T = 0.3, 0.5, 100
    noise = torch.randn(T - 1, 1, generator=torch.Generator().manual_seed(0))

    scm = ContinuousSCM([OUMechanism(theta=theta, sigma=sigma)])
    times, dts = regular_schedule(T, dt=1.0)
    _, traj = scm.simulate(times, dts, noise=noise)

    # Manual AR(1): x[t+1] = (1 - theta) * x[t] + sigma * Z
    x_ar = torch.zeros(T)
    for i in range(T - 1):
        x_ar[i + 1] = (1 - theta) * x_ar[i] + sigma * noise[i, 0]

    assert torch.allclose(traj.squeeze(-1), x_ar, atol=1e-5)


def test_deterministic_with_seed():
    scm = ContinuousSCM.sample_random(n_vars=4, generator=torch.Generator().manual_seed(1))
    times, dts = regular_schedule(30, dt=0.5)

    _, t1 = scm.simulate(times, dts, generator=torch.Generator().manual_seed(42))
    _, t2 = scm.simulate(times, dts, generator=torch.Generator().manual_seed(42))
    assert torch.allclose(t1, t2)


def test_simulate_shapes():
    scm = ContinuousSCM.sample_random(n_vars=5, generator=torch.Generator().manual_seed(0))
    times, dts = regular_schedule(20, dt=0.5)
    out_times, traj = scm.simulate(times, dts, generator=torch.Generator().manual_seed(0))
    assert out_times.shape == (20,)
    assert traj.shape == (20, 5)


def test_simulate_rejects_bad_shapes():
    scm = ContinuousSCM([OUMechanism(theta=1.0, sigma=0.5)])
    times, dts = regular_schedule(10, dt=1.0)
    with pytest.raises(ValueError):
        scm.simulate(times.unsqueeze(0), dts)
    with pytest.raises(ValueError):
        scm.simulate(times, dts[:-1])


def test_lagged_parents_affect_simulation():
    """Changing parent weights should change downstream trajectory."""
    mech_root = OUMechanism(theta=1.0, sigma=0.1)
    mech_child_weak = OUMechanism(
        theta=1.0, sigma=0.1, parent_weights=torch.tensor([0.1]), parents=(0,)
    )
    mech_child_strong = OUMechanism(
        theta=1.0, sigma=0.1, parent_weights=torch.tensor([2.0]), parents=(0,)
    )

    times, dts = regular_schedule(30, dt=0.5)
    noise = torch.randn(29, 2, generator=torch.Generator().manual_seed(0))

    _, traj_weak = ContinuousSCM([mech_root, mech_child_weak]).simulate(
        times, dts, noise=noise
    )
    _, traj_strong = ContinuousSCM([mech_root, mech_child_strong]).simulate(
        times, dts, noise=noise
    )

    # Root trajectory must be identical (same noise, same mechanism).
    assert torch.allclose(traj_weak[:, 0], traj_strong[:, 0])
    # Child trajectories must differ (different weights).
    assert (traj_weak[:, 1] - traj_strong[:, 1]).abs().max() > 1e-3


# ---------------------------------------------------------------------- interventions


def test_hard_intervention_clamps_target_during_window():
    scm = ContinuousSCM.sample_random(n_vars=3, generator=torch.Generator().manual_seed(2))
    times, dts = regular_schedule(40, dt=0.5)
    intv = ContinuousIntervention(
        target=0, t_start=5.0, t_end=10.0, kind=InterventionKind.HARD, value=3.14
    )
    _, traj = scm.simulate(times, dts, intervention=intv, generator=torch.Generator().manual_seed(3))
    mask = (times >= intv.t_start) & (times < intv.t_end)
    assert torch.allclose(traj[mask, 0], torch.full((int(mask.sum()),), 3.14))


def test_time_varying_intervention():
    scm = ContinuousSCM([OUMechanism(theta=1.0, sigma=0.1)])
    times, dts = regular_schedule(20, dt=0.5)
    intv = ContinuousIntervention(
        target=0,
        t_start=2.0,
        t_end=5.0,
        kind=InterventionKind.TIME_VARYING,
        value=lambda t: t * 0.5,
    )
    _, traj = scm.simulate(times, dts, intervention=intv)
    mask = (times >= intv.t_start) & (times < intv.t_end)
    expected = times[mask] * 0.5
    assert torch.allclose(traj[mask, 0], expected)


def test_counterfactual_pair_shares_noise_outside_window():
    """Counterfactual pair: before intervention, obs and cf must match exactly."""
    scm = ContinuousSCM.sample_random(n_vars=3, generator=torch.Generator().manual_seed(4))
    times, dts = regular_schedule(50, dt=0.5)
    intv = ContinuousIntervention(
        target=0, t_start=10.0, t_end=15.0, kind=InterventionKind.HARD, value=2.0
    )
    _, x_obs, x_cf = scm.sample_counterfactual_pair(
        times, dts, intv, generator=torch.Generator().manual_seed(13)
    )
    pre_mask = times < intv.t_start
    assert torch.allclose(x_obs[pre_mask], x_cf[pre_mask], atol=1e-6)


def test_interventional_pair_uses_independent_noise():
    """Interventional pair: obs and int differ even before intervention window."""
    scm = ContinuousSCM.sample_random(n_vars=3, generator=torch.Generator().manual_seed(5))
    times, dts = regular_schedule(50, dt=0.5)
    intv = ContinuousIntervention(
        target=0, t_start=10.0, t_end=15.0, kind=InterventionKind.HARD, value=2.0
    )
    _, x_obs, x_int = scm.sample_interventional_pair(
        times, dts, intv, generator=torch.Generator().manual_seed(13)
    )
    pre_mask = times < intv.t_start
    assert (x_obs[pre_mask] - x_int[pre_mask]).abs().max() > 1e-3


# ---------------------------------------------------------------------- picklability


def test_scm_is_picklable():
    """Multiprocessing dataloaders require the SCM to be picklable."""
    scm = ContinuousSCM.sample_random(n_vars=3, generator=torch.Generator().manual_seed(0))
    payload = pickle.dumps(scm)
    restored = pickle.loads(payload)
    assert restored.n_vars == scm.n_vars


def test_intervention_is_picklable():
    intv = ContinuousIntervention(
        target=0, t_start=0.0, t_end=1.0, kind=InterventionKind.HARD, value=1.0
    )
    pickle.loads(pickle.dumps(intv))
