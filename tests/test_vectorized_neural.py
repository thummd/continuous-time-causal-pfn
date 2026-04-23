"""Tests for vectorized neural-drift simulation (bmm path).

Compares vectorize=True (bmm) vs vectorize=False (loop) for neural-only
SCMs.  The bmm path has ~1e-7 floating-point divergence from the loop
due to different accumulation order — all tests use atol=1e-5.
"""

import math
import time

import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotime.prior.continuous.continuous_scm import (
    ContinuousIntervention,
    ContinuousSCM,
    InterventionKind,
)
from dotime.prior.continuous.time_schedule import regular_schedule, jittered_schedule


ATOL = 1e-5

INTERVENTIONS = {
    "no_intervention": None,
    "hard_intervention": ContinuousIntervention(
        target=0, t_start=3.0, t_end=7.0,
        kind=InterventionKind.HARD, value=2.0,
    ),
    "soft_intervention": ContinuousIntervention(
        target=1, t_start=2.0, t_end=6.0,
        kind=InterventionKind.SOFT, value=0.5,
    ),
    "time_varying_intervention": ContinuousIntervention(
        target=0, t_start=2.5, t_end=8.0,
        kind=InterventionKind.TIME_VARYING,
        value=lambda t: math.sin(t),
    ),
}


def _make_neural_scm(n_vars=5, seed=42, vectorize=False):
    return ContinuousSCM.sample_random(
        n_vars, edge_prob=0.5, mechanism_kind="neural",
        generator=torch.Generator().manual_seed(seed),
        vectorize=vectorize,
    )


def _compare(n_vars, seed, intervention, substeps):
    """Run loop vs bmm and return max absolute difference."""
    scm_loop = _make_neural_scm(n_vars, seed, vectorize=False)
    scm_vec = _make_neural_scm(n_vars, seed, vectorize=True)
    T = 20
    times, dts = regular_schedule(T, dt=0.5)
    n_fine = (T - 1) * substeps
    noise = scm_loop._draw_noise(n_fine, generator=torch.Generator().manual_seed(123))
    x0 = torch.zeros(n_vars)

    _, traj_loop = scm_loop.simulate(times, dts, intervention=intervention,
                                      x0=x0.clone(), noise=noise.clone(), substeps=substeps)
    _, traj_vec = scm_vec.simulate(times, dts, intervention=intervention,
                                    x0=x0.clone(), noise=noise.clone(), substeps=substeps)
    return (traj_loop - traj_vec).abs().max().item()


def test_neural_vectorized_no_intervention():
    for substeps in [1, 5]:
        diff = _compare(5, seed=42, intervention=None, substeps=substeps)
        assert diff < ATOL, f"no_intv substeps={substeps}: diff={diff:.2e}"


def test_neural_vectorized_hard_intervention():
    for substeps in [1, 5]:
        diff = _compare(5, seed=42, intervention=INTERVENTIONS["hard_intervention"],
                        substeps=substeps)
        assert diff < ATOL, f"hard substeps={substeps}: diff={diff:.2e}"


def test_neural_vectorized_soft_intervention():
    for substeps in [1, 5]:
        diff = _compare(5, seed=42, intervention=INTERVENTIONS["soft_intervention"],
                        substeps=substeps)
        assert diff < ATOL, f"soft substeps={substeps}: diff={diff:.2e}"


def test_neural_vectorized_time_varying():
    for substeps in [1, 5]:
        diff = _compare(5, seed=42, intervention=INTERVENTIONS["time_varying_intervention"],
                        substeps=substeps)
        assert diff < ATOL, f"tv substeps={substeps}: diff={diff:.2e}"


def test_neural_vectorized_counterfactual_pair():
    """Pre-intervention trajectories match between vectorized and loop paths."""
    scm_loop = _make_neural_scm(5, seed=42, vectorize=False)
    scm_vec = _make_neural_scm(5, seed=42, vectorize=True)
    T = 20
    times, dts = regular_schedule(T, dt=0.5)
    intv = INTERVENTIONS["hard_intervention"]

    gen_loop = torch.Generator().manual_seed(99)
    gen_vec = torch.Generator().manual_seed(99)
    _, obs_loop, cf_loop = scm_loop.sample_counterfactual_pair(
        times, dts, intv, generator=gen_loop, substeps=3)
    _, obs_vec, cf_vec = scm_vec.sample_counterfactual_pair(
        times, dts, intv, generator=gen_vec, substeps=3)

    # Both paths: obs and cf match pre-intervention
    pre = times < intv.t_start
    assert (obs_loop[pre] - obs_vec[pre]).abs().max().item() < ATOL
    assert (cf_loop[pre] - cf_vec[pre]).abs().max().item() < ATOL


def test_mixed_falls_through_to_loop():
    """Mixed mechanism SCMs use the loop path even with vectorize=True."""
    scm = ContinuousSCM.sample_random(
        5, edge_prob=0.5, mechanism_kind="mixed", p_neural=0.5,
        generator=torch.Generator().manual_seed(42), vectorize=True,
    )
    assert scm._all_linear is False
    assert scm._all_neural is False
    # Should still work (falls through to _step loop)
    times, dts = regular_schedule(15, dt=0.5)
    noise = scm._draw_noise(14, generator=torch.Generator().manual_seed(0))
    _, traj = scm.simulate(times, dts, noise=noise)
    assert traj.isfinite().all()


def test_vectorize_false_uses_loop():
    """With vectorize=False, _all_linear and _all_neural are always False."""
    scm_ou = ContinuousSCM.sample_random(4, edge_prob=0.4, mechanism_kind="linear",
                                          vectorize=False)
    assert scm_ou._all_linear is False

    scm_neural = ContinuousSCM.sample_random(4, edge_prob=0.4, mechanism_kind="neural",
                                              vectorize=False)
    assert scm_neural._all_neural is False


def test_neural_timing():
    """Vectorized neural path should be faster than loop path."""
    n_vars, T, substeps, n_repeats = 8, 100, 10, 10
    seed = 77
    times, dts = regular_schedule(T, dt=0.3)
    n_fine = (T - 1) * substeps
    x0 = torch.zeros(n_vars)
    intv = ContinuousIntervention(target=0, t_start=5.0, t_end=15.0,
                                   kind=InterventionKind.HARD, value=1.5)

    scm_loop = _make_neural_scm(n_vars, seed, vectorize=False)
    scm_vec = _make_neural_scm(n_vars, seed, vectorize=True)

    # Warmup
    for _ in range(3):
        noise = scm_loop._draw_noise(n_fine, generator=None)
        scm_loop.simulate(times, dts, intervention=intv, x0=x0.clone(),
                          noise=noise.clone(), substeps=substeps)
        scm_vec.simulate(times, dts, intervention=intv, x0=x0.clone(),
                         noise=noise.clone(), substeps=substeps)

    loop_times = []
    for _ in range(n_repeats):
        noise = scm_loop._draw_noise(n_fine, generator=None)
        t0 = time.perf_counter()
        scm_loop.simulate(times, dts, intervention=intv, x0=x0.clone(),
                          noise=noise.clone(), substeps=substeps)
        loop_times.append(time.perf_counter() - t0)

    vec_times = []
    for _ in range(n_repeats):
        noise = scm_vec._draw_noise(n_fine, generator=None)
        t0 = time.perf_counter()
        scm_vec.simulate(times, dts, intervention=intv, x0=x0.clone(),
                         noise=noise.clone(), substeps=substeps)
        vec_times.append(time.perf_counter() - t0)

    loop_avg = sum(loop_times) / len(loop_times)
    vec_avg = sum(vec_times) / len(vec_times)
    assert vec_avg < loop_avg, (
        f"vectorized ({vec_avg*1000:.1f}ms) not faster than loop ({loop_avg*1000:.1f}ms)"
    )
