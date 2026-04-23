"""Test script for vectorized OU simulation path.

Creates OU-only SCMs, runs simulate() via the vectorized (gather) path
and the loop (_step) fallback, asserts bit-exact match for all
intervention kinds and substep counts, and prints timing comparison.

Designed to work both as a standalone script (``python tests/test_vectorized_ou.py``)
and under pytest (``pytest tests/test_vectorized_ou.py -v``).
"""

import math
import time

import torch

# ---- patch path so dotime is importable ----
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotime.prior.continuous.continuous_scm import (
    ContinuousIntervention,
    ContinuousSCM,
    InterventionKind,
)
from dotime.prior.continuous.ou_mechanism import OUMechanism
from dotime.prior.continuous.time_schedule import regular_schedule, jittered_schedule


# ------------------------------------------------------------------ helpers


def make_3var_scm(seed=42):
    """3-variable chain: 0 -> 1 -> 2, known parameters."""
    mechs = [
        OUMechanism(theta=1.0, sigma=0.3, parent_weights=torch.empty(0), parents=()),
        OUMechanism(theta=0.8, sigma=0.4, parent_weights=torch.tensor([0.6]), parents=(0,)),
        OUMechanism(theta=1.2, sigma=0.25, parent_weights=torch.tensor([-0.5]), parents=(1,)),
    ]
    return ContinuousSCM(mechs, vectorize=True)


def make_5var_scm(seed=99):
    """5-variable DAG with multiple parents."""
    mechs = [
        OUMechanism(theta=0.7, sigma=0.3, parent_weights=torch.empty(0), parents=()),
        OUMechanism(theta=1.1, sigma=0.35, parent_weights=torch.tensor([0.5]), parents=(0,)),
        OUMechanism(theta=0.9, sigma=0.4, parent_weights=torch.tensor([0.3, -0.4]), parents=(0, 1)),
        OUMechanism(theta=1.3, sigma=0.25, parent_weights=torch.tensor([0.7]), parents=(1,)),
        OUMechanism(theta=0.6, sigma=0.5, parent_weights=torch.tensor([0.2, -0.3, 0.1]), parents=(1, 2, 3)),
    ]
    return ContinuousSCM(mechs, vectorize=True)


def run_loop_simulate(scm, times, dts, intervention, noise, x0, substeps):
    """Run simulate using the original _step loop (fallback path).

    Temporarily sets _all_linear = False to force the loop path.
    """
    saved = scm._all_linear
    scm._all_linear = False
    _, traj = scm.simulate(times, dts, intervention=intervention,
                           x0=x0, noise=noise.clone(), substeps=substeps)
    scm._all_linear = saved
    return traj


def run_vectorized_simulate(scm, times, dts, intervention, noise, x0, substeps):
    """Run simulate using the vectorized (gather) path.

    Ensures _all_linear = True (should already be for OU-only SCMs).
    """
    assert scm._all_linear, "SCM must be all-linear for vectorized path"
    _, traj = scm.simulate(times, dts, intervention=intervention,
                           x0=x0, noise=noise.clone(), substeps=substeps)
    return traj


def _check_parity(scm_factory, interventions, substep_values, seed=123):
    """Check bit-exact match between loop and vectorized paths.

    Returns list of (label, passed) tuples.
    """
    T = 20
    times, dts = regular_schedule(T, dt=0.5)
    results = []

    for substeps in substep_values:
        for int_name, intervention in interventions.items():
            scm = scm_factory()
            x0 = torch.zeros(scm.n_vars)
            n_fine = (T - 1) * substeps
            noise = scm._draw_noise(
                n_fine,
                generator=torch.Generator().manual_seed(seed + hash(int_name) % 1000),
            )

            traj_loop = run_loop_simulate(scm, times, dts, intervention, noise.clone(), x0.clone(), substeps)
            traj_vec = run_vectorized_simulate(scm, times, dts, intervention, noise.clone(), x0.clone(), substeps)

            match = torch.equal(traj_loop, traj_vec)
            max_diff = (traj_loop - traj_vec).abs().max().item()
            label = f"substeps={substeps}, {int_name}"
            results.append((label, match, max_diff))

    return results


# ------------------------------------------------------------------ pytest tests


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


def test_3var_chain_bit_exact():
    """3-variable chain: vectorized == loop for all interventions and substep counts."""
    results = _check_parity(make_3var_scm, INTERVENTIONS, [1, 5])
    for label, match, max_diff in results:
        assert match, f"3-var chain: {label}: max_diff={max_diff:.2e}"


def test_5var_dag_bit_exact():
    """5-variable multi-parent DAG: vectorized == loop for all interventions and substep counts."""
    results = _check_parity(make_5var_scm, INTERVENTIONS, [1, 5])
    for label, match, max_diff in results:
        assert match, f"5-var DAG: {label}: max_diff={max_diff:.2e}"


def test_jittered_schedule_bit_exact():
    """Jittered schedule: vectorized == loop."""
    g = torch.Generator().manual_seed(77)
    T = 25
    times, dts = jittered_schedule(T, dt=0.4, jitter=0.3, generator=g)
    scm = make_3var_scm()
    x0 = torch.zeros(3)

    for substeps in [1, 3]:
        n_fine = (T - 1) * substeps
        noise = scm._draw_noise(n_fine, generator=torch.Generator().manual_seed(88))

        traj_loop = run_loop_simulate(scm, times, dts, None, noise.clone(), x0.clone(), substeps)
        traj_vec = run_vectorized_simulate(scm, times, dts, None, noise.clone(), x0.clone(), substeps)

        assert torch.equal(traj_loop, traj_vec), (
            f"jittered substeps={substeps}: max_diff="
            f"{(traj_loop - traj_vec).abs().max().item():.2e}"
        )


def test_all_linear_flag():
    """_all_linear is True for OU-only and False for neural/mixed."""
    scm_ou = ContinuousSCM.sample_random(4, edge_prob=0.4, mechanism_kind="linear", vectorize=True)
    assert scm_ou._all_linear is True

    scm_neural = ContinuousSCM.sample_random(4, edge_prob=0.4, mechanism_kind="neural", vectorize=True)
    assert scm_neural._all_linear is False


def test_timing():
    """Vectorized path should be faster than loop path (smoke test)."""
    n_vars, T, substeps, n_repeats = 8, 100, 10, 10
    scm = ContinuousSCM.sample_random(n_vars, edge_prob=0.5, mechanism_kind="linear", vectorize=True)
    times, dts = regular_schedule(T, dt=0.3)
    n_fine = (T - 1) * substeps
    x0 = torch.zeros(n_vars)
    intervention = ContinuousIntervention(
        target=0, t_start=5.0, t_end=15.0,
        kind=InterventionKind.HARD, value=1.5,
    )

    # Warm up
    for _ in range(3):
        noise = scm._draw_noise(n_fine, generator=None)
        run_loop_simulate(scm, times, dts, intervention, noise.clone(), x0.clone(), substeps)
        run_vectorized_simulate(scm, times, dts, intervention, noise.clone(), x0.clone(), substeps)

    loop_times_list = []
    for _ in range(n_repeats):
        noise = scm._draw_noise(n_fine, generator=None)
        t0 = time.perf_counter()
        run_loop_simulate(scm, times, dts, intervention, noise.clone(), x0.clone(), substeps)
        t1 = time.perf_counter()
        loop_times_list.append(t1 - t0)

    vec_times_list = []
    for _ in range(n_repeats):
        noise = scm._draw_noise(n_fine, generator=None)
        t0 = time.perf_counter()
        run_vectorized_simulate(scm, times, dts, intervention, noise.clone(), x0.clone(), substeps)
        t1 = time.perf_counter()
        vec_times_list.append(t1 - t0)

    loop_avg = sum(loop_times_list) / len(loop_times_list)
    vec_avg = sum(vec_times_list) / len(vec_times_list)
    assert vec_avg < loop_avg, (
        f"vectorized ({vec_avg*1000:.1f}ms) not faster than loop ({loop_avg*1000:.1f}ms)"
    )


# ------------------------------------------------------------------ standalone


if __name__ == "__main__":
    print("=" * 60)
    print("  Vectorized OU Correctness + Timing Tests")
    print("=" * 60)

    all_pass = True
    for factory, label in [(make_3var_scm, "3-var chain"), (make_5var_scm, "5-var DAG")]:
        print(f"\n--- {label} ---")
        results = _check_parity(factory, INTERVENTIONS, [1, 5])
        for lbl, match, max_diff in results:
            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False
            print(f"  {lbl:45s}  {status}  max_diff={max_diff:.2e}")

    # Jittered schedule
    print("\n--- Jittered schedule ---")
    g = torch.Generator().manual_seed(77)
    T = 25
    times, dts = jittered_schedule(T, dt=0.4, jitter=0.3, generator=g)
    scm = make_3var_scm()
    x0 = torch.zeros(3)
    for substeps in [1, 3]:
        n_fine = (T - 1) * substeps
        noise = scm._draw_noise(n_fine, generator=torch.Generator().manual_seed(88))
        traj_loop = run_loop_simulate(scm, times, dts, None, noise.clone(), x0.clone(), substeps)
        traj_vec = run_vectorized_simulate(scm, times, dts, None, noise.clone(), x0.clone(), substeps)
        match = torch.equal(traj_loop, traj_vec)
        max_diff = (traj_loop - traj_vec).abs().max().item()
        if not match:
            all_pass = False
        print(f"  substeps={substeps}, jittered, no_intervention       {'PASS' if match else 'FAIL'}  max_diff={max_diff:.2e}")

    # Timing
    print("\n--- Timing (n_vars=8, T=100, substeps=10) ---")
    n_vars, T_t, substeps_t, n_repeats = 8, 100, 10, 20
    scm_t = ContinuousSCM.sample_random(n_vars, edge_prob=0.5, mechanism_kind="linear")
    times_t, dts_t = regular_schedule(T_t, dt=0.3)
    n_fine_t = (T_t - 1) * substeps_t
    x0_t = torch.zeros(n_vars)
    int_t = ContinuousIntervention(target=0, t_start=5.0, t_end=15.0,
                                   kind=InterventionKind.HARD, value=1.5)
    for _ in range(3):  # warm up
        noise = scm_t._draw_noise(n_fine_t, generator=None)
        run_loop_simulate(scm_t, times_t, dts_t, int_t, noise.clone(), x0_t.clone(), substeps_t)
        run_vectorized_simulate(scm_t, times_t, dts_t, int_t, noise.clone(), x0_t.clone(), substeps_t)

    loop_ms, vec_ms = [], []
    for _ in range(n_repeats):
        noise = scm_t._draw_noise(n_fine_t, generator=None)
        t0 = time.perf_counter(); run_loop_simulate(scm_t, times_t, dts_t, int_t, noise.clone(), x0_t.clone(), substeps_t); loop_ms.append((time.perf_counter() - t0) * 1000)
        noise = scm_t._draw_noise(n_fine_t, generator=None)
        t0 = time.perf_counter(); run_vectorized_simulate(scm_t, times_t, dts_t, int_t, noise.clone(), x0_t.clone(), substeps_t); vec_ms.append((time.perf_counter() - t0) * 1000)

    l_avg = sum(loop_ms) / len(loop_ms)
    v_avg = sum(vec_ms) / len(vec_ms)
    print(f"  Loop:       {l_avg:.1f} ms")
    print(f"  Vectorized: {v_avg:.1f} ms")
    print(f"  Speedup:    {l_avg / v_avg:.1f}x")

    print(f"\n{'='*60}")
    print(f"  {'ALL CORRECTNESS TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")
