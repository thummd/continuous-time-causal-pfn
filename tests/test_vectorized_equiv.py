"""The vectorised simulator must reproduce the reference loop exactly.

This is the correctness gate for `vectorized_sim.py`. It compares
`simulate_vectorized` against `ContinuousSCM.simulate`'s per-variable loop on
OU, neural, and mixed graphs, with every intervention kind, and under the
shared-noise counterfactual pairing that the training target depends on.
"""
import torch
import pytest

from dotime.prior.continuous.continuous_scm import (
    ContinuousSCM, ContinuousIntervention, InterventionKind,
)
from dotime.prior.continuous.vectorized_sim import (
    _VectorizedPlan, simulate_vectorized, can_vectorize,
)

TOL = 1e-5  # bmm vs sequential matmul differ only in the last few bits


def _make_scm(n, p_neural, seed):
    g = torch.Generator().manual_seed(seed)
    scm = ContinuousSCM.sample_random(
        n_vars=n, edge_prob=0.5, theta_range=(0.1, 0.5), sigma_range=(0.2, 0.6),
        weight_scale=0.3, mechanism_kind=("neural" if p_neural else "linear"),
        p_neural=p_neural, neural_hidden_dim=8, neural_out_scale_range=(0.5, 2.0),
        generator=g,
    )
    return scm


def _attach_plan(scm):
    scm._vec_plan = _VectorizedPlan(scm.mechanisms, scm.device, scm.dtype)


def _run_both(scm, intervention, num_substeps, seed):
    T = 12
    times = torch.arange(T, dtype=torch.float32)
    dts = times[1:] - times[:-1]
    fine = (T - 1) * num_substeps
    noise = torch.randn(fine, scm.n_vars, generator=torch.Generator().manual_seed(seed))
    x0 = torch.randn(scm.n_vars, generator=torch.Generator().manual_seed(seed + 1))
    _, ref = scm.simulate(times, dts, intervention=intervention, x0=x0.clone(),
                          noise=noise.clone(), num_substeps=num_substeps)
    _attach_plan(scm)
    _, vec = simulate_vectorized(scm, times, dts, intervention, x0.clone(),
                                 noise.clone(), num_substeps)
    return ref, vec


@pytest.mark.parametrize("p_neural", [0.0, 1.0, 0.5])
@pytest.mark.parametrize("num_substeps", [1, 8])
def test_no_intervention(p_neural, num_substeps):
    scm = _make_scm(6, p_neural, seed=100 + int(p_neural * 10))
    assert can_vectorize(scm.mechanisms)
    ref, vec = _run_both(scm, None, num_substeps, seed=7)
    assert torch.allclose(ref, vec, atol=TOL, rtol=TOL), (ref - vec).abs().max()


@pytest.mark.parametrize("p_neural", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("kind", [InterventionKind.HARD, InterventionKind.SOFT,
                                  InterventionKind.TIME_VARYING])
def test_with_intervention(p_neural, kind):
    scm = _make_scm(7, p_neural, seed=200 + int(p_neural * 10))
    if kind is InterventionKind.TIME_VARYING:
        val = lambda t: 0.5 * t  # noqa: E731
    else:
        val = 1.3
    interv = ContinuousIntervention(target=2, t_start=3.0, t_end=8.0, kind=kind, value=val)
    ref, vec = _run_both(scm, interv, num_substeps=8, seed=11)
    assert torch.allclose(ref, vec, atol=TOL, rtol=TOL), (ref - vec).abs().max()


def test_counterfactual_pairing_preserved():
    """Shared noise -> the vectorised obs/int pair matches the reference pair,
    and the pre-intervention windows coincide (the property the target needs)."""
    scm = _make_scm(6, 0.5, seed=321)
    T = 12
    times = torch.arange(T, dtype=torch.float32)
    dts = times[1:] - times[:-1]
    noise = torch.randn((T - 1) * 8, scm.n_vars, generator=torch.Generator().manual_seed(5))
    x0 = torch.zeros(scm.n_vars)
    interv = ContinuousIntervention(target=1, t_start=5.0, t_end=9.0,
                                    kind=InterventionKind.HARD, value=2.0)
    # reference obs/int with shared noise
    _, ref_obs = scm.simulate(times, dts, intervention=None, x0=x0.clone(),
                              noise=noise.clone(), num_substeps=8)
    _, ref_int = scm.simulate(times, dts, intervention=interv, x0=x0.clone(),
                              noise=noise.clone(), num_substeps=8)
    _attach_plan(scm)
    _, vec_obs = simulate_vectorized(scm, times, dts, None, x0.clone(), noise.clone(), 8)
    _, vec_int = simulate_vectorized(scm, times, dts, interv, x0.clone(), noise.clone(), 8)
    assert torch.allclose(ref_obs, vec_obs, atol=TOL, rtol=TOL)
    assert torch.allclose(ref_int, vec_int, atol=TOL, rtol=TOL)
    # pre-intervention (t < 5) obs and int coincide under shared noise
    assert torch.allclose(vec_obs[:5], vec_int[:5], atol=TOL, rtol=TOL)
