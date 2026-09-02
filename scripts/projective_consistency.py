#!/usr/bin/env python
"""Directly measure the projective inconsistency of finite EM simulators.

Definition 3.1 of the paper requires a continuous-time prior's laws at all
observation schedules to be restrictions of one stochastic process, so in
particular the marginal law of ``X(T)`` must not depend on which interior
points were observed.  The paper's main experiment measures downstream PFN
loss; this script measures the defined object itself, with no PFN involved.

Protocol (simulator-only):
  1. Draw SCMs from the same prior configuration as the paper's ablation
     (``configs/continuous_ablation_grid.yaml``), one mechanism family at
     a time.
  2. For each SCM, simulate many trajectories on the nested dyadic
     schedules ``{0,T}``, ``{0,T/2,T}``, ``{0,T/4,...,T}``, ``{0,T/8,...,T}``
     -- same endpoints, increasingly many interior observations -- at
     per-gap EM resolutions ``s`` in {1,2,4,8,16,32}.
  3. Compare the empirical marginals of ``X(T)`` across schedules with the
     coordinate-averaged 1-D Wasserstein-1 distance.  A projectively
     consistent simulator gives identical marginals for every schedule;
     the measured distance is the schedule dependence at resolution ``s``.
  4. Report a same-law noise floor: the W1 between two independent
     replicates of the *same* (schedule, s) configuration, which is what
     a perfectly consistent simulator would score at this sample size.

Expected result: schedule dependence decreases roughly like 1/s toward the
noise floor for both mechanism families (Euler-Maruyama is not exact even
for linear drift), which validates calling s=1 and s=8 *coarse* and
*refined* EM approximations of one SDE law rather than discrete- versus
continuous-time simulators.

Reproduction:
    python scripts/projective_consistency.py \
        --config configs/continuous_ablation_grid.yaml \
        --mechanism linear --scms 8 --paths 512 \
        --save-json results/projective_consistency/linear.json
    (and again with --mechanism neural)

Raises
------
SystemExit
    If the prior constructor silently rejects configured knobs (same
    guard as scripts/oracle_floor.py; this caught a real bug once).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os

import torch
import yaml

from dotime.prior.continuous.random_sampler import RandomContinuousExtendedPrior
from dotime.prior.continuous.vectorized_sim import _VectorizedPlan, can_vectorize


class _Capturing(RandomContinuousExtendedPrior):
    """Expose the sampled SCM (same pattern as scripts/oracle_floor.py)."""

    def _sample_scm_context(self):
        ctx = super()._sample_scm_context()
        self._ctx = ctx
        return ctx


def _build_prior(config_path: str, mechanism: str, seed: int) -> _Capturing:
    """Construct the paper's prior with MRO-checked kwargs.

    Args:
        config_path: YAML config whose ``prior`` block defines the knobs.
        mechanism: ``linear`` (OU) or ``neural`` drift family.
        seed: Prior RNG seed.

    Returns:
        A ``_Capturing`` prior ready to sample SCMs.

    Raises:
        SystemExit: If a configured knob is not accepted by any class in
            the constructor MRO (silent default fallback would corrupt
            the experiment).
    """
    cfg = yaml.safe_load(open(config_path))["prior"]
    accepted = set()
    for klass in _Capturing.__mro__:
        init = getattr(klass, "__init__", None)
        if init is None or klass is object:
            continue
        accepted |= set(inspect.signature(init).parameters)
    accepted -= {"self", "kwargs", "args"}
    kw = {k: v for k, v in cfg.items() if k in accepted}
    missing = [k for k in ("theta_range", "sigma_range", "weight_scale")
               if k in cfg and k not in kw]
    if missing:
        raise SystemExit(f"prior knobs not accepted by the constructor: {missing}")
    kw.pop("tscm_structure", None)
    kw.pop("num_substeps", None)
    kw["n_min"] = cfg.get("n_min_prior", 3)
    kw.update(mechanism_kind=mechanism,
              p_neural=0.0 if mechanism == "linear" else kw.get("p_neural", 0.0),
              seed=seed)
    return _Capturing(**kw)


def _batched_drift(plan: _VectorizedPlan, x: torch.Tensor) -> torch.Tensor:
    """Path-batched version of ``_VectorizedPlan.drift``.

    Args:
        plan: The per-SCM tensor plan (theta/sigma/W and padded MLP batch).
        x: States of shape ``(paths, n_vars)``.

    Returns:
        Drift of shape ``(paths, n_vars)``, numerically identical per path
        to the reference single-path drift (validated at start-up).
    """
    d = -plan.theta * x + x @ plan.W.T
    if plan.has_neural:
        inp = x[:, plan.gather_idx] * plan.gather_mask        # (P, nn, d_max)
        h = torch.tanh(torch.einsum("nhd,pnd->pnh", plan.W1, inp) + plan.b1)
        o = torch.tanh(torch.einsum("noh,pnh->pno", plan.W2, h) + plan.b2)
        nn_drift = plan.out_scale * o.squeeze(-1)             # (P, nn)
        d = d.clone()
        d[:, plan.neural_rows] = (-plan.theta[plan.neural_rows]
                                  * x[:, plan.neural_rows] + nn_drift)
    return d


def _simulate_endpoints(plan: _VectorizedPlan, times: torch.Tensor,
                        dts: torch.Tensor, num_substeps: int, paths: int,
                        gen: torch.Generator,
                        noise: torch.Tensor | None = None) -> torch.Tensor:
    """Simulate ``paths`` observational EM trajectories, returning X(T).

    Mirrors the reference loop's noise indexing and snap-to-observation
    substep times; x0 = 0 for every path (the simulator's default).

    Args:
        plan: Per-SCM tensor plan.
        times: Observation times ``(T,)``.
        dts: Gaps ``(T-1,)``.
        num_substeps: EM substeps per observed gap.
        paths: Number of independent trajectories.
        gen: RNG for the noise (ignored when ``noise`` is given).
        noise: Optional pre-drawn noise ``(paths, (T-1)*num_substeps, n)``
            for the start-up equivalence check.

    Returns:
        Endpoint states ``X(times[-1])`` of shape ``(paths, n_vars)``.
    """
    n = plan.n
    if noise is None:
        noise = torch.randn(paths, (times.numel() - 1) * num_substeps, n,
                            generator=gen, dtype=plan.dtype)
    x = torch.zeros(paths, n, dtype=plan.dtype)
    for i in range(times.numel() - 1):
        fine_dt = dts[i] / num_substeps
        sqrt_dt = torch.sqrt(fine_dt)
        for k in range(num_substeps):
            x = (x + _batched_drift(plan, x) * fine_dt
                 + plan.sigma * sqrt_dt * noise[:, i * num_substeps + k])
    return x


def _validate_batched(scm, plan: _VectorizedPlan, times: torch.Tensor,
                      dts: torch.Tensor, num_substeps: int) -> None:
    """Assert the batched EM matches ``scm.simulate`` with shared noise.

    Raises:
        SystemExit: If any endpoint coordinate differs by more than 1e-5
            (the repo's vectorisation-equivalence tolerance).
    """
    g = torch.Generator().manual_seed(0)
    noise = torch.randn((times.numel() - 1) * num_substeps, plan.n, generator=g)
    _, ref = scm.simulate(times, dts, noise=noise, num_substeps=num_substeps)
    got = _simulate_endpoints(plan, times, dts, num_substeps, paths=1,
                              gen=g, noise=noise.unsqueeze(0))
    if (got[0] - ref[-1]).abs().max().item() > 1e-5:
        raise SystemExit("batched EM diverged from scm.simulate reference")


def _w1_per_var(a: torch.Tensor, b: torch.Tensor) -> float:
    """Coordinate-averaged 1-D Wasserstein-1 between two sample sets.

    Args:
        a, b: Tensors of shape ``(paths, n_vars)`` -- samples of X(T).

    Returns:
        Mean over variables of the 1-D W1 distance (equal sample sizes,
        so W1 is the mean absolute difference of sorted samples).
    """
    sa, _ = torch.sort(a, dim=0)
    sb, _ = torch.sort(b, dim=0)
    return (sa - sb).abs().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/continuous_ablation_grid.yaml")
    ap.add_argument("--mechanism", choices=["linear", "neural"], required=True)
    ap.add_argument("--scms", type=int, default=8)
    ap.add_argument("--paths", type=int, default=8192)
    ap.add_argument("--T-end", type=float, default=2.0,
                    help="Common endpoint T; gaps of the coarsest schedule "
                         "match the widest mixed-schedule gaps used in "
                         "the paper.")
    ap.add_argument("--substeps", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="Interior subdivisions of [0, T]: 1 -> {0,T}, "
                         "2 -> {0,T/2,T}, ...")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--save-json", required=True)
    a = ap.parse_args()

    prior = _build_prior(a.config, a.mechanism, a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    # rows[(scm, k, s, replicate)] -> (paths, n_vars) samples of X(T).
    results = {f"{ka}v{kb}@s{s}": [] for s in a.substeps
               for i, ka in enumerate(a.gaps) for kb in a.gaps[i + 1:]}
    floors = {f"s{s}": [] for s in a.substeps}

    for m in range(a.scms):
        prior.generate_sample()          # forces a fresh SCM into _ctx
        scm = prior._ctx.scm
        if not can_vectorize(scm.mechanisms):
            raise SystemExit("SCM has a mechanism the plan cannot encode")
        plan = _VectorizedPlan(scm.mechanisms, device=torch.device("cpu"),
                               dtype=torch.float32)
        # Start-up equivalence gate: batched EM vs the reference simulator.
        vt = torch.linspace(0.0, a.T_end, 3)
        _validate_batched(scm, plan, vt, vt[1:] - vt[:-1], 4)
        samples = {}
        for k in a.gaps:
            times = torch.linspace(0.0, a.T_end, k + 1)
            dts = times[1:] - times[:-1]
            for s in a.substeps:
                for rep in range(2):     # two replicates -> noise floor
                    samples[(k, s, rep)] = _simulate_endpoints(
                        plan, times, dts, s, a.paths, gen)
        for s in a.substeps:
            # Schedule dependence: pairwise W1 across schedules (rep 0).
            for i, ka in enumerate(a.gaps):
                for kb in a.gaps[i + 1:]:
                    results[f"{ka}v{kb}@s{s}"].append(
                        _w1_per_var(samples[(ka, s, 0)], samples[(kb, s, 0)]))
            # Noise floor: same schedule, independent replicate, averaged
            # over schedules (the self-distance of a consistent simulator).
            floors[f"s{s}"].append(sum(
                _w1_per_var(samples[(k, s, 0)], samples[(k, s, 1)])
                for k in a.gaps) / len(a.gaps))
        print(f"[{a.mechanism}] SCM {m + 1}/{a.scms} done", flush=True)

    def _agg(v):
        t = torch.tensor(v)
        return {"mean": t.mean().item(), "sd": t.std(unbiased=False).item()}

    out = {
        "config": vars(a),
        "pairwise_w1": {key: _agg(v) for key, v in results.items()},
        "noise_floor": {key: _agg(v) for key, v in floors.items()},
    }
    os.makedirs(os.path.dirname(a.save_json), exist_ok=True)
    json.dump(out, open(a.save_json, "w"), indent=2)
    print(f"wrote {a.save_json}")


if __name__ == "__main__":
    main()
