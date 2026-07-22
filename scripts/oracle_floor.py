"""Monte-Carlo oracle floor for the OU cells (paper appendix "An oracle
floor for the OU cells").

The oracle is given the true SCM and the system state at intervention
onset, then re-simulates M continuations with fresh Brownian noise and
reads off the empirical quantiles of Y at the query time.  Because it
knows the mechanism that the PFN must infer in context, its expected
pinball loss lower-bounds that of any such predictor.  If the trained
models sat at this floor, "no consistent effect for OU" would mean
"both models are already optimal"; they do not, so the OU result is a
null between the two simulators.

Reproduces the appendix table (three configs, ~40 min each on one GPU):

    python scripts/oracle_floor.py --config configs/continuous_ablation_grid.yaml \\
        --schedule regular --substeps 1 --n-samples 150 --mc 200 \\
        --save-json results/oracle_iv1/oracle_regular_s1.json
    python scripts/oracle_floor.py --config configs/continuous_ablation_grid.yaml \\
        --schedule mixed --substeps 1 --n-samples 150 --mc 200 \\
        --save-json results/oracle_iv1/oracle_mixed_s1.json
    python scripts/oracle_floor.py --config configs/continuous_ablation_grid.yaml \\
        --schedule mixed --substeps 8 --n-samples 60 --mc 100 \\
        --save-json results/oracle_iv1/oracle_mixed_s8.json

The published values were produced with the target-normalisation clip at
10.0; this script reads the clip from ``dotime.data.normalization`` so it
tracks whichever tree it runs in.  The clip does not bind for the
stability-respecting prior used in the paper.
"""
import argparse, inspect, json, math, os
import numpy as np, torch, yaml

from dotime.prior.continuous.random_sampler import RandomContinuousExtendedPrior
from dotime.data.normalization import per_variable_normalize

TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]

try:  # module-level in this tree; some older trees hardcode 10.0 inline
    from dotime.data.normalization import Y_NORM_CLIP as CLIP
except ImportError:  # pragma: no cover
    CLIP = 10.0


class _Capturing(RandomContinuousExtendedPrior):
    """Expose the sampled SCM and the exact intervention object it was given."""

    def _sample_scm_context(self):
        ctx = super()._sample_scm_context()
        self._ctx = ctx
        scm = ctx.scm
        if not getattr(scm, "_oracle_wrapped", False):
            orig = scm.simulate
            def recording(*a, **kw):
                if kw.get("intervention") is not None:
                    self._interv = kw["intervention"]
                return orig(*a, **kw)
            scm.simulate = recording
            scm._oracle_wrapped = True
            scm._oracle_orig = orig
        return ctx


def pinball(y, qs):
    return sum(max(t * (y - q), (t - 1) * (y - q)) for t, q in zip(TAUS, qs)) / len(TAUS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--schedule", default="regular")
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--mc", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260508)
    ap.add_argument("--intervention-value-scale", type=float, default=None)
    ap.add_argument("--save-json", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))["prior"]
    # Collect accepted kwargs from every class in the MRO: the graph knobs
    # live on RandomContinuousExtendedPrior while the mechanism, schedule
    # and intervention knobs live on ContinuousExtendedPrior.  Filtering
    # against only one of them silently drops the other's settings and the
    # prior falls back to defaults.
    accepted = set()
    for klass in _Capturing.__mro__:
        init = getattr(klass, "__init__", None)
        if init is None or klass is object:
            continue
        accepted |= set(inspect.signature(init).parameters)
    accepted -= {"self", "kwargs", "args"}
    kw = {k: v for k, v in cfg.items() if k in accepted}
    missing = [k for k in ("theta_range", "sigma_range", "weight_scale", "t_range",
                           "intervention_value_scale") if k in cfg and k not in kw]
    if missing:
        raise SystemExit(f"prior knobs not accepted by the constructor: {missing}")
    kw.pop("tscm_structure", None); kw.pop("num_substeps", None)
    kw["n_min"] = cfg.get("n_min_prior", 3)
    kw.update(schedule=a.schedule, mechanism_kind="linear", p_neural=0.0, seed=a.seed)
    if a.intervention_value_scale is not None:
        kw["intervention_value_scale"] = a.intervention_value_scale
    prior = _Capturing(**kw); prior.num_substeps = a.substeps

    losses, checked, skipped = [], 0, 0
    for _ in range(a.n_samples):
        s = prior.generate_sample()
        scm, interv = prior._ctx.scm, getattr(prior, "_interv", None)
        perm = list(prior._ctx.canonical_perm)
        n = int(s["num_vars"])
        times, dts = s["times"], s["dts"]
        onset = int(s["int_onset_idx"])
        t_q = float(s["t_query"])
        q_idx = int(torch.argmin((times - t_q).abs()).item())
        q_canon = int(torch.as_tensor(s["query_target"]).item())
        if interv is None or q_idx <= onset or q_canon >= n:
            skipped += 1; continue

        # canonical (padded) -> topological, the order scm.simulate works in
        x0 = torch.zeros(n, dtype=s["X_obs"].dtype)
        for j in range(n):
            x0[perm[j]] = s["X_obs"][onset, j]
        q_topo = perm[q_canon]

        # validate the index/permutation logic against the generator's own output
        if checked < 5:
            assert abs(float(s["X_int"][q_idx, q_canon]) - float(s["Y_true"])) < 1e-4, \
                "index/permutation mismatch vs Y_true"
            checked += 1

        ys = []
        raw_sim = getattr(scm, "_oracle_orig", scm.simulate)
        # The substep kwarg is spelled differently across trees.
        sub_kw = ("num_substeps" if "num_substeps" in
                  inspect.signature(raw_sim).parameters else "substeps")
        for _m in range(a.mc):
            _, Xm = raw_sim(times[onset:], dts[onset:], intervention=interv,
                            x0=x0.clone(), **{sub_kw: a.substeps})
            ys.append(float(Xm[q_idx - onset, q_topo]))
        q = np.quantile(np.asarray(ys), TAUS)

        vm = torch.as_tensor(s["variable_mask"]).reshape(1, -1).float()
        _, means, stds = per_variable_normalize(s["X_obs"].unsqueeze(0), vm)
        mu, sd = float(means[0, q_canon]), float(stds[0, q_canon])
        y_n = float(np.clip((float(s["Y_true"]) - mu) / sd, -CLIP, CLIP))
        q_n = np.clip((q - mu) / sd, -CLIP, CLIP)
        losses.append(pinball(y_n, q_n))

    res = {"schedule": a.schedule, "substeps": a.substeps, "mc": a.mc,
           "intervention_value_scale": kw.get("intervention_value_scale"),
           "n_used": len(losses), "n_skipped": skipped,
           "oracle_pinball": float(np.mean(losses)),
           "sem": float(np.std(losses) / math.sqrt(len(losses)))}
    print(json.dumps(res, indent=1))
    if a.save_json:
        json.dump(res, open(a.save_json, "w"), indent=1)


if __name__ == "__main__":
    main()
