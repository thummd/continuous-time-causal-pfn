r"""Wind-tunnel adapter for CausalChamber wt_intake_impulse_v1.

Sister module to :mod:`dotime.data.causal_chamber_ct`, specialised
for the wind-tunnel rig.  The light-tunnel ``lt_walks_v1`` benchmark
ships with white-noise actuators (>99 % of consecutive samples flip
by more than half a standard deviation), and a change-point
detector run on those actuators surfaces "intervention episodes"
that are really cross-sections of a continuously randomised
process.  The Pearson correlation between any predictor and the
queried sensor is statistically zero on that dataset (see
:doc:`PHASE_14B_PLAN.md` and Appendix D of the workshop paper for
the full analysis).

``wt_intake_impulse_v1`` fixes the three structural issues:

1. **Explicit ``intervention`` column.**  Each ``0 \to 1`` pulse is a
   known toggle event on the intake-fan setpoint ``load_in``, with
   the post-pulse value reported in the ``load_in`` column itself.
   No change-point heuristic.
2. **Real treatment dynamics.**  ``load_in`` toggles between 0.01
   and 1.0; downstream variables (``current_in``, ``rpm_in``,
   ``pressure_intake``, ``pressure_downwind``) respond on
   distinguishable timescales.  Pearson :math:`r` against ground
   truth is non-trivial.
3. **Real timestamps.**  Per-row ``timestamp`` column (median dt
   ~0.15 s, max ~2.4 s).

This module exposes :func:`load_wt_episodes`, a thin convenience
wrapper that loads the dataset, filters episodes to clean toggles
of ``load_in``, and yields episode dicts compatible with
:func:`build_causal_chamber_batch`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from dotime.data.causal_chamber import (
    WT_SUBGRAPH_5VAR,
    WT_TREATMENT,
    CausalChamberLoader,
)


def load_wt_episodes(
    experiment: str = "load_out_0.5_osr_downwind_4",
    *,
    dataset_name: str = "wt_intake_impulse_v1",
    root: str = "/tmp/causalchamber",
    subgraph_vars: Optional[Sequence[str]] = None,
    treatment_var: str = WT_TREATMENT,
    obs_window: int = 50,
    post_window: int = 20,
    download: bool = True,
    max_episodes: Optional[int] = None,
) -> List[Dict]:
    """Load wind-tunnel intervention episodes ready for the CT batch builder.

    Parameters
    ----------
    experiment : name of the wt experiment within the dataset.  The
        five available v1 experiments are
        ``load_out_{0.01, 0.5, 1}_osr_downwind_4`` and
        ``load_out_0.5_osr_downwind_{2, 4, 8}``; defaults to
        ``load_out_0.5_osr_downwind_4`` (the mid-range default
        recommended by Phase 14b).
    dataset_name : CausalChamber dataset.  Default
        ``wt_intake_impulse_v1`` is the impulse-intervention release.
    root : local directory for the CausalChamber download cache.
    subgraph_vars : variables to read into the SCM.  Defaults to
        :data:`WT_SUBGRAPH_5VAR`.
    treatment_var : the variable whose value is reported as the
        intervention target per episode.  Default
        :data:`WT_TREATMENT` (``load_in``).
    obs_window, post_window : pre/post sample counts; the chamber
        runs at ~6.6 Hz so 50 + 20 covers ~10 s of physical time.
    download : passed through to the CausalChamber loader.
    max_episodes : cap the number of returned episodes.  Useful for
        smoke tests; the v1 release contains ~5000 clean episodes
        per experiment so unfiltered loads are slow.

    Returns
    -------
    list of episode dicts
        Compatible with
        :func:`dotime.data.causal_chamber_ct.build_causal_chamber_batch`.
        Each dict carries the real per-row ``timestamps_obs`` /
        ``timestamps_post`` arrays plus the ``intervention_value``
        post-toggle and ``intervention_value_pre`` for bookkeeping.
    """
    subgraph = list(subgraph_vars) if subgraph_vars else list(WT_SUBGRAPH_5VAR)
    if treatment_var not in subgraph:
        raise ValueError(
            f"treatment_var={treatment_var!r} must appear in "
            f"subgraph_vars (got {subgraph})"
        )

    loader = CausalChamberLoader(
        dataset_name=dataset_name,
        root=root,
        subgraph_vars=subgraph,
        download=download,
    )
    df = loader.load_experiment(experiment)
    eps = loader.extract_episodes(
        df,
        obs_window=obs_window,
        post_window=post_window,
        intervention_source="intervention_column",
        treatment_var=treatment_var,
        require_value_change=True,
    )
    for ep in eps:
        ep.setdefault("experiment", experiment)
    if max_episodes is not None:
        eps = eps[:max_episodes]
    return eps
