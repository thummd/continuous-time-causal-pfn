"""Phase 14b tests: wind-tunnel chamber adapter.

Phase 14b switches the chamber benchmark from
``lt_walks_v1/actuators_white`` (white-noise actuators, Pearson
:math:`r \\approx 0` against ground truth, no causal-effect
signal) to ``wt_intake_impulse_v1`` (explicit intervention column,
real downstream dynamics).  These tests pin the new code path:

1. ``CausalChamberLoader.extract_episodes(intervention_source=
   "intervention_column")`` reads the chamber's flag column directly
   and dispenses with the change-point heuristic.
2. ``require_value_change=True`` filters out decoy pulses where
   ``treatment_var`` does not actually change.
3. Per-row timestamps propagate through to ``timestamps_obs`` /
   ``timestamps_post``.
4. The ``causal_chamber_wt.load_wt_episodes`` wrapper produces
   episodes that flow cleanly into
   :func:`build_causal_chamber_batch`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dotime.data.causal_chamber import (
    WT_SUBGRAPH_5VAR,
    WT_TREATMENT,
    CausalChamberLoader,
)
from dotime.data.causal_chamber_ct import build_causal_chamber_batch


N_MAX = 12
WT_VARS = list(WT_SUBGRAPH_5VAR)


def _fake_wt_dataframe(
    n_rows: int = 400,
    base_dt: float = 0.152,
    jitter: float = 0.005,
    period_rows: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthesise a wt_intake_impulse_v1-shaped DataFrame.

    Mirrors the wt rig: an explicit ``intervention`` column with
    single-row pulses, a binary ``load_in`` treatment that toggles
    at every other pulse (so half the pulses are real transitions
    and half are decoys), and downstream variables that respond to
    ``load_in`` with a slow ramp.
    """
    rng = np.random.RandomState(seed)
    cols = ["timestamp", "intervention"] + WT_VARS
    df = pd.DataFrame({c: np.zeros(n_rows, dtype=np.float64) for c in cols})

    # Irregular timestamps with realistic-looking jitter.
    gaps = rng.uniform(base_dt - jitter, base_dt + jitter, size=n_rows - 1)
    t0 = 1000.0
    df["timestamp"] = np.concatenate(([t0], t0 + np.cumsum(gaps)))

    # Intervention pulses: single-row ones at regular spacing, with
    # an additional decoy pulse halfway between to mimic the rig.
    real_toggles = list(range(period_rows, n_rows, period_rows * 2))
    decoy_toggles = list(range(period_rows + period_rows, n_rows, period_rows * 2))
    pulse_rows = sorted(set(real_toggles + decoy_toggles))
    df.loc[pulse_rows, "intervention"] = 1.0

    # load_in flips at the real toggles only.
    load_in = np.zeros(n_rows, dtype=np.float64)
    load_in[:] = 0.01
    state = 0.01
    for r in real_toggles:
        state = 1.0 if state == 0.01 else 0.01
        load_in[r:] = state
    df["load_in"] = load_in

    # current_in tracks load_in tightly.
    df["current_in"] = 300.0 + 200.0 * load_in + rng.randn(n_rows) * 5.0
    # rpm_in ramps slowly toward a load_in-dependent setpoint.
    target = 600.0 + 2000.0 * load_in
    rpm = np.zeros(n_rows)
    rpm[0] = target[0]
    alpha = 0.15
    for i in range(1, n_rows):
        rpm[i] = (1 - alpha) * rpm[i - 1] + alpha * target[i] + rng.randn() * 5.0
    df["rpm_in"] = rpm
    # Pressure variables: small perturbations + tiny load_in coupling.
    df["pressure_intake"] = 95740.0 + rng.randn(n_rows) * 1.5 + 5.0 * load_in
    df["pressure_downwind"] = 95739.0 + rng.randn(n_rows) * 1.5 + 8.0 * load_in
    return df


def _build_loader() -> CausalChamberLoader:
    """Construct a loader without hitting the network."""
    loader = CausalChamberLoader.__new__(CausalChamberLoader)
    loader.subgraph_vars = WT_VARS
    loader.var_to_idx = {v: i for i, v in enumerate(WT_VARS)}
    loader.n_max = 41
    return loader


# ---------------------------------------------------------------------- extract_episodes contract


def test_intervention_column_finds_pulses_without_change_point_detector():
    df = _fake_wt_dataframe(n_rows=400, period_rows=40)
    loader = _build_loader()
    eps = loader.extract_episodes(
        df,
        obs_window=20,
        post_window=15,
        intervention_source="intervention_column",
        treatment_var=WT_TREATMENT,
        require_value_change=True,
    )
    # Every other pulse is real (load_in flips); the rest are decoys.
    expected_real_toggles = [40, 120, 200, 280, 360]
    found_cps = [ep["changepoint"] for ep in eps]
    assert found_cps == expected_real_toggles, (
        f"expected real toggles {expected_real_toggles}, got {found_cps}"
    )


def test_decoy_pulses_dropped_when_require_value_change_true():
    df = _fake_wt_dataframe(n_rows=400, period_rows=40)
    loader = _build_loader()
    eps_filtered = loader.extract_episodes(
        df, obs_window=20, post_window=15,
        intervention_source="intervention_column",
        treatment_var=WT_TREATMENT, require_value_change=True,
    )
    eps_unfiltered = loader.extract_episodes(
        df, obs_window=20, post_window=15,
        intervention_source="intervention_column",
        treatment_var=WT_TREATMENT, require_value_change=False,
    )
    # Decoys are exactly the pulses where pre-pulse and post-pulse
    # load_in agree.  The synthetic frame has 5 real toggles + 4
    # decoys; the filter must drop the decoys.
    assert len(eps_unfiltered) == 9
    assert len(eps_filtered) == 5
    for ep in eps_filtered:
        assert ep["intervention_value"] != ep["intervention_value_pre"]


def test_intervention_value_is_post_pulse_load_in():
    df = _fake_wt_dataframe(n_rows=400, period_rows=40)
    loader = _build_loader()
    eps = loader.extract_episodes(
        df, obs_window=20, post_window=15,
        intervention_source="intervention_column",
        treatment_var=WT_TREATMENT, require_value_change=True,
    )
    # First real toggle: load_in goes 0.01 -> 1.0.
    assert eps[0]["intervention_value"] == pytest.approx(1.0)
    assert eps[0]["intervention_value_pre"] == pytest.approx(0.01)
    # Second real toggle: load_in goes 1.0 -> 0.01.
    assert eps[1]["intervention_value"] == pytest.approx(0.01)
    assert eps[1]["intervention_value_pre"] == pytest.approx(1.0)


def test_real_timestamps_propagate_to_episode():
    df = _fake_wt_dataframe(n_rows=400, period_rows=40)
    loader = _build_loader()
    eps = loader.extract_episodes(
        df, obs_window=20, post_window=15,
        intervention_source="intervention_column",
        treatment_var=WT_TREATMENT, require_value_change=True,
    )
    ep = eps[0]
    assert "timestamps_obs" in ep
    assert "timestamps_post" in ep
    assert ep["timestamps_obs"].shape == (20,)
    assert ep["timestamps_post"].shape == (15,)
    # Anchored to seconds-since-episode-start.
    assert ep["timestamps_obs"][0] == pytest.approx(0.0, abs=1e-6)
    # intervention_dt straddles the change-point.
    assert ep["intervention_dt"] == pytest.approx(
        float(ep["timestamps_post"][0] - ep["timestamps_obs"][-1]), abs=1e-6
    )
    assert ep["intervention_dt"] > 0.0


def test_extract_episodes_rejects_missing_intervention_column():
    df = _fake_wt_dataframe(n_rows=200, period_rows=40).drop(columns=["intervention"])
    loader = _build_loader()
    with pytest.raises(ValueError, match="intervention"):
        loader.extract_episodes(
            df, obs_window=20, post_window=15,
            intervention_source="intervention_column",
            treatment_var=WT_TREATMENT,
        )


def test_extract_episodes_rejects_missing_treatment_var():
    df = _fake_wt_dataframe(n_rows=200, period_rows=40)
    loader = _build_loader()
    with pytest.raises(ValueError, match="treatment_var"):
        loader.extract_episodes(
            df, obs_window=20, post_window=15,
            intervention_source="intervention_column",
            treatment_var=None,
        )


def test_extract_episodes_actuator_step_path_unchanged():
    """Default mode still uses the change-point detector and
    ignores the intervention column -- backward compat for lt
    benchmarks."""
    df = _fake_wt_dataframe(n_rows=200, period_rows=40)
    loader = _build_loader()
    # actuator_step works only on LT_ACTUATORS-named columns; the
    # WT subgraph contains none, so we should get zero episodes
    # (rather than the load_in-toggle list).
    eps = loader.extract_episodes(
        df, obs_window=20, post_window=15,
        intervention_source="actuator_step",
    )
    assert eps == []


# ---------------------------------------------------------------------- batch builder integration


def test_episode_flows_into_build_causal_chamber_batch():
    df = _fake_wt_dataframe(n_rows=400, period_rows=40)
    loader = _build_loader()
    eps = loader.extract_episodes(
        df, obs_window=30, post_window=15,
        intervention_source="intervention_column",
        treatment_var=WT_TREATMENT, require_value_change=True,
    )
    assert len(eps) >= 1
    batch = build_causal_chamber_batch(eps[0], query_var="rpm_in", n_max=N_MAX)
    # 14 query offsets (T_post - 1 = 14).
    B = batch["Y_true"].shape[0]
    assert B == 14
    # Times come from real timestamps, not a uniform 0.1 s grid.
    times = batch["times"][0].numpy()
    expected = np.concatenate([eps[0]["timestamps_obs"], eps[0]["timestamps_post"]])
    np.testing.assert_allclose(times, expected, rtol=0, atol=1e-6)
    # int_onset_idx is the first post-intervention sample.
    assert int(batch["int_onset_idx"][0].item()) == 30
    # Intervention target is load_in (first column).
    assert int(batch["intervention_target"][0].item()) == WT_VARS.index(WT_TREATMENT)


def test_rejects_episode_too_close_to_dataframe_edge():
    df = _fake_wt_dataframe(n_rows=80, period_rows=10)
    loader = _build_loader()
    eps = loader.extract_episodes(
        df, obs_window=30, post_window=15,
        intervention_source="intervention_column",
        treatment_var=WT_TREATMENT, require_value_change=True,
    )
    # First real toggle at row 10 has fewer than 30 pre-rows; last
    # at row >=66 has fewer than 15 post-rows.  Some episodes survive
    # in the middle.
    cps = [ep["changepoint"] for ep in eps]
    for cp in cps:
        assert cp >= 30
        assert cp + 15 <= 80
