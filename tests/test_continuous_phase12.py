"""Phase 12 tests: real CausalChamber timestamps end-to-end.

Earlier versions of the CT adapter discarded the per-row ``timestamp``
column and synthesised a uniform 10 Hz grid, which is exactly the
schedule conflation Definition 3.1 of the workshop paper critiques.
Phase 12 propagates real timestamps from ``extract_episodes`` through
``build_causal_chamber_batch``.  These tests pin the new contract.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from dotime.data.causal_chamber import CausalChamberLoader, LT_SUBGRAPH_5VAR
from dotime.data.causal_chamber_ct import build_causal_chamber_batch


N_MAX = 12


# ---------------------------------------------------------------------- extract_episodes


def _fake_chamber_dataframe(
    n_rows: int = 200,
    base_dt: float = 0.147,
    jitter: float = 0.01,
    big_gap_every: int = 17,
    big_gap_extra: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthesise a CausalChamber-shaped DataFrame with irregular timestamps.

    Mirrors the column layout of ``lt_walks_v1`` (the meta + actuator +
    sensor union) and embeds at least one actuator change-point so
    :meth:`CausalChamberLoader.extract_episodes` can find it.
    """
    rng = np.random.RandomState(seed)
    cols = ["timestamp", "config", "counter", "flag", "intervention",
            "pol_1", "red", "green", "blue", "osr_c"]
    df = pd.DataFrame(
        {c: np.zeros(n_rows, dtype=np.float64) for c in cols}
    )

    # Irregular timestamps: nominal cadence + uniform jitter, with a
    # bigger gap every ``big_gap_every`` rows.
    gaps = rng.uniform(base_dt - jitter, base_dt + jitter, size=n_rows - 1)
    for k in range(0, n_rows - 1, big_gap_every):
        gaps[k] = base_dt + big_gap_extra
    t0 = 1000.0  # arbitrary epoch
    df["timestamp"] = np.concatenate(([t0], t0 + np.cumsum(gaps)))

    # Actuator: step from 0 -> 1 at row n_rows // 2 (the changepoint).
    df["pol_1"] = (np.arange(n_rows) >= n_rows // 2).astype(np.float64)

    # Sensors: random noise; "red" jumps ~ pol_1 to make the
    # post-intervention statistics visibly different from pre.
    for c in ("red", "green", "blue", "osr_c"):
        df[c] = rng.randn(n_rows) * 0.05
    df["red"] += df["pol_1"] * 1.0

    return df


def test_extract_episodes_attaches_real_timestamps():
    df = _fake_chamber_dataframe()
    loader = CausalChamberLoader.__new__(CausalChamberLoader)
    loader.subgraph_vars = LT_SUBGRAPH_5VAR
    loader.var_to_idx = {v: i for i, v in enumerate(LT_SUBGRAPH_5VAR)}
    eps = loader.extract_episodes(df, obs_window=40, post_window=20)
    assert len(eps) >= 1, "synthetic dataframe should produce one episode"

    ep = eps[0]
    assert "timestamps_obs" in ep
    assert "timestamps_post" in ep
    assert ep["timestamps_obs"].shape == (40,)
    assert ep["timestamps_post"].shape == (20,)
    # Both arrays anchored to seconds-since-episode-start.
    assert ep["timestamps_obs"][0] == pytest.approx(0.0, abs=1e-6)
    # Real CausalChamber data has irregular gaps; the synthetic DF
    # injects a bigger gap every 17 rows, so over 60 rows we should
    # see meaningful std of inter-row gaps.
    full_ts = np.concatenate([ep["timestamps_obs"], ep["timestamps_post"]])
    diffs = np.diff(full_ts)
    assert diffs.std() > 0.005, "synthetic schedule should be irregular"
    # The intervention_dt field reports the gap straddling the
    # change-point.
    assert ep["intervention_dt"] > 0.0
    assert ep["intervention_dt"] == pytest.approx(
        float(ep["timestamps_post"][0] - ep["timestamps_obs"][-1]), abs=1e-6,
    )


def test_extract_episodes_without_timestamp_column_is_silent():
    df = _fake_chamber_dataframe().drop(columns=["timestamp"])
    loader = CausalChamberLoader.__new__(CausalChamberLoader)
    loader.subgraph_vars = LT_SUBGRAPH_5VAR
    loader.var_to_idx = {v: i for i, v in enumerate(LT_SUBGRAPH_5VAR)}
    eps = loader.extract_episodes(df, obs_window=40, post_window=20)
    assert len(eps) >= 1
    ep = eps[0]
    # Backward compatibility: the loader does not invent timestamps
    # when the source DF lacks them.
    assert "timestamps_obs" not in ep
    assert "timestamps_post" not in ep


# ---------------------------------------------------------------------- build_causal_chamber_batch


def _episode_with_real_timestamps(T_obs: int = 30, T_post: int = 15, seed: int = 1) -> dict:
    rng = np.random.RandomState(seed)
    var_names = ["pol_1", "red", "green", "blue", "osr_c"]
    N = len(var_names)
    X_obs = rng.randn(T_obs, N).astype(np.float32) * 0.1
    X_post = rng.randn(T_post, N).astype(np.float32) * 0.1 + 1.5

    gaps = rng.uniform(0.142, 0.152, size=T_obs + T_post - 1)
    gaps[5] = 0.25   # one big gap inside the obs window
    gaps[T_obs - 1] = 0.20  # straddling the changepoint
    gaps[T_obs + 4] = 0.25   # one big gap inside the post window
    cumulative = np.concatenate(([0.0], np.cumsum(gaps))).astype(np.float32)

    return {
        "X_obs": X_obs, "X_post": X_post,
        "intervention_var": "pol_1", "intervention_var_idx": 0,
        "intervention_value": 1.0, "var_names": var_names,
        "changepoint": T_obs,
        "timestamps_obs": cumulative[:T_obs],
        "timestamps_post": cumulative[T_obs:],
        "intervention_dt": float(cumulative[T_obs] - cumulative[T_obs - 1]),
    }


def test_batch_uses_real_timestamps_not_uniform_grid():
    ep = _episode_with_real_timestamps()
    batch = build_causal_chamber_batch(ep, query_var="red", n_max=N_MAX)
    times = batch["times"][0].numpy()
    dts = batch["dts"][0].numpy()
    expected = np.concatenate([ep["timestamps_obs"], ep["timestamps_post"]])
    np.testing.assert_allclose(times, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(dts, np.diff(expected), rtol=0, atol=1e-6)


def test_batch_is_irregular_when_source_is_irregular():
    """The dts the encoder sees must reflect the chamber's real
    sampling jitter, not a uniform 10 Hz grid."""
    ep = _episode_with_real_timestamps()
    batch = build_causal_chamber_batch(ep, query_var="red", n_max=N_MAX)
    dts = batch["dts"][0].numpy()
    # We injected gaps spanning ~0.142 - 0.25; the std must reflect
    # that range, not be ~zero (which would be the case under the
    # old uniform-grid code).
    assert dts.std() > 0.01, f"expected irregular dts, got std={dts.std():.4f}"
    assert dts.max() > 0.20, f"expected at least one big gap, got max={dts.max():.4f}"


def test_intervention_window_tracks_real_chamber_time():
    """t_int_end - t_int_start should match the real time spanned by
    the first ``intervention_width_steps`` post-intervention samples,
    not ``intervention_width_steps * dt_seconds``."""
    ep = _episode_with_real_timestamps()
    batch = build_causal_chamber_batch(
        ep, query_var="red", n_max=N_MAX, intervention_width_steps=3,
    )
    t_int_start = float(batch["t_int_start"][0].item())
    t_int_end = float(batch["t_int_end"][0].item())
    expected_end = float(ep["timestamps_post"][2])  # offset 3 - 1
    assert t_int_end == pytest.approx(expected_end, abs=1e-6)
    # Width is the actual physical interval, not 3 * 0.1.
    physical_width = expected_end - float(ep["timestamps_post"][0])
    assert (t_int_end - t_int_start) == pytest.approx(physical_width, abs=1e-6)


def test_batch_falls_back_to_uniform_grid_with_warning():
    """Synthetic episodes without timestamps still work, but emit a
    DeprecationWarning-style notice so callers know."""
    ep = _episode_with_real_timestamps()
    ep_no_ts = {k: v for k, v in ep.items() if not k.startswith("timestamps")}
    ep_no_ts.pop("intervention_dt", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        batch = build_causal_chamber_batch(
            ep_no_ts, query_var="red", n_max=N_MAX, dt_seconds=0.2,
        )
        assert any("CausalChamber" in str(warning.message) for warning in w)
    # And the fallback grid is uniform at the requested dt_seconds.
    dts = batch["dts"][0].numpy()
    np.testing.assert_allclose(dts, 0.2 * np.ones_like(dts), rtol=0, atol=1e-5)


def test_real_timestamps_and_old_dt_seconds_arg_disagree():
    """Passing dt_seconds=0.1 alongside real timestamps must NOT overwrite
    them.  This pins the precedence rule introduced in Phase 12."""
    ep = _episode_with_real_timestamps()
    batch = build_causal_chamber_batch(
        ep, query_var="red", n_max=N_MAX, dt_seconds=0.1,  # legacy default
    )
    times = batch["times"][0].numpy()
    np.testing.assert_allclose(
        times,
        np.concatenate([ep["timestamps_obs"], ep["timestamps_post"]]),
        rtol=0, atol=1e-6,
    )
    # And NOT a uniform 0.1 s grid.
    uniform = np.arange(times.size, dtype=np.float32) * 0.1
    assert not np.allclose(times, uniform, atol=1e-3)


def test_rejects_episode_with_inconsistent_timestamp_shapes():
    ep = _episode_with_real_timestamps(T_obs=30, T_post=15)
    bad = dict(ep)
    bad["timestamps_obs"] = bad["timestamps_obs"][:20]  # shape mismatch
    with pytest.raises(ValueError):
        build_causal_chamber_batch(bad, query_var="red", n_max=N_MAX)


def test_rejects_non_positive_dt():
    """A corrupted episode where two rows share a timestamp (dt = 0)
    must surface as an error rather than silently producing degenerate
    Fourier features."""
    ep = _episode_with_real_timestamps()
    bad = dict(ep)
    bad["timestamps_obs"] = bad["timestamps_obs"].copy()
    bad["timestamps_obs"][5] = bad["timestamps_obs"][4]  # duplicate
    with pytest.raises(ValueError):
        build_causal_chamber_batch(bad, query_var="red", n_max=N_MAX)
