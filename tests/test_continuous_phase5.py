"""Phase 5 tests: random-graph prior + CausalChamber continuous-time adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dotime.data.causal_chamber_ct import (
    build_causal_chamber_batch,
    iter_causal_chamber_batches,
)
from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.eval.continuous_chamber_eval import (
    evaluate_episode,
    evaluate_episodes,
    format_summary,
    metrics_to_dict,
)
from dotime.model.continuous import ContinuousDoOverTimePFN
from dotime.prior.continuous import (
    ContinuousSCM,
    RandomContinuousExtendedPrior,
    RandomContinuousSCMSampler,
)
from dotime.prior.continuous.random_sampler import _pick_intervention_and_outcome


N_MAX = 12
EMBED = 32
CONTEXT_WINDOW = 32


def _tiny_model(n_max: int = N_MAX) -> ContinuousDoOverTimePFN:
    return ContinuousDoOverTimePFN(
        n_max=n_max,
        embed_size=EMBED,
        n_heads=2,
        n_encoder_layers=1,
        n_cross_attn_heads=2,
        head_type="quantile",
        tau_levels=[0.1, 0.5, 0.9],
        context_window=CONTEXT_WINDOW,
        num_time_frequencies=8,
    )


def _synthetic_chamber_episode(T_obs: int = 40, T_post: int = 20, seed: int = 0) -> dict:
    """Fabricate an episode dict with the same shape as CausalChamberLoader output.

    Mirrors the Phase-12 contract: real ``CausalChamberLoader`` episodes
    carry per-row ``timestamps_obs`` / ``timestamps_post`` arrays in
    seconds-since-episode-start.  We synthesise an irregular schedule
    around the real ``lt_walks_v1`` median (~0.147 s) with mild jitter
    plus a few longer gaps.
    """
    rng = np.random.RandomState(seed)
    var_names = ["pol_1", "red", "green", "blue", "osr_c"]
    N = len(var_names)
    X_obs = rng.randn(T_obs, N).astype(np.float32) * 0.1
    X_post = rng.randn(T_post, N).astype(np.float32) * 0.1 + 2.0

    base_dt = 0.147
    gaps = rng.uniform(base_dt - 0.005, base_dt + 0.005, size=T_obs + T_post - 1)
    # Sprinkle in occasional jumbo gaps to mimic real chamber jitter.
    for stride in (7, 13):
        gaps[stride :: stride] = base_dt + 0.1
    cumulative = np.concatenate(([0.0], np.cumsum(gaps))).astype(np.float32)
    timestamps_obs = cumulative[:T_obs]
    timestamps_post = cumulative[T_obs:]

    return {
        "X_obs": X_obs,
        "X_post": X_post,
        "intervention_var": "pol_1",
        "intervention_var_idx": 0,
        "intervention_value": 3.0,
        "var_names": var_names,
        "changepoint": T_obs,
        "experiment": f"synthetic_{seed}",
        "timestamps_obs": timestamps_obs,
        "timestamps_post": timestamps_post,
        "intervention_dt": float(timestamps_post[0] - timestamps_obs[-1]),
    }


# ---------------------------------------------------------------------- random-graph sampler


def test_pick_intervention_and_outcome_respects_topology():
    rng = np.random.RandomState(0)
    for _ in range(20):
        a, y = _pick_intervention_and_outcome(n_vars=5, rng=rng)
        assert 0 <= a < y < 5


def test_pick_intervention_requires_min_two_vars():
    with pytest.raises(ValueError):
        _pick_intervention_and_outcome(n_vars=1, rng=np.random.RandomState(0))


def test_random_sampler_rejects_invalid_hyperparams():
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(n_min=1, n_max_prior=10)
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(n_min=3, n_max_prior=2)
    with pytest.raises(ValueError):
        RandomContinuousSCMSampler(n_min=3, n_max_prior=5, edge_prob=1.5)


def test_random_sampler_produces_valid_scms():
    s = RandomContinuousSCMSampler(n_min=3, n_max_prior=6, seed=42)
    for _ in range(10):
        scm, n_vars, a, y, hidden = s.sample()
        assert isinstance(scm, ContinuousSCM)
        assert 3 <= n_vars <= 6
        assert len(scm.mechanisms) == n_vars
        assert 0 <= a < n_vars
        assert 0 <= y < n_vars
        assert a != y
        # With default hidden_prob=0.0 no nodes are hidden.
        assert hidden == []


def test_random_sampler_is_reproducible():
    s1 = RandomContinuousSCMSampler(n_min=3, n_max_prior=6, seed=17)
    s2 = RandomContinuousSCMSampler(n_min=3, n_max_prior=6, seed=17)
    (_, n1, a1, y1, h1) = s1.sample()
    (_, n2, a2, y2, h2) = s2.sample()
    assert (n1, a1, y1, tuple(h1)) == (n2, a2, y2, tuple(h2))


def test_random_sampler_covers_N_range():
    """Over many samples we should see the full [n_min, n_max_prior] range."""
    s = RandomContinuousSCMSampler(n_min=3, n_max_prior=7, seed=5)
    Ns_seen = {s.sample()[1] for _ in range(200)}
    assert Ns_seen == {3, 4, 5, 6, 7}


# ---------------------------------------------------------------------- random extended prior


def test_random_extended_prior_varies_N_across_samples():
    p = RandomContinuousExtendedPrior(
        n_min=3, n_max_prior=8, t_range=(20, 20), seed=11,
    )
    Ns_seen = {int(p.generate_sample(T=20)["num_vars"].item()) for _ in range(50)}
    assert len(Ns_seen) >= 3  # at least 3 distinct N values


def test_random_extended_prior_produces_finite_trajectories():
    p = RandomContinuousExtendedPrior(
        n_min=3, n_max_prior=6, t_range=(40, 40), seed=12,
    )
    s = p.generate_sample(T=40)
    assert torch.isfinite(s["X_obs"]).all()
    assert torch.isfinite(s["X_int"]).all()


def test_random_extended_prior_canonical_permutation_places_A_at_index_0():
    p = RandomContinuousExtendedPrior(
        n_min=3, n_max_prior=6, t_range=(20, 20), seed=13,
    )
    s = p.generate_sample(T=20)
    assert int(s["intervention_target"].item()) == 0  # A is always at canonical index 0


def test_dataloader_supports_random_prior_mode():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=3,
        prior_mode="random", n_min_prior=3, n_max_prior=5,
        t_range=(24, 24), n_max=N_MAX, seed=1,
    )
    batch = next(iter(loader))
    assert batch["X_obs_norm"].shape == (3, 24, N_MAX)
    assert batch["times"].shape == (3, 24)
    # Every sample's num_vars should be in [3, 5]
    Ns = batch["num_vars"].tolist()
    assert all(3 <= n <= 5 for n in Ns)


def test_dataloader_rejects_bogus_prior_mode():
    with pytest.raises(ValueError):
        ContinuousTemporalInterventionDataLoader(
            num_steps=1, batch_size=1, prior_mode="bogus",
        )


def test_model_forward_on_random_prior_batch():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=2, prior_mode="random",
        n_min_prior=3, n_max_prior=6, t_range=(20, 20),
        n_max=N_MAX, seed=2,
    )
    batch = next(iter(loader))
    model = _tiny_model()
    out = model(batch)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_random_prior_loss_backprop():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=2, prior_mode="random",
        n_min_prior=3, n_max_prior=6, t_range=(20, 20),
        n_max=N_MAX, seed=3,
    )
    batch = next(iter(loader))
    model = _tiny_model()
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    # Time embedding should receive gradients through the encoder.
    time_grad = model.temporal_encoder.time_embedding.projection.weight.grad
    assert time_grad is not None and time_grad.abs().sum() > 0


# ---------------------------------------------------------------------- chamber adapter


def test_chamber_adapter_shape_contract():
    ep = _synthetic_chamber_episode()
    batch = build_causal_chamber_batch(ep, query_var="red", n_max=N_MAX)

    T_total = ep["X_obs"].shape[0] + ep["X_post"].shape[0]
    n_queries = ep["X_post"].shape[0] - 1  # offsets 1..T_post-1
    assert batch["X_obs_norm"].shape == (n_queries, T_total, N_MAX)
    assert batch["times"].shape == (n_queries, T_total)
    assert batch["dts"].shape == (n_queries, T_total - 1)


def test_chamber_adapter_causal_masking_zeros_post_intervention():
    ep = _synthetic_chamber_episode()
    batch = build_causal_chamber_batch(ep, query_var="red", n_max=N_MAX)
    onset = int(batch["int_onset_idx"][0].item())
    post_mask_sum = batch["X_obs"][0, onset:, :].abs().sum().item()
    assert post_mask_sum == 0.0


def test_chamber_adapter_int_onset_and_times_align():
    ep = _synthetic_chamber_episode()
    batch = build_causal_chamber_batch(
        ep, query_var="red", n_max=N_MAX, dt_seconds=0.1,
    )
    onset = int(batch["int_onset_idx"][0].item())
    assert onset == ep["X_obs"].shape[0]
    # t_int_start should equal times[onset]
    assert float(batch["t_int_start"][0].item()) == pytest.approx(
        float(batch["times"][0, onset].item()), abs=1e-5,
    )


def test_chamber_adapter_rejects_unknown_query_var():
    ep = _synthetic_chamber_episode()
    with pytest.raises(ValueError):
        build_causal_chamber_batch(ep, query_var="nonsense", n_max=N_MAX)


def test_chamber_adapter_rejects_small_n_max():
    ep = _synthetic_chamber_episode()
    with pytest.raises(ValueError):
        build_causal_chamber_batch(ep, query_var="red", n_max=2)


def test_chamber_adapter_custom_offsets():
    ep = _synthetic_chamber_episode()
    batch = build_causal_chamber_batch(
        ep, query_var="red", n_max=N_MAX, query_time_offsets=[2, 5, 8],
    )
    assert batch["Y_true"].shape == (3,)
    assert batch["_query_time_offset"].tolist() == [2, 5, 8]


def test_chamber_adapter_raises_on_empty_offsets():
    ep = _synthetic_chamber_episode(T_post=1)
    with pytest.raises(ValueError):
        build_causal_chamber_batch(ep, query_var="red", n_max=N_MAX)


def test_iter_causal_chamber_batches_yields_one_per_episode():
    eps = [_synthetic_chamber_episode(seed=i) for i in range(3)]
    batches = list(iter_causal_chamber_batches(eps, query_var="red", n_max=N_MAX))
    assert len(batches) == 3


# ---------------------------------------------------------------------- chamber evaluator


def test_evaluate_episode_end_to_end():
    ep = _synthetic_chamber_episode()
    model = _tiny_model()
    r = evaluate_episode(model, ep, query_var="red", n_max=N_MAX)
    m = r["metrics"]
    assert m.query_var == "red"
    assert m.n_queries == ep["X_post"].shape[0] - 1
    assert float(m.rmse) >= 0.0


def test_evaluate_episodes_aggregates():
    eps = [_synthetic_chamber_episode(seed=i) for i in range(3)]
    model = _tiny_model()
    result = evaluate_episodes(model, eps, query_var="red", n_max=N_MAX)
    assert len(result["per_episode"]) == 3
    agg = result["aggregate"]
    assert agg.n_episodes == 3
    assert agg.pooled_rmse >= 0.0


def test_evaluate_episodes_empty_input_raises():
    model = _tiny_model()
    with pytest.raises(ValueError):
        evaluate_episodes(model, [], query_var="red", n_max=N_MAX)


def test_chamber_summary_and_json_serialisation_run():
    import json
    eps = [_synthetic_chamber_episode(seed=i) for i in range(2)]
    model = _tiny_model()
    result = evaluate_episodes(model, eps, query_var="red", n_max=N_MAX)
    s = format_summary(result)
    assert "CausalChamber" in s
    d = json.dumps(metrics_to_dict(result))
    assert "aggregate" in d
