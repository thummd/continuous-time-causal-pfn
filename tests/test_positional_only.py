"""Tests for the ``positional_only`` mode on ``ContinuousDoOverTimePFN``.

Covers:
- The override rewrites every time-bearing field to sequence-index
  coordinates.
- The override is idempotent (the sentinel guards against double-apply).
- ``t_int_start`` after override matches ``int_onset_idx`` exactly, so
  the encoder's relative-time signal is consistent with downstream
  masking.
- The "back-door leak" test from EXPERIMENT_PLAN_v2: two episodes with
  the same query step but distinct real schedules must yield identical
  ``query_time`` (mixer scalar) under ``positional_only=True``, and
  *different* ``query_time`` without the override.
- The model's ``forward()`` routes through the override and produces
  finite outputs end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.continuous import ContinuousDoOverTimePFN


N_MAX = 8
EMBED_SIZE = 32
CONTEXT_WINDOW = 32


def _tiny_model(positional_only: bool = False) -> ContinuousDoOverTimePFN:
    return ContinuousDoOverTimePFN(
        n_max=N_MAX,
        embed_size=EMBED_SIZE,
        n_heads=2,
        n_encoder_layers=1,
        n_cross_attn_heads=2,
        head_type="quantile",
        tau_levels=[0.1, 0.5, 0.9],
        context_window=CONTEXT_WINDOW,
        num_time_frequencies=8,
        positional_only=positional_only,
    )


def _hand_batch(times: torch.Tensor, query_step: int, onset_step: int) -> dict:
    """Build the minimal time-bearing fields used by the override.

    ``times`` shape: (B, T).  All other fields derived so the schedule
    can be arbitrary while ``int_onset_idx`` and ``query_time_idx`` are
    fixed step indices.
    """
    B, T = times.shape
    onset_step = int(onset_step)
    query_step = int(query_step)
    end_step = min(T - 1, onset_step + max(1, T // 5))
    return {
        "times": times,
        "dts": times.diff(dim=-1),
        "int_onset_idx": torch.full((B,), onset_step, dtype=torch.long),
        "t_int_start": times[:, onset_step].clone(),
        "t_int_end": times[:, end_step].clone(),
        "t_query": times[:, query_step].clone(),
        # Pre-compute the un-overridden mixer scalars exactly the way
        # ``ContinuousExtendedPrior.generate_sample`` does (line 525-528):
        "intervention_time_start": (
            (times[:, onset_step] - times[:, 0]) / (times[:, -1] - times[:, 0]).clamp(min=1e-6)
        ),
        "intervention_time_end": (
            (times[:, end_step] - times[:, 0]) / (times[:, -1] - times[:, 0]).clamp(min=1e-6)
        ),
        "query_time": (
            (times[:, query_step] - times[:, 0]) / (times[:, -1] - times[:, 0]).clamp(min=1e-6)
        ),
    }


# --------------------------------------------------------- override semantics


def test_override_rewrites_all_time_fields():
    times = torch.tensor([[0.0, 0.5, 1.5, 3.5, 7.5, 15.5]])  # geometric
    batch = _hand_batch(times, query_step=4, onset_step=2)
    model = _tiny_model(positional_only=True)
    out = model._apply_positional_override(batch)

    T = times.shape[-1]
    assert torch.allclose(out["times"][0], torch.arange(T, dtype=torch.float32))
    assert torch.allclose(out["dts"][0], torch.ones(T - 1))
    assert out["t_int_start"].item() == 2.0
    assert out["t_int_end"].item() == out["t_int_end"].long().item()  # integer-valued
    assert out["t_query"].item() == 4.0
    assert out["query_time"].item() == pytest.approx(4.0 / (T - 1), rel=1e-5)
    assert out["intervention_time_start"].item() == pytest.approx(2.0 / (T - 1), rel=1e-5)
    assert out["_positional_applied"] is True


def test_override_is_idempotent():
    times = torch.tensor([[0.0, 1.0, 3.0, 7.0, 15.0]])
    batch = _hand_batch(times, query_step=3, onset_step=1)
    model = _tiny_model(positional_only=True)
    once = model._apply_positional_override(batch)
    twice = model._apply_positional_override(once)
    # Idempotency: second application is a no-op (returns the same dict).
    for k in once:
        if isinstance(once[k], torch.Tensor):
            assert torch.equal(once[k], twice[k]), f"field {k} changed on second apply"
        else:
            assert once[k] == twice[k]


def test_t_int_start_matches_int_onset_idx():
    # t_int_start could fall between samples in a real batch.  We force
    # this by constructing times that don't include the onset value.
    times = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    batch = _hand_batch(times, query_step=4, onset_step=3)
    # Now perturb t_int_start so it lies between samples 2 and 3, but
    # int_onset_idx remains 3.
    batch = {**batch, "t_int_start": torch.tensor([2.4])}
    model = _tiny_model(positional_only=True)
    out = model._apply_positional_override(batch)
    assert out["t_int_start"].item() == float(batch["int_onset_idx"].item())


# --------------------------------------------------------- back-door leak


def test_back_door_leak_blocked_under_positional():
    """Two episodes with identical T and identical query_time_idx but
    different real schedule shapes.  Under ``positional_only=True``,
    the mixer's ``query_time`` scalar must be identical (sequence-index
    based).  Without the override, it differs (real-time based)."""
    T = 21
    query_step = 10  # midpoint
    onset_step = 5

    # Episode A: linear schedule (uniform gaps).
    times_a = torch.arange(T, dtype=torch.float32).unsqueeze(0)
    # Episode B: super-linear schedule (early gaps tight, late gaps wide).
    times_b = (torch.arange(T, dtype=torch.float32) ** 1.5).unsqueeze(0)

    batch_a = _hand_batch(times_a, query_step=query_step, onset_step=onset_step)
    batch_b = _hand_batch(times_b, query_step=query_step, onset_step=onset_step)

    # Without override: query_time differs because real schedules differ.
    qa_real = batch_a["query_time"].item()
    qb_real = batch_b["query_time"].item()
    assert abs(qa_real - qb_real) > 1e-3, (
        f"sanity check failed: real-time query_time should differ "
        f"({qa_real} vs {qb_real})"
    )

    # With override: query_time is identical across both batches.
    model = _tiny_model(positional_only=True)
    out_a = model._apply_positional_override(batch_a)
    out_b = model._apply_positional_override(batch_b)
    assert out_a["query_time"].item() == out_b["query_time"].item()
    assert out_a["query_time"].item() == query_step / (T - 1)


# --------------------------------------------------------- forward smoke


def test_forward_routes_through_override():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        tscm_structure="back_door",
        schedule="jittered",
        dt=1.0,
        jitter=0.3,
        t_range=(20, 20),
        n_max=N_MAX,
        seed=0,
    )
    batch = next(iter(loader))
    model_pos = _tiny_model(positional_only=True)
    out = model_pos(batch)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()
    # Sanity: same model class without the flag also returns finite outputs.
    model_time = _tiny_model(positional_only=False)
    out_time = model_time(batch)
    assert torch.isfinite(out_time).all()


def test_forward_invariant_to_real_schedule_under_positional():
    """A positional_only model must give the same output for two
    batches whose only difference is the real schedule (X_obs identical,
    same int_onset_idx, same query step, same end step).

    Both batches have their event times snapped to known sample steps
    in advance so the override sees identical step-index inputs."""
    B = 1
    T = 20
    base_loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=B,
        tscm_structure="back_door",
        schedule="regular",
        dt=1.0,
        t_range=(T, T),
        n_max=N_MAX,
        seed=99,
    )
    batch = next(iter(base_loader))
    onset = int(batch["int_onset_idx"][0].item())
    end_step = min(T - 1, onset + 4)
    q_idx = (batch["times"][0] - batch["t_query"][0]).abs().argmin().item()

    def _build(times: torch.Tensor) -> dict:
        out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out["times"] = times
        out["dts"] = times.diff(dim=-1)
        out["t_int_start"] = times[:, onset].clone()
        out["t_int_end"] = times[:, end_step].clone()
        out["t_query"] = times[:, q_idx].clone()
        span = (times[:, -1] - times[:, 0]).clamp(min=1e-6)
        out["intervention_time_start"] = (times[:, onset] - times[:, 0]) / span
        out["intervention_time_end"] = (times[:, end_step] - times[:, 0]) / span
        out["query_time"] = (times[:, q_idx] - times[:, 0]) / span
        return out

    times_a = torch.arange(T, dtype=torch.float32).unsqueeze(0)
    times_b = (torch.arange(T, dtype=torch.float32) ** 1.5).unsqueeze(0)
    batch_a = _build(times_a)
    batch_b = _build(times_b)

    model = _tiny_model(positional_only=True).eval()
    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        f"positional_only outputs differ: {out_a} vs {out_b}"
    )


def test_positional_default_off_preserves_behaviour():
    """A freshly constructed model with positional_only=False (default)
    must NOT touch the batch — the override path is skipped."""
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        tscm_structure="back_door",
        schedule="regular",
        dt=1.0,
        t_range=(20, 20),
        n_max=N_MAX,
        seed=42,
    )
    batch = next(iter(loader))
    saved_times = batch["times"].clone()
    model = _tiny_model(positional_only=False)
    _ = model(batch)
    # batch["times"] is the unmodified original (no in-place edit).
    assert torch.equal(batch["times"], saved_times)
    assert "_positional_applied" not in batch
