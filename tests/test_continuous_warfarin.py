"""Phase 7 tests: Warfarin PK/PD loader + 3-variable adapter + evaluator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from dotime.data.pk_pd.warfarin import WarfarinSubject, load_warfarin
from dotime.data.pk_pd.warfarin_adapter import (
    build_warfarin_batch,
    iter_warfarin_batches,
)
from dotime.eval.continuous_warfarin_eval import (
    evaluate_warfarin_dataset,
    evaluate_warfarin_subject,
    format_summary,
    metrics_to_dict,
)
from dotime.model.continuous import ContinuousDoOverTimePFN


N_MAX = 6  # must be >= 3
EMBED = 16
CONTEXT_WINDOW = 16


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


# ---------------------------------------------------------------------- loader


def test_load_warfarin_has_many_subjects():
    subjects = load_warfarin()
    assert len(subjects) >= 30
    for s in subjects:
        assert s.dose_mg > 0
        assert s.weight_kg > 0
        # Every subject must have at least some observations.
        assert s.n_cp + s.n_pca > 0


def test_warfarin_subject_times_strictly_increasing():
    subjects = load_warfarin()
    for s in subjects:
        if s.n_cp > 1:
            assert (s.cp_times.diff() > 0).all()
        if s.n_pca > 1:
            assert (s.pca_times.diff() > 0).all()


def test_warfarin_subject_union_times_include_zero():
    subjects = load_warfarin()
    for s in subjects:
        union = s.union_times()
        assert float(union[0].item()) == 0.0
        assert (union.diff() > 0).all()


def test_warfarin_subject_rejects_bad_shapes():
    with pytest.raises(ValueError):
        WarfarinSubject(
            subject_id=1, dose_mg=100, weight_kg=70, age_years=40, sex="male",
            cp_times=torch.tensor([0.0, 2.0]),
            cp_values=torch.tensor([1.0]),  # mismatched length
        )


def test_warfarin_subject_rejects_non_monotone():
    with pytest.raises(ValueError):
        WarfarinSubject(
            subject_id=1, dose_mg=100, weight_kg=70, age_years=40, sex="male",
            cp_times=torch.tensor([0.0, 2.0, 1.0]),
            cp_values=torch.tensor([1.0, 2.0, 3.0]),
        )


def test_load_warfarin_missing_file_is_explicit():
    with pytest.raises(FileNotFoundError):
        load_warfarin(Path("/nonexistent/warfarin.csv"))


# ---------------------------------------------------------------------- adapter


def test_adapter_produces_expected_batch_size():
    subjects = load_warfarin()
    s = subjects[0]
    batch = build_warfarin_batch(s, n_max=N_MAX)
    B = batch["Y_true"].shape[0]
    assert B == s.n_cp + s.n_pca


def test_adapter_query_targets_are_cp_and_pca_only():
    s = load_warfarin()[0]
    batch = build_warfarin_batch(s, n_max=N_MAX)
    # canonical index 1 = cp, 2 = pca (0 = Dose never queried)
    targets = batch["query_target"].unique().tolist()
    assert set(targets).issubset({1, 2})
    assert all(t in {1, 2} for t in targets)


def test_adapter_intervention_is_hard_on_dose():
    s = load_warfarin()[0]
    batch = build_warfarin_batch(s, n_max=N_MAX)
    assert (batch["intervention_target"] == 0).all()
    assert (batch["intervention_type"] == 0).all()  # HARD
    assert (batch["int_onset_idx"] == 0).all()


def test_adapter_variable_name_tag_has_correct_length():
    s = load_warfarin()[0]
    batch = build_warfarin_batch(s, n_max=N_MAX)
    names = batch["_query_variable_name"]
    assert len(names) == batch["Y_true"].shape[0]
    assert sorted(set(names)) == ["cp", "pca"]


def test_adapter_rejects_too_small_n_max():
    s = load_warfarin()[0]
    with pytest.raises(ValueError):
        build_warfarin_batch(s, n_max=2)  # must be >= 3


def test_adapter_trajectory_carries_cp_and_pca_at_obs_times():
    s = load_warfarin()[0]
    batch = build_warfarin_batch(s, n_max=N_MAX)
    X_int = batch["X_int"][0]  # (T, N_max), same for all B entries
    times = batch["times"][0]

    # For each cp observation, X_int at that row/col 1 should match the value.
    for t_obs, v_obs in zip(s.cp_times.tolist(), s.cp_values.tolist()):
        idx = (times == t_obs).nonzero(as_tuple=True)[0]
        assert idx.numel() == 1
        assert abs(float(X_int[idx[0], 1].item()) - v_obs) < 1e-4


def test_iter_warfarin_batches_yields_per_subject():
    subjects = load_warfarin()[:3]
    batches = list(iter_warfarin_batches(subjects, n_max=N_MAX))
    assert len(batches) == 3
    for s, b in zip(subjects, batches):
        assert b["_subject_id"][0].item() == s.subject_id


# ---------------------------------------------------------------------- evaluator


def test_evaluate_subject_returns_split_metrics():
    model = _tiny_model()
    s = load_warfarin()[0]
    r = evaluate_warfarin_subject(model, s, n_max=N_MAX)
    m = r["metrics"]
    assert m.subject_id == s.subject_id
    assert m.n_cp == s.n_cp
    assert m.n_pca == s.n_pca
    assert m.rmse_cp >= 0.0
    assert m.rmse_pca >= 0.0
    assert r["pred_cp"].shape == r["gt_cp"].shape
    assert r["pred_pca"].shape == r["gt_pca"].shape


def test_evaluate_dataset_aggregates_and_lifts():
    model = _tiny_model()
    subjects = load_warfarin()[:4]
    result = evaluate_warfarin_dataset(model, subjects, n_max=N_MAX)
    assert len(result["per_subject"]) == 4
    agg = result["aggregate"]
    assert agg.n_subjects == 4
    assert agg.n_cp_queries + agg.n_pca_queries > 0
    assert agg.pooled_rmse_cp >= 0.0
    assert agg.pooled_rmse_pca >= 0.0
    # Lift semantics: NaN only if naive RMSE ~= 0; on a 4-subject tiny model
    # lift is simply computed, not asserted to be positive.
    assert -10.0 <= agg.lift_over_naive_cp <= 10.0
    assert -10.0 <= agg.lift_over_naive_pca <= 10.0


def test_evaluate_dataset_raises_on_empty_subjects():
    model = _tiny_model()
    with pytest.raises(ValueError):
        evaluate_warfarin_dataset(model, [], n_max=N_MAX)


def test_metrics_to_dict_serialisable():
    model = _tiny_model()
    result = evaluate_warfarin_dataset(model, load_warfarin()[:2], n_max=N_MAX)
    d = metrics_to_dict(result)
    s = json.dumps(d)
    assert "pooled_rmse_cp" in s
    assert "pooled_rmse_pca" in s


def test_format_summary_mentions_both_variables():
    model = _tiny_model()
    result = evaluate_warfarin_dataset(model, load_warfarin()[:2], n_max=N_MAX)
    s = format_summary(result)
    assert "Concentration" in s
    assert "PCA" in s


# ---------------------------------------------------------------------- checkpoint round-trip


def test_warfarin_eval_over_checkpoint_roundtrip():
    """Train a tiny CT model, save, reload, evaluate on Warfarin."""
    from dotime.training.continuous_trainer import train_continuous

    with tempfile.TemporaryDirectory() as tmp:
        train_continuous(
            n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=CONTEXT_WINDOW,
            num_time_frequencies=8,
            tau_levels=[0.1, 0.5, 0.9],
            tscm_structure="front_door",
            schedule="regular", dt=1.0, pair_mode="counterfactual",
            intervention_kind_probs=(1.0, 0.0, 0.0),
            t_range=(16, 16),
            batch_size=2, lr=2e-4, warmup_steps=2, total_steps=6,
            grad_clip=1.0, eval_every=3, eval_num_steps=1,
            seed=0, device="cpu", save_dir=tmp, prefetch=0,
        )
        ckpt_path = Path(tmp) / "continuous_do_over_time_pfn_best.pt"
        assert ckpt_path.exists()

        from scripts.ct_evaluate import _load_model as load_model  # type: ignore
        model = load_model(ckpt_path, device="cpu")
        result = evaluate_warfarin_dataset(
            model, load_warfarin()[:2], device="cpu", n_max=N_MAX,
        )
        assert result["aggregate"].n_subjects == 2
