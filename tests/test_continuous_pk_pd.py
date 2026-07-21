"""Phase 4 tests: Theophylline PK loader, adapter, and zero-shot evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from dotime.data.pk_pd.theophylline import (
    TheophSubject,
    load_theophylline,
)
from dotime.data.pk_pd.theophylline_adapter import (
    build_theophylline_batch,
    concat_batches,
    iter_theophylline_batches,
)
from dotime.eval.continuous_pk_eval import (
    evaluate_dataset,
    evaluate_subject,
    format_summary,
    metrics_to_dict,
)
from dotime.model.continuous import ContinuousDoOverTimePFN


N_MAX = 4  # Compact: 2 real vars + 2 padding.
EMBED = 16
CONTEXT_WINDOW = 16


def _tiny_model() -> ContinuousDoOverTimePFN:
    return ContinuousDoOverTimePFN(
        n_max=N_MAX,
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


def test_load_theophylline_has_12_subjects():
    subjects = load_theophylline()
    assert len(subjects) == 12
    for s in subjects:
        assert 1 <= s.subject_id <= 12
        assert s.weight_kg > 0
        assert s.dose_mg_per_kg > 0
        assert s.n_observations >= 10


def test_load_theophylline_times_are_strictly_increasing():
    subjects = load_theophylline()
    for s in subjects:
        assert (s.times.diff() > 0).all(), f"subject {s.subject_id} has non-monotone times"


def test_load_theophylline_first_observation_is_near_zero():
    subjects = load_theophylline()
    for s in subjects:
        assert float(s.times[0].item()) == 0.0


def test_theoph_subject_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        TheophSubject(
            subject_id=1, weight_kg=1.0, dose_mg_per_kg=1.0,
            times=torch.tensor([[1.0]]),  # 2-D
            concentrations=torch.tensor([1.0]),
        )
    with pytest.raises(ValueError):
        TheophSubject(
            subject_id=1, weight_kg=1.0, dose_mg_per_kg=1.0,
            times=torch.tensor([0.0, 1.0, 0.5]),  # non-monotone
            concentrations=torch.tensor([0.0, 1.0, 2.0]),
        )


def test_load_theophylline_missing_file_is_explicit():
    with pytest.raises(FileNotFoundError):
        load_theophylline(Path("/nonexistent/theoph.csv"))


# ---------------------------------------------------------------------- adapter


def test_adapter_produces_expected_fields():
    subj = load_theophylline()[0]
    batch = build_theophylline_batch(subj, n_max=N_MAX)

    expected_keys = {
        "X_obs", "X_obs_norm", "X_int", "variable_mask", "num_vars",
        "_norm_means", "_norm_stds",
        "times", "dts",
        "int_onset_idx", "intervention_target", "intervention_type",
        "intervention_value", "intervention_time_start", "intervention_time_end",
        "t_int_start", "t_int_end",
        "query_target", "query_time", "t_query",
        "Y_true", "Y_true_norm",
        "_subject_id", "_query_time_hours",
    }
    assert set(batch.keys()) == expected_keys


def test_adapter_shapes_match_n_queries():
    subj = load_theophylline()[0]
    T = subj.n_observations  # 11
    batch = build_theophylline_batch(subj, n_max=N_MAX)
    B = batch["Y_true"].shape[0]
    assert B == T - 1  # every post-dose observation becomes a query
    assert batch["X_obs_norm"].shape == (B, T, N_MAX)
    assert batch["times"].shape == (B, T)
    assert batch["dts"].shape == (B, T - 1)


def test_adapter_custom_query_time_indices():
    subj = load_theophylline()[0]
    batch = build_theophylline_batch(
        subj, n_max=N_MAX, query_time_indices=[1, 5, 10],
    )
    assert batch["Y_true"].shape == (3,)
    assert torch.allclose(batch["_query_time_hours"], subj.times[[1, 5, 10]])


def test_adapter_rejects_empty_queries():
    subj = load_theophylline()[0]
    with pytest.raises(ValueError):
        build_theophylline_batch(subj, n_max=N_MAX, query_time_indices=[])


def test_adapter_rejects_too_small_n_max():
    subj = load_theophylline()[0]
    with pytest.raises(ValueError):
        build_theophylline_batch(subj, n_max=1)


def test_adapter_fixed_normalisation_honoured():
    subj = load_theophylline()[0]
    batch = build_theophylline_batch(
        subj, n_max=N_MAX,
        normalize_from="fixed",
        fixed_means=(0.0, 5.0),  # conc_mean = 5 mg/L
        fixed_stds=(2.0, 3.0),   # conc_std  = 3 mg/L
    )
    concs_idx = 1  # canonical: A=0 (Dose), Y=1 (conc)
    expected = torch.clamp((subj.concentrations[1:] - 5.0) / 3.0, -10.0, 10.0)
    assert torch.allclose(batch["Y_true_norm"], expected, atol=1e-5)


def test_adapter_fixed_mode_requires_means_and_stds():
    subj = load_theophylline()[0]
    with pytest.raises(ValueError):
        build_theophylline_batch(subj, n_max=N_MAX, normalize_from="fixed")


def test_adapter_intervention_is_hard_at_dose_variable():
    subj = load_theophylline()[0]
    batch = build_theophylline_batch(subj, n_max=N_MAX)
    # All queries share the same intervention
    assert (batch["intervention_target"] == 0).all()  # Dose
    assert (batch["query_target"] == 1).all()         # conc
    assert (batch["intervention_type"] == 0).all()    # HARD
    assert (batch["int_onset_idx"] == 0).all()


def test_concat_batches_stacks_along_batch_dim():
    subjects = load_theophylline()[:3]
    batches = [build_theophylline_batch(s, n_max=N_MAX) for s in subjects]
    full = concat_batches(batches)
    expected_B = sum(b["Y_true"].shape[0] for b in batches)
    assert full["Y_true"].shape == (expected_B,)
    assert full["X_obs_norm"].shape == (expected_B, 11, N_MAX)


def test_concat_batches_rejects_key_mismatch():
    subj = load_theophylline()[0]
    a = build_theophylline_batch(subj, n_max=N_MAX)
    b = {k: v for k, v in a.items()}
    del b["Y_true"]
    with pytest.raises(ValueError):
        concat_batches([a, b])


def test_iter_theophylline_batches_yields_per_subject():
    subjects = load_theophylline()[:4]
    count = 0
    for batch in iter_theophylline_batches(subjects, n_max=N_MAX):
        count += 1
        assert batch["_subject_id"][0].item() == subjects[count - 1].subject_id
    assert count == 4


# ---------------------------------------------------------------------- evaluator


def test_evaluate_subject_runs_end_to_end():
    model = _tiny_model()
    subj = load_theophylline()[0]
    r = evaluate_subject(model, subj, n_max=N_MAX)
    m = r["metrics"]
    assert m.subject_id == subj.subject_id
    assert m.n_queries == subj.n_observations - 1
    assert r["predictions"].shape == r["ground_truth"].shape
    assert float(m.rmse) >= 0.0
    assert -1.0 <= float(m.pearson_r) <= 1.0 or m.pearson_r != m.pearson_r  # nan allowed


def test_evaluate_dataset_aggregates_metrics():
    model = _tiny_model()
    subjects = load_theophylline()[:3]
    result = evaluate_dataset(model, subjects, n_max=N_MAX)

    assert len(result["per_subject"]) == 3
    agg = result["aggregate"]
    assert agg.n_subjects == 3
    assert agg.n_queries == sum(s.n_observations - 1 for s in subjects)
    assert agg.pooled_rmse >= 0.0
    assert agg.naive_pooled_rmse >= 0.0
    # Untrained model: naive baseline is typically comparable or better.
    # We only check the sign semantics are computed, not that the model wins.
    assert -1.1 <= agg.lift_over_naive <= 1.1


def test_metrics_to_dict_is_json_serialisable():
    model = _tiny_model()
    subjects = load_theophylline()[:2]
    result = evaluate_dataset(model, subjects, n_max=N_MAX)
    d = metrics_to_dict(result)
    s = json.dumps(d)
    restored = json.loads(s)
    assert restored["aggregate"]["n_subjects"] == 2
    assert "rmse" in restored["per_subject"][0]


def test_format_summary_mentions_key_numbers():
    model = _tiny_model()
    subjects = load_theophylline()[:2]
    result = evaluate_dataset(model, subjects, n_max=N_MAX)
    summary = format_summary(result)
    assert "Pooled RMSE" in summary
    assert "Lift over naive" in summary
    assert "subject" in summary


# ---------------------------------------------------------------------- checkpoint round-trip


def test_checkpoint_roundtrip_then_evaluate():
    """Train a tiny CT model for a handful of steps, save, reload, evaluate."""
    from dotime.training.continuous_trainer import train_continuous

    with tempfile.TemporaryDirectory() as tmp:
        train_continuous(
            n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=CONTEXT_WINDOW,
            num_time_frequencies=8,
            tau_levels=[0.1, 0.5, 0.9],
            tscm_structure="bivariate",
            schedule="regular", dt=1.0, pair_mode="counterfactual",
            intervention_kind_probs=(1.0, 0.0, 0.0),
            t_range=(16, 16),
            batch_size=2, lr=2e-4, warmup_steps=2, total_steps=6,
            grad_clip=1.0, eval_every=3, eval_num_steps=1,
            seed=0, device="cpu", save_dir=tmp, prefetch=0,
        )
        ckpt_path = Path(tmp) / "continuous_do_over_time_pfn_best.pt"
        assert ckpt_path.exists()

        # Reload via the CLI helper
        from scripts.ct_evaluate import _load_model as load_model  # type: ignore

        model = load_model(ckpt_path, device="cpu")
        subjects = load_theophylline()[:2]
        result = evaluate_dataset(model, subjects, n_max=N_MAX)
        assert result["aggregate"].n_queries > 0
