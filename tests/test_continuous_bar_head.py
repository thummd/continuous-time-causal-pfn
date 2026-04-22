"""Phase 7 tests: bar-distribution head variant for ContinuousDoOverTimePFN."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from dotime.data.continuous_dataloader import ContinuousTemporalInterventionDataLoader
from dotime.model.bar_head import (
    FullSupportBarDistribution,
    calibrate_bar_distribution,
)
from dotime.model.continuous import ContinuousDoOverTimePFN
from dotime.training.continuous_trainer import train_continuous


N_MAX = 6
EMBED = 16
CONTEXT_WINDOW = 16


# ---------------------------------------------------------------------- model


def test_bar_head_model_construction():
    model = ContinuousDoOverTimePFN(
        n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
        n_cross_attn_heads=2, head_type="bar", n_buckets=64,
        context_window=CONTEXT_WINDOW, num_time_frequencies=8,
    )
    assert model.head_type == "bar"
    assert model.bar_head is not None
    assert model.quantile_head is None
    # Quantile head is not constructed.
    assert model.head is model.bar_head


def test_quantile_head_model_construction():
    model = ContinuousDoOverTimePFN(
        n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
        n_cross_attn_heads=2, head_type="quantile",
        tau_levels=[0.1, 0.5, 0.9],
        context_window=CONTEXT_WINDOW, num_time_frequencies=8,
    )
    assert model.head_type == "quantile"
    assert model.quantile_head is not None
    assert model.bar_head is None
    assert model.head is model.quantile_head


# ---------------------------------------------------------------------- calibration


def test_bar_head_needs_calibration_before_loss():
    """Forward works without calibration (returns raw logits), but loss
    should raise because no bucket borders are set."""
    model = ContinuousDoOverTimePFN(
        n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
        n_cross_attn_heads=2, head_type="bar", n_buckets=32,
        context_window=CONTEXT_WINDOW, num_time_frequencies=8,
    )
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=2, tscm_structure="back_door",
        schedule="regular", dt=1.0, t_range=(16, 16), n_max=N_MAX, seed=0,
    )
    batch = next(iter(loader))
    with pytest.raises(RuntimeError):
        model.loss(batch)


def test_calibrate_bar_distribution_works_on_continuous_loader():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=5, batch_size=4, tscm_structure="back_door",
        schedule="regular", dt=1.0, t_range=(16, 16), n_max=N_MAX, seed=7,
    )
    bar_dist, borders = calibrate_bar_distribution(loader, n_buckets=32)
    assert isinstance(bar_dist, FullSupportBarDistribution)
    assert borders.shape == (33,)  # n_buckets + 1
    assert (borders.diff() > 0).all(), "borders must be strictly increasing"


def test_bar_head_forward_and_backward_after_calibration():
    model = ContinuousDoOverTimePFN(
        n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
        n_cross_attn_heads=2, head_type="bar", n_buckets=32,
        context_window=CONTEXT_WINDOW, num_time_frequencies=8,
    )
    cal_loader = ContinuousTemporalInterventionDataLoader(
        num_steps=5, batch_size=4, tscm_structure="back_door",
        schedule="regular", dt=1.0, t_range=(16, 16), n_max=N_MAX, seed=3,
    )
    bar_dist, borders = calibrate_bar_distribution(cal_loader, n_buckets=32)
    model.bar_head.set_bar_distribution(bar_dist, borders)

    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=3, tscm_structure="back_door",
        schedule="regular", dt=1.0, t_range=(16, 16), n_max=N_MAX, seed=4,
    )
    batch = next(iter(loader))
    out = model(batch)
    assert out.shape == (3, 32)  # (B, n_buckets)
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()

    # Check gradients reach the encoder, mixer, and head.
    enc_grad_sum = sum(
        (p.grad is not None and p.grad.abs().sum() > 0)
        for p in model.temporal_encoder.parameters()
    )
    head_grad_sum = sum(
        (p.grad is not None and p.grad.abs().sum() > 0)
        for p in model.bar_head.parameters()
    )
    assert enc_grad_sum > 0
    assert head_grad_sum > 0


def test_bar_head_predict_mean_shape():
    model = ContinuousDoOverTimePFN(
        n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
        n_cross_attn_heads=2, head_type="bar", n_buckets=32,
        context_window=CONTEXT_WINDOW, num_time_frequencies=8,
    )
    cal_loader = ContinuousTemporalInterventionDataLoader(
        num_steps=5, batch_size=4, tscm_structure="back_door",
        schedule="regular", dt=1.0, t_range=(16, 16), n_max=N_MAX, seed=9,
    )
    _, borders = calibrate_bar_distribution(cal_loader, n_buckets=32)
    model.bar_head.set_bar_distribution(FullSupportBarDistribution(borders), borders)

    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=3, tscm_structure="back_door",
        schedule="regular", dt=1.0, t_range=(16, 16), n_max=N_MAX, seed=10,
    )
    batch = next(iter(loader))
    pred = model.predict(batch)
    assert pred.shape == (3,)
    assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------- trainer


def test_train_continuous_with_bar_head():
    with tempfile.TemporaryDirectory() as tmp:
        train_continuous(
            n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=CONTEXT_WINDOW,
            num_time_frequencies=8,
            head_type="bar",
            n_buckets=32,
            bucket_calibration_samples=80,  # small for test speed
            tscm_structure="back_door",
            schedule="regular", dt=1.0, pair_mode="counterfactual",
            intervention_kind_probs=(1.0, 0.0, 0.0),
            t_range=(16, 16),
            batch_size=4, lr=2e-4, warmup_steps=2, total_steps=6,
            grad_clip=1.0, eval_every=3, eval_num_steps=1,
            seed=0, device="cpu", save_dir=tmp, prefetch=0,
        )
        ckpt_path = Path(tmp) / "continuous_do_over_time_pfn_best.pt"
        assert ckpt_path.exists()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["config"]["head_type"] == "bar"
        assert ckpt["borders"] is not None
        assert ckpt["borders"].ndim == 1
        assert ckpt["borders"].numel() == 33  # n_buckets + 1


def test_train_continuous_rejects_invalid_head_type():
    with pytest.raises(ValueError):
        train_continuous(
            n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=CONTEXT_WINDOW,
            head_type="bogus",
            tscm_structure="back_door",
            t_range=(16, 16),
            batch_size=2, total_steps=2,
            seed=0, device="cpu", save_dir="/tmp/unused",
        )


def test_bar_head_checkpoint_roundtrip_via_ct_evaluate_helper():
    """Train with bar head, load via scripts.ct_evaluate._load_model."""
    with tempfile.TemporaryDirectory() as tmp:
        train_continuous(
            n_max=N_MAX, embed_size=EMBED, n_heads=2, n_encoder_layers=1,
            n_cross_attn_heads=2, context_window=CONTEXT_WINDOW,
            num_time_frequencies=8,
            head_type="bar", n_buckets=32, bucket_calibration_samples=80,
            tscm_structure="back_door",
            schedule="regular", dt=1.0, pair_mode="counterfactual",
            intervention_kind_probs=(1.0, 0.0, 0.0),
            t_range=(16, 16),
            batch_size=4, lr=2e-4, warmup_steps=2, total_steps=6,
            grad_clip=1.0, eval_every=3, eval_num_steps=1,
            seed=0, device="cpu", save_dir=tmp, prefetch=0,
        )
        ckpt_path = Path(tmp) / "continuous_do_over_time_pfn_best.pt"
        from scripts.ct_evaluate import _load_model as load_model  # type: ignore
        model = load_model(ckpt_path, device="cpu")
        assert model.head_type == "bar"

        # Verify the loaded model can produce predictions (i.e., borders rehydrated).
        loader = ContinuousTemporalInterventionDataLoader(
            num_steps=1, batch_size=2, tscm_structure="back_door",
            schedule="regular", dt=1.0, t_range=(16, 16), n_max=N_MAX, seed=99,
        )
        batch = next(iter(loader))
        pred = model.predict(batch)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
