"""Tests for model architecture."""

import torch
import pytest
from dotime.model.do_over_time_pfn import DoOverTimePFN
from dotime.model.bar_head import calibrate_bar_distribution
from dotime.data.temporal_dataloader import TemporalInterventionDataLoader


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    model = DoOverTimePFN(
        n_max=41, embed_size=32, n_heads=4, n_encoder_layers=1,
        n_cross_attn_heads=2, n_buckets=50, encoder_backend='transformer',
    )

    # Calibrate bar distribution
    cal_loader = TemporalInterventionDataLoader(
        num_steps=10, batch_size=16, seed=0,
        n_max_prior=5, t_range=(20, 30),
    )
    bar_dist, borders = calibrate_bar_distribution(cal_loader, n_buckets=50)
    model.bar_head.set_bar_distribution(bar_dist, borders)

    return model


@pytest.fixture
def sample_batch():
    """Generate a sample batch."""
    loader = TemporalInterventionDataLoader(
        num_steps=1, batch_size=4, seed=42,
        n_max_prior=5, t_range=(20, 30),
    )
    return next(iter(loader))


def test_forward_shape(small_model, sample_batch):
    """Test that forward pass produces correct output shape."""
    logits = small_model(sample_batch)
    assert logits.shape == (4, 50)


def test_loss_scalar(small_model, sample_batch):
    """Test that loss is a scalar."""
    loss = small_model.loss(sample_batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_backward(small_model, sample_batch):
    """Test that gradients flow through the model."""
    loss = small_model.loss(sample_batch)
    loss.backward()

    has_grad = False
    for p in small_model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad


def test_predict(small_model, sample_batch):
    """Test that predict returns correct shape."""
    preds = small_model.predict(sample_batch)
    assert preds.shape == (4,)


def test_model_parameters():
    """Test parameter count is reasonable."""
    model = DoOverTimePFN(
        n_max=41, embed_size=512, n_heads=4, n_encoder_layers=10,
        n_cross_attn_heads=4, n_buckets=1000, encoder_backend='transformer',
    )
    n_params = sum(p.numel() for p in model.parameters())
    # Should be in the millions range
    assert n_params > 1_000_000


# --- Quantile head tests ---

@pytest.fixture
def quantile_model():
    """Create a small model with quantile head."""
    model = DoOverTimePFN(
        n_max=41, embed_size=32, n_heads=4, n_encoder_layers=1,
        n_cross_attn_heads=2, encoder_backend='transformer',
        head_type='quantile', tau_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
    )
    return model


def test_quantile_forward_shape(quantile_model, sample_batch):
    """Test quantile head output shape."""
    preds = quantile_model(sample_batch)
    assert preds.shape == (4, 5)


def test_quantile_loss_scalar(quantile_model, sample_batch):
    """Test quantile loss is a finite scalar."""
    loss = quantile_model.loss(sample_batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_quantile_backward(quantile_model, sample_batch):
    """Test gradients flow through quantile head."""
    loss = quantile_model.loss(sample_batch)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in quantile_model.parameters()
    )
    assert has_grad


def test_quantile_predict(quantile_model, sample_batch):
    """Test quantile predict_mean returns correct shape."""
    preds = quantile_model.predict(sample_batch)
    assert preds.shape == (4,)


def test_quantile_fewer_params():
    """Quantile head should have far fewer params than bar head."""
    bar_model = DoOverTimePFN(
        n_max=12, embed_size=64, n_heads=4, n_encoder_layers=1,
        n_cross_attn_heads=2, n_buckets=1000, encoder_backend='transformer',
        head_type='bar',
    )
    q_model = DoOverTimePFN(
        n_max=12, embed_size=64, n_heads=4, n_encoder_layers=1,
        n_cross_attn_heads=2, encoder_backend='transformer',
        head_type='quantile',
    )
    bar_params = sum(p.numel() for p in bar_model.parameters())
    q_params = sum(p.numel() for p in q_model.parameters())
    assert q_params < bar_params


def test_head_property(quantile_model):
    """Test the head property returns correct head type."""
    assert quantile_model.head is quantile_model.quantile_head
    assert quantile_model.bar_head is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
