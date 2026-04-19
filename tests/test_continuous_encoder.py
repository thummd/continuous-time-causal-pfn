"""Tests for Delta-t aware time embeddings (dotime.model.continuous)."""

from __future__ import annotations

import pytest
import torch

from dotime.model.continuous import (
    DeltaTEmbedding,
    FourierTimeEmbedding,
    relative_to_intervention,
)


def test_fourier_embedding_forward_shape():
    embed = FourierTimeEmbedding(embed_size=32, num_frequencies=8)
    t = torch.randn(3, 10)
    out = embed(t)
    assert out.shape == (3, 10, 32)


def test_fourier_embedding_handles_scalars():
    embed = FourierTimeEmbedding(embed_size=16, num_frequencies=4)
    t = torch.tensor(0.5)
    out = embed(t)
    assert out.shape == (16,)


def test_fourier_embedding_is_equivariant_along_batch():
    """Same time value should give the same embedding regardless of batch position."""
    embed = FourierTimeEmbedding(embed_size=8, num_frequencies=4)
    t = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    out = embed(t)
    assert torch.allclose(out[0, 0], out[1, 0], atol=1e-6)


def test_fourier_embedding_rejects_bad_hyperparams():
    with pytest.raises(ValueError):
        FourierTimeEmbedding(embed_size=4, num_frequencies=0)
    with pytest.raises(ValueError):
        FourierTimeEmbedding(embed_size=4, min_freq=2.0, max_freq=1.0)


def test_delta_t_embedding_shape_and_zero_padding():
    embed = DeltaTEmbedding(embed_size=16, num_frequencies=4)
    dt = torch.rand(2, 5)
    out = embed(dt)
    assert out.shape == (2, 5, 16)

    # dt == 0 -> log1p(0) == 0, so sine features all vanish and output
    # depends only on the cosine features and the bias. Check that the
    # embedding is the same for all batch entries with dt == 0.
    zeros = torch.zeros(3, 4)
    out_zeros = embed(zeros)
    assert torch.allclose(out_zeros[0, 0], out_zeros[1, 2], atol=1e-6)


def test_delta_t_embedding_clamps_negative():
    embed = DeltaTEmbedding(embed_size=8)
    # negative dt must not produce NaN (log1p of a negative < -1 would)
    out = embed(torch.tensor([-1.0, -0.5, 0.0, 0.5]))
    assert torch.isfinite(out).all()


def test_relative_to_intervention_shapes():
    times = torch.arange(10).float().unsqueeze(0).expand(4, 10).clone()
    t_int = torch.tensor([2.0, 3.0, 5.0, 7.0])
    rel = relative_to_intervention(times, t_int)
    assert rel.shape == times.shape
    # each row should be centered on its respective t_int
    for i, onset in enumerate(t_int):
        assert rel[i, int(onset.item())] == 0.0


def test_relative_to_intervention_passthrough_when_none():
    times = torch.arange(5).float()
    out = relative_to_intervention(times, None)
    assert torch.equal(out, times)


def test_fourier_embedding_is_backpropable():
    """Linear projection should produce gradients on its parameters."""
    embed = FourierTimeEmbedding(embed_size=8, num_frequencies=4)
    t = torch.randn(2, 6)
    out = embed(t)
    loss = out.pow(2).sum()
    loss.backward()
    assert embed.projection.weight.grad is not None
    assert embed.projection.weight.grad.abs().sum() > 0
