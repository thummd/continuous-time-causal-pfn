"""Tests for dataloader and normalization."""

import torch
import pytest
from dotime.data.temporal_dataloader import TemporalInterventionDataLoader
from dotime.data.normalization import per_variable_normalize, normalize_target


def test_dataloader_iteration():
    """Test that dataloader produces correct number of batches."""
    loader = TemporalInterventionDataLoader(
        num_steps=3, batch_size=4, seed=42,
        n_max_prior=5, t_range=(20, 30),
    )
    batches = list(loader)
    assert len(batches) == 3


def test_dataloader_batch_keys():
    """Test that batches contain all required keys."""
    loader = TemporalInterventionDataLoader(
        num_steps=1, batch_size=4, seed=42,
    )
    batch = next(iter(loader))

    required_keys = [
        'X_obs', 'X_int', 'variable_mask',
        'intervention_target', 'intervention_type', 'intervention_value',
        'intervention_time_start', 'intervention_time_end',
        'query_target', 'query_time', 'Y_true', 'num_vars',
        'X_obs_norm', 'Y_true_norm',
    ]
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"


def test_normalization():
    """Test per-variable normalization."""
    B, T, N = 4, 30, 10
    X = torch.randn(B, T, N) * 10 + 5
    mask = torch.ones(B, N)
    mask[:, 7:] = 0

    X_norm, means, stds = per_variable_normalize(X, mask)

    # Normalized active variables should have ~0 mean, ~1 std
    for b in range(B):
        for j in range(7):
            col = X_norm[b, :, j]
            assert abs(col.mean().item()) < 0.01
            assert abs(col.std().item() - 1.0) < 0.01

    # Padded variables should be zero
    assert (X_norm[:, :, 7:] == 0).all()


def test_normalize_target():
    """Test target normalization consistency."""
    B, N = 4, 10
    means = torch.randn(B, N)
    stds = torch.ones(B, N) * 2.0
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
    query_idx = torch.tensor([0, 1, 2, 3])

    normed = normalize_target(targets, query_idx, means, stds)

    for b in range(B):
        expected = (targets[b] - means[b, query_idx[b]]) / stds[b, query_idx[b]]
        assert abs(normed[b].item() - expected.item()) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
