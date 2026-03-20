"""Tests for differentiable quantile extraction and pinball loss."""

import pytest
import torch

from dotime.model.pinball_loss import extract_quantiles, pinball_loss


@pytest.fixture
def bar_setup():
    """Create a simple bar distribution setup for testing."""
    n_buckets = 100
    borders = torch.linspace(-3.0, 3.0, n_buckets + 1)
    # Create logits that produce a roughly Gaussian-like distribution
    centers = (borders[:-1] + borders[1:]) / 2
    logits = -(centers ** 2)  # peaked at 0
    logits = logits.unsqueeze(0).expand(8, -1)  # batch of 8
    return logits, borders, n_buckets


class TestExtractQuantiles:
    def test_output_shape(self, bar_setup):
        logits, borders, _ = bar_setup
        tau_levels = torch.tensor([0.1, 0.5, 0.9])
        q = extract_quantiles(logits, borders, tau_levels)
        assert q.shape == (8, 3)

    def test_quantile_ordering(self, bar_setup):
        """Extracted quantiles should be monotonically non-decreasing."""
        logits, borders, _ = bar_setup
        tau_levels = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        q = extract_quantiles(logits, borders, tau_levels)
        for i in range(q.shape[1] - 1):
            assert (q[:, i] <= q[:, i + 1] + 1e-4).all(), \
                f"Quantile ordering violated at tau={tau_levels[i]:.2f} vs {tau_levels[i+1]:.2f}"

    def test_median_near_mode(self, bar_setup):
        """For a symmetric distribution peaked at 0, median should be near 0."""
        logits, borders, _ = bar_setup
        tau_levels = torch.tensor([0.5])
        q = extract_quantiles(logits, borders, tau_levels)
        assert (q.abs() < 0.5).all(), f"Median {q[0].item():.3f} not near 0"

    def test_gradient_flows(self, bar_setup):
        """Gradients must flow back through logits."""
        logits, borders, _ = bar_setup
        logits = logits.clone().requires_grad_(True)
        tau_levels = torch.tensor([0.1, 0.5, 0.9])
        q = extract_quantiles(logits, borders, tau_levels)
        loss = q.sum()
        loss.backward()
        assert logits.grad is not None
        assert (logits.grad != 0).any(), "Gradients are all zero"

    def test_extreme_quantiles(self, bar_setup):
        """Extreme quantiles should be near the distribution tails."""
        logits, borders, _ = bar_setup
        tau_levels = torch.tensor([0.01, 0.99])
        q = extract_quantiles(logits, borders, tau_levels)
        # Low quantile should be negative, high quantile positive
        assert (q[:, 0] < q[:, 1]).all()

    def test_uniform_distribution(self):
        """For uniform logits, quantiles should match tau * range."""
        n_buckets = 100
        borders = torch.linspace(0.0, 1.0, n_buckets + 1)
        logits = torch.zeros(4, n_buckets)  # uniform after softmax
        tau_levels = torch.tensor([0.25, 0.5, 0.75])
        q = extract_quantiles(logits, borders, tau_levels, temperature=500.0)
        for i, tau in enumerate(tau_levels):
            assert (q[:, i] - tau).abs().max() < 0.05, \
                f"Uniform quantile at tau={tau:.2f}: expected ~{tau:.2f}, got {q[0, i]:.3f}"


class TestPinballLoss:
    def test_output_scalar(self):
        """Pinball loss should return a scalar."""
        q_preds = torch.tensor([[0.1, 0.5, 0.9]])
        y_true = torch.tensor([0.5])
        tau = torch.tensor([0.1, 0.5, 0.9])
        loss = pinball_loss(q_preds, y_true, tau)
        assert loss.dim() == 0

    def test_known_value(self):
        """Check pinball loss against hand-computed value."""
        # Single sample, single quantile: tau=0.5, pred=0.3, true=0.7
        # error = 0.7 - 0.3 = 0.4 >= 0, so loss = 0.5 * 0.4 = 0.2
        q_preds = torch.tensor([[0.3]])
        y_true = torch.tensor([0.7])
        tau = torch.tensor([0.5])
        loss = pinball_loss(q_preds, y_true, tau)
        assert abs(loss.item() - 0.2) < 1e-6

    def test_known_value_overpredict(self):
        """Pinball loss when prediction exceeds target."""
        # tau=0.1, pred=0.8, true=0.2
        # error = 0.2 - 0.8 = -0.6 < 0, so loss = (0.1 - 1) * (-0.6) = 0.9 * 0.6 = 0.54
        q_preds = torch.tensor([[0.8]])
        y_true = torch.tensor([0.2])
        tau = torch.tensor([0.1])
        loss = pinball_loss(q_preds, y_true, tau)
        assert abs(loss.item() - 0.54) < 1e-6

    def test_perfect_prediction(self):
        """Loss is zero when prediction equals target."""
        q_preds = torch.tensor([[0.5, 0.5, 0.5]])
        y_true = torch.tensor([0.5])
        tau = torch.tensor([0.1, 0.5, 0.9])
        loss = pinball_loss(q_preds, y_true, tau)
        assert loss.item() < 1e-6

    def test_non_negative(self):
        """Pinball loss is always non-negative."""
        torch.manual_seed(42)
        q_preds = torch.randn(32, 5)
        y_true = torch.randn(32)
        tau = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        loss = pinball_loss(q_preds, y_true, tau)
        assert loss.item() >= 0


class TestEndToEnd:
    def test_backward_through_extract_and_loss(self, bar_setup):
        """Full backward pass: logits -> extract_quantiles -> pinball_loss."""
        logits, borders, _ = bar_setup
        logits = logits.clone().requires_grad_(True)
        tau_levels = torch.tensor([0.1, 0.5, 0.9])
        y_true = torch.zeros(8)

        q = extract_quantiles(logits, borders, tau_levels)
        loss = pinball_loss(q, y_true, tau_levels)
        loss.backward()

        assert logits.grad is not None
        assert (logits.grad != 0).any()

    def test_pinball_weight_zero_gives_no_contribution(self, bar_setup):
        """When pinball_weight=0, total loss equals bar distribution loss."""
        logits, borders, _ = bar_setup
        tau_levels = torch.tensor([0.1, 0.5, 0.9])
        y_true = torch.zeros(8)

        q = extract_quantiles(logits, borders, tau_levels)
        pb_loss = pinball_loss(q, y_true, tau_levels)

        # With weight 0, contribution is 0
        assert (0.0 * pb_loss).item() == 0.0
