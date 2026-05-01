"""Tests for the mixed observation schedule used in EXPERIMENT_PLAN_v2."""

from __future__ import annotations

import pytest
import torch

from dotime.prior.continuous import mixed_schedule
from dotime.prior.continuous.extended_prior import _build_schedule


def test_mixed_schedule_shapes_and_monotonicity():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        times, dts = mixed_schedule(
            T=100, dt=1.0, jitter=0.3, rate=1.0, generator=g,
        )
        assert times.shape == (100,)
        assert dts.shape == (99,)
        assert (times.diff() > 0).all()
        assert (dts > 0).all()


def test_mixed_schedule_covers_all_three_families():
    """With uniform weights, regular / jittered / exponential each appear
    over many draws.  We sniff each family by its gap-distribution
    signature: regular has zero gap variance, jittered has small
    bounded variance, exponential has CV close to 1."""
    g = torch.Generator().manual_seed(123)
    families_seen = {"regular": False, "jittered": False, "exponential": False}
    for _ in range(200):
        _, dts = mixed_schedule(
            T=200, dt=1.0, jitter=0.3, rate=1.0, generator=g,
        )
        cv = (dts.std() / dts.mean()).item()
        if cv < 1e-6:
            families_seen["regular"] = True
        elif cv < 0.4:  # jittered: dt*(1 + 0.3*U), CV ~= 0.3 / sqrt(3) ~= 0.17
            families_seen["jittered"] = True
        else:  # exponential: CV = 1
            families_seen["exponential"] = True
    assert all(families_seen.values()), f"missing families: {families_seen}"


def test_mixed_schedule_uniform_default_weights():
    """Family counts should be within 3-sigma of uniform under default weights."""
    g = torch.Generator().manual_seed(7)
    counts = {"regular": 0, "jittered": 0, "exponential": 0}
    n = 600
    for _ in range(n):
        _, dts = mixed_schedule(T=200, dt=1.0, jitter=0.3, rate=1.0, generator=g)
        cv = (dts.std() / dts.mean()).item()
        if cv < 1e-6:
            counts["regular"] += 1
        elif cv < 0.4:
            counts["jittered"] += 1
        else:
            counts["exponential"] += 1
    expected = n / 3
    sigma = (n * (1 / 3) * (2 / 3)) ** 0.5  # binomial std for one category
    for k, c in counts.items():
        assert abs(c - expected) < 4 * sigma, f"{k}={c}, expected ~{expected:.0f} ±{sigma:.1f}"


def test_mixed_schedule_weights_zeroing_a_family():
    """Setting a weight to 0 should never produce that family."""
    g = torch.Generator().manual_seed(11)
    # Disable the exponential family — the produced gaps should never
    # have CV near 1.
    for _ in range(50):
        _, dts = mixed_schedule(
            T=200, dt=1.0, jitter=0.3, rate=1.0,
            weights=(1.0, 1.0, 0.0), generator=g,
        )
        cv = (dts.std() / dts.mean()).item()
        assert cv < 0.5, f"exponential leaked: CV={cv}"


def test_mixed_schedule_rejects_invalid_weights():
    with pytest.raises(ValueError):
        mixed_schedule(T=10, weights=(1.0, 1.0))  # wrong length
    with pytest.raises(ValueError):
        mixed_schedule(T=10, weights=(0.0, 0.0, 0.0))  # zero sum
    with pytest.raises(ValueError):
        mixed_schedule(T=10, weights=(-1.0, 1.0, 1.0))  # negative


def test_build_schedule_dispatches_mixed():
    """The prior's _build_schedule should accept 'mixed' and produce
    valid (times, dts) of the right shape."""
    g = torch.Generator().manual_seed(42)
    times, dts = _build_schedule(
        schedule="mixed", T=80, dt=1.0, jitter=0.3, exp_rate=1.0, generator=g,
    )
    assert times.shape == (80,)
    assert dts.shape == (79,)
    assert (times.diff() > 0).all()
