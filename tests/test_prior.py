"""Tests for extended prior and TSCM sampler."""

import torch
import pytest
from dotime.prior.extended_prior import ExtendedCausalTimePrior
from dotime.prior.tscm_sampler import TSCMSampler, TSCMStructure


def test_extended_prior_shapes():
    """Test that extended prior produces correct shapes."""
    prior = ExtendedCausalTimePrior(n_max=41, seed=42)
    sample = prior.generate_sample(T=50)

    assert sample['X_obs'].shape == (50, 41)
    assert sample['X_int'].shape == (50, 41)
    assert sample['variable_mask'].shape == (41,)
    assert sample['intervention_target'].dtype == torch.long
    assert sample['intervention_type'].dtype == torch.long
    assert 0 <= sample['intervention_type'].item() <= 2
    assert 0 <= sample['intervention_time_start'].item() <= 1
    assert 0 <= sample['intervention_time_end'].item() <= 1
    assert 0 <= sample['query_time'].item() <= 1
    assert sample['num_vars'].item() >= 3


def test_extended_prior_batch():
    """Test batch generation."""
    prior = ExtendedCausalTimePrior(n_max=41, seed=42)
    batch = prior.generate_batch(batch_size=8, T=30)

    assert batch['X_obs'].shape == (8, 30, 41)
    assert batch['variable_mask'].shape == (8, 41)
    assert batch['Y_true'].shape == (8,)


def test_variable_mask_consistency():
    """Test that variable mask correctly reflects actual variables."""
    prior = ExtendedCausalTimePrior(n_max=41, seed=42)
    sample = prior.generate_sample(T=30)

    n_vars = sample['num_vars'].item()
    mask = sample['variable_mask']
    assert mask[:n_vars].sum() == n_vars
    assert mask[n_vars:].sum() == 0

    # Padded variables should be zero in X_obs
    assert (sample['X_obs'][:, n_vars:] == 0).all()


def test_tscm_all_structures():
    """Test that all TSCM structures can be sampled and simulated."""
    gen = torch.Generator().manual_seed(42)

    for structure in TSCMStructure:
        sampler = TSCMSampler(structure, max_lag=1)
        scm = sampler.sample(generator=gen)
        X_obs = scm.sample_observational(T=20, burn_in=10, generator=gen)
        assert X_obs.shape[0] == 20
        assert X_obs.shape[1] >= 3


def test_tscm_hidden_vars():
    """Test that unobserved confounder and front-door have hidden vars."""
    s1 = TSCMSampler(TSCMStructure.UNOBSERVED_CONFOUNDER)
    assert s1.get_hidden_vars() == [0]

    s2 = TSCMSampler(TSCMStructure.FRONT_DOOR)
    assert s2.get_hidden_vars() == [0]

    s3 = TSCMSampler(TSCMStructure.OBSERVED_CONFOUNDER)
    assert s3.get_hidden_vars() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
