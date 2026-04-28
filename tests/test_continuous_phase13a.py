"""Phase 13a tests: synthetic pre-baseline padding in PK adapters.

Earlier the Theophylline / Warfarin adapters returned batches with
``int_onset_idx == 0`` because pharmacokinetic datasets have no
pre-dose observations -- but training never produced such samples,
so the cross-variable mixer was extrapolating on those benchmarks.
Phase 13a is the eval-side counterpart to Phase 13b: instead of
re-training, prepend ``pre_baseline_n`` synthetic pre-dose
observations at uniformly spaced negative times so the encoder sees
a real pre-intervention window.

These tests pin the new contract on both adapters.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dotime.data.pk_pd.theophylline import TheophSubject
from dotime.data.pk_pd.theophylline_adapter import build_theophylline_batch
from dotime.data.pk_pd.warfarin import WarfarinSubject
from dotime.data.pk_pd.warfarin_adapter import build_warfarin_batch


N_MAX = 8


def _theoph_subject(subject_id: int = 7) -> TheophSubject:
    times = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 24.0])
    concs = torch.tensor([0.0, 2.5, 5.0, 6.5, 8.0, 7.0, 5.0, 3.0, 1.0])
    return TheophSubject(
        subject_id=subject_id,
        weight_kg=70.0,
        dose_mg_per_kg=4.0,
        times=times,
        concentrations=concs,
    )


def _warfarin_subject(subject_id: int = 3) -> WarfarinSubject:
    cp_times = torch.tensor([0.5, 1.0, 2.0, 6.0, 24.0])
    cp_values = torch.tensor([0.5, 1.2, 2.0, 3.0, 1.5])
    pca_times = torch.tensor([0.0, 6.0, 24.0])
    pca_values = torch.tensor([100.0, 70.0, 40.0])
    return WarfarinSubject(
        subject_id=subject_id, weight_kg=70.0, dose_mg=10.0,
        age_years=45, sex="M",
        cp_times=cp_times, cp_values=cp_values,
        pca_times=pca_times, pca_values=pca_values,
    )


# ---------------------------------------------------------------------- Theophylline


def test_theoph_pre_baseline_zero_preserves_backward_compat():
    subj = _theoph_subject()
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=0)
    # Without padding the existing contract holds: int_onset_idx == 0.
    assert int(a["int_onset_idx"][0].item()) == 0
    # And the times array starts at the dose (t=0).
    assert float(a["times"][0, 0].item()) == pytest.approx(0.0, abs=1e-6)


def test_theoph_pre_baseline_n_shifts_int_onset():
    subj = _theoph_subject()
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=5)
    assert int(a["int_onset_idx"][0].item()) == 5
    # The first 5 timestamps must be strictly negative (pre-dose).
    pre = a["times"][0, :5].numpy()
    assert (pre < 0).all(), f"expected negative pre-dose times, got {pre}"
    # Then the dose time at index 5 must be zero.
    assert float(a["times"][0, 5].item()) == pytest.approx(0.0, abs=1e-6)


def test_theoph_pre_baseline_uses_first_post_dose_gap_by_default():
    subj = _theoph_subject()
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=4)
    # First post-dose gap of the fixture is 0.25 h.
    pre = a["times"][0, :4].numpy()
    expected = np.array([-1.00, -0.75, -0.50, -0.25])
    np.testing.assert_allclose(pre, expected, rtol=0, atol=1e-6)


def test_theoph_pre_baseline_dt_argument_overrides_default():
    subj = _theoph_subject()
    a = build_theophylline_batch(
        subj, n_max=N_MAX, pre_baseline_n=3, pre_baseline_dt_hours=2.0,
    )
    pre = a["times"][0, :3].numpy()
    np.testing.assert_allclose(pre, np.array([-6.0, -4.0, -2.0]),
                                rtol=0, atol=1e-6)


def test_theoph_x_int_post_dose_concentrations_match_real_subject():
    subj = _theoph_subject()
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=4)
    # The padded X_int should have zero concentration in the pre-baseline
    # rows and the real subject concentrations from row 4 onwards
    # (variable index _CONC_IDX = 1 in the canonical ordering).
    pre_conc = a["X_int"][0, :4, 1].numpy()
    post_conc = a["X_int"][0, 4:, 1].numpy()
    np.testing.assert_allclose(pre_conc, np.zeros(4), atol=1e-6)
    np.testing.assert_allclose(
        post_conc, subj.concentrations.numpy(), atol=1e-6,
    )


def test_theoph_x_obs_norm_has_baseline_in_pre_window_only():
    subj = _theoph_subject()
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=3)
    n_vars = int(a["num_vars"][0].item())
    # Pre-baseline rows: the pre-dose normalised baseline is
    # (-mean)/std for each variable.
    means = a["_norm_means"][0, :n_vars]
    stds = a["_norm_stds"][0, :n_vars]
    expected_baseline = (-means) / stds.clamp_min(1e-6)
    # Theophylline keeps X_obs_norm zero AND fills it from raw zeros;
    # confirm the pre-baseline rows hold the normalised baseline (not
    # just zeros).
    pre_norm = a["X_obs_norm"][0, :3, :n_vars]
    for i in range(3):
        torch.testing.assert_close(pre_norm[i], expected_baseline)
    # Post-dose rows are causally masked to zero.
    post_norm = a["X_obs_norm"][0, 3:, :n_vars]
    assert torch.all(post_norm == 0.0)


def test_theoph_query_indices_track_real_observations():
    subj = _theoph_subject()
    # 8 post-dose observations (real_times[1:]) so 8 queries by default.
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=3)
    assert a["Y_true"].shape == (8,)
    # The queried Y_true values are the real subject's post-dose
    # concentrations (excluding t=0, which has conc=0 in this fixture).
    np.testing.assert_allclose(
        a["Y_true"].numpy(), subj.concentrations[1:].numpy(), atol=1e-6,
    )


def test_theoph_t_int_start_is_zero_in_real_time():
    subj = _theoph_subject()
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=4)
    # The dose time stays at t = 0 in absolute hours, regardless of
    # how many synthetic pre-dose rows we prepend.
    assert float(a["t_int_start"][0].item()) == pytest.approx(0.0, abs=1e-6)


def test_theoph_rejects_negative_pre_baseline_n():
    subj = _theoph_subject()
    with pytest.raises(ValueError):
        build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=-1)


def test_theoph_rejects_non_positive_pre_baseline_dt():
    subj = _theoph_subject()
    with pytest.raises(ValueError):
        build_theophylline_batch(
            subj, n_max=N_MAX, pre_baseline_n=2, pre_baseline_dt_hours=0.0,
        )
    with pytest.raises(ValueError):
        build_theophylline_batch(
            subj, n_max=N_MAX, pre_baseline_n=2, pre_baseline_dt_hours=-0.5,
        )


# ---------------------------------------------------------------------- Warfarin


def test_warfarin_pre_baseline_zero_preserves_backward_compat():
    subj = _warfarin_subject()
    a = build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=0)
    assert int(a["int_onset_idx"][0].item()) == 0
    assert float(a["times"][0, 0].item()) == pytest.approx(0.0, abs=1e-6)


def test_warfarin_pre_baseline_n_shifts_int_onset_and_prepends_negative_times():
    subj = _warfarin_subject()
    a = build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=4)
    assert int(a["int_onset_idx"][0].item()) == 4
    pre = a["times"][0, :4].numpy()
    assert (pre < 0).all()
    assert float(a["times"][0, 4].item()) == pytest.approx(0.0, abs=1e-6)


def test_warfarin_pre_baseline_uses_median_post_dose_gap_default():
    subj = _warfarin_subject()
    # Real union-grid times for the fixture:
    # [0.0, 0.5, 1.0, 2.0, 6.0, 24.0]; gaps [0.5, 0.5, 1.0, 4.0, 18.0],
    # median = 1.0 h.
    a = build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=3)
    pre = a["times"][0, :3].numpy()
    np.testing.assert_allclose(pre, np.array([-3.0, -2.0, -1.0]), atol=1e-6)


def test_warfarin_pre_baseline_keeps_real_observations_intact():
    subj = _warfarin_subject()
    a = build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=2)
    # The real cp values appear at the cp times (shifted by 2 in the
    # padded grid).  Find them via the union-grid coordinates and
    # verify X_int is consistent.
    times_padded = a["times"][0].numpy()
    # Find the index of t=0.5 (first cp observation) in the padded grid.
    idx_half = int(np.argmin(np.abs(times_padded - 0.5)))
    cp_idx_in_n_max = 1
    assert float(a["X_int"][0, idx_half, cp_idx_in_n_max].item()) == pytest.approx(
        0.5, abs=1e-6,
    )


def test_warfarin_x_obs_norm_has_baseline_in_pre_window_only():
    subj = _warfarin_subject()
    a = build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=3)
    n_vars = int(a["num_vars"][0].item())
    means = a["_norm_means"][0, :n_vars]
    stds = a["_norm_stds"][0, :n_vars]
    expected_baseline = (-means) / stds.clamp_min(1e-6)
    pre_norm = a["X_obs_norm"][0, :3, :n_vars]
    for i in range(3):
        torch.testing.assert_close(pre_norm[i], expected_baseline)
    post_norm = a["X_obs_norm"][0, 3:, :n_vars]
    assert torch.all(post_norm == 0.0)


def test_warfarin_t_int_start_is_zero_in_real_time():
    subj = _warfarin_subject()
    a = build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=2)
    assert float(a["t_int_start"][0].item()) == pytest.approx(0.0, abs=1e-6)


def test_warfarin_rejects_invalid_pre_baseline_args():
    subj = _warfarin_subject()
    with pytest.raises(ValueError):
        build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=-1)
    with pytest.raises(ValueError):
        build_warfarin_batch(
            subj, n_max=N_MAX, pre_baseline_n=2, pre_baseline_dt_hours=0.0,
        )


# ---------------------------------------------------------------------- shapes


def test_theoph_padded_times_shape_and_dts_relationship():
    subj = _theoph_subject()
    a = build_theophylline_batch(subj, n_max=N_MAX, pre_baseline_n=4)
    T = a["times"].shape[1]
    assert a["dts"].shape[1] == T - 1
    # Total length = original (9) + 4 padding rows = 13.
    assert T == subj.times.numel() + 4


def test_warfarin_padded_times_shape_and_dts_relationship():
    subj = _warfarin_subject()
    a = build_warfarin_batch(subj, n_max=N_MAX, pre_baseline_n=3)
    T = a["times"].shape[1]
    assert a["dts"].shape[1] == T - 1
