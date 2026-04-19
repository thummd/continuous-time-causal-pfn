"""Regression tests for the discrete-time hidden-variable masking fix.

The discrete-time :class:`ExtendedCausalTimePrior` previously had three
related bugs around hidden confounders:

1. The sequential ``generate_sample`` path built ``variable_mask`` as
   ``[1] * N`` without excluding hidden variables at all.
2. Both sequential and batched paths applied the canonical permutation
   (A -> 0, Y -> N-1) to ``X_obs`` / ``X_int`` but left
   ``variable_mask`` in topological order, so the mask and the
   trajectory tensors referred to different columns.
3. Neither path zeroed ``X_obs`` / ``X_int`` at hidden columns, so the
   per-variable temporal encoder still processed the hidden variable's
   trajectory through shared weights (training-data leak via gradient
   flow even though the final ``h * variable_mask`` zeroed the output).

These tests lock the fix in place for both code paths across all three
TSCM structures that carry a hidden ``U``.
"""

from __future__ import annotations

import pytest
import torch

from dotime.prior.extended_prior import ExtendedCausalTimePrior


def _read_sample(s: dict) -> tuple:
    N = int(s["num_vars"].item())
    return (
        N,
        s["variable_mask"][:N],
        s["X_obs"][:, :N],
        s["X_int"][:, :N],
    )


def _read_batched(b: dict) -> tuple:
    N = int(b["num_vars"][0].item())
    return (
        N,
        b["variable_mask"][0, :N],
        b["X_obs"][0, :, :N],
        b["X_int"][0, :, :N],
    )


# ---------------------------------------------------------------------- structure tests


@pytest.mark.parametrize(
    "structure, expected_hidden_count",
    [
        ("back_door", 0),
        ("rct_no_confounding", 0),
        ("front_door", 1),
        ("instrumental_variable", 1),
        ("unobserved_confounder", 1),
    ],
)
def test_sequential_path_hidden_mask_count(structure, expected_hidden_count):
    """generate_sample produces variable_mask with the right number of zeros."""
    p = ExtendedCausalTimePrior(tscm_structure=structure, t_range=(30, 30), seed=0)
    s = p.generate_sample(T=30)
    _, vmask, _, _ = _read_sample(s)
    hidden_count = int((vmask == 0).sum().item())
    assert hidden_count == expected_hidden_count


@pytest.mark.parametrize(
    "structure, expected_hidden_count",
    [
        ("back_door", 0),
        ("rct_no_confounding", 0),
        ("front_door", 1),
        ("instrumental_variable", 1),
        ("unobserved_confounder", 1),
    ],
)
def test_batched_path_hidden_mask_count(structure, expected_hidden_count):
    """generate_batch produces variable_mask with the right number of zeros."""
    p = ExtendedCausalTimePrior(tscm_structure=structure, t_range=(30, 30), seed=0)
    b = p.generate_batch(batch_size=1)
    _, vmask, _, _ = _read_batched(b)
    hidden_count = int((vmask == 0).sum().item())
    assert hidden_count == expected_hidden_count


# ---------------------------------------------------------------------- X_obs / X_int zeroing


@pytest.mark.parametrize("structure", ["front_door", "instrumental_variable", "unobserved_confounder"])
@pytest.mark.parametrize("path", ["sequential", "batched"])
def test_hidden_position_has_zero_trajectories(structure, path):
    """Wherever variable_mask is 0, X_obs and X_int must be entirely zero."""
    p = ExtendedCausalTimePrior(tscm_structure=structure, t_range=(30, 30), seed=1)
    if path == "sequential":
        s = p.generate_sample(T=30)
        N, vmask, X_obs, X_int = _read_sample(s)
    else:
        b = p.generate_batch(batch_size=1)
        N, vmask, X_obs, X_int = _read_batched(b)

    hidden_positions = (vmask == 0).nonzero(as_tuple=True)[0].tolist()
    assert hidden_positions, f"{structure}/{path}: expected at least one hidden position"
    for h in hidden_positions:
        assert torch.all(X_obs[:, h] == 0.0), f"{structure}/{path}: X_obs leaks at canon {h}"
        assert torch.all(X_int[:, h] == 0.0), f"{structure}/{path}: X_int leaks at canon {h}"


# ---------------------------------------------------------------------- alignment with canonical perm


def test_variable_mask_aligned_with_canonical_permutation():
    """After canonical perm, the hidden position in variable_mask must match
    the column in X_obs / X_int that carries the hidden variable.

    For front_door the topological order is [U, A, M, Y] with U hidden.
    canonical_perm sends A -> 0 and Y -> N-1, leaving U somewhere in the
    middle (canon index 1 or 2 depending on the "middle" ordering).  In
    both cases the mask zero and the all-zero X_obs column must coincide.
    """
    p = ExtendedCausalTimePrior(tscm_structure="front_door", t_range=(40, 40), seed=2)
    perm = p.prior.canonical_perm
    # Expected hidden canonical index: U is topo idx 0.
    # perm maps canon_idx -> topo_idx, so hidden_canon = perm.index(0).
    expected_hidden_canon = perm.index(0)

    for path_name, reader in [
        ("sequential", lambda: _read_sample(p.generate_sample(T=40))),
        ("batched", lambda: _read_batched(p.generate_batch(batch_size=1))),
    ]:
        N, vmask, X_obs, X_int = reader()
        hidden_positions = (vmask == 0).nonzero(as_tuple=True)[0].tolist()
        assert hidden_positions == [expected_hidden_canon], (
            f"{path_name}: hidden_positions={hidden_positions} vs expected {[expected_hidden_canon]}"
        )
        assert torch.all(X_obs[:, expected_hidden_canon] == 0.0)
        assert torch.all(X_int[:, expected_hidden_canon] == 0.0)


def test_a_and_y_stay_observable():
    """A (canon 0) and Y (canon N-1) must never be marked hidden."""
    for struct in ("front_door", "instrumental_variable", "unobserved_confounder"):
        p = ExtendedCausalTimePrior(tscm_structure=struct, t_range=(30, 30), seed=3)
        for path_name, reader in [
            ("sequential", lambda: _read_sample(p.generate_sample(T=30))),
            ("batched", lambda: _read_batched(p.generate_batch(batch_size=1))),
        ]:
            N, vmask, _, _ = reader()
            assert vmask[0].item() == 1.0, f"{struct}/{path_name}: A marked hidden"
            assert vmask[N - 1].item() == 1.0, f"{struct}/{path_name}: Y marked hidden"


def test_back_door_has_no_hidden_positions():
    """back_door has no hidden confounder, so all variables stay observable."""
    p = ExtendedCausalTimePrior(tscm_structure="back_door", t_range=(30, 30), seed=4)
    for reader in (
        lambda: _read_sample(p.generate_sample(T=30)),
        lambda: _read_batched(p.generate_batch(batch_size=1)),
    ):
        N, vmask, _, _ = reader()
        assert int((vmask == 0).sum().item()) == 0
