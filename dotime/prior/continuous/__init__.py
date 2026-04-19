"""Continuous-time extensions to the CausalTimePrior.

This subpackage hosts SDE-based mechanism samplers, Ornstein-Uhlenbeck
parameterisations, and variable Delta-t scheduling used by the ICML FMSD
2026 workshop paper.  Discrete-time code in the parent
:mod:`dotime.prior` package stays untouched.

Public API
----------
:class:`OUMechanism`, :func:`sample_ou_mechanism`
    Per-variable linear-drift OU mechanism specification.
:class:`ContinuousSCM`
    Multivariate SCM that integrates OU mechanisms via Euler-Maruyama
    on an arbitrary observation schedule, with support for hard, soft,
    and time-varying interventions, as well as true counterfactual
    pairs via shared noise.
:class:`ContinuousIntervention`, :class:`InterventionKind`
    Intervention specification consumed by :class:`ContinuousSCM`.
:mod:`time_schedule`
    Helpers for regular, jittered, and Poisson-irregular observation
    grids.
"""

from .continuous_scm import (
    ContinuousIntervention,
    ContinuousSCM,
    InterventionKind,
)
from .ou_mechanism import OUMechanism, sample_ou_mechanism
from .time_schedule import (
    exponential_schedule,
    from_times,
    jittered_schedule,
    regular_schedule,
)

__all__ = [
    "ContinuousIntervention",
    "ContinuousSCM",
    "InterventionKind",
    "OUMechanism",
    "exponential_schedule",
    "from_times",
    "jittered_schedule",
    "regular_schedule",
    "sample_ou_mechanism",
]
