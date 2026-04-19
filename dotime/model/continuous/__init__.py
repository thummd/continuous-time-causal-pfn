"""Delta-t aware encoder and continuous-time model variants.

Hosts encoder/mixer/head modifications for continuous-time inputs. The
parent :mod:`dotime.model` package holds the discrete-time reference.

Public API
----------
:class:`FourierTimeEmbedding`
    Sinusoidal features over absolute / relative time.
:class:`DeltaTEmbedding`
    Log-scale embedding of inter-observation gaps.
:func:`relative_to_intervention`
    Re-center a time tensor on the intervention onset.
"""

from .time_embedding import (
    DeltaTEmbedding,
    FourierTimeEmbedding,
    relative_to_intervention,
)

__all__ = [
    "DeltaTEmbedding",
    "FourierTimeEmbedding",
    "relative_to_intervention",
]
