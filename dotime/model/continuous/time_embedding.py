"""Delta-t aware time embeddings for continuous-time causal PFNs.

The existing :class:`dotime.model.encoder.TemporalEncoder` indexes
positions by integer step count.  In continuous time the spacing
between observations varies, so we need to condition the encoder on
the actual observation times rather than their index.

Two related embeddings are provided:

- :class:`FourierTimeEmbedding` maps a 1-D time scalar to a sinusoidal
  feature vector ``[sin(2 pi f_k t), cos(2 pi f_k t)]_k`` followed by a
  learnable linear projection.  Frequencies ``f_k`` are drawn from a
  geometric progression; by default they cover five decades in
  frequency so the embedding is effectively scale-free over the
  trajectories we simulate (typical ``dt`` between 0.01 and 10).
- :class:`DeltaTEmbedding` embeds the inter-observation gap
  ``dt_i = t_i - t_{i-1}`` on a log scale.  Useful as an auxiliary
  signal for transformers that already process order but not spacing.

Both modules are pure functions of time tensors and are trivially
picklable, making them safe to use in multiprocessing dataloaders.

Why relative-to-intervention time
---------------------------------
The downstream task is in-context causal effect estimation, so the
physically meaningful time coordinate is the gap to the intervention
onset, not absolute clock time.  The helper :func:`relative_to_intervention`
centers a ``(B, T)`` tensor of times on ``(B,)`` onset times without
changing dtype or device.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal features over scalar time, projected to ``embed_size``.

    The embedding applies ``K`` fixed (non-learnable) frequencies drawn
    from a geometric progression between ``min_freq`` and ``max_freq``
    to a scalar time ``t``::

        features = [sin(2 pi f_k t), cos(2 pi f_k t)]_{k=1..K}
        output   = Linear(2 K -> embed_size)(features)

    Parameters
    ----------
    embed_size : int
        Output dimension of the embedding.
    num_frequencies : int
        Number of Fourier frequencies ``K``.  The raw feature vector
        has dimension ``2 * K``.
    min_freq, max_freq : float
        Lower / upper bound of the geometric frequency progression.
        Defaults cover dt values roughly in ``[0.01, 10]``.
    """

    def __init__(
        self,
        embed_size: int,
        num_frequencies: int = 64,
        min_freq: float = 0.01,
        max_freq: float = 10.0,
    ) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}")
        if not 0 < min_freq < max_freq:
            raise ValueError(
                f"require 0 < min_freq < max_freq, got min_freq={min_freq}, max_freq={max_freq}"
            )

        freqs = torch.logspace(
            math.log10(min_freq), math.log10(max_freq), num_frequencies
        )
        # Register as buffer: moves with the module but does not receive gradients.
        self.register_buffer("frequencies", freqs, persistent=True)
        self.projection = nn.Linear(2 * num_frequencies, embed_size, bias=True)
        self.embed_size = embed_size
        self.num_frequencies = num_frequencies

    def forward(self, t: Tensor) -> Tensor:
        """Embed a time tensor.

        Parameters
        ----------
        t : Tensor
            Any shape; time values in the same units as the frequencies.

        Returns
        -------
        Tensor
            Shape ``t.shape + (embed_size,)``.
        """
        # t: (...), frequencies: (K,) -> scaled: (..., K)
        scaled = 2.0 * math.pi * t.unsqueeze(-1) * self.frequencies.to(t)
        features = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return self.projection(features)


class DeltaTEmbedding(nn.Module):
    """Log-scale embedding of inter-observation gaps ``dt``.

    Wraps :class:`FourierTimeEmbedding` after a ``log1p`` transform so
    that small and large gaps receive comparable resolution.
    ``dt = 0`` maps to a zero log-time, which is a safe pad value.

    Parameters
    ----------
    embed_size, num_frequencies, min_freq, max_freq :
        Forwarded to :class:`FourierTimeEmbedding`.  The default
        ``max_freq`` is smaller here because ``log1p(dt)`` has a
        compressed range relative to raw ``t``.
    """

    def __init__(
        self,
        embed_size: int,
        num_frequencies: int = 32,
        min_freq: float = 0.1,
        max_freq: float = 5.0,
    ) -> None:
        super().__init__()
        self.fourier = FourierTimeEmbedding(
            embed_size=embed_size,
            num_frequencies=num_frequencies,
            min_freq=min_freq,
            max_freq=max_freq,
        )
        self.embed_size = embed_size

    def forward(self, dt: Tensor) -> Tensor:
        """Embed ``dt >= 0``.  Negative inputs will be clamped to zero."""
        return self.fourier(torch.log1p(dt.clamp(min=0.0)))


def relative_to_intervention(
    times: Tensor,
    t_int: Optional[Tensor],
) -> Tensor:
    """Return ``times - t_int``, broadcasting over leading dims.

    Parameters
    ----------
    times : Tensor
        Shape ``(B, T)`` or ``(T,)``.
    t_int : Tensor or None
        Shape ``(B,)`` or scalar.  If ``None``, returns ``times``
        unchanged (useful for observational-only encoders).

    Returns
    -------
    Tensor
        Same shape as ``times``.
    """
    if t_int is None:
        return times
    if times.dim() == 1:
        return times - t_int
    if t_int.dim() == 1:
        return times - t_int.unsqueeze(-1)
    return times - t_int
