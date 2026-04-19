"""Continuous-time :class:`TemporalEncoder` variant.

Uses a Fourier embedding of the absolute observation times (relative to
the intervention onset) in place of the discrete-time
``rel_pos_encoding`` learnable embedding.  Everything else -- the value
embedding, the transformer / GatedDeltaProduct backbone, the mask-aware
pooling -- is inherited unchanged from
:class:`dotime.model.encoder.TemporalEncoder`.

Forward signature differs from the base class: ``times`` and
``t_int_start`` replace ``int_onset_idx``.  Because of this,
:class:`dotime.model.continuous.ContinuousDoOverTimePFN` provides a
thin model wrapper that overrides ``encode`` to pass the right tensors
from the batch dict.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from dotime.model.encoder import TemporalEncoder

from .time_embedding import FourierTimeEmbedding


class ContinuousTemporalEncoder(TemporalEncoder):
    """Per-variable encoder driven by absolute observation times.

    Parameters
    ----------
    n_max, embed_size, n_heads, n_layers, backend, encoder_config, context_window
        Forwarded to :class:`TemporalEncoder`.
    num_time_frequencies : int
        Number of Fourier frequencies in :class:`FourierTimeEmbedding`.
    time_min_freq, time_max_freq : float
        Range of Fourier frequencies.  Defaults are scale-free over
        ``dt`` in roughly ``[0.01, 10]``, but should be tuned if the
        application domain uses very different time scales (e.g., PK
        data with hours or days).
    """

    def __init__(
        self,
        n_max: int = 41,
        embed_size: int = 512,
        n_heads: int = 4,
        n_layers: int = 10,
        backend: str = "transformer",
        encoder_config: Optional[dict] = None,
        context_window: int = 200,
        num_time_frequencies: int = 64,
        time_min_freq: float = 0.01,
        time_max_freq: float = 10.0,
    ) -> None:
        super().__init__(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_heads,
            n_layers=n_layers,
            backend=backend,
            encoder_config=encoder_config,
            context_window=context_window,
        )
        self.time_embedding = FourierTimeEmbedding(
            embed_size=embed_size,
            num_frequencies=num_time_frequencies,
            min_freq=time_min_freq,
            max_freq=time_max_freq,
        )

    # NOTE: we intentionally override only the forward path; the parent's
    # integer-indexed ``rel_pos_encoding`` parameter stays allocated but
    # unused.  That's ~200 * embed_size extra parameters, a negligible
    # overhead; keeping it simplifies state-dict compatibility if a
    # checkpoint is round-tripped through the base class.

    def forward(  # type: ignore[override]
        self,
        X_obs: Tensor,
        variable_mask: Tensor,
        times: Optional[Tensor] = None,
        t_int_start: Optional[Tensor] = None,
        int_onset_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode observational trajectories with Fourier-time positional embedding.

        Parameters
        ----------
        X_obs : Tensor
            Shape ``(B, T, N_max)``.  Normalised observational series.
        variable_mask : Tensor
            Shape ``(B, N_max)``.
        times : Tensor
            Shape ``(B, T)``.  Absolute observation times in the same
            units as ``t_int_start``.  Required when the model is used
            in continuous-time mode.
        t_int_start : Tensor
            Shape ``(B,)``.  Absolute intervention-onset time per
            sample.
        int_onset_idx : Tensor
            Shape ``(B,)``.  Integer index of the intervention onset
            per sample (needed for truncation to ``context_window``
            pre-intervention observations, which is index-based in the
            continuous case too -- we look at the last ``context_window``
            observations before the intervention regardless of their
            spacing).

        Returns
        -------
        h_vars : Tensor
            Shape ``(B, N_max, embed_size)`` per-variable representations.
        """
        if times is None or t_int_start is None or int_onset_idx is None:
            raise ValueError(
                "ContinuousTemporalEncoder.forward requires times, "
                "t_int_start, and int_onset_idx"
            )

        B, T, N = X_obs.shape
        cw = self.context_window
        device = X_obs.device
        dtype = X_obs.dtype

        # 1. Truncate to the last `cw` pre-intervention observations.
        #    This mirrors TemporalEncoder._forward_relative but we also
        #    gather the corresponding times so we can compute Fourier
        #    features on them.
        X_trunc = torch.zeros(B, cw, N, device=device, dtype=dtype)
        rel_times = torch.zeros(B, cw, device=device, dtype=dtype)
        time_mask = torch.zeros(B, cw, device=device, dtype=torch.bool)

        for b in range(B):
            onset = int(int_onset_idx[b].item())
            start = max(0, onset - cw)
            length = onset - start
            if length > 0:
                X_trunc[b, cw - length:cw, :] = X_obs[b, start:onset, :]
                rel_times[b, cw - length:cw] = (
                    times[b, start:onset] - t_int_start[b]
                )
                time_mask[b, cw - length:cw] = True

        # 2. Value embedding (shared with the base class).
        x = self.expand_values(X_trunc.unsqueeze(-1))  # (B, cw, N, E)

        # 3. Fourier-time positional embedding.
        time_embed = self.time_embedding(rel_times)  # (B, cw, E)
        x = x + time_embed.unsqueeze(2)  # broadcast over N

        # 4. Vectorize (B, cw, N, E) -> (B*N, cw, E) and encode.
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, cw, self.embed_size)
        if self.backend == "gdp":
            x = self._forward_gdp(x, B, N)
        else:
            x = self._forward_transformer(x)

        # 5. Mask-aware pool (same as base class).
        tm = time_mask.unsqueeze(1).expand(B, N, cw).reshape(B * N, cw).unsqueeze(-1)
        h = (x * tm).sum(dim=1) / tm.sum(dim=1).clamp(min=1)
        h = h.view(B, N, self.embed_size)
        h = h * variable_mask.unsqueeze(-1)
        return h
