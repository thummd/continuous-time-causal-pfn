"""Continuous-time :class:`DoOverTimePFN` variant.

Drop-in for :class:`dotime.model.do_over_time_pfn.DoOverTimePFN` that
swaps the discrete-time integer-indexed :class:`TemporalEncoder` for
the continuous-time Fourier-time-indexed
:class:`ContinuousTemporalEncoder` and overrides ``encode`` to pass
``times`` / ``t_int_start`` / ``int_onset_idx`` from the batch dict.

Everything else -- the cross-variable mixer, the output heads, the
``loss`` / ``predict`` / ``forward`` plumbing -- is inherited unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from dotime.model.cross_variable_mixer import CrossVariableMixer
from dotime.model.do_over_time_pfn import DoOverTimePFN
from dotime.model.bar_head import BarDistributionHead
from dotime.model.quantile_head import QuantileHead

from .encoder import ContinuousTemporalEncoder


class ContinuousDoOverTimePFN(DoOverTimePFN):
    """Continuous-time analogue of :class:`DoOverTimePFN`.

    Parameters
    ----------
    All parameters from :class:`DoOverTimePFN`, plus:
    num_time_frequencies, time_min_freq, time_max_freq
        Forwarded to :class:`ContinuousTemporalEncoder`'s
        :class:`FourierTimeEmbedding`.  Tune ``time_max_freq`` up for
        sub-second data (e.g. high-frequency sensors) or down for
        long-timescale data (e.g. daily clinical observations).
    positional_only
        If True, every time-bearing field in the batch is rewritten to
        sequence-index coordinates inside :meth:`forward` before either
        the encoder or the mixer sees it.  Realises the
        positional-encoder ablation tier from EXPERIMENT_PLAN_v2: the
        Fourier embedding sees ``arange(T)`` instead of real timestamps,
        and the mixer's normalised ``query_time`` /
        ``intervention_time_*`` scalars are recomputed from those
        sequence indices so the encoder time coordinate equals the
        mixer query coordinate.  Default False (real-time mode).
    """

    def __init__(
        self,
        n_max: int = 41,
        embed_size: int = 512,
        n_heads: int = 4,
        n_encoder_layers: int = 10,
        n_cross_attn_heads: int = 4,
        n_buckets: int = 1000,
        encoder_backend: str = "transformer",
        encoder_config: Optional[dict] = None,
        head_type: str = "quantile",
        tau_levels: Optional[List[float]] = None,
        n_mixer_layers: int = 1,
        context_window: int = 200,
        num_time_frequencies: int = 64,
        time_min_freq: float = 0.01,
        time_max_freq: float = 10.0,
        positional_only: bool = False,
    ) -> None:
        # Skip DoOverTimePFN.__init__ to avoid constructing a
        # TemporalEncoder we'd only replace. Initialise the nn.Module
        # base and then build each component in order.
        torch.nn.Module.__init__(self)
        self.head_type = head_type
        self.positional_only = bool(positional_only)

        self.temporal_encoder = ContinuousTemporalEncoder(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            backend=encoder_backend,
            encoder_config=encoder_config,
            context_window=context_window,
            num_time_frequencies=num_time_frequencies,
            time_min_freq=time_min_freq,
            time_max_freq=time_max_freq,
        )

        self.cross_variable_mixer = CrossVariableMixer(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_cross_attn_heads,
            n_mixer_layers=n_mixer_layers,
        )

        if head_type == "quantile":
            self.quantile_head = QuantileHead(
                embed_size=embed_size,
                tau_levels=tau_levels,
            )
            self.bar_head = None
        else:
            self.bar_head = BarDistributionHead(
                embed_size=embed_size,
                n_buckets=n_buckets,
            )
            self.quantile_head = None

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 1 with continuous-time positional encoding.

        Batch must contain ``X_obs_norm``, ``variable_mask``, ``times``,
        ``t_int_start``, and ``int_onset_idx``.
        """
        return self.temporal_encoder(
            batch["X_obs_norm"],
            batch["variable_mask"],
            times=batch["times"],
            t_int_start=batch["t_int_start"],
            int_onset_idx=batch["int_onset_idx"],
        )

    def _apply_positional_override(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Rewrite every time-bearing field to sequence-index coordinates.

        Idempotent: a batch carrying ``_positional_applied=True`` is
        returned unchanged.
        """
        if batch.get("_positional_applied", False):
            return batch

        times = batch["times"]
        B, T = times.shape
        device, dtype = times.device, times.dtype
        seq = torch.arange(T, device=device, dtype=dtype)
        new_times = seq.unsqueeze(0).expand(B, T).contiguous()
        new_dts = torch.ones(B, T - 1, device=device, dtype=dtype)
        span = max(float(T - 1), 1.0)

        # Use int_onset_idx for t_int_start so the encoder's relative-time
        # signal lines up exactly with the onset-mask used downstream.
        new_t_int_start = batch["int_onset_idx"].to(dtype)

        # t_int_end / t_query: nearest-step argmin against the original
        # absolute times.  t_query coincides exactly with a sample step
        # (it *is* times[query_time_idx]); t_int_end may fall between
        # samples and gets snapped to the nearest one.  Handles both
        # single-query (B,) and multi-query (B, Q) shapes.
        def _to_step(t_abs: torch.Tensor) -> torch.Tensor:
            if t_abs.dim() == 1:
                return (times - t_abs.unsqueeze(-1)).abs().argmin(dim=-1).to(dtype)
            # (B, Q): broadcast against (B, 1, T)
            diff = times.unsqueeze(1) - t_abs.unsqueeze(-1)
            return diff.abs().argmin(dim=-1).to(dtype)

        new_t_int_end = _to_step(batch["t_int_end"])
        new_t_query = _to_step(batch["t_query"])

        return {
            **batch,
            "times": new_times,
            "dts": new_dts,
            "t_int_start": new_t_int_start,
            "t_int_end": new_t_int_end,
            "t_query": new_t_query,
            "intervention_time_start": new_t_int_start / span,
            "intervention_time_end": new_t_int_end / span,
            "query_time": new_t_query / span,
            "_positional_applied": True,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode + query, applying the positional override first if enabled."""
        if self.positional_only:
            batch = self._apply_positional_override(batch)
        return super().forward(batch)
