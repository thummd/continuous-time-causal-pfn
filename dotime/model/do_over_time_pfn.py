"""Do-Over-Time-PFN: Main model for temporal causal effect estimation.

Two-stage architecture:
1. Per-variable temporal encoding via GatedDeltaProduct or Transformer
2. Cross-variable causal reasoning via attention with intervention/query context
3. Bar distribution output (1000 buckets)
"""

import torch
import torch.nn as nn
from typing import Dict

from dotime.model.encoder import TemporalEncoder
from dotime.model.cross_variable_mixer import CrossVariableMixer
from dotime.model.bar_head import BarDistributionHead


class DoOverTimePFN(nn.Module):
    """In-context causal effect estimation for temporal data.

    Predicts P(X_j^{do}(t_query) | X_obs, intervention_spec) as a
    bar distribution over 1000 buckets.
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
        encoder_config: dict = None,
    ):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            backend=encoder_backend,
            encoder_config=encoder_config,
        )

        self.cross_variable_mixer = CrossVariableMixer(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_cross_attn_heads,
        )

        self.bar_head = BarDistributionHead(
            embed_size=embed_size,
            n_buckets=n_buckets,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        batch : dict with keys:
            X_obs_norm: (B, T, N_max)
            variable_mask: (B, N_max)
            intervention_target: (B,)
            intervention_type: (B,)
            intervention_value: (B,)
            intervention_time_start: (B,)
            intervention_time_end: (B,)
            query_target: (B,)
            query_time: (B,)

        Returns
        -------
        logits : (B, n_buckets) raw logits for bar distribution
        """
        # Stage 1: Per-variable temporal encoding
        h_vars = self.temporal_encoder(
            batch['X_obs_norm'],
            batch['variable_mask'],
        )  # (B, N_max, E)

        # Stage 2: Cross-variable causal reasoning
        h_causal = self.cross_variable_mixer(
            h_vars=h_vars,
            intervention_target=batch['intervention_target'],
            intervention_type=batch['intervention_type'],
            intervention_value=batch['intervention_value'],
            intervention_time_start=batch['intervention_time_start'],
            intervention_time_end=batch['intervention_time_end'],
            query_target=batch['query_target'],
            query_time=batch['query_time'],
            variable_mask=batch['variable_mask'],
        )  # (B, E)

        # Stage 3: Bar distribution output
        logits = self.bar_head(h_causal)  # (B, n_buckets)

        return logits

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute bar distribution loss for a batch.

        Parameters
        ----------
        batch : dict with all required keys including Y_true_norm

        Returns
        -------
        loss : scalar tensor
        """
        logits = self.forward(batch)
        return self.bar_head.loss(logits, batch['Y_true_norm'])

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict mean values for a batch.

        Returns
        -------
        predictions : (B,) predicted mean values (normalized)
        """
        logits = self.forward(batch)
        return self.bar_head.predict_mean(logits)
