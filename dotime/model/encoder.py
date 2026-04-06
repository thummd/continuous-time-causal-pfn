"""Per-variable temporal encoder for Do-Over-Time-PFN.

Stage 1 of the two-stage architecture: encodes each variable's time series
independently, producing per-variable temporal representations.

Supports two backends:
- "gdp": GatedDeltaProductEncoder from TempoPFN (requires GPU + fla)
- "transformer": Standard nn.TransformerEncoder (CPU-compatible fallback)

Both backends follow the same interface: (B, T, N_max) -> (B, N, E)
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalEncoder(nn.Module):
    """Per-variable temporal encoder.

    Encodes each variable's time series independently via either
    GatedDeltaProduct (TempoPFN) or standard Transformer layers,
    then pools over time to produce per-variable representations.
    """

    def __init__(
        self,
        n_max: int = 41,
        embed_size: int = 512,
        n_heads: int = 4,
        n_layers: int = 10,
        backend: str = "transformer",
        encoder_config: Optional[dict] = None,
    ):
        super().__init__()
        self.n_max = n_max
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.backend = backend

        # Per-scalar value embedding (from TempoPFN pattern)
        self.expand_values = nn.Linear(1, embed_size, bias=True)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1024, embed_size) * 0.02  # max T=1024
        )

        if backend == "gdp":
            self._init_gdp_layers(embed_size, n_heads, n_layers, encoder_config or {})
        elif backend == "transformer":
            self._init_transformer_layers(embed_size, n_heads, n_layers)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_gdp_layers(self, embed_size, n_heads, n_layers, encoder_config):
        """Initialize GatedDeltaProduct encoder layers (TempoPFN)."""
        from src.models.blocks import GatedDeltaProductEncoder

        config = {
            "expand_v": 1.0,
            "use_short_conv": True,
            "conv_size": 32,
            "use_gate": True,
            "use_forget_gate": True,
            "num_householder": 4,
        }
        config.update(encoder_config)

        self.encoder_layers = nn.ModuleList([
            GatedDeltaProductEncoder(
                layer_idx=i,
                token_embed_dim=embed_size,
                num_heads=n_heads,
                **config,
            )
            for i in range(n_layers)
        ])

        # Learnable initial hidden states (weaving, from TempoPFN)
        head_k_dim = embed_size // n_heads
        expand_v = config.get("expand_v", 1.0)
        head_v_dim = int(head_k_dim * expand_v)
        self.initial_hidden_state = nn.ParameterList([
            nn.Parameter(
                torch.randn(1, n_heads, head_k_dim, head_v_dim) / head_k_dim,
                requires_grad=True,
            )
            for _ in range(n_layers)
        ])
        self.weaving = config.get("weaving", True)

    def _init_transformer_layers(self, embed_size, n_heads, n_layers):
        """Initialize standard Transformer encoder layers."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=n_heads,
            dim_feedforward=embed_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        X_obs: torch.Tensor,
        variable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observational time series per-variable.

        Parameters
        ----------
        X_obs : (B, T, N_max) normalized observational series
        variable_mask : (B, N_max) binary mask for real variables

        Returns
        -------
        h_vars : (B, N_max, E) per-variable temporal representations
        """
        B, T, N = X_obs.shape

        # Time mask: which timesteps have data (pre-intervention, non-zero)
        # (B, T) — True for timesteps with any non-zero variable
        time_mask = (X_obs.abs().sum(dim=-1) > 0)  # (B, T)

        # Embed per-variable: (B, T, N) -> (B, T, N, 1) -> (B, T, N, E)
        x = self.expand_values(X_obs.unsqueeze(-1))  # (B, T, N, E)

        # Add positional encoding (shared across variables)
        x = x + self.pos_encoding[:, :T, :].unsqueeze(2)  # broadcast over N

        # Vectorize: (B, T, N, E) -> (B*N, T, E)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, T, E)
        x = x.view(B * N, T, self.embed_size)

        # Encode
        if self.backend == "gdp":
            x = self._forward_gdp(x, B, N)
        else:
            x = self._forward_transformer(x)

        # Mask-aware pool: average only over non-zero (pre-intervention) timesteps
        # Expand time_mask from (B, T) to (B*N, T, 1) for broadcasting
        tm = time_mask.unsqueeze(1).expand(B, N, T)  # (B, N, T)
        tm = tm.reshape(B * N, T).unsqueeze(-1)      # (B*N, T, 1)
        h = (x * tm).sum(dim=1) / tm.sum(dim=1).clamp(min=1)  # (B*N, E)

        # Reshape: (B*N, E) -> (B, N, E)
        h = h.view(B, N, self.embed_size)

        # Mask out padded variables
        h = h * variable_mask.unsqueeze(-1)

        return h

    def _forward_gdp(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Forward through GatedDeltaProduct layers with state weaving."""
        if self.weaving:
            hidden_state = torch.zeros_like(
                self.initial_hidden_state[0].repeat(B * N, 1, 1, 1)
            )
            for i, layer in enumerate(self.encoder_layers):
                x, hidden_state = layer(
                    x,
                    hidden_state + self.initial_hidden_state[i].repeat(B * N, 1, 1, 1),
                )
        else:
            for i, layer in enumerate(self.encoder_layers):
                init_state = self.initial_hidden_state[i].repeat(B * N, 1, 1, 1)
                x, _ = layer(x, init_state)
        return x

    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through standard Transformer encoder."""
        return self.transformer(x)
