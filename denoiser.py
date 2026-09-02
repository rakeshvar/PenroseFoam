"""The fixed class/time-conditioned four-channel Direct Transformer."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError(f"Expected time shape (B,), got {tuple(time.shape)}")
        half = self.dimension // 2
        index = torch.arange(half, device=time.device, dtype=time.dtype)
        frequencies = (1000.0 * math.pi) ** (-index / (half - 1))
        phase = time[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class DirectTransformer(nn.Module):
    """Predict joint XYA and nonnegative occupancy velocity per tile."""

    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_global_tokens = config.num_global_tokens
        self.input_projection = nn.Linear(4, config.d_model)
        self.color_embedding = nn.Embedding(2, config.d_model)
        self.class_embedding = nn.Embedding(num_classes, config.class_embed_dim)
        self.class_projection = nn.Linear(config.class_embed_dim, config.d_model)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_embed_dim),
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim * 2, config.d_model),
        )
        self.global_tokens = nn.Parameter(
            torch.randn(1, config.num_global_tokens, config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 4),
        )

    def forward(
        self,
        state: torch.Tensor,
        colors: torch.Tensor,
        time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 3 or state.shape[-1] != 4:
            raise ValueError(f"Expected state shape (B,N,4), got {tuple(state.shape)}")
        if colors.shape != state.shape[:2]:
            raise ValueError("colors must have shape (B,N)")
        batch_size = state.shape[0]
        if time.shape != (batch_size,) or class_labels.shape != (batch_size,):
            raise ValueError("time and class_labels must both have shape (B,)")
        tiles = self.input_projection(state) + self.color_embedding(colors.long())
        hidden = torch.cat(
            (self.global_tokens.expand(batch_size, -1, -1), tiles), dim=1
        )
        condition = self.time_embedding(time).unsqueeze(1)
        condition = condition + self.class_projection(
            self.class_embedding(class_labels.long())
        ).unsqueeze(1)
        raw = self.output_projection(
            self.output_norm(
                self.encoder(hidden + condition)[:, self.num_global_tokens:]
            )
        )
        return torch.cat((raw[..., :3], F.softplus(raw[..., 3:4])), dim=-1)
